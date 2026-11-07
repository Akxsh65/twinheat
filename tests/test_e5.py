"""E5 pipeline, baselines, energy-form loss and OOD score (M2.3; CLAUDE.md §11.5, §12, §6.7).

Pipeline checks on a 64² fake dataset with 1-epoch models: they test that the script computes and
writes what E5 needs, not how accurate untrained models are.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data import generate as gen
from data.raster import rasterise
from data.sampler import sample_board
from experiments import e5_accuracy as e5
from experiments import energy_ablation as ea
from models import train_ensemble as te
from models import zoo
from models.energy import energy, energy_loss, stencil_apply
from models.fno import FNO2d
from models.metrics import board_metrics
from models.normalise import Normaliser
from models.ood_score import OODScore, pooled_features
from solvers.fvm import assemble_board, coefficients, solve_direct

N = 64
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------ models

@pytest.mark.parametrize("arch", sorted(zoo.ARCHS))
def test_every_architecture_maps_13_to_4_channels(arch):
    m = zoo.build(zoo.config(arch))
    assert m(torch.randn(2, 13, N, N)).shape == (2, 4, N, N)


def test_unet_has_4_levels_32_base_channels_and_groupnorm():
    m = zoo.build(zoo.config("unet"))
    convs = [c.out_channels for c in m.modules() if isinstance(c, torch.nn.Conv2d)]
    assert {32, 64, 128, 256} <= set(convs) and 512 not in convs
    assert any(isinstance(c, torch.nn.GroupNorm) for c in m.modules())
    with pytest.raises(ValueError, match="divisible by 8"):
        m(torch.randn(1, 13, 60, 60))


def test_fno_local_is_the_fno_plus_a_3x3_branch():
    fno, loc = zoo.build(zoo.config("fno")), zoo.build(zoo.config("fno_local"))
    extra = sum(p.numel() for p in loc.parameters()) - sum(p.numel() for p in fno.parameters())
    assert extra == 4 * (48 * 48 * 9 + 48)  # four 3×3 convs of width 48 with bias
    assert isinstance(zoo.build({k: v for k, v in zoo.config("fno").items() if k != "arch"}), FNO2d)


# ------------------------------------------------------------------------------ energy

@pytest.fixture(scope="module")
def board():
    rec = sample_board(4, "id")
    r = rasterise(rec, gen.SPEC, 16)
    c = coefficients(gen.SPEC, r, rec["dirichlet_edges"])
    A, b = assemble_board(gen.SPEC, r, rec["dirichlet_edges"])
    ct = {k: torch.from_numpy(v)[None] for k, v in c.items()}  # float64 [1, 4, 16, 16]
    return ct, A, b, solve_direct(A, b)


def test_matrix_free_stencil_equals_the_assembled_matrix(board):
    ct, A, _, _ = board
    x = np.random.default_rng(0).standard_normal(A.shape[0])
    got = stencil_apply(ct, torch.from_numpy(x.reshape(1, 4, 16, 16))).numpy().ravel()
    want = A @ x
    assert np.abs(got - want).max() <= 1e-12 * np.abs(want).max()


def test_the_energy_is_minimised_at_the_solver_solution(board):
    ct, A, b, theta = board
    t = torch.from_numpy(theta.reshape(1, 4, 16, 16)).requires_grad_(True)
    e0 = energy({k: v.double() for k, v in ct.items()}, t.double())
    e0.sum().backward()  # gradient = A θ − b, zero at the solution
    assert t.grad.abs().max() <= 1e-8 * np.abs(b).max()
    rng = np.random.default_rng(1)
    for _ in range(5):
        d = torch.from_numpy(rng.standard_normal((1, 4, 16, 16)))
        assert energy(ct, torch.from_numpy(theta.reshape(1, 4, 16, 16)) + d) > e0.detach()


def test_energy_loss_is_dimensionless_and_refuses_a_board_without_heat(board):
    ct, _, _, theta = board
    t = torch.from_numpy(theta.reshape(1, 4, 16, 16))
    assert torch.isfinite(energy_loss(ct, t, 10.0)).all()
    zero = {**ct, "rhs": torch.zeros_like(ct["rhs"])}
    with pytest.raises(ValueError, match="heat"):
        energy_loss(zero, t, 10.0)


# ------------------------------------------------------------------------------ metrics

def test_metrics_on_known_fields():
    true = np.ones((1, 4, 10, 10))
    true[0, 1, 2, 3] = 50.0  # global peak on L2 at (y 2, x 3)
    pred = true.copy()
    pred[0, 1, 2, 3] = 40.0
    pred[0, 0, 8, 7] = 45.0  # predicted global peak on L1 at (y 8, x 7)
    m = board_metrics(pred, true, board_mm=100.0)  # cell 10 mm
    assert m["peak_error_K"]["all"][0] == pytest.approx(5.0)
    assert m["peak_dist_mm"]["all"][0] == pytest.approx(10.0 * np.hypot(6, 4))
    assert m["peak_dist_mm"]["L2"][0] == 0.0 and m["peak_error_K"]["L2"][0] == pytest.approx(10.0)
    assert m["rel_l2"]["L3"][0] == 0.0 and m["rel_l2"]["all"][0] > 0
    assert m["hotspot_l1_K"]["L3"][0] == 0.0


# ------------------------------------------------------------------------------ OOD score

def test_mahalanobis_score_flags_shifted_features_and_5_percent_of_validation():
    rng = np.random.default_rng(0)
    train = [rng.standard_normal((2000, 8)) for _ in range(2)]
    val = [rng.standard_normal((2000, 8)) for _ in range(2)]
    shifted = [rng.standard_normal((500, 8)) + 3.0 for _ in range(2)]
    s = OODScore.fit(train)
    s.set_threshold(val)
    assert abs(s.flag(val).mean() - 0.05) <= 0.002
    assert s.flag(shifted).mean() > 0.95
    assert s.flag([rng.standard_normal((2000, 8)) for _ in range(2)]).mean() < 0.1


def test_ood_score_round_trips_and_features_come_from_the_fno_head(tmp_path):
    rng = np.random.default_rng(1)
    s = OODScore.fit([rng.standard_normal((50, 4))])
    s.set_threshold([rng.standard_normal((50, 4))])
    s.save(tmp_path / "o.json")
    t = OODScore.load(tmp_path / "o.json")
    f = [rng.standard_normal((7, 4))]
    np.testing.assert_allclose(t.score(f), s.score(f))
    assert t.threshold == s.threshold
    fno = zoo.build(zoo.config("fno"))
    assert pooled_features(fno, torch.randn(3, 13, N, N)).shape == (3, 48)
    with pytest.raises(TypeError, match="FNO"):
        pooled_features(zoo.build(zoo.config("unet")), torch.randn(1, 13, N, N))


# ------------------------------------------------------------------------------ end to end

@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("e5data")
    kw = {"N": N, "shard_size": 4, "batch": 4, "device": "cpu", "allow_dirty": True,
          "log": lambda *a, **k: None}
    for split, count in (("train", 8), ("val", 6), ("test", 4), ("ood_1", 4)):
        gen.generate(split, count, 9, root / split, **kw)
    stats = root / "norm_stats.json"
    Normaliser.fit(root, N=N).save(stats)
    ck = tmp_path_factory.mktemp("ck")
    reg = ck / "registry.json"

    def train(arch: str, member: int, stage: str, init: Path | None = None) -> Path:
        d = ck / f"{arch}{member}_{stage}"
        argv = ["--member", str(member), "--seed", str(10 + member), "--stage", stage,
                "--arch", arch, "--epochs", "1", "--data", str(root), "--norm-stats", str(stats),
                "--registry", str(reg), "--ckpt_dir", str(d / "ckpt"), "--backup_dir",
                str(d / "bk"), "--N", str(N), "--device", "cpu", "--batch", "2", "--workers", "0"]
        te.main(argv + (["--init", str(init)] if init else []))
        return d / "ckpt" / "best.pt"
    for arch, members in (("fno", (1, 2)), ("unet", (1,)), ("fno_local", (1,))):
        for m in members:
            train(arch, m, "ft256", init=train(arch, m, "pre128"))
    return {"root": root, "stats": stats, "registry": reg, "ck": ck}


def make_config(world, tmp_path) -> Path:
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / "e5.yaml").read_text(encoding="utf-8"))
    cfg.update(registry=str(world["registry"]), norm_stats=str(world["stats"]), N=N, batch=2,
               splits=["test", "ood_1"])
    cfg["energy_ablation"].update(seed=3, epochs=1, energy_weight=1.0,
                                  runs=[{"labels": 0, "energy": True},
                                        {"labels": 2, "energy": False}])
    p = tmp_path / "e5.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_e5_produces_the_table_figure_ood_and_go_no_go(world, tmp_path):
    out = tmp_path / "out"
    e5.main(["--config", str(make_config(world, tmp_path)), "--data", str(world["root"]),
             "--out", str(out), "--device", "cpu", "--allow-dirty"])
    for f in ("e5_metrics.csv", "e5_ood.csv", "e5_tables.tex", "e5_figure.png", "go_no_go.md",
              "run_info.json", "ood_score.json"):
        assert (out / f).exists(), f
    with (out / "e5_metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models = {r["model"] for r in rows}
    assert models == {"ensemble", "fno_pre128_at_256", "unet", "fno_local", "member1", "member2"}
    assert {r["split"] for r in rows} == {"test", "ood_1"}
    assert {r["metric"] for r in rows} == {"rel_l2", "peak_error_K", "hotspot_l1_K",
                                           "peak_dist_mm"}
    assert {r["layer"] for r in rows} == {"all", "L1", "L2", "L3", "L4"}
    assert all(int(r["n"]) == 4 and float(r["p95"]) >= 0 for r in rows)
    with (out / "e5_ood.csv").open(encoding="utf-8") as f:
        ood = {r["split"]: r for r in csv.DictReader(f)}
    assert set(ood) == {"test", "ood_1"} and ood["test"]["kind"].startswith("false-positive")
    gate = (out / "go_no_go.md").read_text(encoding="utf-8")
    ens = next(r for r in rows if (r["model"], r["split"], r["metric"], r["layer"])
               == ("ensemble", "test", "rel_l2", "all"))
    assert f"{float(ens['mean']):.4f}" in gate and ("GO" in gate)
    tex = (out / "e5_tables.tex").read_text(encoding="utf-8")
    assert r"fno\_local" in tex and r"ood\_1" in tex
    assert json.loads((out / "run_info.json").read_text(encoding="utf-8"))["N"] == N


def test_e5_refuses_a_config_with_uncommitted_changes_or_missing_fields(world, tmp_path,
                                                                        monkeypatch):
    p = ROOT / "experiments" / "configs" / "e5.yaml"
    monkeypatch.setattr(e5, "_git", lambda *a: " M experiments/configs/e5.yaml")
    with pytest.raises(e5.E5Error, match="uncommitted"):
        e5.load_config(p, allow_dirty=False)
    outside = make_config(world, tmp_path)
    monkeypatch.undo()
    with pytest.raises(e5.E5Error, match="inside the repo"):
        e5.load_config(outside, allow_dirty=False)
    bad = tmp_path / "bad.yaml"
    bad.write_text("N: 64\n", encoding="utf-8")
    with pytest.raises(e5.E5Error, match="registry"):
        e5.load_config(bad, allow_dirty=True)


def test_energy_ablation_needs_its_settings_then_trains_and_scores_each_run(world, tmp_path):
    shipped = e5.load_config(ROOT / "experiments" / "configs" / "e5.yaml", allow_dirty=True)
    with pytest.raises(e5.E5Error, match="seed"):
        ea.run(shipped, world["root"], tmp_path / "a", tmp_path / "b", "cpu", tmp_path / "r.json")
    cfg = e5.load_config(make_config(world, tmp_path), allow_dirty=True)
    rows = ea.run(cfg, world["root"], tmp_path / "ck", tmp_path / "bk", "cpu",
                  tmp_path / "reg.json", ["--N", str(N), "--batch", "2", "--workers", "0"])
    assert [r["variant"] for r in rows] == ["energy_L0", "data_L2"]
    assert all(np.isfinite(r["rel_l2_mean"]) and r["n_test"] == 4 for r in rows)
    with (tmp_path / "ck" / "energy_L0" / "log.csv").open(encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    assert float(log[0]["train_energy"]) != 0.0  # the energy term was trained on
