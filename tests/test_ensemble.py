"""FNO ensemble training and prediction (M2.2; CLAUDE.md §12, §4, §5).

fake_dataset: a real tiny dataset at 64² written by data/generate.py (train 8, val 4) with
normalisation stats fitted on its train split. Everything runs on the CPU here; the real
members train on the GPU (gpu_instructions.md).
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from data import generate as gen
from models import ensemble as ens
from models import losses
from models import train_ensemble as te
from models.normalise import Normaliser

N = 64


@pytest.fixture(scope="module")
def fake(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("fake_dataset")
    kw = {"N": N, "shard_size": 4, "batch": 4, "device": "cpu", "allow_dirty": True,
          "log": lambda *a, **k: None}
    gen.generate("train", 8, 5, root / "train", **kw)
    gen.generate("val", 4, 5, root / "val", **kw)
    stats = root / "norm_stats.json"
    Normaliser.fit(root, N=N).save(stats)
    return {"root": root, "stats": stats}


def args(fake, tmp: Path, *extra: str, member: int = 1, seed: int = 11, epochs: int = 2,
         stage: str = "pre128", name: str = "m") -> list[str]:
    return ["--member", str(member), "--seed", str(seed), "--stage", stage,
            "--ckpt_dir", str(tmp / name / "ckpt"), "--backup_dir", str(tmp / name / "backup"),
            "--epochs", str(epochs), "--data", str(fake["root"]),
            "--norm-stats", str(fake["stats"]),
            "--registry", str(tmp / "registry.json"), "--N", str(N), "--device", "cpu",
            "--batch", "2", "--workers", "0", *extra]


def log_rows(ckpt: Path) -> list[dict[str, str]]:
    with (ckpt / "log.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------ losses

def test_rel_l2_and_hotspot_l1():
    true = torch.zeros(2, 4, 8, 8)
    true[0, 2, 3, 4] = 10.0  # the single hottest cell of sample 0 is on L3
    true[1] = 1.0
    pred = true.clone()
    pred[0, 0, 0, 0] = 5.0  # error in a cold cell only
    k = 1
    assert losses.hotspot_l1(pred, true, k=k)[0] == 0.0  # top-1 true cell is exact
    pred[0, 2, 3, 4] = 7.0
    assert losses.hotspot_l1(pred, true, k=k)[0] == pytest.approx(3.0)
    rl = losses.rel_l2(pred, true)
    want = torch.sqrt(torch.tensor(5.0**2 + 3.0**2)) / 10.0
    assert rl.shape == (2,) and rl[0] == pytest.approx(float(want))
    assert losses.HOT_CELLS == 64  # CLAUDE.md §12 / §8.1: m = 64
    with pytest.raises(ValueError, match="cells"):
        losses.hotspot_l1(pred, true, k=4 * 8 * 8 + 1)


def test_hotspot_uses_true_cells_not_predicted_ones():
    true = torch.zeros(1, 4, 4, 4)
    true[0, 0, 0, 0] = 1.0
    pred = torch.zeros(1, 4, 4, 4)
    pred[0, 3, 3, 3] = 100.0  # a predicted hotspot elsewhere is not "the hotspot"
    assert losses.hotspot_l1(pred, true, k=1)[0] == pytest.approx(1.0)


# ------------------------------------------------------------------------------ training

def test_two_epoch_smoke_run_writes_checkpoints_summary_and_registry(fake, tmp_path):
    te.main(args(fake, tmp_path))
    ckpt, backup = tmp_path / "m" / "ckpt", tmp_path / "m" / "backup"
    for name in ("best.pt", "last.pt"):
        assert (ckpt / name).exists() and (backup / name).exists()
    rows = log_rows(ckpt)
    assert [r["epoch"] for r in rows] == ["0", "1"]
    for key in ("train_loss", "train_rel_l2", "train_hotspot", "val_rel_l2", "val_hotspot_K"):
        assert all(np.isfinite(float(r[key])) for r in rows), key
    s = json.loads((ckpt / "summary.json").read_text(encoding="utf-8"))
    assert s["member"] == 1 and s["seed"] == 11 and s["stage"] == "pre128" and s["N"] == N
    assert s["epochs_run"] == 2 and s["best_val_hotspot_K"] > 0 and s["best_val_rel_l2"] > 0
    assert s["gpu_hours"] == 0.0 and s["device"] == "cpu"  # GPU-hours count CUDA time only
    reg = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    (entry,) = reg["members"]
    assert entry["sha256"] == hashlib.sha256((ckpt / "best.pt").read_bytes()).hexdigest()
    assert entry["norm_stats_sha256"] == hashlib.sha256(fake["stats"].read_bytes()).hexdigest()
    assert entry["member"] == 1 and entry["stage"] == "pre128"


def test_resume_continues_the_same_curve(fake, tmp_path):
    te.main(args(fake, tmp_path, epochs=3, name="full"))

    class Crash(RuntimeError):
        pass

    def crash(epoch: int) -> None:
        if epoch == 1:
            raise Crash
    with pytest.raises(Crash):
        te.train_member(te.parse(args(fake, tmp_path, epochs=3, name="cut")), on_epoch=crash)
    te.main(args(fake, tmp_path, epochs=3, name="cut"))
    full, cut = log_rows(tmp_path / "full" / "ckpt"), log_rows(tmp_path / "cut" / "ckpt")
    keys = ("epoch", "train_loss", "train_rel_l2", "train_hotspot", "val_rel_l2", "val_hotspot_K",
            "lr")
    assert [[r[k] for k in keys] for r in cut] == [[r[k] for k in keys] for r in full]


def test_early_stopping_on_validation_hotspot(fake, tmp_path, monkeypatch):
    vals = iter([3.0, 2.0, 2.5, 2.6, 2.7, 2.8])
    monkeypatch.setattr(te, "evaluate", lambda *a, **k: (0.1, next(vals)))
    te.main(args(fake, tmp_path, "--patience", "2", epochs=6))
    s = json.loads((tmp_path / "m" / "ckpt" / "summary.json").read_text(encoding="utf-8"))
    assert s["stopped_early"] and s["epochs_run"] == 4 and s["best_epoch"] == 1
    assert s["best_val_hotspot_K"] == 2.0


def test_fine_tuning_needs_and_loads_an_init_checkpoint(fake, tmp_path):
    with pytest.raises(SystemExit):
        te.main(args(fake, tmp_path, stage="ft256", name="ft"))
    te.main(args(fake, tmp_path, epochs=1, name="pre"))
    init = tmp_path / "pre" / "ckpt" / "best.pt"
    cfg = te.parse(args(fake, tmp_path, "--init", str(init), stage="ft256", epochs=1, name="ft"))
    model = te.build_model(cfg, torch.device("cpu"))
    want = torch.load(init, map_location="cpu", weights_only=False)["model"]
    for k, v in model.state_dict().items():
        assert torch.equal(v, want[k]), k
    assert cfg.lr == te.STAGES["ft256"]["lr"] == 3e-4 and te.STAGES["pre128"]["lr"] == 1e-3


def test_batch_probe_picks_the_largest_power_of_two_that_fits():
    def fits(b: int) -> None:
        if b > 6:
            raise torch.cuda.OutOfMemoryError("probe")
    assert te.probe_batch(fits, limit=64) == 4
    assert te.probe_batch(lambda b: None, limit=16) == 16
    with pytest.raises(RuntimeError, match="batch 1"):
        te.probe_batch(lambda b: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")), limit=8)


def test_grad_accumulation_splits_the_requested_batch():
    assert te.plan_batches(requested=16, fits=4) == (4, 4)
    assert te.plan_batches(requested=6, fits=4) == (3, 2)  # micro-batches of equal size
    assert te.plan_batches(requested=None, fits=8) == (8, 1)


# ------------------------------------------------------------------------------ ensemble

@pytest.fixture(scope="module")
def trained(fake, tmp_path_factory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("members")
    te.main(args(fake, tmp, epochs=1, name="one"))
    return {"tmp": tmp, "ckpt": tmp / "one" / "ckpt"}


def identical_registry(fake, trained, n: int = 3) -> Path:
    """A registry of n members that are copies of the same checkpoint."""
    tmp = trained["tmp"]
    reg = json.loads((tmp / "registry.json").read_text(encoding="utf-8"))
    base = reg["members"][0]
    members = []
    for m in range(1, n + 1):
        p = tmp / f"copy{m}.pt"
        shutil.copy2(trained["ckpt"] / "best.pt", p)
        members.append({**base, "member": m, "checkpoint": str(p)})
    out = tmp / f"registry_identical_{n}.json"
    out.write_text(json.dumps({**reg, "members": members}), encoding="utf-8")
    return out


def test_ensemble_sd_is_zero_when_all_members_are_identical(fake, trained):
    e = ens.Ensemble.load(identical_registry(fake, trained), fake["stats"], stage="pre128",
                          device="cpu")
    from models.dataset import TwinHeatDataset
    x = np.stack([TwinHeatDataset(fake["root"], "val", N=N)[i][0] for i in range(2)])
    mu, sd = e.predict(x)
    assert mu.shape == (2, 4, N, N) and sd.shape == (2, 4, N, N)
    assert mu.dtype == np.float32 and sd.dtype == np.float32
    assert np.all(sd == 0)
    one = ens.Ensemble.load(identical_registry(fake, trained, n=1), fake["stats"], stage="pre128",
                            device="cpu")
    np.testing.assert_allclose(one.predict(x)[0], mu, rtol=1e-6)  # mu in kelvin, denormalised
    assert float(mu.max()) > 1.0  # kelvin, not normalised units


def test_ensemble_refuses_a_tampered_checkpoint_or_other_norm_stats(fake, trained, tmp_path):
    reg = identical_registry(fake, trained, n=2)
    members = json.loads(reg.read_text(encoding="utf-8"))["members"]
    p = Path(members[1]["checkpoint"])
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.write_bytes(bytes(data))
    with pytest.raises(ens.EnsembleError, match="SHA-256"):
        ens.Ensemble.load(reg, fake["stats"], stage="pre128", device="cpu")
    other = tmp_path / "other_stats.json"
    other.write_text(fake["stats"].read_text(encoding="utf-8").replace('"N": 64', '"N": 65'),
                     encoding="utf-8")
    with pytest.raises(ens.EnsembleError, match="norm"):
        ens.Ensemble.load(identical_registry(fake, trained, n=1), other, stage="pre128",
                          device="cpu")
