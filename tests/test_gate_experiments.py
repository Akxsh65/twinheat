"""The E-scripts end to end (M3.5) on a synthetic input folder in the real file formats
(M3.3 trajectories, N3.2 labels, N3.3 traces, p₀ calibration): every script runs, writes its
CSVs, verdicts and footed figure; E8's feedback loop feeds verified peaks back to the proposer."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_run_gates import FROZEN  # noqa: E402

from experiments import e7_validity as E7
from experiments import e8_efficiency as E8
from experiments import e14_cert_audit as E14
from experiments import e15_groups as E15
from experiments import e16_active as E16
from experiments import run_gates as RG
from experiments import run_trajectories as RT
from gate import groups as G
from gate import proposers as P

T = 160
SEEDS = [0, 1]


def _write_inputs(root: Path) -> None:
    rng = np.random.default_rng(0)
    (root / "trajectories").mkdir(parents=True)
    lab_h, lab_y, lab_it = [], [], []
    tr_h, tr_it, tr_lo, tr_hi = [], [], [], []

    def label(h, true):
        lab_h.append(h)
        lab_y.append(true)
        lab_it.append(60)
        w = np.array([8.0, 2.0, 0.5, 0.1, 0.0])
        tr_h.append(h)
        tr_it.append([12, 24, 36, 48, 60])
        tr_lo.append((true - w).tolist())
        tr_hi.append((true + w).tolist())

    for name in ("P1", "P2", "P3"):
        for level in (1, 2, 3):
            for seed in SEEDS:
                X = rng.random((T, P.DIM))
                s = rng.uniform(0.3, 1.5, T)
                err = rng.uniform(0, 1.0 + 1.5 * (level - 1), T) * s  # worse at higher levels
                rows = []
                for t in range(T):
                    h = P.x_hash(X[t])
                    peak = 50.0 + rng.normal()
                    label(h, peak - err[t] * (1 if t % 2 else -1))
                    rows.append({
                        "schema_version": "1.0", "run_id": f"traj_{name}_a{level}-abc1234-{seed}",
                        "seed": seed, "t": t, "proposer": name, "aggressiveness": level,
                        "group": G.encode(G.group_names(name, level, 1 + t % 3, t % 5 == 0)),
                        "x_hash": h, "r_hot": float(s[t]) - FROZEN["eps_K"], "sd_hot": 0.0,
                        **dict.fromkeys(RT._NULL_GATE_COLS), "design_id": f"t{t}",
                        "design_vector": X[t].tolist(), "peak_hat": peak, "r_hot_raw": 1.0,
                        "ood_score": 1.0, "ood_tercile": 1 + t % 3, "edge_hotspot": t % 5 == 0,
                        "round": t})
                meta = {"config": {"lambda_per_K": [1.0, 3.0, 10.0]}, "level": level,
                        "m_hot": 64}
                tab = pa.Table.from_pylist(rows, schema=RT.SCHEMA).replace_schema_metadata(
                    {"twinheat": json.dumps(meta)})
                pq.write_table(tab, root / "trajectories" / f"traj_{name}_a{level}_s{seed}.parquet")
    Xc = rng.random((400, P.DIM))
    cs = rng.uniform(0.3, 1.5, 400)
    ch = [P.x_hash(x) for x in Xc]
    pk = 50.0 + rng.normal(size=400)
    for h, y in zip(ch, pk - rng.uniform(0, 1.0, 400) * cs, strict=True):
        label(h, float(y))
    pq.write_table(pa.table({"x_hash": ch, "peak_hat": pk, "r_hot": cs - FROZEN["eps_K"],
                             "sd_hot": np.zeros(400), "design_vector": Xc.tolist()}),
                   root / "calibration.parquet")
    pq.write_table(pa.table({"x_hash": lab_h, "oracle_peak": lab_y, "iters_full": lab_it}),
                   root / "labels.parquet")
    pq.write_table(pa.table({"x_hash": tr_h, "check_iters": tr_it, "T_lo": tr_lo, "T_hi": tr_hi}),
                   root / "audit_traces.parquet")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("inputs")
    _write_inputs(root)
    fz = copy.deepcopy(FROZEN)
    fz.update(seeds=SEEDS, W=100, _run_commit="abc1234", _frozen_commit="f" * 40)
    fz["bootstrap"] = {"resamples": 500, "seed": 3}
    return fz, RG.load_inputs(root, traces=True)


def test_e7_runs_end_to_end_and_prints_the_h1_verdict(world, tmp_path, capsys) -> None:
    fz, data = world
    (v,) = E7.analyse(fz, data, tmp_path)
    assert v["hypothesis"] == "H1" and v["verdict"] in ("SUPPORTED", "NOT SUPPORTED")
    assert "P4" in v["missing"]
    assert "H1:" in capsys.readouterr().out
    for f in ("e7_runs.csv", "e7_groups.csv", "e7_ci.csv", "e7_verdicts.json", "e7_m1.png"):
        assert (tmp_path / f).exists(), f
    runs = (tmp_path / "e7_runs.csv").read_text("utf-8").splitlines()
    n_cfg = len(E7.configs(fz))
    assert len(runs) - 1 == n_cfg * 9 * len(SEEDS)
    meta = json.loads((tmp_path / "e7_verdicts.json").read_text("utf-8"))
    assert meta["frozen_commit"] == "f" * 40 and "commit abc1234" in meta["footer"]


def test_e9_is_the_same_code_path_with_verdict_h3(world, tmp_path) -> None:
    fz, data = world
    (v,) = E7.analyse(fz, data, tmp_path, name="e9", hypothesis="H3")
    assert v["hypothesis"] == "H3" and (tmp_path / "e9_m1.png").exists()


def test_e14_e15_e16_run_end_to_end(world, tmp_path, capsys) -> None:
    fz, data = world
    (v14,) = E14.analyse(fz, data, tmp_path)
    assert v14["hypothesis"] == "H7" and v14["verdict"] in ("SUPPORTED", "NOT SUPPORTED")
    assert 0 < v14["parts"]["iteration_saving"] < 1
    (v15,) = E15.analyse(fz, data, tmp_path)
    assert v15["verdict"].endswith("(empirical)")
    (v16,) = E16.analyse(fz, data, tmp_path)
    assert v16["hypothesis"] == "H9"
    out = capsys.readouterr().out
    assert "H7:" in out and "H8:" in out and "H9:" in out
    for f in ("e14_eps_S.png", "e15_groups.png", "e16_p_min.png"):
        assert (tmp_path / f).exists(), f


# ------------------------------------------------------------------------------------ E8


def _fake_evaluate(xs):
    out = []
    for x in xs:
        x = np.asarray(x, dtype=np.float64)
        out.append(P.Prediction(x=x, x_hash=P.x_hash(x), peak_hat=40 + 10 * float(x[0]),
                                r_hot=0.2 + float(x[1]), sd_hot=0.0, r_hot_raw=0.1, ood_score=1.0,
                                hot_cell=(0, 5, 5), edge_hotspot=False))
    return out


BIAS = 0.5  # K: the surrogate is optimistic, but below tau = 1 K


def _exact(x, h):
    return 40 + BIAS + 10 * float(x[0])


class Spy(P.SimulatedAnnealing):
    def next(self, history):
        self.seen = list(history.entries)
        return super().next(history)


def test_feedback_run_shows_verified_peaks_to_the_proposer(world) -> None:
    from gate.coin import AuditCoin

    fz, data = world
    prop = Spy(T0=1.0, step=0.05, steps=80, T_end_ratio=0.01, seed=0, level=2)
    coin = AuditCoin(seed=0)
    dummy = RG.Trajectory("e8", "P2", 2, 0, None, [], np.zeros((0, P.DIM)),
                          *[np.zeros(0)] * 4, [])
    gate = RG.build_gate("G5", fz, data.cal, dummy, coin, 0, p=0.5, p_min=0.05, f=None,
                         ratio_traj=None)
    r = E8.feedback_run(prop, _fake_evaluate, lambda p_, q: "proposer=P2|aggr=2|ood=1|edge=0",
                        gate, coin, _exact, fz)
    assert r["solves"] == r["flagged"] + r["audited"] - sum(
        1 for t in range(r["T"]) if coin.audited(f"t{t}") and prop.seen[t].decision == "flag")
    for e in prop.seen:
        exact = _exact(np.array(e.x), e.x_hash)
        verified = e.decision == "flag" or coin.audited(e.design_id)
        assert e.peak_hat == pytest.approx(exact if verified else exact - BIAS)
    assert r["curve_best"][-1] == r["final_best"]


def test_e8_pairs_g1_at_g5_budget_and_tests_noninferiority(world) -> None:
    fz, data = world

    def make(name, level, seed):
        return P.SimulatedAnnealing(T0=1.0, step=0.05, steps=steps, T_end_ratio=0.01,
                                    seed=seed, level=level)

    steps = 60  # shorter than G1's window of 100: G1 never flags, so E8 must refuse
    with pytest.raises(RG.InputError, match="verified no design"):
        E8.run_pairs(make, _fake_evaluate, lambda p_, q: "proposer=P2|aggr=1|ood=1|edge=0",
                     _exact, {**fz, "p_audit": 0.2}, data.cal, [("P2", 1)], [0, 1])
    steps = 260

    runs, curves, tests = E8.run_pairs(make, _fake_evaluate, lambda p_, q: (
        f"proposer=P2|aggr={q.level}|ood=1|edge=0"), _exact, {**fz, "p_audit": 0.2}, data.cal,
        [("P2", 1)], [0, 1, 2])
    by = {(r["gate"], r["seed"]): r for r in runs}
    for s in (0, 1, 2):
        g5, g1 = by[("G5", s)], by[("G1", s)]
        assert g1["f"] == pytest.approx(g5["solves"] / g5["T"])
    assert set(tests) == {"P2_a1"} and "upper95_K" in tests["P2_a1"]
    v = E8.verdict_h2(tests)
    assert v["hypothesis"] == "H2" and v["verdict"] in ("SUPPORTED", "NOT SUPPORTED")
    assert len(curves) == 6


def test_noninferiority_on_known_differences() -> None:
    ok = E8.noninferiority([0.0, 0.05, -0.05, 0.02, 0.0], margin=0.2, resamples=2000, seed=0)
    assert ok["noninferior"] and ok["upper95_K"] < 0.2
    bad = E8.noninferiority([0.5, 0.4, 0.6, 0.45, 0.5], margin=0.2, resamples=2000, seed=0)
    assert not bad["noninferior"]
    with pytest.raises(ValueError):
        E8.noninferiority([0.1], margin=0.2, resamples=10, seed=0)
