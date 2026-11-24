"""Audit traces for E14 (experiments/audit_traces.py; CLAUDE.md §8.3, D14, B7, B8).

Failure modes covered: a trace set that misses an audited design (coin and replay disagree), a
trace check that does not contain the oracle peak (B7, must raise), and the H7 baseline
(iters_full) not reaching the replay.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse.linalg as spla
from test_run_gates import FROZEN, make_cal, make_traj

from data.raster import rasterise
from data.sampler import design_board
from experiments import audit_traces as AT
from experiments import run_gates as RG
from gate.proposers import sample_p0
from solvers.fvm import assemble_board
from twinheat import spec as specmod

SPEC = specmod.load(RG.ROOT / "config" / "spec_v2.yaml")


def test_audited_set_from_the_coin_equals_the_replay() -> None:
    trajs, labels = [], RG.Labels(oracle={}, iters_full={}, traces={})
    for seed in range(3):
        tr, lab = make_traj(np.linspace(0, 2, 60), proposer="P2", level=2, seed=seed)
        trajs.append(tr)
        labels.oracle.update(lab.oracle)
    cal, _ = make_cal(np.full(999, 0.5))
    data = RG.Inputs(trajs, labels, cal, "0" * 64)
    by_coin = AT.audited_by_coin(trajs, FROZEN["p_audit"])
    assert by_coin == AT.audited_by_replay(data, FROZEN)
    assert sum(len(v) for v in by_coin.values()) > 0
    assert AT.audited(data, FROZEN) == sorted({h for v in by_coin.values() for h in v})


def _design(n=32):
    x = sample_p0(np.random.Generator(np.random.PCG64(4)))
    rec = design_board(x)
    A, b = assemble_board(SPEC, rasterise(rec, SPEC, n), rec["dirichlet_edges"])
    theta = spla.spsolve(A.tocsc(), b).reshape(4, n, n)
    return x, theta


def test_trace_contains_the_oracle_and_reports_iters_full() -> None:
    x, theta = _design()
    mu = theta * 1.02
    row = AT._trace(("h", list(x), mu, float(theta.max()), 5, AT.D14_MAXIT, 60))
    assert row["stop_reason"] == "converged" and row["iters_full"] == row["check_iters"][-1]
    assert all(lo <= theta.max() <= hi for lo, hi in zip(row["T_lo"], row["T_hi"], strict=True))
    with pytest.raises(RG.CertificateViolation, match="B7"):
        AT._trace(("h", list(x), mu, float(theta.max()) + 50.0, 5, AT.D14_MAXIT, 60))


def test_iters_full_reaches_the_labels(tmp_path) -> None:
    lab = tmp_path / "labels.parquet"
    pq.write_table(pa.table({"x_hash": ["a", "b"], "oracle_peak": [1.0, 2.0],
                             "iters_full": pa.array([None, None], pa.int64())}), lab)
    tr = tmp_path / "audit_traces.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"x_hash": "a", "check_iters": [0, 5], "T_lo": [0.0, 0.9], "T_hi": [2.0, 1.1],
         "iters_full": 5, "stop_reason": "converged"}], schema=AT.SCHEMA), tr)
    labels = RG.Labels.from_parquet(lab, tr)
    assert labels.iters_full == {"a": 5, "b": None} and "a" in labels.traces
