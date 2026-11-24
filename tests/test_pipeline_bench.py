"""E4 benchmark plumbing on the CPU (bench/pipeline_bench.py; CLAUDE.md §11.4 H5, B7).

Small sampler boards (N = 24) through the sequential loop and the batched path with NumPy: every
board certified with a true bound, rows and verdict written, a lying bound caught. The GPU rates
themselves come only from rvcse (--device gpu).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from bench import pipeline_bench as PB
from certify.kappa import kappa_hat
from certify.peak_interval import row_abs_sum_bound
from data.raster import rasterise
from data.sampler import sample_board
from solvers.fvm import assemble_board, coefficients
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 24


def boards(n: int) -> list[dict]:
    out = []
    for i in range(n):
        rec = sample_board(100 + i, "id")
        r = rasterise(rec, SPEC, N)
        A, _ = assemble_board(SPEC, r, rec["dirichlet_edges"])
        c = coefficients(SPEC, r, rec["dirichlet_edges"])
        theta = spla.spsolve(A.tocsc(), c["rhs"].ravel()).reshape(4, N, N)
        rng = np.random.Generator(np.random.PCG64(i))
        mu = theta * (1 + 0.03 * rng.standard_normal(theta.shape))
        out.append({"i": i, "id": rec["id"], "record": rec, "coeffs": c, "mu": mu,
                    "theta": theta, "kappa": kappa_hat(A, 60, accel="cg"),
                    "S": row_abs_sum_bound(c), "prep_s": 0.0, "kappa_s": 0.0})
    return out


def test_sequential_and_batched_cpu(tmp_path: Path) -> None:
    bs = boards(4)
    seq = PB.sequential(bs, 0.1)
    rows = [PB.batched(bs, b, tau_v=0.1, check_every=5, maxit=5000, dtype="float64",
                       device="cpu") for b in (1, 2, 4)]
    assert seq["boards"] == 4 and seq["designs_per_s"] > 0
    assert [r["boards"] for r in rows] == [4, 4, 4] and all(r["maxit_stops"] == 0 for r in rows)
    v = PB.write(tmp_path / "pb.csv", tmp_path / "pb.png", seq, rows,
                 {"tau_v": 0.1, "footer": "test"})
    assert set(v["speedup_by_batch"]) == {1, 2, 4} and v["passed"] is False  # no batch >= 64
    assert (tmp_path / "pb.png").exists()
    assert json.loads((tmp_path / "pb.json").read_text())["H5"]["criterion"].startswith(">= 10")


def test_lying_bound_raises() -> None:
    bd = boards(1)[0]
    with pytest.raises(PB.B7Violation, match="B7"):
        PB._b7(bd, bd["mu"], 1e-12, "test")


def test_verdict_rule() -> None:
    seq = {"designs_per_s": 1.0}
    rows = [{"batch": 8, "designs_per_s": 3.0}, {"batch": 64, "designs_per_s": 12.0}]
    v = PB.verdict(seq, rows)
    assert v["passed"] and v["crossover_batch"] == 8
    rows[1]["designs_per_s"] = 9.0
    assert not PB.verdict(seq, rows)["passed"]
