"""E13: certified early stopping on the 4-layer board (CLAUDE.md §8.4, §8.6, §11.4 H6, D1, B7).

Small boards (N = 32, sampler boards with and without heat-sunk edges) with emulated warm starts
μ = θ·(1 + 3 % smooth noise), as §8.6. Failure modes covered: a bound below the true error
(B7, must raise), a certified stop later than the fixed-tolerance stop, a board without a
certificate silently dropped, the loose-tolerance comparison computed at the wrong tolerance,
a wrong H6 / week-6 verdict, and the run touching a protected split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from certify.stopping import CertificateViolation
from data.raster import rasterise
from data.sampler import sample_board
from experiments import e13_certificate as E
from solvers.fvm import assemble_board, coefficients
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 32


def board(seed: int):
    rec = sample_board(seed, "id")
    r = rasterise(rec, SPEC, N)
    A, b = assemble_board(SPEC, r, rec["dirichlet_edges"])
    c = coefficients(SPEC, r, rec["dirichlet_edges"])
    theta = spla.spsolve(A.tocsc(), b)
    rng = np.random.Generator(np.random.PCG64(seed))
    y, x = np.mgrid[0:N, 0:N] / N
    f = sum(rng.uniform(-1, 1) * np.sin(2 * np.pi * (k * x + j * y) + rng.uniform(0, 6))
            for k, j in ((1, 2), (2, 1), (3, 1)))
    mu = (theta.reshape(4, N, N) * (1 + 0.03 * f / np.abs(f).max())[None]).ravel()
    return A, b, c, mu, theta, rec["dirichlet_edges"]


@pytest.mark.parametrize("precond", ["linez", "jacobi"])
def test_zero_violations_and_certified_stops_no_later(precond: str) -> None:
    certified = 0
    for seed in range(8):
        A, b, c, mu, theta, edges = board(seed)
        row, _ = E.analyse_board(A, b, c, mu, theta, board_id=f"b{seed}", precond=precond,
                                 pointwise_every=5, maxit=20000)
        assert row["iters_fixed_warm"] > 0
        if not row["certificate"]:  # 3 V-cycles gave no z > 0 with A z > 0: counted, not dropped
            assert row["kappa"] is None and "bound_0.1K" not in row
            continue
        certified += 1
        for tag in E.TAU_TAGS.values():
            assert row[f"iters_cert_kappa_{tag}"] <= row["iters_fixed_warm"]
            assert row[f"bound_{tag}"] >= row[f"true_err_{tag}"]  # B7
            assert row[f"iters_cert_best_{tag}"] <= row[f"iters_cert_kappa_{tag}"]
        assert row["iters_fixed_warm"] <= row["iters_fixed_cold"] + 50
        for s in E.GS_SWEEPS:
            assert row[f"pw_cert_gs{s}"] >= row[f"pw_true_gs{s}"]
    assert certified >= 4, f"only {certified} of 8 small boards had a certificate"


def test_violation_raises(monkeypatch) -> None:
    A, b, c, mu, theta, _ = board(0)

    def lying(A, b, x0, Minv, kappa, tau_v, maxit=20000, **kw):
        return np.asarray(x0, float).copy(), 0, 1e-9  # claims 1e-9 K at the warm start
    monkeypatch.setattr(E, "pcg_certified", lying)
    with pytest.raises(CertificateViolation, match="b0"):
        E.analyse_board(A, b, c, mu, theta, board_id="b0", precond="linez",
                        pointwise_every=5, maxit=20000)


def test_board_without_certificate_is_counted(monkeypatch) -> None:
    A, b, c, mu, theta, _ = board(1)
    monkeypatch.setattr(E, "kappa_from_z", lambda A, z: None)
    row, tr = E.analyse_board(A, b, c, mu, theta, board_id="b1", precond="linez",
                              pointwise_every=5, maxit=20000)
    assert row["certificate"] is False and row["iters_fixed_warm"] > 0
    s = E.summarise([row], [tr])
    assert s["n_boards"] == 1 and s["n_no_certificate"] == 1 and s["n_certified"] == 0


def test_loose_tolerance_at_equal_mean_iterations() -> None:
    # two boards whose relres falls by 10x per iteration; errors equal to relres
    rel = [10.0 ** -np.arange(0, 12), 10.0 ** -np.arange(0, 12)]
    err = [r * 10 for r in rel]
    traces = [{"relres": r, "err": e} for r, e in zip(rel, err, strict=True)]
    tol, iters, miss = E.loose_tolerance(traces, target_mean_iters=4.0, tau=5e-4)
    assert iters == pytest.approx(4.0) and tol == pytest.approx(1e-4)  # the loosest such tol
    assert miss == 1.0  # err at iteration 4 is 1e-3 > 5e-4
    assert E.loose_tolerance(traces, target_mean_iters=4.0, tau=1e-2)[2] == 0.0


def test_summary_verdicts() -> None:
    def row(fixed, best, kappa, bound, true):
        r = {"certificate": True, "iters_fixed_warm": fixed, "iters_fixed_cold": fixed + 10}
        for tag in E.TAU_TAGS.values():
            r.update({f"iters_cert_best_{tag}": best, f"iters_cert_kappa_{tag}": kappa,
                      f"iters_cert_pw_{tag}": best, f"bound_{tag}": bound,
                      f"true_err_{tag}": true})
        return r
    tr = {"relres": np.array([1.0, 1e-9]), "err": np.array([1.0, 0.0])}
    good = E.summarise([row(100, 40, 60, 0.09, 0.01)] * 3, [tr] * 3)
    assert good["H6"]["passed"] and good["week6"]["use_pointwise"] is False
    bad = E.summarise([row(100, 80, 90, 0.09, 0.001)] * 3, [tr] * 3)
    assert not bad["H6"]["passed"] and bad["week6"]["use_pointwise"]  # saving 10 %, ratio 90×


@pytest.mark.parametrize("split", ["train", "val", "cal"])
def test_protected_and_training_splits_refused(split: str) -> None:
    with pytest.raises(E.E13Error, match="split"):
        E.check_split(split)
    E.check_split("test")
    E.check_split("ood_3oz")


def test_cg_z_gives_a_valid_kappa_where_vcycles_do_not() -> None:
    # 4-layer board without a heat-sunk edge, N = 64: the AMG-CG kappa is a true and tight
    # upper bound on ||A^-1||_inf (A^-1 >= 0, so the norm is max of A^-1 1, by a direct solve)
    from certify.kappa import kappa_hat
    rec = next(r for r in (sample_board(s, "id") for s in range(40)) if not r["dirichlet_edges"])
    r = rasterise(rec, SPEC, 64)
    A, _ = assemble_board(SPEC, r, rec["dirichlet_edges"])
    exact = float(spla.spsolve(A.tocsc(), np.ones(A.shape[0])).max())
    k = kappa_hat(A, 60, accel="cg")
    assert k is not None and exact <= k <= 1.05 * exact
    with pytest.raises(ValueError, match="accel"):
        kappa_hat(A, 3, accel="gmres")
