"""CPU certificate (Lemma 2, CLAUDE.md §8.4, §9.1; invariant B7) on the 64² plate.

The perturbations here are test perturbations of our own (a smooth blob plus noise). They are
NOT the §8.6 emulated-error models, which live in bench/prelim/stop.js and cert.js (not yet in
the repo); bench/prelim/reproduce.py ports those.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from certify.kappa import kappa_from_z, kappa_hat, multigrid
from certify.pointwise import gauss_seidel_smooth, peak_interval, pointwise_bound
from certify.stopping import CertificateViolation, check_bound, jacobi, pcg_certified
from solvers.fvm import assemble_plate, solve_direct
from solvers.mmatrix_test import assert_m_matrix
from tests.test_fvm_plate import BOARDS

ORACLE_RTOL = 1e-12  # the prompt's oracle: 1e-12 direct solve


def _true_inv_norm(A: sp.csr_matrix, block: int = 512) -> float:
    """‖A⁻¹‖∞ = max row sum of |A⁻¹|, by solving A X = I in column blocks (64² only)."""
    n = A.shape[0]
    lu = spla.splu(sp.csc_matrix(A))
    rows = np.zeros(n)
    for j0 in range(0, n, block):
        j1 = min(n, j0 + block)
        E = np.zeros((n, j1 - j0))
        E[np.arange(j0, j1), np.arange(j1 - j0)] = 1.0
        rows += np.abs(lu.solve(E)).sum(axis=1)
    return float(rows.max())


def _case(i: int):
    rec, (G, H, q, dx, edges) = BOARDS[i]
    A, b = assemble_plate(G, H, q, dx, edges)
    assert_m_matrix(A, context=rec["id"])
    theta = solve_direct(A, b, rtol=ORACLE_RTOL)
    return rec, A, b, theta


def _perturbed(theta: np.ndarray, level: float, seed: int) -> np.ndarray:
    """θ + a smooth blob of `level` × peak + 1 % of that as white noise (test perturbation)."""
    n = int(np.sqrt(theta.size))
    rng = np.random.Generator(np.random.PCG64(seed))
    c = (np.arange(n) + 0.5) / n
    Y, X = np.meshgrid(c, c, indexing="ij")
    cx, cy = rng.uniform(0.2, 0.8, 2)
    blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 0.1**2)).ravel()
    amp = level * theta.max()
    return theta + amp * blob + 0.01 * amp * rng.standard_normal(theta.size)


CASES = [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------- Lemma 2(a): κ̂

@pytest.mark.parametrize("i", CASES)
def test_kappa_hat_bounds_the_true_inverse_norm(i):
    rec, A, _, _ = _case(i)
    k = kappa_hat(A)
    true = _true_inv_norm(A)
    assert k is not None, rec["id"]
    print(f"\n{rec['id']} edges {rec['dirichlet_edges']}: kappa_hat / true = {k / true:.4f}")
    assert k >= true, f"{rec['id']}: kappa_hat {k} < true ‖A⁻¹‖∞ {true}"


def test_kappa_from_z_rejects_a_non_positive_z():
    _, A, _, _ = _case(0)
    z = multigrid(A).solve(np.ones(A.shape[0]), maxiter=3, tol=0.0)
    assert kappa_from_z(A, z) is not None
    for bad in (0.0, -1e-3):
        zb = z.copy()
        zb[17] = bad
        assert kappa_from_z(A, zb) is None


def test_kappa_from_z_rejects_a_non_positive_Az():
    _, A, _, _ = _case(0)
    z = np.ones(A.shape[0])
    z[100] = 50.0  # a spike makes (Az) negative at the spike's neighbours
    assert (A @ z).min() <= 0
    assert kappa_from_z(A, z) is None


def test_kappa_with_z_equal_ones_is_varah():
    """§18.5: z = 1 gives 1 / min row margin (Varah 1975) for a strictly dominant M-matrix."""
    _, A, _, _ = _case(1)
    margin = np.asarray(A.sum(axis=1)).ravel()
    assert margin.min() > 0
    k = kappa_from_z(A, np.ones(A.shape[0]))
    assert k == pytest.approx(1.0 / margin.min(), rel=2e-6)  # the (1 + 1e-6) placeholder margin


def test_kappa_hat_raises_on_wrong_input():
    with pytest.raises(ValueError, match="square"):
        kappa_hat(sp.csr_matrix(np.ones((3, 4))))
    with pytest.raises(ValueError, match="n_vcycles"):
        kappa_hat(sp.identity(4, format="csr"), n_vcycles=0)


# ---------------------------------------------------------------------- Lemma 2(b): certified PCG

@pytest.mark.parametrize("i", CASES)
@pytest.mark.parametrize("level", [0.02, 0.05])
def test_every_pcg_iterate_bound_is_above_the_true_error(i, level):
    rec, A, b, theta = _case(i)
    k = kappa_hat(A)
    x0 = _perturbed(theta, level, seed=i)
    seen = []

    def on_iterate(it, x, bound):
        err = float(np.abs(theta - x).max())
        check_bound(rec["id"], it, bound, err)  # raises CertificateViolation
        seen.append((it, bound, err))

    x, iters, cert = pcg_certified(A, b, x0, jacobi(A), k, tau_v=0.1, history=on_iterate)
    assert cert is not None and cert <= 0.1, (rec["id"], iters, cert)
    err = float(np.abs(theta - x).max())
    assert err <= cert
    assert len(seen) == iters + 1  # iterate 0 (the warm start) and every PCG step
    assert seen[-1][1] == pytest.approx(cert)


@pytest.mark.parametrize("i", [0, 2, 5])
def test_bound_is_nearly_tight_for_the_worst_case_error(i):
    """x₀ = θ − c·A⁻¹1 gives r = c·1, so ‖e‖∞ = c‖A⁻¹‖∞ and the bound κ̂·c is only κ̂/‖A⁻¹‖∞
    above the error: any under-reported bound (e.g. a lost factor) violates here."""
    rec, A, b, theta = _case(i)
    k = kappa_hat(A)
    u = solve_direct(A, np.ones(A.shape[0]), rtol=ORACLE_RTOL)  # A⁻¹1 ≥ 0
    x0 = theta - 0.01 * u
    got = {}

    def on_iterate(it, x, bound):
        err = float(np.abs(theta - x).max())
        check_bound(rec["id"], it, bound, err)
        got.setdefault(it, (bound, err))

    pcg_certified(A, b, x0, jacobi(A), k, tau_v=1e-3, maxit=5, history=on_iterate)
    bound0, err0 = got[0]
    assert err0 <= bound0 <= 1.5 * err0, (rec["id"], bound0, err0)


def test_cheap_path_returns_the_same_certificate_as_history():
    rec, A, b, theta = _case(2)
    k = kappa_hat(A)
    x0 = _perturbed(theta, 0.02, seed=9)
    x1, it1, c1 = pcg_certified(A, b, x0, jacobi(A), k, tau_v=1.0)
    x2, it2, c2 = pcg_certified(A, b, x0, jacobi(A), k, tau_v=1.0, history=lambda *a: None)
    assert it1 == it2 and c1 == pytest.approx(c2) and np.allclose(x1, x2)


def test_returns_none_when_maxit_is_hit():
    _, A, b, theta = _case(0)
    k = kappa_hat(A)
    x, iters, cert = pcg_certified(A, b, _perturbed(theta, 0.05, 1), jacobi(A), k, tau_v=1e-9,
                                   maxit=3)
    assert cert is None and iters == 3


def test_check_bound_raises_with_board_iterate_bound_and_error():
    with pytest.raises(CertificateViolation, match=r"board b7, iterate 12: bound 0\.5 < .*0\.6"):
        check_bound("b7", 12, 0.5, 0.6)
    check_bound("b7", 12, 0.6, 0.6)  # equality is not a violation


# ---------------------------------------------------------------------- Lemma 2(c): pointwise

@pytest.mark.parametrize("i", CASES)
@pytest.mark.parametrize("sweeps", [0, 5, 10])
def test_pointwise_bound_covers_every_cell_and_the_peak(i, sweeps):
    rec, A, b, theta = _case(i)
    x = gauss_seidel_smooth(A, b, _perturbed(theta, 0.03, seed=i), sweeps)
    r = b - A @ x
    w = pointwise_bound(A, r)
    err = np.abs(theta - x)
    bad = np.flatnonzero(err > w)
    assert bad.size == 0, f"{rec['id']} sweeps {sweeps}: |e| > w at {bad[:5]}"
    lo, hi = peak_interval(x, w)
    assert lo <= theta.max() <= hi


def test_gauss_seidel_lowers_the_error():
    _, A, b, theta = _case(3)
    x0 = _perturbed(theta, 0.03, seed=3)
    e0 = np.abs(theta - x0).max()
    e5 = np.abs(theta - gauss_seidel_smooth(A, b, x0, 5)).max()
    assert e5 < e0
    np.testing.assert_array_equal(gauss_seidel_smooth(A, b, x0, 0), x0)


def test_pointwise_bound_validity_is_asserted():
    """A residual the approximate solve cannot cover must raise, not return a bad w."""
    _, A, b, _ = _case(0)
    r = np.zeros(A.shape[0])
    r[5] = 1.0
    w = pointwise_bound(A, r)
    assert np.all(A @ w >= np.abs(r)) and w.min() >= 0
    with pytest.raises(ValueError, match="M-matrix"):
        pointwise_bound(-A, r)
