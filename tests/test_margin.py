"""Derived FP64 rounding margin for the Lemma 2 certificate (prompt N4.1; CLAUDE.md §8.4, B7).

The reference is 80-digit mpmath arithmetic on the STORED float64 A, b, x, z: the certificate the
float pipeline returns must be >= the exact κ̂·‖b − A x‖∞ with κ̂ = max z / min(A z) computed
exactly, on 5 boards, for a rough and a fully converged iterate, and for two matvec orders.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from certify import margin as mg
from certify.kappa import positive_z
from data.raster import rasterise
from data.sampler import sample_board
from solvers.fvm import assemble_board, coefficients
from twinheat import spec as specmod

mpmath = pytest.importorskip("mpmath")

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 24  # 4 x 24² = 2,304 unknowns: small enough for 80-digit arithmetic
U = Fraction(1, 2**53)


def _board(seed: int):
    rec = sample_board(seed, "id")
    r = rasterise(rec, SPEC, N)
    A, b = assemble_board(SPEC, r, rec["dirichlet_edges"])
    return sp.csr_matrix(A), b, coefficients(SPEC, r, rec["dirichlet_edges"])


def _stencil_matvec(c):
    """A x in the GPU kernels' order: diag·x − Ge·x_e − Gw·x_w − Gn·x_n − Gs·x_s − gup − gdn."""
    def mv(v: np.ndarray) -> np.ndarray:
        x = v.reshape(c["diag"].shape)
        y = c["diag"] * x
        y[:, :, :-1] -= c["Ge"][:, :, :-1] * x[:, :, 1:]
        y[:, :, 1:] -= c["Gw"][:, :, 1:] * x[:, :, :-1]
        y[:, :-1, :] -= c["Gn"][:, :-1, :] * x[:, 1:, :]
        y[:, 1:, :] -= c["Gs"][:, 1:, :] * x[:, :-1, :]
        y[:-1] -= c["gup"][:-1] * x[1:]
        y[1:] -= c["gdn"][1:] * x[:-1]
        return y.ravel()
    return mv


def _exact_certificate(A: sp.csr_matrix, b: np.ndarray, x: np.ndarray, z: np.ndarray):
    """κ̂·‖b − A x‖∞ with every operation exact (80 digits) on the stored float64 data."""
    mpmath.mp.dps = 80
    mpf = mpmath.mpf
    indptr, idx, data = A.indptr, A.indices, A.data
    zz, xx, bb = [mpf(float(v)) for v in z], [mpf(float(v)) for v in x], [mpf(float(v)) for v in b]
    min_az, r_inf = None, mpf(0)
    for i in range(A.shape[0]):
        cols, vals = idx[indptr[i]:indptr[i + 1]], data[indptr[i]:indptr[i + 1]]
        az = mpmath.fsum(mpf(float(a)) * zz[j] for a, j in zip(vals, cols, strict=True))
        ax = mpmath.fsum(mpf(float(a)) * xx[j] for a, j in zip(vals, cols, strict=True))
        min_az = az if min_az is None or az < min_az else min_az
        r_inf = max(r_inf, abs(bb[i] - ax))
    assert min_az > 0, "exact min(Az) must be positive for Lemma 2(a)"
    return max(zz) / min_az * r_inf


# ------------------------------------------------------------------ the constants


@pytest.mark.parametrize("n", range(1, 9))
def test_gamma_bounds_n_roundings(n: int) -> None:
    g = mg.gamma(n)
    assert g == n * U / (1 - n * U)
    assert (1 + U) ** n - 1 <= g and 1 - (1 - U) ** n <= g


def test_certified_rounds_up() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        c, eps, delta = rng.uniform(1e-6, 1e3), rng.uniform(0, 1e-9), rng.uniform(0, 1e-6)
        m = mg.Margin(eps_total=eps, delta_abs=delta, min_az_threshold=0.0, x_inf_max=1.0)
        out = mg.certified(c, m)
        assert Fraction(out) >= Fraction(c) * (1 + Fraction(eps)) + Fraction(delta)


def _stats(**over):
    s = {"row_abs_sum_max": 0.4, "z_max": 2e4, "min_az": 1.0, "kappa": 2e4, "b_inf": 1e-2,
         "x_inf_max": 100.0}
    s.update(over)
    return s


def test_no_certificate_when_min_az_is_at_rounding_level() -> None:
    m = mg.margin(_stats())
    assert m is not None and m.eps_total > 0 and m.delta_abs > 0
    tiny = m.min_az_threshold / 2
    assert mg.margin(_stats(min_az=tiny, kappa=2e4 / tiny)) is None
    assert mg.margin(_stats(min_az=-1.0)) is None


def test_margin_grows_with_the_iterate_cap_and_kappa() -> None:
    a, b = mg.margin(_stats()), mg.margin(_stats(x_inf_max=1e4))
    assert b.delta_abs > a.delta_abs and b.eps_total == a.eps_total
    c = mg.margin(_stats(kappa=4e4))
    assert c.delta_abs > a.delta_abs


@pytest.mark.parametrize("key", ["row_abs_sum_max", "z_max", "b_inf", "x_inf_max", "kappa"])
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_bad_stats_raise(key: str, bad: float) -> None:
    with pytest.raises(ValueError, match=key):
        mg.margin(_stats(**{key: bad}))


def test_kappa_below_the_computed_quotient_is_refused() -> None:
    with pytest.raises(ValueError, match="kappa"):
        mg.margin(_stats(kappa=1e4))  # z_max / min_az = 2e4


def test_iterate_above_the_cap_is_refused() -> None:
    m = mg.margin(_stats())
    with pytest.raises(ValueError, match="x_inf"):
        mg.certified(1.0, m, x_inf=200.0)


def test_row_abs_sum_bound_is_an_upper_bound() -> None:
    A, _, _ = _board(0)
    s = mg.row_abs_sum_bound(A)
    mpmath.mp.dps = 50
    exact = max(mpmath.fsum(abs(mpmath.mpf(float(v))) for v in A.data[A.indptr[i]:A.indptr[i + 1]])
                for i in range(A.shape[0]))
    assert mpmath.mpf(s) >= exact


# ------------------------------------------------------------------ against 80-digit arithmetic


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("order", ["csr", "stencil"])
def test_certificate_is_at_least_the_exact_value(seed: int, order: str) -> None:
    A, b, c = _board(seed)
    pz = positive_z(A)
    assert pz is not None
    z = pz[0]
    theta = spla.spsolve(A.tocsc(), b)
    rng = np.random.default_rng(seed)
    rough = theta + 0.02 * theta.max() * rng.standard_normal(theta.size)
    matvec = None if order == "csr" else _stencil_matvec(c)
    for x, tight in ((rough, True), (theta, False)):  # theta: residual at rounding level
        bound = mg.certify_residual(A, b, x, z, matvec=matvec)
        assert bound is not None
        exact = _exact_certificate(A, b, x, z)
        assert mpmath.mpf(bound) >= exact, f"seed {seed} {order}: {bound} < {exact} (B7)"
        if tight:  # the margin is a rounding-level correction, not a loosening
            assert mpmath.mpf(bound) <= exact * (1 + mpmath.mpf("1e-8"))


def test_certificate_refuses_a_non_positive_z() -> None:
    A, b, _ = _board(1)
    z = np.ones(A.shape[0])
    z[7] = 0.0
    assert mg.certify_residual(A, b, np.zeros_like(b), z) is None
