"""The derived FP64 margin (certify/margin.py, N4.1) wired into certify/kappa.py, stopping.py and
pointwise.py (N4.2 part; CLAUDE.md §8.4, invariant B7).

Before this change κ̂ carried a (1 + 1e-6) PLACEHOLDER, the stopping bound was the plain float
κ̂·max|b − A x|, the pointwise check compared floats, and the peak interval was rounded to nearest.
On a fully converged iterate the plain float certificate can come out below the exact value.
References here are 80-digit mpmath on the STORED float64 A, b, x, z, w (as tests/test_margin.py).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from certify import margin as mg
from certify.kappa import kappa_from_z, kappa_hat, positive_z
from certify.pointwise import gauss_seidel_smooth, peak_interval, pointwise_bound
from certify.stopping import jacobi, pcg_certified
from data.raster import rasterise
from data.sampler import sample_board
from solvers.fvm import assemble_board
from twinheat import spec as specmod

mpmath = pytest.importorskip("mpmath")

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 24  # 4 x 24² = 2,304 unknowns: small enough for 80-digit arithmetic
SEEDS = [0, 1, 2]


def _board(seed: int) -> tuple[sp.csr_matrix, np.ndarray]:
    rec = sample_board(seed, "id")
    A, b = assemble_board(SPEC, rasterise(rec, SPEC, N), rec["dirichlet_edges"])
    return sp.csr_matrix(A), np.asarray(b, dtype=np.float64)


def _exact_matvec(A: sp.csr_matrix, v: np.ndarray) -> list:
    """(A v)_i with every operation exact (80 digits) on the stored float64 values."""
    mpmath.mp.dps = 80
    mpf = mpmath.mpf
    vv = [mpf(float(t)) for t in v]
    out = []
    for i in range(A.shape[0]):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        out.append(mpmath.fsum(mpf(float(a)) * vv[j]
                               for a, j in zip(A.data[lo:hi], A.indices[lo:hi], strict=True)))
    return out


def _exact_residual(A: sp.csr_matrix, b: np.ndarray, x: np.ndarray) -> list:
    return [mpmath.mpf(float(bi)) - ax for bi, ax in zip(b, _exact_matvec(A, x), strict=True)]


def _exact_kappa(A: sp.csr_matrix, z: np.ndarray):
    """max z / min(A z), exact: an upper bound on ‖A⁻¹‖∞ for this z (Lemma 2(a))."""
    m = min(_exact_matvec(A, z))
    assert m > 0
    return mpmath.mpf(float(z.max())) / m


# ------------------------------------------------------------------ κ̂ (kappa.py)

@pytest.mark.parametrize("seed", SEEDS)
def test_kappa_is_at_least_the_exact_quotient(seed: int) -> None:
    A, _ = _board(seed)
    z = positive_z(A)[0]
    k = kappa_from_z(A, z)
    assert k is not None and mpmath.mpf(k) >= _exact_kappa(A, z)


def test_kappa_carries_no_placeholder_inflation() -> None:
    A, _ = _board(0)
    z = positive_z(A)[0]
    k = kappa_from_z(A, z)
    q = float(z.max()) / float((A @ z).min())
    assert 0 <= k / q - 1 < 1e-10  # the derived factor, not the former 1e-6


def test_no_kappa_when_min_az_is_within_rounding_of_zero() -> None:
    """Row margins of 1e-15 against entries of 1: min(A z) > 0 in floats, but not certifiable."""
    a = 1.0 - 1e-15
    A = sp.csr_matrix(np.array([[1.0, -a], [-a, 1.0]]))
    z = np.ones(2)
    assert float((A @ z).min()) > 0
    assert kappa_from_z(A, z) is None


def test_kappa_upper_and_residual_bound_refuse_bad_input() -> None:
    with pytest.raises(ValueError, match="z_max"):
        mg.kappa_upper(1.0, float("nan"), 1.0)
    with pytest.raises(ValueError, match="r_inf"):
        mg.residual_bound(1.0, -1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="kappa"):
        mg.residual_bound(0.0, 1.0, 1.0, 1.0, 1.0)


def test_residual_bound_is_at_least_kappa_times_the_float_residual() -> None:
    for k, r in [(3.0, 0.1), (1e3, 1e-20), (0.7, 0.0)]:
        got = mg.residual_bound(k, r, 2.0, 1.0, 50.0)
        assert Fraction(got) >= Fraction(k) * Fraction(r)
        assert got > 0  # the rounding term stays even for a zero computed residual


# ------------------------------------------------------------------ PCG bound (stopping.py)

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("start", ["converged", "rough"])
def test_every_pcg_bound_is_at_least_the_exact_value(seed: int, start: str) -> None:
    A, b = _board(seed)
    z = positive_z(A)[0]
    k = kappa_from_z(A, z)
    x0 = spla.spsolve(sp.csc_matrix(A), b)
    if start == "rough":
        x0 = x0 * (1 + 0.02 * np.sin(np.arange(x0.size)))
    seen: list[tuple[np.ndarray, float]] = []
    pcg_certified(A, b, x0, jacobi(A), k, tau_v=1e3, maxit=3,
                  history=lambda it, x, c: seen.append((x, c)))
    kz = _exact_kappa(A, z)
    for it, (x, c) in enumerate(seen):
        ref = kz * max(abs(t) for t in _exact_residual(A, b, x))
        assert mpmath.mpf(c) >= ref, (seed, start, it, c, float(ref))


# ------------------------------------------------------------------ pointwise (pointwise.py)

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("start", ["converged", "smoothed"])
def test_pointwise_w_covers_the_exact_residual(seed: int, start: str) -> None:
    A, b = _board(seed)
    x = spla.spsolve(sp.csc_matrix(A), b)
    if start == "smoothed":
        x = gauss_seidel_smooth(A, b, x * (1 + 0.03 * np.cos(np.arange(x.size))), 5)
    w = pointwise_bound(A, b, x)
    assert w.min() >= 0
    aw, r = _exact_matvec(A, w), _exact_residual(A, b, x)
    bad = [i for i in range(len(r)) if not aw[i] >= abs(r[i])]
    assert not bad, f"seed {seed} {start}: A w < |r| exactly at {bad[:5]}"


def test_pointwise_bound_raises_for_a_non_m_matrix() -> None:
    A, b = _board(0)
    with pytest.raises(ValueError, match="M-matrix"):
        pointwise_bound(-A, b, np.zeros_like(b))


def test_peak_interval_rounds_outward() -> None:
    """x ± w rounded to nearest would give 1.0 on both sides; the exact ends are 1 ± 2⁻⁶⁰."""
    lo, hi = peak_interval(np.array([1.0, 0.5]), np.array([2.0**-60, 0.0]))
    assert Fraction(hi) >= 1 + Fraction(1, 2**60)
    assert Fraction(lo) <= 1 - Fraction(1, 2**60)


def test_kappa_hat_still_returns_a_float() -> None:
    A, _ = _board(1)
    k = kappa_hat(A)
    assert isinstance(k, float) and k > 0
