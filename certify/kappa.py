"""Certificate constant κ̂ ≥ ‖A⁻¹‖∞ (Lemma 2(a), CLAUDE.md §8.4).

For a nonsingular M-matrix A and any z > 0 with Az > 0 componentwise,
    ‖A⁻¹‖∞ ≤ max_i z_i / min_i (Az)_i,
because A⁻¹ ≥ 0 and Az ≥ min(Az)·1 give A⁻¹1 ≤ z / min(Az). Any positive z is valid; z ≈ A⁻¹1
(a few multigrid V-cycles on the vector of ones) only makes the bound tight. On the 4-layer
board without a heat-sunk edge, plain V-cycles barely reduce the near-constant mode of A z = 1
(min(A z) < 0 in ~90k of 262k cells after 40 cycles), so accel="cg" runs n_vcycles iterations of
CG preconditioned by the same V-cycle instead; the positivity checks below are unchanged, so the
certificate is equally rigorous either way (docs/decisions.md, "kappa-hat z by AMG-CG"). The checks
min z > 0 and min Az > 0 are done in float64 (§8.4). The caller must already know that A is a
nonsingular M-matrix (solvers/mmatrix_test.py, invariant B1); nothing here re-proves it.

Rounding: κ̂ is certify.margin.kappa_upper (derivation Step 7a): z_max / (m̂(1 − ρ)) evaluated
exactly and rounded up, where m̂ = min fl(A z) and ρ bounds the rounding of A z relative to m̂. It
is a guaranteed upper bound on ‖A⁻¹‖∞ for the stored A, and None when m̂ is within rounding of
zero. It replaces the (1 + 1e-6) placeholder of CLAUDE.md §9.1.
"""

from __future__ import annotations

import numpy as np
import pyamg
import scipy.sparse as sp

from certify.margin import kappa_upper, row_abs_sum_bound

__all__ = ["kappa_from_z", "kappa_hat", "multigrid", "positive_z"]


def _check_square(A: sp.spmatrix) -> sp.csr_matrix:
    A = sp.csr_matrix(A, dtype=np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    return A


def multigrid(A: sp.spmatrix) -> pyamg.multilevel.MultilevelSolver:
    """Classical (Ruge–Stüben) AMG hierarchy for A; its V-cycle uses symmetric Gauss–Seidel."""
    return pyamg.ruge_stuben_solver(_check_square(A))


def positive_z(
    A: sp.spmatrix, n_vcycles: int = 3, ml: pyamg.multilevel.MultilevelSolver | None = None,
    accel: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(z, Az) with z ≈ A⁻¹1, or None if z or Az has an entry <= 0.

    accel None: n_vcycles stationary V-cycles (§8.4). accel "cg": n_vcycles iterations of CG
    preconditioned by one V-cycle (module docstring). Either way from x0 = 0."""
    A = _check_square(A)
    if n_vcycles < 1:
        raise ValueError(f"n_vcycles must be >= 1, got {n_vcycles}")
    if accel not in (None, "cg"):
        raise ValueError(f"accel must be None or 'cg', got {accel!r}")
    ml = multigrid(A) if ml is None else ml
    n = A.shape[0]
    if accel is None:  # tol = 0: run exactly n_vcycles cycles
        z = ml.solve(np.ones(n), x0=np.zeros(n), maxiter=n_vcycles, tol=0.0)
    else:  # a tiny tol, so CG's own stop does not end it first: n_vcycles iterations
        z = ml.solve(np.ones(n), x0=np.zeros(n), maxiter=n_vcycles, tol=1e-14, accel="cg")
    z = np.asarray(z, dtype=np.float64)
    Az = A @ z
    if not (np.all(np.isfinite(z)) and z.min() > 0 and Az.min() > 0):
        return None
    return z, Az


def kappa_from_z(A: sp.spmatrix, z: np.ndarray) -> float | None:
    """Guaranteed κ̂ ≥ ‖A⁻¹‖∞ from z (K/W), or None if z ≤ 0 somewhere or min(Az) is not
    certifiably positive (within rounding of zero)."""
    A = _check_square(A)
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (A.shape[0],):
        raise ValueError(f"z must have shape ({A.shape[0]},), got {z.shape}")
    if not np.all(np.isfinite(z)):
        return None
    Az = A @ z
    if z.min() <= 0 or Az.min() <= 0:
        return None
    return kappa_upper(row_abs_sum_bound(A), float(z.max()), float(Az.min()))


def kappa_hat(
    A: sp.spmatrix, n_vcycles: int = 3, ml: pyamg.multilevel.MultilevelSolver | None = None,
    accel: str | None = None,
) -> float | None:
    """κ̂ ≥ ‖A⁻¹‖∞ from z ≈ A⁻¹1 (positive_z), or None if no certificate exists.

    None means z or Az had a non-positive entry: do not certify, solve to tolerance instead.
    """
    pz = positive_z(A, n_vcycles, ml, accel)
    if pz is None:
        return None
    return kappa_from_z(A, pz[0])
