"""Certificate constant κ̂ ≥ ‖A⁻¹‖∞ (Lemma 2(a), CLAUDE.md §8.4).

For a nonsingular M-matrix A and any z > 0 with Az > 0 componentwise,
    ‖A⁻¹‖∞ ≤ max_i z_i / min_i (Az)_i,
because A⁻¹ ≥ 0 and Az ≥ min(Az)·1 give A⁻¹1 ≤ z / min(Az). Any positive z is valid; z ≈ A⁻¹1
(a few multigrid V-cycles on the vector of ones) only makes the bound tight. The checks
min z > 0 and min Az > 0 are done in float64 (§8.4). The caller must already know that A is a
nonsingular M-matrix (solvers/mmatrix_test.py, invariant B1); nothing here re-proves it.

PLACEHOLDER margin: the result is multiplied by (1 + 1e-6) as in CLAUDE.md §9.1. The derived
rounding margin (certify/margin.py, prompt N4.1) replaces it; until then κ̂ is not a
rounding-safe bound.
"""

from __future__ import annotations

import numpy as np
import pyamg
import scipy.sparse as sp

__all__ = ["PLACEHOLDER_MARGIN", "kappa_from_z", "kappa_hat", "multigrid", "positive_z"]

PLACEHOLDER_MARGIN = 1e-6  # CLAUDE.md §9.1; to be replaced by certify/margin.py


def _check_square(A: sp.spmatrix) -> sp.csr_matrix:
    A = sp.csr_matrix(A, dtype=np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    return A


def multigrid(A: sp.spmatrix) -> pyamg.multilevel.MultilevelSolver:
    """Classical (Ruge–Stüben) AMG hierarchy for A; its V-cycle uses symmetric Gauss–Seidel."""
    return pyamg.ruge_stuben_solver(_check_square(A))


def positive_z(
    A: sp.spmatrix, n_vcycles: int = 3, ml: pyamg.multilevel.MultilevelSolver | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """(z, Az) with z ≈ A⁻¹1 from n_vcycles V-cycles, or None if z or Az has an entry <= 0."""
    A = _check_square(A)
    if n_vcycles < 1:
        raise ValueError(f"n_vcycles must be >= 1, got {n_vcycles}")
    ml = multigrid(A) if ml is None else ml
    n = A.shape[0]
    # tol = 0: run exactly n_vcycles cycles from x0 = 0
    z = np.asarray(ml.solve(np.ones(n), x0=np.zeros(n), maxiter=n_vcycles, tol=0.0),
                   dtype=np.float64)
    Az = A @ z
    if not (np.all(np.isfinite(z)) and z.min() > 0 and Az.min() > 0):
        return None
    return z, Az


def kappa_from_z(A: sp.spmatrix, z: np.ndarray) -> float | None:
    """max(z) / min(Az) · (1 + placeholder margin), or None if z or Az has an entry <= 0."""
    A = _check_square(A)
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (A.shape[0],):
        raise ValueError(f"z must have shape ({A.shape[0]},), got {z.shape}")
    if not np.all(np.isfinite(z)):
        return None
    Az = A @ z
    if z.min() <= 0 or Az.min() <= 0:
        return None
    return float(z.max() / Az.min() * (1 + PLACEHOLDER_MARGIN))


def kappa_hat(
    A: sp.spmatrix, n_vcycles: int = 3, ml: pyamg.multilevel.MultilevelSolver | None = None
) -> float | None:
    """κ̂ ≥ ‖A⁻¹‖∞ from z ≈ A⁻¹1 (n_vcycles V-cycles), or None if no certificate exists.

    None means z or Az had a non-positive entry: do not certify, solve to tolerance instead.
    """
    pz = positive_z(A, n_vcycles, ml)
    if pz is None:
        return None
    return kappa_from_z(A, pz[0])
