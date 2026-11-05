"""Pointwise certificate (Lemma 2(c), CLAUDE.md §8.4).

If w ≥ 0 and Aw ≥ |r| componentwise (r = b − A x̃, A a nonsingular M-matrix), then
|θ − x̃| ≤ w in every cell and max(x̃ − w) ≤ T_peak ≤ max(x̃ + w).

Construction (§8.4: "approximate A w = |r| solve (4 V-cycles) + two-SpMV fix-up, validity
asserted"):
  1. w₀ = max(V-cycles(|r|), 0)                    approximate A⁻¹|r|, made non-negative
  2. d  = A w₀ − |r|                                SpMV 1: where d < 0, w₀ does not cover r
  3. w  = w₀ + t z,  t = max(0, max_i(−d_i / (Az)_i))·(1 + 1e-6)
     with z > 0, Az > 0 from certify/kappa.positive_z (SpMV 2 is A z), so Aw = A w₀ + t Az ≥ |r|
  4. assert w ≥ 0 and A w ≥ |r| with a fresh float64 SpMV; raise otherwise.
bench/prelim/cert.js does the same with two differences: it does not clip w₀ at 0, and it
uses one scalar t = max_i(−d_i) / min(Az) instead of the per-cell ratio, which is never smaller
than ours (so ours is at least as tight). It also takes z from the same number of V-cycles as
w₀. Validity does not depend on these choices: step 4 checks the Lemma 2(c) hypothesis
directly. The 1e-6 factor and the float64 check are PLACEHOLDERS for the derived rounding
margin (certify/margin.py, N4.1).
"""

from __future__ import annotations

import numpy as np
import pyamg
import scipy.sparse as sp
from pyamg.relaxation.relaxation import gauss_seidel

from certify.kappa import multigrid, positive_z

__all__ = ["gauss_seidel_smooth", "peak_interval", "pointwise_bound"]

FIXUP_INFLATE = 1e-6  # PLACEHOLDER (see module docstring)


def gauss_seidel_smooth(
    A: sp.spmatrix, b: np.ndarray, x: np.ndarray, sweeps: int, sweep: str = "forward",
    colours: np.ndarray | None = None,
) -> np.ndarray:
    """x after `sweeps` Gauss–Seidel sweeps on A x = b (a copy; sweeps = 0 returns x unchanged).

    sweep: "forward" | "backward" | "symmetric" (lexicographic, pyamg), or "red_black": each
    sweep updates every colour-0 unknown, then every colour-1 unknown, from the latest values.
    That is §8.6's smoother (bench/prelim/cert.js, core.js rbSweep with ω = 1, colour (x + y)
    mod 2 on the plate). red_black needs `colours` (0/1 per unknown) with no coupling between
    unknowns of the same colour; this is checked.
    """
    if sweeps < 0:
        raise ValueError(f"sweeps must be >= 0, got {sweeps}")
    A = sp.csr_matrix(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.array(x, dtype=np.float64)
    if sweep != "red_black":
        if colours is not None:
            raise ValueError(f"colours only applies to sweep='red_black', not {sweep!r}")
        if sweeps:
            gauss_seidel(A, out, b, iterations=sweeps, sweep=sweep)
        return out
    if colours is None:
        raise ValueError("sweep='red_black' needs colours (0/1 per unknown)")
    colours = np.asarray(colours)
    if colours.shape != (A.shape[0],) or not np.isin(colours, (0, 1)).all():
        raise ValueError(f"colours must be 0/1 of length {A.shape[0]}")
    C = A.tocoo()
    off = C.row != C.col
    if np.any(colours[C.row[off]] == colours[C.col[off]]):
        raise ValueError("colours: two coupled unknowns share a colour (not a red-black "
                         "colouring of A)")
    d = A.diagonal()
    R = A - sp.diags(d)
    masks = (colours == 0, colours == 1)
    for _ in range(sweeps):
        for m in masks:
            out[m] = ((b - R @ out) / d)[m]
    return out


def pointwise_bound(
    A: sp.spmatrix,
    r: np.ndarray,
    n_vcycles: int = 4,
    ml: pyamg.multilevel.MultilevelSolver | None = None,
    z_and_Az: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """w ≥ 0 with A w ≥ |r| (K per cell), validity asserted; raises ValueError if impossible.

    r is the residual b − A x̃ in W; A must be a nonsingular M-matrix. ml and z_and_Az may be
    passed in to reuse the hierarchy and the κ̂ vector z ≈ A⁻¹1 of the same A.
    """
    A = sp.csr_matrix(A, dtype=np.float64)
    if not A.diagonal().min() > 0:
        raise ValueError("pointwise_bound needs a nonsingular M-matrix; A has a non-positive "
                         "diagonal entry")
    abs_r = np.abs(np.asarray(r, dtype=np.float64))
    if abs_r.shape != (A.shape[0],) or not np.all(np.isfinite(abs_r)):
        raise ValueError(f"r must be a finite vector of length {A.shape[0]}")
    ml = multigrid(A) if ml is None else ml
    pz = positive_z(A, ml=ml) if z_and_Az is None else z_and_Az
    if pz is None:
        raise ValueError("no z > 0 with Az > 0 found: A is not a certified M-matrix here")
    z, Az = pz
    n = A.shape[0]
    w0 = np.maximum(ml.solve(abs_r, x0=np.zeros(n), maxiter=n_vcycles, tol=0.0), 0.0)
    d = A @ w0 - abs_r
    t = max(0.0, float(np.max(-d / Az))) * (1 + FIXUP_INFLATE)
    w = w0 + t * z
    slack = A @ w - abs_r
    if not (w.min() >= 0 and slack.min() >= 0):
        raise ValueError(f"pointwise certificate invalid: min w {w.min():.3e}, "
                         f"min (Aw − |r|) {slack.min():.3e}")
    return w


def peak_interval(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Certified interval [max(x − w), max(x + w)] for the true peak (K)."""
    x = np.asarray(x, dtype=np.float64)
    return float((x - w).max()), float((x + w).max())
