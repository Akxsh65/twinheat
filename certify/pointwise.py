"""Pointwise certificate (Lemma 2(c), CLAUDE.md §8.4).

If w ≥ 0 and Aw ≥ |r| componentwise (r = b − A x̃, A a nonsingular M-matrix), then
|θ − x̃| ≤ w in every cell and max(x̃ − w) ≤ T_peak ≤ max(x̃ + w).

Construction (§8.4: "approximate A w = |r| solve (4 V-cycles) + two-SpMV fix-up, validity
asserted"):
  0. r̂ = fl(b − A x̃); τ_r bounds |r̂_i − r_i| (certify/margin.py, derivation Step 4)
  1. w₀ = max(V-cycles(|r̂|), 0)                    approximate A⁻¹|r̂|, made non-negative
  2. d  = A w₀ − (|r̂| + h)                          SpMV 1: where d < 0, w₀ does not cover r
  3. w  = w₀ + t z,  t = max(0, max_i(−d_i / (Az)_i))
     with z > 0, Az > 0 from certify/kappa.positive_z (SpMV 2 is A z); h is a headroom of a few
     rounding bounds, so that step 4 passes
  4. check w ≥ 0 and fl(A w) ≥ up(|r̂| + τ_w + τ_r) with a fresh float64 SpMV, τ_w the rounding
     bound of that SpMV (Step 7c): this implies A w ≥ |r| exactly. If it fails, h is doubled
     (FIXUP_RETRIES attempts in all), then ValueError.
bench/prelim/cert.js does the same with two differences: it does not clip w₀ at 0, and it
uses one scalar t = max_i(−d_i) / min(Az) instead of the per-cell ratio, which is never smaller
than ours (so ours is at least as tight). It also takes z from the same number of V-cycles as
w₀. Validity does not depend on these choices: step 4 checks the Lemma 2(c) hypothesis
directly, with the derived rounding margin (no placeholder factor). peak_interval rounds both
ends outward (Step 7d).
"""

from __future__ import annotations

import numpy as np
import pyamg
import scipy.sparse as sp
from pyamg.relaxation.relaxation import gauss_seidel

from certify.kappa import multigrid, positive_z
from certify.margin import matvec_tau, residual_tau, row_abs_sum_bound

__all__ = ["gauss_seidel_smooth", "peak_interval", "pointwise_bound"]

FIXUP_RETRIES = 4  # attempts of steps 2-4, doubling the headroom each time (module docstring)


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
    b: np.ndarray,
    x: np.ndarray,
    n_vcycles: int = 4,
    ml: pyamg.multilevel.MultilevelSolver | None = None,
    z_and_Az: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """w ≥ 0 with A w ≥ |b − A x| exactly (K per cell), validity asserted; ValueError otherwise.

    b (W) and the approximation x (K) define the residual, recomputed here in float64 so that its
    rounding is bounded (Step 4). A must be a nonsingular M-matrix. ml and z_and_Az may be passed
    in to reuse the hierarchy and the κ̂ vector z ≈ A⁻¹1 of the same A.
    """
    A = sp.csr_matrix(A, dtype=np.float64)
    if not A.diagonal().min() > 0:
        raise ValueError("pointwise_bound needs a nonsingular M-matrix; A has a non-positive "
                         "diagonal entry")
    b, x = np.asarray(b, dtype=np.float64), np.asarray(x, dtype=np.float64)
    for name, v in (("b", b), ("x", x)):
        if v.shape != (A.shape[0],) or not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must be a finite vector of length {A.shape[0]}")
    ml = multigrid(A) if ml is None else ml
    pz = positive_z(A, ml=ml) if z_and_Az is None else z_and_Az
    if pz is None:
        raise ValueError("no z > 0 with Az > 0 found: A is not a certified M-matrix here")
    z, Az = pz
    n = A.shape[0]
    S = row_abs_sum_bound(A)
    abs_r = np.abs(b - A @ x)
    tau_r = residual_tau(S, float(np.abs(b).max()), float(np.abs(x).max()))
    w0 = np.maximum(ml.solve(abs_r, x0=np.zeros(n), maxiter=n_vcycles, tol=0.0), 0.0)
    Aw0 = A @ w0

    def tau_total(v: np.ndarray) -> float:  # τ_r + τ_w for ‖w‖∞ = max v, rounded up
        return float(np.nextafter(tau_r + matvec_tau(S, float(v.max())), np.inf))

    w, need = w0, abs_r
    for attempt in range(FIXUP_RETRIES):
        head = tau_total(w) * 2.0 ** (attempt + 2)  # headroom for the rounding of steps 2-3
        t = max(0.0, float(np.max((abs_r + head - Aw0) / Az)))
        w = w0 + t * z
        need = np.nextafter(abs_r + tau_total(w), np.inf)  # up(|r̂| + τ_r + τ_w), per cell
        if w.min() >= 0 and np.all(A @ w >= need):
            return w
    slack = A @ w - need
    raise ValueError(f"pointwise certificate invalid after {FIXUP_RETRIES} attempts: min w "
                     f"{w.min():.3e}, min (fl(Aw) − up(|r̂| + τ)) {slack.min():.3e}")


def peak_interval(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Certified interval [max(x − w), max(x + w)] for the true peak (K), rounded outward."""
    x, w = np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)
    lo = float(np.nextafter(x - w, -np.inf).max())
    hi = float(np.nextafter(x + w, np.inf).max())
    return lo, hi
