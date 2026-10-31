"""PyAMG baseline: Ruge–Stüben AMG preconditioning a Krylov method (N1.5; CLAUDE.md §7.3, §11.5).

amg_solve(A, b) builds the classical Ruge–Stüben hierarchy for A (pyamg.ruge_stuben_solver) and
runs GMRES (default) or CG preconditioned by one V-cycle per iteration, until the TRUE relative
residual ‖b − Aθ‖₂ / ‖b‖₂ ≤ tol.

Why a loop: PyAMG's Krylov accelerators stop on their own residual norm (for GMRES the
preconditioned one), which on the 4-layer board sits about 10x above the true residual: tol =
1e-8 returns a true 1.3e-7. A second call from the same iterate stops at once, because the stop is
relative to ‖b‖ in that norm. So amg_solve re-solves from the current iterate with the inner
tolerance scaled by tol / (true relres), until the true residual meets tol; no progress, or more
than maxiter iterations in total, raises AMGNotConverged.

`seconds` = setup + solve: the cost of one oracle label, since every board has its own A and so
its own hierarchy. Timing is wall-clock (time.perf_counter); the caller fixes the thread count
(one core per process for oracle labels, N3.2).

strength_theta: None keeps PyAMG's default classical strength threshold (0.25 in pyamg 5.3).
Smaller values keep weaker couplings in the coarsening; on the layered board θ = 0.1 cut
iterations about 5x in a one-board trial (N1.5). Which value the oracle uses is N's decision, not
fixed here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pyamg
import scipy.sparse as sp

__all__ = ["AMGNotConverged", "AMGResult", "amg_solve"]


class AMGNotConverged(RuntimeError):
    """The true relative residual did not reach tol."""


@dataclass(frozen=True)
class AMGResult:
    theta: np.ndarray  # solution, same index order as b (K for the board)
    iters: int  # Krylov iterations over all inner calls
    seconds: float  # setup + solve wall-clock, s
    setup_seconds: float
    solve_seconds: float
    relres: float  # true ‖b − Aθ‖₂ / ‖b‖₂
    levels: int  # levels in the AMG hierarchy
    inner_calls: int


def amg_solve(
    A: sp.spmatrix, b: np.ndarray, tol: float = 1e-8, accel: str = "gmres",
    strength_theta: float | None = None, x0: np.ndarray | None = None, maxiter: int = 1000,
) -> AMGResult:
    """Solve A θ = b with PyAMG Ruge–Stüben + `accel` to a true relative residual <= tol."""
    A = sp.csr_matrix(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError(f"A must be square and b of length {A.shape[0]}: A {A.shape}, b {b.shape}")
    if not np.all(np.isfinite(b)):
        raise ValueError("b has non-finite entries")
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0:
        raise ValueError("b is zero: the solution is 0 and a relative tolerance is undefined")
    if not (np.isfinite(tol) and 0 < tol < 1):
        raise ValueError(f"tol must be in (0, 1), got {tol}")
    if accel not in ("gmres", "cg"):
        raise ValueError(f"accel must be 'gmres' or 'cg', got {accel!r}")
    kw = {} if strength_theta is None else {"strength": ("classical", {"theta": strength_theta})}

    t0 = time.perf_counter()
    ml = pyamg.ruge_stuben_solver(A, **kw)
    t1 = time.perf_counter()
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64)
    if x.shape != (n,):
        raise ValueError(f"x0 must have length {n}, got {x.shape}")
    iters, calls, inner_tol = 0, 0, tol
    rel = float(np.linalg.norm(b - A @ x)) / bnorm
    while rel > tol:
        if iters >= maxiter:
            raise AMGNotConverged(f"maxiter {maxiter} reached: true relres {rel:.3e} > tol {tol}")
        res: list[float] = []
        x = ml.solve(b, x0=x, tol=inner_tol, accel=accel, maxiter=maxiter - iters,
                     residuals=res)
        calls += 1
        step = max(len(res) - 1, 0)
        iters += step
        new = float(np.linalg.norm(b - A @ x)) / bnorm
        if step == 0 and new >= rel:
            if inner_tol < np.finfo(float).eps:
                raise AMGNotConverged(f"no progress: true relres {new:.3e} > tol {tol} after "
                                      f"{iters} iterations")
        if new > tol:  # scale the inner tolerance by how far the true residual is from tol
            inner_tol *= tol / new
        rel = new
    t2 = time.perf_counter()
    return AMGResult(theta=x, iters=iters, seconds=t2 - t0, setup_seconds=t1 - t0,
                     solve_seconds=t2 - t1, relres=rel, levels=len(ml.levels), inner_calls=calls)
