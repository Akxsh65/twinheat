"""Warm-started PCG, single and batched, with preconditioners (N1.4; CLAUDE.md §9.1, §18.5).

Array-module generic: CuPy arrays on the GPU, NumPy arrays on the CPU (CI tests the logic there).
Fields are [layer, y, x] for pcg and [B, layer, y, x] for batched_pcg (rule 7); θ in K, b in W,
operators in W/K. `Aop(x)` returns A x with the shape of x; stencil_operator() builds one from
the Numba kernels in solvers/cuda_kernels.py.

Semantics follow the §9.1 reference (relative stop ‖r‖ ≤ tol·‖b‖, x0 = warm start) with these
fixes:
  * The residual is checked before the first iteration, so an exact warm start returns after 0
    iterations (the reference divides 0 by 0 on its first step).
  * The recursive residual r ← r − αAp only triggers the stop. Before returning, the true
    residual b − A x is recomputed, with `Aop_check` (e.g. the FP64 operator) when given. If it
    misses tol, PCG restarts from the true residual; if the true residual stops falling between
    restarts it raises PCGNotConverged ("stalled"), because tol is below what the iteration
    precision can reach (§18.5: the FP32 recursive residual keeps falling after the true one
    has stalled).
  * Not converging within maxit raises PCGNotConverged carrying the iterate. A non-positive
    pᵀAp or rᵀM⁻¹r raises PCGBreakdown: A or M⁻¹ is not SPD (§18.5, D10).
Dot products accumulate in float64 whatever the iteration dtype.

batched_pcg keeps one α, β per board and a convergence mask: a converged board gets α = 0, so its
x and r stop changing while the others continue (no compaction; Aop_b still runs on every board).
pcg is batched_pcg with B = 1, so the two use the same arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import cupy as _cp
except ImportError:  # CPU-only machines (CI)
    _cp = None

Array = Any  # numpy.ndarray or cupy.ndarray
Op = Callable[[Array], Array]


class PCGNotConverged(RuntimeError):
    """PCG stopped before ‖b − A x‖ ≤ tol·‖b‖. Carries the iterate, iterations and the true
    relative residual reached (per board for batched_pcg)."""

    def __init__(self, message: str, x: Array, iters, relres):
        super().__init__(message)
        self.x, self.iters, self.relres = x, iters, relres


class PCGBreakdown(RuntimeError):
    """pᵀAp ≤ 0 or rᵀM⁻¹r ≤ 0 (or non-finite): A or M⁻¹ is not symmetric positive definite."""


def _xp(a: Array):
    return _cp.get_array_module(a) if _cp is not None else np


def _host(a: Array) -> np.ndarray:
    return _cp.asnumpy(a) if _cp is not None and isinstance(a, _cp.ndarray) else np.asarray(a)


def _bdot(xp, a: Array, b: Array) -> Array:
    """Per-board dot products [B], accumulated in float64."""
    return xp.sum((a * b).reshape(a.shape[0], -1), axis=1, dtype=xp.float64)


def _check_args(b: Array, x0: Array, tol: float, maxit: int, what: str) -> None:
    if x0.shape != b.shape:
        raise ValueError(f"{what}: x0 shape {x0.shape} != b shape {b.shape}")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(f"{what}: tol must be finite and > 0, got {tol}")
    if not (isinstance(maxit, int | np.integer) and maxit >= 1):
        raise ValueError(f"{what}: maxit must be an int >= 1, got {maxit!r}")
    xp = _xp(b)
    for name, a in (("b", b), ("x0", x0)):
        if not bool(xp.all(xp.isfinite(a))):
            raise ValueError(f"{what}: {name} has non-finite entries")


def batched_pcg(
    Aop_b: Op, b: Array, x0: Array, Minv: Op, tol: float = 1e-8, maxit: int = 20000,
    Aop_check: Op | None = None,
) -> tuple[Array, np.ndarray]:
    """Solve A_k x_k = b_k for every board k of b [B, ...]; returns (x [B, ...], iters [B]).

    Aop_b, Minv and Aop_check act on [B, ...] arrays. Board k stops when its recursive residual
    and then its true residual (Aop_check, or Aop_b) satisfy ‖r_k‖ ≤ tol·‖b_k‖. iters[k] counts
    the A applications made while board k was active (0 for a converged warm start).
    """
    xp = _xp(b)
    _check_args(b, x0, tol, maxit, "batched_pcg")
    if b.ndim < 2:
        raise ValueError(f"batched_pcg: b must be [B, ...], got shape {b.shape}")
    B = b.shape[0]
    dtype = b.dtype
    col = (B,) + (1,) * (b.ndim - 1)  # broadcast per-board scalars over a board's field

    bnorm = _host(xp.sqrt(_bdot(xp, b, b)))
    if np.any(bnorm == 0):
        raise ValueError(f"batched_pcg: b is zero for boards {np.flatnonzero(bnorm == 0).tolist()} "
                         "(the solution is 0; a relative tolerance is undefined)")
    target = tol * bnorm
    b_check = b if Aop_check is None else b.astype(xp.float64)
    check = Aop_b if Aop_check is None else Aop_check

    x = x0.astype(dtype, copy=True)
    r = b - Aop_b(x)
    iters = np.zeros(B, dtype=np.int64)
    done = np.zeros(B, dtype=bool)
    last_true = np.full(B, np.inf)  # true residual norm at the previous failed check
    z = Minv(r)
    rz = _bdot(xp, r, z)
    p = z.copy()
    it = 0

    while True:
        active = ~done
        conv = active & (_host(xp.sqrt(_bdot(xp, r, r))) <= target)
        if conv.any():  # the recursive residual says stop: confirm on the true residual
            rt = b_check - check(x.astype(b_check.dtype, copy=False))
            tn = _host(xp.sqrt(_bdot(xp, rt, rt)))
            ok = conv & (tn <= target)
            restart = conv & ~ok
            stalled = restart & (tn >= last_true)
            if stalled.any():
                k = np.flatnonzero(stalled).tolist()
                raise PCGNotConverged(
                    f"batched_pcg: true residual stalled at relres {(tn / bnorm)[k].tolist()} "
                    f"> tol {tol} for boards {k} after {iters[k].tolist()} iterations; tol is "
                    f"below what {dtype} iterations can reach (CLAUDE.md §18.5)",
                    x, iters, tn / bnorm)
            done |= ok
            if restart.any():  # continue from the true residual with a fresh direction
                last_true[restart] = tn[restart]
                m = xp.asarray(restart)
                r[m] = rt[m].astype(dtype)
                z = Minv(r)
                rz_new = _bdot(xp, r, z)
                p[m] = z[m]
                rz[m] = rz_new[m]
            active = ~done
        if not active.any():
            return x, iters
        if it >= maxit:
            k = np.flatnonzero(active).tolist()
            rel = _host(xp.sqrt(_bdot(xp, r, r))) / bnorm
            raise PCGNotConverged(
                f"batched_pcg: maxit {maxit} reached with boards {k} unconverged "
                f"(recursive relres {rel[k].tolist()}, tol {tol})", x, iters, rel)

        a_dev = xp.asarray(active)
        Ap = Aop_b(p)
        pAp = _bdot(xp, p, Ap)
        _check_positive(pAp, active, "pᵀAp", "A", it)
        alpha = xp.where(a_dev, rz / xp.where(a_dev, pAp, 1.0), 0.0)
        alpha = alpha.astype(dtype).reshape(col)
        x += alpha * p
        r -= alpha * Ap
        iters[active] += 1
        it += 1

        z = Minv(r)
        rz_new = _bdot(xp, r, z)
        still = active & (_host(xp.sqrt(_bdot(xp, r, r))) > target)
        _check_positive(rz_new, still, "rᵀM⁻¹r", "M⁻¹", it)
        beta = xp.where(a_dev, rz_new / xp.where(a_dev, rz, 1.0), 0.0)
        p = z + beta.astype(dtype).reshape(col) * p
        rz = rz_new


def _check_positive(v: Array, mask: np.ndarray, name: str, who: str, it: int) -> None:
    h = _host(v)
    bad = mask & ~(np.isfinite(h) & (h > 0))
    if bad.any():
        k = np.flatnonzero(bad).tolist()
        raise PCGBreakdown(f"{name} = {h[k].tolist()} at iteration {it} for boards {k}: "
                           f"{who} is not symmetric positive definite")


def pcg(
    Aop: Op, b: Array, x0: Array, Minv: Op, tol: float = 1e-8, maxit: int = 20000,
    Aop_check: Op | None = None,
) -> tuple[Array, int]:
    """Solve A x = b for one board, warm-started from x0; returns (x, iterations).

    Stops when ‖b − A x‖ ≤ tol·‖b‖ (2-norm), confirmed on the true residual (Aop_check if
    given). Raises PCGNotConverged or PCGBreakdown; see the module docstring.
    """
    _check_args(b, x0, tol, maxit, "pcg")

    def lift(op: Op | None) -> Op | None:
        return None if op is None else (lambda xb: op(xb[0])[None])

    try:
        x, iters = batched_pcg(lift(Aop), b[None], x0[None], lift(Minv), tol, maxit,
                               lift(Aop_check))
    except PCGNotConverged as e:
        raise PCGNotConverged(str(e).replace("batched_pcg", "pcg"), e.x[0], int(e.iters[0]),
                              float(e.relres[0])) from None
    return x[0], int(iters[0])


# ------------------------------------------------------------------------------ preconditioners

def jacobi(diag: Array) -> Op:
    """M⁻¹ r = r / diag A. diag [.., L, N, N] (W/K) > 0; its dtype sets the output dtype."""
    xp = _xp(diag)
    if not bool(xp.all(diag > 0)):
        raise ValueError("jacobi: diag must be > 0 everywhere (an M-matrix has a positive "
                         "diagonal)")
    inv = 1.0 / diag
    return lambda r: r * inv


def line_z_block_jacobi(diag: Array, gup: Array) -> Op:
    """M⁻¹ = exact inverse of each (y, x) column's L×L tridiagonal block (Thomas algorithm).

    diag, gup [.., L, N, N] from solvers/fvm.py coefficients() (gup[l] couples l and l + 1, so
    the block is tridiag(−gup[l−1], diag[l], −gup[l])). diag also holds the cell's in-plane
    couplings (> 0 for N > 1), so each block is strictly diagonally dominant: Thomas needs no
    pivoting (positive pivots are still asserted), and as the block-diagonal part of an SPD A it
    is SPD. Factors are computed in float64 and stored in diag's dtype.
    """
    xp = _xp(diag)
    if diag.shape != gup.shape or diag.ndim < 3:
        raise ValueError(f"line_z_block_jacobi: diag {diag.shape} and gup {gup.shape} must be "
                         "equal shapes [.., L, N, N]")
    if bool(xp.any(gup[..., -1, :, :] != 0)):
        raise ValueError("line_z_block_jacobi: gup on the last layer must be 0 (no layer below)")
    if not (bool(xp.all(diag > 0)) and bool(xp.all(gup >= 0))):
        raise ValueError("line_z_block_jacobi: need diag > 0 and gup >= 0 (M-matrix couplings)")
    L = diag.shape[-3]
    d = diag.astype(xp.float64)
    g = gup.astype(xp.float64)
    cprime = xp.empty_like(d)  # c'_l = −gup[l] / m_l
    minv = xp.empty_like(d)  # 1 / m_l, m_l = diag[l] − gup[l−1]² / m_{l−1}
    m = d[..., 0, :, :]
    for layer in range(L):  # loop over the 4 layers, vectorised over every column
        if layer > 0:
            m = d[..., layer, :, :] + g[..., layer - 1, :, :] * cprime[..., layer - 1, :, :]
        if not bool(xp.all(m > 0)):
            raise ValueError(f"line_z_block_jacobi: non-positive pivot on layer {layer}")
        minv[..., layer, :, :] = 1.0 / m
        cprime[..., layer, :, :] = -g[..., layer, :, :] / m
    dtype = diag.dtype
    cprime, minv, gl = cprime.astype(dtype), minv.astype(dtype), gup.astype(dtype)

    def Minv(r: Array) -> Array:
        y = xp.empty_like(r)
        y[..., 0, :, :] = r[..., 0, :, :] * minv[..., 0, :, :]
        for layer in range(1, L):  # forward: d'_l = (r_l + gup[l−1] d'_{l−1}) / m_l
            y[..., layer, :, :] = (r[..., layer, :, :]
                                   + gl[..., layer - 1, :, :] * y[..., layer - 1, :, :]
                                   ) * minv[..., layer, :, :]
        for layer in range(L - 2, -1, -1):  # back: θ_l = d'_l − c'_l θ_{l+1}
            y[..., layer, :, :] -= cprime[..., layer, :, :] * y[..., layer + 1, :, :]
        return y
    return Minv


# ------------------------------------------------------------------------- GPU stencil operator

def stencil_operator(coeffs, kernel: str = "tiled") -> Op:
    """Aop on CuPy arrays [L, N, N] or [B, L, N, N] from uploaded stencil coefficients.

    coeffs: cuda_kernels.upload_coefficients(c, dtype, to_device=cupy.asarray); its dtype is the
    operator's (float32 for iterations, float64 for residual checks). kernel: "naive" | "tiled".
    """
    if _cp is None:
        raise RuntimeError("stencil_operator needs CuPy")
    from solvers import cuda_kernels as ck

    apply = {"naive": ck.stencil_apply_naive, "tiled": ck.stencil_apply_tiled}.get(kernel)
    if apply is None:
        raise ValueError(f"kernel must be 'naive' or 'tiled', got {kernel!r}")

    def Aop(x: Array) -> Array:
        out = _cp.empty_like(x)
        apply(coeffs, x, out=out)
        return out
    return Aop
