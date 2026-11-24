"""Batched certified PCG: one launch, each board leaves when its certificate settles it
(N4.2 GPU path; CLAUDE.md §8.1 step 5, §8.4, §9.1, §9.2; invariant B7).

    res = batched_pcg_certified(Aop, Aop64, b, x0, Minv, kappa=..., S=..., tau_v=...,
                                check_every=..., maxit=...)

Arrays are [B, L, N, N] (rule 7), NumPy or CuPy (the same arithmetic; CI runs NumPy). The
iteration runs in b.dtype with one α, β per board and a done mask (a finished board gets α = 0,
so its x stops changing; no compaction), as solvers/pcg.batched_pcg. Every check_every
iterations, at the warm start (k = 0) and at maxit, every active board's TRUE residual
b − A x is recomputed with Aop64 in float64 and certified on the host with
certify.margin.residual_bound (κ̂ (R̂ + τ_r), rounded up; Lemma 2(b)): c bounds |θ − x| in every
cell. The recursive residual never certifies anything (§8.4, §18.5).

A board stops when its certificate settles its decision (§8.1 step 5):
  flagged board (tau_v finite)   c ≤ tau_v                          -> "certified"
  audited board (tau_v = NaN)    the score interval of [max x − c, max x + c] (Lemma 1′) is
                                 decided by certify.peak_interval.decide   -> "below_q" |
                                 "above_q" | "width"
  either, after maxit iterations (the label-independent cap, D14)  -> "maxit"
The returned cert, T_lo, T_hi (and S_lo, S_hi for audits) belong to the returned x.

Iterating in float32 leaves the float64 true residual of the float32 iterate at about
S·‖x‖·2⁻²⁴; with κ̂ ≈ 1.5e6 K/W on the 4-layer board that is ~1 K, so certificates of 0.1 K need
float64 iterations (bench/pipeline_bench.py takes the dtype as a flag and records it).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from certify.margin import residual_bound
from certify.peak_interval import decide, score_interval

try:
    import cupy as _cp
except ImportError:  # CPU-only machines (CI)
    _cp = None

__all__ = ["BatchResult", "batched_pcg_certified"]

Array = Any
Op = Callable[[Array], Array]


def _xp(a: Array):
    return _cp.get_array_module(a) if _cp is not None else np


def _host(a: Array) -> np.ndarray:
    return _cp.asnumpy(a) if _cp is not None and isinstance(a, _cp.ndarray) else np.asarray(a)


def _per_board(xp, a: Array, fn: str) -> np.ndarray:
    return _host(getattr(xp, fn)(a.reshape(a.shape[0], -1), axis=1))


def _bdot(xp, a: Array, b: Array) -> Array:
    return xp.sum((a * b).reshape(a.shape[0], -1), axis=1, dtype=xp.float64)


@dataclass
class BatchResult:
    x: Array  # [B, L, N, N], iteration dtype
    iters: np.ndarray  # [B] PCG iterations while active
    cert: np.ndarray  # [B] certified bound on max|θ − x| (K) at the returned x
    T_lo: np.ndarray  # [B] certified interval for the true peak (K)
    T_hi: np.ndarray
    S_lo: np.ndarray  # [B] certified score interval (audited boards; NaN for flagged)
    S_hi: np.ndarray
    stop: list[str]


def _host_vec(name: str, v, B: int, *, allow_nan: bool = False) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (B,):
        raise ValueError(f"{name} must have shape ({B},), got {a.shape}")
    ok = np.isfinite(a) | (np.isnan(a) if allow_nan else False)
    if not np.all(ok):
        raise ValueError(f"{name} has non-finite entries")
    return a


def batched_pcg_certified(
    Aop: Op, Aop64: Op, b: Array, x0: Array, Minv: Op, *, kappa, S, tau_v,
    check_every: int, maxit: int, peak_hat=None, s=None, q=None, eps_S=None,
) -> BatchResult:
    """Certified, warm-started batched PCG (module docstring).

    kappa [B]: guaranteed κ̂ ≥ ‖A⁻¹‖∞ per board (certify.kappa); S [B]: row_abs_sum_bound of
    each board's A (W/K); tau_v [B]: certificate level (K), NaN for an audited board, which
    then needs peak_hat (max μ, K), s (K), q (≥ 0, may be inf) and eps_S [B].
    """
    xp = _xp(b)
    if b.ndim < 2 or x0.shape != b.shape:
        raise ValueError(f"b and x0 must be equal-shape [B, ...], got {b.shape}, {x0.shape}")
    for name, v in (("check_every", check_every), ("maxit", maxit)):
        if isinstance(v, bool) or not isinstance(v, int | np.integer) or v < 1:
            raise ValueError(f"{name} must be an int >= 1, got {v!r}")
    B = b.shape[0]
    kappa, S = _host_vec("kappa", kappa, B), _host_vec("S", S, B)
    if not (np.all(kappa > 0) and np.all(S > 0)):
        raise ValueError("kappa and S must be > 0 for every board (None kappa: no certificate)")
    tau = _host_vec("tau_v", tau_v, B, allow_nan=True)
    audit = np.isnan(tau)
    if audit.any():
        if any(v is None for v in (peak_hat, s, q, eps_S)):
            raise ValueError("audited boards (tau_v NaN) need peak_hat, s, q and eps_S")
        peak_hat = _host_vec("peak_hat", peak_hat, B)
        s = _host_vec("s", s, B)
        q = np.asarray(q, dtype=np.float64)
        eps_S = _host_vec("eps_S", eps_S, B)
        if q.shape != (B,) or np.any(np.isnan(q)) or np.any(q < 0):
            raise ValueError("q must be [B], >= 0 (inf allowed)")
    if np.any(tau[~audit] <= 0):
        raise ValueError("tau_v must be > 0 K for flagged boards")

    dtype = b.dtype
    col = (B,) + (1,) * (b.ndim - 1)
    b64 = b.astype(xp.float64)
    b_inf = _per_board(xp, xp.abs(b64), "max")
    x = x0.astype(dtype, copy=True)
    iters = np.zeros(B, dtype=np.int64)
    done = np.zeros(B, dtype=bool)
    nan = np.full(B, np.nan)
    cert, T_lo, T_hi, S_lo, S_hi = nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy()
    stop = [""] * B

    def check(final: bool) -> None:
        x64 = x.astype(xp.float64)
        R = _per_board(xp, xp.abs(b64 - Aop64(x64)), "max")
        X = _per_board(xp, xp.abs(x64), "max")
        xm = _per_board(xp, x64, "max")
        for j in np.flatnonzero(~done):
            c = residual_bound(float(kappa[j]), float(R[j]), float(S[j]), float(b_inf[j]),
                               float(X[j]))
            cert[j] = c
            T_lo[j] = float(np.nextafter(xm[j] - c, -math.inf))
            T_hi[j] = float(np.nextafter(xm[j] + c, math.inf))
            why = None
            if audit[j]:
                S_lo[j], S_hi[j] = score_interval(float(peak_hat[j]), T_lo[j], T_hi[j],
                                                  float(s[j]))
                why = decide(S_lo[j], S_hi[j], float(q[j]), float(eps_S[j]))
            elif c <= tau[j]:
                why = "certified"
            if why is None and final:
                why = "maxit"
            if why is not None:
                done[j] = True
                stop[j] = why

    r = b - Aop(x)
    z = Minv(r)
    rz = _bdot(xp, r, z)
    p = z.copy()
    check(final=False)
    k = 0
    while not done.all():
        active = xp.asarray(~done)
        Ap = Aop(p)
        pAp = _bdot(xp, p, Ap)
        rz_h, pAp_h = _host(rz), _host(pAp)
        bad = ~done & (rz_h > 0) & ~(np.isfinite(pAp_h) & (pAp_h > 0))
        if bad.any():
            raise ValueError(f"PCG breakdown at iteration {k + 1} for boards "
                             f"{np.flatnonzero(bad).tolist()}: pᵀAp <= 0 (A or M⁻¹ not SPD)")
        go = active & (rz > 0)
        alpha = xp.where(go, rz / xp.where(go, pAp, 1.0), 0.0).astype(dtype).reshape(col)
        x += alpha * p
        r -= alpha * Ap
        iters[~done] += 1
        k += 1
        z = Minv(r)
        rz_new = _bdot(xp, r, z)
        go = rz > 0
        beta = xp.where(go, rz_new / xp.where(go, rz, 1.0), 0.0).astype(dtype).reshape(col)
        p = z + beta * p
        rz = rz_new
        if k % check_every == 0 or k >= maxit:
            check(final=k >= maxit)
    return BatchResult(x, iters, cert, T_lo, T_hi, S_lo, S_hi, stop)
