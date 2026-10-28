"""Certified early stopping for PCG on the CPU (Lemma 2(b), CLAUDE.md §8.4, §9.1).

At iterate k, c_k = κ̂ · ‖b − A x_k‖∞ bounds the error |θ − x_k| in every cell, where Aθ = b and
κ̂ ≥ ‖A⁻¹‖∞ (certify/kappa.py). Following §9.1, the recursive residual is only a cheap trigger;
a certificate is issued only on the residual recomputed in float64. This CPU version runs the
whole recursion in float64; the GPU version (N1.4/N4.2) runs it in FP32.

The certificate inherits κ̂'s PLACEHOLDER rounding margin (certify/kappa.py) and adds none of
its own for the float64 residual; certify/margin.py (N4.1) supplies the derived margin.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

__all__ = ["CertificateViolation", "check_bound", "jacobi", "pcg_certified"]

Precond = Callable[[np.ndarray], np.ndarray]
History = Callable[[int, np.ndarray, float], None]


class CertificateViolation(AssertionError):
    """A reported bound is below the true error (invariant B7: blocks results)."""


def check_bound(board: str, iterate: int, bound: float, error: float) -> None:
    """Raise CertificateViolation naming board, iterate, bound and error if bound < error."""
    if not bound >= error:
        raise CertificateViolation(
            f"board {board}, iterate {iterate}: bound {bound:.6g} < true error {error:.6g} K")


def jacobi(A: sp.spmatrix) -> Precond:
    """Diagonal (Jacobi) preconditioner r ↦ r / diag(A). SPD whenever A is (D10 is still open)."""
    d = sp.csr_matrix(A).diagonal()
    if not d.min() > 0:
        raise ValueError(f"Jacobi needs a positive diagonal, min is {d.min()}")
    return lambda r: r / d


def pcg_certified(
    A: sp.spmatrix,
    b: np.ndarray,
    x0: np.ndarray,
    Minv: Precond,
    kappa: float,
    tau_v: float,
    maxit: int = 20000,
    *,
    history: History | None = None,
) -> tuple[np.ndarray, int, float | None]:
    """Warm-started PCG that stops once κ̂‖b − Ax‖∞ ≤ tau_v (K). Returns (x, iters, cert).

    cert is the certified bound on max |θ − x| (K) at the returned x, or None if maxit was hit.
    history(k, x_k, c_k), if given, is called at the warm start (k = 0) and after every step
    with the float64 bound c_k; it costs one extra SpMV per step (tests and benchmarks only).
    The stopping rule is the same with or without history.
    """
    if kappa is None or not (np.isfinite(kappa) and kappa > 0):
        raise ValueError(f"kappa must be a positive certificate constant, got {kappa} "
                         "(None means no certificate: solve to tolerance instead)")
    if not tau_v > 0:
        raise ValueError(f"tau_v must be > 0 K, got {tau_v}")
    A = sp.csr_matrix(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.array(x0, dtype=np.float64)

    def certificate(xk: np.ndarray) -> float:
        return float(kappa * np.abs(b - A @ xk).max())

    r = b - A @ x
    if history is not None:
        history(0, x.copy(), certificate(x))
    z = Minv(r)
    p = z.copy()
    rz = float(r @ z)
    for it in range(maxit):
        Ap = A @ p
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        c = certificate(x) if history is not None else None
        if history is not None:
            history(it + 1, x.copy(), c)
        if kappa * float(np.abs(r).max()) <= tau_v:  # cheap test on the recursive residual
            c = certificate(x) if c is None else c  # float64 true residual before certifying
            if c <= tau_v:
                return x, it + 1, c
        z = Minv(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit, None
