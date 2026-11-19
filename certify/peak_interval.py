"""Certified peak interval and decision-sufficient audits (N3.3; CLAUDE.md §8.3, §8.4, B7).

peak_interval(x, coeffs, b, kappa_hat, w) bounds the true peak T_peak = max θ of A θ = b, for any
approximation x, from one FP64 residual b − A x:
  * Lemma 2(b), kappa_hat = κ̂ ≥ ‖A⁻¹‖∞: c = certify.margin.residual_bound(κ̂, ‖r̂‖∞, ...) bounds
    |θ − x| in every cell (FP64 residual plus its derived rounding margin, Step 7b), so
    T_peak ∈ [max x − c, max x + c].
  * Lemma 2(c), w: the hypothesis w ≥ 0, A w ≥ |r| is CHECKED here in FP64 with the derived
    margin, exactly as certify/pointwise.pointwise_bound does (Step 7c); a w that fails raises.
    Then T_peak ∈ [max(x − w), max(x + w)].
  * both: the intersection of the two (each is valid, so their intersection is).
Every end point is rounded outward. The certificate is relative to the exact solution of the stored
float64 system (certify/margin.py), the system the 1e-8 oracle solves.

The operator is the 7-point layered stencil of solvers/fvm.coefficients ([4, N, N] arrays, W/K):
(A x)_P = diag x_P − Ge x_E − Gw x_W − Gn x_N − Gs x_S − gup x_{l+1} − gdn x_{l−1}. These are the
stored values assemble_board puts in A (off-diagonals −G), so the margin's 7-products-per-row model
applies in any evaluation order. A must be a nonsingular M-matrix (B1): the coefficients are
checked to be an M-matrix stencil (diag > 0, couplings ≥ 0 and symmetric, none off the grid);
nonsingularity is B1's (solvers/mmatrix_test.py), and κ̂ must come from the same A
(certify/kappa.kappa_hat).

audit_until_decided runs FP64 PCG from the prediction μ (line-z block Jacobi by default, as
data/generate.py; the production preconditioner is open decision D10, and it changes only the
speed) and checks the Lemma 2(b) interval every check_every iterations, starting at μ itself
(k = 0). With T̂ = max μ and S = |T̂ − T_peak| / s, the certified score interval is
    S⁺ = max(|T̂ − T⁻|, |T̂ − T⁺|) / s,   S⁻ = 0 if T̂ ∈ [T⁻, T⁺], else min(...) / s,
rounded outward (S⁺ ≥ S: Lemma 1′, §8.3). It stops at the first check where
    "below_q"  S⁺ ≤ q_t      (S ≤ q_t certainly)
    "above_q"  S⁻ > q_t      (S > q_t certainly)
    "width"    S⁺ − S⁻ < ε_S
in that order, else at "converged" (true relative residual ≤ 1e-8, the oracle tolerance) or
"maxit". The stop is label-dependent; maxit is the label-independent cap of D14 (open: the value is
not fixed here; 20,000 is solvers/pcg.py's default). audit_trace runs the same iteration with no
decision, to 1e-8 or maxit, and returns every check: the N3.3 trace that experiments/run_gates.py
replays for E14 with score_interval and decide from this module, so a replayed audit stops at the
same check, with the same interval, as a live one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from certify.margin import NNZ_ROW, gamma, matvec_tau, residual_bound, residual_tau
from solvers.pcg import line_z_block_jacobi

__all__ = ["AuditTrace", "TOL_FULL", "apply_A", "audit_trace", "audit_until_decided", "decide",
           "peak_interval", "row_abs_sum_bound", "score_interval"]

FIELDS = ("Ge", "Gw", "Gn", "Gs", "gup", "gdn", "diag")
TOL_FULL = 1e-8  # an undecided audit stops at the oracle tolerance (§11.5, gate_experiment_inputs)
DEFAULT_MAXIT = 20000  # solvers/pcg.py's default; the audit cap itself is D14 (open)

Precond = Callable[[np.ndarray], np.ndarray]


# ------------------------------------------------------------------------------ the operator

def _coeffs(coeffs: dict) -> dict[str, np.ndarray]:
    missing = [k for k in FIELDS if k not in coeffs]
    if missing:
        raise ValueError(f"coeffs needs {FIELDS} (solvers/fvm.coefficients), missing {missing}")
    c = {k: np.asarray(coeffs[k], dtype=np.float64) for k in FIELDS}
    shape = c["diag"].shape
    if len(shape) != 3 or any(v.shape != shape for v in c.values()):
        raise ValueError(f"coeffs must all be [L, N, N] of one shape, got "
                         f"{ {k: v.shape for k, v in c.items()} }")
    if not all(np.all(np.isfinite(v)) for v in c.values()):
        raise ValueError("coeffs must be finite")
    off_grid = (c["Ge"][:, :, -1], c["Gw"][:, :, 0], c["Gn"][:, -1, :], c["Gs"][:, 0, :],
                c["gup"][-1], c["gdn"][0])
    symmetric = (np.array_equal(c["Ge"][:, :, :-1], c["Gw"][:, :, 1:])
                 and np.array_equal(c["Gn"][:, :-1, :], c["Gs"][:, 1:, :])
                 and np.array_equal(c["gup"][:-1], c["gdn"][1:]))
    if not (c["diag"].min() > 0 and all(c[k].min() >= 0 for k in FIELDS[:-1])
            and all(not v.any() for v in off_grid) and symmetric):
        raise ValueError("coeffs are not a symmetric M-matrix stencil (need diag > 0, couplings "
                         ">= 0 and symmetric, none pointing off the grid)")
    return c


def apply_A(coeffs: dict, x: np.ndarray) -> np.ndarray:
    """A x in FP64 for x [L, N, N] (K) → W, from stencil coefficients (unchecked; see _coeffs)."""
    c = coeffs
    y = c["diag"] * x
    y[:, :, :-1] -= c["Ge"][:, :, :-1] * x[:, :, 1:]
    y[:, :, 1:] -= c["Gw"][:, :, 1:] * x[:, :, :-1]
    y[:, :-1, :] -= c["Gn"][:, :-1, :] * x[:, 1:, :]
    y[:, 1:, :] -= c["Gs"][:, 1:, :] * x[:, :-1, :]
    y[:-1] -= c["gup"][:-1] * x[1:]
    y[1:] -= c["gdn"][1:] * x[:-1]
    return y


def _up(q: Fraction) -> float:
    f = float(q)
    return f if Fraction(f) >= q else float(np.nextafter(f, math.inf))


def row_abs_sum_bound(coeffs: dict) -> float:
    """S ≥ max_i Σ_j |a_ij| (W/K): fl(row sum of the 7 coefficients) / (1 − γ_7), rounded up
    (certify/margin.py Step 2; the coefficients are the |a_ij|)."""
    c = coeffs
    s = c["diag"] + c["Ge"] + c["Gw"] + c["Gn"] + c["Gs"] + c["gup"] + c["gdn"]
    return _up(Fraction(float(s.max())) / (1 - gamma(NNZ_ROW)))


def _field(name: str, v, shape: tuple[int, ...]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape} (the coefficients'), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _kappa(k) -> float:
    if k is None or isinstance(k, bool) or not (math.isfinite(float(k)) and float(k) > 0):
        raise ValueError(f"kappa_hat must be a positive certificate constant, got {k!r} (None "
                         "means no certificate: solve to tolerance instead)")
    return float(k)


# --------------------------------------------------------------------------- peak interval

def _kappa_interval(c: dict, b: np.ndarray, x: np.ndarray, S: float, k: float
                    ) -> tuple[float, float, float]:
    """Lemma 2(b) interval and the true relative residual ‖b − A x‖₂ / ‖b‖₂."""
    r = b - apply_A(c, x)
    cb = residual_bound(k, float(np.abs(r).max()), S, float(np.abs(b).max()),
                        float(np.abs(x).max()))
    xm = float(x.max())
    lo = float(np.nextafter(xm - cb, -math.inf))
    hi = float(np.nextafter(xm + cb, math.inf))
    return lo, hi, float(np.linalg.norm(r) / np.linalg.norm(b))


def peak_interval(x: np.ndarray, coeffs: dict, b: np.ndarray, kappa_hat: float | None = None,
                  w: np.ndarray | None = None) -> tuple[float, float, str]:
    """(T_lo, T_hi, method) with T_lo ≤ max θ ≤ T_hi (K), method "kappa" | "pointwise" | "both".

    x: approximation [L, N, N] (K); b: right-hand side (W), same shape; kappa_hat ≥ ‖A⁻¹‖∞
    (K/W) for the same A; w ≥ 0 [L, N, N] (K), verified here. At least one of the two.
    """
    if kappa_hat is None and w is None:
        raise ValueError("peak_interval needs kappa_hat or w (Lemma 2(b) or 2(c))")
    c = _coeffs(coeffs)
    shape = c["diag"].shape
    x, b = _field("x", x, shape), _field("b", b, shape)
    S = row_abs_sum_bound(c)
    out = []
    if w is not None:
        w = _field("w", w, shape)
        if not w.min() >= 0:
            raise ValueError(f"w must be >= 0 (Lemma 2(c)), min is {w.min():.3e}")
        abs_r = np.abs(b - apply_A(c, x))
        tau = float(np.nextafter(residual_tau(S, float(np.abs(b).max()), float(np.abs(x).max()))
                                 + matvec_tau(S, float(w.max())), math.inf))
        need = np.nextafter(abs_r + tau, math.inf)  # up(|r̂| + τ_r + τ_w): A w ≥ |r| exactly
        bad = int(np.count_nonzero(~(apply_A(c, w) >= need)))
        if bad:
            raise ValueError(f"Lemma 2(c) hypothesis fails: A w >= |r| (FP64, with the derived "
                             f"rounding margin) does not hold in {bad} cells")
        out.append((float(np.nextafter(x - w, -math.inf).max()),
                    float(np.nextafter(x + w, math.inf).max())))
    if kappa_hat is not None:
        lo, hi, _ = _kappa_interval(c, b, x, S, _kappa(kappa_hat))
        out.append((lo, hi))
    if len(out) == 2:
        return max(out[0][0], out[1][0]), min(out[0][1], out[1][1]), "both"
    return out[0][0], out[0][1], "pointwise" if w is not None else "kappa"


# -------------------------------------------------------------------------- score, decision

def _abs_diff(a: float, b: float, up: bool) -> float:
    d = abs(a - b)  # nearest float to |a − b|: one step outward contains the exact value
    return float(np.nextafter(d, math.inf)) if up else max(float(np.nextafter(d, 0.0)), 0.0)


def score_interval(peak_hat: float, T_lo: float, T_hi: float, s: float) -> tuple[float, float]:
    """[S⁻, S⁺] ∋ |peak_hat − T_peak| / s for every T_peak ∈ [T_lo, T_hi], rounded outward."""
    if not (math.isfinite(s) and s > 0):
        raise ValueError(f"s must be finite and > 0 K, got {s!r}")
    if not (math.isfinite(T_lo) and math.isfinite(T_hi) and T_lo <= T_hi):
        raise ValueError(f"need finite T_lo <= T_hi, got [{T_lo}, {T_hi}]")
    s_hi = max(_abs_diff(peak_hat, T_lo, True), _abs_diff(peak_hat, T_hi, True))
    s_hi = float(np.nextafter(s_hi / s, math.inf))
    if T_lo <= peak_hat <= T_hi:
        return 0.0, s_hi
    near = min(_abs_diff(peak_hat, T_lo, False), _abs_diff(peak_hat, T_hi, False))
    return max(float(np.nextafter(near / s, 0.0)), 0.0), s_hi


def decide(S_lo: float, S_hi: float, q_t: float, eps_S: float) -> str | None:
    """'below_q' (S⁺ ≤ q_t), 'above_q' (S⁻ > q_t), 'width' (S⁺ − S⁻ < ε_S), else None."""
    if S_hi <= q_t:
        return "below_q"
    if S_lo > q_t:
        return "above_q"
    if S_hi - S_lo < eps_S:
        return "width"
    return None


# ---------------------------------------------------------------------------- the iteration

def _check_run(check_every: int, maxit: int) -> None:
    for name, v in (("check_every", check_every), ("maxit", maxit)):
        if isinstance(v, bool) or not isinstance(v, int | np.integer) or v < 1:
            raise ValueError(f"{name} must be an int >= 1, got {v!r}")


def _checks(mu, coeffs: dict, b, kappa_hat, check_every: int, maxit: int,
            Minv: Precond | None) -> Iterator[tuple[int, float, float, bool, bool]]:
    """(k, T_lo, T_hi, last, converged) at k = 0, every check_every iterations, and the final
    iterate (true relres ≤ TOL_FULL, or maxit). FP64 PCG warm-started from μ."""
    _check_run(check_every, maxit)
    k_hat = _kappa(kappa_hat)
    c = _coeffs(coeffs)
    shape = c["diag"].shape
    x, b = _field("mu", mu, shape).copy(), _field("b", b, shape)
    if not np.abs(b).max() > 0:
        raise ValueError("b is zero: the solution is 0")
    S = row_abs_sum_bound(c)
    M = line_z_block_jacobi(c["diag"], c["gup"]) if Minv is None else Minv

    def check(k: int) -> tuple[int, float, float, bool, bool]:
        lo, hi, rel = _kappa_interval(c, b, x, S, k_hat)
        return k, lo, hi, rel <= TOL_FULL or k >= maxit, rel <= TOL_FULL

    first = check(0)
    yield first
    if first[3]:
        return
    r = b - apply_A(c, x)
    z = M(r)
    p = z.copy()
    rz = float(np.vdot(r, z))
    for k in range(1, maxit + 1):
        Ap = apply_A(c, p)
        pAp = float(np.vdot(p, Ap))
        if not (math.isfinite(pAp) and pAp > 0):
            raise ValueError(f"PCG breakdown at iteration {k}: pᵀAp = {pAp} (A or M⁻¹ not SPD)")
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        z = M(r)
        rz_new = float(np.vdot(r, z))
        if k % check_every == 0 or k == maxit or rz_new == 0.0:
            out = check(k)
            if rz_new == 0.0 and not out[3]:  # recursive residual exactly zero, true one not small
                raise ValueError(f"PCG stalled at iteration {k}: the recursive residual is 0 but "
                                 f"the true relative residual is above {TOL_FULL}")
            yield out
            if out[3]:
                return
        if not (math.isfinite(rz_new) and rz_new > 0):
            raise ValueError(f"PCG breakdown at iteration {k}: rᵀM⁻¹r = {rz_new}")
        p = z + (rz_new / rz) * p
        rz = rz_new


@dataclass(frozen=True)
class AuditTrace:
    """Every check of one audit solve: PCG iteration, certified [T_lo, T_hi] (K), stop reason
    ('converged' at true relres ≤ 1e-8, or 'maxit')."""

    iters: list[int]
    T_lo: list[float]
    T_hi: list[float]
    stop_reason: str


def audit_trace(mu: np.ndarray, coeffs: dict, b: np.ndarray, kappa_hat: float,
                check_every: int = 5, *, maxit: int = DEFAULT_MAXIT,
                Minv: Precond | None = None) -> AuditTrace:
    """The certified interval at every check of PCG from μ, to 1e-8 or maxit (the E14 trace)."""
    its, los, his, conv = [], [], [], []
    for k, lo, hi, _last, cv in _checks(mu, coeffs, b, kappa_hat, check_every, maxit, Minv):
        its.append(k)
        los.append(lo)
        his.append(hi)
        conv.append(cv)
    return AuditTrace(its, los, his, "converged" if conv[-1] else "maxit")


def audit_until_decided(mu: np.ndarray, coeffs: dict, b: np.ndarray, s: float, q_t: float,
                        eps_S: float, kappa_hat: float, check_every: int = 5, *,
                        maxit: int = DEFAULT_MAXIT, Minv: Precond | None = None
                        ) -> tuple[float, float, float, float, int, str]:
    """(T_lo, T_hi, S_lo, S_hi, iters, stop_reason) of a certified audit (Lemma 1′, §8.3, §8.4).

    mu: prediction [L, N, N] (K), the warm start and T̂ = max μ; s: label-free scale (K);
    q_t ≥ 0 (may be inf); eps_S > 0. stop_reason: below_q | above_q | width | converged | maxit.
    """
    if not (math.isfinite(s) and s > 0):
        raise ValueError(f"s must be finite and > 0 K, got {s!r}")
    if math.isnan(q_t) or q_t < 0:
        raise ValueError(f"q_t must be >= 0 (inf allowed), got {q_t!r}")
    if not (math.isfinite(eps_S) and eps_S > 0):
        raise ValueError(f"eps_S must be finite and > 0, got {eps_S!r}")
    _check_run(check_every, maxit)
    _kappa(kappa_hat)
    peak_hat = float(np.max(np.asarray(mu, dtype=np.float64)))
    for k, lo, hi, last, conv in _checks(mu, coeffs, b, kappa_hat, check_every, maxit, Minv):
        s_lo, s_hi = score_interval(peak_hat, lo, hi, s)
        why = decide(s_lo, s_hi, q_t, eps_S)
        if why is not None:
            return lo, hi, s_lo, s_hi, k, why
        if last:
            return lo, hi, s_lo, s_hi, k, "converged" if conv else "maxit"
    raise AssertionError("unreachable: _checks always ends with a last check")
