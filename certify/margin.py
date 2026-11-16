"""Derived FP64 rounding margin for the Lemma 2 certificate (prompt N4.1; CLAUDE.md §8.4, B7).

Lemma 2(b) says ‖e‖∞ ≤ κ̂_exact·‖r‖∞ with κ̂_exact = max z / min(A z) and r = b − A x, all exact.
The pipeline computes A z and b − A x in IEEE binary64. This module bounds what that rounding can
hide, so the reported certificate is a guaranteed upper bound:

    certified(c, m) = c·(1 + eps_total) + delta_abs  ≥  κ̂_exact·‖r_exact‖∞,

where c = fl(κ̂·R̂) is the certificate the code already forms (κ̂ ≥ fl(max z / m̂), R̂ = max |r̂|),
and no certificate is issued when the computed min(A z) is within rounding of zero.

DERIVATION (for the Lemma 2 checker, M; every step is checked numerically in tests/test_margin.py
against 80-digit arithmetic on 5 boards)

Data. A, b, x, z are the STORED float64 arrays. A has at most 7 non-zeros per row (diagonal, four
in-plane neighbours, two vertical neighbours; CLAUDE.md §7.2). "Exact" below means exact real
arithmetic on these stored values: the certificate is relative to the exact solution of the stored
float64 system, which is the discrete system the solvers and the 1e-8 oracle solve.
UNSURE (for M): assembly rounding (stored A vs the A of exact coefficient formulas) is outside
this margin by that definition; the report should say "exact discrete solution of the assembled
system".

Model. u = 2⁻⁵³. Every operation is fl(a ∘ b) = (a ∘ b)(1 + δ) + η with |δ| ≤ u; η = 0 for + and −
(gradual underflow makes subnormal sums exact) and |η| ≤ 2⁻¹⁰⁷⁵ for × (a subnormal product).
No overflow. abs and max are exact. γ_n := n·u / (1 − n·u), and (1 + u)ⁿ − 1 ≤ γ_n and
1 − (1 − u)ⁿ ≤ γ_n for n·u < 1 (tested exactly for n ≤ 8).

Step 1, a row of n products. In ANY summation order, with or without FMA, each product p_k q_k
passes through at most n roundings (one product, at most n − 1 additions; an FMA merges a product
with an addition into one rounding, so never more). Hence
    |fl(Σ_k p_k q_k) − Σ_k p_k q_k| ≤ γ_n Σ_k |p_k q_k| + n·2⁻¹⁰⁷⁴,
the last term covering at most n underflowed products, each later multiplied by at most
(1 + u)ⁿ⁻¹ < 2.

Step 2, row sums of |A|. s_i = fl(Σ_j |a_ij|) adds at most 7 non-negative terms, so the true sum is
at most s_i / (1 − γ_7). S := up(max_i s_i / (1 − γ_7)) ≥ max_i Σ_j |a_ij|. The shortcut
Σ_j≠i |a_ij| ≤ a_ii (B1's weak dominance) is NOT used: on stored rows the computed row sum can be
a few 1e-17 W/K below zero.

Step 3, min(A z). With n = 7 and z > 0: |fl(Az)_i − (Az)_i| ≤ γ_7 (|A| z)_i + 7·2⁻¹⁰⁷⁴
≤ γ_7 S z_max + 7·2⁻¹⁰⁷⁴ =: τ_az. So min(Az)_exact ≥ m̂ − τ_az with m̂ = min_i fl(Az)_i.
No certificate unless m̂ ≥ 2 τ_az (min_az_threshold). Then ρ := τ_az / m̂ ≤ ½,
min(Az)_exact ≥ m̂(1 − ρ) > 0 (Lemma 2(a)'s hypothesis holds exactly; z > 0 is checked on the
stored values, exactly), and κ̂_exact = z_max / min(Az)_exact ≤ K / (1 − ρ), K := z_max / m̂.

Step 4, the residual. r̂_i = fl(b_i − (A x)_i) is a sum of at most 8 terms (b_i and 7 products), so
|r̂_i − r_i| ≤ γ_8 (|b_i| + (|A||x|)_i) + 8·2⁻¹⁰⁷⁴ ≤ γ_8 (‖b‖∞ + S·X) + 8·2⁻¹⁰⁷⁴ =: τ_r,
for any cap X ≥ ‖x‖∞. Hence ‖r‖∞ ≤ R̂ + τ_r.

Step 5, combine. c_exact := κ̂_exact‖r‖∞ ≤ K (R̂ + τ_r) / (1 − ρ). The code's κ̂ satisfies
κ̂ ≥ fl(K) ≥ K(1 − u), and c = fl(κ̂ R̂) ≥ κ̂ R̂ (1 − u), so K R̂ ≤ c / (1 − u)² and
K τ_r ≤ κ̂ τ_r / (1 − u). Therefore
    c_exact ≤ c / ((1 − u)²(1 − ρ)) + κ̂ τ_r / ((1 − u)(1 − ρ)),
    eps_total := 1 / ((1 − u)²(1 − ρ)) − 1,   delta_abs := κ̂ τ_r / ((1 − u)(1 − ρ)).
eps_total is relative and dimensionless; delta_abs is in kelvin. Both are computed in exact rational
arithmetic and rounded up to float64.

Step 6, evaluation. certified() forms c·(1 + eps) + delta with each floating-point operation
rounded up (next float toward +∞), so the returned float is ≥ the real value of that expression.

Step 7, the wiring into certify/ (N4.2; for the Lemma 2 checker, M). Three corollaries of Steps
1-4 used by kappa.py, stopping.py and pointwise.py; each is evaluated in exact rational arithmetic
and rounded up once, so no further rounding analysis is needed.
  (a) kappa_upper: ‖A⁻¹‖∞ ≤ κ̂_exact ≤ z_max / (m̂(1 − ρ)) by Step 3 (no certificate unless
      m̂ ≥ 2 τ_az). The returned κ̂ is that rational rounded up, so it is itself a guaranteed
      upper bound on ‖A⁻¹‖∞, and it is >= fl(z_max / m̂), as margin() requires.
  (b) residual_bound: for κ ≥ ‖A⁻¹‖∞ and R̂ = max_i |fl(b − A x)_i|, Lemma 2(b) and Step 4 give
      ‖θ − x‖∞ ≤ ‖A⁻¹‖∞ ‖r‖∞ ≤ κ (R̂ + τ_r). Returned: up(κ (R̂ + τ_r)).
  (c) matvec_tau: for any v, |fl(A v)_i − (A v)_i| ≤ γ_7 (|A||v|)_i + 7·2⁻¹⁰⁷⁴ ≤ γ_7 S ‖v‖∞ +
      7·2⁻¹⁰⁷⁴ (Step 1 with n = 7, then Step 2). pointwise.py uses it with v = w ≥ 0: if
      fl(A w)_i ≥ |r̂_i| + τ_w + τ_r for every i (the right side rounded up), then
      (A w)_i ≥ fl(A w)_i − τ_w ≥ |r̂_i| + τ_r ≥ |r_i|, which is Lemma 2(c)'s hypothesis A w ≥ |r|.
  (d) Outward rounding of the peak interval: fl(x_i ± w_i) is one rounding to nearest, so the next
      float away from it (toward ±∞) bounds the exact x_i ± w_i.

Assumptions, in the order used: binary64, round to nearest; no overflow; finite inputs; at most 7
non-zeros per row; z > 0 entrywise; κ̂ ≥ fl(z_max / m̂) is the value used to form c; ‖x‖∞ ≤ X at
the certified iterate (certified() checks it when given ‖x‖∞); the FP64 recomputation uses the
FP64 coefficient arrays (an FP32 iteration may run between checks, §8.4, but is never certified on).
Reduction order and FMA do not matter (Step 1).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

__all__ = ["Margin", "certified", "certify_residual", "gamma", "kappa_upper", "margin",
           "matvec_tau", "residual_bound", "residual_tau", "row_abs_sum_bound"]

U = Fraction(1, 2**53)  # unit roundoff of binary64
ETA = Fraction(1, 2**1074)  # 2·(underflow error of one product)
NNZ_ROW = 7  # 7-point layered stencil (CLAUDE.md §7.2)


def gamma(n: int) -> Fraction:
    """γ_n = n·u / (1 − n·u): the relative error bound of n roundings (exact rational)."""
    if not (isinstance(n, int) and 0 < n and n * U < 1):
        raise ValueError(f"gamma needs an int n >= 1 with n·u < 1, got {n!r}")
    return n * U / (1 - n * U)


def _up(q: Fraction) -> float:
    """The smallest float >= q (q is exact)."""
    f = float(q)
    return f if Fraction(f) >= q else float(np.nextafter(f, math.inf))


def _down(q: Fraction) -> float:
    f = float(q)
    return f if Fraction(f) <= q else float(np.nextafter(f, -math.inf))


@dataclass(frozen=True)
class Margin:
    """eps_total (1), delta_abs (K), min_az_threshold (W), x_inf_max (K); see the docstring."""

    eps_total: float
    delta_abs: float
    min_az_threshold: float
    x_inf_max: float


def _stat(stats: dict, key: str) -> Fraction:
    if key not in stats:
        raise ValueError(f"margin stats need {key!r}")
    v = float(stats[key])
    if not math.isfinite(v) or (v < 0 and key != "min_az"):
        raise ValueError(f"margin stats: {key} must be finite and >= 0, got {stats[key]!r}")
    return Fraction(v)


def _tau_az(S: Fraction, z_max: Fraction) -> Fraction:
    """Step 3: |fl(A z)_i − (A z)_i| ≤ γ_7 S z_max + 7·2⁻¹⁰⁷⁴ (W)."""
    return gamma(NNZ_ROW) * S * z_max + NNZ_ROW * ETA


def _tau_r(S: Fraction, b_inf: Fraction, X: Fraction) -> Fraction:
    """Step 4: |fl(b − A x)_i − (b − A x)_i| ≤ γ_8 (‖b‖∞ + S·X) + 8·2⁻¹⁰⁷⁴ (W)."""
    return gamma(NNZ_ROW + 1) * (b_inf + S * X) + (NNZ_ROW + 1) * ETA


def _nonneg(name: str, v: float) -> Fraction:
    v = float(v)
    if not (math.isfinite(v) and v >= 0):
        raise ValueError(f"{name} must be finite and >= 0, got {v!r}")
    return Fraction(v)


def kappa_upper(row_abs_sum_max: float, z_max: float, min_az: float) -> float | None:
    """Guaranteed κ̂ ≥ ‖A⁻¹‖∞ (K/W) from z > 0 and m̂ = min fl(A z), or None (Step 7a).

    row_abs_sum_max is S from row_abs_sum_bound(A). The caller has checked z > 0 on the stored
    values and that A is a nonsingular M-matrix (B1). None: m̂ is within rounding of zero.
    """
    S, zm = _nonneg("row_abs_sum_max", row_abs_sum_max), _nonneg("z_max", z_max)
    m_hat = float(min_az)
    if not math.isfinite(m_hat):
        raise ValueError(f"min_az must be finite, got {min_az!r}")
    m_hat = Fraction(m_hat)
    tau_az = _tau_az(S, zm)
    if m_hat <= 0 or m_hat < 2 * tau_az or zm <= 0:
        return None
    return _up(zm / (m_hat * (1 - tau_az / m_hat)))


def residual_tau(row_abs_sum_max: float, b_inf: float, x_inf: float) -> float:
    """τ_r (W), rounded up: |fl(b − A x)_i − (b − A x)_i| ≤ τ_r for ‖x‖∞ ≤ x_inf (Step 4)."""
    return _up(_tau_r(_nonneg("row_abs_sum_max", row_abs_sum_max), _nonneg("b_inf", b_inf),
                      _nonneg("x_inf", x_inf)))


def matvec_tau(row_abs_sum_max: float, v_inf: float) -> float:
    """τ (W), rounded up: |fl(A v)_i − (A v)_i| ≤ τ for any v with ‖v‖∞ ≤ v_inf (Step 7c)."""
    return _up(_tau_az(_nonneg("row_abs_sum_max", row_abs_sum_max), _nonneg("v_inf", v_inf)))


def residual_bound(kappa: float, r_inf: float, row_abs_sum_max: float, b_inf: float,
                   x_inf: float) -> float:
    """Guaranteed bound on ‖θ − x‖∞ (K): up(κ (R̂ + τ_r)) (Step 7b).

    kappa must be a guaranteed upper bound on ‖A⁻¹‖∞ (kappa_upper); r_inf = R̂ = max |fl(b − A x)|
    from the FP64 recomputation; x_inf = ‖x‖∞ of that iterate.
    """
    k = Fraction(float(kappa))
    if not (math.isfinite(float(kappa)) and k > 0):
        raise ValueError(f"kappa must be finite and > 0, got {kappa!r}")
    R = _nonneg("r_inf", r_inf)
    tau_r = _tau_r(_nonneg("row_abs_sum_max", row_abs_sum_max), _nonneg("b_inf", b_inf),
                   _nonneg("x_inf", x_inf))
    return _up(k * (R + tau_r))


def margin(stats: dict) -> Margin | None:
    """eps_total, delta_abs and the min(A z) threshold for one design, or None (no certificate).

    stats (float64 values from the FP64 computation): row_abs_sum_max (S, W/K, an upper bound, see
    row_abs_sum_bound), z_max, min_az (m̂ = min fl(A z), W), kappa (the κ̂ used to form c, K/W),
    b_inf (‖b‖∞, W), x_inf_max (a cap X on ‖x‖∞ at the certified iterate, K).
    """
    S, z_max = _stat(stats, "row_abs_sum_max"), _stat(stats, "z_max")
    m_hat, kappa = _stat(stats, "min_az"), _stat(stats, "kappa")
    b_inf, X = _stat(stats, "b_inf"), _stat(stats, "x_inf_max")
    tau_az = _tau_az(S, z_max)
    if m_hat < 2 * tau_az or m_hat <= 0:
        return None  # min(Az) is within rounding of zero: no certificate (§8.4)
    if kappa < Fraction(float(z_max / m_hat)) * (1 - U):  # κ̂ must be >= fl(z_max / m̂)
        raise ValueError(f"kappa {float(kappa)} is below fl(z_max / min_az) = "
                         f"{float(z_max / m_hat)}")
    rho = tau_az / m_hat
    tau_r = _tau_r(S, b_inf, X)
    eps = 1 / ((1 - U) ** 2 * (1 - rho)) - 1
    delta = kappa * tau_r / ((1 - U) * (1 - rho))
    return Margin(eps_total=_up(eps), delta_abs=_up(delta), min_az_threshold=_up(2 * tau_az),
                  x_inf_max=float(X))


def certified(c: float, m: Margin, x_inf: float | None = None) -> float:
    """c·(1 + eps_total) + delta_abs, every operation rounded up (K).

    c is fl(κ̂·‖r̂‖∞) from the FP64 residual; x_inf, if given, is ‖x‖∞ of the certified iterate
    and must not exceed the cap the margin was derived for.
    """
    if not (math.isfinite(c) and c >= 0):
        raise ValueError(f"certificate c must be finite and >= 0, got {c}")
    if x_inf is not None and x_inf > m.x_inf_max:
        raise ValueError(f"x_inf = {x_inf} exceeds the margin's cap {m.x_inf_max}: recompute it")
    up = math.inf
    one_eps = float(np.nextafter(1.0 + m.eps_total, up))  # each operation rounded up separately:
    a = float(np.nextafter(c * one_eps, up))  # one next-float step covers one rounding
    return float(np.nextafter(a + m.delta_abs, up))


def row_abs_sum_bound(A: sp.spmatrix) -> float:
    """S ≥ max_i Σ_j |a_ij| of the stored A (W/K), via fl(|A|·1) / (1 − γ_7), rounded up."""
    A = sp.csr_matrix(A, dtype=np.float64)
    if np.diff(A.indptr).max() > NNZ_ROW:
        raise ValueError(f"A has a row with more than {NNZ_ROW} non-zeros; the margin assumes "
                         "the 7-point layered stencil")
    s = float((abs(A) @ np.ones(A.shape[0])).max())
    return _up(Fraction(s) / (1 - gamma(NNZ_ROW)))


def certify_residual(A: sp.spmatrix, b: np.ndarray, x: np.ndarray, z: np.ndarray, *,
                     matvec: Callable[[np.ndarray], np.ndarray] | None = None,
                     kappa: float | None = None) -> float | None:
    """Guaranteed upper bound on ‖θ − x‖∞ (K) for A θ = b, from one FP64 pass, or None.

    z > 0 is any positive vector (e.g. certify.kappa.positive_z); matvec computes A·v in FP64
    (default: CSR A @ v; any order or FMA is covered). kappa defaults to fl(max z / min(A z)).
    None: z is not positive or min(A z) is within rounding of zero (solve to tolerance instead).
    """
    A = sp.csr_matrix(A, dtype=np.float64)
    b, x, z = (np.asarray(v, dtype=np.float64) for v in (b, x, z))
    for name, v in (("b", b), ("x", x), ("z", z)):
        if v.shape != (A.shape[0],) or not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must be a finite vector of length {A.shape[0]}")
    if z.min() <= 0:
        return None
    mv = matvec or (lambda v: A @ v)
    az = mv(z)
    m_hat, z_max = float(az.min()), float(z.max())
    if m_hat <= 0:
        return None
    k = z_max / m_hat if kappa is None else float(kappa)
    x_inf = float(np.abs(x).max())
    mrg = margin({"row_abs_sum_max": row_abs_sum_bound(A), "z_max": z_max, "min_az": m_hat,
                  "kappa": k, "b_inf": float(np.abs(b).max()), "x_inf_max": x_inf})
    if mrg is None:
        return None
    c = k * float(np.abs(b - mv(x)).max())
    return certified(c, mrg, x_inf=x_inf)
