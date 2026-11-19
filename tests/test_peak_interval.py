"""Certified peak interval and decision-sufficient audits (N3.3; CLAUDE.md §8.3, §8.4, B7).

The reference is a direct sparse solve (relative residual ~1e-13, stricter than the 1e-8 oracle of
§11.5) at small N, so that an interval narrower than the oracle's own error cannot be mistaken
for a violation. Warm starts emulate surrogate errors (§8.6): the exact field times a smooth
random factor of a few per cent.

κ̂ comes from z = A⁻¹1 by a direct solve and certify.margin.kappa_upper (Lemma 2(a) holds for any
z > 0; this z gives the tightest κ̂, so the intervals are the narrowest and B7 the hardest to
meet). w comes from a direct solve of A w = |r̂| + headroom; peak_interval verifies it either way.
κ̂ and w from the production V-cycle code (certify/kappa.py, pointwise.py) are tested in
test_certify_rigorous.py; the one comparison with pointwise.py here skips where pyamg cannot load.

Failure modes covered: an interval that misses the true peak at any iterate (B7, 200 boards,
both methods), an unverified w accepted, a stopping rule that decides on the wrong side of q_t,
an S⁺ below the true score (Lemma 1′), a live audit that differs from its E14 replay, and
iteration counts / stop reasons that are off by one check.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from certify import margin as mg
from certify import peak_interval as PI
from data.raster import rasterise
from data.sampler import design_board, sample_board
from experiments.run_gates import CertificateViolation, Trace, certified_audit
from gate.proposers import sample_p0
from solvers.fvm import assemble_board, coefficients
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 32
N_BOARDS = 200


@cache
def board(i: int):
    """Board i: even i a sampler board (ID mode, Dirichlet edges included), odd i a p₀ design."""
    if i % 2 == 0:
        rec = sample_board(1000 + i, "id")
    else:
        rec = design_board(sample_p0(np.random.Generator(np.random.PCG64(i))))
    r = rasterise(rec, SPEC, N)
    edges = rec["dirichlet_edges"]
    A, bflat = assemble_board(SPEC, r, edges)
    c = coefficients(SPEC, r, edges)
    lu = spla.splu(A.tocsc())
    theta = lu.solve(bflat).reshape(4, N, N)
    z = lu.solve(np.ones(A.shape[0]))
    assert z.min() > 0 and (A @ z).min() > 0
    k = mg.kappa_upper(mg.row_abs_sum_bound(A), float(z.max()), float((A @ z).min()))
    assert k is not None
    return A, c, c["rhs"], theta, k


def pointwise_w(A, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """w ≈ A⁻¹(|r̂| + 4τ) by a direct solve (A⁻¹ ≥ 0, so w ≥ 0 up to rounding)."""
    S = mg.row_abs_sum_bound(A)
    abs_r = np.abs(b.ravel() - A @ x.ravel())
    w0 = spla.spsolve(A.tocsc(), abs_r)
    tau = mg.residual_tau(S, float(np.abs(b).max()), float(np.abs(x).max())) + mg.matvec_tau(
        S, 2 * float(w0.max()))
    w = np.maximum(spla.spsolve(A.tocsc(), abs_r + 4 * tau), 0.0)
    return w.reshape(x.shape)


def emulated(theta: np.ndarray, i: int, rel: float = 0.03) -> np.ndarray:
    """θ times a smooth random factor 1 + rel·f, |f| ≤ 1 (an emulated surrogate error)."""
    rng = np.random.Generator(np.random.PCG64(10_000 + i))
    y, x = np.mgrid[0:N, 0:N] / N
    f = np.zeros((4, N, N))
    for _ in range(4):
        a, ph = rng.uniform(-1, 1), rng.uniform(0, 6)
        kx, ky = rng.integers(1, 4, size=2)
        f += a * np.sin(2 * np.pi * (kx * x + ky * y) + ph)[None]
    return theta * (1 + rel * f / max(np.abs(f).max(), 1e-12))


# ----------------------------------------------------------------------------------- B7

def test_b7_every_check_contains_the_true_peak_on_200_boards():
    violations, checks = [], 0
    for i in range(N_BOARDS):
        _, c, b, theta, k = board(i)
        peak = float(theta.max())
        tr = PI.audit_trace(emulated(theta, i), c, b, k, check_every=5)
        for it, lo, hi in zip(tr.iters, tr.T_lo, tr.T_hi, strict=True):
            checks += 1
            if not lo <= peak <= hi:
                violations.append((i, it, lo, peak, hi))
    print(f"B7: {len(violations)} violations in {checks} checks on {N_BOARDS} boards")
    assert violations == [], violations[:5]
    assert checks > 5 * N_BOARDS


def test_b7_pointwise_and_combined_methods_on_200_boards():
    violations = 0
    for i in range(N_BOARDS):
        A, c, b, theta, k = board(i)
        mu = emulated(theta, i)
        w = pointwise_w(A, b, mu)
        peak = float(theta.max())
        for kw in ({"w": w}, {"w": w, "kappa_hat": k}):
            lo, hi, _ = PI.peak_interval(mu, c, b, **kw)
            violations += not lo <= peak <= hi
    print(f"B7 (Lemma 2(c) and combined): {violations} violations")
    assert violations == 0


def test_width_shrinks_on_average_along_iterates():
    K = 12  # checks compared: iterates 0, 5, ..., 55
    widths = np.empty((N_BOARDS, K))
    for i in range(N_BOARDS):
        _, c, b, theta, k = board(i)
        tr = PI.audit_trace(emulated(theta, i), c, b, k, check_every=5)
        w = np.asarray(tr.T_hi) - np.asarray(tr.T_lo)
        widths[i] = np.concatenate([w, np.full(max(K - len(w), 0), w[-1])])[:K]
    mean = widths.mean(axis=0)
    print("mean width (K) at iterates 0, 5, ...:", np.array2string(mean, precision=3))
    assert np.all(np.diff(mean) < 0), mean
    assert mean[-1] < 0.01 * mean[0]


# ------------------------------------------------------------------------- peak_interval

def test_kappa_interval_matches_the_stopping_bound():
    A, c, b, theta, k = board(0)
    mu = emulated(theta, 0)
    lo, hi, method = PI.peak_interval(mu, c, b, kappa_hat=k)
    r = b - PI.apply_A(c, mu)  # the same FP64 residual peak_interval forms
    cb = mg.residual_bound(k, float(np.abs(r).max()), PI.row_abs_sum_bound(c),
                           float(np.abs(b).max()), float(np.abs(mu).max()))
    assert method == "kappa"
    assert lo <= mu.max() - cb and hi >= mu.max() + cb
    assert np.isclose(hi - lo, 2 * cb, rtol=1e-12)


def test_w_interval_matches_pointwise_and_combined_is_the_intersection():
    pw = pytest.importorskip("certify.pointwise", exc_type=ImportError,
                             reason="pyamg cannot load here (needed by certify/pointwise.py)")
    A, c, b, theta, k = board(1)
    mu = emulated(theta, 1)
    w = pw.pointwise_bound(A, b.ravel(), mu.ravel()).reshape(mu.shape)
    lo, hi, method = PI.peak_interval(mu, c, b, w=w)
    assert method == "pointwise" and (lo, hi) == pw.peak_interval(mu.ravel(), w.ravel())
    lk, hk, _ = PI.peak_interval(mu, c, b, kappa_hat=k)
    lb, hb, mb = PI.peak_interval(mu, c, b, kappa_hat=k, w=w)
    assert mb == "both" and (lb, hb) == (max(lo, lk), min(hi, hk))


def test_combined_is_the_intersection():
    A, c, b, theta, k = board(1)
    mu = emulated(theta, 1)
    w = pointwise_w(A, b, mu)
    lw, hw, mw = PI.peak_interval(mu, c, b, w=w)
    lk, hk, mk = PI.peak_interval(mu, c, b, kappa_hat=k)
    lb, hb, mb = PI.peak_interval(mu, c, b, kappa_hat=k, w=w)
    assert (mw, mk, mb) == ("pointwise", "kappa", "both")
    assert (lb, hb) == (max(lw, lk), min(hw, hk))


def test_unverified_w_is_refused():
    A, c, b, theta, _ = board(2)
    mu = emulated(theta, 2)
    w = pointwise_w(A, b, mu)
    PI.peak_interval(mu, c, b, w=w)  # the constructed w passes
    with pytest.raises(ValueError, match=r"A w >= \|r\|"):
        PI.peak_interval(mu, c, b, w=0.5 * w)
    with pytest.raises(ValueError, match="w must be >= 0"):
        PI.peak_interval(mu, c, b, w=w - 2 * w.max())


def test_bad_arguments_are_refused():
    _, c, b, theta, k = board(0)
    with pytest.raises(ValueError, match="kappa_hat or w"):
        PI.peak_interval(theta, c, b)
    with pytest.raises(ValueError, match="kappa_hat"):
        PI.peak_interval(theta, c, b, kappa_hat=-1.0)
    with pytest.raises(ValueError, match="shape"):
        PI.peak_interval(theta[:2], c, b, kappa_hat=k)
    bad = dict(c, Ge=-c["Ge"])
    with pytest.raises(ValueError, match="M-matrix"):
        PI.peak_interval(theta, bad, b, kappa_hat=k)
    for kw, msg in (({"s": 0.0}, "s must"), ({"eps_S": 0.0}, "eps_S"),
                    ({"check_every": 0}, "check_every"), ({"kappa_hat": None}, "kappa_hat")):
        args = {"s": 1.0, "q_t": 1.0, "eps_S": 0.1, "kappa_hat": k, **kw}
        with pytest.raises(ValueError, match=msg):
            PI.audit_until_decided(theta, c, b, **args)


def test_stencil_matvec_matches_the_assembled_matrix():
    A, c, _, theta, _ = board(3)
    x = emulated(theta, 3)
    S = PI.row_abs_sum_bound(c)
    assert S >= mg.row_abs_sum_bound(A) * (1 - 1e-15)
    # two FP64 evaluations of the same exact A x, each within matvec_tau of it
    diff = np.abs(PI.apply_A(c, x).ravel() - A @ x.ravel())
    assert diff.max() <= 2 * mg.matvec_tau(S, float(np.abs(x).max()))


def test_score_interval_is_rounded_outward_and_contains_the_score():
    rng = np.random.Generator(np.random.PCG64(5))
    for _ in range(2000):
        lo = rng.uniform(0, 80)
        hi = lo + rng.exponential(0.5)
        t, peak_hat, s = rng.uniform(lo, hi), rng.uniform(0, 80), rng.uniform(0.01, 3)
        s_lo, s_hi = PI.score_interval(peak_hat, lo, hi, s)
        assert s_lo <= abs(peak_hat - t) / s <= s_hi


# ------------------------------------------------------------------------------- audits

def test_audit_decisions_are_on_the_true_side_of_q():
    wrong = 0
    for i in range(0, N_BOARDS, 4):
        _, c, b, theta, k = board(i)
        mu = emulated(theta, i)
        peak_hat, peak = float(mu.max()), float(theta.max())
        s = 0.5
        S = abs(peak_hat - peak) / s
        for q in (0.25 * S, 0.9 * S, 1.1 * S, 4 * S):
            lo, hi, s_lo, s_hi, iters, why = PI.audit_until_decided(mu, c, b, s, q, 1e-6, k)
            assert lo <= peak <= hi and s_lo <= S <= s_hi  # B7 and Lemma 1′ (S⁺ ≥ S)
            if why == "below_q":
                wrong += S > q
            elif why == "above_q":
                wrong += S <= q
            assert why in ("below_q", "above_q", "width", "converged")
    assert wrong == 0


def test_audit_stop_reasons():
    _, c, b, theta, k = board(4)
    mu = emulated(theta, 4)
    r = PI.audit_until_decided(mu, c, b, 1.0, float("inf"), 0.01, k)
    assert r[4] == 0 and r[5] == "below_q"
    lo, hi, _ = PI.peak_interval(mu, c, b, kappa_hat=k)
    s_lo, s_hi = PI.score_interval(float(mu.max()), lo, hi, 1.0)
    assert s_lo < s_hi
    mid = 0.5 * (s_lo + s_hi)  # undecided at iterate 0, so only the width rule can stop there
    r = PI.audit_until_decided(mu, c, b, 1.0, mid, 1e9, k)
    assert r[4] == 0 and r[5] == "width"
    r = PI.audit_until_decided(theta + 5.0, c, b, 1.0, 0.0, 1e-9, k)
    assert r[5] == "above_q" and r[2] > 0
    tr = PI.audit_trace(mu, c, b, k, check_every=5, maxit=7)
    assert tr.iters == [0, 5, 7] and tr.stop_reason == "maxit"
    r = PI.audit_until_decided(mu, c, b, 1.0, 0.0, 1e-30, k, maxit=7)
    assert (r[5] == "maxit" and r[4] == 7) or (r[5] == "above_q" and r[4] in (0, 5, 7))
    r = PI.audit_until_decided(theta, c, b, 1.0, 0.0, 1e-30, k)
    assert r[5] == "converged" and r[0] <= theta.max() <= r[1]


def test_audits_stop_early_on_average():
    full, used = 0, 0
    for i in range(0, N_BOARDS, 4):
        _, c, b, theta, k = board(i)
        mu = emulated(theta, i)
        tr = PI.audit_trace(mu, c, b, k, check_every=5)
        s = 0.5
        S = abs(float(mu.max()) - float(theta.max())) / s
        used += PI.audit_until_decided(mu, c, b, s, 2 * S, 0.05, k)[4]
        full += tr.iters[-1]
    print(f"audit iterations: {used} certified-decided vs {full} to 1e-8")
    assert used < full


def test_live_audit_equals_the_e14_replay_of_its_trace():
    for i in range(0, 40, 2):
        _, c, b, theta, k = board(i)
        mu = emulated(theta, i)
        tr = PI.audit_trace(mu, c, b, k, check_every=5)
        trace = Trace(np.asarray(tr.iters), np.asarray(tr.T_lo), np.asarray(tr.T_hi))
        peak_hat, s = float(mu.max()), 0.3
        S = abs(peak_hat - float(theta.max())) / s
        for q, eps in ((0.5 * S, 0.01), (2 * S, 0.01), (S, 0.2), (S, 1e-9)):
            lo, hi, _, _, iters, why = PI.audit_until_decided(mu, c, b, s, q, eps, k)
            rlo, rhi, rit, rwhy = certified_audit(peak_hat, s, q, eps, trace, float(theta.max()))
            assert (lo, hi, iters) == (rlo, rhi, rit)
            assert rwhy == {"below_q": "decided", "above_q": "decided", "width": "width",
                            "converged": "end", "maxit": "end"}[why]


def test_replay_raises_on_a_trace_that_misses_the_oracle():
    trace = Trace(np.array([0, 5]), np.array([10.0, 10.5]), np.array([12.0, 11.0]))
    with pytest.raises(CertificateViolation, match="B7"):
        certified_audit(11.0, 1.0, 0.1, 1e-6, trace, 11.2)
