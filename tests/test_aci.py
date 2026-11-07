"""Audited ACI gate G5 (CLAUDE.md §8.1–§8.3, §9.3, §18.1; decisions D7, D8) and the label-free
scale (gate/scale.py)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from contracts.validate import validate
from data.raster import rasterise
from gate.aci_audit import AuditedACIGate, conformal_quantile
from gate.base import AuditRecord, GateInput, certified_score
from gate.coin import AuditCoin
from gate.scale import apply_stencil, label_free_scale
from solvers.fvm import assemble_board, coefficients
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]


def _gate(coin, *, alpha=0.1, gamma=0.01, W=500, tau=1.0, p=0.2, cal=None):
    cal = np.random.default_rng(0).exponential(size=1000) if cal is None else cal
    return AuditedACIGate(alpha=alpha, gamma=gamma, W=W, tau=tau, p=p, cal_scores=cal, coin=coin)


def _input(i: int, s: float, peak: float) -> GateInput:
    return GateInput(design_id=f"d{i}", x_hash=hashlib.sha256(f"d{i}".encode()).hexdigest(),
                     peak_hat=peak, s=s, r_hot=s / 2, sd_hot=s / 2, proposer="P2",
                     aggressiveness=3, group=None)


def _lemma1_C(alpha1: float, gamma: float) -> float:
    return max(alpha1, 1 - alpha1) + gamma


def _run(gate: AuditedACIGate, coin: AuditCoin, T: int, error_fn, rng, s_fn=None, width=0.0):
    """Drive the loop of §8.1 for T steps. error_fn(t) -> signed true peak error (K) of step t.

    Returns (accepted_large_error_count, audited err list, alpha_t trajectory)."""
    bad, errs, alphas = 0, [], []
    for t in range(T):
        s = 1.0 if s_fn is None else s_fn(t)
        peak_hat = 50.0
        true_peak = peak_hat - error_fn(t)
        dec = gate.decide(_input(t, s, peak_hat))
        if dec.decision == "accept" and abs(peak_hat - true_peak) > gate.tau:
            bad += 1
        coin.commit(f"d{t}")
        if coin.draw(f"d{t}", dec.p_audit):
            lo = true_peak - width * rng.random()
            hi = true_peak + width * rng.random()
            (out,) = gate.update([AuditRecord(design_id=f"d{t}", audited=True, peak_hat=peak_hat,
                                              s=s, T_lo=lo, T_hi=hi, iters_used=1)])
            errs.append(out.err)
            alphas.append(gate.alpha_t)
    return bad, errs, alphas


# ------------------------------------------------------------- test 3: Prop. 4.1 on the audits


def test_iid_scores_miscoverage_on_audits_within_C_over_gamma_na() -> None:
    rng = np.random.default_rng(1)
    coin = AuditCoin(seed=1)
    gate = _gate(coin, alpha=0.1, gamma=0.005, p=0.2)
    _, errs, alphas = _run(gate, coin, 20_000, lambda t: rng.exponential(), rng)
    n_a = len(errs)
    C = _lemma1_C(0.1, 0.005)
    for k in range(1, n_a + 1):  # deterministic: after EVERY audit, not just at the end
        assert abs(np.mean(errs[:k]) - 0.1) <= C / (0.005 * k) + 1e-12
    assert min(alphas) >= -0.005 - 1e-12 and max(alphas) <= 1 + 0.005 + 1e-12


def test_adversarial_scores_keep_the_deterministic_bound_and_alpha_range() -> None:
    # The adversary sees q_t (but not the coin) and puts the score just above or below it.
    coin = AuditCoin(seed=2)
    gamma = 0.05
    gate = _gate(coin, alpha=0.2, gamma=gamma, W=50, p=0.5)
    errs, alphas = [], []
    for t in range(5_000):
        q = gate.q_t
        dec = gate.decide(_input(t, 1.0, 50.0))
        miss = (t // 7) % 2 == 0
        if math.isfinite(q):
            S = q + 1.0 if miss else 0.5 * q
        else:
            S = 1e6 if miss else 0.0
        coin.commit(f"d{t}")
        if coin.draw(f"d{t}", dec.p_audit):
            T = 50.0 - S
            (out,) = gate.update([AuditRecord(f"d{t}", True, 50.0, 1.0, T, T, 1)])
            errs.append(out.err)
            alphas.append(gate.alpha_t)
    C = _lemma1_C(0.2, gamma)
    for k in range(1, len(errs) + 1):
        assert abs(np.mean(errs[:k]) - 0.2) <= C / (gamma * k) + 1e-12
    assert min(alphas) >= -gamma - 1e-12 and max(alphas) <= 1 + gamma + 1e-12


def test_boundaries_q_inf_below_zero_and_err_forced_at_one() -> None:
    coin = AuditCoin(seed=3)
    gate = _gate(coin, alpha=0.1, gamma=0.5, W=10, p=1.0, cal=np.ones(10))
    # two zero-error audits push alpha_t up: 0.1 -> 0.15 -> ... ; keep feeding S = 0
    seen_ge1 = False
    for t in range(40):
        gate.decide(_input(t, 1.0, 50.0))
        coin.commit(f"d{t}")
        assert coin.draw(f"d{t}", 1.0)
        a = gate.alpha_t
        (out,) = gate.update([AuditRecord(f"d{t}", True, 50.0, 1.0, 50.0, 50.0, 1)])  # S_hi = 0
        if a >= 1:
            seen_ge1 = True
            assert out.err == 1  # D7: empty set, err = 1 even though S_hi = 0
            assert out.q_t == 0.0
        elif a <= 0:
            assert out.err == 0 and out.q_t == math.inf
    assert seen_ge1
    # far below zero: q = inf, so every prediction is flagged (B = inf > tau)
    gate2 = _gate(AuditCoin(seed=4), alpha=0.1, gamma=0.5, W=10, p=1.0, cal=np.ones(10))
    gate2.load_state_dict({**gate2.state_dict(), "alpha_t": -0.2})
    assert gate2.q_t == math.inf
    dec = gate2.decide(_input(0, 1.0, 50.0))
    assert dec.B == math.inf and dec.decision == "flag"


# ------------------------------------------------------------------- test 4: Lemma 1' (S_hi ≥ S)


def test_certified_score_upper_bounds_the_true_score() -> None:
    rng = np.random.default_rng(4)
    for _ in range(10_000):
        T = rng.normal(40, 20)
        T_hat = T + rng.normal(0, 5)
        lo = T - rng.exponential(2) * rng.integers(0, 2)
        hi = T + rng.exponential(2) * rng.integers(0, 2)
        s = rng.uniform(1e-3, 3)
        S = abs(T_hat - T) / s
        assert certified_score(T_hat, lo, hi, s) >= S - 1e-12 * max(1.0, S)


@pytest.mark.parametrize("bad", [(41.0, 40.0, 1.0), (40.0, 41.0, 0.0), (40.0, float("nan"), 1.0)])
def test_certified_score_rejects_bad_intervals(bad) -> None:
    lo, hi, s = bad
    with pytest.raises(ValueError):
        certified_score(40.0, lo, hi, s)


# ------------------------------------------------------- test 5: adversarial drift, G5 vs static


def test_drift_g5_stays_within_lemma1_slack_and_static_split_conformal_does_not() -> None:
    T, alpha, gamma, p, delta, tau = 40_000, 0.1, 0.01, 0.1, 0.05, 1.0
    rng = np.random.default_rng(5)
    cal = rng.uniform(0, 1, size=1000)  # calibration world: |error| / s ~ U[0, 1]
    s_fn = lambda t: 0.5 + (t * 0.618034) % 1.0  # noqa: E731  label-free scale in [0.5, 1.5)

    def error(t: int) -> float:  # after 10 % of the run the surrogate is 10x worse
        u = rng.uniform(0, 1) if t < T // 10 else rng.uniform(0, 10)
        return u * s_fn(t) * (1 if t % 2 else -1)

    coin = AuditCoin(seed=5)
    g5 = AuditedACIGate(alpha=alpha, gamma=gamma, W=500, tau=tau, p=p, cal_scores=cal, coin=coin)
    bad, errs, _ = _run(g5, coin, T, error, rng, s_fn=s_fn, width=0.2)
    slack = _lemma1_C(alpha, gamma) / (gamma * p * T) + (2 / p) * math.sqrt(math.log(4 / delta) / T)
    W_T = bad / T
    assert W_T <= alpha + slack, f"G5 W_T = {W_T:.3f} > {alpha + slack:.3f}"

    # static split conformal (G2-like): q from the calibration scores once, never updated;
    # fresh errors from the same drifting process
    q_static = conformal_quantile(cal, alpha)
    bad_static = sum(1 for t in range(T)
                     if q_static * s_fn(t) <= tau and abs(error(t)) > tau)
    assert bad_static / T > alpha + slack, f"static W_T = {bad_static / T:.3f}"


# ----------------------------------------------------------------------- quantile (D8)


def test_conformal_quantile_is_the_split_conformal_level_with_higher_interpolation() -> None:
    x = np.arange(1, 20, dtype=float)  # n = 19 scores 1..19
    assert conformal_quantile(x, 0.05) == 19.0  # ceil(20 * 0.95) = 19 -> the 19th smallest
    assert conformal_quantile(x, 0.1) == 18.0  # ceil(20 * 0.9) = 18
    assert conformal_quantile(x, 0.04) == math.inf  # ceil(20 * 0.96) = 20 > n
    assert conformal_quantile(x[::-1].copy(), 0.1) == 18.0  # order does not matter
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.1)
    with pytest.raises(ValueError):
        conformal_quantile(np.array([1.0, np.nan]), 0.1)


def test_calibration_coverage_matches_the_beta_binomial_mean() -> None:
    # §18.2: re-split cached scores; mean coverage ≈ 1 − l/(n+1), l = floor((n+1) α)
    rng = np.random.default_rng(6)
    n, alpha = 99, 0.1
    cover = []
    for _ in range(4_000):
        sc = rng.exponential(size=n + 200)
        q = conformal_quantile(sc[:n], alpha)
        cover.append(np.mean(sc[n:] <= q))
    l_ = math.floor((n + 1) * alpha)
    assert abs(np.mean(cover) - (1 - l_ / (n + 1))) < 0.005


def test_initial_q_comes_from_the_calibration_scores() -> None:
    cal = np.random.default_rng(7).exponential(size=1000)
    g = _gate(AuditCoin(seed=7), alpha=0.1, W=200, cal=cal)
    assert g.q_t == conformal_quantile(cal, 0.1)  # all calibration scores at the start (§9.3)
    assert g.alpha_t == 0.1


# ------------------------------------------------------------------- constructor, state, logs


def test_hyperparameters_have_no_defaults_and_are_checked() -> None:
    cal = np.ones(10)
    coin = AuditCoin(seed=0)
    with pytest.raises(TypeError):
        AuditedACIGate(cal_scores=cal, coin=coin)  # D4: nothing silently becomes "the" value
    ok = dict(alpha=0.1, gamma=0.01, W=10, tau=1.0, p=0.1, cal_scores=cal, coin=coin)
    for k, v in [("alpha", 0.6), ("alpha", 0.0), ("gamma", 0.0), ("W", 0), ("W", 2.5),
                 ("tau", -1.0), ("p", 0.0), ("p", 1.5)]:
        with pytest.raises(ValueError, match=k):
            AuditedACIGate(**{**ok, k: v})
    with pytest.raises(ValueError, match="cal_scores"):
        AuditedACIGate(**{**ok, "cal_scores": np.array([1.0, -1.0])})


def test_state_dict_round_trip_reproduces_the_run() -> None:
    def make():
        coin = AuditCoin(seed=8)
        return coin, _gate(coin, p=0.5, W=50)

    rng = np.random.default_rng(8)
    errors = rng.exponential(size=400)
    c1, g1 = make()
    _run(g1, c1, 200, lambda t: errors[t], rng)
    c2, g2 = make()
    g2.load_state_dict(json.loads(json.dumps(g1.state_dict())))  # survives JSON
    c2.load_state_dict(json.loads(json.dumps(c1.state_dict())))
    for t in range(200, 400):
        d1, d2 = g1.decide(_input(t, 1.0, 50.0)), g2.decide(_input(t, 1.0, 50.0))
        assert d1 == d2
        for c, g in ((c1, g1), (c2, g2)):
            c.commit(f"d{t}")
            if c.draw(f"d{t}", g.p):
                T = 50.0 - errors[t]
                g.update([AuditRecord(f"d{t}", True, 50.0, 1.0, T, T, 1)])
    assert g1.state_dict() == g2.state_dict()
    with pytest.raises(ValueError, match="gamma"):
        _gate(AuditCoin(seed=0), gamma=0.02).load_state_dict(g1.state_dict())


def test_log_row_is_a_valid_audit_log_row() -> None:
    coin = AuditCoin(seed=9)
    g = _gate(coin, p=1.0)
    inp = _input(0, 0.4, 50.0)
    dec = g.decide(inp)
    coin.commit("d0")
    assert coin.draw("d0", dec.p_audit)
    (out,) = g.update([AuditRecord("d0", True, 50.0, 0.4, 49.0, 49.5, 37)])
    row = g.log_row(inp, dec, run_id="g5-abc1234-0", seed=0, t=0, outcome=out, iters_used=37,
                    oracle_peak=49.2)
    validate(row, "audit_log_row")
    assert row["gate"] == "G5" and row["audited"] is True and row["err"] == out.err
    assert row["S_hi"] == pytest.approx(max(1.0, 0.5) / 0.4)
    assert row["alpha_t"] == dec.alpha_t and row["q_t"] == dec.q_t
    # an unaudited row: interval, score and err empty
    inp1 = _input(1, 0.4, 50.0)
    row1 = g.log_row(inp1, g.decide(inp1), run_id="g5-abc1234-0", seed=0, t=1, outcome=None,
                     iters_used=0)
    validate(row1, "audit_log_row")
    assert row1["audited"] is False and row1["T_lo"] is None and row1["err"] is None
    # q = inf is logged as null (schema: null = +inf)
    g.load_state_dict({**g.state_dict(), "alpha_t": -0.5})
    inp2 = _input(2, 0.4, 50.0)
    row2 = g.log_row(inp2, g.decide(inp2), run_id="g5-abc1234-0", seed=0, t=2, outcome=None,
                     iters_used=0)
    validate(row2, "audit_log_row")
    assert row2["q_t"] is None and row2["B"] is None and row2["decision"] == "flag"


# ---------------------------------------------------------------------------- label-free scale


@pytest.fixture(scope="module")
def board():
    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    rec = json.loads((ROOT / "contracts/examples/board_record/valid_2.json").read_text("utf-8"))
    raster = rasterise(rec, spec, 16)
    c = coefficients(spec, raster, rec["dirichlet_edges"])
    A, b = assemble_board(spec, raster, rec["dirichlet_edges"])
    return c, A, b


def test_stencil_matches_the_assembled_matrix(board) -> None:
    c, A, b = board
    x = np.random.default_rng(10).normal(size=c["diag"].shape)
    np.testing.assert_allclose(apply_stencil(c, x).ravel(), A @ x.ravel(), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(c["rhs"].ravel(), b)


def test_scale_is_zero_residual_at_the_exact_solution_and_kelvin_otherwise(board) -> None:
    import scipy.sparse.linalg as spla

    c, A, b = board
    theta = spla.spsolve(A.tocsc(), b).reshape(c["diag"].shape)
    sd = np.zeros_like(theta)
    s, r_hot, sd_hot = label_free_scale(theta, sd, c, c["rhs"], m=64, lam=1.0, eps=1e-6)
    assert r_hot < 1e-9 and sd_hot == 0 and s == pytest.approx(1e-6, abs=1e-9)
    # shift the field by 1 K: the kelvin residual (A·1)/diag lies in [0, 1] per cell
    s1, r1, _ = label_free_scale(theta + 1.0, sd, c, c["rhs"], m=64, lam=1.0, eps=1e-6)
    assert 0 < r1 <= 1.0
    # hot cells are the m hottest of mu over ALL layers; sd is averaged over the same cells
    sd2 = np.zeros_like(theta)
    flat = np.argsort(-theta.ravel(), kind="stable")[:64]
    sd2.ravel()[flat] = 2.0
    _, _, sdh = label_free_scale(theta, sd2, c, c["rhs"], m=64, lam=0.5, eps=1e-6)
    assert sdh == pytest.approx(2.0)
    s3, r3, sd3 = label_free_scale(theta, sd2, c, c["rhs"], m=64, lam=0.5, eps=1e-6)
    assert s3 == pytest.approx(r3 + 0.5 * sd3 + 1e-6)


@pytest.mark.parametrize("kw,match", [
    ({"m": 0}, "m"), ({"eps": 0.0}, "eps"), ({"lam": -1.0}, "lam"), ({"m": 10**6}, "m"),
])
def test_scale_rejects_bad_arguments(board, kw, match) -> None:
    c, _, _ = board
    mu = np.ones(c["diag"].shape)
    args = {"m": 64, "lam": 1.0, "eps": 1e-6, **kw}
    with pytest.raises(ValueError, match=match):
        label_free_scale(mu, np.zeros_like(mu), c, c["rhs"], **args)
    with pytest.raises(ValueError, match="shape"):
        label_free_scale(mu[:2], np.zeros_like(mu[:2]), c, c["rhs"], m=64, lam=1.0, eps=1e-6)
    with pytest.raises(ValueError, match="sd"):
        label_free_scale(mu, -np.ones_like(mu), c, c["rhs"], m=64, lam=1.0, eps=1e-6)
