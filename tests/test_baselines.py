"""Comparison gates G0-G4, G6, G7 (prompt M3.2; CLAUDE.md §8.5, §11.1, §18.2-§18.4; D7, D8, D9,
D13). One synthetic test of each gate's defining behaviour, plus the protocol rules every gate
shares: decide() is pure (B8), commit() is once per design, labels reach only the gates and
designs they are allowed to reach.

Sources: docs/refs/deepoheat_v2_alg1.txt (G1), docs/refs/angelopoulos_bates_cp.txt (G2, G3),
docs/refs/bastani2022_mvp.txt (G6). G4 and G7 are CLAUDE.md §11.1 / §8.5.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from contracts.validate import validate
from gate import active_audit as aa
from gate import baselines as bl
from gate import multivalid as mv
from gate.aci_audit import AuditedACIGate, conformal_quantile
from gate.base import AuditRecord, GateInput
from gate.coin import AuditCoin
from gate.scale import apply_stencil, hot_residual_raw

ALPHA = 0.1


def _inp(i: int, s: float = 1.0, r_hot: float = 0.1, peak: float = 50.0, r_hot_raw=None,
         group=None) -> GateInput:
    return GateInput(design_id=f"d{i}", x_hash=hashlib.sha256(f"d{i}".encode()).hexdigest(),
                     peak_hat=peak, s=s, r_hot=r_hot, sd_hot=0.0, proposer="P1",
                     aggressiveness=1, group=group, r_hot_raw=r_hot_raw)


def _rec(i: int, S: float, s: float = 1.0, peak: float = 50.0, audited: bool = True):
    """A record whose certified interval is a point at the true peak, so S⁺ = S exactly."""
    return AuditRecord(design_id=f"d{i}", audited=audited, peak_hat=peak, s=s,
                       T_lo=peak - S * s, T_hi=peak - S * s, iters_used=1)


# ====================================================================== G0


def test_g0_accepts_everything_and_takes_no_labels() -> None:
    g = bl.NoGate()
    for i, s in enumerate([1e-3, 1.0, 1e6]):
        d = g.commit(_inp(i, s=s))
        assert d.decision == "accept" and d.p_audit == 0.0 and math.isinf(d.B)
    with pytest.raises(ValueError, match="G0"):
        g.update([_rec(0, 0.5)])
    assert g.update([]) == []


# ====================================================================== G1


def _g1(**kw):
    args = {"f": 0.3, "window": 100, "residual": "scaled", "c_events": None, "delta_n": None}
    args.update(kw)
    return bl.DeepOHeatV2Gate(**args)


def test_g1_flag_rate_is_f_after_the_bootstrap() -> None:
    g, rng = _g1(), np.random.default_rng(0)
    flags = [g.commit(_inp(i, r_hot=float(r))).decision == "flag"
             for i, r in enumerate(rng.exponential(size=20_100))]
    assert not any(flags[:100])  # Algorithm 1 line 2: |W| no-flag bootstrap steps
    assert abs(np.mean(flags[100:]) - 0.3) < 0.015


def test_g1_window_includes_the_current_value_and_uses_the_higher_quantile() -> None:
    g = _g1(f=0.3, window=10)
    for i in range(10):  # bootstrap: values 1..10
        g.commit(_inp(i, r_hot=float(i + 1)))
    # push 10.5: window = 2..10, 10.5; ⌈0.7·10⌉ = 7th smallest = 8; 10.5 > 8 flags
    assert g.commit(_inp(10, r_hot=10.5)).decision == "flag"
    # push 0.5: window = 3..10, 10.5, 0.5; 7th smallest = 8; 0.5 does not flag
    assert g.commit(_inp(11, r_hot=0.5)).decision == "accept"


def test_g1_preview_matches_commit_and_changes_nothing() -> None:
    g, rng = _g1(window=20), np.random.default_rng(1)
    for i in range(60):
        inp = _inp(i, r_hot=float(rng.exponential()))
        before = g.state_dict()
        preview = [g.decide(inp) for _ in range(3)]
        assert g.state_dict() == before and preview[0] == preview[1] == preview[2]
        assert g.commit(inp) == preview[0]
    with pytest.raises(ValueError, match="already committed"):
        g.commit(_inp(5))


def test_g1_c_rule_stops_gating_after_c_events() -> None:
    g, rng = _g1(c_events=3, delta_n=50), np.random.default_rng(2)
    decisions = [g.commit(_inp(i, r_hot=float(r))).decision
                 for i, r in enumerate(rng.exponential(size=3000))]
    assert decisions.count("flag") == 150  # C · ΔN refinements, then the block is skipped
    last_flag = max(i for i, d in enumerate(decisions) if d == "flag")
    assert g.events == 3 and all(d == "accept" for d in decisions[last_flag + 1:])


def test_g1_raw_residual_switch_reads_r_hot_raw() -> None:
    g = _g1(residual="raw", window=5)
    for i in range(5):
        g.commit(_inp(i, r_hot=1.0, r_hot_raw=float(i)))
    assert g.commit(_inp(5, r_hot=0.0, r_hot_raw=100.0)).decision == "flag"
    with pytest.raises(ValueError, match="r_hot_raw"):
        g.decide(_inp(6, r_hot=1.0))


@pytest.mark.parametrize("kw", [{"f": 0.0}, {"f": 1.0}, {"window": 0}, {"residual": "W"},
                                {"c_events": 3}, {"c_events": 0, "delta_n": 50}])
def test_g1_bad_settings_raise(kw) -> None:
    with pytest.raises(ValueError):
        _g1(**kw)


def test_g1_uses_no_labels_and_budget_helper() -> None:
    with pytest.raises(ValueError, match="G1"):
        _g1().update([_rec(0, 0.5)])
    assert bl.flag_rate_for_budget(300, 1000) == 0.3
    with pytest.raises(ValueError):
        bl.flag_rate_for_budget(0, 1000)


def test_hot_residual_raw_is_unscaled_mean_over_the_hottest_cells() -> None:
    rng = np.random.default_rng(3)
    shape = (4, 6, 6)
    c = {k: rng.uniform(0.1, 1.0, shape) for k in ("Ge", "Gw", "Gn", "Gs", "gup", "gdn")}
    c["diag"] = sum(c.values()) + 0.5
    mu, b = rng.uniform(0, 50, shape), rng.uniform(0, 1, shape)
    r = (apply_stencil(c, mu) - b).reshape(-1)
    hot = np.argsort(-mu.reshape(-1), kind="stable")[:7]
    assert hot_residual_raw(mu, c, b, m=7) == pytest.approx(np.abs(r[hot]).mean(), rel=1e-15)


# ====================================================================== G2


def test_g2_coverage_check_matches_the_beta_binomial_law() -> None:
    """Appendix C of docs/refs/angelopoulos_bates_cp.txt: re-split cached scores R times."""
    rng = np.random.default_rng(4)
    scores = rng.exponential(size=2000)
    n, n_val, R = 1000, 1000, 400
    cov = []
    for _ in range(R):
        perm = rng.permutation(scores)
        g = bl.StaticSplitConformal(alpha=ALPHA, tau=1.0, cal_scores=perm[:n])
        cov.append(np.mean(perm[n:] <= g.q_t))
    ell = math.floor((n + 1) * ALPHA)
    mean = 1 - ell / (n + 1)
    sd = math.sqrt(ell * (n + 1 - ell) * (n + n_val + 1) / (n_val * R * (n + 1) ** 2 * (n + 2)))
    assert abs(np.mean(cov) - mean) < 4 * sd


def test_g2_is_calibrated_once() -> None:
    cal = np.random.default_rng(5).exponential(size=500)
    g = bl.StaticSplitConformal(alpha=ALPHA, tau=1.0, cal_scores=cal)
    assert g.q_t == conformal_quantile(cal, ALPHA)
    d = g.commit(_inp(0, s=0.5))
    assert d.B == pytest.approx(g.q_t * 0.5) and d.p_audit == 0.0
    assert d.decision == ("accept" if d.B <= 1.0 else "flag")
    with pytest.raises(ValueError, match="calibrated once"):
        g.update([_rec(0, 0.1)])


# ====================================================================== G3


@pytest.mark.parametrize(("n", "alpha"), [(19, 0.05), (20, 0.05), (999, 0.1), (9, 0.05), (50, 0.3)])
def test_g3_equal_weights_give_the_split_conformal_quantile(n: int, alpha: float) -> None:
    s = np.random.default_rng(n).exponential(size=n)
    q = bl.weighted_conformal_quantile(s, np.zeros(n), 0.0, alpha)
    assert q == conformal_quantile(s, alpha)


def test_g3_quantile_is_invariant_to_a_common_log_weight_shift() -> None:
    rng = np.random.default_rng(6)
    s, lw = rng.exponential(size=300), rng.normal(size=300)
    for lt in (-2.0, 0.0, 3.0):
        a = bl.weighted_conformal_quantile(s, lw, lt, ALPHA)
        assert a == bl.weighted_conformal_quantile(s, lw + 700.0, lt + 700.0, ALPHA)


def test_g3_restores_coverage_under_a_known_covariate_shift() -> None:
    """Theorem 3: cal x ~ U(0,1), test x ~ density 2x, so w(x) = 2x; the score scale grows with
    x, so the unweighted (G2) threshold under-covers on the test distribution."""
    rng = np.random.default_rng(7)
    n, n_test = 2000, 20_000
    sig = lambda x: 0.1 + 3.0 * x  # noqa: E731
    xc = rng.uniform(size=n)
    sc = rng.exponential(sig(xc))
    xt = np.sqrt(rng.uniform(size=n_test))  # inverse CDF of density 2x
    st = rng.exponential(sig(xt))
    g3 = bl.WeightedSplitConformal(alpha=ALPHA, tau=1.0, cal_scores=sc, cal_log_w=np.log(2 * xc),
                                   test_log_w=lambda inp: float(np.log(2 * inp.peak_hat)))
    q3 = np.array([g3.decide(_inp(i, peak=float(x))).q_t for i, x in enumerate(xt)])
    g2 = bl.StaticSplitConformal(alpha=ALPHA, tau=1.0, cal_scores=sc)
    assert np.mean(st <= q3) >= 1 - ALPHA - 0.01
    assert np.mean(st <= g2.q_t) < 1 - ALPHA - 0.02


def test_g3_tilted_weights_for_p1() -> None:
    peak = np.array([40.0, 60.0, 80.0])
    assert np.array_equal(bl.tilted_log_weights(peak, 3.0), -3.0 * peak)
    with pytest.raises(ValueError):
        bl.tilted_log_weights(peak, -1.0)


def test_g3_classifier_ratio_recovers_a_log_linear_shift_and_is_clipped() -> None:
    rng = np.random.default_rng(8)
    d, beta = 36, 2.0
    cal = rng.uniform(size=(4000, d))
    u = rng.uniform(size=4000)  # x_0 with density ∝ exp(β x_0) by inverse CDF
    prop = rng.uniform(size=(4000, d))
    prop[:, 0] = np.log1p(u * np.expm1(beta)) / beta
    est = bl.ClassifierDensityRatio(cal, prop, C=1e3)
    xs = rng.uniform(size=(2000, d))
    lw = est.log_weight(xs)
    true = beta * xs[:, 0] - math.log(np.expm1(beta) / beta)
    assert np.corrcoef(lw, true)[0, 1] > 0.95
    assert np.sqrt(np.mean((lw - true) ** 2)) < 0.2  # 36-dim fit on 8,000 points: RMS, not max
    lo, hi = bl.RATIO_CLIP
    prop[:, 0] = np.log1p(u * np.expm1(8.0)) / 8.0  # ratio down to ~1e-3: must be clipped
    w = np.exp(bl.ClassifierDensityRatio(cal, prop, C=1e3).log_weight(xs))
    assert w.min() >= lo - 1e-12 and w.max() <= hi + 1e-12 and np.isclose(w.min(), lo)


def test_g3_is_calibrated_once() -> None:
    g = bl.WeightedSplitConformal(alpha=ALPHA, tau=1.0, cal_scores=np.ones(10),
                                  cal_log_w=np.zeros(10), test_log_w=lambda inp: 0.0)
    with pytest.raises(ValueError, match="calibrated once"):
        g.update([_rec(0, 0.1)])
    with pytest.raises(ValueError, match="cal_log_w"):
        bl.WeightedSplitConformal(alpha=ALPHA, tau=1.0, cal_scores=np.ones(10),
                                  cal_log_w=np.zeros(9), test_log_w=lambda inp: 0.0)


# ====================================================================== G4


def _g4(**kw):
    args = {"alpha": ALPHA, "gamma": 0.01, "W": 500, "tau": 1.0,
            "cal_scores": np.random.default_rng(9).exponential(size=1000),
            "ablation_flagged_only": True}
    args.update(kw)
    return bl.ACIFlaggedOnly(**args)


def test_g4_needs_the_named_ablation_flag() -> None:
    with pytest.raises(ValueError, match="ablation_flagged_only"):
        _g4(ablation_flagged_only=False)


def test_g4_under_covers_when_flagged_and_accepted_errors_differ() -> None:
    """Two design types: small-scale designs (accepted) have heavy scores, large-scale designs
    (flagged) have light ones. G4 calibrates on the flagged type only and under-covers overall;
    G5, fed uniform audits of the same stream, holds α (Prop. B.14 of Tu & Giesecke: the
    selection-bias term, CLAUDE.md §18.1)."""
    rng = np.random.default_rng(10)
    T = 20_000
    kind = rng.uniform(size=T) < 0.5
    s = np.where(kind, 0.1, 10.0)
    S = np.where(kind, rng.exponential(1.0, T), rng.exponential(0.1, T))
    g4 = _g4()
    coin = AuditCoin(seed=11)
    g5 = AuditedACIGate(alpha=ALPHA, gamma=0.01, W=500, tau=1.0, p=0.2,
                        cal_scores=np.random.default_rng(9).exponential(size=1000), coin=coin)
    miss4 = miss5 = 0
    for t in range(T):
        inp = _inp(t, s=float(s[t]))
        d4, d5 = g4.commit(inp), g5.commit(inp)
        miss4 += S[t] > d4.q_t
        miss5 += S[t] > d5.q_t
        if d4.decision == "flag":
            g4.update([_rec(t, float(S[t]), s=float(s[t]))])
        coin.commit(f"d{t}")
        if coin.draw(f"d{t}", d5.p_audit):
            g5.update([_rec(t, float(S[t]), s=float(s[t]))])
    assert miss4 / T > 3 * ALPHA
    assert abs(miss5 / T - ALPHA) < 0.02


def test_g4_takes_only_labels_of_designs_it_flagged_once_each() -> None:
    g = _g4(tau=1e-9)  # everything flags
    g.commit(_inp(0))
    with pytest.raises(ValueError, match="not flagged"):
        g.update([_rec(1, 0.5)])
    g.update([_rec(0, 0.5)])
    with pytest.raises(ValueError, match="already"):
        g.update([_rec(0, 0.5)])
    acc = _g4(tau=1e9)  # everything accepted
    acc.commit(_inp(0))
    with pytest.raises(ValueError, match="not flagged"):
        acc.update([_rec(0, 0.5)])


# ====================================================================== G6


def test_mvp_constants() -> None:
    for eps in (0.5, 1.0, 2.0):
        K, lo, hi = mv.k_eps(eps)
        assert lo <= K <= hi and (hi - lo) / K < 1e-6
        n = np.arange(200_000)
        assert K > np.sum(1.0 / mv.f(n, eps) ** 2)
    assert mv.f(np.array([0]), 1.0)[0] == pytest.approx(math.sqrt(math.log(2) ** 2))
    K = mv.k_eps(1.0)[0]
    assert mv.eta(3, 40, 1.0) == pytest.approx(math.sqrt(math.log(120) / (2 * K * 120)))


def _g6(coin, **kw):
    args = {"alpha": ALPHA, "tau": 1.0, "p": 1.0, "groups": ["all", "g1", "g2"], "m": 10,
            "r": 1000, "eps": 1.0, "noise_width": 0.0, "coin": coin, "seed": 0,
            "membership": _members}
    args.update(kw)
    return mv.MultiValidGate(**args)


def _members(inp: GateInput) -> list[str]:
    i = int(inp.design_id[1:])
    return ["all"] + (["g1"] if i % 2 == 0 else []) + (["g2"] if i % 3 == 0 else [])


def _stream(T: int, seed: int) -> np.ndarray:
    """Scores whose law depends on group membership: g1 ∩ g2 is 4× heavier, g1 alone 2×."""
    rng = np.random.default_rng(seed)
    i = np.arange(T)
    scale = np.where((i % 2 == 0) & (i % 3 == 0), 4.0, np.where(i % 2 == 0, 2.0, 1.0))
    return rng.exponential(scale)


def test_g6_covers_each_of_two_intersecting_groups_where_g5_does_not() -> None:
    T = 30_000
    S = _stream(T, 12)
    coin6, coin5 = AuditCoin(seed=13), AuditCoin(seed=14)
    g6 = _g6(coin6)
    g5 = AuditedACIGate(alpha=ALPHA, gamma=0.01, W=1000, tau=1.0, p=1.0,
                        cal_scores=np.random.default_rng(15).exponential(size=1000), coin=coin5)
    miss6 = {"all": [], "g1": [], "g2": []}
    miss5 = {"all": [], "g1": [], "g2": []}
    for t in range(T):
        inp = _inp(t)
        for coin in (coin6, coin5):
            coin.commit(inp.design_id)
        d6, d5 = g6.commit(inp), g5.commit(inp)
        for g in _members(inp):
            miss6[g].append(S[t] > d6.q_t)
            miss5[g].append(S[t] > d5.q_t)
        assert coin6.draw(inp.design_id, d6.p_audit) and coin5.draw(inp.design_id, d5.p_audit)
        g6.update([_rec(t, float(S[t]))])
        g5.update([_rec(t, float(S[t]))])
    for g in miss6:
        assert abs(np.mean(miss6[g]) - ALPHA) < 0.015, (g, np.mean(miss6[g]))
    assert np.mean(miss5["g2"]) > ALPHA + 0.05


def test_g6_preview_is_the_deterministic_upper_candidate_and_pure() -> None:
    coin = AuditCoin(seed=16)
    g = _g6(coin, r=4)
    S = _stream(400, 17)
    for t in range(400):  # build up some state
        coin.commit(f"d{t}")
        g.commit(_inp(t))
        coin.draw(f"d{t}", 1.0)
        g.update([_rec(t, float(S[t]))])
    before = g.state_dict()
    prev = [g.decide(_inp(1000)) for _ in range(5)]
    assert g.state_dict() == before and all(p == prev[0] for p in prev)
    coin.commit("d1000")
    k_up = g._candidates(_members(_inp(1000)))[1]
    assert prev[0].q_t == mv.q_from_grid(k_up, g.r * g.m)
    committed = g.commit(_inp(1000))
    assert committed.q_t <= prev[0].q_t  # the committed draw is one of the two candidates


def test_g6_protocol_errors() -> None:
    coin = AuditCoin(seed=18)
    g = _g6(coin)
    with pytest.raises(ValueError, match="coin.commit"):
        g.commit(_inp(0))  # D9 rule 2: MVP draws only after the design is committed
    coin.commit("d0")
    g.commit(_inp(0))
    with pytest.raises(ValueError, match="already committed"):
        g.commit(_inp(0))
    coin.draw("d0", 0.0)  # coin says 0: not audited
    with pytest.raises(ValueError, match="came up 0"):
        g.update([_rec(0, 0.5)])
    coin.commit("d1")
    coin.draw("d1", 1.0)
    with pytest.raises(ValueError, match="not committed to G6"):
        g.update([_rec(1, 0.5)])
    with pytest.raises(ValueError, match="no group"):
        _g6(AuditCoin(seed=1), membership=lambda inp: []).decide(_inp(0))
    with pytest.raises(ValueError, match="unknown group"):
        _g6(AuditCoin(seed=1), membership=lambda inp: ["zz"]).decide(_inp(0))


def test_g6_smoothing_noise_never_lowers_the_score() -> None:
    rng = np.random.default_rng(19)
    S, U = rng.exponential(size=1000), rng.uniform(0, 0.5, size=1000)
    t = mv.smoothed_score(S, U)
    assert np.all(t >= S / (1 + S)) and np.all(t < 1)
    assert mv.q_from_grid(40, 40) == math.inf and mv.q_from_grid(0, 40) == 0.0
    assert mv.q_from_grid(20, 40) == pytest.approx(1.0)  # q̃ = 1/2 ↔ q = 1


# ====================================================================== G7


def _g7(coin, **kw):
    rng = np.random.default_rng(20)
    args = {"alpha": ALPHA, "gamma": 0.01, "W": 500, "tau": 1.0, "p_bar": 0.1, "p_min": 0.02,
            "kappa": 0.8, "u_cal": rng.lognormal(0, 0.5, size=1000),
            "cal_scores": rng.exponential(size=1000), "coin": coin}
    args.update(kw)
    return aa.ActiveAuditGate(**args)


def test_g7_probabilities_and_weights_have_the_right_means() -> None:
    coin, rng = AuditCoin(seed=21), np.random.default_rng(22)
    g = _g7(coin)
    T = 40_000
    s = rng.lognormal(0, 0.5, size=T)  # same law as u_cal
    ps, ipw = [], []
    for t in range(T):
        coin.commit(f"d{t}")
        d = g.commit(_inp(t, s=float(s[t])))
        a = coin.draw(f"d{t}", d.p_audit)
        ps.append(d.p_audit)
        ipw.append(a / d.p_audit)
    ps = np.array(ps)
    assert ps.min() >= 0.02 and ps.max() <= 1.0
    assert abs(ps.mean() - 0.1) < 0.005
    assert abs(np.mean(ipw) - 1.0) < 0.05  # A_t / p_t has conditional mean 1
    hi, lo = g.decide(_inp(T + 1, s=5.0)), g.decide(_inp(T + 2, s=0.2))
    assert hi.p_audit > lo.p_audit


def test_g7_update_is_inverse_probability_weighted() -> None:
    coin = AuditCoin(seed=23)
    g = _g7(coin)
    for t in range(10_000):  # commit designs until one is audited
        coin.commit(f"d{t}")
        d = g.commit(_inp(t, s=3.0))
        if coin.draw(f"d{t}", d.p_audit):
            break
    a0 = g.alpha_t
    out = g.update([_rec(t, S=1e6, s=3.0)])[0]
    assert out.err == 1
    assert g.alpha_t == pytest.approx(a0 + 0.01 * (0.1 / d.p_audit) * (ALPHA - 1))


def test_g7_with_kappa_zero_is_g5() -> None:
    coin5, coin7 = AuditCoin(seed=24), AuditCoin(seed=24)
    cal = np.random.default_rng(25).exponential(size=1000)
    g5 = AuditedACIGate(alpha=ALPHA, gamma=0.01, W=500, tau=1.0, p=0.1, cal_scores=cal,
                        coin=coin5)
    g7 = _g7(coin7, kappa=0.0, cal_scores=cal)
    S = np.random.default_rng(26).exponential(size=3000)
    for t in range(3000):
        for g, coin in ((g5, coin5), (g7, coin7)):
            coin.commit(f"d{t}")
            d = g.commit(_inp(t, s=float(1 + t % 5)))
            assert d.p_audit == 0.1
            if coin.draw(f"d{t}", d.p_audit):
                g.update([_rec(t, float(S[t]), s=float(1 + t % 5))])
        assert g5.alpha_t == g7.alpha_t and g5.q_t == g7.q_t


def test_g7_settings_and_protocol() -> None:
    with pytest.raises(ValueError, match="p_min"):
        _g7(AuditCoin(seed=1), p_min=0.2)
    with pytest.raises(ValueError, match="kappa"):
        _g7(AuditCoin(seed=1), kappa=1.5)
    coin = AuditCoin(seed=27)
    g = _g7(coin)
    coin.commit("d0")
    coin.draw("d0", 1.0)
    with pytest.raises(ValueError, match="not committed to G7"):
        g.update([_rec(0, 0.5)])


# ====================================================================== shared rules


def _all_gates():
    coin = AuditCoin(seed=30)
    cal = np.random.default_rng(31).exponential(size=200)
    return coin, [
        bl.NoGate(), _g1(window=3), bl.StaticSplitConformal(alpha=ALPHA, tau=1.0, cal_scores=cal),
        bl.WeightedSplitConformal(alpha=ALPHA, tau=1.0, cal_scores=cal, cal_log_w=np.zeros(200),
                                  test_log_w=lambda inp: 0.0),
        _g4(), _g6(coin), _g7(coin),
    ]


def test_decide_is_pure_and_log_rows_validate_for_every_gate() -> None:
    coin, gates = _all_gates()
    for t in range(6):
        inp = _inp(t, s=0.3, r_hot=0.1 * t)
        coin.commit(inp.design_id)
        for g in gates:
            before = g.state_dict()
            g.decide(inp)
            assert g.state_dict() == before, g.gate_id
            d = g.commit(inp)
            row = g.log_row(inp, d, run_id="e7-abc1234-0", seed=0, t=t, outcome=None, iters_used=0)
            validate(row, "audit_log_row")
            assert row["gate"] == g.gate_id


def test_state_round_trips_for_every_gate() -> None:
    coin, gates = _all_gates()
    for t in range(6):
        inp = _inp(t, s=0.3, r_hot=0.1 * t)
        coin.commit(inp.design_id)
        for g in gates:
            g.commit(inp)
    _, fresh = _all_gates()
    for g, h in zip(gates, fresh, strict=True):
        h.load_state_dict(g.state_dict())
        assert h.state_dict() == g.state_dict(), g.gate_id
        with pytest.raises(ValueError):
            h.load_state_dict({**g.state_dict(), "gate": "G9"})
