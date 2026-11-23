"""Offline gate replay (M3.5; CLAUDE.md §11.3–§11.5, §8.2, §4): metrics against hand counts, the
frozen-config guard, identical proposals for every gate, certified audits, bootstrap CIs and the
hypothesis verdicts, on small synthetic trajectories with known errors."""

from __future__ import annotations

import copy
import math
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments import run_gates as RG
from gate import groups as G
from gate import proposers as P

FROZEN = {
    "alpha": 0.1, "tau_K": 1.0, "gamma": 0.01, "W": 200, "lam": 1.0, "eps_K": 1e-6,
    "m_hot": 64, "p_audit": 0.2, "p_sweep": [0.05, 0.1, 0.2], "delta": 0.05,
    "eps_S": 0.05, "eps_S_sweep": [0.01, 0.05, 0.2], "p_min": 0.05, "p_min_sweep": [0.02, 0.05],
    "kappa": 0.5, "seeds": [0, 1, 2, 3, 4],
    "g1": {"window": 100, "residual": "scaled", "c_events": "off", "delta_n": "off"},
    "g3": {"classifier_C": 1.0},
    "g6": {"m_buckets": 10, "r": 2, "eps": 0.5, "noise_width": 0.01},
    "bootstrap": {"resamples": 2000, "seed": 7},
}


def make_traj(err, s=None, *, proposer="P2", level=2, seed=0, groups=None, peak=50.0,
              lam=None) -> tuple[RG.Trajectory, RG.Labels]:
    """Trajectory with sd_hot = 0, r_hot = s (so s = r_hot + eps) and labels with known errors."""
    err = np.asarray(err, dtype=np.float64)
    T = len(err)
    s = np.ones(T) if s is None else np.asarray(s, dtype=np.float64)
    rng = np.random.default_rng(seed + 100 * level)
    X = rng.random((T, P.DIM))
    hashes = [P.x_hash(x) for x in X]
    groups = groups or [G.encode(G.group_names(proposer, level, 1 + t % 3, t % 2 == 0))
                        for t in range(T)]
    traj = RG.Trajectory(name=f"traj_{proposer}_a{level}_s{seed}", proposer=proposer,
                         level=level, seed=seed, lam=lam, x_hash=hashes, design_vector=X,
                         peak_hat=np.full(T, peak), r_hot=s - FROZEN["eps_K"],
                         sd_hot=np.zeros(T), r_hot_raw=s, group=groups)
    labels = RG.Labels(oracle=dict(zip(hashes, peak - err, strict=True)), iters_full={},
                       traces={})
    return traj, labels


def make_cal(scores, seed=99) -> tuple[RG.Calibration, dict[str, float]]:
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    X = np.random.default_rng(seed).random((n, P.DIM))
    cal = RG.Calibration(peak_hat=np.full(n, 50.0), oracle=50.0 - scores,
                         r_hot=np.ones(n) - FROZEN["eps_K"], sd_hot=np.zeros(n),
                         design_vector=X)
    return cal, {}


# ------------------------------------------------------------------- M1 against a hand count


def test_m1_equals_a_hand_count() -> None:
    # G2: q = 0.5 from calibration; B = 0.5·s; accept iff s < 2 (tau = 1 K)
    s = np.array([1, 1, 3, 3, 1, 1.9, 4, 1, 1, 1.9])  # 1.9, not 2: B = 0.95, clear of tau
    err = np.array([0.2, 1.5, 2.0, 0.1, 1.2, 1.01, 5.0, 0.99, 3.0, 0.0])
    traj, labels = make_traj(err, s)
    cal_scores = np.full(999, 0.5)
    cal, _ = make_cal(cal_scores)
    run = RG.replay(traj, labels, cal, FROZEN, "G2", seed=0)
    accepted = s <= 2.0
    large = err > 1.0
    hand = int(np.sum(accepted & large))  # t = 1, 4, 5, 8 -> 4
    assert hand == 4
    assert run.summary["accepted"] == int(accepted.sum()) == 7
    assert run.summary["accepted_large"] == hand
    assert run.summary["M1_all"] == pytest.approx(hand / 10)
    assert run.summary["M1_accepted"] == pytest.approx(hand / 7)
    g0 = RG.replay(traj, labels, cal, FROZEN, "G0", seed=0)
    assert g0.summary["accepted"] == 10 and g0.summary["M1_all"] == pytest.approx(large.mean())
    assert g0.summary["solves"] == 1  # G0: the final design is verified once


def test_group_metrics_match_a_hand_count() -> None:
    s = np.ones(9)
    err = np.array([2, 0, 2, 2, 0, 0, 2, 2, 2], dtype=float)
    groups = [G.encode(G.group_names("P1", 3, k, e)) for k, e in
              [(1, 0), (1, 0), (2, 1), (3, 1), (3, 0), (2, 0), (1, 1), (1, 1), (2, 0)]]
    traj, labels = make_traj(err, s, proposer="P1", level=3, groups=groups)
    cal, _ = make_cal(np.full(99, 0.1))  # q = 0.1 -> accept all
    run = RG.replay(traj, labels, cal, FROZEN, "G2", seed=0)
    by = {r["group"]: r for r in run.group_rows}
    # ood=1: t = 0, 1, 6, 7 -> large at 0, 6, 7 -> 3 / 4
    assert by["ood=1"]["T_g"] == 4 and by["ood=1"]["accepted_large_g"] == 3
    assert by["ood=1"]["M1g"] == pytest.approx(3 / 4)
    assert by["edge=1"]["T_g"] == 4 and by["edge=1"]["accepted_large_g"] == 4
    assert by["proposer=P1"]["T_g"] == 9
    slack = RG.lemma1_slack(alpha=0.1, gamma=0.01, p=0.2, T=4, delta=0.05)
    assert by["ood=1"]["slack_g"] == pytest.approx(slack)
    assert by["ood=1"]["slack_kind"] == "empirical target (Lemma 1 form, not proved per group)"


# ----------------------------------------------------------------------- Lemma 1 bound


def test_lemma1_bound_formula() -> None:
    a, g, p, T, d = 0.1, 0.01, 0.1, 10_000, 0.05
    C = max(a, 1 - a) + g
    want = a + C / (g * p * T) + (2 / p) * math.sqrt(math.log(4 / d) / T)
    assert RG.lemma1_bound(alpha=a, gamma=g, p=p, T=T, delta=d) == pytest.approx(want)
    with pytest.raises(ValueError):
        RG.lemma1_bound(alpha=a, gamma=g, p=0.0, T=T, delta=d)


# -------------------------------------------------------- every gate, the same proposals


def test_every_gate_sees_the_same_proposals_in_order() -> None:
    rng = np.random.default_rng(1)
    err = rng.exponential(0.6, size=300)
    s = rng.uniform(0.5, 1.5, size=300)
    traj, labels = make_traj(err, s, proposer="P1", level=1, lam=1.0)
    cal, _ = make_cal(rng.exponential(0.6, size=500))
    other, _ = make_traj(err, s, proposer="P1", level=1, seed=1, lam=1.0)
    seqs = {}
    for g in RG.GATES:
        kw = {"f": 0.3} if g == "G1" else {}
        run = RG.replay(traj, labels, cal, FROZEN, g, seed=0, ratio_traj=other, **kw)
        seqs[g] = [r["x_hash"] for r in run.rows]
        assert [r["t"] for r in run.rows] == list(range(300))
        assert all(r["gate"] == g for r in run.rows)
    first = seqs["G0"]
    assert all(v == first for v in seqs.values())


def test_g4_learns_only_from_its_flagged_labels_and_g5_only_from_audits() -> None:
    rng = np.random.default_rng(2)
    traj, labels = make_traj(rng.exponential(1.0, 400), rng.uniform(0.2, 3, 400))
    cal, _ = make_cal(rng.exponential(1.0, 500))
    g4 = RG.replay(traj, labels, cal, FROZEN, "G4", seed=0)
    assert g4.summary["label_updates"] == g4.summary["flagged"] > 0
    assert g4.summary["audited"] == 0
    g5 = RG.replay(traj, labels, cal, FROZEN, "G5", seed=0)
    assert g5.summary["label_updates"] == g5.summary["audited"] > 0
    assert g5.summary["solves"] == len({r["t"] for r in g5.rows
                                        if r["audited"] or r["decision"] == "flag"})


def test_g1_flag_rate_comes_from_the_g5_budget() -> None:
    rng = np.random.default_rng(3)
    traj, labels = make_traj(rng.exponential(1.0, 600), rng.uniform(0.2, 3, 600))
    cal, _ = make_cal(rng.exponential(1.0, 500))
    g5 = RG.replay(traj, labels, cal, FROZEN, "G5", seed=0)
    f = RG.g1_flag_rate(g5.summary)
    assert f == pytest.approx(g5.summary["solves"] / 600)
    g1 = RG.replay(traj, labels, cal, FROZEN, "G1", seed=0, f=f)
    assert g1.summary["audited"] == 0 and g1.summary["flagged"] > 0


# ------------------------------------------------- validity under drift (the H1 mechanism)


def test_g5_stays_within_the_bound_under_drift_and_g2_does_not() -> None:
    T = 6000
    rng = np.random.default_rng(4)
    s = rng.uniform(0.5, 1.5, T)
    u = np.where(np.arange(T) < T // 10, rng.uniform(0, 1, T), rng.uniform(0, 10, T))
    traj, labels = make_traj(u * s, s, proposer="P3", level=3)
    cal, _ = make_cal(rng.uniform(0, 1, 1000))
    fz = {**FROZEN, "p_audit": 0.2}
    g5 = RG.replay(traj, labels, cal, fz, "G5", seed=0)
    g2 = RG.replay(traj, labels, cal, fz, "G2", seed=0)
    assert g5.summary["within_bound"] and g5.summary["M1_all"] <= g5.summary["bound"]
    assert not g2.summary["within_bound"]
    assert g2.summary["bound"] == pytest.approx(g5.summary["bound"])  # same reference line


# --------------------------------------------------------------------- certified audits


def test_certified_audit_stops_when_decided_and_counts_iterations() -> None:
    # true peak 49.0, prediction 50.0, s = 1 -> S = 1.0
    trace = RG.Trace(iters=np.array([5, 10, 15, 20]), T_lo=np.array([40.0, 47.0, 48.6, 48.99]),
                     T_hi=np.array([60.0, 52.0, 49.3, 49.01]))
    # q = 0.5: decided once S_lo > 0.5, i.e. T_hi < 49.5 -> check 3 (T_hi = 49.3)
    lo, hi, iters, why = RG.certified_audit(50.0, 1.0, 0.5, 0.01, trace, oracle=49.0)
    assert (lo, hi, iters, why) == (48.6, 49.3, 15, "decided")
    # q = 3.001: S_hi <= q once max(|50 - lo|, |50 - hi|) <= 3 -> check 2 (lo 47, hi 52: S_hi = 3)
    assert RG.certified_audit(50.0, 1.0, 3.001, 0.01, trace, oracle=49.0)[2:] == (10, "decided")
    # at the exact tie S_hi = q = 3, S_hi is rounded up (certify/peak_interval.score_interval),
    # so the conservative rule waits for the next check
    assert RG.certified_audit(50.0, 1.0, 3.0, 0.01, trace, oracle=49.0)[2:] == (15, "decided")
    # q = 1.0 sits on the true score: never decided, the width rule ends it
    assert RG.certified_audit(50.0, 1.0, 1.0, 0.05, trace, oracle=49.0)[2:] == (20, "width")
    assert RG.certified_audit(50.0, 1.0, 1.0, 0.0, trace, oracle=49.0)[2:] == (20, "end")
    bad = RG.Trace(iters=np.array([5]), T_lo=np.array([49.5]), T_hi=np.array([50.0]))
    with pytest.raises(RG.CertificateViolation, match="B7"):
        RG.certified_audit(50.0, 1.0, 0.5, 0.01, bad, oracle=49.0)


def test_certified_replay_spends_fewer_audit_iterations() -> None:
    rng = np.random.default_rng(5)
    T = 500
    err = rng.exponential(0.5, T)
    traj, labels = make_traj(err, np.ones(T))
    for h, e in zip(traj.x_hash, err, strict=True):
        true = 50.0 - e
        w = np.array([10.0, 3.0, 1.0, 0.3, 0.1, 0.0])
        labels.traces[h] = RG.Trace(iters=np.arange(1, 7) * 10, T_lo=true - w, T_hi=true + w)
        labels.iters_full[h] = 60
    cal, _ = make_cal(rng.exponential(0.5, 500))
    full = RG.replay(traj, labels, cal, FROZEN, "G5", seed=0, audit="exact")
    cert = RG.replay(traj, labels, cal, FROZEN, "G5", seed=0, audit="certified", eps_S=0.05)
    assert full.summary["audit_iters"] == 60 * full.summary["audited"]
    assert cert.summary["audit_iters"] < full.summary["audit_iters"]
    assert cert.summary["audit_iters_full"] == full.summary["audit_iters"]


def test_a_missing_label_raises() -> None:
    traj, labels = make_traj(np.ones(5))
    del labels.oracle[traj.x_hash[3]]
    cal, _ = make_cal(np.full(99, 0.5))
    with pytest.raises(RG.InputError, match="oracle label"):
        RG.replay(traj, labels, cal, FROZEN, "G2", seed=0)


# --------------------------------------------------------------- bootstrap and verdicts


def test_bootstrap_ci_over_seeds() -> None:
    v = [0.08, 0.09, 0.11, 0.10, 0.12]
    m, lo, hi = RG.bootstrap_ci(v, resamples=5000, seed=1)
    assert m == pytest.approx(0.10) and lo < m < hi and lo >= 0.08 and hi <= 0.12
    assert RG.bootstrap_ci(v, resamples=5000, seed=1) == (m, lo, hi)
    assert RG.bootstrap_ci([0.1] * 5, resamples=100, seed=0) == (0.1, 0.1, 0.1)


def _summ(gate, proposer, level, seed, M1, bound=0.2, **kw):
    return {"gate": gate, "proposer": proposer, "level": level, "seed": seed, "M1_all": M1,
            "bound": bound, "within_bound": M1 <= bound, **kw}


def test_h1_verdict() -> None:
    rows = []
    for P_ in ("P1", "P2", "P3"):
        for lv in (1, 2, 3):
            for s in range(5):
                rows.append(_summ("G5", P_, lv, s, 0.1))
                rows.append(_summ("G1", P_, lv, s, 0.1 * lv))  # rises with level, exceeds at 3
                rows.append(_summ("G2", P_, lv, s, 0.12 * lv))
                rows.append(_summ("G4", P_, lv, s, 0.15 * lv))
                rows.append(_summ("G3", P_, lv, s, 0.1 if P_ == "P1" else 0.3))
    v = RG.verdict_h1(rows)
    assert v["verdict"] == "SUPPORTED", v
    bad = [dict(r, M1_all=0.5, within_bound=False) if r["gate"] == "G5" and r["seed"] < 2 else r
           for r in rows]
    assert RG.verdict_h1(bad)["verdict"] == "NOT SUPPORTED"
    assert "P4" in RG.verdict_h1(rows)["missing"]


def test_h7_h8_h9_verdicts() -> None:
    h7 = [_summ("G5c", "P1", 3, s, 0.1, eps_S=0.05, audit_iters=40, audit_iters_full=100)
          for s in range(5)]
    assert RG.verdict_h7(h7, eps_S=0.05)["verdict"] == "SUPPORTED"
    h7b = [dict(r, audit_iters=60) for r in h7]
    assert RG.verdict_h7(h7b, eps_S=0.05)["verdict"] == "NOT SUPPORTED"
    groups = []
    for s in range(5):
        groups += [{"gate": "G6", "level": 3, "seed": s, "group": "ood=3", "M1g": 0.1,
                    "slack_g": 0.2, "alpha": 0.1, "T_g": 100},
                   {"gate": "G5", "level": 3, "seed": s, "group": "ood=3", "M1g": 0.4,
                    "slack_g": 0.2, "alpha": 0.1, "T_g": 100}]
    assert RG.verdict_h8(groups)["verdict"] == "SUPPORTED (empirical)"
    h9 = ([_summ("G5", "P1", 3, s, 0.1, p=0.2, audited=200) for s in range(5)]
          + [_summ("G7", "P1", 3, s, 0.1, p=0.1, p_min=0.05, audited=100) for s in range(5)])
    assert RG.verdict_h9(h9)["verdict"] == "SUPPORTED"


# ------------------------------------------------------------------------ guard and inputs


def _git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True)


def test_frozen_guard(tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    f = repo / "frozen.yaml"
    f.write_text(yaml.safe_dump(FROZEN), "utf-8")
    with pytest.raises(RG.FrozenConfigError, match="committed"):
        RG.check_frozen(f, repo=repo)  # untracked
    _git(repo, "add", "frozen.yaml")
    _git(repo, "commit", "-q", "-m", "freeze")
    cfg = RG.check_frozen(f, repo=repo)
    assert cfg["alpha"] == 0.1 and len(cfg["_frozen_commit"]) == 40
    f.write_text(yaml.safe_dump({**FROZEN, "alpha": 0.2}), "utf-8")  # newer than HEAD
    with pytest.raises(RG.FrozenConfigError, match="uncommitted"):
        RG.check_frozen(f, repo=repo)
    nul = copy.deepcopy(FROZEN)
    nul["gamma"] = None
    f.write_text(yaml.safe_dump(nul), "utf-8")
    _git(repo, "commit", "-q", "-am", "null")
    with pytest.raises(RG.FrozenConfigError, match="gamma"):
        RG.check_frozen(f, repo=repo)
    with pytest.raises(RG.FrozenConfigError, match="exist"):
        RG.check_frozen(repo / "nope.yaml", repo=repo)


def test_shipped_frozen_config_is_complete() -> None:
    # D4 is decided (docs/decisions.md, 2026-11-23): the shipped frozen.yaml must be complete and
    # valid. A null in a template is still refused (test_check_frozen above).
    cfg = RG.validate_frozen(RG.load_yaml(RG.DEFAULT_FROZEN))
    assert cfg["m_hot"] == 64 and cfg["p_audit"] in cfg["p_sweep"]


def test_inputs_load_from_the_m33_and_n32_formats(tmp_path: Path) -> None:
    from experiments import run_trajectories as RT

    traj, labels = make_traj(np.array([0.5, 2.0, 0.1]), proposer="P1", level=1, lam=3.0)
    rows = []
    for t in range(3):
        rows.append({"schema_version": "1.0", "run_id": "traj_P1_a1-abc1234-0", "seed": 0,
                     "t": t, "proposer": "P1", "aggressiveness": 1, "group": traj.group[t],
                     "x_hash": traj.x_hash[t], "r_hot": float(traj.r_hot[t]), "sd_hot": 0.0,
                     **dict.fromkeys(RT._NULL_GATE_COLS), "design_id": f"t{t}",
                     "design_vector": traj.design_vector[t].tolist(), "peak_hat": 50.0,
                     "r_hot_raw": 1.0, "ood_score": 1.0, "ood_tercile": 1, "edge_hotspot": False,
                     "round": t})
    tab = pa.Table.from_pylist(rows, schema=RT.SCHEMA).replace_schema_metadata(
        {"twinheat": '{"config": {"lambda_per_K": [1.0, 3.0, 10.0]}, "level": 1}'})
    tp = tmp_path / "traj_P1_a1_s0.parquet"
    pq.write_table(tab, tp)
    lp = tmp_path / "labels.parquet"
    pq.write_table(pa.table({"x_hash": traj.x_hash, "oracle_peak": [49.5, 48.0, 49.9],
                             "iters_full": [100, None, 80]}), lp)
    t2 = RG.Trajectory.from_parquet(tp)
    assert (t2.proposer, t2.level, t2.seed, t2.lam) == ("P1", 1, 0, 1.0)
    assert t2.x_hash == traj.x_hash and t2.group == traj.group
    lab = RG.Labels.from_parquet(lp)
    assert lab.oracle[traj.x_hash[1]] == 48.0 and lab.iters_full[traj.x_hash[1]] is None
