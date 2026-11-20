"""Offline replay of every gate on identical oracle-labelled trajectories (M3.5; CLAUDE.md §11.1,
§11.3–§11.5, §8.1–§8.3, §4). The E-scripts (e7_validity, e9_darcy, e14_cert_audit, e15_groups,
e16_active) are thin wrappers around replay(); e8_efficiency runs the feedback loop online.

Per step t of a trajectory, every gate sees the same proposal, in the same order, strictly one
at a time (each audit updates the gate before step t + 1 is gated: assumption U1 of
report/lemma1_review.md):

    s_t = r_hot + λ·sd_hot + ε                         (label-free; λ, ε from frozen.yaml, D4)
    coin.commit(t) → binding decision (commit() where the gate has one, else decide())
    A_t = coin.draw(t, p_t)                            (one draw per step, so every gate on the
                                                        same trajectory and seed uses the same
                                                        uniform stream: common random numbers)
    audited → AuditRecord with [T⁻, T⁺]: the oracle label ("exact" audits) or the certified
              interval at the first check where the score interval is decided relative to the
              q_t that gated the design, or narrower than ε_S ("certified" audits, E14)
    G4 only → its flagged designs' labels (the selection-bias ablation)
    row = gate.log_row(...)                            (a §10.4 row, schema-validated)

Inputs (formats in docs/gate_experiment_inputs.md): M3.3 trajectory parquet files, the N3.2
oracle labels keyed by x_hash, the 1,000 labelled p₀ calibration designs (decision 2026-11-18),
and for E14 the N3.3 certified-interval traces. Every hyperparameter comes from
experiments/configs/frozen.yaml, which must be committed (and so predate the run) and complete:
check_frozen() refuses anything else. A missing label, a duplicate with a different value, or a
certified interval that misses the oracle (B7) raises.

Metrics (§11.3): M1 = accepted large errors (|T̂_peak − T_peak| > τ) over all proposals and over
accepted ones; M1g per group; M2 = exact solves (flagged ∪ audited; G0: the final design, 1) and
audit PCG iterations. Beside M1: the §8.2 bound α + C/(γpT) + (2/p)√(ln(4/δ)/T),
C = max(α₁, 1 − α₁) + γ with α₁ = α (p = p̄ for G7; the gates without audits are compared with
the bound at the same p). Per group: the same formula with the group's own step count T_g, an
empirical target only (no per-group bound is proved: report/lemma1_review.md §6).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from certify.peak_interval import decide, score_interval
from gate import groups as G
from gate.aci_audit import AuditedACIGate
from gate.active_audit import ActiveAuditGate
from gate.base import AuditRecord, BaseGate, GateInput
from gate.baselines import (
    ACIFlaggedOnly,
    ClassifierDensityRatio,
    DeepOHeatV2Gate,
    NoGate,
    StaticSplitConformal,
    WeightedSplitConformal,
    flag_rate_for_budget,
    tilted_log_weights,
)
from gate.coin import AuditCoin
from gate.multivalid import MultiValidGate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = ROOT / "experiments" / "configs" / "frozen.yaml"
GATES = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7")
PROPOSERS = ("P1", "P2", "P3", "P4")

# Success criteria, CLAUDE.md §11.4 (not hyperparameters: fixed by the plan, not D4)
RUN_FRACTION = 0.95  # H1, H3, H7, H8: "in >= 95 % of runs"
H7_ITER_SAVING = 0.50  # ">= 50 % fewer audit iterations"
H9_AUDIT_SAVING = 0.30  # ">= 30 % fewer audits"
H2_MARGIN_K = 0.2  # non-inferiority margin
SLACK_KIND = "empirical target (Lemma 1 form, not proved per group)"

FROZEN_KEYS = {
    "alpha": float, "tau_K": float, "gamma": float, "W": int, "lam": float, "eps_K": float,
    "m_hot": int, "p_audit": float, "p_sweep": list, "delta": float, "eps_S": float,
    "eps_S_sweep": list, "p_min": float, "p_min_sweep": list, "kappa": float, "seeds": list,
    "g1": dict, "g3": dict, "g6": dict, "bootstrap": dict,
}
FROZEN_SUB = {"g1": ("window", "residual", "c_events", "delta_n"), "g3": ("classifier_C",),
              "g6": ("m_buckets", "r", "eps", "noise_width"), "bootstrap": ("resamples", "seed")}


class FrozenConfigError(RuntimeError):
    """frozen.yaml is missing, uncommitted, incomplete or malformed (CLAUDE.md §4, D4)."""


class InputError(RuntimeError):
    """A trajectory, label, trace or calibration input is missing or inconsistent."""


class CertificateViolation(RuntimeError):
    """A certified interval does not contain the oracle peak: invariant B7. Blocks results."""


# ----------------------------------------------------------------------------- frozen config


def _git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_frozen(cfg: Any) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise FrozenConfigError("frozen.yaml must be a mapping")
    missing = [k for k in FROZEN_KEYS if k not in cfg]
    nulls = [k for k in FROZEN_KEYS if k in cfg and cfg[k] is None]
    for k, subs in FROZEN_SUB.items():
        if isinstance(cfg.get(k), dict):
            missing += [f"{k}.{s}" for s in subs if s not in cfg[k]]
            nulls += [f"{k}.{s}" for s in subs if s in cfg[k] and cfg[k][s] is None]
    if missing or nulls:
        raise FrozenConfigError(f"frozen.yaml: missing {missing}, null {nulls}: D4 must fix every "
                                "value before a reported run")
    for k, typ in FROZEN_KEYS.items():
        v = cfg[k]
        ok = isinstance(v, typ) or (typ is float and isinstance(v, int) and not isinstance(v, bool))
        if not ok:
            raise FrozenConfigError(f"frozen.yaml: {k} must be {typ.__name__}, got {v!r}")
    for k in ("p_sweep", "eps_S_sweep", "p_min_sweep", "seeds"):
        if not cfg[k]:
            raise FrozenConfigError(f"frozen.yaml: {k} must be a non-empty list")
    g1 = cfg["g1"]
    if (g1["c_events"] == "off") != (g1["delta_n"] == "off"):
        raise FrozenConfigError("frozen.yaml: g1.c_events and g1.delta_n must both be 'off' or "
                                "both be ints")
    if g1["residual"] not in ("scaled", "raw"):
        raise FrozenConfigError("frozen.yaml: g1.residual must be 'scaled' or 'raw' (D13)")
    if cfg["p_audit"] not in cfg["p_sweep"]:
        raise FrozenConfigError("frozen.yaml: p_audit must be one of p_sweep")
    if cfg["eps_S"] not in cfg["eps_S_sweep"] or cfg["p_min"] not in cfg["p_min_sweep"]:
        raise FrozenConfigError("frozen.yaml: eps_S / p_min must be in their sweeps")
    return cfg


def check_frozen(path: Path = DEFAULT_FROZEN, *, repo: Path = ROOT,
                 now: float | None = None) -> dict[str, Any]:
    """The frozen config, refusing unless it exists, is committed with no local changes (so its
    commit predates the run), and is complete. Adds _frozen_commit and _frozen_time."""
    path = Path(path)
    if not path.exists():
        raise FrozenConfigError(f"{path} does not exist: write and commit frozen.yaml (D4) first")
    try:
        rel = str(path.resolve().relative_to(Path(repo).resolve()))
        _git(repo, "ls-files", "--error-unmatch", rel)
    except (ValueError, subprocess.CalledProcessError) as e:
        raise FrozenConfigError(f"{path} must be committed before the run (CLAUDE.md §4)") from e
    if _git(repo, "status", "--porcelain", "--", rel):
        raise FrozenConfigError(f"{path} has uncommitted changes: it is newer than HEAD; commit "
                                "it before the run")
    sha, ts = _git(repo, "log", "-1", "--format=%H %ct", "--", rel).split()
    if float(ts) > (time.time() if now is None else now):
        raise FrozenConfigError(f"{path}: its commit {sha[:7]} is dated after the run")
    cfg = validate_frozen(load_yaml(path))
    return {**cfg, "_frozen_commit": sha, "_frozen_time": int(ts)}


def code_commit(repo: Path = ROOT) -> str:
    """Short HEAD; refuses a dirty tree (CLAUDE.md §0 rule 11)."""
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise FrozenConfigError("the tree has uncommitted changes: commit before a reported run")
    return _git(repo, "rev-parse", "--short=7", "HEAD")


# ------------------------------------------------------------------------------------ inputs


@dataclass(frozen=True)
class Trace:
    """Certified peak intervals (K) at successive PCG checks of one design's audit (N3.3)."""

    iters: np.ndarray
    T_lo: np.ndarray
    T_hi: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.iters)
        if n == 0 or len(self.T_lo) != n or len(self.T_hi) != n:
            raise InputError("a trace needs >= 1 check with iters, T_lo, T_hi of equal length")
        if np.any(np.diff(self.iters) <= 0) or np.any(np.asarray(self.T_lo) > self.T_hi):
            raise InputError("trace iterations must increase and T_lo <= T_hi at every check")


@dataclass
class Trajectory:
    name: str
    proposer: str
    level: int | None
    seed: int
    lam: float | None  # P1's λ (1/K) for G3's exact tilted weights; None otherwise
    x_hash: list[str]
    design_vector: np.ndarray
    peak_hat: np.ndarray
    r_hot: np.ndarray
    sd_hot: np.ndarray
    r_hot_raw: np.ndarray
    group: list[str]
    m_hot: int | None = None  # hot cells r_hot was computed with (M3.3 metadata)

    def __len__(self) -> int:
        return len(self.x_hash)

    @classmethod
    def from_parquet(cls, path: Path) -> Trajectory:
        tab = pq.read_table(path)
        d = tab.to_pydict()
        t = d["t"]
        if t != list(range(len(t))):
            raise InputError(f"{path}: rows must be t = 0, 1, ... in order")
        one = {}
        for c in ("proposer", "aggressiveness", "seed"):
            vals = set(d[c])
            if len(vals) != 1:
                raise InputError(f"{path}: column {c} must be constant, got {sorted(vals)}")
            one[c] = vals.pop()
        meta = json.loads((tab.schema.metadata or {}).get(b"twinheat", b"{}"))
        lam = None
        if one["proposer"] == "P1":
            lam = float(meta["config"]["lambda_per_K"][one["aggressiveness"] - 1])
        return cls(name=Path(path).stem, proposer=one["proposer"], level=one["aggressiveness"],
                   seed=int(one["seed"]), lam=lam, x_hash=list(d["x_hash"]),
                   design_vector=np.asarray(d["design_vector"], dtype=np.float64),
                   peak_hat=np.asarray(d["peak_hat"], dtype=np.float64),
                   r_hot=np.asarray(d["r_hot"], dtype=np.float64),
                   sd_hot=np.asarray(d["sd_hot"], dtype=np.float64),
                   r_hot_raw=np.asarray(d["r_hot_raw"], dtype=np.float64),
                   group=list(d["group"]), m_hot=meta.get("m_hot"))


@dataclass
class Labels:
    oracle: dict[str, float]  # x_hash -> true peak rise (K), 1e-8 solve (N3.2)
    iters_full: dict[str, int | None]  # PCG iterations from μ to 1e-8 (for H7's baseline)
    traces: dict[str, Trace] = field(default_factory=dict)  # N3.3, E14 only

    @classmethod
    def from_parquet(cls, path: Path, traces_path: Path | None = None) -> Labels:
        d = pq.read_table(path).to_pydict()
        oracle: dict[str, float] = {}
        iters: dict[str, int | None] = {}
        for k, (h, v) in enumerate(zip(d["x_hash"], d["oracle_peak"], strict=True)):
            if v is None or not math.isfinite(v):
                raise InputError(f"{path}: row {k} ({h[:12]}…) has no finite oracle_peak")
            if h in oracle and oracle[h] != v:
                raise InputError(f"{path}: x_hash {h[:12]}… has two different labels")
            oracle[h] = float(v)
            it = d.get("iters_full", [None] * len(d["x_hash"]))[k]
            iters[h] = None if it is None else int(it)
        traces = {}
        if traces_path is not None:
            t = pq.read_table(traces_path).to_pydict()
            for h, it, lo, hi in zip(t["x_hash"], t["check_iters"], t["T_lo"], t["T_hi"],
                                     strict=True):
                traces[h] = Trace(np.asarray(it), np.asarray(lo, dtype=np.float64),
                                  np.asarray(hi, dtype=np.float64))
        return cls(oracle, iters, traces)


@dataclass
class Calibration:
    """Labelled designs drawn from p₀ (design-space density): calibration scores for G2/G3 and
    the starting q of G4–G7, G3's calibration weights, G7's mean scale."""

    peak_hat: np.ndarray
    oracle: np.ndarray
    r_hot: np.ndarray
    sd_hot: np.ndarray
    design_vector: np.ndarray

    @classmethod
    def from_parquet(cls, path: Path, labels: Labels) -> Calibration:
        d = pq.read_table(path).to_pydict()
        try:
            oracle = [labels.oracle[h] for h in d["x_hash"]]
        except KeyError as e:
            raise InputError(f"{path}: no oracle label for calibration design {e}") from e
        return cls(np.asarray(d["peak_hat"], dtype=np.float64), np.asarray(oracle),
                   np.asarray(d["r_hot"], dtype=np.float64),
                   np.asarray(d["sd_hot"], dtype=np.float64),
                   np.asarray(d["design_vector"], dtype=np.float64))


def scale(r_hot: np.ndarray, sd_hot: np.ndarray, fz: dict[str, Any]) -> np.ndarray:
    """s = r_hot + λ·sd_hot + ε (K), CLAUDE.md §8.1."""
    s = np.asarray(r_hot) + fz["lam"] * np.asarray(sd_hot) + fz["eps_K"]
    if not (np.isfinite(s).all() and (s > 0).all()):
        raise InputError("the label-free scale s must be finite and > 0")
    return s


def cal_scores(cal: Calibration, fz: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    s = scale(cal.r_hot, cal.sd_hot, fz)
    return np.abs(cal.peak_hat - cal.oracle) / s, s


# ----------------------------------------------------------------------------- bounds, audits


def lemma1_slack(*, alpha: float, gamma: float, p: float, T: int, delta: float) -> float:
    """C/(γpT) + (2/p)√(ln(4/δ)/T), C = max(α, 1 − α) + γ (α₁ = α): CLAUDE.md §8.2."""
    if not (0 < alpha < 1 and gamma > 0 and 0 < p <= 1 and 0 < delta < 1):
        raise ValueError(f"need 0 < alpha < 1, gamma > 0, 0 < p <= 1, 0 < delta < 1; got "
                         f"{alpha}, {gamma}, {p}, {delta}")
    if T <= 0:
        return math.inf
    C = max(alpha, 1 - alpha) + gamma
    return C / (gamma * p * T) + (2 / p) * math.sqrt(math.log(4 / delta) / T)


def lemma1_bound(*, alpha: float, gamma: float, p: float, T: int, delta: float) -> float:
    return alpha + lemma1_slack(alpha=alpha, gamma=gamma, p=p, T=T, delta=delta)


def certified_audit(peak_hat: float, s: float, q: float, eps_S: float, trace: Trace,
                    oracle: float) -> tuple[float, float, int, str]:
    """(T⁻, T⁺, iterations, stop reason) of an audit replayed from its trace (Lemma 1′, §8.3):
    stop at the first check whose score interval [S⁻, S⁺] lies entirely ≤ q or > q
    ("decided"), or is narrower than ε_S ("width"), else at the last check ("end"). The score
    interval and the rule are certify/peak_interval's, so the replay stops where a live
    audit_until_decided would. Every check up to the stop must contain the oracle peak (B7)."""
    for k in range(len(trace.iters)):
        lo, hi = float(trace.T_lo[k]), float(trace.T_hi[k])
        if not lo <= oracle <= hi:
            raise CertificateViolation(f"B7 violated: check {k} interval [{lo}, {hi}] K does not "
                                       f"contain the oracle peak {oracle} K")
        s_lo, s_hi = score_interval(peak_hat, lo, hi, s)
        it = int(trace.iters[k])
        why = decide(s_lo, s_hi, q, eps_S)
        if why is not None:
            return lo, hi, it, "width" if why == "width" else "decided"
    return lo, hi, it, "end"


# ------------------------------------------------------------------------------- gate factory


def g1_flag_rate(g5_summary: dict[str, Any]) -> float:
    """G1's f at G5's realised exact-solve budget on the same trajectory (§11.1)."""
    return flag_rate_for_budget(int(g5_summary["solves"]), int(g5_summary["T"]))


def build_gate(name: str, fz: dict[str, Any], cal: Calibration, traj: Trajectory,
               coin: AuditCoin, seed: int, *, p: float, p_min: float, f: float | None,
               ratio_traj: Trajectory | None) -> BaseGate:
    a, tau, gam, W = fz["alpha"], fz["tau_K"], fz["gamma"], fz["W"]
    scores, cal_s = cal_scores(cal, fz)
    if name == "G0":
        return NoGate()
    if name == "G1":
        if f is None:
            raise ValueError("G1 needs f (g1_flag_rate of the G5 run on the same trajectory)")
        g1 = fz["g1"]
        off = g1["c_events"] == "off"
        return DeepOHeatV2Gate(f=f, window=int(g1["window"]), residual=g1["residual"],
                               c_events=None if off else int(g1["c_events"]),
                               delta_n=None if off else int(g1["delta_n"]))
    if name == "G2":
        return StaticSplitConformal(alpha=a, tau=tau, cal_scores=scores)
    if name == "G3":
        if traj.proposer == "P1":
            if traj.lam is None:
                raise InputError(f"{traj.name}: P1 trajectory without its λ")
            lam = traj.lam
            cal_lw = tilted_log_weights(cal.peak_hat, lam)

            def test_lw(gi: GateInput) -> float:
                return float(tilted_log_weights([gi.peak_hat], lam)[0])
        else:
            if ratio_traj is None or ratio_traj.seed == traj.seed:
                raise InputError(f"{traj.name}: G3's classifier ratio is fitted on another seed's "
                                 "trajectory of the same proposer and level (cross-fitting)")
            ratio = ClassifierDensityRatio(cal.design_vector, ratio_traj.design_vector,
                                           C=float(fz["g3"]["classifier_C"]))
            cal_lw = ratio.log_weight(cal.design_vector)
            lw = ratio.log_weight(traj.design_vector)
            by_id = {f"t{t}": float(lw[t]) for t in range(len(traj))}

            def test_lw(gi: GateInput) -> float:
                return by_id[gi.design_id]
        return WeightedSplitConformal(alpha=a, tau=tau, cal_scores=scores, cal_log_w=cal_lw,
                                      test_log_w=test_lw)
    if name == "G4":
        return ACIFlaggedOnly(alpha=a, gamma=gam, W=W, tau=tau, cal_scores=scores,
                              ablation_flagged_only=True)
    if name == "G5":
        return AuditedACIGate(alpha=a, gamma=gam, W=W, tau=tau, p=p, cal_scores=scores, coin=coin)
    if name == "G6":
        g6 = fz["g6"]
        return MultiValidGate(alpha=a, tau=tau, p=p, groups=G.ALL_GROUPS, m=int(g6["m_buckets"]),
                              r=int(g6["r"]), eps=float(g6["eps"]),
                              noise_width=float(g6["noise_width"]), coin=coin, seed=seed,
                              membership=G.membership)
    if name == "G7":
        return ActiveAuditGate(alpha=a, gamma=gam, W=W, tau=tau, p_bar=p, p_min=p_min,
                               kappa=float(fz["kappa"]), u_cal=cal_s, cal_scores=scores,
                               coin=coin)
    raise ValueError(f"unknown gate {name!r}; expected one of {GATES}")


# ------------------------------------------------------------------------------------ replay


@dataclass
class Run:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    group_rows: list[dict[str, Any]]


def replay(traj: Trajectory, labels: Labels, cal: Calibration, fz: dict[str, Any], gate: str,
           seed: int, *, p: float | None = None, p_min: float | None = None,
           f: float | None = None, audit: str = "exact", eps_S: float | None = None,
           ratio_traj: Trajectory | None = None, tag: str = "replay") -> Run:
    """Replay one gate on one trajectory with coin seed `seed` (see the module docstring)."""
    if audit not in ("exact", "certified"):
        raise ValueError(f"audit must be 'exact' or 'certified', got {audit!r}")
    if audit == "certified" and eps_S is None:
        raise ValueError("certified audits need eps_S")
    p = float(fz["p_audit"] if p is None else p)
    p_min = float(fz["p_min"] if p_min is None else p_min)
    T = len(traj)
    s = scale(traj.r_hot, traj.sd_hot, fz)
    try:
        oracle = np.array([labels.oracle[h] for h in traj.x_hash])
    except KeyError as e:
        raise InputError(f"{traj.name}: no oracle label for x_hash {e}") from e
    coin = AuditCoin(seed=seed)
    g = build_gate(gate, fz, cal, traj, coin, seed, p=p, p_min=p_min, f=f, ratio_traj=ratio_traj)
    commit_fn: Callable | None = getattr(g, "commit", None)
    commit7 = str(fz.get("_run_commit", "0000000"))
    run_id = f"{tag}_{gate}_{traj.name}-{commit7}-{seed}"
    tau = fz["tau_K"]
    rows: list[dict[str, Any]] = []
    n = {"accepted": 0, "flagged": 0, "audited": 0, "solves": 0, "accepted_large": 0,
         "label_updates": 0, "audit_iters": 0, "audit_iters_full": 0, "iters_unknown": 0}
    stops: dict[str, int] = {}
    acc = np.zeros(T, dtype=bool)
    large = np.abs(traj.peak_hat - oracle) > tau
    for t in range(T):
        h = traj.x_hash[t]
        gi = GateInput(design_id=f"t{t}", x_hash=h, peak_hat=float(traj.peak_hat[t]),
                       s=float(s[t]), r_hot=float(traj.r_hot[t]), sd_hot=float(traj.sd_hot[t]),
                       proposer=traj.proposer, aggressiveness=traj.level, group=traj.group[t],
                       r_hot_raw=float(traj.r_hot_raw[t]))
        coin.commit(gi.design_id)
        dec = commit_fn(gi) if commit_fn is not None else g.decide(gi)
        audited = coin.draw(gi.design_id, dec.p_audit)
        outcome, iters = None, 0
        full = labels.iters_full.get(h)
        if audited:
            if audit == "exact":
                lo = hi = float(oracle[t])
                iters = 0 if full is None else full
            else:
                if h not in labels.traces:
                    raise InputError(f"{traj.name} t = {t}: no certified-audit trace for {h[:12]}…")
                lo, hi, iters, why = certified_audit(gi.peak_hat, gi.s, dec.q_t, float(eps_S),
                                                     labels.traces[h], float(oracle[t]))
                stops[why] = stops.get(why, 0) + 1
            (outcome,) = g.update([AuditRecord(gi.design_id, True, gi.peak_hat, gi.s, lo, hi,
                                               int(iters))])
            n["label_updates"] += 1
            n["audited"] += 1
            n["audit_iters"] += int(iters)
            if full is None:
                n["iters_unknown"] += 1
            else:
                n["audit_iters_full"] += full
        elif gate == "G4" and dec.decision == "flag":
            o = float(oracle[t])  # a flagged design is verified; G4 learns from that label
            g.update([AuditRecord(gi.design_id, False, gi.peak_hat, gi.s, o, o, 0)])
            n["label_updates"] += 1
        flag = dec.decision == "flag"
        acc[t] = not flag
        n["accepted"] += int(not flag)
        n["flagged"] += int(flag)
        n["solves"] += int(flag or audited)
        n["accepted_large"] += int((not flag) and large[t])
        rows.append(g.log_row(gi, dec, run_id=run_id, seed=seed, t=t, outcome=outcome,
                              iters_used=int(iters), oracle_peak=float(oracle[t])))
    if gate == "G0":
        n["solves"] = 1  # §11.1: no gate; the final design is verified once
    bound = lemma1_bound(alpha=fz["alpha"], gamma=fz["gamma"], p=p, T=T, delta=fz["delta"])
    m1 = n["accepted_large"] / T
    summary = {
        "gate": gate, "trajectory": traj.name, "proposer": traj.proposer, "level": traj.level,
        "seed": seed, "T": T, "p": p, "p_min": p_min if gate == "G7" else None,
        "f": f if gate == "G1" else None, "audit": audit, "eps_S": eps_S, **n,
        "M1_all": m1,
        "M1_accepted": n["accepted_large"] / n["accepted"] if n["accepted"] else math.nan,
        "bound": bound, "within_bound": bool(m1 <= bound), "alpha": fz["alpha"],
        "stops": json.dumps(stops, sort_keys=True),
    }
    group_rows = []
    member = [set(gs.split(G.SEP)) for gs in traj.group]
    for name in G.ALL_GROUPS:
        idx = np.array([name in m for m in member])
        T_g = int(idx.sum())
        if T_g == 0:
            continue
        a_g = int((acc & idx).sum())
        al_g = int((acc & idx & large).sum())
        slack = lemma1_slack(alpha=fz["alpha"], gamma=fz["gamma"], p=p, T=T_g, delta=fz["delta"])
        group_rows.append({
            "gate": gate, "trajectory": traj.name, "proposer": traj.proposer,
            "level": traj.level, "seed": seed, "p": p, "group": name, "T_g": T_g,
            "accepted_g": a_g, "accepted_large_g": al_g, "M1g": al_g / T_g,
            "M1g_accepted": al_g / a_g if a_g else math.nan, "alpha": fz["alpha"],
            "slack_g": slack, "within_g": bool(al_g / T_g <= fz["alpha"] + slack),
            "slack_kind": SLACK_KIND})
    return Run(rows, summary, group_rows)


# ------------------------------------------------------------------------- statistics, verdicts


def bootstrap_ci(values: Sequence[float], *, resamples: int, seed: int,
                 level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap of the mean over seeds (CLAUDE.md §11.5)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("bootstrap needs >= 1 finite value")
    if np.all(v == v[0]):
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.Generator(np.random.PCG64(seed))
    means = v[rng.integers(0, v.size, size=(resamples, v.size))].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.percentile(means, [100 * a, 100 * (1 - a)])
    return float(v.mean()), float(lo), float(hi)


def _frac(rows: list[dict[str, Any]], key: str = "within_bound") -> float:
    return float(np.mean([bool(r[key]) for r in rows])) if rows else math.nan


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else math.nan


def verdict_h1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """H1 (§11.4): G5 within the Lemma 1 bound in >= 95 % of runs for every proposer and level;
    G1, G2, G4 exceed it as aggressiveness rises; G3 holds only for P1. `rows` are run summaries
    at the headline p."""
    present = sorted({r["proposer"] for r in rows})
    missing = [p for p in PROPOSERS if p not in present]
    parts: dict[str, Any] = {}
    g5 = [r for r in rows if r["gate"] == "G5"]
    parts["G5_fraction_within"] = _frac(g5)
    parts["G5_ok"] = bool(g5) and parts["G5_fraction_within"] >= RUN_FRACTION
    for g in ("G1", "G2", "G4"):
        hits = []
        for p in present:
            r1 = [r for r in rows if r["gate"] == g and r["proposer"] == p and r["level"] == 1]
            r3 = [r for r in rows if r["gate"] == g and r["proposer"] == p and r["level"] == 3]
            if r1 and r3 and _mean(r3, "M1_all") > _mean(r3, "bound") \
                    and _mean(r3, "M1_all") > _mean(r1, "M1_all"):
                hits.append(p)
        parts[f"{g}_exceeds_rising_for"] = hits
        parts[f"{g}_ok"] = bool(hits)
    g3 = {p: _frac([r for r in rows if r["gate"] == "G3" and r["proposer"] == p])
          for p in present}
    parts["G3_fraction_within"] = g3
    others = [p for p in present if p != "P1"]
    parts["G3_ok"] = ("P1" in g3 and g3["P1"] >= RUN_FRACTION and bool(others)
                      and any(g3[p] < RUN_FRACTION for p in others))
    ok = all(parts[k] for k in ("G5_ok", "G1_ok", "G2_ok", "G4_ok", "G3_ok"))
    return {"hypothesis": "H1", "verdict": "SUPPORTED" if ok else "NOT SUPPORTED",
            "parts": parts, "missing": missing,
            "note": "M1 over all proposals vs the §8.2 bound at the headline p; 'exceeds as "
                    "aggressiveness rises' = mean M1 at level 3 above the bound and above level 1"}


def verdict_h7(rows: list[dict[str, Any]], *, eps_S: float) -> dict[str, Any]:
    """H7: certified audits (at the frozen ε_S) keep M1 within the bound in >= 95 % of runs and
    spend >= 50 % fewer audit iterations than audits solved to 1e-8."""
    sel = [r for r in rows if r.get("eps_S") == eps_S]
    full = sum(r["audit_iters_full"] for r in sel)
    if not sel or full == 0:
        return {"hypothesis": "H7", "verdict": "UNTESTABLE",
                "note": "no certified runs at this eps_S, or no iters_full in the labels"}
    saving = 1 - sum(r["audit_iters"] for r in sel) / full
    frac = _frac(sel)
    ok = frac >= RUN_FRACTION and saving >= H7_ITER_SAVING
    return {"hypothesis": "H7", "verdict": "SUPPORTED" if ok else "NOT SUPPORTED",
            "parts": {"fraction_within": frac, "iteration_saving": saving, "eps_S": eps_S}}


def verdict_h8(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """H8 (empirical): G6 keeps M1g <= α + slack_g for every group in >= 95 % of runs, and G5
    misses that in >= 1 group under the aggressive (level 3) proposers."""
    def ok_frac(rows):
        return float(np.mean([r["M1g"] <= r["alpha"] + r["slack_g"] for r in rows]))

    g6 = [r for r in group_rows if r["gate"] == "G6"]
    g5 = [r for r in group_rows if r["gate"] == "G5" and r["level"] == 3]
    names = sorted({r["group"] for r in g6})
    g6_by = {n: ok_frac([r for r in g6 if r["group"] == n]) for n in names}
    g5_by = {n: ok_frac([r for r in g5 if r["group"] == n])
             for n in sorted({r["group"] for r in g5})}
    g6_ok = bool(g6_by) and all(v >= RUN_FRACTION for v in g6_by.values())
    g5_miss = [n for n, v in g5_by.items() if v < RUN_FRACTION]
    ok = g6_ok and bool(g5_miss)
    return {"hypothesis": "H8",
            "verdict": ("SUPPORTED" if ok else "NOT SUPPORTED") + " (empirical)",
            "parts": {"G6_fraction_within_by_group": g6_by, "G5_level3_groups_missed": g5_miss},
            "note": SLACK_KIND}


def verdict_h9(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """H9 (exploratory): some G7 setting meets H1's criterion with >= 30 % fewer audits than the
    cheapest G5 setting that meets it (interpretation of §11.4; M to confirm)."""
    def cheapest(gate, keys):
        best = None
        configs = sorted({tuple(r[k] for k in keys) for r in rows if r["gate"] == gate})
        for c in configs:
            sel = [r for r in rows if r["gate"] == gate and tuple(r[k] for k in keys) == c]
            if _frac(sel) >= RUN_FRACTION:
                m = _mean(sel, "audited")
                if best is None or m < best[1]:
                    best = (c, m)
        return best

    g5, g7 = cheapest("G5", ("p",)), cheapest("G7", ("p", "p_min"))
    if g5 is None or g7 is None:
        return {"hypothesis": "H9", "verdict": "NOT SUPPORTED",
                "parts": {"G5_cheapest_valid": g5, "G7_cheapest_valid": g7},
                "note": "no valid G5 or G7 setting"}
    ok = g7[1] <= (1 - H9_AUDIT_SAVING) * g5[1]
    return {"hypothesis": "H9", "verdict": "SUPPORTED" if ok else "NOT SUPPORTED",
            "parts": {"G5_cheapest_valid": g5, "G7_cheapest_valid": g7,
                      "audit_ratio": g7[1] / g5[1]},
            "note": "exploratory; 'fewer audits' read as: cheapest valid G7 vs cheapest valid G5"}


def ci_table(rows: list[dict[str, Any]], keys: Sequence[str], metric: str,
             fz: dict[str, Any]) -> list[dict[str, Any]]:
    """Mean and bootstrap 95 % CI over seeds of `metric`, per combination of `keys`."""
    out = []
    for c in sorted({tuple(r[k] for k in keys) for r in rows}, key=str):
        vals = [r[metric] for r in rows if tuple(r[k] for k in keys) == c]
        m, lo, hi = bootstrap_ci(vals, resamples=int(fz["bootstrap"]["resamples"]),
                                 seed=int(fz["bootstrap"]["seed"]))
        out.append({**dict(zip(keys, c, strict=True)), "metric": metric, "n_seeds": len(vals),
                    "mean": m, "ci_lo": lo, "ci_hi": hi})
    return out


# ------------------------------------------------------------------------------- experiment IO


@dataclass
class Inputs:
    trajectories: list[Trajectory]
    labels: Labels
    cal: Calibration
    digest: str  # SHA-256 over every input file, for figure footers


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(root: Path, *, traces: bool = False) -> Inputs:
    """root/trajectories/*.parquet (M3.3), root/labels.parquet (N3.2), root/calibration.parquet
    (p₀ designs), root/audit_traces.parquet (N3.3, E14 only)."""
    root = Path(root)
    files = sorted((root / "trajectories").glob("traj_*.parquet"))
    need = [root / "labels.parquet", root / "calibration.parquet"]
    if traces:
        need.append(root / "audit_traces.parquet")
    absent = [str(p) for p in need if not p.exists()] + ([] if files else ["trajectories/*"])
    if absent:
        raise InputError(f"missing inputs: {absent} (docs/gate_experiment_inputs.md)")
    labels = Labels.from_parquet(root / "labels.parquet",
                                 root / "audit_traces.parquet" if traces else None)
    trajs = [Trajectory.from_parquet(f) for f in files]
    cal = Calibration.from_parquet(root / "calibration.parquet", labels)
    h = hashlib.sha256()
    for f in [*files, *need]:
        h.update(_sha(f).encode())
    return Inputs(trajs, labels, cal, h.hexdigest())


def partner(trajs: list[Trajectory], traj: Trajectory) -> Trajectory | None:
    """Another seed's trajectory of the same proposer and level (G3's cross-fitted ratio)."""
    same = sorted((t for t in trajs if t.proposer == traj.proposer and t.level == traj.level
                   and t.seed != traj.seed), key=lambda t: t.seed)
    later = [t for t in same if t.seed > traj.seed]
    return (later or same or [None])[0]


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def footer(fz: dict[str, Any], inputs_digest: str) -> str:
    """CLAUDE.md §4 figure footer: commit, data hash, seeds, hardware, date."""
    hw = platform.processor() or platform.machine()
    frozen = str(fz.get("_frozen_commit", "?"))[:7]
    return (f"commit {fz.get('_run_commit', '?')} · frozen {frozen} · inputs {inputs_digest[:12]}"
            f" · seeds {fz['seeds']} · {hw} · {time.strftime('%Y-%m-%d')}")


def prepare(frozen: Path, inputs: Path, *, traces: bool = False
            ) -> tuple[dict[str, Any], Inputs]:
    """Guard + inputs for an E-script: committed tree, committed complete frozen.yaml."""
    commit = code_commit()
    fz = check_frozen(frozen)
    fz["_run_commit"] = commit
    data = load_inputs(inputs, traces=traces)
    bad = sorted({t.m_hot for t in data.trajectories if t.m_hot != fz["m_hot"]}, key=str)
    if bad:
        raise InputError(f"trajectories were computed with m_hot {bad}, frozen.yaml has "
                         f"{fz['m_hot']}: r_hot would mean something else")
    return fz, data


def run_matrix(fz: dict[str, Any], data: Inputs, configs: list[dict[str, Any]], *,
               tag: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Every config on every trajectory, coin seed = the trajectory's seed. A config is
    {"gate", optional "p", "p_min", "audit", "eps_S"}; G1 takes f from the G5 run at the same p
    on the same trajectory (which is run first if no config asks for it)."""
    summaries, groups, logs = [], [], []
    for traj in data.trajectories:
        if traj.seed not in fz["seeds"]:
            continue
        ratio = partner(data.trajectories, traj)
        g5_cache: dict[float, dict[str, Any]] = {}
        for c in configs:
            kw = {k: c[k] for k in ("p", "p_min", "audit", "eps_S") if k in c}
            if c["gate"] == "G1":
                p1 = float(c.get("p", fz["p_audit"]))
                if p1 not in g5_cache:
                    g5_cache[p1] = replay(traj, data.labels, data.cal, fz, "G5", traj.seed,
                                          p=p1, tag=tag).summary
                kw["f"] = g1_flag_rate(g5_cache[p1])
            run = replay(traj, data.labels, data.cal, fz, c["gate"], traj.seed,
                         ratio_traj=ratio if c["gate"] == "G3" else None, tag=tag, **kw)
            if c["gate"] == "G5" and kw.get("audit", "exact") == "exact":
                g5_cache.setdefault(run.summary["p"], run.summary)
            summaries.append(run.summary)
            groups += run.group_rows
            logs += run.rows
    return summaries, groups, logs


def write_outputs(out: Path, name: str, summaries: list[dict], groups: list[dict],
                  verdicts: list[dict], fz: dict[str, Any], data: Inputs,
                  ci_keys: Sequence[str], logs: list[dict] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(summaries, out / f"{name}_runs.csv")
    if groups:
        write_csv(groups, out / f"{name}_groups.csv")
    ci = []
    for metric in ("M1_all", "M1_accepted", "solves", "audited", "audit_iters"):
        ok = [r for r in summaries if isinstance(r.get(metric), int | float)
              and math.isfinite(r[metric])]
        ci += ci_table(ok, ci_keys, metric, fz)
    write_csv(ci, out / f"{name}_ci.csv")
    if logs:
        write_csv(logs, out / f"{name}_audit_log.csv")
    (out / f"{name}_verdicts.json").write_text(json.dumps({
        "verdicts": verdicts, "run_commit": fz.get("_run_commit"),
        "frozen_commit": fz.get("_frozen_commit"), "inputs_sha256": data.digest,
        "footer": footer(fz, data.digest)}, indent=2, default=str) + "\n", encoding="utf-8")


def print_verdicts(verdicts: list[dict[str, Any]]) -> None:
    for v in verdicts:
        print(f"{v['hypothesis']}: {v['verdict']}")
        print("   " + json.dumps(v.get("parts", {}), default=str))
