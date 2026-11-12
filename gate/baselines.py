"""Comparison gates G0-G4 (CLAUDE.md §11.1; prompt M3.2). All share gate/base.py's interface and
are compared with G5 at equal exact-solve budget (E7, E8).

G0 NoGate                 accept everything; the run verifies only its final design (§11.1).
G1 DeepOHeatV2Gate        DeepOHeat-v2 Algorithm 1 (Yu et al., arXiv 2608.16080v1;
                          docs/refs/deepoheat_v2_alg1.txt), with the D13 choices.
G2 StaticSplitConformal   split conformal calibrated once on p₀ (Angelopoulos & Bates, arXiv
                          2107.07511v6, Sec. 1.1 and Theorem 1; docs/refs/angelopoulos_bates_cp.txt;
                          quantile level D8).
G3 WeightedSplitConformal weighted split conformal under covariate shift (Tibshirani et al. 2019, as
                          given in Angelopoulos & Bates Sec. 4.5 and Theorem 3; same refs file).
                          Weights: exact for P1 (tilted_log_weights), else ClassifierDensityRatio.
G4 ACIFlaggedOnly         ACI updated on flagged labels only: the selection-bias ablation (§11.1,
                          §18.1 Prop. B.14). It deliberately breaks B8 and says so by a named flag.

None of G0-G4 uses the audit coin: their p_audit is 0. Hyperparameters have no defaults (D4).
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from gate.aci_audit import _CEIL_TOL, _check_scores, conformal_quantile
from gate.base import AuditOutcome, AuditRecord, BaseGate, GateDecision, GateInput

__all__ = ["ACIFlaggedOnly", "ClassifierDensityRatio", "DeepOHeatV2Gate", "NoGate",
           "RATIO_CLIP", "StaticSplitConformal", "WeightedSplitConformal",
           "flag_rate_for_budget", "tilted_log_weights", "weighted_conformal_quantile"]

RATIO_CLIP = (0.05, 20.0)  # classifier density-ratio clip, from the M3.2 prompt


def _no_labels(gate_id: str, records: list[AuditRecord], why: str) -> list[AuditOutcome]:
    if records:
        raise ValueError(f"{gate_id} takes no labels ({why}); got {len(records)} record(s)")
    return []


def _check_state_gate(state: dict[str, Any], gate_id: str) -> None:
    if state.get("gate") != gate_id:
        raise ValueError(f"state is for gate {state.get('gate')!r}, not {gate_id}")


def _check_tau(tau: float) -> float:
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be finite and > 0 (K), got {tau}")
    return float(tau)


def _check_alpha(alpha: float) -> float:
    if not (math.isfinite(alpha) and 0 < alpha < 1):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return float(alpha)


def _bound_decision(inp: GateInput, q: float, tau: float, p_audit: float = 0.0,
                    alpha_t: float | None = None) -> GateDecision:
    B = math.inf if math.isinf(q) else q * inp.s
    return GateDecision(design_id=inp.design_id, B=B, decision="accept" if B <= tau else "flag",
                        p_audit=p_audit, group=inp.group, q_t=q, alpha_t=alpha_t)


# ====================================================================== G0


class NoGate(BaseGate):
    """G0: no gate. Every proposal is accepted on the surrogate; the run verifies its final
    design once (CLAUDE.md §11.1). B and q are not applicable (∞, logged as null)."""

    gate_id = "G0"

    def decide(self, inp: GateInput) -> GateDecision:
        return GateDecision(design_id=inp.design_id, B=math.inf, decision="accept", p_audit=0.0,
                            group=inp.group, q_t=math.inf, alpha_t=None)

    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]:
        return _no_labels(self.gate_id, records, "no gate")

    def state_dict(self) -> dict[str, Any]:
        return {"gate": self.gate_id}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _check_state_gate(state, self.gate_id)


# ====================================================================== G1


def flag_rate_for_budget(n_solves: int, n_proposals: int) -> float:
    """f such that G1 flags about n_solves of n_proposals (§11.1: f tuned to the solve budget).
    The bootstrap steps and the C rule (if on) change the realised count; the E8 runner checks
    the realised number of solves against the budget."""
    if not (isinstance(n_solves, int) and isinstance(n_proposals, int)
            and 0 < n_solves < n_proposals):
        raise ValueError(f"need 0 < n_solves < n_proposals (ints), got {n_solves}, {n_proposals}")
    return n_solves / n_proposals


class DeepOHeatV2Gate(BaseGate):
    """G1: DeepOHeat-v2's hotspot trust gate, Algorithm 1 lines 2 and 5-17 (Yu et al., arXiv
    2608.16080v1; docs/refs/deepoheat_v2_alg1.txt), with the D13 choices (docs/decisions.md):

      - signal: GateInput.r_hot (K, divided by diag A, shared m) when residual="scaled"; the
        paper's raw residual GateInput.r_hot_raw (W) when residual="raw".
      - bootstrap (line 2): the first `window` committed proposals are pushed and accepted.
      - then (lines 6-7): push r_hot of every committed proposal into the FIFO window, so the
        window includes the current value, and flag if r_hot > Q_{1−f}(W), with Q the
        ⌈(1−f)|W|⌉-th smallest value ("higher", as in D8).
      - C rule (lines 5, 10-14), off when c_events is None: each flag is one refinement; an
        event happens when the refinement count reaches (e+1)·ΔN; after c_events events the
        gate accepts everything and stops pushing.

    The threshold is label-free and controls the flag rate, not the error rate. G1 uses no
    labels and no audits (p_audit = 0); B and q are not applicable. decide() returns exactly
    what commit() would return for the same proposal, without changing the window.
    """

    gate_id = "G1"

    def __init__(self, *, f: float, window: int, residual: Literal["scaled", "raw"],
                 c_events: int | None, delta_n: int | None) -> None:
        if not (math.isfinite(f) and 0 < f < 1):
            raise ValueError(f"f must lie in (0, 1), got {f}")
        if not (isinstance(window, int) and not isinstance(window, bool) and window >= 1):
            raise ValueError(f"window must be an int >= 1, got {window!r}")
        if residual not in ("scaled", "raw"):
            raise ValueError(f"residual must be 'scaled' or 'raw', got {residual!r}")
        if c_events is not None:
            if not (isinstance(c_events, int) and c_events >= 1):
                raise ValueError(f"c_events must be None or an int >= 1, got {c_events!r}")
            if not (isinstance(delta_n, int) and delta_n >= 1):
                raise ValueError(f"c_events needs delta_n (an int >= 1), got {delta_n!r}")
        elif delta_n is not None:
            raise ValueError("delta_n is only used with c_events; pass None for both")
        self.f, self.window, self.residual = float(f), window, residual
        self.c_events, self.delta_n = c_events, delta_n
        self._k = math.ceil((1 - self.f) * window - _CEIL_TOL)  # rank of Q_{1−f} in a full window
        self._W: deque[float] = deque(maxlen=window)
        self._committed: set[str] = set()
        self.n_flags = 0  # refinements so far (Algorithm 1: |D_online|)
        self.events = 0  # e

    def _signal(self, inp: GateInput) -> float:
        if self.residual == "scaled":
            return inp.r_hot
        if inp.r_hot_raw is None:
            raise ValueError(f"G1 with residual='raw' needs GateInput.r_hot_raw for "
                             f"{inp.design_id!r} (gate/scale.hot_residual_raw)")
        return inp.r_hot_raw

    @property
    def active(self) -> bool:
        return self.c_events is None or self.events < self.c_events

    def _outcome(self, r: float) -> tuple[bool, bool]:
        """(push, flag) for signal r given the current state."""
        if not self.active:
            return False, False  # line 5: solver no longer active
        if len(self._W) < self.window:
            return True, False  # line 2: bootstrap
        w = sorted([*list(self._W)[1:], r])  # the push drops the oldest value
        return True, r > w[self._k - 1]

    def _decision(self, inp: GateInput, flag: bool) -> GateDecision:
        return GateDecision(design_id=inp.design_id, B=math.inf,
                            decision="flag" if flag else "accept", p_audit=0.0, group=inp.group,
                            q_t=math.inf, alpha_t=None)

    def decide(self, inp: GateInput) -> GateDecision:
        return self._decision(inp, self._outcome(self._signal(inp))[1])

    def commit(self, inp: GateInput) -> GateDecision:
        if inp.design_id in self._committed:
            raise ValueError(f"design {inp.design_id!r} already committed to G1")
        r = self._signal(inp)
        push, flag = self._outcome(r)
        self._committed.add(inp.design_id)
        if push:
            self._W.append(r)
        if flag:
            self.n_flags += 1
            if self.c_events is not None and self.n_flags >= (self.events + 1) * self.delta_n:
                self.events += 1  # lines 10-14 (the retraining itself happens outside)
        return self._decision(inp, flag)

    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]:
        return _no_labels(self.gate_id, records, "its threshold is a label-free percentile")

    def state_dict(self) -> dict[str, Any]:
        return {"gate": self.gate_id, "f": self.f, "window": self.window,
                "residual": self.residual, "c_events": self.c_events, "delta_n": self.delta_n,
                "W": list(self._W), "committed": sorted(self._committed),
                "n_flags": self.n_flags, "events": self.events}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _check_state_gate(state, self.gate_id)
        for k in ("f", "window", "residual", "c_events", "delta_n"):
            if state[k] != getattr(self, k):
                raise ValueError(f"state has {k} = {state[k]!r} but this gate has "
                                 f"{getattr(self, k)!r}; settings are fixed (D4, D13)")
        if len(state["W"]) > self.window:
            raise ValueError(f"state window has {len(state['W'])} values > {self.window}")
        self._W = deque((float(v) for v in state["W"]), maxlen=self.window)
        self._committed = {str(c) for c in state["committed"]}
        self.n_flags, self.events = int(state["n_flags"]), int(state["events"])


# ====================================================================== G2


class StaticSplitConformal(BaseGate):
    """G2: split conformal, calibrated once (Angelopoulos & Bates Sec. 1.1, Theorem 1;
    docs/refs/angelopoulos_bates_cp.txt). q = the ⌈(n+1)(1−α)⌉-th smallest calibration score
    (D8), B = q·s, accept if B ≤ τ. The guarantee lapses once the proposer leaves p₀ or the
    model is retrained (CLAUDE.md §18.2); that is what E7 shows."""

    gate_id = "G2"

    def __init__(self, *, alpha: float, tau: float,
                 cal_scores: Sequence[float] | np.ndarray) -> None:
        self.alpha, self.tau = _check_alpha(alpha), _check_tau(tau)
        self.q_t = conformal_quantile(_check_scores(cal_scores, "cal_scores"), self.alpha)

    def decide(self, inp: GateInput) -> GateDecision:
        return _bound_decision(inp, self.q_t, self.tau)

    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]:
        return _no_labels(self.gate_id, records, "G2 is calibrated once")

    def state_dict(self) -> dict[str, Any]:
        return {"gate": self.gate_id, "alpha": self.alpha, "tau": self.tau, "q_t": self.q_t}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _check_state_gate(state, self.gate_id)
        if (state["alpha"], state["tau"], state["q_t"]) != (self.alpha, self.tau, self.q_t):
            raise ValueError("G2 state does not match this gate's calibration")


# ====================================================================== G3


def weighted_conformal_quantile(scores: Sequence[float] | np.ndarray,
                                log_w: Sequence[float] | np.ndarray, log_w_test: float,
                                alpha: float) -> float:
    """Weighted split-conformal quantile (Angelopoulos & Bates Sec. 4.5):

        p_i = w(X_i) / (Σ_j w(X_j) + w(x)),   q̂(x) = inf{ s_j : Σ_{i: s_i ≤ s_j} p_i ≥ 1 − α },

    +∞ when no s_j qualifies (the test point's own mass sits above every score). Weights are
    given as logs and shifted by their maximum, so a common constant cancels (P1's normalising
    constant is never needed). The comparison allows a relative 1e-12 of the total weight, so
    equal weights reproduce conformal_quantile exactly (e.g. 19 · (1/20) summed in floating
    point is 0.95 − 1e-16)."""
    s = _check_scores(scores, "scores")
    lw = np.asarray(log_w, dtype=np.float64).reshape(-1)
    if lw.shape != s.shape:
        raise ValueError(f"log_w has {lw.size} entries for {s.size} scores")
    lt = float(log_w_test)
    if np.isnan(lw).any() or np.isposinf(lw).any() or math.isnan(lt) or lt == math.inf:
        raise ValueError("log weights must be finite or -inf (w = 0)")
    alpha = _check_alpha(alpha)
    ref = max(float(lw.max()), lt)
    if ref == -math.inf:
        raise ValueError("every weight is 0: the weighted quantile is undefined")
    w, wt = np.exp(lw - ref), math.exp(lt - ref)
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(w[order])
    total = cum[-1] + wt
    target = (1 - alpha) * total * (1 - 1e-12)
    j = int(np.searchsorted(cum, target, side="left"))
    if j >= s.size:
        return math.inf
    return float(s[order][j])


def tilted_log_weights(peak_hat: Sequence[float] | np.ndarray, lam: float) -> np.ndarray:
    """log w for P1, p(x) ∝ p₀(x)·exp(−λ·T̂_peak(x)) (CLAUDE.md §11.2): dP/dp₀ ∝ exp(−λ T̂), so
    log w = −λ·T̂_peak up to a constant that cancels in the quantile. λ in 1/K, T̂ in K, both
    from the model the proposer used."""
    t = np.asarray(peak_hat, dtype=np.float64)
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError(f"lam must be finite and >= 0 (1/K), got {lam}")
    if not np.isfinite(t).all():
        raise ValueError("peak_hat must be finite")
    return -lam * t


class ClassifierDensityRatio:
    """Estimated w(x) = dP_prop/dp₀ for proposers without a known density (CLAUDE.md §11.1 G3
    row; M3.2 prompt): logistic regression of "proposer" (1) vs "calibration" (0) on design
    vectors, then by Bayes' rule

        w(x) = [P(1|x) / P(0|x)] · (n₀ / n₁),

    clipped to RATIO_CLIP = [0.05, 20]. C is sklearn's inverse regularisation strength (no
    default: D4). A non-converged fit raises."""

    def __init__(self, cal_x: np.ndarray, prop_x: np.ndarray, *, C: float) -> None:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        a = np.asarray(cal_x, dtype=np.float64)
        b = np.asarray(prop_x, dtype=np.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or len(a) < 2 or len(b) < 2:
            raise ValueError(f"cal_x {a.shape} and prop_x {b.shape} must be 2-D with the same "
                             "width and >= 2 rows each")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("design vectors must be finite")
        if not (math.isfinite(C) and C > 0):
            raise ValueError(f"C must be finite and > 0, got {C}")
        X = np.vstack([a, b])
        y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        clf = LogisticRegression(C=C, max_iter=10_000)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                clf.fit(X, y)
            except ConvergenceWarning as e:
                raise RuntimeError(f"density-ratio classifier did not converge: {e}") from e
        self._clf, self.dim = clf, a.shape[1]
        self._log_prior = math.log(len(a) / len(b))

    def log_weight(self, x: np.ndarray) -> np.ndarray:
        """log w(x) for design vectors x [k, d] (or one vector [d]), after clipping."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] != self.dim or not np.isfinite(x).all():
            raise ValueError(f"x must be finite with {self.dim} columns, got {x.shape}")
        logit = self._clf.decision_function(x)  # log P(1|x)/P(0|x)
        lo, hi = RATIO_CLIP
        return np.clip(logit + self._log_prior, math.log(lo), math.log(hi))


class WeightedSplitConformal(BaseGate):
    """G3: weighted split conformal (Tibshirani et al. 2019, via Angelopoulos & Bates Sec. 4.5,
    Theorem 3; docs/refs/angelopoulos_bates_cp.txt). q(x) = weighted_conformal_quantile with the
    calibration log-weights fixed at construction and the test log-weight from test_log_w(inp)
    (the runner maps a design to its weight: tilted_log_weights for P1, ClassifierDensityRatio
    otherwise). Theorem 3 is for one test point with known weights; an adaptive proposal
    sequence is not covered (CLAUDE.md §18.2)."""

    gate_id = "G3"

    def __init__(self, *, alpha: float, tau: float, cal_scores: Sequence[float] | np.ndarray,
                 cal_log_w: Sequence[float] | np.ndarray,
                 test_log_w: Callable[[GateInput], float]) -> None:
        self.alpha, self.tau = _check_alpha(alpha), _check_tau(tau)
        self._s = _check_scores(cal_scores, "cal_scores")
        lw = np.asarray(cal_log_w, dtype=np.float64).reshape(-1)
        if lw.shape != self._s.shape:
            raise ValueError(f"cal_log_w has {lw.size} entries for {self._s.size} scores")
        self._lw = lw
        if not callable(test_log_w):
            raise TypeError("test_log_w must be callable: GateInput -> log w(x)")
        self._test_log_w = test_log_w

    def decide(self, inp: GateInput) -> GateDecision:
        q = weighted_conformal_quantile(self._s, self._lw, float(self._test_log_w(inp)),
                                        self.alpha)
        return _bound_decision(inp, q, self.tau)

    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]:
        return _no_labels(self.gate_id, records, "G3 is calibrated once")

    def state_dict(self) -> dict[str, Any]:
        return {"gate": self.gate_id, "alpha": self.alpha, "tau": self.tau,
                "cal_scores": self._s.tolist(), "cal_log_w": self._lw.tolist()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _check_state_gate(state, self.gate_id)
        if (state["alpha"], state["tau"]) != (self.alpha, self.tau) or \
                state["cal_scores"] != self._s.tolist() or state["cal_log_w"] != self._lw.tolist():
            raise ValueError("G3 state does not match this gate's calibration")


# ====================================================================== G4


class ACIFlaggedOnly(BaseGate):
    """G4: the update of G5 (Gibbs & Candès update (2), with D7 and D8) fed with the certified
    labels of FLAGGED designs only (CLAUDE.md §11.1). This is the documented selection-bias
    ablation: q learns from the population the gate sends to the solver, not the one it accepts
    (Tu & Giesecke Prop. B.14, CLAUDE.md §18.1). It deliberately bypasses B8's audited-only
    rule, so the constructor requires ablation_flagged_only=True by name.

    commit() records which designs were flagged; update() accepts a record only for a design
    this gate flagged, once. The record's `audited` field is ignored (a flagged design may or may
    not also have been audited). No audits: p_audit = 0."""

    gate_id = "G4"

    def __init__(self, *, alpha: float, gamma: float, W: int, tau: float,
                 cal_scores: Sequence[float] | np.ndarray, ablation_flagged_only: bool) -> None:
        if ablation_flagged_only is not True:
            raise ValueError("G4 updates q from flagged labels, which B8 forbids for every real "
                             "gate; construct it with ablation_flagged_only=True to say this is "
                             "the selection-bias ablation")
        if not (math.isfinite(alpha) and 0 < alpha <= 0.5):
            raise ValueError(f"alpha must lie in (0, 0.5], got {alpha}")
        if not (math.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be finite and > 0, got {gamma}")
        if not (isinstance(W, int) and not isinstance(W, bool) and W >= 1):
            raise ValueError(f"W must be an int >= 1, got {W!r}")
        cal = _check_scores(cal_scores, "cal_scores")
        self.alpha, self.gamma, self.W, self.tau = float(alpha), float(gamma), W, _check_tau(tau)
        self.alpha_t = self.alpha
        self._scores = [float(x) for x in cal]
        self.q_t = conformal_quantile(cal, self.alpha)
        self._committed: set[str] = set()
        self._flagged: set[str] = set()
        self._used: set[str] = set()

    def decide(self, inp: GateInput) -> GateDecision:
        return _bound_decision(inp, self.q_t, self.tau, alpha_t=self.alpha_t)

    def commit(self, inp: GateInput) -> GateDecision:
        if inp.design_id in self._committed:
            raise ValueError(f"design {inp.design_id!r} already committed to G4")
        d = self.decide(inp)
        self._committed.add(inp.design_id)
        if d.decision == "flag":
            self._flagged.add(inp.design_id)
        return d

    def _threshold(self, a: float) -> float:
        if a <= 0:
            return math.inf
        if a >= 1:
            return 0.0
        return conformal_quantile(self._scores[-self.W:], a)

    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]:
        seen: set[str] = set()
        for i, r in enumerate(records):
            if not isinstance(r, AuditRecord):
                raise TypeError(f"records[{i}] is a {type(r).__name__}, not an AuditRecord")
            if r.design_id in self._used or r.design_id in seen:
                raise ValueError(f"records[{i}] ({r.design_id!r}) was already used to update q")
            if r.design_id not in self._flagged:
                raise ValueError(f"records[{i}] ({r.design_id!r}) was not flagged by G4: the "
                                 "ablation learns from flagged labels only")
            seen.add(r.design_id)
        out = []
        for r in records:
            S, a, q = r.S_hi, self.alpha_t, self.q_t
            err = 1 if a >= 1 else 0 if a <= 0 else int(S > q)  # D7
            self.alpha_t = a + self.gamma * (self.alpha - err)
            self._scores = [*self._scores, S][-self.W:]
            self._used.add(r.design_id)
            self.q_t = self._threshold(self.alpha_t)
            out.append(AuditOutcome(design_id=r.design_id, T_lo=float(r.T_lo),
                                    T_hi=float(r.T_hi), S_hi=S, err=err, q_t=q, alpha_t=a))
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"gate": self.gate_id, "alpha": self.alpha, "gamma": self.gamma, "W": self.W,
                "tau": self.tau, "alpha_t": self.alpha_t, "q_t": self.q_t,
                "scores": list(self._scores), "committed": sorted(self._committed),
                "flagged": sorted(self._flagged), "used": sorted(self._used)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _check_state_gate(state, self.gate_id)
        for k in ("alpha", "gamma", "W", "tau"):
            if state[k] != getattr(self, k):
                raise ValueError(f"state has {k} = {state[k]} but this gate has "
                                 f"{getattr(self, k)} (D4)")
        self.alpha_t = float(state["alpha_t"])
        self._scores = [float(x) for x in _check_scores(state["scores"], "state scores")]
        self.q_t = float(state["q_t"])
        self._committed = {str(x) for x in state["committed"]}
        self._flagged = {str(x) for x in state["flagged"]}
        self._used = {str(x) for x in state["used"]}
