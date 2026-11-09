"""G5: audited adaptive conformal gate with uniform, certified audits (CLAUDE.md §8.1–§8.3, §9.3).

Per committed design t (label-free):
    B_t = q_t · s_t (K);  accept if B_t ≤ τ, else flag;  audit probability p (constant: A2).
Per audit k (in audit order; only designs whose coin came up 1):
    S⁺_k = max(|T̂ − T⁻|, |T̂ − T⁺|) / s                      certified score (Lemma 1′)
    err_k = 1[S⁺_k > q_k];  forced to 1 when α_k ≥ 1 (D7), 0 when α_k ≤ 0 (q = ∞)
    α_{k+1} = α_k + γ (α − err_k)                               (Gibbs & Candès update (2))
    q_{k+1} = ∞ if α_{k+1} ≤ 0;  0 if α_{k+1} ≥ 1;  else the split-conformal (1 − α_{k+1})
              quantile of the last W scores (D8)
At the start q₁ is the split-conformal (1 − α) quantile of all calibration scores (§9.3).

Lemma 1's assumptions are enforced here, not by convention (table in the G-M3.1 report):
  A1  coin after commit and never shown to a proposer: gate/coin.py raises on draw-before-commit;
      decide() returns no coin and cannot see one.
  A2  A_t ~ Bernoulli(p), p constant and independent of the design: p is fixed at construction
      and decide() returns it unchanged; the coin's RNG never sees the design.
  A3  q moves only on audited steps: update() raises ValueError on any record with
      audited == False or whose coin did not come up 1, and on any design seen before; the batch
      is checked in full before anything changes. decide() never changes state.
Deterministic consequences, tested as hard invariants (§18.1): α_k ∈ [−γ, 1 + γ] for all k, and
|mean(err) − α| ≤ (max(α₁, 1 − α₁) + γ)/(γ n_a) after every audit, for any score sequence.
Lemma 1 is stated for α ≤ ½, so the constructor refuses α > ½.

D7 (2026-11-07): when α_k ≥ 1 the prediction set is empty, so err = 1 whatever S⁺ is; q is
reported as 0 there (B = 0), as in §8.2. D8 (2026-11-07): every quantile, at calibration and in
the window, uses the ⌈(n+1)(1 − α)⌉/n level with "higher" interpolation, ∞ when that level is
above 1.
Hyperparameters α, γ, W, τ, p have no defaults: D4 freezes them in experiments/configs/frozen.yaml.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from gate.base import AuditOutcome, AuditRecord, BaseGate, GateDecision, GateInput
from gate.coin import AuditCoin, CoinError

__all__ = ["AuditedACIGate", "conformal_quantile"]

# ⌈(n+1)(1−α)⌉ is taken after subtracting this, so 20 · 0.95 = 19.000000000000004 gives 19, not 20.
_CEIL_TOL = 1e-9
_HYPER = ("alpha", "gamma", "W", "tau", "p")


def _check_scores(scores: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(scores, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(a).all() or a.min() < 0:
        raise ValueError(f"{name} must be finite and >= 0 (scores are |error| / s)")
    return a


def conformal_quantile(scores: Sequence[float] | np.ndarray, alpha: float) -> float:
    """Split-conformal quantile (D8): the ⌈(n+1)(1−α)⌉-th smallest of n scores, ∞ if that rank
    exceeds n. 0 < α < 1; the ACI boundaries α ≤ 0 and α ≥ 1 are handled by the caller."""
    a = _check_scores(scores, "scores")
    if not (math.isfinite(alpha) and 0 < alpha < 1):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    n = a.size
    k = math.ceil((n + 1) * (1 - alpha) - _CEIL_TOL)
    if k > n:
        return math.inf
    return float(np.partition(a, k - 1)[k - 1])


class AuditedACIGate(BaseGate):
    gate_id = "G5"

    def __init__(self, *, alpha: float, gamma: float, W: int, tau: float, p: float,
                 cal_scores: Sequence[float] | np.ndarray, coin: AuditCoin) -> None:
        if not (math.isfinite(alpha) and 0 < alpha <= 0.5):
            raise ValueError(f"alpha must lie in (0, 0.5] (Lemma 1 needs alpha <= 1/2), "
                             f"got {alpha}")
        if not (math.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be finite and > 0, got {gamma}")
        if not (isinstance(W, int) and not isinstance(W, bool) and W >= 1):
            raise ValueError(f"W must be an int >= 1, got {W!r}")
        if not (math.isfinite(tau) and tau > 0):
            raise ValueError(f"tau must be finite and > 0 (K), got {tau}")
        if not (math.isfinite(p) and 0 < p <= 1):
            raise ValueError(f"p must lie in (0, 1] (uniform audit probability), got {p}")
        if not isinstance(coin, AuditCoin):
            raise TypeError(f"coin must be an AuditCoin, got {type(coin).__name__}")
        cal = _check_scores(cal_scores, "cal_scores")
        self.alpha, self.gamma, self.W, self.tau, self.p = float(alpha), float(gamma), W, \
            float(tau), float(p)
        self.coin = coin
        self.alpha_t = self.alpha
        self._scores: list[float] = [float(x) for x in cal]
        self._q = conformal_quantile(cal, self.alpha)  # all calibration scores at the start
        self._used: set[str] = set()
        self.n_audits = 0
        self.n_err = 0

    # ------------------------------------------------------------------ threshold

    @property
    def q_t(self) -> float:
        return self._q

    def _threshold(self, alpha_t: float) -> float:
        if alpha_t <= 0:
            return math.inf
        if alpha_t >= 1:
            return 0.0
        return conformal_quantile(self._scores[-self.W:], alpha_t)

    # ------------------------------------------------------------------ decide (pure)

    def decide(self, inp: GateInput) -> GateDecision:
        if not isinstance(inp, GateInput):
            raise TypeError(f"decide() takes a GateInput, got {type(inp).__name__}")
        q = self._q
        B = math.inf if math.isinf(q) else q * inp.s
        return GateDecision(design_id=inp.design_id, B=B,
                            decision="accept" if B <= self.tau else "flag",
                            p_audit=self.p, group=inp.group, q_t=q, alpha_t=self.alpha_t)

    # ------------------------------------------------------------------ update (audits only)

    def _check(self, records: Sequence[AuditRecord]) -> list[float]:
        seen: set[str] = set()
        scores = []
        for i, r in enumerate(records):
            if not isinstance(r, AuditRecord):
                raise TypeError(f"records[{i}] is a {type(r).__name__}, not an AuditRecord")
            if r.audited is not True:
                raise ValueError(f"records[{i}] ({r.design_id!r}) is not audited: only audited "
                                 "designs update q; flagged labels never do (Lemma 1 A3, B8)")
            if r.design_id in self._used or r.design_id in seen:
                raise ValueError(f"records[{i}] ({r.design_id!r}) was already used to update q")
            try:
                drawn = self.coin.audited(r.design_id)
            except CoinError as e:
                raise ValueError(f"records[{i}] ({r.design_id!r}): no audit coin was drawn for "
                                 f"this design ({e})") from e
            if not drawn:
                raise ValueError(f"records[{i}] ({r.design_id!r}) claims audited but its coin "
                                 "came up 0 (a flagged label cannot update q, A3)")
            if not (isinstance(r.iters_used, int) and r.iters_used >= 0):
                raise ValueError(f"records[{i}].iters_used must be an int >= 0")
            seen.add(r.design_id)
            scores.append(r.S_hi)  # validates the certified interval
        return scores

    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]:
        scores = self._check(records)  # everything is checked before anything changes
        out = []
        for r, S in zip(records, scores, strict=True):
            a, q = self.alpha_t, self._q
            if a >= 1:
                err = 1  # D7: empty prediction set
            elif a <= 0:
                err = 0  # q = ∞ covers every finite score
            else:
                err = int(S > q)
            self.alpha_t = a + self.gamma * (self.alpha - err)
            self._scores.append(S)
            self._scores = self._scores[-self.W:]
            self._used.add(r.design_id)
            self.n_audits += 1
            self.n_err += err
            self._q = self._threshold(self.alpha_t)
            out.append(AuditOutcome(design_id=r.design_id, T_lo=float(r.T_lo), T_hi=float(r.T_hi),
                                    S_hi=S, err=err, q_t=q, alpha_t=a))
        return out

    # ------------------------------------------------------------------ state

    def state_dict(self) -> dict[str, Any]:
        return {"gate": self.gate_id, **{k: getattr(self, k) for k in _HYPER},
                "alpha_t": self.alpha_t, "q_t": self._q, "scores": list(self._scores),
                "used": sorted(self._used), "n_audits": self.n_audits, "n_err": self.n_err}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("gate") != self.gate_id:
            raise ValueError(f"state is for gate {state.get('gate')!r}, not {self.gate_id}")
        for k in _HYPER:
            if state[k] != getattr(self, k):
                raise ValueError(f"state has {k} = {state[k]} but this gate has "
                                 f"{k} = {getattr(self, k)}; hyperparameters are fixed (D4)")
        scores = [float(x) for x in _check_scores(state["scores"], "state scores")]
        self.alpha_t = float(state["alpha_t"])
        self._scores = scores
        self._used = {str(u) for u in state["used"]}
        self.n_audits, self.n_err = int(state["n_audits"]), int(state["n_err"])
        # recompute rather than trust the stored q, so an edited alpha_t is honoured
        self._q = (float(state["q_t"]) if self.n_audits == 0 and self.alpha_t == self.alpha
                   else self._threshold(self.alpha_t))
