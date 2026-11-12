"""G7: audited ACI with risk-adaptive audits (CLAUDE.md §8.5 Extension B, §9.3, §11.1; H9).
Reported as an ablation, not a headline.

    p_t = min(1, max(p_min, π_t)),   π_t = p̄ · ((1 − κ) + κ · s_t / s̄_t)
    α_{k+1} = α_k + γ · (p̄ / p_t) · (α − err_k)          on audited steps only

π mixes uniform sampling (κ = 0: p_t = p̄, which is G5 exactly) with sampling proportional to the
label-free scale s_t (κ = 1). s_t already carries the ensemble spread (s = r_hot + λ·σ_hot + ε,
gate/scale.py), so π depends on (s, σ) through s; it does not use the group g (raising p for rare
groups is the §15 fallback, not implemented here). s̄_t is the running mean of s over the
calibration designs (u_cal) and every design committed before t: label-free, fixed before the
coin. Its role is to keep mean p_t near p̄ when the proposer's scales drift; the floor p_min and
the cap 1 make the mean only approximately p̄.

The weights p̄/p_t have conditional mean 1 on the audit indicator (E[A_t · p̄/p_t | past] = p̄)
and are bounded by p̄/p_min. This parametrisation of π is TwinHeat's, not taken from Zrnic &
Candès 2024 or Li, Zrnic & Candès 2025 (neither is in docs/refs). G7 has no cited validity
argument: neither Gibbs & Candès (fixed γ) nor Tu & Giesecke (no IPW update, constant π) covers
the variable step (CLAUDE.md §8.5, §18.1).

Everything else is G5 (gate/aci_audit.py): D7, D8, the audited-only and coin checks of update().
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from gate.aci_audit import AuditedACIGate
from gate.base import AuditOutcome, AuditRecord, GateDecision, GateInput
from gate.coin import AuditCoin

__all__ = ["ActiveAuditGate"]


class ActiveAuditGate(AuditedACIGate):
    gate_id = "G7"

    def __init__(self, *, alpha: float, gamma: float, W: int, tau: float, p_bar: float,
                 p_min: float, kappa: float, u_cal: Sequence[float] | np.ndarray,
                 cal_scores: Sequence[float] | np.ndarray, coin: AuditCoin) -> None:
        super().__init__(alpha=alpha, gamma=gamma, W=W, tau=tau, p=p_bar, cal_scores=cal_scores,
                         coin=coin)
        if not (math.isfinite(p_min) and 0 < p_min <= self.p):
            raise ValueError(f"p_min must lie in (0, p_bar = {self.p}], got {p_min}")
        if not (math.isfinite(kappa) and 0 <= kappa <= 1):
            raise ValueError(f"kappa must lie in [0, 1], got {kappa}")
        u = np.asarray(u_cal, dtype=np.float64).reshape(-1)
        if u.size == 0 or not np.isfinite(u).all() or u.min() <= 0:
            raise ValueError("u_cal (label-free scales s of the calibration designs, K) must be "
                             "non-empty, finite and > 0")
        self.p_bar, self.p_min, self.kappa = self.p, float(p_min), float(kappa)
        self._u_sum, self._u_n = float(u.sum()), int(u.size)
        self._p_of: dict[str, float] = {}

    def _p(self, s: float) -> float:
        pi = self.p_bar * ((1 - self.kappa) + self.kappa * s / (self._u_sum / self._u_n))
        return min(1.0, max(self.p_min, pi))

    def decide(self, inp: GateInput) -> GateDecision:
        d = super().decide(inp)
        return GateDecision(design_id=d.design_id, B=d.B, decision=d.decision,
                            p_audit=self._p(inp.s), group=d.group, q_t=d.q_t, alpha_t=d.alpha_t)

    def commit(self, inp: GateInput) -> GateDecision:
        if inp.design_id in self._p_of:
            raise ValueError(f"design {inp.design_id!r} already committed to G7")
        d = self.decide(inp)  # p_t from the designs committed before this one
        self._p_of[inp.design_id] = d.p_audit
        self._u_sum += inp.s
        self._u_n += 1
        return d

    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]:
        for i, r in enumerate(records):
            if isinstance(r, AuditRecord) and r.design_id not in self._p_of:
                raise ValueError(f"records[{i}] ({r.design_id!r}) was not committed to G7, so "
                                 "its audit probability p_t is unknown")
        scores = self._check(records)  # G5's checks: audited, coin came up 1, used once
        out = []
        for r, S in zip(records, scores, strict=True):
            a, q = self.alpha_t, self._q
            err = 1 if a >= 1 else 0 if a <= 0 else int(S > q)  # D7
            w = self.p_bar / self._p_of[r.design_id]
            self.alpha_t = a + self.gamma * w * (self.alpha - err)
            self._scores = [*self._scores, S][-self.W:]
            self._used.add(r.design_id)
            self.n_audits += 1
            self.n_err += err
            self._q = self._threshold(self.alpha_t)
            out.append(AuditOutcome(design_id=r.design_id, T_lo=float(r.T_lo),
                                    T_hi=float(r.T_hi), S_hi=S, err=err, q_t=q, alpha_t=a))
        return out

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "p_min": self.p_min, "kappa": self.kappa,
                "u_sum": self._u_sum, "u_n": self._u_n, "p_of": dict(sorted(self._p_of.items()))}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("gate") != self.gate_id:
            raise ValueError(f"state is for gate {state.get('gate')!r}, not {self.gate_id}")
        for k in ("p_min", "kappa"):
            if state[k] != getattr(self, k):
                raise ValueError(f"state has {k} = {state[k]} but this gate has "
                                 f"{getattr(self, k)} (D4)")
        super().load_state_dict(state)
        self._u_sum, self._u_n = float(state["u_sum"]), int(state["u_n"])
        self._p_of = {str(k): float(v) for k, v in state["p_of"].items()}
