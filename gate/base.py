"""Gate interface (CLAUDE.md §8.1, §10.4).

A gate sees only label-free quantities when it decides, and only certified audit labels when it
updates:
  decide(GateInput) -> GateDecision    pure: never changes gate state (a `query` preview calls it
                                       freely, B8). The decision is accept or flag; it carries the
                                       audit probability p_t but never the coin.
  commit(GateInput) -> GateDecision    the binding decision, once per committed design; may
                                       change label-free state only (default: decide()).
  update(list[AuditRecord]) -> list[AuditOutcome]
                                       audited designs only; the gate checks each claim against
                                       its AuditCoin and refuses the whole batch on any bad record.
  state_dict() / load_state_dict()     JSON-safe state for checkpoints and replay.
  log_row(...)                         one §10.4 audit-log row, validated against
                                       contracts/schemas/audit_log_row.json.

Scores are S = |T̂_peak − T_peak| / s. An audit returns a certified interval [T⁻, T⁺] ∋ T_peak
(Lemma 2(c), certify/peak_interval.py) and the gate uses S⁺ = max(|T̂ − T⁻|, |T̂ − T⁺|) / s ≥ S
(Lemma 1′, §8.3): |T̂ − T| is convex in T, so its maximum over the interval is at an end.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any, Literal

from contracts.validate import validate

__all__ = ["AuditOutcome", "AuditRecord", "BaseGate", "GateDecision", "GateInput",
           "certified_score"]

PROPOSERS = ("P1", "P2", "P3", "P4")


def _finite(name: str, v: float, *, positive: bool = False, nonneg: bool = False) -> float:
    v = float(v)
    if not math.isfinite(v) or (positive and v <= 0) or (nonneg and v < 0):
        cond = " > 0" if positive else " >= 0" if nonneg else ""
        raise ValueError(f"{name} must be finite{cond}, got {v}")
    return v


def certified_score(peak_hat: float, T_lo: float, T_hi: float, s: float) -> float:
    """S⁺ = max(|T̂ − T⁻|, |T̂ − T⁺|) / s (Lemma 1′); T̂, T⁻, T⁺ in K, s > 0 in K."""
    peak_hat = _finite("peak_hat", peak_hat)
    T_lo, T_hi = _finite("T_lo", T_lo), _finite("T_hi", T_hi)
    s = _finite("s", s, positive=True)
    if T_lo > T_hi:
        raise ValueError(f"certified interval is empty: T_lo {T_lo} > T_hi {T_hi}")
    return max(abs(peak_hat - T_lo), abs(peak_hat - T_hi)) / s


@dataclass(frozen=True)
class GateInput:
    """Label-free description of one committed or previewed design (§8.1 step 2)."""

    design_id: str
    x_hash: str  # SHA-256 of the design, lowercase hex
    peak_hat: float  # T̂_peak = max μ over all layers, K
    s: float  # label-free scale, K (gate/scale.py)
    r_hot: float  # K
    sd_hot: float  # K
    proposer: str  # P1..P4
    aggressiveness: int | None  # 1..3, None for P4
    group: str | None  # label-free group (G6); None for single-threshold gates
    # mean |A μ − b| over the hottest cells in W, without the diag A scaling; only the
    # paper-faithful G1 variant reads it (D13, gate/scale.hot_residual_raw)
    r_hot_raw: float | None = None

    def __post_init__(self) -> None:
        if not (isinstance(self.design_id, str) and self.design_id):
            raise ValueError(f"design_id must be a non-empty str, got {self.design_id!r}")
        _finite("peak_hat", self.peak_hat)
        _finite("s", self.s, positive=True)
        _finite("r_hot", self.r_hot, nonneg=True)
        _finite("sd_hot", self.sd_hot, nonneg=True)
        if self.r_hot_raw is not None:
            _finite("r_hot_raw", self.r_hot_raw, nonneg=True)
        if self.proposer not in PROPOSERS:
            raise ValueError(f"proposer must be one of {PROPOSERS}, got {self.proposer!r}")


@dataclass(frozen=True)
class GateDecision:
    """What the gate says before the coin. B = q_t · s (K, inf when q_t = inf)."""

    design_id: str
    B: float
    decision: Literal["accept", "flag"]
    p_audit: float
    group: str | None
    q_t: float  # threshold used, dimensionless (inf allowed)
    alpha_t: float | None  # ACI level at decision time (None for gates without one)


@dataclass(frozen=True)
class AuditRecord:
    """A certified audit label for a committed design whose coin came up 1."""

    design_id: str
    audited: bool
    peak_hat: float  # K, the prediction that was gated
    s: float  # K, the scale it was gated with
    T_lo: float  # K, certified lower bound on the true peak
    T_hi: float  # K, certified upper bound
    iters_used: int

    @property
    def S_hi(self) -> float:
        return certified_score(self.peak_hat, self.T_lo, self.T_hi, self.s)


@dataclass(frozen=True)
class AuditOutcome:
    """What one audit did to the gate: err_k = 1[S⁺ > q] with the q in force at that audit."""

    design_id: str
    T_lo: float  # K
    T_hi: float  # K
    S_hi: float
    err: int
    q_t: float  # threshold the audit was scored against
    alpha_t: float  # level before this audit's update


class BaseGate(abc.ABC):
    gate_id: str  # "G0".."G7" (§11.1)

    @abc.abstractmethod
    def decide(self, inp: GateInput) -> GateDecision: ...

    def commit(self, inp: GateInput) -> GateDecision:
        """The binding decision for a committed design; the server calls it once per design,
        after coin.commit(design_id) and before drawing the coin. Gates with label-free
        per-design state override it (G1 pushes r_hot into its window, G4 records what it
        flagged, G6 draws MVP's randomisation, G7 fixes p_t). It never sees a label or a coin
        value. Default: the same decision as decide(), with no state change."""
        return self.decide(inp)

    @abc.abstractmethod
    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]: ...

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def log_row(
        self, inp: GateInput, dec: GateDecision, *, run_id: str, seed: int, t: int,
        outcome: AuditOutcome | None, iters_used: int, oracle_peak: float | None = None,
    ) -> dict[str, Any]:
        """One §10.4 row. `decision` is the gate's accept/flag; `audited` says whether the coin
        came up 1 (an audited row has an outcome). q_t and B = inf are written as null."""
        if dec.design_id != inp.design_id or (outcome and outcome.design_id != inp.design_id):
            raise ValueError(f"log_row: input {inp.design_id!r}, decision {dec.design_id!r} and "
                             f"outcome {outcome and outcome.design_id!r} disagree")
        audited = outcome is not None
        row = {
            "schema_version": "1.0", "run_id": run_id, "gate": self.gate_id, "seed": seed, "t": t,
            "proposer": inp.proposer, "aggressiveness": inp.aggressiveness, "group": dec.group,
            "x_hash": inp.x_hash, "s": inp.s, "r_hot": inp.r_hot, "sd_hot": inp.sd_hot,
            "q_t": dec.q_t if math.isfinite(dec.q_t) else None,
            "B": dec.B if math.isfinite(dec.B) else None,
            "decision": dec.decision, "p_t": dec.p_audit, "audited": audited,
            "T_lo": outcome.T_lo if audited else None, "T_hi": outcome.T_hi if audited else None,
            "S_hi": outcome.S_hi if audited else None, "err": outcome.err if audited else None,
            "alpha_t": dec.alpha_t, "iters_used": iters_used, "oracle_peak": oracle_peak,
        }
        validate(row, "audit_log_row")
        return row
