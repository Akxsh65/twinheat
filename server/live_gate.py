"""The audited ACI gate (G5) inside the live server (F7 live demo; CLAUDE.md §8.1, §8.3, B8).

    LiveGate.from_files(frozen_yaml, gate_inputs_dir, seed)

builds gate/aci_audit.AuditedACIGate exactly as experiments/run_gates.py does for G5: α, γ, W, τ,
p, λ, ε and m from a frozen config (experiments/configs/frozen*.yaml), and the starting scores
from the 1,000 labelled p₀ calibration designs (calibration.parquet + labels.parquet). Per
message:
  preview(...)   query: gate.decide() only. Pure: no coin, no state change (B8).
  commit(...)    design: coin.commit(id) -> gate.commit() -> coin.draw(id, p). Returns the
                 decision and whether the design is audited; the coin is drawn after commit and
                 never shown before it (B8).
  update(...)    an audited design's certified interval [T⁻, T⁺] for the true peak (Lemma 1′);
                 flagged-only designs never update q (Lemma 1, A3).
Design ids are made unique on the server ("<client id>#<n>"), so a reconnecting client that
reuses ids cannot commit a design twice to the coin.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gate.aci_audit import AuditedACIGate
from gate.base import AuditRecord, GateDecision, GateInput
from gate.coin import AuditCoin
from gate.scale import label_free_scale


class LiveGate:
    def __init__(self, fz: dict[str, Any], cal_scores: np.ndarray, seed: int) -> None:
        self.fz = fz
        self.coin = AuditCoin(seed=seed)
        self.gate = AuditedACIGate(alpha=float(fz["alpha"]), gamma=float(fz["gamma"]),
                                   W=int(fz["W"]), tau=float(fz["tau_K"]),
                                   p=float(fz["p_audit"]), cal_scores=cal_scores, coin=self.coin)
        self._n = 0

    @classmethod
    def from_files(cls, frozen: Path, gate_inputs: Path, seed: int) -> LiveGate:
        from experiments import run_gates as RG

        fz = RG.validate_frozen(RG.load_yaml(frozen))
        labels = RG.Labels.from_parquet(Path(gate_inputs) / "labels.parquet")
        cal = RG.Calibration.from_parquet(Path(gate_inputs) / "calibration.parquet", labels)
        scores, _ = RG.cal_scores(cal, fz)
        return cls(fz, scores, seed)

    def gate_input(self, msg_id: str, record: dict[str, Any], mu: np.ndarray, sd: np.ndarray,
                   coeffs: dict[str, np.ndarray]) -> GateInput:
        s, r_hot, sd_hot = label_free_scale(mu, sd, coeffs, coeffs["rhs"], m=int(self.fz["m_hot"]),
                                            lam=float(self.fz["lam"]), eps=float(self.fz["eps_K"]))
        x = record.get("design_vector")
        payload = (np.asarray(x, dtype="<f8").tobytes() if x is not None
                   else json.dumps(record, sort_keys=True).encode())
        return GateInput(design_id=msg_id, x_hash=hashlib.sha256(payload).hexdigest(),
                         peak_hat=float(np.max(mu)), s=float(s), r_hot=float(r_hot),
                         sd_hot=float(sd_hot), proposer="P4", aggressiveness=None, group=None)

    def unique_id(self, msg_id: str) -> str:
        self._n += 1
        return f"{msg_id}#{self._n}"

    def preview(self, gi: GateInput) -> GateDecision:
        return self.gate.decide(gi)  # pure (gate/base.py): a query never changes anything

    def commit(self, gi: GateInput) -> tuple[GateDecision, bool]:
        self.coin.commit(gi.design_id)
        dec = self.gate.commit(gi)
        return dec, self.coin.draw(gi.design_id, dec.p_audit)

    def update(self, gi: GateInput, T_lo: float, T_hi: float, iters: int) -> None:
        self.gate.update([AuditRecord(gi.design_id, True, gi.peak_hat, gi.s, T_lo, T_hi,
                                      int(iters))])

    def state(self) -> dict[str, Any]:
        q = self.gate.q_t
        return {"q_t": None if math.isinf(q) else q, "alpha_t": self.gate.alpha_t,
                "audits": self.gate.n_audits, "tau_K": float(self.fz["tau_K"]),
                "p": float(self.fz["p_audit"])}
