"""The audit coin A_t ~ Bernoulli(p_t) (CLAUDE.md §8.1 step 4, Lemma 1 A1–A2, invariant B8).

The server owns one AuditCoin per run. The protocol is enforced here, not by convention:
  commit(design_id)    the design is fixed. Once per id: a committed design cannot be re-proposed
                       under the same id to get a second draw.
  draw(design_id, p)   raises CoinError unless commit(design_id) came first, and raises if this id
                       was already drawn (no re-rolls). Returns A_t to the caller (the server),
                       which passes it to the verifier and the gate's update(); it never goes into
                       a GateDecision, a `pred` or anything else a proposer reads.
  audited(design_id)   the drawn value, so a gate can check that a record claiming "audited"
                       really was (gate/aci_audit.py does).

Draws come from one numpy Generator(PCG64(seed)) in commit/draw order, so the coin is
independent of the design itself (A2: MCAR) and a run is reproducible from its seed. The seed
stays on the server.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["AuditCoin", "CoinError"]


class CoinError(RuntimeError):
    """The commit-then-draw protocol was broken."""


class AuditCoin:
    def __init__(self, seed: int) -> None:
        if not (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0):
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        self._seed = seed
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._committed: set[str] = set()
        self._drawn: dict[str, bool] = {}

    def commit(self, design_id: str) -> None:
        if not (isinstance(design_id, str) and design_id):
            raise ValueError(f"design_id must be a non-empty str, got {design_id!r}")
        if design_id in self._committed:
            raise CoinError(f"design {design_id!r} already committed: one commit per design")
        self._committed.add(design_id)

    def draw(self, design_id: str, p: float) -> bool:
        if design_id not in self._committed:
            raise CoinError(f"draw({design_id!r}) before commit({design_id!r}): the coin is drawn "
                            "only after the design is committed (Lemma 1, A1)")
        if design_id in self._drawn:
            raise CoinError(f"coin for {design_id!r} already drawn: no re-draws")
        p = float(p)
        if not (math.isfinite(p) and 0.0 <= p <= 1.0):
            raise ValueError(f"audit probability p must lie in [0, 1], got {p} for {design_id!r}")
        a = bool(self._rng.random() < p)
        self._drawn[design_id] = a
        return a

    def committed(self, design_id: str) -> bool:
        """Whether commit(design_id) has happened (G6 draws its own randomisation only after
        it, D9)."""
        return design_id in self._committed

    def audited(self, design_id: str) -> bool:
        if design_id not in self._drawn:
            raise CoinError(f"the coin for {design_id!r} has not been drawn")
        return self._drawn[design_id]

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self._seed, "rng": self._rng.bit_generator.state,
                "committed": sorted(self._committed), "drawn": dict(sorted(self._drawn.items()))}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        drawn = {str(k): bool(v) for k, v in state["drawn"].items()}
        committed = {str(c) for c in state["committed"]}
        if not set(drawn) <= committed:
            raise ValueError("coin state has draws for designs that were never committed")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = state["rng"]
        self._seed, self._rng = int(state["seed"]), rng
        self._committed, self._drawn = committed, drawn
