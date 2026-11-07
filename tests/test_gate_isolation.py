"""Invariant B8, gate isolation (CLAUDE.md §5, §8.1, §8.2 A1–A3): q_t changes only on audited
steps; the audit coin is drawn only after commit(design) and never reaches a proposer; decide()
(what a `query` preview calls) never touches gate state."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from gate.aci_audit import AuditedACIGate
from gate.base import AuditRecord, GateDecision, GateInput
from gate.coin import AuditCoin, CoinError

CAL = np.random.default_rng(0).exponential(size=1000)


def _gate(coin: AuditCoin, p: float = 0.5) -> AuditedACIGate:
    return AuditedACIGate(alpha=0.1, gamma=0.01, W=500, tau=1.0, p=p, cal_scores=CAL, coin=coin)


def _input(i: int, s: float = 0.2, peak: float = 40.0) -> GateInput:
    return GateInput(design_id=f"d{i}", x_hash=hashlib.sha256(f"d{i}".encode()).hexdigest(),
                     peak_hat=peak, s=s, r_hot=s / 2, sd_hot=s / 2, proposer="P1",
                     aggressiveness=1, group=None)


def _record(i: int, audited: bool, peak: float = 40.0, s: float = 0.2) -> AuditRecord:
    return AuditRecord(design_id=f"d{i}", audited=audited, peak_hat=peak, s=s,
                       T_lo=peak - 0.5, T_hi=peak + 0.5, iters_used=10)


def _audited_and_not(coin: AuditCoin, gate: AuditedACIGate) -> tuple[list[int], list[int]]:
    yes, no = [], []
    for i in range(200):
        gate.decide(_input(i))
        coin.commit(f"d{i}")
        (yes if coin.draw(f"d{i}", gate.p) else no).append(i)
    assert yes and no
    return yes, no


# --------------------------------------------------------------------- test 1: only audits move q


def test_a_flagged_non_audited_record_raises_and_q_is_unchanged() -> None:
    coin = AuditCoin(seed=1)
    gate = _gate(coin)
    _, not_audited = _audited_and_not(coin, gate)
    before = gate.state_dict()
    i = not_audited[0]
    with pytest.raises(ValueError, match="audited"):
        gate.update([_record(i, audited=False)])
    assert gate.state_dict() == before
    assert gate.q_t == before["q_t"]


def test_a_record_claiming_audited_when_the_coin_said_no_raises() -> None:
    coin = AuditCoin(seed=2)
    gate = _gate(coin)
    _, not_audited = _audited_and_not(coin, gate)
    before = gate.state_dict()
    with pytest.raises(ValueError, match="coin"):
        gate.update([_record(not_audited[0], audited=True)])  # a flagged label dressed as an audit
    assert gate.state_dict() == before


def test_a_batch_with_one_bad_record_changes_nothing() -> None:
    coin = AuditCoin(seed=3)
    gate = _gate(coin)
    yes, no = _audited_and_not(coin, gate)
    before = gate.state_dict()
    with pytest.raises(ValueError):
        gate.update([_record(yes[0], True), _record(no[0], False), _record(yes[1], True)])
    assert gate.state_dict() == before  # validated in full before anything is applied
    gate.update([_record(yes[0], True)])
    assert gate.state_dict() != before


def test_a_design_updates_q_at_most_once_and_only_after_its_coin() -> None:
    coin = AuditCoin(seed=4)
    gate = _gate(coin)
    yes, _ = _audited_and_not(coin, gate)
    gate.update([_record(yes[0], True)])
    with pytest.raises(ValueError, match="already"):
        gate.update([_record(yes[0], True)])
    with pytest.raises(ValueError, match="coin"):
        gate.update([_record(10_000, True)])  # never committed, never drawn


def test_decide_never_touches_gate_state() -> None:
    coin = AuditCoin(seed=5)
    gate = _gate(coin)
    before = gate.state_dict()
    for i in range(100):  # a drag sends many queries: previews call decide() only
        gate.decide(_input(i, s=0.01 + i / 10))
    assert gate.state_dict() == before
    assert coin.state_dict() == AuditCoin(seed=5).state_dict()  # and draws no coin


def test_the_decision_a_proposer_sees_carries_no_coin() -> None:
    gate = _gate(AuditCoin(seed=6))
    dec = gate.decide(_input(0))
    fields = {f.name for f in dataclasses.fields(GateDecision)}
    assert dec.decision in ("accept", "flag")
    assert not fields & {"audited", "audit", "coin", "A_t"}


# --------------------------------------------------------------------- test 2: coin after commit


def test_draw_before_commit_raises() -> None:
    coin = AuditCoin(seed=7)
    with pytest.raises(CoinError, match="commit"):
        coin.draw("d0", 0.1)
    assert coin.state_dict() == AuditCoin(seed=7).state_dict()  # the failed draw consumed nothing


def test_commit_is_once_and_the_coin_cannot_be_redrawn() -> None:
    coin = AuditCoin(seed=8)
    coin.commit("d0")
    with pytest.raises(CoinError, match="already committed"):
        coin.commit("d0")
    first = coin.draw("d0", 0.5)
    with pytest.raises(CoinError, match="already drawn"):
        coin.draw("d0", 0.5)
    assert coin.audited("d0") is first
    with pytest.raises(CoinError, match="not been drawn"):
        coin.audited("d1")


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_draw_rejects_a_bad_probability(p: float) -> None:
    coin = AuditCoin(seed=9)
    coin.commit("d0")
    with pytest.raises(ValueError, match="p"):
        coin.draw("d0", p)


def test_coins_are_bernoulli_p_and_reproducible_from_the_seed() -> None:
    def run(seed: int) -> list[bool]:
        c = AuditCoin(seed=seed)
        out = []
        for i in range(20_000):
            c.commit(f"d{i}")
            out.append(c.draw(f"d{i}", 0.1))
        return out

    a = run(11)
    assert a == run(11)
    assert a != run(12)
    rate = np.mean(a)
    assert abs(rate - 0.1) < 4 * np.sqrt(0.1 * 0.9 / len(a))


def test_coin_state_round_trips() -> None:
    c = AuditCoin(seed=13)
    for i in range(5):
        c.commit(f"d{i}")
        c.draw(f"d{i}", 0.3)
    c.commit("pending")
    d = AuditCoin(seed=0)
    d.load_state_dict(c.state_dict())
    with pytest.raises(CoinError, match="already committed"):
        d.commit("d0")
    assert [d.audited(f"d{i}") for i in range(5)] == [c.audited(f"d{i}") for i in range(5)]
    d.commit("next"), c.commit("next")
    assert d.draw("pending", 0.3) == c.draw("pending", 0.3)
