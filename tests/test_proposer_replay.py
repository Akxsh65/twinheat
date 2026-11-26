"""server/proposer_replay.py: the chain of a recorded P1 / P2 run is recovered exactly (and a
tampered run is refused), P3 rounds are grouped by start, frames are spread evenly, and every
message written is contract-valid (B6) with a board record for each frame."""

from __future__ import annotations

import math

import numpy as np
import pytest

from contracts.validate import validate
from gate import proposers as P
from server import proposer_replay as R


def _peak(x: np.ndarray) -> float:  # a stand-in surrogate: smooth in the design vector
    return 40.0 + 10.0 * float(x[0]) + 5.0 * float(x[1] ** 2)


def _record_run(prop) -> list[dict]:
    """What run_trajectories.run_one stores per proposal (the columns the replay reads)."""
    hist, rows, rnd = P.History(), [], 0
    while not prop.done:
        xs = prop.next(hist)
        if not xs:
            break
        for x in xs:
            t = len(hist)
            row = {"t": t, "design_id": f"t{t}", "design_vector": list(map(float, x)),
                   "x_hash": P.x_hash(x), "peak_hat": _peak(x), "r_hot": 0.1, "sd_hot": 0.2,
                   "r_hot_raw": 0.3, "ood_score": 1.0, "group": "g", "round": rnd}
            hist.append(P.HistoryEntry(
                t=t, design_id=row["design_id"], x=tuple(row["design_vector"]),
                x_hash=row["x_hash"], peak_hat=row["peak_hat"], r_hot=0.1, sd_hot=0.2,
                r_hot_raw=0.3, ood_score=1.0, group="g", decision=None, B=None))
            rows.append(row)
        rnd += 1
    return rows


def _p1(seed=11):
    return P.TiltedMCMC(lam=3.0, step=0.05, n_proposals=12, seed=seed, level=2)


def _p2(seed=11):
    return P.SimulatedAnnealing(T0=1.0, step=0.05, steps=12, T_end_ratio=0.01, seed=seed, level=2)


@pytest.mark.parametrize("make", [_p1, _p2])
def test_walker_chain_recovers_the_recorded_decisions(make):
    rows = _record_run(make())
    chain = R.walker_chain(rows, make())
    assert len(chain) == len(rows)
    assert chain[0]["kind"] == "start" and chain[0]["prob"] is None
    current = rows[0]["peak_hat"]
    for row, c in zip(rows[1:], chain[1:], strict=True):
        assert c["kind"] in ("accepted", "rejected")
        if row["peak_hat"] <= current:
            assert c["kind"] == "accepted" and c["prob"] == 1.0
        if c["kind"] == "accepted":
            current = row["peak_hat"]
        assert c["current"] == current
    assert chain[-1]["best"] == min(r["peak_hat"] for r in rows)
    if make is _p2:
        Ts = [c["T"] for c in chain]
        assert all(a >= b for a, b in zip(Ts, Ts[1:], strict=False)), "P2 cools"
        assert math.isclose(Ts[0], 1.0)


def test_walker_chain_refuses_a_run_it_cannot_reproduce():
    rows = _record_run(_p1())
    with pytest.raises(R.ReplayError, match="different design"):
        R.walker_chain(rows, _p1(seed=12))
    rows[5]["design_vector"][3] += 1e-9
    with pytest.raises(R.ReplayError, match="t = 5"):
        R.walker_chain(rows, _p1())


def test_p3_chain_shows_the_start_with_the_best_final_peak():
    starts, rounds = 3, 4
    final = {0: 50.0, 1: 42.0, 2: 47.0}
    rows = [{"t": r * starts + i, "round": r, "peak_hat": final[i] + (rounds - 1 - r)}
            for r in range(rounds) for i in range(starts)]
    shown, chain = R.p3_chain(rows, starts)
    assert shown == 1
    kinds = [c["kind"] for c in chain]
    assert kinds[1] == "start" and kinds[4] == "step" and kinds[0] == "probe"
    assert chain[-1]["current"] == 42.0 and chain[-1]["best"] == 42.0
    with pytest.raises(R.ReplayError, match="whole rounds"):
        R.p3_chain(rows[:-1], starts)


def test_pick_frames_is_even_and_keeps_the_required_rows():
    f = R.pick_frames(range(0, 100), 5, [37])
    assert f == [0, 25, 37, 50, 74, 99] or f == [0, 25, 37, 50, 75, 99]
    assert R.pick_frames([2, 5, 8], 10, []) == [2, 5, 8]
    with pytest.raises(ValueError):
        R.pick_frames(range(5), 1, [])


def test_build_replay_writes_valid_messages_and_a_record_per_frame():
    rows = _record_run(_p2())
    chain = R.walker_chain(rows, _p2())
    frames = R.pick_frames(range(len(rows)), 4, [0])
    n = 8

    # design_board's record id is "design-" + the first 16 hex of the x_hash
    peak_of = {"design-" + row["x_hash"][:16]: row["peak_hat"] for row in rows}

    def predict(records):
        out = []
        for r in records:
            mu = np.full((4, n, n), peak_of[r["id"]] - 1.0)
            mu[0, 3, 3] = peak_of[r["id"]]
            out.append((mu, np.full((4, n, n), 0.1)))
        return out

    oracle = {rows[k]["x_hash"]: rows[k]["peak_hat"] + 0.5 for k in range(len(rows))}
    msgs, meta = R.build_replay(rows, chain, frames, name="P2", level=2, seed=0, predict=predict,
                                oracle=oracle, verify=None, source={"file": "x"}, settings={},
                                x_of=lambda k: k, x_label="proposal")
    for m in msgs:
        validate(m, m["type"])
        assert m["decision"] == "preview" and m["id"] in meta["records"]
    best_t = int(np.argmin([r["peak_hat"] for r in rows]))
    assert meta["best"]["t"] == best_t and meta["best"]["id"].endswith(f"-t{best_t}")
    assert len(meta["trace"]["points"]) == len(rows)
    assert meta["peak_check_max_abs_K"] < 1e-3  # float16-free: mu.max() is the peak itself
    assert all(f["oracle_peak"] == f["peak_hat"] + 0.5 for f in meta["frames"])
