"""server/app.py v0 (G2.3; CLAUDE.md §10.3, B6, B8) and server/model_stub.py.

The websocket tests drive the FastAPI app with Starlette's TestClient. Inference is injected
(create_app(predict=...)) so the queue tests can hold the worker thread still while the client
floods the socket.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from contracts.f16 import decode
from contracts.validate import ContractError, validate
from data.raster import rasterise
from server import model_stub
from server.app import Inbox, create_app
from solvers.fvm import assemble_board, solve_direct
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
RECORD = json.loads((ROOT / "contracts/examples/board_record/valid_1.json").read_text("utf-8"))
N = 16  # small grid: the tests check plumbing, not accuracy


def _msg(kind: str, i: int, record: dict | None = None) -> dict:
    return {"schema_version": "1.0", "type": kind, "id": f"{kind}-{i}",
            "record": copy.deepcopy(record or RECORD)}


def _fake_predict(n: int = N):
    """A predict(spec, raster, edges, record) that returns a constant field, counting calls."""
    calls: list[str] = []

    def predict(spec, raster, edges, record):
        calls.append(record["id"])
        f = np.full((4, n, n), 1.0, dtype=np.float32)
        return f, 0.1 * f

    return predict, calls


# ----------------------------------------------------------------------------------- inbox


def test_inbox_keeps_only_the_newest_query_and_every_design_in_order() -> None:
    import asyncio

    async def run() -> list[str]:
        box = Inbox()
        for i in range(100):
            box.put(_msg("query", i))
            assert box.queued_queries() <= 1
        box.put(_msg("design", 0))
        box.put(_msg("query", 100))
        box.put(_msg("design", 1))
        assert box.queued_queries() == 1
        assert box.dropped == 100  # q0..q98 replaced by later queries, q99 by q100
        return [(await box.get())["id"] for _ in range(3)]

    # q99 was dropped when q100 arrived (at most one queued query, the newest); designs kept
    assert asyncio.run(run()) == ["design-0", "query-100", "design-1"]


# ------------------------------------------------------------------------------- websocket


def test_20_queries_then_a_design_reply_in_order_and_schema_valid() -> None:
    app = create_app(spec=SPEC, n=N, dev=True)
    replies = []
    with TestClient(app) as client, client.websocket_connect("/") as ws:
        for i in range(20):  # one at a time, so nothing is dropped
            ws.send_json(_msg("query", i))
            replies.append(ws.receive_json())
        ws.send_json(_msg("design", 0))
        replies.append(ws.receive_json())
    assert [r["id"] for r in replies] == [f"query-{i}" for i in range(20)] + ["design-0"]
    for r in replies:
        validate(r, "pred")  # B6
        assert r["B"] is None and r["group"] is None
        assert r["units"] == {"mu": "K", "sd": "K", "B": "K", "peak": "K"}
        assert r["shape"] == [4, N, N]
    assert [r["decision"] for r in replies] == ["preview"] * 20 + ["accept"]


def test_pred_carries_the_stub_field_and_its_peak() -> None:
    app = create_app(spec=SPEC, n=N, dev=True)
    with TestClient(app) as client, client.websocket_connect("/") as ws:
        ws.send_json(_msg("design", 7))
        r = ws.receive_json()
    mu = decode(r["mu"], r["shape"])
    sd = decode(r["sd"], r["shape"])
    raster = rasterise(RECORD, SPEC, N)
    want_mu, want_sd = model_stub.predict(SPEC, raster, RECORD["dirichlet_edges"], RECORD)
    np.testing.assert_allclose(mu, want_mu, rtol=1e-3, atol=1e-2)  # float16 on the wire
    np.testing.assert_allclose(sd, want_sd, rtol=1e-3, atol=1e-2)
    assert r["peak"] == pytest.approx(float(want_mu.max()), rel=1e-6)
    assert isinstance(r["t_server_ns"], int)


def test_flooding_100_queries_leaves_at_most_one_queued() -> None:
    entered, release = threading.Event(), threading.Event()
    predict, calls = _fake_predict()

    def slow(spec, raster, edges, record):
        entered.set()
        release.wait(timeout=30)
        return predict(spec, raster, edges, record)

    app = create_app(spec=SPEC, n=N, dev=True, predict=slow)
    with TestClient(app) as client, client.websocket_connect("/") as ws:
        ws.send_json(_msg("query", 0))
        assert entered.wait(timeout=20), "the worker never started query-0"
        for i in range(1, 100):
            ws.send_json(_msg("query", i))
        ws.send_json(_msg("design", 0))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and app.state.received < 101:
            time.sleep(0.01)
        assert app.state.received == 101
        (box,) = app.state.inboxes
        assert box.max_queued_queries <= 1
        assert box.queued_queries() <= 1
        release.set()
        replies = [ws.receive_json() for _ in range(3)]
    # q0 was already in the worker; q1..q98 were replaced; q99 and the design are answered
    assert [r["id"] for r in replies] == ["query-0", "query-99", "design-0"]
    assert len(calls) == 3  # three inferences: nothing dropped was ever run
    assert box.dropped == 98


def test_designs_are_never_dropped_under_a_flood() -> None:
    release = threading.Event()
    predict, _ = _fake_predict()

    def slow(*a):
        release.wait(timeout=30)
        return predict(*a)

    app = create_app(spec=SPEC, n=N, dev=True, predict=slow)
    sent = []
    with TestClient(app) as client, client.websocket_connect("/") as ws:
        for i in range(50):
            ws.send_json(_msg("query", i))
            if i % 10 == 9:
                ws.send_json(_msg("design", i))
                sent.append(f"design-{i}")
        while app.state.received < 55:
            time.sleep(0.01)
        release.set()
        got = []
        while len([g for g in got if g.startswith("design")]) < len(sent):
            got.append(ws.receive_json()["id"])
    assert [g for g in got if g.startswith("design")] == sent


def test_invalid_message_gets_an_error_reply_and_the_connection_survives() -> None:
    app = create_app(spec=SPEC, n=N, dev=True)
    bad = _msg("query", 0)
    bad["record"]["chips"][0]["P"] = -1.0
    with TestClient(app) as client, client.websocket_connect("/") as ws:
        ws.send_text("not json")
        e1 = ws.receive_json()
        ws.send_json({"schema_version": "1.0", "type": "verified", "id": "x"})
        e2 = ws.receive_json()
        ws.send_json(bad)
        e3 = ws.receive_json()
        ws.send_json(_msg("query", 1))
        ok = ws.receive_json()
    for e in (e1, e2, e3):
        validate(e, "error")
    assert e1["id"] is None and "JSON" in e1["message"]
    assert "verified" in e2["message"]
    assert e3["id"] == "query-0" and "$.record.chips[0].P" in e3["message"]
    assert ok["type"] == "pred" and ok["id"] == "query-1"


def test_dev_mode_raises_on_an_invalid_outgoing_message() -> None:
    def broken(spec, raster, edges, record):  # wrong layer count: pred.shape must be [4, n, n]
        f = np.ones((3, N, N), dtype=np.float32)
        return f, f

    app = create_app(spec=SPEC, n=N, dev=True, predict=broken)
    with TestClient(app) as client, client.websocket_connect("/") as ws:
        ws.send_json(_msg("query", 0))
        with pytest.raises((ContractError, ValueError, WebSocketDisconnect)):
            ws.receive_json()
    assert isinstance(app.state.last_error, (ContractError, ValueError))


def test_query_and_design_never_touch_gate_state_in_v0() -> None:
    # v0 has no gate: B is null, nothing is audited, and no coin exists to draw (B8)
    app = create_app(spec=SPEC, n=N, dev=True)
    assert not hasattr(app.state, "gate") and not hasattr(app.state, "coin")
    with TestClient(app) as client, client.websocket_connect("/") as ws:
        ws.send_json(_msg("query", 0))
        ws.send_json(_msg("design", 0))
        r = [ws.receive_json(), ws.receive_json()]
    assert all(x["decision"] in ("preview", "accept") and x["B"] is None for x in r)


def test_rejects_a_bad_grid() -> None:
    with pytest.raises(ValueError, match="n"):
        create_app(spec=SPEC, n=0)


# ------------------------------------------------------------------------------ model_stub


def test_stub_is_the_solver_field_plus_two_percent_smooth_noise() -> None:
    raster = rasterise(RECORD, SPEC, 32)
    A, b = assemble_board(SPEC, raster, RECORD["dirichlet_edges"])
    theta = solve_direct(A, b).reshape(4, 32, 32)
    mu, sd = model_stub.predict(SPEC, raster, RECORD["dirichlet_edges"], RECORD)
    assert mu.shape == sd.shape == (4, 32, 32) and mu.dtype == np.float32
    rel = (mu - theta) / theta
    assert np.abs(rel).max() <= 4 * model_stub.NOISE_REL  # 4 std of a unit-variance field
    assert 0.5 * model_stub.NOISE_REL < rel.std() < 2 * model_stub.NOISE_REL
    # smooth: neighbouring cells' relative errors are strongly correlated
    r0 = rel[0]
    assert np.corrcoef(r0[:, :-1].ravel(), r0[:, 1:].ravel())[0, 1] > 0.8
    np.testing.assert_allclose(sd, model_stub.NOISE_REL * np.abs(theta), rtol=1e-5)


def test_stub_noise_is_fixed_by_the_design_not_by_the_message_id() -> None:
    raster = rasterise(RECORD, SPEC, N)
    a = copy.deepcopy(RECORD)
    a["id"] = "other-id"
    mu1, _ = model_stub.predict(SPEC, raster, RECORD["dirichlet_edges"], RECORD)
    mu2, _ = model_stub.predict(SPEC, raster, RECORD["dirichlet_edges"], a)
    np.testing.assert_array_equal(mu1, mu2)
    moved = copy.deepcopy(RECORD)
    moved["chips"][0]["x0"] += 1.0
    mu3, _ = model_stub.predict(SPEC, rasterise(moved, SPEC, N), moved["dirichlet_edges"], moved)
    assert not np.array_equal(mu1, mu3)
