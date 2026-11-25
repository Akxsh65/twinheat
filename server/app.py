"""TwinHeat server v0 (G2.3): FastAPI websocket, query/design -> pred (CLAUDE.md §10.3).

    python -m server.app [--n 32] [--host 127.0.0.1] [--port 8765] [--no-dev]

The viewer connects to ws://<host>:<port>/ (viewer/src/main.js default port 8765, or ?ws=).

Per message (client -> server, one JSON object per websocket text frame):
  1. parse and validate against contracts/schemas (query or design; B6). A bad message gets an
     `error` reply carrying its id (null if it had none) and the connection stays open.
  2. queue it in the client's Inbox: designs are always kept, in order; a new query replaces the
     one already queued (at most one queued query per client, the newest), so a fast drag never
     builds a backlog and latency stays flat.
  3. a per-client consumer takes the next message and runs inference in a worker thread, one
     inference at a time across all clients (one model, one GPU later): rasterise
     (data/raster.py, mm -> SI at N = n) -> predict (server/model_stub.py until the FNO
     ensemble) -> pred.
  4. reply pred: decision "preview" for a query, "accept" for a design (no gate in v0), B null,
     group null, peak = max of mu over all layers and cells (K), t_server_ns = monotonic clock.

No gate exists in v0, so neither message can touch gate state or draw an audit coin (B8); v1 adds
the gate bridge, and query must stay prediction-only there too.

--verify (F7, anytime verified rendering): a `design` is answered with its pred (decision "flag":
every committed design is verified) and then, from the same worker thread, the certified PCG
stream of server/iterate_stream.py: `iterate` messages carrying the guaranteed band, then
`verified`. A `query` never starts a verification (B8). While a design is verifying, the client's
next message waits in its Inbox.

dev mode (default on): every outgoing message is validated against its schema and a failure
raises ContractError. That error is a server bug, not a client one: it is logged with its
traceback, kept in app.state.last_error, and the connection is closed with code 1011.
A record the rasteriser refuses (for example off-grid geometry at this n) gets an error reply.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import anyio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from contracts.f16 import encode
from contracts.validate import ContractError, validate
from data.raster import rasterise
from server import model_stub
from server.iterate_stream import verification_messages
from twinheat import spec as specmod

__all__ = ["DEFAULT_N", "DEFAULT_PORT", "Inbox", "create_app", "main"]

log = logging.getLogger("twinheat.server")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8765  # viewer/src/main.js connects here by default
# Grid of the stub field. The direct solve behind model_stub costs ~0.1 s at n = 32 and ~0.6 s at
# n = 64 on a laptop core (informal, G2.3), so 32 keeps 10 drags/s answered one by one; larger n
# still works, with intermediate queries dropped. The FNO replaces this with n = 256 (§7.3).
DEFAULT_N = 32
CLIENT_KINDS = ("query", "design")
DECISION = {"query": "preview", "design": "accept"}  # v0: no gate yet

Predict = Callable[[Any, Any, Sequence[str], dict[str, Any]], tuple[np.ndarray, np.ndarray]]


class Inbox:
    """A client's queue: every design kept in arrival order, at most one (the newest) query."""

    def __init__(self) -> None:
        self._items: deque[dict[str, Any]] = deque()
        self._ready = asyncio.Event()
        self.dropped = 0  # queries replaced by a newer one before they were started
        self.max_queued_queries = 0

    def queued_queries(self) -> int:
        return sum(1 for m in self._items if m["type"] == "query")

    def put(self, msg: dict[str, Any]) -> None:
        if msg["type"] == "query":
            before = len(self._items)
            self._items = deque(m for m in self._items if m["type"] != "query")
            self.dropped += before - len(self._items)
        self._items.append(msg)
        self.max_queued_queries = max(self.max_queued_queries, self.queued_queries())
        self._ready.set()

    async def get(self) -> dict[str, Any]:
        while not self._items:
            self._ready.clear()
            await self._ready.wait()
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)


def _error(msg_id: str | None, message: str) -> dict[str, Any]:
    return {"schema_version": "1.0", "type": "error", "id": msg_id, "message": message,
            "t_server_ns": time.monotonic_ns()}


def create_app(
    *, spec: Any = None, n: int = DEFAULT_N, dev: bool = True, predict: Predict | None = None,
    verify: dict[str, Any] | None = None,
) -> FastAPI:
    """The v0 app. spec: SpecV2 (default config/spec_v2.yaml); n: grid for rasterise/predict;
    predict(spec, raster, dirichlet_edges, record) -> (mu, sd) float [4, n, n] in K.
    verify: None, or {"tau_v", "every", "kappa_iters"} to stream the verification of designs."""
    if not (isinstance(n, int) and not isinstance(n, bool) and n >= 1):
        raise ValueError(f"n must be a positive int, got {n!r}")
    spec = spec if spec is not None else specmod.load(ROOT / "config" / "spec_v2.yaml")
    if not isinstance(spec, specmod.SpecV2):
        raise TypeError(f"spec must be a SpecV2 (config/spec_v2.yaml), got {type(spec).__name__}")
    predict = predict or model_stub.predict
    if verify is not None and set(verify) != {"tau_v", "every", "kappa_iters"}:
        raise ValueError(f"verify needs tau_v, every, kappa_iters; got {sorted(verify)}")
    decision = dict(DECISION, design="flag") if verify is not None else DECISION
    limiter = anyio.CapacityLimiter(1)  # one inference at a time across clients (one model)

    app = FastAPI(title="TwinHeat server v0")
    app.state.inboxes = []
    app.state.received = 0
    app.state.last_error = None
    app.state.n = n

    def outgoing(obj: dict[str, Any]) -> str:
        if dev:
            validate(obj, obj["type"])  # B6: raises ContractError naming the field
        return json.dumps(obj)

    def infer(msg: dict[str, Any], send_from_thread: Callable | None = None) -> dict[str, Any]:
        """Runs in the inference thread. Returns the reply (pred, or error for a refused record).
        With verify and a design, sends the pred and the verification stream itself through
        send_from_thread and returns None."""
        record = msg["record"]
        try:
            raster = rasterise(record, spec, n)
        except ValueError as e:  # includes ContractError; the record was already validated
            return _error(msg["id"], f"cannot rasterise record {record['id']!r} at n = {n}: {e}")
        mu, sd = predict(spec, raster, record["dirichlet_edges"], record)
        mu, sd = np.asarray(mu), np.asarray(sd)
        if mu.shape != sd.shape:
            raise ValueError(f"predict returned mu {mu.shape} and sd {sd.shape}")
        pred = {
            "schema_version": "1.0", "type": "pred", "id": msg["id"],
            "mu": encode(mu), "sd": encode(sd), "B": None, "decision": decision[msg["type"]],
            "group": None, "peak": float(mu.max()),
            "units": {"mu": "K", "sd": "K", "B": "K", "peak": "K"},
            "shape": list(mu.shape), "t_server_ns": time.monotonic_ns(),
        }
        if verify is None or msg["type"] != "design" or send_from_thread is None:
            return pred  # query: prediction only, never a verification (B8)
        send_from_thread(pred)
        for m in verification_messages(record, mu, spec, n, msg_id=msg["id"], **verify):
            send_from_thread(m)
        return None

    @app.websocket("/")
    async def endpoint(ws: WebSocket) -> None:
        await ws.accept()
        box = Inbox()
        app.state.inboxes.append(box)
        send_lock = anyio.Lock()

        async def send(obj: dict[str, Any]) -> None:
            text = outgoing(obj)
            async with send_lock:
                await ws.send_text(text)

        async def receive() -> None:
            while True:
                text = await ws.receive_text()
                app.state.received += 1
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError as e:
                    await send(_error(None, f"message is not valid JSON: {e}"))
                    continue
                msg_id = msg.get("id") if isinstance(msg, dict) else None
                msg_id = msg_id if isinstance(msg_id, str) and msg_id else None
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind not in CLIENT_KINDS:
                    await send(_error(msg_id, f"type {kind!r} is not a client message; "
                                              f"expected one of {CLIENT_KINDS}"))
                    continue
                try:
                    validate(msg, kind)
                except ContractError as e:
                    await send(_error(msg_id, str(e)))
                    continue
                box.put(msg)

        async def consume() -> None:
            while True:
                msg = await box.get()

                def send_from_thread(obj: dict[str, Any]) -> None:
                    anyio.from_thread.run(send, obj)

                reply = await anyio.to_thread.run_sync(infer, msg, send_from_thread,
                                                       limiter=limiter, abandon_on_cancel=True)
                if reply is not None:
                    await send(reply)

        try:
            async with anyio.create_task_group() as tg:

                async def receive_then_stop() -> None:
                    try:
                        await receive()
                    except WebSocketDisconnect:
                        pass
                    tg.cancel_scope.cancel()  # client gone: stop the consumer too

                tg.start_soon(receive_then_stop)
                tg.start_soon(consume)
        except* (ContractError, ValueError, TypeError, FloatingPointError, RuntimeError) as eg:
            exc = eg.exceptions[0]
            app.state.last_error = exc
            log.error("server error, closing the connection", exc_info=exc)
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await ws.close(code=1011, reason=type(exc).__name__[:120])
        finally:
            app.state.inboxes.remove(box)

    return app


def main(argv: Sequence[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="grid of the stub field")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-dev", action="store_true", help="skip outgoing schema validation")
    ap.add_argument("--verify", action="store_true",
                    help="stream the certified verification of every design (F7)")
    ap.add_argument("--tau-v", type=float, default=None, help="K; required with --verify")
    ap.add_argument("--every", type=int, default=None, help="PCG iterations between iterates")
    ap.add_argument("--kappa-iters", type=int, default=None, help="AMG-CG iterations for z")
    ap.add_argument("--model", choices=("stub", "ensemble"), default="stub",
                    help="stub: server/model_stub.py (a direct solve, slow at n = 256); "
                         "ensemble: the FNO ensemble of models/registry.json (stage ft256)")
    ap.add_argument("--device", default="cuda:0", help="torch device for --model ensemble")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    verify = None
    if args.verify:
        if None in (args.tau_v, args.every, args.kappa_iters):
            ap.error("--verify needs --tau-v, --every and --kappa-iters")
        verify = {"tau_v": args.tau_v, "every": args.every, "kappa_iters": args.kappa_iters}
    predict = None
    if args.model == "ensemble":
        from models import inputs as inp
        from models.ensemble import Ensemble
        ens = Ensemble.load(ROOT / "models" / "registry.json", ROOT / "models" / "norm_stats.json",
                            stage="ft256", device=args.device)

        def predict(spec, raster, edges, record):  # noqa: ARG001  (the raster holds everything)
            mu, sd = ens.predict(inp.build(raster, spec)[None])
            return mu[0], sd[0]
    app = create_app(n=args.n, dev=not args.no_dev, verify=verify, predict=predict)
    uvicorn.run(app, host=args.host, port=args.port, ws="websockets", log_level="info")


if __name__ == "__main__":
    main()
