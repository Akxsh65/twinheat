"""Mock WebSocket server: replays a .jsonl of server messages (CLAUDE.md §10.3) at a fixed rate.

Lets the viewer run without the solver or the model. Every line is validated against
contracts/schemas when the file is loaded, and a bad line stops the server with its line number,
unless --allow-invalid is given (to exercise the viewer's rejection path). Each client that
connects gets the whole stream from the first line; with --loop it repeats until the client
leaves, without it the connection stays open and silent after the last message. Lines are sent
exactly as they appear in the file. Messages from clients are read and dropped.

    python -m server.ws_mock_server stream.jsonl --rate 10 --loop [--host 127.0.0.1] [--port 8765]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from websockets.asyncio.server import Server, ServerConnection
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed

from contracts.validate import ContractError, validate

__all__ = ["SERVER_KINDS", "load_messages", "replay", "serve", "main"]

SERVER_KINDS = ("pred", "iterate", "verified", "batch_progress", "error")
DEFAULT_PORT = 8765  # viewer/src/main.js connects here by default


def _check_line(line: str) -> None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON ({e})") from e
    if not isinstance(obj, dict):
        raise ValueError(f"message must be a JSON object, got {type(obj).__name__}")
    kind = obj.get("type")
    if kind not in SERVER_KINDS:
        raise ValueError(f"type {kind!r} is not a server message ({', '.join(SERVER_KINDS)})")
    validate(obj, kind)


def load_messages(path: Path | str, *, allow_invalid: bool = False) -> list[str]:
    """Read one message per line (blank lines skipped); raise ValueError naming the bad line."""
    path = Path(path)
    messages: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            _check_line(line)
        except (ValueError, ContractError) as e:
            if not allow_invalid:
                raise ValueError(f"{path}: line {lineno}: {e}") from e
            print(f"warning: {path}: line {lineno} is invalid, sending it anyway: {e}",
                  file=sys.stderr)
        messages.append(line)
    if not messages:
        raise ValueError(f"{path}: no messages")
    return messages


def _check_rate(rate: float) -> None:
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError(f"rate must be a positive finite number of messages/s, got {rate}")


async def replay(ws: ServerConnection, messages: Sequence[str], rate: float, loop: bool) -> None:
    """Send messages to one client at `rate` Hz on an absolute schedule (no drift)."""
    period = 1.0 / rate
    clock = asyncio.get_running_loop().time
    start = clock()
    i = 0
    while True:
        for m in messages:
            delay = start + i * period - clock()
            if delay > 0:
                await asyncio.sleep(delay)
            await ws.send(m)
            i += 1
        if not loop:
            return


async def _drain(ws: ServerConnection) -> None:
    async for _ in ws:  # client -> server messages are not part of the mock
        pass


async def serve(messages: Sequence[str], *, rate: float, loop: bool, host: str,
                port: int) -> Server:
    """Start the server (port 0 = any free port) and return it; it runs until closed."""
    _check_rate(rate)
    if not messages:
        raise ValueError("no messages to replay")
    messages = list(messages)

    async def handler(ws: ServerConnection) -> None:
        drain = asyncio.create_task(_drain(ws))
        try:
            await replay(ws, messages, rate, loop)
            await ws.wait_closed()
        except ConnectionClosed:
            pass  # the client left; nothing to clean up beyond the drain task
        finally:
            drain.cancel()

    # compression=None: deflating ~1.4 MB of base64 per 256² pred costs more than the 100 ms
    # budget at 10 Hz and gains little on base64; max_size=None: the mock only sends
    return await ws_serve(handler, host, port, max_size=None, compression=None)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", type=Path, help="one server message per line")
    ap.add_argument("--rate", type=float, default=10.0, help="messages per second (default 10)")
    ap.add_argument("--loop", action="store_true", help="repeat the stream for each client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--allow-invalid", action="store_true",
                    help="send lines that fail validation (to test the viewer's error path)")
    args = ap.parse_args(argv)
    _check_rate(args.rate)
    messages = load_messages(args.jsonl, allow_invalid=args.allow_invalid)

    async def run() -> None:
        server = await serve(messages, rate=args.rate, loop=args.loop, host=args.host,
                             port=args.port)
        print(f"replaying {len(messages)} messages from {args.jsonl} at {args.rate} Hz"
              f"{' (loop)' if args.loop else ''} on ws://{args.host}:{args.port}", flush=True)
        await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
