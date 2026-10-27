"""server/ws_mock_server.py: replays a .jsonl of contract-valid server messages, in order."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest
from websockets.asyncio.client import connect

from contracts.f16 import decode
from contracts.validate import validate
from server import demo_stream
from server.ws_mock_server import SERVER_KINDS, load_messages, serve

EXAMPLES = Path(__file__).resolve().parents[1] / "contracts" / "examples"


def _example_lines() -> list[str]:
    """One valid example of every server kind, plus a second pred: 6 distinct messages."""
    names = [(k, "valid_1") for k in SERVER_KINDS] + [("pred", "valid_3")]
    return [json.dumps(json.loads((EXAMPLES / k / f"{n}.json").read_text("utf-8")))
            for k, n in names]


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "stream.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


async def _receive(messages: list[str], n: int, rate: float, loop: bool) -> list[str]:
    server = await serve(messages, rate=rate, loop=loop, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            return [await asyncio.wait_for(ws.recv(), timeout=5) for _ in range(n)]
    finally:
        server.close()
        await server.wait_closed()


def test_sends_all_n_messages_in_order(tmp_path: Path) -> None:
    lines = _example_lines()
    messages = load_messages(_write(tmp_path, lines))
    assert messages == lines
    got = asyncio.run(_receive(messages, len(lines), rate=500.0, loop=False))
    assert got == lines


def test_loop_repeats_the_stream(tmp_path: Path) -> None:
    lines = _example_lines()
    got = asyncio.run(_receive(load_messages(_write(tmp_path, lines)), 2 * len(lines), 500.0, True))
    assert got == lines + lines


def test_without_loop_nothing_follows_the_last_message(tmp_path: Path) -> None:
    lines = _example_lines()[:2]

    async def run() -> None:
        server = await serve(lines, rate=500.0, loop=False, host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                assert [await ws.recv(), await ws.recv()] == lines
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(ws.recv(), timeout=0.2)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


def test_every_client_gets_the_stream_from_the_start(tmp_path: Path) -> None:
    lines = _example_lines()

    async def run() -> tuple[list[str], list[str]]:
        server = await serve(lines, rate=500.0, loop=False, host="127.0.0.1", port=0)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        try:
            async with connect(url) as a, connect(url) as b:
                ra = [await a.recv() for _ in lines]
                rb = [await b.recv() for _ in lines]
                return ra, rb
        finally:
            server.close()
            await server.wait_closed()

    ra, rb = asyncio.run(run())
    assert ra == lines and rb == lines


def test_rate_sets_the_spacing(tmp_path: Path) -> None:
    lines = _example_lines()  # 6 messages at 20 Hz: last one no earlier than 5/20 s after the first
    t0 = time.monotonic()
    asyncio.run(_receive(lines, len(lines), rate=20.0, loop=False))
    assert time.monotonic() - t0 >= 5 / 20 - 0.02


@pytest.mark.parametrize("rate", [0.0, -1.0, float("inf"), float("nan")])
def test_rate_must_be_positive_and_finite(rate: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        asyncio.run(serve(_example_lines(), rate=rate, loop=False, host="127.0.0.1", port=0))


def test_invalid_line_raises_with_its_line_number(tmp_path: Path) -> None:
    lines = _example_lines()
    bad = json.loads((EXAMPLES / "pred" / "invalid_1.json").read_text("utf-8"))
    p = _write(tmp_path, [lines[0], json.dumps(bad), lines[1]])
    with pytest.raises(ValueError, match=r"line 2: .*pred: \$\.decision"):
        load_messages(p)


def test_allow_invalid_keeps_the_line_for_client_error_paths(tmp_path: Path) -> None:
    lines = _example_lines()
    bad = (EXAMPLES / "pred" / "invalid_1.json").read_text("utf-8").replace("\n", "")
    assert load_messages(_write(tmp_path, [lines[0], bad]), allow_invalid=True) == [lines[0], bad]


@pytest.mark.parametrize(
    "line, match",
    [
        ("{not json", "line 1: not JSON"),
        ("[1, 2]", "line 1: .*JSON object"),
        ('{"schema_version": "1.0", "type": "query", "id": "x"}', "line 1: .*not a server message"),
    ],
)
def test_malformed_lines_raise(tmp_path: Path, line: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_messages(_write(tmp_path, [line]))


def test_blank_lines_are_skipped_and_empty_files_refused(tmp_path: Path) -> None:
    lines = _example_lines()[:2]
    assert load_messages(_write(tmp_path, ["", lines[0], "  ", lines[1]])) == lines
    with pytest.raises(ValueError, match="no messages"):
        load_messages(_write(tmp_path, ["", " "]))


def test_demo_stream_is_contract_valid_and_changes_every_layer() -> None:
    msgs = [json.loads(m) for m in demo_stream.messages(frames=3, n=16, seed=0)]
    assert [m["type"] for m in msgs] == ["pred", "pred", "pred"]
    for m in msgs:
        validate(m, "pred")
        assert m["shape"] == [4, 16, 16]
        assert m["id"].startswith("demo-")
    a, b = (decode(m["mu"], m["shape"]) for m in msgs[:2])
    assert all(np.any(a[layer] != b[layer]) for layer in range(4))
