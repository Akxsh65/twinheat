"""E11: anytime-verification latency of the live server (CLAUDE.md §11.5 E11, §13 targets; F7).

    python -m server.app --n 256 --verify --tau-v 0.1 --every 10 --kappa-iters 60   # elsewhere
    python -m bench.e11_latency --url ws://127.0.0.1:8765 --designs 10 --label "n256 stub cpu"

Sends --designs committed designs, one at a time: --pattern random draws each from p₀ (seed
--seed); --pattern drag starts from one p₀ design and moves its first chip (and the via array
under it) by +1 mm in x per design, like a user dragging a chip (the case the z warm start of
server/iterate_stream.py speeds up). For each it records
for each, from the server's monotonic t_server_ns stamps: the time from the pred (the gate's
answer) to the first iterate and to the verified field, the iterations and the final band; plus
the client round trip to the pred. §13 targets: first iterate < 100 ms after the gate fires,
certified field < 2 s; the summary says met or missed for each and never adjusts them.
Writes bench/results/e11_latency.csv and .json (commit, label, hardware, date).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import statistics as st
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIRST_ITERATE_MS, CERTIFIED_MS = 100.0, 2000.0  # CLAUDE.md §13


def _drag(rec: dict, dx: float) -> dict:
    import copy
    r = copy.deepcopy(rec)
    c = r["chips"][0]
    for a in r["via_arrays"]:
        if (a["x0"], a["y0"], a["w"], a["h"]) == (c["x0"], c["y0"], c["w"], c["h"]):
            a["x0"] = round(a["x0"] + dx, 3)
    c["x0"] = round(c["x0"] + dx, 3)
    return r


def run(url: str, designs: int, seed: int, pattern: str = "random") -> list[dict]:
    from websockets.sync.client import connect

    from data.sampler import design_board
    from gate.proposers import sample_p0

    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    with connect(url, max_size=None) as ws:
        base = design_board(sample_p0(rng))
        for i in range(designs):
            rec = design_board(sample_p0(rng)) if pattern == "random" else _drag(base, float(i))
            mid = f"e11-{i}"
            t0 = time.perf_counter()
            ws.send(json.dumps({"schema_version": "1.0", "type": "design", "id": mid,
                                "record": rec}))
            pred = first = None
            n_it = 0
            while True:
                m = json.loads(ws.recv())
                if m.get("id") != mid:
                    continue
                if m["type"] == "error":
                    raise RuntimeError(f"server error for {mid}: {m['message']}")
                if m["type"] == "pred":
                    pred, t_pred = m, time.perf_counter()
                elif m["type"] == "iterate":
                    n_it += 1
                    first = first or m
                elif m["type"] == "verified":
                    if pred is None or first is None:
                        raise RuntimeError(f"{mid}: verified before pred / iterate (server not "
                                           "started with --verify?)")
                    rows.append({
                        "design": mid, "pattern": pattern, "decision": pred["decision"],
                        "client_pred_ms": 1000 * (t_pred - t0),
                        "first_iterate_ms": (first["t_server_ns"] - pred["t_server_ns"]) / 1e6,
                        "verified_ms": (m["t_server_ns"] - pred["t_server_ns"]) / 1e6,
                        "iterates": n_it, "iters": m["iters"], "cert_K": m["cert"],
                        "stop_reason": m["stop_reason"]})
                    break
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--designs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pattern", choices=("random", "drag"), default="random")
    ap.add_argument("--label", required=True, help="server settings, e.g. 'n256 stub cpu'")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "e11_latency.csv")
    a = ap.parse_args(argv)
    rows = run(a.url, a.designs, a.seed, a.pattern)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fi = st.median(r["first_iterate_ms"] for r in rows)
    ve = st.median(r["verified_ms"] for r in rows)
    commit = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    summary = {
        "label": a.label, "pattern": a.pattern, "designs": len(rows),
        "median_first_iterate_ms": fi,
        "median_verified_ms": ve,
        "first_iterate_target":
            f"< {FIRST_ITERATE_MS} ms: {'met' if fi < FIRST_ITERATE_MS else 'missed'}",
        "certified_target": f"< {CERTIFIED_MS} ms: {'met' if ve < CERTIFIED_MS else 'missed'}",
        "commit": commit, "client": f"{platform.node()} {platform.processor()}",
        "date": dt.date.today().isoformat()}
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
