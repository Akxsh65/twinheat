"""Varied demo boards for the live design panel (viewer ?records=; F7 live demo).

    python -m server.demo_records [--out viewer/public/demo/records.json] [--max-seed 3000]

The design-task boards all share one stack, chip set, convection and boundary (decision D2), so
they look alike. This picks boards from the dataset's ID sampler (data/sampler.sample_board, the
distribution the FNO was trained on), one per feature, scanning seeds in order:
  one-chip, four-chips, heat-sunk-edges (two edges), forced-air (a forced-convection zone),
  traces (a copper trace on L1), discs (a copper disc on L1), thick-copper (2 oz outer layers),
and keeps from an existing output file the boards whose id starts with --keep (default: the
first design of the recorded stream, server/iterate_stream.py record mode; the design-task
boards are near-identical, so one is enough). Each board then
gets the dataset's power envelope (data/envelope.apply_envelope: peak rise <= THETA_MAX), using
a direct solve at --n (64 by default, a fast estimate of the 256² peak; display data, not a
result). Ids are "demo-<feature>-<seed>". The output directory is ignored by git.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FEATURES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "one-chip": lambda r: len(r["chips"]) == 1,
    "four-chips": lambda r: len(r["chips"]) == 4,
    "heat-sunk-edges": lambda r: len(r["dirichlet_edges"]) == 2,
    "forced-air": lambda r: r["convection"]["forced_zone"] is not None,
    "traces": lambda r: any(s["type"] == "trace" for s in r["layers"][0]["shapes"]),
    "discs": lambda r: any(s["type"] == "disc" for s in r["layers"][0]["shapes"]),
    "thick-copper": lambda r: r["stack"]["outer_oz"] == 2,
}


def pick(max_seed: int) -> dict[str, dict[str, Any]]:
    from data.sampler import sample_board

    out: dict[str, dict[str, Any]] = {}
    used: set[int] = set()
    for name, has in FEATURES.items():
        for seed in range(max_seed):
            if seed in used:
                continue
            rec = sample_board(seed, "id")
            if has(rec):
                out[f"demo-{name}-{seed}"] = rec
                used.add(seed)
                break
        else:
            raise RuntimeError(f"no ID board with feature {name!r} in seeds 0..{max_seed - 1}")
    return out


def envelope(records: dict[str, dict[str, Any]], n: int) -> dict[str, dict[str, Any]]:
    from data.envelope import THETA_MAX, apply_envelope
    from data.raster import rasterise
    from solvers.fvm import assemble_board, solve_direct
    from twinheat import spec as specmod

    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    out = {}
    for rid, rec in records.items():
        A, b = assemble_board(spec, rasterise(rec, spec, n), rec["dirichlet_edges"])
        theta = solve_direct(A, b)
        rec2, theta2, scaled = apply_envelope(rec, theta, THETA_MAX)
        chips = ", ".join(f"{c['P']:.2g} W" for c in rec2["chips"])
        print(f"{rid:28s} chips [{chips}]  edges {rec2['dirichlet_edges'] or '-'}  "
              f"forced {'yes' if rec2['convection']['forced_zone'] else 'no'}  "
              f"peak ~{float(theta2.max()):.1f} K at n={n}{'  (power scaled)' if scaled else ''}")
        out[rid] = rec2
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "viewer" / "public" / "demo" / "records.json")
    ap.add_argument("--max-seed", type=int, default=3000)
    ap.add_argument("--n", type=int, default=64, help="grid of the envelope solve")
    ap.add_argument("--keep", default="demo-ensemble-0-",
                    help="id prefix of boards to keep from an existing --out ('' keeps none)")
    a = ap.parse_args(argv)
    recs = envelope(pick(a.max_seed), a.n)
    if a.out.exists() and a.keep:
        old = json.loads(a.out.read_text(encoding="utf-8"))
        kept = {k: v for k, v in old.items() if k.startswith(a.keep)}
        recs.update(kept)
        print(f"kept {sorted(kept)} from {a.out}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(recs), encoding="utf-8")
    print(f"wrote {len(recs)} boards to {a.out}")


if __name__ == "__main__":
    main()
