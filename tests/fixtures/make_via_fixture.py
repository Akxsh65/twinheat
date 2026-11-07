"""Write viewer/test/fixtures/via_board.json: a real board and its solved field for the via tests.

    python -m tests.fixtures.make_via_fixture

A sampler board with at least two top-side chips, edited so that ONE dense via array sits under
chip 0 and nothing under the others, with light background vias everywhere; then solved by
solvers/fvm.py at N = 32. The JS test checks that the columns under chip 0 carry the most heat.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.raster import rasterise
from data.sampler import sample_board
from solvers.fvm import assemble_board, solve_direct
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "viewer" / "test" / "fixtures" / "via_board.json"
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 32
ARRAY_DENSITY = 0.8  # vias/mm², inside the §7.4 range U[0.2, 1]
BACKGROUND = 0.02  # vias/mm², inside the §7.4 range 0–0.05


def record() -> dict:
    for seed in range(200):
        rec = sample_board(seed, "id")
        if len(rec["chips"]) >= 2 and all(c["side"] == "top" for c in rec["chips"]):
            break
    else:
        raise RuntimeError("no sampler board with two top chips in seeds 0..199")
    c0 = rec["chips"][0]
    rec["id"] = f"via-fixture-{rec['id']}"
    rec["via_arrays"] = [{k: c0[k] for k in ("x0", "y0", "w", "h")} | {"density_per_mm2":
                                                                         ARRAY_DENSITY}]
    rec["background_via_density_per_mm2"] = BACKGROUND
    return rec


def solve(rec: dict, n: int) -> np.ndarray:
    A, b = assemble_board(SPEC, rasterise(rec, SPEC, n), rec["dirichlet_edges"])
    return solve_direct(A, b).reshape(4, n, n)


def main() -> None:
    rec = record()
    theta = solve(rec, N)
    OUT.write_text(json.dumps({"record": rec, "N": N, "theta": [round(float(v), 7)
                                                                 for v in theta.ravel()]}) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"wrote {OUT}: {rec['id']}, {len(rec['chips'])} chips, peak {theta.max():.2f} K")


if __name__ == "__main__":
    main()
