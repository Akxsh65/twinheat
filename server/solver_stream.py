"""Write verified messages carrying REAL 4-layer solver fields, for the viewer (G1.3 DONE WHEN).

    python -m server.solver_stream --out server/demo/solver.jsonl [--n 256] [--seeds 0 1 2]
    python -m server.ws_mock_server server/demo/solver.jsonl --rate 0.2 --loop

Each board is sampler board `seed` (mode "id"), rasterised at n x n, assembled by
solvers/fvm.assemble_board and solved by SciPy's direct solver (about 100 s per board at 256²
on one core). θ is sent as float16 (contracts §10), so it is not the float64 solution to the
last bit; `cert` is null because no certificate is computed for the rounded field, and
stop_reason says it was a direct solve. Ids are "solver-id-<seed>".

--orientation adds one SYNTHETIC field (id "synthetic-orientation-hot-SE", stop_reason says so):
a gradient with a hot cell at x = n-1, y = 0 on L1. In the viewer it must appear at the
bottom-right corner of the top layer from the default camera (CLAUDE.md §0 rule 7).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from contracts.f16 import encode
from contracts.validate import validate
from data.raster import rasterise
from data.sampler import sample_board
from solvers.fvm import assemble_board, solve_direct
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N_FULL = SPEC.geometry.N.value
# synthetic orientation field (display test only, not physics): gradient + one hot cell, K
_ORIENT_BASE_K, _ORIENT_SLOPE_K, _ORIENT_HOT_K = 10.0, 20.0, 80.0


def _verified(mid: str, theta: np.ndarray, ms: float, stop_reason: str) -> str:
    n = theta.shape[1]
    msg = {"schema_version": "1.0", "type": "verified", "id": mid, "theta": encode(theta),
           "cert": None, "iters": 0, "ms": ms, "stop_reason": stop_reason, "shape": [4, n, n]}
    validate(msg, "verified")
    return json.dumps(msg)


def orientation_field(n: int) -> np.ndarray:
    """[4, n, n]: a west-to-east gradient on every layer and, on L1, a hot square of n/16 cells
    (at least 1) whose corner cell is (x = n-1, y = 0), the south-east corner."""
    x = np.arange(n) / max(n - 1, 1)
    f = np.broadcast_to(_ORIENT_BASE_K + _ORIENT_SLOPE_K * x, (4, n, n)).copy()
    k = max(1, n // 16)
    f[0, :k, n - k:] = _ORIENT_HOT_K  # [layer, y, x]: rows y < k, columns x >= n - k
    return f


def messages(seeds: Sequence[int] = (0, 1, 2), n: int = N_FULL, orientation: bool = False
             ) -> Iterator[str]:
    for seed in seeds:
        rec = sample_board(seed, "id")
        A, b = assemble_board(SPEC, rasterise(rec, SPEC, n), rec["dirichlet_edges"])
        t0 = time.perf_counter()
        theta = solve_direct(A, b).reshape(4, n, n)
        ms = 1e3 * (time.perf_counter() - t0)
        yield _verified(f"solver-id-{seed}", theta, ms,
                        f"direct solve (SciPy spsolve), board {rec['id']}, edges "
                        f"{''.join(rec['dirichlet_edges']) or 'none'}, float16 for transport")
    if orientation:
        yield _verified("synthetic-orientation-hot-SE", orientation_field(n), 0.0,
                        "SYNTHETIC orientation test field, not a solve: hot cell at x = N-1, "
                        "y = 0 on L1 must show bottom-right")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=N_FULL)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-orientation", action="store_true")
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for line in messages(args.seeds, args.n, not args.no_orientation):
            f.write(line + "\n")
            print(json.loads(line)["id"], flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
