"""Achieved memory bandwidth of the naive and tiled stencil kernels (N1.4; CLAUDE.md §9.4).

    python -m bench.stencil_bw [--out bench/results/stencil_bw.csv]

Needs a CUDA GPU (run on Kaggle). For each kernel, dtype and batch size, times A x on the
256² × 4 board with CUDA events (10 warm-ups, median of 20, as N1.5 specifies) and reports
    GB/s = (7 coefficient arrays + x read + y written) × cells × itemsize / median time,
the compulsory traffic, against the T4's nominal ~320 GB/s (the N1.4 prompt's figure). Re-reads
of neighbours that miss in cache are not counted, so the figure is a lower bound on the traffic
the kernel actually moves. Board 0 of the sampler is replicated B times: bandwidth does not depend
on the coefficient values. One row per configuration, with commit, GPU and date (CLAUDE.md §4).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
from numba import config, cuda

from data.raster import rasterise
from data.sampler import sample_board
from solvers import cuda_kernels as ck
from solvers.fvm import coefficients
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
PEAK_GBS = 320.0  # T4 nominal, from the N1.4 prompt
WARMUP, REPS = 10, 20
BATCHES = (1, 16, 64)
DTYPES = (np.float32, np.float64)
STREAMS = len(ck.FIELDS) + 2  # 7 coefficients + x + y


def _commit() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def time_kernel(apply, c, x, out) -> float:
    """Median milliseconds of apply(c, x, out) over REPS launches after WARMUP launches."""
    for _ in range(WARMUP):
        apply(c, x, out)
    cuda.synchronize()
    ms = []
    for _ in range(REPS):
        start, stop = cuda.event(), cuda.event()
        start.record()
        apply(c, x, out)
        stop.record()
        stop.synchronize()
        ms.append(cuda.event_elapsed_time(start, stop))
    return float(np.median(ms))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "stencil_bw.csv")
    args = ap.parse_args()
    if config.ENABLE_CUDASIM or not cuda.is_available():
        raise SystemExit("stencil_bw needs a CUDA GPU (not the simulator)")
    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    n = spec.geometry.N.value
    rec = sample_board(0, "id")
    c = coefficients(spec, rasterise(rec, spec, n), rec["dirichlet_edges"])
    gpu = cuda.get_current_device().name
    gpu = gpu.decode() if isinstance(gpu, bytes) else str(gpu)
    rows = []
    for dtype in DTYPES:
        item = np.dtype(dtype).itemsize
        for B in BATCHES:
            dc = ck.upload_coefficients([c] * B, dtype)
            x = cuda.to_device(np.random.default_rng(0).uniform(0, 100, (B, 4, n, n))
                               .astype(dtype))
            out = cuda.device_array_like(x)
            cells = B * 4 * n * n
            for name in ("naive", "tiled"):
                ms = time_kernel(getattr(ck, f"stencil_apply_{name}"), dc, x, out)
                gbs = STREAMS * cells * item / (ms * 1e-3) / 1e9
                rows.append({
                    "date": dt.date.today().isoformat(), "commit": _commit(), "gpu": gpu,
                    "kernel": name, "dtype": np.dtype(dtype).name, "B": B, "N": n, "layers": 4,
                    "median_ms": f"{ms:.4f}", "achieved_GBs": f"{gbs:.1f}",
                    "peak_GBs_nominal": PEAK_GBS, "pct_of_peak": f"{100 * gbs / PEAK_GBS:.1f}",
                    "bytes_model": f"{STREAMS} streams x cells x {item} B",
                    "timing": f"CUDA events, {WARMUP} warm-ups, median of {REPS}",
                })
                print(f"{name:5s} {np.dtype(dtype).name:7s} B={B:3d}: {ms:8.3f} ms "
                      f"{gbs:7.1f} GB/s ({100 * gbs / PEAK_GBS:.0f} % of {PEAK_GBS:.0f})")
            del dc, x, out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
