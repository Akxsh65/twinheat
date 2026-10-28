"""Write data/figs/raster_check.png: 12 sampled boards rasterised at N = 256 (M1.1 visual check).

    python -m data.raster_check [--out data/figs/raster_check.png] [--n 256]

Rows: seeds 0..8 in mode "id", 0..2 in mode "design". Columns: φ for L1..L4, q on L1 (W/m²) and ν
(vias/mm², the same on all three gaps). Origin lower left, x right, y up (CLAUDE.md §0 rule 7);
the Dirichlet edges are drawn in cyan. Footer per CLAUDE.md §4. A second person reviews it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
from pathlib import Path

import numpy as np

from data.raster import rasterise
from data.sampler import sample_board
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
BOARDS = [(s, "id") for s in range(9)] + [(s, "design") for s in range(3)]
EDGE_LINES = {"W": ([0, 0], [0, 1]), "E": ([1, 1], [0, 1]), "S": ([0, 1], [0, 0]),
              "N": ([0, 1], [1, 1])}  # fmt: skip


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                               capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"figure footer needs the git commit: {e}") from e
    return sha + ("-dirty" if dirty else "")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "figs" / "raster_check.png")
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    commit = _git_commit()
    cols = ["φ L1", "φ L2", "φ L3", "φ L4", "q L1 (W/m²)", "ν (vias/mm²)"]
    fig, axes = plt.subplots(len(BOARDS), len(cols), figsize=(11, 22), constrained_layout=True)
    for row, (seed, mode) in enumerate(BOARDS):
        rec = sample_board(seed, mode)
        r = rasterise(rec, spec, args.n)
        panels = [r.phi[0], r.phi[1], r.phi[2], r.phi[3], r.q, r.nu[0] / 1e6]
        for col, (ax, img) in enumerate(zip(axes[row], panels, strict=True)):
            vmax = 1.0 if col < 4 else max(float(img.max()), 1e-12)
            im = ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap="inferno", vmin=0,
                           vmax=vmax, interpolation="nearest")
            for e in rec["dirichlet_edges"]:
                ax.plot(*EDGE_LINES[e], color="cyan", lw=2.5, clip_on=False)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(cols[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{mode} seed {seed}\nedges {''.join(rec['dirichlet_edges']) or '-'}",
                              fontsize=8)
            if col >= 4:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        total = r.q.sum() * r.dx**2
        want = sum(c["P"] for c in rec["chips"])
        if not np.isclose(total, want, rtol=1e-12, atol=0):
            raise AssertionError(f"{rec['id']}: Σ q·Δx² = {total} W, chips total {want} W")
    footer = (f"commit {commit} · dataset none (sampler seeds above) · N = {args.n} · "
              f"{platform.platform()} {platform.processor()} · {dt.date.today().isoformat()}")
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.988))
    fig.text(0.01, 0.003, footer, fontsize=6, color="0.35")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}; {footer}")


if __name__ == "__main__":
    main()
