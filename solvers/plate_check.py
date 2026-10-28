"""Write bench/figs/plate_check.png: θ for one centred chip on the 64² plate (N1.1 visual check).

    python -m solvers.plate_check [--out bench/figs/plate_check.png]

Board: N = 64, L from config/spec_v1.yaml; G = one full 1 oz plane, bare (W/K); H = 10 W/(m²·K)
per face; one 1 W chip on the centred 25 x 25 mm square (whole cells); all edges Neumann. The
field should be smooth and symmetric under x-flip, y-flip and transpose. Footer per CLAUDE.md §4.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
from pathlib import Path

import numpy as np

from solvers.fvm import assemble_plate, energy_balance, solve_direct
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]


def centred_chip_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(G W/K, H W/(m²·K), q W/m², dx m) for the check board described in the module docstring."""
    s = specmod.load(ROOT / "config" / "spec_v1.yaml")
    n = s.geometry.N.value
    dx = s.dx
    G = np.full((n, n), s.sheet_conductance(1, 0.0))
    H = np.full((n, n), 2 * 10.0)
    q = np.zeros((n, n))
    lo, hi = n * 3 // 8, n * 5 // 8
    q[lo:hi, lo:hi] = 1.0 / (((hi - lo) * dx) ** 2)
    return G, H, q, dx


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
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "figs" / "plate_check.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G, H, q, dx = centred_chip_case()
    n = G.shape[0]
    A, b = assemble_plate(G, H, q, dx, [])
    theta = solve_direct(A, b).reshape(n, n)
    eb = energy_balance(theta, G, H, q, dx, [])
    asym = np.abs(theta - theta[:, ::-1]).max() / theta.max()
    L_mm = n * dx * 1e3

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    im = ax0.imshow(theta, origin="lower", extent=(0, L_mm, 0, L_mm), cmap="inferno")
    ax0.contour(theta, levels=12, origin="lower", extent=(0, L_mm, 0, L_mm), colors="w",
                linewidths=0.5)
    ax0.set(title="θ (K), centred 1 W chip, all edges Neumann", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im, ax=ax0, label="θ (K)")
    c = (np.arange(n) + 0.5) * dx * 1e3
    ax1.plot(c, theta[n // 2], label="row y = N/2")
    ax1.plot(c, theta[:, n // 2], "--", label="column x = N/2")
    ax1.set(title="centre-line profiles", xlabel="position (mm)", ylabel="θ (K)")
    ax1.legend()
    footer = (f"commit {_git_commit()} · dataset none (analytic board) · seed none · "
              f"{platform.platform()} {platform.processor()} · {dt.date.today().isoformat()} · "
              f"energy balance {eb:.1e} · x-flip asymmetry {asym:.1e}")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))  # keep the footer clear of the axes
    fig.text(0.01, 0.008, footer, fontsize=6, color="0.35")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}; peak {theta.max():.3f} K; {footer}")


if __name__ == "__main__":
    main()
