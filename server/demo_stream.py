"""Write a synthetic demo stream of pred messages for the mock server (viewer development only).

The fields are SYNTHETIC: moving Gaussian blobs, smoother and weaker on the deeper layers.
They are not solver or model output and carry no physical meaning. Their ids start with "demo-"
and every message is checked against contracts/schemas before it is written.

    python -m server.demo_stream --out server/demo/demo.jsonl [--frames 60] [--n 256] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from contracts.f16 import encode
from contracts.validate import validate

__all__ = ["fields", "messages", "main"]

# Shape of the synthetic picture (display only, not physics): blob count, peak rise of the top
# layer (K), blob width as a fraction of the board, per-layer attenuation and widening.
_BLOBS = 3
_PEAK_K = 60.0
_WIDTH = 0.08
_LAYER_SCALE = (1.0, 0.7, 0.5, 0.4)
_LAYER_WIDEN = (1.0, 1.6, 2.2, 2.8)
_SD_FRACTION = 0.05


def fields(frames: int, n: int, seed: int) -> Iterator[np.ndarray]:
    """Yield `frames` synthetic [4, n, n] float64 arrays (K), row-major [layer, y, x]."""
    if frames < 1 or n < 1:
        raise ValueError(f"frames and n must be >= 1, got frames={frames}, n={n}")
    rng = np.random.Generator(np.random.PCG64(seed))
    centre = rng.uniform(0.3, 0.7, size=(_BLOBS, 2))
    radius = rng.uniform(0.1, 0.2, size=_BLOBS)
    phase = rng.uniform(0, 2 * np.pi, size=_BLOBS)
    amp = rng.uniform(0.5, 1.0, size=_BLOBS)
    c = (np.arange(n) + 0.5) / n  # cell centres, board units
    y, x = np.meshgrid(c, c, indexing="ij")
    for k in range(frames):
        ang = phase + 2 * np.pi * k / frames
        bx = centre[:, 0] + radius * np.cos(ang)
        by = centre[:, 1] + radius * np.sin(ang)
        out = np.empty((4, n, n))
        for layer in range(4):
            w = _WIDTH * _LAYER_WIDEN[layer]
            d2 = (x[None] - bx[:, None, None]) ** 2 + (y[None] - by[:, None, None]) ** 2
            out[layer] = _LAYER_SCALE[layer] * _PEAK_K * np.sum(
                amp[:, None, None] * np.exp(-d2 / (2 * w * w)), axis=0)
        yield out


def messages(frames: int = 60, n: int = 256, seed: int = 0) -> Iterator[str]:
    """Yield contract-valid pred messages (JSON text, one per frame)."""
    for k, mu in enumerate(fields(frames, n, seed)):
        msg = {
            "schema_version": "1.0", "type": "pred", "id": f"demo-{seed}-{k:04d}",
            "mu": encode(mu), "sd": encode(_SD_FRACTION * mu),
            "B": None, "decision": "preview", "group": None, "peak": float(mu.max()),
            "units": {"mu": "K", "sd": "K", "B": "K", "peak": "K"}, "shape": [4, n, n],
        }
        validate(msg, "pred")
        yield json.dumps(msg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for line in messages(args.frames, args.n, args.seed):
            f.write(line + "\n")
    print(f"wrote {args.frames} synthetic pred messages ({args.n}x{args.n}) to {args.out}")


if __name__ == "__main__":
    main()
