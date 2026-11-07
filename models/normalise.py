"""Input and output normalisation fitted on the TRAIN split only (M2.1; CLAUDE.md §7.5, §5).

    norm = Normaliser.fit(root, N=256)            # reads root/train only; hashes verified first
    norm.save("models/norm_stats.json")
    norm = Normaliser.load("models/norm_stats.json", dataset_hash(root / "train"))

Per channel (§7.5): log10 G, log(1 + ν), h_top, h_bot -> (x − mean) / std; q -> q / P99 (no
shift); Dirichlet mask, x/L, y/L unchanged. Statistics are over every cell of every train board,
in float64; P99 is numpy's linear-interpolation percentile over all cells, computed exactly
(q >= 0, so zero cells sort first and only nonzero values need storing).
Output: θ_normalised = θ / σ_θ, σ_θ the std of every stored train θ value (CLAUDE.md §6.8 "θ
normalised by train std"; UNSURE: §7.5 does not restate it for v2, and a per-layer scale would
also fit the words). No mean shift, so θ = 0 stays 0 and the linearity of θ in power (B4) is
kept by the normalised target.
The stats carry the train split's dataset hash (SHA-256 over its manifest's per-file hashes);
load() refuses stats made from other data. Denormalisation inverts normalisation up to float32
rounding. Stats fitted at one N are meant for both resolutions (128² pretraining, 256²
fine-tuning, §12), so training does not switch scales between stages; N is recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from data import generate as gen
from data.raster import rasterise
from models import inputs as inp

__all__ = ["Normaliser", "NormStatsError", "dataset_hash"]

SCHEMA = "twinheat-norm-stats/1"
Q_PERCENTILE = 99.0  # §7.5: q is divided by the train P99


class NormStatsError(ValueError):
    """Stats cannot be fitted, or do not belong to this dataset."""


def dataset_hash(split_dir: Path) -> str:
    """SHA-256 identifying a split's content: its name, N and every shard file's SHA-256."""
    man = json.loads((Path(split_dir) / "manifest.json").read_text(encoding="utf-8"))
    ident = {"split": man["split"], "N": man["N"], "count": man["count"],
             "files": [[s["name"], s["files"]] for s in man["shards"]]}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


def _percentile_with_zeros(nonzero: np.ndarray, zeros: int, pct: float) -> float:
    """np.percentile(all, pct) for all = zeros x 0.0 plus `nonzero` (all > 0), linear method."""
    nz = np.sort(nonzero)
    n = zeros + nz.size
    pos = pct / 100.0 * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)

    def value(i: int) -> float:
        return 0.0 if i < zeros else float(nz[i - zeros])
    return value(lo) + (pos - lo) * (value(hi) - value(lo))


class Normaliser:
    def __init__(self, offset: np.ndarray, scale: np.ndarray, theta_scale: float,
                 dataset_hash: str, N: int, n_boards: int):
        self.offset = np.asarray(offset, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.theta_scale = float(theta_scale)
        self.dataset_hash = dataset_hash
        self.N = int(N)
        self.n_boards = int(n_boards)
        if self.offset.shape != (inp.N_CHANNELS,) or self.scale.shape != (inp.N_CHANNELS,):
            raise NormStatsError(f"offset and scale must have {inp.N_CHANNELS} entries")
        if not (np.isfinite(self.scale).all() and (self.scale > 0).all()
                and math.isfinite(self.theta_scale) and self.theta_scale > 0):
            raise NormStatsError("every scale must be finite and > 0")

    # ------------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, root: Path, N: int = 256, max_boards: int | None = None) -> Normaliser:
        """Fit on root/train only (hash-verified); max_boards: use the first boards only."""
        split_dir = Path(root) / "train"
        man = gen.verify(split_dir)
        if man["split"] != "train" or not man["complete"]:
            raise NormStatsError(f"{split_dir}: need a complete train split, got split "
                                 f"{man['split']!r}, complete = {man['complete']}")
        factor = _factor(man["N"], N)
        s1 = np.zeros(inp.N_CHANNELS)
        s2 = np.zeros(inp.N_CHANNELS)
        q_nonzero: list[np.ndarray] = []
        q_zeros = 0
        t1 = t2 = 0.0
        cells = t_count = boards = 0
        for sh in man["shards"]:
            d = split_dir / sh["name"]
            fields = np.load(d / "fields.npy", mmap_mode="r")
            lines = (d / "records.jsonl").read_text(encoding="utf-8").splitlines()
            for j, line in enumerate(lines):
                if max_boards is not None and boards >= max_boards:
                    break
                rec = json.loads(line)["record"]
                x = inp.build(rasterise(rec, gen.SPEC, N), gen.SPEC).astype(np.float64)
                flat = x.reshape(inp.N_CHANNELS, -1)
                s1 += flat.sum(axis=1)
                s2 += (flat * flat).sum(axis=1)
                q = flat[inp.Q_CHANNEL]
                q_nonzero.append(q[q > 0])
                q_zeros += int((q == 0).sum())
                if (q < 0).any():
                    raise NormStatsError(f"{rec['id']}: negative heat flux q")
                theta = _target(np.asarray(fields[j], dtype=np.float64), factor)
                t1 += float(theta.sum())
                t2 += float((theta * theta).sum())
                t_count += theta.size
                cells += flat.shape[1]
                boards += 1
        if boards == 0:
            raise NormStatsError(f"{split_dir}: no boards")
        mean = s1 / cells
        std = np.sqrt(np.maximum(s2 / cells - mean * mean, 0.0))
        offset = np.zeros(inp.N_CHANNELS)
        scale = np.ones(inp.N_CHANNELS)
        for c in inp.MEAN_STD_CHANNELS:
            if not std[c] > 0:
                raise NormStatsError(f"channel {c} ({inp.CHANNELS[c]}) is constant over the train "
                                     "split: it cannot be standardised")
            offset[c], scale[c] = mean[c], std[c]
        p99 = _percentile_with_zeros(np.concatenate(q_nonzero), q_zeros, Q_PERCENTILE)
        if not p99 > 0:
            raise NormStatsError(f"train P{Q_PERCENTILE:g} of q is {p99}: more than "
                                 f"{100 - Q_PERCENTILE:g} % of cells must carry heat")
        scale[inp.Q_CHANNEL] = p99
        t_mean = t1 / t_count
        theta_std = math.sqrt(max(t2 / t_count - t_mean * t_mean, 0.0))
        return cls(offset, scale, theta_std, dataset_hash(split_dir), N, boards)

    # ------------------------------------------------------------------- transform
    def _cols(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        if a.shape[-3] != inp.N_CHANNELS:
            raise ValueError(f"input must be [.., {inp.N_CHANNELS}, N, N], got {a.shape}")
        return v.reshape(inp.N_CHANNELS, 1, 1)

    def normalise_x(self, x: np.ndarray) -> np.ndarray:
        return ((x - self._cols(x, self.offset)) / self._cols(x, self.scale)).astype(np.float32)

    def denormalise_x(self, xn: np.ndarray) -> np.ndarray:
        return (xn * self._cols(xn, self.scale) + self._cols(xn, self.offset)).astype(np.float32)

    def normalise_y(self, theta: np.ndarray) -> np.ndarray:
        return (np.asarray(theta) / self.theta_scale).astype(np.float32)

    def denormalise_y(self, yn: np.ndarray) -> np.ndarray:
        """Network output -> θ in K (the server's step, §7.5)."""
        return (np.asarray(yn) * self.theta_scale).astype(np.float32)

    # ------------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "dataset_hash": self.dataset_hash, "fitted_on": "train",
                "N": self.N, "n_boards": self.n_boards, "channels": list(inp.CHANNELS),
                "offset": self.offset.tolist(), "scale": self.scale.tolist(),
                "theta_scale_K": self.theta_scale,
                "rules": {"mean_std": list(inp.MEAN_STD_CHANNELS), "divide_by_p99": inp.Q_CHANNEL,
                          "raw": list(inp.RAW_CHANNELS), "theta": "theta / theta_scale_K"}}

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, expected_dataset_hash: str) -> Normaliser:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("schema") != SCHEMA:
            raise NormStatsError(f"{path}: schema {d.get('schema')!r} != {SCHEMA}")
        if d["channels"] != list(inp.CHANNELS):
            raise NormStatsError(f"{path}: channel list differs from models/inputs.py")
        if d["dataset_hash"] != expected_dataset_hash:
            raise NormStatsError(f"{path}: dataset hash {d['dataset_hash'][:12]}… does not match "
                                 f"{expected_dataset_hash[:12]}…: stats were fitted on other data")
        return cls(d["offset"], d["scale"], d["theta_scale_K"], d["dataset_hash"], d["N"],
                   d["n_boards"])


def main(argv: list[str] | None = None) -> None:
    """python -m models.normalise --root DATASET_ROOT [--N 256] [--out models/norm_stats.json]"""
    import argparse
    ap = argparse.ArgumentParser(description="Fit normalisation stats on the train split")
    ap.add_argument("--root", type=Path, required=True, help="folder with one folder per split")
    ap.add_argument("--N", type=int, default=gen.SPEC.geometry.N.value)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "norm_stats.json")
    args = ap.parse_args(argv)
    norm = Normaliser.fit(args.root, N=args.N)
    norm.save(args.out)
    print(f"wrote {args.out}: {norm.n_boards} train boards at N = {norm.N}, dataset hash "
          f"{norm.dataset_hash[:12]}…, theta scale {norm.theta_scale:.4g} K")


def _factor(n_stored: int, n: int) -> int:
    """1 for the stored resolution, 2 for half of it (2×2 area averaging, §12)."""
    if n == n_stored:
        return 1
    if 2 * n == n_stored:
        return 2
    raise ValueError(f"N = {n}: the data is stored at N = {n_stored}; only N = {n_stored} or "
                     f"N = {n_stored // 2} (2×2 area averages) are supported")


def _target(theta: np.ndarray, factor: int) -> np.ndarray:
    """[4, n, n] stored θ, 2×2 area-averaged when factor = 2."""
    if factor == 1:
        return theta
    n = theta.shape[-1] // 2
    return theta.reshape(*theta.shape[:-2], n, 2, n, 2).mean(axis=(-3, -1))


if __name__ == "__main__":
    main()
