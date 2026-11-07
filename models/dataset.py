"""TwinHeat dataset: stored boards -> 13-channel input, 4-layer target (M2.1; CLAUDE.md §7.5, §12).

    ds = TwinHeatDataset(root, "train", N=256, augment=True, normaliser=norm)
    x, y = ds[i]          # float32 [13, N, N] and [4, N, N]; normalised if a normaliser is given

root holds one folder per split, each written by data/generate.py. At init the split's shard
hashes, counts and shapes are verified (data.generate.verify), and protected splits (cal, test,
ood_*) are refused unless allow_eval=True (models/data_guard.py, CLAUDE.md §5).

Inputs are re-rasterised from the stored records at the requested N (§7.4), optionally cached on
disk as <record SHA-256>-N<N>.npy (raw, before normalisation). Targets are the stored float16
fields in K; N = half the stored N gives 2×2 area averages (128² pretraining, §12). Augmentation
draws one of the 8 square symmetries per item and applies it to the record (transform_record,
then re-rasterise) and to the target (transform_field), so input and target move together; the
x/L and y/L channels are regenerated, never transformed. The symmetry of item i is a hash of
(seed, epoch, i): reproducible, and different each epoch after set_epoch().
Returned arrays are NumPy; torch's default collate turns them into tensors.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from data import generate as gen
from data.raster import rasterise
from data.sampler import transform_record
from models import inputs as inp
from models.data_guard import open_split
from models.normalise import Normaliser, _factor, _target
from solvers.fvm import coefficients

__all__ = ["TwinHeatDataset"]


class TwinHeatDataset:
    def __init__(self, root: Path, split: str, N: int = 256, augment: bool = False,
                 allow_eval: bool = False, normaliser: Normaliser | None = None,
                 cache_dir: Path | None = None, seed: int = 0):
        self.split = open_split(split, lambda s: s, allow_eval=allow_eval)
        self.dir = Path(root) / split
        man = gen.verify(self.dir)  # hashes, counts, shapes
        if not man["complete"]:
            raise gen.ManifestError(f"{self.dir}: split is incomplete "
                                    f"({len(man['shards'])} shards)")
        self.factor = _factor(man["N"], N)
        self.N = N
        self.augment = augment
        self.normaliser = normaliser
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.epoch = 0
        self._records: list[dict] = []
        self._where: list[tuple[int, int]] = []  # (shard index, row in shard)
        self._fields = []
        for k, sh in enumerate(man["shards"]):
            d = self.dir / sh["name"]
            self._fields.append(np.load(d / "fields.npy", mmap_mode="r"))
            lines = (d / "records.jsonl").read_text(encoding="utf-8").splitlines()
            for j, line in enumerate(lines):
                self._records.append(json.loads(line)["record"])
                self._where.append((k, j))

    def __len__(self) -> int:
        return len(self._records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def symmetry(self, i: int) -> int:
        """The symmetry applied to item i this epoch (0 when augment is off)."""
        if not self.augment:
            return 0
        h = hashlib.sha256(f"{self.seed}\0{self.epoch}\0{i}".encode()).digest()
        return int.from_bytes(h[:8], "big") % 8

    def record(self, i: int, sym: int = 0) -> dict:
        rec = self._records[i]
        return transform_record(rec, sym) if sym else rec

    def _inputs(self, rec: dict) -> np.ndarray:
        if self.cache_dir is None:
            return inp.build(rasterise(rec, gen.SPEC, self.N), gen.SPEC)
        path = self.cache_dir / f"{gen.record_hash(rec)}-N{self.N}.npy"
        if path.exists():
            return np.load(path)
        x = inp.build(rasterise(rec, gen.SPEC, self.N), gen.SPEC)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            np.save(f, x, allow_pickle=False)
        os.replace(tmp, path)
        return x

    def sample(self, i: int, sym: int) -> tuple[np.ndarray, np.ndarray]:
        """(input, target) of item i under symmetry sym, normalised if a normaliser is set."""
        if not 0 <= i < len(self):
            raise IndexError(f"item {i} out of range 0..{len(self) - 1}")
        x = self._inputs(self.record(i, sym))
        k, j = self._where[i]
        y = _target(np.asarray(self._fields[k][j], dtype=np.float32), self.factor)
        y = inp.transform_field(y, sym).astype(np.float32)
        if self.normaliser is not None:
            x, y = self.normaliser.normalise_x(x), self.normaliser.normalise_y(y)
        return x, y

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.sample(i, self.symmetry(i))

    def coefficients(self, i: int, sym: int) -> dict[str, np.ndarray]:
        """Stencil coefficients and rhs (solvers/fvm.coefficients, W/K and W) of item i under
        symmetry sym, at this N, float32 [4, N, N]: what the energy loss (models/energy.py) needs.
        Re-rasterised from the transformed record, so they match the augmented sample."""
        rec = self.record(i, sym)
        r = rasterise(rec, gen.SPEC, self.N)
        c = coefficients(gen.SPEC, r, rec["dirichlet_edges"])
        return {k: v.astype(np.float32) for k, v in c.items()}
