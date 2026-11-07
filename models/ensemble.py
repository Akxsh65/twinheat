"""The FNO ensemble: load registered members, predict mean and spread in kelvin (M2.2; §8.1, §12).

    e = Ensemble.load("models/registry.json", "models/norm_stats.json", stage="ft256")
    mu, sd = e.predict(x)        # x raw inputs [B, 13, N, N] (models/inputs.build); K, float32

Load checks, for every member of the stage: the checkpoint's SHA-256 against the registry, the
norm_stats file's SHA-256 against the one the member was trained with, and the stats' dataset
hash; any mismatch raises EnsembleError. predict() normalises x, runs every member, denormalises
each output to kelvin and returns mu = the member mean and sd = the member standard deviation
(population, over members). Both are computed from deviations from the first member, so members
that are bit-identical give sd = 0 exactly and mu equal to that member.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models import zoo
from models.normalise import Normaliser, NormStatsError

__all__ = ["Ensemble", "EnsembleError"]


class EnsembleError(RuntimeError):
    """A registered member cannot be loaded as recorded."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ensemble:
    def __init__(self, models: list[torch.nn.Module], norm: Normaliser, device: torch.device,
                 members: list[dict]):
        if not models:
            raise EnsembleError("an ensemble needs at least one member")
        self.models, self.norm, self.device, self.members = models, norm, device, members

    @classmethod
    def load(cls, registry: Path, norm_stats: Path, stage: str = "ft256", device: str = "cpu",
             arch: str = "fno", variant: str = "base", members: list[int] | None = None
             ) -> Ensemble:
        """The registered models of (stage, arch, variant); `members` picks some of them (e.g.
        [3] for member 3 alone). Entries written before arch/variant existed are fno / base."""
        registry = Path(registry)
        reg = json.loads(registry.read_text(encoding="utf-8"))
        sel = [m for m in reg["members"] if m["stage"] == stage and m.get("arch", "fno") == arch
               and m.get("variant", "base") == variant
               and (members is None or m["member"] in members)]
        members = sorted(sel, key=lambda m: m["member"])
        if not members:
            raise EnsembleError(f"{registry}: no members for stage {stage!r}, arch {arch!r}, "
                                f"variant {variant!r}")
        norm_sha = _sha256(Path(norm_stats))
        hashes = {m["dataset_hash"] for m in members}
        if len(hashes) != 1 or any(m["norm_stats_sha256"] != norm_sha for m in members):
            raise EnsembleError(f"{norm_stats}: SHA-256 {norm_sha[:12]}… is not the norm stats "
                                "every member was trained with (or members disagree)")
        try:
            norm = Normaliser.load(norm_stats, hashes.pop())
        except NormStatsError as e:
            raise EnsembleError(f"norm stats: {e}") from e
        dev = torch.device(device)
        models = []
        for m in members:
            p = Path(m["checkpoint"])
            p = p if p.is_absolute() else registry.parent / p
            if not p.exists():
                raise EnsembleError(f"member {m['member']}: checkpoint {p} not found")
            got = _sha256(p)
            if got != m["sha256"]:
                raise EnsembleError(f"member {m['member']}: {p} SHA-256 {got[:12]}… != registry "
                                    f"{m['sha256'][:12]}…")
            state = torch.load(p, map_location=dev, weights_only=False)
            model = zoo.build(state["model_config"]).to(dev)
            model.load_state_dict(state["model"])
            model.eval()
            models.append(model)
        return cls(models, norm, dev, members)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """mu, sd [B, 4, N, N] in kelvin (float32) for raw inputs x [B, 13, N, N]."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"x must be [B, 13, N, N], got {x.shape}")
        xt = torch.from_numpy(self.norm.normalise_x(x)).to(self.device)
        outs = [m(xt).float() * self.norm.theta_scale for m in self.models]  # K
        first = outs[0]
        dev = torch.stack([o - first for o in outs])  # exact 0 for identical members
        shift = dev.mean(dim=0)
        mu = first + shift
        sd = (dev - shift).square().mean(dim=0).sqrt()
        return mu.cpu().numpy().astype(np.float32), sd.cpu().numpy().astype(np.float32)
