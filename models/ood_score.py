"""Learned OOD score: Mahalanobis distance of pooled FNO features (M2.3; CLAUDE.md §6.7 check (2)).

Features: the input of the FNO's projection head, i.e. the output of the last Fourier layer
(width 48 channels), averaged over the board -> a 48-vector per board and member (UNSURE: §6.7
says "pooled FNO latents"; the projection's 128-wide hidden layer would also fit). For each
member: mean and Ledoit–Wolf shrunk covariance of the TRAIN features (scikit-learn's
LedoitWolf, not re-derived here), distance d = sqrt((f − μ)ᵀ Σ⁻¹ (f − μ)). The board's score is
the mean of the members' distances. The flag threshold is the 95th percentile of the VALIDATION
scores (§6.7); flagged boards go to the exact solver and their bands are shown as not guaranteed.
Fit on train, threshold on val: test and OOD splits are only scored (CLAUDE.md §5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch import nn

__all__ = ["OODScore", "pooled_features"]

THRESHOLD_PERCENTILE = 95.0  # §6.7: validation 95th percentile


@torch.no_grad()
def pooled_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """[B, C] board-averaged input of model.proj (the last Fourier layer's output)."""
    if not hasattr(model, "proj"):
        raise TypeError(f"{type(model).__name__} has no projection head: OOD features are FNO-only")
    got: list[torch.Tensor] = []
    handle = model.proj.register_forward_hook(lambda m, inputs, out: got.append(inputs[0]))
    try:
        model(x)
    finally:
        handle.remove()
    return got[0].float().mean(dim=(-2, -1))


class OODScore:
    def __init__(self, means: list[np.ndarray], precisions: list[np.ndarray],
                 threshold: float | None = None):
        self.means = [np.asarray(m, np.float64) for m in means]
        self.precisions = [np.asarray(p, np.float64) for p in precisions]
        self.threshold = threshold

    @classmethod
    def fit(cls, train_features: list[np.ndarray]) -> OODScore:
        """train_features: one [n, C] array per member."""
        means, precs = [], []
        for f in train_features:
            f = np.asarray(f, np.float64)
            if f.ndim != 2 or len(f) < 2:
                raise ValueError(f"need [n >= 2, C] features per member, got {f.shape}")
            lw = LedoitWolf().fit(f)
            means.append(lw.location_)
            precs.append(lw.precision_)
        return cls(means, precs)

    def score(self, features: list[np.ndarray]) -> np.ndarray:
        """Mean over members of the Mahalanobis distance, [n]."""
        if len(features) != len(self.means):
            raise ValueError(f"{len(features)} feature sets for {len(self.means)} members")
        ds = []
        for f, mu, prec in zip(features, self.means, self.precisions, strict=True):
            d = np.asarray(f, np.float64) - mu
            ds.append(np.sqrt(np.einsum("ni,ij,nj->n", d, prec, d)))
        return np.mean(ds, axis=0)

    def set_threshold(self, val_features: list[np.ndarray],
                      percentile: float = THRESHOLD_PERCENTILE) -> float:
        self.threshold = float(np.percentile(self.score(val_features), percentile))
        return self.threshold

    def flag(self, features: list[np.ndarray]) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("set_threshold() on validation features first")
        return self.score(features) > self.threshold

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "schema": "twinheat-ood-score/1", "threshold": self.threshold,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "means": [m.tolist() for m in self.means],
            "precisions": [p.tolist() for p in self.precisions]}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> OODScore:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["means"], d["precisions"], d["threshold"])
