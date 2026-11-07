"""Training losses for the FNO surrogate (M2.2; CLAUDE.md §12).

    loss = rel_l2(pred, true) + w_hot · hotspot_l1(pred, true)     (w_hot default 0.1)

rel_l2: per sample ‖pred − true‖₂ / ‖true‖₂ over all layers and cells (scale-free).
hotspot_l1: per sample, mean |pred − true| over the HOT_CELLS hottest cells of the TRUE field,
any layer (CLAUDE.md §12; the same m = 64 as the gate's hotspot residual, §8.1). It is in the
units of its inputs: K on denormalised fields (validation, early stopping), normalised units
(K / σ_θ) inside the training loss (models/train_ensemble.py; UNSURE which §12 means, see there).
Both return float32 tensors of shape [B]; inputs are [B, 4, N, N] (rule 7).
"""

from __future__ import annotations

import torch

from models.train import relative_l2

__all__ = ["HOT_CELLS", "W_HOT", "hotspot_l1", "rel_l2"]

HOT_CELLS = 64  # CLAUDE.md §12: top-64 hottest true cells
W_HOT = 0.1  # CLAUDE.md §12: w_hot default


def rel_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return relative_l2(pred, true)


def hotspot_l1(pred: torch.Tensor, true: torch.Tensor, k: int = HOT_CELLS) -> torch.Tensor:
    if pred.shape != true.shape or pred.dim() < 2:
        raise ValueError(f"pred {tuple(pred.shape)} and true {tuple(true.shape)} must match "
                         "and be [B, ...]")
    t = true.float().flatten(1)
    if not 1 <= k <= t.shape[1]:
        raise ValueError(f"k = {k} hottest cells requested, the field has {t.shape[1]} cells")
    idx = t.topk(k, dim=1).indices
    p = pred.float().flatten(1)
    return (p.gather(1, idx) - t.gather(1, idx)).abs().mean(dim=1)
