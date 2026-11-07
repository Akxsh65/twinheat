"""E5 accuracy metrics per board (M2.3; CLAUDE.md §11.5, §12). All inputs are θ in K, [B, 4, N, N].

    rel_l2         ‖pred − true‖₂ / ‖true‖₂, over all layers ("all") or one layer
    peak_error_K   |max pred − max true|: the hotspot temperature error (the quantity the gate
                   scores, T_peak, §8.1), over all layers or per layer
    hotspot_l1_K   mean |pred − true| over the 64 hottest TRUE cells (all layers, or within one
                   layer): the quantity M2.2 early-stops on and the §12 go/no-go reads (UNSURE)
    peak_dist_mm   in-plane distance between the hottest predicted and the hottest true cell, in
                   mm (cell size L/N); over all layers it ignores which layer holds the peak
Each function returns {"all": [B], "L1": [B], ..., "L4": [B]} as NumPy float64 arrays.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HOT_CELLS", "LAYERS", "METRICS", "board_metrics"]

LAYERS = ("L1", "L2", "L3", "L4")
HOT_CELLS = 64  # CLAUDE.md §12, as models/losses.py
METRICS = ("rel_l2", "peak_error_K", "hotspot_l1_K", "peak_dist_mm")


def _check(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, t = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    if p.shape != t.shape or p.ndim != 4 or p.shape[1] != 4:
        raise ValueError(f"pred {p.shape} and true {t.shape} must both be [B, 4, N, N]")
    return p, t


def _rel(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    den = np.sqrt((t * t).reshape(len(t), -1).sum(1))
    if (den == 0).any():
        raise ValueError("relative L2: a true field is identically zero")
    return np.sqrt(((p - t) ** 2).reshape(len(t), -1).sum(1)) / den


def _hot(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    pf, tf = p.reshape(len(p), -1), t.reshape(len(t), -1)
    k = min(HOT_CELLS, tf.shape[1])
    idx = np.argpartition(-tf, k - 1, axis=1)[:, :k]
    return np.abs(np.take_along_axis(pf, idx, 1) - np.take_along_axis(tf, idx, 1)).mean(1)


def _argmax_yx(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(y, x) of the maximum of each a[b] (any leading layer axis is flattened away)."""
    n = a.shape[-1]
    k = a.reshape(len(a), -1).argmax(1) % (n * n)
    return k // n, k % n


def board_metrics(pred: np.ndarray, true: np.ndarray, board_mm: float
                  ) -> dict[str, dict[str, np.ndarray]]:
    p, t = _check(pred, true)
    cell = board_mm / p.shape[-1]
    out: dict[str, dict[str, np.ndarray]] = {m: {} for m in METRICS}

    def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        (ya, xa), (yb, xb) = _argmax_yx(a), _argmax_yx(b)
        return np.hypot(ya - yb, xa - xb) * cell

    out["rel_l2"]["all"] = _rel(p, t)
    out["peak_error_K"]["all"] = np.abs(p.reshape(len(p), -1).max(1) - t.reshape(len(t), -1).max(1))
    out["hotspot_l1_K"]["all"] = _hot(p, t)
    out["peak_dist_mm"]["all"] = dist(p, t)
    for layer, name in enumerate(LAYERS):
        pl, tl = p[:, layer:layer + 1], t[:, layer:layer + 1]
        out["rel_l2"][name] = _rel(pl, tl)
        out["peak_error_K"][name] = np.abs(pl.reshape(len(p), -1).max(1)
                                           - tl.reshape(len(t), -1).max(1))
        out["hotspot_l1_K"][name] = _hot(pl, tl)
        out["peak_dist_mm"][name] = dist(pl, tl)
    return out
