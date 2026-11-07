"""Model inputs: a Raster -> the 13-channel tensor of CLAUDE.md §7.5 (M2.1; invariant B6).

    build(raster, spec) -> float32 [13, N, N]

Channel order (§7.5), arrays [y, x] (rule 7):
   0-3  log10 G_l for l = 1..4, sheet conductance in W/K (solvers/fvm.layer_conductances), > 0
   4    q on L1, W/m²
   5-7  log(1 + ν) for the gaps L1-L2, L2-L3, L3-L4, ν in vias/mm² (the raster holds vias/m²)
   8    h_top, W/(m²·K)
   9    h_bot, W/(m²·K)
  10    Dirichlet mask: 1 on cells next to a heat-sunk edge (§6.6)
  11    x/L at cell centres, (i + 0.5)/N
  12    y/L at cell centres
Values are raw here; models/normalise.py standardises them. The model sees no L4 heat channel:
bottom-side chips (q_bot) are invisible to it, so the runtime rule check sends such boards to the
exact solver (docs/decisions.md, OOD sets v2).

transform_field(a, sym) applies square symmetry sym (0..7) to a [.., y, x] array exactly as
data/raster.transform_array does to a Raster (and data/sampler.transform_record to a record), so
an augmented target is transformed the same way as its re-rasterised input.
"""

from __future__ import annotations

import numpy as np

from solvers.fvm import layer_conductances

__all__ = ["CHANNELS", "MEAN_STD_CHANNELS", "N_CHANNELS", "Q_CHANNEL", "RAW_CHANNELS", "build",
           "coordinate_channels", "transform_field"]

CHANNELS = (
    "log10_G_L1", "log10_G_L2", "log10_G_L3", "log10_G_L4", "q_L1",
    "log1p_nu_L1L2", "log1p_nu_L2L3", "log1p_nu_L3L4", "h_top", "h_bot",
    "dirichlet_mask", "x_over_L", "y_over_L",
)
N_CHANNELS = len(CHANNELS)  # 13, CLAUDE.md §7.5
Q_CHANNEL = 4  # divided by the train P99 (§7.5)
MEAN_STD_CHANNELS = (0, 1, 2, 3, 5, 6, 7, 8, 9)  # standardised with train mean / std (§7.5)
RAW_CHANNELS = (10, 11, 12)  # not normalised (§7.5)
PER_M2_TO_PER_MM2 = 1e-6  # vias/m² -> vias/mm²


def coordinate_channels(n: int) -> np.ndarray:
    """[2, n, n] float32: x/L and y/L at cell centres. Always regenerated, never transformed."""
    c = ((np.arange(n) + 0.5) / n).astype(np.float32)
    return np.stack([np.broadcast_to(c[None, :], (n, n)), np.broadcast_to(c[:, None], (n, n))])


def build(raster, spec) -> np.ndarray:
    """The §7.5 input tensor of one board, float32 [13, N, N], raw (not normalised)."""
    n = raster.N
    G = layer_conductances(spec, raster)  # [4, N, N] W/K
    if not np.all(np.isfinite(G)) or not (G > 0).all():
        k, y, x = np.unravel_index(np.argmin(np.where(np.isfinite(G), G, -np.inf)), G.shape)
        raise ValueError(f"sheet conductance G must be > 0: L{k + 1} cell (x {x}, y {y}) has "
                         f"{G[k, y, x]} W/K")
    out = np.empty((N_CHANNELS, n, n), dtype=np.float32)
    out[0:4] = np.log10(G)
    out[4] = raster.q
    out[5:8] = np.log1p(np.asarray(raster.nu) * PER_M2_TO_PER_MM2)
    out[8] = raster.h_top
    out[9] = raster.h_bot
    out[10] = raster.dirichlet_mask
    out[11:13] = coordinate_channels(n)
    return out


def transform_field(a: np.ndarray, sym: int) -> np.ndarray:
    """a [.., y, x] under square symmetry sym = r + 4 f: mirror x if f, then r quarter turns
    counter-clockwise (the map of data/raster.transform_array). Returns a contiguous copy."""
    if isinstance(sym, bool) or not isinstance(sym, int | np.integer) or not 0 <= sym < 8:
        raise ValueError(f"sym must be an int in 0..7, got {sym!r}")
    r, f = int(sym) % 4, int(sym) // 4
    if f:
        a = a[..., ::-1]
    return np.ascontiguousarray(np.rot90(a, k=-r, axes=(-2, -1)))
