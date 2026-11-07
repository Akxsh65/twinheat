"""FNO with a local 3×3 branch (M2.3; CLAUDE.md §12, the ablation for sharp interfaces).

FNO2d (models/fno.py) with one extra term per Fourier layer: x <- σ(K x + W x + C x), C a 3×3
convolution (zero padding) next to the spectral K and pointwise W paths. The spectral path is
smooth by construction (24 modes); the 3×3 path can represent the jumps in G at copper edges.
Same lifting, width, modes, layer count and projection as FNO2d, so the two differ only in C.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.fno import SpectralConv2d

__all__ = ["FNOLocal2d"]


class FNOLocal2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, width: int = 48, modes: int = 24,
                 n_layers: int = 4, proj_hidden: int = 128) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(n_layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(n_layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(n_layers))
        self.proj = nn.Sequential(nn.Conv2d(width, proj_hidden, 1), nn.GELU(),
                                  nn.Conv2d(proj_hidden, out_ch, 1))  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        last = len(self.spectral) - 1
        for i, (k, w, c) in enumerate(zip(self.spectral, self.pointwise, self.local, strict=True)):
            x = k(x) + w(x) + c(x)
            if i < last:
                x = F.gelu(x)
        return self.proj(x)
