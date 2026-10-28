"""Fourier neural operator (prompt M1.3; CLAUDE.md §12): FNO2d(in_ch, out_ch, width, modes, ...).

Tensors are [B, C, y, x] (CLAUDE.md §0 rule 7, batch first). Each Fourier layer is
x -> GELU(K(x) + W x) with K a truncated spectral convolution and W a 1x1 convolution; the last
layer has no GELU. A 1x1 lifting maps in_ch -> width; a two-layer projection maps width -> out_ch.

SpectralConv2d keeps `modes` frequencies per axis: along x (rfft2's half spectrum) the lowest
`modes`, and along y BOTH the lowest positive and the lowest negative `modes` (rows 0..m-1 and
-m..-1), each block with its own complex weights. The FFT runs in float32 with autocast off, so
AMP never sees half-precision complex numbers.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["FNO2d", "SpectralConv2d"]


class SpectralConv2d(nn.Module):
    """Truncated spectral convolution, [B, in_ch, H, W] -> [B, out_ch, H, W]."""

    def __init__(self, in_ch: int, out_ch: int, modes: int) -> None:
        super().__init__()
        if min(in_ch, out_ch, modes) < 1:
            raise ValueError(f"in_ch, out_ch, modes must be >= 1, got {in_ch}, {out_ch}, {modes}")
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        shape = (in_ch, out_ch, modes, modes)
        self.w_pos = nn.Parameter(scale * torch.rand(shape, dtype=torch.cfloat))  # y >= 0
        self.w_neg = nn.Parameter(scale * torch.rand(shape, dtype=torch.cfloat))  # y < 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        m = self.modes
        if c != self.in_ch:
            raise ValueError(f"expected {self.in_ch} channels, got {c}")
        if 2 * m > h or m > w // 2 + 1:
            raise ValueError(f"modes = {m} needs H >= {2 * m} and W >= {2 * (m - 1)}, got {h}x{w}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.float(), norm="ortho")  # [B, C, H, W//2 + 1]
            out = torch.zeros(b, self.out_ch, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
            out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w_pos)
            out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w_neg)
            return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class FNO2d(nn.Module):
    """FNO with `n_layers` Fourier layers of `width` channels and `modes` modes per axis.

    CLAUDE.md §12 defaults: width 48, modes 24, 4 layers. proj_hidden is the projection MLP's
    hidden width (not fixed by CLAUDE.md).
    """

    def __init__(self, in_ch: int, out_ch: int, width: int = 48, modes: int = 24,
                 n_layers: int = 4, proj_hidden: int = 128) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(n_layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(n_layers))
        self.proj = nn.Sequential(nn.Conv2d(width, proj_hidden, 1), nn.GELU(),
                                  nn.Conv2d(proj_hidden, out_ch, 1))  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        last = len(self.spectral) - 1
        for i, (k, w) in enumerate(zip(self.spectral, self.pointwise, strict=True)):
            x = k(x) + w(x)
            if i < last:
                x = F.gelu(x)
        return self.proj(x)
