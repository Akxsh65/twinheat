"""U-Net baseline (M2.3; CLAUDE.md §12: 4 levels, 32 base channels, GroupNorm).

The local baseline for sharp copper edges. Four resolution levels with 32, 64, 128, 256 channels;
each level is two (3×3 conv, GroupNorm, GELU) blocks; 2×2 max-pool down, 2×2 transposed conv up,
skip connections by concatenation; a 1×1 output conv. N must be divisible by 8 (three halvings).
GN_GROUPS = 8 divides every channel count; it is not fixed by CLAUDE.md (UNSURE).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["UNet"]

GN_GROUPS = 8  # UNSURE (see module docstring)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(GN_GROUPS, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(GN_GROUPS, cout), nn.GELU())


class UNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 32, levels: int = 4) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        widths = [base * 2**k for k in range(levels)]
        self.levels = levels
        self.down = nn.ModuleList(_block(c_in, c) for c_in, c in zip([in_ch, *widths[:-1]], widths,
                                                                    strict=True))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList(nn.ConvTranspose2d(widths[k + 1], widths[k], 2, stride=2)
                                for k in reversed(range(levels - 1)))
        self.dec = nn.ModuleList(_block(2 * widths[k], widths[k])
                                 for k in reversed(range(levels - 1)))
        self.out = nn.Conv2d(widths[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 2 ** (self.levels - 1)
        if x.shape[-1] % m or x.shape[-2] % m:
            raise ValueError(f"U-Net with {self.levels} levels needs N divisible by {m}, "
                             f"got {tuple(x.shape[-2:])}")
        skips = []
        for k, block in enumerate(self.down):
            x = block(x)
            if k < self.levels - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec in zip(self.up, self.dec, strict=True):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.out(x)
