"""Differentiable design inputs for P3's gradient (M3.3; CLAUDE.md §7.4, §7.5, §11.2).

DesignInputs(spec, n)(x) maps design vectors x [B, 36] (torch, in [0, 1]) to raw model inputs
[B, 13, n, n] (torch), the same channels as models/inputs.build(data/raster.rasterise(
data/sampler.design_board(x))), but differentiable in x:

  only x-dependent channels are recomputed:
    0     log10 G_1 = log10(a·φ_1 + c), φ_1 = 1 − Π_k (1 − f_k) over the six L1 rectangles
    4     q = Σ_i P_i f_i / (w_i h_i)                       (W/m², chips do not overlap)
    5–7   log(1 + ν), ν = background + Σ_i d_i f_i          (vias/mm², via arrays = chip footprints)
  every other channel is taken once from inputs.build of a reference design (they do not depend
  on x: the stack, L2–L4, convection and edges are fixed by config/design_task.yaml, D2).

f is the exact area fraction of each cell covered by an axis-aligned rectangle,
overlap_x ⊗ overlap_y, which is piecewise linear in the rectangle's edges, so autograd gives the
true derivative almost everywhere. The coefficients a = k_Cu t_Cu,1 and c (the FR-4 term) come
from solvers/fvm.layer_conductances on the reference raster, not from a second copy of §7.2.

Differences from the real rasteriser, by design (this is a search direction, not a prediction):
  · exact area fractions instead of 4 × 4 supersampling (differs on edge cells only),
  · no rounding of positions to the geometry grid,
  · the L1 union 1 − Π(1 − f) is exact except in cells cut by the edges of two overlapping
    rectangles,
  · infeasible x (overlapping chips, L1 coverage out of range) are not refused here; P3 checks
    feasibility with design_board before it proposes anything.
Every design P3 proposes is rasterised and predicted by the exact path (gate/proposers.py), and
tests/test_proposers.py compares the two channel by channel.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from data import sampler
from data.raster import rasterise
from models import inputs as inp
from solvers.fvm import layer_conductances

__all__ = ["DesignInputs"]

MM2_TO_M2 = 1e-6
_DEPENDENT = (0, inp.Q_CHANNEL, 5, 6, 7)


def _reference_design() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(0))
    for _ in range(10_000):
        x = rng.random(int(sampler._S2.design_space.dimension.value))
        try:
            sampler.design_board(x)
            return x
        except sampler.InfeasibleDesignError:
            continue
    raise RuntimeError("no feasible reference design in 10,000 draws")


class DesignInputs(torch.nn.Module):
    def __init__(self, spec, n: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if not (isinstance(n, int) and n >= 2):
            raise ValueError(f"n must be an int >= 2, got {n!r}")
        task = sampler._design_task()
        self.n, self.dtype = n, dtype
        self.L = float(sampler.L_MM)
        self.margin = float(sampler._S1.chip_edge_margin_min.value)
        self.n_rects = int(sampler._S2.design_space.copper_rect_count.value)
        self.chips = [(float(c["w"]), float(c["h"]), float(c["P"])) for c in task["chips"]]
        self.rect_lo, self.rect_hi = (float(v) for v in task["mapping"]["rect_side"])
        self.via_lo, self.via_hi = (float(v) for v in task["mapping"]["via_density"])
        self.bg_via = float(task["background_via_density_per_mm2"])
        self.dim = 2 * len(self.chips) + 4 * self.n_rects + len(self.chips)

        rec = sampler.design_board(_reference_design())
        ras = rasterise(rec, spec, n)
        base = inp.build(ras, spec).astype(np.float64)
        phi0, phi1 = ras.phi.copy(), ras.phi.copy()
        phi0[0], phi1[0] = 0.0, 1.0
        g0 = layer_conductances(spec, dataclasses.replace(ras, phi=phi0))[0]
        g1 = layer_conductances(spec, dataclasses.replace(ras, phi=phi1))[0]
        c, a = float(g0.mean()), float((g1 - g0).mean())
        if not (np.allclose(g0, c, rtol=1e-12) and np.allclose(g1 - g0, a, rtol=1e-12)):
            raise ValueError("L1 sheet conductance is not a·φ + c with constant a, c")
        self.a, self.c = a, c
        self.register_buffer("base", torch.tensor(base, dtype=dtype))
        edges = torch.arange(n + 1, dtype=dtype) * (self.L / n)
        self.register_buffer("lo", edges[:-1])
        self.register_buffer("hi", edges[1:])

    def _overlap(self, a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
        """[..., n] fraction of each cell interval covered by [a0, a1] (mm)."""
        w = self.L / self.n
        top = torch.minimum(a1[..., None], self.hi)
        bot = torch.maximum(a0[..., None], self.lo)
        return torch.clamp(top - bot, min=0.0) / w

    def _frac(self, x0, y0, w, h) -> torch.Tensor:
        """[B, K, n, n] area fractions (rows = y from the south edge, columns = x)."""
        return self._overlap(y0, y0 + h)[..., :, None] * self._overlap(x0, x0 + w)[..., None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must be [B, {self.dim}], got {tuple(x.shape)}")
        x = x.to(self.dtype)
        B, L, m = x.shape[0], self.L, self.margin
        nc = len(self.chips)
        cw = torch.tensor([c[0] for c in self.chips], dtype=self.dtype, device=x.device)
        ch = torch.tensor([c[1] for c in self.chips], dtype=self.dtype, device=x.device)
        cp = torch.tensor([c[2] for c in self.chips], dtype=self.dtype, device=x.device)
        cx0 = m + x[:, 0:2 * nc:2] * (L - 2 * m - cw)
        cy0 = m + x[:, 1:2 * nc:2] * (L - 2 * m - ch)
        f_chip = self._frac(cx0, cy0, cw.expand(B, nc), ch.expand(B, nc))  # [B, 4, n, n]

        r = x[:, 2 * nc:2 * nc + 4 * self.n_rects].reshape(B, self.n_rects, 4)
        rw = self.rect_lo + r[..., 2] * (self.rect_hi - self.rect_lo)
        rh = self.rect_lo + r[..., 3] * (self.rect_hi - self.rect_lo)
        f_rect = self._frac(r[..., 0] * (L - rw), r[..., 1] * (L - rh), rw, rh)
        phi1 = 1.0 - torch.prod(1.0 - f_rect, dim=1)

        dens = self.via_lo + x[:, 2 * nc + 4 * self.n_rects:] * (self.via_hi - self.via_lo)
        nu = self.bg_via + (dens[:, :, None, None] * f_chip).sum(dim=1)  # vias/mm²
        q = (cp / (cw * ch * MM2_TO_M2))[None, :, None, None] * f_chip

        out = self.base.expand(B, -1, -1, -1).clone()
        out[:, 0] = torch.log10(self.a * phi1 + self.c)
        out[:, inp.Q_CHANNEL] = q.sum(dim=1)
        lognu = torch.log1p(nu)
        out[:, 5], out[:, 6], out[:, 7] = lognu, lognu, lognu
        return out
