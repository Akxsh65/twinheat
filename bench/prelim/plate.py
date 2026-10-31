"""Port of the parts of bench/prelim/core.js that the §8.6 tables use (N1.3).

core.js is the 5 Oct 64² development plate: a fin equation, cell-centred FV, harmonic-mean faces,
2 G_P at heat-sunk faces. Its board is a toy model, NOT config/spec_v1.yaml: k_FR4 = 0.9, one
combined face loss hBase = 24 W/(m²·K), copper sheet conductance 390 · 17.5 µm · cuOz W/K (cuOz
counts 17.5 µm layers), chips placed in cell units. The README there calls these toy results;
§8.6 reports them as such. These constants are copied from core.js, not from the spec.

Every per-cell expression keeps core.js's operation order, so assemble() reproduces core.js's
arrays bit for bit (tests/test_certificate_cpu.py compares them when Node is installed).
Index i = y·N + x (y = 0 bottom row), as in core.js and solvers/fvm.py.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

__all__ = ["Board", "PlateSystem", "colours", "default_board", "gauss", "rng"]

# core.js constants (toy model, see the module docstring)
L = 0.1  # board side, m
T_FR4 = 1.6e-3  # m
K_FR4 = 0.9  # W/(m·K)
K_CU = 390  # W/(m·K)
CU_LAYER = 17.5e-6  # m per unit of cuOz
G_LAM = K_FR4 * T_FR4  # laminate sheet conductance, W/K
_U32 = 0xFFFFFFFF


def rng(seed: int) -> Callable[[], float]:
    """core.js rng: 32-bit xorshift (13, 17, 5), returns s / 2³² in [0, 1)."""
    s = (seed & _U32) or 1

    def next_() -> float:
        nonlocal s
        s ^= (s << 13) & _U32
        s ^= s >> 17
        s ^= (s << 5) & _U32
        return s / 4294967296
    return next_


def gauss(r: Callable[[], float]) -> float:
    """core.js gauss: Box–Muller from two draws of r (u redrawn while 0)."""
    u = 0.0
    while u == 0:
        u = r()
    v = r()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


@dataclass
class Chip:
    x: float  # centre, cells
    y: float
    w: float  # size, cells
    h: float
    P: float  # W


@dataclass
class PlateSystem:
    """core.js assemble() output plus the CSR matrix (W/K) and b (W)."""
    N: int
    G: np.ndarray
    b: np.ndarray
    Hd: np.ndarray
    gE: np.ndarray
    gN: np.ndarray
    dterm: np.ndarray
    diag: np.ndarray
    A: sp.csr_matrix


@dataclass
class Board:
    """core.js Board. edges = [left, top, right, bottom] heat-sunk (top is y = N − 1)."""
    N: int = 64
    phi: np.ndarray = field(default_factory=lambda: np.zeros(64 * 64))
    chips: list[Chip] = field(default_factory=list)
    edges: list[bool] = field(default_factory=lambda: [True, False, False, False])
    hBase: float = 24
    cuOz: float = 2

    def assemble(self) -> PlateSystem:
        N, n = self.N, self.N * self.N
        dx = L / N
        gcu = K_CU * CU_LAYER * self.cuOz
        phi = np.asarray(self.phi, dtype=np.float32).astype(np.float64)  # Float32Array
        G = G_LAM + gcu * phi
        Hd = np.full(n, self.hBase * dx * dx)
        q = np.zeros(n)
        for c in self.chips:
            area = (c.w * L / N) * (c.h * L / N)
            qd = c.P / area
            x0, x1 = c.x - c.w / 2, c.x + c.w / 2
            y0, y1 = c.y - c.h / 2, c.y + c.h / 2
            for y in range(max(0, math.floor(y0)), min(N, math.ceil(y1))):
                for x in range(max(0, math.floor(x0)), min(N, math.ceil(x1))):
                    ox = min(x + 1, x1) - max(x, x0)
                    oy = min(y + 1, y1) - max(y, y0)
                    if ox > 0 and oy > 0:
                        q[y * N + x] += qd * ox * oy * dx * dx
        Gy = G.reshape(N, N)
        gE = np.zeros((N, N))
        gN = np.zeros((N, N))
        a, b = Gy[:, :-1], Gy[:, 1:]
        gE[:, :-1] = 2 * a * b / (a + b)
        a, b = Gy[:-1, :], Gy[1:, :]
        gN[:-1, :] = 2 * a * b / (a + b)
        d = np.zeros((N, N))  # added in core.js's order: left, top, right, bottom
        if self.edges[0]:
            d[:, 0] += 2 * Gy[:, 0]
        if self.edges[1]:
            d[N - 1, :] += 2 * Gy[N - 1, :]
        if self.edges[2]:
            d[:, N - 1] += 2 * Gy[:, N - 1]
        if self.edges[3]:
            d[0, :] += 2 * Gy[0, :]
        diag = Hd.reshape(N, N) + d + gE + gN
        diag[:, 1:] += gE[:, :-1]
        diag[1:, :] += gN[:-1, :]

        k = np.arange(n).reshape(N, N)
        rows = [k.ravel(), k[:, :-1].ravel(), k[:, 1:].ravel(), k[:-1, :].ravel(),
                k[1:, :].ravel()]
        cols = [k.ravel(), k[:, 1:].ravel(), k[:, :-1].ravel(), k[1:, :].ravel(),
                k[:-1, :].ravel()]
        vals = [diag.ravel(), -gE[:, :-1].ravel(), -gE[:, :-1].ravel(), -gN[:-1, :].ravel(),
                -gN[:-1, :].ravel()]
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
        return PlateSystem(N=N, G=G, b=q, Hd=Hd, gE=gE.ravel(), gN=gN.ravel(), dterm=d.ravel(),
                           diag=diag.ravel(), A=A)


def default_board() -> Board:
    """core.js defaultBoard(): pours, CPU spreader and traces; CPU, PMIC, LDO; left edge sunk."""
    N = 64
    y, x = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    p = ((x < 22) & (y > 34)) | ((x >= 18) & (x < 34) & (y >= 20) & (y < 36)) \
        | ((y >= 27) & (y < 29) & (x >= 34) & (x < 50)) | ((x >= 46) & (x < 48) & (y >= 12)
                                                          & (y < 28)) \
        | ((y >= 46) & (y < 48) & (x >= 22) & (x < 56))
    return Board(N=N, phi=p.astype(np.float64).ravel(), chips=[
        Chip(x=26, y=28, w=10, h=10, P=1.2),  # U1 CPU
        Chip(x=50, y=18, w=7, h=7, P=0.8),  # U2 PMIC
        Chip(x=46, y=50, w=5, h=4, P=0.5),  # U3 LDO
    ])


def colours(N: int) -> np.ndarray:
    """core.js rbSweep colouring: colour (x + y) mod 2, colour 0 swept first."""
    y, x = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return ((x + y) % 2).ravel()
