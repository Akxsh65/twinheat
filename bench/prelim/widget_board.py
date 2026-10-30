"""The 5 Oct explainer widget's board and emulated prediction in Python (bench/prelim/README.md).

Ported line by line from explainer_cert_widget.js so N's solver and certify/ can be checked against
the surviving 5 Oct artifact. This is the widget's board and its "noisier" error model, NOT a
spec_v1 board and NOT the lost stop.js / cert.js models. Arrays are [y, x] (index y·N + x, as in
the widget). The widget's edges [left, top, right, bottom] map to W, N, E, S; only the left edge
is heat-sunk.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

__all__ = ["CHIPS", "EDGES", "N", "board_fields", "emulated_prediction", "xorshift32"]

N = 64
L = 0.1  # m, board side (widget TH.L)
K_FR4, T_FR4, K_CU, OZ_T, CU_OZ, H_BASE = 0.9, 1.6e-3, 390.0, 17.5e-6, 2, 24.0  # widget constants
# chips: centre x, centre y, w, h (cells), P (W)
CHIPS = ((26, 28, 10, 10, 1.2), (50, 18, 7, 7, 0.8), (46, 50, 5, 4, 0.5))
EDGES = ("W",)


def board_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """(phi, G W/K, H W/(m²·K), q W/m², dx m) of TH.defaultBoard(), for fvm.assemble_plate."""
    dx = L / N
    y, x = np.mgrid[0:N, 0:N]
    phi = (((x < 22) & (y > 34))                          # pour near the heat-sunk edge
           | ((x >= 18) & (x < 34) & (y >= 20) & (y < 36))   # spreader under the CPU
           | ((y >= 27) & (y < 29) & (x >= 34) & (x < 50))   # trace to the PMIC
           | ((x >= 46) & (x < 48) & (y >= 12) & (y < 28))
           | ((y >= 46) & (y < 48) & (x >= 22) & (x < 56)))  # fmt: skip
    phi = phi.astype(np.float64)
    G = K_FR4 * T_FR4 + K_CU * OZ_T * CU_OZ * phi
    H = np.full((N, N), H_BASE)
    q = np.zeros((N, N))
    for cx, cy, w, h, P in CHIPS:  # fractional cell overlap, as Board.assemble()
        qd = P / ((w * dx) * (h * dx))
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        ox = np.clip(np.minimum(np.arange(N) + 1, x1) - np.maximum(np.arange(N), x0), 0, None)
        oy = np.clip(np.minimum(np.arange(N) + 1, y1) - np.maximum(np.arange(N), y0), 0, None)
        q += qd * oy[:, None] * ox[None, :]
    return phi, G, H, q, dx


def xorshift32(seed: int) -> Callable[[], float]:
    """TH.rng: xorshift32 in [0, 1), bit-exact with the JS (uint32 arithmetic)."""
    s = (seed & 0xFFFFFFFF) or 1

    def r() -> float:
        nonlocal s
        s = (s ^ (s << 13)) & 0xFFFFFFFF
        s = (s ^ (s >> 17)) & 0xFFFFFFFF
        s = (s ^ (s << 5)) & 0xFFFFFFFF
        return s / 4294967296

    return r


def _gauss(r: Callable[[], float]) -> float:
    u = 0.0
    while u == 0:
        u = r()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * r())


def emulated_prediction(
    theta: np.ndarray, phi: np.ndarray, lvl: float, seed: int = 7
) -> np.ndarray:
    """The widget's μ = θ + lvl·peak·(smooth + 0.4·g·ringing + 0.08·noise), flat [N²], seed 7."""
    t, ph = np.asarray(theta, dtype=np.float64).ravel(), np.asarray(phi).ravel()
    peak, r, mu = float(t.max()), xorshift32(seed), np.empty(N * N)
    for i in range(N * N):  # sequential: the noise stream is drawn in cell order, as in the JS
        X, Y = i % N, i // N
        g = 0.0
        if 0 < X < N - 1 and 0 < Y < N - 1:
            g = abs(ph[i + 1] - ph[i - 1]) + abs(ph[i + N] - ph[i - N])
        mu[i] = t[i] + lvl * peak * (math.sin(X / 7 + 1.3) * math.cos(Y / 9 + 1)
                                     + 0.4 * g * math.sin(1.7 * X + 0.9 * Y) + 0.08 * _gauss(r))
    return mu
