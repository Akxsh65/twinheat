"""Model stub for the server until the FNO ensemble lands (plan "Stubs" table: solver field + 2 %
smooth noise, fake σ; built by M, used by G and N, replaced in week 5).

predict(spec, raster, dirichlet_edges, record) -> (mu, sd), each float32 [4, n, n] in K
(temperature rise, CLAUDE.md §0 rule 7 order [layer, y, x]):

    θ  = the exact discrete solution (solvers/fvm.assemble_board + solve_direct)
    mu = θ · (1 + NOISE_REL · ξ)     ξ a smooth random field, zero mean and unit std over [4, n, n]
    sd = NOISE_REL · |θ|             a fake ensemble spread with the same 2 % scale

ξ is white noise low-pass filtered by a Gaussian of standard deviation CORR_MM (in mm on the
board) in Fourier space, independently per layer, then standardised over the whole array. Its
seed is the SHA-256 of the record without its "id", so the same design always gets the same
"prediction" (a drag that returns to a position shows the same field) and the message id does not
matter.

This is a test double, not a model: the field error it fakes is not a property of any surrogate.
CORR_MM is a stub choice (not in CLAUDE.md or the spec YAMLs); M owns this file and may change
it. Nothing here may appear in a reported result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from typing import Any

import numpy as np

from solvers.fvm import N_LAYERS, assemble_board, solve_direct

__all__ = ["CORR_MM", "NOISE_REL", "predict"]

NOISE_REL = 0.02  # "2 % smooth noise" (plan, Stubs table)
CORR_MM = 10.0  # STUB CHOICE: correlation length of the fake error, mm on the 100 mm board


def _seed(record: dict[str, Any]) -> int:
    body = {k: v for k, v in record.items() if k != "id"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _smooth_field(n: int, board_mm: float, rng: np.random.Generator) -> np.ndarray:
    """[4, n, n] Gaussian-filtered white noise, standardised to zero mean and unit std."""
    white = rng.standard_normal((N_LAYERS, n, n))
    k = np.fft.fftfreq(n, d=board_mm / n)  # cycles per mm
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    filt = np.exp(-2.0 * (np.pi * CORR_MM) ** 2 * k2)  # Fourier transform of a Gaussian kernel
    xi = np.fft.ifft2(np.fft.fft2(white, axes=(1, 2)) * filt, axes=(1, 2)).real
    xi -= xi.mean()
    std = xi.std()
    if not std > 0:
        raise FloatingPointError(f"smooth noise field has std {std} at n = {n}")
    return xi / std


def predict(
    spec, raster: Any, dirichlet_edges: Collection[str], record: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sd) in K, float32 [4, n, n], for a rasterised board (data/raster.rasterise)."""
    n = int(raster.N)
    A, b = assemble_board(spec, raster, dirichlet_edges)
    theta = solve_direct(A, b).reshape(N_LAYERS, n, n)
    board_mm = spec.geometry.L.value * 1e3
    xi = _smooth_field(n, board_mm, np.random.Generator(np.random.PCG64(_seed(record))))
    mu = theta * (1.0 + NOISE_REL * xi)
    sd = NOISE_REL * np.abs(theta)
    return mu.astype(np.float32), sd.astype(np.float32)
