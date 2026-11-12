"""Label-free scale s_t (CLAUDE.md §8.1 step 2, §9.2).

    r      = (A μ − b) / diag A                       per cell, kelvin
    hot    = the m hottest cells of μ over all layers  (ties broken by index, stable)
    r_hot  = mean |r| over hot,  sd_hot = mean σ over hot
    s      = r_hot + λ · sd_hot + ε

A μ is applied with the 7-point layered stencil of solvers/fvm.coefficients (W/K; b in W), so
r is in K. Nothing here uses a label: μ, σ, the stencil and b are all known before any solve.
λ, ε and m are arguments (D4 freezes them); m = 64 is CLAUDE.md §8.1's default and is shared by
all gates.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["apply_stencil", "hot_residual_raw", "label_free_scale"]

_STENCIL = ("Ge", "Gw", "Gn", "Gs", "gup", "gdn", "diag")


def apply_stencil(coeffs: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """(A x) [4, N, N] in W for x in K, A given by solvers/fvm.coefficients (W/K)."""
    x = np.asarray(x, dtype=np.float64)
    shape = coeffs["diag"].shape
    if x.shape != shape:
        raise ValueError(f"field shape {x.shape} does not match the stencil shape {shape}")
    y = coeffs["diag"] * x
    y[:, :, :-1] -= coeffs["Ge"][:, :, :-1] * x[:, :, 1:]
    y[:, :, 1:] -= coeffs["Gw"][:, :, 1:] * x[:, :, :-1]
    y[:, :-1, :] -= coeffs["Gn"][:, :-1, :] * x[:, 1:, :]
    y[:, 1:, :] -= coeffs["Gs"][:, 1:, :] * x[:, :-1, :]
    y[:-1] -= coeffs["gup"][:-1] * x[1:]
    y[1:] -= coeffs["gdn"][1:] * x[:-1]
    return y


def label_free_scale(
    mu: np.ndarray, sd: np.ndarray, coeffs: Mapping[str, np.ndarray], b: np.ndarray, *,
    m: int = 64, lam: float, eps: float,
) -> tuple[float, float, float]:
    """(s, r_hot, sd_hot), all in K, for one board. mu, sd [4, N, N] in K; b [4, N, N] in W."""
    missing = [k for k in _STENCIL if k not in coeffs]
    if missing:
        raise ValueError(f"coeffs is missing {missing} (use solvers/fvm.coefficients)")
    shape = coeffs["diag"].shape
    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if mu.shape != shape or sd.shape != shape or b.size != mu.size:
        raise ValueError(f"mu {mu.shape}, sd {sd.shape}, b ({b.size}) must match the stencil "
                         f"shape {shape}")
    if not (isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= mu.size):
        raise ValueError(f"m must be an int in [1, {mu.size}], got {m!r}")
    if not (np.isfinite(lam) and lam >= 0):
        raise ValueError(f"lam must be finite and >= 0, got {lam}")
    if not (np.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and > 0 (s > 0 is required), got {eps}")
    if not (np.isfinite(mu).all() and np.isfinite(sd).all()):
        raise ValueError("mu and sd must be finite")
    if sd.min() < 0:
        raise ValueError(f"sd must be >= 0, min is {sd.min()}")
    diag = coeffs["diag"]
    if diag.min() <= 0:
        raise ValueError(f"diag A must be > 0 (M-matrix, B1), min is {diag.min()}")
    r = (apply_stencil(coeffs, mu).reshape(-1) - b) / diag.reshape(-1)  # K
    hot = np.argsort(-mu.reshape(-1), kind="stable")[:m]
    r_hot = float(np.abs(r[hot]).mean())
    sd_hot = float(sd.reshape(-1)[hot].mean())
    return r_hot + lam * sd_hot + eps, r_hot, sd_hot


def hot_residual_raw(
    mu: np.ndarray, coeffs: Mapping[str, np.ndarray], b: np.ndarray, *, m: int,
) -> float:
    """DeepOHeat-v2's r_hot (Definition 5, docs/refs/deepoheat_v2_alg1.txt): mean |A μ − b| over
    the m hottest cells of μ, in W, with no diag A scaling. Only the paper-faithful G1 variant
    uses it (D13); every gate's main signal is label_free_scale's r_hot in K."""
    shape = coeffs["diag"].shape
    mu = np.asarray(mu, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if mu.shape != shape or b.size != mu.size:
        raise ValueError(f"mu {mu.shape} and b ({b.size}) must match the stencil shape {shape}")
    if not (isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= mu.size):
        raise ValueError(f"m must be an int in [1, {mu.size}], got {m!r}")
    if not np.isfinite(mu).all():
        raise ValueError("mu must be finite")
    r = apply_stencil(coeffs, mu).reshape(-1) - b
    hot = np.argsort(-mu.reshape(-1), kind="stable")[:m]
    return float(np.abs(r[hot]).mean())
