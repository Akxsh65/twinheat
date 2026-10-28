"""Envelope rule (CLAUDE.md §6.5, §7.4): cap the peak rise at θ_max by scaling every chip power.

    apply_envelope(record, theta) -> (record', theta', scaled)

θ is linear in q for fixed G, H and edges (§6.2, invariant B4), so if max θ > θ_max every P_i is
multiplied by s = θ_max / max θ and θ by the same s: no re-solve. θ_max comes from spec_v2
envelope.theta_max (100 K). data/generate.py records `scaled` and the running rescale fraction
(above 20 % the power range is lowered instead, §6.5).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from contracts.validate import validate
from twinheat import spec as specmod

__all__ = ["THETA_MAX", "apply_envelope"]

ROOT = Path(__file__).resolve().parents[1]
THETA_MAX = specmod.load(ROOT / "config" / "spec_v2.yaml").envelope.theta_max.value  # K


def apply_envelope(
    record: dict[str, Any], theta: np.ndarray, max_rise: float = THETA_MAX
) -> tuple[dict[str, Any], np.ndarray, bool]:
    """(record', θ', scaled) with max θ' <= max_rise.

    record is a board_record and theta its solved rise (K, any shape, e.g. [4, N, N]). If
    max θ <= max_rise both come back as copies and scaled is False. Raises ValueError for a
    non-finite θ, a negative peak or a non-positive max_rise.
    """
    validate(record, "board_record")
    theta = np.asarray(theta, dtype=np.float64)
    if theta.size == 0 or not np.isfinite(theta).all():
        raise ValueError(f"{record['id']}: theta must be non-empty and finite")
    if not (np.isfinite(max_rise) and max_rise > 0):
        raise ValueError(f"max_rise must be finite and > 0, got {max_rise}")
    peak = float(theta.max())
    if peak < 0:
        raise ValueError(f"{record['id']}: peak rise {peak} K < 0; θ is a rise above ambient")
    out = copy.deepcopy(record)
    if peak <= max_rise:
        return out, theta.copy(), False
    s = max_rise / peak
    for c in out["chips"]:
        c["P"] = c["P"] * s
    return out, theta * s, True
