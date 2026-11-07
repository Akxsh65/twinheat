"""Model architectures by name, with the CLAUDE.md §12 configurations (M2.2, M2.3).

build(config) makes a model from the `model_config` a checkpoint stores; a config without "arch"
is an FNO (checkpoints written before the zoo). Every model maps [B, 13, N, N] -> [B, 4, N, N].
"""

from __future__ import annotations

from typing import Any

from torch import nn

from models.fno import FNO2d
from models.fno_local import FNOLocal2d
from models.unet import UNet

__all__ = ["ARCHS", "build", "config"]

IO = {"in_ch": 13, "out_ch": 4}  # §7.5
ARCHS: dict[str, tuple[type[nn.Module], dict[str, Any]]] = {
    "fno": (FNO2d, {"width": 48, "modes": 24, "n_layers": 4}),  # §12
    "fno_local": (FNOLocal2d, {"width": 48, "modes": 24, "n_layers": 4}),  # §12 ablation
    "unet": (UNet, {"base": 32, "levels": 4}),  # §12
}


def config(arch: str) -> dict[str, Any]:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; have {sorted(ARCHS)}")
    return {"arch": arch, **IO, **ARCHS[arch][1]}


def build(cfg: dict[str, Any]) -> nn.Module:
    cfg = dict(cfg)
    arch = cfg.pop("arch", "fno")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; have {sorted(ARCHS)}")
    return ARCHS[arch][0](**cfg)
