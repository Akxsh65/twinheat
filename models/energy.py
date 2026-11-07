"""Energy-form physics loss, matrix-free (M2.3 ablation; CLAUDE.md §12, §18.4).

    E(θ) = ½ θᵀ A θ − bᵀ θ            per board, A and b from solvers/fvm.coefficients()

A is SPD, so E is minimised exactly at the solver's θ (Aθ = b); its Hessian is A (condition
κ₂(A)), against 2A² for the squared residual (§18.4). θ in K, A in W/K, b in W, so E is in W·K.
stencil_apply(c, θ) is A θ from the coefficient arrays [B, 4, N, N] (Ge, Gw, Gn, Gs, gup, gdn,
diag); entries pointing off the grid are 0 in the coefficients, and shifted-in values are 0.
energy_loss divides E by Σ|b| · σ_θ (W·K, σ_θ the train θ std from the normaliser) so it is
dimensionless and of order one per board; the scaling changes no minimiser (UNSURE: CLAUDE.md does
not fix a normalisation). Used only as an ablation row (§12), never as a claim (§1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["COEFF_KEYS", "energy", "energy_loss", "stencil_apply"]

COEFF_KEYS = ("Ge", "Gw", "Gn", "Gs", "gup", "gdn", "diag", "rhs")


def stencil_apply(c: dict[str, torch.Tensor], theta: torch.Tensor) -> torch.Tensor:
    """A θ for θ [B, 4, N, N] (rule 7: [layer, y, x]); east/north = increasing x/y index."""
    t = theta
    east = F.pad(t[..., :, 1:], (0, 1))  # θ_{x+1}
    west = F.pad(t[..., :, :-1], (1, 0))  # θ_{x-1}
    north = F.pad(t[..., 1:, :], (0, 0, 0, 1))  # θ_{y+1}
    south = F.pad(t[..., :-1, :], (0, 0, 1, 0))  # θ_{y-1}
    up = F.pad(t[:, 1:], (0, 0, 0, 0, 0, 1))  # θ_{l+1} (towards L4)
    down = F.pad(t[:, :-1], (0, 0, 0, 0, 1, 0))  # θ_{l-1}
    return (c["diag"] * t - c["Ge"] * east - c["Gw"] * west - c["Gn"] * north - c["Gs"] * south
            - c["gup"] * up - c["gdn"] * down)


def energy(c: dict[str, torch.Tensor], theta: torch.Tensor) -> torch.Tensor:
    """E(θ) = ½ θᵀAθ − bᵀθ per board, shape [B], in W·K; float16/bfloat16 input (AMP) is
    promoted to float32, float32 and float64 keep their precision."""
    t = theta.float() if theta.dtype in (torch.float16, torch.bfloat16) else theta
    return (0.5 * (t * stencil_apply(c, t)) - c["rhs"] * t).flatten(1).sum(dim=1)


def energy_loss(c: dict[str, torch.Tensor], theta: torch.Tensor,
                theta_scale: float) -> torch.Tensor:
    """E / (Σ|b| · σ_θ) per board, dimensionless (see module docstring)."""
    scale = c["rhs"].abs().flatten(1).sum(dim=1) * theta_scale
    if not bool((scale > 0).all()):
        raise ValueError("energy_loss: a board has no heat input (b = 0)")
    return energy(c, theta) / scale
