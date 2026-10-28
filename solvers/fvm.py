"""Cell-centred finite-volume solver for the single plate (CLAUDE.md §6.2–§6.4).

Solves −∇·(G ∇θ) + H θ = q on the square plate, θ = temperature rise in K. For cell P the
equation is, in watts (§6.4):

    Σ_f G_f (θ_P − θ_Nf) + H_P Δx² θ_P = q_P Δx²

with harmonic-mean face conductances G_f = 2 G_P G_N / (G_P + G_N) (W/K; the face width Δx
cancels the centre distance Δx). A Dirichlet face (θ = 0 on a heat-sunk edge, half a cell away)
adds 2 G_P to the diagonal; a Neumann face adds nothing. Fields are arrays [y, x] (y index 0 =
south edge, x index 0 = west edge, CLAUDE.md §0 rule 7) and the unknown index is k = y·N + x.

The assembled A is symmetric, a Z-matrix with weak row dominance (strict where H_P > 0 or at a
Dirichlet face) and a connected 5-point graph, hence a nonsingular M-matrix (invariant B1) as
long as H > 0 somewhere or an edge is heat-sunk. assemble_plate refuses inputs that break this.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["EDGES", "assemble_plate", "energy_balance", "solve_direct"]

EDGES = ("W", "E", "S", "N")
# Relative residual a direct solve must reach: the data-generation tolerance, CLAUDE.md §6.4.
DIRECT_RTOL = 1e-10


def _check_inputs(
    G: np.ndarray, H: np.ndarray, q: np.ndarray, dx: float, dirichlet_edges: Collection[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, frozenset[str]]:
    G, H, q = (np.asarray(a, dtype=np.float64) for a in (G, H, q))
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be a square [N, N] array, got shape {G.shape}")
    for name, a in (("H", H), ("q", q)):
        if a.shape != G.shape:
            raise ValueError(f"{name} must have the same shape as G {G.shape}, got {a.shape}")
    for name, a in (("G", G), ("H", H), ("q", q)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite; {int((~np.isfinite(a)).sum())} cells are not")
    if not G.min() > 0:
        raise ValueError(f"G must be > 0 in every cell (W/K), min is {G.min()}")
    if not H.min() >= 0:
        raise ValueError(f"H must be >= 0 in every cell (W/(m²·K)), min is {H.min()}")
    if not (np.isfinite(dx) and dx > 0):
        raise ValueError(f"dx must be a positive cell size in m, got {dx}")
    if isinstance(dirichlet_edges, str):
        raise ValueError(
            f"dirichlet_edges must be a collection of edge names, got {dirichlet_edges!r}")
    edges = frozenset(dirichlet_edges)
    if not edges <= set(EDGES):
        raise ValueError(f"dirichlet_edges must be a subset of {EDGES}, got {sorted(edges)}")
    if not edges and H.max() == 0:
        raise ValueError("H = 0 everywhere and no Dirichlet edge: the system is singular "
                         "(CLAUDE.md §6.2 needs H > 0 or a heat-sunk edge)")
    return G, H, q, edges


def _edge_mask(n: int, edges: frozenset[str]) -> np.ndarray:
    """Number of Dirichlet faces of each cell, [y, x] (0, 1 or 2 at corners)."""
    m = np.zeros((n, n))
    if "W" in edges:
        m[:, 0] += 1
    if "E" in edges:
        m[:, -1] += 1
    if "S" in edges:
        m[0, :] += 1
    if "N" in edges:
        m[-1, :] += 1
    return m


def assemble_plate(
    G: np.ndarray,
    H: np.ndarray,
    q: np.ndarray,
    dx: float,
    dirichlet_edges: Collection[str],
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Assemble A θ = b for the plate.

    G: sheet conductance [N, N], W/K. H: h_top + h_bot [N, N], W/(m²·K). q: heat input
    [N, N], W/m². dx: cell size, m. dirichlet_edges: subset of {"W", "E", "S", "N"} held at θ = 0.
    Returns A (CSR, N² × N², W/K) and b (N², W), both float64, unknown index k = y·N + x.
    """
    G, H, q, edges = _check_inputs(G, H, q, dx, dirichlet_edges)
    n = G.shape[0]
    k = np.arange(n * n).reshape(n, n)

    # interior faces: x-faces between (y, x) and (y, x+1); y-faces between (y, x) and (y+1, x)
    gx = 2 * G[:, :-1] * G[:, 1:] / (G[:, :-1] + G[:, 1:])
    gy = 2 * G[:-1, :] * G[1:, :] / (G[:-1, :] + G[1:, :])
    pa = np.concatenate([k[:, :-1].ravel(), k[:-1, :].ravel()])
    pb = np.concatenate([k[:, 1:].ravel(), k[1:, :].ravel()])
    gf = np.concatenate([gx.ravel(), gy.ravel()])

    diag = H * dx * dx + 2 * G * _edge_mask(n, edges)
    diag = diag.ravel() + np.bincount(pa, gf, n * n) + np.bincount(pb, gf, n * n)

    rows = np.concatenate([k.ravel(), pa, pb])
    cols = np.concatenate([k.ravel(), pb, pa])
    vals = np.concatenate([diag, -gf, -gf])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n * n, n * n)).tocsr()
    b = (q * dx * dx).ravel()
    return A, b


def solve_direct(A: sp.spmatrix, b: np.ndarray, rtol: float = DIRECT_RTOL) -> np.ndarray:
    """Solve A θ = b with SciPy's sparse direct solver; θ in K, same index order as b.

    Raises if the result is not finite or its relative residual ‖b − Aθ‖₂ / ‖b‖₂ exceeds rtol.
    """
    b = np.asarray(b, dtype=np.float64)
    if A.shape != (b.size, b.size):
        raise ValueError(f"A is {A.shape} but b has {b.size} entries")
    theta = spla.spsolve(sp.csc_matrix(A), b)
    if not np.all(np.isfinite(theta)):
        raise FloatingPointError("direct solve returned non-finite values (singular A?)")
    bnorm = np.linalg.norm(b)
    res = np.linalg.norm(b - A @ theta) / bnorm if bnorm > 0 else np.linalg.norm(A @ theta)
    if res > rtol:
        raise FloatingPointError(f"direct solve relative residual {res:.2e} exceeds {rtol:.0e}")
    return theta


def energy_balance(
    theta: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    q: np.ndarray,
    dx: float,
    dirichlet_edges: Collection[str],
) -> float:
    """Relative global energy-balance error |P_in − P_out| / P_in (CLAUDE.md §6.2).

    P_in = Σ q Δx² (W); P_out = Σ H Δx² θ (convection) + Σ over Dirichlet faces 2 G_P θ_P (edge
    heat sink). theta is [N, N] in K; the other arguments as for assemble_plate.
    """
    G, H, q, edges = _check_inputs(G, H, q, dx, dirichlet_edges)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != G.shape:
        raise ValueError(f"theta must have shape {G.shape}, got {theta.shape}")
    p_in = float((q * dx * dx).sum())
    if p_in <= 0:
        raise ValueError(f"energy balance needs a positive total heat input, got {p_in} W")
    p_conv = (H * dx * dx * theta).sum()
    p_edge = (2 * G * _edge_mask(G.shape[0], edges) * theta).sum()
    p_out = float(p_conv + p_edge)
    return abs(p_in - p_out) / p_in

