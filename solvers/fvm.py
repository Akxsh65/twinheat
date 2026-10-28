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

__all__ = [
    "EDGES", "N_LAYERS", "assemble_board", "assemble_plate", "board_energy_balance",
    "coefficients", "energy_balance", "layer_conductances", "solve_direct",
]

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


# ---------------------------------------------------------------------------- 4-layer board
# CLAUDE.md §7.2. Unknowns θ_l(y, x) for layers l = 0..3 (L1..L4), index k = l·N² + y·N + x.
# Cell equation in watts:
#   Σ_in-plane faces G_f (θ_P − θ_N) + Σ_{m=l±1} g_lm Δx² (θ_P − θ_m) + h_l Δx² θ_P = q_l Δx²
# G_l = k_Cu t_Cu,l φ_l + k_FR4,xy ½ (d_{l−1,l} + d_{l,l+1}) with the missing outer dielectric = 0
# (D3, spec.conventions.outer_dielectric_term), g_lm = (k_FR4,z + ν k_Cu A_v) / d_lm, h_1 = h_top,
# h_4 = h_bot, h_2 = h_3 = 0, q on L1 only. Heat-sunk edges hold every layer at θ = 0.
#
# `raster` is any object with these attributes (SI, arrays [.., y, x]); data/raster.py (M1.1)
# must provide them: phi [4,N,N] (1) · q [N,N] W/m² on L1 · nu [3,N,N] vias/m² for the gaps
# L1–L2, L2–L3, L3–L4 · h_top, h_bot [N,N] W/(m²·K) · t_cu [4] m per layer · k_fr4_xy W/(m·K).

N_LAYERS = 4
_RASTER_FIELDS = ("phi", "q", "nu", "h_top", "h_bot", "t_cu", "k_fr4_xy")


def _check_raster(raster: object) -> tuple[int, dict[str, np.ndarray]]:
    f = {name: np.asarray(getattr(raster, name), dtype=np.float64) for name in _RASTER_FIELDS}
    phi = f["phi"]
    if phi.ndim != 3 or phi.shape[0] != N_LAYERS or phi.shape[1] != phi.shape[2]:
        raise ValueError(f"raster.phi must be [4, N, N], got {phi.shape}")
    n = phi.shape[1]
    want = {"q": (n, n), "nu": (3, n, n), "h_top": (n, n), "h_bot": (n, n), "t_cu": (4,),
            "k_fr4_xy": ()}
    for name, shape in want.items():
        if f[name].shape != shape:
            raise ValueError(f"raster.{name} must have shape {shape}, got {f[name].shape}")
    for name, a in f.items():
        if not np.all(np.isfinite(a)):
            raise ValueError(f"raster.{name} must be finite")
    if phi.min() < 0 or phi.max() > 1:
        raise ValueError(f"raster.phi must lie in [0, 1], got [{phi.min()}, {phi.max()}]")
    for name in ("nu", "h_top", "h_bot"):
        if f[name].min() < 0:
            raise ValueError(f"raster.{name} must be >= 0, min is {f[name].min()}")
    if f["t_cu"].min() <= 0 or f["k_fr4_xy"] <= 0:
        raise ValueError(f"raster.t_cu and raster.k_fr4_xy must be > 0, got {f['t_cu']}, "
                         f"{f['k_fr4_xy']}")
    return n, f


def _check_edges(dirichlet_edges: Collection[str]) -> frozenset[str]:
    if isinstance(dirichlet_edges, str) or not set(dirichlet_edges) <= set(EDGES):
        raise ValueError(f"dirichlet_edges must be a subset of {EDGES}, got {dirichlet_edges!r}")
    return frozenset(dirichlet_edges)


def layer_conductances(spec, raster: object) -> np.ndarray:
    """In-plane sheet conductance G_l [4, N, N] in W/K (CLAUDE.md §7.2; D3 outer term = 0).

    Uses the raster's per-board t_cu and k_fr4_xy (the spec's scalar in_plane_conductance uses
    the nominal k_fr4_xy).
    """
    _, f = _check_raster(raster)
    if spec.conventions.outer_dielectric_term != "zero":
        raise ValueError(f"outer_dielectric_term {spec.conventions.outer_dielectric_term!r} "
                         "is not implemented (D3)")
    gaps = np.array([0.0, spec.stack.d12.value, spec.stack.d23.value, spec.stack.d34.value, 0.0])
    half_dielectric = 0.5 * (gaps[:-1] + gaps[1:])  # [4], m
    k_cu = spec.materials.k_cu.value
    return (k_cu * f["t_cu"][:, None, None] * f["phi"]
            + f["k_fr4_xy"] * half_dielectric[:, None, None])


def _gap_conductances(spec, nu: np.ndarray) -> np.ndarray:
    """g_lm [3, N, N] in W/(m²·K) for the gaps L1–L2, L2–L3, L3–L4 (ν in vias/m²)."""
    d = np.array([spec.stack.d12.value, spec.stack.d23.value, spec.stack.d34.value])
    k_z = spec.materials.k_fr4_z.value
    k_cu = spec.materials.k_cu.value
    return (k_z + nu * k_cu * spec.via_area) / d[:, None, None]


def coefficients(spec, raster: object, dirichlet_edges: Collection[str]) -> dict[str, np.ndarray]:
    """Stencil coefficients for the GPU kernels, each [4, N, N] float64, in W/K (rhs in W).

    (A θ)_P = diag_P θ_P − Ge θ_{x+1} − Gw θ_{x−1} − Gn θ_{y+1} − Gs θ_{y−1}
              − gup θ_{l+1} − gdn θ_{l−1}, and rhs = b.
    East/north = increasing x/y index; gup couples layer l to l + 1 (towards L4), gdn to l − 1
    (towards L1), as in CLAUDE.md §9.4. Entries pointing off the grid are 0; a Dirichlet face
    contributes only 2 G_P to diag.
    """
    n, f = _check_raster(raster)
    edges = _check_edges(dirichlet_edges)
    if not edges and f["h_top"].max() == 0 and f["h_bot"].max() == 0:
        raise ValueError("h_top = h_bot = 0 everywhere and no Dirichlet edge: the system is "
                         "singular (CLAUDE.md §7.2 needs h > 0 or a heat-sunk edge)")
    dx = spec.geometry.L.value / n
    G = layer_conductances(spec, raster)
    z = np.zeros((N_LAYERS, n, n))
    Ge, Gw, Gn, Gs, gup, gdn, h, rhs = (z.copy() for _ in range(8))
    gx = 2 * G[:, :, :-1] * G[:, :, 1:] / (G[:, :, :-1] + G[:, :, 1:])
    gy = 2 * G[:, :-1, :] * G[:, 1:, :] / (G[:, :-1, :] + G[:, 1:, :])
    Ge[:, :, :-1] = gx
    Gw[:, :, 1:] = gx
    Gn[:, :-1, :] = gy
    Gs[:, 1:, :] = gy
    g = _gap_conductances(spec, f["nu"]) * dx * dx  # W/K; the Δx² is mandatory (§7.2)
    gup[:-1] = g
    gdn[1:] = g
    h[0] = f["h_top"] * dx * dx
    h[-1] = f["h_bot"] * dx * dx
    diag = Ge + Gw + Gn + Gs + gup + gdn + h + 2 * G * _edge_mask(n, edges)[None]
    rhs[0] = f["q"] * dx * dx
    return {"Ge": Ge, "Gw": Gw, "Gn": Gn, "Gs": Gs, "gup": gup, "gdn": gdn, "diag": diag,
            "rhs": rhs}


def assemble_board(
    spec, raster: object, dirichlet_edges: Collection[str]
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Assemble A θ = b for the 4-layer board; A CSR (4N² × 4N², W/K), b (4N², W).

    spec: SpecV2 (twinheat.spec.load("config/spec_v2.yaml")); raster: see the block comment
    above; dirichlet_edges ⊆ {"W", "E", "S", "N"}. Δx = L / N with N from the raster.
    """
    c = coefficients(spec, raster, dirichlet_edges)
    n = c["diag"].shape[1]
    k = np.arange(N_LAYERS * n * n).reshape(N_LAYERS, n, n)
    rows, cols, vals = [k.ravel()], [k.ravel()], [c["diag"].ravel()]
    for name, sl, step in (
        ("Ge", np.s_[:, :, :-1], 1), ("Gw", np.s_[:, :, 1:], -1),
        ("Gn", np.s_[:, :-1, :], n), ("Gs", np.s_[:, 1:, :], -n),
        ("gup", np.s_[:-1], n * n), ("gdn", np.s_[1:], -n * n),
    ):
        kk = k[sl].ravel()
        rows.append(kk)
        cols.append(kk + step)
        vals.append(-c[name][sl].ravel())
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(k.size, k.size),
    ).tocsr()
    return A, c["rhs"].ravel()


def board_energy_balance(
    spec, raster: object, dirichlet_edges: Collection[str], theta: np.ndarray
) -> float:
    """Relative energy-balance error |P_in − P_out| / P_in for a 4-layer θ [4, N, N] (K).

    P_in = Σ q Δx²; P_out = Σ h_top Δx² θ_L1 + Σ h_bot Δx² θ_L4 + Σ_layers Σ_Dirichlet 2 G_l θ.
    """
    n, f = _check_raster(raster)
    edges = _check_edges(dirichlet_edges)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (N_LAYERS, n, n):
        raise ValueError(f"theta must have shape {(N_LAYERS, n, n)}, got {theta.shape}")
    dx = spec.geometry.L.value / n
    p_in = float((f["q"] * dx * dx).sum())
    if p_in <= 0:
        raise ValueError(f"energy balance needs a positive total heat input, got {p_in} W")
    G = layer_conductances(spec, raster)
    p_conv = ((f["h_top"] * theta[0]).sum() + (f["h_bot"] * theta[-1]).sum()) * dx * dx
    p_edge = (2 * G * _edge_mask(n, edges)[None] * theta).sum()
    return abs(p_in - float(p_conv + p_edge)) / p_in
