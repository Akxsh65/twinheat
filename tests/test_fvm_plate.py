"""64² plate solver (CLAUDE.md §6.2–§6.4; invariants B1, B2, B4, B5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg  # noqa: F401  (sp.linalg.norm)

from data.sampler import sample_board
from solvers.fvm import assemble_plate, energy_balance, solve_direct
from solvers.mmatrix_test import assert_m_matrix
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v1.yaml")
L = SPEC.geometry.L.value  # m
N64 = SPEC.geometry.N.value
SS = SPEC.discretisation.supersample_per_axis.value
OZ_M = SPEC.materials.t_cu.value  # 1 oz copper thickness, m (CLAUDE.md §6.3)
T = SPEC.geometry.t.value
K_CU = SPEC.materials.k_cu.value
ENERGY_TOL = SPEC.discretisation.energy_balance_tol.value  # 1e-3 = 0.1 % (CLAUDE.md §6.4)
SLOPE_LO, SLOPE_HI = SPEC.discretisation.mms_slope.range
N_PLANE_CHOICES = SPEC.sampler.n_plane.value
N_PLANE_P = SPEC.sampler.n_plane.p


# ---------------------------------------------------------------------- test rasteriser
# Minimal stand-in until data/raster.py exists (M1.1): §6.4 4 x 4 supersampling, point sampling
# at (i + 0.5)/4 inside each cell, capsule rule for traces (docs/decisions.md). Plate G uses the
# L1 painted coverage only, with n_plane drawn from the §6.5 distribution.

def _points(n: int) -> tuple[np.ndarray, np.ndarray]:
    c = (np.arange(n * SS) + 0.5) * (L * 1e3) / (n * SS)  # mm
    return np.meshgrid(c, c, indexing="ij")[::-1]  # X, Y with arrays [y, x]


def _shape_mask(s: dict[str, Any], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if s["type"] == "rect":
        return (X >= s["x0"]) & (X < s["x0"] + s["w"]) & (Y >= s["y0"]) & (Y < s["y0"] + s["h"])
    if s["type"] == "disc":
        return (X - s["cx"]) ** 2 + (Y - s["cy"]) ** 2 <= s["r"] ** 2
    if s["type"] == "trace":
        out = np.zeros_like(X, dtype=bool)
        for (ax, ay), (bx, by) in zip(s["points"], s["points"][1:], strict=False):
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            u = 0.0 if seg2 == 0 else np.clip(((X - ax) * dx + (Y - ay) * dy) / seg2, 0, 1)
            out |= (X - (ax + u * dx)) ** 2 + (Y - (ay + u * dy)) ** 2 <= (s["width"] / 2) ** 2
        return out
    raise ValueError(f"test rasteriser: unknown shape {s}")


def _cell_mean(mask: np.ndarray, n: int) -> np.ndarray:
    return mask.reshape(n, SS, n, SS).mean(axis=(1, 3))


def _plate_fields(record: dict[str, Any], n: int, n_plane: int):
    """(G W/K, H W/(m²·K), q W/m²) on [n, n] from a board record (geometry in mm)."""
    X, Y = _points(n)
    l1 = next(layer for layer in record["layers"] if layer["layer"] == 1)
    painted = np.zeros_like(X, dtype=bool)
    for s in l1["shapes"]:
        painted |= _shape_mask(s, X, Y)
    phi = _cell_mean(painted, n)
    t_cu = record["stack"]["outer_oz"] * OZ_M
    G = record["stack"]["k_fr4_xy"] * T + K_CU * t_cu * (n_plane + phi)
    conv = record["convection"]
    h_top = np.full_like(X, conv["h_top"], dtype=float)
    if conv["forced_zone"] is not None:
        z = conv["forced_zone"]
        zone = {"type": "rect", **{k: z[k] for k in ("x0", "y0", "w", "h")}}
        h_top[_shape_mask(zone, X, Y)] = z["h_top"]
    H = _cell_mean(h_top, n) + conv["h_bot"]
    q = np.zeros((n, n))
    for c in record["chips"]:
        area_m2 = c["w"] * c["h"] * 1e-6
        q += c["P"] / area_m2 * _cell_mean(_shape_mask({"type": "rect", **c}, X, Y), n)
    return G, H, q


def _boards(count: int = 20):
    rng = np.random.Generator(np.random.PCG64(20261028))
    for seed in range(count):
        rec = sample_board(seed, "id")
        n_plane = int(rng.choice(N_PLANE_CHOICES, p=N_PLANE_P))
        yield rec, (*_plate_fields(rec, N64, n_plane), L / N64, rec["dirichlet_edges"])


BOARDS = list(_boards())


# ---------------------------------------------------------------------- 1. manufactured solution

def _mms_error(n: int) -> float:
    dx = L / n
    c = (np.arange(n) + 0.5) * dx
    Yc, Xc = np.meshgrid(c, c, indexing="ij")
    G0 = SPEC.sheet_conductance(0, 1.0)  # one 1 oz layer, W/K (§6.3 check value 1.50e-2)
    H0 = 2 * 10.0  # W/(m²·K): 10 per face, inside the natural-convection training range
    exact = np.sin(np.pi * Xc / L) * np.sin(np.pi * Yc / L)  # K; zero on all four edges
    q = (2 * G0 * np.pi**2 / L**2 + H0) * exact  # -G lap(theta) + H theta = q, W/m²
    A, b = assemble_plate(np.full((n, n), G0), np.full((n, n), H0), q, dx, ["W", "E", "S", "N"])
    theta = solve_direct(A, b).reshape(n, n)
    return float(np.abs(theta - exact).max())


def test_manufactured_solution_second_order():
    ns = np.array([32, 64, 128, 256])
    errs = np.array([_mms_error(int(n)) for n in ns])
    slope = -np.polyfit(np.log(ns), np.log(errs), 1)[0]
    table = dict(zip(ns.tolist(), errs.tolist(), strict=True))
    print(f"\nMMS max errors {table}, slope {slope:.4f}")
    assert np.all(np.diff(errs) < 0), errs
    assert SLOPE_LO <= slope <= SLOPE_HI, f"MMS slope {slope:.4f} outside [{SLOPE_LO}, {SLOPE_HI}]"


# ---------------------------------------------------------------------- 2.–4. sampled boards

@pytest.mark.parametrize("i", range(len(BOARDS)))
def test_energy_balance_on_sampled_boards(i):
    rec, (G, H, q, dx, edges) = BOARDS[i]
    A, b = assemble_plate(G, H, q, dx, edges)
    theta = solve_direct(A, b).reshape(G.shape)
    err = energy_balance(theta, G, H, q, dx, edges)
    assert err <= ENERGY_TOL, f"{rec['id']} edges {edges}: energy balance error {err:.2e}"


@pytest.mark.parametrize("i", range(len(BOARDS)))
def test_m_matrix_on_sampled_boards(i):
    rec, (G, H, q, dx, edges) = BOARDS[i]
    A, _ = assemble_plate(G, H, q, dx, edges)
    assert_m_matrix(A, context=f"plate {rec['id']} edges {edges}")


@pytest.mark.parametrize("i", range(len(BOARDS)))
def test_symmetric(i):
    _, (G, H, q, dx, edges) = BOARDS[i]
    A, _ = assemble_plate(G, H, q, dx, edges)
    assert sp.linalg.norm(A - A.T) == 0.0


def test_sampled_boards_cover_the_edge_cases():
    edge_counts = {len(e) for _, (*_, e) in BOARDS}
    assert edge_counts >= {0, 1}, edge_counts  # pure-Neumann and heat-sunk boards both tested
    assert max(G.max() / G.min() for _, (G, *_rest) in BOARDS) > 5  # real copper contrast


# ---------------------------------------------------------------------- 5. linearity (B4)

def test_linearity_in_q():
    _, (G, H, q, dx, edges) = BOARDS[0]
    A, b = assemble_plate(G, H, q, dx, edges)
    A3, b3 = assemble_plate(G, H, 3 * q, dx, edges)
    assert (A != A3).nnz == 0
    t1, t3 = solve_direct(A, b), solve_direct(A3, b3)
    assert np.abs(t3 - 3 * t1).max() <= 1e-10 * np.abs(3 * t1).max()


# ---------------------------------------------------------------------- 6. symmetry (B5, one case)

def _centred_chip(n: int = N64):
    dx = L / n
    G = np.full((n, n), SPEC.sheet_conductance(1, 0.0))
    H = np.full((n, n), 2 * 10.0)
    q = np.zeros((n, n))
    lo, hi = n * 3 // 8, n * 5 // 8  # 37.5–62.5 mm: centred, whole cells
    q[lo:hi, lo:hi] = 1.0 / (((hi - lo) * dx) ** 2)  # 1 W over the footprint, W/m²
    return G, H, q, dx


def test_centred_chip_all_neumann_symmetric_under_x_flip():
    G, H, q, dx = _centred_chip()
    A, b = assemble_plate(G, H, q, dx, [])
    theta = solve_direct(A, b).reshape(G.shape)
    assert np.abs(theta - theta[:, ::-1]).max() <= 1e-10 * np.abs(theta).max()
    assert theta.min() > 0
    assert energy_balance(theta, G, H, q, dx, []) <= ENERGY_TOL


# ---------------------------------------------------------------------- orientation, inputs

def test_index_order_and_dirichlet_edges_are_where_they_say():
    """k = y·N + x; W = x index 0, S = y index 0 (CLAUDE.md §0 rule 7)."""
    n = 8
    G = np.ones((n, n)) * 1e-2
    H = np.ones((n, n))
    q = np.ones((n, n))
    for edge, cells in {"W": (slice(None), 0), "E": (slice(None), -1),
                        "S": (0, slice(None)), "N": (-1, slice(None))}.items():
        A, _ = assemble_plate(G, H, q, 1e-3, [edge])
        A0, _ = assemble_plate(G, H, q, 1e-3, [])
        extra = (A - A0).diagonal().reshape(n, n)
        want = np.zeros((n, n))
        want[cells] = 2 * G[cells]
        np.testing.assert_allclose(extra, want, rtol=0, atol=1e-15, err_msg=edge)
    # an x-neighbour coupling sits at k and k+1, a y-neighbour at k and k+N
    A, _ = assemble_plate(G, H, q, 1e-3, [])
    assert A[0, 1] < 0 and A[0, n] < 0 and A[0, n + 1] == 0


def test_harmonic_face_conductance():
    G = np.array([[1.0, 3.0]])
    G = np.vstack([G, G])
    A, _ = assemble_plate(G, np.ones((2, 2)), np.zeros((2, 2)), 1e-3, [])
    assert A[0, 1] == pytest.approx(-2 * 1.0 * 3.0 / (1.0 + 3.0), rel=1e-15)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"G": np.ones((4, 5))}, "square"),
        ({"H": np.ones((3, 3))}, "same shape"),
        ({"G": -np.ones((4, 4))}, "G must be > 0"),
        ({"H": -np.ones((4, 4))}, "H must be >= 0"),
        ({"q": np.full((4, 4), np.nan)}, "finite"),
        ({"dx": 0.0}, "dx"),
        ({"edges": ["X"]}, "dirichlet_edges"),
        ({"H": np.zeros((4, 4)), "edges": []}, "singular"),
    ],
)
def test_bad_inputs_raise(kwargs, match):
    args = {"G": np.ones((4, 4)), "H": np.ones((4, 4)), "q": np.ones((4, 4)), "dx": 1e-3,
            "edges": ["W"]}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        assemble_plate(args["G"], args["H"], args["q"], args["dx"], args["edges"])


def test_energy_balance_detects_a_wrong_field():
    _, (G, H, q, dx, edges) = BOARDS[1]
    A, b = assemble_plate(G, H, q, dx, edges)
    theta = solve_direct(A, b).reshape(G.shape)
    assert energy_balance(1.01 * theta, G, H, q, dx, edges) > ENERGY_TOL
