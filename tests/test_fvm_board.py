"""4-layer board assembly (CLAUDE.md §7.1–§7.3; invariants B1, B2)."""

from __future__ import annotations

import csv
import datetime as dt
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from data.sampler import sample_board
from solvers.fvm import (
    assemble_board,
    assemble_plate,
    board_energy_balance,
    coefficients,
    layer_conductances,
    solve_direct,
)
from solvers.mmatrix_test import assert_m_matrix
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
SPEC1 = specmod.load(ROOT / "config" / "spec_v1.yaml")
L = SPEC.geometry.L.value  # m
N_FULL = SPEC.geometry.N.value  # 256
OZ_M = SPEC1.materials.t_cu.value  # 1 oz = 35 µm (CLAUDE.md §6.3)
SS = SPEC1.discretisation.supersample_per_axis.value  # 4 x 4 points per cell (§6.4)
ENERGY_TOL = SPEC1.discretisation.energy_balance_tol.value  # 0.1 % (§6.4)
PER_MM2 = 1e6  # vias/mm² -> vias/m²


# ---------------------------------------------------------------------- minimal Raster stand-in
# The prompt's stand-in until data/raster.py (M1.1) exists. Fields, SI, arrays [.., y, x]:
#   phi [4,N,N] copper coverage · q [N,N] W/m² on L1 · nu [3,N,N] vias/m² per gap (L1–L2, L2–L3,
#   L3–L4) · h_top, h_bot [N,N] W/(m²·K) · t_cu [4] m · k_fr4_xy W/(m·K).

@dataclass(frozen=True)
class Raster:
    phi: np.ndarray
    q: np.ndarray
    nu: np.ndarray
    h_top: np.ndarray
    h_bot: np.ndarray
    t_cu: np.ndarray
    k_fr4_xy: float


def uniform_raster(n: int, *, phi=(0.5, 0.9, 1.0, 0.2), nu=0.0, q=1e3, h_top=10.0, h_bot=8.0,
                   oz=(1, 1, 1, 1), k_fr4_xy=0.81) -> Raster:
    full = np.ones((n, n))
    return Raster(
        phi=np.stack([p * full for p in phi]), q=q * full, nu=np.stack([nu * full] * 3),
        h_top=h_top * full, h_bot=h_bot * full, t_cu=np.array(oz, dtype=float) * OZ_M,
        k_fr4_xy=k_fr4_xy,
    )


# Test rasteriser from board records: §6.4 supersampling, capsule traces (docs/decisions.md).
# Through-hole vias (arrays and background) cross all three gaps.

def _points(n: int) -> tuple[np.ndarray, np.ndarray]:
    c = (np.arange(n * SS) + 0.5) * (L * 1e3) / (n * SS)  # mm
    Y, X = np.meshgrid(c, c, indexing="ij")
    return X, Y


def _mask(s: dict[str, Any], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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


def _rect(d: dict[str, Any]) -> dict[str, Any]:
    return {"type": "rect", **{k: d[k] for k in ("x0", "y0", "w", "h")}}


def raster_from_record(rec: dict[str, Any], n: int) -> Raster:
    X, Y = _points(n)

    def cells(m: np.ndarray) -> np.ndarray:
        return m.reshape(n, SS, n, SS).mean(axis=(1, 3))

    layers = {layer["layer"]: layer for layer in rec["layers"]}
    union = [np.zeros_like(X, dtype=bool) for _ in range(4)]
    for s in layers[1]["shapes"]:
        union[0] |= _mask(s, X, Y)
    union[1][:] = layers[2]["plane"]
    for s in layers[2]["cutouts"]:
        union[1] &= ~_mask(s, X, Y)
    for s in layers[3]["split_rects"]:
        union[2] |= _mask(s, X, Y)
    for s in layers[4]["shapes"]:
        union[3] |= _mask(s, X, Y)
    phi = np.stack([cells(u) for u in union])

    via = np.full_like(X, rec["background_via_density_per_mm2"] * PER_MM2, dtype=float)
    for a in rec["via_arrays"]:
        via[_mask(_rect(a), X, Y)] += a["density_per_mm2"] * PER_MM2
    nu = np.stack([cells(via)] * 3)

    conv = rec["convection"]
    h_top = np.full_like(X, conv["h_top"], dtype=float)
    if conv["forced_zone"] is not None:
        h_top[_mask(_rect(conv["forced_zone"]), X, Y)] = conv["forced_zone"]["h_top"]
    q = np.zeros((n, n))
    for c in rec["chips"]:
        if c["side"] != "top":
            raise ValueError(f"{rec['id']}: test rasteriser handles top-side chips only")
        q += c["P"] / (c["w"] * c["h"] * 1e-6) * cells(_mask(_rect(c), X, Y))
    st = rec["stack"]
    oz = (st["outer_oz"], st["inner_oz"], st["inner_oz"], st["outer_oz"])
    return Raster(phi=phi, q=q, nu=nu, h_top=cells(h_top), h_bot=np.full((n, n), conv["h_bot"]),
                  t_cu=np.array(oz, dtype=float) * OZ_M, k_fr4_xy=st["k_fr4_xy"])


def _boards(count: int, n: int):
    for seed in range(count):
        rec = sample_board(seed, "id")
        yield rec, raster_from_record(rec, n)


# ---------------------------------------------------------------------- 1. 1-D through-thickness

def test_uniform_board_matches_series_resistances():
    """Uniform φ, ν, h, q and Neumann edges: the field is uniform in-plane, so per unit area
    (all terms per m², the Δx² cancels only if it is on g_lm AND h AND q):

        L1:  g12 (θ1 − θ2) + h_top θ1 = q
        L2:  g12 (θ2 − θ1) + g23 (θ2 − θ3) = 0
        L3:  g23 (θ3 − θ2) + g34 (θ3 − θ4) = 0
        L4:  g34 (θ4 − θ3) + h_bot θ4 = 0

    L2 and L3 carry no source or sink, so the downward flux f is the same through every gap:
    the bottom path is the series resistance R_b = 1/g12 + 1/g23 + 1/g34 + 1/h_bot (m²·K/W),
    in parallel with 1/h_top. Hence θ1 = q / (h_top + 1/R_b), f = θ1 / R_b,
    θ2 = θ1 − f/g12, θ3 = θ2 − f/g23, θ4 = θ3 − f/g34 (= f / h_bot).
    """
    n = 8
    nu = 0.3 * PER_MM2
    r = uniform_raster(n, nu=nu)
    A, b = assemble_board(SPEC, r, [])
    theta = solve_direct(A, b).reshape(4, n, n)

    g12, g23, g34 = (SPEC.through_plane_conductance(g, nu) for g in (1, 2, 3))
    h_top, h_bot, q = 10.0, 8.0, 1e3
    R_b = 1 / g12 + 1 / g23 + 1 / g34 + 1 / h_bot
    t1 = q / (h_top + 1 / R_b)
    f = t1 / R_b
    want = np.array([t1, t1 - f / g12, t1 - f / g12 - f / g23, f / h_bot])
    assert abs(want[2] - f / g34 - want[3]) <= 1e-12 * want[3]  # the derivation closes
    got = theta.mean(axis=(1, 2))
    np.testing.assert_allclose(got, want, rtol=1e-8)
    assert np.ptp(theta, axis=(1, 2)).max() <= 1e-10 * want.max()  # uniform in-plane


# ---------------------------------------------------------------------- 2. decoupled L1 = plate

@pytest.mark.parametrize("edges", [[], ["W"], ["S", "E"]])
def test_l1_block_is_the_plate_operator_plus_coupling(edges):
    """With g_lm → 0 L1 decouples; exactly, A's L1 block = assemble_plate(G1, h_top) + g12·Δx²."""
    n = 32
    rec = sample_board(3, "id")
    r = raster_from_record(rec, n)
    dx = L / n
    A, b = assemble_board(SPEC, r, edges)
    G = layer_conductances(SPEC, r)
    Ap, bp = assemble_plate(G[0], r.h_top, r.q, dx, edges)
    c = coefficients(SPEC, r, edges)
    g12_dx2 = c["gup"][0].ravel()
    block = A[: n * n, : n * n] - sp.diags(g12_dx2)
    diff = abs(block - Ap).max()
    assert diff <= 1e-10 * abs(Ap).max()
    np.testing.assert_allclose(b[: n * n], bp, rtol=1e-10, atol=0)
    assert not np.any(b[n * n:])  # q on L1 only


def test_l1_solution_approaches_the_plate_when_coupling_vanishes():
    n = 16
    r = uniform_raster(n)
    kz = SPEC.materials.k_fr4_z
    small = SPEC.model_copy(update={"materials": SPEC.materials.model_copy(
        update={"k_fr4_z": kz.model_copy(update={"value": kz.value * 1e-15})})})
    # the L1 field moves by about g12 / h_top relative (~1e-13 here), well inside rtol 1e-10
    A, b = assemble_board(small, r, ["W"])
    theta = solve_direct(A, b).reshape(4, n, n)
    G = layer_conductances(small, r)
    Ap, bp = assemble_plate(G[0], r.h_top, r.q, L / n, ["W"])
    plate = solve_direct(Ap, bp).reshape(n, n)
    np.testing.assert_allclose(theta[0], plate, rtol=1e-10)


# ---------------------------------------------------------------------- 3. M-matrix, 4. energy

@pytest.mark.slow
@pytest.mark.parametrize("i", range(50))
def test_m_matrix_on_sampled_boards(i):
    rec = sample_board(i, "id")
    A, _ = assemble_board(SPEC, raster_from_record(rec, N_FULL), rec["dirichlet_edges"])
    assert_m_matrix(A, context=f"4-layer {rec['id']} edges {rec['dirichlet_edges']}")


def test_m_matrix_on_a_few_boards_fast():
    for rec, r in _boards(5, 32):
        A, _ = assemble_board(SPEC, r, rec["dirichlet_edges"])
        assert_m_matrix(A, context=rec["id"])
        assert abs(A - A.T).max() == 0.0


BOARDS64 = list(_boards(20, 64))


@pytest.mark.parametrize("i", range(20))
def test_energy_balance_on_sampled_boards(i):
    rec, r = BOARDS64[i]
    edges = rec["dirichlet_edges"]
    A, b = assemble_board(SPEC, r, edges)
    theta = solve_direct(A, b).reshape(4, 64, 64)
    err = board_energy_balance(SPEC, r, edges, theta)
    assert err <= ENERGY_TOL, f"{rec['id']} edges {edges}: {err:.2e}"


def test_energy_balance_detects_a_wrong_field():
    rec, r = BOARDS64[0]
    A, b = assemble_board(SPEC, r, rec["dirichlet_edges"])
    theta = solve_direct(A, b).reshape(4, 64, 64)
    theta[3] *= 1.05
    assert board_energy_balance(SPEC, r, rec["dirichlet_edges"], theta) > ENERGY_TOL


# ---------------------------------------------------------------------- 5. coefficients = A

def _stencil_apply(c: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """(A x) from the GPU-kernel coefficients, with explicit neighbour shifts ([l, y, x])."""
    out = c["diag"] * x
    out[:, :, :-1] -= c["Ge"][:, :, :-1] * x[:, :, 1:]  # east  = x + 1
    out[:, :, 1:] -= c["Gw"][:, :, 1:] * x[:, :, :-1]  # west  = x − 1
    out[:, :-1, :] -= c["Gn"][:, :-1, :] * x[:, 1:, :]  # north = y + 1
    out[:, 1:, :] -= c["Gs"][:, 1:, :] * x[:, :-1, :]  # south = y − 1
    out[:-1] -= c["gup"][:-1] * x[1:]  # layer l + 1 (towards L4)
    out[1:] -= c["gdn"][1:] * x[:-1]  # layer l − 1 (towards L1)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coefficients_reproduce_A_times_x(seed):
    rec, r = BOARDS64[seed]
    edges = rec["dirichlet_edges"]
    A, b = assemble_board(SPEC, r, edges)
    c = coefficients(SPEC, r, edges)
    assert set(c) == {"Ge", "Gw", "Gn", "Gs", "gup", "gdn", "diag", "rhs"}
    assert all(v.shape == (4, 64, 64) and v.dtype == np.float64 for v in c.values())
    x = np.random.Generator(np.random.PCG64(seed)).standard_normal((4, 64, 64))
    ref = (A @ x.ravel()).reshape(4, 64, 64)
    assert np.abs(_stencil_apply(c, x) - ref).max() <= 1e-12 * np.abs(ref).max()
    np.testing.assert_array_equal(c["rhs"].ravel(), b)
    # boundary entries that point off the grid are zero (Neumann; Dirichlet lives on diag)
    assert not c["Ge"][:, :, -1].any() and not c["Gw"][:, :, 0].any()
    assert not c["Gn"][:, -1, :].any() and not c["Gs"][:, 0, :].any()
    assert not c["gup"][-1].any() and not c["gdn"][0].any()


def test_index_order_layer_then_y_then_x():
    """k = l·N² + y·N + x; an L1–L2 coupling sits at (k, k + N²)."""
    n = 4
    r = uniform_raster(n)
    A, _ = assemble_board(SPEC, r, [])
    k = 1 * n + 2  # L1, y = 1, x = 2
    assert A[k, k + n * n] < 0 and A[k, k + 1] < 0 and A[k, k + n] < 0
    assert A[k, k + 2 * n * n] == 0  # no L1–L3 coupling


# ---------------------------------------------------------------------- conductance formulas

def test_layer_conductances_match_the_spec_formulas():
    r = uniform_raster(4, phi=(1.0, 1.0, 1.0, 1.0))
    G = layer_conductances(SPEC, r)
    for layer in (1, 2, 3, 4):
        want = SPEC.in_plane_conductance(layer, 1.0, t_cu=r.t_cu[layer - 1])
        np.testing.assert_allclose(G[layer - 1], want, rtol=1e-15)


def test_record_copper_weights_and_k_fr4_xy_are_used():
    thin = uniform_raster(4, oz=(1, 1, 1, 1), k_fr4_xy=0.81)
    thick = uniform_raster(4, oz=(2, 1, 1, 2), k_fr4_xy=1.0)
    G1, G2 = layer_conductances(SPEC, thin), layer_conductances(SPEC, thick)
    assert np.all(G2[[0, 3]] > G1[[0, 3]]) and np.all(G2[[1, 2]] > G1[[1, 2]])  # k_xy raises all
    k_cu = SPEC.materials.k_cu.value
    d12 = SPEC.stack.d12.value
    np.testing.assert_allclose(G2[0], k_cu * 2 * OZ_M * 0.5 + 1.0 * 0.5 * d12, rtol=1e-15)


@pytest.mark.parametrize(
    "field, bad, match",
    [
        ("phi", lambda r: r.phi[:3], "phi"),
        ("phi", lambda r: r.phi * 1.5, "phi"),
        ("nu", lambda r: -r.nu - 1, "nu"),
        ("q", lambda r: np.full_like(r.q, np.inf), "q"),
        ("t_cu", lambda r: r.t_cu[:3], "t_cu"),
        ("h_bot", lambda r: -r.h_bot, "h_bot"),
    ],
)
def test_bad_rasters_raise(field, bad, match):
    r = uniform_raster(4)
    kwargs = {f: getattr(r, f) for f in Raster.__dataclass_fields__}
    kwargs[field] = bad(r)
    with pytest.raises(ValueError, match=match):
        assemble_board(SPEC, Raster(**kwargs), [])


# ---------------------------------------------------------------------- 6. κ₂ by Lanczos

CONDITIONING_CSV = ROOT / "bench" / "results" / "conditioning.csv"


def _git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_condition_number_by_lanczos(seed):
    rec = sample_board(seed, "id")
    edges = rec["dirichlet_edges"]
    A, _ = assemble_board(SPEC, raster_from_record(rec, N_FULL), edges)
    lam_max = spla.eigsh(A, k=1, which="LA", tol=1e-8, return_eigenvectors=False)[0]
    # smallest eigenvalue by shift-invert Lanczos about 0 (A is SPD)
    lam_min = spla.eigsh(A, k=1, sigma=0, which="LM", tol=1e-8, return_eigenvectors=False)[0]
    assert lam_min > 0
    kappa = lam_max / lam_min
    new = not CONDITIONING_CSV.exists()
    CONDITIONING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CONDITIONING_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "commit", "board_id", "N", "layers", "dirichlet_edges",
                        "lambda_min_W_per_K", "lambda_max_W_per_K", "kappa2", "method", "hardware"])
        w.writerow([dt.date.today().isoformat(), _git_commit(), rec["id"], N_FULL, 4,
                    "".join(edges) or "-", f"{lam_min:.6e}", f"{lam_max:.6e}", f"{kappa:.6e}",
                    "scipy eigsh LA + shift-invert sigma=0, tol 1e-8", platform.platform()])
    print(f"\n{rec['id']} edges {edges}: lambda_min {lam_min:.3e}, lambda_max {lam_max:.3e}, "
          f"kappa2 {kappa:.3e}")
