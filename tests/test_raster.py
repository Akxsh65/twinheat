"""Rasteriser (prompt M1.1; CLAUDE.md §6.4, §6.6, §7.1–§7.4, §10.1): records -> SI fields."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contracts.validate import ContractError
from data.raster import Raster, rasterise, transform_array
from data.sampler import sample_board, transform_record
from solvers.fvm import assemble_board
from solvers.mmatrix_test import assert_m_matrix
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
SPEC1 = specmod.load(ROOT / "config" / "spec_v1.yaml")
L_MM = SPEC.geometry.L.value * 1e3
OZ_M = SPEC1.materials.t_cu.value  # 1 oz = 35 µm (CLAUDE.md §6.3)
NS = (64, 128, 256)
SS = SPEC1.discretisation.supersample_per_axis.value
FIELDS = ("phi", "q", "q_bot", "nu", "h_top", "h_bot", "dirichlet_mask")


def _board(**over: Any) -> dict[str, Any]:
    """A hand-built, schema-valid record: one shape family per layer, nothing overlapping."""
    rec: dict[str, Any] = {
        "schema_version": "1.0", "id": "test-board", "seed": 0, "mode": "id",
        "stack": {"outer_oz": 2, "inner_oz": 1, "k_fr4_xy": 0.9},
        "layers": [
            {"layer": 1, "shapes": [
                {"type": "rect", "x0": 10.0, "y0": 10.0, "w": 20.0, "h": 10.0},
                {"type": "disc", "cx": 70.0, "cy": 70.0, "r": 8.0},
                {"type": "trace", "points": [[10.0, 80.0], [40.0, 80.0]], "width": 4.0},
            ]},
            {"layer": 2, "plane": True,
             "cutouts": [{"type": "rect", "x0": 60.0, "y0": 5.0, "w": 30.0, "h": 15.0}]},
            {"layer": 3, "split_rects": [
                {"type": "rect", "x0": 0.0, "y0": 0.0, "w": 40.0, "h": 100.0},
                {"type": "rect", "x0": 40.0, "y0": 0.0, "w": 60.0, "h": 100.0},
            ]},
            {"layer": 4, "shapes": [{"type": "disc", "cx": 30.0, "cy": 40.0, "r": 10.0}]},
        ],
        "chips": [
            {"x0": 41.3, "y0": 30.7, "w": 10.9, "h": 7.3, "P": 1.25, "side": "top"},
            {"x0": 70.0, "y0": 40.0, "w": 6.0, "h": 6.0, "P": 0.5, "side": "top"},
        ],
        "via_arrays": [{"x0": 41.3, "y0": 30.7, "w": 10.9, "h": 7.3, "density_per_mm2": 0.5}],
        "background_via_density_per_mm2": 0.02,
        "convection": {"h_top": 10.0, "h_bot": 8.0,
                       "forced_zone": {"x0": 0.0, "y0": 50.0, "w": 50.0, "h": 50.0, "h_top": 40.0}},
        "dirichlet_edges": ["W", "N"],
        "design_vector": None,
    }
    rec.update(over)
    return rec


def _cell_area_mm2(n: int) -> float:
    return (L_MM / n) ** 2


def _sampled_boards() -> list[dict[str, Any]]:
    """ID and design boards, chosen to include every Dirichlet-edge count and a forced zone."""
    boards = [sample_board(seed, "id") for seed in range(12)]
    boards += [sample_board(seed, "design") for seed in range(3)]
    n_edges = {len(b["dirichlet_edges"]) for b in boards}
    assert n_edges == {0, 1, 2}, f"seeds no longer cover 0/1/2 Dirichlet edges: {n_edges}"
    assert any(b["convection"]["forced_zone"] for b in boards), "no forced zone in the seeds"
    return boards


BOARDS = _sampled_boards()


# ------------------------------------------------------------------ 1. copper area vs analytic


@pytest.mark.parametrize("n", NS)
def test_copper_area_matches_the_analytic_area(n: int) -> None:
    """Area from φ vs the analytic area, within the 4 x 4 supersampling bound.

    A sample is misclassified only near the boundary, so |error| <= h·perimeter + 4h² with h the
    sample spacing. At N = 64, h = 0.39 mm, so a 10 mm side can be off by ~4 %: the plan's flat
    1 % (M1.1 test 1) only holds for large shapes, and is checked on the two plane layers.
    """
    r = rasterise(_board(), SPEC, n)
    a, h = _cell_area_mm2(n), L_MM / (n * SS)
    shapes = {  # layer: (analytic area mm², perimeter mm)
        0: (20.0 * 10.0 + math.pi * 8.0**2 + 4.0 * 30.0 + math.pi * 2.0**2,
            60.0 + 2 * math.pi * 8.0 + 2 * 30.0 + 2 * math.pi * 2.0),
        1: (L_MM**2 - 30.0 * 15.0, 90.0),
        2: (L_MM**2, 0.0),
        3: (math.pi * 10.0**2, 2 * math.pi * 10.0),
    }
    for layer, (area, perimeter) in shapes.items():
        got = r.phi[layer].sum() * a
        bound = h * perimeter + 4 * h * h
        assert abs(got - area) <= bound, f"N={n} L{layer + 1}: {got} vs {area} ± {bound} mm²"
    for layer in (1, 2):  # whole-board planes: the plan's 1 %
        assert r.phi[layer].sum() * a == pytest.approx(shapes[layer][0], rel=0.01)


def test_copper_area_agrees_across_resolutions_on_sampled_boards() -> None:
    for rec in BOARDS[:5]:
        areas = {n: rasterise(rec, SPEC, n).phi.sum(axis=(1, 2)) * _cell_area_mm2(n) for n in NS}
        for n in NS[:-1]:
            np.testing.assert_allclose(areas[n], areas[256], rtol=0.01, atol=1.0,
                                       err_msg=f"{rec['id']}: N={n} vs 256")


# ------------------------------------------------------------------ 2. power is conserved


@pytest.mark.parametrize("n", NS)
def test_total_power_is_conserved(n: int) -> None:
    for rec in [_board(), *BOARDS[:6]]:
        r = rasterise(rec, SPEC, n)
        total = sum(c["P"] for c in rec["chips"])
        got = (r.q.sum() + r.q_bot.sum()) * r.dx**2
        assert got == pytest.approx(total, rel=1e-12), f"{rec['id']} N={n}: {got} vs {total} W"


def test_q_is_power_over_footprint_inside_a_chip() -> None:
    n = 256
    r = rasterise(_board(), SPEC, n)
    c = _board()["chips"][1]  # 6 x 6 mm
    iy, ix = int((c["y0"] + 3.0) / L_MM * n), int((c["x0"] + 3.0) / L_MM * n)
    want = c["P"] / (c["w"] * c["h"] * 1e-6)
    assert r.q[iy, ix] == pytest.approx(want, rel=0.02)
    assert r.q[0, 0] == 0.0


# ------------------------------------------------------------------ 3. symmetries commute


@pytest.mark.parametrize("sym", range(8))
def test_rasterise_commutes_with_every_symmetry(sym: int) -> None:
    for rec in [_board(), *BOARDS]:
        a = rasterise(transform_record(rec, sym), SPEC, 64)
        b = transform_array(rasterise(rec, SPEC, 64), sym)
        for f in FIELDS:
            assert np.array_equal(getattr(a, f), getattr(b, f)), f"{rec['id']} sym {sym}: {f}"
        assert np.array_equal(a.t_cu, b.t_cu) and a.k_fr4_xy == b.k_fr4_xy


@pytest.mark.parametrize("sym", [1, 4, 7])
def test_symmetry_is_exact_at_full_resolution(sym: int) -> None:
    rec = _board()
    a = rasterise(transform_record(rec, sym), SPEC, 256)
    b = transform_array(rasterise(rec, SPEC, 256), sym)
    for f in FIELDS:
        assert np.array_equal(getattr(a, f), getattr(b, f)), f"sym {sym}: {f}"


def test_transform_array_round_trips() -> None:
    from data.sampler import inverse_sym

    r = rasterise(_board(), SPEC, 64)
    for sym in range(8):
        back = transform_array(transform_array(r, sym), inverse_sym(sym))
        for f in FIELDS:
            assert np.array_equal(getattr(back, f), getattr(r, f)), f"sym {sym}: {f}"


# ------------------------------------------------------------------ 4. ranges, dtypes, shapes


@pytest.mark.parametrize("n", [64, 256])
def test_fields_are_finite_float64_and_in_range(n: int) -> None:
    for rec in [_board(), *BOARDS[:6]]:
        r = rasterise(rec, SPEC, n)
        assert r.N == n and r.dx == pytest.approx(SPEC.geometry.L.value / n, rel=1e-15)
        shapes = {"phi": (4, n, n), "q": (n, n), "nu": (3, n, n), "h_top": (n, n),
                  "h_bot": (n, n), "dirichlet_mask": (n, n)}
        for f, shape in shapes.items():
            v = getattr(r, f)
            assert v.dtype == np.float64 and v.shape == shape, f"{rec['id']} {f}"
            assert np.isfinite(v).all(), f"{rec['id']} {f} has NaN or inf"
        assert r.phi.min() >= 0.0 and r.phi.max() <= 1.0
        assert r.q.min() >= 0.0 and r.nu.min() >= 0.0
        assert r.h_top.min() > 0.0 and r.h_bot.min() > 0.0
        assert set(np.unique(r.dirichlet_mask)) <= {0.0, 1.0}


# ------------------------------------------------------------------ 5. small features survive


def test_a_3mm_disc_has_coverage_at_n64() -> None:
    rec = _board(layers=[
        {"layer": 1, "shapes": [{"type": "disc", "cx": 33.3, "cy": 61.7, "r": 3.0}]},
        {"layer": 2, "plane": True, "cutouts": []},
        {"layer": 3, "split_rects": []},
        {"layer": 4, "shapes": []},
    ])
    r = rasterise(rec, SPEC, 64)
    assert r.phi[0].sum() > 0.0
    assert r.phi[0].sum() * _cell_area_mm2(64) == pytest.approx(math.pi * 9.0, rel=0.1)


# ------------------------------------------------------------------ orientation and units


def test_array_order_is_layer_y_x_with_origin_lower_left() -> None:
    n = 64
    rec = _board(layers=[
        {"layer": 1, "shapes": [{"type": "rect", "x0": 0.0, "y0": 0.0, "w": 10.0, "h": 5.0}]},
        {"layer": 2, "plane": True, "cutouts": []},
        {"layer": 3, "split_rects": []},
        {"layer": 4, "shapes": []},
    ], dirichlet_edges=["W"])
    r = rasterise(rec, SPEC, n)
    assert r.phi[0, 0, 0] == 1.0  # south-west corner cell
    assert r.phi[0, n - 1, 0] == 0.0 and r.phi[0, 0, n - 1] == 0.0
    # 10 mm along x = 6.4 cells, 5 mm along y = 3.2 cells (dx = 1.5625 mm): wider than tall
    assert r.phi[0, 0, 5] == 1.0 and r.phi[0, 0, 7] == 0.0
    assert r.phi[0, 2, 0] == 1.0 and r.phi[0, 4, 0] == 0.0
    assert (r.dirichlet_mask[:, 0] == 1.0).all() and r.dirichlet_mask[:, 1:].sum() == 0.0


def test_mask_marks_cells_next_to_each_dirichlet_edge() -> None:
    n = 32
    r = rasterise(_board(dirichlet_edges=["W", "E", "S", "N"]), SPEC, n)
    want = np.zeros((n, n))
    want[:, 0] = want[:, -1] = want[0, :] = want[-1, :] = 1.0
    assert np.array_equal(r.dirichlet_mask, want)
    assert rasterise(_board(dirichlet_edges=[]), SPEC, n).dirichlet_mask.sum() == 0.0


def test_units_vias_per_m2_convection_and_copper_thickness() -> None:
    n = 256
    rec = _board()
    r = rasterise(rec, SPEC, n)
    bg = rec["background_via_density_per_mm2"] * 1e6
    assert r.nu[:, 0, n - 1] == pytest.approx([bg] * 3, rel=1e-12)  # far from the array
    iy, ix = int(34.0 / L_MM * n), int(46.0 / L_MM * n)  # inside chip 0's via array
    assert r.nu[:, iy, ix] == pytest.approx([bg + 0.5e6] * 3, rel=1e-12)
    assert r.h_top[n - 1, 0] == 40.0 and r.h_top[0, n - 1] == 10.0  # forced zone is NW quarter
    assert (r.h_bot == 8.0).all()
    assert r.t_cu == pytest.approx([2 * OZ_M, OZ_M, OZ_M, 2 * OZ_M], rel=1e-15)
    assert r.k_fr4_xy == 0.9


def test_overlapping_shapes_are_a_union_not_a_sum() -> None:
    rect = {"type": "rect", "x0": 20.0, "y0": 20.0, "w": 30.0, "h": 30.0}
    rec = _board(layers=[
        {"layer": 1, "shapes": [rect, copy.deepcopy(rect),
                                {"type": "disc", "cx": 35.0, "cy": 35.0, "r": 5.0}]},
        {"layer": 2, "plane": True, "cutouts": []},
        {"layer": 3, "split_rects": []},
        {"layer": 4, "shapes": []},
    ])
    r = rasterise(rec, SPEC, 64)
    alone = rasterise(_board(layers=[{**rec["layers"][0], "shapes": [rect]}, *rec["layers"][1:]]),
                      SPEC, 64)
    assert r.phi.max() <= 1.0
    assert np.array_equal(r.phi[0], alone.phi[0])  # the copy and the inner disc add nothing


def test_trace_caps_are_round_capsule_rule() -> None:
    n = 256
    trace = {"type": "trace", "points": [[40.0, 50.0], [60.0, 50.0]], "width": 6.0}
    rec = _board(layers=[
        {"layer": 1, "shapes": [trace]},
        {"layer": 2, "plane": True, "cutouts": []},
        {"layer": 3, "split_rects": []},
        {"layer": 4, "shapes": []},
    ])
    phi = rasterise(rec, SPEC, n).phi[0]
    dx = L_MM / n
    corner = (int((50.0 + 2.9) / dx), int((40.0 - 2.9) / dx))  # in a square cap, not a round
    assert phi[corner] == 0.0
    tip = (int(50.0 / dx), int((40.0 - 2.5) / dx))  # on the axis, inside the round cap
    assert phi[tip] > 0.0


# ------------------------------------------------------------------ fits the solver (N1.2)


def test_rasters_assemble_into_m_matrices() -> None:
    for rec in [_board(), *BOARDS[:4]]:
        r = rasterise(rec, SPEC, 32)
        A, _ = assemble_board(SPEC, r, rec["dirichlet_edges"])
        assert_m_matrix(A)


# ------------------------------------------------------------------ fails loudly


# ------------------------------------------------------------------ OOD support (M1.2)


@pytest.mark.parametrize("n", NS)
def test_bottom_side_chips_go_to_q_bot(n: int) -> None:
    rec = _board()
    rec["chips"][0]["side"] = "bottom"
    r = rasterise(rec, SPEC, n)
    top, bot = rec["chips"][1]["P"], rec["chips"][0]["P"]
    assert r.q.sum() * r.dx**2 == pytest.approx(top, rel=1e-12)
    assert r.q_bot.sum() * r.dx**2 == pytest.approx(bot, rel=1e-12)
    assert not np.any((r.q > 0) & (r.q_bot > 0))  # the two chips do not overlap


def test_top_only_boards_have_zero_q_bot() -> None:
    for rec in BOARDS:
        assert not rasterise(rec, SPEC, 32).q_bot.any(), rec["id"]


@pytest.mark.parametrize("n", NS)
def test_45_degree_trace_area_matches_the_capsule(n: int) -> None:
    c = 20.0  # per-axis run: length 20·√2 mm
    trace = {"type": "trace", "points": [[30.0, 30.0], [30.0 + c, 30.0 + c]], "width": 5.0}
    rec = _board(layers=[
        {"layer": 1, "shapes": [trace]},
        {"layer": 2, "plane": True, "cutouts": []},
        {"layer": 3, "split_rects": []},
        {"layer": 4, "shapes": []},
    ])
    r = rasterise(rec, SPEC, n)
    length, half, h = c * math.sqrt(2), 2.5, L_MM / (n * SS)
    area = 2 * half * length + math.pi * half**2
    perimeter = 2 * length + 2 * math.pi * half
    got = r.phi[0].sum() * _cell_area_mm2(n)
    assert abs(got - area) <= h * perimeter + 4 * h * h, f"N={n}: {got} vs {area} mm²"


def test_45_degree_trace_is_symmetric_under_every_symmetry() -> None:
    trace = {"type": "trace", "points": [[12.3, 20.0], [32.3, 40.0], [52.3, 20.0], [52.3, 9.1]],
             "width": 4.4}
    rec = _board()
    rec["layers"][0]["shapes"][2] = trace
    for sym in range(8):
        a = rasterise(transform_record(rec, sym), SPEC, 64)
        b = transform_array(rasterise(rec, SPEC, 64), sym)
        assert np.array_equal(a.phi, b.phi), f"sym {sym}"


@pytest.mark.parametrize("sym", range(8))
def test_ood_boards_commute_with_symmetries(sym: int) -> None:
    for mode in ("ood_3", "ood_4", "ood_bottom_chips", "ood_dense_vias"):
        for seed in range(3):
            rec = sample_board(seed, mode)
            a = rasterise(transform_record(rec, sym), SPEC, 64)
            b = transform_array(rasterise(rec, SPEC, 64), sym)
            for f in FIELDS:
                assert np.array_equal(getattr(a, f), getattr(b, f)), f"{rec['id']} {sym} {f}"


@pytest.mark.xfail(strict=True, reason="needs N: solvers/fvm.py does not read Raster.q_bot yet "
                   "(docs/decisions.md OOD sets v2, point 2). Remove this mark when it does.")
def test_solver_counts_bottom_chip_heat() -> None:
    rec = _board()
    for c in rec["chips"]:
        c["side"] = "bottom"
    r = rasterise(rec, SPEC, 32)
    _, b = assemble_board(SPEC, r, rec["dirichlet_edges"])
    total = sum(c["P"] for c in rec["chips"])
    assert b.sum() == pytest.approx(total, rel=1e-9)  # b is in W: Σ (q + q_bot)·Δx²


def test_off_grid_geometry_raises() -> None:
    rec = _board()
    rec["layers"][0]["shapes"][0]["x0"] = 10.0004
    with pytest.raises(ValueError, match="geometry_resolution"):
        rasterise(rec, SPEC, 64)


def test_invalid_record_raises_with_its_json_path() -> None:
    rec = _board()
    del rec["convection"]["h_bot"]
    with pytest.raises(ContractError, match="convection"):
        rasterise(rec, SPEC, 64)


@pytest.mark.parametrize("n", [0, -4, 64.0, True])
def test_bad_resolution_raises(n: Any) -> None:
    with pytest.raises(ValueError, match="N"):
        rasterise(_board(), SPEC, n)


def test_wrong_spec_raises() -> None:
    with pytest.raises(TypeError, match="SpecV2"):
        rasterise(_board(), SPEC1, 64)


def test_raster_is_a_frozen_dataclass() -> None:
    r = rasterise(_board(), SPEC, 32)
    assert isinstance(r, Raster)
    with pytest.raises(AttributeError):
        r.N = 64  # type: ignore[misc]
