"""Board sampler v0 (CLAUDE.md §6.5, §6.6, §7.4, §10.1): determinism, constraints, symmetries."""

from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Any

import numpy as np
import pytest
from scipy.stats import binomtest, chisquare

from contracts.validate import validate
from data import sampler
from data.sampler import coverage, design_board, inverse_sym, sample_board, transform_record

N_RECORDS = 1000
RES = 1.0e-3  # mm, sampler.v0_defaults.geometry_resolution
EDGES = ("W", "E", "S", "N")


def _json(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True).encode("utf-8")


@pytest.fixture(scope="module")
def records() -> list[dict[str, Any]]:
    return [sample_board(seed, "id") for seed in range(N_RECORDS)]


@pytest.fixture(scope="module")
def design_records() -> list[dict[str, Any]]:
    return [sample_board(seed, "design") for seed in range(50)]


def _feasible_x() -> np.ndarray:
    x = np.full(36, 0.5)
    x[0:8] = [0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9]  # chips near the four corners
    return x


# --------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("mode", ["id", "design"])
@pytest.mark.parametrize("seed", [0, 1, 12345, 2**31 - 1])
def test_same_seed_gives_byte_identical_json(seed: int, mode: str) -> None:
    assert _json(sample_board(seed, mode)) == _json(sample_board(seed, mode))


def test_different_seeds_give_different_boards(records: list[dict[str, Any]]) -> None:
    assert len({_json({**r, "id": "", "seed": 0}) for r in records}) == N_RECORDS


def test_record_carries_its_seed_and_mode(records: list[dict[str, Any]]) -> None:
    for seed, r in enumerate(records):
        assert r["seed"] == seed
        assert r["mode"] == "id"
        assert r["design_vector"] is None


# --------------------------------------------------------------------------- contract


def test_1000_records_validate(records: list[dict[str, Any]]) -> None:
    for r in records:
        validate(r, "board_record")


def test_design_records_validate(design_records: list[dict[str, Any]]) -> None:
    for r in design_records:
        validate(r, "board_record")
        assert r["mode"] == "design"
        assert len(r["design_vector"]) == 36


def test_geometry_is_quantised(records: list[dict[str, Any]]) -> None:
    def coords(r: dict[str, Any]) -> list[float]:
        out: list[float] = []
        for c in r["chips"] + r["via_arrays"]:
            out += [c["x0"], c["y0"], c["w"], c["h"]]
        for layer in r["layers"]:
            for s in layer.get("shapes", []) + layer.get("cutouts", []) + layer.get(
                "split_rects", []
            ):
                for k in ("x0", "y0", "w", "h", "cx", "cy", "r", "width"):
                    if k in s:
                        out.append(s[k])
                for p in s.get("points", []):
                    out += p
        return out

    for r in records[:200]:
        for v in coords(r):
            assert round(v / RES) * RES == pytest.approx(v, abs=1e-12), (r["id"], v)


# --------------------------------------------------------------------------- chips


def _gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    dx = max(b["x0"] - (a["x0"] + a["w"]), a["x0"] - (b["x0"] + b["w"]))
    dy = max(b["y0"] - (a["y0"] + a["h"]), a["y0"] - (b["y0"] + b["h"]))
    return max(dx, dy)


def test_chip_constraints_hold_for_all(records: list[dict[str, Any]]) -> None:
    for r in records:
        chips = r["chips"]
        assert 1 <= len(chips) <= 4, r["id"]
        for c in chips:
            assert c["side"] == "top"
            assert c["x0"] >= 3.0 and c["y0"] >= 3.0, r["id"]
            assert c["x0"] + c["w"] <= 97.0 + 1e-9 and c["y0"] + c["h"] <= 97.0 + 1e-9, r["id"]
            short, long_ = sorted((c["w"], c["h"]))
            assert 4.7 - RES <= short <= 20.0 + RES, r["id"]
            assert 1.0 - 1e-9 <= long_ / short <= 2.0 + 2 * RES / 4.7, r["id"]
        for a, b in combinations(chips, 2):
            assert _gap(a, b) >= 2.0 - 1e-9, r["id"]


def test_chip_powers_respect_range_and_total(records: list[dict[str, Any]]) -> None:
    rescaled = 0
    for r in records:
        powers = [c["P"] for c in r["chips"]]
        total = sum(powers)
        assert total <= 6.0 * (1 + 1e-12), r["id"]
        if total < 6.0 * (1 - 1e-9):
            assert all(0.1 <= p <= 3.0 for p in powers), r["id"]
        else:
            rescaled += 1
            assert all(0 < p <= 3.0 for p in powers), r["id"]
    assert rescaled > 0  # with up to 4 chips of up to 3 W, the cap must bind sometimes


def test_chip_count_histogram_is_uniform(records: list[dict[str, Any]]) -> None:
    counts = np.bincount([len(r["chips"]) for r in records], minlength=5)[1:]
    assert chisquare(counts).pvalue > 0.01, counts


def test_via_arrays_sit_under_chips_with_p_0_6(records: list[dict[str, Any]]) -> None:
    n_chips = n_arrays = 0
    for r in records:
        footprints = {(c["x0"], c["y0"], c["w"], c["h"]) for c in r["chips"]}
        for v in r["via_arrays"]:
            assert (v["x0"], v["y0"], v["w"], v["h"]) in footprints, r["id"]
            assert 0.2 <= v["density_per_mm2"] <= 1.0
        n_chips += len(r["chips"])
        n_arrays += len(r["via_arrays"])
    assert binomtest(n_arrays, n_chips, 0.6).pvalue > 0.01


# --------------------------------------------------------------------------- layers, BCs, air


def test_layer_coverages_are_in_range(records: list[dict[str, Any]]) -> None:
    for r in records[:200]:
        l1, l2, l3, l4 = r["layers"]
        assert 0.10 <= coverage(l1["shapes"]) <= 0.60, r["id"]
        assert 2 <= len(l1["shapes"]) <= 8
        assert 1.0 - coverage(l2["cutouts"]) >= 0.7, r["id"]
        assert 0 <= len(l2["cutouts"]) <= 3
        assert l2["plane"] is True
        assert 2 <= len(l3["split_rects"]) <= 3
        assert coverage(l3["split_rects"]) == pytest.approx(1.0)
        assert 0.05 <= coverage(l4["shapes"]) <= 0.40, r["id"]


def test_l3_split_rects_tile_the_board_without_overlap(records: list[dict[str, Any]]) -> None:
    for r in records[:200]:
        rects = r["layers"][2]["split_rects"]
        assert sum(s["w"] * s["h"] for s in rects) == pytest.approx(1.0e4)
        for a, b in combinations(rects, 2):
            assert _gap(a, b) >= -1e-9, r["id"]  # touching, never overlapping


def test_painted_shapes_use_the_v1_families(records: list[dict[str, Any]]) -> None:
    types = []
    for r in records:
        for s in r["layers"][0]["shapes"]:
            types.append(s["type"])
            if s["type"] == "disc":
                assert 3.0 - RES <= s["r"] <= 15.0 + RES
            elif s["type"] == "trace":
                assert 3.2 - RES <= s["width"] <= 6.0 + RES
                pts = s["points"]
                assert 2 <= len(pts) <= 5  # 2-4 segments, fewer if clipping removed one
                for p, q in zip(pts, pts[1:], strict=False):
                    assert p[0] == q[0] or p[1] == q[1], "segments must be axis-aligned"
    frac_rect = types.count("rect") / len(types)
    assert 0.40 < frac_rect < 0.60  # p = 0.50 before coverage rejection


def test_dirichlet_edges(records: list[dict[str, Any]]) -> None:
    counts = np.zeros(3, dtype=int)
    for r in records:
        edges = r["dirichlet_edges"]
        assert len(set(edges)) == len(edges) <= 2
        assert edges == [e for e in EDGES if e in edges], "canonical W, E, S, N order"
        counts[len(edges)] += 1
    for n, p in enumerate((0.5, 0.3, 0.2)):
        assert binomtest(int(counts[n]), N_RECORDS, p).pvalue > 0.01, counts


def test_convection_and_forced_zone(records: list[dict[str, Any]]) -> None:
    n_forced = 0
    for r in records:
        cv = r["convection"]
        assert 5.0 <= cv["h_top"] <= 15.0 and 5.0 <= cv["h_bot"] <= 15.0
        fz = cv["forced_zone"]
        if fz is not None:
            n_forced += 1
            assert 25.0 <= fz["h_top"] <= 50.0
            assert 20.0 - RES <= fz["w"] <= 100.0 and 20.0 - RES <= fz["h"] <= 100.0
            assert fz["x0"] + fz["w"] <= 100.0 + 1e-9 and fz["y0"] + fz["h"] <= 100.0 + 1e-9
    assert binomtest(n_forced, N_RECORDS, 0.3).pvalue > 0.01


def test_stack(records: list[dict[str, Any]]) -> None:
    outer = [r["stack"]["outer_oz"] for r in records]
    assert set(outer) == {1, 2}
    assert binomtest(outer.count(2), N_RECORDS, 0.5).pvalue > 0.01
    for r in records:
        assert r["stack"]["inner_oz"] == 1
        assert 0.81 <= r["stack"]["k_fr4_xy"] <= 1.06
        assert 0.0 <= r["background_via_density_per_mm2"] <= 0.05


# --------------------------------------------------------------------------- coverage helper


def test_coverage_of_simple_shapes() -> None:
    assert coverage([]) == 0.0
    assert coverage([{"type": "rect", "x0": 0, "y0": 0, "w": 100, "h": 100}]) == 1.0
    assert coverage([{"type": "rect", "x0": 0, "y0": 0, "w": 50, "h": 100}]) == 0.5
    disc = coverage([{"type": "disc", "cx": 50, "cy": 50, "r": 10}])
    assert disc == pytest.approx(math.pi * 100 / 1e4, abs=2e-3)
    half_disc = coverage([{"type": "disc", "cx": 0, "cy": 50, "r": 10}])  # clipped by the board
    assert half_disc == pytest.approx(disc / 2, abs=2e-3)
    trace = coverage([{"type": "trace", "points": [[20, 50], [80, 50]], "width": 4}])
    assert trace == pytest.approx((60 * 4 + math.pi * 4) / 1e4, abs=2e-3)  # capsule
    union = coverage([{"type": "rect", "x0": 0, "y0": 0, "w": 50, "h": 100}] * 2)
    assert union == 0.5


# --------------------------------------------------------------------------- design board


def test_design_board_is_deterministic_in_x() -> None:
    x = _feasible_x()
    assert _json(design_board(x)) == _json(design_board(x.copy()))
    y = x.copy()
    y[8] = 0.25
    assert _json(design_board(y)) != _json(design_board(x))


def test_design_board_maps_x_and_validates() -> None:
    x = _feasible_x()
    r = design_board(x)
    validate(r, "board_record")
    assert r["mode"] == "design"
    assert r["design_vector"] == x.tolist()
    assert len(r["chips"]) == 4
    assert len(r["layers"][0]["shapes"]) == 6
    assert all(s["type"] == "rect" for s in r["layers"][0]["shapes"])
    assert len(r["via_arrays"]) == 4
    # chip 0 at x = (0.1, 0.1) sits in the lower-left quarter, chip 3 at (0.9, 0.9) upper-right
    c0, c3 = r["chips"][0], r["chips"][3]
    assert c0["x0"] < 50 and c0["y0"] < 50 and c3["x0"] > 50 and c3["y0"] > 50


def test_design_board_rejects_overlapping_chips() -> None:
    x = _feasible_x()
    x[2:4] = x[0:2]  # chip 1 on top of chip 0
    with pytest.raises(ValueError, match="chip"):
        design_board(x)


@pytest.mark.parametrize(
    "x",
    [np.full(35, 0.5), np.full((6, 6), 0.5), np.full(36, 1.5), np.full(36, -0.1),
     np.full(36, np.nan)],
)  # fmt: skip
def test_design_board_rejects_bad_x(x: np.ndarray) -> None:
    with pytest.raises(ValueError, match="x"):
        design_board(x)


# --------------------------------------------------------------------------- symmetries


@pytest.mark.parametrize("sym", range(8))
def test_symmetry_then_inverse_returns_original(
    sym: int, records: list[dict[str, Any]], design_records: list[dict[str, Any]]
) -> None:
    for r in records[:100] + design_records[:20]:
        t = transform_record(r, sym)
        validate(t, "board_record")
        assert _json(transform_record(t, inverse_sym(sym))) == _json(r), (r["id"], sym)


def test_inverse_table() -> None:
    assert [inverse_sym(k) for k in range(8)] == [0, 3, 2, 1, 4, 5, 6, 7]


@pytest.mark.parametrize(
    "sym,expected",
    [(0, ["W", "S", "E", "N"]), (1, ["S", "E", "N", "W"]), (4, ["E", "S", "W", "N"])],
)
def test_dirichlet_edges_are_mapped(sym: int, expected: list[str]) -> None:
    r = {**sample_board(0, "id"), "dirichlet_edges": ["W", "S", "E", "N"]}
    assert transform_record(r, sym)["dirichlet_edges"] == expected


def test_rotation_is_90_degrees_ccw_about_the_centre() -> None:
    r = sample_board(3, "id")
    t = transform_record(r, 1)
    for c, tc in zip(r["chips"], t["chips"], strict=True):
        cx, cy = c["x0"] + c["w"] / 2, c["y0"] + c["h"] / 2
        tcx, tcy = tc["x0"] + tc["w"] / 2, tc["y0"] + tc["h"] / 2
        assert (tcx, tcy) == pytest.approx((100 - cy, cx))
        assert (tc["w"], tc["h"]) == (c["h"], c["w"])


def test_four_quarter_turns_are_the_identity() -> None:
    r = sample_board(5, "id")
    t = r
    for _ in range(4):
        t = transform_record(t, 1)
    assert _json(t) == _json(r)


def test_symmetry_suffix_tracks_the_net_transform() -> None:
    r = sample_board(7, "id")
    assert transform_record(r, 0) == r
    assert transform_record(r, 1)["id"] == r["id"] + "-sym1"
    assert transform_record(transform_record(r, 1), 1)["id"] == r["id"] + "-sym2"


def test_transform_does_not_mutate_input() -> None:
    r = sample_board(9, "id")
    before = _json(r)
    transform_record(r, 6)
    assert _json(r) == before


@pytest.mark.parametrize("sym", [-1, 8, 1.0, True])
def test_bad_symmetry_index_raises(sym: Any) -> None:
    with pytest.raises(ValueError, match="sym"):
        transform_record(sample_board(0, "id"), sym)


# --------------------------------------------------------------------------- modes and seeds


@pytest.mark.parametrize("mode", sampler.OOD_MODES)
def test_ood_modes_are_not_implemented_yet(mode: str) -> None:
    with pytest.raises(NotImplementedError, match=mode):
        sample_board(0, mode)


def test_ood_modes_come_from_spec_v2() -> None:
    assert len(sampler.OOD_MODES) == 9
    assert "ood_bottom_chips" in sampler.OOD_MODES


@pytest.mark.parametrize("mode", ["ID", "", "ood_99"])
def test_unknown_mode_raises(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        sample_board(0, mode)


@pytest.mark.parametrize("seed", [-1, 1.5, True, "1"])
def test_bad_seed_raises(seed: Any) -> None:
    with pytest.raises(ValueError, match="seed"):
        sample_board(seed, "id")
