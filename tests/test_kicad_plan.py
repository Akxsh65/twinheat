"""Hardware board generation, the parts that run without KiCad (G1.1; CLAUDE.md §14, §7.1, §10.1).

hardware/kicad/plan.py turns a board record into zones, footprints, tracks and vias; gen_board.py
only copies that plan into pcbnew. These tests use TEST values for the fab rules and the
footprint pad geometry (marked below); the real ones come from fab.json and KiCad's library.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from contracts.validate import validate
from data.raster import rasterise
from hardware.kicad import check_outputs, plan
from solvers.fvm import assemble_board, solve_direct
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
BOARDS = ROOT / "hardware" / "boards"
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
DESIGN_TASK = yaml.safe_load((ROOT / "config" / "design_task.yaml").read_text(encoding="utf-8"))


def record(k: int) -> dict:
    return json.loads((BOARDS / f"board{k}.json").read_text(encoding="utf-8"))


@pytest.fixture()
def cfg() -> dict:
    """The repo's fab.json with its nulls replaced by TEST values (not a real fab or part)."""
    c = json.loads((ROOT / "hardware" / "kicad" / "fab.json").read_text(encoding="utf-8"))
    c["resistor"].update(part="TEST PART", rated_W=1.0)
    c["supply_track_width_mm"] = 0.5
    c["fab"].update(name="TEST FAB", min_clearance_mm=0.2, min_track_mm=0.15,
                    min_annular_ring_mm=0.1, min_hole_to_hole_mm=0.25,
                    copper_edge_clearance_mm=0.3)
    return c


# TEST footprint geometry (KiCad's real pads are read by gen_board.py at run time)
R2512 = plan.FootprintSpec("R_2512_test", (plan.Pad("1", -2.9, 0.0, 1.5, 3.4, "rect"),
                                          plan.Pad("2", 2.9, 0.0, 1.5, 3.4, "rect")), 7.8, 4.0)
TP = plan.FootprintSpec("TP_test", (plan.Pad("1", 0.0, 0.0, 2.0, 2.0, "circle"),), 2.5, 2.5)


def make(k: int, cfg: dict, rec: dict | None = None) -> plan.Plan:
    return plan.plan_board(record(k) if rec is None else rec, cfg, R2512, TP)


# ------------------------------------------------------------------------------ the records

@pytest.mark.parametrize("k", [1, 2, 3])
def test_records_are_valid_and_the_solver_runs_on_them(k):
    """The PCB is generated from the SAME records the solver uses (the prompt's goal)."""
    rec = record(k)
    validate(rec, "board_record")
    r = rasterise(rec, SPEC, 64)
    A, b = assemble_board(SPEC, r, rec["dirichlet_edges"])
    theta = solve_direct(A, b)
    assert 0 < theta.max() < 100  # inside the §6.5 envelope without rescaling


def test_records_differ_only_in_pours_and_via_arrays():
    def rest(r):
        r = copy.deepcopy(r)
        r["layers"][0]["shapes"] = None
        r.pop("via_arrays")
        r.pop("id")
        return r
    recs = [record(k) for k in (1, 2, 3)]
    assert rest(recs[0]) == rest(recs[1]) == rest(recs[2])
    assert recs[0]["layers"][0] != recs[1]["layers"][0] != recs[2]["layers"][0]
    assert len({json.dumps(r["via_arrays"]) for r in recs}) == 3


def test_records_use_the_d2_base_stack_and_rig():
    """docs/decisions.md D2: the E17 boards share the design task's fixed parameters."""
    r = record(1)
    assert r["stack"] == DESIGN_TASK["stack"]
    cutouts = DESIGN_TASK["layers"]["l2_cutouts"]
    assert r["layers"][1] == {"layer": 2, "plane": True, "cutouts": cutouts}
    assert r["layers"][2]["split_rects"] == DESIGN_TASK["layers"]["l3_split_rects"]
    assert r["layers"][3]["shapes"] == DESIGN_TASK["layers"]["l4_shapes"]
    assert r["background_via_density_per_mm2"] == DESIGN_TASK["background_via_density_per_mm2"]
    assert r["convection"] == DESIGN_TASK["convection"]
    assert r["dirichlet_edges"] == DESIGN_TASK["dirichlet_edges"] == []


def test_committed_stackup_matches_the_spec_and_records():
    from hardware.kicad import stackup
    path = ROOT / "hardware" / "kicad" / "stackup.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == stackup.build()
    for s in committed.values():
        assert math.isclose(s["total_mm"], 1.6, abs_tol=1e-9)  # CLAUDE.md §7.1: 1.6 mm board


# ------------------------------------------------------------------------------ configuration

def test_the_shipped_config_is_refused_until_its_nulls_are_filled():
    with pytest.raises(plan.ConfigError) as e:
        plan.load_config(ROOT / "hardware" / "kicad" / "fab.json")
    for key in ("resistor.rated_W", "resistor.part", "fab.name", "fab.min_clearance_mm",
                "supply_track_width_mm"):
        assert key in str(e.value)


def test_a_complete_config_loads(cfg, tmp_path):
    p = tmp_path / "fab.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert plan.load_config(p)["resistor"]["rated_W"] == 1.0


# ------------------------------------------------------------------------------ geometry

def polygon_area(poly) -> float:
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1],
                                                                        strict=True)))


def test_disc_is_an_equal_area_64_gon():
    poly = plan.disc_poly(30.0, 40.0, 8.0)
    assert len(poly) == plan.DISC_SIDES == 64
    assert math.isclose(polygon_area(poly), math.pi * 64.0, rel_tol=1e-12)


def test_trace_segments_become_capsules_of_the_given_width():
    s = {"type": "trace", "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 30.0]], "width": 4.0}
    polys = plan.shape_polys(s)
    assert len(polys) == 2
    caps = math.pi * 2.0**2
    assert math.isclose(polygon_area(polys[0]), 30 * 4 + caps, rel_tol=1e-12)
    assert math.isclose(polygon_area(polys[1]), 20 * 4 + caps, rel_tol=1e-12)


def test_kicad_coordinates_flip_y():
    assert plan.to_kicad(5.0, 5.0, 100.0) == (5.0, 95.0)
    assert plan.to_kicad(95.0, 95.0, 100.0) == (95.0, 5.0)


# ------------------------------------------------------------------------------ heaters

def test_resistor_count_keeps_each_at_or_below_half_rated(cfg):
    p = make(1, cfg)
    assert not p.errors, p.errors
    for chip in record(1)["chips"]:
        rs = [h for h in p.heaters if h.chip == record(1)["chips"].index(chip)]
        assert len(rs) == math.ceil(chip["P"] / (0.5 * cfg["resistor"]["rated_W"]))
        assert all(h.nominal_W <= h.max_W + 1e-12 for h in rs)
        assert math.isclose(sum(h.nominal_W for h in rs), chip["P"], rel_tol=1e-12)


def test_heaters_that_do_not_fit_the_chip_footprint_are_errors_not_resized(cfg):
    cfg["resistor"]["rated_W"] = 0.25  # 0.5 W needs 4 resistors at <= 0.125 W: no fit in 3.2 mm
    p = make(1, cfg)
    assert p.errors and any("do not fit" in e for e in p.errors)
    assert not p.buildable


def test_every_heater_has_two_test_pads_and_supply_tracks(cfg):
    p = make(2, cfg)
    for i in range(len(record(2)["chips"])):
        tps = [f for f in p.test_pads if f.chip == i]
        assert sorted(f.pad_nets["1"] for f in tps) == sorted([f"HTR{i + 1}", "GND"])
        assert any(t.net == f"HTR{i + 1}" for t in p.tracks)
    nets = {h.pad_nets["1"] for h in p.heaters} | {h.pad_nets["2"] for h in p.heaters}
    assert nets == {"HTR1", "HTR2", "HTR3", "GND"}


def test_fiducials_are_at_the_four_corners(cfg):
    p = make(1, cfg)
    assert [(f.x, f.y) for f in p.fiducials] == [(5.0, 5.0), (95.0, 5.0), (95.0, 95.0), (5.0, 95.0)]


# ------------------------------------------------------------------------------ vias

def clear_of(px, py, item, clearance) -> bool:
    return plan.distance_to_item(px, py, item) >= clearance - 1e-9


@pytest.mark.parametrize("k", [1, 2, 3])
def test_kept_vias_respect_supply_copper_pads_edges_and_fiducials(cfg, k):
    p = make(k, cfg)
    r_via = cfg["via"]["pad_mm"] / 2
    clr = cfg["fab"]["min_clearance_mm"]
    for v in p.vias:
        for item in p.supply_copper():  # HTR+ pads, tracks, test pads
            assert clear_of(v.x, v.y, item, clr + r_via), (v, item)
        for item in p.all_pads():  # no via in or touching any pad
            assert clear_of(v.x, v.y, item, r_via), (v, item)
        edge = cfg["fab"]["copper_edge_clearance_mm"] + r_via
        assert edge <= v.x <= 100 - edge and edge <= v.y <= 100 - edge
        for f in p.fiducials:
            assert math.hypot(v.x - f.x, v.y - f.y) >= 1.0 + clr + r_via


def test_via_arrays_report_requested_and_achieved_density(cfg):
    p = make(2, cfg)
    arrays = [s for s in p.via_stats if s["source"].startswith("via_arrays")]
    assert len(arrays) == 3
    for s in arrays:
        # the grid is the requested density (array + background) up to rounding of the counts
        assert abs(s["grid_density"] - s["requested_density"]) <= 0.15 * s["requested_density"]
        assert s["kept"] + s["dropped"] == s["placed"]
        assert s["achieved_density"] == pytest.approx(s["kept"] / s["area_mm2"])
    assert any(s["dropped"] > 0 for s in arrays)  # heater copper sits inside the arrays
    prefixes = ("via", "L1", "heater", "fiducial", "L3", "record")
    assert all(d.startswith(prefixes) for d in p.deviations)


def test_too_dense_a_via_array_is_unmanufacturable(cfg):
    rec = record(1)
    rec["via_arrays"][0]["density_per_mm2"] = 4.0  # 0.5 mm pitch: holes 0.2 mm apart < 0.25
    p = make(1, cfg, rec)
    assert any("via" in e and "hole-to-hole" in e for e in p.errors)


def test_annular_ring_below_the_fab_minimum_is_an_error(cfg):
    cfg["fab"]["min_annular_ring_mm"] = 0.2  # 0.3 mm drill in a 0.6 mm pad leaves 0.15 mm
    p = make(1, cfg)
    assert any("annular" in e for e in p.errors)


def test_l1_pour_cleared_around_supply_copper_is_reported(cfg):
    p = make(2, cfg)  # board 2's pours surround heaters 1 and 2
    cleared = [d for d in p.deviations if d.startswith("L1") and "cleared" in d]
    assert cleared


# ------------------------------------------------------------------------------ CSVs + checker

def test_csvs_round_trip_and_the_checker_accepts_them(cfg, tmp_path):
    p = make(1, cfg)
    plan.write_csvs(p, tmp_path)
    with (tmp_path / "heaters.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["ref", "x_mm", "y_mm", "nominal_W", "max_W"]
    assert len(rows) == len(p.heaters)
    assert check_outputs.check_board(record(1), cfg, tmp_path) == []


@pytest.mark.parametrize("tamper", ["move_heater", "drop_fiducial", "power", "max"])
def test_the_checker_catches_csvs_that_do_not_match_the_record(cfg, tmp_path, tamper):
    plan.write_csvs(make(1, cfg), tmp_path)
    name = "fiducials.csv" if tamper == "drop_fiducial" else "heaters.csv"
    path = tmp_path / name
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if tamper == "move_heater":
        rows[0]["x_mm"] = str(float(rows[0]["x_mm"]) + 10)
    elif tamper == "drop_fiducial":
        rows.pop()
    elif tamper == "power":
        rows[0]["nominal_W"] = str(float(rows[0]["nominal_W"]) * 1.1)
    else:
        rows[0]["max_W"] = "9"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert check_outputs.check_board(record(1), cfg, tmp_path)


def test_the_checker_compares_the_pcb_footprints_with_the_csvs(cfg, tmp_path):
    p = make(1, cfg)
    plan.write_csvs(p, tmp_path)
    fps = [(h.ref, *plan.to_kicad(h.x, h.y, 100.0)) for h in p.heaters]
    fps += [(f.ref, *plan.to_kicad(f.x, f.y, 100.0)) for f in p.fiducials]
    body = "\n".join(f'  (footprint "X" (layer "F.Cu")\n    (at {x:.4f} {y:.4f} 0)\n'
                     f'    (property "Reference" "{ref}" (at 0 0 0))\n  )' for ref, x, y in fps)
    (tmp_path / "board.kicad_pcb").write_text(f"(kicad_pcb\n{body}\n)\n", encoding="utf-8")
    assert check_outputs.check_board(record(1), cfg, tmp_path) == []
    moved = body.replace(f"(at {fps[0][1]:.4f}", f"(at {fps[0][1] + 1:.4f}", 1)
    (tmp_path / "board.kicad_pcb").write_text(f"(kicad_pcb\n{moved}\n)\n", encoding="utf-8")
    assert check_outputs.check_board(record(1), cfg, tmp_path)


def test_grid_points_are_deterministic(cfg):
    a, b = make(3, cfg), make(3, cfg)
    assert [(v.x, v.y) for v in a.vias] == [(v.x, v.y) for v in b.vias]
    assert np.isfinite([v.x for v in a.vias]).all()
