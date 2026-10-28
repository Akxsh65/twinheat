"""OOD generators (prompt M1.2; CLAUDE.md §6.7, §7.4; docs/decisions.md "OOD sets v2").

Each mode must move its own axis outside the training range and keep a sample of the other axes
inside it. 200 records per mode.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contracts.validate import validate
from data.raster import rasterise
from data.sampler import OOD_MODES, sample_board
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N_RECORDS = 200
Record = dict[str, Any]


def _l1(r: Record) -> list[dict[str, Any]]:
    return r["layers"][0]["shapes"]


def _diagonal(trace: dict[str, Any]) -> bool:
    pts = trace["points"]
    return any(a[0] != b[0] and a[1] != b[1] for a, b in zip(pts, pts[1:], strict=False))


def _corners(r: dict[str, Any]) -> set[tuple[float, float]]:
    x0, y0, x1, y1 = r["x0"], r["y0"], round(r["x0"] + r["w"], 3), round(r["y0"] + r["h"], 3)
    return {(x0, y0), (x1, y0), (x0, y1), (x1, y1)}


def _l_pair(shapes: list[dict[str, Any]]) -> bool:
    """Two rects meeting at a shared corner with equal arm width (to the 0.001 mm grid)."""
    rects = [s for s in shapes if s["type"] == "rect"]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            if _corners(a) & _corners(b) and abs(a["h"] - b["w"]) <= 2e-3:
                return True
            if _corners(a) & _corners(b) and abs(a["w"] - b["h"]) <= 2e-3:
                return True
    return False


# Training (ID) ranges of every axis, CLAUDE.md §6.5–§6.6, §7.4. Each returns True if r is ID there.
ID_AXES: dict[str, Callable[[Record], bool]] = {
    "chip_count": lambda r: 1 <= len(r["chips"]) <= 4,
    "chip_power": lambda r: max(c["P"] for c in r["chips"]) <= 3.0 + 1e-12
    and sum(c["P"] for c in r["chips"]) <= 6.0 + 1e-9,
    "geometry": lambda r: all(s["type"] in ("rect", "disc", "trace") for s in _l1(r))
    and not any(s["type"] == "trace" and _diagonal(s) for s in _l1(r))
    and not _l_pair(_l1(r)),
    "thin": lambda r: all(s["width"] >= 3.2 for s in _l1(r) if s["type"] == "trace"),
    "boundary": lambda r: len(r["dirichlet_edges"]) <= 2,
    "convection": lambda r: 5.0 <= r["convection"]["h_top"] <= 15.0
    and 5.0 <= r["convection"]["h_bot"] <= 15.0
    and (r["convection"]["forced_zone"] is None
         or 25.0 <= r["convection"]["forced_zone"]["h_top"] <= 50.0),
    "side": lambda r: all(c["side"] == "top" for c in r["chips"]),
    "copper": lambda r: r["stack"]["outer_oz"] in (1, 2) and r["stack"]["inner_oz"] == 1,
    "vias": lambda r: all(a["density_per_mm2"] <= 1.0 for a in r["via_arrays"]),
}


def _ood3(r: Record) -> bool:
    """L-shapes (rects) and 45° traces only; no disc, no Manhattan trace; something non-ID."""
    shapes = _l1(r)
    if any(s["type"] == "disc" or (s["type"] == "trace" and not _diagonal(s)) for s in shapes):
        return False
    return any(s["type"] == "trace" for s in shapes) or _l_pair(shapes)


# mode -> (axis it moves, predicate "outside the training range")
MOVED: dict[str, tuple[str, Callable[[Record], bool]]] = {
    "ood_1": ("chip_count", lambda r: 5 <= len(r["chips"]) <= 8),
    "ood_2": ("chip_power", lambda r: max(c["P"] for c in r["chips"]) > 3.0
              and sum(c["P"] for c in r["chips"]) <= 12.0 + 1e-9),
    "ood_3": ("geometry", _ood3),
    "ood_4": ("thin", lambda r: any(s["type"] == "trace" for s in _l1(r))
              and all(0.8 <= s["width"] <= 2.0 for s in _l1(r) if s["type"] == "trace")),
    "ood_6": ("boundary", lambda r: len(r["dirichlet_edges"]) in (3, 4)),
    "ood_7": ("convection", lambda r: 2.0 <= r["convection"]["h_top"] <= 5.0
              and 2.0 <= r["convection"]["h_bot"] <= 5.0
              and (r["convection"]["forced_zone"] is None
                   or 50.0 <= r["convection"]["forced_zone"]["h_top"] <= 100.0)),
    "ood_bottom_chips": ("side", lambda r: all(c["side"] == "bottom" for c in r["chips"])),
    "ood_3oz": ("copper", lambda r: r["stack"]["outer_oz"] == 3 and r["stack"]["inner_oz"] == 1),
    "ood_dense_vias": ("vias", lambda r: len(r["via_arrays"]) >= 1
                       and all(1.0 < a["density_per_mm2"] <= 1.5 for a in r["via_arrays"])),
}


@pytest.fixture(scope="module", params=sorted(MOVED))
def mode_records(request: pytest.FixtureRequest) -> tuple[str, list[Record]]:
    mode = request.param
    return mode, [sample_board(seed, mode) for seed in range(N_RECORDS)]


def test_every_spec_mode_has_a_generator_and_a_test() -> None:
    assert set(OOD_MODES) == set(MOVED)


def test_moved_axis_is_outside_training(mode_records: tuple[str, list[Record]]) -> None:
    mode, records = mode_records
    axis, outside = MOVED[mode]
    for r in records:
        assert r["mode"] == mode and r["id"].startswith(mode)
        assert outside(r), f"{r['id']}: axis {axis} is not out of distribution"
        assert not ID_AXES[axis](r), f"{r['id']}: axis {axis} still looks ID"


def test_other_axes_stay_in_training(mode_records: tuple[str, list[Record]]) -> None:
    mode, records = mode_records
    moved = MOVED[mode][0]
    for r in records:
        for axis, inside in ID_AXES.items():
            if axis != moved:
                assert inside(r), f"{r['id']}: axis {axis} left training too ({mode} moves {moved})"


def test_records_validate_and_rasterise(mode_records: tuple[str, list[Record]]) -> None:
    mode, records = mode_records
    for r in records[:40]:
        validate(r, "board_record")
        ras = rasterise(r, SPEC, 32)
        total = sum(c["P"] for c in r["chips"])
        got = (ras.q.sum() + ras.q_bot.sum()) * ras.dx**2
        assert got == pytest.approx(total, rel=1e-12), r["id"]
        assert np.isfinite(ras.phi).all() and ras.phi.max() <= 1.0


def test_same_seed_gives_identical_json() -> None:
    for mode in MOVED:
        a, b = sample_board(7, mode), sample_board(7, mode)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), mode


def test_ood_3_traces_are_exactly_45_degrees() -> None:
    for seed in range(N_RECORDS):
        for s in _l1(sample_board(seed, "ood_3")):
            if s["type"] != "trace":
                continue
            for a, b in zip(s["points"], s["points"][1:], strict=False):
                dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
                assert dx == 0 or dy == 0 or dx == pytest.approx(dy, abs=1e-12), s


def test_ood_3_mixes_both_families() -> None:
    kinds = {"L": 0, "45": 0}
    for seed in range(N_RECORDS):
        shapes = _l1(sample_board(seed, "ood_3"))
        kinds["L"] += any(s["type"] == "rect" for s in shapes)
        kinds["45"] += any(s["type"] == "trace" for s in shapes)
    assert kinds["L"] > N_RECORDS // 4 and kinds["45"] > N_RECORDS // 4, kinds


def test_ood_1_covers_5_to_8_chips() -> None:
    counts = {len(sample_board(seed, "ood_1")["chips"]) for seed in range(N_RECORDS)}
    assert counts == {5, 6, 7, 8}
