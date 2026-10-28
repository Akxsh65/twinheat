"""Board sampler v0: seeds and design vectors -> board records (CLAUDE.md §6.5, §6.6, §7.4, §10.1).

sample_board(seed, mode) draws an ID board ("id") or a design-space board ("design") with
numpy Generator(PCG64(seed)); design_board(x) maps x in [0, 1]^36 to a board; transform_record
applies one of the 8 square symmetries. No rasterisation and no solver calls: the envelope rule
of §6.5 (rescale powers after solving) belongs to data/generate.py.

Units: records use mm for geometry, W for power, W/(m²·K) for convection and vias/mm² for via
density (CLAUDE.md §10.1). Geometry is rounded to sampler.v0_defaults.geometry_resolution so
symmetries round-trip exactly.

Numbers come from config/spec_v1.yaml (the §6.5 painted copper, chips, convection), spec_v2.yaml
(the §7.4 additions and sampler.v0_defaults, which are UNSURE defaults M must confirm) and
config/design_task.yaml (decision D2, docs/decisions.md).

Layer coverage checks (§6.5 mean phi, §7.1 phi_2 and phi_3, §7.4 L4 5-40 %) use coverage(), a
point-grid estimate on coverage_grid_points² points; Phase 1 replaces it with data/raster.py.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from twinheat import spec as specmod

__all__ = [
    "OOD_MODES",
    "InfeasibleDesignError",
    "coverage",
    "design_board",
    "inverse_sym",
    "sample_board",
    "transform_record",
]

ROOT = Path(__file__).resolve().parents[1]
DESIGN_TASK_PATH = ROOT / "config" / "design_task.yaml"

_V1 = specmod.load(ROOT / "config" / "spec_v1.yaml")
_V2 = specmod.load(ROOT / "config" / "spec_v2.yaml")
_S1 = _V1.sampler
_S2 = _V2.sampler
_DEF = _S2.v0_defaults
_O2 = _V2.ood


def _o1(name: str, param: str) -> specmod.Quantity:
    """A §6.7 OOD range from spec_v1.yaml (CLAUDE.md §7.4: v2 reuses OOD-1…7)."""
    return _V1.ood[name].params[param]


OOD_MODES: tuple[str, ...] = tuple(_V2.ood.names.value)
L_MM = _V2.geometry.L.value * 1.0e3  # board side in record units (mm)
_EDGE_ORDER = ("W", "E", "S", "N")
_MAX_TRIES = int(_DEF.resample_max_tries.value)

_RES = _DEF.geometry_resolution.value
_NDEC = round(-math.log10(_RES))
if not math.isclose(10.0**-_NDEC, _RES, rel_tol=1e-12):
    raise ValueError(f"geometry_resolution must be a power of ten in mm, got {_RES}")

_G = int(_DEF.coverage_grid_points.value)
_AXIS = (np.arange(_G) + 0.5) / _G * L_MM
_X, _Y = np.meshgrid(_AXIS, _AXIS)  # [y, x], index 0 = south / west edge

# The 8 square symmetries as 2x2 maps of centred coordinates: sym = r + 4 f is a mirror x -> L - x
# (if f = 1) followed by r quarter turns counter-clockwise. 90° CCW maps W -> S -> E -> N -> W.
_ROT = np.array([[0, -1], [1, 0]])
_FLIP = np.array([[-1, 0], [0, 1]])
_SYMS = [np.linalg.matrix_power(_ROT, r) @ np.linalg.matrix_power(_FLIP, f)
         for f in (0, 1) for r in range(4)]  # fmt: skip
_EDGE_NORMALS = {"W": (-1, 0), "E": (1, 0), "S": (0, -1), "N": (0, 1)}
_DIRS8 = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))  # 45° steps, CCW
_SYM_SUFFIX = re.compile(r"^(.*)-sym([1-7])$")


class InfeasibleDesignError(ValueError):
    """A design vector whose chips break the §6.5 placement rules (gap or edge margin)."""


# ---------------------------------------------------------------------------- small helpers


def _q(v: float) -> float:
    """Round a length (mm) to the geometry grid."""
    return round(float(v), _NDEC) + 0.0


def _q_floor(v: float) -> float:
    """Round a length (mm) down to the geometry grid (keeps lower-left corners inside bounds)."""
    scale = 10.0**_NDEC
    return round(math.floor(float(v) * scale + 1e-6) / scale, _NDEC) + 0.0


def _num(v: float) -> int | float:
    """1.0 -> 1 for counts such as copper ounces, so records read like CLAUDE.md §10.1."""
    return int(v) if float(v).is_integer() else float(v)


def _uniform(rng: np.random.Generator, q: specmod.Quantity) -> float:
    lo, hi = q.value
    return float(rng.uniform(lo, hi))


def _uniform_int(rng: np.random.Generator, q: specmod.Quantity) -> int:
    lo, hi = q.value
    return int(rng.integers(int(lo), int(hi) + 1))


def _loguniform(rng: np.random.Generator, q: specmod.Quantity) -> float:
    lo, hi = q.value
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _categorical(rng: np.random.Generator, values: Sequence[Any], p: Sequence[float]) -> Any:
    return values[int(rng.choice(len(values), p=p))]


def _bernoulli(rng: np.random.Generator, q: specmod.Quantity) -> bool:
    return bool(rng.random() < q.value)


def _rect(x0: float, y0: float, x1: float, y1: float) -> dict[str, Any]:
    """Rect record from corners (mm), clipped to the board and rounded to the grid."""
    x0, y0 = _q(max(0.0, x0)), _q(max(0.0, y0))
    x1, y1 = _q(min(L_MM, x1)), _q(min(L_MM, y1))
    return {"type": "rect", "x0": x0, "y0": y0, "w": _q(x1 - x0), "h": _q(y1 - y0)}


def _in_range(v: float, q: specmod.Quantity) -> bool:
    lo, hi = q.value
    return lo <= v <= hi


# ---------------------------------------------------------------------------- coverage


def _mask(shape: dict[str, Any]) -> np.ndarray:
    t = shape["type"]
    if t == "rect":
        return (
            (_X >= shape["x0"]) & (_X < shape["x0"] + shape["w"])
            & (_Y >= shape["y0"]) & (_Y < shape["y0"] + shape["h"])
        )  # fmt: skip
    if t == "disc":
        return (_X - shape["cx"]) ** 2 + (_Y - shape["cy"]) ** 2 <= shape["r"] ** 2
    if t == "trace":  # capsule rule (v0_defaults.trace_cap): within width/2 of a segment
        half = shape["width"] / 2
        out = np.zeros_like(_X, dtype=bool)
        pts = shape["points"]
        for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            s = 0.0 if seg2 == 0 else np.clip(((_X - ax) * dx + (_Y - ay) * dy) / seg2, 0, 1)
            out |= (_X - (ax + s * dx)) ** 2 + (_Y - (ay + s * dy)) ** 2 <= half * half
        return out
    raise ValueError(f"unknown shape type {t!r} in {shape}")


def coverage(shapes: Sequence[dict[str, Any]]) -> float:
    """Fraction of the board covered by the union of shapes (point-grid estimate, in [0, 1])."""
    covered = np.zeros_like(_X, dtype=bool)
    for shape in shapes:
        covered |= _mask(shape)
    return float(covered.mean())


# ---------------------------------------------------------------------------- copper layers


def _trace(rng: np.random.Generator, width_q: specmod.Quantity, diagonal: bool) -> dict[str, Any]:
    """A §6.5 trace: Manhattan, or (OOD-3) turning ±45° between the 8 compass directions."""
    for _ in range(_MAX_TRIES):
        n_seg = _uniform_int(rng, _S1.trace_segment_count)
        width = _q(_uniform(rng, width_q))
        x, y = (_q(v) for v in rng.uniform(0, L_MM, size=2))
        points = [[x, y]]
        if diagonal:
            d = int(rng.integers(len(_DIRS8)))
            for i in range(n_seg):
                if i > 0:
                    ccw = _bernoulli(rng, _DEF.trace_turn_ccw_probability)
                    d = (d + (1 if ccw else -1)) % len(_DIRS8)
                ux, uy = _DIRS8[d]
                length = _uniform(rng, _S1.trace_segment_length)
                c = length / math.sqrt(2) if ux and uy else length  # step per axis
                # shorten to stay on the board: clamping each axis separately would bend the 45°
                room = min((L_MM - x if ux > 0 else x) if ux else math.inf,
                           (L_MM - y if uy > 0 else y) if uy else math.inf)  # fmt: skip
                c = _q_floor(min(c, room))
                x, y = _q(x + ux * c), _q(y + uy * c)
                if [x, y] != points[-1]:
                    points.append([x, y])
        else:
            horizontal = _bernoulli(rng, _DEF.trace_first_horizontal_probability)
            sign = 1.0 if rng.random() < 0.5 else -1.0  # the board has no preferred direction
            for i in range(n_seg):
                if i > 0:  # turn +-90 degrees
                    ccw = _bernoulli(rng, _DEF.trace_turn_ccw_probability)
                    sign = sign * (1.0 if ccw == horizontal else -1.0)
                    horizontal = not horizontal
                length = _uniform(rng, _S1.trace_segment_length)
                if horizontal:
                    x = _q(min(L_MM, max(0.0, x + sign * length)))
                else:
                    y = _q(min(L_MM, max(0.0, y + sign * length)))
                if [x, y] != points[-1]:
                    points.append([x, y])
        # a 45° trace whose diagonal segments were all shortened to nothing is redrawn
        has_diagonal = any(a[0] != b[0] and a[1] != b[1]
                           for a, b in zip(points, points[1:], strict=False))  # fmt: skip
        if len(points) >= 2 and (has_diagonal or not diagonal):
            return {"type": "trace", "points": points, "width": width}
    raise RuntimeError(f"no non-degenerate trace in {_MAX_TRIES} tries")


def _l_shape(rng: np.random.Generator) -> list[dict[str, Any]]:
    """OOD-3 L-shape: two arms of width t along two sides of a §6.5 box, meeting at a corner."""
    for _ in range(_MAX_TRIES):
        w, h = _uniform(rng, _S1.rect_side), _uniform(rng, _S1.rect_side)
        cx, cy = rng.uniform(0, L_MM, size=2)
        t = _uniform(rng, _O2.l_shape_arm_width)
        corner = int(rng.integers(4))  # 0 SW, 1 SE, 2 NE, 3 NW
        if t >= min(w, h):
            continue
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        south, west = corner in (0, 1), corner in (0, 3)
        arm_x = (x0, y0, x1, y0 + t) if south else (x0, y1 - t, x1, y1)
        arm_y = (x0, y0, x0 + t, y1) if west else (x1 - t, y0, x1, y1)
        arms = [r for r in (_rect(*arm_x), _rect(*arm_y)) if r["w"] > 0 and r["h"] > 0]
        if len(arms) == 2:  # one arm clipped away would leave a plain (ID) rect
            return arms
    raise RuntimeError(f"no L-shape with both arms on the board in {_MAX_TRIES} tries")


def _painted_shape(
    rng: np.random.Generator,
    family: specmod.Quantity | None = None,
    trace_width: specmod.Quantity | None = None,
) -> list[dict[str, Any]]:
    """One painted region as record shapes: §6.5 rect, disc or Manhattan trace (centre uniform,
    clipped), or with an OOD-3 family an L-shape (two rects) or a 45° trace."""
    fam = family or _S1.painted_region_type
    kind = _categorical(rng, fam.value, fam.p)
    if kind == "rect":
        w, h = _uniform(rng, _S1.rect_side), _uniform(rng, _S1.rect_side)
        cx, cy = rng.uniform(0, L_MM, size=2)
        return [_rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]
    if kind == "disc":
        cx, cy = rng.uniform(0, L_MM, size=2)
        r = _q(_uniform(rng, _S1.disc_radius))
        return [{"type": "disc", "cx": _q(cx), "cy": _q(cy), "r": r}]
    if kind == "trace":
        return [_trace(rng, trace_width or _S1.trace_width, diagonal=False)]
    if kind == "L_shape":
        return _l_shape(rng)
    if kind == "trace_45deg":
        return [_trace(rng, _S1.trace_width, diagonal=True)]
    raise ValueError(f"unknown painted region type {kind!r}")


def _painted_layer(
    rng: np.random.Generator,
    count: specmod.Quantity,
    phi: specmod.Quantity,
    name: str,
    family: specmod.Quantity | None = None,
    trace_width: specmod.Quantity | None = None,
) -> list[dict[str, Any]]:
    """Painted regions until the layer coverage lies in phi's range (§6.5 'otherwise resample').

    With trace_width (OOD-4) the layer must also hold at least one trace.
    """
    for _ in range(_MAX_TRIES):
        shapes = [s for _ in range(_uniform_int(rng, count))
                  for s in _painted_shape(rng, family, trace_width)]  # fmt: skip
        if trace_width is not None and not any(s["type"] == "trace" for s in shapes):
            continue
        if _in_range(coverage(shapes), phi):
            return shapes
    raise RuntimeError(f"{name}: no coverage in {phi.value} after {_MAX_TRIES} tries")


def _l2_cutouts(rng: np.random.Generator) -> list[dict[str, Any]]:
    phi_min = _V2.stack.layers[1].phi.value[0]
    for _ in range(_MAX_TRIES):
        cutouts = []
        for _ in range(_uniform_int(rng, _S2.l2_cutout_count)):
            w, h = _uniform(rng, _DEF.l2_cutout_side), _uniform(rng, _DEF.l2_cutout_side)
            cx, cy = rng.uniform(0, L_MM, size=2)
            cutouts.append(_rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        if 1.0 - coverage(cutouts) >= phi_min:
            return cutouts
    raise RuntimeError(f"L2: no cut-outs with phi_2 >= {phi_min} after {_MAX_TRIES} tries")


def _l3_split_rects(rng: np.random.Generator) -> list[dict[str, Any]]:
    """2 or 3 rectangles tiling the board: one full cut, then optionally one half cut again."""
    n = _uniform_int(rng, _S2.l3_split_rect_count)
    cut_x = _bernoulli(rng, _DEF.l3_split_axis_x_probability)
    c = _q(_uniform(rng, _DEF.l3_split_position))
    halves = [(0.0, 0.0, c, L_MM), (c, 0.0, L_MM, L_MM)] if cut_x else [
        (0.0, 0.0, L_MM, c), (0.0, c, L_MM, L_MM)]  # fmt: skip
    if n == 3:
        k = int(rng.integers(2))  # which half is cut again; uniform
        x0, y0, x1, y1 = halves.pop(k)
        d = _q(_uniform(rng, _DEF.l3_split_position))
        parts = [(x0, y0, x1, d), (x0, d, x1, y1)] if cut_x else [(x0, y0, d, y1), (d, y0, x1, y1)]
        halves[k:k] = parts
    return [_rect(*r) for r in halves]


# ---------------------------------------------------------------------------- chips


def _chip_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Clearance (mm) between two rectangles: > 0 apart, <= 0 touching or overlapping."""
    dx = max(b["x0"] - (a["x0"] + a["w"]), a["x0"] - (b["x0"] + b["w"]))
    dy = max(b["y0"] - (a["y0"] + a["h"]), a["y0"] - (b["y0"] + b["h"]))
    return max(dx, dy)


def _chip_violation(chips: Sequence[dict[str, Any]]) -> str | None:
    """None if chips meet the §6.5 edge margin and gap, else a message naming the problem."""
    margin, gap = _S1.chip_edge_margin_min.value, _S1.chip_gap_min.value
    for i, c in enumerate(chips):
        if (c["x0"] < margin or c["y0"] < margin
                or c["x0"] + c["w"] > L_MM - margin + 1e-9
                or c["y0"] + c["h"] > L_MM - margin + 1e-9):  # fmt: skip
            return f"chip {i} is closer than {margin} mm to an edge"
    for i in range(len(chips)):
        for j in range(i + 1, len(chips)):
            g = _chip_gap(chips[i], chips[j])
            if g < gap - 1e-9:
                return f"chips {i} and {j} are {g:.3f} mm apart (< {gap} mm)"
    return None


def _chip_layout(
    rng: np.random.Generator,
    count: specmod.Quantity | None = None,
    power: specmod.Quantity | None = None,
    cap_q: specmod.Quantity | None = None,
) -> list[dict[str, Any]]:
    """§6.5 chips by rejection: 100 tries of a whole layout, then resample the count.

    count, power (log-uniform per chip) and cap_q (total) default to ID; OOD-1 and OOD-2 move them.
    """
    margin = _S1.chip_edge_margin_min.value
    count = count or _S2.chip_count
    power = power or _S2.chip_power
    cap_q = cap_q or _S2.total_power_max
    for _ in range(_MAX_TRIES):
        n = _uniform_int(rng, count)
        for _ in range(int(_S1.chip_placement_max_tries.value)):
            chips = []
            for _ in range(n):
                short = _uniform(rng, _S1.chip_short_side)
                long_ = short * _uniform(rng, _S1.chip_aspect)
                along_x = _bernoulli(rng, _DEF.chip_long_axis_x_probability)
                w, h = (_q(long_), _q(short)) if along_x else (_q(short), _q(long_))
                x0 = _q_floor(rng.uniform(margin, L_MM - margin - w))
                y0 = _q_floor(rng.uniform(margin, L_MM - margin - h))
                chips.append({"x0": x0, "y0": y0, "w": w, "h": h, "P": 0.0, "side": "top"})
            if _chip_violation(chips) is None:
                powers = [_loguniform(rng, power) for _ in chips]
                total, cap = sum(powers), cap_q.value
                scale = cap / total if total > cap else 1.0  # §6.5: rescale all if exceeded
                for c, p in zip(chips, powers, strict=True):
                    c["P"] = p * scale
                return chips
    raise RuntimeError(f"no valid chip layout after {_MAX_TRIES} chip counts")


def _via_arrays(
    rng: np.random.Generator, chips: Sequence[dict[str, Any]], dense: bool = False
) -> list[dict[str, Any]]:
    """A via array under each chip with p = 0.6 (§7.4). dense (ood_dense_vias): density on
    (1.0, 1.5] vias/mm² (spec_v2 ood.dense_via_density) and at least one array on the board."""
    for _ in range(_MAX_TRIES):
        out = []
        for c in chips:
            if _bernoulli(rng, _S2.thermal_via_probability):
                if dense:
                    lo, hi = _O2.dense_via_density.value
                    d = hi - float(rng.uniform(0.0, hi - lo))  # (lo, hi]: strictly above lo
                else:
                    d = _uniform(rng, _S2.thermal_via_density)
                out.append({"x0": c["x0"], "y0": c["y0"], "w": c["w"], "h": c["h"],
                            "density_per_mm2": d})  # fmt: skip
        if out or not dense:
            return out
    raise RuntimeError(f"no via array drawn in {_MAX_TRIES} tries")


# ---------------------------------------------------------------------------- whole boards


def _convection(rng: np.random.Generator, ood: bool = False) -> dict[str, Any]:
    """§6.5 convection; ood (OOD-7) draws natural h from 2–5 and forced h from 50–100."""
    natural = _o1("ood_7", "h_natural") if ood else None
    forced_q = _o1("ood_7", "h_forced") if ood else _S1.forced_zone_h_top
    out: dict[str, Any] = {
        "h_top": _uniform(rng, natural or _S1.h_top_base),
        "h_bot": _uniform(rng, natural or _S1.h_bot_base),
        "forced_zone": None,
    }
    if _bernoulli(rng, _S1.forced_zone_probability):
        w, h = _q(_uniform(rng, _S1.forced_zone_side)), _q(_uniform(rng, _S1.forced_zone_side))
        out["forced_zone"] = {
            "x0": _q_floor(rng.uniform(0, L_MM - w)),
            "y0": _q_floor(rng.uniform(0, L_MM - h)),
            "w": w,
            "h": h,
            "h_top": _uniform(rng, forced_q),
        }
    return out


def _dirichlet_edges(rng: np.random.Generator, ood: bool = False) -> list[str]:
    """§6.6 heat-sunk edges; ood (OOD-6): 3 or 4 whole edges, uniform."""
    q = _V2.boundary.n_dirichlet_edges
    if ood:
        n = _uniform_int(rng, _o1("ood_6", "n_dirichlet_edges"))
    else:
        n = int(_categorical(rng, q.value, q.p))
    picked = rng.choice(len(_EDGE_ORDER), size=n, replace=False)
    return [_EDGE_ORDER[i] for i in sorted(int(i) for i in picked)]


def _random_board(seed: int, rng: np.random.Generator, mode: str = "id") -> dict[str, Any]:
    """An "id" board, or an OOD board that moves exactly one axis (spec_v2 ood.names):

    ood_1 chip count 5–8 · ood_2 chip power 3–6 W each, total <= 12 W · ood_3 L1 regions are
    L-shapes or 45° traces · ood_4 L1 traces 0.8–2 mm wide (at least one) · ood_6 3–4 heat-sunk
    edges · ood_7 natural h 2–5, forced 50–100 · ood_bottom_chips every chip on the bottom ·
    ood_3oz outer copper 3 oz · ood_dense_vias arrays on (1, 1.5] vias/mm², at least one.
    Ranges: CLAUDE.md §6.7 via spec_v1 ood; choices: docs/decisions.md "OOD sets v2".
    """
    if mode == "ood_3oz":
        outer = _O2.ood_3oz_outer_oz.value
    else:
        outer = _categorical(rng, _S2.outer_oz.value, _DEF.outer_oz_p.value)
    stack = {
        "outer_oz": _num(outer),
        "inner_oz": _num(_S2.inner_oz.value),
        "k_fr4_xy": _uniform(rng, _S1.k_fr4_xy),
    }
    l1 = _painted_layer(
        rng, _S1.painted_region_count, _V2.stack.layers[0].phi, "L1",
        family=_O2.ood_3_shape_types if mode == "ood_3" else None,
        trace_width=_o1("ood_4", "trace_width") if mode == "ood_4" else None,
    )  # fmt: skip
    cutouts = _l2_cutouts(rng)
    splits = _l3_split_rects(rng)
    l4 = _painted_layer(rng, _DEF.l4_region_count, _V2.stack.layers[3].phi, "L4")
    if mode == "ood_1":
        chips = _chip_layout(rng, count=_o1("ood_1", "chip_count"))
    elif mode == "ood_2":
        chips = _chip_layout(rng, power=_o1("ood_2", "chip_power"),
                             cap_q=_o1("ood_2", "total_power_max"))  # fmt: skip
    else:
        chips = _chip_layout(rng)
    if mode == "ood_bottom_chips":
        if _O2.bottom_chips_side.value != "all":
            raise ValueError(f"bottom_chips_side {_O2.bottom_chips_side.value!r} not implemented")
        for c in chips:
            c["side"] = "bottom"
    vias = _via_arrays(rng, chips, dense=mode == "ood_dense_vias")
    assert _DEF.background_via_density_dist.value == "uniform"
    background = _uniform(rng, _S2.background_via_density)
    return {
        "schema_version": "1.0",
        "id": f"{mode}-{seed:010d}",
        "seed": seed,
        "mode": mode,
        "stack": stack,
        "layers": [
            {"layer": 1, "shapes": l1},
            {"layer": 2, "plane": True, "cutouts": cutouts},
            {"layer": 3, "split_rects": splits},
            {"layer": 4, "shapes": l4},
        ],
        "chips": chips,
        "via_arrays": vias,
        "background_via_density_per_mm2": background,
        "convection": _convection(rng, ood=mode == "ood_7"),
        "dirichlet_edges": _dirichlet_edges(rng, ood=mode == "ood_6"),
        "design_vector": None,
    }


def _check_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    return int(seed)


def sample_board(seed: int, mode: str) -> dict[str, Any]:
    """Board record for `seed` (int >= 0) in `mode` ("id" or "design"); deterministic in both.

    OOD modes (spec_v2 ood.names) move one axis outside training and draw the rest as "id"
    (see _random_board). "design" draws x ~ U[0, 1]^36 by rejection until design_board(x) is
    feasible.
    """
    seed = _check_seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    if mode == "id" or mode in OOD_MODES:
        return _random_board(seed, rng, mode)
    if mode == "design":
        dim = int(_S2.design_space.dimension.value)
        for _ in range(_MAX_TRIES):
            try:
                return design_board(rng.random(dim), seed=seed)
            except InfeasibleDesignError:
                continue
        raise RuntimeError(f"seed {seed}: no feasible design vector in {_MAX_TRIES} draws")
    raise ValueError(f"unknown mode {mode!r}; expected 'id', 'design' or one of {OOD_MODES}")


# ---------------------------------------------------------------------------- design space


def _pair(v: Any, key: str) -> tuple[float, float]:
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int | float) for x in v)
            and v[0] <= v[1]):  # fmt: skip
        raise ValueError(f"{DESIGN_TASK_PATH}: {key} must be [low, high], got {v!r}")
    return float(v[0]), float(v[1])


@cache
def _design_task() -> dict[str, Any]:
    """config/design_task.yaml, checked (decision D2, docs/decisions.md)."""
    task = yaml.safe_load(DESIGN_TASK_PATH.read_text(encoding="utf-8"))
    want = {"design_task_version", "mapping", "chips", "stack", "layers",
            "background_via_density_per_mm2", "convection", "dirichlet_edges"}  # fmt: skip
    if not isinstance(task, dict) or set(task) != want:
        got = sorted(task) if isinstance(task, dict) else task
        raise ValueError(f"{DESIGN_TASK_PATH}: keys must be {sorted(want)}, got {got}")
    m = task["mapping"]
    _pair(m.get("rect_side"), "mapping.rect_side")
    _pair(m.get("via_density"), "mapping.via_density")
    if not isinstance(m.get("enforce_l1_coverage"), bool):
        raise ValueError(f"{DESIGN_TASK_PATH}: mapping.enforce_l1_coverage must be a bool")
    n_chips = int(_S2.design_space.chip_position_dims.value) // 2
    chips = task["chips"]
    if not isinstance(chips, list) or len(chips) != n_chips or not all(
        isinstance(c, dict) and set(c) == {"w", "h", "P"} and all(c[k] > 0 for k in c)
        for c in chips
    ):
        raise ValueError(f"{DESIGN_TASK_PATH}: chips must be {n_chips} x {{w, h, P}} > 0")
    total = sum(c["P"] for c in chips)
    cap = _S2.total_power_max.value
    if total > cap:
        raise ValueError(f"{DESIGN_TASK_PATH}: chip powers total {total} W > {cap} W")
    if set(task["layers"]) != {"l2_cutouts", "l3_split_rects", "l4_shapes"}:
        raise ValueError(
            f"{DESIGN_TASK_PATH}: layers must be l2_cutouts, l3_split_rects, l4_shapes"
        )
    return task


def design_board(x: np.ndarray, *, seed: int = 0) -> dict[str, Any]:
    """Board record for design vector x in [0, 1]^36 (CLAUDE.md §7.4); deterministic in x.

    x[0:8] chip centres, x[8:32] six L1 rectangles (x, y, w, h), x[32:36] via densities; the
    mapping and every fixed parameter come from config/design_task.yaml (decision D2).
    seed records the generator that drew x (0 when x was given directly). Raises
    InfeasibleDesignError if chips break the §6.5 gap or edge margin, ValueError for a bad x.
    """
    seed = _check_seed(seed)
    dim = int(_S2.design_space.dimension.value)
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (dim,):
        raise ValueError(f"x must have shape ({dim},), got {x.shape}")
    if not (np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()):
        raise ValueError(f"x must be finite and in [0, 1], got min {x.min()} max {x.max()}")
    task = _design_task()
    m = task["mapping"]
    margin = _S1.chip_edge_margin_min.value
    n_chips = len(task["chips"])
    n_rects = int(_S2.design_space.copper_rect_count.value)

    chips = []
    for i, c in enumerate(task["chips"]):
        w, h = _q(c["w"]), _q(c["h"])
        x0 = _q_floor(margin + x[2 * i] * (L_MM - 2 * margin - w))
        y0 = _q_floor(margin + x[2 * i + 1] * (L_MM - 2 * margin - h))
        chips.append({"x0": x0, "y0": y0, "w": w, "h": h, "P": float(c["P"]), "side": "top"})
    problem = _chip_violation(chips)
    if problem is not None:
        raise InfeasibleDesignError(f"design vector is infeasible: {problem}")

    lo, hi = _pair(m["rect_side"], "mapping.rect_side")
    rects = []
    base = 2 * n_chips
    for k in range(n_rects):
        ax, ay, aw, ah = x[base + 4 * k : base + 4 * k + 4]
        w, h = _q(lo + aw * (hi - lo)), _q(lo + ah * (hi - lo))
        x0, y0 = _q_floor(ax * (L_MM - w)), _q_floor(ay * (L_MM - h))
        rects.append({"type": "rect", "x0": x0, "y0": y0, "w": w, "h": h})
    if m["enforce_l1_coverage"] and not _in_range(coverage(rects), _V2.stack.layers[0].phi):
        raise InfeasibleDesignError("design vector is infeasible: L1 coverage out of range")

    vlo, vhi = _pair(m["via_density"], "mapping.via_density")
    base += 4 * n_rects
    vias = [{"x0": c["x0"], "y0": c["y0"], "w": c["w"], "h": c["h"],
             "density_per_mm2": float(vlo + x[base + i] * (vhi - vlo))}
            for i, c in enumerate(chips)]  # fmt: skip

    layers = task["layers"]
    return {
        "schema_version": "1.0",
        "id": "design-" + hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()[:16],
        "seed": seed,
        "mode": "design",
        "stack": copy.deepcopy(task["stack"]),
        "layers": [
            {"layer": 1, "shapes": rects},
            {"layer": 2, "plane": True, "cutouts": copy.deepcopy(layers["l2_cutouts"])},
            {"layer": 3, "split_rects": copy.deepcopy(layers["l3_split_rects"])},
            {"layer": 4, "shapes": copy.deepcopy(layers["l4_shapes"])},
        ],
        "chips": chips,
        "via_arrays": vias,
        "background_via_density_per_mm2": float(task["background_via_density_per_mm2"]),
        "convection": copy.deepcopy(task["convection"]),
        "dirichlet_edges": list(task["dirichlet_edges"]),
        "design_vector": x.tolist(),
    }


# ---------------------------------------------------------------------------- symmetries


def _sym_index(mat: np.ndarray) -> int:
    for k, s in enumerate(_SYMS):
        if np.array_equal(s, mat):
            return k
    raise AssertionError(f"not a square symmetry: {mat.tolist()}")


def inverse_sym(sym: int) -> int:
    """Index of the symmetry that undoes `sym` (0..7)."""
    return _sym_index(_SYMS[_check_sym(sym)].T)


def _check_sym(sym: Any) -> int:
    if isinstance(sym, bool) or not isinstance(sym, int | np.integer) or not 0 <= sym < 8:
        raise ValueError(f"sym must be an int in 0..7, got {sym!r}")
    return int(sym)


def transform_record(record: dict[str, Any], sym: int) -> dict[str, Any]:
    """Copy of record under square symmetry sym (0..7): geometry AND dirichlet_edges.

    sym = r + 4 f: mirror x -> L - x if f = 1, then r quarter turns counter-clockwise about the
    board centre (90° CCW maps edges W -> S, S -> E, E -> N, N -> W). The id carries the net
    symmetry as a '-sym<k>' suffix, so a symmetry followed by its inverse restores the record
    exactly. design_vector is kept as provenance: it describes the board before the symmetry.
    """
    mat = _SYMS[_check_sym(sym)]
    swap = mat[0, 0] == 0  # quarter turns exchange the x and y extents
    c = L_MM / 2

    def point(px: float, py: float) -> list[float]:
        u, v = px - c, py - c
        return [_q(c + mat[0, 0] * u + mat[0, 1] * v), _q(c + mat[1, 0] * u + mat[1, 1] * v)]

    def rect(r: dict[str, Any]) -> dict[str, Any]:
        (ax, ay), (bx, by) = point(r["x0"], r["y0"]), point(r["x0"] + r["w"], r["y0"] + r["h"])
        w, h = (r["h"], r["w"]) if swap else (r["w"], r["h"])
        return {**r, "x0": min(ax, bx), "y0": min(ay, by), "w": w, "h": h}

    def shape(s: dict[str, Any]) -> dict[str, Any]:
        if s["type"] == "rect":
            return rect(s)
        if s["type"] == "disc":
            cx, cy = point(s["cx"], s["cy"])
            return {**s, "cx": cx, "cy": cy}
        if s["type"] == "trace":
            return {**s, "points": [point(px, py) for px, py in s["points"]]}
        raise ValueError(f"unknown shape type {s['type']!r} in record {record.get('id')!r}")

    def edge(e: str) -> str:
        n = mat @ np.array(_EDGE_NORMALS[e])
        return next(k for k, v in _EDGE_NORMALS.items() if (n == v).all())

    out = copy.deepcopy(record)
    l1, l2, l3, l4 = out["layers"]
    l1["shapes"] = [shape(s) for s in l1["shapes"]]
    l2["cutouts"] = [rect(s) for s in l2["cutouts"]]
    l3["split_rects"] = [rect(s) for s in l3["split_rects"]]
    l4["shapes"] = [shape(s) for s in l4["shapes"]]
    out["chips"] = [rect(ch) for ch in out["chips"]]
    out["via_arrays"] = [rect(v) for v in out["via_arrays"]]
    if out["convection"]["forced_zone"] is not None:
        out["convection"]["forced_zone"] = rect(out["convection"]["forced_zone"])
    out["dirichlet_edges"] = [edge(e) for e in out["dirichlet_edges"]]

    match = _SYM_SUFFIX.match(out["id"])
    base, prior = (match.group(1), int(match.group(2))) if match else (out["id"], 0)
    net = _sym_index(mat @ _SYMS[prior])
    out["id"] = base if net == 0 else f"{base}-sym{net}"
    return out
