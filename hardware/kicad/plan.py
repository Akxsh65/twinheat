"""Board plan: a board record -> everything the PCB needs (G1.1; CLAUDE.md §14, §7.1, §10.1).

Standard library only, so it runs inside KiCad's bundled Python (gen_board.py) and in CI (tests).
All geometry here is in RECORD coordinates: mm, origin at the lower-left corner, y up. gen_board.py
converts to KiCad's y-down frame with to_kicad() at the very end.

What the plan contains, from the record and fab.json:
  zones      every record shape as a copper zone on its layer, net GND: L1 shapes (F.Cu), the L2
             plane (In1.Cu, cut-outs as copper keep-out rule areas), the L3 split rects (In2.Cu),
             L4 shapes (B.Cu). Discs are 64-gons with the disc's area; each trace segment is a
             capsule (rectangle + two half-discs of 32 sides each) of the given width.
  heaters    each chip -> k 2512 resistors in parallel, k = ceil(P / (max_fraction · rated_W)), so
             each runs at <= 50 % of rated (CLAUDE.md §14). They sit inside the chip footprint,
             long axis along the chip's long side, stacked across it; if they do not fit, that is an
             ERROR (the record is never resized). Pad 1 -> net HTR<i> (supply), pad 2 -> GND.
  supply     per heater: a test pad on HTR<i> and one on GND just outside the footprint, straight
             tracks of supply_track_width_mm joining the pad columns to them, and one GND via next
             to the GND test pad tying it to the L2 plane.
  fiducials  4 footprints at fab.json positions, with an F.Cu copper keep-out of the mask opening.
  vias       0.3 mm drill / 0.6 mm pad on a square grid at the record's density: inside each via
             array at (array + background) density, elsewhere at the background density (the
             rasteriser adds the two the same way). Vias are net GND, so they join every zone.
Rule for vias near pads (decided for G1.1): a via is DROPPED if it would touch any pad, come within
min_clearance of supply (HTR) copper, the board edge clearance or a fiducial opening. The record
is not changed; every dropped via and every other difference from the record is listed in
`deviations`, and achieved vs requested density is in `via_stats`.
Anything the fab cannot build (annular ring, hole spacing, track width, heaters that do not fit,
bottom-side heaters, test pads off the board) is listed in `errors`; gen_board.py writes no PCB
for a board with errors (prompt: report, never silently change).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ConfigError", "FootprintSpec", "Pad", "Placed", "Plan", "disc_poly", "distance_to_item",
    "load_config", "plan_board", "shape_polys", "to_kicad", "write_csvs",
]

DISC_SIDES = 64  # the prompt: discs as 64-gons
CAP_SIDES = 32  # per half-disc end of a trace segment
AREA_STEP_MM = 0.05  # sampling step for the "pour cleared" area estimates (reporting only)
REQUIRED = (
    ("resistor", "part"), ("resistor", "rated_W"), ("supply_track_width_mm",), ("fab", "name"),
    ("fab", "min_clearance_mm"), ("fab", "min_track_mm"), ("fab", "min_annular_ring_mm"),
    ("fab", "min_hole_to_hole_mm"), ("fab", "copper_edge_clearance_mm"),
)

Point = tuple[float, float]
Poly = list[Point]
Item = tuple  # ("rect", cx, cy, w, h) | ("circle", cx, cy, r) | ("capsule", ax, ay, bx, by, hw)


class ConfigError(ValueError):
    """fab.json is incomplete or malformed."""


def load_config(path: Path) -> dict[str, Any]:
    """Read fab.json; raise ConfigError naming every required field that is missing or null."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = []
    for keys in REQUIRED:
        node: Any = cfg
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if node is None:
            missing.append(".".join(keys))
    if missing:
        raise ConfigError(f"{path}: fill these fields before generating boards (datasheet / fab "
                          f"capability page): {', '.join(missing)}")
    return cfg


# ---------------------------------------------------------------------------------- geometry

def to_kicad(x: float, y: float, size: float) -> Point:
    """Record (y up, origin lower-left) -> KiCad page coordinates (y down), both in mm."""
    return x, size - y


def rect_poly(x0: float, y0: float, w: float, h: float) -> Poly:
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def disc_poly(cx: float, cy: float, r: float, sides: int = DISC_SIDES) -> Poly:
    """Regular polygon with the same area as the disc (circumradius scaled up slightly)."""
    R = r * math.sqrt(2 * math.pi / (sides * math.sin(2 * math.pi / sides)))
    return [(cx + R * math.cos(2 * math.pi * k / sides), cy + R * math.sin(2 * math.pi * k / sides))
            for k in range(sides)]


def capsule_poly(a: Point, b: Point, width: float) -> Poly:
    """Segment a-b of the given width with half-disc ends, the area of the exact capsule."""
    (ax, ay), (bx, by) = a, b
    ang = math.atan2(by - ay, bx - ax)
    hw = width / 2
    # Each half-disc: CAP_SIDES fan triangles of angle δ. The two end vertices stay at hw so the
    # straight sides keep the exact width; the interior radius R solves
    #   (CAP_SIDES − 2)/2·sin δ·R² + hw·sin δ·R = π hw²/2   (half-disc area).
    s = math.sin(math.pi / CAP_SIDES)
    qa, qb, qc = (CAP_SIDES - 2) / 2 * s, hw * s, -math.pi * hw * hw / 2
    R = (-qb + math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    pts: Poly = []
    for (cx, cy), start in (((bx, by), ang - math.pi / 2), ((ax, ay), ang + math.pi / 2)):
        for k in range(CAP_SIDES + 1):
            t = start + math.pi * k / CAP_SIDES
            rr = hw if k in (0, CAP_SIDES) else R
            pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
    return pts


def shape_polys(s: dict[str, Any]) -> list[Poly]:
    if s["type"] == "rect":
        return [rect_poly(s["x0"], s["y0"], s["w"], s["h"])]
    if s["type"] == "disc":
        return [disc_poly(s["cx"], s["cy"], s["r"])]
    if s["type"] == "trace":
        p = [tuple(q) for q in s["points"]]
        return [capsule_poly(a, b, s["width"]) for a, b in zip(p, p[1:], strict=False)]
    raise ValueError(f"unknown shape type {s['type']!r}")


def _seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_item(px: float, py: float, item: Item) -> float:
    """Distance from a point to an item's copper (0 inside)."""
    kind = item[0]
    if kind == "rect":
        _, cx, cy, w, h = item
        dx = max(abs(px - cx) - w / 2, 0.0)
        dy = max(abs(py - cy) - h / 2, 0.0)
        return math.hypot(dx, dy)
    if kind == "circle":
        _, cx, cy, r = item
        return max(math.hypot(px - cx, py - cy) - r, 0.0)
    if kind == "capsule":
        _, ax, ay, bx, by, hw = item
        return max(_seg_dist(px, py, ax, ay, bx, by) - hw, 0.0)
    raise ValueError(f"unknown item {item!r}")


def _in_shape(px: float, py: float, s: dict[str, Any]) -> bool:
    """Record inside-tests (data/raster.py rules: rect half-open, disc closed, trace capsule)."""
    if s["type"] == "rect":
        return s["x0"] <= px < s["x0"] + s["w"] and s["y0"] <= py < s["y0"] + s["h"]
    if s["type"] == "disc":
        return (px - s["cx"]) ** 2 + (py - s["cy"]) ** 2 <= s["r"] ** 2
    p = s["points"]
    return any(_seg_dist(px, py, *a, *b) <= s["width"] / 2 for a, b in zip(p, p[1:], strict=False))


# ------------------------------------------------------------------------------- plan objects

@dataclass(frozen=True)
class Pad:
    number: str
    x: float  # pad centre in the unrotated footprint, mm, y UP (gen_board flips KiCad's y)
    y: float
    w: float
    h: float
    shape: str  # "rect" (incl. roundrect) or "circle"


@dataclass(frozen=True)
class FootprintSpec:
    name: str
    pads: tuple[Pad, ...]
    courtyard_w: float
    courtyard_h: float

    def pad(self, number: str) -> Pad:
        for p in self.pads:
            if p.number == number:
                return p
        raise ValueError(f"{self.name}: no pad {number!r}")


@dataclass
class Placed:
    ref: str
    footprint: str
    x: float  # centre, record coordinates
    y: float
    rot_deg: float  # counter-clockwise, as KiCad's orientation
    pad_nets: dict[str, str]
    pads: list[tuple[str, Item]] = field(default_factory=list)  # (number, world copper item)
    chip: int = -1
    nominal_W: float = 0.0
    max_W: float = 0.0


@dataclass
class Track:
    net: str
    layer: str
    a: Point
    b: Point
    width: float

    def item(self) -> Item:
        return ("capsule", *self.a, *self.b, self.width / 2)


@dataclass
class Via:
    x: float
    y: float
    net: str
    source: str


@dataclass
class Zone:
    layer: str
    net: str
    polygon: Poly
    source: str


@dataclass
class RuleArea:
    layer: str
    polygon: Poly
    reason: str


@dataclass
class Plan:
    board_id: str
    size: float
    zones: list[Zone] = field(default_factory=list)
    rule_areas: list[RuleArea] = field(default_factory=list)
    heaters: list[Placed] = field(default_factory=list)
    test_pads: list[Placed] = field(default_factory=list)
    fiducials: list[Placed] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    via_stats: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)

    @property
    def buildable(self) -> bool:
        return not self.errors

    def all_pads(self) -> list[Item]:
        return [it for f in self.heaters + self.test_pads for _, it in f.pads]

    def supply_copper(self) -> list[Item]:
        """Copper on a heater supply net (HTR<i>): pads, test pads and tracks."""
        out = [it for f in self.heaters + self.test_pads for n, it in f.pads
               if f.pad_nets[n].startswith("HTR")]
        return out + [t.item() for t in self.tracks if t.net.startswith("HTR")]


def _world_pad(fp: FootprintSpec, pad: Pad, cx: float, cy: float, rot: float) -> Item:
    if rot == 0:
        x, y, w, h = pad.x, pad.y, pad.w, pad.h
    elif rot == 90:  # counter-clockwise: (x, y) -> (-y, x)
        x, y, w, h = -pad.y, pad.x, pad.h, pad.w
    else:
        raise ValueError(f"rotation {rot} not supported")
    if pad.shape == "circle":
        return ("circle", cx + x, cy + y, w / 2)
    return ("rect", cx + x, cy + y, w, h)


def _place(fp: FootprintSpec, ref: str, x: float, y: float, rot: float, nets: dict[str, str],
           **kw: Any) -> Placed:
    p = Placed(ref, fp.name, x, y, rot, nets, **kw)
    p.pads = [(pad.number, _world_pad(fp, pad, x, y, rot)) for pad in fp.pads]
    return p


# ---------------------------------------------------------------------------------- planning

def plan_board(rec: dict[str, Any], cfg: dict[str, Any], resistor: FootprintSpec,
               test_pad: FootprintSpec) -> Plan:
    """The full plan for one record. Inspect `errors` before building anything."""
    size = float(cfg["board"]["size_mm"])
    p = Plan(rec["id"], size)
    fab = cfg["fab"]
    drill, vpad = float(cfg["via"]["drill_mm"]), float(cfg["via"]["pad_mm"])
    track_w = float(cfg["supply_track_width_mm"])

    # fab rules that do not depend on the record
    ring = (vpad - drill) / 2
    if ring < fab["min_annular_ring_mm"] - 1e-12:
        p.errors.append(f"via annular ring {ring:.3f} mm ({drill} mm drill, {vpad} mm pad) < fab "
                        f"minimum {fab['min_annular_ring_mm']} mm")
    if track_w < fab["min_track_mm"] - 1e-12:
        p.errors.append(f"supply track {track_w} mm < fab minimum track {fab['min_track_mm']} mm")

    _zones(rec, p)
    _heaters(rec, cfg, resistor, test_pad, p)
    _fiducials(cfg, p)
    _vias(rec, cfg, p)
    _cleared_areas(rec, cfg, p)
    return p


def _zones(rec: dict[str, Any], p: Plan) -> None:
    l1, l2, l3, l4 = rec["layers"]
    for i, s in enumerate(l1["shapes"]):
        for poly in shape_polys(s):
            p.zones.append(Zone("F.Cu", "GND", poly, f"layers[0].shapes[{i}] {s['type']}"))
    if l2["plane"]:
        p.zones.append(Zone("In1.Cu", "GND", rect_poly(0, 0, p.size, p.size), "L2 plane"))
        for i, s in enumerate(l2["cutouts"]):
            for poly in shape_polys(s):
                p.rule_areas.append(RuleArea("In1.Cu", poly, f"L2 cut-out {i}"))
    for i, s in enumerate(l3["split_rects"]):
        p.zones.append(Zone("In2.Cu", "GND", rect_poly(s["x0"], s["y0"], s["w"], s["h"]),
                            f"L3 split rect {i}"))
    if len(l3["split_rects"]) > 1:
        p.deviations.append("L3: the split rects share net GND; where they touch with no gap "
                            "(the record has none) they fill as one plane, which is also what the "
                            "solver sees")
    for i, s in enumerate(l4["shapes"]):
        for poly in shape_polys(s):
            p.zones.append(Zone("B.Cu", "GND", poly, f"layers[3].shapes[{i}] {s['type']}"))


def _heaters(rec: dict[str, Any], cfg: dict[str, Any], rfp: FootprintSpec, tfp: FootprintSpec,
             p: Plan) -> None:
    res = cfg["resistor"]
    rated, frac = float(res["rated_W"]), float(res["max_fraction_of_rated"])
    bw, bh = (float(v) for v in res["body_mm"])
    clr = float(cfg["fab"]["min_clearance_mm"])
    edge = float(cfg["fab"]["copper_edge_clearance_mm"])
    track_w = float(cfg["supply_track_width_mm"])
    vpad = float(cfg["via"]["pad_mm"])
    n_ref = 0
    for i, c in enumerate(rec["chips"]):
        name = f"chips[{i}]"
        if c["side"] != "top":
            p.errors.append(f"{name}: side {c['side']!r}; heaters are only built on F.Cu")
            continue
        k = math.ceil(c["P"] / (frac * rated) - 1e-12)
        long_x = c["w"] >= c["h"]
        rot = 0 if long_x else 90
        long_side, short_side = (c["w"], c["h"]) if long_x else (c["h"], c["w"])
        pitch = rfp.courtyard_h  # stacked resistors: courtyards side by side
        extent = (k - 1) * pitch + bh
        if bw > long_side + 1e-9 or extent > short_side + 1e-9:
            p.errors.append(
                f"{name}: {k} resistors of {bw} x {bh} mm (each <= {frac:g} x {rated:g} W for "
                f"P = {c['P']} W) do not fit in the {c['w']} x {c['h']} mm footprint")
            continue
        cx, cy = c["x0"] + c["w"] / 2, c["y0"] + c["h"] / 2
        ux, uy = (0.0, 1.0) if long_x else (1.0, 0.0)  # the stacking (short) axis
        offs = [(j - (k - 1) / 2) * pitch for j in range(k)]
        net = f"HTR{i + 1}"
        for o in offs:
            n_ref += 1
            p.heaters.append(_place(rfp, f"R{n_ref}", cx + ux * o, cy + uy * o, rot,
                                    {"1": net, "2": "GND"}, chip=i, nominal_W=c["P"] / k,
                                    max_W=frac * rated))
        mine = [h for h in p.heaters if h.chip == i]
        p.deviations.append(
            f"heater {i + 1}: {k} x {rfp.name} ({c['P'] / k:.3g} W each), its pads, two test pads, "
            f"{track_w} mm supply tracks and one GND tie via are copper not in the record")

        # test pads outward along the stacking axis, beyond the last resistor; else the other side
        reach = rfp.courtyard_h / 2 + clr + tfp.courtyard_h / 2
        placed = False
        for sign in (1.0, -1.0):
            last = mine[-1] if sign > 0 else mine[0]
            tps, tracks = [], []
            for pad_no, tp_net in (("1", net), ("2", "GND")):
                col = [it for h in mine for n, it in h.pads if n == pad_no]
                px, py = (col[-1] if sign > 0 else col[0])[1:3]
                lx, ly = last.x + ux * sign * reach, last.y + uy * sign * reach
                tx, ty = (px, ly) if long_x else (lx, py)
                tps.append(_place(tfp, f"TP{i + 1}{'P' if tp_net != 'GND' else 'N'}", tx, ty, 0,
                                  {"1": tp_net}, chip=i))
                ends = [it[1:3] for it in col]
                far = max(ends, key=lambda e: -sign * (e[0] * ux + e[1] * uy))
                tracks.append(Track(tp_net, "F.Cu", far, (tx, ty), track_w))
            r_tp = max(tfp.courtyard_w, tfp.courtyard_h) / 2
            gnd_tp = tps[1]
            vx, vy = gnd_tp.x + ux * sign * (r_tp + clr + vpad / 2), gnd_tp.y + uy * sign * (
                r_tp + clr + vpad / 2)
            inside = all(edge + r_tp <= t.x <= p.size - edge - r_tp
                         and edge + r_tp <= t.y <= p.size - edge - r_tp for t in tps)
            inside &= edge + vpad / 2 <= vx <= p.size - edge - vpad / 2
            inside &= edge + vpad / 2 <= vy <= p.size - edge - vpad / 2
            if inside:
                p.test_pads += tps
                p.tracks += tracks
                p.vias.append(Via(vx, vy, "GND", f"heater {i + 1} GND tie"))
                placed = True
                break
        if not placed:
            p.errors.append(f"{name}: no room for the test pads inside the board edge clearance")


def _fiducials(cfg: dict[str, Any], p: Plan) -> None:
    fid = cfg["fiducial"]
    r_mask = float(fid["mask_diameter_mm"]) / 2
    clr = float(cfg["fab"]["min_clearance_mm"])
    for j, (x, y) in enumerate(fid["positions_mm"]):
        p.fiducials.append(Placed(f"FID{j + 1}", fid["footprint"], float(x), float(y), 0, {}))
        p.rule_areas.append(RuleArea("F.Cu", disc_poly(x, y, r_mask + clr),
                                     f"fiducial FID{j + 1} copper keep-out"))


def _grid(x0: float, y0: float, w: float, h: float, d: float) -> list[Point]:
    """Square-ish grid at density d (per mm²) centred in the rectangle."""
    nx, ny = max(1, round(w * math.sqrt(d))), max(1, round(h * math.sqrt(d)))
    return [(x0 + (a + 0.5) * w / nx, y0 + (b + 0.5) * h / ny)
            for b in range(ny) for a in range(nx)]


def _vias(rec: dict[str, Any], cfg: dict[str, Any], p: Plan) -> None:
    fab = cfg["fab"]
    clr, edge = float(fab["min_clearance_mm"]), float(fab["copper_edge_clearance_mm"])
    drill, vpad = float(cfg["via"]["drill_mm"]), float(cfg["via"]["pad_mm"])
    r_mask = float(cfg["fiducial"]["mask_diameter_mm"]) / 2
    bg = float(rec["background_via_density_per_mm2"])
    supply, pads = p.supply_copper(), p.all_pads()
    ties = [(v.x, v.y) for v in p.vias]

    def reason(x: float, y: float) -> str | None:
        if not (edge + vpad / 2 <= x <= p.size - edge - vpad / 2
                and edge + vpad / 2 <= y <= p.size - edge - vpad / 2):
            return "edge clearance"
        if any(math.hypot(x - f.x, y - f.y) < r_mask + clr + vpad / 2 for f in p.fiducials):
            return "fiducial opening"
        if any(distance_to_item(x, y, it) < vpad / 2 for it in pads):
            return "touches a pad"
        if any(distance_to_item(x, y, it) < clr + vpad / 2 for it in supply):
            return "supply-net clearance"
        if any(math.hypot(x - tx, y - ty) < drill + float(fab["min_hole_to_hole_mm"])
               for tx, ty in ties):
            return "GND tie via"
        return None

    def run(points: list[Point], source: str, requested: float, area: float, grid_d: float):
        kept, why = 0, {}
        for x, y in points:
            r = reason(x, y)
            if r is None:
                p.vias.append(Via(x, y, "GND", source))
                kept += 1
            else:
                why[r] = why.get(r, 0) + 1
        dropped = len(points) - kept
        p.via_stats.append({"source": source, "requested_density": requested,
                            "grid_density": grid_d, "placed": len(points), "kept": kept,
                            "dropped": dropped, "area_mm2": area,
                            "achieved_density": kept / area if area else 0.0, "drop_reasons": why})
        if dropped or abs(grid_d - requested) > 1e-12:
            p.deviations.append(
                f"via {source}: requested {requested:.4g} /mm², grid {grid_d:.4g} /mm², "
                f"kept {kept}/{len(points)} -> {kept / area if area else 0:.4g} /mm² "
                f"(dropped: {why or 'none'})")

    rects = [(a["x0"], a["y0"], a["w"], a["h"]) for a in rec["via_arrays"]]
    for i, a in enumerate(rec["via_arrays"]):
        d = a["density_per_mm2"] + bg
        if d <= 0:
            continue
        pts = _grid(a["x0"], a["y0"], a["w"], a["h"], d)
        nx = max(1, round(a["w"] * math.sqrt(d)))
        ny = max(1, round(a["h"] * math.sqrt(d)))
        pitch = min(a["w"] / nx, a["h"] / ny)
        if pitch - drill < float(fab["min_hole_to_hole_mm"]) - 1e-12:
            p.errors.append(f"via_arrays[{i}]: pitch {pitch:.3f} mm leaves {pitch - drill:.3f} mm "
                            f"hole-to-hole < fab minimum {fab['min_hole_to_hole_mm']} mm "
                            f"(density {d:g} /mm² is too high for this fab)")
            continue
        run(pts, f"via_arrays[{i}]", d, a["w"] * a["h"], nx * ny / (a["w"] * a["h"]))
    if bg > 0:
        pts = [(x, y) for x, y in _grid(0, 0, p.size, p.size, bg)
               if not any(x0 <= x < x0 + w and y0 <= y < y0 + h for x0, y0, w, h in rects)]
        area = p.size**2 - sum(w * h for _, _, w, h in rects)  # arrays do not overlap here
        n = max(1, round(p.size * math.sqrt(bg)))
        run(pts, "background", bg, area, n * n / p.size**2)


def _cleared_areas(rec: dict[str, Any], cfg: dict[str, Any], p: Plan) -> None:
    """Report record L1 copper that the zone fill will clear: around supply copper (clearance)
    and inside the fiducial keep-outs. Estimated on an AREA_STEP_MM sampling grid."""
    clr = float(cfg["fab"]["min_clearance_mm"])
    r_mask = float(cfg["fiducial"]["mask_diameter_mm"]) / 2
    shapes = rec["layers"][0]["shapes"]
    step = AREA_STEP_MM

    def area(items: list[Item], margin: float) -> float:
        n = 0
        for it in items:
            if it[0] == "rect":
                xs = (it[1] - it[3] / 2, it[1] + it[3] / 2)
                ys = (it[2] - it[4] / 2, it[2] + it[4] / 2)
            elif it[0] == "circle":
                xs, ys = (it[1] - it[3], it[1] + it[3]), (it[2] - it[3], it[2] + it[3])
            else:
                xs = (min(it[1], it[3]) - it[5], max(it[1], it[3]) + it[5])
                ys = (min(it[2], it[4]) - it[5], max(it[2], it[4]) + it[5])
            x = xs[0] - margin + step / 2
            while x < xs[1] + margin:
                y = ys[0] - margin + step / 2
                while y < ys[1] + margin:
                    if (distance_to_item(x, y, it) <= margin
                            and any(_in_shape(x, y, s) for s in shapes)):
                        n += 1
                    y += step
                x += step
        return n * step * step  # overlapping items are counted twice: an upper estimate

    for i in range(len(rec["chips"])):
        net = f"HTR{i + 1}"
        items = [it for f in p.heaters + p.test_pads for n, it in f.pads if f.pad_nets[n] == net]
        items += [t.item() for t in p.tracks if t.net == net]
        a = area(items, clr)
        if a > 0:
            p.deviations.append(f"L1 pour cleared around heater {i + 1} supply copper (pads, test "
                                f"pad, track + {clr} mm clearance): about {a:.1f} mm² of record "
                                "copper")
    for f in p.fiducials:
        a = area([("circle", f.x, f.y, r_mask)], clr)
        if a > 0:
            p.deviations.append(f"fiducial {f.ref}: about {a:.1f} mm² of record L1 copper cleared")


# -------------------------------------------------------------------------------------- CSVs

def write_csvs(p: Plan, out_dir: Path) -> None:
    """fiducials.csv (ref, x_mm, y_mm) and heaters.csv (ref, x_mm, y_mm, nominal_W, max_W), in
    record coordinates (origin lower-left, y up), one row per footprint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "fiducials.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ref", "x_mm", "y_mm"])
        for fd in p.fiducials:
            w.writerow([fd.ref, f"{fd.x:.4f}", f"{fd.y:.4f}"])
    with (out_dir / "heaters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ref", "x_mm", "y_mm", "nominal_W", "max_W"])
        for h in p.heaters:
            w.writerow([h.ref, f"{h.x:.4f}", f"{h.y:.4f}", f"{h.nominal_W:.6g}", f"{h.max_W:.6g}"])
