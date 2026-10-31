"""Generate the KiCad 8 boards from the hardware board records (G1.1; CLAUDE.md §14).

Run with KiCad 8's Python from the repo root (pcbnew must import):
    Windows:  "C:\\Program Files\\KiCad\\8.0\\bin\\python.exe" -m hardware.kicad.gen_board ^
                  --footprints "C:\\Program Files\\KiCad\\8.0\\share\\kicad\\footprints"
    Linux:    python3 -m hardware.kicad.gen_board --footprints /usr/share/kicad/footprints
Writes hardware/boards/board<k>/: board.kicad_pcb, fiducials.csv, heaters.csv (only for boards
without errors) and report.md (always). Then run DRC and the checker (hardware/kicad/README.md).

All geometry comes from hardware/kicad/plan.py (tested without KiCad); this file only copies the
plan into pcbnew, reads the real pad geometry of the library footprints back into the plan, and
fills the zones. Never hand-edit the generated PCB: change the record or fab.json and regenerate.
UNSURE: written against the KiCad 8 Python API on a machine without KiCad, so it has never run.
Calls marked UNSURE are the ones most likely to need a rename on the first run; each fails
with an AttributeError naming the call, and none is skipped silently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pcbnew

from hardware.kicad import plan as P

ROOT = Path(__file__).resolve().parents[2]
LAYERS = {"F.Cu": pcbnew.F_Cu, "In1.Cu": pcbnew.In1_Cu, "In2.Cu": pcbnew.In2_Cu,
          "B.Cu": pcbnew.B_Cu}
EDGE_LINE_MM = 0.1  # Edge.Cuts drawing width (cosmetic)


def mm(v: float) -> int:
    return pcbnew.FromMM(float(v))


def at(x: float, y: float, size: float) -> pcbnew.VECTOR2I:
    kx, ky = P.to_kicad(x, y, size)
    return pcbnew.VECTOR2I(mm(kx), mm(ky))


def load_footprint(lib_dir: Path, full_name: str) -> pcbnew.FOOTPRINT:
    lib, name = full_name.split(":", 1)
    path = lib_dir / f"{lib}.pretty"
    fp = pcbnew.FootprintLoad(str(path), name)
    if fp is None:
        raise FileNotFoundError(f"footprint {full_name} not found in {path} (check fab.json)")
    return fp


def spec_of(fp: pcbnew.FOOTPRINT, full_name: str) -> P.FootprintSpec:
    """Pad geometry of a library footprint in plan coordinates (mm, y up, unrotated)."""
    pads = []
    for pad in fp.Pads():
        pos = pad.GetFPRelativePosition()  # UNSURE: KiCad 7+ name
        size = pad.GetSize()
        shape = "circle" if pad.GetShape() == pcbnew.PAD_SHAPE_CIRCLE else "rect"
        pads.append(P.Pad(str(pad.GetNumber()), pcbnew.ToMM(pos.x), -pcbnew.ToMM(pos.y),
                          pcbnew.ToMM(size.x), pcbnew.ToMM(size.y), shape))
    fp.BuildCourtyardCaches()  # UNSURE
    box = fp.GetCourtyard(pcbnew.F_CrtYd).BBox()  # UNSURE
    w, h = pcbnew.ToMM(box.GetWidth()), pcbnew.ToMM(box.GetHeight())
    if w <= 0 or h <= 0:
        raise ValueError(f"{full_name}: no F.CrtYd courtyard; cannot space footprints")
    return P.FootprintSpec(full_name, tuple(pads), w, h)


def apply_rules(board: pcbnew.BOARD, cfg: dict[str, Any], stack: dict[str, Any]) -> None:
    """Fab rules and stack-up into the board's design settings (what DRC checks against)."""
    fab = cfg["fab"]
    ds = board.GetDesignSettings()
    ds.m_MinClearance = mm(fab["min_clearance_mm"])  # UNSURE attribute names (KiCad 8)
    ds.m_TrackMinWidth = mm(fab["min_track_mm"])
    ds.m_ViasMinAnnularWidth = mm(fab["min_annular_ring_mm"])
    ds.m_HoleToHoleMin = mm(fab["min_hole_to_hole_mm"])
    ds.m_CopperEdgeClearance = mm(fab["copper_edge_clearance_mm"])
    ds.m_MinThroughDrill = mm(cfg["via"]["drill_mm"])
    nc = ds.m_NetSettings.m_DefaultNetClass  # UNSURE
    nc.SetClearance(mm(fab["min_clearance_mm"]))
    nc.SetTrackWidth(mm(cfg["supply_track_width_mm"]))
    nc.SetViaDiameter(mm(cfg["via"]["pad_mm"]))
    nc.SetViaDrill(mm(cfg["via"]["drill_mm"]))
    ds.SetBoardThickness(mm(stack["total_mm"]))
    items = ds.GetStackupDescriptor().GetList()  # UNSURE: BOARD_STACKUP API
    want = [layer["thickness_mm"] for layer in stack["layers"]]
    got = [it for it in items if it.GetType() in (pcbnew.BS_ITEM_TYPE_COPPER,
                                                  pcbnew.BS_ITEM_TYPE_DIELECTRIC)]
    if len(got) != len(want):
        raise RuntimeError(f"stack-up has {len(got)} copper/dielectric items, stackup.json "
                           f"{len(want)}: set the layer count before applying the stack-up")
    for it, t in zip(got, want, strict=True):
        it.SetThickness(mm(t))


def build(p: P.Plan, cfg: dict[str, Any], lib_dir: Path, stack: dict[str, Any]) -> pcbnew.BOARD:
    size = p.size
    board = pcbnew.BOARD()
    board.SetCopperLayerCount(len(cfg["board"]["copper_layers"]))
    apply_rules(board, cfg, stack)

    nets: dict[str, pcbnew.NETINFO_ITEM] = {}
    names = {"GND"} | {n for f in p.heaters + p.test_pads for n in f.pad_nets.values()}
    for name in sorted(names):
        ni = pcbnew.NETINFO_ITEM(board, name)
        board.Add(ni)
        nets[name] = ni

    corners = [(0, 0), (size, 0), (size, size), (0, size)]
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        s = pcbnew.PCB_SHAPE(board)
        s.SetShape(pcbnew.SHAPE_T_SEGMENT)
        s.SetStart(at(*a, size))
        s.SetEnd(at(*b, size))
        s.SetLayer(pcbnew.Edge_Cuts)
        s.SetWidth(mm(EDGE_LINE_MM))
        board.Add(s)

    def outline(z: pcbnew.ZONE, poly: P.Poly) -> None:
        o = z.Outline()
        o.NewOutline()
        for x, y in poly:
            v = at(x, y, size)
            o.Append(v.x, v.y)

    for zd in p.zones:
        z = pcbnew.ZONE(board)
        z.SetLayer(LAYERS[zd.layer])
        z.SetNet(nets[zd.net])
        z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)  # solid: the pour is the record's copper
        z.SetZoneName(zd.source[:60])
        outline(z, zd.polygon)
        board.Add(z)
    for ra in p.rule_areas:
        z = pcbnew.ZONE(board)
        z.SetIsRuleArea(True)
        z.SetLayer(LAYERS[ra.layer])
        z.SetDoNotAllowCopperPour(True)
        z.SetDoNotAllowVias(False)
        z.SetDoNotAllowTracks(False)
        z.SetDoNotAllowPads(False)
        z.SetDoNotAllowFootprints(False)
        z.SetZoneName(ra.reason[:60])
        outline(z, ra.polygon)
        board.Add(z)

    for f in p.heaters + p.test_pads + p.fiducials:
        fp = load_footprint(lib_dir, f.footprint)
        fp.SetReference(f.ref)
        fp.SetPosition(at(f.x, f.y, size))
        fp.SetOrientationDegrees(f.rot_deg)
        for pad in fp.Pads():
            net = f.pad_nets.get(str(pad.GetNumber()))
            if net is not None:
                pad.SetNet(nets[net])
        board.Add(fp)

    for t in p.tracks:
        tr = pcbnew.PCB_TRACK(board)
        tr.SetStart(at(*t.a, size))
        tr.SetEnd(at(*t.b, size))
        tr.SetWidth(mm(t.width))
        tr.SetLayer(LAYERS[t.layer])
        tr.SetNet(nets[t.net])
        board.Add(tr)
    for v in p.vias:
        via = pcbnew.PCB_VIA(board)
        via.SetViaType(pcbnew.VIATYPE_THROUGH)
        via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        via.SetPosition(at(v.x, v.y, size))
        via.SetDrill(mm(cfg["via"]["drill_mm"]))
        via.SetWidth(mm(cfg["via"]["pad_mm"]))
        via.SetNet(nets[v.net])
        board.Add(via)

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    return board


def report(p: P.Plan, rec: dict[str, Any], cfg: dict[str, Any],
           specs: list[P.FootprintSpec]) -> str:
    lines = [f"# {p.board_id}: generation report (G1.1)", "",
             f"Record `{rec['id']}` (mode `{rec['mode']}`), fab `{cfg['fab']['name']}`, resistor "
             f"`{cfg['resistor']['part']}` rated {cfg['resistor']['rated_W']} W.", "",
             "## Unmanufacturable (no PCB written if any)", ""]
    lines += [f"- {e}" for e in p.errors] or ["- none"]
    lines += ["", "## Differences between the PCB and the record (the solver's board)", ""]
    lines += [f"- {d}" for d in p.deviations] or ["- none"]
    lines += ["", "## Via grids", "", "| source | requested /mm² | grid /mm² | placed | kept | "
              "achieved /mm² | dropped (reason: count) |", "|---|---|---|---|---|---|---|"]
    for s in p.via_stats:
        lines.append(f"| {s['source']} | {s['requested_density']:.4g} | {s['grid_density']:.4g} | "
                     f"{s['placed']} | {s['kept']} | {s['achieved_density']:.4g} | "
                     f"{s['drop_reasons'] or '-'} |")
    lines += ["", "## Heaters", "", "| ref | x_mm | y_mm | nominal_W | max_W |",
              "|---|---|---|---|---|"]
    lines += [f"| {h.ref} | {h.x:.3f} | {h.y:.3f} | {h.nominal_W:.4g} | {h.max_W:.4g} |"
              for h in p.heaters]
    lines += ["", "## Footprints read from the KiCad library", ""]
    lines += [f"- `{s.name}`: courtyard {s.courtyard_w:.3f} x {s.courtyard_h:.3f} mm, pads "
              + ", ".join(f"{q.number} ({q.x:+.3f}, {q.y:+.3f}) {q.w:.3f}x{q.h:.3f} {q.shape}"
                          for q in s.pads) for s in specs]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boards", type=Path, default=ROOT / "hardware" / "boards")
    ap.add_argument("--config", type=Path, default=ROOT / "hardware" / "kicad" / "fab.json")
    ap.add_argument("--stackup", type=Path, default=ROOT / "hardware" / "kicad" / "stackup.json")
    ap.add_argument("--footprints", type=Path,
                    default=os.environ.get("KICAD8_FOOTPRINT_DIR"),
                    help="KiCad 8 footprint library folder (default: $KICAD8_FOOTPRINT_DIR)")
    args = ap.parse_args()
    if args.footprints is None or not Path(args.footprints).is_dir():
        sys.exit("give --footprints <KiCad 8 share/kicad/footprints> (or set KICAD8_FOOTPRINT_DIR)")
    lib_dir = Path(args.footprints)
    cfg = P.load_config(args.config)
    stacks = json.loads(args.stackup.read_text(encoding="utf-8"))
    rname, tname = cfg["resistor"]["footprint"], cfg["test_pad"]["footprint"]
    rspec = spec_of(load_footprint(lib_dir, rname), rname)
    tspec = spec_of(load_footprint(lib_dir, tname), tname)
    failed = []
    for path in sorted(args.boards.glob("board*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        out = args.boards / path.stem
        out.mkdir(parents=True, exist_ok=True)
        p = P.plan_board(rec, cfg, rspec, tspec)
        (out / "report.md").write_text(report(p, rec, cfg, [rspec, tspec]), encoding="utf-8")
        if not p.buildable:
            failed.append(rec["id"])
            print(f"{rec['id']}: NOT generated, {len(p.errors)} unmanufacturable item(s); see "
                  f"{out / 'report.md'}")
            continue
        if rec["id"] not in stacks:
            sys.exit(f"{rec['id']} missing from {args.stackup}: "
                     "run python -m hardware.kicad.stackup")
        board = build(p, cfg, lib_dir, stacks[rec["id"]])
        pcbnew.SaveBoard(str(out / "board.kicad_pcb"), board)
        P.write_csvs(p, out)
        print(f"{rec['id']}: wrote {out / 'board.kicad_pcb'} ({len(p.vias)} vias, "
              f"{len(p.deviations)} deviation(s) in report.md)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
