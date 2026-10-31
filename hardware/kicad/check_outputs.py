"""Check generated board outputs against the records (G1.1 DONE WHEN, second half).

    python -m hardware.kicad.check_outputs [--boards DIR] [--config hardware/kicad/fab.json]

For each hardware/boards/board<k>.json with an output folder hardware/boards/board<k>/:
  * the record is a valid board_record (contracts, invariant B6);
  * fiducials.csv has exactly the fab.json fiducial positions, in order;
  * heaters.csv: every resistor lies inside a top-side chip footprint of the record; each chip
    has k = ceil(P / (max_fraction · rated_W)) resistors whose nominal_W sum to P; every max_W is
    max_fraction · rated_W and every nominal_W <= max_W;
  * if board.kicad_pcb exists, every CSV ref is a footprint there at the same position (y flipped).
The checks use the record and fab.json directly, not plan.py, so a planning bug cannot hide here.
Exit status 1 on any problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from hardware.kicad.plan import load_config, to_kicad

ROOT = Path(__file__).resolve().parents[2]
POS_TOL_MM = 1e-3  # the record's geometry_resolution (config/spec_v2.yaml)
REL_TOL = 1e-6  # CSV values are written with 6 significant figures


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pcb_footprints(path: Path) -> dict[str, tuple[float, float]]:
    """{reference: (x, y)} of every footprint in a .kicad_pcb (KiCad page coordinates, mm)."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, tuple[float, float]] = {}
    for block in re.split(r"\(footprint\s+\"", text)[1:]:
        at = re.search(r"\(at\s+(-?[\d.]+)\s+(-?[\d.]+)", block)
        ref = re.search(r"\(property\s+\"Reference\"\s+\"([^\"]+)\"", block)
        if at and ref:
            out[ref.group(1)] = (float(at.group(1)), float(at.group(2)))
    return out


def check_board(rec: dict[str, Any], cfg: dict[str, Any], out_dir: Path) -> list[str]:
    """Problems found for one board (empty list = all good)."""
    out_dir = Path(out_dir)
    bad: list[str] = []
    rid = rec["id"]

    fids = _rows(out_dir / "fiducials.csv")
    want = cfg["fiducial"]["positions_mm"]
    if len(fids) != len(want):
        bad.append(f"{rid}: fiducials.csv has {len(fids)} rows, fab.json {len(want)}")
    for row, (x, y) in zip(fids, want, strict=False):
        if math.hypot(float(row["x_mm"]) - x, float(row["y_mm"]) - y) > POS_TOL_MM:
            bad.append(f"{rid}: fiducial {row['ref']} at ({row['x_mm']}, {row['y_mm']}), "
                       f"expected ({x}, {y})")

    res = cfg["resistor"]
    cap = float(res["max_fraction_of_rated"]) * float(res["rated_W"])
    heaters = _rows(out_dir / "heaters.csv")
    if len({h["ref"] for h in heaters}) != len(heaters):
        bad.append(f"{rid}: duplicate refs in heaters.csv")
    per_chip: dict[int, list[dict[str, str]]] = {}
    for h in heaters:
        x, y = float(h["x_mm"]), float(h["y_mm"])
        owner = [i for i, c in enumerate(rec["chips"])
                 if c["x0"] - POS_TOL_MM <= x <= c["x0"] + c["w"] + POS_TOL_MM
                 and c["y0"] - POS_TOL_MM <= y <= c["y0"] + c["h"] + POS_TOL_MM]
        if len(owner) != 1:
            bad.append(f"{rid}: heater {h['ref']} at ({x}, {y}) is in {len(owner)} chip footprints")
            continue
        per_chip.setdefault(owner[0], []).append(h)
        if not math.isclose(float(h["max_W"]), cap, rel_tol=REL_TOL):
            bad.append(f"{rid}: heater {h['ref']} max_W {h['max_W']} != {cap:.6g} "
                       f"(max_fraction x rated_W)")
        if float(h["nominal_W"]) > float(h["max_W"]) * (1 + REL_TOL):
            bad.append(f"{rid}: heater {h['ref']} nominal_W {h['nominal_W']} > max_W {h['max_W']}")
    for i, c in enumerate(rec["chips"]):
        rows = per_chip.get(i, [])
        k = math.ceil(c["P"] / cap - 1e-12)
        if len(rows) != k:
            bad.append(f"{rid}: chip {i} needs {k} resistors (P = {c['P']} W), heaters.csv has "
                       f"{len(rows)}")
        total = sum(float(r["nominal_W"]) for r in rows)
        if rows and not math.isclose(total, c["P"], rel_tol=REL_TOL):
            bad.append(f"{rid}: chip {i} nominal_W sum {total:.6g} != P {c['P']}")

    pcb = out_dir / "board.kicad_pcb"
    if pcb.exists():
        fps = pcb_footprints(pcb)
        size = float(cfg["board"]["size_mm"])
        for row in fids + heaters:
            if row["ref"] not in fps:
                bad.append(f"{rid}: {row['ref']} is in the CSVs but not in board.kicad_pcb")
                continue
            kx, ky = to_kicad(float(row["x_mm"]), float(row["y_mm"]), size)
            px, py = fps[row["ref"]]
            if math.hypot(px - kx, py - ky) > POS_TOL_MM:
                bad.append(f"{rid}: {row['ref']} at ({px}, {py}) in the PCB, CSV says ({kx}, {ky})")
    return bad


def main() -> None:
    from contracts.validate import validate  # needs jsonschema: normal Python, not KiCad's

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boards", type=Path, default=ROOT / "hardware" / "boards")
    ap.add_argument("--config", type=Path, default=ROOT / "hardware" / "kicad" / "fab.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    problems: list[str] = []
    records = sorted(args.boards.glob("board*.json"))
    if not records:
        sys.exit(f"no board*.json in {args.boards}")
    for path in records:
        rec = json.loads(path.read_text(encoding="utf-8"))
        validate(rec, "board_record")
        out = args.boards / path.stem
        if not (out / "heaters.csv").exists():
            problems.append(f"{rec['id']}: no outputs in {out} (run gen_board.py first)")
            continue
        found = check_board(rec, cfg, out)
        problems += found
        print(f"{rec['id']}: {'OK' if not found else f'{len(found)} problem(s)'}"
              f"{' (CSV + PCB)' if (out / 'board.kicad_pcb').exists() else ' (CSV only)'}")
    for p in problems:
        print("  " + p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
