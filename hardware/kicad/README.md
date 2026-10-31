# Hardware boards from records (G1.1)

The three E17 boards are generated from the same board records the solver uses
(`hardware/boards/board{1,2,3}.json`), so simulation and hardware match by construction. Board 1 is
the fit board (h_top, h_bot are fitted on it only, CLAUDE.md §14); boards 2 and 3 are held out.
**Never hand-edit a generated PCB**: change the record or `fab.json` and regenerate.

| File | Runs in | What it does |
|---|---|---|
| `plan.py` | any Python (stdlib only) | record + `fab.json` → zones, heaters, test pads, tracks, vias, errors, deviations |
| `gen_board.py` | KiCad 8's Python | copies the plan into pcbnew, fills zones, writes `board.kicad_pcb`, CSVs, `report.md` |
| `check_outputs.py` | repo Python | CSVs (and the PCB's footprint positions) vs the records and `fab.json` |
| `stackup.py` | repo Python | `stackup.json` from `config/spec_v2.yaml` + the records (KiCad's Python has no YAML) |
| `fab.json` | config | part, fab rules, footprints; **nulls block generation until filled** |

## Before the first run (G)

1. Choose the 2512 part and the fab. Fill every `null` in `fab.json` from the datasheet and the fab's
   capability page, and write the sources in `docs/equipment.md`:
   `resistor.part`, `resistor.rated_W`, `fab.name`, `fab.min_clearance_mm`, `fab.min_track_mm`,
   `fab.min_annular_ring_mm`, `fab.min_hole_to_hole_mm`, `fab.copper_edge_clearance_mm`, and
   `supply_track_width_mm` (from the heater current once the supply voltage is fixed).
2. Check the three footprint names exist in your KiCad 8 library (`fab.json`, marked CHECK).
3. If the fab's stack-up differs from CLAUDE.md §7.1, change `config/spec_v2.yaml` (decision logged,
   §7.1), then `python -m hardware.kicad.stackup`.

## Generate, DRC, check

From the repo root:

```
"C:\Program Files\KiCad\8.0\bin\python.exe" -m hardware.kicad.gen_board --footprints "C:\Program Files\KiCad\8.0\share\kicad\footprints"
for k in 1 2 3; do
  kicad-cli pcb drc --format json --severity-all --exit-code-violations \
    --output hardware/boards/board$k/drc.json hardware/boards/board$k/board.kicad_pcb
done
python -m hardware.kicad.check_outputs
```

DONE WHEN (G1.1): all three boards pass DRC (exit code 0) and `check_outputs` prints OK for all
three. Commit `board.kicad_pcb`, the CSVs, `report.md` and `drc.json` for each board.

`gen_board.py` has never run (no KiCad on the machine it was written on). The calls most likely to
need a rename on the first run are marked UNSURE; each fails loudly, none is skipped.

## Design rules the generator follows

- Every record shape is a copper zone on its layer, net GND, solid pad connection: L1 shapes (F.Cu),
  the L2 plane minus cut-outs (In1.Cu), the L3 split rects (In2.Cu), L4 shapes (B.Cu). Discs are
  equal-area 64-gons; each trace segment is an equal-area capsule of the record width.
- Each chip → k = ceil(P / (0.5 × rated_W)) 2512 resistors in parallel inside the chip footprint.
  Pad 1 is the heater's supply net `HTR<i>`, pad 2 is GND. Two test pads per heater (supply and
  GND) just outside the footprint, straight supply tracks, one GND via tying the GND test pad to L2.
- Via arrays: 0.3 mm drill / 0.6 mm pad on a square grid at (array + background) density;
  background vias elsewhere. A via that would touch any pad, come within clearance of supply copper,
  the board edge or a fiducial opening is **dropped and reported**; the record is not changed.
- Fiducials: `Fiducial_1mm_Mask2mm` at (5,5), (95,5), (95,95), (5,95) mm with an F.Cu copper keep-out.

## Known differences from the records (see each board's report.md)

- Heater copper (pads, tracks, test pads, one GND via) is not in the records; record L1 pour is
  cleared around the supply copper.
- The record via arrays are rectangles around the heaters, so part of each grid lands on the
  resistor pads and is dropped: with test geometry, board 2 keeps about 91 of 120 vias per array
  (0.76 /mm² built where the record says 1.02 /mm²). **Proposal (G to decide):** redraw the records'
  via arrays as rectangles beside the pads (for example above and below each resistor), so the
  record describes exactly the vias that are built.
- The 2512 heaters are 6.3 × 3.2 mm; the sampler's chip short side is at least 4.7 mm, so the
  surrogate sees these boards as out of distribution (`mode: ood_hardware`; docs/pcb_realism.md
  finding 1, proposal R1 is M's). E17 compares the solver with the camera, which is unaffected.
