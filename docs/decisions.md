# Decisions log

Format: decision, date, rationale, who must confirm. Entries here are the record that CLAUDE.md §17
points to. Changing CLAUDE.md itself needs a PR plus a line in this file.

## D1 · How H6 is judged

- **Decision (2026-10-24):** H6 is judged at τ_v = 0.1 K, counting the pointwise certificate (Lemma 2c).
- **Why:** matches H6 as written in CLAUDE.md §11.4. The 50 % bar at 0.1 K is flagged there as at risk;
  if it fails, the fallback is the pointwise variant, keeping Lemma 2 for CG-2.
- **Owners to confirm:** N + M. Must be final before E13.

## D2 · Fixed parameters of the design task

- **Decision (2026-10-28, M):** `config/design_task.yaml` version 1.
  - Chips: 12×12 mm 2.0 W, 10×8 mm 1.5 W, 8×8 mm 1.0 W, 6×6 mm 0.5 W (total 5 W, cap 6 W). One dominant
    chip and unequal sizes make placement matter; all inside the §6.5 ranges.
  - Stack 1 oz on every layer, k_FR4,xy = 0.81 (nominal). L2 full plane; L3 two halves split at x = 50 mm;
    L4 one 40 × 50 mm pour (20 %, inside the §7.4 L4 range 5–40 %, which a bare L4 would leave).
  - Background vias 0.02 /mm² (middle of 0–0.05). Still air, h_top = h_bot = 10 W/(m²·K), no forced zone,
    **no heat-sunk edge**: matches the E17 rig (boards on standoffs in still air, §14) and the pilot.
  - Mapping: rect sides 5–40 mm and via density 0–1 /mm² (the ID ranges); `enforce_l1_coverage: true`.
- **Evidence (decision support only, not a report number; 150 feasible random x per option, N = 64,
  `solvers/fvm.py` direct solve, scratch script, 2026-10-28):**

  | Option | Feasible x | Peak rise P5 / P50 / P95 (K) | > 100 K |
  |---|---|---|---|
  | v0 placeholders (4 W) | 71 % | 34 / 40 / 48 | 0 % |
  | **chosen (5 W, no heat sink)** | 65 % | 42 / 49 / 58 | 0 % |
  | chosen + heat-sunk W edge | 65 % | 29 / 40 / 53 | 0 % |
  | chosen at 6 W | 65 % | 49 / 59 / 71 | 0 % |

  No design reached the 100 K envelope, so rescaling (which would change the task's fixed powers) never
  triggers. A heat-sunk edge spreads peaks more (P95/P5 ≈ 1.8 vs 1.4) but makes "push the chips to the
  edge" the obvious optimum. The L1 coverage check rejected no extra designs.
- **Owners to confirm:** M (decided); G (that the fabricated E17 boards and the pilot use the same setting).

- **Sampler v0 behaviour (2026-10-24, M):** `design_board(x)` raises `InfeasibleDesignError` when chips break
  the §6.5 gap or edge margin, and `sample_board(seed, "design")` draws x ~ U[0,1]³⁶ by rejection. p₀ is
  therefore uniform on the **feasible** part of the cube, not the whole cube. That matters for G3's "exact"
  weights for P1 (§7.4). With D2 fixed, about 35 % of uniform x are infeasible (table above).

## Sampler v0 defaults (2026-10-24, UNSURE, M to confirm)

CLAUDE.md §6.5 and §7.4 leave these open. They live in `config/spec_v2.yaml` → `sampler.v0_defaults`, so
changing one is a YAML edit plus a line here. Records generated before a change are a different version.
- Outer copper 1 / 2 oz with p = 0.5 / 0.5. Background via density uniform on [0, 0.05] vias/mm².
- L2: 0–3 cut-outs, sides U[5, 40] mm, centre uniform, clipped; resample until φ₂ ≥ 0.7.
- L3: one full cut at U[30, 70] mm (x or y, p = 0.5); for 3 rectangles a perpendicular cut splits one half.
  The pieces tile the board (φ₃ = 1).
- L4: 1–8 regions of the §6.5 families; resample until φ₄ ∈ [0.05, 0.40].
- Manhattan traces: start uniform, first direction H/V with p = 0.5, each later segment turns ±90° with
  p = 0.5; points clipped to the board. **Trace copper = points within width/2 of a segment (capsule);
  `data/raster.py` must use the same rule.**
- Chips: long side along x or y with p = 0.5; lower-left uniform inside the 3 mm margin; one try = the whole
  layout, 100 tries, then resample the count. Forced zone: lower-left uniform so the zone fits on the board.
- Coverage checks use a 256 × 256 point grid (the §6.4 4 × 4 rule at N = 64) until `data/raster.py`
  exists. "Otherwise resample" loops stop after 1,000 tries and raise. Record geometry is rounded to
  1e-3 mm so the 8 symmetries round-trip exactly.
- `transform_record` keeps `design_vector` as provenance (it describes the board before the symmetry) and
  marks the net symmetry in the id as `-sym<k>`.

## D3 · Outer-layer G_l dielectric term

- **Decision (2026-10-24):** the missing outer dielectric term is 0, as written in CLAUDE.md §7.2.
- **Implementation:** config flag `outer_dielectric_term: "zero"`, so another convention can be added
  later without touching callers.
- **Owner to confirm:** N.

## D4 · Frozen gate hyperparameters

Open. Owner M, due week 6. Goes to `experiments/configs/frozen.yaml`.

## D5 · Week-3 go/no-go outcome

Open. Owners N + M, due week 3 (exact solve ≥ 0.3 s keeps 256² × 4 layers).

## D6 · Week-1 start date

- **Decision (2026-10-24):** week 1 is the week of Monday 19 October 2026. Today (24 Oct) is its last day.
- **Consequence:** week 3 is 2 to 8 November, and the week-12 end is 10 January 2027.
- **Owners to confirm:** all three.

## Other decisions

- **GPU access (2026-10-24):** Kaggle and Colab runs of `scripts/check_gpu.py` are skipped for now, since
  college GPU access may be available. One Kaggle 2×T4 run is on record in `docs/compute.md`.
- 2026-10-24 · CLAUDE.md: added §18 (implementation notes from the literature survey, distilled from the eight study guides), open decisions D7–D14 in §17, and cross-references in §8.2 (Tu & Giesecke v2 numbering note), §8.4, §8.5, §11.1 (G1, G2 rows) and §16.2. No existing constant, invariant or decision changed. Branch `m/claude-md-survey-notes`; needs PR review by N and M.
