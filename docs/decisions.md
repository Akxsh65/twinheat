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

## OOD sets v2 (M1.2)

- **Decision (2026-10-28, M):** five points CLAUDE.md §6.7 / §7.4 left open, decided before M1.2.
  1. **No annuli (OOD-3) and no partial-edge Dirichlet (OOD-6) in OOD v1.** The board-record schema can
     express neither, and contracts are frozen. OOD-3 keeps L-shapes (two rects) and 45° traces; OOD-6 keeps
     3–4 whole heat-sunk edges. Both stay proposed for a schema v1.1 (`docs/contracts.md` A6, plus partial edges).
  2. **Bottom-side chips: proposed `Raster.q_bot` [N,N] W/m² on L4**, zero except on `ood_bottom_chips`;
     `q` stays [N,N] on L1, so the change is additive. **Needs N's agreement** (it changes `solvers/fvm.py`).
     The model input (§7.5) has no L4 heat channel, so the surrogate cannot see bottom chips; the runtime
     rule check flags `side = bottom` and sends those boards to the exact solver.
  3. **`ood_5` is dropped from v2 (it stays for the v1 plate).** On 4 layers, L1 sheet contrast in
     training already reaches ≈ 170× (1 oz) to ≈ 340× (2 oz) by §7.2/§7.3, above OOD-5's 130–160, so
     OOD-5 is not out of distribution there. `ood_3oz` (≈ 500× at φ = 1) is the v2 contrast set. v2 has
     9 OOD sets: CLAUDE.md §7.4 and §11.5 updated to match.
  4. **`ood_3oz` = 3 oz on the outer layers (L1, L4); inner layers stay 1 oz.** Training varies outer copper
     (1 or 2 oz) and fixes inner copper, so this moves one axis only.
  5. **`ood_dense_vias`: via-array density uniform on (1.0, 1.5] vias/mm².** Upper bound from
     manufacturability: 0.3 mm drill, 0.6 mm pad (G1.1) and a 0.1–0.2 mm pad gap (UNVERIFIED, check the fab
     rules from G1.2) give a minimum pitch of ≈ 0.75–0.8 mm, i.e. ≤ ≈ 1.6–1.8 vias/mm². Raise it if the
     fab allows a tighter pitch.
- **Owners to confirm:** M (decided); N (point 2, and the CLAUDE.md change); G (point 5 against the fab rules).
- **Sampling defaults chosen in M1.2** (CLAUDE.md does not fix them; UNSURE, M to confirm). Each OOD mode
  draws everything else exactly as `id`.
  - `ood_1`: chip count uniform on 5–8; placement and powers as ID (total still ≤ 6 W).
  - `ood_2`: per-chip power log-uniform on [3, 6] W (the ID family); if the total exceeds 12 W all are
    rescaled, as in ID. The moved axis is the largest chip power (> 3 W).
  - `ood_3`: every L1 region is an L-shape or a 45° trace, p = 0.5 / 0.5. L-shape: a §6.5-sized box with
    arms of width U[3.2, 6] mm (the ID trace width, so thinness does not also move) along the two sides at
    a random corner; both arms must stay on the board. 45° trace: §6.5 segment counts, lengths and
    widths; start in one of 8 directions, each later segment turns ±45°; a segment is shortened to stay on
    the board (clamping one axis would bend the angle); at least one diagonal segment.
  - `ood_4`: L1 trace widths U[0.8, 2.0] mm and at least one trace on L1; L4 traces stay ID.
  - `ood_6`: 3 or 4 whole heat-sunk edges, uniform.
  - `ood_7`: h_top and h_bot U[2, 5]; forced zone (p = 0.3 as ID) with h_top U[50, 100].
  - `ood_bottom_chips`: every chip on the bottom (`bottom_chips_side: all`).
  - `ood_dense_vias`: p = 0.6 per chip as ID, redrawn until at least one array exists.

## §8.6 preliminary evidence (N1.3)

- **Proposed (2026-10-30, M); N to confirm by approving PR #15.**
- **Finding: the scripts were never saved.** `bench/prelim/stop.js` and `cert.js`, named by plan v3.1 (5 Oct) and by
  CLAUDE.md §8.6, were never saved. They are in no repo branch, on no local disk, and in neither source chat; the
  numbers were most likely computed inline in a chat's analysis tool. What survives is the explainer's solver and
  certificate widget (`bench/prelim/explainer_cert_widget.js`, verbatim from `TwinHeat_Pitch.html`) and its
  `emulate()` error model.
- **Finding: the table is genuine.** Its κ̂ ÷ true ‖A⁻¹‖∞ column depends only on A. On the surviving solver it
  reproduces to three decimals for the default (1.067), no-heat-sink (1.069) and four-chip (1.067) variants. The
  widget's "noisier errors" figures in §8.6's last paragraph reproduce exactly, and N's `solvers/` + `certify/`
  match the widget iteration for iteration (`tests/test_prelim_widget.py`).
- **Decision: no reproduction claim.** The four-variant table and the pointwise table are reported only as "5 Oct,
  scripts lost". The milder error model and variants 3–4 are not reconstructed from the numbers (CLAUDE.md §0 rule 12).
- **Decision: a declared re-run is the N1.3 deliverable.** `bench/prelim/reproduce.py` fixes every setting in code,
  committed before its first run:
  - the `emulate()` blob-and-ringing shape (ported, checked against the JS to about 1e-16), plus noise at 2 % of its
    maximum, scaled so max|μ − θ| equals the row's warm-start error;
  - variant 3 is a 5 × 5-cell, 0.5 W chip near the N-E corner;
  - variant 4 is 17.5 µm copper and H = 10 W/(m²·K) combined.

  Its CSVs are new preliminary evidence and are compared with §8.6 in the PR, not fitted to it. E13 (4 layers,
  trained FNO) remains the real test of H6.
- **Owners:** N + M.

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
