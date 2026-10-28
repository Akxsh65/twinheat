# Decisions log

Format: decision, date, rationale, who must confirm. Entries here are the record that CLAUDE.md §17
points to. Changing CLAUDE.md itself needs a PR plus a line in this file.

## D1 · How H6 is judged

- **Decision (2026-10-24):** H6 is judged at τ_v = 0.1 K, counting the pointwise certificate (Lemma 2c).
- **Why:** matches H6 as written in CLAUDE.md §11.4. The 50 % bar at 0.1 K is flagged there as at risk;
  if it fails, the fallback is the pointwise variant, keeping Lemma 2 for CG-2.
- **Owners to confirm:** N + M. Must be final before E13.

## D2 · Fixed parameters of the design task

Open. Owner M, due week 2. Placeholders live in `config/design_task.yaml` until then.

- **Sampler v0 behaviour (2026-10-24, M):** `design_board(x)` raises `InfeasibleDesignError` when chips break
  the §6.5 gap or edge margin, and `sample_board(seed, "design")` draws x ~ U[0,1]³⁶ by rejection. p₀ is
  therefore uniform on the **feasible** part of the cube, not the whole cube. That matters for G3's "exact"
  weights for P1 (§7.4), and must be revisited when D2 is fixed.

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
