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

## D3 · Outer-layer G_l dielectric term

- **Decision (2026-10-24):** the missing outer dielectric term is 0, as written in CLAUDE.md §7.2.
- **Implementation:** config flag `outer_dielectric_term: "zero"`, so another convention can be added
  later without touching callers.
- **Owner to confirm:** N.

## D4 · Frozen gate hyperparameters

Open. Owner M, due week 6. Goes to `experiments/configs/frozen.yaml`.

## D5 · Week-3 go/no-go outcome

- **Decision (2026-11-05; proposed by M, N to confirm by approving the PR): keep 256² × 4 layers.** No
  move to 6 layers or 384².
- **Evidence:** `bench/results/solver_table_v1.csv` (N1.5, commit c4f7b6f), 20 fixed boards at 256² × 4
  (262,144 unknowns), measured on the machine that will run the oracle labels (rvcse lab machine: Intel Xeon
  Gold 6226R, 1 thread, as §7.3 states "one CPU core"; GPU rows on an A100-PCIE-40GB). PyAMG-GMRES to 1e-8:
  median 4.55 s with default settings, 1.62 s with strength θ = 0.1, all 20 boards converged. Both are far
  above the 0.3 s threshold (CLAUDE.md §7.3, §15), so an exact solve is expensive enough to justify the
  surrogate at the planned size.
- **Caveats:**
  - The table's `commit` column reads `7ad689a-dirty` (the run started from an uncommitted tree). That is
    acceptable for this go/no-go, but the numbers are not report numbers until re-run on a tagged commit
    (CLAUDE.md §0 rule 11).
  - The Jacobi and red-black rows did not converge within their caps (0/20), so their times are not solve
    times.
  - The `gpu_model` field carries trailing NUL bytes; fix this in `bench/solver_bench.py`.
- **Owners:** N + M.

## D6 · Week-1 start date

- **Decision (2026-10-24):** week 1 is the week of Monday 19 October 2026. Today (24 Oct) is its last day.
- **Consequence:** week 3 is 2 to 8 November, and the week-12 end is 10 January 2027.
- **Owners to confirm:** all three.

## D7 · ACI upper boundary

- **Decision (2026-11-07, M):** when α_k ≥ 1 the prediction set is empty, so **err = 1** whatever S⁺ is,
  as Gibbs & Candès Lemma 4.1 requires. q is still reported as 0 there (B = 0), as in CLAUDE.md §8.2.
- **Consequence:** α_k ∈ [−γ, 1 + γ] and |mean(err) − α| ≤ C/(γ n_a) hold exactly, and are hard unit tests
  (`tests/test_aci.py`).
- **Implemented in:** `gate/aci_audit.py` (M3.1).

## D8 · Quantile level

- **Decision (2026-11-07, M):** every quantile, both at calibration and in the ACI window, is the
  split-conformal one: the ⌈(n+1)(1−α)⌉-th smallest of n scores ("higher" interpolation), and ∞ when that
  rank exceeds n (CLAUDE.md §18.2).
- **Implemented in:** `gate/aci_audit.py::conformal_quantile`. G2 and G3 (`gate/baselines.py`) use the same
  function.
- **Note:** CLAUDE.md §9.3's `quantile(scores, 1 − alpha_t)` is to be read with this level. The §17 rows
  for D7 and D8 should be marked decided in a CLAUDE.md PR (done 2026-11-12, branch m/d13-d9-decisions).

## D9 · G6 / MVP settings

Source for every MVP term below: `docs/refs/bastani2022_mvp.txt` (Bastani et al. 2022, arXiv 2206.01067v1).

- **Decision (2026-11-12, M), design rules (fixed now):**
  1. **Previews never show a randomised threshold.** A `query` reads the G6 state as it stood after the last
     commit and shows the deterministic upper candidate q = i*/m_MVP (the more conservative of MVP's two
     candidates, so the previewed B is never below the committed one). It never draws MVP's coin, never
     draws score noise, and never writes gate state (B8).
  2. **MVP's randomisation is drawn once per committed design,** by the server, after `coin.commit(id)`,
     from a seeded `numpy.random.Generator(PCG64)` stream owned by the G6 gate (separate from the audit
     coin's stream, so adding G6 does not change the G5 coin sequence). The draw is logged in the audit log
     and never re-rolled.
  3. **Only audited steps update MVP** (B8, CLAUDE.md §18.3), with the certified upper score S⁺ (Lemma 1′)
     mapped by S̃ = S⁺/(1 + S⁺) (paper footnote 10). Thresholds map back by q = q̃/(1 − q̃).
  4. **Smoothness noise, if used, is non-negative and added before the map:** S̃ = (S⁺ + U)/(1 + S⁺ + U),
     U ~ Uniform[0, w], drawn after commit and hidden like the coin. This keeps S̃ in [0, 1) without
     clipping (CLAUDE.md §0 rule 5), and because S⁺ + U ≥ S⁺ ≥ S, an accepted large error still counts as
     a miss (the Lemma 1′ direction). UNSURE: the checker (§2) should confirm this argument; the smoothness
     bound ρ ≤ 1/(r·m_MVP·w) of CLAUDE.md §18.3 is for noise on S̃, not on S, and needs re-deriving.
- **Numeric settings: not decided here.** m_MVP, r, ϵ, w (and η, which Theorem 3.1 fixes once K_ϵ, |𝒢| and
  m_MVP are known) go in `experiments/configs/frozen.yaml` under D4 (week 6), chosen on validation
  trajectories. Development default for m_MVP only: 40, the bucket count used in every experiment of the
  paper. The paper states no values of r, ϵ or η for its experiments, so the code takes them as required
  arguments with no defaults.
- **Owner:** M. Due before E15.

## D13 · G1 details

Source: `docs/refs/deepoheat_v2_alg1.txt` (DeepOHeat-v2, arXiv 2608.16080v1, Algorithm 1).

- **Decision (2026-11-12, M):** the G1 used in E7 and E8 is DeepOHeat-v2's gate on TwinHeat's shared
  signal, with no C-event cutoff:
  - **Residual:** r_hot divided by diag A (kelvin), over the m hottest predicted cells, with the m shared
    by all gates (CLAUDE.md §8.1, default 64). The paper uses the raw residual A T̂ − b (watts) and m = 100.
  - **Signal:** r_hot alone, as in the paper (no ensemble spread; that is G5's s_t).
  - **Threshold:** FIFO window of |W| = 100 (CLAUDE.md §11.1). r_hot of **every** proposal is pushed
    before the comparison, so the window includes the current value (Algorithm 1, lines 6–7). Flag if
    r_hot > Q_{1−f}(W), with Q the ⌈(1−f)·|W|⌉-th smallest value ("higher", as in D8). The first |W|
    proposals are not flagged (line 2) and count as accepted in M1. f is tuned to the exact-solve budget
    (§11.1).
  - **C rule: off.** The gate stays active for the whole run.
- **Why:**
  - Same signal for every gate, so E7/E8 compare threshold rules (flag-rate percentile vs. audited
    conformal), not residual definitions. Dividing by diag A matters on our boards: the diagonal varies
    about 56× across cells (3.9e-3 to 0.219 W/K, measured with `solvers.fvm` on 50 sampler boards
    while writing N4.1; a design check, not a report number), so the raw residual
    in watts ranks cells by copper coverage as much as by error. Because the scaling is per cell, it
    changes which proposals G1 flags; it is not a monotone rescaling of r_hot.
  - The C rule counts retraining events. E7 scores all gates offline on one oracle-labelled trajectory
    set with no retraining (CLAUDE.md §11.5), so the rule has nothing to count there. In E8, stopping after
    C = 3 events (ΔN = 50, i.e. 150 refinements) would leave the rest of the run unchecked. At equal
    exact-solve budget, that measures the cutoff, not the gate.
- **Paper-faithful variant:** M3.2 builds both differences into G1 in `gate/baselines.py` as switches
  (`residual="raw"`, `c_events=3`, with m = 100, f = 0.30, ΔN = 50 from the paper's Table III). If E8's budget allows, it is
  reported as one sensitivity row labelled "G1 (DeepOHeat-v2 settings)", never as the main G1.
- **Owner:** M. Due before E7.

## Other decisions

- **GPU access (2026-10-24):** Kaggle and Colab runs of `scripts/check_gpu.py` are skipped for now, since
  college GPU access may be available. One Kaggle 2×T4 run is on record in `docs/compute.md`.
- 2026-10-24 · CLAUDE.md: added §18 (implementation notes from the literature survey, distilled from the eight study guides), open decisions D7–D14 in §17, and cross-references in §8.2 (Tu & Giesecke v2 numbering note), §8.4, §8.5, §11.1 (G1, G2 rows) and §16.2. No existing constant, invariant or decision changed. Branch `m/claude-md-survey-notes`; needs PR review by N and M.

- 2026-11-12 · CLAUDE.md §17: rows D7, D8, D9 and D13 now say where their decision is recorded (this file).
  D9 is marked partly decided: its design rules are fixed, its numbers go to D4. No constant, invariant or
  other row changed. Branch `m/d13-d9-decisions`; needs PR review by N.

## Dataset v1 split seeds (N2.1)

- **Decision (2026-11-04):** every released split uses seed `2026` (per-board seed =
  SHA-256(split, seed, i), so the split name separates splits). Seed 0 is the 200-board pilot
  only and is never released. Splits: train 6,000, val 1,000, cal 1,000, test 1,000; OOD 500 each
  for ood_1, ood_2, ood_3, ood_4, ood_6, ood_7, ood_3oz, ood_dense_vias. Not generated:
  ood_5 (dropped on 4 layers, see "OOD sets v2") and ood_bottom_chips (generator refuses until
  solvers/fvm.py reads q_bot).
- **Pilot (seed 0, 200 boards, A100 GPU 0 on rvcse, commit 743c726):** reviewed by M and N.
- **Owners to confirm:** N + M.

## Dataset v1: OOD-2 envelope rescale above 20 % (N2.1), 2026-11-04
ood_2 (3-6 W per chip, total <= 12 W): 41.0 % of 500 boards rescaled to theta_max = 100 K
(dataset_v1, commit da2c2b6); the generator exited with status 2 as designed. All other
splits <= 7.0 % (ID splits 0.4-1.2 %, ood_7 7.0 %). All 12 splits complete and hash-verified.
Decision: ood_2 accepted as is (N + M).
Reason: the 20 % rule of CLAUDE.md 6.5 is written for the ID sampler; 6.7 says the envelope
still applies to OOD-2, and lowering its power range would remove the shift the set tests.

## M1.3 Darcy validation result, 2026-11-09
Run darcy_validate-2d5a346-0 (rvcse A100, seed 0, 100 epochs, batch 32, lr 1e-3, wd 1e-4):
test rel. L2 0.0213 (val 0.0201, best epoch 96/100). Published PDEBench FNO nRMSE 0.064
(docs/refs/pdebench_darcy_fno.txt): ratio 0.33, outside the prompt's two-sided 20 % band, on the
better side. Not a like-for-like reproduction: PDEBench trained the original FNO implementation and
hyperparameters, 500 epochs, Adam, lr 1e-3 halved every 100 epochs, 90/10 split (arXiv 2210.07182v7,
Sec. 4.1 and App. C); ours is FNO2d width 48, modes 24 (CLAUDE.md 12), AdamW + cosine, x/y channels,
train-only normalisation, split 8500/500/1000 (last 1000 = test, disjoint by test).
Decision: M1.3 accepted. The FNO and trainer learn Darcy below the published FNO error; for this
check "within 20 %" is read as "test rel. L2 <= the published value". Owner: M.

## Ensemble training settings (M2.2), 2026-11-09
Members 1-5: seed = member number. Effective batch 32 (--batch 32; gradient accumulation where a
GPU cannot fit 32). pre128: 100 epochs. ft256: 50 epochs from that member's pre128 best.pt.
Early stopping patience 10 on validation hotspot error (K), as coded. lr 1e-3 / 3e-4, weight decay
1e-4, w_hot 0.1, AMP: models/train_ensemble.py defaults. Basis: rvcse A100 timing run (eb1b3ac),
about 14 s/epoch at 128^2 and 2.6 min/epoch at 256^2. Revisit ft256 epochs after member 1 if its
validation curve has not flattened. All members train on rvcse GPU 0. Owner: M.

## Ensemble members 2-5 run (M2.2), 2026-11-12
Members 2-5 trained at commit 157bfb8 (code identical to c1b0692; the tree was dirty only through
models/registry.json, which the runs write). Deviation from the settings entry of 2026-11-09:
members ran two at a time, 2 and 4 on rvcse GPU 0 and 3 and 5 on rvcse GPU 2 (both
A100-PCIE-40GB, CUDA_DEVICE_ORDER=PCI_BUS_ID). Commands and settings as for member 1. Paired
ft256 runs were disk-bound, so wall-clock gpu_hours are not comparable with member 1's.
Open item (M): every member's best val rel. L2 is above the week-5 target of CLAUDE.md 12 (3 %);
val hotspot error is below 1.5 K for all five. Summaries: experiments/results/ensemble/.
Decision pending.

## Storage of data and checkpoints: no Kaggle Datasets, 2026-11-12
All GPU work so far (dataset v1, verification, norm stats, M1.3, the five ensemble members) ran on
the lab GPU server rvcse, and the remaining data-reading GPU work (E5, gate experiments, E9, N6.1)
can run there too: rvcse has two A100s, enough for N5.3's two-GPU DDP run. A Kaggle Dataset copy
is therefore not needed. Changed CLAUDE.md 4 ("Large files"), the plan's Backups line, N2.2
checklist and N6.1 fetch locations, and gpu_instructions.md N2.2 release steps: data and
checkpoints live on rvcse with an off-server copy on Drive, hashes in data/manifest.json and
models/registry.json as before. The Drive copy stays required (rvcse is shared and is the only
copy). Unchanged: Kaggle T4 runs that need no dataset (N1.4 bandwidth table, @gpu tests) and
notebooks/train_member.ipynb (kept as a fallback template). Needs review by N and M.

## Dataset v1: ood_bottom_chips added (N2.1), 2026-11-12
Generated after the generator fix (branch n/ood-bottom-chips) at commit 23ff3f3, seed 2026, 500 boards,
rvcse A100 GPU 2; 0.4 % of boards rescaled to theta_max = 100 K. data.verify_dataset on the whole
dataset: PASS, 13 splits, 13,500 boards, 0 problems (data/verify_report.md). This replaces "Not
generated: ood_bottom_chips" in the split-seeds entry above. The full-dataset hash changes; the
train-split hash (norm stats, ensemble members) does not. experiments/configs/e5.yaml now lists
the split. Owners to confirm: N + M.

## GPU runs on the college A100 (rvcse), 2026-11-12
Every remaining GPU run moves from Kaggle T4 to the college GPU server rvcse (`rvcecsegpuserver`,
NVIDIA A100-PCIE-40GB), which already ran every GPU job so far (N1.4 tests, N1.5, dataset v1, N2.2,
M1.3, M2.2). This extends the storage entry above, whose "Unchanged: Kaggle T4 runs that need no
dataset (N1.4 bandwidth table, @gpu tests)" no longer holds. Reason: one machine of record, so the
PAGP chapter does not put a T4 bandwidth table next to A100 solver timings. Kaggle stays a
fallback only.
- `bench/stencil_bw.py`: the hard-coded T4 peak (320 GB/s) is removed. `--peak-gbs` and
  `--peak-source` are now required and written into every CSV row. The A100 value is taken from
  NVIDIA's datasheet and recorded in `docs/refs/` by a person, not filled in from memory
  (CLAUDE.md §0 rule 1).
- Changed: CLAUDE.md §4 (Compute line, GPU tests, torch/requirements wording), the plan's N1.4
  DONE WHEN, `gpu_instructions.md` (machines, open and done items, section 4), `docs/compute.md`
  (rvcse section), Makefile and README `gpu-test`, docstrings in `bench/solver_bench.py` and
  `tests/test_gpu_solvers.py`.
- Owners to confirm: N + M.

## M2.3 baselines trained (U-Net, FNO+local), 2026-11-13
U-Net and FNO+local (CLAUDE.md 12) trained with the ensemble settings of 2026-11-09: member 1,
seed 1, effective batch 32, pre128 100 epochs then ft256 50 epochs, train_ensemble.py defaults.
Commit 20b65c7 (tree dirty only through models/registry.json, written by the runs); U-Net on
rvcse GPU 0 and FNO+local on GPU 2, side by side. Summaries: experiments/results/baselines/.
Both reach a lower best val rel. L2 and val hotspot error than every FNO ensemble member, and
FNO+local is the closest to the 3 % target; the U-Net's best epoch was its last. Single seed
each; E5 on the test split is the comparison of record. Owner: M.

## No hardware validation, 2026-11-13
Decision (team): no boards are fabricated and no hardware is used. Dropped: E17, H10, RQ6, M7,
T1.9 (measured overlay), CLAUDE.md 14 (now a stub), the hardware weeks, the "two boards" cut and
the camera and board fallbacks, and the fab_stackup / camera_datasheet refs. Plan prompts G0.3,
G1.1, G1.2, G5.4 and M5.2 are marked DROPPED. Contribution 4 now reads "shared evidence: rigorous
per-iterate bands in anytime rendering and the TwinHeat-Traj v1 release (E18)". The spec_v2 stack
(CLAUDE.md 7.1) is the modelled stack. Consequence: the model-to-reality gap is not measured, and
the report states it as a limitation. Left for G: contracts/schemas/hardware_record.json and its
examples (removal is a contracts change needing the three signatures, CLAUDE.md 10), the hardware/
package stub, and branch g/g1.1-kicad-boards (close unmerged).

## Calibration split stays as in v2-data, 2026-11-13
Decision (team): the cal split of dataset v1 (tag v2-data, ID mode) is kept; no new dataset version.
Open point for M: CLAUDE.md 9.3 and 11.1 initialise q and calibrate G2 on boards drawn from the
design-space density p0, and G3's exact weights for P1 are relative to p0. ID-mode cal boards are
not draws from p0. Lemma 1 (G5-G7) holds for any initial q, so those gates are unaffected. For G2
and G3, M decides where p0 calibration scores come from, for example design-space boards labelled
with the oracle solves of N3.2 (experiment data, not a dataset version), or states that G2 and G3
are calibrated on the ID distribution.

## Week-5 go/no-go: proceed with the FNO ensemble at 256², 2026-11-16
E5 (run at 45913b6; experiments/results/e5/): ensemble test rel. L2 5.41 % (target < 3 %), hotspot
error 0.76 K (target < 1.5 K): NO-GO on rel. L2. Decision (N + M, approved 16 Nov): keep the trained
5-member FNO ensemble at 256²; no retraining; do not move the gate experiments to 128².
Reasons: Lemma 1's validity does not depend on surrogate accuracy; the 128² fallback does not fix
the miss (pre128 val rel. L2 5.5-5.6 %); the hotspot target is met.
Consequences: more flagged designs and exact solves (H2, RQ2, H5); the report states the miss and
that a single FNO+local (test rel. L2 4.78 %) outperformed the ensemble on test.
Note: run_info.json reads 45913b6-dirty only because E5 took the commit after writing its own
results into the repo (fixed on this branch); the tree was clean when the run started.

## M3.3 proposer settings, 2026-11-18
Values not fixed by CLAUDE.md or the M3.3 prompt, chosen by M (experiments/configs/trajectories.yaml);
to be confirmed or changed when D4 freezes the gate hyperparameters, and before the trajectories
are labelled:
- P1 (tilted MCMC): 2,000 proposals per run, so P1 + P2 + P3 over 3 levels x 5 seeds is about
  93k proposals, close to the ~1e5 oracle solves of CLAUDE.md 11.5. Lambda {1, 3, 10}/K and step
  0.05 are the prompt's. The start is one draw from p0.
- P2 (simulated annealing): 1,000 iterations (the prompt's), geometric cooling T_k = T0 *
  0.01^(k/999); presets (T0, Gaussian step) = (3 K, 0.02), (1 K, 0.05), (0.3 K, 0.1) for levels 1-3.
  An infeasible move is a rejected iteration, so a run has at most 1,000 proposals.
- P3 (batched gradient): 64 starts x 50 rounds (the prompt's; round 0 proposes the starts);
  x <- reflect(x - eta * g / max|g|), eta = 0.01 / 0.03 / 0.1 for levels 1-3; an infeasible step
  is halved up to 5 times, then that start stays. g comes from a differentiable rasteriser
  (models/diff_inputs.py: exact area fractions of the design-dependent channels) through the
  ensemble mean, not from finite differences, which would need 37 exact rasterise + predict
  passes per start-step (about 118k per run). The gradient only steers the search; every
  proposed design is rasterised and predicted by the exact path.
- All proposers: reflected moves on [0, 1]^36, p0 = uniform on the feasible cube (as the sampler's
  design mode), seeds 0-4, one PCG64 stream per (seed, proposer, level).
- Groups (gate/groups.py): OOD-score tercile cut points = 1/3 and 2/3 percentiles of the val split's
  scores (experiments/results/trajectories/ood_terciles.json); edge group = predicted hotspot cell
  centre within 5 mm of an edge (CLAUDE.md 8.1).
- Trajectories store r_hot and sd_hot but not s: lambda and eps are D4 values.
Owner: M; N to note the expected label count for N3.2.

## M3.5 gate experiments: settings decided by M, 2026-11-18
- Calibration source (the open item of 13 Nov): G2's q, G3's weighted q and the starting q of
  G4-G7 come from 1,000 fresh draws from the design-space density p0
  (`run_trajectories calib --n 1000`), labelled by N3.2 with the trajectories (1,000 extra oracle
  solves). This matches CLAUDE.md 9.3 and 11.1, and G3's exact P1 weights are relative to p0.
- H8's per-group slack (undefined in CLAUDE.md): slack_g = C/(gamma p T_g) +
  (2/p) sqrt(ln(4/delta)/T_g), Lemma 1's form with the group's step count T_g. It is labelled an
  empirical target in every output, because no per-group bound is proved
  (report/lemma1_review.md section 6).
- E8 feedback: when a design is verified (flagged or audited), the proposer's history shows its
  exact peak instead of T-hat. G1's flag rate f is G5's realised solves / proposals on the same
  (proposer, level, seed). E8 refuses a cell where G5 verifies every proposal or where either gate
  verifies none.
- H9 reading (to confirm): "G7 keeps validity with >= 30 % fewer audits" is judged as the cheapest
  G7 setting (p-bar, p_min) meeting H1's criterion against the cheapest G5 setting (p) meeting it.
- G3 for P2-P4: the classifier density ratio for seed s is fitted on another seed's trajectory of
  the same proposer and level (cross-fitting), never on the trajectory it weights.
- Audit replay: one design at a time, each audit applied before the next design is gated
  (assumption U1 of the M3.4 review), so the batch-scoring issue of report/lemma1_review.md
  section 5 does not arise offline.
Open, for M before E7 is reported: the Lemma 1 bound is vacuous at the trajectory lengths
(T = 1,000-3,200). For example, T = 2,000 with p = 0.1 gives about 1.4 if gamma = 0.01. H1's
criterion "M1 <= alpha + slack" is then trivially met by G5 and cannot be exceeded by G1, G2 or G4.


## N3.2 oracle settings, proposed 2026-11-19 (N to confirm by approving the PR)
experiments/oracle_label.py. Choices not fixed by CLAUDE.md or the N3.2 prompt:
- PyAMG strength threshold theta = 0.1 (solvers/amg_baseline.py leaves it to N). Speed only: the
  stop is the true relative residual <= 1e-8 whatever theta is, and every label is re-checked
  (relres, energy balance <= 0.1 %). Median 1.62 s vs 4.55 s per board in solver_table_v1.csv (D5).
  `--amg-strength-theta pyamg` keeps PyAMG's default instead; one label set uses one setting.
- Cold start (x0 = 0) for both solvers, and no envelope rescale: the label is the exact discrete
  solution of the very board design_board(x) builds, which is what the surrogate predicted.
- Dedupe key: x_hash (SHA-256 of the design vector). design_board is deterministic in x, so this
  is the board-record hash up to the record's `seed` field, which does not change the physics.
- GPU worker: FP64 batched PCG with the line-z block-Jacobi preconditioner (as data/generate.py),
  one GPU per labeller (pick it with CUDA_VISIBLE_DEVICES), batch 64 (prompt).
- Cross-check sample: ceil(0.5 % of the labels), drawn with PCG64(seed 0) over the sorted x_hashes;
  an AMG label is re-solved by PCG (on the GPU if --gpu, else NumPy on the CPU), a PCG label by AMG.
- iters_full (PCG iterations from the ensemble mean, H7's baseline) is left null: it needs mu,
  so it comes with the N3.3 audit traces, which already warm-start from mu.

## D4 · Frozen gate hyperparameters, 2026-11-23
`experiments/configs/frozen.yaml`. M asked Claude to use CLAUDE.md's suggested values and to pick
the rest; M owns the decision. Every value below is fixed before any reported run (CLAUDE.md §4).

From CLAUDE.md or earlier decisions:
- tau_K = 1 K (§8.1 "e.g. τ = 1 K"); p_audit = 0.1 (§8.1 "e.g. 0.1"), p_sweep {0.05, 0.1, 0.2}
  (§8.5); lam = 1.0 and eps_K = 1e-6 (§9.2 reference code); m_hot = 64 (§8.1 default, equal to
  the trajectories'); p_min = 0.05 (§9.2); seeds 0-4 (§4); G1 window 100 (§11.1), residual
  "scaled", c_events and delta_n "off" (D13); G6 m_buckets = 40 (D9 development default, the
  paper's bucket count).

Picked (CLAUDE.md gives no value; the reason for each):
- alpha = 0.1: the usual conformal level; §8.2 needs alpha <= 1/2.
- gamma = 0.05: the Lemma 1 slack falls steeply up to about 0.05 and flattens after (the
  table in the M3.5 note); larger gamma moves alpha_t more per audit, which §15 warns can make q
  oscillate. Fallback if q oscillates: smaller gamma (§15).
- W = 200: about 20/alpha scores, so the (1 - alpha) quantile rests on about 20 scores above it;
  small enough that after about 200 audits the window holds audited scores only (P1 at p = 0.1
  has about 200 audits per run), so the threshold can follow the proposer.
- delta = 0.05: 95 % confidence, the same level as the bootstrap CIs.
- eps_S = 0.1, eps_S_sweep {0.05, 0.1, 0.2}: score units; q is of order tau / s, so 0.1 is a
  tenth of a typical threshold, with one step either side for E14.
- p_min_sweep {0.02, 0.05}: both <= every p of p_sweep, so E16 runs every (p, p_min) pair.
- kappa = 0.5: halfway between uniform (0) and scale-proportional (1) audits; G7 is an ablation.
- G3 classifier_C = 1.0: scikit-learn's default regularisation.
- G6 r = 1 (the coarsest grid: MVP randomises between neighbouring bucket edges), eps = 0.1,
  noise_width = 0.05 (small next to q; D9 rule 4 keeps the Lemma 1' direction for any w >= 0).
  UNSURE (D9): the smoothness bound rho has not been re-derived for this w; H8 stays empirical.
- bootstrap: 10,000 resamples, seed 0.

Which slack H1 is judged against: the §8.2 bound, as CLAUDE.md states H1, unchanged in code.
At these settings it is above 1 for most p = 0.1 runs (P1 T = 2,000: 1.13; P3 T = 3,200: 0.90)
and informative only at p = 0.2 (0.50-0.87), so the report gives H1 at p = 0.2 alongside the
headline p = 0.1 and says the p = 0.1 bound is vacuous. The tighter bound (4) with C+ and the
in-expectation form (report/lemma1_review.md §3) need N's sign-off (N3.1) before they are used
in a verdict.

## kappa-hat z by AMG-CG on the 4-layer board, proposed 2026-11-23 (N to confirm)
- **Finding:** the E13 smoke run at 7710fb8 gave no certificate on any of 5 test boards. CLAUDE.md
  §8.4 builds z ≈ A⁻¹1 from 3 V-cycles; on the 256² × 4 board without a heat-sunk edge the
  Ruge–Stüben V-cycle barely reduces the near-constant mode of A z = 1 (A⁻¹1 ≈ 1.5e6 K/W): on
  two laptop test boards, min(A z) stayed < 0 in about 92k–99k of 262k cells after 3 to 40
  cycles, so no κ̂ exists. The §8.6 prototype passed with 3 cycles because it ran on the 64²
  single-layer plate.
- **Proposal:** E13 (and every caller that opts in) computes z with CG preconditioned by the same
  V-cycle (`certify.kappa.positive_z(..., accel="cg")`), 60 iterations. In the same laptop check
  this gave min(A z) between 0.98 and 1.0000 and κ̂ within about 2 % of max z on three boards,
  in about 5 s per board. Rigour is unchanged: Lemma 2(a) holds for any z > 0 with A z > 0,
  and both are still checked in FP64 with the derived margin (certify/margin.py) before κ̂ is
  used. The default of positive_z stays 3 V-cycles (§8.4); E13 states its choice in
  `--kappa-accel` / `--kappa-iters` and in every output footer.
- **D10 (production PCG preconditioner), proposed:** line-z block Jacobi
  (`solvers/pcg.line_z_block_jacobi`), the one data/generate.py and the oracle GPU worker use. It
  is the block-diagonal part of the SPD A (exact inverse of each column's 4 × 4 tridiagonal
  block), so it is SPD as PCG requires (§18.5). E13 passes `--precond linez`.
- **Owner:** N.

## D4 amendment: τ sensitivity run, 2026-11-24 (M)
- **Finding (E7 at v3-freeze, τ = 1 K):** the ensemble's peak error exceeds 1 K on only 0.1 % of
  the 92,461 trajectory proposals (|T̂_peak − T_peak| quantiles: median 0.115 K, 90 % 0.302 K,
  99 % 0.587 K; computed on rvcse from labels.parquet and the trajectories). With no gate at all
  (G0) M1 is 0.11 %, so every gate meets α = 0.1 trivially and E7/E15/E16 cannot tell the gates
  apart. The run stays the pre-registered headline and is reported as such.
- **Amendment:** one sensitivity run with `experiments/configs/frozen_tau0.2.yaml`, identical to
  frozen.yaml except τ = 0.2 K, where 25.3 % of proposals have a peak error above τ (2.5× α), so
  the large-error event is common enough for the gates to differ. Chosen after seeing the
  τ = 1 K results and the error distribution; reported next to the headline with this reason,
  never instead of it. Outputs: experiments/results/{e7,e15,e16}_tau0.2.
- **Unchanged:** the H1/H8 verdict lines still compare M1 with the §8.2 bound, which is vacuous at
  p = 0.1 (D4 entry); the report reads M1 against α directly as the empirical evidence.

## Pointwise w by AMG-CG, proposed 2026-11-24 (N to confirm)
- **Finding:** E13 at v3-freeze (1,000 test boards) saved 45 % of the PCG iterations at 0.1 K
  with the best certificate, below H6's 50 % bar; the §15 week-6 rule asks for the pointwise
  variant. That variant was crippled by the same V-cycle weakness as κ̂ (decision "kappa-hat z by
  AMG-CG"): w₀ from 4 V-cycles is far from A⁻¹|r̂|, so the fix-up t·z dominates w (median
  pointwise certificate on μ 8.75e3 K for a true error of 2.51 K).
- **Change:** `certify.pointwise.pointwise_bound(..., accel="cg")` computes w₀ with n iterations
  of CG preconditioned by the same V-cycle; E13 takes `--pointwise-accel` / `--pointwise-iters`.
  Validity is unchanged: Step 4 still checks w ≥ 0 and fl(A w) ≥ up(|r̂| + τ) before w is used.
  The default stays 4 V-cycles (§8.4).
- **Laptop check (one 256² board without a heat-sunk edge, not a report number):** the bound on μ
  fell from 216 K (4 V-cycles) to 23 K (60 CG iterations; true error 1.56 K); along PCG from μ the
  pointwise certificate reached 0.1 K at iteration 460 against 840 for Lemma 2(b).
- **Re-run:** E13 again from a new tag with `--pointwise-accel cg --pointwise-iters 60`; the
  v3-freeze run stays in the repo and is reported as the Lemma 2(b) / 4-V-cycle row. D1 already
  judges H6 at 0.1 K counting the pointwise certificate.
- **Owner:** N.
