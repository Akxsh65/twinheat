# CLAUDE.md — TwinHeat project context

> Standing context for every Claude session on this project.
> **Claude Code** loads this file automatically from the repo root.
> **Claude chat**: add this file to the Project's knowledge, or paste it as the first message of a new chat.
> Task prompts live in `docs/TwinHeat_Implementation_Plan.md` and point at the numbered sections here (for example "CLAUDE.md §6.5").
> When this file and any other document disagree, this file wins. Change it only by PR, with a line in `docs/decisions.md`.

---

## 0 · Rules for Claude (read first)

1. **Never invent a constant, formula, threshold or citation.** Every number comes from this file or from `config/spec_v1.yaml` / `config/spec_v2.yaml`. If something is missing, stop and ask, or list it under "Missing information" in your answer.
2. **Open decisions (§17) are not yours to make.** If a task depends on one, say so and ask.
3. **Tests first.** For every function, write the pytest tests that would catch the failure modes listed in the task, then the implementation, then list 3 ways it could still be wrong.
4. **Small diffs.** Touch only the files the task names. Show the file tree of changes first.
5. **Fail loudly.** No bare `except`, no silent fallbacks, no clipping that hides a bug. Raise with a message that names the input.
6. **Units.** SI internally (m, W, K, W/K, W/(m²·K)). Board records use **mm** for geometry. Convert at the boundary (`raster.py`) and nowhere else. θ is temperature *rise* above ambient in K. Display °C = 25 + θ.
7. **Array order** is always `[layer, y, x]` (batch first when present: `[B, layer, y, x]`). Index 0 of y is the bottom edge (S), index 0 of x is the left edge (W).
8. **The invariants in §5 must never be weakened** to make a test pass. If one fails, report it.
9. **Maths:** you may check and draft proofs, but a named human checker signs them off (§2). Mark anything you are unsure of as UNSURE.
10. **Literature:** never cite a paper the user has not given you or that is not in §16. Label suggestions from memory UNVERIFIED.
11. **Numbers for the report** come only from scripts run on a tagged commit. Never write a result number into prose or code comments yourself.
12. When a task needs material that is not in this file (paper text, an algorithm from a PDF, a datasheet table), look in `docs/refs/` (§16.2). If it is not there, ask for it. Do not reconstruct it from memory.

---

## 1 · What TwinHeat is

**One line.** Certified neural surrogates for interactive circuit-board thermal design.

**Pipeline.** A 5-member FNO ensemble predicts the steady temperature of a 4-layer, 100 × 100 mm PCB (256 × 256 × 4 = 262,144 unknowns) in milliseconds. An **audited adaptive-conformal gate** decides, for each proposed design, whether to accept the prediction or flag it to the exact solver. A random audit (coin `A_t ~ Bernoulli(p)`, drawn **after** the design is committed) is the only thing that updates the gate's threshold. Flagged and audited designs are verified in one batched, warm-started GPU PCG solve. Each board stops as soon as a deterministic **M-matrix certificate** settles its decision. A WebXR viewer shows the field converging inside a guaranteed band. Fabricated boards under a thermal camera check the solver against reality.

**Course context.** Final-year project at RVCE serving three subjects (CGVR, PAGP and Deep Learning; 120 marks, assumed 40 each). Team of 3, 12 weeks.

**Contributions (what we claim).**
1. M-matrix certificate with certified early stopping of verifications and audits (Lemma 2, H6, H7).
2. Per-proposer and per-region guarantees, people included, with certified audit labels (Lemma 1′, gate G6, H8, H4). The per-group result is a *target to be proved*, with an empirical fallback.
3. GPU certify-on-demand: one batched warm-started PCG launch, each board leaving when certified (H5, H6).
4. Evidence anchored to reality and shared: rigorous per-iterate bands in anytime rendering, 2–3 fabricated boards under a thermal camera, TwinHeat-Traj v1 benchmark release (H10, E17, E18).

**What we do NOT claim (never write these as ours).** Conformal bands for neural operators; online conformal inference with intermittently revealed labels, including the all-round transfer that Lemma 1 restates (Gibbs & Candès 2021; Tu & Giesecke 2026); audits as a source of unbiased labels (Ma & Zhao 2026); active labelling for sequential risk control (Xu et al. 2024; Zrnic & Candès 2024); gating a surrogate to a warm-started solver (DeepOHeat-v1/v2); warm starts (NOWS); energy-form physics losses; learned preconditioners; energy- and A-norm error bounds; the inequality ‖e‖ ≤ ‖A⁻¹‖·‖r‖. Learned certificates failed our preliminary test and are not claimed.

**Research questions.** RQ1 validity for any proposer and per group · RQ2 cost in exact solves · RQ3 certificate settles decisions early · RQ4 certified designs per second · RQ5 effect of the calibrated signal on designers · RQ6 solver vs real boards.

---

## 2 · Team and ownership

| Role | Person | Owns (folders) | Primary examiner | Checks |
|---|---|---|---|---|
| **N** numerics and parallel lead | `<name>` | `solvers/ certify/ pipeline/ bench/ data/generate.py` | PAGP | Lemmas 1 and 1′ (M's) |
| **M** ML lead | `<name>` | `data/sampler.py data/raster.py models/ gate/ experiments/ release/ server/` (shared) | Deep Learning | Lemma 2 + FP margin (N's) |
| **G** graphics, VR and hardware lead | `<name>` | `viewer/ hardware/ study/ server/` (shared), deck | CGVR | Contract compliance of server output; figure readability |

A file is changed only by its owner, or by a PR the owner reviews.

---

## 3 · Repository layout

```
twinheat/
  CLAUDE.md                      # this file
  config/spec_v1.yaml            # 64² development plate (§6)
  config/spec_v2.yaml            # 4-layer board (§7)
  config/design_task.yaml        # fixed parameters of the 36-dim design space (§17 open decision D2)
  contracts/schemas/*.json       # §10, validated in Python and JS
  contracts/examples/            # 3 valid + 3 invalid examples per type
  twinheat/spec.py               # loads and validates the YAMLs
  solvers/   fvm.py  mmatrix_test.py  cuda_kernels.py  pcg.py  amg_baseline.py
  certify/   kappa.py  margin.py  pcg_batched.py  peak_interval.py  pointwise.py  stopping.py
  data/      sampler.py  raster.py  envelope.py  generate.py  verify_dataset.py  manifest.json
  models/    fno.py  unet.py  inputs.py  dataset.py  losses.py  train.py  train_ensemble.py  ensemble.py  ood_score.py  registry.json  norm_stats.json
  gate/      base.py  scale.py  coin.py  aci_audit.py  certified_audit.py  multivalid.py  active_audit.py
             baselines.py (G0–G4)  proposers.py (P1–P4)  groups.py
  pipeline/  certify_batch.py (PyTorch reference)  certify.cu  scan.cu  gate_bridge.py
  server/    app.py  iterate_stream.py  ws_mock_server.py  model_stub.py
  viewer/    src/ (Three.js, GLSL, WebXR)  study/  overlay/
  hardware/  kicad/  rig/  analysis/ (register, steady, fit_convection, compare)  boards/ results/
  bench/     solver_bench.py  pipeline_bench.py  roofline.py  server_profile.py  particles_cuda.py  ddp_scaling.py  prelim/ (stop.js, cert.js)  results/  figs/
  experiments/ configs/frozen.yaml  run_gates.py  oracle_label.py  e5_accuracy.py  e7_validity.py  e8_efficiency.py
               e9_darcy.py  e11_latency/  e13_certificate.py  e14_cert_audit.py  e15_groups.py  e16_active.py  e17_hardware.py  p4_import.py
  study/     protocol.md  prereg.md  stimuli.json  analysis.py  select_stimuli.py
  release/   build_release.py  verify_release.py  traj_v1/
  report/    lemmas.tex  pagp.tex  dl.tex  cgvr.tex  claude_log.md
  docs/      TwinHeat_Implementation_Plan.md  contracts.md  decisions.md  compute.md  equipment.md  novelty_search.md  future_work.md  refs/
  tests/
```

---

## 4 · Environment and conventions

- **Python 3.11**: numpy, scipy, pyamg, numba, pytorch (CUDA matching the Kaggle driver), cupy-cuda12x, fastapi, uvicorn, websockets, pydantic ≥ 2, jsonschema, pyyaml, pandas, pyarrow, matplotlib, pytest, ruff. Exact pins in `environment.yml` and `requirements.txt` (Kaggle).
- **Node 20**: Vite, three (pinned version), ajv. WebGL2 inside the headset. WebGPU only in desktop mode, optional.
- **Compute**: Kaggle 2×T4 (about 30 GPU-h per account per week; check), Colab, laptop GPU, and a multi-core lab CPU for PyAMG and oracle solves. Real quotas are in `docs/compute.md`.
- **pytest markers**: `slow`, `gpu`. CI runs `pytest -m "not slow and not gpu"` + `ruff` on every push, and slow tests nightly. GPU tests are run by hand on Kaggle and their log is pasted into the PR.
- **Git**: `main` always green. Branches `n/<task>`, `m/<task>`, `g/<task>`. PRs need CI + one reviewer, and the reviewer runs tests locally for `solvers/ certify/ gate/`. Tags: `v0-contracts` (wk 1), `v1-solver` (wk 3), `v2-data` (wk 5), `v3-freeze` (wk 10), `v1.0-release` (wk 11).
- **Large files** never go in git. Data and checkpoints live on Kaggle Datasets + Drive, with hashes in `data/manifest.json` and `models/registry.json`.
- **Figures**: every figure script writes a footer: git commit, dataset hash, seed, hardware, date. No footer, no figure.
- **Experiments**: configs in `experiments/configs/`. Hyperparameters for reported runs come only from `frozen.yaml`, committed before the runs. A run id is `<config>-<commit>-<seed>`. 5 seeds for every gate experiment.
- **Code style**: type hints; docstrings state units; vectorised numpy (no Python loops over cells); deterministic seeds (`numpy.random.Generator(PCG64)`, torch seeds, Philox on device).

---

## 5 · Invariants (barriers). Never weaken these.

| ID | Invariant | How it is checked |
|---|---|---|
| B1 | Every assembled A (1- and 4-layer) is a nonsingular **M-matrix**: off-diagonals ≤ 0, diagonal > 0, every row sum ≥ 0 with at least one > 0, coupling graph connected | `solvers/mmatrix_test.py` on 50 random boards per configuration |
| B2 | Solver correctness: manufactured-solution error slope 2.0 ± 0.1 on N = 32…256; energy balance error ≤ 0.1 % per board | `tests/test_fvm_*.py` |
| B3 | CUDA PCG, PyAMG-GMRES and SciPy direct agree to 1e-6 relative on 20 fixed boards (incl. Dirichlet edges) | GPU test |
| B4 | Linearity: scaling all chip powers by s scales θ by s (1e-9 rel) | test |
| B5 | The 8 square symmetries commute with the solver, with the Dirichlet edges and mask transformed too | test |
| B6 | All model I/O and messages validate against `contracts/schemas` (Python and JS) | CI |
| B7 | **Every certificate is a true upper bound**: certified bound ≥ true error vs a 1e-8 oracle. **Any violation fails CI and blocks results** | `tests/test_certificate.py`, and every E13/E14 run |
| B8 | **Gate isolation**: q_t changes only on audited steps; the audit coin is drawn after `commit(design)` and is never shown to a proposer; `query` messages never touch gate state | `tests/test_gate_isolation.py`, `tests/test_equivalence.py` |

**Data hygiene.** Splits are drawn from recorded seeds. Shard SHA-256 hashes live in `data/manifest.json`, and loaders refuse mismatches. The **calibration split** is used only for conformal initialisation: loaders raise if it's requested from training or early stopping. Test and OOD splits are read only by `experiments/` (`allow_eval=True`). Normalisation statistics come from **train only** and carry the dataset hash. Released splits are never edited; a fix means a new version.

---

## 6 · Physics spec v1: development plate (TwinHeat-2D v1.0, 64²)

Used for unit tests, fast iteration and the preliminary certificate reproduction. Lives in `config/spec_v1.yaml`.

### 6.1 Assumptions
| ID | Assumption | Check |
|---|---|---|
| A1 | Steady state | design target is the worst-case steady temperature |
| A2 | Temperature uniform through thickness | Bi = h·t/k_z ≤ 0.3 → with t = 1.6 mm, k_z ≈ 0.29, per-face h ≤ about 55 W/(m²·K) |
| A3 | Linear, temperature-independent properties | FR-4 and Cu vary little over 25–125 °C |
| A4 | Chip power deposited directly into the board under the footprint | T_J = T_board + P·θ_JB in post-processing only |
| A5 | Radiation linearised into h or neglected | adds about 5–6 W/(m²·K) per face near room temperature |
| A6 | Edge (side-wall) convection neglected | edge area is 3.2 % of the two-face area |

### 6.2 PDE (θ = T − T∞ on Ω = (0,L)²)
−∇·(G ∇θ) + H θ = q in Ω, with
G(x) = k_FR4,xy·t + k_Cu·t_Cu·(n_plane + φ(x)),  H(x) = h_top(x) + h_bot(x).
φ ∈ [0,1] is painted copper coverage on the signal layer; n_plane ∈ {0,1,2} full internal planes.
BCs: θ = 0 on Γ_D (heat-sunk edges); −G ∂θ/∂n = 0 on Γ_N (free edges). H ≥ 2h_min > 0, so the problem is well posed even with Γ_D = ∅.
Discrete operator: symmetric positive definite **and** a nonsingular M-matrix (B1).
Global energy balance: ∫ q dA = ∫ H θ dA + ∫_Γ_D (−G ∂θ/∂n) ds.
Linearity: for fixed G, H and edges, θ is linear in q. T∞ is not a model input.

### 6.3 Constants (SI)
| Symbol | Value | Range / options |
|---|---|---|
| L | 0.100 m | — |
| t | 1.6e-3 m | — |
| t_Cu | 35e-6 m (1 oz) | 17.5e-6 (0.5 oz), 70e-6 (2 oz); 3 oz = 105e-6 is OOD |
| k_FR4,xy | 0.81 W/(m·K) nominal | sampler U[0.81, 1.06] |
| k_FR4,z | 0.29 W/(m·K) | 0.29–0.343 |
| k_Cu | 390 W/(m·K) | 385–400 |
| T∞ (display only) | 25 °C | — |
| Natural convection | 5–15 W/(m²·K) per face (training) | gases 2–25 |
| Forced convection | 25–50 W/(m²·K) (training) | gases 25–250 |
Sheet conductance check values (1.6 mm board): bare FR-4 G = 1.30e-3 W/K (k_eff 0.81); one 1 oz layer φ=1: 1.50e-2 (9.3); two 2 oz planes + painted 2 oz: 8.32e-2 (52). Max training contrast G_max/G_min ≈ 64.

### 6.4 Discretisation
| Rule | Value |
|---|---|
| Grid | N = 64 cell-centred FV, Δx = L/N = 1.5625 mm; transfer tests N ∈ {32, 128, 256} |
| Minimum feature | ≥ 2 cells at N = 64 (3.125 mm); chips ≥ 3 cells; smaller is OOD |
| Rasterisation | analytic shapes, 4×4 supersampling per cell (sample points at (i+0.5)/4) → φ = area fraction, clipped to [0,1] |
| Face conductance | harmonic mean G_f = 2 G_P G_N / (G_P + G_N) |
| Cell equation (W) | Σ_f G_f (θ_P − θ_Nf) + H_P Δx² θ_P = q_P Δx² |
| Dirichlet face | half-cell distance: adds 2 G_P θ_P to the left side (2 G_P on the diagonal) |
| Neumann face | no contribution |
| Unknown index | k = y·N + x |
| Data-generation tolerance | relative residual ≤ 1e-10 |
| Verification | MMS slope 2.0 ± 0.1 on N = 32…256; energy balance ≤ 0.1 % |

### 6.5 Board sampler v1 (ID distribution; one recorded seed per shard)
- Copper weight (all layers): {0.5, 1, 2} oz with p = {0.2, 0.6, 0.2}. Internal planes n_plane ∈ {0,1,2}, p = {0.4, 0.4, 0.2}. k_FR4,xy ~ U[0.81, 1.06].
- Painted copper: regions ~ U{2..8}; type rect 0.50 / disc 0.25 / Manhattan trace 0.25. Rect sides U[5, 40] mm, axis-aligned. Disc radius U[3, 15] mm. Trace 2–4 axis-aligned segments, width U[3.2, 6] mm, segment length U[10, 50] mm. Centre uniform over the board, clipped. Total mean φ ∈ [0.10, 0.60], otherwise resample.
- Chips: count U{1..4}; rectangle short side U[4.7, 20] mm, aspect U[1, 2]; non-overlapping with ≥ 2 mm gap, ≥ 3 mm from edges (rejection, max 100 tries, then resample the count). Power log-uniform [0.1, 3] W; total ≤ 6 W (rescale all if exceeded); q = P_i/A_i uniform over the footprint.
- Convection: base h_top, h_bot each U[5, 15]; with p = 0.3 one forced-air rectangle (sides U[20, 100] mm) where h_top ~ U[25, 50].
- **Envelope rule**: after solving, if max θ > 100 K, scale every P_i by 100/max θ (θ scales exactly, no re-solve). Record the fraction rescaled; if it's above 20 %, lower the power range instead.

### 6.6 Boundary conditions
Default: all four edges Neumann. Heat-sunk edges n_D ∈ {0,1,2} with p = {0.5, 0.3, 0.2}, which edges uniform (opposite or adjacent allowed), θ = 0. Partial-edge Dirichlet is OOD-6. Symmetry augmentation must transform the edge set and mask. Model encoding: mask = 1 on cells adjacent to a Dirichlet edge.

### 6.7 OOD sets (each moves ONE axis; the rest stays ID)
| Set | Axis | Training support | OOD values |
|---|---|---|---|
| OOD-1 | chip count | 1–4 | 5–8 |
| OOD-2 | chip power | 0.1–3 W, total ≤ 6 W | 3–6 W per chip, total ≤ 12 W (envelope still applied) |
| OOD-3 | copper geometry | rect, disc, Manhattan trace | L-shapes, annuli, 45° traces |
| OOD-4 | thin features | ≥ 3.2 mm | traces 0.8–2 mm |
| OOD-5 | contrast | G_max/G_min ≤ 64 | 3 oz copper, 3–4 planes (≈130–160) |
| OOD-6 | boundary | whole-edge Dirichlet, ≤ 2 edges | 3–4 edges, or partial-edge segments |
| OOD-7 | convection | natural 5–15, forced 25–50 | natural 2–5, forced 50–100 |
| OOD-8 | resolution | N = 64 | N = 128, 256 (same ID boards); a discretisation test |
Runtime OOD flag: (1) rule check, any sampler parameter outside its training range; (2) feature check, Mahalanobis distance of pooled FNO latents above the validation 95th percentile. Flagged boards always go to the exact solver; bands are shown but marked not guaranteed.

### 6.8 v1 model I/O (64² development model only)
6 channels: log10 G (train mean/std), q (÷ train P99), H (train mean/std), Dirichlet mask, x/L, y/L. Output θ in K, normalised by train std.
v1 splits: 7,000 / 1,000 / 1,000 / 1,000, OOD-1…7 × 1,000, resolution 200 at each of N = 128, 256.

---

## 7 · Physics spec v2: TwinHeat-Board v2.0 (4 layers, main model)

Lives in `config/spec_v2.yaml`. The 64² plate (§6) stays as the development configuration.

### 7.1 Stack (1.6 mm)
| Layer | Copper | Role | Dielectric below |
|---|---|---|---|
| L1 top | 35 or 70 µm, painted φ₁ | signal; chips (heat sources); convective face h_top | prepreg 0.20 mm |
| L2 | 35 µm, plane φ₂ ∈ [0.7, 1] with cut-outs | ground plane | core 1.06 mm |
| L3 | 35 µm, split plane φ₃ ∈ [0.5, 1] | power plane | prepreg 0.20 mm |
| L4 bottom | 35 or 70 µm, painted φ₄ | signal; convective face h_bot | — |
If the fab's stack differs, the YAML is changed to the **as-fabricated** stack (decision logged).

### 7.2 Equations (θ_l = rise on layer l = 1…4)
−∇·(G_l ∇θ_l) + Σ_{m=l±1} g_lm (θ_l − θ_m) + h_l θ_l = q_l
G_l = k_Cu · t_Cu,l · φ_l + k_FR4,xy · ½ (d_{l−1,l} + d_{l,l+1})   [W/K]  (for L1/L4 the missing outer dielectric term is 0, see open decision D3)
g_lm = (k_FR4,z + ν(x,y) · k_Cu · A_v) / d_lm   [W/(m²·K)], with ν in vias per m²
h_1 = h_top, h_4 = h_bot, h_2 = h_3 = 0. q only on L1.
A_v = π (D·t_p − t_p²) = 2.16e-8 m² (drill D = 0.3 mm, plating t_p = 25 µm).
Discrete cell equation for cell (l,y,x), in W: Σ_in-plane faces G_f(θ_P − θ_N) + Σ_{m=l±1} g_lm Δx² (θ_P − θ_m) + h_l Δx² θ_P = q_l Δx². **The Δx² on g_lm and h_l is mandatory.**
Edges follow §6.6 and apply to all layers. Unknown index k = l·N² + y·N + x. The system is SPD and a nonsingular M-matrix: all couplings (G_f, g_lm) are non-negative → a Z-matrix; rows are weakly diagonally dominant, strictly where h_l > 0 or a heat-sunk edge exists; the graph is connected → irreducibly diagonally dominant. The property **fails** for temperature-dependent conductivity and for central-differenced advection (outside the certified scope).

### 7.3 Numbers that matter
| Quantity | Value |
|---|---|
| Grid | 256 × 256 × 4 = 262,144 unknowns; Δx = 0.390625 mm; 7-point layered stencil |
| Material contrast | k_Cu/k_FR4,z ≈ 1,345× |
| In-plane sheet contrast on L1 | ≈ 170× (35 µm) to ≈ 340× (70 µm) |
| Through-plane contrast | ≈ 30× between no vias and 1 via/mm² |
| Single via, 1.6 mm board | ≈ 190 K/W by our formula (an online calculator gives 407 K/W with another convention; we use ours) |
| κ̂ ≥ ‖A⁻¹‖∞ | measured in week 2 from z ≈ A⁻¹1 (3 V-cycles) |
| κ₂(A) | measured by Lanczos in week 2; expected > 1e4 |
| Exact solve cost | expected 0.3–2 s with PyAMG on one CPU core. **Go/no-go week 3: below 0.3 s → 6 layers or 384²** |

### 7.4 Sampler v2.0 (additions to §6.5)
- L1 copper as §6.5. L4 painted coverage 5–40 %. L2 plane with 0–3 rectangular cut-outs. L3 split into 2–3 rectangles.
- Thermal via arrays under each chip with p = 0.6, density U[0.2, 1] via/mm². Background signal vias 0–0.05 via/mm².
- Outer copper 1 or 2 oz; inner 1 oz. Chips 1–4 on L1, 0.1–3 W each, total ≤ 6 W. Envelope θ_max ≤ 100 K by linear rescale.
- **Design-space component**: half of all training boards come from x ∈ [0,1]³⁶: 4 chip positions (8 numbers), 6 copper rectangles on L1 (24 = 6 × (x, y, w, h)), via density under each chip (4). Fixed parameters (chip sizes, powers, other layers, h, edges) live in `config/design_task.yaml` (open decision D2). Uniform density p₀ on the cube, so the weighted-conformal baseline is exact for the tilted proposer.
- OOD sets v2 (9 × 500): OOD-1…4, 6, 7 of §6.7 + `ood_bottom_chips` + `ood_3oz` (3 oz on L1/L4) + `ood_dense_vias` (1–1.5 via/mm²). OOD-5 is not OOD on 4 layers (L1 contrast is ≈ 170–340× in training); `ood_3oz` is the v2 contrast set (`docs/decisions.md`, "OOD sets v2").
- Splits v2: train 6,000 / val 1,000 / cal 1,000 / test 1,000 + 9 OOD × 500. Fields stored as float16 (about 0.5 MB/board, about 7 GB total); inputs re-rasterised from records.

### 7.5 Model I/O v2 (13 → 4 channels)
Input `[B, 13, 256, 256]` float32, channel order:
0–3 log10 G_l (l = 1..4) · 4 q on L1 (W/m²) · 5–7 log(1 + ν) for gaps L1–L2, L2–L3, L3–L4 (ν in vias/mm²) · 8 h_top · 9 h_bot · 10 Dirichlet mask · 11 x/L · 12 y/L.
Normalisation per channel from train stats (mean/std; q ÷ train P99), stored in `models/norm_stats.json` with the dataset hash. Channels 10–12 are not normalised.
Output `[B, 4, 256, 256]` float32 = θ_l in K (network predicts normalised θ; the server denormalises).

---

## 8 · Method and guarantees

Two layers. The **statistical layer** bounds how often a badly wrong prediction is accepted, for any proposer. It applies known online-conformal results. The **deterministic layer** uses the M-matrix structure to certify any approximate field, so each exact solve stops when its decision is settled. Both compare the surrogate with the **exact discrete solver**. The solver-vs-real-board gap is measured separately (H10).

### 8.1 The certified loop (one step t)
1. **Propose.** Any proposer produces design x_t (tilted sampler, annealing, gradient population, or a person). It may use all history, including the current model, but **never the audit coin**.
2. **Predict.** The 5-member FNO ensemble gives mean μ and spread σ. One stencil pass gives the hotspot residual r_hot = mean |(Aμ − b)/diag A| over the m hottest cells of μ (any layer; m = 64 default, the same for all gates). The label-free scale is s_t = r_hot + λ·σ_hot + ε. Group g_t (proposer, aggressiveness, OOD tercile, hotspot within 5 mm of an edge) comes from label-free quantities.
3. **Gate.** B_t = q_t · s_t (per-group q_{t,g} in G6). Accept if B_t ≤ τ (e.g. τ = 1 K), else flag.
4. **Audit.** After x_t is committed, the server draws A_t ~ Bernoulli(p_t). G5: p_t = p (e.g. 0.1). G7: p_t = max(p_min, π(s_t, σ_t, g_t)), label-free. The coin is never shown before commit.
5. **Verify.** Flagged ∪ audited designs go into one batched warm-started GPU solve. Each board leaves when its certificate settles its decision: a flagged board when its whole field is certified within τ_v (e.g. 0.1 K); an audited board when its certified score interval lies on one side of q_t, or is narrower than ε_S.
6. **Update.** Only audited designs update q, using the certified upper score S⁺ (Lemma 1′). Flagged labels may retrain the model but **never** touch q. Oracle solves for evaluation always run to 1e-8.

Score: S = |T̂_peak − T_peak| / s.

### 8.2 Lemma 1 (proposer-agnostic control of accepted errors; after Gibbs & Candès 2021 and Tu & Giesecke 2026)
Assumptions: (A1) the proposer is any process (adversarial or human allowed) whose x_t depends only on history before A_t. (A2) A_t ~ Bernoulli(p), independent of that history and of x_t (missing completely at random). (A3) q_t is updated only on audited steps by α_{k+1} = α_k + γ(α − err_k), with q = ∞ when α_k ≤ 0 and q = 0 when α_k ≥ 1. The model may be retrained between steps.
Claim: let W_T = fraction of the T proposals accepted while their true peak error exceeded τ. For α ≤ ½ and δ ∈ (0,1), with probability ≥ 1 − δ:
**W_T ≤ α + C/(γ p T) + (2/p)·√(ln(4/δ)/T),  C = max(α₁, 1 − α₁) + γ.**
With n_a ≈ pT audits, the slack is O(1/(γ n_a)) + O(√(ln(1/δ)/(p n_a))).
Proof sketch: (1) an accepted design with error > τ has error > B_t, so W_T ≤ M_T, the miscoverage over all steps [ours]; (2) on audited steps, ACI bounds miscoverage within C/(γ n_a) of α deterministically (Gibbs & Candès Prop. 4.1, applied to the revealed subsequence as in Tu & Giesecke Thm 5.4) [cited]; (3) transfer to all steps under MCAR revelation is Tu & Giesecke Thm 5.6, in expectation [cited]. Here it's made high-probability because m_t is fixed before the coin, so (A_t − p)·m_t is a bounded martingale difference → Azuma–Hoeffding, with Hoeffding controlling n_a/(pT) [ours].
Corollary: among accepted predictions, the error rate ≤ W_T / accepted fraction. Report both.
Limits: a long-run average, not per step (errors can cluster); says nothing about final design quality (H2); holds only if the proposer can't see the coin before committing.
**Citation numbering (open, D12):** in the provided v2 PDF (arXiv 2605.17705v2), Thm 5.4 is the spatial similarity-weight bound and 5.6 is an *assumption*. The results meant in steps (2)–(3) are **Prop. 5.5** (revealed-round identity) and **Thm 5.9** (eq. 10, revealed rounds w.h.p.; eq. 11, all rounds in expectation). Exact statements and what they force on the code: §18.1.

### 8.3 Lemma 1′ (certified audit scores)
If an audit returns a certified interval [T⁻, T⁺] for the true peak instead of the exact value, record S⁺ = max(|T̂_peak − T⁻|, |T̂_peak − T⁺|)/s ≥ S, and use S⁺ for err_t = 1[S⁺ > q_t] and in the score window. Lemma 1 holds for the S⁺ stream: an accepted large error implies S > q_t, hence S⁺ > q_t. Price: q_t slightly conservative when intervals are wide. The audit stops when the interval is decided relative to q_t or narrower than ε_S (E14).

### 8.4 Lemma 2 (M-matrix certificate)
Let A be a nonsingular M-matrix, x̃ any approximation of θ with Aθ = b, r = b − Ax̃, e = θ − x̃.
(a) If z > 0 and Az > 0 componentwise, then ‖A⁻¹‖∞ ≤ κ̂ := max_i z_i / min_i (Az)_i.
(b) Hence ‖e‖∞ ≤ κ̂ · ‖r‖∞.
(c) If w ≥ 0 and Aw ≥ |r| componentwise, then |e| ≤ w, and max_i(x̃_i − w_i) ≤ T_peak ≤ max_i(x̃_i + w_i).
Proof: A⁻¹ ≥ 0 and Az ≥ min(Az)·1 ⇒ A⁻¹1 ≤ z/min(Az), and ‖A⁻¹‖∞ = max entry of A⁻¹1 → (a). (b) from e = A⁻¹r. (c): |e| = |A⁻¹r| ≤ A⁻¹|r| ≤ A⁻¹Aw = w.
**Decision-sufficient stopping.** Per design: z ≈ A⁻¹1 by 3 multigrid V-cycles (any positive z is valid; accuracy only affects tightness); check min(Az) > 0 and min z > 0 once in **FP64**. At PCG iterate k, c_k = κ̂·‖r_k‖∞ bounds every cell's error. Flagged design: stop when c_k ≤ τ_v. Audit: stop when the certified score interval is decided relative to q_t or narrower than ε_S. Residual and Az in FP64 with a **derived** rounding margin (`certify/margin.py`). Never certify on the FP32 recursive residual alone. Pointwise variant (c) where a sharper bound is worth an extra approximate solve: Gauss–Seidel smoothing of the prediction, approximate A w = |r| solve (4 V-cycles) + two-SpMV fix-up, validity asserted.
Scope: certifies distance to the exact discrete solution only. Needs the M-matrix property (true for 4-layer conduction, Darcy, upwinded advection–diffusion). Where it fails, fall back to A-norm CG bounds (Meurant & Tichý). Caveats on that fallback, Varah's z = 1 special case and PCG details: §18.5 (and D11).

### 8.5 Extensions
- **G6, per-group (Extension A).** Groups 𝒢: proposer P1–P4, aggressiveness (3 levels), OOD-score tercile, hotspot within 5 mm of an edge; groups may intersect. Per-group thresholds q_{t,g} updated on audited steps with MultiValid Prediction (Bastani et al. 2022). Target statement (to be proved, week 7): for every group with n_{a,g} audits, the accepted-error rate in g ≤ α + slack(n_{a,g}). Argument: membership is fixed before the coin, so each group's audits are an unbiased subsample, and Lemma 1's transfer applies per group. **Fallback by week 8: report as empirical.** MVP algorithm, its assumptions and the gaps the proof must close: §18.3.
- **G7, risk-adaptive audits (Extension B).** p_t = max(p_min, π(s_t, σ_t, g_t)), mean p_t = p̄; π never sees the label. Update α_{t+1} = α_t + γ·(p̄/p_t)·A_t·(α − err_t). Weights have conditional mean 1, bounded by 1/p_min. π interpolates between uniform and uncertainty-proportional sampling (Zrnic & Candès 2024; Li, Zrnic & Candès 2025). Reported as an ablation, not a headline. Neither Gibbs & Candès (fixed γ) nor Tu & Giesecke (no IPW update) covers this variable step: G7 has no cited validity argument yet (§18.1).
- **Why audits:** updating q from flagged labels only calibrates on the wrong population (G4 shows the bias). Sweep p ∈ {0.05, 0.1, 0.2}.

### 8.6 Preliminary evidence (5 Oct 2026, 64² plate, emulated errors; to be reproduced in Phase 1 and repeated on 4 layers in E13)
Certified early stopping, Lemma 2(b), warm start from the emulated prediction:
| Variant | Warm-start error | PCG iters to 1e-8 | Certified ≤ 1 K | Certified ≤ 0.1 K | Saved at 1 K | κ̂ ÷ true ‖A⁻¹‖∞ | Bound ÷ true error at stop |
|---|---|---|---|---|---|---|---|
| Default, one heat-sunk edge | 2 % | 170 | 32 | 65 | 81 % | 1.067 | 24× |
| No heat-sunk edge | 2 % | 184 | 30 | 76 | 84 % | 1.069 | 17× |
| Four chips, one near a corner | 5 % | 174 | 43 | 74 | 75 % | 1.067 | 26× |
| Thin copper, weak convection | 2 % | 160 | 29 | 61 | 82 % | 1.005 | 14× |
Pointwise certificate, Lemma 2(c), on the prediction itself:
| Emulated error | GS sweeps | True worst error (K) | Varah bound (K) | Best pointwise cert (K) | Cert ÷ true |
|---|---|---|---|---|---|
| 1 % | 0 | 1.26 | 350 | 40.0 | 32× |
| 1 % | 5 | 0.81 | 24 | 0.98 | 1.2× |
| 1 % | 10 | 0.78 | 13 | 0.82 | 1.05× |
| 3 % | 0 | 2.95 | 1,200 | 125 | 42× |
| 3 % | 5 | 1.87 | 40 | 3.16 | 1.7× |
| 3 % | 10 | 1.69 | 35 | 2.56 | 1.5× |
With noisier emulated errors: 59–66 % fewer iterations at 1 K, 44–51 % at 0.1 K, zero violations (H6's 50 % bar at 0.1 K is at risk, see D1). Learned certificates failed validity in most trials and are not claimed. The 5 Oct scripts (`stop.js`, `cert.js`) were never saved; `bench/prelim/` holds the surviving widget, a cross-check, and `reproduce.py`, a declared re-run that is new evidence, not a reproduction (`docs/decisions.md`, "§8.6 preliminary evidence").

---

## 9 · Reference code (behavioural spec; copy semantics, fix noted bugs)

### 9.1 Warm-started PCG and certified variant (CuPy; batched versions add a leading batch axis + per-board stop masks)
```python
def pcg(Aop, b, x0, Minv, tol=1e-8, maxit=20000):
    x = x0.copy(); r = b - Aop(x); z = Minv(r); p = z.copy()
    rz = cp.vdot(r, z); bnorm = cp.linalg.norm(b)
    for it in range(maxit):
        Ap = Aop(p); alpha = rz / cp.vdot(p, Ap)
        x += alpha * p; r -= alpha * Ap
        if cp.linalg.norm(r) < tol * bnorm: return x, it + 1
        z = Minv(r); rz_new = cp.vdot(r, z)
        p = z + (rz_new / rz) * p; rz = rz_new
    return x, maxit

def certificate_constant(Aop64, vcycles, n_cycles=3):          # Lemma 2(a), once per design
    z = vcycles(cp.ones(n), n_cycles)                          # z ≈ A⁻¹1; any positive z is valid
    Az = Aop64(z); m = float(Az.min())
    if m <= 0 or float(z.min()) <= 0: return None              # no certificate: solve to tolerance
    return float(z.max()) / m * (1 + 1e-6)                     # PLACEHOLDER margin: replace with certify/margin.py

def pcg_certified(Aop, Aop64, b, x0, Minv, kappa, tau_v, maxit=20000):
    x = x0.copy(); r = b - Aop(x); z = Minv(r); p = z.copy(); rz = cp.vdot(r, z)
    for it in range(maxit):
        Ap = Aop(p); alpha = rz / cp.vdot(p, Ap)
        x += alpha * p; r -= alpha * Ap
        if kappa * float(cp.abs(r).max()) <= tau_v:            # cheap test on the recursive residual
            c = kappa * float(cp.abs(b - Aop64(x)).max())       # FP64 true residual before certifying
            if c <= tau_v: return x, it + 1, c                  # Lemma 2(b): |error| ≤ c in every cell
        z = Minv(r); rz_new = cp.vdot(r, z)
        p = z + (rz_new / rz) * p; rz = rz_new
    return x, maxit, None
```

### 9.2 Certify-on-demand step (PyTorch reference; P5 fuses residual → decision into one CUDA kernel)
```python
def certify_batch(models, x, A, b, q_t, tau, p_bar=0.10, p_min=0.05, lam=1.0, m=64, tau_v=0.1):
    preds = torch.stack([f(x) for f in models])            # [M, B, 4, N, N]
    mu, sd = preds.mean(0), preds.std(0)
    r = (A.apply(mu) - b) / A.diag                          # residual in kelvin
    hot = mu.flatten(1).topk(m, dim=1).indices              # hottest cells, any layer
    r_hot  = r.flatten(1).abs().gather(1, hot).mean(1)
    sd_hot = sd.flatten(1).gather(1, hot).mean(1)
    s = r_hot + lam * sd_hot + 1e-6                         # label-free scale
    B = q_t * s                                             # bound on peak error (K)
    flag  = B > tau
    p_aud = audit_prob(s, sd_hot, p_bar, p_min)             # = p_bar for uniform audits (G5); label-free
    audit = torch.rand_like(B) < p_aud                      # coin drawn after design is fixed
    ids = torch.nonzero(flag | audit).squeeze(1)            # stream compaction (scan)
    cert = batched_pcg_certified(A[ids], b[ids], x0=mu[ids],
                                 tau_v=tau_v, q=q_t, s=s[ids], audit=audit[ids])
    return mu, B, flag, audit, p_aud, ids, cert, s
```

### 9.3 Gate loop G5 with certified, weighted audits
```python
# initialise q from 1,000 calibration boards drawn from the design-space density p0
alpha_t, scores = alpha, list(cal_scores)
q_t = quantile(scores, 1 - alpha_t)
for step in search:                                    # any proposer
    mu, B, flag, audit, p_aud, ids, cert, s = certify_batch(models, x, A, b, q_t, tau)
    for k in audited(ids):                             # audit stream only (Lemma 1, A3)
        T_lo, T_hi = cert[k].peak_interval             # certified, early-stopped (Lemma 2)
        S_hi = max(abs(peak(mu[k]) - T_lo), abs(peak(mu[k]) - T_hi)) / s[k]
        err = float(S_hi > q_t)                        # conservative score (Lemma 1')
        w = p_bar / p_aud[k]                           # = 1 for uniform audits (G5); IPW weight (G7)
        alpha_t = alpha_t + gamma * w * (alpha - err)  # Gibbs & Candès update
        scores.append(S_hi)
        q_t = inf if alpha_t <= 0 else (0.0 if alpha_t >= 1 else quantile(scores[-W:], 1 - alpha_t))
    retrain_buffer.extend(flagged(ids))                # certified labels: model only, never q
# G6 keeps one threshold per group and replaces the alpha_t update with MultiValid Prediction
```

### 9.4 Layered Jacobi sweep, shared-memory tile (Numba CUDA). **Known bug: skips boundary cells; production code must handle them (separate boundary kernel or guarded halo).**
```python
TB = 16
@cuda.jit
def jacobi_layered(T, Tn, Ge, Gw, Gn, Gs, gup, gdn, diag, rhs):   # arrays [L, N, N]
    s = cuda.shared.array((TB + 2, TB + 2), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    i, j = cuda.grid(2)
    L, n0, n1 = T.shape
    for l in range(L):
        if i < n0 and j < n1: s[tx+1, ty+1] = T[l, i, j]
        if tx == 0    and i > 0:    s[0, ty+1]    = T[l, i-1, j]
        if tx == TB-1 and i < n0-1: s[TB+1, ty+1] = T[l, i+1, j]
        if ty == 0    and j > 0:    s[tx+1, 0]    = T[l, i, j-1]
        if ty == TB-1 and j < n1-1: s[tx+1, TB+1] = T[l, i, j+1]
        cuda.syncthreads()
        if 0 < i < n0-1 and 0 < j < n1-1:
            v = (Ge[l,i,j]*s[tx+1,ty+2] + Gw[l,i,j]*s[tx+1,ty]
               + Gn[l,i,j]*s[tx,ty+1]   + Gs[l,i,j]*s[tx+2,ty+1])
            if l > 0:     v += gdn[l,i,j] * T[l-1, i, j]
            if l < L - 1: v += gup[l,i,j] * T[l+1, i, j]
            Tn[l, i, j] = (v + rhs[l, i, j]) / diag[l, i, j]
        cuda.syncthreads()
```

### 9.5 Blelloch exclusive scan, one block (CUDA C; add a block-sums pass for large batches; pad for bank conflicts)
```c
__global__ void scan_block(const int* in, int* out, int n) {
  extern __shared__ int t[]; int i = threadIdx.x, off = 1;
  t[2*i] = (2*i < n) ? in[2*i] : 0; t[2*i+1] = (2*i+1 < n) ? in[2*i+1] : 0;
  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads(); if (i < d) { int a = off*(2*i+1)-1, b = off*(2*i+2)-1; t[b] += t[a]; } off *= 2; }
  if (i == 0) t[n-1] = 0;
  for (int d = 1; d < n; d *= 2) {
    off >>= 1; __syncthreads();
    if (i < d) { int a = off*(2*i+1)-1, b = off*(2*i+2)-1; int v = t[a]; t[a] = t[b]; t[b] += v; } }
  __syncthreads(); if (2*i < n) out[2*i] = t[2*i]; if (2*i+1 < n) out[2*i+1] = t[2*i+1];
}
```

### 9.6 Particle advection (WebGL2, GLSL ES 3.0)
```glsl
uniform sampler2D uPos, uFlux; uniform float uDt, uSeed; in vec2 vUv; out vec4 o;
vec2 F(vec2 p){ return texture(uFlux, p).xy; }            // flux in board UV units
void main(){
  vec4 s = texture(uPos, vUv); vec2 p = s.xy; float age = s.z;
  vec2 mid = p + 0.5*uDt*F(p);                              // RK2 midpoint
  p += uDt*F(mid); age += uDt;
  bool dead = any(lessThan(p, vec2(0.))) || any(greaterThan(p, vec2(1.)))
           || length(F(p)) < 1e-4 || age > 3.0;
  if (dead) { p = respawn(vUv, uSeed); age = 0.; }          // importance-sampled near sources
  o = vec4(p, age, 1.);
}
```

---

## 10 · Contracts v1.0 (frozen in week 1, tag `v0-contracts`)

General: every record and message has `"schema_version": "1.0"`. Field arrays (`mu`, `sd`, `theta`) are **base64 of little-endian float16**, row-major `[layer, y, x]`, with an explicit `"shape": [4, n, n]`. A `"units"` object states kelvin / metres / watts where relevant. Changes need all three signatures in `docs/contracts.md`. Additive changes bump the minor version; existing fields never change meaning.

### 10.1 Board record (JSON; geometry in **mm**, origin lower-left, x right, y up)
```json
{
  "schema_version": "1.0", "id": "str", "seed": 0, "mode": "id | design | ood_<name>",
  "stack": {"outer_oz": 1, "inner_oz": 1, "k_fr4_xy": 0.81},
  "layers": [
    {"layer": 1, "shapes": [{"type": "rect", "x0": 0, "y0": 0, "w": 10, "h": 5},
                            {"type": "disc", "cx": 50, "cy": 50, "r": 6},
                            {"type": "trace", "points": [[10,10],[40,10],[40,30]], "width": 4}]},
    {"layer": 2, "plane": true, "cutouts": [{"type": "rect", "x0": 0, "y0": 0, "w": 0, "h": 0}]},
    {"layer": 3, "split_rects": [{"type": "rect", "x0": 0, "y0": 0, "w": 0, "h": 0}]},
    {"layer": 4, "shapes": []}
  ],
  "chips": [{"x0": 40, "y0": 40, "w": 10, "h": 8, "P": 1.2, "side": "top"}],
  "via_arrays": [{"x0": 40, "y0": 40, "w": 10, "h": 8, "density_per_mm2": 0.5}],
  "background_via_density_per_mm2": 0.02,
  "convection": {"h_top": 10.0, "h_bot": 8.0, "forced_zone": null},
  "dirichlet_edges": ["W"],
  "design_vector": null
}
```
`forced_zone` = null or `{"x0","y0","w","h","h_top"}`. `dirichlet_edges` ⊆ {W, E, S, N}. `design_vector` = 36 floats in [0,1] or null. `side` = "top" (ID) or "bottom" (`ood_bottom_chips`).

### 10.2 Model I/O
```
Model input   [B, 13, 256, 256] float32   (§7.5 channel order)
Model output  [B, 4, 256, 256]  float32   temperature rise per layer, kelvin (after denormalisation)
```

### 10.3 WebSocket messages
```
client → server
  {type:"query",  id, record}          # while dragging: prediction only; NEVER touches gate state, NEVER draws a coin
  {type:"design", id, record}          # commit: server calls coin.commit(id) first, then gates
server → client
  {type:"pred",     id, mu, sd, B, decision, group, peak, units}
                    decision ∈ preview | accept | flag | audit   (preview only for query)
  {type:"iterate",  id, k, theta, cert_k}        # every k PCG iterations; cert_k in K (Lemma 2 bound)
  {type:"verified", id, theta, cert, iters, ms, stop_reason}
  {type:"batch_progress", batch_id, active_ids, done:[{id, iters, cert, reason}],
                    stage_ms:{gate, scan, pcg, scatter}, designs_per_s}   # throughput panel
  {type:"error",    id, message}
```
Server timestamps for E11: every server message may carry `t_server_ns` (monotonic).

### 10.4 Audit log (CSV, one row per proposal)
`run_id, gate, seed, t, proposer, aggressiveness, group, x_hash, s, r_hot, sd_hot, q_t, B, decision, p_t, audited, T_lo, T_hi, S_hi, err, alpha_t, iters_used, oracle_peak (offline only)`

### 10.5 Hardware record (JSON)
`board_id, record_id, power_per_resistor:[{ref, V, I}], ambient_C, camera_frame_id, camera_model, emissivity, sensor_readings:[{id, x_mm, y_mm, T_C}], steady_state:{criterion:"<0.1 K/min over 5 min", met: bool}, timestamp`

### 10.6 Session log (study; JSONL, one action per line)
`participant (code, never a name), condition, t_ms (relative), action ∈ grab|move|release|brush|via_stamp|undo|request_verify|accept|query|commit, design_hash (at commit), server_decision`

---

## 11 · Gates, proposers, metrics, hypotheses, experiments

### 11.1 Gates (compared at equal exact-solve budget)
| ID | Gate | Source |
|---|---|---|
| G0 | no gate; final design verified once | — |
| G1 | hotspot-residual gate, sliding-window (FIFO, 100) percentile, fixed flag rate f (tuned to match budget) | DeepOHeat-v2 Algorithm 1 (text in `docs/refs/`); details and differences §18.4, D13 |
| G2 | static split conformal on s, calibrated once on p₀ | quantile level §18.2, D8 |
| G3 | weighted split conformal | Tibshirani et al. 2019; Fannjiang et al. 2022; Choi 2026. Exact weights for P1; classifier density ratio (clipped) for others |
| G4 | ACI updated on **flagged** labels only (selection-bias ablation) | — |
| G5 | audited ACI, uniform audits, certified audits (Lemma 1, 1′) | p ∈ {0.05, 0.1, 0.2} |
| G6 | audited multivalid, one threshold per group | Bastani et al. 2022 (algorithm text in `docs/refs/`) |
| G7 | audited ACI with risk-adaptive audits and floor p_min | §8.5 |

### 11.2 Proposers (all over the 36-dim design space; none ever sees coins)
| ID | Proposer | Density known? | Aggressiveness knob (3 levels) |
|---|---|---|---|
| P1 | tilted sampler p(x) ∝ p₀(x)·exp(−λ·T̂_peak(x)), MCMC | yes, up to a constant | λ ∈ {1, 3, 10} per K |
| P2 | simulated annealing, 1,000 steps | no | start temperature + step size presets |
| P3 | batched gradient descent through the surrogate, 64 starts × 50 steps | no | step size, steps |
| P4 | logged human sessions from the pilot, replayed | no | — |

### 11.3 Metrics
M1 accepted-error rate over all proposals (Lemma 1) and among accepted; M1g the same per group · M2 exact solves used (flags + audits) and PCG iterations spent · M3 best verified peak vs cumulative exact solves (curve, AUC, value at a fixed budget) · M4 surrogate–true gap at the returned design · M5 certified designs/s and per-stage wall-clock · M6 certificate: iterations to certified τ_v, certificate ÷ true error, violations vs oracle (**must be 0**) · M7 model-form error: hotspot and field error, verified solver vs thermal camera on held-out boards, with an uncertainty budget.

### 11.4 Hypotheses and success criteria
| H | Hypothesis | Success | If it fails |
|---|---|---|---|
| H1 | G5 keeps M1 at target for P1–P4 at every aggressiveness; G1, G2, G4 exceed it as aggressiveness rises; G3 holds only for P1 | M1 ≤ α + Lemma 1 slack in ≥ 95 % of runs (5 seeds) | check A1 (coin visible?) and small n_a |
| H2 | at equal M2, G5's final verified peak is no worse than G1's | non-inferiority margin 0.2 K, paired over seeds | report the trade-off curve over p |
| H3 | H1 replicates on Darcy | same criterion | report conditions as a limitation |
| H4 | calibrated signal lowers the accepted-wrong rate in the pilot | pre-registered direction + effect size; exploratory | null pilot; design a larger study |
| H5 | GPU pipeline certifies much faster than the sequential CPU loop | ≥ 10× certified designs/s at batch ≥ 64 | report crossover batch + time breakdown |
| H6 | certified early stopping cuts PCG iterations on 4 layers | ≥ 50 % fewer than fixed 1e-8 at τ_v = 0.1 K (judging rule: D1), zero violations; vs fixed loose tolerance at equal mean iterations, report how often the uncertified field misses τ_v | pointwise variant; keep Lemma 2 for CG-2 |
| H7 | certified audits keep validity at lower cost | M1 within Lemma 1′ slack; ≥ 50 % fewer audit iterations | report cost–width trade-off over ε_S |
| H8 | G6 holds per group where G5 doesn't | M1g ≤ α + slack(n_{a,g}) for every group in ≥ 95 % of runs; G5 exceeds in ≥ 1 group under aggressive proposers | merge sparse groups; empirical if proof incomplete |
| H9 | G7 keeps validity with fewer audits (exploratory) | H1's criterion with ≥ 30 % fewer audits | report negative; uniform stays default |
| H10 | verified solver predicts real boards | after fitting h_top, h_bot on board 1 only, held-out hotspot rise within max(2 K, 10 %) | report model-form error with budget |

### 11.5 Experiments
E1 solver verification (MMS, energy balance, κ₂ by Lanczos, M-matrix) · E2 kernel roofline · E3 solver comparison (direct, PyAMG-GMRES, Jacobi, red-black, PCG cold/warm; latency + batch throughput) · E4 certify pipeline vs sequential CPU loop (H5) · E5 surrogate accuracy (FNO vs U-Net, ID + 9 OOD, 128→256, energy-loss ablation) · E6 static coverage of conformal bounds ID/OOD · E7 gate validity G0–G7 × P1–P4 × 3 × 5 seeds (H1) · E8 efficiency at equal budget + audit-rate sweep (H2) · E9 Darcy replication (H3) · E10 rendering performance (frame time, headset fps) · E11 anytime-verification latency · E12 pre-registered pilot (H4) · E13 certificate on 4 layers (H6) · E14 certified audits, ε_S sweep (H7) · E15 per-group validity (H8) · E16 risk-adaptive audits, p_min sweep (H9) · E17 hardware validation (H10) · E18 TwinHeat-Traj v1 release.
For H1, H8 and H9, proposers run on surrogate predictions, so one oracle-labelled trajectory set scores all gates offline. H2 reruns G1 and G5 with feedback. Oracle solves: about 1e5, always to 1e-8.

### 11.6 Darcy second task
PDEBench 2D Darcy (128², 10,000 samples, u = 0 on all edges, constant forcing β). Design task: place 4 high-permeability inclusions in a parametric space to minimise peak u. Same FV code with H = 0, all edges Dirichlet (an M-matrix, so Lemma 2 applies). Scope: same operator family; shows transfer across dataset, BC and task, not new physics.

---

## 12 · Surrogate configuration
| Model | Role | Config |
|---|---|---|
| FNO (layers as channels) | main surrogate, 13 → 4 | 4 Fourier layers, width 48, 24 modes; 5-member ensemble (different seeds and data order) |
| U-Net | local baseline at sharp copper edges | 4 levels, 32 base channels, GroupNorm |
| FNO + local 3×3 branch | ablation for sharp interfaces | — |
Training: pretrain 128² (2×2 area-average downsampling of fields; inputs re-rasterised at 128), fine-tune 256². Loss = relative L2 + w_hot · mean |pred − true| over the top-64 hottest true cells (w_hot default 0.1; log both terms). AdamW, cosine with warm-up, AMP with FP32 loss, grad clip 1.0, early stop on validation hotspot error (K). 8-symmetry augmentation, transforming the mask and the x/y channels. Energy-form loss ½θᵀAθ − bᵀθ is one ablation row at 0 / 100 / 1k labels (not a claim). FNO validated first on PDEBench Darcy. **Week-5 go/no-go: ID mean relative L2 < 3 % and hotspot error < 1.5 K**, otherwise run the gate experiments at 128² and say so.

---

## 13 · Viewer spec (CGVR)
Three.js supplies the scene graph and WebXR session only; every technique below is written by the team. WebGL2 in the headset. Board shown 1 m wide. Desktop parity for every action.
| ID | Feature | Techniques |
|---|---|---|
| T1.1 | layer peel: 4 planes pulled apart (two-handed / slider); via arrays as instanced emissive columns, brightness ∝ vertical flow ν·k_Cu·A_v·(θ_l − θ_m)/d; FR-4 translucent slab with weighted blended OIT; grabbable cut plane | 3D interaction, instancing, OIT |
| T1.2 | flux particles: F = −k∇T per layer (RG float texture); 262k particles in a 512² float texture, RK2 ping-pong; importance-sampled respawn ∝ |F|; age; trails in an accumulation buffer; layer hops at vias ∝ vertical flux; LIC second mode | GPGPU, vector-field vis |
| T1.3 | LUTs (inferno default, ironbow, white-hot); blackbody LUT (Planck × CIE 1931 → XYZ → linear sRGB), board range mapped to 800–2,000 K and labelled stylised; HDR half-float target; own bloom + ACES/AgX; isolines fract(T/ΔT) with fwidth | transfer functions, colour science, post-processing |
| T1.4 | realism: JFA signed distance field from the copper mask → bevel height → normal map; procedural traces (A*); copper PBR, clear-coat solder mask, silkscreen; instanced packages; IBL | procedural, PBR |
| T1.5 | VR: grab (ray far / direct near), 1 mm snap, ghost footprint, 30 Hz `query` while dragging, `design` on release; probe pen + haptics; copper brush; via stamp; undo; teleport, seated mode | picking, direct manipulation, haptics |
| T1.6 | trust: shimmer (≤ 2 Hz) / hatching / VSUP; badge Predicted (amber) → Verifying (blue) → Verified (green), driven only by server messages | uncertainty vis |
| T1.7 | anytime verified rendering: blend μ → iterate_k → verified on GPU; band = cert_k (rigorous); labelled "guaranteed band" | progressive rendering |
| T1.8 | proposer replay from audit logs, decisions colour-coded, best-verified-peak curve | animation |
| T1.9 | measured overlay: thermal frame → homography (4 fiducials) → projective texture on L1 + difference map | registration, projective texturing |
Targets: desktop ≥ 60 fps; headset ≥ 72 fps; first iterate < 100 ms after the gate fires; certified field < 2 s. Deferred: heat haze, volume ray marching/marching cubes, AR placement, WebGPU inference.
Pilot (CG-1): 8–10 participants with PCB or heat-transfer coursework, within-subjects, 3 counterbalanced conditions (none / DeepOHeat-v2-style residual flag / calibrated bound + anytime verification). Task A: 30 predictions, accept or request verification. Task B: cool a board below a limit in 5 min with a fixed solve budget. Measures: accepted-wrong rate, unnecessary-verification rate, verified peak, time, NASA-TLX. Pre-registered in week 5; effect sizes; exploratory. **Runs only after ethics approval.**

## 14 · Hardware validation (E17)
Boards: 2–3, 100 × 100 mm, 1.6 mm, 4-layer to §7.1, each with 2–3 SMD power resistors (2512, ≤ 50 % rated power each) and 4 fiducials at (5,5), (95,5), (95,95), (5,95) mm. They differ in pours and via arrays. Board 1 is for fitting only; the others are held out. Generated from board records by `hardware/kicad/gen_board.py`, so simulated and physical boards match.
Rig: bench supply with measured V and I, current-limited; still-air enclosure; low-conductivity non-flammable standoffs; matte black coating of known emissivity; thermal camera on the top face; 2–3 NTC thermistors/thermocouples; ambient logged. Steady state = hotspot change < 0.1 K/min (over 5 min). Never unattended.
Analysis: register frames (homography); compare with L1 of the verified solver; fit h_top, h_bot on board 1 only (bounds [2, 50]); report hotspot error, field RMSE and an uncertainty budget (camera accuracy, emissivity, power, ambient, fit). `fit.json` is committed before the held-out comparison.

## 15 · Schedule, decision points, fallbacks
Weeks: P0 wk 1 · P1 wk 1–3 · P2 wk 3–5 · P3 wk 5–8 · P4 wk 4–8 · P5 wk 4–10 (freeze end wk 10) · hardware wk 2–9 · P6 wk 10–12.
Decision points: **wk 3** solve ≥ 0.3 s (else 6 layers or 384²), M-matrix test passes, certificate prototype reproduces §8.6 · **wk 5** surrogate accuracy go/no-go · **wk 6** certificate tightness on 4 layers (saving < 30 % or bound > 50× true error → pointwise variant) · **wk 8** H1 and H6 in hand, G6 proof or "empirical".
Cut list (in order): LIC mode → procedural traces → Darcy validity-only → G7 offline only → pilot of 6 → two boards.
Key fallbacks: q oscillates → smaller γ, larger W, p = 0.2 early then 0.1 · small groups (< ~100 audits) → merge or raise p for rare groups · no camera → 6–8 thermistor array · no boards by wk 8 → E17 to journal extension · headset < 72 fps → fewer particles, half-res post, opaque layers · new prior work → narrow G1.

## 16 · References and reference material
### 16.1 Core citations (verified in Phase 0)
DeepOHeat-v2 (Yu et al., arXiv 2608.16080, 2026) · Gibbs & Candès, ACI, NeurIPS 2021 · Tu & Giesecke, arXiv 2605.17705 (2026) · Ma & Zhao, arXiv 2608.01378 (2026) · Xu, Karampatziakis & Mineiro, NeurIPS 2024 · Zrnic & Candès, ICML 2024 · Li, Zrnic & Candès, NeurIPS 2025 · Bastani et al., Multivalid CP, NeurIPS 2022 · Lekeufack et al., Conformal Decision Theory · Fannjiang et al., PNAS 2022 · Choi, ICML 2026 · Tibshirani et al. 2019 · Meurant & Tichý 2018 · Varah 1975 · Astral (ICLR 2026) · Hillebrecht & Unger 2023 · Chin 2026 · Li et al., FNO, ICLR 2021 · PDEBench (Takamoto et al. 2022) · McGreivy & Hakim 2024 · NOWS 2025 · Cresswell et al., ICML 2024 · Correll, Moritz & Heer, VSUP, CHI 2018 · Cabral & Leedom, LIC 1993 · Zhang & Chen, PLOS ONE 2024 · IC-ThermBench 2026. Full list: Phase 0 §1 and plan §16/§20.
### 16.2 `docs/refs/` (humans add these; Claude reads them, never reconstructs them)
`deepoheat_v2_alg1.txt` (Algorithm 1 text) · `bastani2022_mvp.txt` (MultiValid algorithm) · `gibbs_candes_prop41.txt` (Lemma 4.1 + Prop. 4.1), `tu_giesecke_thm54_56.txt` (theorem statements; must be taken from **v2**: Prop. 5.5, Assumptions 5.6/5.8, Thm 5.9, Props 5.7, B.14, B.15; see D12) · `pdebench_darcy_fno.txt` (published FNO Darcy error) · `cie1931_2deg_5nm.csv` · `planckian_locus_check.csv` (reference chromaticities) · `fab_stackup.txt` · `camera_datasheet.pdf` · `bench/prelim/` (surviving 5 Oct widget; the original `stop.js`, `cert.js` were never saved, see `docs/decisions.md`).

## 17 · Open decisions (Claude must ask, not assume)
| ID | Decision | Owner | Due |
|---|---|---|---|
| D1 | How H6 is judged: at 1 K, or at 0.1 K counting the pointwise certificate. Fixed in `decisions.md` before E13 | N + M | wk 1 |
| D2 | Fixed parameters of the 36-dim design task (chip sizes and powers, other layers, h, edges) → `config/design_task.yaml` | M | wk 2 |
| D3 | Outer-layer G_l dielectric term (§7.2): 0 for the missing outer dielectric, as written, or another convention | N | wk 1 |
| D4 | Frozen gate hyperparameters α, τ, γ, W, λ, m, p, ε_S, p_min, chosen on validation trajectories → `experiments/configs/frozen.yaml` | M | wk 6 |
| D5 | Week-3 go/no-go outcome (keep 256²×4, or 6 layers / 384²) | N + M | wk 3 |
| D6 | Week-1 start date of the 12-week schedule | all | wk 0 |
| D7 | ACI upper boundary: §8.2 sets q = 0 when α_k ≥ 1, so err = 1[S⁺ > 0]; Gibbs & Candès Lemma 4.1 needs err = 1 whenever α_k > 1. Force err = 1 there, or keep q = 0? (differs only if S⁺ = 0 exactly) (§18.1) | M (Lemma 1 checker) | before gate code |
| D8 | G2 initial quantile: plain `quantile(scores, 1 − α)` (§9.3) or the split-conformal level ⌈(n+1)(1−α)⌉/n with "higher" interpolation (§18.2); and whether the ACI path uses the same correction | M | before gate code |
| D9 | G6 / MVP settings: bucket count m_MVP, grid factor r, ε, score-noise width for smoothness; whether they go in `frozen.yaml` under D4; whether `query` previews may show a randomised q_{t,g} or only a value fixed at commit (§18.3) | M | before E15 |
| D10 | Production PCG preconditioner `Minv`, and confirmation that it is SPD (symmetric V-cycle: symmetric smoothing, R = Pᵀ) (§18.5) | N | before `solvers/pcg.py` merges |
| D11 | Narrow the §8.4 fallback wording? Meurant & Tichý needs SPD A and a guaranteed μ ≤ λ_min; it covers neither failure case §7.2 names (§18.5) | N | — |
| D12 | Correct the Tu & Giesecke theorem numbers in §8.2 to v2 (Prop. 5.5, Thm 5.9) and regenerate `docs/refs/tu_giesecke_thm54_56.txt` from v2 | N + M | — |
| D13 | G1 details not fixed by §11.1: residual definition (paper: raw (A T̂ − b); ours: divided by diag A), and whether G1 copies the paper's stop-checking-after-C-events rule (§18.4) | M | before E7 |
| D14 | Audit delay bound: maximum number of steps D an audit solve may take before its label is used, enforced by a label-independent `maxit` (§18.1, Tu & Giesecke Prop. B.15) | N + M | before E14 |

---

## 18 · Implementation notes from the literature survey

Distilled from the eight survey study guides (`grand_guide.pdf`; each guide used only its provided PDF). Paper numbering is the paper's own. Items marked UNSURE are the guide author's inference and need the named checker (§2). Nothing here changes §5 or decides anything in §17.

### 18.1 Lemma 1 sources (G5): exact statements and what they force on the code
**Gibbs & Candès (arXiv 2106.00170v3).** Update (2): α_{t+1} = α_t + γ(α − err_t), err_t = 1[S_t > Q̂_t(1 − α_t)]. Standing convention: α₁ ∈ [0,1], Q̂_t non-decreasing, Q̂_t(x) = −∞ for x < 0 and +∞ for x > 1.
- Lemma 4.1: α_t ∈ [−γ, 1 + γ] for all t, almost surely.
- Prop. 4.1: |(1/T) Σ err_t − α| ≤ (max{α₁, 1 − α₁} + γ)/(Tγ), for **every** sequence. It is deterministic telescoping, so it holds for any proposer and for S⁺ in place of S. It allows S_t and Q̂_t to change every step (retraining OK).
- Thms 4.1/4.2 (HMM, fixed score, spectral gap) do **not** apply to TwinHeat. Never quote them as guarantees. The paper's γ = 0.005 is not ours (D4).
- Code consequences: q = ∞ when α_k ≤ 0 ⇒ err = 0 (matches). When α_k ≥ 1 the paper's set is empty ⇒ err = 1; §8.2's q = 0 gives err = 1[S⁺ > 0] (D7). **Testable invariant for `gate/aci_audit.py`:** on the audit stream, α_k never leaves [−γ, 1 + γ], and |mean(err) − α| ≤ C/(γ n_a) after every audit, for any adversarial 0/1 sequence. This is exact, not statistical, so it can be a hard unit test.

**Tu & Giesecke (arXiv 2605.17705v2).** Dictionary: their R_t ↔ our A_t, π ↔ p, ℓ_t ↔ err_t, "revealed round" ↔ audited step. Level update (5): α_{t+1} = α_t + γ R_t (α − ℓ_t), i.e. only on revealed rounds.
- Assumption 5.8 (MCAR): P(R_t = 1 | ℋ_t) = π, with ℋ_t = ℱ_t ∨ σ(Y_t). ℋ_t contains the **committed design and its label**, so a proposer that sees A_t and then edits x_t breaks it. Showing **past** audit outcomes is allowed (they are in the filtration).
- Thm 5.9 eq. (10), w.h.p., revealed rounds: −(1 − α + γα)/(γ|ℛ_T|) ≤ mean_{ℛ_T} ℓ_t − α ≤ (α + γ(1 − α) + γ log(1/δ)/(8ζ))/(γ|ℛ_T|). Eq. (11), **in expectation**, all rounds: same form with |ℛ_T| → πT and log(1/δ) → 1. The high-probability all-step version in §8.2 is ours.
- Assumption 5.6 (fallback feasibility, ζ) exists only because W-TQA replaces the trivial set. TwinHeat does not need it **as long as q = ∞ is kept** when α_k ≤ 0. Do not "finitize" q (e.g. cap it at the window maximum) without re-deriving Lemma 1.
- Prop. 5.7: without an assumption on the reveal mechanism the all-round miss rate is not identifiable. Prop. B.14: selection bias −(1/(πT)) Σ_t Cov(p_t, ℓ_t). This is the G4 effect and the reason q never sees flagged labels.
- Prop. B.15 (delay ≤ D): constants γα, γ(1 − α) become Dγα, Dγ(1 − α); arrival time may depend on the label, but a label arriving after D is **discarded**, and the discard must itself be MCAR. Certified audits stop at label-dependent times, so every audit needs a label-independent iteration cap (D14).
- Stated for a **fixed** score; time-varying (retrained) scores are not covered (UNSURE). Safe rule: retrain only on data available before step t, never using A_t.
- G7: p_t depends on (s_t, σ_t, g_t), so MCAR with constant π fails; the IPW update is analysed in neither paper.

### 18.2 Split conformal (G2, G3) — Angelopoulos & Bates (arXiv 2107.07511v6)
- q̂ = the ⌈(n+1)(1−α)⌉/n empirical quantile of n calibration scores ("higher" interpolation). If α < 1/(n+1), q̂ = ∞. Under exchangeability: 1 − α ≤ P(Y ∈ C(X)) ≤ 1 − α + 1/(n+1) (upper bound needs continuous scores). The plain (1 − α) quantile slightly under-covers (D8).
- Coverage conditional on one calibration set ~ Beta(n + 1 − l, l), l = ⌊(n+1)α⌋. **Test for G2/G3 code:** repeatedly re-split cached scores into cal/val; mean coverage ≈ 1 − l/(n+1) and the histogram matches the beta-binomial law. A larger deviation is a bug.
- Our S = |T̂_peak − T_peak|/s is the paper's "uncertainty scalar" score |y − f̂(x)|/u(x); ensemble spread is a valid u. Validity does not depend on s_t's quality; tightness does.
- G2's guarantee lapses once the proposer leaves p₀ **or** the model is retrained (old cal scores used a different score function).
- G3 (Thm 3, covariate shift) needs w = dP_test/dP known and P_{Y|X} fixed (true: deterministic solver). Weights are exact only for P1, and the theorem is for one test point, not an adaptive sequence.
- Marginal coverage allows all errors to sit in one group; per-group calibration (Prop. 1) still needs i.i.d. data. This motivates G6.

### 18.3 MultiValid Predictor (G6) — Bastani et al. (arXiv 2206.01067v1)
- Protocol per round: learner fixes score s_t ∈ [0,1]; adversary picks (x_t, y_t); learner sees x_t (not y_t) and outputs q_t; learner sees y_t. Map our unbounded score with **S̃ = S/(1 + S)** (monotone, exact inverse q = q̃/(1 − q̃); not clipping, so §0 rule 5 is respected).
- Buckets B_m(i) = [(i−1)/m, i/m), last one closed; grid 𝒫^{rm}. f(n) = √((n+1) log^{1+ε}(n+2)), K_ε = Σ_{n≥0} 1/f(n)², η = √(ln(|𝒢|m)/(2K_ε|𝒢|m)). Per round MVP finds a sign change i* and **randomises** between i*/m − 1/(rm) and i*/m.
- Thm 3.1, **in expectation** (not w.h.p.): on each (group, bucket) subsequence of length n, |coverage − (1 − δ)| ≤ c_exp·f(n)/n with c_exp ≤ √(4K_ε|𝒢|m ln(|𝒢|m)) + ρT. Groups may intersect; one ACI per group is not possible with intersecting groups (paper Sec. 4.3.2).
- **Smoothness (ρ, rm) fails** for a deterministic solver (score given x is a point mass). Remark 3.1: add small continuous noise to the score. Noise must be drawn by the server **after commit** and hidden, like the coin. Uniform noise of width w gives ρ ≤ 1/(rmw) (guide's arithmetic, UNSURE).
- Feed MVP **only audited steps** (B8). MVP's own randomisation is drawn once per committed design; a `query` preview must not re-roll it (D9).
- Gaps the week-7 per-group proof must close: transfer from audited steps to all steps; S⁺ depends on q_t through early stopping (Thm 3.1 assumes the score is independent of the learner's randomised q_t); expectation vs. high probability; two-sided per-bucket result → one-sided per-group accepted-error rate.
- Name clash: MVP's m (buckets) ≠ our m (hottest cells, default 64); MVP's δ = our α; MVP's α(·) is a function.

### 18.4 G1 baseline and related points — DeepOHeat-v2 (arXiv 2608.16080v1)
- Algorithm 1: r_hot(T̂) = (1/m) Σ_{y∈S_m} |r(y)| with r = A T̂ − b (**not** divided by diag A) over the m predicted-hottest cells; push r_hot into a FIFO window 𝒲 for **every** proposal; flag if r_hot > Q_{1−f}(𝒲). The threshold is label-free and controls the **flag rate**, not the error rate. Paper values (Table III): |𝒲| = 100, m = 100, f_target = 0.30, ΔN = 50; checking stops after C = 3 retraining events. In TwinHeat, G1 uses the shared m (§8.1) and f tuned to budget (§11.1); open points in D13. Canonical text: `docs/refs/deepoheat_v2_alg1.txt`.
- Energy-form loss ½θᵀAθ − bᵀθ has Hessian A (condition κ₂(A)); squared residual has Hessian 2A² (condition κ₂(A)²); same minimiser. In network weights only an upper bound holds (needs full-rank Jacobian, near a good fit). Ablation row only (§12).
- Prop. 7 already bounds the hotspot error using A⁻¹ ≥ 0, but needs hotspot coincidence and row diagonal dominance of A⁻¹ and is "numerically loose". Record it in `docs/novelty_search.md`; Lemma 2 needs neither assumption.

### 18.5 Solver and certificate (Lemma 2) — Shewchuk 1994, Varah 1975, Meurant & Tichý 2018
- **Two different kappas.** κ₂(A) = λ_max/λ_min sets PCG **speed**; κ̂ ≥ ‖A⁻¹‖∞ (Lemma 2a) sets the **error bound**. Only κ̂ enters the certificate.
- **PCG (Shewchuk).** CG minimises ‖e‖_A over the Krylov space; ‖e_i‖_A ≤ 2((√κ − 1)/(√κ + 1))^i ‖e_0‖_A (Eq. 52), so at most ⌈½√κ ln(2/ε)⌉ iterations for an ε reduction in the A-norm (worst case, exact arithmetic; real runs are usually faster). For PCG, κ = κ(M⁻¹A) (UNSURE, guide's derivation) and **M must be SPD**: a V-cycle preconditioner needs symmetric smoothing and R = Pᵀ (D10). Not a certificate: wrong norm, needs λ_min, λ_max and the unknown ‖e_0‖_A.
- **Residual drift.** The recursive residual r ← r − αAp drifts in floating point and can give a "false zero residual"; conjugacy is lost too. Hence the §8.4 rule: the FP32 recursive residual is only a trigger; certify on b − Ax recomputed in FP64 plus the derived margin.
- **Warm starts.** x₀ = μ lowers ‖e_0‖, not the rate. A stopping test relative to ‖r₀‖ throws the head start away; use ‖b‖ (as §9.1 does) or the absolute certificate κ̂‖r_k‖∞ ≤ τ_v.
- **Recommended extra test (not an invariant):** ‖A − Aᵀ‖ = 0 after assembly, alongside B1 (catches one-sided Dirichlet or g_lm Δx² terms).
- **Varah.** For strictly row-diagonally-dominant A, ‖A⁻¹‖∞ ≤ 1/α, α = min_k(|a_kk| − Σ_{j≠k}|a_kj|). For an M-matrix this is exactly Lemma 2(a) with z = 1 ((A1)_k is row k's margin), i.e. the "Varah bound" column of §8.6. **Unit test for `certify/kappa.py`:** with z = 1, κ̂ equals 1/α to rounding. On the v1 plate α > 0 (margin H_P Δx², plus 2G_P at Dirichlet faces), but small, so the bound is loose. On the 4-layer board h₂ = h₃ = 0, so L2/L3 rows away from a heat-sunk edge have margin exactly 0: α = 0, Varah gives nothing, and z ≈ A⁻¹1 is required. Whether Varah's block Theorem 2 works with per-column 4×4 blocks is untested (UNSURE). That the general-z form of Lemma 2(a) is new is UNVERIFIED (may be classical M-matrix theory).
- **Meurant & Tichý.** Thm 1: ‖x − x_k‖_A² < (‖r_k‖²/μ)(‖r_k‖²/‖p_k‖²), valid only for a **guaranteed** 0 < μ ≤ λ_min (of the preconditioned matrix under PCG), exact arithmetic; decreasing in k. The practical version with a Ritz estimate μ_k is **not a bound** (Ritz values over-estimate λ_min; it underestimates early on) and must never be used as a certificate (B7). Delayed variants give iterate k's bound only at k + d. Converting A-norm to per-cell kelvin needs a rigorous λ_min again. The fallback does not cover the two failure cases §7.2 names: central-differenced advection is non-symmetric, temperature-dependent conductivity is nonlinear (D11).

### 18.6 Credit additions (extends §1 "what we do NOT claim")
Varah's ‖A⁻¹‖∞ ≤ 1/α and therefore the z = 1 case of Lemma 2(a) · DeepOHeat-v2's use of A⁻¹ ≥ 0 in an error bound (Prop. 7) · MVP, threshold-calibrated multivalid coverage, the s/(1+s) map and the noise trick (Bastani et al.) · split, weighted and group-balanced conformal (Angelopoulos & Bates summarise these; the main theorem is Vovk et al.'s) · PCG theory, warm-start advice, periodic residual recomputation (Shewchuk; not in §16.1, add it there before citing).
