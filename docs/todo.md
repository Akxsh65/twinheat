# TwinHeat to-do list

Updated 13 Nov 2026, after the no-hardware and cal-split decisions (week 4; week 1 began 19 Oct). Owners follow CLAUDE.md §2: **N** numerics and
parallel, **M** ML, **G** graphics, VR and hardware. "GPU" means the rvcse A100 (`gpu_instructions.md`);
everything else runs on a laptop. Tick items in the PR that finishes them.

## Running now

- [ ] **E5 accuracy and the week-5 go/no-go** (M, GPU). Started 13 Nov on rvcse GPU 0, from 45913b6.
      When it finishes: commit `experiments/results/e5/` and read `go_no_go.md`.

## Week 5 (Gate 2)

- [ ] **Week-5 go/no-go decision** (M + N), from E5's `go_no_go.md`, logged in `docs/decisions.md`.
      On validation, every FNO member is above the 3 % rel. L2 target, hotspot error is below 1.5 K,
      and FNO+local comes closest (`experiments/results/baselines/`). §12's fallback is to run the gate
      experiments at 128² and say so. Using FNO+local as the main surrogate instead would be a team
      decision (five more members, about two days of GPU).
- [ ] **Energy-loss ablation** (M, GPU), the last E5 row (§12). M first sets `seed`, `epochs` and
      `energy_weight` under `energy_ablation` in `experiments/configs/e5.yaml`.
- [ ] **N1.4 bandwidth table** (N, GPU, about 10 min). Overdue for Gate 1. Merge `n/gpu-name-nul`
      first, so the CSV's GPU column has no NUL padding. Read the peak from the card (memory clock
      × bus width), check it against NVIDIA's A100 datasheet, record both in `docs/refs/`, then run
      `bench/stencil_bw.py` with `--peak-gbs` and `--peak-source` on an idle GPU.
- [ ] **Merge the scope change** (`docs/no-hardware`): hardware dropped, cal split kept. All three
      review (it changes CLAUDE.md).
- [ ] **Merge the N4.2 margin wiring** (`n/n42-margin`). N reviews; M checks the derivation, Step 7 of
      `certify/margin.py`.
- [ ] **Pilot pre-registration** (G, §13, "pre-registered in week 5"): `study/prereg.md`. The pilot
      itself runs only after ethics approval.

## Reviews and sign-offs (people, no GPU)

- [ ] **Margin derivation**, `certify/margin.py` Steps 1–7 (M checks, §2). UNSURE: rounding during
      assembly is outside the bound, so the report should say "exact solution of the assembled system".
- [ ] **Refs files checked against their PDFs.**
  - [ ] `docs/refs/deepoheat_v2_alg1.txt`
  - [ ] `docs/refs/bastani2022_mvp.txt` (seven paper typos marked [sic]; our readings are UNSURE)
  - [ ] `docs/refs/angelopoulos_bates_cp.txt`
  - [ ] `docs/refs/pdebench_darcy_fno.txt`
- [ ] **M3.2 risks** (#43).
  - [ ] G6 uses the natural log in f(n) and K_ε, where the paper writes "log".
  - [ ] G6 takes the first sign change when there are several.
  - [ ] G3's classifier weights are linear in the design vector, and the [0.05, 20] clip hides extremes.
  - [ ] G7's audit-probability formula is ours and has no cited validity argument.
- [ ] **D9 smoothing-noise argument**: whether it keeps the Lemma 1′ direction, and re-deriving the
      smoothness bound ρ (checker).
- [ ] **`raster_check.png` second reviewer** (M1.1).
- [ ] **Tag `v0-contracts`**: all three signatures in `docs/contracts.md` first (M0.1).

## Phase 1 leftovers

- [ ] **Solver table**: re-run `solver_table_v1.csv` on a tagged commit (its GPU columns came from the
      dirty 7ad689a) before quoting it (N, GPU). Strip the NUL padding of the numba GPU name in
      `bench/solver_bench.py` and `scripts/check_gpu.py` first.
- [ ] **Preliminary tables**: re-run `python -m bench.prelim.reproduce` on a tagged commit after the
      N4.2 margin wiring merges; the committed `bench/results/prelim_*.csv` used the placeholder margin.

## Phase 3: gates and experiments (Gate 3, end of week 8)

- [ ] **M3.3 proposers and groups** (M). P1–P4 and `gate/groups.py`; trajectories for P1–P3 × 3 levels
      × 5 seeds. P3 needs the GPU.
- [ ] **N3.2 oracle labels** (N). About 1e5 solves to 1e-8 on the CPU pool. The test compares CPU and
      GPU labels on 50 boards.
- [ ] **N3.3 certified audits** (N), using the batched PCG.
- [ ] **Calibration scores for G2 and G3** (M). The cal split stays ID-mode (decided 13 Nov), but
      §9.3 and §11.1 calibrate G2 and G3 on the design-space density p₀. Decide where p₀ scores come
      from (for example design-space boards labelled by N3.2's oracle solves), or say that G2 and G3
      are calibrated on the ID distribution. G5–G7 are unaffected (Lemma 1 holds for any initial q).
- [ ] **M3.4 Lemma 1 and 1′ write-up check** (M drafts, N signs). G6's per-group result is proved by
      week 7, or reported as empirical by week 8.
- [ ] **D4 frozen hyperparameters** in `experiments/configs/frozen.yaml` (M, week 6), committed before
      any reported run. Includes:
  - [ ] α, τ, γ, W, λ, m, p, ε_S, p_min
  - [ ] G6: m, r, ε and noise width
  - [ ] G7: κ
  - [ ] G3: the classifier's C
  - [ ] G1: the flag rate f, tuned to the budget
- [ ] **Experiments, 5 seeds each**: E7, E8, E9 (Darcy replication, needs the GPU), E14, E15, E16.
- [ ] **Week-6 decision point** (§15): certificate tightness on 4 layers. If the saving is < 30 % or the
      bound is > 50× the true error, use the pointwise variant.
- [ ] **Week-8 decision point**: H1 and H6 in hand, and the G6 proof or "empirical".

## Phase 4: GPU certify-on-demand

- [ ] **N4.2 batched certified PCG on the GPU** (N, GPU). `@gpu` tests, zero B7 violations. The CPU
      modules now use the derived margin (`n/n42-margin`); the GPU kernels must use the same margin.
      Needs D10.
- [ ] **N4.3** fused gate kernel and scan compaction (N, GPU).
- [ ] **N4.4 benchmarks** (N, GPU): E3, E4, E13, rooflines, nsys profiles.
- [ ] **M4.2 equivalence test** (M, GPU).
- [ ] **G4.1 / G4.2** anytime rendering and E11 latency (G, needs the GPU server running).

## Phase 5: server, viewer, study

- [ ] **M5.1** server v1 with a GPU worker (M).
- [ ] **N5.1** server profiling (N), **N5.2** CUDA particle twin (N).
- [ ] **N5.3** DDP scaling (N). Needs 2 GPUs: rvcse GPU 0 + GPU 2.
- [ ] **G5.x viewer** (G). Targets: ≥ 60 fps desktop, ≥ 72 fps headset (§13).
- [ ] **Pilot study, E12** (G). Runs only after ethics approval.

## Phase 6: freeze and release

- [ ] **Tag `v3-freeze`** (week 10).
- [ ] **N6.1** clean-room re-run of every result from that tag.
- [ ] **E18 TwinHeat-Traj v1 release**, tag `v1.0-release` (week 11).
- [ ] **Report chapters**: `report/pagp.tex`, `dl.tex`, `cgvr.tex`, `lemmas.tex`.

## Open decisions (CLAUDE.md §17)

Decided: D1, D2, D3, D5, D6, D7, D8, D9, D13 (`docs/decisions.md`). Still open:

- [ ] **D4**: frozen gate hyperparameters (M, week 6). See Phase 3.
- [ ] **D10**: production PCG preconditioner, confirmed SPD (N, before `solvers/pcg.py` merges).
- [ ] **D11**: narrow the §8.4 fallback wording (N).
- [ ] **D12**: Tu & Giesecke theorem numbers in §8.2, and regenerating that refs file from v2 (N + M).
- [ ] **D14**: audit iteration cap (N + M, before E14).

## Deferred

- [ ] **Off-server copy** (N + M), left for later (13 Nov). Copy dataset v1 and the 14 checkpoints in
      `models/registry.json` to Drive (CLAUDE.md §4), and check their hashes. Until then rvcse holds
      the only copy.

## Housekeeping

- [ ] **Hardware clean-up** (G). Hardware validation is dropped (decisions.md, 13 Nov).
  - [ ] Remove `contracts/schemas/hardware_record.json` and its examples. This is a contracts change
        and needs the three signatures (CLAUDE.md §10).
  - [ ] Remove the `hardware/` package stub.
  - [ ] Close branch `g/g1.1-kicad-boards` unmerged.

- [ ] **Credentials on rvcse.**
  - [ ] Set a repo-local `git config user.name` / `user.email`; some commits arrived as "Bindu9468".
  - [ ] Remove the saved GitHub token from `~/.git-credentials` on the shared account, and revoke it on
        GitHub. Keep pushing with the cache-only helper.
- [ ] **`gpu_instructions.md` §0L**: change `pip install -r requirements.txt`, which must not run on
      rvcse (it would replace the CUDA 12.8 torch build), to the filtered install used there.
- [ ] **Pin `h5py`** in `requirements.txt`; it was installed by hand on rvcse for the Darcy run.
- [ ] **Input cache**: `cache_inputs` on rvcse is 194 GB, far above the expected ~35 GB, and 256²
      training is disk-bound. Find out why before any more 256² training.
- [ ] **After local test runs**, run `git checkout -- bench/results`, because the tests append to
      `bench/results/conditioning.csv`.
