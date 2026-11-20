# GPU runs to do (the college A100 server, rvcse)

Runs that need a real CUDA GPU, or that must happen on the lab machine where the data lives.
Everything else runs on a laptop. Paste each "Send back" item into the PR named in its section,
then tick the box. Updated 2026-11-12 (week 4).

Machines:
- **rvcse** (`rvcecsegpuserver`, college GPU server; the machine of record for every GPU run and
  for oracle solves): Intel Xeon Gold 6226R (64 logical), NVIDIA A100-PCIE-40GB (used as GPU 0
  and GPU 2 with `CUDA_DEVICE_ORDER=PCI_BUS_ID`). Holds dataset v1. Setup: section 0L.
- **Kaggle** (2×T4): fallback only, if rvcse is unavailable. Section 0K. No run is planned there
  (docs/decisions.md, "GPU runs on the college A100").

## Open

| # | Prompt | Where | Needs first | Needed by | Status |
|---|---|---|---|---|---|
| 9 | M2.3 U-Net and FNO+local baselines | rvcse GPU 0 / GPU 2 | nothing (running overnight from 2026-11-12) | week 5 | [ ] |
| 10 | M2.3 E5 accuracy and the week-5 go/no-go | rvcse | the two baselines committed; `git pull` (e5.yaml lists `ood_bottom_chips`) | week 5 | [ ] |
| 12 | M3.3 proposer trajectories (P1–P3 × 3 levels × 5 seeds) | rvcse GPU 0 | M3.3 merged; `ood_terciles.json` committed (step 1 of section 9) | N3.2 oracle labels, week 6 | [ ] |
| 13 | M3.5 E8 feedback runs (G1 vs G5, H2) | rvcse GPU 0 + CPU pool | frozen.yaml committed (D4); N3.2 labels + `calibration.parquet` (section 10) | week 8 | [ ] |

For 10: `python -m experiments.e5_accuracy --config experiments/configs/e5.yaml --data <dataset_v1>`
on a clean tree. Every ensemble member's val relative L2 is above the 3 % target of CLAUDE.md §12
(hotspot error is below 1.5 K): docs/decisions.md, "M2.2 members 2-5", decision pending. E5's
`go_no_go.md` is where that decision is made.

## Done

| Was | Prompt | Result |
|---|---|---|
| 4 | N1.4 stencil bandwidth table | `bench/results/stencil_bw.csv` on rvcse GPU 0, 2026-11-16: two runs agree within 2 % at B = 16 and 64; GPU tests 71 passed |
| 11 | `scripts/check_gpu.py` on the A100 | all PASS on rvcse GPU 2, 2026-11-13: `docs/compute.md` |
| 1 | N1.4 GPU tests | ran on the A100 (rvcse); two kernel fixes for a real-GPU compile merged (9434ef0, 7ad689a). Bandwidth table still open: section 4 |
| 2 | N1.5 T4 rows | not needed (section A measured the oracle machine itself) |
| A | N1.5 solver table on the oracle machine | `bench/results/solver_table_v1.csv` + `bench/figs/solver_table_v1.png` (c4f7b6f); D5: keep 256² × 4 (docs/decisions.md) |
| 3 | N2.1 pilot, then dataset v1 | pilot (seed 0, commit 743c726) reviewed by M and N; all splits generated with seed 2026 and hash-verified (docs/decisions.md, "Dataset v1") |
| 5 | N2.2 verification of the full dataset | `data/verify_report.md`: PASS, 13 splits, 0 problems (#29, #38); tag `v2-data` |
| 6 | N2.1 `ood_bottom_chips` split | generated at 23ff3f3 on rvcse GPU 2 (#37) |
| 7 | M1.3 FNO on PDEBench Darcy | results and acceptance in docs/decisions.md (#33) |
| 8 | M2.2 ensemble (5 members) | registry entries, summaries, logs (#34, #35) |

Sections 5–8 below are kept as the record of how those runs were done.

Note on A: the CSV rows carry commit `7ad689a-dirty`. CLAUDE.md §0 rule 11 wants report numbers
from a script run on a tagged commit; if the solver table goes into the report, re-run it on
`v1-solver` (or a later tag) first.

---

## 0K · Kaggle session setup (fallback only)

1. Kaggle → New Notebook → Settings: **Accelerator: GPU T4 x2**, **Internet: on**.
2. If the repo is private: Add-ons → Secrets → add `GITHUB_TOKEN` (fine-grained, read access to
   `Akxsh65/twinheat`), and tick it for this notebook.
3. First cell (`REF` is a branch or tag; each section says which):

```python
REF = "main"
import subprocess
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    url = f"https://{token}@github.com/Akxsh65/twinheat.git"
except Exception:
    url = "https://github.com/Akxsh65/twinheat.git"  # public repo
subprocess.run(["git", "clone", "-q", "-b", REF, url, "twinheat"], check=True)
%cd twinheat
!git log --oneline -1
```

4. Dependencies Kaggle does not have, and the GPU check:

```
!pip install -q pydantic==2.14.1 jsonschema==4.26.0 pyyaml==6.0.3 pytest==9.1.1 pyamg==5.3.0
!python scripts/check_gpu.py | tail -1
```

`check_gpu.py` must report all PASS (torch, CuPy, numba.cuda). If not, stop and send the output.

## 0L · rvcse setup

```
cd <repo> && git fetch origin && git checkout <REF> && git pull
python -m pip install -r requirements.txt
python scripts/check_gpu.py | tail -1        # must be all PASS
git status --short                           # must be empty for anything that goes in the report
```

---

## 4 · N1.4 · stencil bandwidth table

About 5 minutes on rvcse (section 0L), `REF = "main"`, clean tree. The peak bandwidth is a
required argument with its source (the script has no built-in GPU number):

```
python -m pytest tests/test_gpu_solvers.py -q -rs 2>&1 | tail -3
python -m bench.stencil_bw --peak-gbs 1555.2 --peak-source "docs/refs/a100_pcie_40gb_bandwidth.txt (card memory clock x bus width x 2)" 2>&1 | tee n14_bw_log.txt
```

The peak and its source are in `docs/refs/a100_pcie_40gb_bandwidth.txt`: the card's own memory
clock and bus width. The current NVIDIA datasheet lists only the 80 GB models. Never use a number
from memory. Run on one idle GPU (`CUDA_VISIBLE_DEVICES=0` or `2`, nothing else running on it)
while the CPU is quiet (not during E5 or a training run): on 13 Nov, runs during E5 differed by up
to 8× at B = 1 and 1.3× at B = 64. Run it twice; the two runs' B = 16 and 64 rows should agree
within a few per cent before the CSV is committed. B = 1 fits in the A100's 40 MB L2 cache, so
those rows are not DRAM bandwidth.

**Send back:** `bench/results/stencil_bw.csv`, `n14_bw_log.txt`, and the `pytest` summary line.

---

## 5 · N2.2 · verify the full dataset, then release it

Runs on rvcse because that is where dataset v1 is. Mostly CPU: the 1 % re-solve is about 130 of
the 13,000 boards with PyAMG-GMRES to 1e-10 (on rvcse PyAMG took 1.6–4.6 s per board at 1e-8 in
`solver_table_v1.csv`, so roughly 10–20 minutes), plus hashing about 7 GB.
`REF = "main"`, on a clean tree (the script refuses a dirty one for a release).

```
python -m data.verify_dataset --root <DATASET_V1_ROOT> 2>&1 | tee n22_verify_log.txt
echo "exit status $?"
```

`<DATASET_V1_ROOT>` is the folder with one folder per split (train, val, cal, test, ood_*).
DONE WHEN (N2.2): exit status 0, and `data/verify_report.md` + `data/figs/verify_*.png` are
committed (they are written in the repo; open a PR with them).

Then the release steps from the plan's N2.2 checklist:
1. Copy the dataset folder to Drive (no Kaggle Dataset: decisions.md, 2026-11-12).
2. Check the copied hashes match `manifest.json` (`python -m data.verify_dataset
   --root <the copy>` on the copy).
3. Tag the commit `v2-data` (CLAUDE.md §4).

**Send back:** `n22_verify_log.txt`, the PR with the report and figures, the Drive folder name.

**If it fails:** the report lists every failure. A re-solve outside 2e-3 or a cross-split duplicate
is a real problem: do not release; send the report. Never edit released shards (CLAUDE.md §5); a
fix is a new version `v2.1-data`.

---

## 6 · N2.1 · the `ood_bottom_chips` split

`data/generate.py` generates `ood_bottom_chips` once branch `n/ood-bottom-chips` is merged
(PR #20 made `solvers/fvm.py` read `q_bot`). Then, on rvcse with `REF = "main"`:

```
python -m data.generate --split ood_bottom_chips --seed 2026 --out <DATASET_V1_ROOT>/ood_bottom_chips --device gpu 2>&1 | tee n21_bottom_log.txt
python -m data.verify_dataset --root <DATASET_V1_ROOT>
```

Seed 2026 is the dataset v1 seed (docs/decisions.md). Adding a split to a released dataset changes
the release: if `v2-data` is already tagged, this becomes `v2.1-data` (CLAUDE.md §5).

**Send back:** `n21_bottom_log.txt`, the new `manifest.json`, the verification output.

---

## 7 · M1.3 · FNO and trainer on PDEBench Darcy

GPU, about a few GPU-hours (unknown until measured; 100 epochs by default). `REF = "main"`.
Two inputs only a person can provide (CLAUDE.md §0 rule 12):
1. **`docs/refs/pdebench_darcy_fno.txt`**: the published FNO test relative L2 for 2D DarcyFlow
   β = 1.0, copied from the PDEBench paper (Takamoto et al. 2022), as one line
   `fno_test_rel_l2: <number>` plus lines naming the table, page and version. Commit it.
2. **`2D_DarcyFlow_beta1.0_Train.hdf5`** from PDEBench, on rvcse (or as a Kaggle dataset).

```
python -m experiments.darcy_validate --data <path>/2D_DarcyFlow_beta1.0_Train.hdf5 --backup-dir <drive or second disk>/twinheat/darcy 2>&1 | tee m13_darcy_log.txt
```

The script refuses to run without the reference file. lr, weight decay, epochs and batch are
UNSURE defaults (the script says so): record what was used. Checkpoints go to
`checkpoints/darcy` (git-ignored) and `--backup-dir`; a killed session resumes.

DONE WHEN (M1.3): test relative L2 within ~20 % of the published number, with a healthy learning
curve. **Send back:** `m13_darcy_log.txt`, `experiments/results/darcy/` (results JSON + curve PNG).

---

## 8 · M2.2 · ensemble training

Not ready: needs M2.1 (inputs, normaliser, dataset loader) and M2.2's own code
(`models/train_ensemble.py`, Kaggle notebook template). M2.2's DONE WHEN: train **member 1 first
and measure its GPU-hours** before launching members 2–5; plan members 2–5 across rvcse and the
three Kaggle accounts from that measurement. This section gets its commands when that code exists.

---

## 9 · M3.3 · proposer trajectories for the offline gate experiments

What it does: runs P1 (tilted MCMC), P2 (simulated annealing) and P3 (batched gradient) at three
aggressiveness levels and seeds 0–4 on the 5-member FNO ensemble (`experiments/run_trajectories.py`,
settings in `experiments/configs/trajectories.yaml`). No gate and no coin is involved; the output
is the trajectory set that N3.2 labels with oracle solves and every gate is then scored on.
Expected size: P1 2,000 + P2 ≤ 1,000 + P3 3,200 proposals per run, 15 runs each, about 93k rows in
45 parquet files (a few MB each). CPU (rasterising at 256²) dominates: about 0.16 s per proposal on
a laptop core, so use several jobs.

On a clean tree at the merge commit (`git status` empty), with the rvcse environment of section 0L:

1. OOD terciles, once (validation split, about 1,000 boards):
   ```
   export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
   python -m experiments.run_trajectories terciles --data <dataset_v1> --device cuda:0
   ```
   Commit `experiments/results/trajectories/ood_terciles.json` (PR), pull, and check `git status`
   is empty again: the runs refuse a dirty tree.
2. Smoke run, one file (a few minutes); note its wall-clock time:
   ```
   python -m experiments.run_trajectories run --proposer P2 --level 1 --seed 0 --out /tmp/traj_smoke --device cuda:0
   ```
3. All 45 runs:
   ```
   python -m experiments.run_trajectories all --out <traj_dir> --device cuda:0 --jobs 6
   ```
   Each job loads its own copy of the ensemble on the GPU. P3's memory per job (autograd through
   5 members at 256², `grad_chunk: 4`) is not measured yet: watch `nvidia-smi` during the smoke
   run of P3 (`run --proposer P3 --level 1 --seed 0`), and lower `--jobs` or `grad_chunk` if the
   GPU runs out of memory. `--only P3` reruns one
   proposer; the manifest keeps the other entries.

Send back (PR "M3.3 trajectories"): `<traj_dir>/manifest.json` copied to
`experiments/results/trajectories/manifest.json`, the printed row counts, the wall-clock time and
`nvidia-smi` GPU name. Copy `<traj_dir>` to Drive (CLAUDE.md §4: data stays out of git) and tell N
its location: it is the input of N3.2 (oracle labels to 1e-8; labels keyed by `x_hash`, so
repeated designs need one solve).

---

## 10 · M3.5 · gate experiments

E7, E9, E14, E15 and E16 replay recorded trajectories and need no GPU: they run on any machine
with the inputs folder (docs/gate_experiment_inputs.md). Only the p₀ calibration set (step 1) and
E8 (step 3) need the ensemble.

1. With the M3.3 trajectories, the p₀ calibration designs:
   ```
   python -m experiments.run_trajectories calib --seed 2026 --out <traj_dir>/calibration.parquet --device cuda:0
   ```
   Hand them to N with the trajectories: N3.2 labels both into `labels.parquet`.
2. After `experiments/configs/frozen.yaml` is filled and committed (D4), and with a clean tree:
   ```
   python -m experiments.e7_validity    --inputs <inputs>
   python -m experiments.e15_groups     --inputs <inputs>
   python -m experiments.e16_active     --inputs <inputs>
   python -m experiments.e14_cert_audit --inputs <inputs>    # needs N3.3's audit_traces.parquet
   ```
3. E8, online, needs the GPU for the ensemble and CPU for the exact solves of verified designs.
   The number of solves is not measured yet; the AMG solve took about 1.6 s per board at 256²
   on one rvcse core (D5):
   ```
   python -m experiments.e8_efficiency --inputs <inputs> --device cuda:0
   ```

Send back (PR "M3.5 results"): the `experiments/results/e*/` folders and the printed verdicts.

