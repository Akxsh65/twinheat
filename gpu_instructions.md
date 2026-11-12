# GPU runs to do (Kaggle T4 and the lab GPU machine)

Runs that need a real CUDA GPU, or that must happen on the lab machine where the data lives.
Everything else runs on a laptop. Paste each "Send back" item into the PR named in its section,
then tick the box. Updated 2026-11-06 (end of week 3).

Machines:
- **rvcse** (lab GPU machine, oracle machine of record): Intel Xeon Gold 6226R (64 logical),
  NVIDIA A100-PCIE-40GB. Holds dataset v1. Setup: section 0L.
- **Kaggle** (2×T4): section 0K.

## Open

| # | Prompt | Where | Needs first | Needed by | Status |
|---|---|---|---|---|---|
| 4 | N1.4 stencil bandwidth table (`stencil_bw.csv`) | Kaggle T4 (see note) | nothing | overdue (gate G1) | [ ] |
| 5 | N2.2 verification of the full dataset + report; then release | rvcse | nothing | week 4 (tag `v2-data`) | [ ] |
| 6 | N2.1 `ood_bottom_chips` split (500 boards) | rvcse | the small generate.py fix (below) | week 4 | [ ] |
| 7 | M1.3 FNO on PDEBench Darcy | rvcse or Kaggle T4 | `docs/refs/pdebench_darcy_fno.txt` + the HDF5 file | before M2.2 | [ ] |
| 8 | M2.2 ensemble training (5 members) | rvcse + Kaggle | M2.1 code, section 7 passed | week 5 (accuracy go/no-go) | [ ] |

## Done

| Was | Prompt | Result |
|---|---|---|
| 1 | N1.4 GPU tests | ran on the A100 (rvcse); two kernel fixes for a real-GPU compile merged (9434ef0, 7ad689a). Bandwidth table still open: section 4 |
| 2 | N1.5 T4 rows | not needed (section A measured the oracle machine itself) |
| A | N1.5 solver table on the oracle machine | `bench/results/solver_table_v1.csv` + `bench/figs/solver_table_v1.png` (c4f7b6f); D5: keep 256² × 4 (docs/decisions.md) |
| 3 | N2.1 pilot, then dataset v1 | pilot (seed 0, commit 743c726) reviewed by M and N; all 12 splits generated with seed 2026 and hash-verified (docs/decisions.md, "Dataset v1") |

Note on A: the CSV rows carry commit `7ad689a-dirty`. CLAUDE.md §0 rule 11 wants report numbers
from a script run on a tagged commit; if the solver table goes into the report, re-run it on
`v1-solver` (or a later tag) first.

---

## 0K · Kaggle session setup

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

About 5 minutes. `REF = "main"`. The N1.4 DONE WHEN asks for achieved GB/s of the naive and
tiled stencils "vs ~320 GB/s peak", which is the **T4's** nominal figure, and `bench/stencil_bw.py`
hard-codes that 320. So run it on a Kaggle T4:

```
!python -m pytest tests/test_gpu_solvers.py -q -rs 2>&1 | tail -3
!python -m bench.stencil_bw 2>&1 | tee n14_bw_log.txt
```

If you would rather run it on the A100, the `peak_GBs_nominal` and `pct_of_peak` columns will be
wrong: tell me the A100's peak bandwidth from its datasheet and I will make the peak a parameter
(not a number I fill in from memory).

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
