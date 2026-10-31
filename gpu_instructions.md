# GPU runs to do (Kaggle T4 and the lab GPU machine)

Commands that need a real CUDA GPU, plus the N1.5 run on the lab machine. Everything else runs on
a laptop. Run the sections in order, paste each "Send back" item into the PR named there, then
tick the box. The oracle labels (~1e5 solves, N3.2) will run on the **lab GPU machine**, so
the N1.5 go/no-go is measured there (section A); Kaggle (sections 1–2) validates the kernels.

| # | Prompt | Where | Branch | Needed by | Status |
|---|---|---|---|---|---|
| 1 | N1.4 GPU tests + stencil bandwidth | Kaggle T4 | `n/n1.4-gpu-kernels` | gate G1, end of week 3 (Sat 7 Nov 2026); aim for Thu 5 Nov | [ ] |
| 2 | N1.5 GPU rows on a T4 (optional once A is done) | Kaggle T4 | `n/n1.4-gpu-kernels` | after section 1 passes | [ ] |
| A | N1.5 CPU **and** GPU rows on the oracle machine | lab GPU machine | `n/n1.4-gpu-kernels` | gate G1 (Sat 7 Nov 2026); decides D5 | [ ] |

New GPU prompts (N1.5 timing, N2.1, N4.x, M1.3, M2.x, ...) get a section here when their code is
written.

---

## 0 · Session setup (every time)

1. Kaggle → New Notebook → Settings: **Accelerator: GPU T4 x2**, **Internet: on**.
2. If the repo is private: Add-ons → Secrets → add `GITHUB_TOKEN` (a fine-grained token with
   read access to `Akxsh65/twinheat`), and tick it for this notebook.
3. First cell (clone the branch for the run you are doing; `BRANCH` changes per section):

```python
BRANCH = "n/n1.4-gpu-kernels"
import subprocess
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    url = f"https://{token}@github.com/Akxsh65/twinheat.git"
except Exception:
    url = "https://github.com/Akxsh65/twinheat.git"  # public repo
subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, "twinheat"], check=True)
%cd twinheat
!git log --oneline -1
```

4. Dependencies Kaggle does not have, and a check that the GPU stack works:

```
!pip install -q pydantic==2.14.1 jsonschema==4.26.0 pyyaml==6.0.3 pytest==9.1.1
!python scripts/check_gpu.py | tail -1
```

`check_gpu.py` must report all PASS (torch, CuPy, numba.cuda). If CuPy or numba fails, stop
and send the output: nothing below will be valid.

---

## 1 · N1.4 · CUDA stencil, PCG, batched PCG

About 15 minutes. Branch `n/n1.4-gpu-kernels` (must be pushed first).

```
!python -m pytest tests/test_gpu_solvers.py -v -rs 2>&1 | tee n14_gpu_log.txt
!tail -5 n14_gpu_log.txt
!python -m bench.stencil_bw 2>&1 | tee n14_bw_log.txt
```

**Expected:** every test passes; none of the `gpu` tests is skipped (a skip with "no CUDA
device" means numba cannot see the GPU, so the run does not count). The bench prints one line per
kernel, dtype and batch size and writes `bench/results/stencil_bw.csv`.

**Send back** (Output panel → download, or copy the text):
- `n14_gpu_log.txt` (whole file) → pasted into the N1.4 PR (DONE WHEN: "paste log into the PR")
- `bench/results/stencil_bw.csv` → committed on the branch
- `n14_bw_log.txt`

**If something fails:**
- `test_cupy_pcg_float32_iterations_with_fp64_check` says "stalled": FP32 cannot reach 1e-5 on
  this board; the tolerance is too tight, not a kernel bug. Send the log; the stalled relres is in
  the message.
- A stencil test fails only at the edges: boundary handling bug (CLAUDE.md §9.4). Send the log.
- `dot_warp_shuffle` fails: this kernel has only ever run here (the simulator has no warp
  shuffles). Send the log.
- Batched vs single differs by more than 1 iteration: a masking bug. Send the log.
- Anything with "CertificateViolation" or B3: stop and report; never re-run to get a pass.

Do not edit code on Kaggle. Fixes go through the branch, then re-run this section.

---

## 2 · N1.5 · GPU rows of the solver table (E3)

About 20–30 minutes (20 boards; each GPU solve is timed with 10 warm-ups + median of 20).
Run only after section 1 passes. Same session setup (section 0), same branch.

```
!pip install -q pyamg==5.3.0
!python -m bench.solver_bench --part gpu --out bench/results/solver_table_v1_gpu.csv 2>&1 | tee n15_gpu_log.txt
```

The first board also builds SciPy's direct reference solve (about 100 s per board on one core;
Kaggle's CPU may be slower), so the whole run is dominated by those 20 direct solves.
If the session is short on time: `--boards 5` and say so in the PR.

**Send back:** `bench/results/solver_table_v1_gpu.csv` and `n15_gpu_log.txt`.

**If something fails:** `SolverDisagreement` means a GPU solver's answer differs from SciPy's by
more than 1e-6: a real bug, send the log. Do not lower the tolerance.

---

## A · N1.5 · CPU and GPU rows on the lab GPU machine (the oracle machine)

The go/no-go (D5, CLAUDE.md §7.3) counts only if measured on the machine that will run the ~1e5
oracle labels: the lab GPU machine. Run both parts there. The CPU part gives the PyAMG time that
§7.3's 0.3 s rule is written for; the GPU part gives the GPU PCG time, which is what labelling
on that machine will actually cost.

0. Check the GPU stack first (section 1's checks apply here too):

```
git fetch origin && git checkout n/n1.4-gpu-kernels && git pull
python -m pip install -r requirements.txt     # or the conda route in environment.yml
python scripts/check_gpu.py | tail -1         # must be all PASS
python -m pytest tests/test_gpu_solvers.py -q -rs   # the N1.4 tests, on this GPU
```

1. CPU part: about 70 minutes for 20 boards (SciPy's direct solve is ~100 s per board). One
   process, one thread per pool (the script sets OMP/OpenBLAS/MKL to 1 unless you set them).
   Close other heavy jobs first: they distort the timing.

```
python -m bench.solver_bench --part cpu --out bench/results/solver_table_v1.csv 2>&1 | tee n15_cpu_log.txt
```

2. GPU part, into the same CSV (it repeats the 20 direct reference solves: about another 35
   minutes plus the GPU timing):

```
python -m bench.solver_bench --part gpu --out bench/results/solver_table_v1.csv 2>&1 | tee n15_gpu_log.txt
python -m bench.solver_bench --figure --out bench/results/solver_table_v1.csv
```

**Send back:** `bench/results/solver_table_v1.csv`, `bench/figs/solver_table_v1.png`,
`n15_cpu_log.txt`, `n15_gpu_log.txt`, the output of `check_gpu.py`, and the machine's name (it
becomes the oracle machine of record).
