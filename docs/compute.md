# Compute

## rvcse: college GPU server (`rvcecsegpuserver`), machine of record

Every GPU run and every oracle solve happens here (docs/decisions.md, "GPU runs on the college
A100"); it also holds dataset v1 (copy on Drive). Known from runs so far:

| Item | Value |
|---|---|
| GPUs | NVIDIA A100-PCIE-40GB, used as GPU 0 and GPU 2 (`CUDA_DEVICE_ORDER=PCI_BUS_ID`) |
| CPU | Intel Xeon Gold 6226R, 64 logical cores (N1.5 timings use 1 thread) |
| `scripts/check_gpu.py` output | to fill in (gpu_instructions.md, item 11) |
| Peak memory bandwidth (for N1.4) | from the A100 datasheet, kept in `docs/refs/`; not written here from memory |

## Kaggle (fallback only; measured 2026-10-23)

Measured with `scripts/check_gpu.py` (2026-10-23), session "GPU T4 x2":

| Item | Value |
|---|---|
| GPUs | 2 × Tesla T4, sm_75, 15,360 MiB each (about 14.8 GiB usable) |
| Driver | 580.178.04, supports up to CUDA 13.0 |
| Python | 3.13.15 |
| torch | 2.11.0+cu128 (CUDA 12.8 build) |
| cupy | 14.0.1, CUDA runtime 12.9 |
| numba | 0.61.2 |
| numpy / scipy | 2.1.3 / 1.16.3 |
| pyamg | not pre-installed |
| torch fp16 matmul 4096² | 6.56 ms, 20.95 TFLOP/s |
| cupy SpMV, 262,144 rows, 7 diagonals | 0.226 ms, about 109 GB/s, error vs SciPy 1.3e-16 |
| numba.cuda launch | PASS |

Weekly GPU-hour quota: to be filled in from the Kaggle account page.

## Colab

## Laptop GPU

## Lab CPU

## Budget per experiment
