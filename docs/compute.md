# Compute

## rvcse: college GPU server (`rvcecsegpuserver`), machine of record

Every GPU run and every oracle solve happens here (docs/decisions.md, "GPU runs on the college
A100"); it also holds dataset v1 (copy on Drive). Known from runs so far:

| Item | Value |
|---|---|
| GPUs | NVIDIA A100-PCIE-40GB, used as GPU 0 and GPU 2 (`CUDA_DEVICE_ORDER=PCI_BUS_ID`) |
| CPU | Intel Xeon Gold 6226R, 64 logical cores (N1.5 timings use 1 thread) |
| `scripts/check_gpu.py` output | all PASS, 2026-11-13, on GPU 2 (table below) |
| Peak memory bandwidth (for N1.4) | 1555.2 GB/s, from the card's memory clock and bus width; source and the datasheet check in `docs/refs/a100_pcie_40gb_bandwidth.txt` |

Measured with `scripts/check_gpu.py` (2026-11-13), run with `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=2 NUMBA_CUDA_USE_NVIDIA_BINDING=1`, so torch, CuPy and numba each saw one
A100 (GPU 2):

| Item | Value |
|---|---|
| nvidia-smi | 3 GPUs: A100-PCIE-40GB (GPU 0), T400 2 GiB (GPU 1, display), A100-PCIE-40GB (GPU 2) |
| Driver | 580.126.18, supports up to CUDA 13.0 |
| Python | 3.11.15, Linux 6.8.0-138-generic, glibc 2.35 |
| torch | 2.11.0+cu128 (CUDA 12.8 build), capability 8.0, 40,440 MiB visible |
| cupy | 14.0.1, CUDA runtime 12.9 |
| numba | 0.61.2 (`numba.cuda` through the NVIDIA binding, cuda-python 12.9) |
| numpy / scipy / pyamg | 2.1.3 / 1.16.3 / 5.3.0 |
| torch fp16 matmul 4096² | 0.82 ms, 167.6 TFLOP/s |
| cupy SpMV, 262,144 rows, 7 diagonals | 0.178 ms, about 138 GB/s, error vs SciPy 1.2e-16 |
| numba.cuda launch | PASS |

The numba device name comes back padded with NUL bytes (`"NVIDIA A100-PCIE-40GB\u0000..."`);
the same padding reaches `gpu_model` in `bench/solver_bench.py` (handover flag 8). The SpMV GB/s is
check_gpu's rough figure, not the N1.4 bandwidth measurement.

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
