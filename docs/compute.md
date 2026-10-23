# Compute

## Kaggle quotas

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
