# TwinHeat

**Certified neural surrogates for interactive circuit-board thermal design.**

A 5-member FNO ensemble predicts the steady temperature of a 4-layer, 100 × 100 mm PCB (256 × 256 × 4 unknowns) in milliseconds. An audited adaptive-conformal gate decides, for each proposed design, whether to accept the prediction or send it to the exact solver. Flagged and audited designs are verified in one batched, warm-started GPU PCG solve, and each board stops as soon as a deterministic M-matrix certificate settles its decision. A WebXR viewer shows the field converging inside a guaranteed band, and fabricated boards under a thermal camera check the solver against reality.

Final-year project at RVCE, serving CGVR, PAGP and Deep Learning.

> **Status:** repository skeleton, CI and pinned environment are in place; the packages below are still empty. The full specification, invariants and open decisions are in [CLAUDE.md](CLAUDE.md); task-by-task prompts are in [TwinHeat_Implementation_Plan.md](TwinHeat_Implementation_Plan.md).

## Quick start

```bash
python -m pip install -r requirements.txt   # or: conda env create -f environment.yml
python scripts/verify_env.py                # checks imports, pins and (if present) the GPU stack
make test                                   # fast tests:  pytest -m "not slow and not gpu"
make lint                                   # ruff
```

| Command | Runs |
|---|---|
| `make test` | fast tests (what CI runs on every push) |
| `make test-slow` | slow tests (CI runs these nightly) |
| `make lint` | `ruff check .` |
| `make gpu-test` | prints how to run the GPU tests on Kaggle (paste the log into the PR) |

## Layout

| Package | Owner | Purpose |
|---|---|---|
| `solvers/` | N | Finite-volume assembly, M-matrix test, CUDA kernels, PCG, AMG baseline |
| `certify/` | N | Lemma 2 certificate: κ̂, FP64 rounding margin, batched certified PCG, peak intervals |
| `pipeline/` | N | Certify-on-demand step (PyTorch reference and CUDA) |
| `bench/` | N | Solver, pipeline and roofline benchmarks |
| `data/` | M (N: `generate.py`) | Board sampler, rasterisation, dataset generation and verification |
| `models/` | M | FNO / U-Net, training, ensemble, OOD score |
| `gate/` | M | Gates G0–G7, audit coin, proposers P1–P4, groups |
| `experiments/` | M | Experiment scripts and frozen configs |
| `release/` | M | TwinHeat-Traj v1 benchmark release |
| `server/` | M + G | FastAPI / WebSocket server |
| `study/` | G | User-study protocol and analysis |
| `hardware/` | G | Board generation, rig, thermal-camera analysis |
| `twinheat/` | — | Spec loading and validation |
| `docs/` | — | Decisions log, compute quotas, equipment, novelty search, future work |
| `scripts/` | — | Environment and GPU checks |

The planned files inside each package are listed in CLAUDE.md §3.

## Working rules (short version)

- `main` is always green. Work on `n/<task>`, `m/<task>` or `g/<task>` branches; PRs need CI plus one reviewer.
- Invariants B1–B8 (CLAUDE.md §5) are never weakened to make a test pass. B7: every certificate must be a true upper bound.
- Data and checkpoints never go in git; hashes live in `data/manifest.json` and `models/registry.json`.
- Changes to CLAUDE.md go through a PR with a line in [docs/decisions.md](docs/decisions.md).
