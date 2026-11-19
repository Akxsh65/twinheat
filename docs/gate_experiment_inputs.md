# Inputs of the gate experiments (E7, E8, E9, E14, E15, E16)

For N (N3.2 oracle labels, N3.3 certified-audit traces) and M (trajectories, calibration set).
`experiments/run_gates.py` reads these files and refuses anything that does not match. Paths are
relative to the `--inputs` folder passed to every E-script.

## Folder layout

| File | Produced by | Used by |
|---|---|---|
| `trajectories/traj_<P>_a<level>_s<seed>.parquet` | M3.3, `run_trajectories all` | every E-script except E8 |
| `calibration.parquet` | M, `run_trajectories calib --n 1000` | all (calibration scores, G3 weights, G7 mean scale) |
| `labels.parquet` | **N3.2** | all |
| `audit_traces.parquet` | **N3.3** | E14 only |

## `labels.parquet` (N3.2)

One row per **unique** `x_hash`, covering every trajectory file and `calibration.parquet`.

| Column | Type | Meaning |
|---|---|---|
| `x_hash` | string, 64 hex | SHA-256 of the design vector as little-endian float64 (the trajectory's `x_hash`) |
| `oracle_peak` | float64, K | max over all layers of θ, solved to relative residual 1e-8 (§11.5) |
| `iters_full` | int64, nullable | PCG iterations from the ensemble-mean warm start to 1e-8 (the baseline of H7). Null when unknown |

Rebuild each board with `data/sampler.design_board(design_vector)` and rasterise at N = 256. A
duplicate `x_hash` with two different values is refused.

`experiments/oracle_label.py` writes this file (`label`, then `finalise`), with these extra
columns, which run_gates ignores: `source` (first input file the design appears in),
`argmax_layer`, `argmax_y`, `argmax_x`, `iters`, `solver` (`pyamg_gmres` | `pcg_linez`),
`device`, `batch`, `wall_s`, `relres`, `energy_balance`, `theta_sha256` (SHA-256 of θ as
little-endian float32 `[4, N, N]`), `worker`. `iters_full` is null there (it needs μ; N3.3).

## `audit_traces.parquet` (N3.3, E14 only)

One row per design that is audited in some run. Simplest is one row for every design in the
labels.

| Column | Type | Meaning |
|---|---|---|
| `x_hash` | string | as above |
| `check_iters` | list<int64> | PCG iteration at each certificate check (strictly increasing; `check_every`) |
| `T_lo`, `T_hi` | list<float64>, K | the certified interval for the true peak at that check (Lemma 2: `peak_interval`) |

The audit is warm-started from the ensemble mean μ and stopped at 1e-8 or at the
label-independent iteration cap (D14), whichever comes first. The replay picks the stopping check
itself: the first check whose score interval is decided relative to the gating q_t, or narrower
than ε_S. That way one trace serves every gate and every ε_S. **Every check must contain
`oracle_peak` (B7); the replay raises on the first that does not.** If the last checks get
narrower than the oracle's own accuracy, say so in the PR. The comparison is exact, with no
tolerance.

## `calibration.parquet` (M)

1,000 fresh draws from p₀ (decision 2026-11-18), with the columns `x_hash`, `design_vector`,
`peak_hat`, `r_hot`, `sd_hot`, `r_hot_raw`, `ood_score`. Their labels go in `labels.parquet` like
every other design: 1,000 extra oracle solves.
