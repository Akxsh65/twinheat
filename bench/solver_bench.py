"""Solver comparison and cost go/no-go (N1.5; CLAUDE.md §7.3, §11.5 E3, §17 D5).

    python -m bench.solver_bench --part cpu     # on the machine that will run oracle labels
    python -m bench.solver_bench --part gpu     # on a Kaggle T4
    python -m bench.solver_bench --figure       # table -> bench/figs/solver_table_v1.png

Boards: the 20 fixed boards are sampler seeds 0..19 (mode "id", their own Dirichlet edges),
rasterised at 256² × 4. Each part appends one row per (board, solver) to
bench/results/solver_table_v1.csv, with CPU model, logical core count, thread setting and GPU
model in every row (CLAUDE.md §4).

Solvers (tolerance 1e-8 on the true relative residual ‖b − Aθ‖₂/‖b‖₂ unless capped):
  CPU  direct            SciPy spsolve (SuperLU), the reference every other row is checked against
       pyamg_gmres       solvers/amg_baseline.py, PyAMG's default strength threshold
       pyamg_gmres_theta0.1   the same with strength θ = 0.1 (exploratory; not a frozen choice)
       pcg_cold, pcg_warm     solvers/pcg.py, Jacobi preconditioner (D10 open), x0 = 0 / warm start
       jacobi, rb_gs     stationary sweeps; at 256² they cannot reach 1e-8 (κ₂ ≈ 1e5), so they run
                         a fixed number of sweeps (--sweeps) and the row says converged = False
  GPU  pcg_cold_gpu, pcg_warm_gpu   solvers/pcg.py on CuPy, FP64 tiled stencil, Jacobi
       jacobi_gpu, rb_gs_gpu        fixed sweeps with the N1.4 kernels
Warm start (the prompt's "exact + 2 % smooth noise"): x0 = θ + 0.02 · max θ · s, s a seeded smooth
field sin(2π k_x x/N + φ_x) · sin(2π k_y y/N + φ_y), k ∈ {1, 2}, scaled to max|s| = 1, the same
on all four layers.

Timing. CPU: wall-clock (time.perf_counter), median of --cpu-reps runs (default 1: SciPy's
direct solve takes about 100 s per board on one core); thread pools are pinned to 1 thread
(OMP/OpenBLAS/MKL) unless the environment sets otherwise, as the oracle job runs one process per
core (N3.2). GPU: CUDA events, 10 warm-ups, median of 20 (the prompt); host↔device copies of the
coefficients, b and x0 (h2d_s) and of θ back (d2h_s) are separate columns, not in time_s.
Every converged row must agree with the direct solve to 1e-6 relative (‖·‖∞), else
SolverDisagreement is raised and nothing more is written.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # before numpy/scipy load their thread pools

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import functools  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from data.raster import rasterise  # noqa: E402
from data.sampler import sample_board  # noqa: E402
from solvers.amg_baseline import amg_solve  # noqa: E402
from solvers.fvm import assemble_board, coefficients, solve_direct  # noqa: E402
from solvers.pcg import jacobi as jacobi_precond  # noqa: E402
from solvers.pcg import pcg  # noqa: E402
from twinheat import spec as specmod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N_FULL = SPEC.geometry.N.value
N_BOARDS = 20  # the prompt: 20 fixed seeded boards
TOL = 1e-8  # the prompt: PyAMG-GMRES tolerance; used for every converging solver
AGREE_RTOL = 1e-6  # the prompt: all solvers agree to 1e-6 relative
WARM_NOISE = 0.02  # the prompt: exact + 2 % smooth noise
GO_NO_GO_S = 0.3  # CLAUDE.md §7.3: below 0.3 s -> 6 layers or 384²
STRENGTH_EXPLORATORY = 0.1  # pyamg_gmres_theta0.1 (see module docstring)
SWEEPS_DEFAULT = 1000  # stationary rows at 256²: a fixed budget, reported as not converged
STATIONARY_MAX = 1_000_000  # sweeps=None (small test boards): run to TOL or raise
GPU_WARMUPS, GPU_REPS = 10, 20  # the prompt
CPU_SOLVERS = ("direct", "pyamg_gmres", "pyamg_gmres_theta0.1", "pcg_cold", "pcg_warm",
               "jacobi", "rb_gs")
GPU_SOLVERS = ("pcg_cold_gpu", "pcg_warm_gpu", "jacobi_gpu", "rb_gs_gpu")
CSV_PATH = ROOT / "bench" / "results" / "solver_table_v1.csv"
FIG_PATH = ROOT / "bench" / "figs" / "solver_table_v1.png"


class SolverDisagreement(AssertionError):
    """A converged solver differs from the direct solve by more than AGREE_RTOL."""


# ----------------------------------------------------------------------------- boards

@dataclass(frozen=True)
class Case:
    seed: int
    board_id: str
    edges: tuple[str, ...]
    n: int
    A: sp.csr_matrix
    b: np.ndarray  # (4n²,) W
    coeffs: dict[str, np.ndarray]  # solvers/fvm.py coefficients(), for the GPU stencil
    theta_direct: np.ndarray  # (4n²,) K
    direct_seconds: float


@functools.cache
def fixed_board(seed: int, n: int = N_FULL) -> Case:
    """Sampler board `seed` (mode id) at n × n × 4, with its direct solution (timed)."""
    rec = sample_board(seed, "id")
    edges = tuple(rec["dirichlet_edges"])
    r = rasterise(rec, SPEC, n)
    A, b = assemble_board(SPEC, r, edges)
    t0 = time.perf_counter()
    theta = solve_direct(A, b)
    sec = time.perf_counter() - t0
    return Case(seed, rec["id"], edges, n, A.tocsr(), b, coefficients(SPEC, r, edges), theta,
                sec)


def warm_start(theta: np.ndarray, n: int, seed: int) -> np.ndarray:
    """θ + 2 % of max θ times a seeded smooth field (module docstring)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    kx, ky = rng.integers(1, 3, size=2)
    px, py = rng.uniform(0, 2 * np.pi, size=2)
    c = np.arange(n)
    s = np.sin(2 * np.pi * kx * c[None, :] / n + px) * np.sin(2 * np.pi * ky * c[:, None] / n + py)
    s = np.broadcast_to(s / np.abs(s).max(), (4, n, n)).ravel()
    return theta + WARM_NOISE * theta.max() * s


def check_agreement(case: Case, solver: str, theta: np.ndarray, converged: bool) -> float:
    """‖θ − θ_direct‖∞ / ‖θ_direct‖∞; raises SolverDisagreement if a converged row exceeds 1e-6."""
    err = float(np.abs(np.asarray(theta, np.float64) - case.theta_direct).max()
                / np.abs(case.theta_direct).max())
    if converged and not err <= AGREE_RTOL:
        raise SolverDisagreement(f"board {case.board_id} (seed {case.seed}) {solver}: "
                                 f"relative error {err:.3e} vs direct > {AGREE_RTOL}")
    return err


def _relres(case: Case, x: np.ndarray) -> float:
    return float(np.linalg.norm(case.b - case.A @ x) / np.linalg.norm(case.b))


# --------------------------------------------------------------------------- provenance

@functools.cache
def cpu_model() -> str:
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Windows":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    return platform.processor() or platform.machine()


@functools.cache
def gpu_model() -> str:
    try:
        from numba import config, cuda
        if config.ENABLE_CUDASIM or not cuda.is_available():
            return "none"
        name = cuda.get_current_device().name
        return name.decode() if isinstance(name, bytes) else str(name)
    except ImportError:
        return "none"


@functools.cache
def commit() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def _row(case: Case, solver: str, device: str, **fields: Any) -> dict[str, Any]:
    return {
        "date": dt.date.today().isoformat(), "commit": commit(), "board_id": case.board_id,
        "seed": case.seed, "dirichlet_edges": "".join(case.edges) or "-", "N": case.n,
        "layers": 4, "unknowns": case.b.size, "solver": solver, "device": device,
        "precision": "float64", "tol": TOL, "iters": None, "converged": None,
        "relres_true": None, "err_vs_direct": None, "time_s": None, "setup_s": None,
        "s_per_iter": None, "h2d_s": None, "d2h_s": None, "reps": None, "warmups": None,
        "cpu_model": cpu_model(), "cores": os.cpu_count(),
        "threads": os.environ.get("OMP_NUM_THREADS", ""), "gpu_model": gpu_model(),
        "python": platform.python_version(), **fields,
    }


# ---------------------------------------------------------------------------------- CPU

def _stationary(case: Case, kind: str, sweeps: int | None) -> tuple[np.ndarray, int]:
    A, b = case.A, case.b
    d = A.diagonal()
    x = np.zeros_like(b)
    n = case.n
    if kind == "rb_gs":
        R = (A - sp.diags(d)).tocsr()
        idx = np.indices((4, n, n)).sum(axis=0).ravel() % 2  # colour (l + y + x) mod 2
        masks = (idx == 0, idx == 1)
    cap = STATIONARY_MAX if sweeps is None else sweeps
    bnorm = float(np.linalg.norm(b))
    for k in range(1, cap + 1):
        if kind == "jacobi":
            x += (b - A @ x) / d
        else:
            for m in masks:
                x[m] = ((b - R @ x) / d)[m]
        if sweeps is None and np.linalg.norm(b - A @ x) <= TOL * bnorm:
            return x, k
    if sweeps is None:
        raise RuntimeError(f"board {case.board_id}: {kind} did not reach {TOL} in {cap} sweeps")
    return x, cap


def run_cpu(case: Case, sweeps: int | None = SWEEPS_DEFAULT, timing_reps: int = 1
            ) -> list[dict[str, Any]]:
    """One row per CPU solver for `case`. sweeps=None runs Jacobi / red-black GS to TOL."""
    rows = []
    rows.append(_row(case, "direct", "cpu", iters=0, converged=True,
                     relres_true=_relres(case, case.theta_direct), err_vs_direct=0.0,
                     time_s=case.direct_seconds, reps=1))
    x0_warm = warm_start(case.theta_direct, case.n, case.seed)
    jac = jacobi_precond(case.A.diagonal())

    def amg(theta_s):
        return lambda: amg_solve(case.A, case.b, tol=TOL, strength_theta=theta_s)

    runs = {
        "pyamg_gmres": amg(None),
        "pyamg_gmres_theta0.1": amg(STRENGTH_EXPLORATORY),
        "pcg_cold": lambda: pcg(lambda v: case.A @ v, case.b, np.zeros_like(case.b), jac, TOL),
        "pcg_warm": lambda: pcg(lambda v: case.A @ v, case.b, x0_warm, jac, TOL),
        "jacobi": lambda: _stationary(case, "jacobi", sweeps),
        "rb_gs": lambda: _stationary(case, "rb_gs", sweeps),
    }
    for name, fn in runs.items():
        times, setups = [], []
        for _ in range(timing_reps):
            t0 = time.perf_counter()
            out = fn()
            times.append(time.perf_counter() - t0)
            if name.startswith("pyamg"):
                setups.append(out.setup_seconds)
        if name.startswith("pyamg"):
            theta, iters = out.theta, out.iters
        else:
            theta, iters = out
        rel = _relres(case, theta)
        converged = rel <= TOL
        t = statistics.median(times)
        rows.append(_row(case, name, "cpu", iters=iters, converged=converged, relres_true=rel,
                         err_vs_direct=check_agreement(case, name, theta, converged),
                         time_s=t, setup_s=statistics.median(setups) if setups else None,
                         s_per_iter=t / iters if iters else None, reps=timing_reps))
    return rows


# ---------------------------------------------------------------------------------- GPU

def run_gpu(case: Case, sweeps: int = SWEEPS_DEFAULT, warmups: int = GPU_WARMUPS,
            reps: int = GPU_REPS) -> list[dict[str, Any]]:
    """One row per GPU solver (needs CuPy and a CUDA GPU)."""
    import cupy as cp

    from solvers import cuda_kernels as ck
    from solvers.pcg import stencil_operator

    n = case.n
    shape = (4, n, n)

    def timed(fn) -> float:  # seconds between two CUDA events around fn()
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        return cp.cuda.get_elapsed_time(start, stop) / 1e3

    x0_warm_h = warm_start(case.theta_direct, n, case.seed).reshape(shape)
    up: dict[str, Any] = {}

    def upload():
        up["c"] = ck.upload_coefficients(case.coeffs, np.float64, to_device=cp.asarray)
        up["b"] = cp.asarray(case.b.reshape(shape))
        up["x0"] = cp.asarray(x0_warm_h)
        up["diag"] = cp.asarray(case.coeffs["diag"])
    h2d = timed(upload)
    A = stencil_operator(up["c"], "tiled")
    M = jacobi_precond(up["diag"])
    inv_d = 1.0 / up["diag"]
    b = up["b"]

    def pcg_run(x0):
        return pcg(A, b, x0, M, TOL)

    def jacobi_run():
        x = cp.zeros_like(b)
        for _ in range(sweeps):
            x += (b - A(x)) * inv_d
        return x, sweeps

    def rb_run():
        x = cp.zeros_like(b)
        for _ in range(sweeps):
            ck.red_black_gs_sweep(up["c"], x, b)
        return x, sweeps

    runs = {
        "pcg_cold_gpu": lambda: pcg_run(cp.zeros_like(b)),
        "pcg_warm_gpu": lambda: pcg_run(up["x0"]),
        "jacobi_gpu": jacobi_run,
        "rb_gs_gpu": rb_run,
    }
    rows = []
    for name, fn in runs.items():
        for _ in range(warmups):
            fn()
        times = []
        for _ in range(reps):
            box = {}
            times.append(timed(lambda box=box, fn=fn: box.update(out=fn())))
        theta_d, iters = box["out"]
        hold: dict[str, np.ndarray] = {}
        d2h = timed(lambda theta_d=theta_d, hold=hold: hold.update(theta=cp.asnumpy(theta_d)))
        theta = hold["theta"].ravel()
        rel = _relres(case, theta)
        converged = rel <= TOL
        t = statistics.median(times)
        rows.append(_row(case, name, "gpu", iters=iters, converged=converged, relres_true=rel,
                         err_vs_direct=check_agreement(case, name, theta, converged), time_s=t,
                         s_per_iter=t / iters if iters else None, h2d_s=h2d, d2h_s=d2h,
                         reps=reps, warmups=warmups))
    return rows


# --------------------------------------------------------------------------- CSV, figure

def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else v))
                        for k, v in r.items()})


def figure(csv_path: Path = CSV_PATH, fig_path: Path = FIG_PATH) -> Path:
    """Median time per solver over the boards (log scale), each board a dot, the 0.3 s line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    keys = list(dict.fromkeys((r["solver"], r["device"], r["cpu_model"] if r["device"] == "cpu"
                               else r["gpu_model"]) for r in rows))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(keys) + 2.2))
    for k, (solver, device, hw) in enumerate(keys):
        sel = [r for r in rows if (r["solver"], r["device"]) == (solver, device)
               and hw in (r["cpu_model"], r["gpu_model"])]
        t = np.array([float(r["time_s"]) for r in sel])
        conv = all(r["converged"] == "True" for r in sel)
        med = float(np.median(t))
        ax.barh(k, med, color="#4C72B0" if conv else "#BBBBBB",
                hatch=None if conv else "//", edgecolor="black", linewidth=0.5)
        ax.plot(t, np.full(t.size, k), "k.", ms=3)
        note = f"{med:.3g} s" if conv else f"{med:.3g} s, {sel[0]['iters']} sweeps, not converged"
        ax.text(med * 1.15, k, note, va="center", fontsize=7)
    ax.axvline(GO_NO_GO_S, color="red", ls="--", lw=1)
    ax.text(GO_NO_GO_S, len(keys) - 0.4, " go/no-go 0.3 s (CLAUDE.md §7.3)", color="red",
            fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{s} [{d}]" for s, d, _ in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time per solve, s (median over boards; dots = boards)")
    n_boards = len({r["board_id"] for r in rows})
    ax.set_title(f"Solver comparison, 256² × 4, {n_boards} boards, true relres ≤ {TOL:g}")
    hw = sorted({f"CPU {r['cpu_model']} ({r['cores']} logical, {r['threads']} thread)"
                 for r in rows if r["device"] == "cpu"}
                | {f"GPU {r['gpu_model']}" for r in rows if r["device"] == "gpu"})
    footer = (f"commit {', '.join(sorted({r['commit'] for r in rows}))} · data: sampler seeds "
              f"{min(int(r['seed']) for r in rows)}–{max(int(r['seed']) for r in rows)} (mode id;"
              f" no dataset hash before v2-data) · {'; '.join(hw)} · "
              f"{', '.join(sorted({r['date'] for r in rows}))}")
    fig.text(0.01, 0.01, footer, fontsize=6, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("cpu", "gpu"))
    ap.add_argument("--figure", action="store_true", help="draw the figure from the CSV")
    ap.add_argument("--boards", type=int, default=N_BOARDS)
    ap.add_argument("--n", type=int, default=N_FULL)
    ap.add_argument("--sweeps", type=int, default=SWEEPS_DEFAULT)
    ap.add_argument("--cpu-reps", type=int, default=1)
    ap.add_argument("--warmups", type=int, default=GPU_WARMUPS)
    ap.add_argument("--reps", type=int, default=GPU_REPS)
    ap.add_argument("--out", type=Path, default=CSV_PATH)
    args = ap.parse_args()
    if args.part is None and not args.figure:
        ap.error("give --part cpu|gpu and/or --figure")
    if args.part:
        for seed in range(args.boards):
            case = fixed_board(seed, args.n)
            rows = (run_cpu(case, args.sweeps, args.cpu_reps) if args.part == "cpu"
                    else run_gpu(case, args.sweeps, args.warmups, args.reps))
            append_rows(args.out, rows)
            print(f"board {case.board_id}: " + ", ".join(
                f"{r['solver']} {r['time_s']:.3g}s{'' if r['converged'] else ' (capped)'}"
                for r in rows), flush=True)
    if args.figure:
        print(f"wrote {figure(args.out)}")


if __name__ == "__main__":
    main()
