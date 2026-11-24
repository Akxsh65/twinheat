"""E4: GPU certify-on-demand vs the sequential CPU loop (H5; CLAUDE.md §8.1 step 5, §11.4, §11.5).

    python -m bench.pipeline_bench --data DATASET_V1 --mu MU_DIR --n-boards 256 \
        --batches 1 8 16 32 64 128 --seq-boards 32 --tau-v 0.1 --check-every 10 \
        --kappa-accel cg --kappa-iters 60 --dtype float64 --workers 32 [--device gpu]

Boards: the first --n-boards test boards with their FNO-ensemble μ (MU_DIR from
experiments/e13_certificate.py predict; same dataset hash). Every board is a flagged board
certified to τ_v from μ (Lemma 2(b), §8.1 step 5).

Stages, each timed on its own:
  prep (CPU pool)   rasterise + assemble + stencil coefficients    same for both pipelines
  kappa (CPU pool)  κ̂ by certify.kappa (z by --kappa-accel)        same for both pipelines
  oracle (CPU pool) θ to 1e-10 for the B7 check only; not part of either pipeline
  sequential        certify.stopping.pcg_certified, one board at a time on one core, FP64 CSR,
                    the same preconditioner: designs/s over the first --seq-boards boards
  batched (GPU)     certify.pcg_batched.batched_pcg_certified at each batch size, iterations in
                    --dtype, FP64 certification: upload (coefficients, b, μ to the device), pcg,
                    download (x and results back), each synchronised and timed
Every certified board is checked against the oracle: bound < max|θ − x| raises (B7).
H5 passes when batched designs/s (upload + pcg + download) is >= 10× the sequential rate at
some batch >= 64 (§11.4); otherwise the crossover batch and the time breakdown are the result.
Outputs: bench/results/pipeline_bench.csv (one row per batch size and the sequential row),
pipeline_bench.json (H5 verdict, settings), bench/figs/pipeline_bench.png (§4 footer).
--device cpu runs the batched path with NumPy (CI smoke tests only; not an E4 number).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from certify.pcg_batched import batched_pcg_certified
from certify.stopping import pcg_certified
from solvers.pcg import line_z_block_jacobi

ROOT = Path(__file__).resolve().parents[1]
H5_SPEEDUP, H5_MIN_BATCH = 10.0, 64  # §11.4
ORACLE_TOL = 1e-10  # §6.4
FIELDS = ("Ge", "Gw", "Gn", "Gs", "gup", "gdn", "diag")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class BenchError(RuntimeError):
    pass


class B7Violation(AssertionError):
    pass


# ---------------------------------------------------------------------------------- boards

def _prep(args: tuple) -> dict[str, Any]:
    """One board: coefficients, κ̂, S, oracle θ, μ, with the CPU time of each stage."""
    i, rec, mu_path, kappa_accel, kappa_iters = args
    from certify.kappa import kappa_hat, multigrid
    from certify.peak_interval import row_abs_sum_bound
    from data.raster import rasterise
    from solvers.amg_baseline import amg_solve
    from solvers.fvm import assemble_board, coefficients
    from twinheat import spec as specmod

    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    mu = np.array(np.load(mu_path, mmap_mode="r")[i], dtype=np.float64)
    t0 = time.perf_counter()
    r = rasterise(rec, spec, mu.shape[-1])
    A, _ = assemble_board(spec, r, rec["dirichlet_edges"])
    c = coefficients(spec, r, rec["dirichlet_edges"])
    t1 = time.perf_counter()
    k = kappa_hat(A, kappa_iters, ml=multigrid(A), accel=kappa_accel)
    t2 = time.perf_counter()
    theta = amg_solve(A, c["rhs"].ravel(), ORACLE_TOL, strength_theta=0.1).theta
    return {"i": i, "id": rec["id"], "record": rec, "coeffs": c, "mu": mu,
            "theta": theta.reshape(mu.shape), "kappa": k, "S": row_abs_sum_bound(c),
            "prep_s": t1 - t0, "kappa_s": t2 - t1}


def _csr(board: dict[str, Any]):
    from data.raster import rasterise
    from solvers.fvm import assemble_board
    from twinheat import spec as specmod
    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    rec = board["record"]
    return assemble_board(spec, rasterise(rec, spec, board["mu"].shape[-1]),
                          rec["dirichlet_edges"])[0].tocsr()


# ------------------------------------------------------------------------------- pipelines

def sequential(boards: list[dict[str, Any]], tau_v: float) -> dict[str, Any]:
    """certify.stopping.pcg_certified one board at a time (FP64 CSR, line-z): the CPU loop."""
    secs, iters = [], []
    for bd in boards:
        A = _csr(bd)
        shape = bd["mu"].shape
        M = line_z_block_jacobi(bd["coeffs"]["diag"], bd["coeffs"]["gup"])
        t0 = time.perf_counter()
        x, it, c = pcg_certified(A, bd["coeffs"]["rhs"].ravel(), bd["mu"].ravel(),
                                 lambda r, M=M, shape=shape: M(r.reshape(shape)).ravel(),
                                 bd["kappa"], tau_v)
        secs.append(time.perf_counter() - t0)
        iters.append(it)
        _b7(bd, x.reshape(shape), c, "sequential")
    return {"mode": "sequential_cpu", "batch": 1, "boards": len(boards),
            "pcg_s": float(sum(secs)), "total_s": float(sum(secs)),
            "designs_per_s": len(boards) / sum(secs), "mean_iters": float(np.mean(iters))}


def _b7(bd: dict[str, Any], x: np.ndarray, cert: float | None, where: str) -> None:
    err = float(np.abs(bd["theta"] - np.asarray(x, dtype=np.float64)).max())
    if cert is None or not cert >= err:
        raise B7Violation(f"{where}: board {bd['i']} ({bd['id']}): bound {cert} < true error "
                          f"{err} K (B7)")


def batched(boards: list[dict[str, Any]], batch: int, *, tau_v: float, check_every: int,
            maxit: int, dtype: str, device: str) -> dict[str, Any]:
    """Every board through batched_pcg_certified in chunks of `batch` (full chunks only)."""
    n = len(boards) - len(boards) % batch
    if n == 0:
        raise BenchError(f"batch {batch} > {len(boards)} boards")
    dt_ = np.dtype(dtype)
    t_up = t_pcg = t_down = 0.0
    iters, stops = [], []
    for s in range(0, n, batch):
        chunk = boards[s:s + batch]
        sync = _sync(device)
        t0 = time.perf_counter()
        Aop, Aop64, M, b, x0, to_host = _upload(chunk, dt_, device)
        sync()
        t1 = time.perf_counter()
        res = batched_pcg_certified(
            Aop, Aop64, b, x0, M, kappa=np.array([bd["kappa"] for bd in chunk]),
            S=np.array([bd["S"] for bd in chunk]), tau_v=np.full(len(chunk), tau_v),
            check_every=check_every, maxit=maxit)
        sync()
        t2 = time.perf_counter()
        xh = to_host(res.x)
        t3 = time.perf_counter()
        t_up, t_pcg, t_down = t_up + t1 - t0, t_pcg + t2 - t1, t_down + t3 - t2
        for j, bd in enumerate(chunk):
            _b7(bd, xh[j], float(res.cert[j]), f"batched B={batch}")
        iters += res.iters.tolist()
        stops += res.stop
    total = t_up + t_pcg + t_down
    return {"mode": f"batched_{device}", "batch": batch, "boards": n, "upload_s": t_up,
            "pcg_s": t_pcg, "download_s": t_down, "total_s": total,
            "designs_per_s": n / total, "designs_per_s_pcg_only": n / t_pcg,
            "mean_iters": float(np.mean(iters)),
            "maxit_stops": int(sum(st == "maxit" for st in stops))}


def _sync(device: str):
    if device == "gpu":
        import cupy as cp
        return cp.cuda.Device().synchronize
    return lambda: None


def _upload(chunk: list[dict[str, Any]], dt_: np.dtype, device: str):
    cs = [bd["coeffs"] for bd in chunk]
    b = np.stack([c["rhs"] for c in cs]).astype(dt_)
    x0 = np.stack([bd["mu"] for bd in chunk]).astype(dt_)
    if device == "gpu":
        import cupy as cp

        from solvers import cuda_kernels as ck
        from solvers.pcg import stencil_operator
        Aop = stencil_operator(ck.upload_coefficients(cs, dt_, to_device=cp.asarray))
        Aop64 = Aop if dt_ == np.float64 else stencil_operator(
            ck.upload_coefficients(cs, np.float64, to_device=cp.asarray))
        diag = cp.asarray(np.stack([c["diag"] for c in cs]).astype(dt_))
        gup = cp.asarray(np.stack([c["gup"] for c in cs]).astype(dt_))
        return Aop, Aop64, line_z_block_jacobi(diag, gup), cp.asarray(b), cp.asarray(x0), \
            cp.asnumpy
    mats = [_csr(bd) for bd in chunk]

    def make(cast):
        def op(x):
            return np.stack([(A @ xi.ravel().astype(np.float64)).reshape(xi.shape).astype(cast)
                             for A, xi in zip(mats, x, strict=True)])
        return op
    diag = np.stack([c["diag"] for c in cs]).astype(dt_)
    gup = np.stack([c["gup"] for c in cs]).astype(dt_)
    return make(dt_), make(np.float64), line_z_block_jacobi(diag, gup), b, x0, np.asarray


def verdict(seq: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    sp = {r["batch"]: r["designs_per_s"] / seq["designs_per_s"] for r in rows}
    big = {b: v for b, v in sp.items() if b >= H5_MIN_BATCH}
    passed = bool(big) and max(big.values()) >= H5_SPEEDUP
    cross = next((b for b in sorted(sp) if sp[b] >= 1.0), None)
    return {"criterion": f">= {H5_SPEEDUP}x sequential certified designs/s at batch >= "
                         f"{H5_MIN_BATCH} (upload + pcg + download)",
            "speedup_by_batch": sp, "passed": passed, "crossover_batch": cross}


# ------------------------------------------------------------------------------------ main

def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def _hardware(device: str) -> str:
    cpu = platform.processor() or "cpu"
    p = Path("/proc/cpuinfo")
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    gpu = ""
    if device == "gpu":
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        gpu = " · " + (name.decode() if isinstance(name, bytes) else str(name)).strip("\0 ")
    return f"{platform.node()} {cpu}{gpu}"


def write(out_csv: Path, out_fig: Path, seq: dict[str, Any], rows: list[dict[str, Any]],
          info: dict[str, Any]) -> dict[str, Any]:
    v = verdict(seq, rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in [seq, *rows] for k in r))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([seq, *rows])
    out_csv.with_suffix(".json").write_text(json.dumps({"H5": v, **info}, indent=2,
                                                       default=str) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bs = [r["batch"] for r in rows]
    ax.plot(bs, [r["designs_per_s"] for r in rows], "o-", label="batched GPU (with transfers)")
    ax.plot(bs, [r["designs_per_s_pcg_only"] for r in rows], "s--", label="batched GPU (PCG only)")
    ax.axhline(seq["designs_per_s"], color="k", lw=1, label="sequential CPU loop")
    ax.axhline(H5_SPEEDUP * seq["designs_per_s"], color="k", ls=":", lw=1, label="H5 bar (10×)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel(f"certified designs / s (τ_v = {info['tau_v']} K)")
    ax.legend(fontsize=7)
    fig.text(0.01, 0.01, info["footer"], fontsize=6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    return v


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--mu", type=Path, required=True)
    ap.add_argument("--n-boards", type=int, required=True)
    ap.add_argument("--batches", type=int, nargs="+", required=True)
    ap.add_argument("--seq-boards", type=int, required=True)
    ap.add_argument("--tau-v", type=float, required=True, help="K (CLAUDE.md §8.1: e.g. 0.1)")
    ap.add_argument("--check-every", type=int, required=True)
    ap.add_argument("--kappa-accel", choices=("none", "cg"), required=True)
    ap.add_argument("--kappa-iters", type=int, required=True)
    ap.add_argument("--dtype", choices=("float64", "float32"), required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--maxit", type=int, default=20000)
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "pipeline_bench.csv")
    ap.add_argument("--fig", type=Path, default=ROOT / "bench" / "figs" / "pipeline_bench.png")
    a = ap.parse_args(argv)
    from models.dataset import TwinHeatDataset
    from models.normalise import dataset_hash

    if _git("status", "--porcelain", "--untracked-files=no"):
        raise BenchError("the tree has uncommitted changes: commit before a reported run")
    commit = _git("rev-parse", "--short=7", "HEAD")
    meta = json.loads((a.mu / "mu_meta.json").read_text(encoding="utf-8"))
    dh = dataset_hash(a.data / "test")
    if meta["split"] != "test" or meta["dataset_hash"] != dh:
        raise BenchError(f"{a.mu}: μ is for {meta['split']} / {meta['dataset_hash'][:12]}, "
                         f"not test / {dh[:12]}")
    ds = TwinHeatDataset(a.data, "test", N=meta["N"], allow_eval=True)
    n = min(a.n_boards, meta["n"])
    accel = None if a.kappa_accel == "none" else a.kappa_accel
    jobs = [(i, ds.record(i), str(a.mu / "mu.npy"), accel, a.kappa_iters) for i in range(n)]
    old = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update(dict.fromkeys(THREAD_VARS, "1"))
    try:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
            boards = list(ex.map(_prep, jobs, chunksize=1))
        t_pool = time.perf_counter() - t0
        none = [bd["i"] for bd in boards if bd["kappa"] is None]
        if none:
            raise BenchError(f"boards {none[:10]} have no certificate (κ̂ None); E4 needs one")
        seq = sequential(boards[:a.seq_boards], a.tau_v)
    finally:
        for v, val in old.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    seq.update(prep_s_per_board=float(np.mean([bd["prep_s"] for bd in boards])),
               kappa_s_per_board=float(np.mean([bd["kappa_s"] for bd in boards])))
    if a.device == "gpu":  # warm-up: compile the kernels on one full batch, untimed
        batched(boards[:min(a.batches)], min(a.batches), tau_v=a.tau_v,
                check_every=a.check_every, maxit=a.maxit, dtype=a.dtype, device="gpu")
    rows = [batched(boards, bsz, tau_v=a.tau_v, check_every=a.check_every, maxit=a.maxit,
                    dtype=a.dtype, device=a.device) for bsz in a.batches]
    hw = _hardware(a.device)
    footer = (f"commit {commit} · dataset {dh[:12]} · test boards 0..{n - 1} · μ {meta['commit']}"
              f" · {a.dtype} iterations · κ̂ z {a.kappa_accel}×{a.kappa_iters} · {hw} · "
              f"{dt.date.today().isoformat()}")
    info = {"commit": commit, "dataset_hash": dh, "n_boards": n, "tau_v": a.tau_v,
            "check_every": a.check_every, "dtype": a.dtype, "kappa_accel": a.kappa_accel,
            "kappa_iters": a.kappa_iters, "maxit": a.maxit, "device": a.device,
            "prep_pool_s": t_pool, "workers": a.workers, "hardware": hw, "footer": footer,
            "violations": 0}
    v = write(a.out, a.fig, seq, rows, info)
    print(json.dumps({"H5": v, "sequential_designs_per_s": seq["designs_per_s"],
                      "batched": {r["batch"]: round(r["designs_per_s"], 3) for r in rows}},
                     indent=2, default=str))
    if math.isnan(seq["designs_per_s"]):
        raise BenchError("no sequential rate")


if __name__ == "__main__":
    main()
