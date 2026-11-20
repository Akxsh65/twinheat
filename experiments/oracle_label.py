"""Oracle labels: the exact peak rise of every proposed design (N3.2; CLAUDE.md §8.1, §11.5, B3).

    python -m experiments.oracle_label label    --inputs DIR [--cpu-workers K] [--gpu]
    python -m experiments.oracle_label finalise --inputs DIR
    python -m experiments.oracle_label check    --inputs DIR [--gpu | --pcg-cpu-workers K]
    python -m experiments.oracle_label report   --inputs DIR [--out experiments/oracle_report.md]
    python -m experiments.oracle_label all      --inputs DIR [--cpu-workers K] [--gpu]

DIR follows docs/gate_experiment_inputs.md: DIR/trajectories/traj_*.parquet with their
manifest.json (M3.3) and DIR/calibration.parquet (the 1,000 p₀ designs). Every trajectory file
must match its manifest SHA-256, and every row's x_hash must be the SHA-256 of its design_vector
(little-endian float64), or nothing runs.

label: dedupes designs by x_hash across all files (a design keeps the first file it appears in,
trajectories in name order, then calibration), rebuilds each board with
data/sampler.design_board(x) (deterministic in x, so x_hash identifies the board record) and
solves it **to a true relative residual ‖b − Aθ‖₂/‖b‖₂ ≤ 1e-8**, cold-started. No certified early
stopping and no envelope rescale: the label is the exact discrete solution of the very board the
surrogate predicted. Workers pull from one queue:
  * K CPU processes, PyAMG Ruge–Stüben + GMRES (solvers/amg_baseline.py), one thread each
    (OMP/OpenBLAS/MKL_NUM_THREADS=1, set before they start);
  * optionally one GPU process (--gpu): batched FP64 PCG, line-z block-Jacobi preconditioned
    (solvers/pcg.py, as data/generate.py), up to 64 boards per launch.
Each label is checked: finite θ, relres ≤ 1e-8, energy-balance error ≤ 0.1 % (B2). Results are
appended to DIR/oracle/parts/<source>.<seq>.parquet every --flush-every boards (atomic rename), so a
kill loses at most the unflushed boards; rerunning skips every labelled x_hash. A lock file stops
two labellers sharing DIR.

finalise: refuses gaps (a unique design with no label), duplicates and labels for designs not in
the inputs, then writes DIR/labels.parquet (one row per unique x_hash, sorted).

check: re-solves a seeded random 0.5 % of the labels with the OTHER solver (AMG labels by PCG and
the reverse) and requires |Δpeak| < 1e-4 K for every one; writes DIR/oracle/cross_check.{parquet,
json} and raises on any failure.

report: writes experiments/oracle_report.md: counts, solver statistics, throughput (boards/s, CPU
core-hours, GPU-hours, from DIR/oracle/sessions.jsonl), the cross-check, and a footer.

labels.parquet columns: x_hash, source, oracle_peak (K, max over layers of θ), argmax_layer,
argmax_y, argmax_x ([layer, y, x] index of the peak), iters, solver (pyamg_gmres | pcg_linez),
device (cpu | gpu), batch (boards in the launch), wall_s (rasterise + assemble + solve; a PCG
launch's wall time divided by its batch), relres (true), energy_balance, theta_sha256 (SHA-256 of
θ as little-endian float32 [4, N, N], C order), iters_full (null: PCG iterations from the ensemble
mean need μ and come with N3.3), worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data import sampler
from data.raster import rasterise
from solvers.amg_baseline import amg_solve
from solvers.fvm import assemble_board, board_energy_balance, coefficients
from solvers.pcg import batched_pcg, line_z_block_jacobi
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
_S1 = specmod.load(ROOT / "config" / "spec_v1.yaml")
DIM = int(SPEC.sampler.design_space.dimension.value)  # 36 (§7.4)

ORACLE_TOL = 1e-8  # CLAUDE.md §8.1 step 6 and §11.5: oracle solves always to 1e-8
ENERGY_TOL = _S1.discretisation.energy_balance_tol.value  # 1e-3 per board, B2 / §6.4
CROSS_TOL_K = 1e-4  # N3.2 prompt: |Δpeak| between the two solvers
CROSS_FRACTION = 0.005  # N3.2 prompt: re-solve 0.5 % with the other solver
GPU_BATCH = 64  # N3.2 prompt
# Speed only (the 1e-8 true-residual stop fixes the accuracy); proposed in docs/decisions.md
# "N3.2 oracle settings", N to confirm. 1.6 s vs 4.6 s median per board in solver_table_v1.csv.
DEFAULT_STRENGTH_THETA: float | None = 0.1
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")
DEFAULT_REPORT = ROOT / "experiments" / "oracle_report.md"
CALIBRATION = "calibration"

SCHEMA = pa.schema([
    ("x_hash", pa.string()), ("source", pa.string()), ("oracle_peak", pa.float64()),
    ("argmax_layer", pa.int64()), ("argmax_y", pa.int64()), ("argmax_x", pa.int64()),
    ("iters", pa.int64()), ("solver", pa.string()), ("device", pa.string()),
    ("batch", pa.int64()), ("wall_s", pa.float64()), ("relres", pa.float64()),
    ("energy_balance", pa.float64()), ("theta_sha256", pa.string()),
    ("iters_full", pa.int64()), ("worker", pa.string()),
])


class OracleError(RuntimeError):
    """Inputs, labels or a solve failed a check; the message names what and where."""


# ---------------------------------------------------------------------------------- solving

def x_hash(x: np.ndarray) -> str:
    """SHA-256 of x as little-endian float64 (the §10.4 x_hash, as gate/proposers.x_hash)."""
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()


def theta_sha256(theta: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(theta, dtype="<f4").tobytes()).hexdigest()


def _board(x: np.ndarray, N: int):
    rec = sampler.design_board(np.asarray(x, dtype=np.float64))
    return rec, rasterise(rec, SPEC, N), rec["dirichlet_edges"]


def _summary(theta: np.ndarray, raster, edges, relres: float, what: str) -> dict[str, Any]:
    if not np.all(np.isfinite(theta)):
        raise OracleError(f"{what}: θ has non-finite entries")
    if not relres <= ORACLE_TOL:
        raise OracleError(f"{what}: true relative residual {relres:.3e} > {ORACLE_TOL}")
    eb = float(board_energy_balance(SPEC, raster, edges, theta))
    if not eb <= ENERGY_TOL:
        raise OracleError(f"{what}: energy balance error {eb:.3e} > {ENERGY_TOL} (B2)")
    k = int(np.argmax(theta))
    layer, y, x = (int(v) for v in np.unravel_index(k, theta.shape))
    return {"oracle_peak": float(theta.flat[k]), "argmax_layer": layer, "argmax_y": y,
            "argmax_x": x, "relres": float(relres), "energy_balance": eb,
            "theta_sha256": theta_sha256(theta), "iters_full": None}


def solve_amg(x: np.ndarray, *, N: int, strength_theta: float | None) -> dict[str, Any]:
    """Label one design with PyAMG-GMRES on this process's CPU core."""
    t0 = time.perf_counter()
    _, raster, edges = _board(x, N)
    A, b = assemble_board(SPEC, raster, edges)
    res = amg_solve(A, b, ORACLE_TOL, accel="gmres", strength_theta=strength_theta)
    wall = time.perf_counter() - t0
    row = _summary(res.theta.reshape(4, N, N), raster, edges, res.relres,
                   f"design {x_hash(x)[:12]}… (pyamg_gmres)")
    row.update(iters=int(res.iters), solver="pyamg_gmres", device="cpu", batch=1, wall_s=wall)
    return row


def _norms(xp, a) -> np.ndarray:
    v = xp.sqrt(xp.sum((a * a).reshape(a.shape[0], -1), axis=1))
    return v.get() if hasattr(v, "get") else np.asarray(v)


def solve_pcg(xs: list[np.ndarray], *, N: int, device: str) -> list[dict[str, Any]]:
    """Label a batch with FP64 batched PCG (line-z block Jacobi), on "gpu" (CuPy) or "cpu"."""
    if device not in ("gpu", "cpu"):
        raise OracleError(f"device must be 'gpu' or 'cpu', got {device!r}")
    if not xs:
        raise OracleError("solve_pcg: empty batch")
    t0 = time.perf_counter()
    boards = [_board(x, N) for x in xs]
    cs = [coefficients(SPEC, r, e) for _, r, e in boards]
    b = np.stack([c["rhs"] for c in cs])
    diag = np.stack([c["diag"] for c in cs])
    gup = np.stack([c["gup"] for c in cs])
    if device == "gpu":
        import cupy as cp

        from solvers import cuda_kernels as ck
        from solvers.pcg import stencil_operator
        op = stencil_operator(ck.upload_coefficients(cs, np.float64, to_device=cp.asarray))
        bd = cp.asarray(b)
        xd, iters = batched_pcg(op, bd, cp.zeros_like(bd),
                                line_z_block_jacobi(cp.asarray(diag), cp.asarray(gup)),
                                ORACLE_TOL)
        rel = _norms(cp, bd - op(xd)) / _norms(cp, bd)
        theta = cp.asnumpy(xd)
    else:
        mats = [assemble_board(SPEC, r, e)[0] for _, r, e in boards]

        def op(v: np.ndarray) -> np.ndarray:
            return np.stack([(A @ vi.ravel()).reshape(vi.shape)
                             for A, vi in zip(mats, v, strict=True)])
        theta, iters = batched_pcg(op, b, np.zeros_like(b), line_z_block_jacobi(diag, gup),
                                   ORACLE_TOL)
        rel = _norms(np, b - op(theta)) / _norms(np, b)
    wall = (time.perf_counter() - t0) / len(xs)
    rows = []
    for k, (_, raster, edges) in enumerate(boards):
        row = _summary(theta[k], raster, edges, float(rel[k]),
                       f"design {x_hash(xs[k])[:12]}… (pcg_linez, {device})")
        row.update(iters=int(iters[k]), solver="pcg_linez", device=device, batch=len(xs),
                   wall_s=wall)
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------- inputs

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_designs(root: Path) -> tuple[dict[str, tuple[str, np.ndarray]], list[dict]]:
    """{x_hash: (source, x)} over every input file (first appearance wins), and per-file info."""
    root = Path(root)
    td = root / "trajectories"
    files = sorted(td.glob("traj_*.parquet"))
    cal = root / "calibration.parquet"
    if not files:
        raise OracleError(f"no {td}/traj_*.parquet (docs/gate_experiment_inputs.md)")
    if not cal.exists():
        raise OracleError(f"{cal} is missing: label the p₀ calibration designs with the "
                          "trajectories (decision 2026-11-18)")
    man_path = td / "manifest.json"
    if not man_path.exists():
        raise OracleError(f"{man_path} is missing (written by run_trajectories)")
    listed = {e["file"]: e for e in json.loads(man_path.read_text(encoding="utf-8"))["runs"]}
    names = {f.name for f in files}
    if names - set(listed):
        raise OracleError(f"{sorted(names - set(listed))} are not listed in {man_path}")
    if set(listed) - names:
        raise OracleError(f"{man_path} lists {sorted(set(listed) - names)}, not found in {td}")
    info, designs = [], {}
    for name, path in [(f.stem, f) for f in files] + [(CALIBRATION, cal)]:
        sha = _sha(path)
        if name != CALIBRATION and sha != listed[path.name]["sha256"]:
            raise OracleError(f"{path}: sha256 {sha[:12]}… does not match {man_path} "
                              f"({listed[path.name]['sha256'][:12]}…)")
        try:
            t = pq.read_table(path, columns=["x_hash", "design_vector"]).to_pydict()
        except (KeyError, pa.ArrowInvalid) as e:
            raise OracleError(f"{path}: needs columns x_hash, design_vector ({e})") from e
        new = 0
        for k, (h, v) in enumerate(zip(t["x_hash"], t["design_vector"], strict=True)):
            if v is None or len(v) != DIM:
                raise OracleError(f"{path}: row {k} has no {DIM}-float design_vector")
            x = np.asarray(v, dtype=np.float64)
            if x_hash(x) != h:
                raise OracleError(f"{path}: row {k}: x_hash {str(h)[:12]}… is not the SHA-256 "
                                  "of its design_vector")
            if h not in designs:
                designs[h] = (name, x)
                new += 1
        info.append({"source": name, "file": str(path.relative_to(root)),
                     "rows": len(t["x_hash"]), "new_unique": new, "sha256": sha})
    return designs, info


# ------------------------------------------------------------------------------------ parts

def _odir(root: Path) -> Path:
    return Path(root) / "oracle"


def _meta(N: int, strength_theta: float | None) -> dict[str, Any]:
    return {"schema": "twinheat-oracle-labels/1", "N": int(N), "tol": ORACLE_TOL,
            "strength_theta": strength_theta}


def _read_meta(schema: pa.Schema, path: Path) -> dict[str, Any]:
    if schema.remove_metadata() != SCHEMA or not schema.metadata or \
            b"twinheat" not in schema.metadata:
        raise OracleError(f"{path}: not an oracle-label table (schema or metadata differ)")
    return json.loads(schema.metadata[b"twinheat"])


def _same_settings(a: dict, b: dict, what: str) -> None:
    for k in ("N", "tol", "strength_theta"):
        if a[k] != b[k]:
            raise OracleError(f"{what}: {k} = {b[k]!r}, but other labels used {a[k]!r}; "
                              "one label set has one solver setting")


def load_parts(root: Path) -> tuple[dict[str, dict], dict | None]:
    """Every row of DIR/oracle/parts/*.parquet ({x_hash: row}) and their shared settings."""
    pdir = _odir(root) / "parts"
    rows: dict[str, dict] = {}
    where: dict[str, str] = {}
    meta = None
    for p in sorted(pdir.glob("*.parquet")):  # *.parquet.tmp (a killed write) is skipped
        t = pq.read_table(p)
        m = _read_meta(t.schema, p)
        if meta is None:
            meta = {k: m[k] for k in ("schema", "N", "tol", "strength_theta")}
        _same_settings(meta, m, str(p))
        for r in t.to_pylist():
            h = r["x_hash"]
            if h in rows:
                raise OracleError(f"x_hash {h[:12]}… is labelled twice: {where[h]} and {p.name}")
            rows[h], where[h] = r, p.name
    return rows, meta


class _PartWriter:
    def __init__(self, root: Path, meta: dict[str, Any]) -> None:
        self.dir = _odir(root) / "parts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = json.dumps(meta, sort_keys=True)
        self.seq: dict[str, int] = {}
        for p in self.dir.glob("*.parquet"):
            src, s = p.stem.rsplit(".", 1)
            self.seq[src] = max(self.seq.get(src, -1), int(s))

    def write(self, source: str, rows: list[dict]) -> None:
        s = self.seq.get(source, -1) + 1
        self.seq[source] = s
        path = self.dir / f"{source}.{s:05d}.parquet"
        tmp = path.with_name(path.name + ".tmp")
        t = pa.Table.from_pylist(rows, schema=SCHEMA).replace_schema_metadata(
            {"twinheat": self.meta})
        pq.write_table(t, tmp)
        os.replace(tmp, path)


class RunLock:
    """An OS lock on DIR/oracle/.lock, released by the OS if the holder is killed."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.f = None

    def __enter__(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+b")  # noqa: SIM115 (held until __exit__)
        try:
            if sys.platform == "win32":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            f.close()
            raise OracleError(f"another labeller holds {self.path}; wait for it or stop it "
                              f"({e})") from e
        self.f = f
        return self

    def __exit__(self, *exc) -> None:
        f, self.f = self.f, None
        if f is None:
            return
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


# ------------------------------------------------------------------------------------- pool

_EMPTY = object()


def _worker(kind: str, task_q, result_q, N: int, strength_theta: float | None, device: str,
            batch: int) -> None:
    """kind "amg": one board per task. kind "pcg": up to `batch` boards per launch. A task is
    (x_hash, source, x); None ends the worker. Exits if the parent dies (a killed labeller)."""
    parent = mp.parent_process()
    name = f"{kind}-{device}-{os.getpid()}"

    def get(block: bool):
        while True:
            try:
                return task_q.get(timeout=1.0) if block else task_q.get_nowait()
            except queue.Empty:
                if not block:
                    return _EMPTY
                if parent is not None and not parent.is_alive():
                    result_q.cancel_join_thread()  # nobody reads: do not block on exit
                    return None

    while True:
        first = get(True)
        if first is None:
            return
        tasks, stop = [first], False
        while kind == "pcg" and len(tasks) < batch:
            u = get(False)
            if u is _EMPTY:
                break
            if u is None:
                stop = True
                break
            tasks.append(u)
        try:
            xs = [np.asarray(t[2], dtype=np.float64) for t in tasks]
            rows = ([solve_amg(xs[0], N=N, strength_theta=strength_theta)] if kind == "amg"
                    else solve_pcg(xs, N=N, device=device))
        except Exception as e:  # forwarded: the parent aborts the run and re-raises it
            result_q.put(("error", [t[0] for t in tasks],
                          f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
            return
        for t, r in zip(tasks, rows, strict=True):
            r.update(x_hash=t[0], source=t[1], worker=name)
        result_q.put(("ok", rows))
        if stop:
            return


def _run_pool(tasks: list[tuple], *, amg_workers: int, pcg_workers: int, pcg_device: str,
              batch: int, N: int, strength_theta: float | None, shared: bool,
              on_rows: Callable[[list[dict]], None]) -> None:
    """Solve every task. tasks: (x_hash, source, x, want) with want "amg" | "pcg" (ignored when
    shared: every worker pulls from one queue)."""
    if amg_workers < 0 or pcg_workers < 0 or amg_workers + pcg_workers == 0:
        raise OracleError(f"need at least one worker (cpu {amg_workers}, pcg {pcg_workers})")
    if pcg_device == "gpu" and pcg_workers > 1:
        raise OracleError("one GPU worker per process tree; pick the GPU with "
                          "CUDA_VISIBLE_DEVICES")
    if not (isinstance(batch, int) and batch >= 1):
        raise OracleError(f"batch must be an int >= 1, got {batch!r}")
    ctx = mp.get_context("spawn")
    q_amg = ctx.Queue()
    q_pcg = q_amg if shared else ctx.Queue()
    if not shared:
        for kind, n in (("amg", amg_workers), ("pcg", pcg_workers)):
            if n == 0 and any(t[3] == kind for t in tasks):
                raise OracleError(f"tasks need the {kind} solver but no {kind} worker is set")
    rq = ctx.Queue()
    old = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update(dict.fromkeys(THREAD_VARS, "1"))  # inherited by the spawned workers
    procs = []
    try:
        for i in range(amg_workers):
            procs.append(ctx.Process(target=_worker, name=f"amg{i}", daemon=True,
                                     args=("amg", q_amg, rq, N, strength_theta, "cpu", 1)))
        for i in range(pcg_workers):
            procs.append(ctx.Process(target=_worker, name=f"pcg{i}", daemon=True,
                                     args=("pcg", q_pcg, rq, N, strength_theta, pcg_device,
                                           batch)))
        for p in procs:
            p.start()
    finally:
        for v, val in old.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    ok = False
    try:
        for t in tasks:
            (q_pcg if (not shared and t[3] == "pcg") else q_amg).put(
                (t[0], t[1], [float(v) for v in t[2]]))
        for _ in range(amg_workers):
            q_amg.put(None)
        for _ in range(pcg_workers):
            q_pcg.put(None)
        got = 0
        while got < len(tasks):
            try:
                msg = rq.get(timeout=1.0)
            except queue.Empty:
                crashed = [p for p in procs if not p.is_alive() and p.exitcode != 0]
                if crashed:
                    raise OracleError(f"worker {crashed[0].name} died (exit code "
                                      f"{crashed[0].exitcode}) with {len(tasks) - got} boards "
                                      "unlabelled") from None
                if not all(not p.is_alive() for p in procs):
                    continue
                try:  # every worker flushed its results before exiting: drain them
                    msg = rq.get(timeout=5.0)
                except queue.Empty:
                    raise OracleError(f"all workers exited with {len(tasks) - got} boards "
                                      "unlabelled") from None
            if msg[0] == "error":
                raise OracleError(f"solve failed for {[h[:12] for h in msg[1]]}: {msg[2]}")
            got += len(msg[1])
            on_rows(msg[1])
        ok = True
    finally:
        for p in procs:
            if ok:
                p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


# ------------------------------------------------------------------------------- provenance

def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def _commit() -> dict[str, Any]:
    return {"commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))}


def _cpu_model() -> str:
    p = Path("/proc/cpuinfo")
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _gpu_name() -> str:
    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    return (name.decode(errors="replace") if isinstance(name, bytes) else str(name)).strip("\0 ")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _log_session(root: Path, s: dict[str, Any]) -> None:
    p = _odir(root) / "sessions.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(s, sort_keys=True) + "\n")


def _sessions(root: Path) -> list[dict[str, Any]]:
    p = _odir(root) / "sessions.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def _hardware(cpu_workers: int, pcg_workers: int, pcg_device: str) -> dict[str, Any]:
    return {"host": platform.node(), "cpu_model": _cpu_model(), "logical_cpus": os.cpu_count(),
            "gpu_model": _gpu_name() if pcg_workers and pcg_device == "gpu" else None,
            "cpu_workers": cpu_workers, "pcg_workers": pcg_workers, "pcg_device": pcg_device,
            "python": platform.python_version()}


# ------------------------------------------------------------------------------- commands

def label(root: Path, *, N: int | None = None, cpu_workers: int, pcg_workers: int = 0,
          pcg_device: str = "gpu", gpu_batch: int = GPU_BATCH, flush_every: int = 256,
          strength_theta: float | None = DEFAULT_STRENGTH_THETA,
          progress: bool = False) -> dict[str, Any]:
    """Label every unique design of DIR that has no label yet; returns the session record.
    progress: print the count, rate and ETA at every flush."""
    root = Path(root)
    N = int(SPEC.geometry.N.value if N is None else N)
    if not (isinstance(flush_every, int) and flush_every >= 1):
        raise OracleError(f"flush_every must be an int >= 1, got {flush_every!r}")
    designs, info = collect_designs(root)
    meta = _meta(N, strength_theta)
    with RunLock(_odir(root) / ".lock"):
        have, old = load_parts(root)
        if old is not None:
            _same_settings(old, meta, "this run")
        stray = set(have) - set(designs)
        if stray:
            raise OracleError(f"{len(stray)} labels are for designs not in the inputs "
                              f"(e.g. {sorted(stray)[0][:12]}…): the inputs changed")
        todo = [(h, src, x, None) for h, (src, x) in designs.items() if h not in have]
        writer = _PartWriter(root, meta)
        buf: dict[str, list[dict]] = {}
        t0 = time.perf_counter()
        stats = {"labelled_this_run": 0, "by_solver": {}, "busy_s": {"cpu": 0.0, "gpu": 0.0}}

        def flush() -> None:
            for src, rs in buf.items():
                if rs:
                    writer.write(src, rs)
                    stats["labelled_this_run"] += len(rs)
            buf.clear()
            n, el = stats["labelled_this_run"], time.perf_counter() - t0
            if n and progress:
                rate = n / el
                eta = dt.timedelta(seconds=round((len(todo) - n) / rate))
                print(f"oracle: {n}/{len(todo)} boards this run, {rate:.2f} boards/s, "
                      f"ETA {eta}", flush=True)

        def on_rows(rows: list[dict]) -> None:
            for r in rows:
                buf.setdefault(r["source"], []).append(r)
                key = f"{r['solver']}/{r['device']}"
                stats["by_solver"][key] = stats["by_solver"].get(key, 0) + 1
                stats["busy_s"][r["device"]] += r["wall_s"]
            if sum(len(v) for v in buf.values()) >= flush_every:
                flush()

        session = {"kind": "label", "start": _now(), **_commit(), **meta,
                   "hardware": _hardware(cpu_workers, pcg_workers, pcg_device),
                   "gpu_batch": gpu_batch, "unique_designs": len(designs),
                   "already_labelled": len(have), "inputs": info}
        status = "error"
        try:
            if todo:
                _run_pool(todo, amg_workers=cpu_workers, pcg_workers=pcg_workers,
                          pcg_device=pcg_device, batch=gpu_batch, N=N,
                          strength_theta=strength_theta, shared=True, on_rows=on_rows)
            status = "complete"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            flush()
            session.update(stats, status=status, end=_now(),
                           elapsed_s=time.perf_counter() - t0)
            _log_session(root, session)
    return session


def finalise(root: Path) -> Path:
    """DIR/labels.parquet from the parts; refuses gaps, duplicates and stray labels."""
    root = Path(root)
    designs, info = collect_designs(root)
    rows, meta = load_parts(root)
    missing = [h for h in designs if h not in rows]
    if missing:
        raise OracleError(f"{len(missing)} of {len(designs)} unique designs have no label "
                          f"(e.g. {missing[0][:12]}…); rerun `label` to resume")
    stray = set(rows) - set(designs)
    if stray:
        raise OracleError(f"{len(stray)} labels are for designs not in the inputs")
    out = root / "labels.parquet"
    tmp = out.with_name(out.name + ".tmp")
    md = {**meta, "unique_designs": len(designs), "inputs": info, "finalised": _now(),
          "finalised_at": _commit()}
    t = pa.Table.from_pylist([rows[h] for h in sorted(rows)], schema=SCHEMA)
    pq.write_table(t.replace_schema_metadata({"twinheat": json.dumps(md, sort_keys=True)}), tmp)
    os.replace(tmp, out)
    return out


def sample_size(n: int, fraction: float) -> int:
    if not (0 < fraction <= 1) or n < 1:
        raise OracleError(f"need n >= 1 and fraction in (0, 1], got {n}, {fraction}")
    return max(1, math.ceil(round(fraction * n, 9)))


def cross_check(root: Path, *, fraction: float = CROSS_FRACTION, seed: int = 0,
                cpu_workers: int, pcg_workers: int = 0, pcg_device: str = "gpu",
                gpu_batch: int = GPU_BATCH) -> dict[str, Any]:
    """Re-solve a seeded random `fraction` of labels.parquet with the other solver."""
    root = Path(root)
    path = root / "labels.parquet"
    if not path.exists():
        raise OracleError(f"{path} is missing: run `finalise` first")
    t = pq.read_table(path)
    meta = _read_meta(t.schema, path)
    lab = t.to_pydict()
    designs, _ = collect_designs(root)
    if set(lab["x_hash"]) != set(designs):
        raise OracleError(f"{path} does not cover exactly the input designs")
    order = np.argsort(lab["x_hash"])
    k = sample_size(len(order), fraction)
    rng = np.random.Generator(np.random.PCG64(seed))
    pick = sorted(int(order[i]) for i in rng.choice(len(order), size=k, replace=False))
    other = {"pyamg_gmres": "pcg", "pcg_linez": "amg"}
    tasks = [(lab["x_hash"][i], lab["source"][i], designs[lab["x_hash"][i]][1],
              other[lab["solver"][i]]) for i in pick]
    got: dict[str, dict] = {}

    def on_rows(rows: list[dict]) -> None:
        got.update({r["x_hash"]: r for r in rows})

    t0 = time.perf_counter()
    _run_pool(tasks, amg_workers=cpu_workers, pcg_workers=pcg_workers, pcg_device=pcg_device,
              batch=gpu_batch, N=int(meta["N"]), strength_theta=meta["strength_theta"],
              shared=False, on_rows=on_rows)
    elapsed = time.perf_counter() - t0
    out = []
    for i in pick:
        h = lab["x_hash"][i]
        r = got[h]
        out.append({"x_hash": h, "label_solver": lab["solver"][i], "check_solver": r["solver"],
                    "check_device": r["device"], "peak_label": lab["oracle_peak"][i],
                    "peak_check": r["oracle_peak"],
                    "dpeak_K": abs(lab["oracle_peak"][i] - r["oracle_peak"]),
                    "argmax_equal": (lab["argmax_layer"][i], lab["argmax_y"][i],
                                     lab["argmax_x"][i]) == (r["argmax_layer"], r["argmax_y"],
                                                             r["argmax_x"]),
                    "relres_check": r["relres"], "iters_check": r["iters"],
                    "wall_s_check": r["wall_s"]})
    odir = _odir(root)
    odir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), odir / "cross_check.parquet")
    d = np.array([o["dpeak_K"] for o in out])
    n_fail = int((~(d < CROSS_TOL_K)).sum())
    res = {"n_labels": len(order), "n_checked": k, "fraction": fraction, "seed": seed,
           "tol_K": CROSS_TOL_K, "max_dpeak_K": float(d.max()), "n_fail": n_fail,
           "passed": n_fail == 0, "argmax_mismatch": sum(not o["argmax_equal"] for o in out),
           "labels_sha256": _sha(path), "date": _now(), "elapsed_s": elapsed, **_commit()}
    (odir / "cross_check.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    busy = {"cpu": 0.0, "gpu": 0.0}
    for r in got.values():
        busy[r["device"]] += r["wall_s"]
    _log_session(root, {"kind": "check", "start": res["date"], "elapsed_s": elapsed,
                        "status": "complete", "labelled_this_run": 0, "checked": k,
                        "busy_s": busy, **_commit(),
                        "hardware": _hardware(cpu_workers, pcg_workers, pcg_device)})
    if n_fail:
        raise OracleError(f"cross-check failed: {n_fail} of {k} labels differ from the other "
                          f"solver by >= {CROSS_TOL_K} K (max {d.max():.3e} K); see "
                          f"{odir / 'cross_check.parquet'}")
    return res


# ----------------------------------------------------------------------------------- report

def _hours(sessions: list[dict], kind: str | None = None) -> dict[str, float]:
    h = {"elapsed": 0.0, "cpu_core": 0.0, "cpu_busy": 0.0, "gpu": 0.0, "gpu_busy": 0.0}
    for s in sessions:
        if kind is not None and s["kind"] != kind:
            continue
        hw, e = s["hardware"], s["elapsed_s"] / 3600
        cores = hw["cpu_workers"] + (hw["pcg_workers"] if hw["pcg_device"] == "cpu" else 0)
        h["elapsed"] += e
        h["cpu_core"] += cores * e
        h["cpu_busy"] += s["busy_s"]["cpu"] / 3600
        if hw["pcg_workers"] and hw["pcg_device"] == "gpu":
            h["gpu"] += hw["pcg_workers"] * e
            h["gpu_busy"] += s["busy_s"]["gpu"] / 3600
    return h


def _q(v: list[float], p: float) -> float:
    return float(np.quantile(np.asarray(v, dtype=np.float64), p))


def write_report(root: Path, out: Path = DEFAULT_REPORT) -> Path:
    """experiments/oracle_report.md from DIR/labels.parquet, sessions.jsonl and the cross-check."""
    root = Path(root)
    path = root / "labels.parquet"
    if not path.exists():
        raise OracleError(f"{path} is missing: run `finalise` first")
    t = pq.read_table(path)
    meta = _read_meta(t.schema, path)
    lab = t.to_pydict()
    designs, info = collect_designs(root)
    sessions = _sessions(root)
    cpath = _odir(root) / "cross_check.json"
    cc = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else None
    if cc is not None and cc["labels_sha256"] != _sha(path):
        cc = {**cc, "passed": False, "stale": True}
    n_lab, n_uni = len(lab["x_hash"]), len(designs)
    complete = n_lab == n_uni and set(lab["x_hash"]) == set(designs)
    passed = bool(cc and cc["passed"])
    rows_total = sum(i["rows"] for i in info)

    L = ["# Oracle labels (N3.2)", "",
         "Generated by `python -m experiments.oracle_label report`; do not edit by hand. "
         "Every design is solved cold to a true relative residual <= "
         f"{ORACLE_TOL:g} (CLAUDE.md §11.5), without certified early stopping.", ""]
    reasons = ([] if complete else [f"{n_uni - n_lab} designs unlabelled"]) + (
        [] if passed else ["cross-check " + ("stale" if cc and cc.get("stale") else
                                             "not passed" if cc else "not run")])
    L += [f"Status: {'DONE' if not reasons else 'NOT DONE (' + '; '.join(reasons) + ')'}", ""]
    L += ["## Counts", "", "| Quantity | Value |", "|---|---|",
          f"| Input rows (all files) | {rows_total} |", f"| Unique designs | {n_uni} |",
          f"| Labelled | {n_lab} |", f"| Repeated rows (input rows - unique) | "
          f"{rows_total - n_uni} |", f"| Grid N | {meta['N']} |",
          f"| AMG strength theta | {meta['strength_theta']} |", ""]
    L += ["| Source | Rows | New unique designs | sha256 |", "|---|---|---|---|"]
    L += [f"| {i['source']} | {i['rows']} | {i['new_unique']} | `{i['sha256'][:12]}` |"
          for i in info]
    L += ["", "## Solvers", "",
          "| Solver | Device | Boards | Iterations median / p95 / max | Wall s per board "
          "median / mean |", "|---|---|---|---|---|"]
    keys = sorted({(s, d) for s, d in zip(lab["solver"], lab["device"], strict=True)})
    for s, d in keys:
        idx = [i for i in range(n_lab) if lab["solver"][i] == s and lab["device"][i] == d]
        it = [lab["iters"][i] for i in idx]
        w = [lab["wall_s"][i] for i in idx]
        L.append(f"| {s} | {d} | {len(idx)} | {_q(it, 0.5):.0f} / {_q(it, 0.95):.0f} / "
                 f"{max(it)} | {_q(w, 0.5):.3f} / {float(np.mean(w)):.3f} |")
    pk = lab["oracle_peak"]
    L += ["", "## Labels", "", "| Quantity | Value |", "|---|---|",
          f"| Peak rise min / median / max (K) | {min(pk):.4g} / {_q(pk, 0.5):.4g} / "
          f"{max(pk):.4g} |", f"| Max true relative residual | {max(lab['relres']):.3e} |",
          f"| Max energy-balance error | {max(lab['energy_balance']):.3e} |", ""]
    L += ["## Throughput", "",
          "| Session | Kind | Status | Elapsed h | Boards | boards/s | CPU workers | GPU |",
          "|---|---|---|---|---|---|---|---|"]
    for s in sessions:
        n = s.get("labelled_this_run", 0) or s.get("checked", 0)
        rate = f"{n / s['elapsed_s']:.3f}" if n and s["elapsed_s"] > 0 else "-"
        hw = s["hardware"]
        L.append(f"| {s['start']} | {s['kind']} | {s['status']} | {s['elapsed_s'] / 3600:.3f} | "
                 f"{n} | {rate} | {hw['cpu_workers']} | {hw['gpu_model'] or '-'} |")
    hl, hall = _hours(sessions, "label"), _hours(sessions)
    n_run = sum(s.get("labelled_this_run", 0) for s in sessions if s["kind"] == "label")
    overall = f"{n_run / (hl['elapsed'] * 3600):.3f}" if hl["elapsed"] > 0 and n_run else "-"
    L += ["", "| Quantity | Labelling | Labelling + cross-check |", "|---|---|---|",
          f"| Wall-clock hours | {hl['elapsed']:.3f} | {hall['elapsed']:.3f} |",
          f"| Labelling boards/s | {overall} | |",
          f"| CPU core-hours (workers x wall-clock) | {hl['cpu_core']:.3f} | "
          f"{hall['cpu_core']:.3f} |",
          f"| CPU core-hours busy (sum of per-board times) | {hl['cpu_busy']:.3f} | "
          f"{hall['cpu_busy']:.3f} |",
          f"| GPU-hours (GPU workers x wall-clock) | {hl['gpu']:.3f} | {hall['gpu']:.3f} |",
          f"| GPU-hours busy (sum of launch times) | {hl['gpu_busy']:.3f} | "
          f"{hall['gpu_busy']:.3f} |", ""]
    L += ["## Cross-check", "", "| Quantity | Value |", "|---|---|"]
    if cc is None:
        L.append("| Cross-check | not run |")
    else:
        L += [f"| Cross-check | {'PASS' if cc['passed'] else 'FAIL'}"
              f"{' (stale: labels changed since)' if cc.get('stale') else ''} |",
              f"| Re-solved with the other solver | {cc['n_checked']} of {cc['n_labels']} "
              f"({cc['fraction']:g}, seed {cc['seed']}) |",
              f"| Max abs peak difference (K) | {cc['max_dpeak_K']:.3e} (limit "
              f"{cc['tol_K']:g}) |", f"| Peak cell differs | {cc['argmax_mismatch']} |"]
    commits = sorted({f"{s['commit']}{'-dirty' if s['dirty'] else ''}" for s in sessions})
    hw = sorted({f"{s['hardware']['host']}: {s['hardware']['cpu_model']}"
                 f"{', ' + s['hardware']['gpu_model'] if s['hardware']['gpu_model'] else ''}"
                 for s in sessions})
    L += ["", "---", "",
          f"commit {', '.join(commits) or 'unknown'} · labels sha256 `{_sha(path)[:16]}` · "
          f"cross-check seed {cc['seed'] if cc else '-'} · hardware {'; '.join(hw) or '-'} · "
          f"generated {_now()}"]
    if any(s["dirty"] for s in sessions):
        L += ["", "A session ran from an uncommitted tree: these are not report numbers "
              "(CLAUDE.md §0 rule 11)."]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# -------------------------------------------------------------------------------------- CLI

def _theta(v: str) -> float | None:
    return None if v == "pyamg" else float(v)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sub.add_parser(c) for c in ("label", "finalise", "check", "report", "all")}
    for p in ps.values():
        p.add_argument("--inputs", type=Path, required=True)
    default_cpu = max(1, (os.cpu_count() or 2) - 2)
    for c in ("label", "check", "all"):
        p = ps[c]
        p.add_argument("--cpu-workers", type=int, default=default_cpu,
                       help=f"PyAMG processes, one core each (default {default_cpu})")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--gpu", action="store_true", help="add one batched-PCG GPU worker")
        g.add_argument("--pcg-cpu-workers", type=int, default=0,
                       help="PCG workers on the CPU (no GPU)")
        p.add_argument("--gpu-batch", type=int, default=GPU_BATCH)
    for c in ("label", "all"):
        ps[c].add_argument("--N", type=int, default=None, help="grid (default spec_v2, 256)")
        ps[c].add_argument("--flush-every", type=int, default=256)
        ps[c].add_argument("--amg-strength-theta", type=_theta, default=DEFAULT_STRENGTH_THETA,
                           help="float, or 'pyamg' for PyAMG's default")
    for c in ("check", "all"):
        ps[c].add_argument("--fraction", type=float, default=CROSS_FRACTION)
        ps[c].add_argument("--seed", type=int, default=0)
    for c in ("report", "all"):
        ps[c].add_argument("--out", type=Path, default=DEFAULT_REPORT)
    a = ap.parse_args(argv)

    def workers() -> dict[str, Any]:
        return {"cpu_workers": a.cpu_workers, "pcg_workers": 1 if a.gpu else a.pcg_cpu_workers,
                "pcg_device": "gpu" if a.gpu else "cpu", "gpu_batch": a.gpu_batch}

    if a.cmd in ("label", "all"):
        s = label(a.inputs, N=a.N, flush_every=a.flush_every,
                  strength_theta=a.amg_strength_theta, progress=True, **workers())
        print(f"labelled {s['labelled_this_run']} boards in {s['elapsed_s']:.1f} s "
              f"({s['already_labelled']} already labelled, {s['unique_designs']} unique)")
    if a.cmd in ("finalise", "all"):
        print(f"wrote {finalise(a.inputs)}")
    if a.cmd in ("check", "all"):
        r = cross_check(a.inputs, fraction=a.fraction, seed=a.seed, **workers())
        print(f"cross-check PASS: {r['n_checked']} re-solved, max |dpeak| "
              f"{r['max_dpeak_K']:.3e} K")
    if a.cmd in ("report", "all"):
        print(f"wrote {write_report(a.inputs, a.out)}")


if __name__ == "__main__":
    main()
