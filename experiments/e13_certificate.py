"""E13: certified early stopping on the 4-layer board (H6; CLAUDE.md §8.4, §8.6, §11.4, §15, D1).

    python -m experiments.e13_certificate predict --data DATASET_V1 --split test \
        --out MU_DIR --device cuda:0
    python -m experiments.e13_certificate run --data DATASET_V1 --split test --mu MU_DIR \
        --precond linez --pointwise-every 10 --workers 32 [--limit 200] \
        [--out experiments/results/e13]

predict (GPU): the FNO-ensemble mean μ of every board of the split, MU_DIR/mu.npy float32
[n, 4, N, N] with mu_meta.json (dataset hash, registry hash, commit). Only `test` and `ood_*`
splits are accepted, through TwinHeatDataset(allow_eval=True) (CLAUDE.md §5 data hygiene).

run (CPU pool, one thread per process): per board, rebuilt from its stored record,
  oracle θ    PyAMG-GMRES to a true relative residual 1e-10 (the §6.4 data tolerance, stricter
              than the 1e-8 oracle of §11.5)
  κ̂           certify.kappa from z ≈ A⁻¹1 (3 V-cycles); None means "no certificate": the board
              is counted, never dropped
  fixed       PCG to the fixed relative residual 1e-8 from μ (H6's baseline) and from 0
  Lemma 2(b)  certify.stopping.pcg_certified from μ to τ_v ∈ {1 K, 0.1 K}: iterations, bound,
              true error at the stop. bound < true error raises CertificateViolation (B7)
  Lemma 2(c)  the pointwise certificate (certify.pointwise.pointwise_bound) on the PCG iterates
              from μ every --pointwise-every iterations (and at μ): the first check whose
              max w ≤ τ_v; |θ − x| ≤ w is checked in every cell (B7). D1 judges H6 at 0.1 K
              counting this certificate: best = min(Lemma 2(b), Lemma 2(c)) iterations
  §8.6 table 2  μ after {0, 5, 10} red-black Gauss–Seidel sweeps: true worst error, the
              Varah bound (κ̂ from z = 1; ∞ when min(A1) is not certifiably > 0, as on L2/L3
              rows without a heat-sunk edge, §18.5) and the pointwise certificate
  loose tol   the loosest fixed tolerance whose mean iterations are >= the certified runs' mean,
              and the fraction of boards whose uncertified field then misses τ_v (§11.4 H6)
Outputs in --out: e13_boards.csv, e13_summary.json (H6 and week-6 verdicts), e13_tables.md (§8.6
layout), e13_saving.png (footer: commit, dataset hash, hardware, date), run_info.json.
H6 passes when the 0.1 K saving of the best certificate vs the fixed 1e-8 run, from μ, is >= 50 %
with zero violations; the §15 week-6 rule asks for the pointwise variant when the Lemma 2(b)
saving at 0.1 K is < 30 % or its median bound ÷ true error is > 50×.
The preconditioner is open decision D10, so --precond has no default: "linez" (line-z block
Jacobi, solvers/pcg.py, as data/generate.py) or "jacobi" (certify/stopping.py).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from certify.kappa import kappa_from_z, multigrid, positive_z
from certify.margin import residual_bound, row_abs_sum_bound
from certify.pointwise import gauss_seidel_smooth, pointwise_bound
from certify.stopping import CertificateViolation, check_bound, jacobi, pcg_certified
from solvers.pcg import line_z_block_jacobi

ROOT = Path(__file__).resolve().parents[1]
ORACLE_TOL = 1e-10  # §6.4 data-generation tolerance
FIXED_TOL = 1e-8  # §11.4 H6: "fixed 1e-8"
TAU_TAGS = {1.0: "1K", 0.1: "0.1K"}  # §8.6 columns; D1 judges H6 at 0.1 K
GS_SWEEPS = (0, 5, 10)  # §8.6 second table
H6_SAVING = 0.50  # §11.4
WEEK6_SAVING, WEEK6_RATIO = 0.30, 50.0  # §15 week-6 decision point
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DEFAULT_OUT = ROOT / "experiments" / "results" / "e13"

Precond = Callable[[np.ndarray], np.ndarray]


class E13Error(RuntimeError):
    """Inputs refused or inconsistent; the message names what."""


# ------------------------------------------------------------------------------- the analysis

def make_precond(name: str, A: sp.csr_matrix, coeffs: dict) -> Precond:
    if name == "jacobi":
        return jacobi(A)
    if name == "linez":
        diag = np.asarray(coeffs["diag"], dtype=np.float64)
        M = line_z_block_jacobi(diag, np.asarray(coeffs["gup"], dtype=np.float64))
        return lambda r: M(r.reshape(diag.shape)).ravel()
    raise E13Error(f"--precond must be 'linez' or 'jacobi' (open decision D10), got {name!r}")


def pcg_trace(A: sp.csr_matrix, b: np.ndarray, x0: np.ndarray, Minv: Precond,
              theta: np.ndarray, maxit: int,
              callback: Callable[[int, np.ndarray], None] | None = None) -> dict[str, Any]:
    """FP64 PCG to ‖b − Ax‖₂ ≤ 1e-8‖b‖₂ (confirmed on the true residual), recording the
    recursive relative residual and the true error max|θ − x| at every iterate (k = 0 is x0)."""
    x = np.array(x0, dtype=np.float64)
    bn = float(np.linalg.norm(b))
    r = b - A @ x
    rel, err = [float(np.linalg.norm(r)) / bn], [float(np.abs(theta - x).max())]
    if callback is not None:
        callback(0, x)
    z = Minv(r)
    p = z.copy()
    rz = float(r @ z)
    k = 0
    while rel[-1] > FIXED_TOL or float(np.linalg.norm(b - A @ x)) / bn > FIXED_TOL:
        if k >= maxit:
            raise E13Error(f"PCG did not reach {FIXED_TOL} in {maxit} iterations")
        Ap = A @ p
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        k += 1
        rel.append(float(np.linalg.norm(r)) / bn)
        err.append(float(np.abs(theta - x).max()))
        if callback is not None:
            callback(k, x)
        z = Minv(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return {"iters": k, "relres": np.asarray(rel), "err": np.asarray(err)}


def _colours(shape: tuple[int, ...]) -> np.ndarray:
    L, n, _ = shape
    lay, y, x = np.meshgrid(np.arange(L), np.arange(n), np.arange(n), indexing="ij")
    return ((lay + y + x) % 2).ravel()  # a red-black colouring of the 7-point layered stencil


def analyse_board(A: sp.spmatrix, b: np.ndarray, coeffs: dict, mu: np.ndarray,
                  theta: np.ndarray, *, board_id: str, precond: str, pointwise_every: int,
                  maxit: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One board's row of e13_boards.csv and its warm-start trace (relres, err per iterate)."""
    A = sp.csr_matrix(A, dtype=np.float64)
    b, mu, theta = (np.asarray(v, dtype=np.float64).ravel() for v in (b, mu, theta))
    shape = np.asarray(coeffs["diag"]).shape
    Minv = make_precond(precond, A, coeffs)
    ml = multigrid(A)
    pz = positive_z(A, n_vcycles=3, ml=ml)
    kappa = None if pz is None else kappa_from_z(A, pz[0])
    row: dict[str, Any] = {"board": board_id, "unknowns": A.shape[0],
                           "peak_K": float(theta.max()), "kappa": kappa,
                           "certificate": kappa is not None,
                           "warm_err_K": float(np.abs(mu - theta).max())}
    pw_first: dict[str, int | None] = dict.fromkeys(TAU_TAGS.values())

    def pointwise(k: int, x: np.ndarray) -> None:  # Lemma 2(c) on the PCG iterates
        if kappa is None or k % pointwise_every or all(v is not None for v in pw_first.values()):
            return
        w = pointwise_bound(A, b, x, ml=ml, z_and_Az=pz)
        if not np.all(np.abs(theta - x) <= w):
            raise CertificateViolation(f"board {board_id}, iterate {k}: pointwise bound below "
                                       "the true error in some cell (B7)")
        for tau, tag in TAU_TAGS.items():
            if pw_first[tag] is None and float(w.max()) <= tau:
                pw_first[tag] = k

    warm = pcg_trace(A, b, mu, Minv, theta, maxit, pointwise)
    row["iters_fixed_warm"] = warm["iters"]
    row["iters_fixed_cold"] = pcg_trace(A, b, np.zeros_like(b), Minv, theta, maxit)["iters"]
    if kappa is None:
        return row, warm
    for tau, tag in TAU_TAGS.items():
        x, it, c = pcg_certified(A, b, mu, Minv, kappa, tau, maxit)
        true = float(np.abs(theta - x).max())
        if c is None:
            raise E13Error(f"board {board_id}: pcg_certified hit maxit {maxit} at τ_v = {tau} K")
        check_bound(board_id, it, c, true)  # B7: raises CertificateViolation
        best = it if pw_first[tag] is None else min(it, pw_first[tag])
        row.update({f"iters_cert_kappa_{tag}": it, f"bound_{tag}": c, f"true_err_{tag}": true,
                    f"ratio_{tag}": c / true if true > 0 else math.inf,
                    f"iters_cert_pw_{tag}": pw_first[tag], f"iters_cert_best_{tag}": best})
    S, b_inf = row_abs_sum_bound(A), float(np.abs(b).max())
    k_varah = kappa_from_z(A, np.ones(A.shape[0]))  # z = 1 (§18.5)
    for s in GS_SWEEPS:
        xs = gauss_seidel_smooth(A, b, mu, s, sweep="red_black", colours=_colours(shape))
        w = pointwise_bound(A, b, xs, ml=ml, z_and_Az=pz)
        true_cells = np.abs(theta - xs)
        if not np.all(true_cells <= w):
            raise CertificateViolation(f"board {board_id}, {s} GS sweeps: pointwise bound "
                                       "below the true error in some cell (B7)")
        varah = math.inf if k_varah is None else residual_bound(
            k_varah, float(np.abs(b - A @ xs).max()), S, b_inf, float(np.abs(xs).max()))
        row.update({f"pw_true_gs{s}": float(true_cells.max()), f"pw_cert_gs{s}": float(w.max()),
                    f"varah_gs{s}": varah})
    return row, warm


# ------------------------------------------------------------------------------- the summary

def loose_tolerance(traces: list[dict[str, np.ndarray]], target_mean_iters: float,
                    tau: float) -> tuple[float, float, float]:
    """(tolerance, its mean iterations, fraction of boards with true error > tau there) for the
    loosest fixed relative-residual tolerance whose mean iterations are >= target_mean_iters."""
    cm = [np.minimum.accumulate(t["relres"]) for t in traces]  # first reach is monotone in tol

    def iters(tol: float) -> np.ndarray:
        return np.array([min(int(np.searchsorted(-c, -tol)), len(c) - 1) for c in cm])

    cand = np.unique(np.concatenate(cm))[::-1]  # loosest first; mean iterations rise along it
    lo, hi = 0, len(cand) - 1
    if iters(cand[hi]).mean() < target_mean_iters:
        lo = hi
    while lo < hi:
        mid = (lo + hi) // 2
        if iters(cand[mid]).mean() >= target_mean_iters:
            hi = mid
        else:
            lo = mid + 1
    tol = float(cand[lo])
    k = iters(tol)
    miss = np.mean([t["err"][i] > tau for t, i in zip(traces, k, strict=True)])
    return tol, float(k.mean()), float(miss)


def summarise(rows: list[dict[str, Any]], traces: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    cert = [i for i, r in enumerate(rows) if r["certificate"]]
    out: dict[str, Any] = {"n_boards": len(rows), "n_certified": len(cert),
                           "n_no_certificate": len(rows) - len(cert), "violations": 0,
                           "levels": {}}
    if not cert:
        out["H6"] = {"passed": False, "note": "no board has a certificate"}
        out["week6"] = {"use_pointwise": None}
        return out
    fixed = sum(rows[i]["iters_fixed_warm"] for i in cert)
    for tau, tag in TAU_TAGS.items():
        best = [rows[i][f"iters_cert_best_{tag}"] for i in cert]
        kap = [rows[i][f"iters_cert_kappa_{tag}"] for i in cert]
        ratio = [rows[i][f"bound_{tag}"] / rows[i][f"true_err_{tag}"] for i in cert
                 if rows[i][f"true_err_{tag}"] > 0]
        tol, it, miss = loose_tolerance([traces[i] for i in cert], float(np.mean(best)), tau)
        out["levels"][tag] = {
            "tau_K": tau, "mean_iters_fixed_warm": fixed / len(cert),
            "mean_iters_cert_kappa": float(np.mean(kap)),
            "mean_iters_cert_best": float(np.mean(best)),
            "saving_kappa": 1 - sum(kap) / fixed, "saving_best": 1 - sum(best) / fixed,
            "median_bound_over_true": float(np.median(ratio)) if ratio else math.inf,
            "loose_tol": tol, "loose_mean_iters": it, "loose_miss_fraction": miss}
    lv = out["levels"]["0.1K"]
    out["H6"] = {"criterion": f"saving_best at 0.1 K >= {H6_SAVING}, zero violations (D1)",
                 "saving_best_0.1K": lv["saving_best"], "violations": 0,
                 "passed": bool(lv["saving_best"] >= H6_SAVING)}
    out["week6"] = {"criterion": f"Lemma 2(b) saving at 0.1 K < {WEEK6_SAVING} or median bound "
                                 f"/ true > {WEEK6_RATIO} -> pointwise variant (§15)",
                    "saving_kappa_0.1K": lv["saving_kappa"],
                    "median_bound_over_true_0.1K": lv["median_bound_over_true"],
                    "use_pointwise": bool(lv["saving_kappa"] < WEEK6_SAVING
                                          or lv["median_bound_over_true"] > WEEK6_RATIO)}
    return out


# ----------------------------------------------------------------------------------- inputs

def check_split(split: str) -> str:
    if split == "test" or split.startswith("ood_"):
        return split
    raise E13Error(f"split {split!r} refused: E13 reads only test or ood_* (CLAUDE.md §5)")


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def commit_id() -> str:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise E13Error("the tree has uncommitted changes: commit before a reported run (rule 11)")
    return _git("rev-parse", "--short=7", "HEAD")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hardware() -> str:
    cpu = platform.processor() or "cpu"
    p = Path("/proc/cpuinfo")
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return f"{platform.node()} {cpu}"


def predict(data: Path, split: str, out: Path, device: str, batch: int, limit: int | None,
            registry: Path, norm_stats: Path, stage: str) -> Path:
    from models.dataset import TwinHeatDataset
    from models.ensemble import Ensemble
    from models.normalise import dataset_hash

    check_split(split)
    commit = commit_id()
    ens = Ensemble.load(registry, norm_stats, stage=stage, device=device)
    ds = TwinHeatDataset(data, split, N=256, allow_eval=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    out.mkdir(parents=True, exist_ok=True)
    mu = np.lib.format.open_memmap(out / "mu.npy", mode="w+", dtype=np.float32,
                                   shape=(n, 4, ds.N, ds.N))
    for s in range(0, n, batch):
        raw = np.stack([ds.sample(i, 0)[0] for i in range(s, min(s + batch, n))])
        mu[s:s + len(raw)] = ens.predict(raw)[0]
    mu.flush()
    meta = {"split": split, "n": n, "N": ds.N, "dataset_hash": dataset_hash(Path(data) / split),
            "registry_sha256": _sha(registry), "stage": stage, "commit": commit,
            "mu_sha256": _sha(out / "mu.npy"), "date": dt.date.today().isoformat()}
    (out / "mu_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out / "mu.npy"


def _job(args: tuple) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    i, rec, mu_path, precond, pointwise_every, maxit = args
    from data.raster import rasterise
    from solvers.amg_baseline import amg_solve
    from solvers.fvm import assemble_board, coefficients
    from twinheat import spec as specmod

    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    mu = np.load(mu_path, mmap_mode="r")[i]
    r = rasterise(rec, spec, mu.shape[-1])
    A, b = assemble_board(spec, r, rec["dirichlet_edges"])
    c = coefficients(spec, r, rec["dirichlet_edges"])
    theta = amg_solve(A, b, ORACLE_TOL, strength_theta=0.1).theta
    row, tr = analyse_board(A, b, c, mu, theta, board_id=f"{i}:{rec['id']}", precond=precond,
                            pointwise_every=pointwise_every, maxit=maxit)
    row["dirichlet_edges"] = "".join(rec["dirichlet_edges"]) or "-"
    return row, {"relres": tr["relres"].astype(np.float32), "err": tr["err"].astype(np.float32)}


def run(data: Path, split: str, mu_dir: Path, out: Path, *, precond: str, pointwise_every: int,
        workers: int, maxit: int, limit: int | None) -> dict[str, Any]:
    from models.dataset import TwinHeatDataset
    from models.normalise import dataset_hash

    check_split(split)
    if precond not in ("linez", "jacobi"):
        raise E13Error(f"--precond must be 'linez' or 'jacobi' (open decision D10), "
                       f"got {precond!r}")
    commit = commit_id()
    meta = json.loads((mu_dir / "mu_meta.json").read_text(encoding="utf-8"))
    dh = dataset_hash(Path(data) / split)
    if meta["split"] != split or meta["dataset_hash"] != dh:
        raise E13Error(f"{mu_dir}: μ was predicted for {meta['split']} / "
                       f"{meta['dataset_hash'][:12]}, not {split} / {dh[:12]}")
    if _sha(mu_dir / "mu.npy") != meta["mu_sha256"]:
        raise E13Error(f"{mu_dir}/mu.npy does not match its mu_meta.json hash")
    ds = TwinHeatDataset(data, split, N=meta["N"], allow_eval=True)
    n = meta["n"] if limit is None else min(limit, meta["n"])
    jobs = [(i, ds.record(i), str(mu_dir / "mu.npy"), precond, pointwise_every, maxit)
            for i in range(n)]
    old = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update(dict.fromkeys(THREAD_VARS, "1"))
    try:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            res = list(ex.map(_job, jobs, chunksize=1))
    finally:
        for v, val in old.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    rows, traces = [r for r, _ in res], [t for _, t in res]
    summary = summarise(rows, traces)
    info = {"commit": commit, "split": split, "dataset_hash": dh, "mu_meta": meta,
            "precond": precond, "pointwise_every": pointwise_every, "maxit": maxit,
            "workers": workers, "hardware": _hardware(), "date": dt.datetime.now().isoformat(),
            "oracle_tol": ORACLE_TOL, "fixed_tol": FIXED_TOL}
    write_outputs(out, rows, summary, info)
    return summary


# ---------------------------------------------------------------------------------- outputs

def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.3g}"
    return str(v)


def write_outputs(out: Path, rows: list[dict[str, Any]], summary: dict[str, Any],
                  info: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "e13_boards.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "e13_summary.json").write_text(json.dumps({**summary, "run": info}, indent=2,
                                                     default=str) + "\n", encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n",
                                       encoding="utf-8")
    footer = (f"commit {info['commit']} · dataset {info['dataset_hash'][:12]} · "
              f"{info['split']} n = {summary['n_boards']} · precond {info['precond']} · "
              f"{info['hardware']} · {info['date'][:10]}")
    L = ["# E13: certified early stopping on the 4-layer board", "",
         "Generated by experiments/e13_certificate.py; do not edit. " + footer, "",
         f"Boards: {summary['n_boards']} ({summary['n_no_certificate']} without a certificate). "
         f"Violations: {summary['violations']}.", "",
         "| τ_v | PCG iters to 1e-8 from μ (mean) | certified, Lemma 2(b) | certified, best of "
         "2(b)/2(c) | saved, 2(b) | saved, best | median bound ÷ true | loose tol at equal "
         "mean iters: missed τ_v |", "|---|---|---|---|---|---|---|---|"]
    for tag, lv in summary.get("levels", {}).items():
        L.append(f"| {tag} | {lv['mean_iters_fixed_warm']:.0f} | {lv['mean_iters_cert_kappa']:.0f} "
                 f"| {lv['mean_iters_cert_best']:.0f} | {lv['saving_kappa']:.0%} | "
                 f"{lv['saving_best']:.0%} | {lv['median_bound_over_true']:.3g}× | "
                 f"{lv['loose_miss_fraction']:.0%} (tol {lv['loose_tol']:.2g}) |")
    cert = [r for r in rows if r["certificate"]]
    if cert:
        L += ["", "| GS sweeps on μ | true worst error (K, median) | Varah bound (K, median) | "
              "pointwise cert (K, median) | cert ÷ true (median) |", "|---|---|---|---|---|"]
        for s in GS_SWEEPS:
            t = [r[f"pw_true_gs{s}"] for r in cert]
            v = [r[f"varah_gs{s}"] for r in cert]
            c = [r[f"pw_cert_gs{s}"] for r in cert]
            q = [ci / ti for ci, ti in zip(c, t, strict=True) if ti > 0]
            L.append(f"| {s} | {_fmt(float(np.median(t)))} | {_fmt(float(np.median(v)))} | "
                     f"{_fmt(float(np.median(c)))} | {_fmt(float(np.median(q)) if q else None)} |")
    h6, w6 = summary["H6"], summary["week6"]
    rule = "use the pointwise variant" if w6["use_pointwise"] else "keep Lemma 2(b)"
    L += ["", f"H6: {'PASS' if h6['passed'] else 'FAIL'} ({h6.get('criterion', h6.get('note'))})",
          f"Week-6 rule: {rule} ({w6.get('criterion', '')})"]
    (out / "e13_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if cert:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for tag in TAU_TAGS.values():
            sv = [1 - r[f"iters_cert_best_{tag}"] / r["iters_fixed_warm"] for r in cert
                  if r["iters_fixed_warm"] > 0]
            ax.hist(sv, bins=30, alpha=0.6, label=f"τ_v = {tag}")
        ax.axvline(H6_SAVING, color="k", ls="--", lw=1, label="H6 bar (50 %)")
        ax.set_xlabel("PCG iterations saved vs fixed 1e-8 (per board, best certificate)")
        ax.set_ylabel("boards")
        ax.legend(fontsize=8)
        fig.text(0.01, 0.01, footer, fontsize=6)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / "e13_saving.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--registry", type=Path, default=ROOT / "models" / "registry.json")
    p.add_argument("--norm-stats", type=Path, default=ROOT / "models" / "norm_stats.json")
    p.add_argument("--stage", default="ft256")
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--split", default="test")
    r.add_argument("--mu", type=Path, required=True)
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    r.add_argument("--precond", required=True, choices=("linez", "jacobi"),
                   help="open decision D10: no default")
    r.add_argument("--pointwise-every", type=int, required=True,
                   help="PCG iterations between Lemma 2(c) checks")
    r.add_argument("--workers", type=int, required=True)
    r.add_argument("--maxit", type=int, default=20000, help="certify/stopping.py's default")
    r.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        path = predict(a.data, a.split, a.out, a.device, a.batch, a.limit, a.registry,
                       a.norm_stats, a.stage)
        print(f"wrote {path}")
        return
    s = run(a.data, a.split, a.mu, a.out, precond=a.precond, pointwise_every=a.pointwise_every,
            workers=a.workers, maxit=a.maxit, limit=a.limit)
    print(json.dumps({"H6": s["H6"], "week6": s["week6"], "n_boards": s["n_boards"],
                      "n_no_certificate": s["n_no_certificate"]}, indent=2, default=str))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
