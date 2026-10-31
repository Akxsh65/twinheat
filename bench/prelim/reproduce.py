"""Reproduce the §8.6 preliminary tables with certify/ (N1.3; CLAUDE.md §8.4, §8.6).

    python -m bench.prelim.reproduce [--out bench/results]

writes prelim_stop.csv (Lemma 2(b), port of stop.js) and prelim_pointwise.csv (Lemma 2(c), port
of cert.js's first table). Boards, warm starts, emulated errors, seeds and RNG order are ported
exactly from the 5 Oct scripts (bench/prelim/plate.py for core.js). The certificates are ours:
  * κ̂ = certify.kappa (3 pyamg Ruge–Stüben V-cycles on ones); stop.js uses 3 V-cycles of its own
    geometric multigrid, so κ̂ ÷ true differs slightly.
  * The stopping bound is κ̂·‖b − A x_k‖∞ on the float64 TRUE residual (§8.4); stop.js uses the
    recursive residual. iters_cert_1K_recursive reports stop.js's rule for comparison.
  * The pointwise certificate is certify.pointwise.pointwise_bound (4 pyamg V-cycles + fix-up);
    cert.js's "idealCert" (max A⁻¹|r| by an exact solve, the §8.6 "best pointwise cert" column)
    is reported next to it.
Every iterate's bound, the warm start included, is checked against the error vs a 1e-12 direct
solve and raises CertificateViolation on the first violation (invariant B7). cert.js's learned-
certificate tables are not ported: §8.6 reports that test as failed and nothing claims it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from bench.prelim.plate import Board, Chip, colours, default_board, gauss, rng
from certify.kappa import kappa_from_z, multigrid, positive_z
from certify.pointwise import gauss_seidel_smooth, peak_interval, pointwise_bound
from certify.stopping import check_bound, jacobi, pcg_certified
from solvers.fvm import solve_direct

ROOT = Path(__file__).resolve().parents[2]
ORACLE_RTOL = 1e-12  # the N1.3 prompt's oracle: 1e-12 direct solve
REL_TOL = 1e-8  # stop.js: iterate to relative recursive residual 1e-8 ...
MAXIT = 5000  # ... with at most 5000 iterations
TAU_1K, TAU_01K = 1.0, 0.1  # K, the two certificate levels of §8.6
N_VCYCLES_KAPPA = 3  # §8.4
N_VCYCLES_POINTWISE = 4  # §8.4


@dataclass(frozen=True)
class StopVariant:
    label: str
    err_level: float  # fraction of peak θ
    seed: int
    board: Callable[[], Board]


def _no_sink() -> Board:
    b = default_board()
    b.edges = [False, False, False, False]
    return b


def _corner_chip() -> Board:
    b = default_board()
    b.chips.append(Chip(x=8, y=56, w=6, h=6, P=0.6))  # U4
    return b


def _thin_copper() -> Board:
    b = default_board()
    b.cuOz, b.hBase = 0.5, 12
    return b


# stop.js, in its order
STOP_VARIANTS = [
    StopVariant("default, 1 heat-sunk edge", 0.02, 3, default_board),
    StopVariant("no heat sink (worst conditioning)", 0.02, 4, _no_sink),
    StopVariant("4 chips, chip near corner", 0.05, 5, _corner_chip),
    StopVariant("thin copper, weak convection", 0.02, 6, _thin_copper),
]
POINTWISE_SEED = 11  # cert.js
POINTWISE_LEVELS = (1, 3)  # % of peak
POINTWISE_SWEEPS = (0, 2, 5, 10)


def true_inv_norm(A: sp.csr_matrix, block: int = 512) -> float:
    """‖A⁻¹‖∞ = max row sum of |A⁻¹|, by solving A X = I in column blocks (64² only)."""
    n = A.shape[0]
    lu = spla.splu(sp.csc_matrix(A))
    rows = np.zeros(n)
    for j0 in range(0, n, block):
        j1 = min(n, j0 + block)
        E = np.zeros((n, j1 - j0))
        E[np.arange(j0, j1), np.arange(j1 - j0)] = 1.0
        rows += np.abs(lu.solve(E)).sum(axis=1)
    return float(rows.max())


def stop_warm_start(xs: np.ndarray, N: int, err_level: float, seed: int) -> np.ndarray:
    """stop.js: μ_i = θ_i + level·peak·sin(x/7 + 6·R())·cos(y/9 + 1), one R() per cell in order."""
    R = rng(seed)
    draws = np.array([R() for _ in range(xs.size)])
    i = np.arange(xs.size)
    return xs + err_level * xs.max() * np.sin((i % N) / 7 + draws * 6) * np.cos((i // N) / 9 + 1)


def make_err(R: Callable[[], float], level: float, phi: np.ndarray, pk: float, N: int):
    """cert.js makeErr: 6 smooth blobs (~1 % of peak per unit level), ringing at copper edges,
    white noise; draws from R in cert.js's order (blobs first, then one gauss per cell)."""
    blobs = [(R() * 64, R() * 64, 6 + R() * 10, (R() - 0.5) * 2) for _ in range(6)]
    y, x = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float), indexing="ij")
    v = np.zeros((N, N))
    for cx, cy, sg, a in blobs:
        v += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sg * sg))
    p = np.asarray(phi, dtype=np.float32).astype(np.float64).reshape(N, N)
    g = np.zeros((N, N))
    g[1:-1, 1:-1] = (np.abs(p[1:-1, 2:] - p[1:-1, :-2]) + np.abs(p[2:, 1:-1] - p[:-2, 1:-1]))
    noise = np.array([gauss(R) for _ in range(N * N)]).reshape(N, N)
    return (level * pk * (0.01 * v + 0.004 * g * np.sin(1.7 * x + 0.9 * y)
                          + 0.0008 * noise)).ravel()


def run_stop() -> list[dict[str, Any]]:
    """Lemma 2(b) table: one row per stop.js variant. Raises CertificateViolation (B7)."""
    rows = []
    for v in STOP_VARIANTS:
        s = v.board().assemble()
        A, b, N = s.A, s.b, s.N
        xs = solve_direct(A, b, rtol=ORACLE_RTOL)
        pz = positive_z(A, N_VCYCLES_KAPPA)
        if pz is None:
            raise RuntimeError(f"{v.label}: no z > 0 with Az > 0 from {N_VCYCLES_KAPPA} V-cycles")
        kappa = kappa_from_z(A, pz[0])
        inv = true_inv_norm(A)
        mu = stop_warm_start(xs, N, v.err_level, v.seed)

        # stop.js's Jacobi PCG from μ, scanned iterate by iterate
        d = s.diag
        bnorm = float(np.linalg.norm(b))
        x = mu.copy()
        r = b - A @ x
        z = r / d
        p = z.copy()
        rz = float(r @ z)
        it = 0
        first = {"1K": None, "0_1K": None, "1K_recursive": None}
        ratio_1K = None

        def check(k: int, xk: np.ndarray, A=A, b=b, xs=xs, kappa=kappa, label=v.label) -> float:
            c = kappa * float(np.abs(b - A @ xk).max())
            check_bound(label, k, c, float(np.abs(xk - xs).max()))
            return c

        check(0, x)
        checked = 1
        while float(np.linalg.norm(r)) / bnorm > REL_TOL:
            if it >= MAXIT:
                raise RuntimeError(f"{v.label}: PCG did not reach {REL_TOL} in {MAXIT} iterations")
            Ap = A @ p
            a = rz / float(p @ Ap)
            x += a * p
            r -= a * Ap
            z = r / d
            rzn = float(r @ z)
            p = z + (rzn / rz) * p
            rz = rzn
            it += 1
            c = check(it, x)
            checked += 1
            if first["1K"] is None and c <= TAU_1K:
                first["1K"] = it
                ratio_1K = c / max(float(np.abs(x - xs).max()), 1e-12)  # stop.js's floor
            if first["0_1K"] is None and c <= TAU_01K:
                first["0_1K"] = it
            if first["1K_recursive"] is None and kappa * float(np.abs(r).max()) <= TAU_1K:
                first["1K_recursive"] = it
        if first["1K"] is None or first["0_1K"] is None:
            raise RuntimeError(f"{v.label}: never certified at {TAU_01K} K before 1e-8")

        cert_iters = {}
        for key, tau in (("1K", TAU_1K), ("0_1K", TAU_01K)):
            _, k, cert = pcg_certified(A, b, mu, jacobi(A), kappa, tau, maxit=MAXIT)
            if cert is None:
                raise RuntimeError(f"{v.label}: pcg_certified did not certify {tau} K")
            cert_iters[key] = k

        rows.append({
            "variant": v.label, "err_level": v.err_level, "seed": v.seed,
            "peak_K": float(xs.max()), "min_Az_positive": bool(pz[1].min() > 0),
            "kappa_hat": kappa, "true_inv_norm": inv, "kappa_over_true": kappa / inv,
            "iters_to_1e8": it, "iters_cert_1K": first["1K"], "iters_cert_0_1K": first["0_1K"],
            "saved_at_1K_pct": 100 * (1 - first["1K"] / it),
            "saved_at_0_1K_pct": 100 * (1 - first["0_1K"] / it),
            "bound_over_true_err_at_1K": ratio_1K,
            "iters_cert_1K_recursive": first["1K_recursive"],
            "pcg_certified_iters_1K": cert_iters["1K"],
            "pcg_certified_iters_0_1K": cert_iters["0_1K"],
            "iterates_checked": checked, "violations": 0,
        })
    return rows


def run_pointwise() -> list[dict[str, Any]]:
    """Lemma 2(c) table (cert.js's first table): default board, seed 11, levels 1 and 3 %."""
    board = default_board()
    s = board.assemble()
    A, b, N = s.A, s.b, s.N
    xs = solve_direct(A, b, rtol=ORACLE_RTOL)
    pk = float(xs.max())
    R = rng(POINTWISE_SEED)
    ml = multigrid(A)
    pz = positive_z(A, N_VCYCLES_KAPPA, ml)
    if pz is None:
        raise RuntimeError("default board: no z > 0 with Az > 0")
    lu = spla.splu(sp.csc_matrix(A))
    margin = float((s.Hd + s.dterm).min())  # Varah: smallest row-dominance margin (W/K)
    col = colours(N)
    rows = []
    for level in POINTWISE_LEVELS:
        e0 = make_err(R, level, board.phi, pk, N)
        for sweeps in POINTWISE_SWEEPS:
            mu = gauss_seidel_smooth(A, b, xs + e0, sweeps, sweep="red_black", colours=col)
            err = np.abs(mu - xs)
            r = b - A @ mu
            w = pointwise_bound(A, r, N_VCYCLES_POINTWISE, ml=ml, z_and_Az=pz)
            lo, hi = peak_interval(mu, w)
            rows.append({
                "err_level_pct": level, "gs_sweeps": sweeps, "true_max_err_K": float(err.max()),
                "true_peak_err_K": abs(float(mu.max()) - pk),
                "varah_K": float(np.abs(r).max()) / margin,
                "ideal_cert_K": float(lu.solve(np.abs(r)).max()),
                "cert_4v_K": float(w.max()), "cert_4v_over_true": float(w.max() / err.max()),
                "cert_valid": bool(np.all(err <= w)),
                "true_peak_K": pk, "peak_lo_K": lo, "peak_hi_K": hi,
            })
    return rows


def _commit() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def write_csv(path: Path, rows: list[dict[str, Any]], footer: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[*footer, *rows[0]])
        w.writeheader()
        for r in rows:
            w.writerow({**footer, **{k: (f"{v:.6g}" if isinstance(v, float) else v)
                                     for k, v in r.items()}})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results")
    args = ap.parse_args()
    footer = {"date": dt.date.today().isoformat(), "commit": _commit(),
              "hardware": platform.platform()}
    stop = run_stop()
    write_csv(args.out / "prelim_stop.csv", stop, footer)
    pw = run_pointwise()
    write_csv(args.out / "prelim_pointwise.csv", pw, footer)
    print(f"wrote {args.out / 'prelim_stop.csv'} and {args.out / 'prelim_pointwise.csv'}; "
          f"iterates checked {sum(r['iterates_checked'] for r in stop)}, violations 0")


if __name__ == "__main__":
    main()
