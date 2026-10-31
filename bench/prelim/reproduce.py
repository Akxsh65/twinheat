"""Re-run of the §8.6 preliminary evidence with declared settings (prompt N1.3; CLAUDE.md §8.6).

    python -m bench.prelim.reproduce [--out bench/results]

The 5 Oct scripts (stop.js, cert.js) were never saved (docs/decisions.md, "§8.6 preliminary
evidence"). This script is NOT a reproduction of them: every setting below was fixed and
committed BEFORE the first run, from surviving 5 Oct code where it exists, and declared
otherwise. Its output is new preliminary evidence, not a report number (CLAUDE.md §0 rule 11).

Solver and certificates are the project's own: solvers.fvm.assemble_plate, certify.kappa
(κ̂ from 3 pyamg V-cycles), certify.stopping (certified Jacobi-PCG, FP64 residual) and
certify.pointwise (Lemma 2(c), 4 V-cycles + fix-up). Boards and error shape come from the
explainer widget (bench/prelim/widget_board.py, checked against the JS to ~1e-16).

Writes prelim_stop.csv and prelim_pointwise.csv. Every bound is checked against the exact error
at every iterate; any violation is counted and the script exits non-zero (invariant B7).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from bench.prelim.widget_board import CHIPS, board_fields, emulate_shape, xorshift32
from certify.kappa import kappa_hat, multigrid, positive_z
from certify.pointwise import gauss_seidel_smooth, pointwise_bound
from certify.stopping import jacobi, pcg_certified
from solvers.fvm import assemble_plate
from solvers.mmatrix_test import assert_m_matrix

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ declared settings
# Board variants (the four §8.6 rows). The first two are fully determined by the row names; for
# the last two the row name leaves values open, chosen here before any run (docs/decisions.md).
CORNER_CHIP = ("U4", 58, 58, 5, 5, 0.5)  # 5x5 cells, 0.5 W, near the N-E corner (far from the sink)
VARIANTS = {
    "default_one_heat_sunk_edge": dict(chips=CHIPS, edges=("W",), cu_oz=2, h_base=24.0, lvl=0.02),
    "no_heat_sunk_edge": dict(chips=CHIPS, edges=(), cu_oz=2, h_base=24.0, lvl=0.02),
    "four_chips_one_near_a_corner": dict(chips=CHIPS + (CORNER_CHIP,), edges=("W",), cu_oz=2,
                                         h_base=24.0, lvl=0.05),
    # widget cuOz counts 17.5 µm units: 2 = 35 µm (default), 1 = 17.5 µm (0.5 oz, §6.3 minimum);
    # h_base is both faces combined: 10 = 5 per face, the bottom of the §6.3 natural range
    "thin_copper_weak_convection": dict(chips=CHIPS, edges=("W",), cu_oz=1, h_base=10.0,
                                        lvl=0.02),
}
NOISE_FRACTION = 0.02  # noise σ as a fraction of max|blob + ringing| (the "and noise" of the model)
SEED = 7  # xorshift32 seed, as the widget
TAUS = (1.0, 0.1)  # K, the two certified columns of §8.6
PCG_TOL, MAXIT = 1e-8, 4000  # "PCG iterations to 1e-8": ‖r‖₂/‖b‖₂, Jacobi-PCG (widget counter)
POINTWISE_LEVELS = (0.01, 0.03)  # the pointwise table's emulated error levels
GS_SWEEPS = (0, 5, 10)  # forward Gauss–Seidel sweeps on the prediction (certify.pointwise default)


def milder_prediction(theta: np.ndarray, phi: np.ndarray, chips: tuple, lvl: float) -> np.ndarray:
    """μ = θ + e: emulate()'s blobs + mild ringing, plus noise, scaled so max|e| = lvl·max θ."""
    shape = emulate_shape(phi, chips)
    r = xorshift32(SEED)
    u1 = np.array([r() for _ in range(theta.size)])
    u2 = np.array([r() for _ in range(theta.size)])
    u1[u1 == 0] = np.finfo(float).tiny
    noise = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    e = shape + NOISE_FRACTION * np.abs(shape).max() * noise
    return theta + e * (lvl * theta.max() / np.abs(e).max())


def _system(v: dict):
    phi, G, H, q, dx = board_fields(chips=v["chips"], cu_oz=v["cu_oz"], h_base=v["h_base"])
    A, b = assemble_plate(G, H, q, dx, v["edges"])
    assert_m_matrix(A)  # Lemma 2 needs it (B1)
    lu = spla.splu(sp.csc_matrix(A))
    return phi, A, b, lu.solve(b), float(lu.solve(np.ones(A.shape[0])).max())


def _iters_to_tol(A, b, x0) -> int:
    d = A.diagonal()
    x = x0.copy()
    r = b - A @ x
    z = r / d
    p, rz, bn, it = z.copy(), r @ z, np.linalg.norm(b), 0
    while np.linalg.norm(r) / bn > PCG_TOL and it < MAXIT:
        Ap = A @ p
        a = rz / (p @ Ap)
        x += a * p
        r -= a * Ap
        it += 1
        z = r / d
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    if it >= MAXIT:
        raise RuntimeError(f"Jacobi-PCG did not reach {PCG_TOL} in {MAXIT} iterations")
    return it


def stop_table() -> tuple[list[dict], int]:
    rows, violations = [], 0
    for name, v in VARIANTS.items():
        phi, A, b, theta, true_inv = _system(v)
        mu = milder_prediction(theta, phi, v["chips"], v["lvl"])
        kap = kappa_hat(A)
        if kap is None:
            raise RuntimeError(f"{name}: no certificate (z or Az not positive)")
        it8 = _iters_to_tol(A, b, mu)
        row = {"variant": name, "warm_start_error": v["lvl"], "peak_K": theta.max(),
               "iters_to_1e-8": it8, "kappa_over_true_inv": kap / true_inv}
        for tau in TAUS:
            bad: list[bool] = []

            def check(k, xk, c, bad=bad, theta=theta):  # bound vs exact error at every iterate
                bad.append(c < np.abs(xk - theta).max() * (1 - 1e-9))

            x, it, cert = pcg_certified(A, b, mu, jacobi(A), kap, tau, history=check)
            violations += sum(bad)
            if cert is None:
                raise RuntimeError(f"{name}: tau {tau} K not certified in the iteration cap")
            row[f"certified_le_{tau:g}K"] = it
            if tau == TAUS[0]:
                row["saved_at_1K"] = 1 - it / it8
                row["bound_over_true_error_at_stop"] = cert / np.abs(x - theta).max()
        rows.append(row)
    return rows, violations


def pointwise_table() -> tuple[list[dict], int]:
    v = VARIANTS["default_one_heat_sunk_edge"]
    phi, A, b, theta, _ = _system(v)
    ml = multigrid(A)
    pz = positive_z(A, ml=ml)
    off = sp.csr_matrix(A) - sp.diags(A.diagonal())
    alpha = float((A.diagonal() - np.abs(off).sum(axis=1).A1).min())  # Varah: min row margin
    if not alpha > 0:
        raise RuntimeError(f"Varah needs strict row dominance; min margin is {alpha}")
    rows, violations = [], 0
    for lvl in POINTWISE_LEVELS:
        mu = milder_prediction(theta, phi, v["chips"], lvl)
        for sweeps in GS_SWEEPS:
            x = gauss_seidel_smooth(A, b, mu, sweeps)
            r = b - A @ x
            err = np.abs(x - theta)
            w = pointwise_bound(A, r, ml=ml, z_and_Az=pz)
            violations += int(np.any(w < err * (1 - 1e-9)))
            rows.append({"emulated_error": lvl, "gs_sweeps": sweeps,
                         "true_worst_error_K": err.max(), "varah_bound_K": np.abs(r).max() / alpha,
                         "pointwise_cert_K": w.max(), "cert_over_true": w.max() / err.max()})
    return rows, violations


def _git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def _write(path: Path, rows: list[dict], meta: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[*rows[0], *meta])
        w.writeheader()
        for row in rows:
            w.writerow({**{k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in row.items()},
                        **meta})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results")
    args = ap.parse_args(argv)
    meta = {"commit": _git_commit(), "date": dt.date.today().isoformat(),
            "hardware": f"{platform.platform()} {platform.processor()}"}
    stop, v1 = stop_table()
    point, v2 = pointwise_table()
    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "prelim_stop.csv", stop, meta)
    _write(args.out / "prelim_pointwise.csv", point, meta)
    print(f"wrote {args.out}/prelim_stop.csv, prelim_pointwise.csv; commit {meta['commit']}; "
          f"violations {v1 + v2}")
    return 1 if v1 + v2 else 0


if __name__ == "__main__":
    sys.exit(main())
