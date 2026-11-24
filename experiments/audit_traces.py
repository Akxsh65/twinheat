"""Certified-audit traces for E14 (N3.3 / F6; CLAUDE.md §8.3, §18.1 Prop. B.15, D14, B7, B8).

    python -m experiments.audit_traces audited --inputs DIR [--frozen FROZEN_YAML]
    python -m experiments.audit_traces predict --inputs DIR --device cuda:0
    python -m experiments.audit_traces trace   --inputs DIR --workers 32 --check-every 10 \
        --kappa-iters 60 [--maxit 20000]

audited: the designs G5 audits at the frozen headline p on every trajectory (E14 replays only G5
at that p, exact and certified, experiments/e14_cert_audit.py). G5's audit probability is the
constant p, so its coin draws depend only on the coin seed and the step index; the set is computed
twice, by replaying G5 (run_gates.replay) and from the coin alone with no labels, and the two
must agree (B8: the coin never sees a label). Writes DIR/oracle/audited.json.
predict (GPU): the ensemble mean μ of every audited design (design_board -> rasterise -> inputs ->
ensemble), DIR/oracle/audit_mu.npy with its x_hash order; the largest |max μ − peak_hat| against
the trajectories' stored peak_hat is recorded (a consistency check, not a gate).
trace (CPU pool): per audited design, κ̂ (certify.kappa, z by AMG-CG, as E13) and
certify.peak_interval.audit_trace(μ, ...): FP64 PCG from μ, the Lemma 2(b) peak interval at
k = 0 and every --check-every iterations, to a true relative residual of 1e-8 or --maxit, the
label-independent cap of D14 (20,000, docs/decisions.md). Every check is verified against the
oracle peak (B7). Writes DIR/audit_traces.parquet: x_hash, check_iters, T_lo, T_hi (lists),
iters_full (PCG iterations from μ to 1e-8, at check granularity; the H7 baseline), stop_reason.
Resumes: parts under DIR/oracle/trace_parts are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments import run_gates as RG
from gate.coin import AuditCoin

D14_MAXIT = 20000  # docs/decisions.md, D14 (2026-11-24)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
SCHEMA = pa.schema([("x_hash", pa.string()), ("check_iters", pa.list_(pa.int64())),
                    ("T_lo", pa.list_(pa.float64())), ("T_hi", pa.list_(pa.float64())),
                    ("iters_full", pa.int64()), ("stop_reason", pa.string())])


class TraceError(RuntimeError):
    pass


# --------------------------------------------------------------------------- audited set

def audited_by_coin(trajs: list[RG.Trajectory], p: float) -> dict[str, list[str]]:
    """x_hash audited by a constant-p coin on each trajectory (seed = the trajectory seed)."""
    out = {}
    for tr in trajs:
        coin = AuditCoin(seed=tr.seed)
        hits = []
        for t in range(len(tr)):
            coin.commit(f"t{t}")
            if coin.draw(f"t{t}", p):
                hits.append(tr.x_hash[t])
        out[tr.name] = hits
    return out


def audited_by_replay(data: RG.Inputs, fz: dict[str, Any]) -> dict[str, list[str]]:
    out = {}
    for tr in data.trajectories:
        run = RG.replay(tr, data.labels, data.cal, fz, "G5", tr.seed, p=fz["p_audit"],
                        audit="exact")
        out[tr.name] = [r["x_hash"] for r in run.rows if r["audited"]]
    return out


def audited(data: RG.Inputs, fz: dict[str, Any]) -> list[str]:
    by_coin = audited_by_coin(data.trajectories, float(fz["p_audit"]))
    by_replay = audited_by_replay(data, fz)
    if by_coin != by_replay:
        bad = [k for k in by_coin if by_coin[k] != by_replay.get(k)]
        raise TraceError(f"G5's audited designs differ between the coin and the replay on {bad[:3]}"
                         ": the audit coin is not label-independent (B8)")
    return sorted({h for v in by_coin.values() for h in v})


# -------------------------------------------------------------------------------- traces

def _trace(args: tuple) -> dict[str, Any]:
    h, x, mu, oracle, check_every, maxit, kappa_iters = args
    from certify.kappa import kappa_hat
    from certify.peak_interval import audit_trace
    from data.raster import rasterise
    from data.sampler import design_board
    from solvers.fvm import assemble_board, coefficients
    from twinheat import spec as specmod

    spec = specmod.load(RG.ROOT / "config" / "spec_v2.yaml")
    rec = design_board(np.asarray(x, dtype=np.float64))
    r = rasterise(rec, spec, mu.shape[-1])
    A, _ = assemble_board(spec, r, rec["dirichlet_edges"])
    c = coefficients(spec, r, rec["dirichlet_edges"])
    k = kappa_hat(A, kappa_iters, accel="cg")
    if k is None:
        raise TraceError(f"design {h[:12]}…: no certificate (κ̂ None)")
    tr = audit_trace(mu, c, c["rhs"], k, check_every, maxit=maxit)
    for it, lo, hi in zip(tr.iters, tr.T_lo, tr.T_hi, strict=True):
        if not lo <= oracle <= hi:
            raise RG.CertificateViolation(f"design {h[:12]}…, iteration {it}: [{lo}, {hi}] K "
                                          f"does not contain the oracle peak {oracle} K (B7)")
    return {"x_hash": h, "check_iters": list(map(int, tr.iters)), "T_lo": list(tr.T_lo),
            "T_hi": list(tr.T_hi),
            "iters_full": int(tr.iters[-1]) if tr.stop_reason == "converged" else None,
            "stop_reason": tr.stop_reason}


def trace_all(root: Path, *, workers: int, check_every: int, maxit: int,
              kappa_iters: int, chunk: int = 256) -> Path:
    odir = root / "oracle"
    meta = json.loads((odir / "audit_mu.json").read_text(encoding="utf-8"))
    mu_all = np.load(odir / "audit_mu.npy", mmap_mode="r")
    hashes = meta["x_hash"]
    lab = pq.read_table(root / "labels.parquet", columns=["x_hash", "oracle_peak"]).to_pydict()
    oracle = dict(zip(lab["x_hash"], lab["oracle_peak"], strict=True))
    designs = _design_vectors(root)
    parts = odir / "trace_parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in parts.glob("*.parquet"):
        done |= set(pq.read_table(p, columns=["x_hash"]).column("x_hash").to_pylist())
    todo = [i for i, h in enumerate(hashes) if h not in done]
    print(f"traces: {len(done)} done, {len(todo)} to go", flush=True)
    old = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update(dict.fromkeys(THREAD_VARS, "1"))
    try:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            for s in range(0, len(todo), chunk):
                idx = todo[s:s + chunk]
                jobs = [(hashes[i], designs[hashes[i]], np.array(mu_all[i], dtype=np.float64),
                         float(oracle[hashes[i]]), check_every, maxit, kappa_iters) for i in idx]
                rows = list(ex.map(_trace, jobs, chunksize=1))
                tmp = parts / f"part_{len(list(parts.glob('*.parquet'))):05d}.parquet.tmp"
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp)
                os.replace(tmp, tmp.with_suffix(""))
                print(f"traces: {len(done) + s + len(idx)}/{len(hashes)}", flush=True)
    finally:
        for v, val in old.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    tables = [pq.read_table(p) for p in sorted(parts.glob("*.parquet"))]
    t = pa.concat_tables(tables)
    if sorted(t.column("x_hash").to_pylist()) != sorted(hashes):
        raise TraceError("the trace parts do not cover exactly the audited designs")
    out = root / "audit_traces.parquet"
    md = {"check_every": check_every, "maxit": maxit, "kappa_iters": kappa_iters,
          "kappa_accel": "cg", "n": len(hashes), "commit": RG.code_commit()}
    pq.write_table(t.replace_schema_metadata({"twinheat": json.dumps(md)}), out)
    return out


def _design_vectors(root: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for f in sorted((root / "trajectories").glob("traj_*.parquet")):
        d = pq.read_table(f, columns=["x_hash", "design_vector"]).to_pydict()
        out.update(zip(d["x_hash"], d["design_vector"], strict=True))
    return out


def predict(root: Path, device: str, batch: int = 16) -> Path:
    import torch  # noqa: F401  (models need it)

    from data.raster import rasterise
    from data.sampler import design_board
    from models import inputs as inp
    from models.ensemble import Ensemble
    from twinheat import spec as specmod

    cfg = json.loads((root / "oracle" / "audited.json").read_text(encoding="utf-8"))
    hashes = cfg["x_hash"]
    spec = specmod.load(RG.ROOT / "config" / "spec_v2.yaml")
    ens = Ensemble.load(RG.ROOT / "models" / "registry.json", RG.ROOT / "models" /
                        "norm_stats.json", stage="ft256", device=device)
    designs = _design_vectors(root)
    peak_hat = {}
    for f in sorted((root / "trajectories").glob("traj_*.parquet")):
        d = pq.read_table(f, columns=["x_hash", "peak_hat"]).to_pydict()
        peak_hat.update(zip(d["x_hash"], d["peak_hat"], strict=True))
    out = root / "oracle" / "audit_mu.npy"
    mu = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                   shape=(len(hashes), 4, 256, 256))
    worst = 0.0
    for s in range(0, len(hashes), batch):
        hs = hashes[s:s + batch]
        raw = np.stack([inp.build(rasterise(design_board(np.asarray(designs[h])), spec, 256),
                                  spec) for h in hs])
        m = ens.predict(raw)[0]
        mu[s:s + len(hs)] = m
        worst = max(worst, max(abs(float(m[j].max()) - peak_hat[h]) for j, h in enumerate(hs)))
    mu.flush()
    (root / "oracle" / "audit_mu.json").write_text(json.dumps(
        {"x_hash": hashes, "max_abs_peak_hat_diff_K": worst, "commit": RG.code_commit()},
        indent=1), encoding="utf-8")
    print(f"μ for {len(hashes)} audited designs; max |max μ − peak_hat| = {worst:.3e} K")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("audited")
    a1.add_argument("--frozen", type=Path, default=RG.DEFAULT_FROZEN)
    a2 = sub.add_parser("predict")
    a2.add_argument("--device", default="cuda:0")
    a3 = sub.add_parser("trace")
    a3.add_argument("--workers", type=int, required=True)
    a3.add_argument("--check-every", type=int, required=True)
    a3.add_argument("--kappa-iters", type=int, required=True)
    a3.add_argument("--maxit", type=int, default=D14_MAXIT, help="D14 (docs/decisions.md)")
    for p in (a1, a2, a3):
        p.add_argument("--inputs", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "audited":
        fz, data = RG.prepare(a.frozen, a.inputs)
        hs = audited(data, fz)
        path = a.inputs / "oracle" / "audited.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"p_audit": fz["p_audit"], "x_hash": hs}, indent=1),
                        encoding="utf-8")
        print(f"{len(hs)} audited designs (coin and replay agree) -> {path}")
    elif a.cmd == "predict":
        predict(a.inputs, a.device)
    else:
        out = trace_all(a.inputs, workers=a.workers, check_every=a.check_every,
                        maxit=a.maxit, kappa_iters=a.kappa_iters)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
