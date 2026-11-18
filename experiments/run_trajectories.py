"""Proposer trajectories P1–P3 × 3 levels × 5 seeds for the offline gate experiments (M3.3;
CLAUDE.md §11.2, §11.5). Runs on the GPU server (gpu_instructions.md, M3.3).

    python -m experiments.run_trajectories terciles --data DATA_ROOT --device cuda:0
    python -m experiments.run_trajectories run --proposer P1 --level 2 --seed 0 --out OUT
    python -m experiments.run_trajectories all --out OUT --device cuda:0 [--only P3] [--jobs 3]
    python -m experiments.run_trajectories calib --seed 2026 --out OUT/calibration.parquet

`terciles` fits the OOD-score tercile cut points on the validation split (gate/groups.py) and
writes them next to the E5 results; `run` / `all` need them. Proposers see surrogate predictions
only, never a gate or a coin, so one oracle-labelled trajectory set scores every gate offline.

Output per run: OUT/traj_<P>_a<level>_s<seed>.parquet, one row per proposal, with the §10.4
columns except oracle_peak. Gate-dependent columns (gate, s, q_t, B, decision, p_t, audited, T_lo,
T_hi, S_hi, err, alpha_t, iters_used) are null here; experiments/run_gates.py fills them. s is
null because λ and ε are frozen under D4; r_hot and sd_hot are stored. Extra columns:
design_id, design_vector (36 floats, so N3.2 can rebuild the board with data/sampler.design_board),
peak_hat, r_hot_raw, ood_score, ood_tercile, edge_hotspot, round. OUT/manifest.json lists every
file with its SHA-256, row count and settings; it is the hand-off to N3.2 (oracle labels).
The config must be committed (CLAUDE.md §4); the run id is traj_<P>_a<level>-<commit>-<seed>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from gate import groups as G
from gate import proposers as P
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "trajectories.yaml"
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
NAMES = ("P1", "P2", "P3")
LEVELS = (1, 2, 3)
REQUIRED = ("registry", "norm_stats", "stage", "ood_score", "ood_terciles", "N", "m_hot",
            "eval_batch", "seeds", "P1", "P2", "P3")

_NULL_GATE_COLS = {  # §10.4 columns that only a gate can fill
    "gate": pa.string(), "s": pa.float64(), "q_t": pa.float64(), "B": pa.float64(),
    "decision": pa.string(), "p_t": pa.float64(), "audited": pa.bool_(), "T_lo": pa.float64(),
    "T_hi": pa.float64(), "S_hi": pa.float64(), "err": pa.int64(), "alpha_t": pa.float64(),
    "iters_used": pa.int64(),
}
SCHEMA = pa.schema([
    ("schema_version", pa.string()), ("run_id", pa.string()), ("gate", pa.string()),
    ("seed", pa.int64()), ("t", pa.int64()), ("proposer", pa.string()),
    ("aggressiveness", pa.int64()), ("group", pa.string()), ("x_hash", pa.string()),
    ("s", pa.float64()), ("r_hot", pa.float64()), ("sd_hot", pa.float64()),
    ("q_t", pa.float64()), ("B", pa.float64()), ("decision", pa.string()),
    ("p_t", pa.float64()), ("audited", pa.bool_()), ("T_lo", pa.float64()),
    ("T_hi", pa.float64()), ("S_hi", pa.float64()), ("err", pa.int64()),
    ("alpha_t", pa.float64()), ("iters_used", pa.int64()),
    # extras
    ("design_id", pa.string()), ("design_vector", pa.list_(pa.float64(), P.DIM)),
    ("peak_hat", pa.float64()), ("r_hot_raw", pa.float64()), ("ood_score", pa.float64()),
    ("ood_tercile", pa.int64()), ("edge_hotspot", pa.bool_()), ("round", pa.int64()),
])


class TrajectoryError(RuntimeError):
    pass


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def commit_id(config: Path) -> str:
    """Short HEAD; refuses a dirty tree or an untracked config (CLAUDE.md §0 rule 11, §4)."""
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise TrajectoryError("the tree has uncommitted changes: commit before a reported run")
    try:
        _git("ls-files", "--error-unmatch", str(Path(config).resolve().relative_to(ROOT)))
    except (ValueError, subprocess.CalledProcessError) as e:
        raise TrajectoryError(f"{config} must be a committed file inside the repo") from e
    return _git("rev-parse", "--short=7", "HEAD")


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if cfg.get(k) is None]
    if missing:
        raise TrajectoryError(f"{path}: set {', '.join(missing)}")
    if len(cfg["P1"]["lambda_per_K"]) != 3 or len(cfg["P2"]["presets"]) != 3 \
            or len(cfg["P3"]["eta"]) != 3:
        raise TrajectoryError(f"{path}: every proposer needs exactly 3 levels")
    return cfg


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def proposer_seed(seed: int, name: str, level: int) -> int:
    """Independent stream per (seed, proposer, level)."""
    ss = np.random.SeedSequence([int(seed), NAMES.index(name) + 1, int(level)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def make_proposer(cfg: dict[str, Any], name: str, level: int, seed: int, oracle=None):
    k = LEVELS.index(level)
    s = proposer_seed(seed, name, level)
    if name == "P1":
        c = cfg["P1"]
        return P.TiltedMCMC(lam=float(c["lambda_per_K"][k]), step=float(c["step"]),
                            n_proposals=int(c["proposals"]), seed=s, level=level)
    if name == "P2":
        c = cfg["P2"]
        pre = c["presets"][k]
        return P.SimulatedAnnealing(T0=float(pre["T0_K"]), step=float(pre["step"]),
                                    steps=int(c["steps"]), T_end_ratio=float(c["T_end_ratio"]),
                                    seed=s, level=level)
    if name == "P3":
        if oracle is None:
            raise TrajectoryError("P3 needs a gradient oracle")
        c = cfg["P3"]
        return P.BatchedGradient(oracle, eta=float(c["eta"][k]), starts=int(c["starts"]),
                                 steps=int(c["steps"]), halvings=int(c["halvings"]), seed=s,
                                 level=level)
    raise TrajectoryError(f"unknown proposer {name!r}; expected one of {NAMES}")


def run_one(cfg: dict[str, Any], name: str, level: int, seed: int, out: Path, *, ensemble,
            ood, cuts: tuple[float, float], commit: str, oracle=None) -> dict[str, Any]:
    """One trajectory -> OUT/traj_<P>_a<level>_s<seed>.parquet; returns its manifest entry."""
    if level not in LEVELS or seed not in cfg["seeds"]:
        raise TrajectoryError(f"level {level} / seed {seed} not in the config")
    prop = make_proposer(cfg, name, level, seed, oracle)
    evaluate = P.SurrogateEvaluator(ensemble, ood, SPEC, n=int(cfg["N"]), m=int(cfg["m_hot"]),
                                    batch=int(cfg["eval_batch"]))
    grouper = G.make_grouper(cuts)
    hist = P.History()
    preds: list[P.Prediction] = []
    rounds: list[int] = []
    r = 0
    while not prop.done:
        xs = prop.next(hist)
        if not xs:
            break
        ps = evaluate(xs)
        P.record(hist, ps, grouper, prop)
        preds += ps
        rounds += [r] * len(ps)
        r += 1
    run_id = f"traj_{name}_a{level}-{commit}-{seed}"
    rows = []
    for e, p, rd in zip(hist.entries, preds, rounds, strict=True):
        rows.append({
            "schema_version": "1.0", "run_id": run_id, "seed": seed, "t": e.t, "proposer": name,
            "aggressiveness": level, "group": e.group, "x_hash": e.x_hash, "r_hot": e.r_hot,
            "sd_hot": e.sd_hot, **dict.fromkeys(_NULL_GATE_COLS), "design_id": e.design_id,
            "design_vector": list(e.x), "peak_hat": e.peak_hat, "r_hot_raw": e.r_hot_raw,
            "ood_score": e.ood_score, "ood_tercile": G.ood_tercile(e.ood_score, cuts),
            "edge_hotspot": p.edge_hotspot, "round": rd})
    settings = {"proposer": name, "level": level, "seed": seed,
                "proposer_seed": proposer_seed(seed, name, level),
                "config": cfg[name], "N": cfg["N"], "m_hot": cfg["m_hot"], "commit": commit,
                "ood_tercile_cuts": list(cuts), "infeasible_moves": getattr(prop, "infeasible", 0),
                "stuck_steps": getattr(prop, "stuck", 0),
                "members": [m.get("sha256") for m in getattr(ensemble, "members", [])]}
    table = pa.Table.from_pylist(rows, schema=SCHEMA).replace_schema_metadata(
        {"twinheat": json.dumps(settings, sort_keys=True)})
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"traj_{name}_a{level}_s{seed}.parquet"
    pq.write_table(table, path)
    return {"file": path.name, "run_id": run_id, "rows": len(rows),
            "unique_designs": len({e.x_hash for e in hist.entries}), "sha256": _sha256(path),
            **{k: settings[k] for k in ("proposer", "level", "seed", "infeasible_moves",
                                        "stuck_steps")}}


def write_manifest(out: Path, entries: list[dict[str, Any]], cfg_path: Path,
                   commit: str) -> Path:
    path = out / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    keep = {e["file"]: e for e in old["runs"]}
    keep.update({e["file"]: e for e in entries})
    runs = sorted(keep.values(), key=lambda e: e["file"])
    path.write_text(json.dumps({
        "schema": "twinheat-trajectories/1", "commit": commit, "config": str(cfg_path),
        "config_sha256": _sha256(cfg_path), "runs": runs,
        "total_rows": sum(e["rows"] for e in runs),
        "total_unique_designs_per_file": sum(e["unique_designs"] for e in runs),
    }, indent=2) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------- loading (GPU host)


def _load(cfg: dict[str, Any], device: str, need_cuts: bool = True):
    from models.diff_inputs import DesignInputs
    from models.ensemble import Ensemble
    from models.ood_score import OODScore

    ens = Ensemble.load(_path(cfg["registry"]), _path(cfg["norm_stats"]), stage=cfg["stage"],
                        device=device)
    ood = OODScore.load(_path(cfg["ood_score"]))
    cuts = G.load_terciles(_path(cfg["ood_terciles"]), ood_score_path=_path(cfg["ood_score"])) \
        if need_cuts else None
    di = DesignInputs(SPEC, n=int(cfg["N"]))
    oracle = P.GradientOracle(ens, di, chunk=int(cfg["P3"]["grad_chunk"]))
    return ens, ood, cuts, oracle


def fit_terciles(cfg: dict[str, Any], data: Path, device: str, commit: str) -> Path:
    import torch

    from models.dataset import TwinHeatDataset
    from models.normalise import dataset_hash
    from models.ood_score import pooled_features

    ens, ood, _, _ = _load(cfg, device, need_cuts=False)
    ds = TwinHeatDataset(data, "val", N=int(cfg["N"]))
    scores = []
    b = int(cfg["eval_batch"])
    for s in range(0, len(ds), b):
        raw = np.stack([ds.sample(i, 0)[0] for i in range(s, min(s + b, len(ds)))])
        xn = torch.from_numpy(ens.norm.normalise_x(raw)).to(ens.device)
        scores.append(ood.score([pooled_features(m, xn).cpu().numpy() for m in ens.models]))
    v = np.concatenate(scores)
    cuts = G.ood_tercile_cuts(v)
    path = _path(cfg["ood_terciles"])
    G.save_terciles(path, cuts, n_val=len(v), ood_score_path=_path(cfg["ood_score"]),
                    dataset_hash=dataset_hash(Path(data) / "val"), commit=commit)
    return path


def run_calibration(cfg: dict[str, Any], n: int, seed: int, out: Path, *, ensemble, ood,
                    commit: str) -> dict[str, Any]:
    """n fresh draws from p₀ (uniform on the feasible cube), predicted like the trajectories:
    the design-space calibration set of run_gates (decision 2026-11-18). N3.2 labels it."""
    if not (isinstance(n, int) and n >= 2):
        raise TrajectoryError(f"n must be an int >= 2, got {n!r}")
    ss = np.random.SeedSequence([int(seed), 0, 0])  # stream 0: no proposer uses it
    rng = np.random.Generator(np.random.PCG64(ss.generate_state(1, dtype=np.uint64)[0]))
    xs = [P.sample_p0(rng) for _ in range(n)]
    evaluate = P.SurrogateEvaluator(ensemble, ood, SPEC, n=int(cfg["N"]), m=int(cfg["m_hot"]),
                                    batch=int(cfg["eval_batch"]))
    preds = evaluate(xs)
    table = pa.table({
        "x_hash": [p.x_hash for p in preds],
        "design_vector": pa.array([p.x.tolist() for p in preds], pa.list_(pa.float64(), P.DIM)),
        "peak_hat": [p.peak_hat for p in preds], "r_hot": [p.r_hot for p in preds],
        "sd_hot": [p.sd_hot for p in preds], "r_hot_raw": [p.r_hot_raw for p in preds],
        "ood_score": [p.ood_score for p in preds],
    }).replace_schema_metadata({"twinheat": json.dumps(
        {"kind": "calibration_p0", "n": n, "seed": seed, "N": cfg["N"], "m_hot": cfg["m_hot"],
         "commit": commit}, sort_keys=True)})
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return {"file": out.name, "rows": n, "sha256": _sha256(out)}


def _job(args: tuple) -> dict[str, Any]:
    cfg_path, name, level, seed, out, device, commit = args
    cfg = load_config(cfg_path)
    ens, ood, cuts, oracle = _load(cfg, device)
    return run_one(cfg, name, level, seed, Path(out), ensemble=ens, ood=ood, cuts=cuts,
                   commit=commit, oracle=oracle)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("terciles")
    t.add_argument("--data", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--proposer", choices=NAMES, required=True)
    r.add_argument("--level", type=int, choices=LEVELS, required=True)
    r.add_argument("--seed", type=int, required=True)
    c = sub.add_parser("calib")
    c.add_argument("--n", type=int, default=1000, help="p0 designs (decision 2026-11-18: 1,000)")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--out", type=Path, required=True, help="calibration.parquet")
    a = sub.add_parser("all")
    a.add_argument("--only", choices=NAMES, action="append")
    a.add_argument("--jobs", type=int, default=1)
    for p in (r, a):
        p.add_argument("--out", type=Path, required=True)
    for p in (t, r, a, c):
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        p.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    commit = commit_id(args.config)
    cfg = load_config(args.config)
    if args.cmd == "terciles":
        print(f"wrote {fit_terciles(cfg, args.data, args.device, commit)}")
        return
    if args.cmd == "calib":
        ens, ood, _, _ = _load(cfg, args.device, need_cuts=False)
        e = run_calibration(cfg, args.n, args.seed, args.out, ensemble=ens, ood=ood,
                            commit=commit)
        print(f"wrote {args.out}: {e['rows']} p0 designs, sha256 {e['sha256'][:12]}")
        return
    jobs = ([(args.proposer, args.level, args.seed)] if args.cmd == "run" else
            [(n, lv, s) for n in (args.only or NAMES) for lv in LEVELS for s in cfg["seeds"]])
    work = [(args.config, n, lv, s, str(args.out), args.device, commit) for n, lv, s in jobs]
    n_jobs = getattr(args, "jobs", 1)
    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs, mp_context=get_context("spawn")) as ex:
            entries = list(ex.map(_job, work))
    else:
        entries = [_job(w) for w in work]
    for e in entries:
        print(f"{e['file']}: {e['rows']} rows ({e['unique_designs']} unique)")
    print(f"wrote {write_manifest(args.out, entries, args.config, commit)}")


if __name__ == "__main__":
    main()
