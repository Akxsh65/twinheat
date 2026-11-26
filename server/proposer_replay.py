"""Replay of a real offline proposer run in the viewer (F7 demo B; CLAUDE.md §11.2, §13 T1.8).

    python -m server.proposer_replay --inputs GATE_INPUTS --proposer P2 --level 2 --seed 0 \
        --frames 60 --device cuda:0 --tau-v 0.1 --every 10 --kappa-iters 60 \
        --out viewer/public/demo/replay_P2

Reads one trajectory of the gate experiments, GATE_INPUTS/trajectories/traj_<P>_a<level>_s<seed>
.parquet (experiments/run_trajectories.py), and writes two files for viewer/src/replay.js:
  <out>.jsonl  contract-valid server messages (B6): one `pred` (decision "preview": the proposer
               saw predictions only, never a gate or a coin) per frame, then the trajectory's best
               design (lowest predicted peak) as a `pred` followed by its certified verification
               stream (`iterate` ... `verified`, server/iterate_stream.py).
  <out>.json   what the panel draws: the whole trajectory's predicted peaks, the chain's current
               and best peak, each frame's step (P1 / P2: kept or rejected, acceptance
               probability, P2 temperature; P3: round), the board records of the frames, and,
               with GATE_INPUTS/labels.parquet, the oracle (true) peak of each frame (N3.2).

How each proposer's chain is recovered (nothing here is re-randomised or re-tuned):
  P1, P2  the proposer is rebuilt as the run built it (run_trajectories.make_proposer, same seed
          stream, same config) and fed the recorded predictions in order; every design it proposes
          must equal the recorded one exactly, otherwise this stops (the config or the code differs
          from the run's). Its accept / reject decisions and current design are read from it.
  P3      the rows of round r are the `starts` designs in start order; the start with the lowest
          final predicted peak is shown, the other starts are drawn as dots.
Frames are spread evenly over the run (P1 / P2: proposals; P3: rounds of the shown start), plus the
start and the best design. Fields are predicted again by the same ensemble at --n (the recorded
peak_hat and the new prediction may differ by GPU rounding; the largest difference is stored as
peak_check_max_abs_K). Display data for the demo, not a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("P1", "P2", "P3")
INFO = {
    "P1": "P1 tilted MCMC (Metropolis on p0(x)·exp(−λ·T̂_peak)): random moves of all 36 design "
          "numbers; a cooler design is always kept, a hotter one with probability exp(−λ·ΔT). It "
          "samples good designs and keeps wandering.",
    "P2": "P2 simulated annealing: the same random moves; a hotter design is kept with "
          "probability exp(−ΔT/T), and T cools geometrically, so it explores early and settles.",
    "P3": "P3 batched gradient descent: 64 starts, each following the gradient of the predicted "
          "peak through the FNO (autograd), normalised steps η. Smooth and nearly monotone; starts "
          "end in different local minima.",
}


class ReplayError(RuntimeError):
    pass


# ------------------------------------------------------------------------------ chain recovery


def walker_chain(rows: Sequence[dict[str, Any]], proposer) -> list[dict[str, Any]]:
    """P1 / P2: feed the recorded predictions to a freshly built proposer; per row: kind
    (start | accepted | rejected), prob, T (P2), current and best peak after the row (K)."""
    from gate import proposers as P

    hist = P.History()
    out: list[dict[str, Any]] = []
    best = math.inf
    for k, row in enumerate(rows):
        xs = proposer.next(hist)
        x = np.asarray(row["design_vector"], dtype=np.float64)
        if len(xs) != 1 or not np.array_equal(np.asarray(xs[0], dtype=np.float64), x):
            raise ReplayError(f"row t = {row['t']}: the rebuilt {proposer.name} proposes a "
                              "different design than the run recorded (config, seed or code)")
        T = proposer.temperature(proposer.iterations - 1) if proposer.name == "P2" else None
        hist.append(P.HistoryEntry(
            t=len(hist), design_id=row["design_id"], x=tuple(float(v) for v in x),
            x_hash=row["x_hash"], peak_hat=float(row["peak_hat"]), r_hot=float(row["r_hot"]),
            sd_hot=float(row["sd_hot"]), r_hot_raw=float(row["r_hot_raw"]),
            ood_score=float(row["ood_score"]), group=str(row["group"]), decision=None, B=None))
        out.append({"T": T})
        if k > 0:  # the proposer decided on row k - 1 when asked for row k
            _settle(out[k - 1], proposer)
        best = min(best, float(row["peak_hat"]))
        out[k]["best"] = best
    if rows:
        if proposer.next(hist):  # observes the last row
            raise ReplayError(f"the rebuilt {proposer.name} proposes more than the "
                              f"{len(rows)} recorded rows")
        _settle(out[-1], proposer)
    out[0]["kind"], out[0]["prob"] = "start", None
    return out


def _settle(entry: dict[str, Any], proposer) -> None:
    if proposer.last_accepted is None:  # the first row: no decision
        entry["kind"], entry["prob"] = "start", None
    else:
        entry["kind"] = "accepted" if proposer.last_accepted else "rejected"
        entry["prob"] = float(proposer.last_accept_prob)
    entry["current"] = float(proposer.current_peak)


def p3_chain(rows: Sequence[dict[str, Any]], starts: int) -> tuple[int, list[dict[str, Any]]]:
    """P3: (shown start, per-row info). Rows of round r are the starts in order."""
    if starts < 1 or len(rows) % starts:
        raise ReplayError(f"{len(rows)} rows are not whole rounds of {starts} starts")
    rounds = len(rows) // starts
    for k, row in enumerate(rows):
        if int(row["round"]) != k // starts:
            raise ReplayError(f"row t = {row['t']} has round {row['round']}, "
                              f"expected {k // starts}")
    final = [float(rows[(rounds - 1) * starts + i]["peak_hat"]) for i in range(starts)]
    shown = int(np.argmin(final))
    out, best, current = [], math.inf, None
    for k, row in enumerate(rows):
        best = min(best, float(row["peak_hat"]))
        if k % starts == shown:
            current = float(row["peak_hat"])
        out.append({"kind": ("start" if k < starts else "step") if k % starts == shown else "probe",
                    "prob": None, "T": None, "current": current, "best": best})
    return shown, out


def pick_frames(candidates: Sequence[int], frames: int, must: Sequence[int]) -> list[int]:
    """`frames` row indices spread evenly over candidates (in order), plus `must`, sorted."""
    if frames < 2:
        raise ValueError(f"frames must be >= 2, got {frames}")
    c = list(candidates)
    pos = np.unique(np.round(np.linspace(0, len(c) - 1, min(frames, len(c)))).astype(int))
    return sorted({c[i] for i in pos} | set(must))


# ------------------------------------------------------------------------------------ writing


def build_replay(rows: Sequence[dict[str, Any]], chain: Sequence[dict[str, Any]], frame_rows:
                 Sequence[int], *, name: str, level: int, seed: int, predict: Callable,
                 oracle: dict[str, float] | None, verify: Callable | None, source: dict[str, Any],
                 settings: dict[str, Any], x_of: Callable[[int], float], x_label: str
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(messages, meta). predict(records) -> [(mu, sd)]; verify(record, mu, msg_id) -> messages."""
    from contracts.f16 import encode
    from data.sampler import design_board

    tag = f"replay-{name}-a{level}-s{seed}"
    best_k = int(np.argmin([float(r["peak_hat"]) for r in rows]))
    frame_rows = sorted(set(frame_rows) | {best_k})
    records = {k: design_board(np.asarray(rows[k]["design_vector"], dtype=np.float64))
               for k in frame_rows}
    fields = predict([records[k] for k in frame_rows])
    msgs, frames, recs, check = [], [], {}, 0.0

    def pred(mid: str, mu: np.ndarray, sd: np.ndarray) -> dict[str, Any]:
        return {"schema_version": "1.0", "type": "pred", "id": mid, "mu": encode(mu),
                "sd": encode(sd), "B": None, "decision": "preview", "group": None,
                "peak": float(mu.max()), "units": {"mu": "K", "sd": "K", "B": "K", "peak": "K"},
                "shape": list(mu.shape)}

    best_mu = None
    for k, (mu, sd) in zip(frame_rows, fields, strict=True):
        mu, sd = np.asarray(mu), np.asarray(sd)
        mid = f"{tag}-t{rows[k]['t']}"
        check = max(check, abs(float(mu.max()) - float(rows[k]["peak_hat"])))
        msgs.append(pred(mid, mu, sd))
        recs[mid] = records[k]
        c = chain[k]
        frames.append({"id": mid, "x": x_of(k), "t": int(rows[k]["t"]),
                       "peak_hat": float(rows[k]["peak_hat"]), "kind": c["kind"],
                       "prob": c["prob"], "T": c["T"], "current": c["current"], "best": c["best"],
                       "oracle_peak": None if oracle is None else oracle.get(rows[k]["x_hash"])})
        if k == best_k:
            best_mu = (mid, mu, sd)
    best = {"t": int(rows[best_k]["t"]), "x": x_of(best_k),
            "peak_hat": float(rows[best_k]["peak_hat"]),
            "oracle_peak": None if oracle is None else oracle.get(rows[best_k]["x_hash"])}
    if verify is not None:
        mid = f"{tag}-best-t{rows[best_k]['t']}"
        msgs.append(pred(mid, best_mu[1], best_mu[2]))
        recs[mid] = records[best_k]
        msgs += list(verify(records[best_k], best_mu[1], mid))
        best["id"] = mid
    else:
        best["id"] = best_mu[0]
    trace = {
        "points": [[x_of(k), float(r["peak_hat"]), chain[k]["kind"]] for k, r in enumerate(rows)],
        "current": [[x_of(k), c["current"]] for k, c in enumerate(chain)
                    if c["current"] is not None],
        "best": [[x_of(k), c["best"]] for k, c in enumerate(chain)],
        "oracle": [[f["x"], f["oracle_peak"]] for f in frames if f["oracle_peak"] is not None],
    }
    meta = {"schema": "twinheat-replay/1", "proposer": name, "level": level, "seed": seed,
            "info": INFO[name], "settings": settings, "source": source, "x_label": x_label,
            "x_max": x_of(len(rows) - 1), "trace": trace, "frames": frames, "best": best,
            "records": recs, "peak_check_max_abs_K": check}
    return msgs, meta


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = sorted(table.to_pylist(), key=lambda r: r["t"])
    if [r["t"] for r in rows] != list(range(len(rows))):
        raise ReplayError(f"{path}: t is not 0..{len(rows) - 1}")
    md = (table.schema.metadata or {}).get(b"twinheat")
    if md is None:
        raise ReplayError(f"{path}: no twinheat settings in the parquet metadata")
    return rows, json.loads(md)


def _oracle(path: Path) -> dict[str, float]:
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["x_hash", "oracle_peak"]).to_pydict()
    return {h: float(v) for h, v in zip(t["x_hash"], t["oracle_peak"], strict=True)}


def main(argv: list[str] | None = None) -> None:
    from contracts.validate import validate
    from experiments import run_trajectories as RT
    from twinheat import spec as specmod

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True, help="the gate-experiment inputs folder")
    ap.add_argument("--proposer", choices=NAMES, required=True)
    ap.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", choices=("ensemble", "stub"), default="ensemble")
    ap.add_argument("--tau-v", type=float,
                    help="K; with --every and --kappa-iters: verify the best design")
    ap.add_argument("--every", type=int)
    ap.add_argument("--kappa-iters", type=int)
    ap.add_argument("--config", type=Path, default=RT.DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, required=True, help="path prefix: writes .jsonl and .json")
    a = ap.parse_args(argv)
    vargs = (a.tau_v, a.every, a.kappa_iters)
    if any(v is None for v in vargs) and any(v is not None for v in vargs):
        ap.error("--tau-v, --every and --kappa-iters go together")

    path = a.inputs / "trajectories" / f"traj_{a.proposer}_a{a.level}_s{a.seed}.parquet"
    rows, settings = _read_rows(path)
    cfg = RT.load_config(a.config)
    if settings.get("config") != cfg[a.proposer]:
        raise ReplayError(f"{path} was made with {a.proposer} settings {settings.get('config')}, "
                          f"{a.config} now has {cfg[a.proposer]}")
    if a.proposer == "P3":
        starts = int(cfg["P3"]["starts"])
        shown, chain = p3_chain(rows, starts)
        cand = [k for k in range(len(rows)) if k % starts == shown]
        x_of, x_label = (lambda k: k // starts), "round (64 starts each)"
    else:
        prop = RT.make_proposer(cfg, a.proposer, a.level, a.seed)
        chain = walker_chain(rows, prop)
        cand = list(range(len(rows)))
        x_of, x_label = (lambda k: k), "proposal"
    frame_rows = pick_frames(cand, a.frames, [cand[0]])

    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    from server.iterate_stream import _predictor, verification_messages
    one = _predictor(a.model, spec, a.n, a.device)

    def predict(records):
        out = []
        for i, r in enumerate(records):
            out.append(one(r))
            print(f"  predicted {i + 1}/{len(records)}", end="\r", flush=True)
        print()
        return out

    verify = None
    if a.tau_v is not None:
        def verify(record, mu, mid):
            return verification_messages(record, mu, spec, a.n, msg_id=mid, tau_v=a.tau_v,
                                         every=a.every, kappa_iters=a.kappa_iters)
    labels = a.inputs / "labels.parquet"
    oracle = _oracle(labels) if labels.exists() else None
    source = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "commit": settings.get("commit"), "rows": len(rows),
              "labels": labels.name if oracle is not None else None, "model": a.model, "n": a.n}
    msgs, meta = build_replay(rows, chain, frame_rows, name=a.proposer, level=a.level,
                              seed=a.seed, predict=predict, oracle=oracle, verify=verify,
                              source=source, settings=settings["config"], x_of=x_of,
                              x_label=x_label)
    for m in msgs:
        validate(m, m["type"])  # B6
    a.out.parent.mkdir(parents=True, exist_ok=True)
    jl = a.out.with_name(a.out.name + ".jsonl")
    js = a.out.with_name(a.out.name + ".json")
    jl.write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")
    js.write_text(json.dumps(meta), encoding="utf-8")
    last = msgs[-1]
    print(f"{path.name}: {len(rows)} proposals, {len(meta['frames'])} frames, best t = "
          f"{meta['best']['t']} at {meta['best']['peak_hat']:.2f} K predicted"
          + (f" ({meta['best']['oracle_peak']:.2f} K oracle)" if meta["best"]["oracle_peak"]
             is not None else "")
          + (f"; verified: {last['stop_reason']}, cert {last['cert']} K" if last["type"] ==
             "verified" else "")
          + f"; recorded vs new peak differ by <= {meta['peak_check_max_abs_K']:.2e} K")
    print(f"wrote {jl} and {js}")


if __name__ == "__main__":
    main()
