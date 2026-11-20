"""E8: efficiency at equal exact-solve budget, feedback runs of G1 vs G5 (H2; CLAUDE.md §11.3 M3,
§11.4, §11.5). Runs on the GPU server (it calls the ensemble and the exact solver online).

    python -m experiments.e8_efficiency --inputs <dir> --device cuda:0 [--out ...]

Feedback (decision 2026-11-18): a proposer runs live against the gate. When a design is verified
(flagged, or audited after commit), the proposer's history shows its exact peak in place of the
surrogate's T̂; otherwise it shows T̂. The coin itself never reaches the proposer, but a changed
peak in the history reveals that a past accepted design was audited. Past audit outcomes may be
shown (CLAUDE.md §18.1); the current one never is, because the history entry is written after the
decision. P3's gradient comes from the model, not the history, so feedback changes only what P3
gets verified.

Budget: for each (proposer, level, seed), G5 runs first at the headline p; G1 then runs with
f = G5's realised solves / proposals (gate/baselines.flag_rate_for_budget), and both runs'
realised solve counts are reported. M3 = best verified peak against cumulative exact solves. H2
(non-inferiority, paired over seeds): d_s = final best verified peak (G5) − (G1); H2 holds for a
(proposer, level) when the one-sided 95 % bootstrap upper bound of mean d is below 0.2 K.
Exact peaks come from the N3.2 labels when the design is already labelled, else from a 1e-8
PyAMG solve (solvers/amg_baseline.amg_solve), cached by x_hash.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from experiments import figs_gates as F
from experiments import run_gates as RG
from gate import proposers as P
from gate.base import AuditRecord, GateInput

DEFAULT_OUT = RG.ROOT / "experiments" / "results" / "e8"


def feedback_run(proposer: P.Proposer, evaluate, grouper, gate, coin,
                 exact_peak: Callable[[np.ndarray, str], float], fz: dict[str, Any], *,
                 max_proposals: int = 10**9) -> dict[str, Any]:
    """One live run (§8.1 steps 1–6 with verified peaks fed back). Returns the M3 curve."""
    history = P.History()
    solves, best = 0, math.inf
    curve_solves, curve_best = [0], [math.nan]
    n_flag = n_audit = 0
    while not proposer.done and len(history) < max_proposals:
        xs = proposer.next(history)
        if not xs:
            break
        for p in evaluate(xs):
            t = len(history)
            did = f"t{t}"
            group = grouper(p, proposer)
            s = float(RG.scale(np.array([p.r_hot]), np.array([p.sd_hot]), fz)[0])
            gi = GateInput(design_id=did, x_hash=p.x_hash, peak_hat=p.peak_hat, s=s,
                           r_hot=p.r_hot, sd_hot=p.sd_hot, proposer=proposer.name,
                           aggressiveness=proposer.level, group=group, r_hot_raw=p.r_hot_raw)
            coin.commit(did)
            commit = getattr(gate, "commit", None)
            dec = commit(gi) if commit is not None else gate.decide(gi)
            audited = coin.draw(did, dec.p_audit)
            flagged = dec.decision == "flag"
            seen = p.peak_hat
            if flagged or audited:
                true = float(exact_peak(p.x, p.x_hash))
                solves += 1
                best = min(best, true)
                curve_solves.append(solves)
                curve_best.append(best)
                seen = true
                n_flag += int(flagged)
                n_audit += int(audited)
                if audited:
                    gate.update([AuditRecord(did, True, p.peak_hat, s, true, true, 0)])
                elif gate.gate_id == "G4":
                    gate.update([AuditRecord(did, False, p.peak_hat, s, true, true, 0)])
            history.append(P.HistoryEntry(
                t=t, design_id=did, x=tuple(float(v) for v in p.x), x_hash=p.x_hash,
                peak_hat=seen, r_hot=p.r_hot, sd_hot=p.sd_hot, r_hot_raw=p.r_hot_raw,
                ood_score=p.ood_score, group=group, decision=dec.decision, B=dec.B))
    return {"T": len(history), "solves": solves, "flagged": n_flag, "audited": n_audit,
            "final_best": best, "curve_solves": curve_solves, "curve_best": curve_best}


def noninferiority(d: list[float], *, margin: float, resamples: int, seed: int) -> dict[str, Any]:
    """One-sided paired test: mean of d = (G5 − G1) final verified peaks, upper 95 % bootstrap
    bound below `margin` (K) -> non-inferior."""
    v = np.asarray(d, dtype=np.float64)
    if v.size < 2 or not np.isfinite(v).all():
        raise ValueError(f"need >= 2 finite paired differences, got {d}")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = v[rng.integers(0, v.size, size=(resamples, v.size))].mean(axis=1)
    upper = float(np.percentile(means, 95))
    return {"mean_d_K": float(v.mean()), "upper95_K": upper, "margin_K": margin,
            "noninferior": bool(upper < margin), "n_pairs": int(v.size)}


def verdict_h2(tests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = bool(tests) and all(t["noninferior"] for t in tests.values())
    return {"hypothesis": "H2", "verdict": "SUPPORTED" if ok else "NOT SUPPORTED",
            "parts": tests, "note": f"non-inferiority margin {RG.H2_MARGIN_K} K, paired over seeds"}


def run_pairs(make_proposer, evaluate, grouper, exact_peak, fz: dict[str, Any], cal: RG.Calibration,
              cells: list[tuple[str, int]], seeds: list[int]) -> tuple[list, list, dict]:
    """G5 then G1 at G5's budget for every (proposer, level) cell and seed."""
    from gate.coin import AuditCoin

    runs, curves, tests = [], [], {}
    dummy = RG.Trajectory("e8", "P1", 1, 0, None, [], np.zeros((0, P.DIM)), *[np.zeros(0)] * 4,
                          [])
    for name, level in cells:
        diffs = []
        for seed in seeds:
            out = {}
            for gate_name in ("G5", "G1"):
                coin = AuditCoin(seed=seed)
                f = None
                if gate_name == "G1":
                    g5 = out["G5"]
                    if not 0 < g5["solves"] < g5["T"]:
                        raise RG.InputError(
                            f"E8 {name} level {level} seed {seed}: G5 verified {g5['solves']} of "
                            f"{g5['T']} proposals, so G1 cannot be run at the same budget")
                    f = RG.flag_rate_for_budget(int(g5["solves"]), int(g5["T"]))
                gate = RG.build_gate(gate_name, fz, cal, dummy, coin, seed, p=fz["p_audit"],
                                     p_min=fz["p_min"], f=f, ratio_traj=None)
                r = feedback_run(make_proposer(name, level, seed), evaluate, grouper, gate, coin,
                                 exact_peak, fz)
                out[gate_name] = r
                runs.append({"gate": gate_name, "proposer": name, "level": level, "seed": seed,
                             "f": f, **{k: r[k] for k in ("T", "solves", "flagged", "audited",
                                                          "final_best")}})
                curves.append({"gate": gate_name, "proposer": name, "level": level, "seed": seed,
                               "solves": r["curve_solves"], "best": r["curve_best"]})
            for g in ("G5", "G1"):
                if not math.isfinite(out[g]["final_best"]):
                    raise RG.InputError(
                        f"E8 {name} level {level} seed {seed}: {g} verified no design in "
                        f"{out[g]['T']} proposals (G1 accepts its first window of "
                        f"{fz['g1']['window']} proposals), so it has no final verified peak")
            diffs.append(out["G5"]["final_best"] - out["G1"]["final_best"])
        tests[f"{name}_a{level}"] = noninferiority(
            diffs, margin=RG.H2_MARGIN_K, resamples=int(fz["bootstrap"]["resamples"]),
            seed=int(fz["bootstrap"]["seed"]))
    return runs, curves, tests


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - GPU host only
    from data import sampler
    from data.raster import rasterise
    from experiments import run_trajectories as RT
    from gate import groups as G
    from solvers.amg_baseline import amg_solve
    from solvers.fvm import assemble_board

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True, help="labels + calibration (run_gates)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frozen", type=Path, default=RG.DEFAULT_FROZEN)
    ap.add_argument("--traj-config", type=Path, default=RT.DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    commit = RG.code_commit()
    fz = RG.check_frozen(a.frozen)
    fz["_run_commit"] = commit
    labels = RG.Labels.from_parquet(a.inputs / "labels.parquet")
    cal = RG.Calibration.from_parquet(a.inputs / "calibration.parquet", labels)
    tcfg = RT.load_config(a.traj_config)
    ens, ood, cuts, oracle = RT._load(tcfg, a.device)
    evaluate = P.SurrogateEvaluator(ens, ood, RT.SPEC, n=int(tcfg["N"]), m=int(tcfg["m_hot"]),
                                    batch=int(tcfg["eval_batch"]))
    cache = dict(labels.oracle)

    def exact_peak(x: np.ndarray, h: str) -> float:
        if h not in cache:
            rec = sampler.design_board(x)
            A, b = assemble_board(RT.SPEC, rasterise(rec, RT.SPEC, int(tcfg["N"])),
                                  rec["dirichlet_edges"])
            cache[h] = float(amg_solve(A, b, tol=1e-8).theta.max())
        return cache[h]

    def make(name, level, seed):
        return RT.make_proposer(tcfg, name, level, seed, oracle)

    cells = [(n, lv) for n in RT.NAMES for lv in RT.LEVELS]
    runs, curves, tests = run_pairs(make, evaluate, G.make_grouper(cuts), exact_peak, fz, cal,
                                    cells, list(fz["seeds"]))
    v = verdict_h2(tests)
    a.out.mkdir(parents=True, exist_ok=True)
    RG.write_csv(runs, a.out / "e8_runs.csv")
    RG.write_csv([{"cell": k, **t} for k, t in tests.items()], a.out / "e8_noninferiority.csv")
    digest = RG._sha(a.inputs / "labels.parquet")
    F.best_verified_curves(curves, RG.footer(fz, digest), a.out / "e8_m3.png",
                           title="E8: best verified peak vs exact solves (G1 at G5's budget)")
    RG.print_verdicts([v])


if __name__ == "__main__":
    main()
