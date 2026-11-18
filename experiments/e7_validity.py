"""E7: validity of every gate, G0–G7 × P1–P4 × 3 levels × 5 seeds (H1; CLAUDE.md §11.4–§11.5).

    python -m experiments.e7_validity --inputs <dir> [--out experiments/results/e7] [--logs]

Replays each gate on the same oracle-labelled trajectories (experiments/run_gates.py). G5 runs at
every p of the sweep; the other audited gates (G6, G7) at the headline p; G1 at G5's solve budget
at the headline p. Writes e7_runs.csv, e7_groups.csv, e7_ci.csv (mean and bootstrap 95 % CI over
seeds), e7_verdicts.json and e7_m1.png, and prints the H1 verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import figs_gates as F
from experiments import run_gates as RG

DEFAULT_OUT = RG.ROOT / "experiments" / "results" / "e7"


def configs(fz: dict) -> list[dict]:
    p = fz["p_audit"]
    return ([{"gate": "G5", "p": q} for q in fz["p_sweep"]]
            + [{"gate": g} for g in ("G0", "G2", "G3", "G4")]
            + [{"gate": "G1", "p": p}, {"gate": "G6", "p": p},
               {"gate": "G7", "p": p, "p_min": fz["p_min"]}])


def analyse(fz: dict, data: RG.Inputs, out: Path, *, name: str = "e7", hypothesis: str = "H1",
            logs: bool = False) -> list[dict]:
    summaries, groups, rows = RG.run_matrix(fz, data, configs(fz), tag=name)
    headline = [r for r in summaries if r["gate"] != "G5" or r["p"] == fz["p_audit"]]
    v = RG.verdict_h1(headline)
    v["hypothesis"] = hypothesis
    RG.write_outputs(out, name, summaries, groups, [v], fz, data,
                     ("gate", "p", "proposer", "level"), rows if logs else None)
    ci = RG.ci_table(headline, ("gate", "proposer", "level"), "M1_all", fz)
    g5 = [r for r in headline if r["gate"] == "G5"]
    levels = sorted({r["level"] for r in g5 if r["level"] is not None})
    bounds = {lv: sum(r["bound"] for r in g5 if r["level"] == lv)
              / sum(1 for r in g5 if r["level"] == lv) for lv in levels}
    F.m1_vs_level(ci, bounds, RG.footer(fz, data.digest), out / f"{name}_m1.png",
                  title=f"{name.upper()}: accepted large-error rate M1 by gate "
                        "(95 % CI over seeds)")
    RG.print_verdicts([v])
    return [v]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frozen", type=Path, default=RG.DEFAULT_FROZEN)
    ap.add_argument("--logs", action="store_true", help="also write every §10.4 row")
    a = ap.parse_args(argv)
    fz, data = RG.prepare(a.frozen, a.inputs)
    analyse(fz, data, a.out, logs=a.logs)


if __name__ == "__main__":
    main()
