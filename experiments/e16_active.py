"""E16: risk-adaptive audits, p_min sweep at equal audit budget (H9, exploratory; CLAUDE.md §8.5,
§11.4). G7 runs at mean audit rate p̄ = each p of the sweep (the same budget as G5 at that p) and
every p_min <= p̄; G5 at every p. H9: the cheapest G7 setting that meets H1's criterion uses
>= 30 % fewer audits than the cheapest G5 setting that meets it (reading of §11.4, M to confirm).

    python -m experiments.e16_active --inputs <dir> [--out experiments/results/e16]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import figs_gates as F
from experiments import run_gates as RG

DEFAULT_OUT = RG.ROOT / "experiments" / "results" / "e16"


def configs(fz: dict) -> list[dict]:
    return ([{"gate": "G5", "p": p} for p in fz["p_sweep"]]
            + [{"gate": "G7", "p": p, "p_min": m} for p in fz["p_sweep"]
               for m in fz["p_min_sweep"] if m <= p])


def analyse(fz: dict, data: RG.Inputs, out: Path) -> list[dict]:
    summaries, groups, _ = RG.run_matrix(fz, data, configs(fz), tag="e16")
    v = RG.verdict_h9(summaries)
    RG.write_outputs(out, "e16", summaries, groups, [v], fz, data,
                     ("gate", "p", "p_min", "proposer", "level"))
    g7 = [r for r in summaries if r["gate"] == "G7"]
    F.sweep(g7, "p_min", ("M1_all", "audited"), RG.footer(fz, data.digest),
            out / "e16_p_min.png", title="E16: G7, M1 and audits vs p_min (one line per p̄)",
            group="p")
    RG.print_verdicts([v])
    return [v]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frozen", type=Path, default=RG.DEFAULT_FROZEN)
    a = ap.parse_args(argv)
    fz, data = RG.prepare(a.frozen, a.inputs)
    analyse(fz, data, a.out)


if __name__ == "__main__":
    main()
