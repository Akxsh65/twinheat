"""E15: per-group validity, G6 vs G5 (H8; CLAUDE.md §8.5, §11.4). Empirical: no per-group bound
is proved (report/lemma1_review.md §6), so the per-group target α + slack_g (Lemma 1's form with
the group's step count) is labelled an empirical target everywhere.

    python -m experiments.e15_groups --inputs <dir> [--out experiments/results/e15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import figs_gates as F
from experiments import run_gates as RG

DEFAULT_OUT = RG.ROOT / "experiments" / "results" / "e15"


def analyse(fz: dict, data: RG.Inputs, out: Path) -> list[dict]:
    p = fz["p_audit"]
    summaries, groups, _ = RG.run_matrix(fz, data, [{"gate": "G5", "p": p},
                                                    {"gate": "G6", "p": p}], tag="e15")
    v = RG.verdict_h8(groups)
    RG.write_outputs(out, "e15", summaries, groups, [v], fz, data, ("gate", "proposer", "level"))
    F.group_bars(groups, RG.footer(fz, data.digest), out / "e15_groups.png",
                 title="E15: per-group accepted large-error rate M1g, G5 vs G6 (empirical target)")
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
