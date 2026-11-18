"""E9: H1 replicated on the Darcy task (H3; CLAUDE.md §11.4, §11.6).

    python -m experiments.e9_darcy --inputs <darcy dir> [--out experiments/results/e9]

The same code path as E7 (experiments/e7_validity.analyse) on Darcy inputs in the same formats
(docs/gate_experiment_inputs.md): trajectories over the Darcy design space, oracle labels of
peak u, calibration designs. Verdict H3 uses H1's criterion. NOT RUNNABLE YET: the Darcy design
task, its proposers and its labels do not exist (CLAUDE.md §11.6; cut list item "Darcy
validity-only", §15).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import e7_validity as E7
from experiments import run_gates as RG

DEFAULT_OUT = RG.ROOT / "experiments" / "results" / "e9"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frozen", type=Path, default=RG.DEFAULT_FROZEN)
    a = ap.parse_args(argv)
    fz, data = RG.prepare(a.frozen, a.inputs)
    E7.analyse(fz, data, a.out, name="e9", hypothesis="H3")


if __name__ == "__main__":
    main()
