"""E14: certified audits, ε_S sweep (H7; CLAUDE.md §8.3, §11.4).

    python -m experiments.e14_cert_audit --inputs <dir> [--out experiments/results/e14]

G5 at the headline p with audits solved to 1e-8 ("exact") and with certified, early-stopped
audits replayed from the N3.3 traces at every ε_S of the sweep. Every certified interval is
checked against the oracle (B7: any violation raises and blocks results). H7 is judged at the
frozen ε_S: M1 within the Lemma 1 bound in >= 95 % of runs and >= 50 % fewer audit iterations
than the 1e-8 audits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import figs_gates as F
from experiments import run_gates as RG

DEFAULT_OUT = RG.ROOT / "experiments" / "results" / "e14"


def configs(fz: dict) -> list[dict]:
    p = fz["p_audit"]
    return ([{"gate": "G5", "p": p, "audit": "exact"}]
            + [{"gate": "G5", "p": p, "audit": "certified", "eps_S": e}
               for e in fz["eps_S_sweep"]])


def analyse(fz: dict, data: RG.Inputs, out: Path) -> list[dict]:
    summaries, groups, _ = RG.run_matrix(fz, data, configs(fz), tag="e14")
    cert = [r for r in summaries if r["audit"] == "certified"]
    v = RG.verdict_h7(cert, eps_S=fz["eps_S"])
    RG.write_outputs(out, "e14", summaries, groups, [v], fz, data,
                     ("audit", "eps_S", "proposer", "level"))
    for r in cert:
        r["iter_fraction"] = r["audit_iters"] / r["audit_iters_full"] if r["audit_iters_full"] \
            else float("nan")
    F.sweep(cert, "eps_S", ("M1_all", "iter_fraction"), RG.footer(fz, data.digest),
            out / "e14_eps_S.png", title="E14: certified audits, M1 and audit iterations "
            "(fraction of 1e-8 audits) vs ε_S", group="proposer")
    RG.print_verdicts([v])
    return [v]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frozen", type=Path, default=RG.DEFAULT_FROZEN)
    a = ap.parse_args(argv)
    fz, data = RG.prepare(a.frozen, a.inputs, traces=True)
    analyse(fz, data, a.out)


if __name__ == "__main__":
    main()
