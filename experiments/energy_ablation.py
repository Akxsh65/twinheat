"""Energy-form loss ablation at 0 / 100 / 1k labels (M2.3; CLAUDE.md §12, one ablation row).

    python -m experiments.energy_ablation --config experiments/configs/e5.yaml --data ROOT \
        --ckpt-root checkpoints/energy --backup-root BACKUP [--device cuda]

For each run in the config's energy_ablation.runs: train one FNO (models/train_ensemble.py) at the
config's stage where only the first `labels` train boards carry the data loss, plus, if
`energy: true`, the energy-form loss ½θᵀAθ − bᵀθ (models/energy.py, matrix-free from
solvers/fvm.coefficients) on every train board, weighted by energy_weight. Then evaluate each run
on the ID test split at the stage's N with the E5 metrics. Writes energy_ablation.csv. The
energy-form loss is an ablation, not a claim (CLAUDE.md §1, §12). seed, epochs and energy_weight
are null in the committed config until M sets them; the script stops and names them.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.e5_accuracy import BOARD_MM, ROOT, E5Error, evaluate, load_config
from models import train_ensemble as te
from models.dataset import TwinHeatDataset
from models.ensemble import Ensemble

__all__ = ["main", "run"]


def run(cfg: dict[str, Any], data: Path, ckpt_root: Path, backup_root: Path, device: str,
        registry: Path, extra: list[str] | None = None) -> list[dict[str, Any]]:
    ab = cfg["energy_ablation"]
    missing = [k for k in ("seed", "epochs", "stage", "member") if ab.get(k) is None]
    if any(r["energy"] for r in ab["runs"]) and ab.get("energy_weight") is None:
        missing.append("energy_weight")
    if missing:
        raise E5Error(f"energy_ablation: set {', '.join(missing)} in the config first")
    norm = ROOT / cfg["norm_stats"] if not Path(cfg["norm_stats"]).is_absolute() \
        else Path(cfg["norm_stats"])
    rows = []
    for r in ab["runs"]:
        variant = f"energy_L{r['labels']}" if r["energy"] else f"data_L{r['labels']}"
        argv = ["--member", str(ab["member"]), "--seed", str(ab["seed"]), "--stage", ab["stage"],
                "--epochs", str(ab["epochs"]), "--data", str(data),
                "--ckpt_dir", str(ckpt_root / variant), "--backup_dir", str(backup_root / variant),
                "--norm-stats", str(norm), "--registry", str(registry), "--device", device,
                "--labels", str(r["labels"]), "--variant", variant,
                "--energy-weight", str(ab["energy_weight"] if r["energy"] else 0.0), *(extra or [])]
        cfg_run = te.parse(argv)
        te.train_member(cfg_run)
        ens = Ensemble.load(registry, norm, stage=ab["stage"], device=device, variant=variant)
        m = evaluate(ens, TwinHeatDataset(data, "test", N=cfg_run.N, allow_eval=True), cfg["batch"])
        rows.append({"variant": variant, "labels": r["labels"], "energy": r["energy"],
                     "N": cfg_run.N, "rel_l2_mean": float(np.mean(m["rel_l2"]["all"])),
                     "hotspot_l1_K_mean": float(np.mean(m["hotspot_l1_K"]["all"])),
                     "peak_error_K_mean": float(np.mean(m["peak_error_K"]["all"])),
                     "peak_dist_mm_mean": float(np.mean(m["peak_dist_mm"]["all"])),
                     "n_test": len(m["rel_l2"]["all"]), "board_mm": BOARD_MM})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e5.yaml")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--backup-root", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=None,
                    help="default: <ckpt-root>/registry_energy.json (apart from the ensemble)")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "e5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-dirty", action="store_true", help="tests only")
    a, extra = ap.parse_known_args(argv)
    cfg = load_config(a.config, a.allow_dirty)
    registry = a.registry or a.ckpt_root / "registry_energy.json"
    rows = run(cfg, a.data, a.ckpt_root, a.backup_root, a.device, registry, extra)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "energy_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
