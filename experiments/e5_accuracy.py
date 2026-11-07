"""E5: surrogate accuracy on ID test and every OOD set, and the OOD score (M2.3; §11.5, §12, §6.7).

    python -m experiments.e5_accuracy --config experiments/configs/e5.yaml --data DATASET_V1_ROOT \
        [--out experiments/results/e5] [--device cuda]

Every number here comes from this script, run on registered checkpoints with a committed config
(CLAUDE.md §0 rule 11, §4); the config is refused if it has uncommitted changes. Reading test and
OOD splits is allowed only here, in experiments/ (allow_eval=True, §5); nothing is tuned on them.

Models (config `models`, plus one row per ensemble member): predictions in K from
models/ensemble.py (ensemble = mean of the members; a single model is an ensemble of one).
Metrics per board (models/metrics.py), overall and per layer: relative L2, peak-temperature error
(K), hotspot error over the 64 hottest true cells (K), hotspot location error (mm); each reported
as mean and P95 over the split. The 128-only model is evaluated zero-shot at the config N.
OOD score (models/ood_score.py): fitted on train features, threshold = 95th percentile on val,
then the flag rate per split (recall on OOD sets, false-positive rate on ID test).

Outputs in --out: e5_metrics.csv (tidy: model, split, metric, layer, mean, p95, n), e5_ood.csv,
e5_tables.tex, e5_figure.png (footer: commit, dataset hash, config seed, hardware, date),
go_no_go.md (the §12 week-5 decision with its numbers) and run_info.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from models.dataset import TwinHeatDataset
from models.ensemble import Ensemble
from models.metrics import LAYERS, METRICS, board_metrics
from models.normalise import dataset_hash
from models.ood_score import OODScore, pooled_features

ROOT = Path(__file__).resolve().parents[1]
BOARD_MM = 100.0  # board side (CLAUDE.md §7.1, spec_v2 L = 0.100 m)
REQUIRED = ("registry", "norm_stats", "N", "batch", "splits", "models", "go_no_go", "ood")


class E5Error(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def load_config(path: Path, allow_dirty: bool) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if cfg.get(k) is None]
    if missing:
        raise E5Error(f"{path}: set {', '.join(missing)}")
    if not allow_dirty:
        try:
            rel = Path(path).resolve().relative_to(ROOT)
            dirty = _git("status", "--porcelain", "--", str(rel))
        except (ValueError, subprocess.CalledProcessError) as e:
            raise E5Error(f"{path}: must be a committed file inside the repo ({e})") from e
        if dirty:
            raise E5Error(f"{path} has uncommitted changes: commit the config before the run "
                          "(CLAUDE.md §4)")
    return cfg


# ---------------------------------------------------------------------------- evaluation

def evaluate(ens: Ensemble, ds: TwinHeatDataset, batch: int) -> dict[str, dict[str, np.ndarray]]:
    """Per-board metrics of an ensemble's mean prediction over a dataset (raw inputs, K targets)."""
    acc: dict[str, dict[str, list[np.ndarray]]] = {m: {} for m in METRICS}
    for i0 in range(0, len(ds), batch):
        xs, ys = zip(*(ds[i] for i in range(i0, min(i0 + batch, len(ds)))), strict=True)
        mu, _ = ens.predict(np.stack(xs))
        for metric, by_layer in board_metrics(mu, np.stack(ys), BOARD_MM).items():
            for layer, v in by_layer.items():
                acc[metric].setdefault(layer, []).append(v)
    return {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in acc.items()}


def _features(ens: Ensemble, ds: TwinHeatDataset, batch: int, max_boards: int | None
              ) -> list[np.ndarray]:
    n = len(ds) if max_boards is None else min(max_boards, len(ds))
    out: list[list[np.ndarray]] = [[] for _ in ens.models]
    for i0 in range(0, n, batch):
        x = np.stack([ds[i][0] for i in range(i0, min(i0 + batch, n))])
        xt = torch.from_numpy(ens.norm.normalise_x(x)).to(ens.device)
        for k, model in enumerate(ens.models):
            out[k].append(pooled_features(model, xt).cpu().numpy())
    return [np.concatenate(f) for f in out]


def _models(cfg: dict[str, Any], registry: Path, norm: Path, device: str) -> dict[str, Ensemble]:
    out: dict[str, Ensemble] = {}
    for label, sel in cfg["models"].items():
        out[label] = Ensemble.load(registry, norm, stage=sel["stage"], device=device,
                                   arch=sel["arch"], variant=sel["variant"])
    if cfg.get("each_member"):
        base = cfg["models"][cfg["go_no_go"]["model"]]
        for m in out[cfg["go_no_go"]["model"]].members:
            out[f"member{m['member']}"] = Ensemble.load(
                registry, norm, stage=base["stage"], device=device, arch=base["arch"],
                variant=base["variant"], members=[m["member"]])
    return out


# ---------------------------------------------------------------------------- outputs

def _summary(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(np.percentile(values, 95))


def _tex(s: str) -> str:
    return s.replace("_", r"\_")


def _write_tables(rows: list[dict[str, Any]], ood: list[dict[str, Any]], gate: dict[str, Any],
                  path: Path) -> None:
    def get(model: str, split: str, metric: str, stat: str = "mean") -> float:
        for r in rows:
            if (r["model"], r["split"], r["metric"], r["layer"]) == (model, split, metric, "all"):
                return r[stat]
        return float("nan")
    models = list(dict.fromkeys(r["model"] for r in rows))
    splits = list(dict.fromkeys(r["split"] for r in rows))
    id_split, main = gate["split"], gate["model"]
    lines = [r"% E5 (experiments/e5_accuracy.py); every number is computed by the script.",
             r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Model & rel.\ $L_2$ mean & rel.\ $L_2$ P95 & peak err.\ (K) & hotspot err.\ (K) "
             r"& peak loc.\ (mm) \\", r"\hline"]
    for m in models:
        lines.append(f"{_tex(m)} & {get(m, id_split, 'rel_l2'):.4f} & "
                     f"{get(m, id_split, 'rel_l2', 'p95'):.4f} & "
                     f"{get(m, id_split, 'peak_error_K'):.3f} & "
                     f"{get(m, id_split, 'hotspot_l1_K'):.3f} & "
                     f"{get(m, id_split, 'peak_dist_mm'):.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", "", r"\begin{tabular}{lrrrr}", r"\hline",
              f"Split ({_tex(main)}) & rel.\\ $L_2$ mean & hotspot err.\\ (K) & "
              r"peak loc.\ (mm) & OOD flag rate \\", r"\hline"]
    rate = {o["split"]: o["flag_rate"] for o in ood}
    for s in splits:
        lines.append(f"{_tex(s)} & {get(main, s, 'rel_l2'):.4f} & "
                     f"{get(main, s, 'hotspot_l1_K'):.3f} & {get(main, s, 'peak_dist_mm'):.2f} & "
                     f"{rate.get(s, float('nan')):.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(rows: list[dict[str, Any]], footer: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(r["model"] for r in rows))
    splits = list(dict.fromkeys(r["split"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in ((axes[0], "rel_l2", "relative L2 (mean)"),
                              (axes[1], "hotspot_l1_K", "hotspot error, K (mean)")):
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            v = [next((r["mean"] for r in rows if (r["model"], r["split"], r["metric"],
                                                   r["layer"]) == (m, s, metric, "all")), np.nan)
                 for s in splits]
            ax.bar(np.arange(len(splits)) + k * w, v, w, label=m)
        ax.set_xticks(np.arange(len(splits)) + 0.4 - w / 2, splits, rotation=45, ha="right",
                      fontsize=8)
        ax.set_ylabel(label)
        ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.text(0.01, 0.01, footer, fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _go_no_go(rows: list[dict[str, Any]], gate: dict[str, Any]) -> tuple[bool, str]:
    def val(metric: str) -> float:
        for r in rows:
            if (r["model"], r["split"], r["metric"], r["layer"]) == (gate["model"], gate["split"],
                                                                     metric, "all"):
                return r["mean"]
        raise E5Error(f"go/no-go: no {metric} row for {gate['model']} on {gate['split']}")
    rl, hot = val("rel_l2"), val(gate["hotspot_metric"])
    ok = rl < gate["rel_l2_max"] and hot < gate["hotspot_K_max"]
    text = "\n".join([
        "# Week-5 surrogate go/no-go (CLAUDE.md §12), computed by experiments/e5_accuracy.py", "",
        f"- Model: {gate['model']}; split: {gate['split']}",
        f"- Mean relative L2: {rl:.4f} (target < {gate['rel_l2_max']})",
        f"- Mean hotspot error ({gate['hotspot_metric']}): {hot:.3f} K "
        f"(target < {gate['hotspot_K_max']} K)",
        f"- Result: **{'GO' if ok else 'NO-GO'}**",
        "",
        ("Recommendation: run the gate experiments at 256² as planned."
         if ok else "Recommendation: run the gate experiments at 128² and say so (CLAUDE.md §12)."),
        "", "The decision is N and M's; record it in docs/decisions.md."])
    return ok, text


# ---------------------------------------------------------------------------- main

def run(cfg: dict[str, Any], data: Path, out: Path, device: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    registry, norm = ROOT / cfg["registry"], ROOT / cfg["norm_stats"]
    if Path(cfg["registry"]).is_absolute():
        registry, norm = Path(cfg["registry"]), Path(cfg["norm_stats"])
    N, batch = cfg["N"], cfg["batch"]
    models = _models(cfg, registry, norm, device)
    rows: list[dict[str, Any]] = []
    for split in cfg["splits"]:
        ds = TwinHeatDataset(data, split, N=N, allow_eval=True)
        for label, ens in models.items():
            for metric, by_layer in evaluate(ens, ds, batch).items():
                for layer in ("all", *LAYERS):
                    mean, p95 = _summary(by_layer[layer])
                    rows.append({"model": label, "split": split, "metric": metric, "layer": layer,
                                 "mean": mean, "p95": p95, "n": len(by_layer[layer])})
    with (out / "e5_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    o = cfg["ood"]
    feat_ens = Ensemble.load(registry, norm, stage=o["members"]["stage"], device=device,
                             arch=o["members"]["arch"], variant=o["members"]["variant"])
    score = OODScore.fit(_features(feat_ens, TwinHeatDataset(data, o["fit_split"], N=N), batch,
                                   o["fit_max_boards"]))
    thr = score.set_threshold(_features(feat_ens, TwinHeatDataset(data, o["threshold_split"], N=N),
                                        batch, None), o["threshold_percentile"])
    score.save(out / "ood_score.json")
    ood = []
    for split in cfg["splits"]:
        flags = score.flag(_features(feat_ens, TwinHeatDataset(data, split, N=N, allow_eval=True),
                                     batch, None))
        ood.append({"split": split, "n": len(flags), "flagged": int(flags.sum()),
                    "flag_rate": float(flags.mean()), "threshold": thr,
                    "kind": "false-positive rate (ID)" if split == "test" else "recall (OOD)"})
    with (out / "e5_ood.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ood[0]))
        w.writeheader()
        w.writerows(ood)

    ok, text = _go_no_go(rows, cfg["go_no_go"])
    (out / "go_no_go.md").write_text(text + "\n", encoding="utf-8")
    _write_tables(rows, ood, cfg["go_no_go"], out / "e5_tables.tex")
    try:
        commit = _git("rev-parse", "--short", "HEAD") + ("-dirty" if _git("status", "--porcelain")
                                                         else "")
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    dhash = dataset_hash(Path(data) / "train")
    seeds = sorted({m["seed"] for e in models.values() for m in e.members})
    footer = (f"commit {commit} · dataset {dhash[:12]} · member seeds {seeds} · "
              f"{platform.processor() or platform.machine()} / GPU {gpu} · "
              f"{dt.date.today().isoformat()}")
    _write_figure(rows, footer, out / "e5_figure.png")
    info = {"commit": commit, "dataset_hash": dhash, "registry_sha256": _sha256(registry),
            "norm_stats_sha256": _sha256(norm), "N": N, "gpu": gpu, "device": device,
            "date": dt.date.today().isoformat(), "go": ok, "ood_threshold": thr}
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "ood": ood, "go": ok, "go_no_go": text}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "e5.yaml")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "e5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-dirty", action="store_true", help="tests only, never a reported run")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.allow_dirty)
    data = a.data or (Path(cfg["data_root"]) if cfg.get("data_root") else None)
    if data is None:
        raise E5Error("no dataset: set data_root in the config or pass --data")
    res = run(cfg, data, a.out, a.device)
    print(res["go_no_go"])


if __name__ == "__main__":
    main()
