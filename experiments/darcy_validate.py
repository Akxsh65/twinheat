"""Validate the FNO and trainer on PDEBench 2D Darcy, beta = 1.0, 128² (prompt M1.3; CLAUDE.md §12).

    python -m experiments.darcy_validate --data /path/2D_DarcyFlow_beta1.0_Train.hdf5 \
        --backup-dir /drive/twinheat/darcy [--epochs E --batch B --lr LR --weight-decay WD]

File layout (read from the file, 2026-10-28): "nu" [10000, 128, 128] permeability (input),
"tensor" [10000, 1, 128, 128] solution (target), attribute beta = 1.0. Splits: the first
n_train = 9,000 samples train, of which the last n_val are held out for best-checkpoint
selection; the last n_test = 1,000 are the test split, read through models.data_guard with
allow_eval=True. Whether PDEBench's published number used this exact split is UNVERIFIED.

Input channels: normalised nu, x/L, y/L (as TwinHeat's coordinate channels, §7.5). Target and input
are normalised with train-only mean/std. Model: FNO2d(3, 1), width 48, modes 24, 4 layers (§12).
lr, weight decay, epochs and batch size are not fixed by CLAUDE.md: the defaults below are
UNSURE starting points; record what was used.

The published number is read from docs/refs/pdebench_darcy_fno.txt (CLAUDE.md §16.2), which a
human writes. It must contain one line `fno_test_rel_l2: <number>` plus provenance lines. Done when
the test relative L2 is within 20 % of it. Outputs: results JSON, the training log, and a learning
curve PNG with the CLAUDE.md §4 footer. Checkpoints go to --ckpt-dir (git-ignored) and
--backup-dir.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "docs" / "refs" / "pdebench_darcy_fno.txt"
REF_RE = re.compile(r"^fno_test_rel_l2:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
TOLERANCE = 0.20  # prompt M1.3: within ~20 % of the published value
BETA = 1.0


def read_reference(path: Path | None = None) -> float:
    """The published FNO test relative L2 from docs/refs (never reconstructed: §0 rule 12)."""
    path = REF_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: copy the published FNO Darcy (beta = 1.0) "
                                "number from the PDEBench paper into it (CLAUDE.md §16.2)")
    found = REF_RE.findall(path.read_text(encoding="utf-8"))
    if len(found) != 1:
        raise ValueError(f"{path} must have exactly one line 'fno_test_rel_l2: <number>', "
                         f"found {len(found)}")
    value = float(found[0])
    if not 0 < value < 1:
        raise ValueError(f"{path}: fno_test_rel_l2 = {value} is not a relative error in (0, 1)")
    return value


def split_indices(n_total: int, n_train: int, n_val: int, n_test: int
                  ) -> dict[str, np.ndarray]:  # fmt: skip
    """train and val from the first n_train samples (val = their last n_val); test = last n_test."""
    if not (0 < n_val < n_train and n_test > 0 and n_train + n_test <= n_total):
        raise ValueError(f"bad split sizes: total {n_total}, train {n_train}, val {n_val}, "
                         f"test {n_test}")
    idx = {"train": np.arange(0, n_train - n_val), "val": np.arange(n_train - n_val, n_train),
           "test": np.arange(n_total - n_test, n_total)}  # fmt: skip
    for a in idx:
        for b in idx:
            if a < b and np.intersect1d(idx[a], idx[b]).size:
                raise AssertionError(f"splits {a} and {b} overlap")
    return idx


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                               capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"results need the git commit: {e}") from e
    return sha + ("-dirty" if dirty else "")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--backup-dir", type=Path, required=True, help="second copy, e.g. on Drive")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "darcy")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "darcy")
    ap.add_argument("--epochs", type=int, default=100, help="UNSURE default")
    ap.add_argument("--batch", type=int, default=32, help="UNSURE default")
    ap.add_argument("--lr", type=float, default=1e-3, help="UNSURE default")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="UNSURE default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import h5py  # only this script reads HDF5
    import torch

    from models.data_guard import open_split
    from models.fno import FNO2d
    from models.train import TrainConfig, relative_l2, train

    reference = read_reference()  # fail before hours of training, not after
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA GPU is visible")
    commit = _git_commit()
    data_hash = _sha256(args.data)
    run_id = f"darcy_validate-{commit}-{args.seed}"

    with h5py.File(args.data, "r") as f:
        for key in ("nu", "tensor"):
            if key not in f:
                raise KeyError(f"{args.data}: dataset {key!r} missing; found {sorted(f)}")
        if float(f.attrs.get("beta", np.nan)) != BETA:
            raise ValueError(f"{args.data}: beta = {f.attrs.get('beta')}, expected {BETA}")
        nu, sol = f["nu"], f["tensor"]
        n_total, h, w = nu.shape
        if sol.shape != (n_total, 1, h, w):
            raise ValueError(f"tensor shape {sol.shape} does not match nu {nu.shape}")
        idx = split_indices(n_total, args.n_train, args.n_val, args.n_test)

        def load(split: str) -> tuple[np.ndarray, np.ndarray]:
            i = idx[split]
            return nu[i[0]:i[-1] + 1], sol[i[0]:i[-1] + 1, 0]  # contiguous ranges

        raw = {s: open_split(s, load, allow_eval=(s == "test")) for s in ("train", "val", "test")}

    torch.manual_seed(args.seed)
    nu_mean, nu_std = float(raw["train"][0].mean()), float(raw["train"][0].std())
    y_mean, y_std = float(raw["train"][1].mean()), float(raw["train"][1].std())
    yy, xx = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")

    def tensors(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        a, u = raw[split]
        n = len(a)
        x = np.stack([(a - nu_mean) / nu_std, np.broadcast_to(xx, (n, h, w)),
                      np.broadcast_to(yy, (n, h, w))], axis=1).astype(np.float32)  # fmt: skip
        return torch.from_numpy(x), torch.from_numpy(u[:, None].astype(np.float32))

    data = {s: tensors(s) for s in raw}
    model = FNO2d(3, 1, width=48, modes=24, n_layers=4).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(ckpt_dir=args.ckpt_dir, backup_dir=args.backup_dir,
                      log_csv=args.out / f"{run_id}_log.csv", epochs=args.epochs,
                      batch_size=args.batch, lr=args.lr, weight_decay=args.weight_decay,
                      seed=args.seed)  # fmt: skip
    stats = {"nu_mean": nu_mean, "nu_std": nu_std, "dataset_sha256": data_hash}
    summary = train(model, data["train"], data["val"], cfg, y_mean, y_std, extra=stats)

    best = torch.load(args.ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    model.eval()
    errs = []
    with torch.no_grad():
        xt, yt = data["test"]
        for i in range(0, len(xt), args.batch):
            pred = model(xt[i:i + args.batch].to(device)).float() * y_std + y_mean
            errs.append(relative_l2(pred, yt[i:i + args.batch].to(device)).cpu())
    test = float(torch.cat(errs).mean())
    chip = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    result = {
        "run_id": run_id, "commit": commit, "dataset_sha256": data_hash, "seed": args.seed,
        "hardware": f"{platform.platform()} {chip}",
        "date": dt.date.today().isoformat(), "splits": {k: int(v.size) for k, v in idx.items()},
        "hyperparameters": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                            "weight_decay": args.weight_decay, "width": 48, "modes": 24,
                            "n_layers": 4},
        "best_epoch": summary["best_epoch"], "best_val_rel_l2": summary["best_val_rel_l2"],
        "test_rel_l2": test, "published_rel_l2": reference,
        "ratio_to_published": test / reference,
        "within_tolerance": abs(test / reference - 1) <= TOLERANCE,
    }
    (args.out / f"{run_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _curve(cfg.log_csv, args.out / f"{run_id}_curve.png", result)
    print(json.dumps(result, indent=2))
    return result


def _curve(log_csv: Path, out: Path, result: dict[str, Any]) -> None:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(log_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.semilogy(ep, [float(r["train_rel_l2"]) for r in rows], label="train")
    ax.semilogy(ep, [float(r["val_rel_l2"]) for r in rows], label="val")
    ax.axhline(result["published_rel_l2"], color="0.4", ls="--", lw=1, label="published (refs)")
    ax.set(xlabel="epoch", ylabel="relative L2", title="FNO on PDEBench Darcy, beta = 1.0")
    ax.legend()
    footer = (f"commit {result['commit']} · dataset sha256 {result['dataset_sha256'][:12]} · "
              f"seed {result['seed']} · {result['hardware']} · {result['date']}")
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.95))
    fig.text(0.01, 0.01, footer, fontsize=5.5, color="0.35")
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
