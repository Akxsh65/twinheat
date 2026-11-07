"""Train one FNO ensemble member (M2.2; CLAUDE.md §12, §4, §5).

    python -m models.train_ensemble --member 1 --seed S --stage pre128 --epochs E \
        --data DATASET_V1_ROOT --ckpt_dir CK --backup_dir BK [--norm-stats models/norm_stats.json]
    python -m models.train_ensemble --member 1 --seed S --stage ft256 --init CK/best.pt ...

Stages (§12): pre128 trains at 128² (targets 2×2 area averages, inputs re-rasterised, lr 1e-3);
ft256 fine-tunes at 256² from --init (lr 3e-4). Model FNO2d(13 -> 4, width 48, modes 24, 4
layers). Loss = rel_l2 + w_hot · hotspot_l1 (w_hot 0.1) on the normalised output, both terms
logged. UNSURE: §12 does not say whether hotspot_l1 is in K or normalised units inside the loss;
here it is normalised (K / σ_θ), which keeps the two terms of similar size; validation and early
stopping use the hotspot error in K, as §12 says. AdamW, per-step cosine schedule with 5 %
warm-up, AMP (float16 autocast on CUDA) with the loss in FP32, gradient clipping 1.0.

Batch: on CUDA a memory probe finds the largest power-of-two micro-batch that runs a forward and
backward pass at this N (probe_batch); --batch asks for a larger effective batch, reached by
gradient accumulation over equal micro-batches (plan_batches). The plan is stored in the
checkpoint and reused on resume. Without CUDA (tests) the micro-batch is --batch (default 2).
Early stopping on the validation hotspot error (K), patience 10. Every epoch writes last.pt
(best.pt when the validation hotspot improves) to --ckpt_dir and copies it to --backup_dir;
rerunning the same command resumes from last.pt and continues the same curve (shuffle and
augmentation are seeded by (seed, epoch)). GPU-hours = CUDA wall time of training epochs, summed
across resumes. At the end: <ckpt_dir>/summary.json, and the member's best.pt is entered in
--registry (models/registry.json) with its SHA-256 and the norm_stats SHA-256.

Not fixed by CLAUDE.md, so UNSURE defaults that experiments/configs/frozen.yaml should replace for
reported runs (§4): weight decay 1e-4, the probe's upper limit, DataLoader workers.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import losses, zoo
from models.dataset import TwinHeatDataset
from models.energy import energy_loss
from models.normalise import Normaliser, dataset_hash
from models.train import _save, _schedule

__all__ = ["STAGES", "evaluate", "main", "parse", "plan_batches", "probe_batch", "train_member"]

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"pre128": {"N": 128, "lr": 1e-3}, "ft256": {"N": 256, "lr": 3e-4}}  # §12
MODEL = zoo.config("fno")  # FNO2d(13 -> 4, width 48, modes 24, 4 layers), §12
WEIGHT_DECAY = 1e-4  # UNSURE (see module docstring)
WARMUP_FRAC = 0.05  # §12
GRAD_CLIP = 1.0  # §12
PATIENCE = 10  # the M2.2 prompt
PROBE_LIMIT = 64  # UNSURE: largest micro-batch the probe tries
# settings that must match to resume (paths, workers, device and cache may differ between machines)
RESUME_KEYS = ("member", "seed", "stage", "epochs", "N", "lr", "weight_decay", "batch",
               "patience", "w_hot", "arch", "variant", "labels", "energy_weight")
LOG_FIELDS = ("epoch", "train_loss", "train_rel_l2", "train_hotspot", "train_energy",
              "val_rel_l2", "val_hotspot_K", "lr", "seconds", "gpu_hours")


@dataclass(frozen=True)
class Config:
    member: int
    seed: int
    stage: str
    init: str | None
    ckpt_dir: str
    backup_dir: str
    epochs: int
    data: str
    norm_stats: str
    registry: str
    N: int
    lr: float
    weight_decay: float
    batch: int | None
    workers: int
    device: str
    cache_dir: str | None
    patience: int
    w_hot: float
    arch: str = "fno"
    variant: str = "base"
    labels: int | None = None
    energy_weight: float = 0.0

    @property
    def energy_mode(self) -> bool:
        return self.energy_weight > 0 or self.labels is not None


def parse(argv: list[str] | None = None) -> Config:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--member", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", choices=tuple(STAGES), required=True)
    ap.add_argument("--init", default=None, help="checkpoint to start from (required for ft256)")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--backup_dir", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--data", required=True, help="dataset root, one folder per split")
    ap.add_argument("--norm-stats", default=str(ROOT / "models" / "norm_stats.json"))
    ap.add_argument("--registry", default=str(ROOT / "models" / "registry.json"))
    ap.add_argument("--N", type=int, default=None, help="override the stage's N (tests only)")
    ap.add_argument("--lr", type=float, default=None, help="override the stage's lr")
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--batch", type=int, default=None, help="effective batch (accumulates)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=None, help="input cache (models/dataset.py)")
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--w-hot", type=float, default=losses.W_HOT)
    ap.add_argument("--arch", choices=tuple(zoo.ARCHS), default="fno")
    ap.add_argument("--variant", default=None, help="registry label (default base, or energy_L<n>)")
    ap.add_argument("--labels", type=int, default=None,
                    help="energy ablation: only the first n train boards carry the data loss")
    ap.add_argument("--energy-weight", type=float, default=0.0,
                    help="weight of the energy-form loss (models/energy.py)")
    a = ap.parse_args(argv)
    if a.labels is not None and a.labels < 0:
        ap.error("--labels must be >= 0")
    if a.labels == 0 and a.energy_weight <= 0:
        ap.error("--labels 0 trains on the energy loss only: give --energy-weight > 0")
    if a.stage == "ft256" and a.init is None:
        ap.error("--stage ft256 fine-tunes: give --init <pre128 best.pt>")
    if a.member < 1 or a.epochs < 1 or a.patience < 1 or (a.batch is not None and a.batch < 1):
        ap.error("member, epochs, patience and batch must be >= 1")
    if Path(a.ckpt_dir).resolve() == Path(a.backup_dir).resolve():
        ap.error("--backup_dir must differ from --ckpt_dir (it is the second copy)")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    st = STAGES[a.stage]
    variant = a.variant or (f"energy_L{a.labels}" if a.labels is not None
                            else ("energy" if a.energy_weight > 0 else "base"))
    return Config(a.member, a.seed, a.stage, a.init, a.ckpt_dir, a.backup_dir, a.epochs, a.data,
                  a.norm_stats, a.registry, a.N or st["N"], a.lr or st["lr"], a.weight_decay,
                  a.batch, a.workers, device, a.cache_dir, a.patience, a.w_hot, a.arch, variant,
                  a.labels, a.energy_weight)


# ----------------------------------------------------------------------------- pieces

def build_model(cfg: Config, device: torch.device) -> nn.Module:
    torch.manual_seed(cfg.seed)  # the member's own initialisation
    model = zoo.build(zoo.config(cfg.arch)).to(device)
    if cfg.init is not None:
        state = torch.load(cfg.init, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    return model


def probe_batch(try_batch: Callable[[int], None], limit: int = PROBE_LIMIT) -> int:
    """Largest power of two b <= limit for which try_batch(b) raises no CUDA out-of-memory."""
    best, b = 0, 1
    while b <= limit:
        try:
            try_batch(b)
        except torch.cuda.OutOfMemoryError:
            break
        best, b = b, 2 * b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if best == 0:
        raise RuntimeError("memory probe: even batch 1 does not fit on this GPU at this N")
    return best


def plan_batches(requested: int | None, fits: int) -> tuple[int, int]:
    """(micro-batch, accumulation steps): equal micro-batches no larger than `fits`."""
    if requested is None or requested <= fits:
        return (fits if requested is None else requested), 1
    accum = math.ceil(requested / fits)
    return math.ceil(requested / accum), accum


def _try_batch(model: nn.Module, n: int, device: torch.device) -> Callable[[int], None]:
    def run(b: int) -> None:
        x = torch.randn(b, zoo.IO["in_ch"], n, n, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):  # fmt: skip
            out = model(x)
        out.float().square().mean().backward()
        model.zero_grad(set_to_none=True)
    return run


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, theta_scale: float, device: torch.device,
             use_amp: bool) -> tuple[float, float]:
    """(mean relative L2, mean hotspot error in K) over the loader."""
    model.eval()
    rl, hot, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
        pred_k, true_k = out.float() * theta_scale, y.float() * theta_scale
        rl += float(losses.rel_l2(pred_k, true_k).sum())
        hot += float(losses.hotspot_l1(pred_k, true_k).sum())
        n += len(x)
    return rl / n, hot / n


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit() -> str:
    try:
        def git(*a: str) -> str:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain")
                                                      else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _key(m: dict[str, Any]) -> tuple[int, str, str, str]:
    """Registry identity: (member, stage, arch, variant); older entries are fno / base."""
    return m["member"], m["stage"], m.get("arch", "fno"), m.get("variant", "base")


class _EnergyItems:
    """Dataset view for the energy loss: (x, y, coefficients, index) per item."""

    def __init__(self, ds: TwinHeatDataset):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        sym = self.ds.symmetry(i)
        x, y = self.ds.sample(i, sym)
        return x, y, self.ds.coefficients(i, sym), i


def update_registry(registry: Path, entry: dict[str, Any]) -> None:
    """Add or replace the (member, stage) entry of models/registry.json, atomically."""
    registry = Path(registry)
    reg = (json.loads(registry.read_text(encoding="utf-8")) if registry.exists()
           else {"schema": "twinheat-registry/1", "members": []})
    reg["members"] = [m for m in reg["members"] if _key(m) != _key(entry)]
    reg["members"].append(entry)
    reg["members"].sort(key=lambda m: (*_key(m)[1:], m["member"]))
    registry.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, registry)


# ----------------------------------------------------------------------------- training

def train_member(cfg: Config, on_epoch: Callable[[int], None] | None = None) -> dict[str, Any]:
    device = torch.device(cfg.device)
    use_amp = device.type == "cuda"
    data = Path(cfg.data)
    norm = Normaliser.load(cfg.norm_stats, dataset_hash(data / "train"))
    norm_sha = _sha256(Path(cfg.norm_stats))
    train_ds = TwinHeatDataset(data, "train", N=cfg.N, augment=True, normaliser=norm,
                               cache_dir=cfg.cache_dir, seed=cfg.seed)
    val_ds = TwinHeatDataset(data, "val", N=cfg.N, normaliser=norm, cache_dir=cfg.cache_dir)
    ckpt_dir, backup_dir = Path(cfg.ckpt_dir), Path(cfg.backup_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last = ckpt_dir / "last.pt"
    model = build_model(cfg, device)
    state = torch.load(last, map_location=device, weights_only=False) if last.exists() else None

    if state is not None:  # resume: same training settings and data, same batch plan
        want = {k: v for k, v in asdict(cfg).items() if k in RESUME_KEYS}
        got = {k: state["config"].get(k) for k in want}
        if got != want:
            diff = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
            raise ValueError(f"{last} was written with a different configuration: {diff}")
        if (state["norm_stats_sha256"], state["dataset_hash"]) != (norm_sha, norm.dataset_hash):
            raise ValueError(f"{last} was trained with other norm stats or another train split")
        micro, accum = state["micro_batch"], state["accum"]
    elif device.type == "cuda":
        micro, accum = plan_batches(cfg.batch, probe_batch(_try_batch(model, cfg.N, device)))
    else:
        micro, accum = (cfg.batch or 2), 1

    steps = math.ceil(len(train_ds) / (micro * accum))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule(cfg.epochs * steps, WARMUP_FRAC))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, best, best_rl, best_epoch, bad, gpu_s, stopped = 0, math.inf, math.inf, -1, 0, 0.0, False
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"] + 1
        best, best_rl, best_epoch = state["best_val_hotspot_K"], state["best_val_rel_l2"], \
            state["best_epoch"]
        bad, gpu_s, stopped = state["bad_epochs"], state["gpu_seconds"], state["stopped_early"]

    log_path = ckpt_dir / "log.csv"
    kept: list[list[str]] = []
    if state is not None and log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows or tuple(rows[0]) != LOG_FIELDS:
            raise ValueError(f"{log_path} does not start with {LOG_FIELDS}")
        kept = [r for r in rows[1:] if int(r[0]) <= state["epoch"]]
    val_loader = DataLoader(val_ds, batch_size=micro, num_workers=cfg.workers)
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        log = csv.writer(fh)
        log.writerow(LOG_FIELDS)
        log.writerows(kept)
        fh.flush()
        for epoch in range(start, cfg.epochs):
            if stopped:
                break
            t0 = time.perf_counter()
            model.train()
            train_ds.set_epoch(epoch)
            g = torch.Generator().manual_seed(cfg.seed * 100_003 + epoch)
            order = torch.randperm(len(train_ds), generator=g).tolist()
            items = _EnergyItems(train_ds) if cfg.energy_mode else train_ds
            loader = DataLoader(items, batch_size=micro, sampler=order, num_workers=cfg.workers)
            sums = {"loss": 0.0, "rl": 0.0, "hot": 0.0, "energy": 0.0}
            count = 0
            opt.zero_grad(set_to_none=True)
            for b, batch in enumerate(loader):
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=use_amp):  # fmt: skip
                    out = model(x)
                rl = losses.rel_l2(out.float(), y)  # FP32 loss
                hot = losses.hotspot_l1(out.float(), y)
                data = rl + cfg.w_hot * hot
                if cfg.energy_mode:
                    coeffs = {k: v.to(device) for k, v in batch[2].items()}
                    idx = batch[3].to(device)
                    labelled = (torch.ones_like(idx, dtype=torch.bool) if cfg.labels is None
                                else idx < cfg.labels)
                    en = energy_loss(coeffs, out.float() * norm.theta_scale, norm.theta_scale)
                    data_term = (data * labelled).sum() / labelled.sum().clamp(min=1)
                    loss = data_term + cfg.energy_weight * en.mean()
                    sums["energy"] += float(en.detach().sum())
                else:
                    loss = data.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"member {cfg.member} epoch {epoch} batch {b}: "
                                             f"loss is {loss.item()}")
                scaler.scale(loss / accum).backward()
                if (b + 1) % accum == 0 or b + 1 == len(loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                k = len(x)
                sums["loss"] += float(loss.detach()) * k
                sums["rl"] += float(rl.detach().sum())
                sums["hot"] += float(hot.detach().sum())
                count += k
            val_rl, val_hot = evaluate(model, val_loader, norm.theta_scale, device, use_amp)
            sec = time.perf_counter() - t0
            if device.type == "cuda":
                gpu_s += sec
            improved = val_hot < best
            if improved:
                best, best_rl, best_epoch, bad = val_hot, val_rl, epoch, 0
            else:
                bad += 1
            stopped = bad >= cfg.patience
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best_val_hotspot_K": best, "best_val_rel_l2": best_rl,
                     "best_epoch": best_epoch, "bad_epochs": bad, "gpu_seconds": gpu_s,
                     "stopped_early": stopped, "micro_batch": micro, "accum": accum,
                     "config": asdict(cfg), "model_config": zoo.config(cfg.arch),
                     "theta_scale": norm.theta_scale,
                     "norm_stats_sha256": norm_sha, "dataset_hash": norm.dataset_hash}
            _save(state, last, backup_dir)
            if improved:
                _save(state, ckpt_dir / "best.pt", backup_dir)
            log.writerow([epoch, f"{sums['loss'] / count:.6e}", f"{sums['rl'] / count:.6e}",
                          f"{sums['hot'] / count:.6e}", f"{sums['energy'] / count:.6e}",
                          f"{val_rl:.6e}", f"{val_hot:.6e}",
                          f"{sched.get_last_lr()[0]:.6e}", f"{sec:.2f}", f"{gpu_s / 3600:.4f}"])
            fh.flush()
            if on_epoch is not None:
                on_epoch(epoch)

    final = torch.load(last, map_location="cpu", weights_only=False)
    best_pt = ckpt_dir / "best.pt"
    summary = {
        "member": cfg.member, "seed": cfg.seed, "stage": cfg.stage, "N": cfg.N,
        "init": cfg.init, "epochs_planned": cfg.epochs, "epochs_run": final["epoch"] + 1,
        "stopped_early": final["stopped_early"], "best_epoch": final["best_epoch"],
        "best_val_rel_l2": final["best_val_rel_l2"],
        "best_val_hotspot_K": final["best_val_hotspot_K"],
        "gpu_hours": round(final["gpu_seconds"] / 3600, 4), "device": device.type,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "micro_batch": final["micro_batch"], "accum": final["accum"], "lr": cfg.lr,
        "weight_decay": cfg.weight_decay, "w_hot": cfg.w_hot, "model": zoo.config(cfg.arch),
        "arch": cfg.arch, "variant": cfg.variant, "labels": cfg.labels,
        "energy_weight": cfg.energy_weight,
        "commit": _commit(), "dataset_hash": norm.dataset_hash, "norm_stats_sha256": norm_sha,
        "checkpoint_sha256": _sha256(best_pt), "finished": dt.datetime.now(dt.UTC).isoformat(
            timespec="seconds"),
    }
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    reg = Path(cfg.registry)
    try:
        rel = os.path.relpath(best_pt.resolve(), reg.resolve().parent)
    except ValueError:  # another drive on Windows
        rel = str(best_pt.resolve())
    update_registry(reg, {
        "member": cfg.member, "stage": cfg.stage, "arch": cfg.arch, "variant": cfg.variant,
        "seed": cfg.seed, "N": cfg.N, "checkpoint": rel.replace("\\", "/"),
        "sha256": summary["checkpoint_sha256"],
        "norm_stats_sha256": norm_sha, "dataset_hash": norm.dataset_hash,
        "best_val_rel_l2": summary["best_val_rel_l2"],
        "best_val_hotspot_K": summary["best_val_hotspot_K"], "gpu_hours": summary["gpu_hours"],
        "commit": summary["commit"]})
    return summary


def main(argv: list[str] | None = None) -> None:
    summary = train_member(parse(argv))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
