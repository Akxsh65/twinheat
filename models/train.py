"""Training loop (prompt M1.3; CLAUDE.md §12): train(model, train, val, cfg) -> summary dict.

AdamW, per-step cosine schedule with linear warm-up (5 % of steps), AMP on CUDA with the loss in
FP32, gradient clipping at 1.0, relative-L2 loss on the decoded output. Every epoch writes
last.pt (and best.pt when the validation error improves) to cfg.ckpt_dir, copies both to
cfg.backup_dir, and appends a row to cfg.log_csv. train() resumes from ckpt_dir/last.pt if it
exists. Shuffling uses a generator seeded with (seed, epoch), so a resumed run continues the
same loss curve as an uninterrupted one.

Data are (x, y) tensors [n, C, H, W]; x is already normalised by the caller; the model predicts
y normalised by (y_mean, y_std) and the loss decodes it first.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = ["TrainConfig", "relative_l2", "train"]

LOG_FIELDS = ("epoch", "train_rel_l2", "val_rel_l2", "lr", "seconds")


@dataclass(frozen=True)
class TrainConfig:
    """Training settings. lr, weight_decay, epochs and batch_size are not fixed by CLAUDE.md;
    reported runs take them from experiments/configs/frozen.yaml (CLAUDE.md §4)."""

    ckpt_dir: Path
    backup_dir: Path
    log_csv: Path
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    warmup_frac: float = 0.05  # CLAUDE.md §12 / prompt M1.3
    grad_clip: float = 1.0  # CLAUDE.md §12
    amp: bool = True  # used only on CUDA

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError(f"epochs and batch_size must be >= 1, got {self.epochs}, "
                             f"{self.batch_size}")
        if not (self.lr > 0 and self.weight_decay >= 0 and 0 <= self.warmup_frac < 1):
            raise ValueError(f"bad lr / weight_decay / warmup_frac: {self.lr}, "
                             f"{self.weight_decay}, {self.warmup_frac}")
        if Path(self.ckpt_dir).resolve() == Path(self.backup_dir).resolve():
            raise ValueError("backup_dir must differ from ckpt_dir (it is the second copy)")


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Per-sample ‖pred − true‖₂ / ‖true‖₂ over all non-batch dims, in float32, shape [B]."""
    p, t = pred.float().flatten(1), true.float().flatten(1)
    den = t.norm(dim=1)
    if (den == 0).any():
        raise ValueError("relative_l2: a target sample is identically zero")
    return (p - t).norm(dim=1) / den


def _schedule(total_steps: int, warmup_frac: float):
    warm = max(1, round(warmup_frac * total_steps)) if warmup_frac > 0 else 0

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return f


def _save(state: dict[str, Any], path: Path, backup_dir: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a kill never leaves a half-written checkpoint
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)


@torch.no_grad()
def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, y_mean: float, y_std: float,
              batch_size: int, device: torch.device) -> float:
    model.eval()
    errs = []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        errs.append(relative_l2(model(xb).float() * y_std + y_mean, yb))
    return float(torch.cat(errs).mean())


def train(
    model: nn.Module,
    train_data: tuple[torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor],
    cfg: TrainConfig,
    y_mean: float,
    y_std: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train (or resume) and return {"epochs_run", "best_val_rel_l2", "best_epoch", "resumed_from"}.

    `extra` (e.g. normalisation stats, dataset hash) is stored in every checkpoint.
    """
    if not (y_std > 0 and math.isfinite(y_mean)):
        raise ValueError(f"y_std must be > 0 and y_mean finite, got {y_std}, {y_mean}")
    xt, yt = train_data
    if len(xt) != len(yt) or len(xt) == 0:
        raise ValueError(f"train x and y lengths differ or are 0: {len(xt)}, {len(yt)}")
    device = next(model.parameters()).device
    use_amp = cfg.amp and device.type == "cuda"
    ckpt_dir, backup_dir = Path(cfg.ckpt_dir), Path(cfg.backup_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = math.ceil(len(xt) / cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _schedule(cfg.epochs * steps_per_epoch, cfg.warmup_frac))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, best, best_epoch, resumed = 0, math.inf, -1, None
    last = ckpt_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["config"]["seed"] != cfg.seed or state["config"]["epochs"] != cfg.epochs:
            raise ValueError(f"{last} was written with seed/epochs {state['config']['seed']}/"
                             f"{state['config']['epochs']}, not {cfg.seed}/{cfg.epochs}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, best_epoch = state["epoch"] + 1, state["best_val"], state["best_epoch"]
        resumed = state["epoch"]

    log_path = Path(cfg.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[list[str]] = []
    if resumed is not None and log_path.exists():  # drop rows after the checkpoint's epoch
        with open(log_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows or tuple(rows[0]) != LOG_FIELDS:
            raise ValueError(f"{log_path} does not start with the header {LOG_FIELDS}")
        kept = [r for r in rows[1:] if int(r[0]) <= resumed]
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        log = csv.writer(fh)
        log.writerow(LOG_FIELDS)
        log.writerows(kept)
        for epoch in range(start, cfg.epochs):
            t0 = time.perf_counter()
            model.train()
            g = torch.Generator().manual_seed(cfg.seed * 100_003 + epoch)
            order = torch.randperm(len(xt), generator=g)
            total, count = 0.0, 0
            for i in range(0, len(xt), cfg.batch_size):
                idx = order[i:i + cfg.batch_size]
                xb, yb = xt[idx].to(device), yt[idx].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=use_amp):  # fmt: skip
                    out = model(xb)
                loss = relative_l2(out.float() * y_std + y_mean, yb).mean()  # FP32 loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"epoch {epoch} batch {i // cfg.batch_size}: "
                                             f"loss is {loss.item()}")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total, count = total + loss.item() * len(idx), count + len(idx)
            val = _evaluate(model, *val_data, y_mean, y_std, cfg.batch_size, device)
            improved = val < best
            if improved:
                best, best_epoch = val, epoch
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best_val": best, "best_epoch": best_epoch,
                     "config": {k: str(v) if isinstance(v, Path) else v
                                for k, v in asdict(cfg).items()},
                     "y_mean": y_mean, "y_std": y_std, "extra": extra or {}}  # fmt: skip
            _save(state, last, backup_dir)
            if improved:
                _save(state, ckpt_dir / "best.pt", backup_dir)
            log.writerow([epoch, f"{total / count:.6e}", f"{val:.6e}",
                          f"{sched.get_last_lr()[0]:.6e}", f"{time.perf_counter() - t0:.2f}"])
            fh.flush()
    return {"epochs_run": cfg.epochs - start, "best_val_rel_l2": best, "best_epoch": best_epoch,
            "resumed_from": resumed}
