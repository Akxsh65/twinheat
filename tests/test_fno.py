"""FNO, trainer and split guard (prompt M1.3; CLAUDE.md §12, §5)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

import models.train as train_mod
from models.data_guard import SplitAccessError, is_protected, open_split
from models.fno import FNO2d, SpectralConv2d
from models.train import TrainConfig, relative_l2, train


def _grid(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    c = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    return torch.meshgrid(c, c, indexing="ij")  # Y, X: [y, x]


def _identity(conv: SpectralConv2d) -> None:
    with torch.no_grad():
        for w in (conv.w_pos, conv.w_neg):
            w.zero_()
            for i in range(conv.in_ch):
                w[i, i] = 1.0


# ------------------------------------------------------------------ spectral convolution


@pytest.mark.parametrize("n", [32, 64])
def test_identity_weights_pass_a_low_frequency_sine(n: int) -> None:
    """Needs both the positive and the negative y-frequency blocks: sin(2π·3y) has ±3."""
    conv = SpectralConv2d(2, 2, modes=6)
    _identity(conv)
    Y, X = _grid(n)
    f = torch.stack([torch.sin(2 * math.pi * 3 * Y) * torch.cos(2 * math.pi * 2 * X),
                     torch.cos(2 * math.pi * 5 * Y + 0.3)]).float()[None]
    torch.testing.assert_close(conv(f), f, rtol=0, atol=1e-5)


def test_frequencies_above_the_cut_are_removed() -> None:
    conv = SpectralConv2d(1, 1, modes=4)
    _identity(conv)
    Y, X = _grid(32)
    high = torch.sin(2 * math.pi * 9 * Y)[None, None].float()  # |k_y| = 9 > 4
    assert conv(high).abs().max() < 1e-5


def test_negative_y_block_is_used() -> None:
    conv = SpectralConv2d(1, 1, modes=4)
    _identity(conv)
    with torch.no_grad():
        conv.w_neg.zero_()  # without the negative block a real y-sine is not reproduced
    Y, _ = _grid(32)
    f = torch.sin(2 * math.pi * 2 * Y)[None, None].float()
    assert (conv(f) - f).abs().max() > 0.1


def test_modes_larger_than_the_grid_raise() -> None:
    with pytest.raises(ValueError, match="modes"):
        SpectralConv2d(1, 1, modes=24)(torch.zeros(1, 1, 32, 32))


# ------------------------------------------------------------------ FNO2d


@pytest.mark.parametrize("n", [64, 128])
def test_fno_forward_shapes_at_64_and_128(n: int) -> None:
    torch.manual_seed(0)
    model = FNO2d(in_ch=13, out_ch=4, width=48, modes=24, n_layers=4)  # CLAUDE.md §12 config
    out = model(torch.randn(2, 13, n, n))
    assert out.shape == (2, 4, n, n) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_fno_is_resolution_independent_in_parameters() -> None:
    model = FNO2d(3, 1, width=8, modes=4, n_layers=2)
    assert model(torch.randn(1, 3, 16, 16)).shape == (1, 1, 16, 16)
    assert model(torch.randn(1, 3, 40, 40)).shape == (1, 1, 40, 40)


# ------------------------------------------------------------------ loss


def test_relative_l2() -> None:
    t = torch.ones(2, 1, 4, 4)
    p = t.clone()
    p[1] *= 1.5
    torch.testing.assert_close(relative_l2(p, t), torch.tensor([0.0, 0.5]))
    with pytest.raises(ValueError, match="zero"):
        relative_l2(t, torch.zeros_like(t))


# ------------------------------------------------------------------ trainer


def _toy(n: int = 40, size: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A smoothing operator the FNO can learn: y = low-pass(x) + 1."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, size, size, generator=g)
    xf = torch.fft.rfft2(x)
    xf[..., 4:, :] = 0
    xf[..., 3:] = 0
    return x, torch.fft.irfft2(xf, s=(size, size)) + 1.0


def _cfg(tmp: Path, epochs: int = 4) -> TrainConfig:
    return TrainConfig(ckpt_dir=tmp / "ckpt", backup_dir=tmp / "backup", log_csv=tmp / "log.csv",
                       epochs=epochs, batch_size=8, lr=3e-3, weight_decay=1e-4, seed=7)


def _model() -> FNO2d:
    torch.manual_seed(3)
    return FNO2d(1, 1, width=8, modes=4, n_layers=2, proj_hidden=16)


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_training_lowers_the_error_and_writes_checkpoints(tmp_path: Path) -> None:
    x, y = _toy()
    cfg = _cfg(tmp_path, epochs=6)
    out = train(_model(), (x[:32], y[:32]), (x[32:], y[32:]), cfg, float(y.mean()), float(y.std()))
    rows = _rows(cfg.log_csv)
    assert [int(r["epoch"]) for r in rows] == list(range(6))
    assert float(rows[-1]["train_rel_l2"]) < float(rows[0]["train_rel_l2"])
    for name in ("last.pt", "best.pt"):
        assert (cfg.ckpt_dir / name).exists() and (cfg.backup_dir / name).exists()
    assert out["resumed_from"] is None and out["epochs_run"] == 6


def test_resume_after_a_kill_continues_the_same_curve(tmp_path: Path, monkeypatch) -> None:
    x, y = _toy()
    args = ((x[:32], y[:32]), (x[32:], y[32:]))
    stats = (float(y.mean()), float(y.std()))
    full = _cfg(tmp_path / "full")
    train(_model(), *args, full, *stats)

    killed = _cfg(tmp_path / "killed")
    real_eval, calls = train_mod._evaluate, {"n": 0}

    def dying_eval(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:  # dies inside epoch 2, after epochs 0 and 1 were saved
            raise KeyboardInterrupt("simulated kill")
        return real_eval(*a, **k)

    monkeypatch.setattr(train_mod, "_evaluate", dying_eval)
    with pytest.raises(KeyboardInterrupt):
        train(_model(), *args, killed, *stats)
    monkeypatch.setattr(train_mod, "_evaluate", real_eval)
    torch.manual_seed(999)  # a different fresh init must not matter: the checkpoint wins
    fresh = FNO2d(1, 1, width=8, modes=4, n_layers=2, proj_hidden=16)
    out = train(fresh, *args, killed, *stats)

    assert out["resumed_from"] == 1 and out["epochs_run"] == 2
    a, b = _rows(full.log_csv), _rows(killed.log_csv)
    assert [r["epoch"] for r in a] == [r["epoch"] for r in b]
    for ra, rb in zip(a, b, strict=True):
        for col in ("train_rel_l2", "val_rel_l2", "lr"):
            assert float(ra[col]) == pytest.approx(float(rb[col]), rel=1e-6), (ra["epoch"], col)


def test_resume_refuses_a_checkpoint_from_another_config(tmp_path: Path) -> None:
    x, y = _toy()
    cfg = _cfg(tmp_path, epochs=1)
    train(_model(), (x[:32], y[:32]), (x[32:], y[32:]), cfg, 1.0, 1.0)
    other = TrainConfig(**{**cfg.__dict__, "seed": 8})
    with pytest.raises(ValueError, match="seed"):
        train(_model(), (x[:32], y[:32]), (x[32:], y[32:]), other, 1.0, 1.0)


def test_warmup_then_cosine_schedule() -> None:
    f = train_mod._schedule(100, 0.05)
    assert f(0) == pytest.approx(0.2) and f(4) == pytest.approx(1.0)
    assert f(5) == pytest.approx(1.0) and f(99) < 0.01
    assert all(f(i) >= f(i + 1) for i in range(5, 99))


def test_config_rejects_backup_equal_to_checkpoint_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backup_dir"):
        TrainConfig(ckpt_dir=tmp_path, backup_dir=tmp_path, log_csv=tmp_path / "l.csv",
                    epochs=1, batch_size=1, lr=1e-3, weight_decay=0.0, seed=0)


# ------------------------------------------------------------------ data guard


@pytest.mark.parametrize("split", ["cal", "test", "ood_1", "ood_bottom_chips"])
def test_guard_refuses_protected_splits(split: str) -> None:
    assert is_protected(split)
    with pytest.raises(SplitAccessError, match=split):
        open_split(split, lambda s: s)
    assert open_split(split, lambda s: s.upper(), allow_eval=True) == split.upper()


@pytest.mark.parametrize("split", ["train", "val"])
def test_guard_allows_train_and_val(split: str) -> None:
    assert not is_protected(split) and open_split(split, lambda s: s) == split


@pytest.mark.parametrize("split", ["", None, 3])
def test_guard_rejects_bad_names(split) -> None:
    with pytest.raises(ValueError, match="split"):
        open_split(split, lambda s: s)  # type: ignore[arg-type]


# ------------------------------------------------------------------ Darcy script helpers


def test_reference_file_format(tmp_path: Path) -> None:
    from experiments.darcy_validate import read_reference

    ref = tmp_path / "ref.txt"
    with pytest.raises(FileNotFoundError, match="PDEBench"):
        read_reference(ref)
    ref.write_text("source: PDEBench, Table X\nfno_test_rel_l2: 0.05\n", encoding="utf-8")
    assert read_reference(ref) == 0.05
    ref.write_text("fno_test_rel_l2: 0.05\nfno_test_rel_l2: 0.06\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        read_reference(ref)
    ref.write_text("fno_test_rel_l2: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="relative error"):
        read_reference(ref)


def test_darcy_splits_are_disjoint_and_test_is_last() -> None:
    from experiments.darcy_validate import split_indices

    idx = split_indices(10_000, 9_000, 500, 1_000)
    assert [len(idx[k]) for k in ("train", "val", "test")] == [8_500, 500, 1_000]
    assert idx["test"][0] == 9_000 and idx["test"][-1] == 9_999 and idx["val"][-1] == 8_999
    with pytest.raises(ValueError, match="split"):
        split_indices(10_000, 9_500, 500, 1_000)
