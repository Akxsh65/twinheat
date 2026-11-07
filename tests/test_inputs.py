"""Model inputs, normalisation and dataset (M2.1; CLAUDE.md §7.5, §5, §6.6, §12; invariant B6).

The fixture is a real (tiny) dataset written by data/generate.py: train, val and cal at N = 32 on
the CPU, so the loader is tested against genuine shards, manifests and hashes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from data import generate as gen
from data.raster import rasterise, transform_array
from data.sampler import transform_record
from models import inputs as inp
from models.data_guard import SplitAccessError
from models.dataset import TwinHeatDataset
from models.normalise import Normaliser, NormStatsError, dataset_hash
from solvers.fvm import assemble_board, layer_conductances, solve_direct

SPEC = gen.SPEC
N = 32
KW = {"N": N, "shard_size": 4, "batch": 4, "device": "cpu", "allow_dirty": True,
      "log": lambda *a, **k: None}


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    r = tmp_path_factory.mktemp("dataset")
    gen.generate("train", 8, 11, r / "train", **KW)
    gen.generate("val", 4, 11, r / "val", **KW)
    gen.generate("cal", 4, 11, r / "cal", **KW)
    return r


def rows(root: Path, split: str) -> list[dict]:
    return [json.loads(line) for f in sorted((root / split).glob("shard_*/records.jsonl"))
            for line in f.read_text(encoding="utf-8").splitlines()]


def stored(root: Path, split: str) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted((root / split).glob("shard_*/fields.npy"))])


def solve(rec: dict, n: int) -> np.ndarray:
    A, b = assemble_board(SPEC, rasterise(rec, SPEC, n), rec["dirichlet_edges"])
    return solve_direct(A, b).reshape(4, n, n)


# ------------------------------------------------------------------------------ channels

def test_the_13_channels_follow_section_7_5(root):
    rec = rows(root, "train")[1]["record"]
    r = rasterise(rec, SPEC, N)
    x = inp.build(r, SPEC)
    assert x.dtype == np.float32 and x.shape == (13, N, N)
    np.testing.assert_allclose(x[0:4], np.log10(layer_conductances(SPEC, r)), rtol=1e-6)
    np.testing.assert_allclose(x[4], r.q, rtol=1e-6)  # W/m² on L1
    np.testing.assert_allclose(x[5:8], np.log1p(r.nu / 1e6), rtol=1e-6)  # nu in vias/mm²
    np.testing.assert_allclose(x[8], r.h_top, rtol=1e-6)
    np.testing.assert_allclose(x[9], r.h_bot, rtol=1e-6)
    np.testing.assert_array_equal(x[10], r.dirichlet_mask)
    c = (np.arange(N) + 0.5) / N
    np.testing.assert_allclose(x[11], np.broadcast_to(c[None, :], (N, N)), rtol=1e-7)  # x/L
    np.testing.assert_allclose(x[12], np.broadcast_to(c[:, None], (N, N)), rtol=1e-7)  # y/L
    assert inp.CHANNELS[0] == "log10_G_L1" and inp.CHANNELS[12] == "y_over_L"
    assert len(inp.CHANNELS) == inp.N_CHANNELS == 13


def test_build_refuses_a_non_positive_conductance(root, monkeypatch):
    r = rasterise(rows(root, "train")[0]["record"], SPEC, N)
    bad = layer_conductances(SPEC, r).copy()
    bad[2, 5, 7] = 0.0
    monkeypatch.setattr(inp, "layer_conductances", lambda spec, raster: bad)
    with pytest.raises(ValueError, match="L3"):
        inp.build(r, SPEC)


def test_transform_field_is_the_raster_transform(root):
    """transform_array maps every [.., y, x] field of a Raster; transform_field must be the same
    map, or targets and inputs would be transformed differently."""
    r = rasterise(rows(root, "train")[0]["record"], SPEC, N)
    a = np.random.default_rng(0).standard_normal((4, N, N))
    for sym in range(8):
        want = transform_array(r.__class__(**{**r.__dict__, "phi": a}), sym).phi
        np.testing.assert_array_equal(inp.transform_field(a, sym), want)
    with pytest.raises(ValueError, match="sym"):
        inp.transform_field(a, 8)


# ------------------------------------------------------------------------------ augmentation

@pytest.mark.parametrize("k", [0, 1, 2])
def test_augmentation_is_physics_consistent(root, k):
    """For 3 boards and every symmetry: the transformed solution equals the solution of the
    transformed record (float64, within 1e-3 K), and the stored float16 field transformed agrees
    with that re-solve at float16 precision."""
    row = rows(root, "train")[k]
    rec = row["record"]
    theta = solve(rec, N)
    field = stored(root, "train")[k].astype(np.float64)
    for sym in range(1, 8):
        resolved = solve(transform_record(rec, sym), N)
        assert np.abs(inp.transform_field(theta, sym) - resolved).max() <= 1e-3, sym
        f16_step = 2.0**-10 * np.abs(resolved).max()  # float16 spacing at the peak
        assert np.abs(inp.transform_field(field, sym) - resolved).max() <= f16_step, sym


def test_dataset_augmented_sample_is_the_transformed_record_and_field(root):
    ds = TwinHeatDataset(root, "train", N=N, augment=True)
    field = stored(root, "train")
    for i, sym in ((0, 3), (5, 6), (2, 0)):
        x, y = ds.sample(i, sym)
        rec = rows(root, "train")[i]["record"]
        want_x = inp.build(rasterise(transform_record(rec, sym), SPEC, N), SPEC)
        np.testing.assert_array_equal(x, want_x)
        np.testing.assert_array_equal(y, inp.transform_field(field[i].astype(np.float32), sym))
        # coordinates are regenerated, never transformed
        np.testing.assert_array_equal(x[11:], inp.coordinate_channels(N))


def test_augment_draws_a_symmetry_per_item_reproducibly(root):
    a = TwinHeatDataset(root, "train", N=N, augment=True, seed=3)
    b = TwinHeatDataset(root, "train", N=N, augment=True, seed=3)
    syms = [a.symmetry(i) for i in range(len(a))]
    assert syms == [b.symmetry(i) for i in range(len(b))]
    a.set_epoch(1)
    assert syms != [a.symmetry(i) for i in range(len(a))]
    assert set(syms) <= set(range(8))
    plain = TwinHeatDataset(root, "train", N=N, augment=False)
    assert {plain.symmetry(i) for i in range(len(plain))} == {0}


# ------------------------------------------------------------------------------ dataset

def test_shapes_and_dtypes_match_the_model_contract(root):
    """CLAUDE.md §10.2 / §7.5 (B6): input [13, N, N] float32, target [4, N, N] float32 (K)."""
    ds = TwinHeatDataset(root, "train", N=N)
    x, y = ds[0]
    assert x.shape == (13, N, N) and x.dtype == np.float32
    assert y.shape == (4, N, N) and y.dtype == np.float32
    assert len(ds) == 8


def test_half_resolution_targets_are_2x2_area_averages_and_inputs_are_re_rasterised(root):
    ds = TwinHeatDataset(root, "train", N=N // 2)
    x, y = ds.sample(4, 0)
    full = stored(root, "train")[4].astype(np.float32)
    np.testing.assert_allclose(y, full.reshape(4, N // 2, 2, N // 2, 2).mean(axis=(2, 4)),
                               rtol=1e-6)
    rec = rows(root, "train")[4]["record"]
    np.testing.assert_array_equal(x, inp.build(rasterise(rec, SPEC, N // 2), SPEC))


def test_other_resolutions_are_refused(root):
    with pytest.raises(ValueError, match="N"):
        TwinHeatDataset(root, "train", N=24)


def test_protected_splits_need_allow_eval(root):
    with pytest.raises(SplitAccessError, match="cal"):
        TwinHeatDataset(root, "cal", N=N)
    assert len(TwinHeatDataset(root, "cal", N=N, allow_eval=True)) == 4
    with pytest.raises(SplitAccessError, match="ood_3"):
        TwinHeatDataset(root, "ood_3", N=N)


def test_shard_hashes_are_verified_at_init(root, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(root / "val", bad / "val")
    p = bad / "val" / "shard_0000" / "fields.npy"
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.write_bytes(bytes(data))
    with pytest.raises(gen.ManifestError, match="shard_0000"):
        TwinHeatDataset(bad, "val", N=N)


def test_the_input_cache_is_keyed_by_record_hash_and_returns_the_same_tensor(root, tmp_path):
    cache = tmp_path / "cache"
    ds = TwinHeatDataset(root, "val", N=N, cache_dir=cache)
    x1, _ = ds.sample(1, 2)
    files = list(cache.glob("*.npy"))
    assert len(files) == 1
    rec2 = transform_record(rows(root, "val")[1]["record"], 2)
    assert files[0].name == f"{gen.record_hash(rec2)}-N{N}.npy"
    x2, _ = TwinHeatDataset(root, "val", N=N, cache_dir=cache).sample(1, 2)
    np.testing.assert_array_equal(x1, x2)


# ------------------------------------------------------------------------------ normalisation

@pytest.fixture(scope="module")
def norm(root) -> Normaliser:
    return Normaliser.fit(root, N=N)


def test_normaliser_is_fitted_on_train_only_and_records_the_dataset_hash(root, norm):
    assert norm.dataset_hash == dataset_hash(root / "train")
    assert norm.dataset_hash != dataset_hash(root / "val")
    assert norm.n_boards == 8 and norm.N == N


def test_normaliser_statistics_are_the_train_statistics(root, norm):
    xs = np.stack([inp.build(rasterise(r["record"], SPEC, N), SPEC).astype(np.float64)
                   for r in rows(root, "train")])
    for c in inp.MEAN_STD_CHANNELS:
        v = xs[:, c]
        assert norm.offset[c] == pytest.approx(v.mean(), rel=1e-9)
        assert norm.scale[c] == pytest.approx(v.std(), rel=1e-9)
    assert norm.offset[inp.Q_CHANNEL] == 0.0
    assert norm.scale[inp.Q_CHANNEL] == pytest.approx(np.percentile(xs[:, inp.Q_CHANNEL], 99),
                                                      rel=1e-9)
    for c in inp.RAW_CHANNELS:
        assert (norm.offset[c], norm.scale[c]) == (0.0, 1.0)
    theta = stored(root, "train").astype(np.float64)
    assert norm.theta_scale == pytest.approx(theta.std(), rel=1e-9)


def test_normalisation_round_trips(root, norm):
    x, y = TwinHeatDataset(root, "train", N=N)[3]
    xn = norm.normalise_x(x)
    assert xn.dtype == np.float32
    np.testing.assert_array_equal(xn[10:], x[10:])  # mask and coordinates are not normalised
    np.testing.assert_allclose(norm.denormalise_x(xn), x, rtol=1e-6, atol=1e-6)
    yn = norm.normalise_y(y)
    np.testing.assert_allclose(norm.denormalise_y(yn), y, rtol=1e-6, atol=0)
    assert norm.normalise_y(np.zeros(4)).tolist() == [0, 0, 0, 0]  # 0 K stays 0 (B4 linearity)


def test_saved_stats_load_only_for_the_same_dataset(root, norm, tmp_path):
    p = tmp_path / "norm_stats.json"
    norm.save(p)
    back = Normaliser.load(p, dataset_hash(root / "train"))
    np.testing.assert_array_equal(back.scale, norm.scale)
    assert back.theta_scale == norm.theta_scale
    with pytest.raises(NormStatsError, match="hash"):
        Normaliser.load(p, dataset_hash(root / "val"))


def test_a_constant_channel_cannot_be_standardised(root, monkeypatch):
    real = inp.build

    def flat(raster, spec):
        x = real(raster, spec)
        x[8] = 10.0  # h_top identical everywhere
        return x
    monkeypatch.setattr(inp, "build", flat)
    with pytest.raises(NormStatsError, match="h_top"):
        Normaliser.fit(root, N=N)


def test_dataset_applies_a_normaliser_when_given(root, norm):
    raw = TwinHeatDataset(root, "val", N=N)[2]
    normed = TwinHeatDataset(root, "val", N=N, normaliser=norm)[2]
    np.testing.assert_allclose(normed[0], norm.normalise_x(raw[0]), rtol=1e-6)
    np.testing.assert_allclose(normed[1], norm.normalise_y(raw[1]), rtol=1e-6)
