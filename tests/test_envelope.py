"""Envelope rule (prompt M1.2; CLAUDE.md §6.5, §7.4): max θ' <= 100 K, θ'/θ == P'/P."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.envelope import THETA_MAX, apply_envelope
from data.raster import rasterise
from data.sampler import sample_board
from solvers.fvm import assemble_board, solve_direct
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")


def _field(seed: int, peak: float) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    theta = rng.random((4, 16, 16))
    return theta / theta.max() * peak


def test_theta_max_is_100_k_from_the_spec() -> None:
    assert THETA_MAX == 100.0


@pytest.mark.parametrize("peak", [100.0 + 1e-6, 150.0, 1234.5, 1e6])
def test_scaled_boards_meet_the_cap_and_scale_theta_like_power(peak: float) -> None:
    for seed in range(20):
        rec = sample_board(seed, "id")
        theta = _field(seed, peak)
        rec2, theta2, scaled = apply_envelope(rec, theta)
        assert scaled
        assert theta2.max() <= THETA_MAX + 1e-9
        s = theta2.max() / theta.max()
        np.testing.assert_allclose(theta2 / theta, s, rtol=1e-12)
        for c, c2 in zip(rec["chips"], rec2["chips"], strict=True):
            assert c2["P"] / c["P"] == pytest.approx(s, rel=1e-12)


@pytest.mark.parametrize("peak", [0.0, 1.0, 99.9, 100.0])
def test_boards_under_the_cap_are_unchanged_copies(peak: float) -> None:
    rec = sample_board(3, "id")
    theta = _field(3, peak) if peak > 0 else np.zeros((4, 8, 8))
    rec2, theta2, scaled = apply_envelope(rec, theta)
    assert not scaled and rec2 == rec and np.array_equal(theta2, theta)
    assert rec2 is not rec and theta2 is not theta
    rec2["chips"][0]["P"] = -1.0
    assert rec["chips"][0]["P"] > 0  # deep copy: the input record is never edited


def test_input_record_is_not_edited_when_scaled() -> None:
    rec = sample_board(4, "id")
    before = [c["P"] for c in rec["chips"]]
    apply_envelope(rec, _field(4, 500.0))
    assert [c["P"] for c in rec["chips"]] == before


def test_bottom_chips_are_scaled_too() -> None:
    rec = sample_board(5, "ood_bottom_chips")
    rec2, _, scaled = apply_envelope(rec, _field(5, 200.0))
    assert scaled
    assert all(c2["P"] == pytest.approx(c["P"] / 2, rel=1e-12)
               for c, c2 in zip(rec["chips"], rec2["chips"], strict=True))


def test_rescaled_record_solves_to_the_rescaled_field() -> None:
    """Linearity (B4) is what makes the rule exact: re-solving the scaled record gives θ'."""
    rec = sample_board(1, "id")
    n = 32

    def solve(r: dict) -> np.ndarray:
        A, b = assemble_board(SPEC, rasterise(r, SPEC, n), r["dirichlet_edges"])
        return solve_direct(A, b)

    theta = solve(rec)
    rec2, theta2, scaled = apply_envelope(rec, theta, max_rise=theta.max() / 3)
    assert scaled
    np.testing.assert_allclose(solve(rec2), theta2, rtol=1e-9, atol=1e-12 * theta.max())


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf]), np.array([])])
def test_bad_theta_raises(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="theta"):
        apply_envelope(sample_board(0, "id"), bad)


def test_negative_peak_raises() -> None:
    with pytest.raises(ValueError, match="rise"):
        apply_envelope(sample_board(0, "id"), -np.ones(4))


@pytest.mark.parametrize("cap", [0.0, -5.0, np.inf, np.nan])
def test_bad_cap_raises(cap: float) -> None:
    with pytest.raises(ValueError, match="max_rise"):
        apply_envelope(sample_board(0, "id"), np.ones(4), max_rise=cap)
