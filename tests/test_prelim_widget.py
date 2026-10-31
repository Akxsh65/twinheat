"""N's solver and certify/ against the 5 Oct explainer widget (bench/prelim/README.md).

Expected values are read from bench/prelim/explainer_cert_widget_results.csv (the JS output), not
typed in. Same board, same emulated prediction (bit-exact RNG port), same Jacobi-PCG: iteration
counts must match the JS exactly when given the widget's κ̂, and every bound must cover the error.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from bench.prelim.widget_board import CHIPS as CHIPS_DEFAULT
from bench.prelim.widget_board import EDGES, board_fields, emulated_prediction, xorshift32
from certify.kappa import kappa_hat
from certify.stopping import jacobi, pcg_certified
from solvers.fvm import assemble_plate
from solvers.mmatrix_test import assert_m_matrix

ROOT = Path(__file__).resolve().parents[1]
ROWS = list(csv.DictReader(open(ROOT / "bench/prelim/explainer_cert_widget_results.csv",
                                newline="", encoding="utf-8")))


@pytest.fixture(scope="module")
def system():
    phi, G, H, q, dx = board_fields()
    A, b = assemble_plate(G, H, q, dx, EDGES)
    lu = spla.splu(A.tocsc())
    theta, inv1 = lu.solve(b), lu.solve(np.ones(A.shape[0]))
    return phi, A, b, theta, float(inv1.max())


def _jacobi_pcg_iters(A, b, x0, tol=1e-8, maxit=4000) -> int:
    """The widget's TH.pcg counter: Jacobi-PCG until ‖r‖₂/‖b‖₂ <= tol."""
    d = A.diagonal()
    x = x0.copy()
    r = b - A @ x
    z = r / d
    p, rz, bn, it = z.copy(), r @ z, np.linalg.norm(b), 0
    while np.linalg.norm(r) / bn > tol and it < maxit:
        Ap = A @ p
        a = rz / (p @ Ap)
        x += a * p
        r -= a * Ap
        it += 1
        z = r / d
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return it


def test_rng_port_is_bit_exact_xorshift32() -> None:
    # first three uint32 states of TH.rng(7), printed by node v22.19.0 on 2026-10-30
    r = xorshift32(7)
    assert [round(r() * 4294967296) for _ in range(3)] == [1892583, 470389255, 3882205507]


def test_same_matrix_as_the_widget(system) -> None:
    _, A, _, theta, true_inv = system
    assert_m_matrix(A)
    row = ROWS[0]
    assert theta.max() == pytest.approx(float(row["peak_K"]), abs=0.005)
    assert true_inv == pytest.approx(float(row["trueInv"]), abs=0.5)


def test_our_kappa_hat_is_a_valid_and_tighter_bound(system) -> None:
    _, A, _, _, true_inv = system
    k = kappa_hat(A)
    assert k is not None and true_inv <= k <= float(ROWS[0]["kappa_ratio"]) * true_inv


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"lvl{r['lvl']}-tau{r['tau']}")
def test_iterations_match_the_widget_and_bounds_hold(system, row) -> None:
    phi, A, b, theta, true_inv = system
    mu = emulated_prediction(theta, phi, float(row["lvl"]))
    assert _jacobi_pcg_iters(A, b, mu) == int(row["it_1e-8"])
    kappa = float(row["kappa"]) / float(row["trueInv"]) * true_inv  # the widget's κ̂ (CSV rounds it)
    covered: list[bool] = []
    _, it, cert = pcg_certified(
        A, b, mu, jacobi(A), kappa, float(row["tau"]),
        history=lambda k, x, c: covered.append(c >= np.abs(x - theta).max() * (1 - 1e-9)))
    assert it == int(row["it_cert"]) and cert is not None and cert <= float(row["tau"])
    assert all(covered), "a certified bound fell below the true error (B7)"


def test_milder_prediction_has_the_declared_error_level(system) -> None:
    from bench.prelim.reproduce import VARIANTS, milder_prediction
    from bench.prelim.widget_board import CHIPS

    phi, _, _, theta, _ = system
    for lvl in (0.01, 0.02, 0.05):
        mu = milder_prediction(theta, phi, CHIPS, lvl)
        assert np.abs(mu - theta).max() == pytest.approx(lvl * theta.max(), rel=1e-12)
        assert np.array_equal(mu, milder_prediction(theta, phi, CHIPS, lvl))  # deterministic
    assert set(VARIANTS) == {"default_one_heat_sunk_edge", "no_heat_sunk_edge",
                             "four_chips_one_near_a_corner", "thin_copper_weak_convection"}


def test_widget_board_defaults_are_the_widget_board(system) -> None:
    from bench.prelim.widget_board import board_fields as bf

    phi, A, *_ = system
    p2, G, H, q, dx = bf(chips=CHIPS_DEFAULT, cu_oz=2, h_base=24.0)
    A2, _ = assemble_plate(G, H, q, dx, EDGES)
    assert np.array_equal(phi, p2) and (A != A2).nnz == 0
