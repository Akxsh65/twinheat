"""Solver baseline and benchmark (N1.5; CLAUDE.md §7.3, §11.5 E3).

TESTS FIRST criterion: every solver in the table agrees with SciPy's direct solve to 1e-6
relative on the 20 fixed boards. At 256² × 4 that is the slow test (SciPy's direct solve takes
about 100 s per board on one core); bench/solver_bench.py enforces the same check on every row it
writes. Jacobi and red-black GS cannot reach 1e-8 at 256² in a usable number of sweeps (κ₂ ≈ 1e5,
bench/results/conditioning.csv), so their agreement is tested to convergence on 16² × 4 boards,
where the same code paths run in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bench import solver_bench as sb
from solvers.amg_baseline import AMGNotConverged, AMGResult, amg_solve

AGREE_RTOL = 1e-6  # the N1.5 prompt
TOL = 1e-8  # PyAMG-GMRES tolerance (the N1.5 prompt), relative true residual


def relres(A: sp.csr_matrix, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))


# ------------------------------------------------------------------------------ amg_solve

@pytest.mark.parametrize("accel", ["gmres", "cg"])
def test_amg_solve_meets_the_true_residual_tolerance(accel):
    """PyAMG's GMRES stops on a preconditioned norm (about 10x above the true residual here);
    amg_solve must keep going until the TRUE relative residual is <= tol."""
    case = sb.fixed_board(0, 64)
    res = amg_solve(case.A, case.b, tol=TOL, accel=accel)
    assert isinstance(res, AMGResult)
    assert relres(case.A, res.theta, case.b) <= TOL
    assert res.relres <= TOL and res.iters > 0
    assert res.seconds >= res.setup_seconds > 0 and res.seconds >= res.solve_seconds > 0
    assert np.abs(res.theta - case.theta_direct).max() <= AGREE_RTOL * np.abs(
        case.theta_direct).max()


def test_amg_solve_strength_option_changes_the_hierarchy():
    case = sb.fixed_board(1, 64)
    a = amg_solve(case.A, case.b, tol=TOL)
    t = amg_solve(case.A, case.b, tol=TOL, strength_theta=0.1)
    assert a.levels != t.levels or a.iters != t.iters
    assert relres(case.A, t.theta, case.b) <= TOL


def test_amg_solve_raises_when_maxiter_is_too_small():
    case = sb.fixed_board(0, 64)
    with pytest.raises(AMGNotConverged, match="maxiter"):
        amg_solve(case.A, case.b, tol=1e-12, maxiter=2)


@pytest.mark.parametrize("bad", ["zero_b", "nan_b", "shape", "tol"])
def test_amg_solve_rejects_bad_input(bad):
    case = sb.fixed_board(0, 32)
    A, b, kw = case.A, case.b.copy(), {"tol": TOL}
    if bad == "zero_b":
        b[:] = 0
    elif bad == "nan_b":
        b[3] = np.nan
    elif bad == "shape":
        b = b[:-1]
    elif bad == "tol":
        kw["tol"] = 0.0
    with pytest.raises(ValueError):
        amg_solve(A, b, **kw)


# ------------------------------------------------------------------------ agreement (E3)

def test_warm_start_is_exact_plus_two_percent_smooth_noise():
    case = sb.fixed_board(2, 32)
    x0 = sb.warm_start(case.theta_direct, case.n, seed=2)
    d = x0 - case.theta_direct
    assert np.isclose(np.abs(d).max(), sb.WARM_NOISE * case.theta_direct.max(), rtol=1e-12)
    # smooth: neighbouring cells differ by a small fraction of the amplitude
    grid = d.reshape(4, case.n, case.n)
    assert np.abs(np.diff(grid, axis=2)).max() <= 0.5 * np.abs(d).max()
    np.testing.assert_array_equal(x0, sb.warm_start(case.theta_direct, case.n, seed=2))


@pytest.mark.parametrize("seed", range(sb.N_BOARDS))
def test_all_cpu_solvers_agree_on_small_boards(seed):
    """Every CPU solver, stationary ones run to convergence, on the 20 fixed boards at 16² × 4."""
    case = sb.fixed_board(seed, 16)
    rows = sb.run_cpu(case, sweeps=None, timing_reps=1)
    assert {r["solver"] for r in rows} == set(sb.CPU_SOLVERS)
    for r in rows:
        assert r["converged"], r
        assert r["err_vs_direct"] <= AGREE_RTOL, r
        assert r["relres_true"] <= TOL, r


def test_capped_stationary_rows_say_they_did_not_converge():
    case = sb.fixed_board(0, 32)
    rows = {r["solver"]: r for r in sb.run_cpu(case, sweeps=5, timing_reps=1)}
    for name in ("jacobi", "rb_gs"):
        assert rows[name]["iters"] == 5 and not rows[name]["converged"], rows[name]
        assert rows[name]["relres_true"] > TOL
    assert rows["pcg_warm"]["converged"] and rows["pcg_cold"]["converged"]


def test_a_disagreeing_solver_raises():
    case = sb.fixed_board(0, 16)
    with pytest.raises(sb.SolverDisagreement, match="board"):
        sb.check_agreement(case, "fake", case.theta_direct * (1 + 2 * AGREE_RTOL), True)


def test_rows_carry_hardware_and_provenance():
    case = sb.fixed_board(0, 16)
    row = sb.run_cpu(case, sweeps=3, timing_reps=1)[0]
    for key in ("cpu_model", "cores", "threads", "gpu_model", "commit", "date", "board_id",
                "h2d_s", "d2h_s"):
        assert key in row, key
    assert row["cpu_model"] and row["cores"] >= 1


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(sb.N_BOARDS))
def test_converging_solvers_agree_at_full_size(seed):
    """The prompt's test at 256² × 4: direct, PyAMG-GMRES (both settings), PCG cold and warm."""
    case = sb.fixed_board(seed, sb.N_FULL)
    rows = sb.run_cpu(case, sweeps=2, timing_reps=1)
    for r in rows:
        if r["solver"] not in ("jacobi", "rb_gs"):
            assert r["converged"] and r["err_vs_direct"] <= AGREE_RTOL, r
