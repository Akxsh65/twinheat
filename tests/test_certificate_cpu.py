"""CPU certificate (Lemma 2, CLAUDE.md §8.4, §9.1; invariant B7) on the 64² plate.

The perturbations in the first part are test perturbations of our own (a smooth blob plus
noise). The §8.6 emulated-error models live in bench/prelim/stop.js and cert.js (the 5 Oct
scripts); bench/prelim/plate.py and reproduce.py port them, and the last part of this file checks
the port against core.js (when Node is installed) and against the §8.6 tables.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from bench.prelim import plate, reproduce
from bench.prelim.plate import colours as plate_colours
from certify.kappa import kappa_from_z, kappa_hat, multigrid
from certify.pointwise import gauss_seidel_smooth, peak_interval, pointwise_bound
from certify.stopping import CertificateViolation, check_bound, jacobi, pcg_certified
from solvers.fvm import assemble_plate, solve_direct
from solvers.mmatrix_test import assert_m_matrix
from tests.test_fvm_plate import BOARDS

ORACLE_RTOL = 1e-12  # the prompt's oracle: 1e-12 direct solve


def _true_inv_norm(A: sp.csr_matrix, block: int = 512) -> float:
    """‖A⁻¹‖∞ = max row sum of |A⁻¹|, by solving A X = I in column blocks (64² only)."""
    n = A.shape[0]
    lu = spla.splu(sp.csc_matrix(A))
    rows = np.zeros(n)
    for j0 in range(0, n, block):
        j1 = min(n, j0 + block)
        E = np.zeros((n, j1 - j0))
        E[np.arange(j0, j1), np.arange(j1 - j0)] = 1.0
        rows += np.abs(lu.solve(E)).sum(axis=1)
    return float(rows.max())


def _case(i: int):
    rec, (G, H, q, dx, edges) = BOARDS[i]
    A, b = assemble_plate(G, H, q, dx, edges)
    assert_m_matrix(A, context=rec["id"])
    theta = solve_direct(A, b, rtol=ORACLE_RTOL)
    return rec, A, b, theta


def _perturbed(theta: np.ndarray, level: float, seed: int) -> np.ndarray:
    """θ + a smooth blob of `level` × peak + 1 % of that as white noise (test perturbation)."""
    n = int(np.sqrt(theta.size))
    rng = np.random.Generator(np.random.PCG64(seed))
    c = (np.arange(n) + 0.5) / n
    Y, X = np.meshgrid(c, c, indexing="ij")
    cx, cy = rng.uniform(0.2, 0.8, 2)
    blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 0.1**2)).ravel()
    amp = level * theta.max()
    return theta + amp * blob + 0.01 * amp * rng.standard_normal(theta.size)


CASES = [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------- Lemma 2(a): κ̂

@pytest.mark.parametrize("i", CASES)
def test_kappa_hat_bounds_the_true_inverse_norm(i):
    rec, A, _, _ = _case(i)
    k = kappa_hat(A)
    true = _true_inv_norm(A)
    assert k is not None, rec["id"]
    print(f"\n{rec['id']} edges {rec['dirichlet_edges']}: kappa_hat / true = {k / true:.4f}")
    assert k >= true, f"{rec['id']}: kappa_hat {k} < true ‖A⁻¹‖∞ {true}"


def test_kappa_from_z_rejects_a_non_positive_z():
    _, A, _, _ = _case(0)
    z = multigrid(A).solve(np.ones(A.shape[0]), maxiter=3, tol=0.0)
    assert kappa_from_z(A, z) is not None
    for bad in (0.0, -1e-3):
        zb = z.copy()
        zb[17] = bad
        assert kappa_from_z(A, zb) is None


def test_kappa_from_z_rejects_a_non_positive_Az():
    _, A, _, _ = _case(0)
    z = np.ones(A.shape[0])
    z[100] = 50.0  # a spike makes (Az) negative at the spike's neighbours
    assert (A @ z).min() <= 0
    assert kappa_from_z(A, z) is None


def test_kappa_with_z_equal_ones_is_varah():
    """§18.5: z = 1 gives 1 / min row margin (Varah 1975) for a strictly dominant M-matrix."""
    _, A, _, _ = _case(1)
    margin = np.asarray(A.sum(axis=1)).ravel()
    assert margin.min() > 0
    k = kappa_from_z(A, np.ones(A.shape[0]))
    assert k == pytest.approx(1.0 / margin.min(), rel=2e-6)  # the (1 + 1e-6) placeholder margin


def test_kappa_hat_raises_on_wrong_input():
    with pytest.raises(ValueError, match="square"):
        kappa_hat(sp.csr_matrix(np.ones((3, 4))))
    with pytest.raises(ValueError, match="n_vcycles"):
        kappa_hat(sp.identity(4, format="csr"), n_vcycles=0)


# ---------------------------------------------------------------------- Lemma 2(b): certified PCG

@pytest.mark.parametrize("i", CASES)
@pytest.mark.parametrize("level", [0.02, 0.05])
def test_every_pcg_iterate_bound_is_above_the_true_error(i, level):
    rec, A, b, theta = _case(i)
    k = kappa_hat(A)
    x0 = _perturbed(theta, level, seed=i)
    seen = []

    def on_iterate(it, x, bound):
        err = float(np.abs(theta - x).max())
        check_bound(rec["id"], it, bound, err)  # raises CertificateViolation
        seen.append((it, bound, err))

    x, iters, cert = pcg_certified(A, b, x0, jacobi(A), k, tau_v=0.1, history=on_iterate)
    assert cert is not None and cert <= 0.1, (rec["id"], iters, cert)
    err = float(np.abs(theta - x).max())
    assert err <= cert
    assert len(seen) == iters + 1  # iterate 0 (the warm start) and every PCG step
    assert seen[-1][1] == pytest.approx(cert)


@pytest.mark.parametrize("i", [0, 2, 5])
def test_bound_is_nearly_tight_for_the_worst_case_error(i):
    """x₀ = θ − c·A⁻¹1 gives r = c·1, so ‖e‖∞ = c‖A⁻¹‖∞ and the bound κ̂·c is only κ̂/‖A⁻¹‖∞
    above the error: any under-reported bound (e.g. a lost factor) violates here."""
    rec, A, b, theta = _case(i)
    k = kappa_hat(A)
    u = solve_direct(A, np.ones(A.shape[0]), rtol=ORACLE_RTOL)  # A⁻¹1 ≥ 0
    x0 = theta - 0.01 * u
    got = {}

    def on_iterate(it, x, bound):
        err = float(np.abs(theta - x).max())
        check_bound(rec["id"], it, bound, err)
        got.setdefault(it, (bound, err))

    pcg_certified(A, b, x0, jacobi(A), k, tau_v=1e-3, maxit=5, history=on_iterate)
    bound0, err0 = got[0]
    assert err0 <= bound0 <= 1.5 * err0, (rec["id"], bound0, err0)


def test_cheap_path_returns_the_same_certificate_as_history():
    rec, A, b, theta = _case(2)
    k = kappa_hat(A)
    x0 = _perturbed(theta, 0.02, seed=9)
    x1, it1, c1 = pcg_certified(A, b, x0, jacobi(A), k, tau_v=1.0)
    x2, it2, c2 = pcg_certified(A, b, x0, jacobi(A), k, tau_v=1.0, history=lambda *a: None)
    assert it1 == it2 and c1 == pytest.approx(c2) and np.allclose(x1, x2)


def test_returns_none_when_maxit_is_hit():
    _, A, b, theta = _case(0)
    k = kappa_hat(A)
    x, iters, cert = pcg_certified(A, b, _perturbed(theta, 0.05, 1), jacobi(A), k, tau_v=1e-9,
                                   maxit=3)
    assert cert is None and iters == 3


def test_check_bound_raises_with_board_iterate_bound_and_error():
    with pytest.raises(CertificateViolation, match=r"board b7, iterate 12: bound 0\.5 < .*0\.6"):
        check_bound("b7", 12, 0.5, 0.6)
    check_bound("b7", 12, 0.6, 0.6)  # equality is not a violation


# ---------------------------------------------------------------------- Lemma 2(c): pointwise

@pytest.mark.parametrize("i", CASES)
@pytest.mark.parametrize("sweeps", [0, 5, 10])
def test_pointwise_bound_covers_every_cell_and_the_peak(i, sweeps):
    rec, A, b, theta = _case(i)
    x = gauss_seidel_smooth(A, b, _perturbed(theta, 0.03, seed=i), sweeps)
    r = b - A @ x
    w = pointwise_bound(A, r)
    err = np.abs(theta - x)
    bad = np.flatnonzero(err > w)
    assert bad.size == 0, f"{rec['id']} sweeps {sweeps}: |e| > w at {bad[:5]}"
    lo, hi = peak_interval(x, w)
    assert lo <= theta.max() <= hi


def test_gauss_seidel_lowers_the_error():
    _, A, b, theta = _case(3)
    x0 = _perturbed(theta, 0.03, seed=3)
    e0 = np.abs(theta - x0).max()
    e5 = np.abs(theta - gauss_seidel_smooth(A, b, x0, 5)).max()
    assert e5 < e0
    np.testing.assert_array_equal(gauss_seidel_smooth(A, b, x0, 0), x0)


def test_pointwise_bound_validity_is_asserted():
    """A residual the approximate solve cannot cover must raise, not return a bad w."""
    _, A, b, _ = _case(0)
    r = np.zeros(A.shape[0])
    r[5] = 1.0
    w = pointwise_bound(A, r)
    assert np.all(A @ w >= np.abs(r)) and w.min() >= 0
    with pytest.raises(ValueError, match="M-matrix"):
        pointwise_bound(-A, r)


def test_red_black_gauss_seidel_is_one_colour_then_the_other():
    """Red-black GS (what cert.js uses): after one sweep, each colour solves its own rows given
    the other colour's latest values; red (0) first."""
    _, A, b, theta = _case(1)
    n = int(np.sqrt(A.shape[0]))
    colours = plate_colours(n)
    x0 = _perturbed(theta, 0.03, seed=1)
    x1 = gauss_seidel_smooth(A, b, x0, 1, sweep="red_black", colours=colours)
    D = A.diagonal()
    R = A - sp.diags(D)
    red, black = colours == 0, colours == 1
    xr = x0.copy()
    xr[red] = ((b - R @ x0) / D)[red]
    xb = xr.copy()
    xb[black] = ((b - R @ xr) / D)[black]
    np.testing.assert_allclose(x1, xb, rtol=1e-14, atol=0)
    assert np.abs(theta - x1).max() < np.abs(theta - x0).max()


def test_red_black_gauss_seidel_rejects_a_bad_colouring():
    _, A, b, theta = _case(0)
    with pytest.raises(ValueError, match="colour"):
        gauss_seidel_smooth(A, b, theta, 1, sweep="red_black",
                            colours=np.zeros(A.shape[0], dtype=int))
    with pytest.raises(ValueError, match="colours"):
        gauss_seidel_smooth(A, b, theta, 1, sweep="red_black")


# ---------------------------------------------------------------------- §8.6 port (bench/prelim)

PRELIM = Path(__file__).resolve().parents[1] / "bench" / "prelim"

# CLAUDE.md §8.6, first table (the output of bench/prelim/stop.js)
S86_ITERS_1E8 = [170, 184, 174, 160]
S86_ITERS_CERT_1K = [32, 30, 43, 29]
# §8.6 second table (bench/prelim/cert.js): (error %, GS sweeps, true worst error K, Varah K,
# best pointwise cert K). "Best" is cert.js's idealCert, max A⁻¹|r| from an exact solve.
S86_POINTWISE = [(1, 0, 1.26, 350, "40.0"), (1, 5, 0.81, 24, "0.98"), (1, 10, 0.78, 13, "0.82"),
                 (3, 0, 2.95, 1200, "125"), (3, 5, 1.87, 40, "3.16"), (3, 10, 1.69, 35, "2.56")]


def _as_printed(value: float, printed: str) -> bool:
    """value rounds to `printed` at the number of decimals §8.6 prints."""
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return abs(value - float(printed)) <= 0.5 * 10.0**-decimals


def _node_json(script: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed: cannot run bench/prelim/core.js")
    out = subprocess.run([node, "-e", script], cwd=PRELIM, capture_output=True, text=True,
                         check=True, timeout=120)
    return json.loads(out.stdout)


def test_rng_and_gauss_match_core_js():
    js = _node_json("""
        const TH = require('./core.js');
        const out = {};
        for (const seed of [0, 3, 11, 4294967295]) { const R = TH.rng(seed);
          out['u' + seed] = Array.from({length: 2000}, () => R()); }
        const G = TH.rng(11); out.gauss = Array.from({length: 500}, () => TH.gauss(G));
        console.log(JSON.stringify(out));""")
    for seed in (0, 3, 11, 4294967295):
        R = plate.rng(seed)
        assert [R() for _ in range(2000)] == js[f"u{seed}"], f"seed {seed}"
    G = plate.rng(11)
    np.testing.assert_allclose([plate.gauss(G) for _ in range(500)], js["gauss"],
                               rtol=1e-15, atol=0)


def test_assembled_plates_match_core_js_bit_for_bit():
    """The 4 stop.js variants: diag, east/north conductances and b identical to core.js."""
    js = _node_json("""
        const TH = require('./core.js');
        const v = [TH.defaultBoard(), TH.defaultBoard(), TH.defaultBoard(), TH.defaultBoard()];
        v[1].edges = [false, false, false, false];
        v[2].chips.push({ id: 'U4', name: 'U4', x: 8, y: 56, w: 6, h: 6, P: 0.6 });
        v[3].cuOz = 0.5; v[3].hBase = 12;
        console.log(JSON.stringify(v.map(b => { const s = b.assemble();
          return { diag: Array.from(s.diag), gE: Array.from(s.gE), gN: Array.from(s.gN),
                   b: Array.from(s.b), Hd: Array.from(s.Hd), dterm: Array.from(s.dterm) }; })));""")
    for k, (variant, ref) in enumerate(zip(reproduce.STOP_VARIANTS, js, strict=True)):
        s = variant.board().assemble()
        for name in ("diag", "gE", "gN", "b", "Hd", "dterm"):
            np.testing.assert_array_equal(getattr(s, name), ref[name], err_msg=f"{k} {name}")


def test_red_black_smoothing_matches_core_js():
    js = _node_json("""
        const TH = require('./core.js');
        const s = TH.defaultBoard().assemble(), R = TH.rng(5);
        const x0 = Array.from({length: s.n}, () => 50 * R());
        const st = TH.sorStepper(s, x0, 1.0); for (let k = 0; k < 5; k++) st.step();
        console.log(JSON.stringify({x0, x5: Array.from(st.x)}));""")
    s = plate.default_board().assemble()
    x5 = gauss_seidel_smooth(s.A, s.b, np.array(js["x0"]), 5, sweep="red_black",
                             colours=plate_colours(s.N))
    np.testing.assert_allclose(x5, js["x5"], rtol=1e-12, atol=0)


def test_warm_starts_match_the_js_scripts():
    """Run the real stop.js and cert.js under Node, capture the fields they hand to the solvers
    (wrapping core.js functions on the shared module object), and compare with the port.
    The scripts' θ comes from PCG to 1e-13, ours from a direct solve: hence rtol 1e-9."""
    js = _node_json("""
        const TH = require('./core.js');
        const out = { stop: [], cert: [] }, STOP = new Error('captured');
        const pcgStepper = TH.pcgStepper, sorStepper = TH.sorStepper;
        console.table = () => {}; console.log = () => {};
        TH.pcgStepper = (s, mu) => { out.stop.push(Array.from(mu)); return pcgStepper(s, mu); };
        require('./stop.js');
        TH.sorStepper = (s, mu, w) => { out.cert.push(Array.from(mu));
          if (out.cert.length === 8) throw STOP; return sorStepper(s, mu, w); };
        try { require('./cert.js'); } catch (e) { if (e !== STOP) throw e; }
        process.stdout.write(JSON.stringify(out));""")
    assert len(js["stop"]) == len(reproduce.STOP_VARIANTS) and len(js["cert"]) == 8
    for v, mu_js in zip(reproduce.STOP_VARIANTS, js["stop"], strict=True):
        s = v.board().assemble()
        xs = solve_direct(s.A, s.b, rtol=ORACLE_RTOL)
        mu = reproduce.stop_warm_start(xs, s.N, v.err_level, v.seed)
        np.testing.assert_allclose(mu, mu_js, rtol=1e-9, atol=0, err_msg=v.label)
    board = plate.default_board()
    s = board.assemble()
    xs = solve_direct(s.A, s.b, rtol=ORACLE_RTOL)
    R = plate.rng(reproduce.POINTWISE_SEED)
    k = 0
    for level in reproduce.POINTWISE_LEVELS:
        e0 = reproduce.make_err(R, level, board.phi, float(xs.max()), s.N)
        for _ in reproduce.POINTWISE_SWEEPS:  # cert.js smooths a fresh copy of θ + e0 per row
            np.testing.assert_allclose(xs + e0, js["cert"][k], rtol=1e-9, atol=0,
                                       err_msg=f"cert.js level {level} row {k}")
            k += 1


@pytest.fixture(scope="module")
def stop_rows():
    return reproduce.run_stop()  # raises CertificateViolation on any iterate (invariant B7)


def test_stop_port_reproduces_the_warm_starts(stop_rows):
    """Iterations to 1e-8 depend only on the board, the warm start and Jacobi PCG: equal to
    stop.js's means the emulated error (formula, seed, RNG order) is ported exactly."""
    assert [r["iters_to_1e8"] for r in stop_rows] == S86_ITERS_1E8


def test_stop_port_meets_the_n13_done_criteria(stop_rows):
    for r, ref in zip(stop_rows, S86_ITERS_CERT_1K, strict=True):
        assert r["min_Az_positive"]
        assert 1.0 <= r["kappa_over_true"] <= 1.07, r
        assert abs(r["iters_cert_1K"] - ref) <= 0.1 * ref, r
        assert r["violations"] == 0
        assert r["iterates_checked"] >= r["iters_to_1e8"]


def test_stop_port_uses_pcg_certified(stop_rows):
    """The per-iterate scan and certify.stopping.pcg_certified stop at the same iterate."""
    for r in stop_rows:
        assert r["pcg_certified_iters_1K"] == r["iters_cert_1K"], r
        assert r["pcg_certified_iters_0_1K"] == r["iters_cert_0_1K"], r


@pytest.fixture(scope="module")
def pointwise_rows():
    return reproduce.run_pointwise()


def test_pointwise_port_reproduces_section_8_6(pointwise_rows):
    rows = {(r["err_level_pct"], r["gs_sweeps"]): r for r in pointwise_rows}
    for level, sweeps, true_err, varah, best in S86_POINTWISE:
        r = rows[(level, sweeps)]
        assert round(r["true_max_err_K"], 2) == true_err, r
        assert float(f"{r['varah_K']:.2g}") == varah, r
        assert _as_printed(r["ideal_cert_K"], best), r


def test_pointwise_port_certificates_are_valid(pointwise_rows):
    for r in pointwise_rows:
        assert r["cert_valid"], r  # |e| <= w in every cell, checked against the oracle
        assert r["cert_4v_K"] >= r["ideal_cert_K"] * (1 - 1e-9), r  # ideal: best of its type
        assert r["peak_lo_K"] <= r["true_peak_K"] <= r["peak_hi_K"], r
