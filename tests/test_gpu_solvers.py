"""GPU stencil kernels and PCG for the 4-layer operator (N1.4; CLAUDE.md §9.1, §9.4, B3).

Three groups:
  CPU  solvers/pcg.py on NumPy arrays (pcg.py is array-module generic). Runs in CI.
  gpu  solvers/cuda_kernels.py (Numba) against SciPy. Also runs on a laptop without a GPU under
       Numba's CUDA simulator, with small grids:
         NUMBA_ENABLE_CUDASIM=1 TWINHEAT_GPU_TEST_N=20 pytest -m gpu -k "not cupy"
  gpu  solvers/pcg.py on CuPy with the Numba stencil operator ("cupy" in the name). Kaggle T4.

Field arrays are [B, layer, y, x] (rule 7); the stencil coefficients are solvers/fvm.py
coefficients(). Tolerances: 1e-6 relative is B3's; float64 kernels are held to 1e-12.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from data.raster import rasterise
from data.sampler import sample_board
from solvers.fvm import assemble_board, coefficients
from solvers.pcg import (
    PCGBreakdown,
    PCGNotConverged,
    batched_pcg,
    jacobi,
    line_z_block_jacobi,
    pcg,
)
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N_FULL = SPEC.geometry.N.value  # 256
B3_RTOL = 1e-6  # CLAUDE.md §5 B3
FP64_RTOL = 1e-12
N_BOARDS = 20  # B3: 20 fixed boards
# GPU grid sizes: the real 256 and 100 (a multiple of 4 for the rasteriser, not of the 16-cell
# tile, so the last tile is partial). The simulator run passes a small N instead.
GPU_SIZES = [int(s) for s in os.environ.get("TWINHEAT_GPU_TEST_N", f"{N_FULL},100").split(",")]
CPU_N = 32
SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
FIELDS = ("Ge", "Gw", "Gn", "Gs", "gup", "gdn", "diag")


@functools.cache
def board(seed: int, n: int, edges: tuple[str, ...] | None = None):
    """(coefficients dict, A CSR, b [4, n, n]) for sampler board `seed` at n x n."""
    rec = sample_board(seed, "id")
    e = rec["dirichlet_edges"] if edges is None else list(edges)
    r = rasterise(rec, SPEC, n)
    c = coefficients(SPEC, r, e)
    A, b = assemble_board(SPEC, r, e)
    return c, A, b.reshape(4, n, n)


def exact(seed: int, n: int) -> np.ndarray:
    _, A, b = board(seed, n)
    return spla.spsolve(A.tocsc(), b.ravel()).reshape(b.shape)


def scipy_op(A: sp.csr_matrix, dtype=np.float64):
    """Aop on [.., 4, n, n] NumPy arrays (any leading batch axes), in `dtype`."""
    Ad = A.astype(dtype)

    def Aop(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(-1, A.shape[0])
        return np.ascontiguousarray((Ad @ flat.T).T).reshape(x.shape).astype(dtype, copy=False)
    return Aop


def batched_scipy_op(As: list[sp.csr_matrix], dtype=np.float64):
    """Aop_b for boards with different operators: x [B, 4, n, n]."""
    ops = [scipy_op(A, dtype) for A in As]
    return lambda x: np.stack([op(xi) for op, xi in zip(ops, x, strict=True)])


def stack(cs: list[dict], key: str, dtype=np.float64) -> np.ndarray:
    return np.stack([c[key] for c in cs]).astype(dtype)


def relerr(a: np.ndarray, ref: np.ndarray) -> float:
    """Normwise relative error ‖a − ref‖∞ / ‖ref‖∞."""
    return float(np.abs(np.asarray(a, np.float64) - ref).max() / np.abs(ref).max())


# =============================================================================== CPU (CI)
# ------------------------------------------------------------------------- preconditioners

def column_block(A: sp.csr_matrix, n: int) -> sp.csr_matrix:
    """The part of A that couples cells in the same (y, x) column: what line_z inverts."""
    C = A.tocoo()
    same = (C.row % (n * n)) == (C.col % (n * n))
    return sp.csr_matrix((C.data[same], (C.row[same], C.col[same])), shape=A.shape)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_line_z_inverts_the_column_blocks_exactly(seed):
    c, A, _ = board(seed, CPU_N)
    Minv = line_z_block_jacobi(c["diag"], c["gup"])
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((4, CPU_N, CPU_N))
    Bu = (column_block(A, CPU_N) @ u.ravel()).reshape(u.shape)
    assert relerr(Minv(Bu), u) <= FP64_RTOL


def test_line_z_and_jacobi_are_spd_and_batch_aware():
    """PCG needs M⁻¹ symmetric positive definite (CLAUDE.md §18.5); a [B, ...] input must give
    the same per-board result as single boards."""
    cs = [board(s, CPU_N)[0] for s in (0, 1)]
    rng = np.random.default_rng(7)
    u, v = rng.standard_normal((2, 2, 4, CPU_N, CPU_N))
    for make in (lambda d, g: jacobi(d), line_z_block_jacobi):
        Mb = make(stack(cs, "diag"), stack(cs, "gup"))
        for k, c in enumerate(cs):
            M1 = make(c["diag"], c["gup"])
            assert relerr(Mb(u)[k], M1(u[k])) <= FP64_RTOL
            assert abs(np.vdot(M1(u[k]), v[k]) - np.vdot(u[k], M1(v[k]))) <= 1e-12 * abs(
                np.vdot(M1(u[k]), v[k]))
            assert np.vdot(u[k], M1(u[k])) > 0


def test_line_z_rejects_a_coupling_off_the_bottom_layer():
    c, _, _ = board(0, CPU_N)
    gup = c["gup"].copy()
    gup[-1, 3, 4] = 1e-3
    with pytest.raises(ValueError, match="gup"):
        line_z_block_jacobi(c["diag"], gup)


# ------------------------------------------------------------------------------- pcg (NumPy)

@pytest.mark.parametrize("precond", ["jacobi", "line_z"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pcg_matches_spsolve(seed, precond):
    c, A, b = board(seed, CPU_N)
    Minv = jacobi(c["diag"]) if precond == "jacobi" else line_z_block_jacobi(c["diag"], c["gup"])
    x, iters = pcg(scipy_op(A), b, np.zeros_like(b), Minv, tol=1e-10)
    assert relerr(x, exact(seed, CPU_N)) <= B3_RTOL
    assert 0 < iters < 20000
    r = b - (A @ x.ravel()).reshape(b.shape)
    assert np.linalg.norm(r) <= 1e-10 * np.linalg.norm(b)  # true residual, not the recursive one


def test_line_z_needs_no_more_iterations_than_jacobi():
    c, A, b = board(1, CPU_N)
    _, it_j = pcg(scipy_op(A), b, np.zeros_like(b), jacobi(c["diag"]), tol=1e-10)
    _, it_z = pcg(scipy_op(A), b, np.zeros_like(b), line_z_block_jacobi(c["diag"], c["gup"]),
                  tol=1e-10)
    assert it_z <= it_j


def test_warm_start_from_the_exact_solution_takes_at_most_one_iteration():
    c, A, b = board(0, CPU_N)
    x, iters = pcg(scipy_op(A), b, exact(0, CPU_N), jacobi(c["diag"]), tol=1e-8)
    assert iters <= 1
    assert relerr(x, exact(0, CPU_N)) <= 1e-12


def test_warm_start_beats_cold_start():
    c, A, b = board(2, CPU_N)
    rng = np.random.default_rng(2)
    x0 = exact(2, CPU_N) * (1 + 0.02 * rng.standard_normal(b.shape))  # 2 % emulated error
    _, cold = pcg(scipy_op(A), b, np.zeros_like(b), jacobi(c["diag"]), tol=1e-10)
    _, warm = pcg(scipy_op(A), b, x0, jacobi(c["diag"]), tol=1e-10)
    assert warm < cold


def test_float32_with_fp64_check_never_returns_a_falsely_converged_field():
    """CLAUDE.md §18.5: in FP32 the recursive residual keeps falling after the true residual has
    stalled. Asking FP32 iterations for a 1e-12 FP64 residual must raise, not return."""
    c, A, b = board(0, CPU_N)
    with pytest.raises(PCGNotConverged, match="stalled") as ei:
        pcg(scipy_op(A, np.float32), b.astype(np.float32), np.zeros(b.shape, np.float32),
            jacobi(c["diag"].astype(np.float32)), tol=1e-12, Aop_check=scipy_op(A))
    e = ei.value
    assert e.relres > 1e-12
    assert e.x.shape == b.shape


def test_float32_with_fp64_check_meets_an_achievable_tolerance():
    c, A, b = board(0, CPU_N)
    x, _ = pcg(scipy_op(A, np.float32), b.astype(np.float32), np.zeros(b.shape, np.float32),
               jacobi(c["diag"].astype(np.float32)), tol=1e-5, Aop_check=scipy_op(A))
    assert x.dtype == np.float32
    r = b - (A @ x.astype(np.float64).ravel()).reshape(b.shape)
    assert np.linalg.norm(r) <= 1e-5 * np.linalg.norm(b)  # certified by the FP64 check


def test_pcg_raises_at_maxit_with_the_iterate():
    c, A, b = board(0, CPU_N)
    with pytest.raises(PCGNotConverged, match="maxit") as ei:
        pcg(scipy_op(A), b, np.zeros_like(b), jacobi(c["diag"]), tol=1e-10, maxit=3)
    assert ei.value.iters == 3
    assert ei.value.relres > 1e-10


def test_pcg_detects_a_preconditioner_that_is_not_positive_definite():
    c, A, b = board(0, CPU_N)
    with pytest.raises(PCGBreakdown, match="M"):
        pcg(scipy_op(A), b, np.zeros_like(b), lambda r: -r / c["diag"], tol=1e-10)


def test_pcg_detects_an_operator_that_is_not_positive_definite():
    c, A, b = board(0, CPU_N)
    with pytest.raises(PCGBreakdown, match="A"):
        pcg(lambda x: -scipy_op(A)(x), b, np.zeros_like(b), jacobi(c["diag"]), tol=1e-10)


@pytest.mark.parametrize("bad", ["zero_b", "shape", "tol", "maxit", "nan"])
def test_pcg_rejects_bad_inputs(bad):
    c, A, b = board(0, CPU_N)
    x0 = np.zeros_like(b)
    kw = {"tol": 1e-8, "maxit": 100}
    if bad == "zero_b":
        b = np.zeros_like(b)
    elif bad == "shape":
        x0 = np.zeros((4, CPU_N, CPU_N + 1))
    elif bad == "tol":
        kw["tol"] = 0.0
    elif bad == "maxit":
        kw["maxit"] = 0
    elif bad == "nan":
        x0 = x0.copy()
        x0[1, 2, 3] = np.nan
    with pytest.raises(ValueError):
        pcg(scipy_op(A), b, x0, jacobi(c["diag"]), **kw)


# ------------------------------------------------------------------------ batched_pcg (NumPy)

def test_batched_pcg_equals_single_solves():
    seeds = list(range(6))
    cs, As, bs = zip(*(board(s, CPU_N) for s in seeds), strict=True)
    b = np.stack(bs)
    xb, itb = batched_pcg(batched_scipy_op(list(As)), b, np.zeros_like(b),
                          jacobi(stack(list(cs), "diag")), tol=1e-10)
    for k, s in enumerate(seeds):
        x1, it1 = pcg(scipy_op(As[k]), bs[k], np.zeros_like(bs[k]), jacobi(cs[k]["diag"]),
                      tol=1e-10)
        assert itb[k] == it1, f"board {s}"
        assert relerr(xb[k], x1) <= 1e-9
        assert relerr(xb[k], exact(s, CPU_N)) <= B3_RTOL
    assert len(set(itb.tolist())) > 1  # boards really stopped at different iterations


def test_batched_pcg_freezes_converged_boards():
    """A board that starts converged keeps its x0 bit for bit while the others iterate."""
    cs, As, bs = zip(*(board(s, CPU_N) for s in (0, 1, 2)), strict=True)
    b = np.stack(bs)
    x0 = np.zeros_like(b)
    x0[1] = exact(1, CPU_N)
    xb, itb = batched_pcg(batched_scipy_op(list(As)), b, x0, jacobi(stack(list(cs), "diag")),
                          tol=1e-8)
    assert itb[1] == 0
    np.testing.assert_array_equal(xb[1], x0[1])
    assert itb[0] > 0 and itb[2] > 0


def test_batched_pcg_names_the_boards_that_did_not_converge():
    cs, As, bs = zip(*(board(s, CPU_N) for s in (0, 1)), strict=True)
    b = np.stack(bs)
    x0 = np.zeros_like(b)
    x0[0] = exact(0, CPU_N)
    with pytest.raises(PCGNotConverged, match=r"\[1\]"):
        batched_pcg(batched_scipy_op(list(As)), b, x0, jacobi(stack(list(cs), "diag")),
                    tol=1e-10, maxit=2)


# =============================================================================== GPU (Numba)

@pytest.fixture(scope="module")
def ck():
    """solvers.cuda_kernels with a usable CUDA device (real or the Numba simulator)."""
    numba_cuda = pytest.importorskip("numba.cuda")
    if not numba_cuda.is_available():
        pytest.skip("no CUDA device (set NUMBA_ENABLE_CUDASIM=1 to use the simulator)")
    from solvers import cuda_kernels
    return cuda_kernels


def host(a) -> np.ndarray:
    return a.copy_to_host() if hasattr(a, "copy_to_host") else np.asarray(a)


def edge_case_boards(n: int) -> list[tuple[int, tuple[str, ...] | None]]:
    """B3's 20 fixed boards: sampler seeds 0..17 with their own edges, plus seed 0 with all four
    edges heat-sunk and seed 1 with two adjacent ones (every boundary branch is exercised)."""
    return [(s, None) for s in range(N_BOARDS - 2)] + [(0, ("W", "E", "S", "N")), (1, ("S", "W"))]


def test_the_20_boards_cover_0_1_2_and_4_dirichlet_edges():
    counts = {len(sample_board(s, "id")["dirichlet_edges"]) for s, e in edge_case_boards(8)
              if e is None}
    assert {0, 1, 2} <= counts


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["naive", "tiled"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", GPU_SIZES)
def test_stencil_matches_scipy_on_20_boards(ck, kernel, dtype, n):
    """B3 for the operator: A x on the GPU == SciPy A @ x, rel 1e-6 (FP32) / 1e-12 (FP64)."""
    apply = getattr(ck, f"stencil_apply_{kernel}")
    tol = B3_RTOL if dtype == np.float32 else FP64_RTOL
    rng = np.random.default_rng(n)
    for seed, edges in edge_case_boards(n):
        c, A, _ = board(seed, n, edges)
        x = rng.uniform(0.0, 100.0, (4, n, n)).astype(dtype)
        dc = ck.upload_coefficients(c, dtype)
        y = host(apply(dc, ck.cuda.to_device(x)))
        ref = (A @ x.astype(np.float64).ravel()).reshape(x.shape)
        assert y.shape == x.shape and y.dtype == dtype
        assert relerr(y, ref) <= tol, f"seed {seed} edges {edges}"


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["naive", "tiled"])
def test_stencil_boundary_ring_is_computed(ck, kernel):
    """x nonzero only on the outer ring of cells: a kernel that skips boundary cells (the §9.4
    bug) leaves those outputs wrong, and the normwise check is taken over the ring alone."""
    n = GPU_SIZES[-1]
    c, A, _ = board(0, n, ("W", "E", "S", "N"))
    x = np.zeros((4, n, n))
    ring = np.ones((n, n), bool)
    ring[1:-1, 1:-1] = False
    x[:, ring] = np.random.default_rng(0).uniform(1.0, 2.0, (4, ring.sum()))
    y = host(getattr(ck, f"stencil_apply_{kernel}")(ck.upload_coefficients(c, np.float64),
                                                     ck.cuda.to_device(x)))
    ref = (A @ x.ravel()).reshape(x.shape)
    assert relerr(y[:, ring], ref[:, ring]) <= FP64_RTOL
    assert relerr(y, ref) <= FP64_RTOL


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["naive", "tiled"])
def test_stencil_batched_equals_single(ck, kernel):
    n = GPU_SIZES[-1]
    apply = getattr(ck, f"stencil_apply_{kernel}")
    cs = [board(s, n)[0] for s in (0, 1, 2)]
    x = np.random.default_rng(1).uniform(0, 10, (3, 4, n, n)).astype(np.float32)
    yb = host(apply(ck.upload_coefficients(cs, np.float32), ck.cuda.to_device(x)))
    for k, c in enumerate(cs):
        y1 = host(apply(ck.upload_coefficients(c, np.float32), ck.cuda.to_device(x[k])))
        np.testing.assert_array_equal(yb[k], y1)


@pytest.mark.gpu
def test_stencil_rejects_mismatched_inputs(ck):
    n = GPU_SIZES[-1]
    c, _, _ = board(0, n)
    dc = ck.upload_coefficients(c, np.float32)
    with pytest.raises(TypeError, match="dtype"):
        ck.stencil_apply_tiled(dc, ck.cuda.to_device(np.zeros((4, n, n))))
    with pytest.raises(ValueError, match="shape"):
        ck.stencil_apply_tiled(dc, ck.cuda.to_device(np.zeros((4, n, n + 1), np.float32)))
    x = ck.cuda.to_device(np.zeros((4, n, n), np.float32))
    with pytest.raises(ValueError, match="alias"):
        ck.stencil_apply_tiled(dc, x, out=x)
    bad = dict(c)
    bad["diag"] = c["diag"].copy()
    bad["diag"][2, 0, 0] = 0.0
    with pytest.raises(ValueError, match="diag"):
        ck.upload_coefficients(bad, np.float32)


def rb_reference(c: dict, x: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """One red-black Gauss–Seidel sweep in NumPy: colour = (l + y + x) mod 2, red (0) first."""
    x = x.copy()
    L, n, _ = x.shape
    ll, yy, xx = np.meshgrid(np.arange(L), np.arange(n), np.arange(n), indexing="ij")
    for colour in (0, 1):
        nb = np.zeros_like(x)
        nb[:, :, :-1] += c["Ge"][:, :, :-1] * x[:, :, 1:]
        nb[:, :, 1:] += c["Gw"][:, :, 1:] * x[:, :, :-1]
        nb[:, :-1, :] += c["Gn"][:, :-1, :] * x[:, 1:, :]
        nb[:, 1:, :] += c["Gs"][:, 1:, :] * x[:, :-1, :]
        nb[:-1] += c["gup"][:-1] * x[1:]
        nb[1:] += c["gdn"][1:] * x[:-1]
        m = (ll + yy + xx) % 2 == colour
        x[m] = ((rhs + nb) / c["diag"])[m]
    return x


@pytest.mark.gpu
def test_red_black_sweep_matches_numpy_and_converges(ck):
    n = GPU_SIZES[-1]
    c, A, b = board(3, n, ("W",))
    dc = ck.upload_coefficients(c, np.float64)
    x = np.random.default_rng(3).uniform(0, 50, (4, n, n))
    dx = ck.cuda.to_device(x)
    rhs = ck.cuda.to_device(b)
    ck.red_black_gs_sweep(dc, dx, rhs)
    assert relerr(host(dx), rb_reference(c, x, b)) <= FP64_RTOL
    theta = spla.spsolve(A.tocsc(), b.ravel()).reshape(b.shape)

    def a_norm(e):
        return float(np.sqrt(e.ravel() @ (A @ e.ravel())))
    errs = [a_norm(host(dx) - theta)]
    for _ in range(5):
        ck.red_black_gs_sweep(dc, dx, rhs)
        errs.append(a_norm(host(dx) - theta))
    assert all(e1 < e0 for e0, e1 in zip(errs, errs[1:], strict=False)), errs  # GS on SPD


@pytest.mark.gpu
@pytest.mark.parametrize("size", [1, 31, 32, 33, 1000, 4 * 64 * 64 + 7, 4 * N_FULL * N_FULL])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dot_warp_shuffle_matches_float64_dot(ck, size, dtype):
    if SIM:
        pytest.skip("the CUDA simulator has no warp shuffles (shfl_down_sync): T4 only")
    rng = np.random.default_rng(size)
    a, b = rng.standard_normal((2, size)).astype(dtype)
    got = float(host(ck.dot_warp_shuffle(ck.cuda.to_device(a), ck.cuda.to_device(b)))[0])
    want = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    # FP64 accumulation: error ~ size·eps64·Σ|ab|, far inside 1e-12 of Σ|ab|
    assert abs(got - want) <= 1e-12 * float(np.abs(a.astype(np.float64) * b).sum())


@pytest.mark.gpu
def test_dot_warp_shuffle_batched_rows(ck):
    if SIM:
        pytest.skip("the CUDA simulator has no warp shuffles (shfl_down_sync): T4 only")
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((2, 5, 4, 20, 20)).astype(np.float32)
    got = host(ck.dot_warp_shuffle(ck.cuda.to_device(a), ck.cuda.to_device(b)))
    want = np.einsum("bi,bi->b", a.reshape(5, -1).astype(np.float64), b.reshape(5, -1))
    np.testing.assert_allclose(got, want, rtol=1e-12)


# =============================================================================== GPU (CuPy)

@pytest.fixture(scope="module")
def cp(ck):
    cupy = pytest.importorskip("cupy")
    if SIM:
        pytest.skip("CuPy needs a real GPU")
    return cupy


def gpu_operator(cp, c, dtype, kernel="tiled"):
    from solvers import cuda_kernels as ck
    from solvers.pcg import stencil_operator
    return stencil_operator(ck.upload_coefficients(c, dtype, to_device=cp.asarray), kernel)


@pytest.mark.gpu
def test_dot_warp_shuffle_equals_cupy_vdot(ck, cp):
    a, b = cp.random.default_rng(0).standard_normal((2, 4 * N_FULL * N_FULL), dtype=cp.float32)
    got = float(cp.asarray(ck.dot_warp_shuffle(a, b))[0])
    want = float(cp.vdot(a, b))
    assert abs(got - want) <= B3_RTOL * abs(want)


@pytest.mark.gpu
@pytest.mark.parametrize("precond", ["jacobi", "line_z"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cupy_pcg_matches_spsolve(cp, seed, precond):
    """B3 for the solver: CUDA PCG (FP64 operator) vs SciPy direct, rel 1e-6."""
    c, _, b = board(seed, N_FULL)
    diag, gup = cp.asarray(c["diag"]), cp.asarray(c["gup"])
    Minv = jacobi(diag) if precond == "jacobi" else line_z_block_jacobi(diag, gup)
    x, iters = pcg(gpu_operator(cp, c, np.float64), cp.asarray(b), cp.zeros(b.shape), Minv,
                   tol=1e-10)
    assert relerr(cp.asnumpy(x), exact(seed, N_FULL)) <= B3_RTOL
    assert 0 < iters < 20000


@pytest.mark.gpu
def test_cupy_pcg_float32_iterations_with_fp64_check(cp):
    """The production mix: FP32 operator for iterations, FP64 operator for the true residual."""
    c, A, b = board(0, N_FULL)
    x, _ = pcg(gpu_operator(cp, c, np.float32), cp.asarray(b, np.float32),
               cp.zeros(b.shape, np.float32), jacobi(cp.asarray(c["diag"], np.float32)),
               tol=1e-5, Aop_check=gpu_operator(cp, c, np.float64))
    r = b - (A @ cp.asnumpy(x).astype(np.float64).ravel()).reshape(b.shape)
    assert np.linalg.norm(r) <= 1e-5 * np.linalg.norm(b)


@pytest.mark.gpu
def test_cupy_warm_start_from_the_exact_solution(cp):
    c, _, b = board(1, N_FULL)
    x, iters = pcg(gpu_operator(cp, c, np.float64), cp.asarray(b), cp.asarray(exact(1, N_FULL)),
                   jacobi(cp.asarray(c["diag"])), tol=1e-8)
    assert iters <= 1


@pytest.mark.gpu
def test_cupy_batched_pcg_on_16_boards_equals_16_single_solves(cp):
    seeds = list(range(16))
    cs = [board(s, N_FULL)[0] for s in seeds]
    b = cp.asarray(np.stack([board(s, N_FULL)[2] for s in seeds]))
    xb, itb = batched_pcg(gpu_operator(cp, cs, np.float64), b, cp.zeros_like(b),
                          jacobi(cp.asarray(stack(cs, "diag"))), tol=1e-10)
    for k, s in enumerate(seeds):
        x1, it1 = pcg(gpu_operator(cp, cs[k], np.float64), b[k], cp.zeros_like(b[k]),
                      jacobi(cp.asarray(cs[k]["diag"])), tol=1e-10)
        # same arithmetic per board up to the order of the batched reductions
        assert abs(int(itb[k]) - it1) <= 1, f"board {s}: {itb[k]} vs {it1}"
        assert relerr(cp.asnumpy(xb[k]), cp.asnumpy(x1)) <= 1e-8, f"board {s}"
        assert relerr(cp.asnumpy(xb[k]), exact(s, N_FULL)) <= B3_RTOL, f"board {s}"
