"""Numba CUDA kernels for the 4-layer stencil operator (N1.4; CLAUDE.md §7.2, §9.4).

Operator, from solvers/fvm.py coefficients() (W/K; rhs in W):
  (A x)_P = diag_P x_P − Ge x_{x+1} − Gw x_{x−1} − Gn x_{y+1} − Gs x_{y−1}
            − gup x_{l+1} − gdn x_{l−1}
Arrays are [B, L, N, N] = [board, layer, y, x] (rule 7); [L, N, N] inputs are treated as B = 1
and returned in the same shape. Coefficients pointing off the grid are 0 (Neumann faces) and
Dirichlet terms live in diag, so every cell, the boundary ring included, uses the same formula;
the kernels only guard their reads. Threads map threadIdx.x to x, the contiguous axis, so loads
coalesce; grid axis z runs over (board, layer).

dtype: kernels specialise on the array dtype. float32 coefficients for iterations, float64 for
residual checks: upload_coefficients(c, np.float64) is the FP64 operator. Mixing dtypes raises.

Differences from the §9.4 reference kernel:
  1. every cell is computed, including the outer ring (the reference skips it: its known bug);
  2. threadIdx.x indexes x, not y (the reference's tx -> i mapping strides neighbouring threads
     by N, so its loads do not coalesce);
  3. north/south follow coefficients(): Gn couples y + 1 (index 0 of y is the south edge,
     rule 7); the reference reads Gn from row i − 1.
Device arrays may be Numba device arrays or any __cuda_array_interface__ array (CuPy).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np
from numba import cuda, float32, float64

__all__ = [
    "StencilCoeffs", "cuda", "dot_warp_shuffle", "red_black_gs_sweep", "stencil_apply_naive",
    "stencil_apply_tiled", "upload_coefficients",
]

TB = 16  # tile side (§9.4): 16 × 16 threads, an 18 × 18 shared tile with halos
DOT_BLOCK = 256  # threads per block in the dot kernel (8 warps)
WARP = 32
FULL_MASK = 0xFFFFFFFF
DOT_MAX_BLOCKS = 1024  # grid-stride cap for the dot kernel
FIELDS = ("Ge", "Gw", "Gn", "Gs", "gup", "gdn", "diag")


class StencilCoeffs(NamedTuple):
    """Device coefficient arrays, each [B, L, N, N], one dtype."""
    Ge: Any
    Gw: Any
    Gn: Any
    Gs: Any
    gup: Any
    gdn: Any
    diag: Any


def upload_coefficients(
    coeffs: dict | Sequence[dict], dtype, to_device: Callable = cuda.to_device
) -> StencilCoeffs:
    """Stack solvers/fvm.py coefficients() dicts (one per board) into [B, L, N, N] device arrays.

    to_device: cuda.to_device (Numba arrays) or cupy.asarray. Raises if shapes differ, a value is
    not finite, diag <= 0 or an off-diagonal coupling is negative (not an M-matrix stencil).
    """
    cs = [coeffs] if isinstance(coeffs, dict) else list(coeffs)
    if not cs:
        raise ValueError("upload_coefficients: no boards")
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise TypeError(f"dtype must be float32 or float64, got {dtype}")
    host = {}
    for name in FIELDS:
        arrs = [np.asarray(c[name], dtype=np.float64) for c in cs]
        shapes = {a.shape for a in arrs}
        if len(shapes) != 1 or len(next(iter(shapes))) != 3:
            raise ValueError(f"{name}: every board must be [L, N, N] of one shape, got {shapes}")
        a = np.stack(arrs)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: non-finite coefficient")
        if name == "diag" and a.min() <= 0:
            raise ValueError(f"diag must be > 0, min is {a.min()} (board, l, y, x) = "
                             f"{np.unravel_index(a.argmin(), a.shape)}")
        if name != "diag" and a.min() < 0:
            raise ValueError(f"{name} must be >= 0 (M-matrix coupling), min is {a.min()}")
        host[name] = np.ascontiguousarray(a.astype(dtype))
    return StencilCoeffs(*(to_device(host[name]) for name in FIELDS))


# ---------------------------------------------------------------------------------- helpers

def _dev(a, what: str):
    """A Numba device array view of `a` (Numba array, CuPy array, or a simulator array)."""
    if cuda.devicearray.is_cuda_ndarray(a):
        return a
    if hasattr(a, "__cuda_array_interface__"):
        return cuda.as_cuda_array(a)
    raise TypeError(f"{what} must be a device array (Numba or CuPy), got {type(a).__name__}")


def _ptr(a) -> int:
    """Device address of a's data (host address under the CUDA simulator, which has no CAI)."""
    if hasattr(a, "__cuda_array_interface__"):
        return a.__cuda_array_interface__["data"][0]
    return a._ary.ctypes.data


def _as4(a, what: str):
    d = _dev(a, what)
    if d.ndim == 3:
        return d.reshape((1, *d.shape)), True
    if d.ndim == 4:
        return d, False
    raise ValueError(f"{what} must be [L, N, N] or [B, L, N, N], got shape {d.shape}")


def _prepare(c: StencilCoeffs, x, out):
    xd, squeeze = _as4(x, "x")
    diag = _dev(c.diag, "coefficients")
    if xd.shape != diag.shape:
        raise ValueError(f"x shape {tuple(xd.shape)} does not match the coefficients' "
                         f"{tuple(diag.shape)}")
    if xd.dtype != diag.dtype:
        raise TypeError(f"x dtype {xd.dtype} != coefficient dtype {diag.dtype} (upload the "
                        "coefficients in the operator's dtype)")
    if out is None:
        od = cuda.device_array(xd.shape, dtype=xd.dtype)
        ret = od.reshape(od.shape[1:]) if squeeze else od
    else:
        od, _ = _as4(out, "out")
        if od.shape != xd.shape or od.dtype != xd.dtype:
            raise ValueError(f"out shape/dtype {tuple(od.shape)}/{od.dtype} must equal x's")
        if _ptr(od) == _ptr(xd):
            raise ValueError("out must not alias x (the stencil reads neighbours of x)")
        ret = out
    args = tuple(_dev(a, "coefficients") for a in c)
    return xd, od, args, ret


def _grid3(xd) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    B, L, ny, nx = xd.shape
    return ((nx + TB - 1) // TB, (ny + TB - 1) // TB, B * L), (TB, TB, 1)


# ------------------------------------------------------------------------------ naive stencil

@cuda.jit
def _stencil_naive(Ge, Gw, Gn, Gs, gup, gdn, diag, x, y):
    i, j, bl = cuda.grid(3)
    nl, ny, nx = x.shape[1], x.shape[2], x.shape[3]
    if i >= nx or j >= ny or bl >= x.shape[0] * nl:
        return
    b = bl // nl
    l = bl - b * nl  # noqa: E741
    v = diag[b, l, j, i] * x[b, l, j, i]
    if i + 1 < nx:
        v -= Ge[b, l, j, i] * x[b, l, j, i + 1]
    if i > 0:
        v -= Gw[b, l, j, i] * x[b, l, j, i - 1]
    if j + 1 < ny:
        v -= Gn[b, l, j, i] * x[b, l, j + 1, i]
    if j > 0:
        v -= Gs[b, l, j, i] * x[b, l, j - 1, i]
    if l + 1 < nl:
        v -= gup[b, l, j, i] * x[b, l + 1, j, i]
    if l > 0:
        v -= gdn[b, l, j, i] * x[b, l - 1, j, i]
    y[b, l, j, i] = v


def stencil_apply_naive(c: StencilCoeffs, x, out=None):
    """y = A x, one thread per cell, every neighbour read from global memory."""
    xd, od, args, ret = _prepare(c, x, out)
    grid, block = _grid3(xd)
    _stencil_naive[grid, block](*args, xd, od)
    return ret


# ------------------------------------------------------------------------------ tiled stencil

def _make_tiled(nb_dtype):
    @cuda.jit
    def kernel(Ge, Gw, Gn, Gs, gup, gdn, diag, x, y):
        s = cuda.shared.array((TB + 2, TB + 2), nb_dtype)  # [y, x] with a 1-cell halo
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        i = cuda.blockIdx.x * TB + tx
        j = cuda.blockIdx.y * TB + ty
        bl = cuda.blockIdx.z
        nl, ny, nx = x.shape[1], x.shape[2], x.shape[3]
        b = bl // nl
        l = bl - b * nl  # noqa: E741
        inside = i < nx and j < ny
        # every slot of s is written (0 off the grid) before the barrier; no thread returns early
        if inside:
            s[ty + 1, tx + 1] = x[b, l, j, i]
        else:
            s[ty + 1, tx + 1] = 0
        if tx == 0:
            if j < ny and 0 < i <= nx:
                s[ty + 1, 0] = x[b, l, j, i - 1]
            else:
                s[ty + 1, 0] = 0
        if tx == TB - 1:
            if j < ny and i + 1 < nx:
                s[ty + 1, TB + 1] = x[b, l, j, i + 1]
            else:
                s[ty + 1, TB + 1] = 0
        if ty == 0:
            if i < nx and 0 < j <= ny:
                s[0, tx + 1] = x[b, l, j - 1, i]
            else:
                s[0, tx + 1] = 0
        if ty == TB - 1:
            if i < nx and j + 1 < ny:
                s[TB + 1, tx + 1] = x[b, l, j + 1, i]
            else:
                s[TB + 1, tx + 1] = 0
        cuda.syncthreads()
        if inside:
            v = (diag[b, l, j, i] * s[ty + 1, tx + 1]
                 - Ge[b, l, j, i] * s[ty + 1, tx + 2] - Gw[b, l, j, i] * s[ty + 1, tx]
                 - Gn[b, l, j, i] * s[ty + 2, tx + 1] - Gs[b, l, j, i] * s[ty, tx + 1])
            if l + 1 < nl:
                v -= gup[b, l, j, i] * x[b, l + 1, j, i]
            if l > 0:
                v -= gdn[b, l, j, i] * x[b, l - 1, j, i]
            y[b, l, j, i] = v
    return kernel


_TILED = {np.dtype(np.float32): _make_tiled(float32), np.dtype(np.float64): _make_tiled(float64)}


def stencil_apply_tiled(c: StencilCoeffs, x, out=None):
    """y = A x with 16 × 16 shared-memory tiles plus a one-cell halo; all cells computed.

    Off-grid halo slots hold 0 and the matching coefficients are 0, so boundary cells need no
    separate kernel. Layer neighbours (l ± 1) are read from global memory.
    """
    xd, od, args, ret = _prepare(c, x, out)
    kernel = _TILED.get(np.dtype(xd.dtype))
    if kernel is None:
        raise TypeError(f"no tiled kernel for dtype {xd.dtype}")
    grid, block = _grid3(xd)
    kernel[grid, block](*args, xd, od)
    return ret


# ---------------------------------------------------------------------- red-black Gauss–Seidel

@cuda.jit
def _rb_half_sweep(Ge, Gw, Gn, Gs, gup, gdn, diag, x, rhs, colour):
    i, j, bl = cuda.grid(3)
    nl, ny, nx = x.shape[1], x.shape[2], x.shape[3]
    if i >= nx or j >= ny or bl >= x.shape[0] * nl:
        return
    b = bl // nl
    l = bl - b * nl  # noqa: E741
    if (l + j + i) % 2 != colour:
        return
    v = rhs[b, l, j, i]
    if i + 1 < nx:
        v += Ge[b, l, j, i] * x[b, l, j, i + 1]
    if i > 0:
        v += Gw[b, l, j, i] * x[b, l, j, i - 1]
    if j + 1 < ny:
        v += Gn[b, l, j, i] * x[b, l, j + 1, i]
    if j > 0:
        v += Gs[b, l, j, i] * x[b, l, j - 1, i]
    if l + 1 < nl:
        v += gup[b, l, j, i] * x[b, l + 1, j, i]
    if l > 0:
        v += gdn[b, l, j, i] * x[b, l - 1, j, i]
    x[b, l, j, i] = v / diag[b, l, j, i]


def red_black_gs_sweep(c: StencilCoeffs, x, rhs) -> None:
    """One Gauss–Seidel sweep on A x = rhs, in place: colour (l + y + x) mod 2 = 0, then 1.

    The 7-point stencil couples only cells of opposite colour, so each half-sweep updates its
    cells in parallel from the other colour's latest values (two launches per sweep).
    """
    xd, _ = _as4(x, "x")
    rd, _ = _as4(rhs, "rhs")
    if rd.shape != xd.shape or rd.dtype != xd.dtype:
        raise ValueError(f"rhs shape/dtype {tuple(rd.shape)}/{rd.dtype} must equal x's")
    _prepare(c, xd, rd)  # shape and dtype checks against the coefficients
    args = tuple(_dev(a, "coefficients") for a in c)
    grid, block = _grid3(xd)
    for colour in (0, 1):
        _rb_half_sweep[grid, block](*args, xd, rd, colour)


# ------------------------------------------------------------------------- warp-shuffle dot

@cuda.jit
def _dot_rows(a, b, out):
    """out[row] += Σ_k a[row, k]·b[row, k] in float64: grid-stride sum, warp shuffle, then the
    per-warp sums through shared memory, then one atomic add per block."""
    warp_sums = cuda.shared.array(DOT_BLOCK // WARP, float64)
    row = cuda.blockIdx.y
    tid = cuda.threadIdx.x
    n = a.shape[1]
    acc = 0.0
    k = cuda.blockIdx.x * cuda.blockDim.x + tid
    stride = cuda.gridDim.x * cuda.blockDim.x
    while k < n:
        acc += float64(a[row, k]) * float64(b[row, k])
        k += stride
    off = WARP // 2
    while off > 0:
        acc += cuda.shfl_down_sync(FULL_MASK, acc, off)
        off //= 2
    lane = tid % WARP
    warp = tid // WARP
    if lane == 0:
        warp_sums[warp] = acc
    cuda.syncthreads()
    if warp == 0:
        acc = warp_sums[lane] if lane < DOT_BLOCK // WARP else 0.0
        off = WARP // 2
        while off > 0:
            acc += cuda.shfl_down_sync(FULL_MASK, acc, off)
            off //= 2
        if lane == 0:
            cuda.atomic.add(out, row, acc)


def dot_warp_shuffle(a, b, out=None):
    """Dot products accumulated in float64; returns a device float64 array [B].

    a, b: equal-shape device arrays. 1-D inputs give one dot ([1]); otherwise the first axis is
    the batch and the rest is flattened (one dot per board). out: optional float64 [B].
    """
    ad, bd = _dev(a, "a"), _dev(b, "b")
    if ad.shape != bd.shape or ad.dtype != bd.dtype:
        raise ValueError(f"a and b must match: {tuple(ad.shape)}/{ad.dtype} vs "
                         f"{tuple(bd.shape)}/{bd.dtype}")
    if ad.size == 0:
        raise ValueError("dot of empty arrays")
    rows = 1 if ad.ndim == 1 else ad.shape[0]
    cols = ad.size // rows
    a2, b2 = ad.reshape((rows, cols)), bd.reshape((rows, cols))
    if out is None:
        od = cuda.to_device(np.zeros(rows))
        ret = od
    else:
        od = _dev(out, "out")
        if od.shape != (rows,) or od.dtype != np.float64:
            raise ValueError(f"out must be float64 [{rows}], got {tuple(od.shape)}/{od.dtype}")
        od[:] = 0  # the kernel accumulates with atomics
        ret = out
    blocks = min((cols + DOT_BLOCK - 1) // DOT_BLOCK, DOT_MAX_BLOCKS)
    _dot_rows[(blocks, rows), DOT_BLOCK](a2, b2, od)
    return ret
