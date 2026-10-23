"""Report what this machine's GPU environment can run (CLAUDE.md §4).

Prints a human-readable report, runs three micro-tests (torch fp16 matmul, cupy SpMV,
numba.cuda launch), then prints one JSON object as the LAST line of stdout.
Installs nothing. Each section and test is guarded separately, so one failure
does not hide the others. Exit code is 1 if any test FAILs, else 0.

Units: memory in MiB, times in ms, throughput in TFLOP/s and GB/s.
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

MIB = 1024**2


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _version(module: str) -> str:
    try:
        return str(importlib.import_module(module).__version__)
    except Exception as exc:  # the import problem itself is the finding; report it
        return f"IMPORT FAILED ({_err(exc)})"


def _section(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:  # reported in the output, never silent
        return {"error": _err(exc), "traceback": traceback.format_exc(limit=3)}


def system_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {m: _version(m) for m in ("numpy", "scipy", "numba", "pyamg")},
    }


def nvidia_smi_info() -> dict[str, Any]:
    query = "name,driver_version,memory.total,memory.free"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    gpus = []
    for line in out.strip().splitlines():
        name, drv, total, free = (s.strip() for s in line.split(","))
        gpus.append(
            {"name": name, "driver": drv, "total_mib": float(total), "free_mib": float(free)}
        )
    header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    cuda = header.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in header else None
    return {"count": len(gpus), "driver_cuda_version": cuda, "gpus": gpus}


def torch_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "version": torch.__version__,
        "built_for_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": [],
    }
    if info["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            p = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                "free_mib": free / MIB, "total_mib": total / MIB,
            })
    return info


def cupy_info() -> dict[str, Any]:
    import cupy as cp

    n = cp.cuda.runtime.getDeviceCount()
    info: dict[str, Any] = {
        "version": cp.__version__,
        "device_count": n,
        "driver_version": cp.cuda.runtime.driverGetVersion(),
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "devices": [],
    }
    for i in range(n):
        with cp.cuda.Device(i):
            free, total = cp.cuda.runtime.memGetInfo()
        name = cp.cuda.runtime.getDeviceProperties(i)["name"].decode()
        info["devices"].append(
            {"index": i, "name": name, "free_mib": free / MIB, "total_mib": total / MIB}
        )
    return info


def numba_info() -> dict[str, Any]:
    from numba import cuda

    info: dict[str, Any] = {
        "version": _version("numba"),
        "cuda_available": bool(cuda.is_available()),
    }
    if info["cuda_available"]:
        names = [d.name for d in cuda.list_devices()]
        info["devices"] = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return info


# ---------------------------------------------------------------- micro-tests


def test_torch_matmul() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    n, iters = 4096, 20
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    c = a @ b  # warm-up (cuBLAS init)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / iters
    if not bool(torch.isfinite(c).all()):
        raise RuntimeError("fp16 matmul produced non-finite values")
    return {"n": n, "ms_per_matmul": ms, "tflops": 2 * n**3 / (ms * 1e-3) / 1e12}


def test_cupy_spmv() -> dict[str, Any]:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import numpy as np
    import scipy.sparse as sp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("cupy sees 0 devices")
    n, nx = 262_144, 256  # 256 x 256 x 4 layout: offsets of the 7-point layered stencil
    offsets = [-nx * nx, -nx, -1, 0, 1, nx, nx * nx]
    rng = np.random.Generator(np.random.PCG64(0))
    data = rng.standard_normal((7, n))
    x = rng.standard_normal(n)
    host = sp.dia_matrix((data, offsets), shape=(n, n)).tocsr()
    dev = csp.csr_matrix(
        (cp.asarray(host.data), cp.asarray(host.indices), cp.asarray(host.indptr)), shape=(n, n)
    )
    xd = cp.asarray(x)
    y = dev @ xd  # warm-up
    cp.cuda.Device().synchronize()
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        y = dev @ xd
    cp.cuda.Device().synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / iters
    ref = host @ x
    rel = float(np.linalg.norm(cp.asnumpy(y) - ref) / np.linalg.norm(ref))
    if not rel < 1e-10:
        raise RuntimeError(f"SpMV disagrees with SciPy: relative error {rel:.3e}")
    nbytes = host.nnz * (8 + 4) + 2 * n * 8  # values + column indices + x and y (rough)
    return {
        "rows": n, "nnz": int(host.nnz), "ms_per_spmv": ms,
        "approx_gb_per_s": nbytes / (ms * 1e-3) / 1e9, "rel_err_vs_scipy": rel,
    }


def test_numba_cuda() -> dict[str, Any]:
    import numpy as np
    from numba import cuda

    if not cuda.is_available():
        raise RuntimeError("numba.cuda.is_available() is False")

    @cuda.jit
    def add_one(a):
        i = cuda.grid(1)
        if i < a.size:
            a[i] += 1.0

    host = np.arange(1024, dtype=np.float32)
    dev = cuda.to_device(host)
    add_one[4, 256](dev)
    cuda.synchronize()
    if not np.array_equal(dev.copy_to_host(), host + 1.0):
        raise RuntimeError("kernel result is wrong")
    return {"elements": 1024}


def run_test(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        res = {"status": "PASS", **fn()}
    except Exception as exc:  # one test's failure must not hide the others
        res = {"status": "FAIL", "error": _err(exc)}
    detail = {k: v for k, v in res.items() if k != "status"}
    print(f"[{res['status']}] {name}: {detail}")
    return res


def main() -> int:
    report: dict[str, Any] = {"system": _section(system_info)}
    for key, fn in (
        ("nvidia_smi", nvidia_smi_info),
        ("torch", torch_info),
        ("cupy", cupy_info),
        ("numba", numba_info),
    ):
        report[key] = _section(fn)
        print(f"== {key}: {report[key]}")

    print("== micro-tests")
    report["tests"] = {
        "torch_fp16_matmul_4096": run_test("torch fp16 matmul 4096x4096", test_torch_matmul),
        "cupy_spmv_262144_7diag": run_test("cupy SpMV 262144 rows, 7 diagonals", test_cupy_spmv),
        "numba_cuda_launch": run_test("numba.cuda trivial kernel", test_numba_cuda),
    }
    failed = [k for k, v in report["tests"].items() if v["status"] != "PASS"]
    report["all_pass"] = not failed
    print("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
    print(json.dumps(report, default=str))  # last line: machine-readable
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
