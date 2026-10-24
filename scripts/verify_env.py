"""Verify the TwinHeat environment (CLAUDE.md §4). Exits 1 on any failure.

Checks, in order:
  1. every package of CLAUDE.md §4 imports, and its version is printed;
  2. every `name==version` pin in requirements.txt matches the installed distribution
     (a local build tag such as torch's "+cu128" is ignored);
  3. pydantic is >= 2;
  4. if an NVIDIA GPU is visible: torch.cuda.is_available(), cupy sees the same number of
     devices as torch, and numba.cuda is available. Without a GPU these checks are skipped
     and said to be skipped; imports are still required.

Usage: python scripts/verify_env.py
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"

# (import name, distribution name used by pip/conda metadata)
PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pyamg", "pyamg"),
    ("numba", "numba"),
    ("torch", "torch"),
    ("cupy", "cupy-cuda12x"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("websockets", "websockets"),
    ("pydantic", "pydantic"),
    ("jsonschema", "jsonschema"),
    ("yaml", "pyyaml"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest"),
]

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ruff_version() -> str | None:
    """Version of the ruff command, or None. pip installs ruff as a Python module and
    conda-forge as a standalone executable (no pip metadata), so try both."""
    prefix = Path(sys.prefix)
    env_dirs = [str(prefix / sub) for sub in ("Scripts", "Library/bin", "bin")]
    search = os.pathsep.join([*env_dirs, os.environ.get("PATH", "")])
    candidates = [[sys.executable, "-m", "ruff"]]
    exe = shutil.which("ruff", path=search)
    if exe:
        candidates.append([exe])
    for cmd in candidates:
        try:
            out = subprocess.run(
                [*cmd, "--version"], capture_output=True, text=True, timeout=60, check=True
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            continue
        return out.removeprefix("ruff").strip()
    return None


def installed_version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return ruff_version() if dist == "ruff" else None


def check_imports() -> None:
    print("== imports")
    for module, dist in PACKAGES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # any import problem is a failure of this check
            fail(f"import {module}: {type(exc).__name__}: {exc}")
            continue
        print(f"  ok    {module:<11} {installed_version(dist) or 'version unknown'}")
    version = ruff_version()  # ruff is a command-line tool, not an import
    if version is None:
        fail("ruff: no working `python -m ruff` or `ruff` executable found")
    else:
        print(f"  ok    ruff        {version}")


def check_pins() -> None:
    print("== pins in requirements.txt")
    if not REQUIREMENTS.exists():
        fail(f"{REQUIREMENTS} not found")
        return
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if not m:
            continue
        dist, want = m.group(1), m.group(2)
        have = installed_version(dist)
        if have is None:
            fail(f"{dist}=={want} pinned but distribution not installed")
        elif have.split("+")[0] != want:
            fail(f"{dist}: pinned {want}, installed {have}")
        else:
            print(f"  ok    {dist}=={have}")


def check_pydantic_major() -> None:
    import pydantic

    if int(pydantic.VERSION.split(".")[0]) < 2:
        fail(f"pydantic {pydantic.VERSION} < 2 (CLAUDE.md §4 requires >= 2)")


def gpu_visible() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30, check=True)
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def check_gpu() -> None:
    print("== GPU")
    if not gpu_visible():
        print("  skipped: no NVIDIA GPU visible (nvidia-smi missing or failing)")
        return
    import torch

    if not torch.cuda.is_available():
        fail(f"GPU visible but torch.cuda.is_available() is False (torch {torch.__version__}, "
             f"built for CUDA {torch.version.cuda}); likely a CPU-only torch build")
        return
    n_torch = torch.cuda.device_count()
    print(f"  ok    torch {torch.__version__}: {n_torch} device(s), "
          f"{[torch.cuda.get_device_name(i) for i in range(n_torch)]}")
    try:
        import cupy as cp

        n_cupy = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # a broken cupy install is the thing being detected
        fail(f"cupy device query: {type(exc).__name__}: {exc}")
    else:
        if n_cupy < 1:
            fail("cupy sees 0 devices")
        elif n_cupy != n_torch:
            fail(f"cupy sees {n_cupy} device(s), torch sees {n_torch}")
        else:
            print(f"  ok    cupy {cp.__version__}: {n_cupy} device(s)")
    try:
        from numba import cuda

        if cuda.is_available():
            print("  ok    numba.cuda available")
        else:
            fail("numba.cuda.is_available() is False")
    except Exception as exc:  # same: report, then exit non-zero below
        fail(f"numba.cuda: {type(exc).__name__}: {exc}")


def main() -> int:
    print(f"python {sys.version.split()[0]} on {sys.platform}")
    check_imports()
    check_pins()
    if "pydantic" not in " ".join(failures):
        check_pydantic_major()
    check_gpu()
    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        return 1
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
