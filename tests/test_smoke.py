"""Smoke tests: every Python package of CLAUDE.md §3 imports."""

import importlib

import pytest

PACKAGES = [
    "twinheat",
    "solvers",
    "certify",
    "data",
    "models",
    "gate",
    "pipeline",
    "server",
    "experiments",
    "study",
    "release",
    "bench",
    "hardware",
    "hardware.analysis",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.slow
def test_slow_marker_is_registered() -> None:
    """Keeps the nightly `slow and not gpu` selection non-empty."""
    importlib.import_module("twinheat")
