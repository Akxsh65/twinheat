"""bench/stencil_bw.py argument handling (N1.4); the kernels need a GPU (tests/test_gpu_solvers.py).

The peak bandwidth is a required, sourced argument: no GPU's datasheet number is built into the
code (CLAUDE.md §0 rule 1; docs/decisions.md, "GPU runs on the college A100").
"""

from __future__ import annotations

import pytest

from bench import stencil_bw


def test_peak_and_its_source_are_required() -> None:
    with pytest.raises(SystemExit):
        stencil_bw.parse_args([])
    with pytest.raises(SystemExit):
        stencil_bw.parse_args(["--peak-gbs", "100"])
    with pytest.raises(SystemExit):
        stencil_bw.parse_args(["--peak-source", "datasheet"])


@pytest.mark.parametrize("bad", ["0", "-5", "nan", "inf"])
def test_non_positive_or_non_finite_peak_is_refused(bad: str) -> None:
    with pytest.raises(SystemExit):
        stencil_bw.parse_args(["--peak-gbs", bad, "--peak-source", "datasheet"])


def test_blank_source_is_refused() -> None:
    with pytest.raises(SystemExit):
        stencil_bw.parse_args(["--peak-gbs", "100", "--peak-source", "   "])


def test_valid_arguments_parse() -> None:
    a = stencil_bw.parse_args(["--peak-gbs", "123.5", "--peak-source", "docs/refs/x.txt"])
    assert a.peak_gbs == 123.5 and a.peak_source == "docs/refs/x.txt"


def test_no_gpu_peak_is_hard_coded() -> None:
    assert not hasattr(stencil_bw, "PEAK_GBS")
