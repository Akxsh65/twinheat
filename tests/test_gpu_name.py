"""numba returns NUL-padded GPU names (docs/compute.md, rvcse A100); twinheat.gpu cleans them."""

from __future__ import annotations

import pytest

from twinheat.gpu import device_name


@pytest.mark.parametrize("raw", [
    "NVIDIA A100-PCIE-40GB" + "\x00" * 43 + " " * 64,
    b"NVIDIA A100-PCIE-40GB\x00\x00\x00",
    b"NVIDIA A100-PCIE-40GB",
    "  NVIDIA A100-PCIE-40GB  ",
])
def test_padding_is_removed(raw) -> None:
    assert device_name(raw) == "NVIDIA A100-PCIE-40GB"


def test_an_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        device_name("\x00\x00  ")
