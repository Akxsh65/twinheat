"""GPU device names for CSV rows and figure footers (CLAUDE.md §4)."""

from __future__ import annotations

__all__ = ["device_name"]


def device_name(raw: bytes | str) -> str:
    """The device name without the NUL bytes and blanks numba pads it with (rvcse A100).

    numba's cuda.get_current_device().name and cuda.list_devices() return the fixed-size C
    buffer, so a CSV column would otherwise carry dozens of NUL characters.
    """
    s = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    s = s.replace("\x00", "").strip()
    if not s:
        raise ValueError(f"GPU name is empty after removing padding: {raw!r}")
    return s
