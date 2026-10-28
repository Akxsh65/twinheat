"""Split guard (prompt M1.3; CLAUDE.md §5 data hygiene).

Training and early stopping may read train and val only. The calibration split ("cal") is for
conformal initialisation, and test and OOD splits ("test", "ood_*") are read only by experiments/
with allow_eval=True. open_split refuses anything else, naming the split.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

__all__ = ["SplitAccessError", "is_protected", "open_split"]

T = TypeVar("T")
PROTECTED = ("cal", "test")


class SplitAccessError(PermissionError):
    """A protected split was requested without allow_eval=True."""


def is_protected(split: str) -> bool:
    """True for "cal", "test" and every "ood_*" split."""
    return split in PROTECTED or split.startswith("ood_")


def open_split(split: str, loader: Callable[[str], T], *, allow_eval: bool = False) -> T:
    """loader(split), unless split is protected and allow_eval is False (then raise)."""
    if not isinstance(split, str) or not split:
        raise ValueError(f"split must be a non-empty string, got {split!r}")
    if is_protected(split) and not allow_eval:
        raise SplitAccessError(
            f"split {split!r} is protected (CLAUDE.md §5): cal is for conformal initialisation, "
            "test and ood_* are read only by experiments/ with allow_eval=True"
        )
    return loader(split)
