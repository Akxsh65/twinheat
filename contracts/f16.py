"""Field arrays on the wire (CLAUDE.md §10): base64 of little-endian IEEE 754 float16, row-major.

Array order is [layer, y, x] (CLAUDE.md §0 rule 7); values are temperature rises in K. Rounding
is IEEE round-to-nearest-even from the input dtype; viewer/src/f16.js implements the same codec
and tests/fixtures/f16_vectors.json pins both to identical bits.

Fails loudly (CLAUDE.md §0 rule 5): non-finite values, values that overflow float16 (|x| >= 65520
rounds to inf), malformed shapes and byte-length mismatches all raise ValueError.
"""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Sequence

import numpy as np

__all__ = ["decode", "encode", "check_shape"]

_WIRE = np.dtype("<f2")


def check_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    """Return shape as a tuple of 3 positive ints [layer, y, x], or raise ValueError naming it."""
    dims = list(shape)
    if len(dims) != 3 or not all(
        isinstance(d, (int, np.integer)) and not isinstance(d, (bool, np.bool_)) and d > 0
        for d in dims
    ):
        raise ValueError(f"shape must be 3 positive ints [layer, y, x], got {dims!r}")
    return int(dims[0]), int(dims[1]), int(dims[2])


def encode(arr: np.ndarray) -> str:
    """Encode a 3-D float array [layer, y, x] (K) as base64 little-endian float16."""
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"field array must be 3-D [layer, y, x], got shape {a.shape}")
    check_shape(a.shape)
    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError(f"field array must have a floating dtype, got {a.dtype}")
    if not np.isfinite(a).all():
        bad = int((~np.isfinite(a)).sum())
        raise ValueError(f"field array must be finite; {bad} non-finite values")
    with np.errstate(over="ignore"):
        h = np.ascontiguousarray(a, dtype=_WIRE)
    if not np.isfinite(h).all():
        worst = float(np.abs(a).max())
        raise ValueError(
            f"field array must be finite in float16; max |value| {worst} overflows (>= 65520)"
        )
    return base64.b64encode(h.tobytes(order="C")).decode("ascii")


def decode(b64: str, shape: Sequence[int]) -> np.ndarray:
    """Decode base64 little-endian float16 into a float16 array of the given [layer, y, x] shape."""
    dims = check_shape(shape)
    if not isinstance(b64, str) or not b64:
        raise ValueError(f"field must be a non-empty base64 string, got {b64!r:.40}")
    try:
        raw = base64.b64decode(b64, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"field is not valid base64: {exc}") from exc
    expected = 2 * math.prod(dims)
    if len(raw) != expected:
        raise ValueError(
            f"field has {len(raw)} bytes but shape {list(dims)} needs {expected} (2 per value)"
        )
    out = np.frombuffer(raw, dtype=_WIRE).reshape(dims).astype(np.float16)
    if not np.isfinite(out).all():
        bad = int((~np.isfinite(out)).sum())
        raise ValueError(f"field contains {bad} non-finite float16 values")
    return out
