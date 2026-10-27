"""Validate records and messages against contracts/schemas (CLAUDE.md §10, invariant B6).

validate(obj, kind) raises ContractError carrying the JSON path ("$.layers[0].shapes[1].r") of
the first error. After the schema it runs the two checks JSON Schema cannot express, identically
to viewer/src/contracts.js: field-array shapes are square ([4, n, n]) and each base64 field
decodes to exactly 2 * prod(shape) bytes.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

__all__ = ["KINDS", "ContractError", "validate"]

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
KINDS = (
    "board_record", "query", "design", "pred", "iterate", "verified", "batch_progress", "error",
    "audit_log_row", "hardware_record", "session_log_line",
)  # fmt: skip
FIELD_ARRAYS = {"pred": ("mu", "sd"), "iterate": ("theta",), "verified": ("theta",)}


class ContractError(ValueError):
    """An object violates its contract. `path` is the JSON path of the first error."""

    def __init__(self, kind: str, path: str, message: str) -> None:
        super().__init__(f"{kind}: {path}: {message}")
        self.kind = kind
        self.path = path
        self.message = message


@cache
def _validators() -> dict[str, Draft202012Validator]:
    schemas = {}
    for kind in KINDS:
        with (SCHEMA_DIR / f"{kind}.json").open(encoding="utf-8") as fh:
            schemas[kind] = json.load(fh)
    registry: Registry = Registry().with_resources(
        (s["$id"], Resource.from_contents(s)) for s in schemas.values()
    )
    return {k: Draft202012Validator(s, registry=registry) for k, s in schemas.items()}


def _json_path(parts: Any) -> str:
    out = "$"
    for p in parts:
        out += f"[{p}]" if isinstance(p, int) else f".{p}"
    return out


def _base64_bytes(s: str) -> int | None:
    """Decoded length of a padded base64 string, or None if its length is not a multiple of 4."""
    if len(s) % 4:
        return None
    return 3 * len(s) // 4 - (len(s) - len(s.rstrip("=")))


def _check_field_arrays(obj: dict[str, Any], kind: str) -> None:
    _, n_y, n_x = obj["shape"]
    if n_y != n_x:
        raise ContractError(kind, "$.shape", f"field arrays must be [4, n, n], got {obj['shape']}")
    expected = 2 * 4 * n_y * n_x
    for field in FIELD_ARRAYS[kind]:
        got = _base64_bytes(obj[field])
        if got != expected:
            raise ContractError(
                kind,
                f"$.{field}",
                f"decodes to {got} bytes but shape {obj['shape']} needs {expected}",
            )


def validate(obj: Any, kind: str) -> None:
    """Raise ContractError if obj does not satisfy contract `kind`; return None otherwise."""
    if kind not in KINDS:
        raise ContractError(kind, "$", f"unknown contract kind {kind!r}; known: {', '.join(KINDS)}")
    err: ValidationError | None = next(iter(_validators()[kind].iter_errors(obj)), None)
    if err is not None:
        raise ContractError(kind, _json_path(err.absolute_path), err.message)
    if kind in FIELD_ARRAYS:
        _check_field_arrays(obj, kind)
