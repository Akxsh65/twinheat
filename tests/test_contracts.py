"""Contracts v1.0 (CLAUDE.md §10, invariant B6): schemas, examples, validator, float16 codec.

viewer/test/contracts.test.js runs the same examples, expected error paths and float16 vectors
in JS, so both validators are held to one set of files.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from contracts import f16
from contracts.validate import KINDS, ContractError, validate

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "contracts" / "schemas"
EXAMPLES = ROOT / "contracts" / "examples"
F16_VECTORS = ROOT / "tests" / "fixtures" / "f16_vectors.json"

EXPECTED_KINDS = {
    "board_record", "query", "design", "pred", "iterate", "verified", "batch_progress", "error",
    "audit_log_row", "hardware_record", "session_log_line",
}  # fmt: skip
MESSAGE_TYPES = {
    "query": "query", "design": "design", "pred": "pred", "iterate": "iterate",
    "verified": "verified", "batch_progress": "batch_progress", "error": "error",
}  # fmt: skip
EXAMPLE_NAMES = [f"valid_{i}" for i in (1, 2, 3)] + [f"invalid_{i}" for i in (1, 2, 3)]


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _schema(kind: str) -> dict[str, Any]:
    return _load(SCHEMAS / f"{kind}.json")


def _example(kind: str, name: str) -> Any:
    return _load(EXAMPLES / kind / f"{name}.json")


EXPECTED_ERRORS: dict[str, dict[str, str]] = _load(EXAMPLES / "expected_errors.json")
VALID_CASES = [(k, n) for k in sorted(EXPECTED_KINDS) for n in EXAMPLE_NAMES if n.startswith("v")]
INVALID_CASES = [(k, n) for k in sorted(EXPECTED_KINDS) for n in EXAMPLE_NAMES if n.startswith("i")]


# --------------------------------------------------------------------------- schema files


def test_kinds_match_deliverables() -> None:
    assert set(KINDS) == EXPECTED_KINDS
    assert {p.stem for p in SCHEMAS.glob("*.json")} == EXPECTED_KINDS


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_each_kind_has_exactly_six_examples(kind: str) -> None:
    assert sorted(p.stem for p in (EXAMPLES / kind).glob("*.json")) == sorted(EXAMPLE_NAMES)


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_schema_is_valid_draft_2020_12(kind: str) -> None:
    schema = _schema(kind)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == f"https://twinheat.invalid/contracts/1.0/{kind}.json"
    assert schema["title"] == kind
    Draft202012Validator.check_schema(schema)


def _object_schemas(node: Any, where: str = "$") -> list[tuple[str, dict[str, Any]]]:
    """Every subschema that declares type object (alone or with null)."""
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        t = node.get("type")
        if t == "object" or (isinstance(t, list) and "object" in t):
            found.append((where, node))
        for key, value in node.items():
            found += _object_schemas(value, f"{where}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += _object_schemas(value, f"{where}/{i}")
    return found


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_every_object_forbids_additional_properties(kind: str) -> None:
    for where, sub in _object_schemas(_schema(kind)):
        if str(sub.get("$comment", "")).startswith("dispatcher:"):
            # A dispatcher only selects a branch; every branch it can select must be closed.
            for branch in sub["allOf"]:
                target = branch["then"]["$ref"].split("/")[-1]
                assert _schema(kind)["$defs"][target]["additionalProperties"] is False, where
            continue
        assert sub.get("additionalProperties") is False, f"{kind}: {where}"


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_schema_version_required_and_pinned(kind: str) -> None:
    schema = _schema(kind)
    assert schema["required"][0] == "schema_version"
    assert schema["properties"]["schema_version"] == {"$ref": "#/$defs/schema_version"}
    assert schema["$defs"]["schema_version"] == {"const": "1.0"}


def test_shared_defs_are_identical_across_files() -> None:
    seen: dict[str, tuple[str, Any]] = {}
    for kind in sorted(EXPECTED_KINDS):
        for name, body in _schema(kind)["$defs"].items():
            if name in seen:
                assert body == seen[name][1], f"$defs/{name} differs: {kind} vs {seen[name][0]}"
            else:
                seen[name] = (kind, body)


@pytest.mark.parametrize("kind,type_", sorted(MESSAGE_TYPES.items()))
def test_message_type_is_pinned(kind: str, type_: str) -> None:
    schema = _schema(kind)
    assert schema["properties"]["type"] == {"const": type_}
    assert schema["required"][:2] == ["schema_version", "type"]


def test_pred_decision_enum_is_exactly_the_contract() -> None:
    decision = _schema("pred")["properties"]["decision"]
    assert decision["enum"] == ["preview", "accept", "flag", "audit"]


def test_query_and_design_reference_the_board_record() -> None:
    for kind in ("query", "design"):
        assert _schema(kind)["properties"]["record"] == {"$ref": "board_record.json"}


# --------------------------------------------------------------------------- examples


@pytest.mark.parametrize("kind,name", VALID_CASES)
def test_valid_example_passes(kind: str, name: str) -> None:
    validate(_example(kind, name), kind)


def test_expected_errors_cover_exactly_the_invalid_examples() -> None:
    assert set(EXPECTED_ERRORS) == EXPECTED_KINDS
    for kind in EXPECTED_KINDS:
        assert sorted(EXPECTED_ERRORS[kind]) == ["invalid_1", "invalid_2", "invalid_3"]


@pytest.mark.parametrize("kind,name", INVALID_CASES)
def test_invalid_example_fails_at_expected_path(kind: str, name: str) -> None:
    with pytest.raises(ContractError) as info:
        validate(_example(kind, name), kind)
    assert info.value.kind == kind
    assert info.value.path == EXPECTED_ERRORS[kind][name]
    assert info.value.path in str(info.value)
    assert kind in str(info.value)


def test_unknown_kind_raises_naming_it() -> None:
    with pytest.raises(ContractError, match="nonsense"):
        validate({}, "nonsense")


def test_non_object_is_rejected() -> None:
    with pytest.raises(ContractError) as info:
        validate([], "pred")
    assert info.value.path == "$"


def _set(obj: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(obj)
    node: Any = out
    keys = [int(k) if k.isdigit() else k for k in dotted.split(".")]
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    return out


MUTATIONS = [
    # (kind, field, bad value, expected path)
    ("board_record", "dirichlet_edges", ["W", "W"], "$.dirichlet_edges"),
    ("board_record", "dirichlet_edges", ["X"], "$.dirichlet_edges[0]"),
    ("board_record", "mode", "ood-1", "$.mode"),
    ("board_record", "convection.h_top", 0, "$.convection.h_top"),
    ("board_record", "layers.1.layer", 3, "$.layers[1].layer"),
    ("board_record", "chips.0.side", "left", "$.chips[0].side"),
    ("board_record", "chips.0.x0", 100.5, "$.chips[0].x0"),
    ("query", "record.schema_version", "1.1", "$.record.schema_version"),
    ("pred", "B", -0.1, "$.B"),
    ("pred", "units.mu", "degC", "$.units.mu"),
    ("pred", "t_server_ns", 1.5, "$.t_server_ns"),
    ("pred", "mu", "not base64!", "$.mu"),
    ("iterate", "cert_k", -1.0, "$.cert_k"),
    ("batch_progress", "active_ids", ["a", "a"], "$.active_ids"),
    ("audit_log_row", "x_hash", "ABC", "$.x_hash"),
    ("audit_log_row", "run_id", "g5", "$.run_id"),
    ("audit_log_row", "aggressiveness", 4, "$.aggressiveness"),
    ("hardware_record", "power_per_resistor", [], "$.power_per_resistor"),
    ("session_log_line", "server_decision", "reject", "$.server_decision"),
    ("session_log_line", "participant", "P1", "$.participant"),
]


@pytest.mark.parametrize("kind,field,value,path", MUTATIONS)
def test_single_field_mutation_of_valid_1_is_rejected(
    kind: str, field: str, value: Any, path: str
) -> None:
    with pytest.raises(ContractError) as info:
        validate(_set(_example(kind, "valid_1"), field, value), kind)
    assert info.value.path == path


def test_design_vector_accepts_null_and_36_unit_floats_only() -> None:
    rec = _example("board_record", "valid_1")
    validate(_set(rec, "design_vector", None), "board_record")
    validate(_set(rec, "design_vector", [0.5] * 36), "board_record")
    with pytest.raises(ContractError):
        validate(_set(rec, "design_vector", [0.5] * 35 + [1.5]), "board_record")


@pytest.mark.parametrize("kind", ["pred", "iterate", "verified"])
def test_field_array_byte_length_must_match_shape(kind: str) -> None:
    msg = _example(kind, "valid_1")
    n = msg["shape"][1]
    field = "mu" if kind == "pred" else "theta"
    validate(_set(msg, "shape", [4, n, n]), kind)
    with pytest.raises(ContractError) as info:
        validate(_set(msg, "shape", [4, n + 1, n + 1]), kind)
    assert info.value.path == f"$.{field}"


def test_message_validation_does_not_mutate_input() -> None:
    msg = _example("pred", "valid_1")
    before = copy.deepcopy(msg)
    validate(msg, "pred")
    assert msg == before


# --------------------------------------------------------------------------- float16 codec

# (float64 input, expected IEEE 754 binary16 bits); worked by hand, independent of numpy.
HAND_VECTORS = [
    (0.0, 0x0000),
    (-0.0, 0x8000),
    (1.0, 0x3C00),
    (-2.0, 0xC000),
    (0.1, 0x2E66),
    (65504.0, 0x7BFF),  # largest finite half
    (65519.0, 0x7BFF),  # just below the overflow midpoint
    (2.0**-14, 0x0400),  # smallest normal
    (2.0**-24, 0x0001),  # smallest subnormal
    (2.0**-25, 0x0000),  # tie between 0 and 2^-24 -> even (0)
    (3 * 2.0**-25, 0x0002),  # tie between 1 and 2 ulps -> even (2)
    (1 + 2.0**-11, 0x3C00),  # tie -> even mantissa
    (1 + 3 * 2.0**-11, 0x3C02),  # tie -> even mantissa
    (2.0**-14 - 2.0**-25, 0x0400),  # rounds up from subnormal into the smallest normal
]


def _bits(arr: np.ndarray) -> list[int]:
    return np.ascontiguousarray(arr, dtype="<f2").view("<u2").ravel().tolist()


def test_encode_matches_hand_worked_bits() -> None:
    values = np.array([v for v, _ in HAND_VECTORS], dtype=np.float64).reshape(1, 1, -1)
    raw = base64.b64decode(f16.encode(values))
    assert [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)] == [
        b for _, b in HAND_VECTORS
    ]


def test_encoding_is_little_endian() -> None:
    assert base64.b64decode(f16.encode(np.ones((1, 1, 1)))) == b"\x00\x3c"


def test_fixture_vectors_match_python_codec() -> None:
    for case in _load(F16_VECTORS)["cases"]:
        shape = case["shape"]
        values = np.array(case["values"], dtype=np.float64).reshape(shape)
        assert f16.encode(values) == case["b64"], case["name"]
        decoded = f16.decode(case["b64"], shape)
        assert decoded.shape == tuple(shape)
        assert _bits(decoded) == case["bits"], case["name"]


def test_round_trip_is_exact_for_float16_values() -> None:
    rng = np.random.Generator(np.random.PCG64(20261024))
    x = (rng.standard_normal((4, 16, 16)) * 30).astype(np.float16)
    y = f16.decode(f16.encode(x), [4, 16, 16])
    assert y.dtype == np.float16
    assert _bits(y) == _bits(x)


def test_encode_uses_row_major_layer_y_x_even_for_non_contiguous_input() -> None:
    x = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    xt = np.transpose(x, (0, 2, 1))  # non-contiguous view
    assert f16.decode(f16.encode(xt), [2, 3, 3]).tolist() == xt.tolist()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 65520.0, -1e6])
def test_encode_rejects_non_finite_and_overflow(bad: float) -> None:
    x = np.zeros((1, 2, 2))
    x[0, 1, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        f16.encode(x)


@pytest.mark.parametrize("arr", [np.zeros((4, 4)), np.zeros((1, 4, 4, 4)), np.zeros(())])
def test_encode_rejects_non_3d(arr: np.ndarray) -> None:
    with pytest.raises(ValueError, match="3-D"):
        f16.encode(arr)


def test_encode_rejects_non_float_dtype() -> None:
    with pytest.raises(TypeError, match="int64"):
        f16.encode(np.zeros((1, 2, 2), dtype=np.int64))


def test_decode_rejects_shape_mismatch() -> None:
    b64 = f16.encode(np.zeros((4, 8, 8)))
    with pytest.raises(ValueError, match=r"\[4, 8, 9\]"):
        f16.decode(b64, [4, 8, 9])
    with pytest.raises(ValueError, match="shape"):
        f16.decode(b64, [4, 4, 4])


BAD_SHAPES = [[4, 8], [4, 8, 8, 1], [0, 8, 8], [4, -8, -8], [4, 8.0, 8], [True, 1, 1]]


@pytest.mark.parametrize("shape", BAD_SHAPES)
def test_decode_rejects_malformed_shape(shape: list[Any]) -> None:
    with pytest.raises(ValueError, match="shape"):
        f16.decode(f16.encode(np.zeros((1, 1, 1))), shape)


@pytest.mark.parametrize("b64", ["AA=", "AA*A", "AAAA AAAA", ""])
def test_decode_rejects_bad_base64(b64: str) -> None:
    with pytest.raises(ValueError):
        f16.decode(b64, [1, 1, 1])


@pytest.mark.parametrize("bits", [0x7C00, 0xFC00, 0x7E00])
def test_decode_rejects_non_finite_halves(bits: int) -> None:
    b64 = base64.b64encode(bits.to_bytes(2, "little")).decode("ascii")
    with pytest.raises(ValueError, match="finite"):
        f16.decode(b64, [1, 1, 1])
