"""Tests for twinheat.spec and config/spec_v{1,2}.yaml (CLAUDE.md §6, §7).

Expected numbers are taken from CLAUDE.md, not from the code under test:
  - A_v = 2.16e-8 m²                              (§7.2)
  - single via 1.6 mm board ~ 190 K/W              (§7.3)
  - v1 sheet conductances 1.30e-3, 1.50e-2, 8.32e-2 W/K   (§6.3)
  - k_Cu/k_FR4,z ~ 1,345; L1 in-plane contrast ~ 170x (35 um) and ~ 340x (70 um);
    through-plane contrast ~ 30x between no vias and 1 via/mm²   (§7.3)
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from twinheat.spec import (
    Quantity,
    SpecError,
    SpecV1,
    SpecV2,
    export_json,
    iter_quantities,
    load,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"
V1_PATH = CONFIG / "spec_v1.yaml"
V2_PATH = CONFIG / "spec_v2.yaml"


@pytest.fixture(scope="module")
def v1() -> SpecV1:
    spec = load(V1_PATH)
    assert isinstance(spec, SpecV1)
    return spec


@pytest.fixture(scope="module")
def v2() -> SpecV2:
    spec = load(V2_PATH)
    assert isinstance(spec, SpecV2)
    return spec


def _mutated(tmp_path: Path, src: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    mutate(data)
    out = tmp_path / "mutated.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


# ------------------------------------------------------------------ derived numbers


def test_via_area_matches_claude_md(v2: SpecV2) -> None:
    assert v2.via_area == pytest.approx(2.16e-8, rel=0.01)


def test_via_area_formula_from_inputs(v2: SpecV2) -> None:
    d = v2.vias.drill_diameter.value
    t = v2.vias.plating_thickness.value
    assert v2.via_area == pytest.approx(math.pi * (d * t - t * t), rel=1e-12)


def test_single_via_resistance_is_our_convention_not_407(v2: SpecV2) -> None:
    """We use ~190 K/W per via for a 1.6 mm board, not the online calculator's 407 K/W."""
    r = v2.single_via_resistance(1.6e-3)
    assert r == pytest.approx(190.0, rel=0.01)
    assert abs(r - 407.0) > 100.0


@pytest.mark.parametrize(
    ("n_plane", "phi", "t_cu", "expected"),
    [
        (0, 0.0, None, 1.30e-3),  # bare FR-4
        (0, 1.0, 35.0e-6, 1.50e-2),  # one 1 oz layer, phi = 1
        (2, 1.0, 70.0e-6, 8.32e-2),  # two 2 oz planes + painted 2 oz
    ],
)
def test_v1_sheet_conductance(
    v1: SpecV1, n_plane: int, phi: float, t_cu: float | None, expected: float
) -> None:
    assert v1.sheet_conductance(n_plane, phi, t_cu=t_cu) == pytest.approx(expected, rel=0.02)


def test_v1_grid(v1: SpecV1) -> None:
    assert v1.geometry.N.value == 64
    assert v1.dx == pytest.approx(1.5625e-3, rel=1e-12)


def test_v2_grid_and_unknowns(v2: SpecV2) -> None:
    assert v2.geometry.N.value == 256
    assert len(v2.stack.layers) == 4
    assert v2.n_unknowns == 262_144
    assert v2.dx == pytest.approx(0.390625e-3, rel=1e-12)


def test_v2_stack_adds_up_to_1p6_mm_with_35um_copper(v2: SpecV2) -> None:
    dielectric = sum(g.value for g in (v2.stack.d12, v2.stack.d23, v2.stack.d34))
    copper = 4 * 35.0e-6
    assert dielectric + copper == pytest.approx(1.6e-3, rel=1e-9)


def test_v2_material_contrast_cu_over_fr4_z(v2: SpecV2) -> None:
    ratio = v2.materials.k_cu.value / v2.materials.k_fr4_z.value
    assert ratio == pytest.approx(1345, rel=0.01)


@pytest.mark.parametrize(("t_cu", "expected"), [(35.0e-6, 170), (70.0e-6, 340)])
def test_v2_l1_in_plane_contrast(v2: SpecV2, t_cu: float, expected: float) -> None:
    """Copper (phi = 1) over the bare-dielectric part of G_1; confirms D3 'zero' reading."""
    full = v2.in_plane_conductance(1, 1.0, t_cu=t_cu)
    bare = v2.in_plane_conductance(1, 0.0, t_cu=t_cu)
    assert (full - bare) / bare == pytest.approx(expected, rel=0.03)


def test_v2_through_plane_contrast_one_via_per_mm2(v2: SpecV2) -> None:
    bare = v2.through_plane_conductance(1, 0.0)
    dense = v2.through_plane_conductance(1, 1.0e6)  # 1 via/mm² = 1e6 via/m²
    assert dense / bare == pytest.approx(30, rel=0.05)


def test_v2_in_plane_conductance_hand_calc(v2: SpecV2) -> None:
    k_cu, k_xy = 390.0, 0.81
    d12, d23, d34 = 0.20e-3, 1.06e-3, 0.20e-3
    t = 35.0e-6
    # outer layers: missing outer dielectric term = 0 (D3), so only one half-gap counts
    assert v2.in_plane_conductance(1, 0.5) == pytest.approx(k_cu * t * 0.5 + k_xy * 0.5 * d12)
    assert v2.in_plane_conductance(2, 1.0) == pytest.approx(k_cu * t + k_xy * 0.5 * (d12 + d23))
    assert v2.in_plane_conductance(3, 0.7) == pytest.approx(
        k_cu * t * 0.7 + k_xy * 0.5 * (d23 + d34)
    )
    assert v2.in_plane_conductance(4, 0.2) == pytest.approx(k_cu * t * 0.2 + k_xy * 0.5 * d34)


def test_v2_through_plane_conductance_hand_calc(v2: SpecV2) -> None:
    k_z, k_cu, a_v = 0.29, 390.0, math.pi * (0.3e-3 * 25.0e-6 - (25.0e-6) ** 2)
    nu = 5.0e5
    assert v2.through_plane_conductance(1, nu) == pytest.approx((k_z + nu * k_cu * a_v) / 0.20e-3)
    assert v2.through_plane_conductance(2, nu) == pytest.approx((k_z + nu * k_cu * a_v) / 1.06e-3)
    assert v2.through_plane_conductance(3, 0.0) == pytest.approx(k_z / 0.20e-3)


def test_conductance_arguments_are_validated(v2: SpecV2, v1: SpecV1) -> None:
    with pytest.raises(ValueError, match="layer"):
        v2.in_plane_conductance(5, 0.5)
    with pytest.raises(ValueError, match="phi"):
        v2.in_plane_conductance(1, 1.5)
    with pytest.raises(ValueError, match="gap"):
        v2.through_plane_conductance(4, 0.0)
    with pytest.raises(ValueError, match="via density"):
        v2.through_plane_conductance(1, -1.0)
    with pytest.raises(ValueError, match="phi"):
        v1.sheet_conductance(0, -0.1)
    with pytest.raises(ValueError, match="n_plane"):
        v1.sheet_conductance(-1, 0.5)


# ------------------------------------------------------------------ content checks


def test_every_quantity_has_claude_md_source(v1: SpecV1, v2: SpecV2) -> None:
    for spec in (v1, v2):
        found = list(iter_quantities(spec))
        assert len(found) > 40
        for path, q in found:
            assert isinstance(q, Quantity)
            assert q.source.startswith("CLAUDE.md §"), path


def test_v2_design_space_is_36_dimensional(v2: SpecV2) -> None:
    ds = v2.sampler.design_space
    assert ds.chip_position_dims.value + ds.copper_rect_dims.value + ds.via_density_dims.value == 36
    assert ds.copper_rect_dims.value == 6 * 4


def test_v2_has_nine_ood_sets(v2: SpecV2) -> None:
    names = v2.ood.names.value  # ood_5 dropped on 4 layers (docs/decisions.md, OOD sets v2)
    assert len(names) == 9
    assert len(set(names)) == 9
    assert {"ood_bottom_chips", "ood_3oz", "ood_dense_vias"} <= set(names)
    assert "ood_5" not in names
    lo, hi = v2.ood.dense_via_density.value
    assert lo == v2.ood.dense_via_threshold.value and hi > lo


def test_v1_has_eight_ood_axes(v1: SpecV1) -> None:
    assert sorted(v1.ood) == [f"ood_{i}" for i in range(1, 9)]


def test_outer_dielectric_flag_is_zero_per_d3(v2: SpecV2) -> None:
    assert v2.conventions.outer_dielectric_term == "zero"


# ------------------------------------------------------------------ failure modes


def test_negative_k_cu_raises_naming_field(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V1_PATH, lambda d: d["materials"]["k_cu"].update(value=-1.0))
    with pytest.raises(SpecError, match="materials.k_cu"):
        load(path)


def test_missing_field_raises_naming_field(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V1_PATH, lambda d: d["materials"].pop("k_fr4_z"))
    with pytest.raises(SpecError, match="k_fr4_z"):
        load(path)


def test_missing_source_raises(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V1_PATH, lambda d: d["geometry"]["L"].pop("source"))
    with pytest.raises(SpecError, match="geometry.L"):
        load(path)


def test_source_must_point_at_claude_md(tmp_path: Path) -> None:
    path = _mutated(
        tmp_path, V1_PATH, lambda d: d["geometry"]["L"].update(source="my memory of a paper")
    )
    with pytest.raises(SpecError, match="source"):
        load(path)


def test_unknown_field_raises(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V1_PATH, lambda d: d["materials"].update(k_unobtainium={}))
    with pytest.raises(SpecError, match="k_unobtainium"):
        load(path)


def test_unknown_unit_raises(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V1_PATH, lambda d: d["geometry"]["L"].update(unit="furlongs"))
    with pytest.raises(SpecError, match="unit"):
        load(path)


def test_value_outside_range_raises(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V1_PATH, lambda d: d["materials"]["k_cu"].update(value=500.0))
    with pytest.raises(SpecError, match="materials.k_cu"):
        load(path)


def test_inverted_range_raises(tmp_path: Path) -> None:
    path = _mutated(
        tmp_path, V1_PATH, lambda d: d["materials"]["k_cu"].update(range=[400.0, 385.0])
    )
    with pytest.raises(SpecError, match="range"):
        load(path)


def test_numeric_looking_string_raises(tmp_path: Path) -> None:
    """PyYAML reads '35e-6' (no dot) as text; that must fail loudly, not pass as a string."""
    path = tmp_path / "bad.yaml"
    data = yaml.safe_load(V1_PATH.read_text(encoding="utf-8"))
    data["geometry"]["t"]["value"] = "35e-6"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(SpecError, match="geometry.t"):
        load(path)


def test_probabilities_must_sum_to_one(tmp_path: Path) -> None:
    path = _mutated(
        tmp_path, V1_PATH, lambda d: d["sampler"]["copper_oz"].update(p=[0.2, 0.6, 0.3])
    )
    with pytest.raises(SpecError, match="copper_oz"):
        load(path)


def test_outer_dielectric_flag_rejects_other_values(tmp_path: Path) -> None:
    path = _mutated(
        tmp_path, V2_PATH, lambda d: d["conventions"].update(outer_dielectric_term="half")
    )
    with pytest.raises(SpecError, match="outer_dielectric_term"):
        load(path)


def test_wrong_layer_count_raises(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V2_PATH, lambda d: d["stack"]["layers"].pop())
    with pytest.raises(SpecError, match="layers"):
        load(path)


def test_unknown_spec_version_raises(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V1_PATH, lambda d: d.update(spec_version="3"))
    with pytest.raises(SpecError, match="spec_version"):
        load(path)


def test_missing_spec_version_raises(tmp_path: Path) -> None:
    path = _mutated(tmp_path, V1_PATH, lambda d: d.pop("spec_version"))
    with pytest.raises(SpecError, match="spec_version"):
        load(path)


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(SpecError, match="mapping"):
        load(path)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(SpecError, match="YAML"):
        load(path)


def test_missing_file_raises_naming_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load(tmp_path / "nope.yaml")


# ------------------------------------------------------------------ export


@pytest.mark.parametrize("path", [V1_PATH, V2_PATH])
def test_export_json_round_trips(tmp_path: Path, path: Path) -> None:
    spec = load(path)
    out = tmp_path / "spec.json"
    export_json(spec, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["spec_version"] == spec.spec_version
    assert type(spec).model_validate(payload["spec"]) == spec
    assert "dx_m" in payload["derived"]


def test_export_json_v2_has_derived_via_and_conductances(tmp_path: Path, v2: SpecV2) -> None:
    out = tmp_path / "v2.json"
    export_json(v2, out)
    derived = json.loads(out.read_text(encoding="utf-8"))["derived"]
    assert derived["via_area_m2"] == pytest.approx(v2.via_area)
    assert len(derived["in_plane_conductance_full_copper_W_per_K"]) == 4
