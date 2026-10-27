"""Validated loader for config/spec_v1.yaml and config/spec_v2.yaml (CLAUDE.md §6, §7).

Every physical constant lives in the YAML files as a `Quantity`: {value, unit, range, source}.
The solver, sampler, viewer and KiCad generator import this module instead of retyping numbers.

Units: SI internally (m, W, K, W/K, W/(m²·K)). Sampler lengths are in mm because board records
use mm (CLAUDE.md §0 rule 6, §10.1); each Quantity's `unit` field says which. theta is a
temperature rise above ambient in K.

Open decision D3 (outer-layer dielectric term of G_l, CLAUDE.md §7.2): decided as "zero" on
2026-10-24 (docs/decisions.md) and selected by `conventions.outer_dielectric_term` in spec_v2.yaml.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

__all__ = [
    "Quantity",
    "SpecError",
    "SpecV1",
    "SpecV2",
    "export_json",
    "iter_quantities",
    "load",
]

KNOWN_UNITS = frozenset(
    {
        "m", "mm", "W", "K", "degC", "oz", "1", "count", "text",
        "W/(m·K)", "W/(m²·K)", "W/K", "K/W", "m²", "via/mm²",
    }
)  # fmt: skip
SOURCE_RE = re.compile(r"^CLAUDE\.md §\d+(\.\d+)*")
Dist = Literal["uniform", "loguniform", "uniform_int", "categorical", "set", "range"]
Scalar = float | int | str

_EXPORT_SCHEMA_VERSION = "1.0"
_REL_TOL = 1e-9


class SpecError(ValueError):
    """A spec file is invalid. The message names the offending field path."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Quantity(_Model):
    """One constant: value, unit, allowed range and where in CLAUDE.md it comes from.

    `dist` describes how a sampler draws it: uniform / loguniform / uniform_int / range take
    value [lo, hi] (`range` means the range is stated but the distribution is not); categorical
    takes value [choices] and p; set takes value [choices] with no probabilities given.
    """

    value: Scalar | list[Scalar]
    unit: str
    source: str
    range: tuple[float, float] | None = None
    dist: Dist | None = None
    p: list[float] | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> Quantity:
        if self.unit not in KNOWN_UNITS:
            raise ValueError(f"unit {self.unit!r} is not one of {sorted(KNOWN_UNITS)}")
        if not SOURCE_RE.match(self.source):
            raise ValueError(f"source {self.source!r} must start with 'CLAUDE.md §<number>'")
        items = self.items
        if not items:
            raise ValueError("value must not be an empty list")
        if self.unit == "text":
            if not all(isinstance(x, str) for x in items):
                raise ValueError("unit 'text' needs string value(s)")
            if self.range is not None:
                raise ValueError("a text quantity cannot have a range")
        else:
            self._validate_numeric(items)
        self._validate_dist(items)
        return self

    @property
    def items(self) -> list[Scalar]:
        """The value as a list (a scalar becomes a one-element list)."""
        return list(self.value) if isinstance(self.value, list) else [self.value]

    def _validate_numeric(self, items: list[Scalar]) -> None:
        for x in items:
            if isinstance(x, str):
                try:
                    float(x)
                except ValueError:
                    raise ValueError(f"value {x!r} is text but unit is {self.unit!r}") from None
                raise ValueError(
                    f"value {x!r} was read as text; YAML needs a dot and a signed exponent "
                    "(write 35.0e-6, not 35e-6)"
                )
            if not math.isfinite(x):
                raise ValueError(f"value {x} is not finite")
        if self.range is None:
            raise ValueError("a numeric quantity needs a range [min, max]")
        lo, hi = self.range
        if lo > hi:
            raise ValueError(f"range {[lo, hi]} has min > max")
        tol = _REL_TOL * max(abs(lo), abs(hi))
        for x in items:
            if x < lo - tol or x > hi + tol:
                raise ValueError(f"value {x} outside range [{lo}, {hi}]")

    def _validate_dist(self, items: list[Scalar]) -> None:
        if self.p is not None and self.dist != "categorical":
            raise ValueError("p is only allowed with dist: categorical")
        d = self.dist
        if d is None:
            return
        if not isinstance(self.value, list):
            raise ValueError(f"dist {d!r} needs a list value")
        if d in ("uniform", "loguniform", "uniform_int", "range"):
            if len(items) != 2:
                raise ValueError(f"dist {d!r} needs value [low, high], got {len(items)} items")
            lo, hi = items
            if lo > hi:
                raise ValueError(f"dist {d!r}: low {lo} > high {hi}")
            if d == "loguniform" and lo <= 0:
                raise ValueError(f"dist 'loguniform' needs low > 0, got {lo}")
            if d == "uniform_int" and not all(float(x).is_integer() for x in items):
                raise ValueError("dist 'uniform_int' needs integer bounds")
        elif d == "categorical":
            if self.p is None:
                raise ValueError("dist 'categorical' needs probabilities p")
            if len(self.p) != len(items):
                raise ValueError(f"p has {len(self.p)} entries but value has {len(items)}")
            if any(x < 0 for x in self.p) or abs(sum(self.p) - 1.0) > _REL_TOL:
                raise ValueError(f"p must be non-negative and sum to 1, got {self.p}")


# ---------------------------------------------------------------------------- shared sections


class Meta(_Model):
    name: str
    version: str
    description: str


class Boundary(_Model):
    n_dirichlet_edges: Quantity
    dirichlet_theta: Quantity
    edge_choice: Quantity


class OodSet(_Model):
    axis: str
    source: str
    params: dict[str, Quantity]

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        if not SOURCE_RE.match(v):
            raise ValueError(f"source {v!r} must start with 'CLAUDE.md §<number>'")
        return v


# ---------------------------------------------------------------------------- spec v1


class GeometryV1(_Model):
    L: Quantity
    t: Quantity
    N: Quantity
    N_transfer: Quantity


class MaterialsV1(_Model):
    t_cu: Quantity
    k_fr4_xy: Quantity
    k_fr4_z: Quantity
    k_cu: Quantity
    t_inf_display: Quantity


class ConvectionV1(_Model):
    natural_training: Quantity
    forced_training: Quantity
    natural_gases: Quantity
    forced_gases: Quantity
    biot_max: Quantity


class Discretisation(_Model):
    supersample_per_axis: Quantity
    min_feature_cells: Quantity
    min_chip_cells: Quantity
    face_conductance: Quantity
    rel_residual_tol: Quantity
    mms_slope: Quantity
    energy_balance_tol: Quantity


class SamplerV1(_Model):
    copper_oz: Quantity
    n_plane: Quantity
    k_fr4_xy: Quantity
    painted_region_count: Quantity
    painted_region_type: Quantity
    painted_region_centre: Quantity
    rect_side: Quantity
    disc_radius: Quantity
    trace_segment_count: Quantity
    trace_width: Quantity
    trace_segment_length: Quantity
    painted_mean_phi: Quantity
    chip_count: Quantity
    chip_short_side: Quantity
    chip_aspect: Quantity
    chip_gap_min: Quantity
    chip_edge_margin_min: Quantity
    chip_placement_max_tries: Quantity
    chip_power: Quantity
    total_power_max: Quantity
    chip_source_density: Quantity
    h_top_base: Quantity
    h_bot_base: Quantity
    forced_zone_probability: Quantity
    forced_zone_side: Quantity
    forced_zone_h_top: Quantity


class EnvelopeV1(_Model):
    theta_max: Quantity
    max_rescaled_fraction: Quantity


class SpecV1(_Model):
    """TwinHeat-2D v1.0: 64² development plate (CLAUDE.md §6)."""

    spec_version: Literal["1"]
    meta: Meta
    geometry: GeometryV1
    materials: MaterialsV1
    convection: ConvectionV1
    discretisation: Discretisation
    sampler: SamplerV1
    envelope: EnvelopeV1
    boundary: Boundary
    ood: dict[str, OodSet]

    @field_validator("ood")
    @classmethod
    def _ood_axes(cls, v: dict[str, OodSet]) -> dict[str, OodSet]:
        want = {f"ood_{i}" for i in range(1, 9)}
        if set(v) != want:
            raise ValueError(f"ood must have exactly {sorted(want)}, got {sorted(v)}")
        return v

    @property
    def dx(self) -> float:
        """Cell size dx = L/N in m."""
        return self.geometry.L.value / self.geometry.N.value

    def sheet_conductance(
        self,
        n_plane: int,
        phi: float,
        t_cu: float | None = None,
        k_fr4_xy: float | None = None,
    ) -> float:
        """Sheet conductance G in W/K (CLAUDE.md §6.2): k_FR4,xy*t + k_Cu*t_Cu*(n_plane + phi).

        n_plane: number of full internal copper planes (count, >= 0); phi: painted copper
        coverage in [0, 1]; t_cu: copper thickness in m (default materials.t_cu);
        k_fr4_xy: in W/(m·K) (default materials.k_fr4_xy).
        """
        if n_plane < 0 or int(n_plane) != n_plane:
            raise ValueError(f"n_plane must be a non-negative integer, got {n_plane}")
        if not 0.0 <= phi <= 1.0:
            raise ValueError(f"phi must be in [0, 1], got {phi}")
        t_cu = self.materials.t_cu.value if t_cu is None else t_cu
        k_xy = self.materials.k_fr4_xy.value if k_fr4_xy is None else k_fr4_xy
        if t_cu <= 0 or k_xy <= 0:
            raise ValueError(f"t_cu and k_fr4_xy must be positive, got {t_cu}, {k_xy}")
        k_cu = self.materials.k_cu.value
        return k_xy * self.geometry.t.value + k_cu * t_cu * (n_plane + phi)

    def derived(self) -> dict[str, Any]:
        """Derived quantities for the JSON export (units in the key names)."""
        t_cu_2oz = 70.0e-6
        return {
            "dx_m": self.dx,
            "sheet_conductance_W_per_K": {
                "bare_fr4": self.sheet_conductance(0, 0.0),
                "one_1oz_layer_phi_1": self.sheet_conductance(0, 1.0),
                "two_2oz_planes_painted_2oz": self.sheet_conductance(2, 1.0, t_cu=t_cu_2oz),
            },
        }


# ---------------------------------------------------------------------------- spec v2


class Conventions(_Model):
    outer_dielectric_term: str

    @field_validator("outer_dielectric_term")
    @classmethod
    def _only_zero(cls, v: str) -> str:
        if v != "zero":
            raise ValueError(
                f"outer_dielectric_term {v!r} is not implemented; only 'zero' (open decision D3, "
                "decided in docs/decisions.md)"
            )
        return v


class GeometryV2(_Model):
    L: Quantity
    N: Quantity
    n_layers: Quantity


class MaterialsV2(_Model):
    k_fr4_xy: Quantity
    k_fr4_z: Quantity
    k_cu: Quantity


class Vias(_Model):
    drill_diameter: Quantity
    plating_thickness: Quantity


class LayerV2(_Model):
    name: str
    role: str
    t_cu: Quantity
    phi: Quantity


class StackV2(_Model):
    layers: list[LayerV2]
    d12: Quantity
    d23: Quantity
    d34: Quantity

    @field_validator("layers")
    @classmethod
    def _four_layers(cls, v: list[LayerV2]) -> list[LayerV2]:
        if len(v) != 4:
            raise ValueError(f"layers: expected exactly 4, got {len(v)}")
        names = [layer.name for layer in v]
        if names != ["L1", "L2", "L3", "L4"]:
            raise ValueError(f"layers must be named L1..L4 in order, got {names}")
        return v


class ConvectionV2(_Model):
    natural_training: Quantity
    forced_training: Quantity


class DesignSpace(_Model):
    dimension: Quantity
    chip_position_dims: Quantity
    copper_rect_count: Quantity
    copper_rect_dims: Quantity
    via_density_dims: Quantity

    @model_validator(mode="after")
    def _dimensions_add_up(self) -> DesignSpace:
        total = (
            self.chip_position_dims.value
            + self.copper_rect_dims.value
            + self.via_density_dims.value
        )
        if total != self.dimension.value:
            raise ValueError(
                f"component dimensions add to {total}, dimension is {self.dimension.value}"
            )
        if self.copper_rect_dims.value != 4 * self.copper_rect_count.value:
            raise ValueError("copper_rect_dims must be 4 * copper_rect_count (x, y, w, h)")
        return self


class SamplerDefaultsV2(_Model):
    """Sampler v0 choices CLAUDE.md leaves open; UNSURE, M to confirm (docs/decisions.md)."""

    outer_oz_p: Quantity
    background_via_density_dist: Quantity
    l2_cutout_side: Quantity
    l3_split_position: Quantity
    l3_split_axis_x_probability: Quantity
    l4_region_count: Quantity
    chip_long_axis_x_probability: Quantity
    chip_position: Quantity
    trace_first_horizontal_probability: Quantity
    trace_turn_ccw_probability: Quantity
    trace_cap: Quantity
    forced_zone_position: Quantity
    coverage_grid_points: Quantity
    resample_max_tries: Quantity
    geometry_resolution: Quantity


class SamplerV2(_Model):
    l2_cutout_count: Quantity
    l3_split_rect_count: Quantity
    thermal_via_probability: Quantity
    thermal_via_density: Quantity
    background_via_density: Quantity
    outer_oz: Quantity
    inner_oz: Quantity
    chip_count: Quantity
    chip_power: Quantity
    total_power_max: Quantity
    design_fraction: Quantity
    design_space: DesignSpace
    v0_defaults: SamplerDefaultsV2


class EnvelopeV2(_Model):
    theta_max: Quantity


class OodV2(_Model):
    names: Quantity
    samples_per_set: Quantity
    dense_via_threshold: Quantity


class SplitsV2(_Model):
    train: Quantity
    val: Quantity
    cal: Quantity
    test: Quantity


class SpecV2(_Model):
    """TwinHeat-Board v2.0: 4-layer main model (CLAUDE.md §7)."""

    spec_version: Literal["2"]
    meta: Meta
    conventions: Conventions
    geometry: GeometryV2
    materials: MaterialsV2
    vias: Vias
    stack: StackV2
    convection: ConvectionV2
    sampler: SamplerV2
    boundary: Boundary
    envelope: EnvelopeV2
    ood: OodV2
    splits: SplitsV2

    @model_validator(mode="after")
    def _layer_count_matches_geometry(self) -> SpecV2:
        if self.geometry.n_layers.value != len(self.stack.layers):
            raise ValueError("geometry.n_layers does not match the number of stack layers")
        return self

    @property
    def dx(self) -> float:
        """Cell size dx = L/N in m."""
        return self.geometry.L.value / self.geometry.N.value

    @property
    def n_unknowns(self) -> int:
        """Number of unknowns n_layers * N² (CLAUDE.md §7.3)."""
        n = int(self.geometry.N.value)
        return int(self.geometry.n_layers.value) * n * n

    @property
    def via_area(self) -> float:
        """Plated-barrel copper area A_v = π (D·t_p − t_p²) in m² (CLAUDE.md §7.2)."""
        d = self.vias.drill_diameter.value
        t = self.vias.plating_thickness.value
        return math.pi * (d * t - t * t)

    @property
    def board_thickness(self) -> float:
        """Nominal board thickness in m: the three dielectrics plus four nominal copper layers."""
        dielectric = sum(g.value for g in (self.stack.d12, self.stack.d23, self.stack.d34))
        return dielectric + sum(layer.t_cu.value for layer in self.stack.layers)

    def single_via_resistance(self, thickness: float | None = None) -> float:
        """Thermal resistance of one via barrel in K/W: thickness / (k_Cu · A_v).

        thickness in m (default board_thickness). This is our convention, about 190 K/W for a
        1.6 mm board; not the 407 K/W of an online calculator.
        """
        thickness = self.board_thickness if thickness is None else thickness
        if thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        return thickness / (self.materials.k_cu.value * self.via_area)

    def in_plane_conductance(self, layer: int, phi: float, t_cu: float | None = None) -> float:
        """In-plane sheet conductance G_l in W/K of layer l = 1..4 (CLAUDE.md §7.2).

        G_l = k_Cu · t_Cu,l · phi + k_FR4,xy · ½ (d_{l-1,l} + d_{l,l+1}). For L1 and L4 the missing
        outer dielectric term is 0 (open decision D3, flag conventions.outer_dielectric_term).
        phi: copper coverage in [0, 1]; t_cu: copper thickness in m (default the layer's nominal,
        must lie in the layer's allowed range).
        """
        if layer not in (1, 2, 3, 4):
            raise ValueError(f"layer must be 1..4, got {layer}")
        if not 0.0 <= phi <= 1.0:
            raise ValueError(f"phi must be in [0, 1], got {phi}")
        spec_layer = self.stack.layers[layer - 1]
        if t_cu is None:
            t_cu = spec_layer.t_cu.value
        else:
            lo, hi = spec_layer.t_cu.range
            if not lo * (1 - _REL_TOL) <= t_cu <= hi * (1 + _REL_TOL):
                raise ValueError(f"t_cu {t_cu} outside layer L{layer} range [{lo}, {hi}]")
        assert self.conventions.outer_dielectric_term == "zero"  # D3: the only implemented option
        gaps = [0.0, self.stack.d12.value, self.stack.d23.value, self.stack.d34.value, 0.0]
        half_dielectric = 0.5 * (gaps[layer - 1] + gaps[layer])
        k_cu = self.materials.k_cu.value
        k_xy = self.materials.k_fr4_xy.value
        return k_cu * t_cu * phi + k_xy * half_dielectric

    def through_plane_conductance(self, gap: int, via_density_per_m2: float) -> float:
        """Through-plane conductance g_lm in W/(m²·K) across gap 1..3 (L1–L2, L2–L3, L3–L4).

        g = (k_FR4,z + ν · k_Cu · A_v) / d, with ν = via_density_per_m2 in vias/m²
        (1 via/mm² = 1e6 via/m²).
        """
        if gap not in (1, 2, 3):
            raise ValueError(f"gap must be 1..3 (L1-L2, L2-L3, L3-L4), got {gap}")
        if via_density_per_m2 < 0:
            raise ValueError(f"via density must be non-negative, got {via_density_per_m2}")
        d = (self.stack.d12, self.stack.d23, self.stack.d34)[gap - 1].value
        k_z = self.materials.k_fr4_z.value
        k_cu = self.materials.k_cu.value
        return (k_z + via_density_per_m2 * k_cu * self.via_area) / d

    def derived(self) -> dict[str, Any]:
        """Derived quantities for the JSON export (units in the key names)."""
        return {
            "dx_m": self.dx,
            "n_unknowns": self.n_unknowns,
            "via_area_m2": self.via_area,
            "board_thickness_m": self.board_thickness,
            "single_via_resistance_K_per_W": self.single_via_resistance(),
            "in_plane_conductance_full_copper_W_per_K": [
                self.in_plane_conductance(layer, 1.0) for layer in (1, 2, 3, 4)
            ],
            "through_plane_conductance_no_vias_W_per_m2K": [
                self.through_plane_conductance(g, 0.0) for g in (1, 2, 3)
            ],
            "through_plane_conductance_1_via_per_mm2_W_per_m2K": [
                self.through_plane_conductance(g, 1.0e6) for g in (1, 2, 3)
            ],
        }


# ---------------------------------------------------------------------------- loading / export

_MODELS: dict[str, type[SpecV1] | type[SpecV2]] = {"1": SpecV1, "2": SpecV2}


def _format_errors(exc: ValidationError) -> str:
    return "\n".join(
        f"  {'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
    )


def load(path: str | Path) -> SpecV1 | SpecV2:
    """Load and validate a spec YAML; the model is chosen by its `spec_version` key.

    Raises FileNotFoundError (names the path) if the file is missing and SpecError (names the
    field path and the problem) if the YAML or any value is invalid.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    version = data.get("spec_version")
    if not isinstance(version, str) or version not in _MODELS:
        raise SpecError(
            f"{path}: spec_version must be one of {sorted(_MODELS)} (as a quoted string), "
            f"got {version!r}"
        )
    try:
        return _MODELS[version].model_validate(data)
    except ValidationError as exc:
        raise SpecError(f"{path}: invalid spec:\n{_format_errors(exc)}") from exc


def iter_quantities(obj: Any, prefix: str = "") -> Iterator[tuple[str, Quantity]]:
    """Yield (dotted path, Quantity) for every Quantity inside a spec."""
    if isinstance(obj, Quantity):
        yield prefix, obj
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            yield from iter_quantities(getattr(obj, name), f"{prefix}.{name}" if prefix else name)
    elif isinstance(obj, list | tuple):
        for i, item in enumerate(obj):
            yield from iter_quantities(item, f"{prefix}[{i}]")
    elif isinstance(obj, dict):
        for key, item in obj.items():
            yield from iter_quantities(item, f"{prefix}.{key}" if prefix else str(key))


def export_json(spec: SpecV1 | SpecV2, path: str | Path) -> None:
    """Write the spec and its derived quantities as JSON for the viewer.

    Layout: {schema_version, spec_version, spec: <the validated spec>, derived: {...}}. Loading
    `spec` back with `type(spec).model_validate` reproduces the spec exactly.
    """
    payload = {
        "schema_version": _EXPORT_SCHEMA_VERSION,
        "spec_version": spec.spec_version,
        "spec": spec.model_dump(mode="json"),
        "derived": spec.derived(),
    }
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
