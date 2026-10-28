"""Rasteriser: board record (mm) -> per-cell SI fields at any N (M1.1; CLAUDE.md §6.4–§7.4).

rasterise(record, spec, N) returns a Raster with the fields solvers/fvm.py reads (phi, q, nu, h_top,
h_bot, t_cu, k_fr4_xy) plus the model's Dirichlet mask, N and dx. This is the only place where mm
become metres (CLAUDE.md §0 rule 6). Arrays are [layer, y, x] / [y, x]; index 0 of y is the south
edge and index 0 of x the west edge (rule 7).

Coverage is the §6.4 rule: SS x SS supersamples per cell at ((i + 0.5)/SS) cell widths, φ = area
fraction. Inside-tests run in exact integers: record geometry lies on the sampler's
geometry_resolution grid (raises otherwise), and supersample coordinates are odd multiples of a
unit in which every grid value is an integer. So rasterise(transform_record(r, s)) ==
transform_array(rasterise(r), s) holds bit for bit (invariant B5).

Shape rules (same as data/sampler.py and the solver tests):
  rect   half-open [x0, x0 + w) x [y0, y0 + h); a sample exactly on an edge raises (it would break
         the symmetry), which cannot happen for the spec's resolution and N divisible by 4
  disc   closed, (x − cx)² + (y − cy)² <= r²
  trace  capsule: points within width/2 of a segment (docs/decisions.md). Axis-aligned segments
         are tested exactly; other segments (OOD-3 45° traces) compare cross² <= r²·|d|² in
         float64 built only from symmetry-invariant integers, so symmetry stays exact.
Layers: φ1 = union of L1 shapes; φ2 = 1 − union of cut-outs if plane else 0; φ3 = union of split
rects; φ4 = union of L4 shapes. Top chips put P on L1 (q), bottom chips on L4 (q_bot), spread by
area fraction over the *sampled* footprint, so Σ (q + q_bot)·Δx² equals the total power exactly.
Via density (arrays + background) is the same on all three gaps (through-hole vias). The forced
zone sets h_top inside it, by area fraction.

q_bot is proposed in docs/decisions.md ("OOD sets v2", point 2) and needs N: until
solvers/fvm.py reads it, the solver silently ignores bottom-chip heat (tests/test_raster.py has a
strict xfail that flips when it does). Do not generate ood_bottom_chips data before then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contracts.validate import validate
from twinheat import spec as specmod

__all__ = ["Raster", "rasterise", "transform_array"]

ROOT = Path(__file__).resolve().parents[1]
_V1 = specmod.load(ROOT / "config" / "spec_v1.yaml")
SS = int(_V1.discretisation.supersample_per_axis.value)  # 4 per axis (§6.4)
T_CU_1OZ = _V1.materials.t_cu.value  # m; 0.5 / 1 / 2 oz scale linearly (§6.3)
PER_MM2_TO_PER_M2 = 1.0e6
N_LAYERS = 4
N_GAPS = 3


@dataclass(frozen=True)
class Raster:
    """Per-cell fields of one board, SI units, read-only arrays [.., y, x].

    phi [4,N,N] copper coverage (1) · q [N,N] W/m² on L1 · q_bot [N,N] W/m² on L4 (bottom chips,
    zero otherwise) · nu [3,N,N] vias/m² for gaps L1–L2, L2–L3, L3–L4 · h_top, h_bot [N,N]
    W/(m²·K) · dirichlet_mask [N,N] 1 on cells next to a Dirichlet edge · t_cu [4] m per layer ·
    k_fr4_xy W/(m·K) · N cells per side · dx m.
    """

    phi: np.ndarray
    q: np.ndarray
    q_bot: np.ndarray
    nu: np.ndarray
    h_top: np.ndarray
    h_bot: np.ndarray
    dirichlet_mask: np.ndarray
    t_cu: np.ndarray
    k_fr4_xy: float
    N: int
    dx: float


_ARRAY_FIELDS = ("phi", "q", "q_bot", "nu", "h_top", "h_bot", "dirichlet_mask")
_MAX_UNITS = 3_000_000_000  # board extent cap: keeps every int64 product below 2^63


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.setflags(write=False)
    return a


class _Grid:
    """Integer coordinates for one N: 1 unit = resolution / (2M) mm, M = N·SS samples per side.

    Sample k sits at (2k + 1)·Lr units, Lr = L / resolution; a grid value g mm is g/res·2M units.
    """

    def __init__(self, n: int, L_mm: float, res_mm: float) -> None:
        lr = L_mm / res_mm
        if not math.isclose(lr, round(lr), rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"L = {L_mm} mm is not a multiple of geometry_resolution {res_mm} mm")
        self.n, self.m, self.res = n, n * SS, res_mm
        self.lr = round(lr)
        self.two_m = 2 * self.m
        if self.lr * self.two_m > _MAX_UNITS:
            raise ValueError(f"N = {n} is too fine for exact int64 inside-tests")
        self.s = (2 * np.arange(self.m, dtype=np.int64) + 1) * self.lr  # sample coordinates

    def units(self, v_mm: float, what: str) -> int:
        g = v_mm / self.res
        gi = round(g)
        if abs(g - gi) > 1e-6:
            raise ValueError(f"{what} = {v_mm} mm is not on the geometry_resolution grid "
                             f"({self.res} mm, spec_v2 sampler.v0_defaults)")
        return gi * self.two_m

    def krange(self, lo: int, hi: int) -> slice:
        """Sample indices k with lo <= s_k <= hi (may be empty)."""
        step = 2 * self.lr
        kmin = max(0, -((self.lr - lo) // step))  # ceil((lo − Lr) / step)
        kmax = min(self.m - 1, (hi - self.lr) // step)
        return slice(kmin, max(kmin, kmax + 1))

    def check_edge(self, e: int, what: str) -> None:
        if (e - self.lr) % (2 * self.lr) == 0 and 0 <= (e - self.lr) // (2 * self.lr) < self.m:
            raise ValueError(f"{what} lies exactly on a supersample point at N = {self.n}; the "
                             "half-open rect rule would break symmetry invariance (B5)")

    # ---------------------------------------------------------------- inside-tests

    def rect(self, out: np.ndarray, r: dict[str, Any], what: str) -> None:
        x0, y0 = self.units(r["x0"], f"{what}.x0"), self.units(r["y0"], f"{what}.y0")
        x1 = x0 + self.units(r["w"], f"{what}.w")
        y1 = y0 + self.units(r["h"], f"{what}.h")
        for e, name in ((x0, "x0"), (x1, "x0 + w"), (y0, "y0"), (y1, "y0 + h")):
            self.check_edge(e, f"{what} edge {name}")
        kx, ky = self.krange(x0, x1), self.krange(y0, y1)
        sx, sy = self.s[kx], self.s[ky]
        out[ky, kx] |= ((sy >= y0) & (sy < y1))[:, None] & ((sx >= x0) & (sx < x1))[None, :]

    def _disc(self, out: np.ndarray, cx: int, cy: int, r: int) -> None:
        kx, ky = self.krange(cx - r, cx + r), self.krange(cy - r, cy + r)
        dx, dy = self.s[kx] - cx, self.s[ky] - cy
        out[ky, kx] |= dy[:, None] ** 2 + dx[None, :] ** 2 <= r * r

    def disc(self, out: np.ndarray, d: dict[str, Any], what: str) -> None:
        self._disc(out, self.units(d["cx"], f"{what}.cx"), self.units(d["cy"], f"{what}.cy"),
                   self.units(d["r"], f"{what}.r"))

    def trace(self, out: np.ndarray, t: dict[str, Any], what: str) -> None:
        half = self.units(t["width"], f"{what}.width") // 2  # width units are even (2M | units)
        pts = [(self.units(px, f"{what}.points"), self.units(py, f"{what}.points"))
               for px, py in t["points"]]
        for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
            if ax != bx and ay != by:
                self._oblique(out, ax, ay, bx, by, half)
                continue
            self._disc(out, ax, ay, half)
            self._disc(out, bx, by, half)
            x_lo, x_hi = (min(ax, bx), max(ax, bx)) if ay == by else (ax - half, ax + half)
            y_lo, y_hi = (ay - half, ay + half) if ay == by else (min(ay, by), max(ay, by))
            kx, ky = self.krange(x_lo, x_hi), self.krange(y_lo, y_hi)
            out[ky, kx] = True  # closed band along the segment: the rest of the capsule

    def _oblique(self, out: np.ndarray, ax: int, ay: int, bx: int, by: int, half: int) -> None:
        """Capsule of a non-axis-aligned segment (OOD-3 45° traces).

        Near an end: integer distance². Along the segment: 0 <= (p−a)·d <= |d|² and
        cross² <= r²·|d|². cross, (p−a)·d and |d|² are integers that a square symmetry only
        permutes or negates, so the float64 comparison gives identical results under symmetry.
        """
        kx = self.krange(min(ax, bx) - half, max(ax, bx) + half)
        ky = self.krange(min(ay, by) - half, max(ay, by) + half)
        px, py = self.s[kx][None, :] - ax, self.s[ky][:, None] - ay
        dx, dy = bx - ax, by - ay
        d2, r2 = dx * dx + dy * dy, half * half
        dot = px * dx + py * dy
        cross = (px * dy - py * dx).astype(np.float64)
        near_a = px * px + py * py <= r2
        near_b = (px - dx) ** 2 + (py - dy) ** 2 <= r2
        band = (dot >= 0) & (dot <= d2) & (cross * cross <= float(r2) * float(d2))
        out[ky, kx] |= near_a | near_b | band

    def shape(self, out: np.ndarray, s: dict[str, Any], what: str) -> None:
        {"rect": self.rect, "disc": self.disc, "trace": self.trace}[s["type"]](out, s, what)

    # ---------------------------------------------------------------- samples -> cells

    def blank(self) -> np.ndarray:
        return np.zeros((self.m, self.m), dtype=bool)

    def frac(self, samples: np.ndarray) -> np.ndarray:
        """Area fraction per cell [N, N]: count / SS², exact in float64."""
        counts = samples.reshape(self.n, SS, self.n, SS).sum(axis=(1, 3), dtype=np.int64)
        return counts / float(SS * SS)

    def union(self, shapes: list[dict[str, Any]], what: str) -> np.ndarray:
        out = self.blank()
        for i, s in enumerate(shapes):
            self.shape(out, s, f"{what}[{i}]")
        return self.frac(out)

    def rect_frac(self, r: dict[str, Any], what: str) -> np.ndarray:
        out = self.blank()
        self.rect(out, r, what)
        return self.frac(out)


def _check_n(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"N must be a positive int, got {n!r}")
    return int(n)


def _dirichlet_mask(n: int, edges: list[str]) -> np.ndarray:
    """1 on cells adjacent to a Dirichlet edge (§6.6), else 0."""
    m = np.zeros((n, n))
    if "W" in edges:
        m[:, 0] = 1.0
    if "E" in edges:
        m[:, -1] = 1.0
    if "S" in edges:
        m[0, :] = 1.0
    if "N" in edges:
        m[-1, :] = 1.0
    return m


def rasterise(record: dict[str, Any], spec: specmod.SpecV2, N: int) -> Raster:
    """Raster of `record` (a board_record, mm) at N x N cells; `spec` is config/spec_v2.yaml.

    Raises ContractError for an invalid record, ValueError for off-grid geometry or a bad N.
    """
    if not isinstance(spec, specmod.SpecV2):
        raise TypeError(f"spec must be a SpecV2 (config/spec_v2.yaml), got {type(spec).__name__}")
    n = _check_n(N)
    validate(record, "board_record")
    rid = record["id"]
    L_m = spec.geometry.L.value
    g = _Grid(n, L_m * 1e3, spec.sampler.v0_defaults.geometry_resolution.value)
    dx = L_m / n

    l1, l2, l3, l4 = record["layers"]
    phi = np.zeros((N_LAYERS, n, n))
    phi[0] = g.union(l1["shapes"], f"{rid}: layers[0].shapes")
    if l2["plane"]:
        phi[1] = 1.0 - g.union(l2["cutouts"], f"{rid}: layers[1].cutouts")
    phi[2] = g.union(l3["split_rects"], f"{rid}: layers[2].split_rects")
    phi[3] = g.union(l4["shapes"], f"{rid}: layers[3].shapes")

    q, q_bot = np.zeros((n, n)), np.zeros((n, n))
    for i, c in enumerate(record["chips"]):
        what = f"{rid}: chips[{i}]"
        f = g.rect_frac(c, what)
        area = f.sum() * dx * dx  # sampled footprint, m²
        if area <= 0.0:
            raise ValueError(f"{what}: footprint covers no supersample at N = {n}")
        target = {"top": q, "bottom": q_bot}[c["side"]]
        target += c["P"] * f / area

    nu_cells = np.full((n, n), record["background_via_density_per_mm2"] * PER_MM2_TO_PER_M2)
    for i, a in enumerate(record["via_arrays"]):
        nu_cells += g.rect_frac(a, f"{rid}: via_arrays[{i}]") * (
            a["density_per_mm2"] * PER_MM2_TO_PER_M2)
    nu = np.stack([nu_cells] * N_GAPS)

    conv = record["convection"]
    h_top = np.full((n, n), float(conv["h_top"]))
    fz = conv["forced_zone"]
    if fz is not None:
        h_top += g.rect_frac(fz, f"{rid}: convection.forced_zone") * (fz["h_top"] - conv["h_top"])
    h_bot = np.full((n, n), float(conv["h_bot"]))

    st = record["stack"]
    oz = (st["outer_oz"], st["inner_oz"], st["inner_oz"], st["outer_oz"])
    return Raster(
        phi=_frozen(phi), q=_frozen(q), q_bot=_frozen(q_bot), nu=_frozen(nu), h_top=_frozen(h_top),
        h_bot=_frozen(h_bot), dirichlet_mask=_frozen(_dirichlet_mask(n, record["dirichlet_edges"])),
        t_cu=_frozen(np.array(oz, dtype=np.float64) * T_CU_1OZ),
        k_fr4_xy=float(st["k_fr4_xy"]), N=n, dx=dx,
    )


def transform_array(raster: Raster, sym: int) -> Raster:
    """Raster under square symmetry sym (0..7), matching data.sampler.transform_record.

    sym = r + 4 f: mirror x -> L − x if f = 1, then r quarter turns counter-clockwise.
    """
    if isinstance(sym, bool) or not isinstance(sym, int | np.integer) or not 0 <= sym < 8:
        raise ValueError(f"sym must be an int in 0..7, got {sym!r}")
    r, f = int(sym) % 4, int(sym) // 4

    def apply(a: np.ndarray) -> np.ndarray:
        if f:
            a = a[..., ::-1]
        return _frozen(np.rot90(a, k=-r, axes=(-2, -1)))  # CCW in [y, x] with y pointing north

    return Raster(**{**raster.__dict__, **{k: apply(getattr(raster, k)) for k in _ARRAY_FIELDS}})
