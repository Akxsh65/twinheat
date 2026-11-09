// Via columns (CLAUDE.md §13 T1.1, §7.2): one instanced cylinder per via per gap, glowing with the
// vertical heat flux the vias carry there.
//
// Positions come from the board record (the viewer always holds the record it shows: it sends it
// in query / design messages, §10.3): each via array is a square-ish grid at (array + background)
// density, and background vias fill the rest of the board on a grid at the background density
// (the same rule as hardware/kicad/plan.py and the solver's rasteriser, which adds the two).
// Flux at a column in gap l–m: q = ν · k_Cu · A_v · (θ_l − θ_m) / d_lm, in W/m² of board, with ν the
// local via density (vias/m²) and θ read from the CPU copy of the displayed field at the column's
// cell. k_Cu, A_v and d_lm come from src/spec_constants.json (exported from the spec YAML).
// Emissive intensity = |q| / (|q| + q_ref), q_ref = the mean |q| over all columns (a Reinhard tone
// map: a display choice; it keeps the ordering and never saturates).

import * as THREE from "three";

const PER_MM2_TO_PER_M2 = 1e6;
const GAPS = 3;

function grid(x0, y0, w, h, d) {
  const nx = Math.max(1, Math.round(w * Math.sqrt(d)));
  const ny = Math.max(1, Math.round(h * Math.sqrt(d)));
  const out = [];
  for (let b = 0; b < ny; b++) for (let a = 0; a < nx; a++) out.push([x0 + ((a + 0.5) * w) / nx, y0 + ((b + 0.5) * h) / ny]);
  return out;
}

/** Via positions in record mm (origin SW, y north) with their local density in vias/mm². */
export function viaPositions(record, boardMm = 100) {
  const bg = record.background_via_density_per_mm2;
  const arrays = record.via_arrays;
  const out = [];
  for (const a of arrays) {
    const d = a.density_per_mm2 + bg;
    if (d > 0) for (const [x, y] of grid(a.x0, a.y0, a.w, a.h, d)) out.push({ x_mm: x, y_mm: y, density_per_mm2: d });
  }
  if (bg > 0) {
    const inside = (x, y) => arrays.some((a) => x >= a.x0 && x < a.x0 + a.w && y >= a.y0 && y < a.y0 + a.h);
    for (const [x, y] of grid(0, 0, boardMm, boardMm, bg)) {
      if (!inside(x, y)) out.push({ x_mm: x, y_mm: y, density_per_mm2: bg });
    }
  }
  return out;
}

/**
 * One column per via per gap: {x_mm, y_mm, gap (0 = L1–L2), density_per_mm2, flux_W_per_m2,
 * intensity in [0, 1)}. values: Float32Array [4, n, n] of θ in K ([layer, y, x]).
 */
export function viaColumns(record, values, n, spec) {
  const boardMm = spec.L_m * 1000;
  const cols = [];
  for (const p of viaPositions(record, boardMm)) {
    const x = Math.min(n - 1, Math.floor((p.x_mm / boardMm) * n));
    const y = Math.min(n - 1, Math.floor((p.y_mm / boardMm) * n));
    const nu = p.density_per_mm2 * PER_MM2_TO_PER_M2;
    for (let g = 0; g < GAPS; g++) {
      const dtheta = values[(g * n + y) * n + x] - values[((g + 1) * n + y) * n + x];
      const flux = (nu * spec.k_cu_W_per_mK * spec.via_area_m2 * dtheta) / spec.gap_thickness_m[g];
      cols.push({ ...p, gap: g, flux_W_per_m2: flux, intensity: 0 });
    }
  }
  if (cols.length) {
    const ref = cols.reduce((s, c) => s + Math.abs(c.flux_W_per_m2), 0) / cols.length;
    for (const c of cols) c.intensity = ref > 0 ? Math.abs(c.flux_W_per_m2) / (Math.abs(c.flux_W_per_m2) + ref) : 0;
  }
  return cols;
}

// Display choices (not physics).
const RADIUS_M = 0.003; // a 0.3 mm drill drawn 10x larger on the 1 m board, so it stays visible
const COLD = new THREE.Color(0x331a00);
const HOT = new THREE.Color(0xffd27a);

/**
 * InstancedMesh of via columns for the layer stack. update(columns, layerY(l), boardWidth) places
 * each column between the planes of its gap (height = the gap) and colours it by intensity.
 */
export function createViaMesh(maxColumns = 30000) {
  const geom = new THREE.CylinderGeometry(RADIUS_M, RADIUS_M, 1, 10, 1, true);
  const mat = new THREE.MeshBasicMaterial({ toneMapped: false });
  const mesh = new THREE.InstancedMesh(geom, mat, maxColumns);
  mesh.count = 0;
  mesh.frustumCulled = false;
  const m = new THREE.Matrix4();
  const c = new THREE.Color();
  function update(columns, layerY, boardWidth, boardMm = 100) {
    const k = Math.min(columns.length, maxColumns);
    for (let i = 0; i < k; i++) {
      const col = columns[i];
      const top = layerY(col.gap);
      const bottom = layerY(col.gap + 1);
      const h = Math.max(top - bottom, 1e-6);
      m.makeScale(1, h, 1).setPosition((col.x_mm / boardMm - 0.5) * boardWidth, (top + bottom) / 2,
        -(col.y_mm / boardMm - 0.5) * boardWidth);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, c.copy(COLD).lerp(HOT, col.intensity));
    }
    mesh.count = k;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.visible = k > 0;
  }
  return { mesh, update };
}
