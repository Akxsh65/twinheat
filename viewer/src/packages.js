// Chip packages (CLAUDE.md §13 T1.4): an instanced package body on every top-side chip of the record,
// with pins, so the board reads as a populated PCB. Only the record's chips get packages, and each
// body covers exactly the chip footprint where the solver puts its heat (§6.1 A4): the picture never
// shows a heat source the physics does not have. Bottom-side chips are not drawn on the top view.
//
// Style by footprint aspect (long / short side): < 1.5 -> QFN (pads along all four edges), >= 1.5 ->
// SOT-23 style (two gull-wing leads on one long side, one on the other). Pitch, pad and lead sizes
// and body heights are display constants at true scale (the 1 m board is 10x the 100 mm board).

import * as THREE from "three";

// Display constants (mm on the real board).
const QFN_PITCH_MM = 0.5;
const QFN_PAD_MM = [0.25, 0.6]; // across, along the outward direction
const QFN_HEIGHT_MM = 0.9;
const SOT_HEIGHT_MM = 1.1;
const SOT_LEAD_WIDTH_MM = 0.45;
const PAD_HEIGHT_MM = 0.05;
const SOT_ASPECT = 1.5;

/** {bodies: [{kind, x_mm, y_mm, w_mm, h_mm, height_mm}], pins: [{x_mm, y_mm, w_mm, h_mm, height_mm}]} */
export function packageLayout(record) {
  const bodies = [];
  const pins = [];
  for (const c of record.chips) {
    if (c.side !== "top") continue;
    const x = c.x0 + c.w / 2, y = c.y0 + c.h / 2;
    const aspect = Math.max(c.w, c.h) / Math.min(c.w, c.h);
    if (aspect >= SOT_ASPECT) {
      bodies.push({ kind: "sot23", x_mm: x, y_mm: y, w_mm: c.w, h_mm: c.h, height_mm: SOT_HEIGHT_MM });
      const longX = c.w >= c.h; // leads leave the long sides
      const [L, S] = longX ? [c.w, c.h] : [c.h, c.w];
      const leadLen = 0.25 * S;
      const leads = [[-L / 3, -1], [L / 3, -1], [0, 1]]; // (offset along the long side, side)
      for (const [a, side] of leads) {
        const across = side * (S / 2 + leadLen / 2);
        pins.push(longX
          ? { x_mm: x + a, y_mm: y + across, w_mm: SOT_LEAD_WIDTH_MM, h_mm: leadLen, height_mm: PAD_HEIGHT_MM }
          : { x_mm: x + across, y_mm: y + a, w_mm: leadLen, h_mm: SOT_LEAD_WIDTH_MM, height_mm: PAD_HEIGHT_MM });
      }
    } else {
      bodies.push({ kind: "qfn", x_mm: x, y_mm: y, w_mm: c.w, h_mm: c.h, height_mm: QFN_HEIGHT_MM });
      const [padW, padL] = QFN_PAD_MM;
      for (const [len, horizontal] of [[c.w, true], [c.h, false]]) {
        const count = Math.max(2, Math.floor(len / QFN_PITCH_MM) - 1);
        const span = (count - 1) * QFN_PITCH_MM;
        for (let k = 0; k < count; k++) {
          const a = -span / 2 + k * QFN_PITCH_MM;
          for (const side of [-1, 1]) {
            pins.push(horizontal
              ? { x_mm: x + a, y_mm: y + side * (c.h / 2 + padL / 2 - padL * 0.3), w_mm: padW, h_mm: padL, height_mm: PAD_HEIGHT_MM }
              : { x_mm: x + side * (c.w / 2 + padL / 2 - padL * 0.3), y_mm: y + a, w_mm: padL, h_mm: padW, height_mm: PAD_HEIGHT_MM });
          }
        }
      }
    }
  }
  return { bodies, pins };
}

/** Instanced bodies (dark epoxy) and pins (tinned metal); update(layout, boardWidth, topY). */
export function createPackageMeshes(maxBodies = 64, maxPins = 8192) {
  const box = new THREE.BoxGeometry(1, 1, 1);
  const bodies = new THREE.InstancedMesh(box, new THREE.MeshPhysicalMaterial({
    color: 0x1b1b1d, roughness: 0.55, metalness: 0, clearcoat: 0.3, clearcoatRoughness: 0.4 }), maxBodies);
  const pins = new THREE.InstancedMesh(box, new THREE.MeshPhysicalMaterial({
    color: 0xd6d6d0, roughness: 0.3, metalness: 1 }), maxPins);
  for (const m of [bodies, pins]) {
    m.count = 0;
    m.frustumCulled = false;
  }
  const mat = new THREE.Matrix4();
  function place(mesh, items, boardWidth, topY, max) {
    const k = Math.min(items.length, max);
    const s = boardWidth / 100; // m per mm on the 1 m board
    for (let i = 0; i < k; i++) {
      const it = items[i];
      mat.makeScale(it.w_mm * s, it.height_mm * s, it.h_mm * s)
        .setPosition((it.x_mm / 100 - 0.5) * boardWidth, topY + (it.height_mm * s) / 2, -(it.y_mm / 100 - 0.5) * boardWidth);
      mesh.setMatrixAt(i, mat);
    }
    mesh.count = k;
    mesh.instanceMatrix.needsUpdate = true;
  }
  return {
    meshes: [bodies, pins],
    update(layout, boardWidth, topY) {
      place(bodies, layout.bodies, boardWidth, topY, maxBodies);
      place(pins, layout.pins, boardWidth, topY, maxPins);
    },
  };
}
