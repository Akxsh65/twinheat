// G2.2 (CLAUDE.md §13 T1.4): copper mask, jump flooding, bevel normals, packages. Node tests of the CPU
// mirrors of the GPU passes, against an exact distance transform.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { bevelHeight, bevelNormals, JFA_SIZE, jfaCPU, maskFromRecord, stepSizes } from "../src/jfa.js";
import { packageLayout } from "../src/packages.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FX = JSON.parse(readFileSync(path.join(HERE, "fixtures/via_board.json"), "utf-8"));

const rect = (x0, y0, w, h) => ({ type: "rect", x0, y0, w, h });
const board = (shapes, chips = []) => ({ layers: [{ layer: 1, shapes }], chips });

/** Exact signed distance (pixels) to the nearest boundary pixel, brute force. */
function exactSdf(mask, n) {
  const seeds = [];
  const at = (x, y) => (x < 0 || y < 0 || x >= n || y >= n ? 0 : mask[y * n + x]);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const m = at(x, y);
      if (at(x + 1, y) !== m || at(x - 1, y) !== m || at(x, y + 1) !== m || at(x, y - 1) !== m) seeds.push([x, y]);
    }
  const d = new Float32Array(n * n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      let best = Infinity;
      for (const [sx, sy] of seeds) best = Math.min(best, (sx - x) ** 2 + (sy - y) ** 2);
      d[y * n + x] = (mask[y * n + x] ? 1 : -1) * Math.sqrt(best);
    }
  return d;
}

// ------------------------------------------------------------------------------ mask

test("the copper mask follows the record: rows from the south edge, shapes by their record rules", () => {
  const n = 100; // 1 mm pixels on the 100 mm board
  const m = maskFromRecord(board([rect(10, 20, 30, 10)]), n);
  assert.equal(m.length, n * n);
  assert.equal(m[25 * n + 15], 1); // inside: x 15 mm, y 25 mm (row 25 counted from the south)
  assert.equal(m[75 * n + 15], 0); // mirrored in y: outside
  assert.equal(m.reduce((s, v) => s + v, 0), 300); // 30 × 10 mm²
  const disc = maskFromRecord(board([{ type: "disc", cx: 50, cy: 50, r: 10 }]), n);
  assert.ok(Math.abs(disc.reduce((s, v) => s + v, 0) - Math.PI * 100) < 12);
  const trace = maskFromRecord(board([{ type: "trace", points: [[10, 50], [90, 50]], width: 4 }]), n);
  assert.equal(trace[50 * n + 50], 1);
  assert.equal(trace[53 * n + 50], 0);
});

// ------------------------------------------------------------------------------ JFA

test("jump flooding takes log2(N) passes with halving steps", () => {
  assert.deepEqual(stepSizes(JFA_SIZE), [256, 128, 64, 32, 16, 8, 4, 2, 1]);
  assert.equal(stepSizes(512).length, Math.log2(512));
  assert.throws(() => stepSizes(500), /power of two/);
});

for (const [name, shapes] of [
  ["rectangle", [rect(20, 30, 40, 25)]],
  ["disc and trace", [{ type: "disc", cx: 30, cy: 60, r: 14 }, { type: "trace", points: [[50, 20], [85, 20], [85, 70]], width: 6 }]],
  ["the fixture board's L1", FX.record.layers[0].shapes],
]) {
  test(`JFA signed distance matches the exact transform within 1.5 px: ${name}`, () => {
    const n = 64;
    const mask = maskFromRecord(board(shapes), n);
    const { dist } = jfaCPU(mask, n);
    const ref = exactSdf(mask, n);
    let worst = 0;
    for (let i = 0; i < n * n; i++) {
      assert.equal(Math.sign(dist[i]) || 1, Math.sign(ref[i]) || 1, `sign at pixel ${i}`);
      if (Number.isFinite(ref[i])) worst = Math.max(worst, Math.abs(dist[i] - ref[i]));
    }
    assert.ok(worst <= 1.5, `worst JFA error ${worst.toFixed(3)} px`);
  });
}

test("JFA with no copper edge gives -infinity-like distances everywhere (nothing to bevel)", () => {
  const { dist } = jfaCPU(new Uint8Array(16 * 16), 16);
  assert.ok(dist.every((d) => d < -1e3));
});

// ------------------------------------------------------------------------------ bevel

test("bevel height: 0 off the copper, 1 deeper than the bevel width inside, monotone", () => {
  const w = 0.4;
  assert.equal(bevelHeight(-1, w), 0);
  assert.equal(bevelHeight(0, w), 0);
  assert.equal(bevelHeight(w, w), 1);
  assert.equal(bevelHeight(3 * w, w), 1);
  let prev = 0;
  for (let d = 0; d <= w; d += w / 20) {
    const h = bevelHeight(d, w);
    assert.ok(h >= prev);
    prev = h;
  }
});

test("bevel normals point up, and tilt outwards at a copper edge", () => {
  const n = 64;
  const mm = 100 / n;
  const mask = maskFromRecord(board([rect(20, 20, 60, 60)]), n);
  const { dist } = jfaCPU(mask, n);
  const nrm = bevelNormals(dist, n, mm, 3 * mm, 1.0);
  for (let i = 0; i < n * n; i++) assert.ok(nrm[3 * i + 2] > 0, "z up");
  const y = 32;
  const west = Math.round(20 / mm); // west edge of the rectangle, x = 20 mm
  const i = y * n + west + 1; // just inside, on the bevel slope
  assert.ok(nrm[3 * i] < 0, "the normal leans west (outwards) on the west edge");
  const flat = y * n + 32;
  assert.ok(Math.abs(nrm[3 * flat]) < 1e-6 && Math.abs(nrm[3 * flat + 1]) < 1e-6, "flat in the middle");
});

// ------------------------------------------------------------------------------ packages

test("one package per chip, its body on the chip footprint, pins around it", () => {
  const rec = FX.record;
  const pk = packageLayout(rec);
  assert.equal(pk.bodies.length, rec.chips.length);
  pk.bodies.forEach((b, k) => {
    const c = rec.chips[k];
    assert.deepEqual([b.x_mm, b.y_mm, b.w_mm, b.h_mm], [c.x0 + c.w / 2, c.y0 + c.h / 2, c.w, c.h]);
    assert.ok(["qfn", "sot23"].includes(b.kind));
  });
  assert.ok(pk.pins.length >= rec.chips.length * 3);
  for (const p of pk.pins) assert.ok(p.x_mm > 0 && p.x_mm < 100 && p.y_mm > 0 && p.y_mm < 100);
  const elongated = packageLayout(board([], [{ x0: 40, y0: 40, w: 9, h: 4.7, P: 1, side: "top" }]));
  assert.equal(elongated.bodies[0].kind, "sot23");
  assert.equal(elongated.pins.length, 3);
  const square = packageLayout(board([], [{ x0: 40, y0: 40, w: 8, h: 8, P: 1, side: "top" }]));
  assert.equal(square.bodies[0].kind, "qfn");
  assert.ok(square.pins.length % 4 === 0 && square.pins.length > 4);
  assert.equal(packageLayout(board([], [{ x0: 40, y0: 40, w: 8, h: 8, P: 1, side: "bottom" }])).bodies.length, 0);
});
