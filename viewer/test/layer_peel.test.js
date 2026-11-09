// G2.1 (CLAUDE.md §13 T1.1, §7.2): layer peel, via columns, FR-4 OIT, cut plane. Node tests of the
// pure parts; the via fixture is a real board solved by solvers/fvm.py
// (tests/fixtures/make_via_fixture.py), so "brightest" is physics, not a test pattern.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import * as THREE from "three";

import { createPeel, peelStep, setPeelTarget, PEEL_RANGE_M } from "../src/layerPeel.js";
import { heightToLayer, sectionValue } from "../src/cutPlane.js";
import { BLEND, wboitComposite } from "../src/oit.js";
import { viaColumns, viaPositions } from "../src/vias.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SPEC = JSON.parse(readFileSync(path.join(HERE, "../src/spec_constants.json"), "utf-8"));
const FX = JSON.parse(readFileSync(path.join(HERE, "fixtures/via_board.json"), "utf-8"));
const VALUES = Float32Array.from(FX.theta);

const inRect = (p, r) => p.x_mm >= r.x0 && p.x_mm <= r.x0 + r.w && p.y_mm >= r.y0 && p.y_mm <= r.y0 + r.h;

// ------------------------------------------------------------------------------ vias

test("a board with no vias has no columns", () => {
  const rec = { ...FX.record, via_arrays: [], background_via_density_per_mm2: 0 };
  assert.deepEqual(viaPositions(rec), []);
  assert.deepEqual(viaColumns(rec, VALUES, FX.N, SPEC), []);
});

test("vias under one chip only: that chip's columns carry the most heat", () => {
  const cols = viaColumns(FX.record, VALUES, FX.N, SPEC);
  const chip0 = FX.record.chips[0];
  const under = cols.filter((c) => inRect(c, chip0));
  const elsewhere = cols.filter((c) => !inRect(c, chip0));
  assert.ok(under.length > 0 && elsewhere.length > 0, "array and background columns both exist");
  const best = cols.reduce((a, b) => (b.intensity > a.intensity ? b : a));
  assert.ok(inRect(best, chip0), `brightest column at (${best.x_mm}, ${best.y_mm}) is not under chip 0`);
  const maxUnder = Math.max(...under.map((c) => c.intensity));
  const maxElse = Math.max(...elsewhere.map((c) => c.intensity));
  assert.ok(maxUnder > maxElse);
  for (const c of cols) assert.ok(c.intensity >= 0 && c.intensity < 1, "tone-mapped into [0, 1)");
});

test("each column's flux is ν · k_Cu · A_v · (θ_l − θ_m) / d_lm, W/m² (CLAUDE.md §7.2, §13 T1.1)", () => {
  const cols = viaColumns(FX.record, VALUES, FX.N, SPEC);
  const n = FX.N;
  for (const c of [...cols.slice(0, 10), ...cols.slice(-10)]) {
    const x = Math.min(n - 1, Math.floor((c.x_mm / 100) * n));
    const y = Math.min(n - 1, Math.floor((c.y_mm / 100) * n));
    const top = VALUES[(c.gap * n + y) * n + x];
    const bottom = VALUES[((c.gap + 1) * n + y) * n + x];
    const nu = c.density_per_mm2 * 1e6; // vias/m²
    const want = (nu * SPEC.k_cu_W_per_mK * SPEC.via_area_m2 * (top - bottom)) / SPEC.gap_thickness_m[c.gap];
    assert.ok(Math.abs(c.flux_W_per_m2 - want) <= 1e-9 * Math.abs(want) + 1e-12, `gap ${c.gap}`);
  }
  assert.deepEqual([...new Set(cols.map((c) => c.gap))].sort(), [0, 1, 2]); // one column per via per gap
});

test("via positions follow the record's densities (array + background inside, background outside)", () => {
  const rec = FX.record;
  const a = rec.via_arrays[0];
  const pos = viaPositions(rec);
  const inside = pos.filter((p) => inRect(p, a)).length;
  const want = (a.density_per_mm2 + rec.background_via_density_per_mm2) * a.w * a.h;
  assert.ok(Math.abs(inside - want) <= 0.25 * want + 2, `${inside} vias in the array, expected ~${want}`);
  const outside = pos.length - inside;
  const wantOut = rec.background_via_density_per_mm2 * (100 * 100 - a.w * a.h);
  assert.ok(Math.abs(outside - wantOut) <= 0.25 * wantOut + 5, `${outside} background vias, expected ~${wantOut}`);
});

// ------------------------------------------------------------------------------ peel

test("peel: the slider sets the target; the separation follows it smoothly and stays in [0, 1]", () => {
  let st = createPeel(0.2);
  st = setPeelTarget(st, 0.8);
  const seen = [];
  for (let k = 0; k < 60; k++) {
    st = peelStep(st, { bothGrips: false }, 1 / 60);
    seen.push(st.s);
  }
  assert.ok(seen.every((v, i) => i === 0 || v >= seen[i - 1]), "monotone, no overshoot");
  assert.ok(Math.abs(st.s - 0.8) < 1e-3);
  assert.equal(setPeelTarget(st, 7).target, 1);
  assert.equal(setPeelTarget(st, -1).target, 0);
});

test("peel in VR: spreading the controllers while both grips are held opens the stack", () => {
  let st = createPeel(0.3);
  st = peelStep(st, { bothGrips: true, distance: 0.4 }, 1 / 72); // grab starts at 0.4 m
  for (let k = 0; k < 200; k++) st = peelStep(st, { bothGrips: true, distance: 0.4 + PEEL_RANGE_M / 2 }, 1 / 72);
  assert.ok(Math.abs(st.s - 0.8) < 1e-3, `s = ${st.s}`);
  for (let k = 0; k < 10; k++) st = peelStep(st, { bothGrips: false, distance: 0.1 }, 1 / 72);
  assert.ok(Math.abs(st.s - 0.8) < 1e-3, "releasing keeps the separation");
  st = peelStep(st, { bothGrips: true, distance: 0.4 }, 1 / 72); // a new grab: relative to 0.4 m again
  assert.ok(Math.abs(st.s - 0.8) < 1e-3, "starting a grab does not jump");
  for (let k = 0; k < 200; k++) st = peelStep(st, { bothGrips: true, distance: 5 }, 1 / 72);
  assert.ok(st.s <= 1 && st.s > 0.99, "clamped at fully open");
});

// ------------------------------------------------------------------------------ cut plane

test("cut-plane section: layer values at layer heights, linear in z between them", () => {
  const n = FX.N;
  const u = { x: 0.37, y: 0.61 };
  const x = Math.floor(u.x * n), y = Math.floor(u.y * n);
  const layer = (l) => VALUES[(l * n + y) * n + x];
  for (let l = 0; l < 4; l++) assert.equal(sectionValue(VALUES, n, u, l), layer(l));
  const mid = sectionValue(VALUES, n, u, 1.25);
  assert.ok(Math.abs(mid - (0.75 * layer(1) + 0.25 * layer(2))) < 1e-6);
  const gap = 0.05;
  assert.equal(heightToLayer(1.5 * gap, gap), 0); // L1 plane, on top
  assert.equal(heightToLayer(-1.5 * gap, gap), 3); // L4 plane
  assert.equal(heightToLayer(10, gap), 0); // clamped
});

// ------------------------------------------------------------------------------ OIT

test("weighted blended OIT: the result does not depend on draw order", () => {
  const frags = [
    { rgb: [0.9, 0.8, 0.3], a: 0.35, z: 0.3 },
    { rgb: [0.2, 0.7, 0.4], a: 0.35, z: 0.5 },
    { rgb: [0.6, 0.1, 0.9], a: 0.5, z: 0.7 },
  ];
  const bg = [0.1, 0.1, 0.12];
  const ref = wboitComposite(frags, bg);
  for (const order of [[2, 1, 0], [1, 2, 0], [0, 2, 1]]) {
    const out = wboitComposite(order.map((i) => frags[i]), bg);
    out.forEach((v, c) => assert.ok(Math.abs(v - ref[c]) < 1e-12));
  }
});

test("weighted blended OIT: one fragment gives exactly ordinary alpha blending", () => {
  const f = { rgb: [0.9, 0.5, 0.2], a: 0.4, z: 0.42 };
  const bg = [0.1, 0.2, 0.3];
  wboitComposite([f], bg).forEach((v, c) => assert.ok(Math.abs(v - (f.a * f.rgb[c] + (1 - f.a) * bg[c])) < 1e-12));
  wboitComposite([], bg).forEach((v, c) => assert.equal(v, bg[c]));
});

test("OIT blend states are the stated ones (accumulate additively, revealage multiplicatively)", () => {
  assert.deepEqual(BLEND.accum, { src: THREE.OneFactor, dst: THREE.OneFactor, target: "RGBA16F" });
  assert.deepEqual(BLEND.revealage, { src: THREE.ZeroFactor, dst: THREE.OneMinusSrcColorFactor, target: "R8" });
  assert.deepEqual(BLEND.composite, { src: THREE.SrcAlphaFactor, dst: THREE.OneMinusSrcAlphaFactor, target: "screen" });
});
