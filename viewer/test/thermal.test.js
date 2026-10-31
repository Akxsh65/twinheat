// G1.3 (CLAUDE.md §13 T1.3, §0 rule 7): orientation of solver fields on screen, the probe, the LUTs.
// Runs in Node: three's geometry, camera and Raycaster need no WebGL, so the test follows the
// same chain the viewer uses (texture data -> plane uv -> world -> camera), minus the rasteriser.

import assert from "node:assert/strict";
import { test } from "node:test";

import { Mesh, MeshBasicMaterial, Raycaster, Vector2, Vector3 } from "three";

import { fromHalfBits, toHalfBits } from "../src/f16.js";
import { boardCentre, desktopCamera, layerGeometry, layerOffset, GAP_DEFAULT_M } from "../src/layout.js";
import { DEFAULT_LUT, LUT_NAMES, createLutTexture, lutBytes, lutHex, setLut } from "../src/lut.js";
import { LayerTextures, N_LAYERS } from "../src/net.js";
import { AMBIENT_C, cellAt, niceStep, probeHits, probeValue } from "../src/probe.js";

const N = 16;
const HOT = { x: N - 1, y: 0, theta: 80 }; // south-east corner, layer L1
const WARM = { x: 0, y: N - 1, theta: 40 }; // north-west corner, layer L1

/** An asymmetric [4, N, N] field: a gradient plus a hot SE and a warm NW corner on L1. */
function field() {
  const v = new Float32Array(N_LAYERS * N * N);
  for (let l = 0; l < N_LAYERS; l++)
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) v[(l * N + y) * N + x] = 10 + 0.5 * x + 0.25 * y + l;
  v[HOT.y * N + HOT.x] = HOT.theta;
  v[WARM.y * N + WARM.x] = WARM.theta;
  const bits = Uint16Array.from(v, toHalfBits);
  const vals = Float32Array.from(bits, fromHalfBits);
  return { bits, vals, min: Math.min(...vals), max: Math.max(...vals) };
}

function loaded() {
  const f = field();
  const t = new LayerTextures();
  t.apply(f.bits, N, f.min, f.max);
  return { t, f };
}

/** The layer-0 plane exactly as scene.js places it. */
function layerMesh(l = 0) {
  const m = new Mesh(layerGeometry(), new MeshBasicMaterial());
  m.position.copy(boardCentre());
  m.position.y += layerOffset(l, N_LAYERS, GAP_DEFAULT_M);
  m.userData.layer = l;
  m.updateMatrixWorld();
  return m;
}

/** World point of plane uv (u, v): the plane is affine in uv, so use 3 vertices with known uv. */
function worldAt(mesh, u, v) {
  const pos = mesh.geometry.attributes.position;
  const uv = mesh.geometry.attributes.uv;
  const find = (a, b) => {
    for (let i = 0; i < uv.count; i++) if (uv.getX(i) === a && uv.getY(i) === b) return new Vector3().fromBufferAttribute(pos, i);
    throw new Error(`no vertex with uv (${a}, ${b})`);
  };
  const p00 = find(0, 0);
  const du = find(1, 0).sub(p00);
  const dv = find(0, 1).sub(p00);
  return p00.add(du.multiplyScalar(u)).add(dv.multiplyScalar(v)).applyMatrix4(mesh.matrixWorld);
}

const texelUv = (x, y) => [(x + 0.5) / N, (y + 0.5) / N];

// ---------------------------------------------------------------------------- orientation

test("texture row 0 is y = 0 and column x is x: data[y * N + x] holds field[0, y, x]", () => {
  const { t, f } = loaded();
  const tex = t.textures[0];
  assert.equal(tex.flipY, false);
  assert.equal(tex.image.data[HOT.y * N + HOT.x], f.bits[HOT.y * N + HOT.x]);
  assert.equal(fromHalfBits(tex.image.data[HOT.y * N + HOT.x]), HOT.theta);
  assert.equal(fromHalfBits(tex.image.data[WARM.y * N + WARM.x]), WARM.theta);
});

test("the hot corner (x = N-1, y = 0) appears bottom-right, the warm one (0, N-1) top-left", () => {
  const camera = desktopCamera(16 / 9);
  const mesh = layerMesh();
  const hot = worldAt(mesh, ...texelUv(HOT.x, HOT.y)).project(camera);
  const warm = worldAt(mesh, ...texelUv(WARM.x, WARM.y)).project(camera);
  assert.ok(hot.x > 0 && hot.y < 0, `hot corner at NDC (${hot.x.toFixed(3)}, ${hot.y.toFixed(3)})`);
  assert.ok(warm.x < 0 && warm.y > 0, `warm corner at NDC (${warm.x.toFixed(3)}, ${warm.y.toFixed(3)})`);
  // and on screen at all (inside the view frustum)
  for (const p of [hot, warm]) assert.ok(Math.abs(p.x) < 1 && Math.abs(p.y) < 1 && p.z < 1);
});

test("a ray from the camera through the hot pixel hits the hot cell and the probe reads 80 K", () => {
  const { t } = loaded();
  const camera = desktopCamera(16 / 9);
  const mesh = layerMesh();
  const ndc = worldAt(mesh, ...texelUv(HOT.x, HOT.y)).project(camera);
  const ray = new Raycaster();
  ray.setFromCamera(new Vector2(ndc.x, ndc.y), camera);
  const hits = ray.intersectObject(mesh);
  assert.equal(hits.length, 1);
  const p = probeValue(t, 0, hits[0].uv);
  assert.deepEqual([p.x, p.y], [HOT.x, HOT.y]);
  assert.equal(p.theta, HOT.theta);
  assert.equal(p.celsius, AMBIENT_C + HOT.theta);
});

test("probeHits picks the nearest layer under the ray (L1 on top seen from above)", () => {
  const { t } = loaded();
  const camera = desktopCamera(16 / 9);
  const planes = [0, 1, 2, 3].map(layerMesh);
  const ndc = worldAt(planes[0], 0.5, 0.5).project(camera);
  const ray = new Raycaster();
  ray.setFromCamera(new Vector2(ndc.x, ndc.y), camera);
  const p = probeHits(ray.intersectObjects(planes), t);
  assert.equal(p.layer, 0);
  assert.equal(probeHits([], t), null);
});

// ---------------------------------------------------------------------------- probe

test("the probe at 5 known cells equals the field values, in K and in °C = 25 + θ", () => {
  const { t, f } = loaded();
  const cells = [[0, 0, 0], [0, HOT.y, HOT.x], [0, WARM.y, WARM.x], [2, 7, 3], [3, N - 1, N - 1]];
  for (const [l, y, x] of cells) {
    const p = probeValue(t, l, { x: (x + 0.5) / N, y: (y + 0.5) / N });
    const want = f.vals[(l * N + y) * N + x];
    assert.deepEqual([p.layer, p.y, p.x], [l, y, x]);
    assert.equal(p.theta, want, `L${l + 1} (x ${x}, y ${y})`);
    assert.equal(p.celsius, 25 + want);
  }
});

test("cellAt: cell edges, clamping at u or v = 1, and bad input", () => {
  assert.deepEqual(cellAt({ x: 0, y: 0 }, N), { x: 0, y: 0 });
  assert.deepEqual(cellAt({ x: 1 / N, y: 0.999999 }, N), { x: 1, y: N - 1 });
  assert.deepEqual(cellAt({ x: 1, y: 1 }, N), { x: N - 1, y: N - 1 }); // the far edge belongs to the last cell
  assert.throws(() => cellAt({ x: -0.1, y: 0.5 }, N), /uv/);
  assert.throws(() => cellAt({ x: NaN, y: 0.5 }, N), /uv/);
});

test("probeValue rejects a bad layer", () => {
  const { t } = loaded();
  assert.throws(() => probeValue(t, 4, { x: 0.5, y: 0.5 }), /layer/);
});

test("values follow later fields of the same size and of a new size", () => {
  const { t, f } = loaded();
  const bits = f.bits.slice();
  bits[HOT.y * N + HOT.x] = toHalfBits(90);
  t.apply(bits, N, f.min, 90);
  assert.equal(probeValue(t, 0, { x: (HOT.x + 0.5) / N, y: 0.5 / N }).theta, 90);
  const small = new Uint16Array(N_LAYERS * 4).fill(toHalfBits(7));
  t.apply(small, 2, 7, 7);
  assert.equal(probeValue(t, 3, { x: 0.9, y: 0.9 }).theta, 7);
});

// ---------------------------------------------------------------------------- LUTs

const luminance = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b; // sRGB-encoded, relative only

test("three LUTs, inferno default, each 256 entries", () => {
  assert.deepEqual([...LUT_NAMES], ["inferno", "ironbow", "white-hot"]);
  assert.equal(DEFAULT_LUT, "inferno");
  for (const name of LUT_NAMES) assert.equal(lutBytes(lutHex(name)).length, 4 * 256);
  assert.throws(() => lutHex("rainbow"), /rainbow/);
});

test("white-hot is the grey ramp 0..255; ironbow runs black to white with rising luminance", () => {
  const w = lutBytes(lutHex("white-hot"));
  for (let i = 0; i < 256; i++) assert.deepEqual([...w.subarray(4 * i, 4 * i + 3)], [i, i, i]);
  const ib = lutBytes(lutHex("ironbow"));
  assert.deepEqual([...ib.subarray(0, 3)], [0, 0, 0]);
  assert.deepEqual([...ib.subarray(4 * 255, 4 * 255 + 3)], [255, 255, 255]);
  for (let i = 1; i < 256; i++) {
    const a = luminance(ib.subarray(4 * (i - 1), 4 * i - 1));
    const b = luminance(ib.subarray(4 * i, 4 * i + 3));
    assert.ok(b >= a, `ironbow luminance drops at entry ${i}`);
  }
});

test("setLut switches a LUT texture at runtime", () => {
  const tex = createLutTexture();
  const v0 = tex.version;
  setLut(tex, "white-hot");
  assert.equal(tex.version, v0 + 1);
  assert.deepEqual([...tex.image.data.subarray(4 * 200, 4 * 200 + 3)], [200, 200, 200]);
  assert.throws(() => setLut(tex, "jet"), /jet/);
});

// ---------------------------------------------------------------------------- isoline step

test("niceStep picks 1, 2 or 5 x 10^k giving at most ~10 isolines over the range", () => {
  assert.equal(niceStep(80), 10);
  assert.equal(niceStep(27.5), 5);
  assert.equal(niceStep(15.6), 2);
  assert.equal(niceStep(7.3), 1);
  assert.equal(niceStep(0.6), 0.1);
  assert.equal(niceStep(0), 0); // constant field: no isolines
  for (const span of [0.37, 3.3, 41, 960]) {
    const s = niceStep(span);
    assert.ok(span / s <= 10 && span / s > 2, `${span} -> ${s}`);
  }
});
