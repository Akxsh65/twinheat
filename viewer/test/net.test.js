// Message handler (src/net.js): only contract-valid pred / verified messages reach the layer
// textures (CLAUDE.md §10, invariant B6). Uses the shared contracts/examples, so the viewer is held
// to the same files as tests/test_contracts.py. Runs in Node: no WebGL, only three's DataTexture.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { HalfFloatType, RedFormat } from "three";

import { decode, decodeBits } from "../src/f16.js";
import { createHandler, LayerTextures, N_LAYERS, processMessage } from "../src/net.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const raw = (kind, name) => readFileSync(path.join(ROOT, "contracts", "examples", kind, `${name}.json`), "utf-8");
const FIELD = { pred: "mu", verified: "theta" };

const VALID = [
  ["pred", "valid_1"], // [4, 4, 4]
  ["verified", "valid_2"], // [4, 4, 4]
  ["pred", "valid_3"], // [4, 8, 8]: forces new textures
];
const INVALID = [
  ["pred", "invalid_1"], // bad decision
  ["verified", "invalid_2"], // bad iters
  ["pred", "invalid_3"], // mu byte length does not match shape
];

function fresh() {
  const textures = new LayerTextures();
  const logged = [];
  const handle = createHandler({ textures, log: (m) => logged.push(m) });
  return { textures, handle, logged };
}

const snapshot = (textures) =>
  textures.textures.map((t) => ({ tex: t, version: t.version, data: t.image.data.slice() }));

function assertUntouched(textures, before) {
  textures.textures.forEach((t, l) => {
    assert.equal(t, before[l].tex, `layer ${l}: texture object replaced`);
    assert.equal(t.version, before[l].version, `layer ${l}: needsUpdate set`);
    assert.deepEqual(t.image.data, before[l].data, `layer ${l}: data changed`);
  });
}

function assertShows(textures, kind, name) {
  const msg = JSON.parse(raw(kind, name));
  const [, n] = msg.shape;
  const bits = decodeBits(msg[FIELD[kind]], msg.shape);
  const values = decode(msg[FIELD[kind]], msg.shape);
  assert.equal(textures.n, n);
  textures.textures.forEach((t, l) => {
    assert.equal(t.image.width, n);
    assert.equal(t.image.height, n);
    assert.equal(t.type, HalfFloatType);
    assert.equal(t.format, RedFormat);
    assert.equal(t.flipY, false, "row 0 of the texture must stay y = 0 (south edge)");
    assert.deepEqual(t.image.data, bits.subarray(l * n * n, (l + 1) * n * n), `layer ${l} data`);
  });
  assert.equal(textures.min, Math.min(...values));
  assert.equal(textures.max, Math.max(...values));
}

test("3 valid and 3 invalid messages, interleaved: textures change only for the valid ones", () => {
  const { textures, handle } = fresh();
  for (let i = 0; i < 3; i++) {
    const [vk, vn] = VALID[i];
    const before = snapshot(textures);
    const ok = handle(raw(vk, vn));
    assert.equal(ok.ok, true, `${vk}/${vn} rejected: ${ok.error}`);
    assertShows(textures, vk, vn);
    assert.ok(
      textures.textures.some((t, l) => t !== before[l].tex || t.version > before[l].version),
      `${vk}/${vn}: no texture updated`,
    );
    assert.equal(handle.status.lastType, vk);

    const [ik, iname] = INVALID[i];
    const shown = snapshot(textures);
    const bad = handle(raw(ik, iname));
    assert.equal(bad.ok, false, `${ik}/${iname} accepted`);
    assertUntouched(textures, shown);
    assert.match(handle.status.lastError, new RegExp(`^${ik}: \\$`));
  }
  assert.equal(handle.status.received, 6);
  assert.equal(handle.status.rejected, 3);
});

test("invalid messages are logged with kind and JSON path", () => {
  const { handle, logged } = fresh();
  for (const [kind, name] of INVALID) handle(raw(kind, name));
  assert.equal(logged.length, 3);
  assert.match(logged[0], /pred: \$\.decision/);
  assert.match(logged[1], /verified: \$\.iters/);
  assert.match(logged[2], /pred: \$\.mu/);
});

test("needsUpdate only on change: the same message twice bumps no version", () => {
  const { textures, handle } = fresh();
  handle(raw("pred", "valid_1"));
  const before = snapshot(textures);
  assert.equal(handle(raw("pred", "valid_1")).ok, true);
  assertUntouched(textures, before);
});

test("only the layers whose data changed are re-uploaded", () => {
  const { textures, handle } = fresh();
  const msg = JSON.parse(raw("pred", "valid_1"));
  handle(JSON.stringify(msg));
  const before = snapshot(textures);
  const bits = decodeBits(msg.mu, msg.shape);
  bits[2 * 16 + 5] ^= 0x0001; // one value on layer index 2 (L3), still finite
  msg.mu = Buffer.from(bits.buffer).toString("base64");
  assert.equal(handle(JSON.stringify(msg)).ok, true);
  textures.textures.forEach((t, l) => {
    assert.equal(t, before[l].tex);
    assert.equal(t.version, before[l].version + (l === 2 ? 1 : 0), `layer ${l}`);
  });
});

test("a schema-valid field with a non-finite float16 is rejected before any texture changes", () => {
  const { textures, handle } = fresh();
  handle(raw("pred", "valid_1"));
  const before = snapshot(textures);
  const msg = JSON.parse(raw("pred", "valid_2"));
  const bits = decodeBits(msg.mu, msg.shape);
  bits[bits.length - 1] = 0x7c00; // +inf on the last layer
  msg.mu = Buffer.from(bits.buffer).toString("base64");
  const res = handle(JSON.stringify(msg));
  assert.equal(res.ok, false);
  assert.match(res.error, /non-finite/);
  assertUntouched(textures, before);
});

test("non-JSON, untyped, unknown-type and client->server messages are rejected", () => {
  const { textures, handle } = fresh();
  handle(raw("pred", "valid_1"));
  const before = snapshot(textures);
  for (const bad of [
    "{not json",
    "[1, 2]",
    JSON.stringify({ schema_version: "1.0", id: "x" }),
    JSON.stringify({ schema_version: "1.0", type: "hello", id: "x" }),
    raw("query", "valid_1"),
    raw("design", "valid_1"),
  ]) {
    assert.equal(handle(bad).ok, false, bad.slice(0, 60));
  }
  assertUntouched(textures, before);
  assert.equal(handle.status.rejected, 6);
});

test("valid iterate, batch_progress and error messages are accepted but never rendered", () => {
  const { textures, handle } = fresh();
  handle(raw("verified", "valid_1"));
  const before = snapshot(textures);
  for (const kind of ["iterate", "batch_progress", "error"]) {
    const res = handle(raw(kind, "valid_1"));
    assert.equal(res.ok, true, `${kind}: ${res.error}`);
    assert.equal(handle.status.lastType, kind);
  }
  assertUntouched(textures, before);
  assert.equal(handle.status.rejected, 0);
});

test("textures start as N_LAYERS 1x1 R16F textures at zero", () => {
  const { textures, handle } = fresh();
  assert.equal(N_LAYERS, 4);
  assert.equal(textures.textures.length, 4);
  for (const t of textures.textures) {
    assert.equal(t.image.width, 1);
    assert.equal(t.internalFormat, "R16F");
    assert.deepEqual([...t.image.data], [0]);
  }
  assert.equal(handle.status.lastType, null);
  assert.equal(handle.status.lastError, null);
});

test("worker path: a structured-cloned processMessage result renders exactly like handle(raw)", () => {
  const inline = fresh();
  const viaWorker = fresh();
  for (const [kind, name] of [...VALID, ...INVALID]) {
    const r = processMessage(raw(kind, name));
    const transfer = r.ok && r.field ? [r.field.bits.buffer] : [];
    const a = inline.handle(raw(kind, name));
    const b = viaWorker.handle.apply(structuredClone(r, { transfer }));
    assert.equal(b.ok, a.ok, `${kind}/${name}`);
    assert.deepEqual(b.updated, a.updated, `${kind}/${name}`);
  }
  viaWorker.textures.textures.forEach((t, l) => {
    assert.deepEqual(t.image.data, inline.textures.textures[l].image.data, `layer ${l}`);
  });
  assert.deepEqual(viaWorker.handle.status, inline.handle.status);
  assert.equal(inline.handle.status.worker, null); // set by connect(), not by the handler
});
