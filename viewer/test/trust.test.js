// Trust display (src/trust.js, F7; CLAUDE.md §13 T1.6-T1.7): the badge follows message types only,
// iterate fields are rendered, and a band is shown only when the server sent one.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { createHandler, LayerTextures } from "../src/net.js";
import { describe, nextTrust } from "../src/trust.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const raw = (kind, name) => readFileSync(path.join(ROOT, "contracts", "examples", kind, `${name}.json`), "utf-8");

test("the badge goes Predicted -> Verifying -> Verified, by message type only", () => {
  const textures = new LayerTextures();
  const handle = createHandler({ textures, log: () => {} });
  assert.equal(describe(handle.status.trust).cls, "trust-none");
  handle(raw("pred", "valid_1"));
  assert.equal(describe(handle.status.trust).cls, "trust-amber");
  const it = handle(raw("iterate", "valid_1"));
  assert.ok(it.ok && it.updated.length > 0, "an iterate field is drawn");
  assert.equal(handle.status.trust.state, "verifying");
  assert.match(describe(handle.status.trust).text, /guaranteed band ±/);
  handle(raw("verified", "valid_2"));
  assert.equal(describe(handle.status.trust).cls, "trust-green");
});

test("an invalid message never changes the badge", () => {
  const handle = createHandler({ textures: new LayerTextures(), log: () => {} });
  handle(raw("pred", "valid_1"));
  const before = handle.status.trust;
  handle(raw("iterate", "invalid_1"));
  handle(raw("verified", "invalid_2"));
  assert.equal(handle.status.trust, before);
});

test("no certificate: no band is claimed", () => {
  const v = nextTrust(null, "iterate", "d", { k: 3, band: null });
  assert.match(describe(v).text, /no guaranteed band/);
  const done = nextTrust(v, "verified", "d", { band: null, iters: 9, ms: 5, stop: "no certificate" });
  assert.doesNotMatch(describe(done).text, /±/);
  assert.equal(describe(nextTrust(null, "pred", "q", { decision: "preview", peak: 1 })).cls, "trust-amber");
});

test("with the live gate, the badge shows the gate's decision and B", () => {
  const at = (decision) => describe(nextTrust(null, "pred", "d", { decision, peak: 50, B: 0.123 }));
  assert.match(at("accept").text, /ACCEPTED by the gate · B = 0\.12 K ≤ τ · prediction used, not verified/);
  assert.match(at("flag").text, /FLAGGED by the gate · B = 0\.12 K > τ/);
  assert.match(at("audit").text, /AUDITED \(random check\)/);
  assert.match(at("preview").text, /PREVIEW while dragging · B = 0\.12 K/);
  assert.equal(describe(nextTrust(null, "pred", "d", { decision: "flag", peak: 1, B: null })).text.includes("B ="), false);
});
