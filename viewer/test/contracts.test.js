// Contracts v1.0 (CLAUDE.md §10, invariant B6) in JS: the same examples, expected error paths
// and float16 vectors as tests/test_contracts.py, so both validators are held to one set of files.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { ContractError, KINDS, validate } from "../src/contracts.js";
import { decode, decodeBits, encode, fromHalfBits, toHalfBits } from "../src/f16.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const EXAMPLES = path.join(ROOT, "contracts", "examples");
const load = (p) => JSON.parse(readFileSync(p, "utf-8"));
const example = (kind, name) => load(path.join(EXAMPLES, kind, `${name}.json`));
const EXPECTED_ERRORS = load(path.join(EXAMPLES, "expected_errors.json"));
const VECTORS = load(path.join(ROOT, "tests", "fixtures", "f16_vectors.json"));

const EXPECTED_KINDS = [
  "audit_log_row", "batch_progress", "board_record", "design", "error", "hardware_record",
  "iterate", "pred", "query", "session_log_line", "verified",
];
const VALID = ["valid_1", "valid_2", "valid_3"];
const INVALID = ["invalid_1", "invalid_2", "invalid_3"];

// ------------------------------------------------------------------------------ validator

test("kinds match the deliverables and the schema files", () => {
  assert.deepEqual([...KINDS].sort(), EXPECTED_KINDS);
  const files = readdirSync(path.join(ROOT, "contracts", "schemas")).map((f) => f.replace(/\.json$/, ""));
  assert.deepEqual(files.sort(), EXPECTED_KINDS);
});

for (const kind of EXPECTED_KINDS) {
  test(`${kind}: exactly six examples`, () => {
    const names = readdirSync(path.join(EXAMPLES, kind)).map((f) => f.replace(/\.json$/, ""));
    assert.deepEqual(names.sort(), [...INVALID, ...VALID]);
  });

  for (const name of VALID) {
    test(`${kind}/${name} passes`, () => {
      validate(example(kind, name), kind);
    });
  }

  for (const name of INVALID) {
    test(`${kind}/${name} fails at ${EXPECTED_ERRORS[kind][name]}`, () => {
      assert.throws(
        () => validate(example(kind, name), kind),
        (err) => {
          assert.ok(err instanceof ContractError, String(err));
          assert.equal(err.kind, kind);
          assert.equal(err.path, EXPECTED_ERRORS[kind][name]);
          assert.ok(err.message.includes(err.path));
          return true;
        },
      );
    });
  }
}

test("unknown kind raises naming it", () => {
  assert.throws(() => validate({}, "nonsense"), /nonsense/);
});

test("non-object is rejected at $", () => {
  assert.throws(() => validate([], "pred"), (err) => err.path === "$");
});

for (const [kind, field] of [["pred", "mu"], ["iterate", "theta"], ["verified", "theta"]]) {
  test(`${kind}: field byte length must match shape`, () => {
    const msg = example(kind, "valid_1");
    const n = msg.shape[1];
    validate({ ...msg, shape: [4, n, n] }, kind);
    assert.throws(() => validate({ ...msg, shape: [4, n + 1, n + 1] }, kind), (err) => err.path === `$.${field}`);
  });
}

test("validation does not mutate its input", () => {
  const msg = example("pred", "valid_1");
  const before = structuredClone(msg);
  validate(msg, "pred");
  assert.deepEqual(msg, before);
});

// ------------------------------------------------------------------------------ float16 codec

// (float64 input, expected IEEE 754 binary16 bits); same hand-worked list as the Python test.
const HAND_VECTORS = [
  [0.0, 0x0000],
  [-0.0, 0x8000],
  [1.0, 0x3c00],
  [-2.0, 0xc000],
  [0.1, 0x2e66],
  [65504.0, 0x7bff],
  [65519.0, 0x7bff],
  [2 ** -14, 0x0400],
  [2 ** -24, 0x0001],
  [2 ** -25, 0x0000],
  [3 * 2 ** -25, 0x0002],
  [1 + 2 ** -11, 0x3c00],
  [1 + 3 * 2 ** -11, 0x3c02],
  [2 ** -14 - 2 ** -25, 0x0400],
];

test("toHalfBits matches hand-worked bits", () => {
  for (const [x, bits] of HAND_VECTORS) assert.equal(toHalfBits(x), bits, `x = ${x}`);
});

test("encoding is little-endian", () => {
  assert.equal(encode([1.0], [1, 1, 1]), Buffer.from([0x00, 0x3c]).toString("base64"));
});

for (const c of VECTORS.cases) {
  test(`fixture ${c.name}: JS encode equals the Python base64`, () => {
    assert.equal(encode(c.values, c.shape), c.b64);
  });
  test(`fixture ${c.name}: JS decode of the Python base64 gives identical bits`, () => {
    assert.deepEqual(Array.from(decodeBits(c.b64, c.shape)), c.bits);
    const vals = decode(c.b64, c.shape);
    assert.ok(vals instanceof Float32Array);
    assert.equal(vals.length, c.bits.length);
    c.bits.forEach((b, i) => assert.ok(Object.is(vals[i], fromHalfBits(b)), `index ${i}`));
  });
}

test("round trip is exact for float16 values", () => {
  const bits = Array.from({ length: 4 * 16 * 16 }, (_, i) => (i * 2654435761) % 0x7c00);
  const vals = bits.map(fromHalfBits);
  assert.deepEqual(Array.from(decodeBits(encode(vals, [4, 16, 16]), [4, 16, 16])), bits);
});

test("agrees with the platform Float16Array where one exists", { skip: typeof Float16Array === "undefined" && "no Float16Array in this Node" }, () => {
  const xs = [];
  let s = 12345;
  for (let i = 0; i < 20000; i++) {
    s = (s * 1103515245 + 12345) % 2 ** 31;
    xs.push(((s / 2 ** 31) * 2 - 1) * 10 ** ((i % 12) - 7));
  }
  const ref = new Uint16Array(new Float16Array(xs).buffer);
  xs.forEach((x, i) => assert.equal(toHalfBits(x), ref[i], `x = ${x}`));
});

for (const bad of [NaN, Infinity, -Infinity, 65520, -1e6]) {
  test(`encode rejects ${bad}`, () => {
    assert.throws(() => encode([0, bad, 0, 0], [1, 2, 2]), /finite/);
  });
}

test("encode rejects a value count that does not match the shape", () => {
  assert.throws(() => encode(new Float32Array(16), [4, 2, 3]), /shape/);
});

test("decode rejects shape mismatch", () => {
  const b64 = encode(new Float32Array(4 * 8 * 8), [4, 8, 8]);
  assert.throws(() => decode(b64, [4, 8, 9]), /\[4,8,9\]|4, 8, 9/);
  assert.throws(() => decode(b64, [4, 4, 4]), /shape/);
});

for (const shape of [[4, 8], [4, 8, 8, 1], [0, 8, 8], [4, -8, -8], [4, 8.5, 8], [true, 1, 1], "448"]) {
  test(`decode rejects malformed shape ${JSON.stringify(shape)}`, () => {
    assert.throws(() => decode(encode([0], [1, 1, 1]), shape), /shape/);
  });
}

for (const b64 of ["AA=", "AA*A", "AAAA AAAA", ""]) {
  test(`decode rejects bad base64 ${JSON.stringify(b64)}`, () => {
    assert.throws(() => decode(b64, [1, 1, 1]));
  });
}

for (const bits of [0x7c00, 0xfc00, 0x7e00]) {
  test(`decode rejects non-finite half 0x${bits.toString(16)}`, () => {
    const b64 = Buffer.from([bits & 0xff, bits >> 8]).toString("base64");
    assert.throws(() => decode(b64, [1, 1, 1]), /finite/);
  });
}
