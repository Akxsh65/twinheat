// Field arrays on the wire (CLAUDE.md §10): base64 of little-endian IEEE 754 float16, row-major.
//
// Array order is [layer, y, x] (CLAUDE.md §0 rule 7); values are temperature rises in K. Rounding
// is IEEE round-to-nearest-even straight from float64 (no float32 step, so no double rounding);
// contracts/f16.py implements the same codec and tests/fixtures/f16_vectors.json pins both to
// identical bits. Runs in Node >= 20 and in the browser (uses atob/btoa, no Float16Array).
//
// Fails loudly (CLAUDE.md §0 rule 5): non-finite values, float16 overflow (|x| >= 65520),
// malformed shapes and byte-length mismatches all throw.

const BASE64_RE = /^[A-Za-z0-9+/]*={0,2}$/;

/** Return shape as [layer, y, x] of positive integers, or throw naming it. */
export function checkShape(shape) {
  if (
    !Array.isArray(shape) ||
    shape.length !== 3 ||
    !shape.every((d) => typeof d === "number" && Number.isInteger(d) && d > 0)
  ) {
    throw new RangeError(`shape must be 3 positive ints [layer, y, x], got ${JSON.stringify(shape)}`);
  }
  return shape;
}

function roundHalfEven(v) {
  const f = Math.floor(v);
  const d = v - f; // exact: v < 2^11 here
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** IEEE binary16 bits of a finite float64 (round to nearest, ties to even). */
export function toHalfBits(x) {
  if (!Number.isFinite(x)) throw new RangeError(`field array must be finite, got ${x}`);
  const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  const a = Math.abs(x);
  if (a < 2 ** -14) {
    // subnormal: units of 2^-24; a result of 1024 is the smallest normal, same bit pattern
    return sign | roundHalfEven(a * 2 ** 24);
  }
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e -= 1; // log2 can round across a power of two
  if (2 ** (e + 1) <= a) e += 1;
  let mant = roundHalfEven((a / 2 ** e - 1) * 1024); // exact scaling by powers of two
  if (mant === 1024) {
    mant = 0;
    e += 1;
  }
  if (e > 15) throw new RangeError(`field array must be finite in float16; ${x} overflows (>= 65520)`);
  return sign | ((e + 15) << 10) | mant;
}

/** float64 value of IEEE binary16 bits; throws on inf/NaN. */
export function fromHalfBits(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0x1f) throw new RangeError(`field contains a non-finite float16 value (0x${h.toString(16)})`);
  if (e === 0) return s * m * 2 ** -24;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** Encode values (any array-like of numbers, row-major [layer, y, x], K) as base64 float16. */
export function encode(values, shape) {
  const [l, ny, nx] = checkShape(shape);
  const n = l * ny * nx;
  if (values.length !== n) {
    throw new RangeError(`field has ${values.length} values but shape [${shape.join(", ")}] needs ${n}`);
  }
  const bytes = new Uint8Array(2 * n);
  for (let i = 0; i < n; i++) {
    const h = toHalfBits(values[i]);
    bytes[2 * i] = h & 0xff;
    bytes[2 * i + 1] = h >> 8;
  }
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

/** Decode base64 float16 into raw bits (Uint16Array, row-major [layer, y, x]). */
export function decodeBits(b64, shape) {
  const [l, ny, nx] = checkShape(shape);
  if (typeof b64 !== "string" || b64.length === 0 || b64.length % 4 !== 0 || !BASE64_RE.test(b64)) {
    throw new RangeError(`field is not valid padded base64: ${JSON.stringify(String(b64).slice(0, 40))}`);
  }
  const bin = atob(b64);
  const expected = 2 * l * ny * nx;
  if (bin.length !== expected) {
    throw new RangeError(
      `field has ${bin.length} bytes but shape [${shape.join(", ")}] needs ${expected} (2 per value)`,
    );
  }
  const out = new Uint16Array(expected / 2);
  for (let i = 0; i < out.length; i++) out[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
  return out;
}

/** Decode base64 float16 into a Float32Array (exact: every float16 is a float32). */
export function decode(b64, shape) {
  const bits = decodeBits(b64, shape);
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) out[i] = fromHalfBits(bits[i]);
  return out;
}
