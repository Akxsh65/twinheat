// WebSocket client and message handler (CLAUDE.md §10.3, invariant B6).
//
// Every server message is parsed and validated (src/contracts.js) before anything else happens.
// Invalid messages are logged, counted and never rendered. Valid pred / verified messages carry a
// [4, n, n] float16 field (pred: mu, verified: theta, K); it is decoded and checked completely
// before any texture is touched, so a bad field can never leave the board half-updated. iterate,
// batch_progress and error are validated and recorded but not rendered yet (T1.7 comes later).
// Parse + validate + decode cost ~60 ms for a 256 x 256 x 4 pred (two 700 kB base64 fields), so
// in the browser they run in a worker (src/net.worker.js) and only the texture copy is on the
// main thread; Node tests run the same two halves inline through handle(raw).
//
// Texture layout: one R16F DataTexture per layer, fed the raw float16 bits (no conversion).
// Layer l is the slice [l, :, :] of the row-major [layer, y, x] array; texture row 0 is y = 0, the
// south edge (CLAUDE.md §0 rule 7), so flipY stays false and uv (0, 0) is the board's SW corner.

import { DataTexture, HalfFloatType, LinearFilter, RedFormat } from "three";

import { validate } from "./contracts.js";
import { decodeBits, fromHalfBits } from "./f16.js";

export const N_LAYERS = 4;
/** Message kinds a server may send (CLAUDE.md §10.3); anything else is rejected. */
export const SERVER_KINDS = Object.freeze(["pred", "iterate", "verified", "batch_progress", "error"]);
/** Kinds that are rendered, and the field each one renders. */
export const RENDERED_FIELD = Object.freeze({ pred: "mu", verified: "theta" });

function makeTexture(n, data) {
  const t = new DataTexture(data, n, n, RedFormat, HalfFloatType);
  t.internalFormat = "R16F";
  t.flipY = false;
  t.minFilter = LinearFilter; // R16F is filterable in WebGL2
  t.magFilter = LinearFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}

function sameBits(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * The 4 layer textures plus the colour range (K) of the field they show, and a CPU-side Float32
 * copy of the same values ([layer, y, x], K) for the probe (src/probe.js).
 */
export class LayerTextures {
  constructor() {
    this.n = 1;
    this.min = 0;
    this.max = 0;
    this.textures = Array.from({ length: N_LAYERS }, () => makeTexture(1, new Uint16Array(1)));
    this.values = new Float32Array(N_LAYERS);
  }

  /** θ (K) at layer l (0 = L1), row y (0 = south), column x (0 = west), exactly as displayed. */
  valueAt(l, y, x) {
    const n = this.n;
    if (!(Number.isInteger(l) && l >= 0 && l < N_LAYERS)) throw new RangeError(`layer must be 0..${N_LAYERS - 1}, got ${l}`);
    if (!(Number.isInteger(x) && Number.isInteger(y) && x >= 0 && y >= 0 && x < n && y < n)) {
      throw new RangeError(`cell (x ${x}, y ${y}) outside the ${n} x ${n} grid`);
    }
    return this.values[(l * n + y) * n + x];
  }

  /**
   * Show a decoded field. bits: Uint16Array of float16 bits, row-major [layer, y, x]; n: side.
   * Textures are replaced only when n changes; otherwise a layer is re-uploaded (needsUpdate)
   * only if its bits differ from what it already shows. Returns the indices of updated layers.
   */
  apply(bits, n, min, max) {
    const per = n * n;
    if (bits.length !== N_LAYERS * per) {
      throw new RangeError(`field has ${bits.length} values, expected ${N_LAYERS} x ${n} x ${n}`);
    }
    const updated = [];
    if (n !== this.n) {
      for (const t of this.textures) t.dispose();
      this.textures = Array.from({ length: N_LAYERS }, (_, l) => {
        updated.push(l);
        return makeTexture(n, bits.slice(l * per, (l + 1) * per));
      });
      this.n = n;
      this.values = new Float32Array(N_LAYERS * per);
    } else {
      this.textures.forEach((t, l) => {
        const slice = bits.subarray(l * per, (l + 1) * per);
        if (sameBits(t.image.data, slice)) return;
        t.image.data.set(slice);
        t.needsUpdate = true;
        updated.push(l);
      });
    }
    const table = halfTable();
    for (const l of updated) {
      for (let i = l * per; i < (l + 1) * per; i++) this.values[i] = table[bits[i]];
    }
    this.min = min;
    this.max = max;
    return updated;
  }
}

/** Parse and validate one raw message; return {kind, msg} or throw (message names the problem). */
export function parseMessage(raw) {
  let msg;
  try {
    msg = JSON.parse(raw);
  } catch (e) {
    throw new SyntaxError(`message is not JSON (${e.message}): ${String(raw).slice(0, 60)}`);
  }
  if (msg === null || typeof msg !== "object" || Array.isArray(msg)) {
    throw new TypeError(`message must be a JSON object, got ${String(raw).slice(0, 60)}`);
  }
  const kind = msg.type;
  if (!SERVER_KINDS.includes(kind)) {
    throw new TypeError(`message type ${JSON.stringify(kind)} is not a server message (${SERVER_KINDS.join(", ")})`);
  }
  validate(msg, kind); // throws ContractError "<kind>: <path>: <detail>"
  return { kind, msg };
}

// float16 bits -> float32 value, NaN for inf / NaN patterns; built on first use (256 KiB)
let halfToFloat = null;
function halfTable() {
  if (halfToFloat === null) {
    halfToFloat = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) halfToFloat[h] = ((h >> 10) & 0x1f) === 0x1f ? NaN : fromHalfBits(h);
  }
  return halfToFloat;
}

/** Decode the rendered field of a valid pred / verified message; throws on non-finite values. */
export function decodeField(msg, kind) {
  const bits = decodeBits(msg[RENDERED_FIELD[kind]], msg.shape);
  const table = halfTable();
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < bits.length; i++) {
    const v = table[bits[i]];
    if (v !== v) {
      throw new RangeError(`${kind}: $.${RENDERED_FIELD[kind]}: value ${i} is a non-finite float16 (0x${bits[i].toString(16)})`);
    }
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { bits, n: msg.shape[1], min, max };
}

/**
 * The expensive half of message handling (parse, validate, decode); runs in src/net.worker.js
 * or inline. Returns {ok: true, kind, field: {bits, n, min, max} | null} or {ok: false, error}.
 * Never throws, so a bad message cannot kill the worker.
 */
export function processMessage(raw) {
  try {
    const { kind, msg } = parseMessage(raw);
    return { ok: true, kind, field: RENDERED_FIELD[kind] ? decodeField(msg, kind) : null };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

/**
 * Return handle(raw) -> {ok, kind, updated} | {ok: false, error}, which runs processMessage
 * inline. handle.apply(result) takes a processMessage result (e.g. from the worker) and does the
 * cheap half: count it and, if it is valid and rendered, update the textures. handle.status holds
 * what the dev overlay shows: lastType (last valid message), lastError, received, rejected.
 */
export function createHandler({ textures, log = console.warn }) {
  const status = { lastType: null, lastError: null, received: 0, rejected: 0, worker: null };
  function reject(error) {
    status.rejected += 1;
    status.lastError = error;
    log(`rejected message: ${error}`);
    return { ok: false, error };
  }
  function apply(result) {
    status.received += 1;
    if (!result.ok) return reject(result.error);
    let updated = [];
    if (result.field) {
      const f = result.field;
      try {
        updated = textures.apply(f.bits, f.n, f.min, f.max);
      } catch (e) {
        return reject(e.message);
      }
    }
    status.lastType = result.kind;
    return { ok: true, kind: result.kind, updated };
  }
  const handle = (raw) => apply(processMessage(raw));
  handle.apply = apply;
  handle.status = status;
  return handle;
}

/**
 * Keep a WebSocket open to url and hand every text frame to handle. With `worker` (running
 * src/net.worker.js) frames are parsed, validated and decoded off the main thread, in arrival
 * order, and only handle.apply runs here; without it handle(raw) runs inline. Reconnects after a
 * close with exponential back-off (retryMs doubling up to maxRetryMs). Returns {close()}.
 * onState(state) gets "connecting" | "open" | "closed". The worker posts {ready: true} once its
 * modules have loaded; without it after workerTimeoutMs the overlay shows an error.
 */
export function connect(
  url,
  handle,
  { worker = null, onState = () => {}, retryMs = 500, maxRetryMs = 8000, workerTimeoutMs = 10000 } = {},
) {
  let ws = null;
  let delay = retryMs;
  let timer = null;
  let stopped = false;
  if (worker) {
    // a worker that fails to load, hangs or crashes must not look like a quiet stream
    handle.status.worker = "loading";
    const fail = (why) => {
      handle.status.worker = "failed";
      handle.apply({ ok: false, error: `message worker ${why}; reload, or use ?inline` });
    };
    const readyTimer = setTimeout(() => fail(`not ready after ${workerTimeoutMs} ms`), workerTimeoutMs);
    worker.onmessage = (ev) => {
      if (ev.data && ev.data.ready === true) {
        clearTimeout(readyTimer);
        handle.status.worker = "ready";
        return;
      }
      handle.apply(ev.data);
    };
    worker.onerror = (ev) => {
      clearTimeout(readyTimer);
      fail(`failed: ${ev.message ?? "unknown error"}`);
    };
  } else {
    handle.status.worker = "inline";
  }

  function open() {
    onState("connecting");
    ws = new WebSocket(url);
    ws.onopen = () => {
      delay = retryMs;
      onState("open");
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") {
        handle.apply({ ok: false, error: "binary frame received; contracts v1.0 messages are JSON text" });
      } else if (worker) {
        worker.postMessage(ev.data);
      } else {
        handle(ev.data);
      }
    };
    ws.onclose = () => {
      onState("closed");
      if (stopped) return;
      timer = setTimeout(open, delay);
      delay = Math.min(2 * delay, maxRetryMs);
    };
    ws.onerror = () => ws.close();
  }

  open();
  return {
    close() {
      stopped = true;
      clearTimeout(timer);
      if (ws) ws.close();
    },
  };
}
