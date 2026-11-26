// Proposer replay panel helpers (src/replay.js): frames are found by message id, described from
// the meta only, and the chart data carries the whole run with the cursor at the frame.

import assert from "node:assert/strict";
import { test } from "node:test";

import { chartData, frameOf, frameText } from "../src/replay.js";

const meta = {
  schema: "twinheat-replay/1", proposer: "P2", x_label: "proposal", x_max: 9,
  trace: {
    points: [[0, 50, "start"], [1, 49, "accepted"], [2, 52, "rejected"]],
    current: [[0, 50], [1, 49], [2, 49]],
    best: [[0, 50], [1, 49], [2, 49]],
    oracle: [[1, 49.4]],
  },
  frames: [
    { id: "replay-P2-a2-s0-t0", x: 0, t: 0, peak_hat: 50, kind: "start", prob: null, T: 1, current: 50, best: 50, oracle_peak: null },
    { id: "replay-P2-a2-s0-t2", x: 2, t: 2, peak_hat: 52, kind: "rejected", prob: 0.135, T: 0.9, current: 49, best: 49, oracle_peak: 52.3 },
  ],
  best: { id: "replay-P2-a2-s0-best-t1", x: 1, t: 1, peak_hat: 49, oracle_peak: 49.4 },
};

test("frames are found by message id; the verification lines of the best design are not frames", () => {
  assert.equal(frameOf(meta, "replay-P2-a2-s0-t2").x, 2);
  assert.equal(frameOf(meta, meta.best.id), null);
});

test("a frame's text says what the proposer did, with temperature and the oracle peak", () => {
  const t = frameText(meta, meta.frames[1]);
  assert.match(t, /rejected \(p = 0\.14\)/);
  assert.match(t, /T = 0\.90 K/);
  assert.match(t, /oracle 52\.30 K/);
  assert.doesNotMatch(frameText(meta, meta.frames[0]), /oracle/);
});

test("chart data carries the whole run and the cursor", () => {
  const d = chartData(meta, 2);
  assert.equal(d.points.length, 3);
  assert.deepEqual(d.points[2], { x: 2, peak: 52, kind: "rejected" });
  assert.equal(d.cursor, 2);
  assert.equal(d.xMax, 9);
  assert.deepEqual(d.oracle, [[1, 49.4]]);
});
