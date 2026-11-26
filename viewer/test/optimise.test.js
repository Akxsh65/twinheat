// Live optimiser (src/optimise.js): settings come from trajectories.yaml (via spec_constants.json),
// the chip mapping is design_board's, every design sent is feasible (§6.5) and contract-valid,
// and P1 / P2 / P3 follow their acceptance rules.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { validate } from "../src/contracts.js";
import { DEFAULT_STEPS, settingsText } from "../src/live.js";
import { chipViolation, fromUnit, makeRng, reflect, search, settings, span, toUnit } from "../src/optimise.js";
import { yRange } from "../src/trace_chart.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SPEC = JSON.parse(readFileSync(path.join(ROOT, "viewer/src/spec_constants.json"), "utf-8"));
const base = JSON.parse(readFileSync(path.join(ROOT, "contracts/examples/board_record/valid_1.json"), "utf-8"));

// two chips, each with its via array, well apart
const rec = structuredClone(base);
rec.chips = [
  { x0: 20, y0: 20, w: 10, h: 8, P: 1.2, side: "top" },
  { x0: 60, y0: 60, w: 8, h: 8, P: 0.8, side: "top" },
];
rec.via_arrays = rec.chips.map((c) => ({ x0: c.x0, y0: c.y0, w: c.w, h: c.h, density_per_mm2: 0.5 }));

// a smooth stand-in for the surrogate: peak falls as the chips move apart and away from (30, 70)
function fakePeak(r) {
  const [a, b] = r.chips.map((c) => [c.x0 + c.w / 2, c.y0 + c.h / 2]);
  const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
  return 60 - 0.2 * d + 0.002 * ((a[0] - 30) ** 2 + (a[1] - 70) ** 2);
}

async function collect(method, opts = {}) {
  const sent = [];
  const evaluate = async (r) => {
    sent.push(r);
    return fakePeak(r);
  };
  const events = [];
  for await (const ev of search({ method, level: 2, steps: opts.steps ?? 12, record: rec, spec: SPEC, evaluate, seed: 3, ...opts })) {
    events.push(ev);
  }
  return { sent, events };
}

test("settings per level are trajectories.yaml's, through spec_constants.json", () => {
  const yaml = readFileSync(path.join(ROOT, "experiments/configs/trajectories.yaml"), "utf-8");
  assert.match(yaml, /lambda_per_K: \[1\.0, 3\.0, 10\.0\]/);
  assert.match(yaml, /eta: \[0\.01, 0\.03, 0\.1\]/);
  assert.deepEqual(settings("P1", 3, SPEC), { lam: 10, step: 0.05 });
  assert.deepEqual(settings("P2", 1, SPEC), { T0: 3, step: 0.02, ratio: 0.01 });
  assert.deepEqual(settings("P3", 2, SPEC), { eta: 0.03, halvings: 5 });
  assert.throws(() => settings("P1", 4, SPEC), /level/);
  for (const m of ["P1", "P2", "P3"]) assert.ok(settingsText(m, 2, SPEC).length > 0 && DEFAULT_STEPS[m] >= 2);
});

test("design units round-trip on the 0.001 mm grid, and via arrays follow their chips", () => {
  const u = toUnit(rec, SPEC);
  assert.equal(u.length, 4);
  const back = fromUnit(rec, u, SPEC);
  back.chips.forEach((c, i) => {
    assert.ok(Math.abs(c.x0 - rec.chips[i].x0) <= 0.001 + 1e-9);
    assert.ok(Math.abs(c.y0 - rec.chips[i].y0) <= 0.001 + 1e-9);
  });
  const moved = fromUnit(rec, [0, 1, 1, 0], SPEC);
  const m = SPEC.chip_edge_margin_min_mm;
  assert.equal(moved.chips[0].x0, m);
  assert.ok(Math.abs(moved.chips[0].y0 + moved.chips[0].h - (100 - m)) < 0.002);
  moved.chips.forEach((c, i) => {
    assert.equal(moved.via_arrays[i].x0, c.x0);
    assert.equal(moved.via_arrays[i].y0, c.y0);
  });
  assert.equal(rec.chips[0].x0, 20, "the original is not changed");
  validate(moved, "board_record");
  assert.deepEqual(span(rec.chips[0], SPEC), [100 - 2 * m - 10, 100 - 2 * m - 8]);
});

test("chipViolation applies the §6.5 edge margin and chip gap", () => {
  assert.equal(chipViolation(rec.chips, SPEC), null);
  const edge = structuredClone(rec.chips);
  edge[0].x0 = 2.9;
  assert.match(chipViolation(edge, SPEC), /edge/);
  const close = structuredClone(rec.chips);
  close[1].x0 = close[0].x0 + close[0].w + 1.5;
  close[1].y0 = close[0].y0;
  assert.match(chipViolation(close, SPEC), /apart/);
  close[1].x0 = close[0].x0 + close[0].w + 2;
  assert.equal(chipViolation(close, SPEC), null, "exactly the minimum gap is allowed");
});

test("reflect folds into [0, 1] and the generator is seeded", () => {
  assert.deepEqual(reflect([-0.2, 1.3, 0.5, 2.4]).map((v) => +v.toFixed(12)), [0.2, 0.7, 0.5, 0.4]);
  const a = makeRng(5);
  const b = makeRng(5);
  for (let i = 0; i < 10; i++) assert.equal(a.normal(), b.normal());
});

for (const method of ["P1", "P2", "P3"]) {
  test(`${method}: every design sent is feasible and contract-valid; best is the minimum seen`, async () => {
    const { sent, events } = await collect(method);
    assert.ok(sent.length > 3);
    for (const r of sent) {
      assert.equal(chipViolation(r.chips, SPEC), null);
      validate(r, "board_record");
    }
    const minSeen = Math.min(...sent.map(fakePeak));
    const last = events.at(-1);
    assert.equal(last.best, minSeen);
    assert.equal(fakePeak(last.bestRecord), minSeen);
    const queried = events.filter((e) => !["infeasible", "stuck"].includes(e.kind)).length;
    assert.equal(queried, sent.length, "one event per query");
  });
}

test("P1 / P2 keep every downhill move and the current peak only changes on a kept move", async () => {
  for (const method of ["P1", "P2"]) {
    const { events } = await collect(method, { steps: 30 });
    let current = events[0].peak;
    for (const e of events.filter((x) => x.kind === "propose")) {
      if (e.peak <= current) assert.ok(e.accepted && e.prob === 1, `${method} step ${e.k}`);
      assert.equal(e.current, e.accepted ? e.peak : current);
      current = e.current;
    }
  }
});

test("P2 cools geometrically from T0 to T0 · ratio over its iterations", async () => {
  const { events } = await collect("P2", { steps: 12 });
  const props = events.filter((e) => e.kind === "propose" || e.kind === "infeasible");
  assert.equal(props.at(-1).k, 11, "steps counts iterations, the start included");
  const c = settings("P2", 2, SPEC);
  for (const e of events.filter((x) => x.kind === "propose")) {
    assert.ok(Math.abs(e.T - c.T0 * c.ratio ** (e.k / 11)) < 1e-12);
  }
});

test("P3 probes every coordinate, then steps; on a smooth peak it goes downhill", async () => {
  const { events } = await collect("P3", { steps: 4, hMm: 1 });
  const steps = events.filter((e) => e.kind === "step");
  assert.ok(steps.length >= 1);
  for (const s of steps) {
    assert.equal(events.filter((e) => e.kind === "probe" && e.k === s.k).length, 4, "2 coordinates per chip");
  }
  assert.ok(steps.at(-1).peak < events[0].peak);
});

test("shouldStop ends a run, and an infeasible start is refused", async () => {
  let n = 0;
  const { events } = await collect("P1", { steps: 50, shouldStop: () => ++n > 5 });
  assert.ok(events.length < 10);
  const bad = structuredClone(rec);
  bad.chips[1].x0 = bad.chips[0].x0 + 1;
  bad.chips[1].y0 = bad.chips[0].y0;
  await assert.rejects(async () => {
    for await (const _ of search({ method: "P2", level: 1, steps: 5, record: bad, spec: SPEC, evaluate: async () => 1 }));
  }, /infeasible/);
});

test("the chart's y range covers the data", () => {
  assert.deepEqual(yRange([]), [0, 1]);
  const [lo, hi] = yRange([10, 20, NaN]);
  assert.ok(lo < 10 && hi > 20);
  const [a, b] = yRange([5, 5]);
  assert.ok(a < 5 && b > 5);
});
