// Live optimiser demo (F7; CLAUDE.md §11.2, §10.3, B8): the rules of P1, P2 and P3, run in the
// browser over the CHIP POSITIONS of the loaded board, with the server's query previews as the
// surrogate. Every candidate is sent as a `query` (prediction only: never gated, never verified,
// no coin; B8); only the best design is committed at the end, as one `design`.
//
// What is the same as the offline proposers (gate/proposers.py) and what is not:
//   same   the acceptance rules, and the settings per level (lambda, step, T0, cooling ratio, eta,
//          halvings), read from experiments/configs/trajectories.yaml through spec_constants.json;
//          moves in design units u in [0, 1] per chip coordinate, x0 = margin + u·(L − 2·margin − w)
//          as data/sampler.design_board maps them, reflected into [0, 1]; the §6.5 edge margin
//          and chip gap decide feasibility (an infeasible move is rejected, never sent).
//   not    only the chips move (the offline runs also move 6 copper rectangles and 4 via
//          densities); one chain / one start (P3 runs 64); short demo runs (P1 2,000 proposals and
//          P2 1,000 steps offline); P3's gradient is a forward finite difference of previews,
//          spacing h_mm, not autograd through the FNO; the random numbers come from a seeded
//          JavaScript generator, so runs do not reproduce the offline trajectories.
// All peaks are surrogate predictions (K): the demo shows HOW each method searches, not results.

export const METHODS = Object.freeze(["P1", "P2", "P3"]);

export const INFO = Object.freeze({
  P1: "P1 tilted MCMC: move every chip at random; a cooler design is always kept, a hotter one with " +
    "probability exp(−λ·ΔT). It samples good designs and keeps wandering; it does not settle.",
  P2: "P2 simulated annealing: the same random moves, but the tolerance for hotter designs " +
    "(temperature T) cools geometrically: it explores early and settles late.",
  P3: "P3 gradient descent: measure how the predicted peak changes with each chip coordinate, step " +
    "downhill (normalised step η, halved if infeasible). Smooth, nearly monotone, can stall in a " +
    "local minimum.",
});

/** Settings of `method` at level 1..3 from spec_constants.json's proposers block. */
export function settings(method, level, spec) {
  const k = [1, 2, 3].indexOf(level);
  if (k < 0) throw new RangeError(`level must be 1, 2 or 3, got ${level}`);
  const p = spec.proposers?.[method];
  if (!p) throw new RangeError(`no settings for ${method} in spec_constants.json`);
  if (method === "P1") return { lam: p.lambda_per_K[k], step: p.step };
  if (method === "P2") return { T0: p.T0_K[k], step: p.step[k], ratio: p.T_end_ratio };
  return { eta: p.eta[k], halvings: p.halvings };
}

const L_MM = (spec) => spec.L_m * 1000;

/** Travel (mm) of chip c along x and y: L − 2·margin − size (design_board's mapping). */
export function span(c, spec) {
  const m = spec.chip_edge_margin_min_mm;
  const sx = L_MM(spec) - 2 * m - c.w;
  const sy = L_MM(spec) - 2 * m - c.h;
  if (!(sx > 0 && sy > 0)) throw new RangeError(`chip ${c.w} x ${c.h} mm does not fit inside the edge margin`);
  return [sx, sy];
}

/** Design units [u0, v0, u1, v1, ...] of the record's chips, clamped into [0, 1]. */
export function toUnit(record, spec) {
  const m = spec.chip_edge_margin_min_mm;
  const u = [];
  for (const c of record.chips) {
    const [sx, sy] = span(c, spec);
    u.push(Math.min(1, Math.max(0, (c.x0 - m) / sx)), Math.min(1, Math.max(0, (c.y0 - m) / sy)));
  }
  return u;
}

/** A deep copy of record with its chips at design units u (floored to the 0.001 mm grid, as
 * design_board does); a via array exactly under a chip follows it. */
export function fromUnit(record, u, spec) {
  if (u.length !== 2 * record.chips.length) throw new RangeError(`u has ${u.length} values for ${record.chips.length} chips`);
  const r = structuredClone(record);
  const m = spec.chip_edge_margin_min_mm;
  const q = (v) => Math.floor(v * 1000 + 1e-9) / 1000;
  r.chips.forEach((c, i) => {
    const old = record.chips[i];
    const [sx, sy] = span(c, spec);
    c.x0 = q(m + u[2 * i] * sx);
    c.y0 = q(m + u[2 * i + 1] * sy);
    for (const a of r.via_arrays ?? []) {
      if (a.x0 === old.x0 && a.y0 === old.y0 && a.w === old.w && a.h === old.h) {
        a.x0 = c.x0;
        a.y0 = c.y0;
      }
    }
  });
  return r;
}

/** Clearance (mm) between two chips: > 0 apart (data/sampler._chip_gap). */
function gap(a, b) {
  const dx = Math.max(b.x0 - (a.x0 + a.w), a.x0 - (b.x0 + b.w));
  const dy = Math.max(b.y0 - (a.y0 + a.h), a.y0 - (b.y0 + b.h));
  return Math.max(dx, dy);
}

/** null if the chips meet the §6.5 edge margin and gap, else what is wrong (sampler._chip_violation). */
export function chipViolation(chips, spec) {
  const m = spec.chip_edge_margin_min_mm;
  const g = spec.chip_gap_min_mm;
  const L = L_MM(spec);
  for (let i = 0; i < chips.length; i++) {
    const c = chips[i];
    if (c.x0 < m || c.y0 < m || c.x0 + c.w > L - m + 1e-9 || c.y0 + c.h > L - m + 1e-9) {
      return `chip ${i} is closer than ${m} mm to an edge`;
    }
  }
  for (let i = 0; i < chips.length; i++) {
    for (let j = i + 1; j < chips.length; j++) {
      const d = gap(chips[i], chips[j]);
      if (d < g - 1e-9) return `chips ${i} and ${j} are ${d.toFixed(3)} mm apart (< ${g} mm)`;
    }
  }
  return null;
}

/** Fold every value back into [0, 1] by mirror reflection (gate/proposers.reflect). */
export function reflect(u) {
  return u.map((v) => {
    const y = ((v % 2) + 2) % 2;
    return y > 1 ? 2 - y : y;
  });
}

/** Seeded generator: random() in [0, 1) (mulberry32) and normal() (Box–Muller). */
export function makeRng(seed) {
  let a = seed >>> 0;
  const random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    let u = 0;
    while (u === 0) u = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
  };
  return { random, normal };
}

/**
 * Run one search; an async generator of events (the panel draws them):
 *   {kind: "start",   k: 0, record, peak}
 *   {kind: "propose", k, record, peak, accepted, prob, T?}        P1 / P2 candidate (a query)
 *   {kind: "infeasible", k, why}                                   rejected without a query
 *   {kind: "probe",   k, coord, record, peak}                      P3 finite-difference probe
 *   {kind: "step",    k, record, peak, gradNorm, halvings}         P3 accepted step
 *   {kind: "stuck",   k}                                           P3: every halving infeasible
 * Every event also has current, best (peaks, K) and bestRecord. evaluate(record) -> Promise<peak K>.
 * steps: P1 feasible proposals, P2 iterations (start included), P3 gradient steps.
 */
export async function* search({ method, level, steps, record, spec, evaluate, seed = 0, hMm = 1, shouldStop = () => false }) {
  if (!METHODS.includes(method)) throw new RangeError(`method must be one of ${METHODS.join(", ")}, got ${method}`);
  if (!(Number.isInteger(steps) && steps >= 2)) throw new RangeError(`steps must be an integer >= 2, got ${steps}`);
  const cfg = settings(method, level, spec);
  const rng = makeRng(seed);
  const violation = (u) => chipViolation(fromUnit(record, u, spec).chips, spec);
  let cur = toUnit(record, spec);
  const startWhy = violation(cur);
  if (startWhy) throw new RangeError(`the loaded board is infeasible for the search: ${startWhy}`);
  let curRec = fromUnit(record, cur, spec);
  let curPeak = await evaluate(curRec);
  let best = curPeak;
  let bestRecord = curRec;
  const track = (peak, rec) => {
    if (peak < best) {
      best = peak;
      bestRecord = rec;
    }
  };
  yield { kind: "start", k: 0, record: curRec, peak: curPeak, current: curPeak, best, bestRecord };

  if (method === "P1" || method === "P2") {
    const accept = (delta, k) => {
      if (delta <= 0) return 1;
      if (method === "P1") return Math.exp(-cfg.lam * delta);
      return Math.exp(-delta / (cfg.T0 * cfg.ratio ** (k / (steps - 1))));
    };
    let k = 0; // P1: feasible proposals; P2: iterations (the start is iteration 0)
    let tries = 0;
    const maxTries = 100 * steps; // P1 only: a guard against a board where no move is feasible
    while (!shouldStop()) {
      if (method === "P2" && k >= steps - 1) break;
      if (method === "P1" && k >= steps) break;
      if (method === "P1" && ++tries > maxTries) throw new Error(`no feasible move in ${maxTries} tries`);
      const cand = reflect(cur.map((v) => v + cfg.step * rng.normal()));
      if (method === "P2") k += 1;
      const why = violation(cand);
      if (why) {
        yield { kind: "infeasible", k, why, current: curPeak, best, bestRecord };
        continue;
      }
      if (method === "P1") k += 1;
      const rec = fromUnit(record, cand, spec);
      const peak = await evaluate(rec);
      track(peak, rec);
      const prob = accept(peak - curPeak, k);
      const accepted = rng.random() < prob;
      if (accepted) {
        cur = cand;
        curRec = rec;
        curPeak = peak;
      }
      const T = method === "P2" ? cfg.T0 * cfg.ratio ** (k / (steps - 1)) : undefined;
      yield { kind: "propose", k, record: rec, peak, accepted, prob, T, current: curPeak, best, bestRecord };
    }
    return;
  }

  // P3: forward differences in design units, spacing h_mm / travel per coordinate
  const spans = record.chips.flatMap((c) => span(c, spec));
  for (let k = 1; k <= steps && !shouldStop(); k++) {
    const g = new Array(cur.length).fill(0);
    for (let i = 0; i < cur.length && !shouldStop(); i++) {
      const h = hMm / spans[i];
      for (const sgn of [1, -1]) {
        const probe = cur.slice();
        probe[i] += sgn * h;
        if (probe[i] < 0 || probe[i] > 1 || violation(probe)) continue;
        const rec = fromUnit(record, probe, spec);
        const peak = await evaluate(rec);
        track(peak, rec);
        g[i] = (sgn * (peak - curPeak)) / h;
        yield { kind: "probe", k, coord: i, record: rec, peak, current: curPeak, best, bestRecord };
        break;
      }
    }
    if (shouldStop()) return;
    const gmax = Math.max(...g.map(Math.abs));
    if (gmax === 0) {
      yield { kind: "stuck", k, current: curPeak, best, bestRecord };
      continue;
    }
    let moved = false;
    for (let hv = 0; hv <= cfg.halvings; hv++) {
      const y = reflect(cur.map((v, i) => v - cfg.eta * 0.5 ** hv * (g[i] / gmax)));
      if (violation(y)) continue;
      cur = y;
      curRec = fromUnit(record, y, spec);
      curPeak = await evaluate(curRec);
      track(curPeak, curRec);
      yield { kind: "step", k, record: curRec, peak: curPeak, gradNorm: gmax, halvings: hv, current: curPeak, best, bestRecord };
      moved = true;
      break;
    }
    if (!moved) yield { kind: "stuck", k, current: curPeak, best, bestRecord };
  }
}
