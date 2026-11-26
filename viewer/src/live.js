// Live design panel (F7 live demo; CLAUDE.md §10.3, §13 T1.5-T1.7, B8).
//
// Pick a board record (from ?records=...), move a chip in mm steps and watch the server:
//   every move  -> {type: "query",  id, record}  prediction only: never verified, never gated (B8)
//   "Commit"    -> {type: "design", id, record}  the server (python -m server.app --verify) answers
//                  with pred, then iterate messages with the guaranteed band, then verified.
// The records sent are deep copies; geometry stays in mm on the record's 0.001 mm grid (§10.1).
// A move the server cannot rasterise (e.g. a chip pushed off the board) comes back as an error
// message, which the dev overlay shows; the record here is not changed by the server.
//
// Optimiser (src/optimise.js): pick P1 / P2 / P3 and a level, press run. Every candidate is a
// query preview (B8); the chart plots each predicted peak against the number of previews used;
// at the end the best design becomes the panel's record and, if "commit best" is ticked, is sent
// as one design (gate, coin, verification).

import { INFO, METHODS, search, settings } from "./optimise.js";
import { drawTrace } from "./trace_chart.js";

const REPLY_TIMEOUT_MS = 120000; // a cold rvcse server can take tens of seconds for its first reply

/** A deep copy of `record` with chip `chip` moved by (dx, dy) mm, rounded to the 0.001 mm grid. */
export function moveChip(record, chip, dx, dy) {
  const r = structuredClone(record);
  const c = r.chips[chip];
  if (!c) throw new RangeError(`record ${r.id} has no chip ${chip}`);
  const q = (v) => Math.round(v * 1000) / 1000;
  c.x0 = q(c.x0 + dx);
  c.y0 = q(c.y0 + dy);
  // via arrays under a chip follow it (design_board places one array exactly under each chip)
  for (const a of r.via_arrays ?? []) {
    if (a.x0 === record.chips[chip].x0 && a.y0 === record.chips[chip].y0 && a.w === c.w && a.h === c.h) {
      a.x0 = c.x0;
      a.y0 = c.y0;
    }
  }
  return r;
}

/** The client message of kind "query" | "design" for record, with a fresh id. */
export function clientMessage(kind, record, n) {
  if (kind !== "query" && kind !== "design") throw new TypeError(`kind must be query or design, got ${kind}`);
  return { schema_version: "1.0", type: kind, id: `live-${kind[0]}-${n}`, record };
}

/** Default run length per method (previews: P1 / P2 one per step, P3 2·chips + 1 per step). */
export const DEFAULT_STEPS = Object.freeze({ P1: 40, P2: 40, P3: 8 });

/** One-line description of the settings in use, from spec_constants.json (trajectories.yaml). */
export function settingsText(method, level, spec) {
  const c = settings(method, level, spec);
  if (method === "P1") return `λ = ${c.lam}/K, step ${c.step} (design units)`;
  if (method === "P2") return `T0 = ${c.T0} K → ${c.T0 * c.ratio} K, step ${c.step} (design units)`;
  return `η = ${c.eta} (design units), ${c.halvings} halvings, finite-difference h = panel step (mm)`;
}

/**
 * Wire the panel. el: a container; send(obj) -> bool (net.connect's send); getRecords() -> {id:
 * record}; remember(id, record) lets the viewer draw this record's vias for its replies;
 * getStatus() -> the message handler's status (trust, serverError), to read query replies;
 * spec: spec_constants.json.
 */
export function createLivePanel(el, { send, getRecords, remember, getStatus, spec }) {
  el.innerHTML = `
    <div><b>live design</b></div>
    <div><select id="live-rec"></select> <button id="live-load">load</button></div>
    <div>chip <select id="live-chip"></select> step <input id="live-step" type="number" value="1" min="0.1" step="0.1" style="width:4em"> mm</div>
    <div><button data-d="-1,0">←</button><button data-d="1,0">→</button><button data-d="0,1">↑</button><button data-d="0,-1">↓</button>
         <button id="live-commit">commit (verify)</button></div>
    <div id="live-msg" style="max-width:34ch;font-size:12px"></div>
    <div style="margin-top:6px"><b>optimiser</b>
      <select id="opt-method">${METHODS.map((m) => `<option>${m}</option>`).join("")}</select>
      level <select id="opt-level"><option>1</option><option selected>2</option><option>3</option></select>
      steps <input id="opt-steps" type="number" min="2" value="${DEFAULT_STEPS.P1}" style="width:3.5em">
      seed <input id="opt-seed" type="number" min="0" value="0" style="width:3em"></div>
    <div><button id="opt-run">run</button> <button id="opt-stop" disabled>stop</button>
      <label><input id="opt-commit" type="checkbox" checked> commit best</label></div>
    <div id="opt-info" style="max-width:34ch;font-size:11px;color:#ccc"></div>
    <canvas id="opt-chart" width="280" height="110"></canvas>
    <div id="opt-msg" style="max-width:34ch;font-size:12px"></div>`;
  const $ = (id) => el.querySelector(`#${id}`);
  let record = null;
  let counter = 0;

  function say(text) {
    $("live-msg").textContent = text;
  }
  function post(kind, rec = record) {
    if (!rec) {
      say("load a record first");
      return null;
    }
    counter += 1;
    const msg = clientMessage(kind, rec, counter);
    remember(msg.id, rec);
    if (!send(msg)) {
      say("not connected: start python -m server.app --verify");
      return null;
    }
    say(kind === "design" ? `committed ${msg.id}: verifying…` : `preview ${msg.id}`);
    return msg.id;
  }
  function refreshRecords() {
    // the designs to choose from: the loaded records, not the live messages sent from them nor
    // the frames of a proposer replay (src/replay.js)
    const sel = $("live-rec");
    const ids = Object.keys(getRecords()).filter((id) => !id.startsWith("live-") && !id.startsWith("replay-"));
    if (sel.options.length === ids.length) return ids.length;
    sel.innerHTML = "";
    for (const id of ids) sel.add(new Option(id.slice(0, 28), id));
    return ids.length;
  }
  $("live-load").addEventListener("click", () => {
    refreshRecords();
    const rec = getRecords()[$("live-rec").value];
    if (!rec) return say("no records: open the viewer with ?records=/demo/records.json");
    record = structuredClone(rec);
    const chipSel = $("live-chip");
    chipSel.innerHTML = "";
    record.chips.forEach((c, i) => chipSel.add(new Option(`${i} (${c.P} W)`, String(i))));
    post("query");
  });
  for (const b of el.querySelectorAll("button[data-d]")) {
    b.addEventListener("click", () => {
      if (!record) return say("load a record first");
      const [sx, sy] = b.dataset.d.split(",").map(Number);
      const step = Number($("live-step").value) || 1;
      record = moveChip(record, Number($("live-chip").value), sx * step, sy * step);
      post("query");
    });
  }
  $("live-commit").addEventListener("click", () => post("design"));

  // ------------------------------------------------------------------------------ optimiser
  const optSay = (text) => ($("opt-msg").textContent = text);
  let running = false;
  let stopRequested = false;
  function showInfo() {
    const m = $("opt-method").value;
    let text;
    try {
      text = settingsText(m, Number($("opt-level").value), spec);
    } catch (e) {
      text = e.message;
    }
    $("opt-info").textContent = `${INFO[m]} Settings: ${text}.`;
  }
  $("opt-method").addEventListener("change", () => {
    $("opt-steps").value = String(DEFAULT_STEPS[$("opt-method").value]);
    showInfo();
  });
  $("opt-level").addEventListener("change", showInfo);
  showInfo();

  /** Send a query for rec; resolve with the predicted peak (K) of its pred reply. */
  function evaluate(rec) {
    const id = post("query", rec);
    if (!id) return Promise.reject(new Error("not connected"));
    const t0 = performance.now();
    return new Promise((resolve, reject) => {
      const tick = () => {
        const st = getStatus();
        if (st.trust && st.trust.id === id && st.trust.state === "preview") return resolve(st.trust.peak);
        if (st.serverError && st.serverError.id === id) return reject(new Error(`server: ${st.serverError.message}`));
        if (performance.now() - t0 > REPLY_TIMEOUT_MS) return reject(new Error(`no reply to ${id} in ${REPLY_TIMEOUT_MS} ms`));
        setTimeout(tick, 15);
      };
      tick();
    });
  }

  async function run() {
    if (running) return;
    if (!record) return optSay("load a record first");
    const method = $("opt-method").value;
    const level = Number($("opt-level").value);
    const steps = Number($("opt-steps").value);
    const seed = Number($("opt-seed").value) || 0;
    const hMm = Number($("live-step").value) || 1;
    running = true;
    stopRequested = false;
    $("opt-run").disabled = true;
    $("opt-stop").disabled = false;
    const perStep = method === "P3" ? 2 * record.chips.length + 1 : 1;
    const chart = { points: [], current: [], best: [], xMax: steps * perStep, xLabel: "previews used" };
    let q = 0;
    let startPeak = null;
    let last = null;
    try {
      for await (const ev of search({ method, level, steps, record, spec, evaluate, seed, hMm, shouldStop: () => stopRequested })) {
        last = ev;
        if (ev.kind === "infeasible") {
          optSay(`step ${ev.k}: move rejected without a query (${ev.why})`);
          continue;
        }
        if (ev.kind === "stuck") {
          optSay(`step ${ev.k}: stuck (flat gradient, or every halving infeasible)`);
          continue;
        }
        q += 1;
        if (startPeak === null) startPeak = ev.peak;
        const kind = ev.kind === "propose" ? (ev.accepted ? "accepted" : "rejected") : ev.kind;
        chart.points.push({ x: q, peak: ev.peak, kind });
        chart.current.push([q, ev.current]);
        chart.best.push([q, ev.best]);
        chart.xMax = Math.max(chart.xMax, q);
        drawTrace($("opt-chart"), chart);
        const T = ev.T === undefined ? "" : ` · T = ${ev.T.toPrecision(2)} K`;
        const p = ev.prob === undefined ? "" : ` (p = ${ev.prob.toPrecision(2)})`;
        const what = {
          start: "start",
          propose: `${ev.accepted ? "kept" : "rejected"}${p}`,
          probe: `probe of coordinate ${ev.coord}`,
          step: `gradient step (${ev.halvings} halvings)`,
        }[ev.kind];
        optSay(`${method} step ${ev.k}: ${ev.peak.toFixed(2)} K, ${what}${T} · current ${ev.current.toFixed(2)} · best ${ev.best.toFixed(2)} K (predicted)`);
      }
      if (!last) return;
      record = structuredClone(last.bestRecord);
      const summary = `${method} level ${level}: ${q} previews, start ${startPeak.toFixed(2)} K → best ${last.best.toFixed(2)} K (surrogate predictions)`;
      if ($("opt-commit").checked) {
        post("design");
        optSay(`${summary}; best design committed: gate, then verification`);
      } else {
        post("query");
        optSay(`${summary}; best design loaded (not committed)`);
      }
    } catch (e) {
      optSay(`optimiser stopped: ${e.message}`);
    } finally {
      running = false;
      $("opt-run").disabled = false;
      $("opt-stop").disabled = true;
    }
  }
  $("opt-run").addEventListener("click", run);
  $("opt-stop").addEventListener("click", () => (stopRequested = true));

  say("waiting for designs (?records=...)");
  return { refreshRecords, say };
}
