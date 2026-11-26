// Proposer replay panel (F7 demo B; CLAUDE.md §13 T1.8): plays a real offline P1 / P2 / P3 run,
// written by server/proposer_replay.py, through the same message handler as the live stream (B6:
// every line is validated again here), so the board, the trust badge and the guaranteed band
// behave exactly as live. Needs no server: ?replay=<prefix> (e.g. /demo/replay_) loads
// <prefix>P1.json + <prefix>P1.jsonl, and likewise P2 and P3, from the viewer's public folder.
//
// The chart shows the whole run (every proposal's predicted peak; grey rejected, amber kept, blue
// other P3 starts), the chain's current design (white), the best so far (green), the oracle peak
// of each frame (red, when labels were available) and a cursor at the frame on screen. The run ends
// with the best design's certified verification.

import { drawTrace } from "./trace_chart.js";

/** Pure: the frame of meta whose id is `id`, or null. */
export function frameOf(meta, id) {
  return meta.frames.find((f) => f.id === id) ?? null;
}

/** Pure: one line describing a frame (K values are surrogate predictions unless marked oracle). */
export function frameText(meta, f) {
  const what = {
    start: "start",
    accepted: `kept (p = ${f.prob?.toPrecision(2)})`,
    rejected: `rejected (p = ${f.prob?.toPrecision(2)})`,
    step: "gradient step",
    probe: "another start",
  }[f.kind] ?? f.kind;
  const T = f.T === null || f.T === undefined ? "" : ` · T = ${f.T.toPrecision(2)} K`;
  const o = f.oracle_peak === null || f.oracle_peak === undefined ? "" : ` · oracle ${f.oracle_peak.toFixed(2)} K`;
  return `${meta.proposer} ${meta.x_label} ${f.x}: predicted ${f.peak_hat.toFixed(2)} K${o} · ${what}${T} · current ${f.current?.toFixed(2)} · best ${f.best.toFixed(2)} K`;
}

/** Pure: chart data for meta with the cursor at x. */
export function chartData(meta, cursor) {
  return {
    points: meta.trace.points.map(([x, peak, kind]) => ({ x, peak, kind })),
    current: meta.trace.current,
    best: meta.trace.best,
    oracle: meta.trace.oracle,
    xMax: meta.x_max,
    cursor,
    xLabel: meta.x_label,
  };
}

/**
 * el: a container; handle: net.createHandler's handle (handle(raw) validates and shows a line);
 * remember(id, record) lets the viewer draw chips and vias for the frame on screen; prefix: URL
 * prefix of the replay files.
 */
export function createReplayPanel(el, { handle, remember, prefix }) {
  el.innerHTML = `
    <div><b>proposer replay</b> (real offline runs)
      <select id="rp-p"><option>P1</option><option selected>P2</option><option>P3</option></select>
      <button id="rp-load">load</button> <button id="rp-play" disabled>play</button>
      fps <input id="rp-fps" type="number" value="3" min="0.5" max="20" step="0.5" style="width:3.5em"></div>
    <div><input id="rp-pos" type="range" min="0" max="0" value="0" style="width:280px" disabled></div>
    <div id="rp-info" style="max-width:40ch;font-size:11px;color:#ccc"></div>
    <canvas id="rp-chart" width="300" height="110"></canvas>
    <div id="rp-msg" style="max-width:40ch;font-size:12px"></div>`;
  const $ = (id) => el.querySelector(`#${id}`);
  let meta = null;
  let lines = [];
  let pos = 0;
  let timer = null;
  const say = (t) => ($("rp-msg").textContent = t);

  function show(i) {
    pos = i;
    $("rp-pos").value = String(i);
    const r = handle(lines[i]);
    if (!r.ok) return say(`line ${i + 1} rejected: ${r.error}`);
    const f = frameOf(meta, r.id);
    const verifying = r.id === meta.best.id && !f;
    drawTrace($("rp-chart"), chartData(meta, f ? f.x : meta.best.x));
    if (f) say(frameText(meta, f));
    else if (verifying || r.kind !== "pred") {
      const o = meta.best.oracle_peak === null ? "" : ` (oracle ${meta.best.oracle_peak.toFixed(2)} K)`;
      say(`best design of the run (${meta.x_label} ${meta.best.x}, predicted ${meta.best.peak_hat.toFixed(2)} K${o}): ${r.kind === "pred" ? "committed for verification" : r.kind}`);
    }
  }
  function stop() {
    clearInterval(timer);
    timer = null;
    $("rp-play").textContent = "play";
  }
  $("rp-play").addEventListener("click", () => {
    if (timer) return stop();
    if (pos >= lines.length - 1) pos = -1;
    $("rp-play").textContent = "pause";
    timer = setInterval(() => {
      if (pos >= lines.length - 1) return stop();
      show(pos + 1);
    }, 1000 / Math.max(0.5, Number($("rp-fps").value) || 3));
  });
  $("rp-pos").addEventListener("input", () => {
    stop();
    show(Number($("rp-pos").value));
  });
  $("rp-load").addEventListener("click", async () => {
    stop();
    const p = $("rp-p").value;
    say(`loading ${prefix}${p}.json / .jsonl …`);
    try {
      const get = async (url, kind) => {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
        return kind === "json" ? r.json() : r.text();
      };
      const [m, text] = await Promise.all([get(`${prefix}${p}.json`, "json"), get(`${prefix}${p}.jsonl`, "text")]);
      if (m.schema !== "twinheat-replay/1") throw new Error(`${prefix}${p}.json is not a replay (schema ${m.schema})`);
      meta = m;
      lines = text.split(String.fromCharCode(10)).filter((l) => l.trim());
      for (const [id, rec] of Object.entries(meta.records)) remember(id, rec);
      $("rp-info").textContent = `${meta.info} Run ${meta.source.file} (commit ${meta.source.commit}, ${meta.source.rows} proposals, level ${meta.level}, seed ${meta.seed}).`;
      $("rp-pos").max = String(lines.length - 1);
      $("rp-pos").disabled = false;
      $("rp-play").disabled = false;
      show(0);
    } catch (e) {
      say(`could not load the replay: ${e.message} (run python -m server.proposer_replay, see its docstring)`);
    }
  });
  say("pick P1, P2 or P3 and press load");
  return { stop };
}
