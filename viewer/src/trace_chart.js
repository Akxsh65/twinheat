// Peak-vs-step chart for the optimiser demos (live: src/optimise.js; replay: src/replay.js).
// Dots: every design a proposer tried (grey rejected, amber accepted / taken, blue P3 probes or
// other starts); lines: the chain's current design (white) and the best so far (green); a vertical
// cursor marks the frame on screen. Optional red dots: oracle (true) peaks of the replayed run.
// All values are K; the y axis is fitted to the data.

export const COLORS = Object.freeze({
  rejected: "#8a8a8a", accepted: "#ffb300", probe: "#4f8fdc", start: "#ffffff", step: "#ffb300",
  current: "#ffffff", best: "#43c463", oracle: "#ff5252", cursor: "#ff80ab",
});

/** Pure: the y range [lo, hi] (K) covering every value, padded by 5 % (1 K if flat). */
export function yRange(values) {
  const v = values.filter(Number.isFinite);
  if (!v.length) return [0, 1];
  let lo = Math.min(...v);
  let hi = Math.max(...v);
  if (hi - lo < 1e-9) {
    lo -= 0.5;
    hi += 0.5;
  }
  const pad = 0.05 * (hi - lo);
  return [lo - pad, hi + pad];
}

/**
 * Draw on a 2D canvas. data: {points: [{x, peak, kind}], current: [[x, K]], best: [[x, K]],
 * oracle?: [[x, K]], xMax, cursor?: x, xLabel}.
 */
export function drawTrace(canvas, data) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const padL = 34;
  const padB = 14;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fillRect(0, 0, W, H);
  const all = [...data.points.map((p) => p.peak), ...data.current.map((p) => p[1]),
    ...data.best.map((p) => p[1]), ...(data.oracle ?? []).map((p) => p[1])];
  const [lo, hi] = yRange(all);
  const xMax = Math.max(1, data.xMax ?? 1);
  const X = (x) => padL + ((W - padL - 4) * x) / xMax;
  const Y = (y) => 4 + ((H - padB - 8) * (hi - y)) / (hi - lo);
  ctx.fillStyle = "#ccc";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`${hi.toFixed(1)}`, 2, 12);
  ctx.fillText(`${lo.toFixed(1)}`, 2, H - padB);
  ctx.fillText(`K   ${data.xLabel ?? "step"} →`, padL, H - 2);
  for (const p of data.points) {
    ctx.fillStyle = COLORS[p.kind] ?? COLORS.rejected;
    ctx.fillRect(X(p.x) - 1, Y(p.peak) - 1, p.kind === "probe" ? 2 : 3, p.kind === "probe" ? 2 : 3);
  }
  for (const p of data.oracle ?? []) {
    ctx.fillStyle = COLORS.oracle;
    ctx.fillRect(X(p[0]) - 1.5, Y(p[1]) - 1.5, 3, 3);
  }
  const line = (pts, color) => {
    if (pts.length < 1) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.stroke();
  };
  line(data.current, COLORS.current);
  line(data.best, COLORS.best);
  if (Number.isFinite(data.cursor)) {
    ctx.strokeStyle = COLORS.cursor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(data.cursor), 2);
    ctx.lineTo(X(data.cursor), H - padB);
    ctx.stroke();
  }
}
