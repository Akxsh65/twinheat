// Trust display: the Predicted -> Verifying -> Verified badge and the guaranteed band
// (CLAUDE.md §13 T1.6, T1.7; F7).
//
// The state is driven ONLY by the type of the last valid server message (nextTrust), never by
// anything the viewer computes, so the badge cannot claim more than the server sent:
//   pred      -> "predicted" (amber; "preview" while dragging: a query, never verified, B8)
//   iterate   -> "verifying" (blue) with k and the band cert_k
//   verified  -> "verified"  (green) with the final band, iterations and time
// The band is the server's cert_k / cert in K: it bounds |true field − drawn field| in EVERY cell,
// the float16 rounding of the drawn field included (server/iterate_stream.py). A null band means
// the board has no certificate, and the panel says so instead of drawing a band.

/** Scalar fields of a valid message that the trust display needs (null for other kinds). */
export function messageInfo(kind, msg) {
  if (kind === "pred") return { decision: msg.decision, peak: msg.peak };
  if (kind === "iterate") return { k: msg.k, band: msg.cert_k };
  if (kind === "verified") return { band: msg.cert, iters: msg.iters, ms: msg.ms, stop: msg.stop_reason };
  return null;
}

/** The trust state after a valid message of `kind`; other kinds leave it unchanged. */
export function nextTrust(prev, kind, id, info) {
  if (kind === "pred") return { state: info.decision === "preview" ? "preview" : "predicted", id, peak: info.peak };
  if (kind === "iterate") return { state: "verifying", id, k: info.k, band: info.band };
  if (kind === "verified") {
    return { state: "verified", id, band: info.band, iters: info.iters, ms: info.ms, stop: info.stop };
  }
  return prev;
}

const fmt = (band) => (band < 0.01 ? band.toExponential(1) : band.toPrecision(2));

/** {cls, text} for a trust state (null: nothing received yet). */
export function describe(t) {
  if (!t) return { cls: "trust-none", text: "no field yet" };
  if (t.state === "preview") return { cls: "trust-amber", text: "PREDICTED (preview while dragging) · not verified" };
  if (t.state === "predicted") return { cls: "trust-amber", text: "PREDICTED · not verified yet" };
  if (t.state === "verifying") {
    const band = t.band === null ? "no certificate: no guaranteed band" : `guaranteed band ±${fmt(t.band)} K`;
    return { cls: "trust-blue", text: `VERIFYING · PCG iteration ${t.k} · ${band}` };
  }
  if (t.band === null) return { cls: "trust-blue", text: `SOLVED · ${t.stop} · no guaranteed band` };
  return {
    cls: "trust-green",
    text: `VERIFIED · guaranteed band ±${fmt(t.band)} K in every cell · ${t.iters} iterations · ${t.ms.toFixed(0)} ms`,
  };
}

/** A panel that rewrites its DOM only when the trust state changes. */
export function createTrustPanel(el) {
  let last = undefined;
  return {
    update(t) {
      if (t === last) return;
      last = t;
      const { cls, text } = describe(t);
      el.className = cls;
      el.textContent = text;
    },
  };
}
