// Live design panel (F7 live demo; CLAUDE.md §10.3, §13 T1.5-T1.7, B8).
//
// Pick a board record (from ?records=...), move a chip in mm steps and watch the server:
//   every move  -> {type: "query",  id, record}  prediction only: never verified, never gated (B8)
//   "Commit"    -> {type: "design", id, record}  the server (python -m server.app --verify) answers
//                  with pred, then iterate messages with the guaranteed band, then verified.
// The records sent are deep copies; geometry stays in mm on the record's 0.001 mm grid (§10.1).
// A move the server cannot rasterise (e.g. a chip pushed off the board) comes back as an error
// message, which the dev overlay shows; the record here is not changed by the server.

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

/**
 * Wire the panel. el: a container; send(obj) -> bool (net.connect's send); getRecords() -> {id:
 * record}; remember(id, record) lets the viewer draw this record's vias for its replies.
 */
export function createLivePanel(el, { send, getRecords, remember }) {
  el.innerHTML = `
    <div><b>live design</b></div>
    <div><select id="live-rec"></select> <button id="live-load">load</button></div>
    <div>chip <select id="live-chip"></select> step <input id="live-step" type="number" value="1" min="0.1" step="0.1" style="width:4em"> mm</div>
    <div><button data-d="-1,0">←</button><button data-d="1,0">→</button><button data-d="0,1">↑</button><button data-d="0,-1">↓</button>
         <button id="live-commit">commit (verify)</button></div>
    <div id="live-msg" style="max-width:24ch;font-size:12px"></div>`;
  const $ = (id) => el.querySelector(`#${id}`);
  let record = null;
  let counter = 0;

  function say(text) {
    $("live-msg").textContent = text;
  }
  function post(kind) {
    if (!record) return say("load a record first");
    counter += 1;
    const msg = clientMessage(kind, record, counter);
    remember(msg.id, record);
    if (!send(msg)) return say("not connected: start python -m server.app --verify");
    say(kind === "design" ? `committed ${msg.id}: verifying…` : `preview ${msg.id}`);
  }
  function refreshRecords() {
    const sel = $("live-rec");
    const ids = Object.keys(getRecords());
    if (sel.options.length === ids.length) return;
    sel.innerHTML = "";
    for (const id of ids) sel.add(new Option(id.slice(0, 28), id));
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
  return { refreshRecords };
}
