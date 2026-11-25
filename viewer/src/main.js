// Viewer entry point: connect to the server, validate every message, show the 4 layer fields.
//
// URL parameters: ?ws=<url> (default ws://<page host>:8765, the mock server's default port)
//                 ?seated      use the "local" reference space instead of "local-floor"
//                 ?inline      parse / validate / decode on the main thread (no worker; debugging)
//                 ?debug       expose window.twinheat = {view, textures, handle} (console, screenshots)
//                 ?bench=<s>   E10: record every frame interval for <s> seconds, then download
//                              e10_frames.csv (frame_index, ms, look, oit, peel, trust state)
//                 ?hdri=<url>  an equirectangular .hdr for the realistic look's lighting (default:
//                              three's procedural RoomEnvironment)
//                 ?records=<url>  JSON {message id: board record} for the via columns of the mock
//                              stream (server/solver_stream.py --records-out); in production the
//                              viewer holds the record it sent in query / design (§10.3)

import { createDevOverlay } from "./devOverlay.js";
import { DEFAULT_LUT, LUT_NAMES } from "./lut.js";
import { connect, createHandler, LayerTextures } from "./net.js";
import { createProbe } from "./probe.js";
import { createScene, GAP_DEFAULT_M, GAP_MAX_M } from "./scene.js";
import { createLivePanel } from "./live.js";
import { createTrustPanel } from "./trust.js";
import SPEC from "./spec_constants.json";
import { viaColumns } from "./vias.js";

const params = new URLSearchParams(window.location.search);
const wsUrl = params.get("ws") ?? `ws://${window.location.hostname || "localhost"}:8765`;
const seated = params.has("seated");
const inline = params.has("inline");

const textures = new LayerTextures();
const handle = createHandler({ textures });
const view = createScene(document.getElementById("app"), { seated, hdriUrl: params.get("hdri") });
const overlay = createDevOverlay(document.getElementById("dev-overlay"));
const trust = createTrustPanel(document.getElementById("trust"));

const peel = document.getElementById("peel");
const peelLabel = document.getElementById("peel-value");
peel.value = String(GAP_DEFAULT_M / GAP_MAX_M);
function onPeel() {
  const s = Number(peel.value);
  view.setSeparation(s);
  peelLabel.textContent = `${(1000 * s * GAP_MAX_M).toFixed(0)} mm`;
}
peel.addEventListener("input", onPeel);
onPeel();

const cutOn = document.getElementById("cut-on");
const cutAxis = document.getElementById("cut-axis");
const cutPos = document.getElementById("cut-pos");
cutOn.addEventListener("change", () => (view.cut.mesh.visible = cutOn.checked));
cutAxis.addEventListener("change", () => view.cut.setAxis(cutAxis.value));
cutPos.addEventListener("input", () => view.cut.setCut(Number(cutPos.value)));
document.getElementById("look").addEventListener("change", (ev) => view.setLook(ev.target.value));
document.getElementById("sdf").addEventListener("change", (ev) => view.showSdf(ev.target.checked));
document.getElementById("oit").addEventListener("change", (ev) => view.setOit(ev.target.value === "wboit"));

let records = {};
const recordsLoaded = params.get("records")
  ? fetch(params.get("records"))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
      .then((j) => (records = j))
  : Promise.reject(new Error("no ?records= in the URL"));
let viaKey = null;
function refreshVias() {
  const key = `${handle.status.lastId}|${handle.status.received}|${Object.keys(records).length}`;
  if (key === viaKey) return;
  viaKey = key;
  const rec = records[handle.status.lastId];
  view.setRecord(rec ?? null);
  view.setViaColumns(rec && textures.n > 1 ? viaColumns(rec, textures.values, textures.n, SPEC) : []);
}

const lutSelect = document.getElementById("lut");
for (const name of LUT_NAMES) lutSelect.add(new Option(name, name, name === DEFAULT_LUT, name === DEFAULT_LUT));
lutSelect.addEventListener("change", () => view.setLut(lutSelect.value));

const isoSelect = document.getElementById("iso");
isoSelect.addEventListener("change", () => {
  view.setIsoStep(isoSelect.value === "auto" ? "auto" : Number(isoSelect.value));
});

createProbe({
  camera: view.camera,
  planes: view.planes,
  textures,
  dom: view.renderer.domElement,
  tooltip: document.getElementById("probe"),
});

const worker = inline ? null : new Worker(new URL("./net.worker.js", import.meta.url), { type: "module" });
const conn = connect(wsUrl, handle, { worker, onState: (s) => overlay.setConnection(`${s} ${wsUrl}`) });
// live design panel: query on every chip move, design on commit (server: python -m server.app --verify)
const live = createLivePanel(document.getElementById("live"), {
  send: (msg) => conn.send(msg),
  getRecords: () => records,
  remember: (id, rec) => (records[id] = rec),
});
recordsLoaded
  .then(() => live.say(`${live.refreshRecords()} designs loaded: pick one, press load`))
  .catch((e) => {
    console.warn(`could not load records ${params.get("records")}: ${e.message}`);
    live.say(`no designs: ${e.message} (open ?records=/demo/records.json)`);
  });

if (params.has("debug")) window.twinheat = { view, textures, handle }; // ?debug: console / screenshots

// E10 frame-time recorder (?bench=<seconds>): a CSV the report reads, never a number typed in
const benchS = Number(params.get("bench") ?? 0);
const frames = [];
let benchStart = null;
let lastT = null;
function benchFrame(t) {
  if (!(benchS > 0)) return;
  if (benchStart === null) benchStart = t;
  if (lastT !== null) {
    frames.push([frames.length, (t - lastT).toFixed(3), document.getElementById("look").value,
      document.getElementById("oit").value, peel.value, handle.status.trust?.state ?? "none"]);
  }
  lastT = t;
  if (t - benchStart < 1000 * benchS) return;
  const NL = String.fromCharCode(10);
  const csv = ["frame_index,ms,look,oit,peel,trust", ...frames.map((r) => r.join(","))].join(NL);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv + NL], { type: "text/csv" }));
  a.download = "e10_frames.csv";
  a.click();
  benchStart = Infinity; // once
}

view.start((time, cpuMs) => {
  view.update(textures);
  refreshVias();
  overlay.frame(time, cpuMs, handle.status);
  trust.update(handle.status.trust);
  benchFrame(time);
});
