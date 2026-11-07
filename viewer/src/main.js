// Viewer entry point: connect to the server, validate every message, show the 4 layer fields.
//
// URL parameters: ?ws=<url> (default ws://<page host>:8765, the mock server's default port)
//                 ?seated      use the "local" reference space instead of "local-floor"
//                 ?inline      parse / validate / decode on the main thread (no worker; debugging)
//                 ?debug       expose window.twinheat = {view, textures, handle} (console, screenshots)
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
if (params.get("records")) {
  fetch(params.get("records"))
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
    .then((j) => (records = j))
    .catch((e) => console.warn(`could not load records ${params.get("records")}: ${e.message}`));
}
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
connect(wsUrl, handle, { worker, onState: (s) => overlay.setConnection(`${s} ${wsUrl}`) });

if (params.has("debug")) window.twinheat = { view, textures, handle }; // ?debug: console / screenshots

view.start((time, cpuMs) => {
  view.update(textures);
  refreshVias();
  overlay.frame(time, cpuMs, handle.status);
});
