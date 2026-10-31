// Viewer entry point: connect to the server, validate every message, show the 4 layer fields.
//
// URL parameters: ?ws=<url> (default ws://<page host>:8765, the mock server's default port)
//                 ?seated      use the "local" reference space instead of "local-floor"
//                 ?inline      parse / validate / decode on the main thread (no worker; debugging)

import { createDevOverlay } from "./devOverlay.js";
import { DEFAULT_LUT, LUT_NAMES } from "./lut.js";
import { connect, createHandler, LayerTextures } from "./net.js";
import { createProbe } from "./probe.js";
import { createScene, GAP_DEFAULT_M, GAP_MAX_M } from "./scene.js";

const params = new URLSearchParams(window.location.search);
const wsUrl = params.get("ws") ?? `ws://${window.location.hostname || "localhost"}:8765`;
const seated = params.has("seated");
const inline = params.has("inline");

const textures = new LayerTextures();
const handle = createHandler({ textures });
const view = createScene(document.getElementById("app"), { seated });
const overlay = createDevOverlay(document.getElementById("dev-overlay"));

const gap = document.getElementById("gap");
const gapLabel = document.getElementById("gap-value");
gap.max = String(GAP_MAX_M);
gap.value = String(GAP_DEFAULT_M);
function onGap() {
  const m = Number(gap.value);
  view.setGap(m);
  gapLabel.textContent = `${(1000 * m).toFixed(0)} mm`;
}
gap.addEventListener("input", onGap);
onGap();

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

view.start((time, cpuMs) => {
  view.update(textures);
  overlay.frame(time, cpuMs, handle.status);
});
