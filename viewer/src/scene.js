// Scene: the 4 layer planes, the colour bar, desktop orbit camera and WebXR session (CLAUDE.md §13).
//
// Geometry and camera come from src/layout.js (shared with the orientation tests): the board is
// 1 m wide, layers lie horizontal with L1 (layer index 0) on top, board north points away from
// the desktop camera. Three supplies the scene graph and the WebXR session only; the shaders are
// ours (shaders/field.vert, shaders/thermal.frag).
//
// Reference space: "local-floor" (standing; board at table height in front of the user) or
// "local" when seated (origin at the head at session start; board below eye level).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRButton } from "three/addons/webxr/VRButton.js";

import fieldVert from "../shaders/field.vert?raw";
import thermalFrag from "../shaders/thermal.frag?raw";
import { BOARD_WIDTH_M, GAP_DEFAULT_M, GAP_MAX_M, boardCentre, desktopCamera, layerGeometry, layerOffset } from "./layout.js";
import { DEFAULT_LUT, createLutTexture, setLut as switchLut } from "./lut.js";
import { N_LAYERS } from "./net.js";
import { AMBIENT_C, niceStep } from "./probe.js";

export { BOARD_WIDTH_M, GAP_DEFAULT_M, GAP_MAX_M };

// Display choices (presentation only, not physics).
const ISO_WIDTH_PX = 1.5;
const ISO_COLOUR = new THREE.Color(1, 1, 1);
const ISO_ALPHA = 0.55;
const BAR_LENGTH_M = 0.35;
const BAR_WIDTH_M = 0.03;
const BAR_MARGIN_M = 0.08; // gap between the board's east edge and the bar
const LABEL_WIDTH_M = 0.14;
const CAPTION_M = 0.05; // band above the bar for the LUT name and isoline step
const PX_PER_M = 2048; // label canvas resolution

function thermalMaterial(lut, mode) {
  return new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: fieldVert,
    fragmentShader: thermalFrag,
    uniforms: {
      uField: { value: null },
      uLut: { value: lut },
      uMin: { value: 0 },
      uSpan: { value: 0 },
      uIsoStep: { value: 0 },
      uIsoWidth: { value: ISO_WIDTH_PX },
      uIsoColor: { value: ISO_COLOUR },
      uIsoAlpha: { value: ISO_ALPHA },
      uMode: { value: mode },
    },
    side: THREE.DoubleSide,
  });
}

/** Tick labels in °C (= 25 + θ) for the colour bar, drawn on a canvas texture (works in VR). */
function createBarLabels() {
  const canvas = document.createElement("canvas");
  const w = Math.round(LABEL_WIDTH_M * PX_PER_M);
  const cap = Math.round(CAPTION_M * PX_PER_M); // caption band at the top
  const h = Math.round(BAR_LENGTH_M * PX_PER_M); // tick band, aligned with the bar
  [canvas.width, canvas.height] = [w, cap + h];
  const ctx = canvas.getContext("2d");
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  let key = "";
  function draw(min, max, step, lutName) {
    const k = `${min}|${max}|${step}|${lutName}`;
    if (k === key) return;
    key = k;
    ctx.clearRect(0, 0, w, cap + h);
    ctx.fillStyle = "#e8e8e8";
    ctx.textBaseline = "middle";
    ctx.font = "26px system-ui, sans-serif";
    ctx.fillText(lutName, 4, cap * 0.3);
    ctx.fillText(step > 0 ? `lines every ${step} K` : "no isolines", 4, cap * 0.7);
    const span = max - min;
    const yOf = (theta) => cap + (span > 0 ? h - ((theta - min) / span) * h : h);
    const label = (theta) => `${(AMBIENT_C + theta).toFixed(step > 0 && step < 1 ? 1 : 0)} °C`;
    const ticks = [];
    if (step > 0) for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(v);
    ctx.font = "32px system-ui, sans-serif";
    ctx.fillRect(0, cap, 3, h);
    for (const v of ticks) {
      const y = Math.min(cap + h - 18, Math.max(cap + 18, yOf(v)));
      ctx.fillRect(0, y - 1, 16, 2);
      ctx.fillText(label(v), 24, y);
    }
    texture.needsUpdate = true;
  }
  return { texture, draw };
}

/**
 * Build renderer, scene and camera inside `container`. Returns {renderer, camera, planes,
 * setGap(m), setLut(name), setIsoStep("auto" | K | 0), update(layerTextures), start(onFrame)}.
 */
export function createScene(container, { seated = false } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.xr.enabled = true;
  renderer.xr.setReferenceSpaceType(seated ? "local" : "local-floor");
  container.appendChild(renderer.domElement);
  document.body.appendChild(VRButton.createButton(renderer));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202024);

  const centre = boardCentre(seated);
  const camera = desktopCamera(container.clientWidth / container.clientHeight, seated);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(centre);
  controls.update();

  const lut = createLutTexture(DEFAULT_LUT);
  let lutName = DEFAULT_LUT;
  let isoMode = "auto";
  const board = new THREE.Group();
  board.position.copy(centre);
  scene.add(board);

  const geometry = layerGeometry();
  const planes = Array.from({ length: N_LAYERS }, (_, l) => {
    const mesh = new THREE.Mesh(geometry, thermalMaterial(lut, 0));
    mesh.name = `L${l + 1}`;
    mesh.userData.layer = l; // read by the probe
    board.add(mesh);
    return mesh;
  });

  // colour bar: standing at the east edge; uv.x runs bottom (coldest) to top (hottest)
  const barGeometry = new THREE.PlaneGeometry(BAR_LENGTH_M, BAR_WIDTH_M);
  barGeometry.rotateZ(Math.PI / 2);
  const bar = new THREE.Mesh(barGeometry, thermalMaterial(lut, 1));
  bar.position.set(BOARD_WIDTH_M / 2 + BAR_MARGIN_M, BAR_LENGTH_M / 2, 0);
  board.add(bar);
  const labels = createBarLabels();
  const labelMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(LABEL_WIDTH_M, BAR_LENGTH_M + CAPTION_M),
    new THREE.MeshBasicMaterial({ map: labels.texture, transparent: true, side: THREE.DoubleSide }),
  );
  labelMesh.position.set(bar.position.x + BAR_WIDTH_M / 2 + LABEL_WIDTH_M / 2 + 0.005,
    (BAR_LENGTH_M + CAPTION_M) / 2, 0);
  board.add(labelMesh);

  function setGap(gap) {
    if (!(gap >= 0 && gap <= GAP_MAX_M)) throw new RangeError(`layer gap must be in [0, ${GAP_MAX_M}] m, got ${gap}`);
    planes.forEach((mesh, l) => {
      mesh.position.y = layerOffset(l, N_LAYERS, gap);
    });
  }
  setGap(GAP_DEFAULT_M);

  function setLut(name) {
    switchLut(lut, name); // one shared texture: planes and bar switch together
    lutName = name;
  }

  /** "auto" (niceStep of the field's range), a step in K, or 0 for no isolines. */
  function setIsoStep(mode) {
    if (!(mode === "auto" || (typeof mode === "number" && mode >= 0 && Number.isFinite(mode)))) {
      throw new RangeError(`isoline step must be "auto" or a number >= 0 (K), got ${mode}`);
    }
    isoMode = mode;
  }

  /** Bind the current layer textures, colour range and isoline step (cheap; call every frame). */
  function update(layerTextures) {
    const span = layerTextures.max - layerTextures.min;
    const step = isoMode === "auto" ? niceStep(span) : isoMode;
    for (const mesh of [...planes, bar]) {
      const u = mesh.material.uniforms;
      u.uMin.value = layerTextures.min;
      u.uSpan.value = span;
      u.uIsoStep.value = step;
    }
    planes.forEach((mesh, l) => {
      mesh.material.uniforms.uField.value = layerTextures.textures[l]; // rebinds after a resize
    });
    labels.draw(layerTextures.min, layerTextures.max, step, lutName);
  }

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  /**
   * Start the render loop (setAnimationLoop also drives the XR frame loop). onFrame(time, cpuMs)
   * runs before each render; cpuMs is the CPU time the previous frame took (callback + render).
   */
  function start(onFrame) {
    let cpuMs = 0;
    renderer.setAnimationLoop((time) => {
      const t0 = performance.now();
      onFrame(time, cpuMs);
      renderer.render(scene, camera);
      cpuMs = performance.now() - t0;
    });
  }

  return { renderer, camera, planes, setGap, setLut, setIsoStep, update, start };
}
