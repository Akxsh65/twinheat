// Scene: the 4 layer planes, desktop orbit camera and WebXR session (CLAUDE.md §13).
//
// The board is shown 1 m wide (§13). Layers lie horizontal, L1 (layer index 0) on top, stacked
// downwards with an adjustable gap. Board north (+y, uv v = 1) points away from the desktop camera
// (world -z). Three supplies the scene graph and the WebXR session only; the shaders are ours.
//
// Reference space: "local-floor" (standing; board at table height in front of the user) or
// "local" when seated (origin at the head at session start; board below eye level).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRButton } from "three/addons/webxr/VRButton.js";

import fieldFrag from "../shaders/field.frag?raw";
import fieldVert from "../shaders/field.vert?raw";
import { createLutTexture } from "./lut.js";
import { N_LAYERS } from "./net.js";

export const BOARD_WIDTH_M = 1.0; // CLAUDE.md §13
// Viewer layout defaults (presentation only, not physics), in metres.
export const GAP_DEFAULT_M = 0.05;
export const GAP_MAX_M = 0.3;
const BOARD_CENTRE = {
  standing: new THREE.Vector3(0, 1.0, -0.8), // local-floor: y is height above the floor
  seated: new THREE.Vector3(0, -0.35, -0.7), // local: y is relative to the head
};

/**
 * Build renderer, scene and camera inside `container`. Returns
 * {renderer, setGap(m), update(layerTextures), start(onFrame)}.
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

  const centre = seated ? BOARD_CENTRE.seated : BOARD_CENTRE.standing;
  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 50);
  camera.position.set(centre.x, centre.y + 0.9, centre.z + 1.1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(centre);
  controls.update();

  const lut = createLutTexture();
  const board = new THREE.Group();
  board.position.copy(centre);
  scene.add(board);

  const geometry = new THREE.PlaneGeometry(BOARD_WIDTH_M, BOARD_WIDTH_M);
  geometry.rotateX(-Math.PI / 2); // lie flat; local +y (board north) -> world -z
  const planes = Array.from({ length: N_LAYERS }, (_, l) => {
    const material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fieldVert,
      fragmentShader: fieldFrag,
      uniforms: {
        uField: { value: null },
        uLut: { value: lut },
        uMin: { value: 0 },
        uSpan: { value: 0 },
      },
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `L${l + 1}`;
    board.add(mesh);
    return mesh;
  });

  function setGap(gap) {
    if (!(gap >= 0 && gap <= GAP_MAX_M)) throw new RangeError(`layer gap must be in [0, ${GAP_MAX_M}] m, got ${gap}`);
    // L1 on top; the stack stays centred on the board centre
    planes.forEach((mesh, l) => {
      mesh.position.y = ((N_LAYERS - 1) / 2 - l) * gap;
    });
  }
  setGap(GAP_DEFAULT_M);

  /** Bind the current layer textures and colour range (cheap; call every frame). */
  function update(layerTextures) {
    const span = layerTextures.max - layerTextures.min;
    planes.forEach((mesh, l) => {
      const u = mesh.material.uniforms;
      u.uField.value = layerTextures.textures[l]; // rebinds after a resize
      u.uMin.value = layerTextures.min;
      u.uSpan.value = span;
    });
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

  return { renderer, setGap, update, start };
}
