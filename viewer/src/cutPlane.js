// Through-thickness cut plane (CLAUDE.md §13 T1.1): a vertical section through the layer stack.
//
// The plane stands at board coordinate v0 (fraction of the side, 0 = south / west) across the
// whole board, from L4 up to L1. At a point of the plane, its height y (relative to the stack
// centre) gives the layer coordinate s = 1.5 − y / gap ∈ [0, 3] (0 = L1 on top, 3 = L4), and the
// value is the 4 layer textures sampled at that board position and blended with linear "hat"
// weights max(0, 1 − |s − l|): linear interpolation in z between neighbouring layers, exact at the
// layer planes. GLSL ES 3.00 cannot index a sampler array with a variable, so the shader samples
// all four. Colours use the same LUT and range as the layers. sectionValue / heightToLayer are the
// CPU mirror (tested). Desktop: drag the plane (it follows the pointer over the board) or use the
// slider; the VR grab comes with the interaction work (T1.5).

import * as THREE from "three";

const N_LAYERS = 4;

/** Layer coordinate s ∈ [0, 3] at height y (m, relative to the stack centre) for a layer gap. */
export function heightToLayer(y, gap) {
  if (!(gap > 0)) return 0;
  return Math.min(N_LAYERS - 1, Math.max(0, (N_LAYERS - 1) / 2 - y / gap));
}

/** θ at board uv {x, y} ∈ [0, 1]² and layer coordinate s, from values [4, n, n] (nearest cell). */
export function sectionValue(values, n, uv, s) {
  const x = Math.min(n - 1, Math.floor(uv.x * n));
  const y = Math.min(n - 1, Math.floor(uv.y * n));
  let v = 0;
  for (let l = 0; l < N_LAYERS; l++) v += Math.max(0, 1 - Math.abs(s - l)) * values[(l * n + y) * n + x];
  return v;
}

/**
 * The cut-plane mesh (add it to the board group). axis "x": the plane runs east–west at north
 * fraction `cut`; axis "y": north–south at east fraction `cut`. shaders: {vert, frag} sources of
 * shaders/section.{vert,frag} (passed in so this module also loads in Node tests).
 */
export function createCutPlane(boardWidth, lut, shaders) {
  const geometry = new THREE.PlaneGeometry(1, 1); // scaled to (board width, stack height)
  const material = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: shaders.vert,
    fragmentShader: shaders.frag,
    uniforms: {
      uField0: { value: null }, uField1: { value: null }, uField2: { value: null }, uField3: { value: null },
      uLut: { value: lut }, uMin: { value: 0 }, uSpan: { value: 0 },
      uCut: { value: 0.5 }, uAxis: { value: 0 }, uGap: { value: 0.05 },
    },
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "cut plane";
  mesh.visible = false;
  const state = { cut: 0.5, axis: "x", gap: 0.05 };

  function place() {
    const h = N_LAYERS - 1 > 0 ? (N_LAYERS - 1) * state.gap : 0;
    mesh.scale.set(boardWidth, Math.max(h, 1e-6), 1);
    const off = (state.cut - 0.5) * boardWidth;
    if (state.axis === "x") {
      mesh.rotation.set(0, 0, 0);
      mesh.position.set(0, 0, -off); // board north = world −z
    } else {
      mesh.rotation.set(0, Math.PI / 2, 0);
      mesh.position.set(off, 0, 0);
    }
    const u = material.uniforms;
    u.uCut.value = state.cut;
    u.uAxis.value = state.axis === "x" ? 0 : 1;
    u.uGap.value = state.gap;
  }
  place();

  return {
    mesh,
    setCut(v) {
      state.cut = Math.min(1, Math.max(0, v));
      place();
    },
    setAxis(a) {
      if (a !== "x" && a !== "y") throw new RangeError(`cut axis must be "x" or "y", got ${a}`);
      state.axis = a;
      place();
    },
    setGap(gap) {
      state.gap = gap;
      place();
    },
    update(layerTextures) {
      const u = material.uniforms;
      layerTextures.textures.forEach((t, l) => (u[`uField${l}`].value = t));
      u.uMin.value = layerTextures.min;
      u.uSpan.value = layerTextures.max - layerTextures.min;
    },
    get cut() {
      return state.cut;
    },
    get axis() {
      return state.axis;
    },
  };
}

/**
 * Desktop drag: pointer-down on the cut plane, then the plane follows the pointer's position over the
 * board (a ray to the horizontal plane through the stack centre). OrbitControls pause while dragging.
 */
export function attachCutDrag({ cut, camera, dom, controls, boardCentre, boardWidth, onChange }) {
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), -boardCentre.y);
  const hit = new THREE.Vector3();
  let dragging = false;
  const toNdc = (ev) => {
    const r = dom.getBoundingClientRect();
    ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
  };
  dom.addEventListener("pointerdown", (ev) => {
    if (!cut.mesh.visible) return;
    toNdc(ev);
    if (ray.intersectObject(cut.mesh).length) {
      dragging = true;
      controls.enabled = false;
    }
  });
  dom.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    toNdc(ev);
    if (!ray.ray.intersectPlane(ground, hit)) return;
    const v = cut.axis === "x" ? -(hit.z - boardCentre.z) / boardWidth + 0.5 : (hit.x - boardCentre.x) / boardWidth + 0.5;
    cut.setCut(v);
    onChange?.(cut.cut);
  });
  const end = () => {
    dragging = false;
    controls.enabled = true;
  };
  dom.addEventListener("pointerup", end);
  dom.addEventListener("pointerleave", end);
}
