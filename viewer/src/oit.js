// Weighted blended order-independent transparency for the FR-4 slabs (CLAUDE.md §13 T1.1; McGuire &
// Bavoil 2013, "Weighted Blended Order-Independent Transparency").
//
// Passes, per frame (desktop):
//   1 opaque: the scene with the transparent meshes hidden, drawn to the screen as usual (so its
//     colours are exactly those of the non-OIT path), and again into a target whose depth texture
//     the next passes test against.
//   2 accumulation: transparent fragments into an RGBA16F target cleared to 0, blend ONE, ONE
//     (additive): out = (C·α·w, α·w).
//   3 revealage: the same fragments into an R8 target cleared to 1, blend ZERO, ONE_MINUS_SRC_COLOR
//     (multiplicative): r ← r · (1 − α).
//   4 composite: a full-screen pass over the opaque image, blend SRC_ALPHA, ONE_MINUS_SRC_ALPHA:
//     out = (accum.rgb / max(accum.a, ε), 1 − r), i.e. C_avg · (1 − r) + C_opaque · r.
// Passes 2–3 draw only the transparent meshes (each rendered on its own, so no camera layers are
// needed: three reserves layers 1 and 2 for the two eyes in XR). They do the depth test against the opaque depth texture in the shader (fragments behind
// opaque geometry are discarded) and write no depth. Weight w = α · clamp(10 / (1e-5 + (|z|/5)³),
// 1e-2, 3e3), z the view-space depth in m: a depth-decreasing weight of the paper's form; the
// constants are our choice for a 1 m board seen from 0.3–3 m (UNVERIFIED against the paper's
// equations, which are not in docs/refs).
// In an XR session the slabs are drawn with ordinary sorted alpha blending instead: three's WebXR
// path does not support these offscreen passes, and for parallel horizontal slabs the
// distance-to-centre sort three already does is the exact back-to-front order from any viewpoint.

import * as THREE from "three";

/** The blend state of each pass (stated, and tested). */
export const BLEND = Object.freeze({
  accum: { src: THREE.OneFactor, dst: THREE.OneFactor, target: "RGBA16F" },
  revealage: { src: THREE.ZeroFactor, dst: THREE.OneMinusSrcColorFactor, target: "R8" },
  composite: { src: THREE.SrcAlphaFactor, dst: THREE.OneMinusSrcAlphaFactor, target: "screen" },
});

/** The weight function of passes 2–3 (mirrors shaders/oit_accum.frag). z: view depth, m. */
export function oitWeight(alpha, z) {
  return alpha * Math.min(3e3, Math.max(1e-2, 10 / (1e-5 + Math.pow(Math.abs(z) / 5, 3))));
}

/** CPU model of passes 2–4 for one pixel: fragments [{rgb, a, z}] over background rgb. */
export function wboitComposite(fragments, background) {
  let ar = 0, ag = 0, ab = 0, aa = 0, reveal = 1;
  for (const f of fragments) {
    const w = oitWeight(f.a, f.z);
    ar += f.rgb[0] * f.a * w;
    ag += f.rgb[1] * f.a * w;
    ab += f.rgb[2] * f.a * w;
    aa += f.a * w;
    reveal *= 1 - f.a;
  }
  if (fragments.length === 0) return [...background];
  const d = Math.max(aa, 1e-5);
  return [ar / d, ag / d, ab / d].map((c, i) => c * (1 - reveal) + background[i] * reveal);
}

const QUAD_VERT = `precision highp float;
in vec3 position;
out vec2 vUv;
void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const OIT_VERT = `precision highp float;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
in vec3 position;
out float vViewZ;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vViewZ = mv.z;
  gl_Position = projectionMatrix * mv;
}`;

function fullscreen(frag, uniforms, blend) {
  const m = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: QUAD_VERT,
    fragmentShader: frag, uniforms, depthTest: false, depthWrite: false });
  if (blend) Object.assign(m, { transparent: true, blending: THREE.CustomBlending,
    blendSrc: blend.src, blendDst: blend.dst, blendSrcAlpha: blend.src, blendDstAlpha: blend.dst });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), m);
  mesh.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(mesh);
  return scene;
}

/**
 * The desktop WBOIT renderer; render(scene, camera, transparentMeshes) draws the whole frame.
 * shaders: {accum, composite} sources of shaders/oit_accum.frag and shaders/oit_composite.frag.
 */
export function createWBOIT(renderer, { accum: accumFrag, composite: compositeFrag }) {
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  const opaque = new THREE.WebGLRenderTarget(size.x, size.y, { depthTexture: new THREE.DepthTexture(size.x, size.y) });
  const accum = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, depthBuffer: false });
  const reveal = new THREE.WebGLRenderTarget(size.x, size.y, { format: THREE.RedFormat, type: THREE.UnsignedByteType, depthBuffer: false });
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const composite = fullscreen(compositeFrag, { uAccum: { value: accum.texture }, uReveal: { value: reveal.texture } }, BLEND.composite);
  const shared = { uDepth: { value: opaque.depthTexture }, uResolution: { value: size.clone() } };
  const materials = new Map(); // mesh -> {accum, reveal}

  function passMaterial(mesh, mode, blend) {
    const base = mesh.userData.oit; // {color: THREE.Color, alpha}
    const m = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: OIT_VERT, fragmentShader: accumFrag,
      uniforms: { ...shared, uColor: { value: base.color }, uAlpha: { value: base.alpha }, uMode: { value: mode } },
      transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
      blending: THREE.CustomBlending, blendSrc: blend.src, blendDst: blend.dst,
      blendSrcAlpha: blend.src, blendDstAlpha: blend.dst,
    });
    return m;
  }
  function materialsFor(mesh) {
    if (!materials.has(mesh)) {
      materials.set(mesh, { accum: passMaterial(mesh, 0, BLEND.accum), reveal: passMaterial(mesh, 1, BLEND.revealage) });
    }
    return materials.get(mesh);
  }

  function setSize(w, h) {
    opaque.setSize(w, h);
    accum.setSize(w, h);
    reveal.setSize(w, h);
    shared.uResolution.value.set(w, h);
  }

  function render(scene, camera, transparentMeshes) {
    const autoClear = renderer.autoClear;
    const clear = renderer.getClearColor(new THREE.Color());
    const clearAlpha = renderer.getClearAlpha();
    renderer.autoClear = false;
    const shown = transparentMeshes.filter((m) => m.visible);
    // 1 opaque: to the screen, and into `opaque` for its depth only
    shown.forEach((m) => (m.visible = false));
    renderer.setRenderTarget(null);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
    renderer.setRenderTarget(opaque);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
    shown.forEach((m) => (m.visible = true));
    // 2 accumulation, 3 revealage: each transparent mesh on its own (world matrices are current)
    const originals = shown.map((m) => m.material);
    for (const [target, key, colour, alpha] of [[accum, "accum", 0x000000, 0], [reveal, "reveal", 0xffffff, 1]]) {
      shown.forEach((m) => (m.material = materialsFor(m)[key]));
      renderer.setRenderTarget(target);
      renderer.setClearColor(colour, alpha);
      renderer.clear(true, false, false);
      for (const m of shown) renderer.render(m, camera);
    }
    shown.forEach((m, i) => (m.material = originals[i]));
    // 4 composite
    renderer.setRenderTarget(null);
    renderer.setClearColor(clear, clearAlpha);
    renderer.render(composite, quadCam);
    renderer.autoClear = autoClear;
  }

  return { render, setSize };
}
