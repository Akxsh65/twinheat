// Realistic top layer (CLAUDE.md §13 T1.4): copper, solder mask, silkscreen and packages over L1, lit by
// image-based lighting. Built from the board record; the JFA bevel bake runs once per record (per
// edit), never per frame.
//
// Three stacked planes at the L1 height, all MeshPhysicalMaterial:
//   base        FR-4 and L1 copper. Colour, roughness and metalness maps from the copper mask
//               (metalness 1, roughness 0.25 on copper, with low-frequency oxidation noise that darkens
//               and roughens it); normal map = the JFA bevel normals (src/jfa.js).
//   solder mask clear-coated green, lighter where it lies over copper (as on a real board) and carrying
//               the same JFA bevel normals, so the pours read through the mask.
//   silkscreen  white chip outlines and reference designators from a canvas texture (alpha-tested).
// IBL: a small HDRI when ?hdri=<url> is given (RGBE), otherwise three's procedural RoomEnvironment;
// it only lights the physical materials, not the thermal shaders. Debug: showSdf(true) replaces the
// base by a view of the signed distance (orange copper, blue board, white edge, a line every 1 mm).
// Colours, roughness noise, thicknesses and offsets are display choices.

import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";

import { JFA_SIZE, maskFromRecord } from "./jfa.js";
import { createPackageMeshes, packageLayout } from "./packages.js";

// Display choices (mm on the real board, sRGB colours).
const BEVEL_MM = 0.4;
const BEVEL_HEIGHT_MM = 0.05;
const COPPER = [0xb8, 0x73, 0x33];
const COPPER_OXIDE = [0x7a, 0x4a, 0x24];
const FR4 = [0xc9, 0xb2, 0x6b];
const MASK_OVER_FR4 = [0x1a, 0x5c, 0x30];
const MASK_OVER_CU = [0x3c, 0x84, 0x36];
const SILK_PX = 2048;
const LIFT_M = { mask: 0.0006, silk: 0.0011, packages: 0.0012 }; // above the L1 plane, metres

function hashString(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

/** Seeded value noise in [0, 1], [n, n], a few octaves (oxidation pattern). */
export function oxidationNoise(n, seed) {
  const g = 17;
  let s = seed || 1;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  const out = new Float32Array(n * n);
  let amp = 1, total = 0;
  for (let oct = 0; oct < 4; oct++) {
    const cells = g << oct;
    const lat = Float32Array.from({ length: (cells + 1) ** 2 }, rnd);
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const fx = (x / n) * cells, fy = (y / n) * cells;
        const ix = Math.floor(fx), iy = Math.floor(fy), tx = fx - ix, ty = fy - iy;
        const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
        const v = (a, b) => lat[(iy + b) * (cells + 1) + ix + a];
        out[y * n + x] += amp * ((v(0, 0) * (1 - sx) + v(1, 0) * sx) * (1 - sy) + (v(0, 1) * (1 - sx) + v(1, 1) * sx) * sy);
      }
    total += amp;
    amp *= 0.5;
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

function dataTexture(bytes, n, srgb) {
  const t = new THREE.DataTexture(bytes, n, n, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = true;
  t.anisotropy = 8;
  t.needsUpdate = true;
  return t;
}

const SDF_FRAG = `precision highp float;
uniform sampler2D uSdf;
in vec2 vUv;
out vec4 o;
void main() {
  float d = texture(uSdf, vUv).r;                 // mm, + on copper
  vec3 c = d > 0.0 ? vec3(0.95, 0.55, 0.2) : vec3(0.15, 0.35, 0.8);
  float band = 0.5 + 0.5 * cos(6.2831853 * d);   // a line every 1 mm
  c *= mix(0.55, 1.0, band);
  float edge = 1.0 - smoothstep(0.0, 0.08, abs(d));
  o = vec4(mix(c, vec3(1.0), edge), 1.0);
}`;
const SDF_VERT = `precision highp float;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
in vec3 position;
in vec2 uv;
out vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

/**
 * The realistic look. jfa: createJFA(...) baker. Returns {group, setRecord(record), showSdf(on),
 * setVisible(on), setHeight(y), lastBakeMs}.
 */
export function createBoardLook({ renderer, scene, jfa, boardWidth, layerGeometry, hdriUrl = null }) {
  const n = JFA_SIZE;
  const group = new THREE.Group();
  group.name = "realistic L1";
  group.visible = false;

  // image-based lighting
  const pmrem = new THREE.PMREMGenerator(renderer);
  let envMap = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  if (hdriUrl) {
    new RGBELoader().load(hdriUrl, (hdr) => {
      hdr.mapping = THREE.EquirectangularReflectionMapping;
      envMap = pmrem.fromEquirectangular(hdr).texture;
      hdr.dispose();
      if (group.visible) scene.environment = envMap;
    }, undefined, (e) => console.warn(`HDRI ${hdriUrl} not loaded (${e?.message ?? e}); using RoomEnvironment`));
  }

  const base = new THREE.Mesh(layerGeometry, new THREE.MeshPhysicalMaterial({ color: 0xffffff,
    roughness: 1, metalness: 1, normalScale: new THREE.Vector2(1, 1) }));
  const maskPlane = new THREE.Mesh(layerGeometry, new THREE.MeshPhysicalMaterial({
    color: 0xffffff, roughness: 0.45, metalness: 0, clearcoat: 0.5, clearcoatRoughness: 0.4,
    envMapIntensity: 0.45, normalScale: new THREE.Vector2(1, 1) }));
  const silkCanvas = typeof document === "undefined" ? null : document.createElement("canvas");
  const silkTex = silkCanvas ? new THREE.CanvasTexture(silkCanvas) : null;
  if (silkCanvas) {
    silkCanvas.width = silkCanvas.height = SILK_PX;
    silkTex.colorSpace = THREE.SRGBColorSpace;
    silkTex.anisotropy = 8;
  }
  const silk = new THREE.Mesh(layerGeometry, new THREE.MeshPhysicalMaterial({ color: 0xffffff,
    map: silkTex, transparent: true, alphaTest: 0.5, roughness: 0.8, metalness: 0 }));
  const sdfMaterial = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: SDF_VERT,
    fragmentShader: SDF_FRAG, uniforms: { uSdf: { value: null } } });
  const packages = createPackageMeshes();
  group.add(base, maskPlane, silk, ...packages.meshes);
  const physical = base.material;
  let lastBakeMs = null;
  let sdfOn = false;

  function drawSilkscreen(record) {
    if (!silkCanvas) return;
    const ctx = silkCanvas.getContext("2d");
    const px = SILK_PX / 100; // canvas px per mm; canvas row 0 = north (CanvasTexture flips)
    ctx.clearRect(0, 0, SILK_PX, SILK_PX);
    ctx.strokeStyle = ctx.fillStyle = "#f4f4ee";
    ctx.lineWidth = 0.15 * px;
    ctx.font = `${Math.round(1.6 * px)}px monospace`;
    ctx.textBaseline = "bottom";
    record.chips.filter((c) => c.side === "top").forEach((c, k) => {
      const m = 0.6; // outline offset, mm
      ctx.strokeRect((c.x0 - m) * px, (100 - c.y0 - c.h - m) * px, (c.w + 2 * m) * px, (c.h + 2 * m) * px);
      ctx.fillText(`U${k + 1}`, (c.x0 - m) * px, (100 - c.y0 - c.h - m - 0.3) * px);
    });
    ctx.font = `${Math.round(2.2 * px)}px monospace`;
    ctx.fillText(`TwinHeat  ${record.id}`, 3 * px, 97 * px);
    silkTex.needsUpdate = true;
  }

  function setRecord(record) {
    const mask = maskFromRecord(record, n);
    const noise = oxidationNoise(n, hashString(record.id));
    const colour = new Uint8Array(4 * n * n);
    const orm = new Uint8Array(4 * n * n);
    const maskColour = new Uint8Array(4 * n * n);
    for (let i = 0; i < n * n; i++) {
      const cu = mask[i] === 1;
      const ox = cu ? Math.max(0, noise[i] - 0.45) * 1.6 : 0; // patches, not a uniform tint
      const c = cu ? COPPER.map((v, j) => v + (COPPER_OXIDE[j] - v) * ox) : FR4;
      colour.set([c[0], c[1], c[2], 255], 4 * i);
      const g = cu ? MASK_OVER_CU : MASK_OVER_FR4;
      maskColour.set([g[0], g[1], g[2], 255], 4 * i);
      // glTF ORM packing as three reads it: G = roughness, B = metalness
      orm.set([255, Math.round(255 * (cu ? 0.25 + 0.35 * ox : 0.8)), cu ? 255 : 0, 255], 4 * i);
    }
    const bake = jfa.bake(mask, { boardMm: 100, bevelMm: BEVEL_MM, heightMm: BEVEL_HEIGHT_MM });
    lastBakeMs = bake.ms;
    physical.map?.dispose();
    physical.roughnessMap?.dispose();
    physical.map = dataTexture(colour, n, true);
    physical.roughnessMap = physical.metalnessMap = dataTexture(orm, n, false);
    physical.normalMap = bake.normal;
    physical.needsUpdate = true;
    const mm = maskPlane.material;
    mm.map?.dispose();
    mm.map = dataTexture(maskColour, n, true);
    mm.normalMap = bake.normal;
    mm.needsUpdate = true;
    sdfMaterial.uniforms.uSdf.value = bake.sdf;
    drawSilkscreen(record);
    packages.update(packageLayout(record), boardWidth, LIFT_M.packages);
    return bake;
  }

  return {
    group,
    setRecord,
    showSdf(on) {
      sdfOn = Boolean(on);
      base.material = sdfOn ? sdfMaterial : physical;
      maskPlane.visible = silk.visible = !sdfOn;
      packages.meshes.forEach((m) => (m.visible = !sdfOn));
    },
    setVisible(on) {
      group.visible = Boolean(on);
      scene.environment = group.visible ? envMap : null;
    },
    setHeight(y) {
      base.position.y = y;
      maskPlane.position.y = y + LIFT_M.mask;
      silk.position.y = y + LIFT_M.silk;
      packages.meshes.forEach((m) => (m.position.y = y));
    },
    get lastBakeMs() {
      return lastBakeMs;
    },
    get sdfOn() {
      return sdfOn;
    },
  };
}
