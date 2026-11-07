// Jump flooding (Rong & Tan 2006) on the L1 copper mask -> signed distance -> bevel normal map
// (CLAUDE.md §13 T1.4). Baked once per board edit, never per frame.
//
// Passes on an N × N grid (N = JFA_SIZE = 512, a power of two), each a full-screen fragment pass:
//   seed   a pixel whose mask value differs from one of its 4 neighbours (outside the board counts
//          as bare) is a seed and stores its own pixel coordinates; others store "none" (−1).
//   flood  for k = N/2, N/4, ..., 1 (log2 N passes, ping-pong between two RG32F targets): each pixel
//          looks at the 9 pixels at offsets {−k, 0, k}² and keeps the nearest seed it finds.
//   dist   signed distance in mm to the nearest seed: + on copper, − off it (−1e4 if no seed).
//   normal bevel height h = smoothstep(0, w, d) over the bevel width w (mm) inside the copper, 0
//          outside; tangent-space normal n = normalize(−H ∂h/∂x, −H ∂h/∂y, 1) by central differences
//          (H = bevel height, mm), stored as RGB = n/2 + 1/2.
// JFA is approximate: the nearest seed can be missed by a pixel or so (tests bound the error against
// an exact transform). jfaCPU / bevelNormals are the CPU mirror of the GPU passes, used by the tests.
// The mask comes from the record's L1 shapes (maskFromRecord), with data/raster.py's shape rules:
// rect half-open, disc closed, trace = capsule of the given width; row 0 = the south edge (rule 7).

import * as THREE from "three";

export const JFA_SIZE = 512;
const NONE = -1;
const FAR = 1e4; // "no copper edge anywhere", in pixels or mm

export function stepSizes(n) {
  if (!(Number.isInteger(n) && n >= 2 && (n & (n - 1)) === 0)) throw new RangeError(`JFA size must be a power of two, got ${n}`);
  const out = [];
  for (let k = n / 2; k >= 1; k /= 2) out.push(k);
  return out;
}

function inShape(s, x, y) {
  if (s.type === "rect") return x >= s.x0 && x < s.x0 + s.w && y >= s.y0 && y < s.y0 + s.h;
  if (s.type === "disc") return (x - s.cx) ** 2 + (y - s.cy) ** 2 <= s.r ** 2;
  if (s.type === "trace") {
    const r2 = (s.width / 2) ** 2;
    for (let i = 0; i + 1 < s.points.length; i++) {
      const [ax, ay] = s.points[i];
      const [bx, by] = s.points[i + 1];
      const dx = bx - ax, dy = by - ay;
      const L2 = dx * dx + dy * dy;
      const t = L2 === 0 ? 0 : Math.min(1, Math.max(0, ((x - ax) * dx + (y - ay) * dy) / L2));
      if ((x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2 <= r2) return true;
    }
    return false;
  }
  throw new RangeError(`unknown shape type ${s.type}`);
}

/** L1 copper mask Uint8Array [y, x] (row 0 = south), 1 where any L1 shape covers the pixel centre. */
export function maskFromRecord(record, n = JFA_SIZE, boardMm = 100) {
  const shapes = record.layers.find((l) => l.layer === 1)?.shapes ?? [];
  const mask = new Uint8Array(n * n);
  const px = boardMm / n;
  for (const s of shapes) {
    for (let y = 0; y < n; y++) {
      const cy = (y + 0.5) * px;
      for (let x = 0; x < n; x++) if (!mask[y * n + x] && inShape(s, (x + 0.5) * px, cy)) mask[y * n + x] = 1;
    }
  }
  return mask;
}

/** CPU mirror of the seed + flood + dist passes; dist is signed, in pixels. */
export function jfaCPU(mask, n) {
  const at = (x, y) => (x < 0 || y < 0 || x >= n || y >= n ? 0 : mask[y * n + x]);
  let sx = new Float32Array(n * n).fill(NONE);
  let sy = new Float32Array(n * n).fill(NONE);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const m = at(x, y);
      if (at(x + 1, y) !== m || at(x - 1, y) !== m || at(x, y + 1) !== m || at(x, y - 1) !== m) {
        sx[y * n + x] = x;
        sy[y * n + x] = y;
      }
    }
  for (const k of stepSizes(n)) {
    const nx = new Float32Array(n * n), ny = new Float32Array(n * n);
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        let bx = NONE, by = NONE, bd = Infinity;
        for (let j = -1; j <= 1; j++)
          for (let i = -1; i <= 1; i++) {
            const qx = x + i * k, qy = y + j * k;
            if (qx < 0 || qy < 0 || qx >= n || qy >= n) continue;
            const s = qy * n + qx;
            if (sx[s] === NONE) continue;
            const d = (sx[s] - x) ** 2 + (sy[s] - y) ** 2;
            if (d < bd) [bd, bx, by] = [d, sx[s], sy[s]];
          }
        nx[y * n + x] = bx;
        ny[y * n + x] = by;
      }
    sx = nx;
    sy = ny;
  }
  const dist = new Float32Array(n * n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const i = y * n + x;
      const d = sx[i] === NONE ? FAR : Math.hypot(sx[i] - x, sy[i] - y);
      dist[i] = mask[i] ? d : -d;
    }
  return { seedX: sx, seedY: sy, dist };
}

/** Bevel height in [0, 1] at signed distance d (mm) for a bevel width w (mm). */
export function bevelHeight(d, w) {
  if (d <= 0) return 0;
  if (d >= w) return 1;
  const t = d / w;
  return t * t * (3 - 2 * t);
}

/** CPU mirror of the normal pass: Float32Array [3 n²] of unit normals from dist (pixels). */
export function bevelNormals(dist, n, mmPerPx, bevelMm, heightMm) {
  const h = (x, y) => bevelHeight(dist[Math.min(n - 1, Math.max(0, y)) * n + Math.min(n - 1, Math.max(0, x))] * mmPerPx, bevelMm);
  const out = new Float32Array(3 * n * n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const gx = (heightMm * (h(x + 1, y) - h(x - 1, y))) / (2 * mmPerPx);
      const gy = (heightMm * (h(x, y + 1) - h(x, y - 1))) / (2 * mmPerPx);
      const len = Math.hypot(gx, gy, 1);
      const i = 3 * (y * n + x);
      out[i] = -gx / len;
      out[i + 1] = -gy / len;
      out[i + 2] = 1 / len;
    }
  return out;
}

// ---------------------------------------------------------------------------- GPU bake

const QUAD_VERT = `precision highp float;
in vec3 position;
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`;

/**
 * GPU baker. shaders: {seed, step, dist, normal} fragment sources (shaders/jfa_*.frag, passed in so
 * this module loads in Node tests). bake(maskBytes, {boardMm, bevelMm, heightMm}) runs every pass
 * once and returns {sdf, normal} textures (R32F mm, RGBA8 tangent-space normal) plus the time taken.
 */
export function createJFA(renderer, shaders, n = JFA_SIZE) {
  const steps = stepSizes(n);
  const rg = () => new THREE.WebGLRenderTarget(n, n, { type: THREE.FloatType, format: THREE.RGFormat,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false });
  const ping = [rg(), rg()];
  const sdf = new THREE.WebGLRenderTarget(n, n, { type: THREE.FloatType, format: THREE.RedFormat,
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false });
  const normal = new THREE.WebGLRenderTarget(n, n, { minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter, depthBuffer: false });
  normal.texture.colorSpace = THREE.NoColorSpace;
  const maskTex = new THREE.DataTexture(new Uint8Array(n * n), n, n, THREE.RedFormat, THREE.UnsignedByteType);
  maskTex.minFilter = maskTex.magFilter = THREE.NearestFilter;
  const mat = (frag, uniforms) => new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3,
    vertexShader: QUAD_VERT, fragmentShader: frag, uniforms, depthTest: false, depthWrite: false });
  const m = {
    seed: mat(shaders.seed, { uMask: { value: maskTex } }),
    step: mat(shaders.step, { uSeeds: { value: null }, uStep: { value: 1 } }),
    dist: mat(shaders.dist, { uSeeds: { value: null }, uMask: { value: maskTex }, uMmPerPx: { value: 1 } }),
    normal: mat(shaders.normal, { uSdf: { value: sdf.texture }, uMmPerPx: { value: 1 },
      uBevelMm: { value: 1 }, uHeightMm: { value: 1 } }),
  };
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), m.seed);
  quad.frustumCulled = false;
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const pass = (material, target) => {
    quad.material = material;
    renderer.setRenderTarget(target);
    renderer.render(quad, cam);
  };

  function bake(maskBytes, { boardMm, bevelMm, heightMm }) {
    const t0 = performance.now();
    if (maskBytes.length !== n * n) throw new RangeError(`mask must have ${n * n} pixels, got ${maskBytes.length}`);
    // 0/1 -> 0/255: an R8 texture reads byte v as v / 255, and the shaders test > 0.5
    for (let i = 0; i < maskBytes.length; i++) maskTex.image.data[i] = maskBytes[i] ? 255 : 0;
    maskTex.needsUpdate = true;
    const prev = renderer.getRenderTarget();
    const xr = renderer.xr.enabled;
    renderer.xr.enabled = false; // offscreen passes, never per eye
    pass(m.seed, ping[0]);
    let src = 0;
    for (const k of steps) {
      m.step.uniforms.uSeeds.value = ping[src].texture;
      m.step.uniforms.uStep.value = k;
      pass(m.step, ping[1 - src]);
      src = 1 - src;
    }
    const mmPerPx = boardMm / n;
    m.dist.uniforms.uSeeds.value = ping[src].texture;
    m.dist.uniforms.uMmPerPx.value = mmPerPx;
    pass(m.dist, sdf);
    m.normal.uniforms.uMmPerPx.value = mmPerPx;
    m.normal.uniforms.uBevelMm.value = bevelMm;
    m.normal.uniforms.uHeightMm.value = heightMm;
    pass(m.normal, normal);
    renderer.setRenderTarget(prev);
    renderer.xr.enabled = xr;
    return { sdf: sdf.texture, normal: normal.texture, passes: 3 + steps.length, ms: performance.now() - t0 };
  }
  return { bake, size: n };
}
