// Temperature probe and isoline spacing (CLAUDE.md §13 T1.3; §0 rule 6: display °C = 25 + θ).
//
// The probe raycasts the layer planes and reads θ from the CPU-side Float32 copy kept by
// LayerTextures, at the cell under the hit's uv: x = floor(u·N), y = floor(v·N) (uv (0, 0) is the
// SW corner, src/layout.js), so it reads exactly the value the texture shows at that cell's centre.

import { Raycaster, Vector2 } from "three";

/** Display only: θ is the rise above ambient, shown as °C = 25 + θ (CLAUDE.md §0 rule 6, §6.3). */
export const AMBIENT_C = 25;

/** Cell {x, y} under uv for an n x n field; u or v = 1 belongs to the last cell. */
export function cellAt(uv, n) {
  const ok = (v) => Number.isFinite(v) && v >= 0 && v <= 1;
  if (!(ok(uv.x) && ok(uv.y))) throw new RangeError(`uv must lie in [0, 1]², got (${uv.x}, ${uv.y})`);
  return { x: Math.min(n - 1, Math.floor(uv.x * n)), y: Math.min(n - 1, Math.floor(uv.y * n)) };
}

/** {layer, x, y, theta (K), celsius} at uv on layer (0 = L1). */
export function probeValue(textures, layer, uv) {
  const { x, y } = cellAt(uv, textures.n);
  const theta = textures.valueAt(layer, y, x);
  return { layer, x, y, theta, celsius: AMBIENT_C + theta };
}

/** Probe the nearest hit of a Raycaster.intersectObjects() result over the layer planes, or null. */
export function probeHits(hits, textures) {
  const hit = hits.find((h) => Number.isInteger(h.object.userData.layer) && h.uv);
  return hit ? probeValue(textures, hit.object.userData.layer, hit.uv) : null;
}

/**
 * Isoline spacing (K): the smallest 1, 2 or 5 x 10^k that gives at most 10 lines over the span
 * (a display choice); 0 for a constant field (no isolines).
 */
export function niceStep(span, maxLines = 10) {
  if (!(span > 0)) return 0;
  const raw = span / maxLines;
  const k = Math.floor(Math.log10(raw));
  const base = 10 ** k;
  for (const m of [1, 2, 5, 10]) if (m * base >= raw * (1 - 1e-12)) return Number((m * base).toPrecision(12));
  return 10 * base;
}

/**
 * Desktop probe: on pointer move over `dom`, raycast the planes (meshes with userData.layer) from
 * `camera` and show "L<n> x, y · T °C (θ K)" in `tooltip` next to the pointer; hide it otherwise.
 * Returns {dispose}. (The VR probe pen, T1.5, calls probeHits with a controller ray.)
 */
export function createProbe({ camera, planes, textures, dom, tooltip }) {
  const ray = new Raycaster();
  const ndc = new Vector2();
  function move(ev) {
    const r = dom.getBoundingClientRect();
    ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const p = probeHits(ray.intersectObjects(planes.filter((m) => m.visible), false), textures);
    if (p === null) {
      tooltip.hidden = true;
      return;
    }
    tooltip.textContent = `L${p.layer + 1}  x ${p.x}, y ${p.y}  ·  ${p.celsius.toFixed(1)} °C  (θ ${p.theta.toFixed(2)} K)`;
    tooltip.style.left = `${ev.clientX - r.left + 14}px`;
    tooltip.style.top = `${ev.clientY - r.top + 14}px`;
    tooltip.hidden = false;
  }
  function leave() {
    tooltip.hidden = true;
  }
  dom.addEventListener("pointermove", move);
  dom.addEventListener("pointerleave", leave);
  return {
    dispose() {
      dom.removeEventListener("pointermove", move);
      dom.removeEventListener("pointerleave", leave);
    },
  };
}
