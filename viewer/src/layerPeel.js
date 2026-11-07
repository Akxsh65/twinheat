// Layer peel (CLAUDE.md §13 T1.1): separation s ∈ [0, 1] pulls the 4 layer planes apart; the layer gap
// is s · GAP_MAX_M. Desktop: a slider sets the target. VR: while BOTH grips are held, the change in
// the distance between the controllers since the grab began moves the target (PEEL_RANGE_M of
// spread = the full range). The shown separation follows the target with exponential smoothing
// (time constant PEEL_TAU_S), so tracking jitter and slider jumps never make the stack pop.
// State is plain data and peelStep is pure, so it is tested without WebGL or XR.

import { GAP_MAX_M } from "./layout.js";

// Display choices (presentation only).
export const PEEL_TAU_S = 0.12;
export const PEEL_RANGE_M = 0.4;

const clamp01 = (v) => Math.min(1, Math.max(0, v));

export function createPeel(s0 = 0) {
  const s = clamp01(s0);
  return { s, target: s, grab: null };
}

export function setPeelTarget(state, s) {
  return { ...state, target: clamp01(s) };
}

/** Advance by dt seconds. input: {bothGrips: bool, distance: m between controllers}. */
export function peelStep(state, input, dt) {
  let { target, grab } = state;
  if (input.bothGrips) {
    if (grab === null) grab = { d0: input.distance, t0: target };
    target = clamp01(grab.t0 + (input.distance - grab.d0) / PEEL_RANGE_M);
  } else {
    grab = null;
  }
  const a = 1 - Math.exp(-Math.max(dt, 0) / PEEL_TAU_S);
  return { s: state.s + a * (target - state.s), target, grab };
}

export const gapForSeparation = (s) => clamp01(s) * GAP_MAX_M;

/**
 * Read the two XR controllers' grip state and distance each frame. Returns {input()}: the
 * {bothGrips, distance} for peelStep. Uses the "squeeze" (grip) events of three's WebXRManager.
 */
export function xrPeelInput(renderer, scene) {
  const held = [false, false];
  const ctrls = [0, 1].map((i) => {
    const c = renderer.xr.getController(i);
    c.addEventListener("squeezestart", () => (held[i] = true));
    c.addEventListener("squeezeend", () => (held[i] = false));
    c.addEventListener("disconnected", () => (held[i] = false));
    scene.add(c);
    return c;
  });
  return {
    input() {
      const both = held[0] && held[1] && renderer.xr.isPresenting;
      return { bothGrips: both, distance: both ? ctrls[0].position.distanceTo(ctrls[1].position) : 0 };
    },
  };
}
