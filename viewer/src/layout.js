// Board layout shared by the scene and the tests: geometry, layer heights, desktop camera.
// No shaders and no DOM, so Node tests can check orientation with the same code the viewer runs.
//
// Record and field conventions (CLAUDE.md §0 rule 7): arrays are [layer, y, x]; y index 0 is the
// south edge, x index 0 the west edge. A layer plane has uv (0, 0) at the board's SW corner and
// u along +x (east), v along +y (north); texture row 0 is y = 0 (flipY = false). The plane lies
// flat with north pointing away from the desktop camera (world -z), so south-east is bottom-right.

import { PerspectiveCamera, PlaneGeometry, Vector3 } from "three";

export const BOARD_WIDTH_M = 1.0; // CLAUDE.md §13
// Viewer layout defaults (presentation only, not physics), in metres.
export const GAP_DEFAULT_M = 0.05;
export const GAP_MAX_M = 0.3;
const BOARD_CENTRE = {
  standing: new Vector3(0, 1.0, -0.8), // local-floor: y is height above the floor
  seated: new Vector3(0, -0.35, -0.7), // local: y is relative to the head
};
const CAMERA_OFFSET = new Vector3(0, 0.9, 1.1); // desktop view: above and in front (south)

export function boardCentre(seated = false) {
  return (seated ? BOARD_CENTRE.seated : BOARD_CENTRE.standing).clone();
}

/** One layer plane, BOARD_WIDTH_M square, lying flat; local +y (board north) -> world -z. */
export function layerGeometry() {
  const g = new PlaneGeometry(BOARD_WIDTH_M, BOARD_WIDTH_M);
  g.rotateX(-Math.PI / 2);
  return g;
}

/** Height of layer l (0 = L1, on top) above the stack centre, for a layer gap in metres. */
export function layerOffset(l, nLayers, gap) {
  return ((nLayers - 1) / 2 - l) * gap;
}

/** The desktop camera, looking at the board centre from above and in front (south). */
export function desktopCamera(aspect, seated = false) {
  const centre = boardCentre(seated);
  const camera = new PerspectiveCamera(50, aspect, 0.01, 50);
  camera.position.copy(centre).add(CAMERA_OFFSET);
  camera.lookAt(centre);
  camera.updateMatrixWorld();
  return camera;
}
