// Cut plane (src/cutPlane.js): pass the plane's uv; vUv.x runs along the cut, vUv.y bottom (L4) to
// top (L1) of the stack.
precision highp float;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec2 uv;

out vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
