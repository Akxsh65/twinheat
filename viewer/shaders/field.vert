// Layer plane (RawShaderMaterial, GLSL ES 3.0; three prepends "#version 300 es").
// uv (0, 0) is the board's SW corner, matching texture row 0 = y index 0 (CLAUDE.md §0 rule 7).
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
