// Weighted blended OIT, pass 4 (src/oit.js): over the opaque image with blend SRC_ALPHA,
// ONE_MINUS_SRC_ALPHA. out = (average transparent colour, 1 - revealage), so the screen becomes
// C_avg * (1 - r) + C_opaque * r. Pixels no transparent fragment touched (r = 1) are left alone.
precision highp float;

uniform sampler2D uAccum;
uniform sampler2D uReveal;

in vec2 vUv;

out vec4 fragColor;

void main() {
  float r = texture(uReveal, vUv).r;
  if (r >= 1.0) discard;
  vec4 a = texture(uAccum, vUv);
  fragColor = vec4(a.rgb / clamp(a.a, 1e-5, 5e4), 1.0 - r);
}
