// Weighted blended OIT, passes 2 and 3 (src/oit.js). uMode 0 (accumulation, RGBA16F, blend ONE, ONE):
// out = (C * a * w, a * w). uMode 1 (revealage, R8, blend ZERO, ONE_MINUS_SRC_COLOR): out = a, so
// the target holds prod(1 - a). Depth test by hand against the opaque pass's depth texture.
// w = a * clamp(10 / (1e-5 + (|z| / 5)^3), 1e-2, 3e3), z = view depth in m (src/oit.js oitWeight).
precision highp float;

uniform sampler2D uDepth;
uniform vec2 uResolution;
uniform vec3 uColor;
uniform float uAlpha;
uniform int uMode;

in float vViewZ;

out vec4 fragColor;

void main() {
  float opaqueDepth = texture(uDepth, gl_FragCoord.xy / uResolution).r;
  if (gl_FragCoord.z > opaqueDepth) discard;  // hidden behind an opaque surface
  float w = uAlpha * clamp(10.0 / (1e-5 + pow(abs(vViewZ) / 5.0, 3.0)), 1e-2, 3e3);
  fragColor = uMode == 0 ? vec4(uColor * uAlpha * w, uAlpha * w) : vec4(uAlpha);
}
