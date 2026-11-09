// Bevel normal pass (src/jfa.js): height h = smoothstep(0, w, d) inside the copper (0 outside), w the
// bevel width in mm; tangent-space normal by central differences, n = normalize(-H dh/dx, -H dh/dy, 1),
// H the bevel height in mm; stored as RGB = n / 2 + 1 / 2.
precision highp float;

uniform sampler2D uSdf;
uniform float uMmPerPx;
uniform float uBevelMm;
uniform float uHeightMm;

out vec4 fragColor;

float h(ivec2 p) {
  ivec2 n = textureSize(uSdf, 0);
  p = clamp(p, ivec2(0), n - 1);
  float d = texelFetch(uSdf, p, 0).r;
  return d <= 0.0 ? 0.0 : smoothstep(0.0, uBevelMm, d);
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float gx = uHeightMm * (h(p + ivec2(1, 0)) - h(p - ivec2(1, 0))) / (2.0 * uMmPerPx);
  float gy = uHeightMm * (h(p + ivec2(0, 1)) - h(p - ivec2(0, 1))) / (2.0 * uMmPerPx);
  vec3 nrm = normalize(vec3(-gx, -gy, 1.0));
  fragColor = vec4(nrm * 0.5 + 0.5, 1.0);
}
