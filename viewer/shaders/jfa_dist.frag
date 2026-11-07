// JFA distance pass (src/jfa.js): signed distance in mm to the nearest seed, + on copper, - off it;
// -1e4 when no seed exists (no copper edge on the board).
precision highp float;

uniform sampler2D uSeeds;
uniform sampler2D uMask;
uniform float uMmPerPx;

out vec4 fragColor;

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 s = texelFetch(uSeeds, p, 0).xy;
  float sgn = texelFetch(uMask, p, 0).r > 0.5 ? 1.0 : -1.0;
  float d = s.x < 0.0 ? 1e4 : length(s - vec2(p)) * uMmPerPx;
  fragColor = vec4(sgn * d, 0.0, 0.0, 1.0);
}
