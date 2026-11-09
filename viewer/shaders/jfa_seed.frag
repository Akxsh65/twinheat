// JFA seed pass (src/jfa.js): a pixel whose copper mask differs from one of its 4 neighbours (outside
// the grid counts as bare board) is a seed and stores its own pixel coordinates; others store -1.
precision highp float;

uniform sampler2D uMask;

out vec4 fragColor;

float mask(ivec2 p, ivec2 n) {
  if (p.x < 0 || p.y < 0 || p.x >= n.x || p.y >= n.y) return 0.0;
  return texelFetch(uMask, p, 0).r > 0.5 ? 1.0 : 0.0;
}

void main() {
  ivec2 n = textureSize(uMask, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  float m = mask(p, n);
  bool edge = mask(p + ivec2(1, 0), n) != m || mask(p - ivec2(1, 0), n) != m
           || mask(p + ivec2(0, 1), n) != m || mask(p - ivec2(0, 1), n) != m;
  fragColor = edge ? vec4(vec2(p), 0.0, 1.0) : vec4(-1.0, -1.0, 0.0, 1.0);
}
