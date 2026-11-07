// JFA flood pass with step uStep (src/jfa.js): keep the nearest seed among the 9 pixels at offsets
// {-k, 0, k}^2. Seeds are pixel coordinates; -1 means none yet.
precision highp float;

uniform sampler2D uSeeds;
uniform float uStep;

out vec4 fragColor;

void main() {
  ivec2 n = textureSize(uSeeds, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  int k = int(uStep);
  vec2 best = vec2(-1.0);
  float bestD = 1e20;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      ivec2 q = p + ivec2(i, j) * k;
      if (q.x < 0 || q.y < 0 || q.x >= n.x || q.y >= n.y) continue;
      vec2 s = texelFetch(uSeeds, q, 0).xy;
      if (s.x < 0.0) continue;
      vec2 d = s - vec2(p);
      float dd = dot(d, d);
      if (dd < bestD) {
        bestD = dd;
        best = s;
      }
    }
  }
  fragColor = vec4(best, 0.0, 1.0);
}
