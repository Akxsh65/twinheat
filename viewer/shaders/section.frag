// Through-thickness section (src/cutPlane.js). Board uv of this fragment: along the cut and at the
// cut position (uAxis 0: plane runs along x at north fraction uCut; 1: along y at east fraction
// uCut; the +90 deg turn about y sends the plane's +x to world -z = north, so vUv.x runs south to north). Layer coordinate s = 3 (1 - vUv.y): 0 at the top (L1),
// 3 at the bottom (L4). Value = sum over layers of max(0, 1 - |s - l|) * theta_l: linear in z,
// exact at the layer planes. Then the same LUT mapping as shaders/thermal.frag.
precision highp float;

uniform sampler2D uField0;
uniform sampler2D uField1;
uniform sampler2D uField2;
uniform sampler2D uField3;
uniform sampler2D uLut;
uniform float uMin;
uniform float uSpan;
uniform float uCut;
uniform int uAxis;

in vec2 vUv;

out vec4 fragColor;

void main() {
  vec2 board = uAxis == 0 ? vec2(vUv.x, uCut) : vec2(uCut, vUv.x);
  float s = 3.0 * (1.0 - vUv.y);
  float theta = max(0.0, 1.0 - abs(s - 0.0)) * texture(uField0, board).r
              + max(0.0, 1.0 - abs(s - 1.0)) * texture(uField1, board).r
              + max(0.0, 1.0 - abs(s - 2.0)) * texture(uField2, board).r
              + max(0.0, 1.0 - abs(s - 3.0)) * texture(uField3, board).r;
  float t = uSpan > 0.0 ? clamp((theta - uMin) / uSpan, 0.0, 1.0) : 0.0;
  fragColor = vec4(texture(uLut, vec2((t * 255.0 + 0.5) / 256.0, 0.5)).rgb, 1.0);
}
