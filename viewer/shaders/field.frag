// Temperature rise -> colour: t = (theta - theta_min) / (theta_max - theta_min), through the 1D LUT.
// uField is the R16F layer texture (theta in K); uMin and uSpan = theta_max - theta_min come from the
// whole 4-layer field, so all layers share one colour scale. uSpan == 0 (a constant field) maps
// to the coldest entry. The LUT holds sRGB-encoded values and is written out unchanged.
precision highp float;

uniform sampler2D uField;
uniform sampler2D uLut;
uniform float uMin;
uniform float uSpan;

in vec2 vUv;

out vec4 fragColor;

void main() {
  float theta = texture(uField, vUv).r;
  float t = uSpan > 0.0 ? clamp((theta - uMin) / uSpan, 0.0, 1.0) : 0.0;
  fragColor = vec4(texture(uLut, vec2((t * 255.0 + 0.5) / 256.0, 0.5)).rgb, 1.0);
}
