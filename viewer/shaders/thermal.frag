// Temperature rise -> colour through the 1D LUT, with anti-aliased isolines (CLAUDE.md §13 T1.3).
//
// uMode 0 (layer plane): theta (K) is read from uField, the layer's R16F texture.
// uMode 1 (colour bar): theta = uMin + vUv.x * uSpan, so the bar runs through the same mapping
// and shows the same isolines as tick marks.
// t = (theta - uMin) / uSpan picks the LUT entry; uMin and uSpan come from the whole 4-layer
// field, so all layers share one scale; uSpan == 0 (constant field) maps to the coldest entry.
// The LUT holds sRGB-encoded values and is written out unchanged (see src/lut.js).
//
// Isolines every uIsoStep K (uIsoStep <= 0: none). f = theta / uIsoStep is the level coordinate;
// the distance to the nearest integer level, divided by fwidth(f) (the change of f per pixel), is
// a distance in screen pixels, so lines are uIsoWidth px wide with a 1 px smooth edge at any
// distance or angle. Where levels crowd closer than about 4 px (far away, grazing angles, steep
// gradients) the lines fade out instead of aliasing into moiré.
precision highp float;

uniform sampler2D uField;
uniform sampler2D uLut;
uniform float uMin;
uniform float uSpan;
uniform float uIsoStep;
uniform float uIsoWidth;
uniform vec3 uIsoColor;
uniform float uIsoAlpha;
uniform int uMode;

in vec2 vUv;

out vec4 fragColor;

void main() {
  float theta = uMode == 1 ? uMin + vUv.x * uSpan : texture(uField, vUv).r;
  float t = uSpan > 0.0 ? clamp((theta - uMin) / uSpan, 0.0, 1.0) : 0.0;
  vec3 colour = texture(uLut, vec2((t * 255.0 + 0.5) / 256.0, 0.5)).rgb;

  if (uIsoStep > 0.0) {
    float f = theta / uIsoStep;
    float w = max(fwidth(f), 1e-6);           // levels per pixel
    float d = abs(fract(f + 0.5) - 0.5) / w;  // pixels to the nearest level
    float line = clamp(0.5 * uIsoWidth + 0.5 - d, 0.0, 1.0);
    float crowd = clamp(2.0 - 4.0 * w, 0.0, 1.0);  // 1 at >= 4 px between levels, 0 at <= 2 px
    colour = mix(colour, uIsoColor, line * crowd * uIsoAlpha);
  }
  fragColor = vec4(colour, 1.0);
}
