// Inferno colour lookup table (CLAUDE.md §13 T1.3: inferno is the default LUT).
//
// The 256 entries are matplotlib's "inferno" (CC0, van der Walt & Smith), exported from
// matplotlib 3.10.9 and rounded to 8-bit sRGB:
//   python -c "from matplotlib import colormaps; c = colormaps['inferno'];
//     print(''.join('%02x%02x%02x' % tuple(round(255*v) for v in c(i)[:3]) for i in range(256)))"
// The values are sRGB-encoded, and the texture is tagged NoColorSpace so three never converts
// them. field.frag writes them straight out, which is correct for both the canvas and the
// WebXR framebuffer.

import { ClampToEdgeWrapping, DataTexture, LinearFilter, NoColorSpace, RGBAFormat, UnsignedByteType } from "three";

export const LUT_SIZE = 256;

/** 256 x RRGGBB, entry 0 = coldest. */
export const INFERNO_HEX =
  "00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a0722" +
  "0b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48" +
  "210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b0964" +
  "3d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d" +
  "57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e" +
  "71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a" +
  "8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62" +
  "a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655" +
  "bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545" +
  "d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933" +
  "e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711f" +
  "f1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0a" +
  "f98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0f" +
  "fcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932" +
  "f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865" +
  "f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4";

/** Decode a 256-entry RRGGBB string into RGBA8 bytes (alpha 255); throws on a malformed table. */
export function lutBytes(hex = INFERNO_HEX) {
  if (hex.length !== 6 * LUT_SIZE || !/^[0-9a-f]+$/.test(hex)) {
    throw new RangeError(`LUT must be ${LUT_SIZE} RRGGBB entries, got ${hex.length} hex chars`);
  }
  const out = new Uint8Array(4 * LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    for (let c = 0; c < 3; c++) out[4 * i + c] = parseInt(hex.slice(6 * i + 2 * c, 6 * i + 2 * c + 2), 16);
    out[4 * i + 3] = 255;
  }
  return out;
}

/** 256 x 1 RGBA8 texture of the LUT. The shader samples texel centres: u = (t * 255 + 0.5) / 256. */
export function createLutTexture(hex = INFERNO_HEX) {
  const t = new DataTexture(lutBytes(hex), LUT_SIZE, 1, RGBAFormat, UnsignedByteType);
  t.colorSpace = NoColorSpace;
  t.minFilter = LinearFilter;
  t.magFilter = LinearFilter;
  t.wrapS = ClampToEdgeWrapping;
  t.wrapT = ClampToEdgeWrapping;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}
