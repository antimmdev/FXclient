// Ultra HD map renderer.
//
// The game stores the map as a small pixel grid (e.g. 510x510) and stretches it onto the
// screen, so zooming in shows big blocks. This module replaces that with a WebGL2 renderer:
//
//  1. "Bake": once per map, a realistic high resolution terrain texture (up to ~4096px) is
//     generated on the GPU from the game's terrain colors: smooth signed-distance-field
//     coastlines, fractal relief with hillshading, natural vegetation/rock/snow/sand colors,
//     depth-shaded water with foam and a sandy shelf.
//  2. "Composite": every frame, the visible part of the baked terrain is drawn at full canvas
//     resolution, and the territory layer is decoded from the game's own pixel data and drawn
//     with smooth, anti-aliased borders instead of square pixels.
//
// If WebGL2 is unavailable or anything fails, drawMap() returns false and the game falls back
// to its original rendering.

window.__fx = window.__fx || {};
const getSettings = () => window.__fx.settings || {};

const VERT = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const NOISE = `
uint fxHash(uvec2 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * 1664525u; v.y += v.x * 1664525u;
  v ^= v >> 16u;
  v.x += v.y * 1664525u; v.y += v.x * 1664525u;
  v ^= v >> 16u;
  return v.x ^ v.y;
}
vec2 grad(ivec2 i) {
  float a = float(fxHash(uvec2(i) + uvec2(7919u, 104729u))) * (6.28318530718 / 4294967296.0);
  return vec2(cos(a), sin(a));
}
float gnoise(vec2 p) {
  vec2 fl = floor(p);
  ivec2 i = ivec2(fl);
  vec2 f = p - fl;
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a = dot(grad(i), f);
  float b = dot(grad(i + ivec2(1, 0)), f - vec2(1.0, 0.0));
  float c = dot(grad(i + ivec2(0, 1)), f - vec2(0.0, 1.0));
  float d = dot(grad(i + ivec2(1, 1)), f - vec2(1.0, 1.0));
  return 1.4 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
const mat2 ROT = mat2(0.8, -0.6, 0.6, 0.8);
float fbm(vec2 p, int oct) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 8; i++) {
    if (i >= oct) break;
    s += a * gnoise(p);
    p = ROT * p * 2.03 + vec2(17.1, 9.7);
    a *= 0.5;
  }
  return s;
}
float ridged(vec2 p, int oct) {
  float s = 0.0, a = 0.5, w = 1.0;
  for (int i = 0; i < 8; i++) {
    if (i >= oct) break;
    float n = 1.0 - abs(gnoise(p));
    n *= n * w;
    w = clamp(n * 1.5, 0.0, 1.0);
    s += a * n;
    p = ROT * p * 2.07 + vec2(3.3, 21.4);
    a *= 0.5;
  }
  return s;
}
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
`;

const BAKE_FRAG = `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform sampler2D uLandCol;
uniform sampler2D uWaterCol;
uniform sampler2D uSdf;
uniform vec2 uSize;     // map size in texels
uniform vec2 uBake;     // bake texture size in pixels
uniform float uSeed;
${NOISE}

vec4 cubic(float v) {
  vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
  vec4 s = n * n * n;
  float x = s.x, y = s.y - 4.0 * s.x, z = s.z - 4.0 * s.y + 6.0 * s.x;
  return vec4(x, y, z, 6.0 - x - y - z) * (1.0 / 6.0);
}
// smooth B-spline bicubic sampling (p in texels)
vec4 bicubic(sampler2D s, vec2 p) {
  p -= 0.5;
  vec2 f = fract(p);
  p -= f;
  vec4 xc = cubic(f.x), yc = cubic(f.y);
  vec4 c = p.xxyy + vec2(-0.5, 1.5).xyxy;
  vec4 s4 = vec4(xc.xz + xc.yw, yc.xz + yc.yw);
  vec4 o = (c + vec4(xc.yw, yc.yw) / s4) / uSize.xxyy;
  vec4 s0 = texture(s, o.xz), s1 = texture(s, o.yz), s2 = texture(s, o.xw), s3 = texture(s, o.yw);
  float sx = s4.x / (s4.x + s4.y), sy = s4.z / (s4.z + s4.w);
  return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
}
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x);
}

// signed distance to the coast in texels (land > 0, water < 0)
float sdf(vec2 p) {
  vec4 s = texture(uSdf, p / uSize);
  float fine = (s.r * 255.0 - 128.0) / 4.0;
  float big = (s.g * 255.0 - 128.0) * 2.0;
  return abs(fine) < 31.0 ? fine : big;
}
float coastNoise(vec2 p) { return fbm(p * 0.7 + uSeed, 4) * 0.55 + gnoise(p * 3.1) * 0.12; }

// terrain height in "texel" units
float height(vec2 p, float rocky) {
  float d = sdf(p);
  float inland = max(d, 0.0);
  float base = 1.0 - exp(-inland / 14.0);
  float hills = fbm(p * 0.045 + uSeed, 7);
  float mountains = ridged(p * 0.03 + 3.7 + uSeed, 6);
  float relief = texture(uSdf, p / uSize).b;
  float h = base * (1.6 + 2.2 * hills + (1.2 + 3.5 * rocky) * mountains * smoothstep(0.0, 1.0, base));
  h += relief * 2.0 * base;
  h += gnoise(p * 2.3) * 0.06 + gnoise(p * 5.1) * 0.025;
  return h;
}

void main() {
  vec2 frag = gl_FragCoord.xy; // window coordinates already include the tile viewport offset
  vec2 p = frag / uBake * uSize;
  float px = uSize.x / uBake.x; // texels per bake pixel

  vec3 landC = bicubic(uLandCol, p).rgb;
  vec3 waterC = bicubic(uWaterCol, p).rgb;
  vec3 hsv = rgb2hsv(landC);
  float rocky = clamp(1.0 - hsv.y * 2.2, 0.0, 1.0);

  float d = sdf(p) + coastNoise(p);
  float land = smoothstep(-0.6 * px, 0.6 * px, d);

  vec3 L = normalize(vec3(-0.55, -0.7, 0.85));
  vec3 col;

  // ---------- land ----------
  vec3 landCol = vec3(0.0);
  if (d > -2.0) {
    float e = max(px, 0.08);
    float h0 = height(p, rocky);
    float hx = height(p + vec2(e, 0.0), rocky);
    float hy = height(p + vec2(0.0, e), rocky);
    vec3 N = normalize(vec3(-(hx - h0) / e, -(hy - h0) / e, 1.0));
    float slope = 1.0 - N.z;

    // natural color grading of the game's palette
    float l = luma(landC);
    vec3 c = mix(vec3(l), landC, 0.72);
    float green = smoothstep(0.13, 0.2, hsv.x) * (1.0 - smoothstep(0.45, 0.55, hsv.x)) * smoothstep(0.12, 0.3, hsv.y);
    vec3 grass = vec3(0.34, 0.45, 0.19);
    vec3 forest = vec3(0.11, 0.22, 0.08);
    vec3 dry = vec3(0.56, 0.52, 0.31);
    float dens = clamp(1.15 - hsv.z * 1.25 + fbm(p * 0.22 + 4.0, 4) * 0.8, 0.0, 1.0);
    vec3 veg = mix(grass, forest, smoothstep(0.3, 0.75, dens));
    veg = mix(veg, dry, smoothstep(0.15, 0.6, fbm(p * 0.06 + 9.0, 3)) * 0.4);
    veg *= mix(0.8, 1.25, smoothstep(0.15, 0.75, hsv.z));
    c = mix(c, veg * vec3(1.0, 1.06, 0.95), green * 0.72);
    // tree canopy texture
    float canopy = gnoise(p * 2.7) * 0.55 + gnoise(p * 6.3) * 0.3 + gnoise(p * 13.0) * 0.15;
    c *= 1.0 + green * (0.22 * canopy - 0.12 * smoothstep(0.1, 0.6, gnoise(p * 4.1 + 2.0)) * dens);
    // generic multi-scale ground texture
    c *= 0.9 + 0.2 * (0.5 + 0.5 * fbm(p * 0.9, 5));
    // exposed rock on steep slopes and in grey areas
    vec3 rock = vec3(0.45, 0.42, 0.38) * (0.8 + 0.4 * gnoise(p * 3.0 + 1.0));
    float rockAmt = clamp(smoothstep(0.35, 0.75, slope) * 0.8 + rocky * 0.25 * (1.0 - step(0.75, l)), 0.0, 1.0);
    c = mix(c, rock * mix(0.7, 1.25, l), rockAmt * (1.0 - smoothstep(0.75, 0.9, l)));
    // bright unsaturated areas: sand and dunes near the coast, snow high up inland
    float pale = smoothstep(0.62, 0.8, l) * rocky;
    float inlandHigh = smoothstep(10.0, 22.0, d);
    vec3 sand = vec3(0.79, 0.71, 0.52) * (0.92 + 0.16 * fbm(p * 0.5 + 7.0, 3));
    c = mix(c, sand, pale * (1.0 - inlandHigh) * 0.9);
    vec3 lime = vec3(0.64, 0.61, 0.55) * (0.78 + 0.4 * ridged(p * 0.45 + 2.0, 5)) * (0.9 + 0.2 * gnoise(p * 2.0));
    c = mix(c, lime, pale * inlandHigh * 0.85);
    float snow = smoothstep(0.86, 0.95, l) * rocky * inlandHigh;
    c = mix(c, vec3(0.93, 0.95, 0.98), snow * smoothstep(0.3, 0.7, N.z));
    // beaches
    float beach = (1.0 - smoothstep(0.15, 0.9, d + gnoise(p * 2.0) * 0.35)) * (1.0 - rockAmt) * smoothstep(0.15, 0.35, l);
    c = mix(c, vec3(0.80, 0.73, 0.55), beach * 0.85);

    // lighting
    float diff = dot(N, L) / L.z;
    float shade = clamp(diff, 0.25, 1.6);
    float cavity = clamp((h0 - (height(p + vec2(-e * 3.0, 0.0), rocky) + height(p + vec2(0.0, -e * 3.0), rocky)) * 0.5) * 0.6, -0.15, 0.15);
    vec3 sun = vec3(1.04, 1.0, 0.92);
    vec3 sky = vec3(0.55, 0.65, 0.85);
    landCol = c * (sun * shade * 0.82 + sky * 0.22) * (1.0 + cavity);
    // wet sand line right at the coast
    landCol *= 1.0 - 0.25 * (1.0 - smoothstep(0.0, 0.35, d));
  }

  // ---------- water ----------
  vec3 waterCol = vec3(0.0);
  if (d < 2.0) {
    float depth = max(-d, 0.0);
    vec3 shallow = vec3(0.13, 0.58, 0.62);
    vec3 midW = vec3(0.04, 0.33, 0.50);
    vec3 deep = vec3(0.02, 0.13, 0.29);
    vec3 w = mix(shallow, midW, smoothstep(0.0, 4.0, depth));
    w = mix(w, deep, smoothstep(3.0, 38.0, depth));
    // keep some of the map's own water tint (lakes, swamps...)
    vec3 tint = waterC / max(luma(waterC), 0.05);
    w = mix(w, w * clamp(tint, 0.6, 1.6), 0.3);
    // sandy shelf and caustics in the shallows
    float shelf = exp(-depth * 0.9);
    w = mix(w, vec3(0.55, 0.62, 0.48), shelf * 0.5);
    float caus = pow(1.0 - abs(gnoise(p * 2.2 + 3.0)), 6.0) + pow(1.0 - abs(gnoise(p * 3.7 + 9.0)), 6.0);
    w += shelf * caus * 0.05;
    // large scale swell + small waves
    w *= 0.9 + 0.16 * fbm(p * 0.035 + 2.0, 3);
    vec2 wn = vec2(gnoise(p * 1.3 + 3.0), gnoise(p * 1.3 + 11.0)) * 0.5
            + vec2(gnoise(p * 4.7 + 5.0), gnoise(p * 4.7 + 13.0)) * 0.25
            + vec2(gnoise(p * 11.0), gnoise(p * 11.0 + 7.0)) * 0.05;
    vec3 N = normalize(vec3(wn * 0.25, 1.0));
    vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(N, H), 0.0), 90.0);
    w *= 0.92 + 0.16 * dot(N, L);
    w += vec3(1.0, 0.97, 0.9) * spec * 0.35;
    // foam
    float fn = gnoise(p * 3.0) * 0.3 + gnoise(p * 7.0) * 0.15;
    float foam = 1.0 - smoothstep(0.0, 0.45, depth + fn);
    float foam2 = (1.0 - smoothstep(0.0, 0.18, abs(depth - 0.9 + fn * 0.6))) * smoothstep(-0.1, 0.4, gnoise(p * 2.5 + 4.0));
    w = mix(w, vec3(0.9, 0.95, 0.96), clamp(foam * 0.8 + foam2 * 0.35, 0.0, 1.0));
    waterCol = w;
  }

  col = mix(waterCol, landCol, land);
  // mild contrast curve
  col = clamp(col, 0.0, 1.0);
  col = mix(col, col * col * (3.0 - 2.0 * col), 0.3);
  outColor = vec4(clamp(col, 0.0, 1.0), land);
}`;

const DRAW_FRAG = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
out vec4 outColor;
uniform sampler2D uHD;
uniform usampler2D uTer;
uniform sampler2D uPal;
uniform vec2 uSize;     // map size in texels
uniform vec2 uScreen;   // canvas size in pixels
uniform vec3 uXform;    // px per texel, translate x, translate y
uniform float uBakeScale;
uniform float uUiScale; // canvas pixels per css pixel
uniform float uTime;
uniform int uHasTer;
uniform float uDim;
${NOISE}

int ownerAt(ivec2 c, out uvec4 t) {
  if (c.x < 0 || c.y < 0 || c.x >= int(uSize.x) || c.y >= int(uSize.y)) { t = uvec4(0u); return -1; }
  t = texelFetch(uTer, c, 0);
  if (t.a < 208u) return -1;
  return int(((t.r & 3u) << 7) | ((t.g & 3u) << 5) | ((t.b & 3u) << 3) | (t.a & 7u));
}

void main() {
  vec2 frag = vec2(gl_FragCoord.x, uScreen.y - gl_FragCoord.y);
  vec2 p = (frag - uXform.yz) / uXform.x;
  if (p.x < 0.0 || p.y < 0.0 || p.x > uSize.x || p.y > uSize.y) { outColor = vec4(0.0); return; }

  vec4 hd = texture(uHD, p / uSize);
  vec3 col = hd.rgb;
  float landness = hd.a;
  float zoomOverBake = uXform.x / uBakeScale; // screen pixels per baked pixel

  // procedural relief once we are zoomed in further than the baked resolution
  float detail = smoothstep(1.0, 3.0, zoomOverBake);
  if (detail > 0.0) {
    vec3 L = normalize(vec3(-0.55, -0.7, 0.85));
    float land = smoothstep(0.3, 0.7, landness);
    const float S = 0.12;
    if (land > 0.0) {
      // ground grain: rock, soil and vegetation bumps lit by the sun
      vec2 q = p * 4.5;
      float h0 = fbm(q, 5), hx = fbm(q + vec2(S, 0.0), 5), hy = fbm(q + vec2(0.0, S), 5);
      vec3 N = normalize(vec3(-(hx - h0) / S * 0.32, -(hy - h0) / S * 0.32, 1.0));
      float shade = clamp(dot(N, L) / L.z, 0.55, 1.4);
      float grain = gnoise(p * 31.0) * 0.6 + gnoise(p * 67.0) * 0.4;
      col *= mix(1.0, shade * (1.0 + 0.07 * grain), detail * land);
    }
    if (land < 1.0) {
      // animated waves
      vec2 q = p * 2.4;
      vec2 t1 = vec2(0.31, 0.17) * uTime, t2 = vec2(-0.22, 0.29) * uTime;
      float h0 = gnoise(q + t1) * 0.6 + gnoise(q * 2.3 + t2) * 0.4;
      float hx = gnoise(q + vec2(S, 0.0) + t1) * 0.6 + gnoise((q + vec2(S, 0.0)) * 2.3 + t2) * 0.4;
      float hy = gnoise(q + vec2(0.0, S) + t1) * 0.6 + gnoise((q + vec2(0.0, S)) * 2.3 + t2) * 0.4;
      vec3 N = normalize(vec3(-(hx - h0) / S * 0.18, -(hy - h0) / S * 0.18, 1.0));
      vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
      float spec = pow(max(dot(N, H), 0.0), 60.0);
      vec3 w = col * (0.9 + 0.2 * dot(N, L) / L.z) + vec3(1.0, 0.97, 0.9) * spec * 0.25;
      col = mix(col, w, detail * (1.0 - land));
      // break the (magnified) baked foam up into lacy bubbles
      float foamAmt = smoothstep(0.5, 0.8, luma(col)) * (1.0 - land) * detail;
      float lace = gnoise(p * 9.0 + t1 * 0.5) + 0.5 * gnoise(p * 23.0 - t2 * 0.5);
      col = mix(col, col * vec3(0.45, 0.72, 0.8), foamAmt * (1.0 - smoothstep(-0.25, 0.25, lace)) * 0.75);
    }
  }
  // animated glitter on open water
  if (landness < 0.5) {
    vec2 q = p * 1.6;
    float g = gnoise(q + vec2(uTime * 0.11, uTime * 0.07)) + gnoise(q * 2.3 - vec2(uTime * 0.09, -uTime * 0.13));
    float glint = smoothstep(1.05, 1.45, g) * (1.0 - landness);
    col += vec3(0.9, 0.95, 1.0) * glint * 0.12;
    col *= 1.0 + 0.03 * gnoise(p * 0.5 + uTime * 0.05);
  }

  if (uHasTer == 1) {
    float z = uXform.x;
    if (z < 1.6) {
      // zoomed out: plain per-texel colors (like the game), no smoothing needed
      uvec4 t;
      int id = ownerAt(ivec2(floor(p)), t);
      if (id >= 0) {
        vec3 c = vec3(t.rgb) / 255.0;
        if (t.a < 224u) c = texelFetch(uPal, ivec2(id, 0), 0).rgb;
        vec3 tinted = c * (0.55 + 0.75 * luma(col));
        col = mix(col, mix(c, tinted, 0.45), (t.a < 224u ? 0.66 : 0.9) * smoothstep(0.2, 0.6, landness));
      }
    } else {
      // smooth territory: quadratic B-spline coverage over a 3x3 texel neighbourhood
      vec2 q = p - 0.5;
      vec2 fl = floor(q + 0.5);
      vec2 f = q - fl + 0.5; // 0..1
      vec3 wx = vec3(0.5 * (1.0 - f.x) * (1.0 - f.x), 0.75 - (f.x - 0.5) * (f.x - 0.5), 0.5 * f.x * f.x);
      vec3 wy = vec3(0.5 * (1.0 - f.y) * (1.0 - f.y), 0.75 - (f.y - 0.5) * (f.y - 0.5), 0.5 * f.y * f.y);
      ivec2 b = ivec2(fl) - 1;
      int ids[9]; uvec4 ts[9]; float ws[9];
      for (int j = 0; j < 3; j++) for (int i = 0; i < 3; i++) {
        int k = j * 3 + i;
        ids[k] = ownerAt(b + ivec2(i, j), ts[k]);
        ws[k] = wx[i] * wy[j];
      }
      int best = -1; float bestCov = 0.0;
      for (int k = 0; k < 9; k++) {
        if (ids[k] < 0 || ids[k] == best) continue;
        float cv = 0.0;
        for (int m = 0; m < 9; m++) if (ids[m] == ids[k]) cv += ws[m];
        if (cv > bestCov) { bestCov = cv; best = ids[k]; }
      }
      if (best >= 0) {
        vec3 fill = texelFetch(uPal, ivec2(best, 0), 0).rgb;
        vec3 border = vec3(0.0); float bw = 0.0;
        for (int k = 0; k < 9; k++) if (ids[k] == best && ts[k].a >= 224u) { border += vec3(ts[k].rgb) / 255.0 * ws[k]; bw += ws[k]; }
        border = bw > 0.0 ? border / bw : fill * 0.6;
        float fw = max(fwidth(bestCov), 1e-4);
        float alpha = smoothstep(0.5 - fw * 0.75, 0.5 + fw * 0.75, bestCov);
        // territory ends at the (smooth) coastline
        float lw = max(fwidth(landness), 0.01);
        alpha *= smoothstep(0.5 - lw, 0.5 + lw, landness);
        float distPx = (bestCov - 0.5) / fw;               // distance inside the region, in pixels
        float bwPx = clamp(z * 0.16, 1.1 * uUiScale, 4.5 * uUiScale);
        float borderAmt = 1.0 - smoothstep(bwPx - 0.75, bwPx + 0.75, distPx);
        // fill tinted by terrain lighting so the relief stays visible
        vec3 tinted = fill * (0.5 + 0.85 * luma(col));
        vec3 terr = mix(col, mix(fill, tinted, 0.5), 0.62);
        // slightly stronger tint near the edge, like printed political maps
        float glow = 1.0 - smoothstep(bwPx, bwPx + 9.0 * uUiScale + z * 0.6, distPx);
        terr = mix(terr, fill, glow * 0.18);
        vec3 c = mix(terr, border * (0.85 + 0.25 * luma(col)), borderAmt * 0.95);
        col = mix(col, c, alpha);
      }
    }
  }
  col *= uDim;
  outColor = vec4(col, 1.0);
}`;

// ---------- CPU helpers ----------

// 1D squared Euclidean distance transform (Felzenszwalb & Huttenlocher)
function edt1d(f, n, d, v, z) {
  let k = 0;
  v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s;
    while (true) {
      const r = v[k];
      s = ((f[q] + q * q) - (f[r] + r * r)) / (2 * q - 2 * r);
      if (s <= z[k]) { k--; if (k < 0) { k = 0; break; } } else break;
    }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const r = v[k];
    d[q] = (q - r) * (q - r) + f[r];
  }
}
// squared distance from every pixel to the nearest pixel where mask[i] === target
function edt(mask, W, H, target) {
  const INF = 1e20;
  const out = new Float32Array(W * H);
  const n = Math.max(W, H);
  const f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) f[y] = mask[y * W + x] === target ? 0 : INF;
    edt1d(f, H, d, v, z);
    for (let y = 0; y < H; y++) out[y * W + x] = d[y];
  }
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) f[x] = out[row + x];
    edt1d(f, W, d, v, z);
    for (let x = 0; x < W; x++) out[row + x] = d[x];
  }
  return out;
}
// fill pixels that are not of the given class with the average of neighbouring filled pixels
function dilateColors(src, mask, W, H, keep, passes) {
  const col = new Uint8Array(src);
  let filled = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) filled[i] = mask[i] === keep ? 1 : 0;
  for (let pass = 0; pass < passes; pass++) {
    const next = filled.slice();
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (filled[i]) continue;
      let r = 0, g = 0, b = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const j = yy * W + xx;
        if (!filled[j]) continue;
        r += col[j * 4]; g += col[j * 4 + 1]; b += col[j * 4 + 2]; n++;
      }
      if (n) { col[i * 4] = r / n; col[i * 4 + 1] = g / n; col[i * 4 + 2] = b / n; next[i] = 1; }
    }
    filled = next;
  }
  // anything still unfilled (e.g. a map without any water) gets a neutral fallback
  for (let i = 0; i < W * H; i++) if (!filled[i]) {
    if (keep === 1) { col[i * 4] = 70; col[i * 4 + 1] = 120; col[i * 4 + 2] = 60; }
    else { col[i * 4] = 20; col[i * 4 + 1] = 70; col[i * 4 + 2] = 120; }
  }
  return col;
}
function boxBlur(src, W, H, r) {
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    let acc = 0, n = 0;
    for (let x = -r; x < W + r; x++) {
      if (x + r < W) { acc += src[y * W + x + r] ?? 0; n++; }
      if (x - r - 1 >= 0) { acc -= src[y * W + x - r - 1]; n--; }
      if (x >= 0 && x < W) tmp[y * W + x] = acc / n;
    }
  }
  for (let x = 0; x < W; x++) {
    let acc = 0, n = 0;
    for (let y = -r; y < H + r; y++) {
      if (y + r < H) { acc += tmp[(y + r) * W + x]; n++; }
      if (y - r - 1 >= 0) { acc -= tmp[(y - r - 1) * W + x]; n--; }
      if (y >= 0 && y < H) out[y * W + x] = acc / n;
    }
  }
  return out;
}

// ---------- renderer ----------

const hd = {
  failed: false,
  gl: null,
  canvas: null,
  progs: null,
  terrains: new Map(), // terrain canvas -> baked state (a small LRU cache)
  territory: { canvas: null, data: null, tex: null, w: 0, h: 0, dirty: null, palTex: null, pal: new Uint8Array(512 * 4), palDirty: true },
  startTime: performance.now(),
};

function isEnabled() { return getSettings().hdMapRendering !== false; }

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("HD renderer shader error: " + gl.getShaderInfoLog(s));
  return s;
}
function program(gl, frag) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, frag));
  gl.bindAttribLocation(p, 0, "aPos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("HD renderer link error: " + gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}

function init() {
  if (hd.gl) return true;
  if (hd.failed) return false;
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: true, antialias: false, preserveDrawingBuffer: true, depth: false, stencil: false });
    if (!gl) throw new Error("WebGL2 not available");
    hd.canvas = canvas;
    hd.gl = gl;
    hd.progs = { bake: program(gl, BAKE_FRAG), draw: program(gl, DRAW_FRAG) };
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    hd.aniso = gl.getExtension("EXT_texture_filter_anisotropic");
    hd.dummyTer = makeTex(gl, 1, 1, gl.RGBA8UI, gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, new Uint8Array(4), gl.NEAREST);
    hd.dummyPal = makeTex(gl, 1, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4), gl.NEAREST);
    canvas.addEventListener("webglcontextlost", e => { e.preventDefault(); resetGL(); });
    console.log("FX HD renderer initialised");
    return true;
  } catch (e) {
    console.warn("FX HD renderer disabled:", e);
    hd.failed = true;
    hd.gl = null;
    return false;
  }
}
function resetGL() {
  hd.gl = null; hd.progs = null;
  hd.terrains.forEach(T => { T.tex = null; T.bakedVersion = -2; });
  hd.territory.tex = null; hd.territory.palTex = null; hd.territory.palDirty = true;
}

function makeTex(gl, w, h, internal, format, type, data, filter) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function getTerrain(source) {
  let T = hd.terrains.get(source);
  if (T) {
    // mark as most recently used
    hd.terrains.delete(source);
    hd.terrains.set(source, T);
    return T;
  }
  T = { source, version: 0, bakedVersion: -1, tex: null, scale: 1, w: 0, h: 0, lastBake: 0 };
  hd.terrains.set(source, T);
  // keep at most two baked terrains on the GPU
  while (hd.terrains.size > 2) {
    const [oldSource, old] = hd.terrains.entries().next().value;
    if (old.tex && hd.gl) hd.gl.deleteTexture(old.tex);
    hd.terrains.delete(oldSource);
  }
  // re-bake whenever the game draws onto the terrain canvas
  const ctx = source.getContext("2d");
  if (ctx && !ctx.__fxHdWatched) {
    ctx.__fxHdWatched = true;
    for (const name of ["putImageData", "drawImage", "fillRect"]) {
      const orig = ctx[name];
      ctx[name] = function () { const t = hd.terrains.get(source); if (t) t.version++; return orig.apply(this, arguments); };
    }
  }
  return T;
}

function bake(T) {
  const gl = hd.gl, src = T.source;
  const W = src.width, H = src.height;
  if (!W || !H) return false;
  const t0 = performance.now();
  const pixels = src.getContext("2d").getImageData(0, 0, W, H).data;
  // the game paints a 1px grey frame around the map: replace it with the neighbouring pixels
  if (W > 4 && H > 4) {
    const copy = (to, from) => { for (let c = 0; c < 4; c++) pixels[to * 4 + c] = pixels[from * 4 + c]; };
    for (let x = 0; x < W; x++) { copy(x, W + x); copy((H - 1) * W + x, (H - 2) * W + x); }
    for (let y = 0; y < H; y++) { copy(y * W, y * W + 1); copy(y * W + W - 1, y * W + W - 2); }
  }
  const N = W * H;
  const mask = new Uint8Array(N); // 1 = land
  const lum = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
    mask[i] = (b > g && b > r) ? 0 : 1;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    // relief proxy: bright and unsaturated land (rock / snow) is high
    lum[i] = mask[i] ? (mx / 255) * (1 - (mx - mn) / (mx + 1)) : 0;
  }
  const toWater = edt(mask, W, H, 0), toLand = edt(mask, W, H, 1);
  const relief = boxBlur(lum, W, H, 3);
  const sdf = new Uint8Array(N * 4);
  for (let i = 0; i < N; i++) {
    const d = mask[i] ? Math.sqrt(toWater[i]) - 0.5 : -(Math.sqrt(toLand[i]) - 0.5);
    sdf[i * 4] = Math.max(0, Math.min(255, Math.round(128 + d * 4)));
    sdf[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(128 + d * 0.5)));
    sdf[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(relief[i] * 255)));
    sdf[i * 4 + 3] = 255;
  }
  const landCol = dilateColors(pixels, mask, W, H, 1, 4);
  const waterCol = dilateColors(pixels, mask, W, H, 0, 4);

  const maxTex = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), 4096);
  const scale = Math.max(1, Math.min(8, Math.floor(maxTex / Math.max(W, H))));
  const BW = W * scale, BH = H * scale;

  const texLand = makeTex(gl, W, H, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, landCol, gl.LINEAR);
  const texWater = makeTex(gl, W, H, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, waterCol, gl.LINEAR);
  const texSdf = makeTex(gl, W, H, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, sdf, gl.LINEAR);

  if (T.tex) gl.deleteTexture(T.tex);
  const out = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, out);
  const levels = Math.floor(Math.log2(Math.max(BW, BH))) + 1;
  gl.texStorage2D(gl.TEXTURE_2D, levels, gl.RGBA8, BW, BH);
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out, 0);

  const { p, u } = hd.progs.bake;
  gl.useProgram(p);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texLand);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texWater);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, texSdf);
  gl.uniform1i(u.uLandCol, 0); gl.uniform1i(u.uWaterCol, 1); gl.uniform1i(u.uSdf, 2);
  gl.uniform2f(u.uSize, W, H);
  gl.uniform2f(u.uBake, BW, BH);
  gl.uniform1f(u.uSeed, (W * 7 + H * 13) % 97);
  gl.disable(gl.BLEND);
  // render in tiles so a single huge draw call doesn't trip the GPU watchdog
  const TILE = 1024;
  for (let y = 0; y < BH; y += TILE) for (let x = 0; x < BW; x += TILE) {
    const w = Math.min(TILE, BW - x), h = Math.min(TILE, BH - y);
    gl.viewport(x, y, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.flush();
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fb);
  gl.deleteTexture(texLand); gl.deleteTexture(texWater); gl.deleteTexture(texSdf);

  gl.bindTexture(gl.TEXTURE_2D, out);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (hd.aniso) gl.texParameterf(gl.TEXTURE_2D, hd.aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, gl.getParameter(hd.aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));

  T.tex = out; T.scale = scale; T.w = W; T.h = H;
  T.bakedVersion = T.version;
  T.lastBake = performance.now();
  console.log(`FX HD terrain baked: ${W}x${H} -> ${BW}x${BH} in ${Math.round(T.lastBake - t0)} ms`);
  return true;
}

// called by the game whenever the territory layer is (re)created
hd.setTerritory = function (canvas, imageData) {
  const R = hd.territory;
  R.canvas = canvas;
  R.data = imageData;
  R.dirty = { x0: 0, y0: 0, x1: imageData.width, y1: imageData.height };
  R.pal.fill(0);
  R.palDirty = true;
  const ctx = canvas.getContext("2d");
  if (ctx && !ctx.__fxHdWatched) {
    ctx.__fxHdWatched = true;
    const orig = ctx.putImageData;
    ctx.putImageData = function (img, dx, dy, sx, sy, sw, sh) {
      const res = orig.apply(this, arguments);
      if (img !== R.data) { R.data = img; R.dirty = { x0: 0, y0: 0, x1: img.width, y1: img.height }; return res; }
      if (sx === undefined) { sx = 0; sy = 0; sw = img.width; sh = img.height; }
      const x0 = Math.max(0, dx + sx), y0 = Math.max(0, dy + sy);
      const x1 = Math.min(img.width, dx + sx + sw), y1 = Math.min(img.height, dy + sy + sh);
      if (x1 <= x0 || y1 <= y0) return res;
      const d = R.dirty;
      R.dirty = d ? { x0: Math.min(d.x0, x0), y0: Math.min(d.y0, y0), x1: Math.max(d.x1, x1), y1: Math.max(d.y1, y1) } : { x0, y0, x1, y1 };
      return res;
    };
  }
};

function uploadTerritory() {
  const gl = hd.gl, R = hd.territory, img = R.data;
  if (!img) return false;
  const W = img.width, H = img.height;
  if (!R.tex || R.w !== W || R.h !== H) {
    if (R.tex) gl.deleteTexture(R.tex);
    R.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, R.tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8UI, W, H);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    R.w = W; R.h = H;
    R.dirty = { x0: 0, y0: 0, x1: W, y1: H };
  }
  if (!R.palTex) {
    R.palTex = makeTex(gl, 512, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, null, gl.NEAREST);
    R.palDirty = true;
  }
  const d = R.dirty;
  if (d) {
    const data = img.data;
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    gl.bindTexture(gl.TEXTURE_2D, R.tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, W);
    gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, d.x0);
    gl.pixelStorei(gl.UNPACK_SKIP_ROWS, d.y0);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0, gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, bytes);
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
    // learn each player's fill color from interior pixels
    const pal = R.pal;
    for (let y = d.y0; y < d.y1; y++) {
      let i = (y * W + d.x0) * 4;
      for (let x = d.x0; x < d.x1; x++, i += 4) {
        const a = bytes[i + 3];
        if (a < 208) continue;
        const r = bytes[i], g = bytes[i + 1], b = bytes[i + 2];
        const id = ((r & 3) << 7) | ((g & 3) << 5) | ((b & 3) << 3) | (a & 7);
        const o = id * 4;
        if (a < 224) {
          if (pal[o] !== r || pal[o + 1] !== g || pal[o + 2] !== b || pal[o + 3] !== 255) {
            pal[o] = r; pal[o + 1] = g; pal[o + 2] = b; pal[o + 3] = 255; R.palDirty = true;
          }
        } else if (pal[o + 3] !== 255) {
          // only border pixels seen so far: estimate the fill as a lighter border color
          const lr = Math.min(255, r * 1.35 + 20), lg = Math.min(255, g * 1.35 + 20), lb = Math.min(255, b * 1.35 + 20);
          if (pal[o] !== lr) { pal[o] = lr; pal[o + 1] = lg; pal[o + 2] = lb; pal[o + 3] = 128; R.palDirty = true; }
        }
      }
    }
    R.dirty = null;
  }
  if (R.palDirty) {
    gl.bindTexture(gl.TEXTURE_2D, R.palTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 512, 1, gl.RGBA, gl.UNSIGNED_BYTE, R.pal);
    R.palDirty = false;
  }
  return true;
}

/**
 * Draws the map onto `ctx` so that map texel (x, y) lands on canvas pixel
 * (scale * x + tx, scale * y + ty). Returns false if the caller should draw the map itself.
 * @param {CanvasRenderingContext2D} ctx
 * @param {HTMLCanvasElement} terrain
 * @param {HTMLCanvasElement | null} territory
 */
hd.drawMap = function (ctx, scale, tx, ty, terrain, territory, dim = 1) {
  if (!isEnabled() || !terrain || !terrain.width || !init()) return false;
  try {
    const gl = hd.gl, T = getTerrain(terrain);
    if (T.bakedVersion !== T.version) {
      // the terrain canvas changed; rebake at most twice per second
      if (T.tex === null || performance.now() - T.lastBake > 500 || T.w !== terrain.width || T.h !== terrain.height) bake(T);
    }
    if (!T.tex || T.w !== terrain.width || T.h !== terrain.height) return false;

    const R = hd.territory;
    const useTer = !!territory && R.canvas === territory && !!R.data && R.data.width === T.w && R.data.height === T.h && uploadTerritory();

    const cw = ctx.canvas.width, ch = ctx.canvas.height;
    if (hd.canvas.width !== cw || hd.canvas.height !== ch) { hd.canvas.width = cw; hd.canvas.height = ch; }
    gl.viewport(0, 0, cw, ch);
    gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const { p, u } = hd.progs.draw;
    gl.useProgram(p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.tex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, useTer ? R.tex : hd.dummyTer);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, useTer ? R.palTex : hd.dummyPal);
    gl.uniform1i(u.uHD, 0); gl.uniform1i(u.uTer, 1); gl.uniform1i(u.uPal, 2);
    gl.uniform2f(u.uSize, T.w, T.h);
    gl.uniform2f(u.uScreen, cw, ch);
    gl.uniform3f(u.uXform, scale, tx, ty);
    gl.uniform1f(u.uBakeScale, T.scale);
    gl.uniform1f(u.uUiScale, cw / Math.max(1, ctx.canvas.clientWidth || cw));
    gl.uniform1f(u.uTime, (performance.now() - hd.startTime) / 1000);
    gl.uniform1i(u.uHasTer, useTer ? 1 : 0);
    gl.uniform1f(u.uDim, dim);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(hd.canvas, 0, 0);
    ctx.restore();
    // territory data wasn't available: let the game's own territory layer show on top
    if (territory && !useTer) {
      ctx.save();
      ctx.setTransform(scale, 0, 0, scale, tx, ty);
      ctx.imageSmoothingEnabled = scale < 3;
      ctx.drawImage(territory, 0, 0);
      ctx.restore();
    }
    return true;
  } catch (e) {
    console.error("FX HD renderer error, falling back to default rendering", e);
    hd.failed = true;
    return false;
  }
};

/** Multiplier for the game's canvas resolution. */
hd.adjustResolution = function (scale, cssWidth, cssHeight) {
  const mode = getSettings().renderResolution || "4k";
  const dpr = window.devicePixelRatio || 1;
  if (mode === "native") return 1;
  if (mode === "4k") {
    const target = Math.sqrt((3840 * 2160) / Math.max(1, cssWidth * cssHeight));
    const total = Math.max(dpr, Math.min(target, 3));
    return total / dpr;
  }
  return scale;
};

window.__fx.hd = hd;
export default hd;
