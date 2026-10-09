import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MARS_G = 3.71;
const FLAT_R = 25;          // terrain is exactly h = 0 inside this radius
const BLEND_R = 60;         // fully "natural" terrain beyond this radius
const CONE_R = 7000;        // inside this radius h <= 0.035 r must hold
const CONE_SLOPE = 0.032;   // enforced cap (a little below 0.035)
const GRID_N = 192;         // quads per ring edge (divisible by 4)
const GRID_LEVELS = 13;     // ring k spans +-24 * 2^k metres -> ~98 km
const BASE_CELL = 0.25;     // metres, finest ring

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const lerp = (a, b, t) => a + (b - a) * t;
function smoothstep(e0, e1, x) {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Terrain height function (analytic, shared by mesh, rocks and heightAt)
// ---------------------------------------------------------------------------

function ihash(x, y, s) {
  let h = Math.imul(x | 0, 0x27d4eb2d) ^ Math.imul(y | 0, 0x165667b1) ^ Math.imul(s | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

// Smooth value noise, 0..1
function vnoise(x, y, s) {
  const ix = Math.floor(x), iy = Math.floor(y);
  let fx = x - ix, fy = y - iy;
  fx = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  fy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
  const k = 1 / 4294967296;
  const a = ihash(ix, iy, s) * k, b = ihash(ix + 1, iy, s) * k;
  const c = ihash(ix, iy + 1, s) * k, d = ihash(ix + 1, iy + 1, s) * k;
  return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

function vnoise3(x, y, z, s) {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  let fx = x - ix, fy = y - iy, fz = z - iz;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
  const k = 1 / 4294967296;
  const h = (dx, dy, dz) => ihash(ix + dx, (iy + dy) * 73856093 ^ (iz + dz) * 19349663, s) * k;
  const x00 = lerp(h(0, 0, 0), h(1, 0, 0), fx), x10 = lerp(h(0, 1, 0), h(1, 1, 0), fx);
  const x01 = lerp(h(0, 0, 1), h(1, 0, 1), fx), x11 = lerp(h(0, 1, 1), h(1, 1, 1), fx);
  return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz);
}

// Level-of-detail weight: 0 when the feature size is below ~3 cells of the sampling grid
const lodW = (size, c) => (c > 0 ? clamp((size / c - 3) / 3, 0, 1) : 1);

const PLAINS = [[3000, 9], [1100, 4], [420, 1.6], [160, 0.7], [60, 0.28], [22, 0.1], [8, 0.035], [3, 0.012], [1.1, 0.004]];

function plains(x, z, r, c) {
  let h = 0;
  for (let i = 0; i < PLAINS.length; i++) {
    const lam = PLAINS[i][0], amp = PLAINS[i][1];
    const lw = lodW(lam, c);
    if (lw <= 0) break;
    // large undulations are suppressed near the pad so the cone constraint is met without clipping
    const ramp = smoothstep(FLAT_R, Math.max(BLEND_R, 0.6 * lam), r);
    if (ramp > 0) h += amp * lw * ramp * (vnoise(x / lam, z / lam, i + 1) * 2 - 1);
  }
  return h;
}

const WIND = [0.82, 0.57];

// Patches of transverse dunes, crests perpendicular to the wind
function dunes(x, z, r, c) {
  const lam = 38;
  const lw = lodW(lam, c);
  if (lw <= 0) return 0;
  const mask = smoothstep(0.58, 0.72, vnoise(x / 1800 + 7, z / 1800 - 3, 91)) * smoothstep(250, 800, r);
  if (mask <= 0) return 0;
  const warp = vnoise(x / 70, z / 70, 92) * 14;
  const u = (x * WIND[0] + z * WIND[1] + warp) / lam;
  const s = u - Math.floor(u);
  let f = s < 0.72 ? s / 0.72 : (1 - s) / 0.28;   // gentle windward, steep lee face
  f = f * f * (3 - 2 * f);
  return 1.3 * mask * lw * (0.6 + 0.8 * vnoise(x / 400, z / 400, 93)) * f;
}

const CRATER_P = [0.35, 0.5, 0.6, 0.6, 0.6, 0.55, 0.5, 0.45, 0.4, 0.4, 0.4, 0.4];
// Cumulative number ~ D^-2 (equal occupancy per octave, cell ~ 3.3 D)
const CRATER_CLASSES = [3, 6, 12, 24, 48, 96, 192, 400, 800, 1600, 3200, 6400].map((D, j) => ({
  D, j, p: CRATER_P[j], cell: 3.3 * D * Math.pow(D / 3, 0.07),
}));

let lastRim = 0;   // rim-proximity (0..1) of the last terrainHeight() call, used for rock placement

function craterProfile(x, D, age) {
  const fresh = 1 - age;
  const old = D >= 600 ? 0.4 : 1;                 // big craters are degraded: shallow, low rims
  const dep = D * (0.045 + 0.15 * fresh) * old;
  const rimH = D * (0.008 + 0.03 * fresh) * old;
  if (x < 1) {
    const t = 1 - x * x;
    const rw = (x - 1) / 0.22;
    return -dep * Math.pow(t, 1.4) + rimH * Math.exp(-rw * rw) + 0.45 * rimH * smoothstep(0.8, 1, x);
  }
  const rw = (x - 1) / 0.42;
  const taper = 1 - smoothstep(1.1, 1.5, x);
  return taper * (rimH * Math.exp(-rw * rw) + 0.45 * rimH * Math.pow(x, -2.5));   // rim + ejecta blanket
}

function craters(x, z, c) {
  let h = 0, rim = 0;
  for (let j = 0; j < CRATER_CLASSES.length; j++) {
    const k = CRATER_CLASSES[j];
    const lw = lodW(k.D, c / 1.5);
    if (lw <= 0) continue;
    const s = k.cell;
    const ix = Math.floor(x / s), iz = Math.floor(z / s);
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cx = ix + dx, cz = iz + dz;
        const H = ihash(cx, cz, 100 + j);
        if (((H >>> 20) & 63) / 64 >= k.p) continue;
        const D = k.D * (0.75 + 0.5 * (H >>> 26) / 64);
        const R = 0.5 * D;
        const px = (cx + 0.1 + 0.8 * (H & 1023) / 1024) * s;
        const pz = (cz + 0.1 + 0.8 * ((H >>> 10) & 1023) / 1024) * s;
        const ddx = x - px, ddz = z - pz;
        const d2 = ddx * ddx + ddz * ddz;
        if (d2 >= 2.25 * R * R) continue;
        const keep = 1.5 * D + 60;
        if (px * px + pz * pz < keep * keep) continue;   // keep the landing area free of craters
        const age = ((H >>> 3) & 63) / 64;
        const xr = Math.sqrt(d2) / R;
        h += lw * craterProfile(xr, D, age);
        if (D <= 400 && D >= 6) { const q = (xr - 1) / 0.6; rim += lw * Math.exp(-q * q); }
      }
    }
  }
  lastRim = Math.min(1, rim);
  return h;
}

function ridged(x, z, lam, seed) { return 1 - Math.abs(vnoise(x / lam, z / lam, seed) * 2 - 1); }

// Large-scale relief, identically zero for r <= CONE_R: crater-wall mountain range, mesas, hills
function relief(x, z, r, c) {
  const ramp = smoothstep(CONE_R, 12000, r);
  if (ramp <= 0) return 0;
  const a = Math.atan2(z, x), ca = Math.cos(a), sa = Math.sin(a);
  const nA = vnoise(ca * 1.6 + 10, sa * 1.6 + 4, 201);
  const nB = vnoise(ca * 4.0 + 30, sa * 4.0 - 8, 202);
  const ringR = 31000 + 7000 * (nA - 0.5);
  const d = r - ringR;
  const prof = d < 0 ? Math.pow(smoothstep(-13000, 0, d), 1.7) : 1 / (1 + (d / 5500) * (d / 5500));
  const H = 350 + 1700 * smoothstep(0.25, 0.75, nB);
  const rid = 0.7 + 0.5 * ridged(x, z, 7000, 203)
    + 0.3 * lodW(2600, c) * ridged(x, z, 2600, 204)
    + 0.2 * lodW(900, c) * ridged(x, z, 900, 205);
  let h = H * prof * rid * 0.8;

  const cell = 9000;
  const ix = Math.floor(x / cell), iz = Math.floor(z / cell);
  for (let dz = -1; dz <= 1; dz++) {
    for (let dx = -1; dx <= 1; dx++) {
      const cx = ix + dx, cz = iz + dz;
      const Hh = ihash(cx, cz, 300);
      if (((Hh >>> 20) & 63) / 64 >= 0.3) continue;
      const rad = 700 + 1000 * ((Hh & 1023) / 1024);
      if (c > 0 && rad < 3 * c) continue;
      const px = (cx + 0.15 + 0.7 * ((Hh >>> 10) & 1023) / 1024) * cell;
      const pz = (cz + 0.15 + 0.7 * (Hh >>> 26) / 64) * cell;
      const q = Math.hypot(x - px, z - pz) / rad;
      if (q < 1) h += (120 + 300 * ((Hh >>> 3) & 63) / 64) * (1 - smoothstep(0.72, 1, q));
    }
  }
  h += 140 * lodW(5000, c) * ridged(x, z, 5000, 210) * smoothstep(8000, 16000, r);
  return h * ramp;
}

function smin(a, b, k) {
  const t = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - t * t * k * 0.25;
}

/**
 * Terrain height in metres. `c` is the sampling cell size used to drop sub-cell
 * detail for coarse mesh rings; c = 0 gives the full-resolution surface.
 */
export function terrainHeight(x, z, c = 0) {
  const r = Math.hypot(x, z);
  lastRim = 0;
  if (r <= FLAT_R) return 0;
  const w = smoothstep(FLAT_R, BLEND_R, r);
  let h = (plains(x, z, r, c) + dunes(x, z, r, c) + craters(x, z, c)) * w;
  if (r < CONE_R) {
    const cap = CONE_SLOPE * r;                         // stay under the 4 degree glide-slope cone
    if (h > cap - 0.5) h = smin(h, cap, 0.5);
  } else {
    h += relief(x, z, r, c);
  }
  return h;
}

// ---------------------------------------------------------------------------
// Terrain mesh: concentric square rings, cell size doubling per ring
// ---------------------------------------------------------------------------

function buildTerrainLevel(k) {
  const N = GRID_N, half = N / 2, hole = N / 4;
  const c = BASE_CELL * Math.pow(2, k);
  const P = N + 3;
  const hp = new Float32Array(P * P);
  const at = (i, j) => (j + 1) * P + (i + 1);

  for (let j = -1; j <= N + 1; j++) {
    for (let i = -1; i <= N + 1; i++) {
      const ai = Math.abs(i - half), aj = Math.abs(j - half);
      if (k > 0 && ai < hole - 1 && aj < hole - 1) continue;
      // blend the LOD filter toward the next ring near the outer edge so shared vertices agree
      const edge = half - Math.max(ai, aj);
      const blend = k < GRID_LEVELS - 1 ? clamp(1 - edge / 8, 0, 1) : 0;
      hp[at(i, j)] = terrainHeight((i - half) * c, (j - half) * c, c * Math.pow(2, blend));
    }
  }
  if (k < GRID_LEVELS - 1) {
    // odd edge vertices are interpolated so there are no cracks against the coarser neighbour
    for (let t = 1; t < N; t += 2) {
      hp[at(t, 0)] = 0.5 * (hp[at(t - 1, 0)] + hp[at(t + 1, 0)]);
      hp[at(t, N)] = 0.5 * (hp[at(t - 1, N)] + hp[at(t + 1, N)]);
      hp[at(0, t)] = 0.5 * (hp[at(0, t - 1)] + hp[at(0, t + 1)]);
      hp[at(N, t)] = 0.5 * (hp[at(N, t - 1)] + hp[at(N, t + 1)]);
    }
  }

  const V = N + 1;
  const pos = new Float32Array(V * V * 3), nor = new Float32Array(V * V * 3);
  const heights = new Float32Array(V * V);
  for (let j = 0; j <= N; j++) {
    for (let i = 0; i <= N; i++) {
      const v = j * V + i;
      const h = hp[at(i, j)];
      heights[v] = h;
      pos[v * 3] = (i - half) * c; pos[v * 3 + 1] = h; pos[v * 3 + 2] = (j - half) * c;
      const nx = (hp[at(i - 1, j)] - hp[at(i + 1, j)]) / (2 * c);
      const nz = (hp[at(i, j - 1)] - hp[at(i, j + 1)]) / (2 * c);
      const il = 1 / Math.sqrt(nx * nx + 1 + nz * nz);
      nor[v * 3] = nx * il; nor[v * 3 + 1] = il; nor[v * 3 + 2] = nz * il;
    }
  }

  const idx = [];
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      if (k > 0 && i >= hole && i < N - hole && j >= hole && j < N - hole) continue;
      const a = j * V + i, b = (j + 1) * V + i, cc = j * V + i + 1, d = (j + 1) * V + i + 1;
      idx.push(a, b, cc, b, d, cc);
    }
  }
  return { c, E: half * c, pos, nor, heights, index: new Uint16Array(idx) };
}

// ---------------------------------------------------------------------------
// Procedural fallback textures (periodic so they tile)
// ---------------------------------------------------------------------------

function periodicFbm(S, octaves, seed, ridge) {
  const out = new Float32Array(S * S);
  let amp = 1, total = 0;
  for (let o = 0; o < octaves; o++) {
    const f = 4 << o;
    const rnd = mulberry32(seed + o * 101);
    const lat = new Float32Array(f * f);
    for (let i = 0; i < lat.length; i++) lat[i] = rnd();
    for (let y = 0; y < S; y++) {
      const fy = (y / S) * f, y0 = Math.floor(fy), ty0 = fy - y0;
      const ty = ty0 * ty0 * (3 - 2 * ty0);
      for (let x = 0; x < S; x++) {
        const fx = (x / S) * f, x0 = Math.floor(fx), tx0 = fx - x0;
        const tx = tx0 * tx0 * (3 - 2 * tx0);
        const x1 = (x0 + 1) % f, y1 = (y0 + 1) % f;
        let v = lerp(lerp(lat[y0 * f + x0], lat[y0 * f + x1], tx), lerp(lat[y1 * f + x0], lat[y1 * f + x1], tx), ty);
        if (ridge) v = 1 - Math.abs(v * 2 - 1);
        out[y * S + x] += v * amp;
      }
    }
    total += amp;
    amp *= 0.55;
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

function makeProceduralSet(kind) {
  const S = 256;
  const rock = kind === 'rock';
  const hgt = periodicFbm(S, rock ? 6 : 7, rock ? 7 : 3, rock);
  const grain = periodicFbm(S, 3, 99, false);
  const col = new Uint8Array(S * S * 4), nrm = new Uint8Array(S * S * 4), rgh = new Uint8Array(S * S * 4);
  const ao = new Uint8Array(S * S * 4);
  const strength = rock ? 5 : 2.2;
  let sumLin = 0;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const g = clamp(0.5 + (hgt[i] - 0.5) * (rock ? 1.8 : 1.1) + (grain[i] - 0.5) * 0.5, 0.05, 0.98);
      const g8 = Math.round(g * 255);
      col.set([g8, g8, g8, 255], i * 4);
      sumLin += Math.pow(g, 2.2);
      const dx = hgt[y * S + ((x + 1) % S)] - hgt[y * S + ((x + S - 1) % S)];
      const dy = hgt[((y + 1) % S) * S + x] - hgt[((y + S - 1) % S) * S + x];
      let nx = -dx * strength * S * 0.05, ny = -dy * strength * S * 0.05;
      const il = 1 / Math.sqrt(nx * nx + ny * ny + 1);
      nx *= il; ny *= il;
      nrm.set([Math.round((nx * 0.5 + 0.5) * 255), Math.round((ny * 0.5 + 0.5) * 255), Math.round((il * 0.5 + 0.5) * 255), 255], i * 4);
      const r8 = Math.round(clamp(0.82 + (grain[i] - 0.5) * 0.3, 0, 1) * 255);
      rgh.set([r8, r8, r8, 255], i * 4);
      const a8 = Math.round(clamp(0.55 + hgt[i] * 0.6, 0, 1) * 255);
      ao.set([a8, a8, a8, 255], i * 4);
    }
  }
  const mk = (data, srgb) => {
    const t = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.generateMipmaps = true;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    t.needsUpdate = true;
    return t;
  };
  return { map: mk(col, true), normalMap: mk(nrm, false), roughnessMap: mk(rgh, false), aoMap: mk(ao, false), mean: sumLin / (S * S) };
}

function solidTexture(v) {
  const t = new THREE.DataTexture(new Uint8Array([v, v, v, 255]), 1, 1, THREE.RGBAFormat);
  t.needsUpdate = true;
  return t;
}

// Mean linear luminance of a loaded colour texture (used to normalise it before Mars tinting)
function meanLuminance(tex) {
  try {
    const img = tex.image;
    const cv = document.createElement('canvas');
    cv.width = cv.height = 32;
    const ctx = cv.getContext('2d');
    ctx.drawImage(img, 0, 0, 32, 32);
    const d = ctx.getImageData(0, 0, 32, 32).data;
    let s = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = Math.pow(d[i] / 255, 2.2), g = Math.pow(d[i + 1] / 255, 2.2), b = Math.pow(d[i + 2] / 255, 2.2);
      s += 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    return Math.max(s / (d.length / 4), 0.02);
  } catch (e) {
    return 0.25;
  }
}

async function loadGroundSet(def, loader, base, aniso) {
  if (!def || !def.map || !def.normalMap) return null;
  const load = async (p, srgb) => {
    if (!p) return null;
    try {
      const t = await loader.loadAsync(base + p);
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.anisotropy = aniso;
      if (srgb) t.colorSpace = THREE.SRGBColorSpace;
      return t;
    } catch (e) {
      return null;
    }
  };
  const [map, normalMap, roughnessMap, aoMap] = await Promise.all([
    load(def.map, true), load(def.normalMap, false), load(def.roughnessMap, false), load(def.aoMap, false),
  ]);
  if (!map || !normalMap) return null;
  return { map, normalMap, roughnessMap: roughnessMap || solidTexture(255), aoMap: aoMap || solidTexture(255), mean: meanLuminance(map) };
}

// ---------------------------------------------------------------------------
// Terrain material: MeshStandardMaterial patched for world-space multi-scale
// texturing tinted to Mars regolith colours.
// ---------------------------------------------------------------------------

const TERRAIN_FRAG_PARS = /* glsl */`
uniform sampler2D tSD; uniform sampler2D tSN; uniform sampler2D tSR;
uniform sampler2D tRD; uniform sampler2D tRN; uniform sampler2D tRR; uniform sampler2D tRA;
uniform sampler2D tCD; uniform sampler2D tCN; uniform sampler2D tCR; uniform sampler2D tCA;
uniform vec3 uMean;
uniform float uCrack;
uniform vec4 uScour;
varying vec3 vWPos;
varying vec3 vWN;
float tH(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float tN(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(tH(i), tH(i + vec2(1.0, 0.0)), f.x), mix(tH(i + vec2(0.0, 1.0)), tH(i + vec2(1.0, 1.0)), f.x), f.y);
}
float tF(vec2 p) {
  float a = 0.0, b = 0.5;
  for (int i = 0; i < 4; i++) { a += b * tN(p); p = p * 2.03 + vec2(17.0, 9.0); b *= 0.5; }
  return a / 0.9375;
}
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
`;

const TERRAIN_MAP_FRAG = /* glsl */`
vec3 P = vWPos;
vec3 gN = normalize(vWN);
vec2 xz = P.xz;
float dist = length(vViewPosition);
float rr = length(xz);
float slope = 1.0 - gN.y;
float macro = tF(xz * 0.0021 + 3.1);
float macro2 = tF(xz * 0.017 + 11.7);

float scourM = uScour.w * (1.0 - smoothstep(0.35 * uScour.z, uScour.z, length(xz - uScour.xy)));
float rockM = smoothstep(0.035, 0.13, slope + (macro2 - 0.5) * 0.07);
rockM = max(rockM, smoothstep(0.62, 0.78, macro) * 0.85);
rockM = max(rockM * smoothstep(14.0, 30.0, rr), scourM * 0.6);
float crackM = uCrack * smoothstep(0.60, 0.70, tF(xz * 0.0012 + 40.0)) * (1.0 - rockM) * smoothstep(60.0, 220.0, rr);

// sand: planar, three scales (2.4 m, 9.7 m rotated, 83 m) to break up tiling
vec2 suv = vec2(xz.x, -xz.y);
vec2 uvA = suv / 2.4;
vec2 uvB = mat2(0.8, -0.6, 0.6, 0.8) * suv / 9.7;
vec2 uvC = suv / 83.0;
float wAB = 0.15 + 0.7 * smoothstep(0.3, 0.7, tF(xz * 0.09 + 5.0));
float fDist = smoothstep(25.0, 160.0, dist);
float lA = lum(texture2D(tSD, uvA).rgb);
float lB = lum(texture2D(tSD, uvB).rgb);
float lC = lum(texture2D(tSD, uvC).rgb);
float relS = clamp(mix(mix(lA, lB, wAB), lC, fDist * 0.5) / uMean.x, 0.3, 2.2);
vec3 palS = mix(vec3(0.333, 0.105, 0.042), vec3(0.485, 0.181, 0.076), smoothstep(0.30, 0.70, macro));
palS = mix(palS, vec3(0.156, 0.044, 0.0185) * 1.3, smoothstep(0.60, 0.82, macro2) * 0.55);

// aeolian ripples (wavelength 0.7 m and 5 m), crests perpendicular to the wind
vec2 wd = vec2(0.82, 0.57);
float flatness = 1.0 - smoothstep(0.03, 0.10, slope);
float ripAmp = flatness * (1.0 - rockM) * (1.0 - scourM) * smoothstep(0.30, 0.60, tF(xz * 0.006 + 21.0));
float ph1 = dot(xz, wd) * 8.98 + 4.0 * tN(xz * 0.45);
float ph2 = dot(xz, wd) * 1.26 + 2.5 * tN(xz * 0.05);
float f1 = 1.0 - smoothstep(25.0, 110.0, dist);
float f2 = 1.0 - smoothstep(300.0, 1500.0, dist);
vec2 ripGrad = wd * (cos(ph1) * 0.045 * f1 + cos(ph2) * 0.06 * f2) * ripAmp;
vec3 albS = palS * pow(relS, 0.85) * (1.0 + 0.10 * sin(ph1) * ripAmp * f1);

vec3 tA = texture2D(tSN, uvA).xyz * 2.0 - 1.0;
vec3 tB = texture2D(tSN, uvB).xyz * 2.0 - 1.0;
vec2 tnS = tA.xy + tB.xy * mix(0.55, 0.9, wAB);
vec3 nS = normalize(vec3(gN.x + tnS.x - ripGrad.x, gN.y, gN.z - tnS.y - ripGrad.y));
float rS = texture2D(tSR, uvA).g;

// rock / gravel: triplanar so steep crater walls do not smear
vec3 bw = pow(abs(gN), vec3(4.0));
bw /= (bw.x + bw.y + bw.z);
float sR = 1.0 / 3.1;
vec2 uvRy = suv * sR;
vec2 uvRx = vec2(P.z, P.y) * sR;
vec2 uvRz = vec2(P.x, P.y) * sR;
vec3 dR = texture2D(tRD, uvRx).rgb * bw.x + texture2D(tRD, uvRy).rgb * bw.y + texture2D(tRD, uvRz).rgb * bw.z;
float relR = clamp(lum(dR) / uMean.y, 0.3, 2.2);
vec3 palR = mix(vec3(0.075, 0.036, 0.024), vec3(0.20, 0.085, 0.04), smoothstep(0.3, 0.7, macro2));
vec3 albR = palR * pow(relR, 0.9);
vec3 tx = texture2D(tRN, uvRx).xyz * 2.0 - 1.0;
vec3 ty = texture2D(tRN, uvRy).xyz * 2.0 - 1.0;
vec3 tz = texture2D(tRN, uvRz).xyz * 2.0 - 1.0;
vec3 nRx = vec3(abs(tx.z) * gN.x, tx.y + gN.y, tx.x + gN.z);
vec3 nRy = vec3(gN.x + ty.x, gN.y, gN.z - ty.y);
vec3 nRz = vec3(tz.x + gN.x, tz.y + gN.y, abs(tz.z) * gN.z);
vec3 nR = normalize(nRx * bw.x + nRy * bw.y + nRz * bw.z);
float rR = texture2D(tRR, uvRy).g;
float aoR = texture2D(tRA, uvRy).r;

// cracked dry ground patches
vec2 uvK = suv / 5.5;
float relK = clamp(lum(texture2D(tCD, uvK).rgb) / uMean.z, 0.3, 2.2);
vec3 albK = mix(vec3(0.26, 0.105, 0.05), vec3(0.37, 0.15, 0.07), macro) * pow(relK, 0.9);
vec3 tk = texture2D(tCN, uvK).xyz * 2.0 - 1.0;
vec3 nK = normalize(vec3(gN.x + tk.x, gN.y, gN.z - tk.y));
float rK = texture2D(tCR, uvK).g;
float aoK = texture2D(tCA, uvK).r;

vec3 alb = mix(mix(albS, albK, crackM), albR, rockM);
alb *= 1.0 - 0.30 * scourM;
float ao = mix(mix(1.0, aoK, crackM), aoR, rockM);
alb *= mix(1.0, ao, 0.8);
diffuseColor.rgb = alb;

vec3 nD = normalize(mix(mix(nS, nK, crackM), nR, rockM));
vec3 gNd = normalize(mix(gN, nD, 1.0 - smoothstep(250.0, 2500.0, dist)));
float gRough = clamp(0.7 + 0.3 * mix(mix(rS, rK, crackM), rR, rockM), 0.7, 1.0);
`;

function createTerrainMaterial(sand, rock, crack) {
  const uniforms = {
    tSD: { value: sand.map }, tSN: { value: sand.normalMap }, tSR: { value: sand.roughnessMap },
    tRD: { value: rock.map }, tRN: { value: rock.normalMap }, tRR: { value: rock.roughnessMap }, tRA: { value: rock.aoMap },
    tCD: { value: (crack || rock).map }, tCN: { value: (crack || rock).normalMap },
    tCR: { value: (crack || rock).roughnessMap }, tCA: { value: (crack || rock).aoMap },
    uMean: { value: new THREE.Vector3(sand.mean, rock.mean, (crack || rock).mean) },
    uCrack: { value: crack ? 1 : 0 },
    uScour: { value: new THREE.Vector4(0, 0, 1, 0) },
  };
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0 });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWN;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvWPos = position;\nvWN = normal;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\n' + TERRAIN_FRAG_PARS)
      .replace('#include <map_fragment>', TERRAIN_MAP_FRAG)
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = gRough;')
      .replace('#include <normal_fragment_maps>', 'normal = normalize((viewMatrix * vec4(gNd, 0.0)).xyz);');
  };
  material.customProgramCacheKey = () => 'mars-terrain-v1';
  return { material, uniforms };
}

// ---------------------------------------------------------------------------
// Sky
// ---------------------------------------------------------------------------

const HORIZON_COL = new THREE.Color().setRGB(0.80, 0.58, 0.40, THREE.LinearSRGBColorSpace);
const HAZE_BOOST = 1.04;   // sky colour at e = 0 relative to HORIZON_COL; fog uses the product

function createSkyMaterial(sunDir, disc) {
  return new THREE.ShaderMaterial({
    uniforms: { uSun: { value: sunDir }, uDisc: { value: disc } },
    side: THREE.BackSide,
    depthTest: false,
    depthWrite: false,
    fog: false,
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() {
        vDir = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */`
      uniform vec3 uSun;
      uniform float uDisc;
      varying vec3 vDir;
      void main() {
        vec3 d = normalize(vDir);
        float e = d.y;
        vec3 horizon = vec3(0.80, 0.58, 0.40);
        vec3 zenith = vec3(0.40, 0.265, 0.185);
        vec3 col = mix(horizon, zenith, pow(clamp(e, 0.0, 1.0), 0.42));
        col = mix(col, horizon * 1.08, exp(-max(e, 0.0) * 9.0) * 0.5);
        // forward scattering by dust: cool blue aureole plus a wide warm glow
        float ang = acos(clamp(dot(d, uSun), -1.0, 1.0));
        col += vec3(0.42, 0.56, 0.82) * 0.75 * exp(-ang / 0.10) + vec3(1.0, 0.82, 0.6) * 0.25 * exp(-ang / 0.5);
        col += vec3(1.0, 0.95, 0.85) * smoothstep(0.0036, 0.0028, ang) * 25.0 * uDisc;
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
}

// ---------------------------------------------------------------------------
// Rocks
// ---------------------------------------------------------------------------

function makeRockGeometry(detail, seed) {
  let g = new THREE.IcosahedronGeometry(1, detail);
  g.deleteAttribute('normal');
  g.deleteAttribute('uv');
  const rnd = mulberry32(seed);
  const sx = 0.8 + 0.5 * rnd(), sy = 0.5 + 0.35 * rnd(), sz = 0.8 + 0.5 * rnd();
  const o = rnd() * 50;
  const p = g.attributes.position;
  const v = new THREE.Vector3();
  // identical source vertices get identical displacement because it is a pure function of position
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).normalize();
    const n1 = vnoise3(v.x * 1.4 + o, v.y * 1.4, v.z * 1.4, 7);
    const n2 = 1 - Math.abs(vnoise3(v.x * 3.1 + o, v.y * 3.1, v.z * 3.1, 8) * 2 - 1);
    const rad = 1 + 0.30 * (n1 - 0.5) * 2 + 0.12 * (n2 - 0.5) * 2;
    p.setXYZ(i, v.x * rad * sx, Math.max(v.y * rad * sy, -0.32 * sy), v.z * rad * sz);   // flattened base
  }
  g.computeVertexNormals();
  g.computeBoundingBox();
  const bb = g.boundingBox;
  const m = 1 / Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z);
  g.translate(-(bb.max.x + bb.min.x) / 2, -bb.min.y, -(bb.max.z + bb.min.z) / 2);
  g.scale(m, m, m);
  // per-vertex brightness variation
  const pp = g.attributes.position, col = new Float32Array(pp.count * 3);
  for (let i = 0; i < pp.count; i++) {
    const n = 0.75 + 0.4 * vnoise3(pp.getX(i) * 7 + o, pp.getY(i) * 7, pp.getZ(i) * 7, 9);
    col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = n;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}

// Adds a wind-blown dust coat on upward faces and darker contact zone at the base
function patchRockMaterial(mat) {
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying float vLY;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvLY = position.y;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vLY;')
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        float upk = clamp(dot(normal, normalize((viewMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz)), 0.0, 1.0);
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.30, 0.12, 0.055), smoothstep(0.45, 0.95, upk) * 0.55);
        diffuseColor.rgb *= mix(0.5, 1.0, smoothstep(0.0, 0.22, vLY));`);
  };
  mat.customProgramCacheKey = () => 'mars-rock-v1' + (mat.map ? 'm' : 'p') + (mat.vertexColors ? 'c' : '');
  return mat;
}

async function loadRockModels(manifest, base, aniso) {
  const list = (manifest && manifest.rocks) || [];
  const loader = new GLTFLoader();
  const out = [];
  for (const entry of list) {
    if (!entry || !entry.gltf) continue;
    try {
      const gltf = await loader.loadAsync(base + entry.gltf);
      gltf.scene.updateMatrixWorld(true);
      let mesh = null;
      gltf.scene.traverse((o) => { if (!mesh && o.isMesh) mesh = o; });
      if (!mesh) continue;
      const geo = mesh.geometry.clone();
      geo.applyMatrix4(mesh.matrixWorld);
      geo.computeBoundingBox();
      const bb = geo.boundingBox;
      const m = 1 / Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z);
      geo.translate(-(bb.max.x + bb.min.x) / 2, -bb.min.y, -(bb.max.z + bb.min.z) / 2);
      geo.scale(m, m, m);
      const src = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      const mat = new THREE.MeshStandardMaterial({
        map: src.map || null, normalMap: src.normalMap || null, roughnessMap: src.roughnessMap || null,
        aoMap: null, color: new THREE.Color().setRGB(0.62, 0.45, 0.36, THREE.SRGBColorSpace), roughness: 1, metalness: 0,
      });
      for (const t of [mat.map, mat.normalMap, mat.roughnessMap]) if (t) t.anisotropy = aniso;
      const tris = geo.index ? geo.index.count / 3 : geo.attributes.position.count / 3;
      out.push({ geo, mat: patchRockMaterial(mat), tris });
    } catch (e) {
      // model missing: skip, procedural rocks cover for it
    }
  }
  return out;
}

function sampleSize(rnd, dmin, dmax, q) {
  const u = rnd();
  return dmin / Math.pow(1 - u * (1 - Math.pow(dmin / dmax, q)), 1 / q);
}

// Candidate rock list; density is boosted near crater rims
function generateRocks(meshHeight) {
  const rnd = mulberry32(12345);
  const cats = [
    { n: 2600, r0: 16, r1: 90, dmin: 0.04, dmax: 0.5 },
    { n: 3400, r0: 90, r1: 350, dmin: 0.1, dmax: 1.4 },
    { n: 1500, r0: 350, r1: 2500, dmin: 0.5, dmax: 3.2 },
    { n: 450, r0: 700, r1: 6500, dmin: 2.0, dmax: 8 },
  ];
  const rocks = [];
  for (const cat of cats) {
    let made = 0, tries = 0;
    while (made < cat.n && tries < cat.n * 30) {
      tries++;
      const r = Math.sqrt(cat.r0 * cat.r0 + rnd() * (cat.r1 * cat.r1 - cat.r0 * cat.r0));
      const a = rnd() * Math.PI * 2;
      const x = r * Math.cos(a), z = r * Math.sin(a);
      const hExact = terrainHeight(x, z);
      const accept = 0.25 + 0.75 * Math.min(1, lastRim * 1.5);
      if (rnd() > accept) continue;
      const size = sampleSize(rnd, cat.dmin, cat.dmax, 2.2);
      const y = r < 200 ? hExact : meshHeight(x, z);
      rocks.push({ x, y, z, r, size, yaw: rnd() * Math.PI * 2, tx: (rnd() - 0.5) * 0.4, tz: (rnd() - 0.5) * 0.4,
        a: 0.8 + 0.45 * rnd(), b: 0.7 + 0.5 * rnd(), c: 0.8 + 0.45 * rnd(), v: rnd(), tint: rnd() });
      made++;
    }
  }
  return rocks;
}

function buildRocks(rocks, models, group, disposables) {
  const BIG = 0.25;
  const procSmall = [0, 1, 2, 3].map((i) => makeRockGeometry(2, 40 + i));
  const procBig = [0, 1, 2].map((i) => makeRockGeometry(3, 80 + i));
  const procMat = patchRockMaterial(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0, vertexColors: true }));
  disposables.push(...procSmall, ...procBig, procMat);

  // largest apparent size first gets the detailed glTF meshes, within a triangle budget
  const caps = models.map((m) => Math.floor(1.5e6 / Math.max(m.tris, 1)));
  const used = models.map(() => 0);
  const order = rocks.map((r, i) => i).filter((i) => rocks[i].size >= BIG)
    .sort((i, j) => rocks[j].size / Math.max(rocks[j].r, 30) - rocks[i].size / Math.max(rocks[i].r, 30));
  const assign = new Array(rocks.length).fill(null);
  let rr = 0;
  if (models.length) {
    for (const i of order) {
      for (let t = 0; t < models.length; t++) {
        const m = (rr + t) % models.length;
        if (used[m] < caps[m]) { assign[i] = { gltf: m }; used[m]++; rr = m + 1; break; }
      }
    }
  }
  const buckets = new Map();
  const put = (key, geo, mat, rock) => {
    const near = rock.r < 140 ? 1 : 0;
    const k = key + '|' + near;
    if (!buckets.has(k)) buckets.set(k, { geo, mat, near, items: [] });
    buckets.get(k).items.push(rock);
  };
  rocks.forEach((rock, i) => {
    if (assign[i]) put('g' + assign[i].gltf, models[assign[i].gltf].geo, models[assign[i].gltf].mat, rock);
    else if (rock.size >= BIG) { const v = Math.floor(rock.v * procBig.length); put('pb' + v, procBig[v], procMat, rock); }
    else { const v = Math.floor(rock.v * procSmall.length); put('ps' + v, procSmall[v], procMat, rock); }
  });

  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), p = new THREE.Vector3(), s = new THREE.Vector3();
  const basalt = new THREE.Color(0x4a3328), dusty = new THREE.Color(0x7a4a34), col = new THREE.Color();
  for (const b of buckets.values()) {
    const mesh = new THREE.InstancedMesh(b.geo, b.mat, b.items.length);
    b.items.forEach((rk, i) => {
      e.set(rk.tx, rk.yaw, rk.tz, 'YXZ');
      q.setFromEuler(e);
      s.set(rk.size * rk.a, rk.size * rk.b, rk.size * rk.c);
      p.set(rk.x, rk.y - 0.12 * rk.size * rk.b, rk.z);   // sink the base slightly into the regolith
      m4.compose(p, q, s);
      mesh.setMatrixAt(i, m4);
      col.copy(basalt).lerp(dusty, rk.tint * 0.8).multiplyScalar(0.8 + 0.7 * rk.v);
      mesh.setColorAt(i, col);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.castShadow = !!b.near;
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    group.add(mesh);
    disposables.push(mesh);
  }
}

// ---------------------------------------------------------------------------
// Dust ejecta: ring buffer of analytically evolving camera-facing sprites.
// Particle state is a closed-form function of age, so the CPU only writes
// new particles; the vertex shader evaluates motion, drag, gravity and growth.
// ---------------------------------------------------------------------------

const DUST_MAX = 20000;

function createDust({ scene, heightAt, sunDirection, sunRadiance, ambient, terrainUniforms }) {
  const base = new THREE.PlaneGeometry(1, 1);
  const geo = new THREE.InstancedBufferGeometry();
  geo.index = base.index;
  geo.setAttribute('position', base.getAttribute('position'));
  geo.setAttribute('uv', base.getAttribute('uv'));
  const mk = () => new THREE.InstancedBufferAttribute(new Float32Array(DUST_MAX * 4), 4).setUsage(THREE.DynamicDrawUsage);
  const aA = mk(), aV = mk(), aM = mk(), aP = mk();
  for (let i = 0; i < DUST_MAX; i++) { aA.array[i * 4 + 3] = -1e9; aV.array[i * 4 + 3] = 1; aP.array[i * 4] = 1; }
  geo.setAttribute('aA', aA); geo.setAttribute('aV', aV); geo.setAttribute('aM', aM); geo.setAttribute('aP', aP);
  geo.instanceCount = 0;

  const uniforms = THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
    uTime: { value: 0 },
    uSunDir: { value: sunDirection },
    uSunRad: { value: sunRadiance },
    uAmb: { value: ambient },
    uAlbedo: { value: new THREE.Color().setRGB(0.52, 0.25, 0.12, THREE.LinearSRGBColorSpace) },
  }]);
  const material = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    fog: true,
    vertexShader: /* glsl */`
      uniform float uTime;
      uniform vec3 uSunDir;
      uniform vec3 uSunRad;
      uniform vec3 uAmb;
      uniform vec3 uAlbedo;
      attribute vec4 aA;  // spawn position, spawn time
      attribute vec4 aV;  // initial velocity, lifetime
      attribute vec4 aM;  // size, seed, turbulence weight, ground height at spawn
      attribute vec4 aP;  // drag, effective gravity, growth, peak alpha
      varying vec2 vUv;
      varying vec3 vCol;
      varying float vAlpha;
      varying float vSeed;
      #include <fog_pars_vertex>
      void main() {
        float life = aV.w;
        float age = uTime - aA.w;
        float live = step(0.0, age) * step(age, life);
        float t = clamp(age, 0.0, life);
        float k = aP.x;
        float e = exp(-k * t);
        float f = (1.0 - e) / k;
        // velocity decays as exp(-k t); gravity pulls toward terminal speed g/k
        vec3 p = aA.xyz + aV.xyz * f;
        p.y += (aP.y / k) * (f - t);
        float grow = 1.0 - exp(-t / (0.4 * life));
        float s = aM.x * (1.0 + aP.z * grow);
        float ph = aM.y * 40.0;
        vec3 sw = vec3(sin(ph + t * 0.9), 0.35 * sin(ph * 1.7 + t * 0.7), cos(ph * 1.3 + t * 0.8));
        p += sw * s * 0.35 * aM.z * grow;
        p.y = max(p.y, aM.w + 0.3 * s);

        float ang = aM.y * 6.2831 + t * (aM.y - 0.5) * 0.6;
        float ca = cos(ang), sa = sin(ang);
        vec2 q = vec2(ca * position.x - sa * position.y, sa * position.x + ca * position.y);
        vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
        mvPosition.xy += q * s * live;
        gl_Position = projectionMatrix * mvPosition;

        vec3 rd = normalize(p - cameraPosition);
        float fwd = max(dot(rd, uSunDir), 0.0);
        float phase = 0.55 + 2.2 * pow(fwd, 8.0);   // forward-scattering glow when looking toward the sun
        vCol = uAlbedo * (uAmb + uSunRad * phase);
        float fadeIn = smoothstep(0.0, 0.15, t);
        float fadeOut = 1.0 - smoothstep(0.5 * life, life, t);
        float dilution = pow(aM.x / s, 1.1);
        float nearFade = smoothstep(0.4, 4.0, -mvPosition.z);
        vAlpha = aP.w * fadeIn * fadeOut * dilution * nearFade * live;
        vUv = uv;
        vSeed = aM.y;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */`
      varying vec2 vUv;
      varying vec3 vCol;
      varying float vAlpha;
      varying float vSeed;
      #include <fog_pars_fragment>
      float h(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float n(vec2 p) {
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(h(i), h(i + vec2(1.0, 0.0)), f.x), mix(h(i + vec2(0.0, 1.0)), h(i + vec2(1.0, 1.0)), f.x), f.y);
      }
      void main() {
        if (vAlpha < 0.002) discard;
        vec2 q = vUv * 2.0 - 1.0;
        float r = length(q);
        float nz = n(q * 2.2 + vSeed * 37.0) * 0.6 + n(q * 5.0 - vSeed * 13.0) * 0.4;
        float a = smoothstep(1.0, 0.15, r + (nz - 0.5) * 0.55);
        a *= a;
        gl_FragColor = vec4(vCol, a * vAlpha);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
        #include <fog_fragment>
      }`,
  });
  const mesh = new THREE.Mesh(geo, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 10;
  scene.add(mesh);

  let time = 0, head = 0, total = 0;
  const acc = new Float32Array(32);
  const hit = new THREE.Vector3();
  let scour = 0;
  const scourCentre = new THREE.Vector2(1e9, 1e9);
  let spawnedFrame = 0, frameStart = 0;

  // Distance along the exhaust ray to the terrain (Newton on the vertical gap)
  function plumeHit(pos, dir, out) {
    if (dir.y > -0.05) return -1;
    let s = 0;
    for (let i = 0; i < 8; i++) {
      const x = pos.x + dir.x * s, z = pos.z + dir.z * s;
      const err = pos.y + dir.y * s - heightAt(x, z);
      s += err / -dir.y;
      if (Math.abs(err) < 0.02) break;
    }
    if (s < 0) s = 0;
    out.set(pos.x + dir.x * s, 0, pos.z + dir.z * s);
    out.y = heightAt(out.x, out.z);
    return s;
  }

  function spawn(kind, P, strength, H, dir) {
    const i = head;
    head = (head + 1) % DUST_MAX;
    total++;
    spawnedFrame++;
    const R = Math.random;
    let phi = R() * Math.PI * 2;
    const slant = Math.min(Math.hypot(dir.x, dir.z) * 1.5, 0.6);
    if (R() < slant) phi = Math.atan2(dir.z, dir.x) + (R() - 0.5) * 1.4;   // inclined plume favours downstream
    const cp = Math.cos(phi), sp = Math.sin(phi);
    const sq = Math.sqrt(strength);
    let r0, speed, vy, drag, g, size, life, grow, alpha, turb;
    if (kind === 0) {          // fast, low radial sheet
      r0 = (0.4 + 0.25 * H) * (0.3 + 0.7 * R());
      speed = (10 + 45 * sq) * (0.35 + 0.8 * R());
      vy = speed * (0.03 + 0.15 * R());
      drag = 0.45 + 0.55 * R(); g = MARS_G;
      size = (0.5 + 1.6 * R() * R()) * (0.8 + 0.6 * strength);
      life = 1.8 + 3.2 * R(); grow = 2.0; alpha = 0.10 + 0.08 * R(); turb = 0.25;
    } else {                   // slow billowing cloud that settles
      r0 = (1 + 0.3 * H) * R();
      speed = (2 + 9 * sq) * R();
      vy = (1.5 + 8 * sq) * (0.25 + 0.75 * R());
      drag = 0.30 + 0.25 * R(); g = MARS_G * 0.3;
      size = (2 + 4 * R()) * (0.6 + 0.7 * strength);
      life = 5 + 5 * R(); grow = 2.6; alpha = 0.05 + 0.04 * R(); turb = 1;
    }
    const x = P.x + cp * r0, z = P.z + sp * r0;
    const yg = heightAt(x, z);
    aA.array.set([x, yg + 0.05, z, time], i * 4);
    aV.array.set([cp * speed, vy, sp * speed, life], i * 4);
    aM.array.set([size, R(), turb, yg], i * 4);
    aP.array.set([drag, g, grow, alpha], i * 4);
  }

  function flush() {
    if (spawnedFrame === 0) return;
    const n = Math.min(spawnedFrame, DUST_MAX);
    for (const a of [aA, aV, aM, aP]) {
      if (frameStart + n <= DUST_MAX) {
        a.addUpdateRange(frameStart * 4, n * 4);
      } else {
        a.addUpdateRange(frameStart * 4, (DUST_MAX - frameStart) * 4);
        a.addUpdateRange(0, (frameStart + n - DUST_MAX) * 4);
      }
      a.needsUpdate = true;
    }
    geo.instanceCount = Math.min(total, DUST_MAX);
  }

  function update(dt, { nozzles, throttle, active } = {}) {
    dt = Math.min(dt, 0.1);
    time += dt;
    uniforms.uTime.value = time;
    spawnedFrame = 0;
    frameStart = head;
    if (active && nozzles && nozzles.length) {
      let bestH = 1e9, cx = 0, cz = 0, wsum = 0;
      for (let i = 0; i < nozzles.length && i < 32; i++) {
        const nz = nozzles[i];
        const thr = Array.isArray(throttle) ? (throttle[i] ?? 0) : (typeof throttle === 'number' ? throttle : 1);
        if (thr < 0.02) continue;
        const H = plumeHit(nz.position, nz.direction, hit);
        if (H < 0 || H > 40) continue;
        const Hc = Math.max(H, 0.5);
        // jet momentum flux at the ground falls roughly with the square of distance beyond a few metres
        const inten = thr * (1 / (1 + (Hc / 6) * (Hc / 6))) * (1 - smoothstep(18, 40, Hc));
        acc[i] += dt * (450 * inten);
        const nSheet = Math.min(Math.floor(acc[i]), 400);
        acc[i] -= Math.floor(acc[i]);
        for (let n = 0; n < nSheet; n++) spawn(0, hit, inten, Hc, nz.direction);
        acc[16 + i] += dt * (150 * inten);
        const nCloud = Math.min(Math.floor(acc[16 + i]), 150);
        acc[16 + i] -= Math.floor(acc[16 + i]);
        for (let n = 0; n < nCloud; n++) spawn(1, hit, inten, Hc, nz.direction);
        if (Hc < bestH) bestH = Hc;
        cx += hit.x * inten; cz += hit.z * inten; wsum += inten;
      }
      // blast scour: slowly darken and roughen the patch under the plume
      if (wsum > 0.05 && bestH < 12) {
        cx /= wsum; cz /= wsum;
        if (Math.hypot(cx - scourCentre.x, cz - scourCentre.y) > 25) { scour = 0; scourCentre.set(cx, cz); }
        scour += dt * wsum * 0.5;
        const u = terrainUniforms.uScour.value;
        u.set(scourCentre.x, scourCentre.y, 3 + 7 * Math.min(1, scour / 3), 1 - Math.exp(-scour));
      }
    }
    flush();
  }

  return {
    update,
    dispose() {
      scene.remove(mesh);
      geo.dispose();
      base.dispose();
      material.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

export async function createEnvironment({
  renderer, scene, camera, manifest = null, assetBase = '',
  sunElevationDeg = 30, sunAzimuthDeg = 107.5, sunIntensity = 9, shadowMapSize = 4096,
  fogDensity = 3.5e-5, environmentIntensity = 0.9,
} = {}) {
  const disposables = [];
  const tick = () => new Promise((r) => setTimeout(r, 0));
  const aniso = renderer.capabilities.getMaxAnisotropy();

  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // --- sun direction (azimuth measured from +X toward +Z) ---
  const el = THREE.MathUtils.degToRad(sunElevationDeg), az = THREE.MathUtils.degToRad(sunAzimuthDeg);
  const sunDirection = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)).normalize();
  const sunColor = new THREE.Color().setRGB(1.0, 0.92, 0.80, THREE.LinearSRGBColorSpace);

  // --- sky, fog, image-based lighting ---
  const fogColor = HORIZON_COL.clone().multiplyScalar(HAZE_BOOST);
  const ownFog = new THREE.FogExp2(fogColor, fogDensity);
  scene.fog = ownFog;

  const skyMat = createSkyMaterial(sunDirection, 1);
  const sky = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), skyMat);
  sky.frustumCulled = false;
  sky.renderOrder = -1000;
  scene.add(sky);
  disposables.push(sky.geometry, skyMat);

  const envScene = new THREE.Scene();
  const envSkyMat = createSkyMaterial(sunDirection, 0);   // no disc in the reflection map: the directional light provides the sun
  const envSky = new THREE.Mesh(new THREE.SphereGeometry(50, 48, 24), envSkyMat);
  envScene.add(envSky);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envRT = pmrem.fromScene(envScene, 0.02);
  pmrem.dispose();
  scene.environment = envRT.texture;
  scene.environmentIntensity = environmentIntensity;
  disposables.push(envRT, envSky.geometry, envSkyMat);

  // --- lights ---
  const sun = new THREE.DirectionalLight(sunColor, sunIntensity);
  sun.castShadow = true;
  sun.shadow.mapSize.set(shadowMapSize, shadowMapSize);
  sun.shadow.bias = -0.0003;
  sun.shadow.camera.near = 0.5;
  scene.add(sun, sun.target);
  const hemi = new THREE.HemisphereLight(
    new THREE.Color().setRGB(0.80, 0.60, 0.43, THREE.LinearSRGBColorSpace),
    new THREE.Color().setRGB(0.45, 0.26, 0.16, THREE.LinearSRGBColorSpace), 1.0);
  scene.add(hemi);

  // --- textures ---
  const texLoader = new THREE.TextureLoader();
  let m = manifest;
  if (!m) {
    // no manifest: probe the known vendored texture set, silently falling back if absent
    const mk = (id, folder, ao) => ({
      map: `assets/textures/${folder}/${id}_diff_2k.jpg`, normalMap: `assets/textures/${folder}/${id}_nor_gl_2k.jpg`,
      roughnessMap: `assets/textures/${folder}/${id}_rough_2k.jpg`, aoMap: ao ? `assets/textures/${folder}/${id}_ao_2k.jpg` : null,
    });
    m = { ground_primary: mk('red_sand', 'red_sand', false), ground_rocky: mk('red_laterite_soil_stones', 'red_laterite_soil_stones', true) };
  }
  const [sandL, rockL, crackL] = await Promise.all([
    loadGroundSet(m.ground_primary, texLoader, assetBase, aniso),
    loadGroundSet(m.ground_rocky, texLoader, assetBase, aniso),
    loadGroundSet(m.ground_cracked, texLoader, assetBase, aniso),
  ]);
  const sand = sandL || makeProceduralSet('sand');
  const rock = rockL || makeProceduralSet('rock');
  for (const set of [sandL, rockL, crackL, sandL ? null : sand, rockL ? null : rock]) {
    if (set) disposables.push(set.map, set.normalMap, set.roughnessMap, set.aoMap);
  }
  const { material: terrainMat, uniforms: terrainUniforms } = createTerrainMaterial(sand, rock, crackL);
  disposables.push(terrainMat);

  // --- terrain rings ---
  const terrain = new THREE.Group();
  terrain.name = 'MarsTerrain';
  const levelData = [];
  for (let k = 0; k < GRID_LEVELS; k++) {
    const L = buildTerrainLevel(k);
    levelData.push({ c: L.c, E: L.E, heights: L.heights });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(L.pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(L.nor, 3));
    g.setIndex(new THREE.BufferAttribute(L.index, 1));
    g.computeBoundingSphere();
    const mesh = new THREE.Mesh(g, terrainMat);
    mesh.receiveShadow = true;
    mesh.castShadow = k <= 3;
    mesh.matrixAutoUpdate = false;
    terrain.add(mesh);
    disposables.push(g);
    if (k % 3 === 2) await tick();
  }
  scene.add(terrain);

  // height of the rendered mesh (bilinear in the ring grid), for placing rocks on the visible surface
  const V = GRID_N + 1;
  function meshHeight(x, z) {
    const mx = Math.max(Math.abs(x), Math.abs(z));
    let k = 0;
    while (k < GRID_LEVELS - 1 && mx > levelData[k].E) k++;
    const { c, heights } = levelData[k];
    const fx = clamp(x / c + GRID_N / 2, 0, GRID_N - 1e-4), fz = clamp(z / c + GRID_N / 2, 0, GRID_N - 1e-4);
    const i = Math.floor(fx), j = Math.floor(fz), tx = fx - i, tz = fz - j;
    const h = (a, b) => heights[b * V + a];
    return lerp(lerp(h(i, j), h(i + 1, j), tx), lerp(h(i, j + 1), h(i + 1, j + 1), tx), tz);
  }

  // --- rocks ---
  await tick();
  const models = await loadRockModels(m, assetBase, aniso);
  const rockGroup = new THREE.Group();
  rockGroup.name = 'MarsRocks';
  buildRocks(generateRocks(meshHeight), models, rockGroup, disposables);
  scene.add(rockGroup);
  for (const mdl of models) disposables.push(mdl.geo, mdl.mat);

  // --- dust ---
  const sunRadiance = sunColor.clone().multiplyScalar(sunIntensity / Math.PI);
  const ambient = new THREE.Color().setRGB(0.45, 0.33, 0.24, THREE.LinearSRGBColorSpace);
  const dust = createDust({ scene, heightAt: (x, z) => terrainHeight(x, z), sunDirection, sunRadiance, ambient, terrainUniforms });

  // --- shadow follow ---
  const ex = new THREE.Vector3(), ey = new THREE.Vector3(), centre = new THREE.Vector3();
  const UP = new THREE.Vector3(0, 1, 0);
  function updateShadow(focus) {
    const alt = Math.max(0, focus.y - terrainHeight(focus.x, focus.z));
    // lateral half-size grows slowly with altitude; quantised so the texel grid stays stable
    const S = Math.ceil(clamp(28 + 0.07 * alt, 28, 160) / 8) * 8;
    // depth must cover the lander-to-ground distance along the sun ray
    const D = Math.max(S * 1.5, alt / Math.max(Math.sin(el), 0.2) + 30);
    const texel = (2 * S) / shadowMapSize;
    ex.crossVectors(UP, sunDirection).normalize();
    ey.crossVectors(sunDirection, ex);
    const a = Math.round(focus.dot(ex) / texel) * texel;
    const b = Math.round(focus.dot(ey) / texel) * texel;
    centre.copy(ex).multiplyScalar(a).addScaledVector(ey, b).addScaledVector(sunDirection, focus.dot(sunDirection));
    sun.target.position.copy(centre);
    sun.position.copy(centre).addScaledVector(sunDirection, D);
    const cam = sun.shadow.camera;
    if (cam.right !== S || cam.far !== 2 * D) {
      cam.left = -S; cam.right = S; cam.top = S; cam.bottom = -S;
      cam.near = 0.5; cam.far = 2 * D;
      cam.updateProjectionMatrix();
    }
    sun.shadow.normalBias = 2.5 * texel;
  }
  updateShadow(new THREE.Vector3(0, 1, 0));

  function update(dt, { camera: cam = camera, focus } = {}) {
    if (cam) {
      sky.position.copy(cam.position);
      sky.scale.setScalar(Math.min(cam.far * 0.4, 80000));
    }
    if (focus) updateShadow(focus);
  }

  function dispose() {
    scene.remove(terrain, rockGroup, sky, sun, sun.target, hemi);
    if (scene.fog === ownFog) scene.fog = null;
    if (scene.environment === envRT.texture) scene.environment = null;
    dust.dispose();
    sun.shadow.dispose();
    for (const d of disposables) if (d && d.dispose) d.dispose();
  }

  return {
    heightAt: (x, z) => terrainHeight(x, z),
    sunDirection,
    update,
    dust,
    dispose,
    sun,
    fogColor,
    terrainUniforms,
  };
}
