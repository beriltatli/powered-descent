// Procedural Mars powered-descent lander with hydrazine monoprop plumes (Three.js r170).
// Local frame: origin at the centre of the footpad contact plane, +Y = vehicle axis (net thrust direction).
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const DEG = Math.PI / 180;
const N_ENG = 6;
const CANT = 27 * DEG;           // engine cant from the vehicle axis
const HEX_R = 1.7;               // deck circumradius (3.4 m corner to corner)
const APO = HEX_R * Math.cos(30 * DEG);
const DECK_Y0 = 0.95;
const DECK_Y1 = 1.40;
const PAD_R = 2.05;              // radius of the circle through the footpad centres
const PAD_RADIUS = 0.21;
const ENG_R = 1.2;               // throat radial position
const ENG_Y = 0.93;              // throat height
const NOZZLE_LEN = 0.42;
const PLUME_R0 = 0.13;

const Y_AXIS = new THREE.Vector3(0, 1, 0);
const DOWN = new THREE.Vector3(0, -1, 0);

// ---------------------------------------------------------------- small helpers

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function polar(r, a, y) {
  return new THREE.Vector3(r * Math.cos(a), y, r * Math.sin(a));
}

const _e = new THREE.Euler();
const _q = new THREE.Quaternion();
const _one = new THREE.Vector3(1, 1, 1);

// position + euler rotation + scale -> Matrix4
function mx(px, py, pz, rx = 0, ry = 0, rz = 0, sx = 1, sy = sx, sz = sx) {
  _q.setFromEuler(_e.set(rx, ry, rz));
  return new THREE.Matrix4().compose(new THREE.Vector3(px, py, pz), _q, new THREE.Vector3(sx, sy, sz));
}

// unit cylinder (height 1 along Y) stretched between a and b
function segMatrix(a, b, radius) {
  const d = b.clone().sub(a);
  const len = d.length();
  const q = new THREE.Quaternion().setFromUnitVectors(Y_AXIS, d.normalize());
  return new THREE.Matrix4().compose(a.clone().add(b).multiplyScalar(0.5), q, new THREE.Vector3(radius, len, radius));
}

// Collects geometry per material key, then merges each key into a single mesh.
class Batch {
  constructor() { this.parts = new Map(); }
  add(key, geometry, matrix) {
    const g = geometry.index ? geometry.toNonIndexed() : geometry.clone();
    for (const name of Object.keys(g.attributes)) {
      if (name !== 'position' && name !== 'normal' && name !== 'uv') g.deleteAttribute(name);
    }
    g.clearGroups();
    if (matrix) g.applyMatrix4(matrix);
    if (!this.parts.has(key)) this.parts.set(key, []);
    this.parts.get(key).push(g);
  }
  build(materials, parent) {
    for (const [key, list] of this.parts) {
      const merged = mergeGeometries(list, false);
      list.forEach((g) => g.dispose());
      const mesh = new THREE.Mesh(merged, materials[key]);
      mesh.name = `lander_${key}`;
      mesh.castShadow = mesh.receiveShadow = true;
      parent.add(mesh);
    }
  }
}

function hexShape(R) {
  const s = new THREE.Shape();
  for (let k = 0; k < 6; k++) {
    const x = R * Math.cos(k * 60 * DEG), y = R * Math.sin(k * 60 * DEG);
    if (k === 0) s.moveTo(x, y); else s.lineTo(x, y);
  }
  s.closePath();
  return s;
}

function hexSlab(R, y0, y1, bevel) {
  const depth = y1 - y0 - 2 * bevel;
  const g = new THREE.ExtrudeGeometry(hexShape(R - bevel), {
    depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 2, curveSegments: 1,
  });
  g.translate(0, 0, bevel);
  g.rotateX(-Math.PI / 2);
  g.translate(0, y0, 0);
  return g;
}

function ringGeo(R, tube, radial = 8, tubular = 40) {
  const g = new THREE.TorusGeometry(R, tube, radial, tubular);
  g.rotateX(Math.PI / 2);
  return g;
}

// Box whose front (+Z) face bulges like a stuffed thermal blanket. yOff positions this piece within the pillow.
function bulge(geo, W, H, yOff, amp) {
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    if (p.getZ(i) <= 0) continue;
    const x = p.getX(i), y = p.getY(i) + yOff;
    const cx = Math.cos(Math.PI * THREE.MathUtils.clamp(x / W, -0.5, 0.5));
    const cy = Math.cos(Math.PI * THREE.MathUtils.clamp(y / H, -0.5, 0.5));
    p.setZ(i, p.getZ(i) + amp * cx * cy);
  }
  geo.computeVertexNormals();
  return geo;
}

function pillow(W, H, D, amp) {
  return bulge(new THREE.BoxGeometry(W, H, D, 18, 8, 1), W, H, 0, amp);
}

// ---------------------------------------------------------------- procedural textures

function canvasOf(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

// Tileable crumpled-foil normal map: flat facets (jittered Voronoi cells) at two scales.
function makeFoilNormalMap(size = 256) {
  const cv = canvasOf(size, size);
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(size, size);
  const layers = [{ g: 9, amp: 0.55, seed: 11 }, { g: 26, amp: 0.28, seed: 23 }, { g: 70, amp: 0.1, seed: 5 }].map((L) => {
    const rnd = mulberry32(L.seed);
    const n = L.g * L.g;
    const c = { g: L.g, amp: L.amp, px: new Float32Array(n), py: new Float32Array(n), sx: new Float32Array(n), sy: new Float32Array(n) };
    for (let i = 0; i < n; i++) {
      c.px[i] = rnd(); c.py[i] = rnd();
      c.sx[i] = rnd() * 2 - 1; c.sy[i] = rnd() * 2 - 1;
    }
    return c;
  });
  const slope = new Float32Array(2);
  const cellSlope = (L, u, v) => {
    const cx = Math.floor(u), cy = Math.floor(v);
    let best = 1e9, bi = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const wx = ((cx + dx) % L.g + L.g) % L.g, wy = ((cy + dy) % L.g + L.g) % L.g;
        const i = wy * L.g + wx;
        const ddx = cx + dx + L.px[i] - u, ddy = cy + dy + L.py[i] - v;
        const d = ddx * ddx + ddy * ddy;
        if (d < best) { best = d; bi = i; }
      }
    }
    slope[0] = L.sx[bi] * L.amp; slope[1] = L.sy[bi] * L.amp;
  };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let sx = 0, sy = 0;
      for (const L of layers) {
        cellSlope(L, (x / size) * L.g, (y / size) * L.g);
        sx += slope[0]; sy += slope[1];
      }
      const inv = 1 / Math.hypot(sx, sy, 1);
      const o = (y * size + x) * 4;
      img.data[o] = (-sx * inv * 0.5 + 0.5) * 255;
      img.data[o + 1] = (sy * inv * 0.5 + 0.5) * 255;
      img.data[o + 2] = (inv * 0.5 + 0.5) * 255;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
}

function makeSolarTexture() {
  const cv = canvasOf(256, 256);
  const c = cv.getContext('2d');
  const grad = c.createRadialGradient(100, 90, 10, 128, 128, 190);
  grad.addColorStop(0, '#1d2f5c'); grad.addColorStop(1, '#0c1630');
  c.fillStyle = grad; c.fillRect(0, 0, 256, 256);
  c.strokeStyle = '#8f9bb0'; c.lineWidth = 2;
  for (let i = 0; i <= 8; i++) {
    c.beginPath(); c.moveTo(i * 32, 0); c.lineTo(i * 32, 256); c.stroke();
    c.beginPath(); c.moveTo(0, i * 32); c.lineTo(256, i * 32); c.stroke();
  }
  c.strokeStyle = '#c9ced8'; c.lineWidth = 1;
  for (let i = 0; i < 8; i++) {
    c.beginPath(); c.moveTo(i * 32 + 16, 0); c.lineTo(i * 32 + 16, 256); c.stroke();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

// Nozzle colour (heat-tinted throat) and emissive weighting. Lathe v: 0 at exit, 1 at chamber.
function makeNozzleMaps() {
  const col = canvasOf(4, 64), em = canvasOf(4, 64);
  const a = col.getContext('2d'), b = em.getContext('2d');
  // canvas top = v 1 (chamber/throat), bottom = v 0 (exit)
  const g1 = a.createLinearGradient(0, 0, 0, 64);
  g1.addColorStop(0, '#8d8f94'); g1.addColorStop(0.18, '#9b8a62'); g1.addColorStop(0.3, '#5f6f90');
  g1.addColorStop(0.5, '#5a5c61'); g1.addColorStop(1, '#43454a');
  a.fillStyle = g1; a.fillRect(0, 0, 4, 64);
  const g2 = b.createLinearGradient(0, 0, 0, 64);
  g2.addColorStop(0, '#000000'); g2.addColorStop(0.2, '#707070'); g2.addColorStop(1, '#ffffff');
  b.fillStyle = g2; b.fillRect(0, 0, 4, 64);
  const map = new THREE.CanvasTexture(col);
  map.colorSpace = THREE.SRGBColorSpace;
  const emissiveMap = new THREE.CanvasTexture(em);
  emissiveMap.colorSpace = THREE.SRGBColorSpace;
  return { map, emissiveMap };
}

// ---------------------------------------------------------------- materials

function makeMaterials(foilNormal, nozzleMaps) {
  const std = (o) => new THREE.MeshStandardMaterial(o);
  const nGold = foilNormal.clone(); nGold.repeat.set(3, 1.6); nGold.needsUpdate = true;
  const nSilver = foilNormal.clone(); nSilver.repeat.set(2.4, 1.4); nSilver.needsUpdate = true;
  return {
    white: new THREE.MeshPhysicalMaterial({ color: 0xe9e7e1, metalness: 0.05, roughness: 0.52, clearcoat: 0.15, clearcoatRoughness: 0.5 }),
    dark: std({ color: 0x2a2d33, metalness: 0.65, roughness: 0.45 }),
    ti: std({ color: 0x9a9ca1, metalness: 0.85, roughness: 0.32 }),
    steel: std({ color: 0xd0d3d8, metalness: 0.9, roughness: 0.22 }),
    alu: std({ color: 0xbcc0c6, metalness: 0.8, roughness: 0.4 }),
    gold: std({ color: 0xe0b04a, metalness: 0.8, roughness: 0.34, normalMap: nGold, normalScale: new THREE.Vector2(1, 1) }),
    silver: std({ color: 0xe6e9ed, metalness: 0.8, roughness: 0.28, normalMap: nSilver, normalScale: new THREE.Vector2(0.9, 0.9) }),
    tape: std({ color: 0xe0a020, metalness: 0.2, roughness: 0.45 }),
    cable: std({ color: 0x141516, metalness: 0.0, roughness: 0.6 }),
    glass: std({ color: 0x05070a, metalness: 0.2, roughness: 0.08 }),
    red: std({ color: 0xb02418, metalness: 0.0, roughness: 0.6 }),
    solar: std({ color: 0xffffff, map: makeSolarTexture(), metalness: 0.5, roughness: 0.3 }),
    _foil: { gold: nGold, silver: nSilver },
    _nozzle: nozzleMaps,
  };
}

// ---------------------------------------------------------------- plume shader

const PLUME_VERT = /* glsl */`
uniform float uLen;
uniform float uR0;
uniform float uSpread;
varying vec3 vN;
varying vec3 vV;
varying float vT;
varying vec2 vDir;
void main() {
  float t = position.y;                                   // 0 at nozzle exit, 1 at plume tip
  vec2 d = normalize(position.xz + 1e-6);
  float s = t + 0.02;
  float r = uR0 + uLen * uSpread * (pow(s, 0.75) - pow(0.02, 0.75));  // broad, under-expanded spread
  float drdt = uLen * uSpread * 0.75 * pow(s, -0.25);
  vec3 p = vec3(d.x * r, -t * uLen, d.y * r);
  vec3 n = normalize(vec3(d.x * uLen, drdt, d.y * uLen));
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vN = normalize(normalMatrix * n);
  vV = -mv.xyz;
  vT = t;
  vDir = d;
  gl_Position = projectionMatrix * mv;
}`;

const PLUME_FRAG = /* glsl */`
uniform float uTime;
uniform float uLen;
uniform float uIntensity;
varying vec3 vN;
varying vec3 vV;
varying float vT;
varying vec2 vDir;

float hash(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 3; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; }
  return s;
}

void main() {
  float t = clamp(vT, 0.0, 1.0);
  float z = t * uLen;                                     // metres from the exit plane
  float n = fbm(vec3(vDir * 1.6, z * 0.8 - uTime * 6.0));
  float f = abs(dot(normalize(vN), normalize(vV)));       // thicker through the middle of the volume

  float core = exp(-z * 1.3);                             // hot glow right at the exit
  float body = pow(1.0 - t, 1.6);                         // very faint, nearly transparent bulk
  float shimmer = 0.5 + 0.5 * sin(z * 7.0 - uTime * 16.0 + n * 5.0);
  float a = core * (0.5 + 0.9 * n) * 0.55 + body * (0.03 + 0.07 * n) * (0.6 + 0.4 * shimmer);
  // smoothstep needs edge0 < edge1 (reversed edges are undefined in GLSL and render as dark rings)
  a *= pow(f, 1.2) * (1.0 - smoothstep(0.7, 1.0, t)) * uIntensity;
  a = clamp(a, 0.0, 4.0);

  vec3 hot = vec3(0.70, 0.82, 1.0);                       // bluish-white at the throat
  vec3 warm = vec3(1.0, 0.86, 0.62);                      // pale yellow-white
  vec3 cool = vec3(0.85, 0.82, 0.80);
  vec3 col = mix(hot, warm, smoothstep(0.0, 0.5, z));
  col = mix(col, cool, smoothstep(0.5, 2.5, z));
  gl_FragColor = vec4(col * a, a);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

function makePlumeMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 }, uLen: { value: 3 }, uR0: { value: PLUME_R0 },
      uSpread: { value: 0.45 }, uIntensity: { value: 0 },
    },
    vertexShader: PLUME_VERT,
    fragmentShader: PLUME_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });
}

// ---------------------------------------------------------------- vehicle sections

function buildEngineParts(B, E) {
  // E: engine frame matrix (throat at origin, exhaust along local -Y)
  const m = (local) => E.clone().multiply(local);
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 16, 1);
  const box = new THREE.BoxGeometry(1, 1, 1);
  B.add('ti', cyl, m(mx(0, 0.21, 0, 0, 0, 0, 0.066, 0.18, 0.066)));      // thrust chamber
  B.add('dark', cyl, m(mx(0, 0.34, 0, 0, 0, 0, 0.088, 0.07, 0.088)));    // injector head
  B.add('dark', cyl, m(mx(0, 0.1, 0, 0, 0, 0, 0.1, 0.022, 0.1)));        // flange
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    B.add('steel', new THREE.CylinderGeometry(1, 1, 1, 6, 1), m(mx(0.083 * Math.cos(a), 0.1, 0.083 * Math.sin(a), 0, 0, 0, 0.011, 0.028, 0.011)));
  }
  B.add('dark', box, m(mx(0.1, 0.24, 0, 0, 0, 0, 0.06, 0.08, 0.05)));    // propellant valve
  B.add('ti', cyl, m(mx(0.15, 0.24, 0, 0, 0, Math.PI / 2, 0.025, 0.05, 0.025)));
  B.add('ti', ringGeo(0.128, 0.008, 8, 32), m(mx(0, -NOZZLE_LEN, 0)));   // exit lip
  B.add('ti', ringGeo(0.087, 0.006, 6, 32), m(mx(0, -0.2, 0)));          // stiffener rings
  B.add('ti', ringGeo(0.113, 0.006, 6, 32), m(mx(0, -0.33, 0)));
}

function buildDeck(B) {
  const yc = (DECK_Y0 + DECK_Y1) / 2;
  B.add('white', hexSlab(HEX_R, DECK_Y0 + 0.02, DECK_Y1 - 0.02, 0.025));
  B.add('dark', hexSlab(HEX_R - 0.1, DECK_Y0 - 0.06, DECK_Y0 + 0.02, 0.012));   // bottom plate
  B.add('white', hexSlab(HEX_R - 0.12, DECK_Y1 - 0.02, DECK_Y1 + 0.03, 0.012)); // top plate
  B.add('dark', ringGeo(1.05, 0.02, 8, 48), mx(0, DECK_Y1 + 0.03, 0));
  // radial ribs along the top plate
  for (let k = 0; k < 6; k++) {
    const a = k * 60 * DEG;
    B.add('white', new THREE.BoxGeometry(1, 1, 1), mx(0.8 * Math.cos(a), DECK_Y1 + 0.06, 0.8 * Math.sin(a), 0, -a, 0, 1.2, 0.06, 0.05));
  }
  // thermal blankets on the six faces; leg faces get a narrower silver blanket
  for (let j = 0; j < 6; j++) {
    const a = (30 + 60 * j) * DEG;
    const legFace = j % 2 === 0;
    const W = legFace ? 0.8 : 1.45, H = 0.38;
    const place = (extraZ) => mx(Math.cos(a) * (APO + extraZ), yc, Math.sin(a) * (APO + extraZ), 0, Math.PI / 2 - a, 0);
    B.add(legFace ? 'silver' : 'gold', pillow(W, H, 0.05, 0.035), place(0));
    // kapton tape: horizontal run and two verticals, following the bulge
    const tz = 0.025 + 0.005;
    const hT = bulge(new THREE.BoxGeometry(W + 0.02, 0.035, 0.01, 18, 1, 1), W, H, 0, 0.035);
    B.add('tape', hT, place(0).multiply(mx(0, 0, tz)));
    for (const off of [-W * 0.32, W * 0.32]) {
      const vT = new THREE.BoxGeometry(0.035, H + 0.02, 0.01, 1, 8, 1);
      vT.translate(off, 0, 0);
      B.add('tape', bulge(vT, W, H, 0, 0.035), place(0).multiply(mx(0, 0, tz)));
    }
    if (legFace) {
      // hip bracket plate and clevis
      B.add('dark', new RoundedBoxGeometry(0.5, 0.46, 0.05, 2, 0.01), place(0.07));
      for (const sx of [-0.2, 0.2]) for (const sy of [-0.17, 0.17]) {
        B.add('steel', new THREE.CylinderGeometry(1, 1, 1, 6, 1), place(0.1).multiply(mx(sx, sy, 0, Math.PI / 2, 0, 0, 0.016, 0.014, 0.016)));
      }
    }
  }
  // perimeter bolt heads, top and bottom
  const hexV = (R, y, k) => polar(R, k * 60 * DEG, y);
  const bolt = new THREE.CylinderGeometry(1, 1, 1, 6, 1);
  for (let k = 0; k < 6; k++) {
    for (const [R, y, s] of [[HEX_R - 0.12, DECK_Y1 + 0.03, 1], [HEX_R - 0.1, DECK_Y0 - 0.06, -1]]) {
      const p0 = hexV(R, y, k), p1 = hexV(R, y, k + 1);
      for (let i = 1; i <= 9; i++) {
        const p = p0.clone().lerp(p1, i / 10);
        B.add('steel', bolt, mx(p.x, p.y + s * 0.006, p.z, 0, 0, 0, 0.02, 0.014, 0.02));
      }
    }
  }
}

function buildUnderside(B) {
  const box = new THREE.BoxGeometry(1, 1, 1);
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 16, 1);
  // radar altimeter / velocimeter: slotted patch antenna, framed
  const y = DECK_Y0 - 0.06;
  B.add('dark', new RoundedBoxGeometry(0.62, 0.07, 0.62, 2, 0.015), mx(0, y - 0.035, 0));
  for (let i = -3; i <= 3; i++) {
    B.add('cable', box, mx(0, y - 0.072, i * 0.075, 0, 0, 0, 0.5, 0.006, 0.014));
  }
  for (let k = 0; k < 3; k++) {
    const a = (30 + 120 * k) * DEG;
    const horn = new THREE.CylinderGeometry(0.065, 0.025, 0.11, 14, 1, true);
    const p = polar(0.42, a, y - 0.04);
    B.add('alu', horn, new THREE.Matrix4().compose(p, new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(Math.cos(a + Math.PI / 2), 0, Math.sin(a + Math.PI / 2)), 0.15), _one));
    B.add('cable', cyl, mx(p.x, p.y - 0.04, p.z, 0, 0, 0, 0.06, 0.004, 0.06));
  }
  B.add('glass', cyl, mx(0.1, y - 0.03, 0.62, 0, 0, 0, 0.04, 0.05, 0.04));   // descent camera
  B.add('dark', cyl, mx(0.1, y - 0.015, 0.62, 0, 0, 0, 0.055, 0.04, 0.055));
  // propellant feed manifold and lines to each engine boss
  B.add('ti', ringGeo(0.62, 0.016, 8, 64), mx(0, y + 0.02, 0));
  for (let k = 0; k < N_ENG; k++) {
    const a = k * 60 * DEG;
    B.add('ti', cyl, segMatrix(polar(0.62, a, y + 0.02), polar(ENG_R - 0.08, a, ENG_Y + 0.0), 0.016));
    // engine mounting boss + bolts
    B.add('dark', cyl, mx(ENG_R * Math.cos(a), DECK_Y0 - 0.075, ENG_R * Math.sin(a), 0, 0, 0, 0.16, 0.04, 0.16));
    for (let i = 0; i < 6; i++) {
      const b = i / 6 * Math.PI * 2;
      B.add('steel', new THREE.CylinderGeometry(1, 1, 1, 6, 1), mx(ENG_R * Math.cos(a) + 0.12 * Math.cos(b), DECK_Y0 - 0.1, ENG_R * Math.sin(a) + 0.12 * Math.sin(b), 0, 0, 0, 0.011, 0.012, 0.011));
    }
  }
}

function buildLegs(B) {
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 14, 1);
  const sph = new THREE.SphereGeometry(1, 16, 12);
  const padProfile = [[0, 0], [0.18, 0], [PAD_RADIUS, 0.014], [0.205, 0.034], [0.12, 0.058], [0.055, 0.09], [0, 0.098]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  const padGeo = new THREE.LatheGeometry(padProfile, 28);
  for (let j = 0; j < 3; j++) {
    const a = (30 + 120 * j) * DEG;
    const n = polar(1, a, 0);
    const hip = polar(APO + 0.11, a, 1.12);
    const pad = polar(PAD_R, a, 0);
    const ball = polar(PAD_R - 0.03, a, 0.17);
    const along = (t) => hip.clone().lerp(ball, t);
    // primary strut: painted upper tube, polished inner stage, collar
    B.add('white', cyl, segMatrix(hip, along(0.56), 0.05));
    B.add('steel', cyl, segMatrix(along(0.5), ball, 0.034));
    B.add('dark', cyl, segMatrix(along(0.54), along(0.6), 0.058));
    B.add('dark', cyl, segMatrix(along(0.0), along(0.07), 0.06));
    // crushable secondary struts from the adjacent deck corners
    const q = along(0.62);
    for (const s of [-1, 1]) {
      const v = polar(HEX_R - 0.05, a + s * 30 * DEG, 1.02);
      const mid = v.clone().lerp(q, 0.55);
      B.add('white', cyl, segMatrix(v, mid, 0.03));
      B.add('ti', cyl, segMatrix(mid, q, 0.022));
      B.add('dark', cyl, segMatrix(mid.clone().lerp(v, 0.04), mid.clone().lerp(q, 0.1), 0.036)); // honeycomb stage
      B.add('steel', sph, mx(q.x, q.y, q.z, 0, 0, 0, 0.034));
      B.add('steel', sph, mx(v.x, v.y, v.z, 0, 0, 0, 0.032));
    }
    // footpad
    B.add('steel', sph, mx(ball.x, ball.y, ball.z, 0, 0, 0, 0.05));
    B.add('alu', padGeo, mx(pad.x, 0, pad.z));
    B.add('dark', cyl, mx(pad.x, 0.1, pad.z, 0, 0, 0, 0.03, 0.05, 0.03));
    for (let i = 0; i < 6; i++) {
      const b = i / 6 * Math.PI * 2;
      B.add('dark', new THREE.BoxGeometry(1, 1, 1), mx(pad.x + 0.12 * Math.cos(b), 0.05, pad.z + 0.12 * Math.sin(b), 0, -b, 0, 0.12, 0.05, 0.012));
    }
    // harness along the primary strut
    const side = new THREE.Vector3(-n.z, 0, n.x).multiplyScalar(0.065);
    const pts = [hip.clone().add(side), along(0.3).add(side).add(new THREE.Vector3(0, 0.02, 0)), along(0.6).add(side), ball.clone().add(side)];
    B.add('cable', new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 16, 0.009, 6, false));
  }
}

function buildTanksAndTop(B) {
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 16, 1);
  const sph = new THREE.SphereGeometry(1, 40, 24);
  const box = new THREE.BoxGeometry(1, 1, 1);
  const top = DECK_Y1 + 0.05;
  const R_T = 0.32, TR = 0.72, TY = top + 0.06 + R_T - 0.02;
  for (let k = 0; k < 3; k++) {
    const a = (k * 120) * DEG;
    const c = polar(TR, a, TY);
    B.add('ti', sph, mx(c.x, c.y, c.z, 0, 0, 0, R_T));
    B.add('steel', ringGeo(R_T, 0.007, 6, 48), mx(c.x, c.y, c.z));                                  // equatorial weld
    B.add('steel', ringGeo(R_T * 0.77, 0.005, 6, 40), mx(c.x, c.y + R_T * 0.64, c.z));
    B.add('dark', cyl, mx(c.x, c.y + R_T - 0.005, c.z, 0, 0, 0, 0.065, 0.04, 0.065));              // polar boss
    B.add('steel', cyl, mx(c.x, c.y + R_T + 0.035, c.z, 0, 0, 0, 0.025, 0.05, 0.025));
    B.add('dark', cyl, mx(c.x, top + 0.03, c.z, 0, 0, 0, 0.19, 0.06, 0.19));                        // cradle cup
    B.add('steel', ringGeo(0.19, 0.008, 6, 32), mx(c.x, top + 0.06, c.z));
    // fill/drain valve on the shoulder
    const vd = polar(R_T * 0.8, a + 0.9, 0);
    B.add('dark', cyl, mx(c.x + vd.x, c.y + R_T * 0.6, c.z + vd.z, 0, 0, 0, 0.026, 0.05, 0.026));
    B.add('red', cyl, mx(c.x + vd.x, c.y + R_T * 0.6 + 0.032, c.z + vd.z, 0, 0, 0, 0.02, 0.014, 0.02));
    // tension struts from the deck up to the tank equator
    const t = new THREE.Vector3(-Math.sin(a), 0, Math.cos(a));
    for (const s of [-1, 1]) {
      const lo = c.clone().addScaledVector(t, s * 0.34); lo.y = top;
      const hi = c.clone().addScaledVector(t, s * R_T * 0.97); hi.y = c.y - 0.06;
      B.add('alu', cyl, segMatrix(lo, hi, 0.014));
    }
    // feed line from tank to the central pressurant bottle
    B.add('ti', cyl, segMatrix(polar(TR - R_T * 0.95, a, c.y - 0.1), polar(0.2, a, c.y - 0.1), 0.01));
  }
  // central pressurant bottle wrapped in MLI
  B.add('gold', sph, mx(0, top + 0.26, 0, 0, 0, 0, 0.21));
  B.add('tape', ringGeo(0.212, 0.008, 6, 40), mx(0, top + 0.26, 0));
  B.add('dark', cyl, mx(0, top + 0.04, 0, 0, 0, 0, 0.14, 0.08, 0.14));
  // high-gain antenna on a mast
  const mastTop = top + 0.26 + 0.21 + 0.03;
  B.add('alu', cyl, segMatrix(new THREE.Vector3(0, top + 0.4, 0), new THREE.Vector3(0, mastTop + 0.1, 0), 0.02));
  B.add('dark', cyl, mx(0, mastTop + 0.1, 0, 0, 0, 0, 0.04, 0.05, 0.04));
  const dish = [];
  for (let i = 0; i <= 12; i++) { const r = i * 0.0135; dish.push(new THREE.Vector2(r, 3.2 * r * r)); }
  const dishGeo = new THREE.LatheGeometry(dish, 32);
  const tilt = new THREE.Matrix4().compose(new THREE.Vector3(0, mastTop + 0.13, 0), new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.42, 0.5, 0)), _one);
  B.add('white', dishGeo, tilt);
  B.add('white', new THREE.LatheGeometry(dish.slice().reverse().map((p) => new THREE.Vector2(p.x * 0.999, p.y - 0.004)), 32), tilt);
  B.add('alu', cyl, tilt.clone().multiply(mx(0, 0.1, 0, 0, 0, 0, 0.006, 0.2, 0.006)));
  B.add('dark', cyl, tilt.clone().multiply(mx(0, 0.2, 0, 0, 0, 0, 0.022, 0.04, 0.022)));
  // UHF whip
  B.add('dark', cyl, mx(-0.9, DECK_Y1 + 0.2, -1.0, 0.05, 0, 0.05, 0.006, 0.38, 0.006));
  B.add('dark', cyl, mx(-0.9, DECK_Y1 + 0.04, -1.0, 0, 0, 0, 0.03, 0.04, 0.03));

  // round deck solar panels between the tanks
  for (let k = 0; k < 3; k++) {
    const a = (60 + 120 * k) * DEG;
    const c = polar(0.98, a, DECK_Y1 + 0.085);
    B.add('solar', cyl, mx(c.x, c.y, c.z, 0, 0, 0, 0.36, 0.022, 0.36));
    B.add('alu', ringGeo(0.365, 0.014, 8, 48), mx(c.x, c.y + 0.004, c.z));
    for (const s of [0, 1, 2]) {
      const b = (s / 3) * Math.PI * 2 + 0.4;
      B.add('dark', cyl, mx(c.x + 0.25 * Math.cos(b), DECK_Y1 + 0.055, c.z + 0.25 * Math.sin(b), 0, 0, 0, 0.012, 0.05, 0.012));
    }
  }
  // avionics boxes on the non-leg faces, MLI wrapped or bare
  const wraps = ['gold', 'silver', 'gold'];
  for (let k = 0; k < 3; k++) {
    const a = (90 + 120 * k) * DEG;
    const c = polar(1.22, a, DECK_Y1 + 0.03 + 0.11);
    const rot = new THREE.Matrix4().makeRotationY(Math.PI / 2 - a);
    const place = (px, py, pz) => new THREE.Matrix4().makeTranslation(c.x, c.y, c.z).multiply(rot).multiply(mx(px, py, pz));
    B.add(wraps[k], new RoundedBoxGeometry(0.52, 0.22, 0.34, 3, 0.025), place(0, 0, 0));
    B.add('tape', new RoundedBoxGeometry(0.535, 0.03, 0.352, 2, 0.012), place(0, 0.02, 0));
    B.add('dark', box, place(0.2, 0.0, 0.18).multiply(mx(0, 0, 0, 0, 0, 0, 0.1, 0.08, 0.02)));
    B.add('dark', cyl, place(-0.12, 0.12, 0).multiply(mx(0, 0, 0, 0, 0, 0, 0.03, 0.03, 0.03)));
    // harness from the box to the deck centre
    const p0 = c.clone().add(new THREE.Vector3(0, 0.0, 0));
    const p1 = polar(0.95, a + 0.12, DECK_Y1 + 0.08);
    const p2 = polar(0.5, a + 0.2, DECK_Y1 + 0.08);
    B.add('cable', new THREE.TubeGeometry(new THREE.CatmullRomCurve3([p0.clone().setY(p0.y - 0.05), p1, p2]), 20, 0.012, 6, false));
    B.add('cable', new THREE.TubeGeometry(new THREE.CatmullRomCurve3([p0.clone().setY(p0.y - 0.08).add(new THREE.Vector3(0.05, 0, 0)), p1.clone().add(new THREE.Vector3(0.03, 0.01, 0.03)), p2.clone().add(new THREE.Vector3(0, 0, 0.04))]), 20, 0.008, 6, false));
  }
  // small dark units above the leg faces + warning tag
  for (let k = 0; k < 3; k++) {
    const a = (30 + 120 * k) * DEG;
    const c = polar(1.3, a, DECK_Y1 + 0.03 + 0.07);
    const rot = new THREE.Matrix4().makeRotationY(Math.PI / 2 - a);
    B.add('dark', new RoundedBoxGeometry(0.34, 0.14, 0.2, 2, 0.015), new THREE.Matrix4().makeTranslation(c.x, c.y, c.z).multiply(rot));
    B.add('white', cyl, new THREE.Matrix4().makeTranslation(c.x, c.y + 0.1, c.z).multiply(rot).multiply(mx(0.08, 0, 0, 0, 0, 0, 0.04, 0.06, 0.04)));
  }
  B.add('red', box, mx(1.48, DECK_Y1 + 0.06, 0.2, 0.2, 0, 0.1, 0.12, 0.003, 0.03));
}

function buildRCS(B) {
  const cone = new THREE.CylinderGeometry(0.028, 0.012, 0.07, 10, 1);
  const rb = new RoundedBoxGeometry(0.15, 0.13, 0.15, 2, 0.015);
  for (let k = 0; k < 6; k++) {
    const a = k * 60 * DEG;
    const c = polar(HEX_R - 0.02, a, DECK_Y1 - 0.12);
    B.add('dark', rb, mx(c.x, c.y, c.z, 0, -a, 0));
    const out = new THREE.Vector3(Math.cos(a), 0, Math.sin(a));
    const tan = new THREE.Vector3(-Math.sin(a), 0, Math.cos(a));
    const dirs = [out, tan, tan.clone().negate(), new THREE.Vector3(0, 1, 0)];
    for (const d of dirs) {
      const p = c.clone().addScaledVector(d, 0.1);
      const q = new THREE.Quaternion().setFromUnitVectors(Y_AXIS, d);
      B.add('ti', cone, new THREE.Matrix4().compose(p, q, _one));
    }
  }
}

// ---------------------------------------------------------------- factory

export function createLander({ manifest = null, textureLoader = null } = {}) {
  const object = new THREE.Group();
  object.name = 'lander';

  const foilNormal = makeFoilNormalMap();
  const nozzleMaps = makeNozzleMaps();
  const mats = makeMaterials(foilNormal, nozzleMaps);
  const foil = mats._foil;
  delete mats._foil;
  delete mats._nozzle;

  const body = new THREE.Group();
  body.name = 'lander_body';
  object.add(body);

  const B = new Batch();
  buildDeck(B);
  buildUnderside(B);
  buildLegs(B);
  buildTanksAndTop(B);
  buildRCS(B);

  // engines
  const engines = [];
  const exitLocal = [];
  const dirLocal = [];
  const bellProfile = new THREE.SplineCurve([
    [0.128, -NOZZLE_LEN], [0.118, -0.34], [0.1, -0.25], [0.075, -0.15], [0.05, -0.07],
    [0.036, -0.02], [0.032, 0.0], [0.034, 0.015], [0.05, 0.07], [0.066, 0.12],
  ].map(([r, y]) => new THREE.Vector2(r, y))).getPoints(40);
  const bellGeo = new THREE.LatheGeometry(bellProfile, 32);
  const plumeGeo = new THREE.CylinderGeometry(1, 1, 1, 28, 20, true);
  plumeGeo.translate(0, 0.5, 0);
  const nozzleMats = [];
  const plumeMats = [];

  for (let k = 0; k < N_ENG; k++) {
    const phi = k * 60 * DEG;
    const dir = new THREE.Vector3(Math.sin(CANT) * Math.cos(phi), -Math.cos(CANT), Math.sin(CANT) * Math.sin(phi));
    const q = new THREE.Quaternion().setFromUnitVectors(DOWN, dir);
    const throat = polar(ENG_R, phi, ENG_Y);
    const E = new THREE.Matrix4().compose(throat, q, _one);
    buildEngineParts(B, E);

    const grp = new THREE.Group();
    grp.position.copy(throat);
    grp.quaternion.copy(q);
    body.add(grp);

    const nm = new THREE.MeshStandardMaterial({
      color: 0xffffff, map: nozzleMaps.map, metalness: 0.85, roughness: 0.4, side: THREE.DoubleSide,
      emissive: new THREE.Color(0, 0, 0), emissiveMap: nozzleMaps.emissiveMap, emissiveIntensity: 1,
    });
    nozzleMats.push(nm);
    const bell = new THREE.Mesh(bellGeo, nm);
    bell.castShadow = bell.receiveShadow = true;
    grp.add(bell);

    const pm = makePlumeMaterial();
    plumeMats.push(pm);
    const plume = new THREE.Mesh(plumeGeo, pm);
    plume.position.set(0, -NOZZLE_LEN, 0);
    plume.frustumCulled = false;
    plume.visible = false;
    plume.renderOrder = 10;
    plume.castShadow = false;       // a translucent glow must not shadow the ground
    plume.receiveShadow = false;
    grp.add(plume);

    engines.push({ grp, plume, pm, nm });
    exitLocal.push(throat.clone().addScaledVector(dir, NOZZLE_LEN));
    dirLocal.push(dir.clone());
  }

  B.build(mats, body);

  // faint light from the exhaust glow, centred under the deck
  const light = new THREE.PointLight(0xffd9b0, 0, 9, 2);
  light.position.set(0, 0.45, 0);
  light.castShadow = false;
  body.add(light);

  // ---- optional photographic foil textures, swapped in when loaded
  const extraTextures = [];
  if (manifest && manifest.foil) {
    const loader = textureLoader || new THREE.TextureLoader();
    const resolve = (p) => new URL(`../${p}`, import.meta.url).href;
    const f = manifest.foil;
    const load = (path, apply) => {
      if (!path) return;
      loader.load(resolve(path), (t) => {
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.anisotropy = 8;
        extraTextures.push(t);
        apply(t);
      }, undefined, () => { /* keep procedural fallback */ });
    };
    load(f.normalMap, (t) => { for (const k of ['gold', 'silver']) { mats[k].normalMap = t; mats[k].needsUpdate = true; } });
    load(f.roughnessMap, (t) => { for (const k of ['gold', 'silver']) { mats[k].roughnessMap = t; mats[k].needsUpdate = true; } });
    load(f.map, (t) => { t.colorSpace = THREE.SRGBColorSpace; for (const k of ['gold', 'silver']) { mats[k].map = t; mats[k].needsUpdate = true; } });
  }

  // ---- dynamic state
  const target = new Float32Array(N_ENG);
  const current = new Float32Array(N_ENG);
  const heat = new Float32Array(N_ENG);
  let isActive = false;
  let time = 0;

  function setEngines(throttle, active) {
    isActive = !!active;
    for (let i = 0; i < N_ENG; i++) {
      const v = Array.isArray(throttle) || ArrayBuffer.isView(throttle) ? throttle[i] : throttle;
      target[i] = isActive ? THREE.MathUtils.clamp(Number(v) || 0, 0, 1) : 0;
    }
  }

  const TAU_PLUME = 0.04;      // plume follows throttle within ~0.1 s
  const TAU_HEAT_UP = 7.0, TAU_HEAT_DOWN = 9.0;

  function update(dt) {
    dt = Math.min(Math.max(dt || 0, 0), 0.25);
    time += dt;
    const kp = 1 - Math.exp(-dt / TAU_PLUME);
    let mean = 0;
    for (let i = 0; i < N_ENG; i++) {
      current[i] += (target[i] - current[i]) * kp;
      const c = current[i];
      mean += c / N_ENG;

      // nozzle thermal emission: slow first-order lag
      const heatTarget = target[i] > 0 ? 0.55 + 0.45 * THREE.MathUtils.clamp((target[i] - 0.3) / 0.5, 0, 1) : 0;
      const tau = heatTarget > heat[i] ? TAU_HEAT_UP : TAU_HEAT_DOWN;
      heat[i] += (heatTarget - heat[i]) * (1 - Math.exp(-dt / tau));
      const h = heat[i];
      const nm = engines[i].nm;
      nm.emissive.setRGB(0.55 + 0.45 * h, 0.04 + 0.26 * h * h, 0.005 + 0.05 * h * h);
      nm.emissiveIntensity = h * 1.5;

      const e = engines[i];
      const on = c > 0.004;
      e.plume.visible = on;
      if (on) {
        const u = e.pm.uniforms;
        u.uTime.value = time + i * 3.7;
        u.uLen.value = 0.8 + 6.5 * c;
        u.uIntensity.value = Math.min(1, c / 0.8);
      }
    }
    light.intensity = 9 * mean;
  }

  const exits = Array.from({ length: N_ENG }, () => ({ position: new THREE.Vector3(), direction: new THREE.Vector3() }));
  function nozzleExits() {
    object.updateWorldMatrix(true, false);
    for (let i = 0; i < N_ENG; i++) {
      exits[i].position.copy(exitLocal[i]).applyMatrix4(object.matrixWorld);
      exits[i].direction.copy(dirLocal[i]).transformDirection(object.matrixWorld);
    }
    return exits;
  }

  function dispose() {
    const textures = new Set([foilNormal, foil.gold, foil.silver, nozzleMaps.map, nozzleMaps.emissiveMap, ...extraTextures]);
    object.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
    });
    for (const m of [...Object.values(mats), ...nozzleMats, ...plumeMats]) {
      for (const v of Object.values(m)) if (v && v.isTexture) textures.add(v);
      m.dispose();
    }
    textures.forEach((t) => t.dispose());
    light.dispose?.();
  }

  return {
    object,
    setEngines,
    update,
    nozzleExits,
    footprintRadius: PAD_R,
    footprintOuterRadius: PAD_R + PAD_RADIUS,
    dispose,
  };
}
