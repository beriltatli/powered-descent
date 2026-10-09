import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { createLander } from './lander.js';

// Surface frame of Açıkmeşe & Ploen 2007: r = [altitude, h1, h2]. Three.js is Y-up, so
// (x, y, z) = (h2, altitude, h1): a cyclic permutation, which keeps the frame right-handed.
const toThree = (r, out = new THREE.Vector3()) => out.set(r[2], r[0], r[1]);
const UP = new THREE.Vector3(0, 1, 0);
const $ = (id) => document.getElementById(id);

const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 200000);
camera.position.set(60, 25, -40);

const target = new THREE.WebGLRenderTarget(innerWidth, innerHeight, { samples: 4, type: THREE.HalfFloatType });
const composer = new EffectComposer(renderer, target);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.22, 0.4, 0.96);
composer.addPass(bloom);
composer.addPass(new OutputPass());
// Lens: mild vignette and sensor grain, applied after tone mapping.
const lens = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, time: { value: 0 }, strength: { value: 0.035 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float time; uniform float strength; varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)) + time) * 43758.5453); }
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      vec2 d = vUv - 0.5;
      c.rgb *= mix(1.0, 0.72, smoothstep(0.35, 0.85, length(d * vec2(1.25, 1.0))));
      c.rgb += (hash(vUv * 1000.0) - 0.5) * strength;
      gl_FragColor = c;
    }`,
});
composer.addPass(lens);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enabled = false;

const state = {
  env: null, lander: null, frames: [], latestT: 0, renderT: 0, live: false, ended: false, paused: false,
  speed: 1, hello: null, plan: null, endInfo: null, camMode: 'auto', es: null, runId: null,
  q: new THREE.Quaternion(), pos: new THREE.Vector3(), vel: new THREE.Vector3(), thrust: new THREE.Vector3(),
  mass: 1905, solveMs: [], nResolves: 0, nNominal: 0, overlays: true, idleAngle: 0, landed: false,
};

async function loadManifest() {
  try {
    const res = await fetch('assets/manifest.json');
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

function fallbackEnvironment() {
  scene.background = new THREE.Color(0xc79a74);
  scene.fog = new THREE.FogExp2(0xc39a78, 0.00004);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.4;
  const sun = new THREE.DirectionalLight(0xfff1e0, 2.6);
  sun.position.set(-300, 400, 200);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  scene.add(sun, sun.target, new THREE.HemisphereLight(0xd8b08c, 0x5a3422, 0.6));
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(60000, 60000).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0x8a5034, roughness: 0.95 }));
  ground.receiveShadow = true;
  scene.add(ground);
  return {
    heightAt: () => 0,
    sunDirection: sun.position.clone().normalize(),
    update(dt, { focus }) {
      sun.target.position.copy(focus);
      sun.position.copy(focus).addScaledVector(this.sunDirection, 500);
      const s = THREE.MathUtils.clamp(focus.y * 0.6 + 30, 30, 600);
      Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s });
      sun.shadow.camera.updateProjectionMatrix();
    },
    dust: { update() {} },
    dispose() {},
  };
}

async function loadEnvironment(manifest) {
  try {
    const mod = await import('./environment.js');
    return await mod.createEnvironment({ renderer, scene, camera, manifest, assetBase: '' });
  } catch (err) {
    console.warn('environment.js unavailable, using a flat fallback:', err);
    return fallbackEnvironment();
  }
}

// Guidance overlays: latest plan, flown trail, glide-slope cone, target ring.
const overlay = new THREE.Group();
scene.add(overlay);
const planMat = new LineMaterial({ color: 0x6fe3c1, linewidth: 2.5, transparent: true, opacity: 0.9, dashed: true,
  dashSize: 12, gapSize: 7, worldUnits: false });
const trailMat = new LineMaterial({ color: 0xffd2a6, linewidth: 2, transparent: true, opacity: 0.85 });
let planLine = null;
let trailLine = null;
const trailPts = [];

function setLine(old, pts, mat) {
  if (old) {
    overlay.remove(old);
    old.geometry.dispose();
  }
  if (pts.length < 2) return null;
  const geo = new LineGeometry();
  geo.setPositions(pts.flatMap((p) => [p.x, p.y, p.z]));
  const line = new Line2(geo, mat);
  line.computeLineDistances();
  line.frustumCulled = false;
  overlay.add(line);
  return line;
}

function buildGlideSlope(glideslopeDeg, heightAt) {
  // Cone (11): ||(r2, r3)|| <= tan(theta_alt) r1, apex at the target; drawn up to 220 m altitude.
  const H = 220;
  const R = H * Math.tan(THREE.MathUtils.degToRad(glideslopeDeg));
  const g = new THREE.Group();
  const cone = new THREE.Mesh(new THREE.ConeGeometry(R, H, 128, 1, true).rotateX(Math.PI).translate(0, H / 2, 0),
    new THREE.MeshBasicMaterial({ color: 0x7fd1ff, transparent: true, opacity: 0.045, side: THREE.DoubleSide, depthWrite: false }));
  g.add(cone);
  const ringMat = new THREE.LineBasicMaterial({ color: 0x7fd1ff, transparent: true, opacity: 0.35, depthWrite: false });
  for (let h = 20; h <= H; h += 40) {
    const r = h * Math.tan(THREE.MathUtils.degToRad(glideslopeDeg));
    const pts = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      pts.push(new THREE.Vector3(r * Math.cos(a), h, r * Math.sin(a)));
    }
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
  }
  const ring = new THREE.Mesh(new THREE.RingGeometry(2.6, 2.9, 96).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: 0x6fe3c1, transparent: true, opacity: 0.8, depthWrite: false, polygonOffset: true,
      polygonOffsetFactor: -4 }));
  ring.position.y = heightAt(0, 0) + 0.03;
  const cross = new THREE.Mesh(new THREE.RingGeometry(0.0, 0.25, 32).rotateX(-Math.PI / 2), ring.material);
  cross.position.copy(ring.position);
  g.add(ring, cross);
  return g;
}

function resetOverlays() {
  planLine = setLine(planLine, [], planMat);
  trailLine = setLine(trailLine, [], trailMat);
  trailPts.length = 0;
}

// Sparkline of per-re-solve wall time, with the brief's 50 ms p99 budget drawn as a line.
function drawSpark() {
  const c = $('solve-spark');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height, data = state.solveMs.slice(-90);
  ctx.clearRect(0, 0, w, h);
  const top = Math.max(60, ...data);
  const y = (ms) => h - 4 - (ms / top) * (h - 10);
  ctx.strokeStyle = 'rgba(255,93,79,0.6)';
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(0, y(50)); ctx.lineTo(w, y(50)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,93,79,0.8)';
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.fillText('50 ms', w - 36, y(50) - 3);
  if (data.length < 2) return;
  ctx.strokeStyle = '#7fd1ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  data.forEach((ms, i) => {
    const x = (i / 89) * w;
    i ? ctx.lineTo(x, y(ms)) : ctx.moveTo(x, y(ms));
  });
  ctx.stroke();
}

function log(text, cls = '') {
  const li = document.createElement('li');
  li.textContent = text;
  if (cls) li.className = cls;
  const ol = $('log');
  ol.prepend(li);
  while (ol.children.length > 9) ol.lastChild.remove();
}

function setPhase(text, cls) {
  const p = $('phase');
  p.textContent = text;
  p.className = `pill ${cls || ''}`;
}

const PLAN_LABEL = {
  initial: 'initial (Algorithm 1)', nominal: 'nominal, fixed tf', true_bounds: 'true thrust bounds',
  retimed: 're-timed (Algorithm 1)', soft_recovery: 'soft-state recovery', soft_retimed: 'soft, re-timed', held: 'held previous plan',
};

function onHello(h) {
  state.hello = h;
  state.mass = h.vehicle.m_wet;
  $('eps-show').textContent = `${h.eps >= 0 ? '+' : ''}${(h.eps * 100).toFixed(1)} %`;
  $('solver').textContent = `N = ${h.guidance.N} · ${h.guidance.solver}`;
  overlay.remove(state.cone);
  state.cone = buildGlideSlope(h.guidance.glideslope_deg, state.env.heightAt);
  overlay.add(state.cone);
  toThree(h.r0, state.pos);
  toThree(h.v0, state.vel);
  state.lander.object.visible = true;
  state.snapCam = true;
  log(`${h.scenario}: r0 = [${h.r0.map((x) => x.toFixed(0))}] m`);
  setPhase('PLANNING', 'live');
}

function onPlan(p) {
  state.plan = p;
  const pts = p.r.map((r) => toThree(r));
  planLine = setLine(planLine, pts, planMat);
  if (p.kind !== 'initial') {
    state.nResolves += 1;
    if (p.kind === 'nominal') state.nNominal += 1;
    state.solveMs.push(p.solve_ms);
  }
  $('plan-kind').textContent = PLAN_LABEL[p.kind] || p.kind;
  $('solve-ms').textContent = p.solve_ms.toFixed(1);
  $('resolves').textContent = state.nResolves;
  $('nominal').textContent = `${state.nNominal} / ${state.nResolves - state.nNominal}`;
  drawSpark();
  if (p.kind === 'initial') {
    log(`t=0.0 initial plan tf=${p.tf.toFixed(1)} s (${p.solve_ms.toFixed(0)} ms incl. tf search)`);
    setPhase('POWERED DESCENT', 'live');
  } else if (p.kind !== 'nominal') {
    const bad = p.kind === 'held' || p.kind.startsWith('soft');
    log(`t=${p.t0.toFixed(1)} ${PLAN_LABEL[p.kind]} tf=${p.tf.toFixed(1)} s`, bad ? 'crit' : 'warn');
  }
}

function onStep(f) {
  const fr = { t: f.t, r: toThree(f.r), v: toThree(f.v), T: toThree(f.T), m: f.m, sat: f.saturated };
  state.frames.push(fr);
  if (state.frames.length > 4000) state.frames.splice(0, 1000);
  state.latestT = f.t;
}

function onEnd(e) {
  state.endInfo = e;
  // Steps are decimated at high playback speed; the exact contact state comes with the end event.
  if (e.r_end && e.t_end > state.latestT) {
    onStep({ t: e.t_end, r: e.r_end, v: e.v_end, m: state.hello.vehicle.m_wet - e.fuel_used, T: [0, 0, 0], saturated: false });
  }
  state.live = false;
  state.ended = true;
  state.es?.close();
  const ok = e.status === 'landed_at_tf' && e.landing_error < 5 && e.touchdown_speed < 1;
  const fuelOut = e.fuel_exhausted;
  setPhase(ok ? 'TOUCHDOWN' : fuelOut ? 'FUEL EXHAUSTED' : e.status === 'no_initial_plan' ? 'NO FEASIBLE PLAN' : 'HARD CONTACT',
    ok ? 'done' : 'fail');
  if (e.status === 'no_initial_plan') {
    log('Algorithm 1: no feasible time of flight for this start', 'crit');
  }
  $('result-title').textContent = ok ? 'Touchdown: within 5 m and below 1 m/s' : fuelOut ? 'Fuel exhausted before touchdown'
    : e.status === 'no_initial_plan' ? 'Infeasible start: no plan' : 'Landed outside the success criteria';
  const fmt = (x, d = 2) => (x == null || !isFinite(x) ? '—' : x.toFixed(d));
  $('r-err').textContent = fmt(e.landing_error);
  $('r-spd').textContent = fmt(e.touchdown_speed);
  $('r-fuel').textContent = fmt(e.fuel_used, 1);
  $('r-res').textContent = `${e.n_resolves} (${e.n_feasible})`;
  $('r-ms').textContent = `${fmt(e.solve_ms_p50, 1)} / ${fmt(e.solve_ms_max, 1)}`;
  $('r-rf').textContent = e.recursively_feasible ? 'yes' : 'no';
  setTimeout(() => $('result').classList.remove('hidden'), 1800);
  $('go').textContent = 'Launch again';
}

function launch() {
  state.es?.close();
  Object.assign(state, { frames: [], latestT: 0, renderT: 0, live: true, ended: false, paused: false, plan: null,
    endInfo: null, solveMs: [], nResolves: 0, nNominal: 0, landed: false, crashT: 0 });
  resetOverlays();
  $('log').innerHTML = '';
  $('result').classList.add('hidden');
  $('pause').textContent = 'Pause';
  drawSpark();
  const eps = Number($('eps').value) / 100;
  state.speed = Number($('speed').value);
  const qs = new URLSearchParams({ scenario: $('scenario').value, eps, speed: state.speed });
  const es = new EventSource(`/api/fly?${qs}`);
  state.es = es;
  es.addEventListener('hello', (ev) => { const h = JSON.parse(ev.data); state.runId = h.id; onHello(h); });
  es.addEventListener('plan', (ev) => onPlan(JSON.parse(ev.data)));
  es.addEventListener('step', (ev) => onStep(JSON.parse(ev.data)));
  es.addEventListener('end', (ev) => onEnd(JSON.parse(ev.data)));
  // EventSource reconnects on its own after the stream closes; that would start a new flight.
  es.onerror = () => { es.close(); if (!state.ended) { state.live = false; setPhase('LINK LOST', 'fail'); } };
  setPhase('CONNECTING', 'live');
}

function control(params) {
  if (!state.runId) return;
  fetch(`/api/control?${new URLSearchParams({ id: state.runId, ...params })}`, { method: 'POST' }).catch(() => {});
}

// Cubic Hermite between two plant samples using their velocities: C1-continuous motion at any frame rate.
function sample(t) {
  const F = state.frames;
  if (!F.length) return null;
  if (t <= F[0].t) return F[0];
  if (t >= F[F.length - 1].t) return F[F.length - 1];
  let lo = 0, hi = F.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    F[mid].t <= t ? (lo = mid) : (hi = mid);
  }
  const a = F[lo], b = F[hi], h = b.t - a.t, s = (t - a.t) / h;
  const s2 = s * s, s3 = s2 * s;
  const r = new THREE.Vector3()
    .addScaledVector(a.r, 2 * s3 - 3 * s2 + 1).addScaledVector(a.v, (s3 - 2 * s2 + s) * h)
    .addScaledVector(b.r, -2 * s3 + 3 * s2).addScaledVector(b.v, (s3 - s2) * h);
  const v = a.v.clone().lerp(b.v, s);
  return { t, r, v, T: a.T, m: a.m + (b.m - a.m) * s, sat: a.sat };
}

const camState = { pos: new THREE.Vector3(60, 25, -40), look: new THREE.Vector3(), fov: 55, back: new THREE.Vector3(1, 0, 0) };
const TRACK_SITE = new THREE.Vector3(-70, 0, 55);
const PAD_SITE = new THREE.Vector3(16, 0, -11);

function cameraGoal(dt) {
  const p = state.pos, heightAt = state.env.heightAt;
  let mode = state.camMode;
  if (mode === 'auto') mode = !state.live && !state.ended ? 'idle' : p.y > 140 ? 'chase' : 'ground';
  const goal = { pos: new THREE.Vector3(), look: p.clone(), fov: 55, k: 3.0 };
  if (mode === 'idle') {
    state.idleAngle += dt * 0.04;
    goal.pos.set(Math.cos(state.idleAngle) * 70, 18, Math.sin(state.idleAngle) * 70);
    goal.look.set(0, 2, 0);
    goal.fov = 50;
  } else if (mode === 'chase') {
    const hv = new THREE.Vector3(state.vel.x, 0, state.vel.z);
    if (hv.lengthSq() > 4) camState.back.lerp(hv.normalize().negate(), 1 - Math.exp(-dt * 0.8));
    const d = 16 + Math.min(p.y * 0.01, 10);
    goal.pos.copy(p).addScaledVector(camState.back, d).add(new THREE.Vector3(0, d * 0.16, 0));
    goal.look.add(new THREE.Vector3(state.vel.x, 0, state.vel.z).multiplyScalar(0.12));
    goal.fov = 50;
    goal.k = 4.0;
  } else if (mode === 'ground') {
    goal.pos.copy(TRACK_SITE).setY(heightAt(TRACK_SITE.x, TRACK_SITE.z) + 1.7);
    const dist = goal.pos.distanceTo(p);
    // A tracking telephoto: keep the ~4 m vehicle (plus plume and dust) about a fifth of the frame tall.
    goal.fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(14 / dist)), 2.5, 55);
    goal.k = 6.0;
  } else if (mode === 'pad') {
    goal.pos.copy(PAD_SITE).setY(heightAt(PAD_SITE.x, PAD_SITE.z) + 1.2);
    goal.fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(7 / goal.pos.distanceTo(p))), 4, 62);
    goal.k = 6.0;
  } else if (mode === 'overview') {
    const start = state.frames[0]?.r || p;
    const mid = start.clone().multiplyScalar(0.5);
    const side = new THREE.Vector3(-start.z, 0, start.x).normalize();
    if (!isFinite(side.x)) side.set(0, 0, 1);
    goal.pos.copy(mid).addScaledVector(side, Math.max(start.length() * 1.15, 400)).add(new THREE.Vector3(0, 250, 0));
    goal.look.copy(mid);
    goal.fov = 45;
    goal.k = 1.5;
  }
  return { mode, goal };
}

function updateCamera(dt) {
  if (state.camMode === 'free') {
    controls.target.lerp(state.pos, 1 - Math.exp(-dt * 6));
    controls.update();
    return;
  }
  const { goal } = cameraGoal(dt);
  const a = state.snapCam ? 1 : 1 - Math.exp(-dt * goal.k);
  camState.pos.lerp(goal.pos, a);
  camState.look.lerp(goal.look, state.snapCam ? 1 : 1 - Math.exp(-dt * goal.k * 2));
  state.snapCam = false;
  camState.fov += (goal.fov - camState.fov) * a;
  const floor = state.env.heightAt(camState.pos.x, camState.pos.z) + 0.8;
  if (camState.pos.y < floor) camState.pos.y = floor;
  camera.position.copy(camState.pos);
  camera.lookAt(camState.look);
  if (Math.abs(camera.fov - camState.fov) > 1e-3) {
    camera.fov = camState.fov;
    camera.updateProjectionMatrix();
  }
}

let hudTimer = 0;
function updateHud(f) {
  const h = state.hello;
  if (!h || !f) return;
  const alt = f.r.y;
  const engineOff = (state.ended && state.renderT >= state.latestT - 1e-6) || f.m <= h.vehicle.m_dry + 1e-6;
  const thrustN = engineOff ? 0 : f.T.length();
  const perEngineMax = h.vehicle.engine_max_N;
  const throttle = thrustN / (h.vehicle.n_engines * perEngineMax * Math.cos(THREE.MathUtils.degToRad(h.vehicle.cant_deg)));
  $('t').textContent = f.t.toFixed(1);
  $('tgo').textContent = state.plan && !state.ended ? Math.max(0, state.plan.tf - f.t).toFixed(1) : '—';
  $('alt').textContent = alt.toFixed(alt < 100 ? 1 : 0);
  $('vz').textContent = f.v.y.toFixed(1);
  $('vh').textContent = Math.hypot(f.v.x, f.v.z).toFixed(1);
  $('range').textContent = Math.hypot(f.r.x, f.r.z).toFixed(alt < 100 ? 1 : 0);
  $('thrust').textContent = (thrustN / 1000).toFixed(2);
  $('throttle-bar').style.width = `${Math.min(100, throttle * 100)}%`;
  $('tilt').textContent = thrustN > 1 ? THREE.MathUtils.radToDeg(f.T.angleTo(UP)).toFixed(1) : '—';
  $('mass').textContent = f.m.toFixed(1);
  $('mhat').textContent = ((1 + h.eps) * f.m).toFixed(1);
  const fuel = f.m - h.vehicle.m_dry, cap = h.vehicle.m_wet - h.vehicle.m_dry;
  $('fuel-bar').style.width = `${Math.max(0, (fuel / cap) * 100)}%`;
  $('fuel-bar').classList.toggle('low', fuel / cap < 0.15);
  $('fuel-text').textContent = `fuel ${Math.max(0, fuel).toFixed(1)} kg`;
}

const clock = new THREE.Clock();
const qGoal = new THREE.Quaternion();
const thrustDir = new THREE.Vector3();

const hardContact = () => !!state.endInfo && state.endInfo.status === 'ground_contact' && state.endInfo.touchdown_speed > 3;

function frame(dt) {
  const L = state.lander, env = state.env;
  let f = null;
  if (state.frames.length) {
    if (!state.paused) state.renderT += dt * state.speed;
    // Stay just behind the newest sample so there is always a later one to interpolate toward.
    const lead = state.ended ? 0 : 0.06 * state.speed;
    state.renderT = Math.min(state.renderT, state.latestT - lead);
    if (state.latestT - lead - state.renderT > 0.6 * state.speed) state.renderT = state.latestT - lead - 0.1 * state.speed;
    f = sample(state.renderT);
  }
  let active = false, throttle = 0;
  if (f) {
    state.pos.copy(f.r);
    state.vel.copy(f.v);
    const ground = env.heightAt(f.r.x, f.r.z);
    const touchedDown = state.ended && state.renderT >= state.latestT - 1e-6;
    if (state.pos.y < ground) state.pos.y = ground;
    const thrustN = f.T.length();
    const h = state.hello.vehicle;
    active = thrustN > 1 && !touchedDown && f.m > h.m_dry + 1e-6;
    throttle = thrustN / (h.n_engines * h.engine_max_N * Math.cos(THREE.MathUtils.degToRad(h.cant_deg)));
    // 3-DoF model: the thrust direction stands in for attitude; a finite slew rate keeps it physical.
    if (active) {
      thrustDir.copy(f.T).normalize();
      qGoal.setFromUnitVectors(UP, thrustDir);
    } else if (touchedDown && hardContact()) {
      // Impact well beyond the legs' stroke: the vehicle comes to rest on a crushed leg, tipped over.
      const axis = new THREE.Vector3(f.v.z, 0, -f.v.x);
      qGoal.setFromAxisAngle(axis.lengthSq() > 1e-6 ? axis.normalize() : new THREE.Vector3(1, 0, 0), -0.55);
    } else {
      qGoal.identity();
    }
    state.q.slerp(qGoal, 1 - Math.exp(-dt * (active ? 7 : touchedDown && hardContact() ? 12 : 3)));
    if (touchedDown && hardContact()) {
      state.crashT = (state.crashT || 0) + dt;
      state.pos.y -= Math.min(0.35, state.crashT * 3);
    }
    if (touchedDown && !state.landed) state.landed = true;
    if (state.overlays && (trailPts.length === 0 || trailPts[trailPts.length - 1].distanceToSquared(state.pos) > 4)) {
      trailPts.push(state.pos.clone());
      if (trailPts.length % 4 === 0 || trailPts.length < 8) trailLine = setLine(trailLine, trailPts, trailMat);
    }
  }
  L.object.position.copy(state.pos);
  L.object.quaternion.copy(state.q);
  L.object.updateMatrixWorld(true);
  L.setEngines(throttle, active);
  L.update(dt, camera);
  const burst = hardContact() && state.crashT > 0 && state.crashT < 0.9;
  env.dust.update(dt, { nozzles: L.nozzleExits(), throttle: burst ? 1 : throttle, active: active || burst });
  updateCamera(dt);
  env.update(dt, { camera, focus: state.lander.object.visible ? state.pos : new THREE.Vector3() });
  overlay.visible = state.overlays;
  planMat.resolution.set(innerWidth, innerHeight);
  trailMat.resolution.set(innerWidth, innerHeight);
  lens.uniforms.time.value = (lens.uniforms.time.value + dt) % 100;
  hudTimer += dt;
  if (hudTimer > 0.066 && f) {
    hudTimer = 0;
    updateHud(f);
  }
  composer.render(dt);
}

function animate() {
  frame(Math.min(clock.getDelta(), 0.1));
  requestAnimationFrame(animate);
}

function setCam(mode) {
  state.camMode = mode;
  controls.enabled = mode === 'free';
  if (mode === 'free') {
    controls.target.copy(state.pos);
    camera.position.copy(state.pos).add(new THREE.Vector3(14, 6, 14));
  } else {
    camState.pos.copy(camera.position);
  }
  document.querySelectorAll('.cams button').forEach((b) => b.classList.toggle('on', b.dataset.cam === mode));
}

function bindUi() {
  $('eps').addEventListener('input', (e) => {
    const v = Number(e.target.value);
    $('eps-val').textContent = `${v >= 0 ? '+' : ''}${v.toFixed(1)} %`;
  });
  $('go').addEventListener('click', launch);
  $('pause').addEventListener('click', () => {
    if (!state.live) return;
    state.paused = !state.paused;
    $('pause').textContent = state.paused ? 'Resume' : 'Pause';
    control({ paused: state.paused ? 1 : 0 });
  });
  $('speed').addEventListener('change', (e) => {
    state.speed = Number(e.target.value);
    control({ speed: state.speed });
  });
  document.querySelectorAll('.cams button').forEach((b) => b.addEventListener('click', () => setCam(b.dataset.cam)));
  $('overlays').addEventListener('change', (e) => { state.overlays = e.target.checked; });
  $('close-result').addEventListener('click', () => $('result').classList.add('hidden'));
  const keys = ['auto', 'chase', 'ground', 'pad', 'overview', 'free'];
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key >= '1' && e.key <= '6') setCam(keys[Number(e.key) - 1]);
    if (e.key === 'h' || e.key === 'H') { state.overlays = !state.overlays; $('overlays').checked = state.overlays; }
    if (e.key === ' ') { e.preventDefault(); $('pause').click(); }
  });
  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    composer.setSize(innerWidth, innerHeight);
  });
}

async function init() {
  const manifest = await loadManifest();
  $('loading-text').textContent = 'Building terrain…';
  state.env = await loadEnvironment(manifest);
  state.lander = createLander({ manifest });
  state.lander.object.visible = false;
  scene.add(state.lander.object);
  state.cone = buildGlideSlope(86, state.env.heightAt);
  overlay.add(state.cone);
  bindUi();
  drawSpark();
  $('loading').classList.add('gone');
  animate();
}

init();
// Debug handle: frame(dt) renders one frame without requestAnimationFrame (for throttled or headless tabs).
window.__pd = { THREE, scene, camera, renderer, composer, bloom, lens, state, frame, setCam };
