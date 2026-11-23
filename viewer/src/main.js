import * as THREE from 'three';
import './style.css';
import { createSkyDome, createTerrainMaterial, createWaterMaterial } from './terrainShader.js';
import { buildLakeSystem, buildRiverSystem } from './rivers.js';
import { createGrassField } from './grass.js';
import { buildWorldProps, updatePropsTransform } from './props.js';

// ---------- Renderer & Scene ----------
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.06;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(56, window.innerWidth / window.innerHeight, 0.4, 20000);

// Lights (driving multi-tier trees, boulders & explorer character)
const hemiLight = new THREE.HemisphereLight('#b9dcff', '#4b5a40', 0.92);
scene.add(hemiLight);
const sunLight = new THREE.DirectionalLight('#fff3dc', 1.85);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.near = 20;
sunLight.shadow.camera.far = 900;
sunLight.shadow.camera.left = -220;
sunLight.shadow.camera.right = 220;
sunLight.shadow.camera.top = 220;
sunLight.shadow.camera.bottom = -220;
sunLight.shadow.bias = -0.0005;
scene.add(sunLight);
scene.add(sunLight.target);

const skyDome = createSkyDome();
scene.add(skyDome);

// ---------- World State ----------
let catalog = [];
let world = null; // { meta, heights: Float32Array, masks: Uint8Array, splines }
let terrainMesh = null;
let terrainMat = null;
let waterMesh = null;
let waterMat = null;
let riverObj = null;
let lakeObj = null;
let grassObj = null;
let propsGroup = null;
let heightFloatTex = null;
let maskTex = null;
let climateTex = null;

// ---------- Settings ----------
const state = {
  camMode: 'third', // 'third' | 'first' | 'diorama'
  styleMode: 0, // 0 high-detail | 1 faceted diorama | 2 topo
  sunElevDeg: 28,
  sunAzimDeg: 48,
  vScale: 1.0,
};

// ---------- Explorer Character ----------
const explorer = new THREE.Group();
{
  const bodyMat = new THREE.MeshStandardMaterial({ color: '#2b73d9', roughness: 0.75 });
  const cloakMat = new THREE.MeshStandardMaterial({ color: '#d9483b', roughness: 0.8 });
  const skinMat = new THREE.MeshStandardMaterial({ color: '#f2c7a2', roughness: 0.7 });
  const packMat = new THREE.MeshStandardMaterial({ color: '#6b472b', roughness: 0.85 });

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.55, 1.05, 6, 12), bodyMat);
  torso.position.y = 1.25;
  torso.castShadow = true;
  explorer.add(torso);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.46, 14, 12), skinMat);
  head.position.y = 2.45;
  head.castShadow = true;
  explorer.add(head);

  const hat = new THREE.Mesh(new THREE.ConeGeometry(0.64, 0.88, 10), cloakMat);
  hat.position.set(0, 3.0, -0.04);
  hat.rotation.x = -0.18;
  hat.castShadow = true;
  explorer.add(hat);

  const cape = new THREE.Mesh(new THREE.BoxGeometry(0.84, 1.1, 0.16), cloakMat);
  cape.position.set(0, 1.6, 0.5);
  cape.rotation.x = 0.22;
  cape.name = 'cape';
  cape.castShadow = true;
  explorer.add(cape);

  const pack = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.78, 0.38), packMat);
  pack.position.set(0, 1.48, 0.62);
  pack.castShadow = true;
  explorer.add(pack);

  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(1.0, 20),
    new THREE.MeshBasicMaterial({ color: '#09121c', transparent: true, opacity: 0.35, depthWrite: false })
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.08;
  explorer.add(shadow);
}
scene.add(explorer);

// ---------- Isometric Explorer Pin & Cursor Reticle (visible ONLY in Mode 3 / Diorama) ----------
const explorerPin = new THREE.Group();
explorerPin.visible = false;
{
  // Ground beacon ring
  const baseRing = new THREE.Mesh(
    new THREE.RingGeometry(2.4, 3.6, 36),
    new THREE.MeshBasicMaterial({ color: 0xff6b4a, side: THREE.DoubleSide, transparent: true, opacity: 0.92, depthWrite: false })
  );
  baseRing.rotation.x = -Math.PI / 2;
  baseRing.position.y = 0.35;
  baseRing.name = 'pinBaseRing';
  explorerPin.add(baseRing);

  // Expanding pulse ring
  const pulseRing = new THREE.Mesh(
    new THREE.RingGeometry(3.2, 4.1, 36),
    new THREE.MeshBasicMaterial({ color: 0xffa07a, side: THREE.DoubleSide, transparent: true, opacity: 0.7, depthWrite: false })
  );
  pulseRing.rotation.x = -Math.PI / 2;
  pulseRing.position.y = 0.4;
  pulseRing.name = 'pinPulseRing';
  explorerPin.add(pulseRing);

  // Vertical beacon stem
  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(0.32, 0.2, 11.5, 10),
    new THREE.MeshBasicMaterial({ color: 0xfff0e6, transparent: true, opacity: 0.9 })
  );
  stem.position.y = 6.0;
  explorerPin.add(stem);

  // Floating map-pin teardrop head (high-contrast coral-red + white core)
  const pinHead = new THREE.Group();
  pinHead.position.y = 14.2;
  pinHead.name = 'pinHead';

  const tip = new THREE.Mesh(
    new THREE.ConeGeometry(2.35, 4.2, 18),
    new THREE.MeshBasicMaterial({ color: 0xff4d2e })
  );
  tip.rotation.x = Math.PI;
  tip.position.y = -2.4;
  pinHead.add(tip);

  const bulb = new THREE.Mesh(
    new THREE.SphereGeometry(2.65, 20, 16),
    new THREE.MeshBasicMaterial({ color: 0xff4d2e })
  );
  pinHead.add(bulb);

  const core = new THREE.Mesh(
    new THREE.SphereGeometry(1.15, 14, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff })
  );
  core.position.z = 0;
  pinHead.add(core);

  const halo = new THREE.Mesh(
    new THREE.TorusGeometry(2.95, 0.22, 10, 28),
    new THREE.MeshBasicMaterial({ color: 0x0f172a })
  );
  pinHead.add(halo);

  explorerPin.add(pinHead);
}
scene.add(explorerPin);

// Cursor ground target reticle in Isometric mode (shows where 'T' / double-click will relocate explorer)
const cursorReticle = new THREE.Group();
cursorReticle.visible = false;
{
  const outer = new THREE.Mesh(
    new THREE.RingGeometry(2.5, 3.6, 32),
    new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
      depthWrite: false,
    })
  );
  outer.rotation.x = -Math.PI / 2;
  outer.position.y = 0.8;
  outer.renderOrder = 25;
  cursorReticle.add(outer);

  const dot = new THREE.Mesh(
    new THREE.CircleGeometry(0.95, 20),
    new THREE.MeshBasicMaterial({
      color: 0xe0f2fe,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
      depthWrite: false,
    })
  );
  dot.rotation.x = -Math.PI / 2;
  dot.position.y = 0.9;
  dot.renderOrder = 26;
  cursorReticle.add(dot);
}
scene.add(cursorReticle);

const player = {
  pos: new THREE.Vector3(0, 20, 0),
  velY: 0,
  yaw: 0,
  grounded: true,
  activity: 'WALKING',
};

const camRig = {
  yaw: 0.55,
  pitch: 0.09,
  dist: 18,
  dioramaCenter: new THREE.Vector3(0, 0, 0),
  dioramaDist: 760,
  dioramaYaw: 0.0, // North-Up so Diorama aligns 1:1 with the Stage 1 globe target square!
};

// ---------- Input Handling ----------
const keys = new Set();
let isDragging = false;
let lastMouse = { x: 0, y: 0 };
const pointerNdc = { x: 0, y: 0, inside: false };
const pickRaycaster = new THREE.Raycaster();
let dioramaCursorHit = null;
let pinDropFlash = 0;

// ---------- Drill-Down Zoom State & 3D Box Overlays ----------
const drillState = {
  aiming: false,
  zoomFactor: 4,
  running: false,
  lastResult: null,
  cachedChildren: [],
};
let drillBoxGroup = null;
let drillChildrenGroup = null;

function disposeGroup(grp) {
  if (!grp) return;
  scene.remove(grp);
  grp.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
}

function buildTerrainBoxMesh(cx, cz, boxSpan, colorHex, fillOpacity = 0.0, liftY = 3.5) {
  const grp = new THREE.Group();
  const half = boxSpan * 0.5;
  const segs = 28;
  const pts = [];
  const edges = [
    [-1, -1, 1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, 1],
    [-1, 1, -1, -1],
  ];
  for (const [sx0, sz0, sx1, sz1] of edges) {
    for (let i = 0; i < segs; i++) {
      const f = i / segs;
      const x = cx + (sx0 + (sx1 - sx0) * f) * half;
      const z = cz + (sz0 + (sz1 - sz0) * f) * half;
      const y = Math.max(sampleHeight(x, z), 0) + liftY;
      pts.push(new THREE.Vector3(x, y, z));
    }
  }
  pts.push(pts[0].clone());
  grp.add(
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.95 })
    )
  );

  // Subtle corner ticks for crisp readability over noisy terrain
  for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    const x = cx + sx * half;
    const z = cz + sz * half;
    const y = Math.max(sampleHeight(x, z), 0) + liftY;
    const post = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(boxSpan * 0.011, 2.5), 10, 8),
      new THREE.MeshBasicMaterial({ color: colorHex })
    );
    post.position.set(x, y + 1.5, z);
    grp.add(post);
  }

  if (fillOpacity > 0) {
    const gN = 10;
    const pos = [];
    const idx = [];
    for (let r = 0; r <= gN; r++) {
      const z = cz + (r / gN - 0.5) * boxSpan;
      for (let c = 0; c <= gN; c++) {
        const x = cx + (c / gN - 0.5) * boxSpan;
        const y = Math.max(sampleHeight(x, z), 0) + liftY * 0.6;
        pos.push(x, y, z);
      }
    }
    for (let r = 0; r < gN; r++) {
      for (let c = 0; c < gN; c++) {
        const a = r * (gN + 1) + c;
        const b = a + 1;
        const d = (r + 1) * (gN + 1) + c;
        const e = d + 1;
        idx.push(a, d, b, b, d, e);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    grp.add(
      new THREE.Mesh(
        geo,
        new THREE.MeshBasicMaterial({
          color: colorHex,
          transparent: true,
          opacity: fillOpacity,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
      )
    );
  }

  return grp;
}

function getClampedDrillCenterUV() {
  if (!world) return { u: 0.5, v: 0.5, cx: 0, cz: 0, boxSpan: 1200 };
  const span = world.meta.worldSpan;
  const half = 0.5 / drillState.zoomFactor;
  const rawU = player.pos.x / span + 0.5;
  const rawV = player.pos.z / span + 0.5;
  const u = THREE.MathUtils.clamp(rawU, half, 1.0 - half);
  const v = THREE.MathUtils.clamp(rawV, half, 1.0 - half);
  return {
    u,
    v,
    cx: (u - 0.5) * span,
    cz: (v - 0.5) * span,
    boxSpan: span / drillState.zoomFactor,
  };
}

function rebuildDrillBoxOverlay() {
  disposeGroup(drillBoxGroup);
  drillBoxGroup = null;
  if (!world || !drillState.aiming || state.camMode !== 'diorama') return;
  const { cx, cz, boxSpan } = getClampedDrillCenterUV();
  const liftY = Math.max(world.meta.worldSpan * 0.0022, 1.5);
  drillBoxGroup = buildTerrainBoxMesh(cx, cz, boxSpan, '#38f2d8', 0.10, liftY);
  scene.add(drillBoxGroup);
}

function rebuildDrillChildrenOverlay() {
  disposeGroup(drillChildrenGroup);
  drillChildrenGroup = null;
  if (!world) return;
  const children = catalog.filter((w) => w?.drillOrigin?.parentWorldId === world.meta.id);
  drillState.cachedChildren = children;

  const wrap = document.getElementById('drill-children');
  const list = document.getElementById('drill-children-list');
  if (wrap && list) {
    if (!children.length) {
      wrap.style.display = 'none';
      list.innerHTML = '';
    } else {
      wrap.style.display = 'block';
      list.innerHTML = '';
      children.forEach((ch) => {
        const d = ch.drillOrigin;
        const a = document.createElement('a');
        a.className = 'drill-chip';
        a.href = `/index.html?world=${encodeURIComponent(ch.id)}`;
        a.textContent = `★ ${d.zoomFactor}× (${d.pixelM.toFixed(1)}m/px) ⚡`;
        a.title = `Instant open cached ${d.zoomFactor}× sub-region (${ch.lakeCount ?? 0} lakes, ${ch.riverCount ?? 0} rivers)`;
        list.appendChild(a);
      });
    }
  }

  if (!children.length) return;
  const g = new THREE.Group();
  const span = world.meta.worldSpan;
  const liftY = Math.max(span * 0.0018, 1.2);
  children.forEach((ch) => {
    const d = ch.drillOrigin;
    const cx = (d.centerU - 0.5) * span;
    const cz = (d.centerV - 0.5) * span;
    const boxSpan = span / (d.zoomFactor || 4);
    g.add(buildTerrainBoxMesh(cx, cz, boxSpan, '#fbbf24', 0.08, liftY));
  });
  g.visible = state.camMode === 'diorama';
  drillChildrenGroup = g;
  scene.add(drillChildrenGroup);
}

async function refreshCatalogData() {
  const baseCatalog = await fetch('/worlds/catalog.json', { cache: 'no-store' }).then((r) => r.json());
  let synthCatalog = [];
  try {
    const r = await fetch('/worlds/synthesized_catalog.json', { cache: 'no-store' });
    if (r.ok) synthCatalog = await r.json();
  } catch {
    // ignore
  }
  const seenIds = new Set(baseCatalog.map((w) => w.id));
  catalog = [...baseCatalog, ...synthCatalog.filter((w) => !seenIds.has(w.id))];
  const select = document.getElementById('world-select');
  if (select) {
    const curVal = world?.meta?.id || select.value;
    select.innerHTML = catalog.map((w) => `<option value="${w.id}">${w.title}</option>`).join('');
    if (curVal && catalog.some((w) => w.id === curVal)) select.value = curVal;
  }
}

async function runDrillDownAtPin() {
  if (!world || drillState.running) return;
  const btn = document.getElementById('btn-drill-run');
  const progWrap = document.getElementById('drill-progress-wrap');
  const progFill = document.getElementById('drill-progress-fill');
  const progText = document.getElementById('drill-status-text');
  const resBox = document.getElementById('drill-result');
  const resMetrics = document.getElementById('drill-result-metrics');
  const openBtn = document.getElementById('btn-open-drill');

  drillState.running = true;
  if (btn) {
    btn.disabled = true;
    btn.textContent = `Drilling ${drillState.zoomFactor}× Micro-Terrain…`;
  }
  if (resBox) resBox.style.display = 'none';
  if (progWrap) progWrap.style.display = 'block';
  if (progFill) progFill.style.width = '8%';
  if (progText) progText.textContent = 'Sending sub-region to Terrain Diffusion MPS…';

  const pollId = window.setInterval(async () => {
    try {
      const r = await fetch('/api/status', { cache: 'no-store' });
      if (!r.ok) return;
      const st = await r.json();
      if (st.message && progText) progText.textContent = st.message;
      if (typeof st.progress === 'number' && progFill) {
        progFill.style.width = `${Math.max(10, Math.round(st.progress * 100))}%`;
      }
    } catch {
      // ignore transient poll errors
    }
  }, 550);

  try {
    const { u, v } = getClampedDrillCenterUV();
    const resp = await fetch('/api/drilldown', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        parentWorldId: world.meta.id,
        centerU: u,
        centerV: v,
        zoomFactor: drillState.zoomFactor,
        quality: 'high',
        fidelity: 'exact',
      }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || `Drill-down failed (${resp.status})`);
    }
    drillState.lastResult = data;
    if (progFill) progFill.style.width = '100%';
    if (progText) progText.textContent = `Done in ${data.totalSec}s (MPS ${data.mpsSec}s · Hydro ${data.hydroSec}s)`;
    if (resMetrics) {
      resMetrics.innerHTML =
        `<strong>✓ ${data.zoomFactor}× High-Res Ready!</strong> (${data.pixelM} m/px)<br/>` +
        `IoU <strong>${data.iouPct}%</strong> · <strong>${data.lakeCount}</strong> lakes · <strong>${data.riverCount}</strong> rivers`;
    }
    if (openBtn) openBtn.href = data.openUrl;
    if (resBox) resBox.style.display = 'block';

    await refreshCatalogData();
    rebuildDrillChildrenOverlay();
  } catch (err) {
    if (progFill) progFill.style.width = '100%';
    if (progText) {
      progText.innerHTML = `<span style="color:#fca5a5;">${err.message || 'Stage 2 server unreachable.'}</span>`;
    }
  } finally {
    window.clearInterval(pollId);
    drillState.running = false;
    if (btn) {
      btn.disabled = false;
      btn.textContent = '🔬 Drill Down at Pin (~12s)';
    }
  }
}

function getDioramaPlaneY() {
  return Math.max((world?.meta?.maxElev || 160) * state.vScale * 0.22, 20);
}

function pickTerrainAtNdc(ndcX, ndcY) {
  if (!world) return null;
  pickRaycaster.setFromCamera({ x: ndcX, y: ndcY }, camera);
  const o = pickRaycaster.ray.origin;
  const d = pickRaycaster.ray.direction;
  if (d.y >= -1e-4) return null;

  const halfSpan = world.meta.worldSpan * 0.49;
  const maxY = Math.max(world.meta.maxElev * state.vScale + 30, 60);
  const tStart = Math.max(1.0, (maxY - o.y) / d.y);
  const tEnd = Math.min(20000, (-60 - o.y) / d.y);
  if (tEnd <= tStart) return null;

  const steps = 96;
  const dt = (tEnd - tStart) / steps;
  let prevT = tStart;
  for (let i = 1; i <= steps; i++) {
    const t = tStart + dt * i;
    const x = o.x + d.x * t;
    const y = o.y + d.y * t;
    const z = o.z + d.z * t;
    const surfY = Math.max(sampleHeight(x, z), 0);
    if (y - surfY <= 0) {
      let lo = prevT;
      let hi = t;
      for (let b = 0; b < 10; b++) {
        const mid = (lo + hi) * 0.5;
        const mx = o.x + d.x * mid;
        const my = o.y + d.y * mid;
        const mz = o.z + d.z * mid;
        const mh = Math.max(sampleHeight(mx, mz), 0);
        if (my - mh <= 0) hi = mid;
        else lo = mid;
      }
      const fx = o.x + d.x * hi;
      const fz = o.z + d.z * hi;
      if (Math.abs(fx) > halfSpan || Math.abs(fz) > halfSpan) return null;
      const fy = Math.max(sampleHeight(fx, fz), 0);
      return new THREE.Vector3(fx, fy, fz);
    }
    prevT = t;
  }
  return null;
}

function relocateExplorerTo(x, z) {
  if (!world) return false;
  const halfSpan = world.meta.worldSpan * 0.48;
  const cx = THREE.MathUtils.clamp(x, -halfSpan, halfSpan);
  const cz = THREE.MathUtils.clamp(z, -halfSpan, halfSpan);
  const cy = Math.max(sampleHeight(cx, cz), 0.2);
  player.pos.set(cx, cy, cz);
  player.velY = 0;
  player.grounded = true;
  explorer.position.copy(player.pos);
  pinDropFlash = 1.0;
  rebuildDrillBoxOverlay();
  return true;
}

function updatePointerNdc(e) {
  const rect = canvas.getBoundingClientRect();
  pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  pointerNdc.inside = true;
}

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  keys.add(k);
  if (k === '1') setCameraMode('third');
  if (k === '2') setCameraMode('first');
  if (k === '3') setCameraMode('diorama');
  if (k === 'v') setStyleMode((state.styleMode + 1) % 3);
  if (k === 'h') document.body.classList.toggle('hud-hidden');
  if (k === 't' && state.camMode === 'diorama') {
    const hit = pointerNdc.inside ? pickTerrainAtNdc(pointerNdc.x, pointerNdc.y) : null;
    if (hit) {
      relocateExplorerTo(hit.x, hit.z);
    } else {
      // Fallback: place explorer at the center of the isometric view
      relocateExplorerTo(camRig.dioramaCenter.x, camRig.dioramaCenter.z);
    }
  }
  if (k === 'f' && state.camMode === 'diorama') {
    camRig.dioramaCenter.set(player.pos.x, getDioramaPlaneY(), player.pos.z);
  }
  if (e.code === 'Space') e.preventDefault();
});
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

canvas.addEventListener('pointerdown', (e) => {
  updatePointerNdc(e);
  isDragging = true;
  lastMouse = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', (e) => {
  isDragging = false;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
});
canvas.addEventListener('pointerleave', () => {
  pointerNdc.inside = false;
  dioramaCursorHit = null;
});
canvas.addEventListener('dblclick', (e) => {
  if (state.camMode !== 'diorama') return;
  updatePointerNdc(e);
  const hit = pickTerrainAtNdc(pointerNdc.x, pointerNdc.y);
  if (hit) relocateExplorerTo(hit.x, hit.z);
});
canvas.addEventListener('pointermove', (e) => {
  updatePointerNdc(e);
  if (!isDragging) return;
  const dx = e.clientX - lastMouse.x;
  const dy = e.clientY - lastMouse.y;
  lastMouse = { x: e.clientX, y: e.clientY };

  if (state.camMode === 'diorama') {
    camRig.dioramaYaw -= dx * 0.006;
  } else {
    camRig.yaw -= dx * 0.0055;
    camRig.pitch = THREE.MathUtils.clamp(camRig.pitch + dy * 0.0045, -0.55, 1.25);
  }
});
canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    if (state.camMode === 'diorama') {
      camRig.dioramaDist = THREE.MathUtils.clamp(camRig.dioramaDist * (1 + e.deltaY * 0.001), 160, 7400);
    } else if (state.camMode === 'third') {
      camRig.dist = THREE.MathUtils.clamp(camRig.dist * (1 + e.deltaY * 0.001), 8, 120);
    }
  },
  { passive: false }
);

// ---------- Terrain Sampling ----------
function worldToGrid(x, z) {
  const { size, worldSpan } = world.meta;
  const u = (x / worldSpan + 0.5) * (size - 1);
  const v = (z / worldSpan + 0.5) * (size - 1);
  return {
    u: THREE.MathUtils.clamp(u, 0, size - 1.001),
    v: THREE.MathUtils.clamp(v, 0, size - 1.001),
  };
}

function sampleHeight(x, z) {
  if (!world) return 0;
  const { size } = world.meta;
  const { u, v } = worldToGrid(x, z);
  const x0 = Math.floor(u);
  const y0 = Math.floor(v);
  const x1 = Math.min(x0 + 1, size - 1);
  const y1 = Math.min(y0 + 1, size - 1);
  const tx = u - x0;
  const ty = v - y0;

  const h = world.heights;
  const a = h[y0 * size + x0] * (1 - tx) + h[y0 * size + x1] * tx;
  const b = h[y1 * size + x0] * (1 - tx) + h[y1 * size + x1] * tx;
  return (a * (1 - ty) + b * ty) * state.vScale;
}

function sampleSlopeDeg(x, z) {
  const eps = 3.5;
  const hx = (sampleHeight(x + eps, z) - sampleHeight(x - eps, z)) / (2 * eps);
  const hz = (sampleHeight(x, z + eps) - sampleHeight(x, z - eps)) / (2 * eps);
  return THREE.MathUtils.radToDeg(Math.atan(Math.hypot(hx, hz)));
}

function sampleMasks(x, z) {
  if (!world) return { river: 0, lake: 0, curv: 0.5, moisture: 0.5 };
  const { size } = world.meta;
  const { u, v } = worldToGrid(x, z);
  const idx = (Math.round(v) * size + Math.round(u)) * 4;
  return {
    river: world.masks[idx] / 255,
    lake: world.masks[idx + 1] / 255,
    curv: world.masks[idx + 2] / 255,
    moisture: world.masks[idx + 3] / 255,
  };
}

function sampleClimate(x, z) {
  if (!world || !world.meta.hasClimate || !world.climate) return null;
  const { size, tempRange, precipRange } = world.meta;
  const { u, v } = worldToGrid(x, z);
  const idx = (Math.round(v) * size + Math.round(u)) * 4;
  const tempC = tempRange[0] + (world.climate[idx] / 255) * (tempRange[1] - tempRange[0]);
  const precipMm = precipRange[0] + (world.climate[idx + 1] / 255) * (precipRange[1] - precipRange[0]);
  return { tempC, precipMm };
}

// ---------- Dispose Helper ----------
function disposeWorldScene() {
  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMat.dispose();
    terrainMesh = null;
  }
  if (waterMesh) {
    scene.remove(waterMesh);
    waterMesh.geometry.dispose();
    waterMat.dispose();
    waterMesh = null;
  }
  if (riverObj) {
    scene.remove(riverObj.mesh);
    riverObj.mesh.geometry.dispose();
    riverObj.material.dispose();
    riverObj = null;
  }
  if (lakeObj) {
    scene.remove(lakeObj.mesh);
    lakeObj.mesh.geometry.dispose();
    lakeObj.material.dispose();
    lakeObj = null;
  }
  if (grassObj) {
    scene.remove(grassObj.mesh);
    grassObj.mesh.geometry.dispose();
    grassObj.material.dispose();
    grassObj = null;
  }
  if (propsGroup) {
    scene.remove(propsGroup);
    propsGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    propsGroup = null;
  }
  if (heightFloatTex) {
    heightFloatTex.dispose();
    heightFloatTex = null;
  }
  if (maskTex) {
    maskTex.dispose();
    maskTex = null;
  }
  if (climateTex) {
    climateTex.dispose();
    climateTex = null;
  }
}

// ---------- Load World ----------
async function loadWorld(worldId) {
  const loader = document.getElementById('loading');
  loader.classList.remove('hidden');

  const meta = await fetch(`/worlds/${worldId}.json`).then((r) => r.json());
  const fetches = [
    fetch(`/worlds/${worldId}.height.f32`).then((r) => r.arrayBuffer()),
    fetch(`/worlds/${worldId}.masks.u8`).then((r) => r.arrayBuffer()),
    fetch(`/worlds/${worldId}.rivers.json`).then((r) => r.json()),
    fetch(`/worlds/${worldId}.lakes.json`).then((r) => r.json()),
  ];
  if (meta.hasClimate) {
    fetches.push(fetch(`/worlds/${worldId}.climate.u8`).then((r) => r.arrayBuffer()));
  }
  const [heightBuf, maskBuf, splines, lakes, climateBuf] = await Promise.all(fetches);

  disposeWorldScene();

  const heights = new Float32Array(heightBuf);
  const masks = new Uint8Array(maskBuf);
  const climate = climateBuf ? new Uint8Array(climateBuf) : null;
  world = { meta, heights, masks, climate, splines, lakes };

  document.getElementById('world-subtitle').textContent = `${meta.subtitle} · ${lakes.length} lakes, ${splines.length} rivers`;
  const sel = document.getElementById('world-select');
  if (sel) sel.value = worldId;

  const { size, worldSpan, maxElev } = meta;

  // Float32 height texture (full 512 or 1024 resolution for shadows, ocean foam, lakes & grass!)
  const hasFloatLinear = renderer.extensions.has('OES_texture_float_linear');
  heightFloatTex = new THREE.DataTexture(heights, size, size, THREE.RedFormat, THREE.FloatType);
  heightFloatTex.flipY = true;
  heightFloatTex.minFilter = hasFloatLinear ? THREE.LinearFilter : THREE.NearestFilter;
  heightFloatTex.magFilter = hasFloatLinear ? THREE.LinearFilter : THREE.NearestFilter;
  heightFloatTex.needsUpdate = true;

  // Masks texture: R=riverWet, G=lake, B=curvatureAO, A=moisture
  maskTex = new THREE.DataTexture(masks, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  maskTex.flipY = true;
  maskTex.minFilter = THREE.LinearFilter;
  maskTex.magFilter = THREE.LinearFilter;
  maskTex.needsUpdate = true;

  // Climate texture (WorldClim R=temp, G=precip, B=seasonality) or 1x1 neutral fallback
  if (climate) {
    climateTex = new THREE.DataTexture(climate, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  } else {
    climateTex = new THREE.DataTexture(new Uint8Array([128, 128, 128, 255]), 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  }
  climateTex.flipY = true;
  climateTex.minFilter = THREE.LinearFilter;
  climateTex.magFilter = THREE.LinearFilter;
  climateTex.needsUpdate = true;

  // 1. Terrain Mesh (cap geometry grid at 512x512 for locked 60 FPS even on 1024x1024 worlds!)
  const meshVerts = Math.min(size, 512);
  const geo = new THREE.PlaneGeometry(worldSpan, worldSpan, meshVerts - 1, meshVerts - 1);
  geo.rotateX(-Math.PI / 2);
  const posAttr = geo.attributes.position;
  const meshHeights = new Float32Array(meshVerts * meshVerts);
  if (meshVerts === size) {
    for (let i = 0; i < heights.length; i++) {
      posAttr.setY(i, heights[i]);
      meshHeights[i] = heights[i];
    }
  } else {
    for (let vy = 0; vy < meshVerts; vy++) {
      const fy = (vy / (meshVerts - 1)) * (size - 1);
      const y0 = Math.floor(fy);
      const y1 = Math.min(y0 + 1, size - 1);
      const ty = fy - y0;
      for (let vx = 0; vx < meshVerts; vx++) {
        const fx = (vx / (meshVerts - 1)) * (size - 1);
        const x0 = Math.floor(fx);
        const x1 = Math.min(x0 + 1, size - 1);
        const tx = fx - x0;
        const i00 = y0 * size + x0;
        const i10 = y0 * size + x1;
        const i01 = y1 * size + x0;
        const i11 = y1 * size + x1;
        const a = heights[i00] * (1 - tx) + heights[i10] * tx;
        const b = heights[i01] * (1 - tx) + heights[i11] * tx;
        let hVal = a * (1 - ty) + b * ty;
        // Preserve carved riverbeds during 1024->512 mesh reduction so terrain triangles NEVER bridge over streams!
        const maxRiv = Math.max(masks[i00 * 4], masks[i10 * 4], masks[i01 * 4], masks[i11 * 4]);
        if (maxRiv > 45) {
          const minH = Math.min(heights[i00], heights[i10], heights[i01], heights[i11]);
          const blendMin = Math.min(1.0, (maxRiv - 45) / 110.0);
          hVal = hVal * (1.0 - blendMin) + minH * blendMin;
        }
        const idx = vy * meshVerts + vx;
        posAttr.setY(idx, hVal);
        meshHeights[idx] = hVal;
      }
    }
  }
  geo.computeVertexNormals();

  // Exact triangle-interpolated height sampler of the rendered terrain mesh (for zero-clipping river ribbons!)
  const sampleRenderedMeshHeight = (wx, wz) => {
    const gx = ((wx / worldSpan) + 0.5) * (meshVerts - 1);
    const gz = ((wz / worldSpan) + 0.5) * (meshVerts - 1);
    const x0 = Math.max(0, Math.min(meshVerts - 2, Math.floor(gx)));
    const z0 = Math.max(0, Math.min(meshVerts - 2, Math.floor(gz)));
    const fx = Math.max(0, Math.min(1, gx - x0));
    const fz = Math.max(0, Math.min(1, gz - z0));
    const h00 = meshHeights[z0 * meshVerts + x0];
    const h10 = meshHeights[z0 * meshVerts + x0 + 1];
    const h01 = meshHeights[(z0 + 1) * meshVerts + x0];
    const h11 = meshHeights[(z0 + 1) * meshVerts + x0 + 1];
    // Matches THREE.PlaneGeometry diagonal split (a,c,b / b,c,d -> diagonal from (1,0) to (0,1))
    if (fx + fz <= 1.0) {
      return h00 + fx * (h10 - h00) + fz * (h01 - h00);
    }
    return h11 + (1.0 - fx) * (h01 - h11) + (1.0 - fz) * (h10 - h11);
  };

  terrainMat = createTerrainMaterial(
    maskTex,
    heightFloatTex,
    worldSpan,
    maxElev,
    size,
    climateTex,
    Boolean(meta.hasClimate),
    meta.tempRange || [-12.0, 28.0],
    meta.precipRange || [0.0, 1200.0]
  );
  terrainMat.uniforms.uVScale.value = state.vScale;
  terrainMesh = new THREE.Mesh(geo, terrainMat);
  terrainMesh.scale.y = state.vScale;
  scene.add(terrainMesh);
  syncFogDensity();

  // 2. Ocean Water Plane
  const waterGeo = new THREE.PlaneGeometry(worldSpan * 1.6, worldSpan * 1.6, 128, 128);
  waterGeo.rotateX(-Math.PI / 2);
  waterMat = createWaterMaterial(heightFloatTex, maskTex, worldSpan);
  waterMat.uniforms.uVScale.value = state.vScale;
  waterMesh = new THREE.Mesh(waterGeo, waterMat);
  waterMesh.position.y = 0.0;
  scene.add(waterMesh);

  // 3. Real 3D Flowing River Ribbons & Flat-Surfaced Alpine/Valley Lakes
  riverObj = buildRiverSystem(splines, sampleRenderedMeshHeight);
  riverObj.mesh.scale.y = state.vScale;
  scene.add(riverObj.mesh);

  if (lakes.length > 0) {
    lakeObj = buildLakeSystem(lakes, size, worldSpan, heightFloatTex, maskTex);
    lakeObj.mesh.scale.y = state.vScale;
    lakeObj.material.uniforms.uVScale.value = state.vScale;
    scene.add(lakeObj.mesh);
  }
  syncFogDensity();

  // 4. GPU-Instanced 48,000-Blade Wind-Swaying 3D Grass Ring
  grassObj = createGrassField(heightFloatTex, maskTex, worldSpan, maxElev, 48000, 155.0);
  grassObj.material.uniforms.uVScale.value = state.vScale;
  scene.add(grassObj.mesh);

  // 5. Multi-Tier Conifers, Broadleaf Valley Trees & Craggy Cliff/River Boulders
  propsGroup = buildWorldProps(world, state.vScale);
  scene.add(propsGroup);

  updateSun();
  setStyleMode(state.styleMode);
  teleportTo('plains');

  // Scale default Diorama framing cleanly to worldSpan (4.8km vs 1.2km drilled sub-regions!)
  camRig.dioramaDist = THREE.MathUtils.clamp(worldSpan * 0.45, 420, 2200);

  // Sync URL query so refreshing stays on this exact world
  const curUrl = new URL(window.location.href);
  curUrl.searchParams.set('world', worldId);
  window.history.replaceState(null, '', `${curUrl.pathname}?${curUrl.searchParams.toString()}`);

  // Update resolution badge & drill-down UI state
  const pixelM = worldSpan / size;
  const resBadge = document.getElementById('drill-res-badge');
  if (resBadge) resBadge.textContent = `${pixelM.toFixed(1)}m/px`;
  const drillResBox = document.getElementById('drill-result');
  if (drillResBox) drillResBox.style.display = 'none';
  const drillProg = document.getElementById('drill-progress-wrap');
  if (drillProg) drillProg.style.display = 'none';

  // Breadcrumbs: Planet origin + Parent world (if drilled down!)
  const planetLink = document.getElementById('link-planet');
  if (planetLink) {
    if (meta.planetOrigin) {
      const { seed, archetype } = meta.planetOrigin;
      planetLink.href = `/planet.html?seed=${encodeURIComponent(seed)}&archetype=${encodeURIComponent(archetype || 'earth')}`;
      planetLink.textContent = `← Planet #${seed}`;
    } else {
      planetLink.href = '/planet.html';
      planetLink.textContent = 'Spherical Planet Mode →';
    }
  }
  const parentLink = document.getElementById('link-parent');
  if (parentLink) {
    if (meta.drillOrigin?.parentWorldId) {
      parentLink.href = `/index.html?world=${encodeURIComponent(meta.drillOrigin.parentWorldId)}`;
      parentLink.textContent = `↑ Back to Parent`;
      parentLink.title = meta.drillOrigin.parentTitle || meta.drillOrigin.parentWorldId;
      parentLink.style.display = 'inline-block';
    } else {
      parentLink.style.display = 'none';
    }
  }

  rebuildDrillChildrenOverlay();
  rebuildDrillBoxOverlay();

  loader.classList.add('hidden');
}

// ---------- Teleport & Modes ----------
function teleportTo(landmarkKey) {
  if (!world) return;
  const pt = world.meta.landmarks[landmarkKey] || [0.5, 0.5];
  const x = (pt[0] - 0.5) * world.meta.worldSpan;
  const z = (pt[1] - 0.5) * world.meta.worldSpan;
  const y = Math.max(sampleHeight(x, z), 0.5);
  player.pos.set(x, y, z);
  player.velY = 0;
  player.grounded = true;
  explorer.position.copy(player.pos);
  camRig.dioramaCenter.set(x, getDioramaPlaneY(), z);
  pinDropFlash = 1.0;
  rebuildDrillBoxOverlay();

  // Frame toward highest mountain peak for dramatic horizon
  const peak = world.meta.landmarks.peak || [0.5, 0.5];
  const px = (peak[0] - 0.5) * world.meta.worldSpan;
  const pz = (peak[1] - 0.5) * world.meta.worldSpan;
  if (landmarkKey !== 'peak') {
    const dx = px - x;
    const dz = pz - z;
    camRig.yaw = Math.atan2(dx, dz) + Math.PI;
    player.yaw = Math.atan2(dx, dz);
  }
}

function syncFogDensity() {
  const span = world?.meta?.worldSpan || 2400;
  const spanFactor = Math.min(2400 / Math.max(span, 2400), 1.0);
  const baseFog = 0.00044 * (0.62 + 0.38 * spanFactor);
  const fogVal = state.camMode === 'diorama' ? baseFog * 0.24 : baseFog;
  if (terrainMat) terrainMat.uniforms.uFogDensity.value = fogVal;
  if (waterMat) waterMat.uniforms.uFogDensity.value = fogVal;
  if (riverObj) riverObj.material.uniforms.uFogDensity.value = fogVal;
  if (lakeObj) lakeObj.material.uniforms.uFogDensity.value = fogVal;
}

function setCameraMode(mode) {
  state.camMode = mode;
  document.querySelectorAll('[data-cam]').forEach((b) => b.classList.toggle('active', b.dataset.cam === mode));

  if (mode === 'diorama') {
    camera.fov = 24;
    camRig.dioramaCenter.set(player.pos.x, getDioramaPlaneY(), player.pos.z);
    explorer.visible = true;
    explorer.scale.setScalar(2.4);
    explorerPin.visible = true;
  } else if (mode === 'first') {
    camera.fov = 66;
    camRig.pitch = -0.05;
    explorer.visible = false;
    explorerPin.visible = false;
    cursorReticle.visible = false;
  } else {
    camera.fov = 56;
    camRig.pitch = 0.09;
    explorer.visible = true;
    explorer.scale.setScalar(1.0);
    explorerPin.visible = false;
    cursorReticle.visible = false;
  }
  camera.updateProjectionMatrix();
  syncFogDensity();
  if (grassObj) grassObj.mesh.visible = state.styleMode === 0 && mode !== 'diorama';
  if (drillChildrenGroup) drillChildrenGroup.visible = mode === 'diorama';
  rebuildDrillBoxOverlay();
  updatePropsTransform(propsGroup, state.vScale, mode === 'diorama');
  syncPresetPills();
}

function setStyleMode(styleIdx) {
  state.styleMode = styleIdx;
  if (terrainMat) terrainMat.uniforms.uStyleMode.value = styleIdx;
  if (waterMat) waterMat.uniforms.uStyleMode.value = styleIdx;
  if (riverObj) {
    riverObj.material.uniforms.uStyleMode.value = styleIdx;
    riverObj.mesh.visible = styleIdx !== 2;
  }
  if (lakeObj) {
    lakeObj.material.uniforms.uStyleMode.value = styleIdx;
    lakeObj.mesh.visible = styleIdx !== 2;
  }
  if (grassObj) grassObj.mesh.visible = styleIdx === 0 && state.camMode !== 'diorama';
  if (propsGroup) propsGroup.visible = styleIdx !== 2;
  document.querySelectorAll('[data-style]').forEach((b) => b.classList.toggle('active', Number(b.dataset.style) === styleIdx));
  syncPresetPills();
}

function applyPreset(preset) {
  if (preset === 'zelda') {
    setCameraMode('third');
    setStyleMode(0);
    state.sunElevDeg = 26;
    document.getElementById('sun-slider').value = '26';
    updateSun();
  } else if (preset === 'hike') {
    setCameraMode('first');
    setStyleMode(0);
    state.sunElevDeg = 24;
    document.getElementById('sun-slider').value = '24';
    updateSun();
  } else if (preset === 'diorama') {
    setCameraMode('diorama');
    setStyleMode(1);
    state.sunElevDeg = 44;
    document.getElementById('sun-slider').value = '44';
    updateSun();
  }
  syncPresetPills();
}

function syncPresetPills() {
  let active = '';
  if (state.camMode === 'third' && state.styleMode === 0) active = 'zelda';
  else if (state.camMode === 'first' && state.styleMode === 0) active = 'hike';
  else if (state.camMode === 'diorama' && state.styleMode === 1) active = 'diorama';
  document.querySelectorAll('[data-preset]').forEach((b) => b.classList.toggle('active', b.dataset.preset === active));
}

function updateSun() {
  const elev = THREE.MathUtils.degToRad(state.sunElevDeg);
  const azim = THREE.MathUtils.degToRad(state.sunAzimDeg);
  const dir = new THREE.Vector3(
    Math.cos(elev) * Math.sin(azim),
    Math.sin(elev),
    Math.cos(elev) * Math.cos(azim)
  ).normalize();

  sunLight.userData.dir = dir;
  document.getElementById('sun-val').textContent = `${Math.round(state.sunElevDeg)}°`;

  // Golden Hour -> Crisp Alpine Noon palette
  const t = THREE.MathUtils.smoothstep(state.sunElevDeg, 10, 46);
  const zenith = new THREE.Color('#234c88').lerp(new THREE.Color('#2872cc'), t);
  const horizon = new THREE.Color('#f2bc8a').lerp(new THREE.Color('#b6dcf5'), t);
  const fogCol = new THREE.Color('#e2bea0').lerp(new THREE.Color('#a8d0ea'), t);
  const mistCol = new THREE.Color('#f0d4ba').lerp(new THREE.Color('#d4e6f2'), t);

  sunLight.color.set('#ffbc80').lerp(new THREE.Color('#fff5e2'), t);

  skyDome.material.uniforms.uSunDir.value.copy(dir);
  skyDome.material.uniforms.uZenith.value.copy(zenith);
  skyDome.material.uniforms.uHorizon.value.copy(horizon);

  if (terrainMat) {
    terrainMat.uniforms.uSunDir.value.copy(dir);
    terrainMat.uniforms.uFogColor.value.copy(fogCol);
    terrainMat.uniforms.uMistColor.value.copy(mistCol);
  }
  if (waterMat) {
    waterMat.uniforms.uSunDir.value.copy(dir);
    waterMat.uniforms.uFogColor.value.copy(fogCol);
  }
  if (riverObj) {
    riverObj.material.uniforms.uSunDir.value.copy(dir);
    riverObj.material.uniforms.uFogColor.value.copy(fogCol);
  }
  if (lakeObj) {
    lakeObj.material.uniforms.uSunDir.value.copy(dir);
    lakeObj.material.uniforms.uFogColor.value.copy(fogCol);
  }
  if (grassObj) {
    grassObj.material.uniforms.uSunDir.value.copy(dir);
    grassObj.material.uniforms.uFogColor.value.copy(fogCol);
    grassObj.material.uniforms.uMistColor.value.copy(mistCol);
  }
}

// ---------- UI Wiring ----------
document.querySelectorAll('[data-preset]').forEach((b) =>
  b.addEventListener('click', () => applyPreset(b.dataset.preset))
);
document.querySelectorAll('[data-cam]').forEach((b) =>
  b.addEventListener('click', () => setCameraMode(b.dataset.cam))
);
document.querySelectorAll('[data-style]').forEach((b) =>
  b.addEventListener('click', () => setStyleMode(Number(b.dataset.style)))
);
document.querySelectorAll('[data-tp]').forEach((b) =>
  b.addEventListener('click', () => teleportTo(b.dataset.tp))
);
document.getElementById('sun-slider').addEventListener('input', (e) => {
  state.sunElevDeg = Number(e.target.value);
  updateSun();
});
document.getElementById('vscale-slider').addEventListener('input', (e) => {
  state.vScale = Number(e.target.value);
  document.getElementById('vscale-val').textContent = `${state.vScale.toFixed(2)}×`;
  if (terrainMesh) terrainMesh.scale.y = state.vScale;
  if (terrainMat) terrainMat.uniforms.uVScale.value = state.vScale;
  if (waterMat) waterMat.uniforms.uVScale.value = state.vScale;
  if (riverObj) riverObj.mesh.scale.y = state.vScale;
  if (lakeObj) {
    lakeObj.mesh.scale.y = state.vScale;
    lakeObj.material.uniforms.uVScale.value = state.vScale;
  }
  if (grassObj) grassObj.material.uniforms.uVScale.value = state.vScale;
  if (world) player.pos.y = Math.max(sampleHeight(player.pos.x, player.pos.z), 0.2);
  updatePropsTransform(propsGroup, state.vScale, state.camMode === 'diorama');
  rebuildDrillBoxOverlay();
  rebuildDrillChildrenOverlay();
});
document.getElementById('world-select').addEventListener('change', (e) => loadWorld(e.target.value));
document.getElementById('sel-drill-zoom')?.addEventListener('change', (e) => {
  drillState.zoomFactor = Number(e.target.value) || 4;
  rebuildDrillBoxOverlay();
});
document.getElementById('btn-drill-aim')?.addEventListener('click', () => {
  drillState.aiming = !drillState.aiming;
  document.getElementById('btn-drill-aim')?.classList.toggle('active', drillState.aiming);
  if (drillState.aiming) {
    if (state.camMode !== 'diorama') setCameraMode('diorama');
    if (world) {
      camRig.dioramaCenter.set(player.pos.x, getDioramaPlaneY(), player.pos.z);
      camRig.dioramaDist = Math.max(camRig.dioramaDist, Math.min(world.meta.worldSpan * 0.55, 2400));
    }
  }
  rebuildDrillBoxOverlay();
});
document.getElementById('btn-drill-run')?.addEventListener('click', () => {
  runDrillDownAtPin();
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- Main Loop ----------
const clock = new THREE.Clock();
const elActivity = document.getElementById('stat-activity');
const elElev = document.getElementById('stat-elev');
const elSlope = document.getElementById('stat-slope');
const elTemp = document.getElementById('stat-temp');
const elZone = document.getElementById('stat-zone');

function updateMovement(dt) {
  if (!world) return;
  const halfSpan = world.meta.worldSpan * 0.48;

  const forwardInput = (keys.has('w') || keys.has('arrowup') ? 1 : 0) - (keys.has('s') || keys.has('arrowdown') ? 1 : 0);
  const rightInput = (keys.has('d') || keys.has('arrowright') ? 1 : 0) - (keys.has('a') || keys.has('arrowleft') ? 1 : 0);
  const sprint = keys.has('shift');

  if (state.camMode === 'diorama') {
    const panSpeed = (sprint ? 260 : 140) * (camRig.dioramaDist / 700);
    const fwd = new THREE.Vector3(-Math.sin(camRig.dioramaYaw), 0, -Math.cos(camRig.dioramaYaw));
    const right = new THREE.Vector3(Math.cos(camRig.dioramaYaw), 0, -Math.sin(camRig.dioramaYaw));
    camRig.dioramaCenter.addScaledVector(fwd, forwardInput * panSpeed * dt);
    camRig.dioramaCenter.addScaledVector(right, rightInput * panSpeed * dt);
    camRig.dioramaCenter.x = THREE.MathUtils.clamp(camRig.dioramaCenter.x, -halfSpan, halfSpan);
    camRig.dioramaCenter.z = THREE.MathUtils.clamp(camRig.dioramaCenter.z, -halfSpan, halfSpan);
    // Keep camera glide strictly on a level aerial plane (zero jumping up/down over ridges!)
    camRig.dioramaCenter.y = getDioramaPlaneY();

    const pitch = THREE.MathUtils.degToRad(40);
    const hDist = Math.cos(pitch) * camRig.dioramaDist;
    const vDist = Math.sin(pitch) * camRig.dioramaDist;
    const camX = camRig.dioramaCenter.x + Math.sin(camRig.dioramaYaw) * hDist;
    const camZ = camRig.dioramaCenter.z + Math.cos(camRig.dioramaYaw) * hDist;
    // Only prevent camera itself from ever dipping into a tall mountain when zoomed super tight
    const camY = Math.max(camRig.dioramaCenter.y + vDist, sampleHeight(camX, camZ) + 45);
    camera.position.set(camX, camY, camZ);
    camera.lookAt(camRig.dioramaCenter);

    // Keep explorer & high-contrast map pin synced to player.pos, scaled nicely for current zoom
    const groundAtPlayer = Math.max(sampleHeight(player.pos.x, player.pos.z), 0.2);
    player.pos.y = groundAtPlayer;
    explorer.position.copy(player.pos);

    const t = clock.elapsedTime;
    const pinScale = THREE.MathUtils.clamp(camRig.dioramaDist / 420, 0.85, 11.5);
    explorerPin.visible = true;
    explorerPin.position.set(player.pos.x, groundAtPlayer, player.pos.z);
    explorerPin.scale.setScalar(pinScale);

    const pinHead = explorerPin.getObjectByName('pinHead');
    if (pinHead) {
      const dropBounce = pinDropFlash > 0 ? Math.sin(pinDropFlash * Math.PI) * 5.0 : 0;
      pinHead.position.y = 14.2 + Math.sin(t * 3.0) * 0.85 + dropBounce;
      pinHead.rotation.y = camRig.dioramaYaw;
    }
    const pulseRing = explorerPin.getObjectByName('pinPulseRing');
    if (pulseRing) {
      const phase = (t * 0.75) % 1.0;
      const boost = pinDropFlash > 0 ? 1.0 + (1.0 - pinDropFlash) * 2.4 : 1.0 + phase * 1.45;
      pulseRing.scale.setScalar(boost);
      pulseRing.material.opacity = pinDropFlash > 0 ? pinDropFlash * 0.95 : (1.0 - phase) * 0.65;
    }
    if (pinDropFlash > 0) pinDropFlash = Math.max(0, pinDropFlash - dt * 2.4);

    // Update live terrain cursor reticle for 'T' / double-click relocation
    if (pointerNdc.inside && !isDragging) {
      dioramaCursorHit = pickTerrainAtNdc(pointerNdc.x, pointerNdc.y);
    } else {
      dioramaCursorHit = null;
    }
    if (dioramaCursorHit) {
      cursorReticle.visible = true;
      cursorReticle.position.copy(dioramaCursorHit);
      cursorReticle.scale.setScalar(THREE.MathUtils.clamp(camRig.dioramaDist / 480, 0.75, 9.5));
    } else {
      cursorReticle.visible = false;
    }

    player.activity = 'DIORAMA VIEW';
    return;
  }

  const slopeDeg = sampleSlopeDeg(player.pos.x, player.pos.z);
  const m = sampleMasks(player.pos.x, player.pos.z);
  const groundY = sampleHeight(player.pos.x, player.pos.z);
  const inWater = groundY < 0.8 || m.river > 0.42 || m.lake > 0.4;
  const climbing = slopeDeg > 28 && !inWater && groundY > 2.0;

  let speed = sprint ? 60 : 28;
  if (climbing) speed *= 0.46;
  else if (inWater) speed *= 0.62;

  const move = new THREE.Vector3();
  if (forwardInput !== 0 || rightInput !== 0) {
    const camFwd = new THREE.Vector3(-Math.sin(camRig.yaw), 0, -Math.cos(camRig.yaw));
    const camRight = new THREE.Vector3(Math.cos(camRig.yaw), 0, -Math.sin(camRig.yaw));
    move.addScaledVector(camFwd, forwardInput).addScaledVector(camRight, rightInput).normalize();
    player.pos.x = THREE.MathUtils.clamp(player.pos.x + move.x * speed * dt, -halfSpan, halfSpan);
    player.pos.z = THREE.MathUtils.clamp(player.pos.z + move.z * speed * dt, -halfSpan, halfSpan);
    player.yaw = Math.atan2(move.x, move.z);
  }

  const floorY = Math.max(sampleHeight(player.pos.x, player.pos.z), -0.4);
  if (keys.has(' ') && player.grounded) {
    player.velY = 22;
    player.grounded = false;
  }
  player.velY -= 58 * dt;
  player.pos.y += player.velY * dt;
  if (player.pos.y <= floorY) {
    player.pos.y = floorY;
    player.velY = 0;
    player.grounded = true;
  }

  const t = clock.elapsedTime;
  const moving = move.lengthSq() > 0.01;
  explorer.position.copy(player.pos);
  explorer.rotation.y = player.yaw + Math.PI;
  explorer.rotation.x = climbing && moving ? 0.25 : 0;
  if (moving && player.grounded) {
    explorer.position.y += Math.abs(Math.sin(t * (sprint ? 13 : 9))) * (climbing ? 0.22 : 0.4);
  }
  const cape = explorer.getObjectByName('cape');
  if (cape) cape.rotation.x = 0.2 + (moving ? 0.35 : 0.05) + Math.sin(t * 6) * 0.08;

  if (!player.grounded) player.activity = 'JUMPING';
  else if (inWater) player.activity = m.river > 0.42 ? 'CROSSING RIVER' : 'WADING';
  else if (climbing) player.activity = 'CLIMBING';
  else if (moving) player.activity = sprint ? 'SPRINTING' : 'WALKING';
  else player.activity = 'EXPLORING';

  if (state.camMode === 'first') {
    const eye = new THREE.Vector3(player.pos.x, player.pos.y + 4.2, player.pos.z);
    const lookDir = new THREE.Vector3(
      -Math.sin(camRig.yaw) * Math.cos(camRig.pitch),
      -Math.sin(camRig.pitch),
      -Math.cos(camRig.yaw) * Math.cos(camRig.pitch)
    ).normalize();
    const aheadH = sampleHeight(eye.x + lookDir.x * 8, eye.z + lookDir.z * 8);
    if (aheadH + 2.0 > eye.y) eye.y = aheadH + 2.0;
    camera.position.copy(eye);
    camera.lookAt(eye.clone().add(lookDir));
  } else {
    const target = new THREE.Vector3(player.pos.x, player.pos.y + 4.2, player.pos.z);
    const hDist = Math.cos(camRig.pitch) * camRig.dist;
    const vDist = Math.sin(camRig.pitch) * camRig.dist;
    const camPos = new THREE.Vector3(
      target.x + Math.sin(camRig.yaw) * hDist,
      target.y + vDist,
      target.z + Math.cos(camRig.yaw) * hDist
    );
    const camFloor = sampleHeight(camPos.x, camPos.z) + 2.5;
    if (camPos.y < camFloor) camPos.y = camFloor;
    camera.position.copy(camPos);
    camera.lookAt(target);
  }
}

function updateTelemetry() {
  if (!world) return;
  const pt =
    state.camMode === 'diorama'
      ? dioramaCursorHit || camRig.dioramaCenter
      : player.pos;
  const elev = sampleHeight(pt.x, pt.z);
  const slope = sampleSlopeDeg(pt.x, pt.z);
  const m = sampleMasks(pt.x, pt.z);
  const clim = sampleClimate(pt.x, pt.z);
  const hNorm = elev / Math.max(world.meta.maxElev * state.vScale, 1);

  elElev.textContent = `${Math.round(elev)} m`;
  elSlope.textContent = `${Math.round(slope)}°`;
  if (elTemp) {
    elTemp.textContent = clim ? `${Math.round(clim.tempC)}°C` : '—';
  }
  elActivity.textContent = player.activity;
  elActivity.className = 'v badge';
  if (player.activity === 'CLIMBING') elActivity.classList.add('climbing');
  else if (player.activity.includes('RIVER') || player.activity === 'WADING') elActivity.classList.add('wading');
  else if (player.activity.includes('DIORAMA')) elActivity.classList.add('diorama');

  let zone = 'Meadow Plains';
  if (elev <= 0.5) zone = 'Ocean Shallows';
  else if (m.river > 0.35) zone = 'River Valley';
  else if (m.lake > 0.4) zone = 'Highland Lake';
  else if (clim) {
    if (clim.tempC < 1.2) zone = 'Glacial Alpine';
    else if (slope > 28) zone = 'Mountain Cliff';
    else if (elev < 8) zone = 'Sandy Coast';
    else if (clim.tempC < 6.5) zone = 'Boreal Highlands';
    else if (clim.tempC > 11 && clim.precipMm < 260) zone = 'Warm Dry Steppe';
    else if (clim.precipMm > 340) zone = 'Temperate Forest';
    else zone = 'Coastal Meadow';
  } else {
    if (hNorm > 0.65) zone = 'Alpine Summit';
    else if (slope > 28) zone = 'Mountain Cliff';
    else if (elev < 8) zone = 'Sandy Coast';
    else if (m.moisture < 0.38) zone = 'Arid Steppe';
    else if (m.moisture > 0.64) zone = 'Woodland Valley';
  }
  elZone.textContent = zone;
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.elapsedTime;

  updateMovement(dt);

  // Follow player with directional shadow frustum & 48K GPU grass ring
  const focus = state.camMode === 'diorama' ? camRig.dioramaCenter : player.pos;
  if (sunLight.userData.dir) {
    sunLight.target.position.copy(focus);
    sunLight.position.copy(focus).addScaledVector(sunLight.userData.dir, 360);
  }

  if (terrainMat) terrainMat.uniforms.uTime.value = t;
  if (waterMat) waterMat.uniforms.uTime.value = t;
  if (riverObj) riverObj.material.uniforms.uTime.value = t;
  if (lakeObj) lakeObj.material.uniforms.uTime.value = t;
  if (grassObj) {
    grassObj.material.uniforms.uTime.value = t;
    grassObj.material.uniforms.uFocusPos.value.copy(focus);
    grassObj.material.uniforms.uPlayerPos.value.copy(player.pos);
  }

  updateTelemetry();
  renderer.render(scene, camera);
}

// ---------- Boot ----------
async function init() {
  const baseCatalog = await fetch('/worlds/catalog.json', { cache: 'no-store' }).then((r) => r.json());
  let synthCatalog = [];
  try {
    const r = await fetch('/worlds/synthesized_catalog.json', { cache: 'no-store' });
    if (r.ok) synthCatalog = await r.json();
  } catch {
    // No locally synthesized worlds yet
  }
  const seenIds = new Set(baseCatalog.map((w) => w.id));
  catalog = [...baseCatalog, ...synthCatalog.filter((w) => !seenIds.has(w.id))];

  const select = document.getElementById('world-select');
  select.innerHTML = catalog.map((w) => `<option value="${w.id}">${w.title}</option>`).join('');

  const reqWorld = new URLSearchParams(window.location.search).get('world');
  const startId = reqWorld && catalog.some((w) => w.id === reqWorld) ? reqWorld : catalog[0].id;
  select.value = startId;

  await loadWorld(startId);
  applyPreset('zelda');
  animate();
}

window.__terragrok = {
  applyPreset,
  loadWorld,
  teleportTo,
  setCameraMode,
  setStyleMode,
  relocateExplorer: relocateExplorerTo,
  getPlayerPos: () => ({ x: player.pos.x, y: player.pos.y, z: player.pos.z }),
  getDioramaCenter: () => ({ x: camRig.dioramaCenter.x, y: camRig.dioramaCenter.y, z: camRig.dioramaCenter.z }),
  getCameraY: () => camera.position.y,
  isPinVisible: () => explorerPin.visible,
  setDrillAim(on, zoomFactor = null) {
    drillState.aiming = !!on;
    if (zoomFactor) {
      drillState.zoomFactor = zoomFactor;
      const sel = document.getElementById('sel-drill-zoom');
      if (sel) sel.value = String(zoomFactor);
    }
    document.getElementById('btn-drill-aim')?.classList.toggle('active', drillState.aiming);
    rebuildDrillBoxOverlay();
  },
  runDrillDown: runDrillDownAtPin,
  getDrillResult: () => drillState.lastResult,
  focusDiorama(x, z, dist = 340) {
    setCameraMode('diorama');
    camRig.dioramaCenter.set(x, getDioramaPlaneY(), z);
    camRig.dioramaDist = dist;
  },
};

init();
