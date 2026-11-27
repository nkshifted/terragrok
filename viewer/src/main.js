import * as THREE from 'three';
import './style.css';
import { createSkyDome, createTerrainMaterial, createWaterMaterial } from './terrainShader.js';
import { buildLakeSystem, buildRiverSystem } from './rivers.js';
import { createGrassField } from './grass.js';
import { buildWorldProps, updatePropsTransform } from './props.js';
import { buildTerrainClipmapGeometry } from './clipmap.js';
import { createTileStreamController } from './tileStream.js';
import {
  buildSeamAtlas,
  sampleAtlasClimate,
  sampleAtlasHeight,
  sampleAtlasMasks,
  seamOffsetAt,
  tileAt,
} from './seamAtlas.js';

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

// ---------- Streamed chart tiles ----------
// A world synthesized from the planet can grow: the Stage 2 server streams the tiles around the
// player and the explorer stitches the 3×3 window under them into one seam atlas. While `atlas`
// is set, terrain / water / grass / sampling all read it instead of the home tile's own rasters.
let tileStream = null;
/** @type {null | ReturnType<typeof buildSeamAtlas>} */
let atlas = null;
let atlasDirty = false;
/** Per-tile rivers, lakes and props in atlas mode: key -> { sig, river, lake, props }. */
const tileDecor = new Map();
/**
 * Tiles outside the stitched 3×3, drawn from low-detail copies so explored terrain and the
 * ghosts ahead stay on screen: key -> { sig, mesh, mat, textures, lake, props }.
 */
const farDecor = new Map();
let farGeo = null;
let homeMeanClimate = [128, 128, 128];
let subtitleBase = '';

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
skyDome.frustumCulled = false;
skyDome.renderOrder = -10;
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
// UI is hidden (index.html #section-drilldown[hidden]); keep helpers for possible future streaming LOD.
const DRILLDOWN_UI_ENABLED = false;
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
  if (!DRILLDOWN_UI_ENABLED || !world || !drillState.aiming || state.camMode !== 'diorama') return;
  const { cx, cz, boxSpan } = getClampedDrillCenterUV();
  const liftY = Math.max(world.meta.worldSpan * 0.0022, 1.5);
  drillBoxGroup = buildTerrainBoxMesh(cx, cz, boxSpan, '#38f2d8', 0.10, liftY);
  scene.add(drillBoxGroup);
}

function rebuildDrillChildrenOverlay() {
  disposeGroup(drillChildrenGroup);
  drillChildrenGroup = null;
  if (!DRILLDOWN_UI_ENABLED || !world) return;
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
  // Streamed tiles and ghosts belong to their home world; they are not worlds to open on their own
  catalog = [...baseCatalog, ...synthCatalog.filter((w) => !seenIds.has(w.id) && !w.tileOrigin)];
  const select = document.getElementById('world-select');
  if (select) {
    const curVal = world?.meta?.id || select.value;
    select.innerHTML = catalog.map((w) => `<option value="${w.id}">${w.title}</option>`).join('');
    if (curVal && catalog.some((w) => w.id === curVal)) select.value = curVal;
  }
}

async function runDrillDownAtPin() {
  if (!DRILLDOWN_UI_ENABLED || !world || drillState.running) return;
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

  const bounds = worldBounds();
  const maxY = Math.max(Math.max(world.meta.maxElev, atlas ? 460 : 0) * state.vScale + 30, 60);
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
      if (fx < bounds.minX || fx > bounds.maxX || fz < bounds.minZ || fz > bounds.maxZ) return null;
      const fy = Math.max(sampleHeight(fx, fz), 0);
      return new THREE.Vector3(fx, fy, fz);
    }
    prevT = t;
  }
  return null;
}

function relocateExplorerTo(x, z) {
  if (!world) return false;
  // A pin dropped on a ghost / unsynthesized tile is ignored rather than snapped elsewhere
  if (atlas && !canWalk(x, z)) return false;
  const { x: cx, z: cz } = clampToWalkable(x, z);
  const cy = Math.max(sampleHeight(cx, cz), 0.2);
  player.pos.set(cx, cy, cz);
  player.velY = 0;
  player.grounded = true;
  explorer.position.copy(player.pos);
  pinDropFlash = 1.0;
  rebuildDrillBoxOverlay();
  // Landed on a far tile: recentre the stitched window on it now, not at the next heartbeat
  if (atlas) tileStream.poke();
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

// ---------- Terrain Sampling / Walk Bounds ----------
/** Extent of everything drawn: the home tile, or the 3×3 seam atlas around the player. */
function worldBounds() {
  if (!world) return { minX: 0, maxX: 0, minZ: 0, maxZ: 0 };
  if (atlas) {
    const h = (tileStream.farRing + 0.5) * atlas.meta.worldSpan;
    const { centerX: cx, centerZ: cz } = atlas.meta;
    return { minX: cx - h, maxX: cx + h, minZ: cz - h, maxZ: cz + h };
  }
  const lim = world.meta.worldSpan * 0.5;
  return { minX: -lim, maxX: lim, minZ: -lim, maxZ: lim };
}

function canWalk(x, z) {
  if (!world) return false;
  if (atlas) return tileStream.isWalkable(x, z);
  const lim = world.meta.worldSpan * 0.5 - 1;
  return Math.abs(x) <= lim && Math.abs(z) <= lim;
}

/** Nearest walkable spot: (x, z) itself, else clamped into the tile the player stands on. */
function clampToWalkable(x, z) {
  if (!world) return { x, z };
  const span = world.meta.worldSpan;
  if (atlas) {
    if (canWalk(x, z)) return { x, z };
    const here = canWalk(player.pos.x, player.pos.z) ? tileAt(span, player.pos.x, player.pos.z) : { qi: 0, qj: 0 };
    const lim = span * 0.5 - 1;
    return {
      x: THREE.MathUtils.clamp(x, here.qi * span - lim, here.qi * span + lim),
      z: THREE.MathUtils.clamp(z, here.qj * span - lim, here.qj * span + lim),
    };
  }
  const lim = Math.max(0, span * 0.5 - 1);
  return { x: THREE.MathUtils.clamp(x, -lim, lim), z: THREE.MathUtils.clamp(z, -lim, lim) };
}

/** One walking step with wall-sliding against tiles that are not synthesized yet. */
function resolveWalk(x, z, nextX, nextZ) {
  if (!atlas) {
    const c = clampToWalkable(nextX, nextZ);
    return { x: c.x, z: c.z, blocked: c.x !== nextX || c.z !== nextZ };
  }
  if (canWalk(nextX, nextZ)) return { x: nextX, z: nextZ, blocked: false };
  if (canWalk(nextX, z)) return { x: nextX, z, blocked: true };
  if (canWalk(x, nextZ)) return { x, z: nextZ, blocked: true };
  return { x, z, blocked: true };
}

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
  if (atlas) {
    // Stitched surface near the player; low-detail far copies beyond it; open sea elsewhere
    const t = tileAt(atlas.meta.worldSpan, x, z);
    let raw = atlas.slots.has(`${t.qi},${t.qj}`) ? sampleAtlasHeight(atlas, x, z) : null;
    if (raw == null) raw = tileStream.sampleFarHeight(x, z);
    return (raw == null ? -30 : raw) * state.vScale;
  }
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

// ---------- Seam atlas (streamed tiles) ----------
function makeAtlasTexture(data, size, format, type, linear) {
  const tex = new THREE.DataTexture(data, size, size, format, type);
  tex.flipY = true;
  tex.minFilter = linear ? THREE.LinearFilter : THREE.NearestFilter;
  tex.magFilter = linear ? THREE.LinearFilter : THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}

function replaceTextureData(tex, data) {
  tex.image.data = data;
  tex.needsUpdate = true;
}

/** Re-stitch the 3×3 window around the player and rebuild whatever decor it invalidated. */
function rebuildChart() {
  atlasDirty = false;
  if (!world || !tileStream?.isLive() || !terrainMat) return;
  const t0 = performance.now();
  const win = tileStream.getWindow();
  const span = world.meta.worldSpan;
  const entering = !atlas;
  atlas = buildSeamAtlas({
    centerQi: win.centerQi,
    centerQj: win.centerQj,
    tiles: win.tiles,
    tileN: tileStream.tileN,
    climN: tileStream.climN,
    worldSpan: span,
    fallbackClimate: homeMeanClimate,
  });
  const { size: W, climateSize: CW, atlasSpan, centerX, centerZ } = atlas.meta;

  if (entering) {
    // Home-only rasters and decor give way to the atlas and per-tile decor
    disposeHomeDecor();
    heightFloatTex.dispose();
    maskTex.dispose();
    climateTex.dispose();
    const hasFloatLinear = renderer.extensions.has('OES_texture_float_linear');
    heightFloatTex = makeAtlasTexture(atlas.heights, W, THREE.RedFormat, THREE.FloatType, hasFloatLinear);
    maskTex = makeAtlasTexture(atlas.masks, W, THREE.RGBAFormat, THREE.UnsignedByteType, true);
    climateTex = makeAtlasTexture(atlas.climate, CW, THREE.RGBAFormat, THREE.UnsignedByteType, true);
    terrainMesh.geometry.dispose();
    terrainMesh.geometry = buildTerrainClipmapGeometry(atlasSpan, W, 128);
  } else {
    replaceTextureData(heightFloatTex, atlas.heights);
    replaceTextureData(maskTex, atlas.masks);
    replaceTextureData(climateTex, atlas.climate);
  }

  const tu = terrainMat.uniforms;
  tu.uHeightTex.value = heightFloatTex;
  tu.uMasks.value = maskTex;
  tu.uClimate.value = climateTex;
  tu.uHasClimate.value = 1;
  tu.uWorldSpan.value = atlasSpan;
  tu.uGridSize.value = W;
  tu.uCellReg.value = 1;
  tu.uTileOrigin.value.set(centerX, centerZ);

  const wu = waterMat.uniforms;
  wu.uHeightTex.value = heightFloatTex;
  wu.uMasks.value = maskTex;
  wu.uWorldSpan.value = atlasSpan;
  wu.uGridSize.value = W;
  wu.uCellReg.value = 1;
  wu.uTexOrigin.value.set(centerX, centerZ);
  waterMesh.position.set(centerX, 0, centerZ);
  // One sea for everything on screen, far tiles included
  waterMesh.scale.setScalar(((2 * tileStream.farRing + 1.5) * span) / (span * 1.6));
  waterMesh.scale.y = 1;

  if (grassObj) {
    const gu = grassObj.material.uniforms;
    gu.uHeightTex.value = heightFloatTex;
    gu.uMasks.value = maskTex;
    gu.uWorldSpan.value = atlasSpan;
    gu.uTexOrigin.value.set(centerX, centerZ);
  }

  const built = rebuildTileDecor(win);
  const farBuilt = rebuildFarTiles();
  for (const d of tileDecor.values()) {
    if (!d.lake) continue;
    const lu = d.lake.material.uniforms;
    lu.uHeightTex.value = heightFloatTex;
    lu.uMasks.value = maskTex;
    lu.uWorldSpan.value = atlasSpan;
    lu.uTexOrigin.value.set(centerX, centerZ);
  }
  updateSun();
  setStyleMode(state.styleMode);
  syncFogDensity();
  updatePropsTransformAll();
  if (player.grounded) player.pos.y = Math.max(sampleHeight(player.pos.x, player.pos.z), 0.2);
  console.info(
    `[seam-atlas] centre ${win.centerQi},${win.centerQj} · ${win.tiles.size} tiles · ` +
      `${built} decor rebuilt · ${farDecor.size} far tiles (${farBuilt} new) · ${(performance.now() - t0).toFixed(0)} ms`
  );
}

function disposeDecor(d) {
  if (d.river) {
    scene.remove(d.river.mesh);
    d.river.mesh.geometry.dispose();
    d.river.material.dispose();
  }
  if (d.lake) {
    scene.remove(d.lake.mesh);
    d.lake.mesh.geometry.dispose();
    d.lake.material.dispose();
  }
  if (d.props) {
    scene.remove(d.props);
    d.props.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
  }
}

function disposeHomeDecor() {
  disposeDecor({ river: riverObj, lake: lakeObj, props: propsGroup });
  riverObj = null;
  lakeObj = null;
  propsGroup = null;
}

function clearTileDecor() {
  for (const d of tileDecor.values()) disposeDecor(d);
  tileDecor.clear();
  for (const d of farDecor.values()) disposeFar(d);
  farDecor.clear();
}

function disposeFar(d) {
  scene.remove(d.mesh);
  d.mat.dispose();
  for (const t of d.textures) t.dispose();
  disposeDecor({ river: null, lake: d.lake, props: d.props });
}

/**
 * Low-detail stand-ins for every known tile outside the stitched window: the real terrain
 * shader over a 256² copy of the tile, its lakes, and a thinned-out scatter of trees and
 * rocks. No rivers (ribbons need the full-resolution bed) and no walking: stepping or
 * teleporting onto one recentres the window, which swaps in the full tile.
 */
function rebuildFarTiles() {
  const span = world.meta.worldSpan;
  const hasFloatLinear = renderer.extensions.has('OES_texture_float_linear');
  const want = new Map(tileStream.getFarTiles().map((e) => [e.key, e]));
  for (const [key, d] of farDecor) {
    if (want.get(key)?.version !== d.sig) {
      disposeFar(d);
      farDecor.delete(key);
    }
  }
  if (!farGeo) farGeo = buildTerrainClipmapGeometry(span, 256, 64);
  let built = 0;
  for (const [key, e] of want) {
    if (farDecor.has(key)) continue;
    const ox = e.qi * span;
    const oz = e.qj * span;
    const n = e.size;
    const cn = e.climateSize;

    const hTex = makeAtlasTexture(e.heights, n, THREE.RedFormat, THREE.FloatType, hasFloatLinear);
    let masks = e.masks;
    if (!masks) {
      masks = new Uint8Array(n * n * 4);
      for (let i = 0; i < n * n; i++) {
        masks[i * 4 + 2] = 128;
        masks[i * 4 + 3] = 110;
      }
    }
    const mTex = makeAtlasTexture(masks, n, THREE.RGBAFormat, THREE.UnsignedByteType, true);
    let climate = e.climate;
    if (!climate || e.ghost) {
      // Ghosts carry no climate: borrow the home's average and flag them (alpha 0) for the haze
      climate = new Uint8Array(cn * cn * 4);
      for (let i = 0; i < cn * cn; i++) {
        climate[i * 4] = homeMeanClimate[0];
        climate[i * 4 + 1] = homeMeanClimate[1];
        climate[i * 4 + 2] = homeMeanClimate[2];
        climate[i * 4 + 3] = e.ghost ? 0 : 255;
      }
    }
    const cTex = makeAtlasTexture(climate, cn, THREE.RGBAFormat, THREE.UnsignedByteType, true);

    const mat = createTerrainMaterial(
      mTex,
      hTex,
      span,
      world.meta.maxElev,
      n,
      cTex,
      true,
      world.meta.tempRange || [-16.0, 32.0],
      world.meta.precipRange || [0.0, 1800.0]
    );
    mat.uniforms.uCellReg.value = 1;
    mat.uniforms.uTileOrigin.value.set(ox, oz);
    // The clipmap stays centred on its own tile: uniform (coarse) detail across it
    mat.uniforms.uFocusPos.value.set(ox, 0, oz);
    mat.uniforms.uVScale.value = state.vScale;
    const mesh = new THREE.Mesh(farGeo, mat);
    mesh.frustumCulled = false;
    scene.add(mesh);

    const d = { sig: e.version, mesh, mat, textures: [hTex, mTex, cTex], lake: null, props: null };
    if (!e.ghost) {
      if (e.lakes?.length) {
        d.lake = buildLakeSystem(e.lakes, e.meta.lakeGridSize || e.meta.size, span, hTex, mTex);
        d.lake.mesh.position.set(ox, 0, oz);
        d.lake.mesh.scale.y = state.vScale;
        d.lake.material.uniforms.uVScale.value = state.vScale;
        d.lake.material.uniforms.uTexOrigin.value.set(ox, oz);
        scene.add(d.lake.mesh);
      }
      d.props = buildWorldProps(
        {
          meta: { ...e.meta, size: n, climateSize: cn, hasClimate: Boolean(e.climate) },
          heights: e.heights,
          masks,
          climate: e.climate,
        },
        state.vScale,
        { originX: ox, originZ: oz, seed: (4242 + e.qi * 7919 + e.qj * 104729) >>> 0, density: 0.3 }
      );
      scene.add(d.props);
    }
    farDecor.set(key, d);
    built++;
  }
  return built;
}

/** Terrain materials of the far tiles (the stitched terrain's own is `terrainMat`). */
function farMats() {
  const out = [];
  for (const d of farDecor.values()) out.push(d.mat);
  return out;
}

/**
 * Rivers, lakes and props for every real tile in the window, positioned on the stitched
 * surface. A tile is rebuilt only when it or a neighbour changed (neighbours move its seams).
 */
function rebuildTileDecor(win) {
  const span = world.meta.worldSpan;
  const surface = (x, z) => {
    const h = sampleAtlasHeight(atlas, x, z);
    return h == null ? 0 : h;
  };
  let built = 0;
  for (const [key, d] of tileDecor) {
    if (!win.tiles.has(key) || win.tiles.get(key).ghost) {
      disposeDecor(d);
      tileDecor.delete(key);
    }
  }
  for (const [key, e] of win.tiles) {
    if (e.ghost) continue;
    let sig = `${e.version}`;
    for (let dj = -1; dj <= 1; dj++) {
      for (let di = -1; di <= 1; di++) {
        if (di === 0 && dj === 0) continue;
        sig += `|${win.tiles.get(`${e.qi + di},${e.qj + dj}`)?.version ?? '-'}`;
      }
    }
    const prev = tileDecor.get(key);
    if (prev?.sig === sig) continue;
    if (prev) disposeDecor(prev);

    const ox = e.qi * span;
    const oz = e.qj * span;
    const d = { sig, river: null, lake: null, props: null };

    // River points are tile-local; move them into the world and onto the stitched surface
    const rivers = (e.rivers || []).map((r) => {
      const pts = (Array.isArray(r) ? r : r.points).map((p) => {
        const x = p[0] + ox;
        const z = p[1] + oz;
        return [x, z, p[2] + seamOffsetAt(atlas, x, z), ...p.slice(3)];
      });
      return Array.isArray(r) ? pts : { ...r, points: pts };
    });
    if (rivers.length) {
      d.river = buildRiverSystem(rivers, surface);
      d.river.mesh.scale.y = state.vScale;
      scene.add(d.river.mesh);
    }

    const lakes = (e.lakes || []).map((l) => ({ ...l, level: l.level + seamOffsetAt(atlas, l.cx + ox, l.cz + oz) }));
    if (lakes.length) {
      d.lake = buildLakeSystem(lakes, e.meta.lakeGridSize || e.meta.size, span, heightFloatTex, maskTex);
      d.lake.mesh.position.set(ox, 0, oz);
      d.lake.mesh.scale.y = state.vScale;
      d.lake.material.uniforms.uVScale.value = state.vScale;
      scene.add(d.lake.mesh);
    }

    d.props = buildWorldProps(
      {
        meta: { ...e.meta, size: tileStream.tileN, climateSize: tileStream.climN, hasClimate: Boolean(e.slot.climate) },
        heights: e.slot.grid,
        masks: e.slot.masks,
        climate: e.slot.climate,
      },
      state.vScale,
      { originX: ox, originZ: oz, surfaceAt: surface, seed: (4242 + e.qi * 7919 + e.qj * 104729) >>> 0 }
    );
    scene.add(d.props);

    tileDecor.set(key, d);
    built++;
  }
  return built;
}

function allRivers() {
  const out = riverObj ? [riverObj] : [];
  for (const d of tileDecor.values()) if (d.river) out.push(d.river);
  return out;
}

function allLakes() {
  const out = lakeObj ? [lakeObj] : [];
  for (const d of tileDecor.values()) if (d.lake) out.push(d.lake);
  for (const d of farDecor.values()) if (d.lake) out.push(d.lake);
  return out;
}

function allProps() {
  const out = propsGroup ? [propsGroup] : [];
  for (const d of tileDecor.values()) if (d.props) out.push(d.props);
  for (const d of farDecor.values()) if (d.props) out.push(d.props);
  return out;
}

function updatePropsTransformAll() {
  for (const g of allProps()) updatePropsTransform(g, state.vScale, state.camMode === 'diorama');
}

function sampleSlopeDeg(x, z) {
  const eps = 3.5;
  const hx = (sampleHeight(x + eps, z) - sampleHeight(x - eps, z)) / (2 * eps);
  const hz = (sampleHeight(x, z + eps) - sampleHeight(x, z - eps)) / (2 * eps);
  return THREE.MathUtils.radToDeg(Math.atan(Math.hypot(hx, hz)));
}

function sampleMasks(x, z) {
  if (!world) return { river: 0, lake: 0, curv: 0.5, moisture: 0.5 };
  if (atlas) {
    const m = sampleAtlasMasks(atlas, x, z) || [0, 0, 128, 128];
    return { river: m[0] / 255, lake: m[1] / 255, curv: m[2] / 255, moisture: m[3] / 255 };
  }
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
  let t;
  let p;
  if (atlas) {
    const c = sampleAtlasClimate(atlas, x, z);
    if (!c) return null;
    [t, p] = c;
  } else {
    const climateSize = world.meta.climateSize || size;
    const { u, v } = worldToGrid(x, z);
    // Climate may be packed at a coarser grid (e.g. 512 on 2K/4K worlds)
    const cu = THREE.MathUtils.clamp((u / Math.max(size - 1, 1)) * (climateSize - 1), 0, climateSize - 1.001);
    const cv = THREE.MathUtils.clamp((v / Math.max(size - 1, 1)) * (climateSize - 1), 0, climateSize - 1.001);
    const idx = (Math.round(cv) * climateSize + Math.round(cu)) * 4;
    t = world.climate[idx];
    p = world.climate[idx + 1];
  }
  const tempC = tempRange[0] + (t / 255) * (tempRange[1] - tempRange[0]);
  const precipMm = precipRange[0] + (p / 255) * (precipRange[1] - precipRange[0]);
  return { tempC, precipMm };
}

// ---------- Dispose Helper ----------
function disposeWorldScene() {
  if (tileStream) tileStream.stop();
  atlas = null;
  atlasDirty = false;
  clearTileDecor();
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

  subtitleBase = `${meta.subtitle} · ${lakes.length} lakes, ${splines.length} rivers`;
  document.getElementById('world-subtitle').textContent = subtitleBase;
  const sel = document.getElementById('world-select');
  if (sel) sel.value = worldId;
  syncWorldCacheActions();

  const { size, worldSpan, maxElev } = meta;
  const climateSize = meta.climateSize || size;
  const lakeGridSize = meta.lakeGridSize || size;

  // Float32 height texture at full 1K/2K/4K resolution (clipmap samples this for native detail!)
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

  // Climate texture (may be compact 512x512 even on 2K/4K worlds)
  if (climate) {
    climateTex = new THREE.DataTexture(climate, climateSize, climateSize, THREE.RGBAFormat, THREE.UnsignedByteType);
  } else {
    climateTex = new THREE.DataTexture(new Uint8Array([128, 128, 128, 255]), 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  }
  climateTex.flipY = true;
  climateTex.minFilter = THREE.LinearFilter;
  climateTex.magFilter = THREE.LinearFilter;
  climateTex.needsUpdate = true;

  // 1. GPU Geometry Clipmap — native 1.17–4.69 m triangles around the walker, coarse on the horizon
  const geo = buildTerrainClipmapGeometry(worldSpan, size, size >= 2048 ? 160 : 128);
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
  terrainMat.uniforms.uFocusPos.value.set(0, 0, 0);
  terrainMesh = new THREE.Mesh(geo, terrainMat);
  // Clipmap displacement is done in the vertex shader via uVScale — do NOT scale the mesh Y
  terrainMesh.scale.set(1, 1, 1);
  terrainMesh.frustumCulled = false;
  scene.add(terrainMesh);
  syncFogDensity();

  // 2. Ocean Water Plane
  const waterGeo = new THREE.PlaneGeometry(worldSpan * 1.6, worldSpan * 1.6, 128, 128);
  waterGeo.rotateX(-Math.PI / 2);
  waterMat = createWaterMaterial(heightFloatTex, maskTex, worldSpan, size);
  waterMat.uniforms.uVScale.value = state.vScale;
  waterMesh = new THREE.Mesh(waterGeo, waterMat);
  waterMesh.position.y = 0.0;
  scene.add(waterMesh);

  // 3. Real 3D Flowing River Ribbons & Flat-Surfaced Alpine/Valley Lakes
  // Sample full-resolution heightfield (not a downsampled mesh) so ribbons hug 2K/4K riverbeds
  const sampleRenderedMeshHeight = (wx, wz) => sampleHeight(wx, wz) / Math.max(state.vScale, 1e-6);
  riverObj = buildRiverSystem(splines, sampleRenderedMeshHeight);
  riverObj.mesh.scale.y = state.vScale;
  scene.add(riverObj.mesh);

  if (lakes.length > 0) {
    lakeObj = buildLakeSystem(lakes, lakeGridSize, worldSpan, heightFloatTex, maskTex);
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

  // Planet-synthesized worlds grow: stream the tiles around the player from the Stage 2 server.
  // (Demo worlds and drill-down children are not part of a planet chart; start() ignores them.)
  if (!tileStream) {
    tileStream = createTileStreamController({
      onChange: () => {
        atlasDirty = true;
      },
      getPlayerPos: () => ({ x: player.pos.x, z: player.pos.z }),
    });
  }
  if (climate) {
    const sum = [0, 0, 0];
    const n = climate.length / 4;
    for (let i = 0; i < climate.length; i += 4) {
      sum[0] += climate[i];
      sum[1] += climate[i + 1];
      sum[2] += climate[i + 2];
    }
    homeMeanClimate = sum.map((v) => Math.round(v / n));
  } else {
    homeMeanClimate = [128, 128, 128];
  }
  tileStream.start(meta, { heights, masks, climate, rivers: splines, lakes });

  loader.classList.add('hidden');
}

// ---------- Teleport & Modes ----------
function teleportTo(landmarkKey) {
  if (!world) return;
  const pt = world.meta.landmarks[landmarkKey] || [0.5, 0.5];
  const rawX = (pt[0] - 0.5) * world.meta.worldSpan;
  const rawZ = (pt[1] - 0.5) * world.meta.worldSpan;
  const { x, z } = clampToWalkable(rawX, rawZ);
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
  for (const m of farMats()) m.uniforms.uFogDensity.value = fogVal;
  if (waterMat) waterMat.uniforms.uFogDensity.value = fogVal;
  for (const r of allRivers()) r.material.uniforms.uFogDensity.value = fogVal;
  for (const l of allLakes()) l.material.uniforms.uFogDensity.value = fogVal;
}

function setCameraMode(mode) {
  // Leaving the isometric view: keep looking the way it was turned, so the compass does not jump
  if (state.camMode === 'diorama' && mode !== 'diorama') {
    camRig.yaw = camRig.dioramaYaw;
    player.yaw = camRig.dioramaYaw + Math.PI;
  }
  state.camMode = mode;
  document.querySelectorAll('[data-cam]').forEach((b) => b.classList.toggle('active', b.dataset.cam === mode));

  // Diorama pin can sit on the visual rim; snap inward before walking so the AABB clamp isn't an "invisible wall"
  if (mode !== 'diorama' && world) {
    const { x, z } = clampToWalkable(player.pos.x, player.pos.z);
    player.pos.x = x;
    player.pos.z = z;
    player.pos.y = Math.max(sampleHeight(x, z), 0.2);
  }

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
  updatePropsTransformAll();
}

function setStyleMode(styleIdx) {
  state.styleMode = styleIdx;
  if (terrainMat) terrainMat.uniforms.uStyleMode.value = styleIdx;
  for (const m of farMats()) m.uniforms.uStyleMode.value = styleIdx;
  if (waterMat) waterMat.uniforms.uStyleMode.value = styleIdx;
  for (const r of allRivers()) {
    r.material.uniforms.uStyleMode.value = styleIdx;
    r.mesh.visible = styleIdx !== 2;
  }
  for (const l of allLakes()) {
    l.material.uniforms.uStyleMode.value = styleIdx;
    l.mesh.visible = styleIdx !== 2;
  }
  if (grassObj) grassObj.mesh.visible = styleIdx === 0 && state.camMode !== 'diorama';
  for (const g of allProps()) g.visible = styleIdx !== 2;
  document.querySelectorAll('[data-style]').forEach((b) => b.classList.toggle('active', Number(b.dataset.style) === styleIdx));
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

  for (const m of terrainMat ? [terrainMat, ...farMats()] : []) {
    m.uniforms.uSunDir.value.copy(dir);
    m.uniforms.uFogColor.value.copy(fogCol);
    m.uniforms.uMistColor.value.copy(mistCol);
  }
  if (waterMat) {
    waterMat.uniforms.uSunDir.value.copy(dir);
    waterMat.uniforms.uFogColor.value.copy(fogCol);
  }
  for (const o of [...allRivers(), ...allLakes()]) {
    o.material.uniforms.uSunDir.value.copy(dir);
    o.material.uniforms.uFogColor.value.copy(fogCol);
  }
  if (grassObj) {
    grassObj.material.uniforms.uSunDir.value.copy(dir);
    grassObj.material.uniforms.uFogColor.value.copy(fogCol);
    grassObj.material.uniforms.uMistColor.value.copy(mistCol);
  }
}

// ---------- UI Wiring ----------
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
  // Clipmap height is displaced in-shader; mesh Y scale stays 1
  if (terrainMat) terrainMat.uniforms.uVScale.value = state.vScale;
  for (const m of farMats()) m.uniforms.uVScale.value = state.vScale;
  if (waterMat) waterMat.uniforms.uVScale.value = state.vScale;
  for (const r of allRivers()) r.mesh.scale.y = state.vScale;
  for (const l of allLakes()) {
    l.mesh.scale.y = state.vScale;
    l.material.uniforms.uVScale.value = state.vScale;
  }
  if (grassObj) grassObj.material.uniforms.uVScale.value = state.vScale;
  if (world) player.pos.y = Math.max(sampleHeight(player.pos.x, player.pos.z), 0.2);
  updatePropsTransformAll();
  rebuildDrillBoxOverlay();
  rebuildDrillChildrenOverlay();
});
document.getElementById('world-select').addEventListener('change', (e) => loadWorld(e.target.value));

function syncWorldCacheActions() {
  const wrap = document.getElementById('world-cache-actions');
  const bmBtn = document.getElementById('btn-bookmark-world');
  if (!wrap || !world) return;
  const isSynth = Boolean(world.meta?.planetOrigin || world.meta?.drillOrigin);
  wrap.style.display = isSynth ? 'flex' : 'none';
  if (bmBtn) {
    bmBtn.textContent = world.meta?.bookmarked ? '★ Bookmarked' : '☆ Bookmark';
  }
}

document.getElementById('btn-bookmark-world')?.addEventListener('click', async () => {
  if (!world?.meta?.id) return;
  try {
    const next = !world.meta.bookmarked;
    const resp = await fetch(`/api/worlds/${encodeURIComponent(world.meta.id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bookmarked: next }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || 'Bookmark failed');
    world.meta.bookmarked = !!data.world?.bookmarked;
    const catEntry = catalog.find((w) => w.id === world.meta.id);
    if (catEntry) catEntry.bookmarked = world.meta.bookmarked;
    syncWorldCacheActions();
  } catch (err) {
    window.alert(err.message || 'Bookmark failed (is Stage 2 server running?)');
  }
});

document.getElementById('btn-delete-world')?.addEventListener('click', async () => {
  if (!world?.meta?.id) return;
  if (!world.meta.planetOrigin && !world.meta.drillOrigin) {
    window.alert('Only synthesized / drilled worlds can be deleted from here.');
    return;
  }
  if (!window.confirm(`Delete "${world.meta.title || world.meta.id}" from disk?`)) return;
  try {
    const resp = await fetch(`/api/worlds/${encodeURIComponent(world.meta.id)}?cascade=1`, { method: 'DELETE' });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || 'Delete failed');
    catalog = catalog.filter((w) => !data.deletedIds.includes(w.id));
    const select = document.getElementById('world-select');
    if (select) {
      select.innerHTML = catalog.map((w) => `<option value="${w.id}">${w.title}</option>`).join('');
    }
    const nextId = catalog[0]?.id;
    if (nextId) await loadWorld(nextId);
    else window.alert('World deleted. No remaining worlds in catalog.');
  } catch (err) {
    window.alert(err.message || 'Delete failed (is Stage 2 server running?)');
  }
});
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
const elCompassRose = document.getElementById('compass-rose');
const elCompassHeading = document.getElementById('compass-heading');

function updateMovement(dt) {
  if (!world) return;
  const bounds = worldBounds();

  const forwardInput = (keys.has('w') || keys.has('arrowup') ? 1 : 0) - (keys.has('s') || keys.has('arrowdown') ? 1 : 0);
  const rightInput = (keys.has('d') || keys.has('arrowright') ? 1 : 0) - (keys.has('a') || keys.has('arrowleft') ? 1 : 0);
  const sprint = keys.has('shift');

  if (state.camMode === 'diorama') {
    const panSpeed = (sprint ? 260 : 140) * (camRig.dioramaDist / 700);
    const fwd = new THREE.Vector3(-Math.sin(camRig.dioramaYaw), 0, -Math.cos(camRig.dioramaYaw));
    const right = new THREE.Vector3(Math.cos(camRig.dioramaYaw), 0, -Math.sin(camRig.dioramaYaw));
    camRig.dioramaCenter.addScaledVector(fwd, forwardInput * panSpeed * dt);
    camRig.dioramaCenter.addScaledVector(right, rightInput * panSpeed * dt);
    camRig.dioramaCenter.x = THREE.MathUtils.clamp(camRig.dioramaCenter.x, bounds.minX, bounds.maxX);
    camRig.dioramaCenter.z = THREE.MathUtils.clamp(camRig.dioramaCenter.z, bounds.minZ, bounds.maxZ);
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
  let hitMapEdge = false;
  if (forwardInput !== 0 || rightInput !== 0) {
    const camFwd = new THREE.Vector3(-Math.sin(camRig.yaw), 0, -Math.cos(camRig.yaw));
    const camRight = new THREE.Vector3(Math.cos(camRig.yaw), 0, -Math.sin(camRig.yaw));
    move.addScaledVector(camFwd, forwardInput).addScaledVector(camRight, rightInput).normalize();
    const nextX = player.pos.x + move.x * speed * dt;
    const nextZ = player.pos.z + move.z * speed * dt;
    const step = resolveWalk(player.pos.x, player.pos.z, nextX, nextZ);
    hitMapEdge = step.blocked;
    player.pos.x = step.x;
    player.pos.z = step.z;
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
  else if (hitMapEdge && moving) player.activity = atlas ? 'TERRAIN LOADING' : 'MAP EDGE';
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

  // Compass: the rose turns so its N points at chart north (-Z) as seen from the camera
  const yaw = state.camMode === 'diorama' ? camRig.dioramaYaw : camRig.yaw;
  const heading = (((-THREE.MathUtils.radToDeg(yaw)) % 360) + 360) % 360;
  if (elCompassRose) elCompassRose.style.transform = `rotate(${(-heading).toFixed(1)}deg)`;
  if (elCompassHeading) {
    const names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const label = `${names[Math.round(heading / 45) % 8]} ${String(Math.round(heading) % 360).padStart(3, '0')}°`;
    if (elCompassHeading.textContent !== label) elCompassHeading.textContent = label;
  }

  // Tile-stream progress rides along on the subtitle line
  const stream = tileStream?.describe() || '';
  const sub = stream ? `${subtitleBase} · ${stream}` : subtitleBase;
  const elSub = document.getElementById('world-subtitle');
  if (elSub && elSub.textContent !== sub) elSub.textContent = sub;
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.elapsedTime;

  // Tiles landed or the player crossed into a new tile: re-stitch before this frame draws
  if (atlasDirty) rebuildChart();
  updateMovement(dt);

  // Follow player with directional shadow frustum & 48K GPU grass ring
  const focus = state.camMode === 'diorama' ? camRig.dioramaCenter : player.pos;
  if (sunLight.userData.dir) {
    sunLight.target.position.copy(focus);
    sunLight.position.copy(focus).addScaledVector(sunLight.userData.dir, 360);
  }

  if (terrainMat) {
    terrainMat.uniforms.uTime.value = t;
    terrainMat.uniforms.uFocusPos.value.copy(focus);
  }
  for (const m of farMats()) m.uniforms.uTime.value = t;
  if (waterMat) waterMat.uniforms.uTime.value = t;
  for (const r of allRivers()) r.material.uniforms.uTime.value = t;
  for (const l of allLakes()) l.material.uniforms.uTime.value = t;
  if (grassObj) {
    grassObj.material.uniforms.uTime.value = t;
    grassObj.material.uniforms.uFocusPos.value.copy(focus);
    grassObj.material.uniforms.uPlayerPos.value.copy(player.pos);
  }

  // The sky is a finite dome: keep it centred on the camera so walking (or flying the
  // isometric view) several tiles from home never leaves it and shows black space instead.
  skyDome.position.copy(camera.position);

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
  // Streamed tiles and ghosts belong to their home world; they are not worlds to open on their own
  catalog = [...baseCatalog, ...synthCatalog.filter((w) => !seenIds.has(w.id) && !w.tileOrigin)];

  const select = document.getElementById('world-select');
  select.innerHTML = catalog.map((w) => `<option value="${w.id}">${w.title}</option>`).join('');

  const reqWorld = new URLSearchParams(window.location.search).get('world');
  const startId = reqWorld && catalog.some((w) => w.id === reqWorld) ? reqWorld : catalog[0].id;
  select.value = startId;

  await loadWorld(startId);
  setCameraMode('third');
  setStyleMode(0);
  state.sunElevDeg = 26;
  document.getElementById('sun-slider').value = '26';
  updateSun();
  animate();
}

window.__terragrok = {
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
