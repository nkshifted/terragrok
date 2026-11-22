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
const camera = new THREE.PerspectiveCamera(56, window.innerWidth / window.innerHeight, 0.4, 10000);

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
  dioramaYaw: Math.PI * 0.25,
};

// ---------- Input Handling ----------
const keys = new Set();
let isDragging = false;
let lastMouse = { x: 0, y: 0 };

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  keys.add(k);
  if (k === '1') setCameraMode('third');
  if (k === '2') setCameraMode('first');
  if (k === '3') setCameraMode('diorama');
  if (k === 'v') setStyleMode((state.styleMode + 1) % 3);
  if (k === 'h') document.body.classList.toggle('hud-hidden');
  if (e.code === 'Space') e.preventDefault();
});
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

canvas.addEventListener('pointerdown', (e) => {
  isDragging = true;
  lastMouse = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', (e) => {
  isDragging = false;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
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
      camRig.dioramaDist = THREE.MathUtils.clamp(camRig.dioramaDist * (1 + e.deltaY * 0.001), 160, 5200);
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
  if (meshVerts === size) {
    for (let i = 0; i < heights.length; i++) {
      posAttr.setY(i, heights[i]);
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
        const a = heights[y0 * size + x0] * (1 - tx) + heights[y0 * size + x1] * tx;
        const b = heights[y1 * size + x0] * (1 - tx) + heights[y1 * size + x1] * tx;
        posAttr.setY(vy * meshVerts + vx, a * (1 - ty) + b * ty);
      }
    }
  }
  geo.computeVertexNormals();

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
  riverObj = buildRiverSystem(splines);
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
  camRig.dioramaCenter.set(x, y, z);

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
    camRig.dioramaCenter.copy(player.pos);
    explorer.visible = true;
    explorer.scale.setScalar(2.4);
  } else if (mode === 'first') {
    camera.fov = 66;
    camRig.pitch = -0.05;
    explorer.visible = false;
  } else {
    camera.fov = 56;
    camRig.pitch = 0.09;
    explorer.visible = true;
    explorer.scale.setScalar(1.0);
  }
  camera.updateProjectionMatrix();
  syncFogDensity();
  if (grassObj) grassObj.mesh.visible = state.styleMode === 0 && mode !== 'diorama';
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
});
document.getElementById('world-select').addEventListener('change', (e) => loadWorld(e.target.value));

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
    camRig.dioramaCenter.y = Math.max(sampleHeight(camRig.dioramaCenter.x, camRig.dioramaCenter.z), 0);

    const pitch = THREE.MathUtils.degToRad(40);
    const hDist = Math.cos(pitch) * camRig.dioramaDist;
    const vDist = Math.sin(pitch) * camRig.dioramaDist;
    camera.position.set(
      camRig.dioramaCenter.x + Math.sin(camRig.dioramaYaw) * hDist,
      camRig.dioramaCenter.y + vDist,
      camRig.dioramaCenter.z + Math.cos(camRig.dioramaYaw) * hDist
    );
    camera.lookAt(camRig.dioramaCenter);
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
  const pt = state.camMode === 'diorama' ? camRig.dioramaCenter : player.pos;
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
  catalog = await fetch('/worlds/catalog.json').then((r) => r.json());
  const select = document.getElementById('world-select');
  select.innerHTML = catalog.map((w) => `<option value="${w.id}">${w.title}</option>`).join('');
  await loadWorld(catalog[0].id);
  applyPreset('zelda');
  animate();
}

window.__terragrok = {
  applyPreset,
  loadWorld,
  teleportTo,
  setCameraMode,
  setStyleMode,
  focusDiorama(x, z, dist = 340) {
    setCameraMode('diorama');
    camRig.dioramaCenter.set(x, sampleHeight(x, z), z);
    camRig.dioramaDist = dist;
  },
};

init();
