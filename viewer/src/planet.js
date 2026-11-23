import * as THREE from 'three';
import {
  CUBE_FACES as TEC_FACES,
  generateTectonicPlanet,
  faceUVToUnitDir as tecFaceUVToUnitDir,
  unitDirToFaceUV as tecUnitDirToFaceUV,
} from './tectonics.js';

const CUBE_FACES = TEC_FACES.map((f) => ({
  name: f.name,
  n: new THREE.Vector3(...f.n),
  u: new THREE.Vector3(...f.u),
  v: new THREE.Vector3(...f.v),
}));

// ---------- Renderer & Scene ----------
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.10;

const scene = new THREE.Scene();
scene.background = new THREE.Color('#04070f');

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1.0, 50000);

const sunDir = new THREE.Vector3(0.75, 0.30, 0.59).normalize();
const sunLight = new THREE.DirectionalLight('#fff5e4', 2.0);
sunLight.position.copy(sunDir).multiplyScalar(15000);
scene.add(sunLight);
scene.add(new THREE.HemisphereLight('#88b8ff', '#1b2638', 0.48));

// ---------- Starfield ----------
function buildStarfield() {
  const count = 2400;
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  let s = 1337;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < count; i++) {
    const theta = rnd() * Math.PI * 2;
    const u = rnd() * 2 - 1;
    const r = Math.sqrt(1 - u * u);
    const dist = 22000 + rnd() * 6000;
    pos[i * 3] = r * Math.cos(theta) * dist;
    pos[i * 3 + 1] = u * dist;
    pos[i * 3 + 2] = r * Math.sin(theta) * dist;
    const tint = 0.72 + rnd() * 0.28;
    col[i * 3] = tint * (0.9 + rnd() * 0.1);
    col[i * 3 + 1] = tint * (0.92 + rnd() * 0.08);
    col[i * 3 + 2] = tint;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mat = new THREE.PointsMaterial({ size: 22, vertexColors: true, sizeAttenuation: true, fog: false });
  return new THREE.Points(geo, mat);
}
scene.add(buildStarfield());

// ---------- State ----------
let currentPlanet = null; // { meta, faces }
let planetGroup = null;
let poleGroup = null;
let faceMeshes = [];
let oceanMesh = null;
let atmoMesh = null;
let explorerGroup = null;
let homeBeacon = null;
let targetPatchGroup = null;

// Active planet generation settings (also supports ?seed=&archetype= restore from index.html!)
const urlParams = new URLSearchParams(window.location.search);
const initialSeed = Number(urlParams.get('seed')) || 523;
const initialArch = urlParams.get('archetype') || 'earth';
const genState = {
  seed: initialSeed, // Starts on a rich multi-continent Earth-like tectonic seed!
  planetNumber: 1,
  archetype: ['auto', 'earth', 'rifting', 'supercontinent', 'archipelago'].includes(initialArch)
    ? initialArch
    : 'earth',
  seaLevelBias: 0.0,
  reliefScale: 1.0,
  showTectonics: false,
  magNorth: new THREE.Vector3(0.16, 0.98, 0.12).normalize(),
};

const stage2 = {
  targetDir: new THREE.Vector3(0, 0, 1),
  spanDeg: 46,
  serverReady: false,
  serverDevice: 'mps',
  synthesizing: false,
  lastResult: null,
};

const nav = {
  mode: 'orbit', // 'orbit' | 'surface'
  orbitYaw: 0.35,
  orbitPitch: 0.24,
  orbitDist: 8600,
  surfDir: new THREE.Vector3(0.12, 0.25, 0.96).normalize(),
  surfForward: new THREE.Vector3(0, 1, 0),
  surfPitch: 0.20,
  surfCamDist: 38,
  homeDir: null,
  speedMult: 2,
};

// ---------- Input ----------
const keys = new Set();
let pointerDown = false;
let pointerLastX = 0;
let pointerLastY = 0;
let pointerDownX = 0;
let pointerDownY = 0;
let pointerMoved = 0;

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  keys.add(k);
  if (k === 'o') setMode(nav.mode === 'orbit' ? 'surface' : 'orbit');
  if (k === 'r') generateNewPlanet();
  if (k === 'h') {
    const hud = document.getElementById('hud-left');
    hud.style.display = hud.style.display === 'none' ? '' : 'none';
  }
});
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

canvas.addEventListener('pointerdown', (e) => {
  pointerDown = true;
  pointerLastX = e.clientX;
  pointerLastY = e.clientY;
  pointerDownX = e.clientX;
  pointerDownY = e.clientY;
  pointerMoved = 0;
});
window.addEventListener('pointerup', (e) => {
  const wasClick = pointerDown && pointerMoved < 5 && e.target === canvas;
  pointerDown = false;
  if (wasClick && nav.mode === 'orbit' && currentPlanet && faceMeshes.length > 0) {
    const ndc = new THREE.Vector2(
      (e.clientX / window.innerWidth) * 2 - 1,
      -(e.clientY / window.innerHeight) * 2 + 1
    );
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(faceMeshes, false);
    if (hits.length > 0) {
      setStage2Target(hits[0].point.clone().normalize());
    }
  }
});
window.addEventListener('pointermove', (e) => {
  if (!pointerDown) return;
  const dx = e.clientX - pointerLastX;
  const dy = e.clientY - pointerLastY;
  pointerMoved += Math.hypot(dx, dy);
  pointerLastX = e.clientX;
  pointerLastY = e.clientY;

  if (nav.mode === 'orbit') {
    nav.orbitYaw -= dx * 0.0055;
    nav.orbitPitch = THREE.MathUtils.clamp(nav.orbitPitch + dy * 0.0055, -1.48, 1.48);
  } else {
    const qYaw = new THREE.Quaternion().setFromAxisAngle(nav.surfDir, -dx * 0.0045);
    nav.surfForward.applyQuaternion(qYaw).normalize();
    nav.surfPitch = THREE.MathUtils.clamp(nav.surfPitch + dy * 0.004, -0.35, 0.85);
  }
});

canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    if (nav.mode === 'orbit') {
      nav.orbitDist = THREE.MathUtils.clamp(nav.orbitDist * (1 + e.deltaY * 0.0009), 3600, 17500);
    } else {
      nav.surfCamDist = THREE.MathUtils.clamp(nav.surfCamDist * (1 + e.deltaY * 0.0008), 16, 220);
    }
  },
  { passive: false }
);

// Double-click any continent, island arc, or isthmus in Orbit mode to land right there!
const raycaster = new THREE.Raycaster();
canvas.addEventListener('dblclick', (e) => {
  if (!currentPlanet || faceMeshes.length === 0) return;
  const ndc = new THREE.Vector2(
    (e.clientX / window.innerWidth) * 2 - 1,
    -(e.clientY / window.innerHeight) * 2 + 1
  );
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(faceMeshes, false);
  if (hits.length > 0) {
    const hitDir = hits[0].point.clone().normalize();
    landAtDirection(hitDir);
  }
});

// ---------- Cube-Sphere Coordinate & Sampling Math ----------
function faceUVToUnitDir(faceIdx, u, v, target = new THREE.Vector3()) {
  const out = tecFaceUVToUnitDir(faceIdx, u, v);
  return target.set(out[0], out[1], out[2]);
}

function unitDirToFaceUV(dir) {
  return tecUnitDirToFaceUV(dir.x, dir.y, dir.z);
}

/** Bilinearly sample elevation (m), climate (tempC, moisture), river & lake masks anywhere on S^2. */
function sampleSphereSurface(unitDir) {
  if (!currentPlanet) return { elev: 0, tempC: 15, moisture: 0.5, river: 0, lake: 0, bound: 0 };
  const N = currentPlanet.meta.faceSize;
  const { faceIdx, u, v } = unitDirToFaceUV(unitDir);
  const f = currentPlanet.faces[faceIdx];
  const gx = (u * 0.5 + 0.5) * (N - 1);
  const gy = (v * 0.5 + 0.5) * (N - 1);
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const x1 = Math.min(x0 + 1, N - 1);
  const y1 = Math.min(y0 + 1, N - 1);
  const tx = gx - x0;
  const ty = gy - y0;

  const i00 = y0 * N + x0;
  const i10 = y0 * N + x1;
  const i01 = y1 * N + x0;
  const i11 = y1 * N + x1;

  const h00 = f.heights[i00];
  const h10 = f.heights[i10];
  const h01 = f.heights[i01];
  const h11 = f.heights[i11];
  const hRaw = (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty;
  const elev = hRaw > 0 ? hRaw * genState.reliefScale : hRaw;

  const c00 = i00 * 4;
  const tempNorm = f.climate[c00] / 255.0;
  const moisture = f.climate[c00 + 1] / 255.0;
  const tempC = tempNorm * 60.0 - 20.0;
  const river = f.masks[c00] / 255.0;
  const lake = f.masks[c00 + 1] / 255.0;
  const bound = (f.masks[c00 + 2] / 255.0) * 2.0 - 1.0;

  return { elev, tempC, moisture, river, lake, bound };
}

// ---------- Seamless 3D Planet Surface Shader (with Tectonic Overlay + Continental Shelves) ----------
function createPlanetFaceMaterial(maskTex, climTex, maxElev) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uMasks: { value: maskTex },
      uClimate: { value: climTex },
      uSunDir: { value: sunDir.clone() },
      uMaxElev: { value: maxElev },
      uShowTectonics: { value: genState.showTectonics ? 1.0 : 0.0 },
    },
    vertexShader: /* glsl */ `
      attribute float aElev;
      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vUnitPos;
      varying vec3 vWorldPos;
      varying float vElev;

      void main() {
        vUv = uv;
        vElev = aElev;
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vUnitPos = normalize(position);
        vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uMasks;
      uniform sampler2D uClimate;
      uniform vec3 uSunDir;
      uniform float uMaxElev;
      uniform float uShowTectonics;

      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vUnitPos;
      varying vec3 vWorldPos;
      varying float vElev;

      // Seamless 3D value noise directly on unit sphere coordinates!
      float hash3(vec3 p) {
        p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
      }
      float noise3(vec3 x) {
        vec3 i = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(mix(hash3(i + vec3(0,0,0)), hash3(i + vec3(1,0,0)), f.x),
              mix(hash3(i + vec3(0,1,0)), hash3(i + vec3(1,1,0)), f.x), f.y),
          mix(mix(hash3(i + vec3(0,0,1)), hash3(i + vec3(1,0,1)), f.x),
              mix(hash3(i + vec3(0,1,1)), hash3(i + vec3(1,1,1)), f.x), f.y),
          f.z
        );
      }
      float fbm3(vec3 p) {
        float v = 0.0;
        float a = 0.5;
        for (int i = 0; i < 4; i++) {
          v += a * noise3(p);
          p *= 2.07;
          a *= 0.5;
        }
        return v;
      }

      void main() {
        vec3 N = normalize(vNormalW);
        vec3 U = normalize(vUnitPos);
        float slope = clamp(1.0 - max(dot(N, U), 0.0), 0.0, 1.0);
        float hNorm = clamp(vElev / max(uMaxElev, 1.0), 0.0, 1.0);

        vec4 m = texture2D(uMasks, vUv);
        float riverWet = m.r;
        float lake = m.g;
        float tectBound = m.b * 2.0 - 1.0; // >0 convergent mountain/subduction, <0 divergent rift

        vec4 c = texture2D(uClimate, vUv);
        float tempC = c.r * 60.0 - 20.0;
        float moist = c.g;
        float orogeny = c.b;

        float nMacro = fbm3(U * 34.0);
        float nFine  = fbm3(U * 155.0);

        // Bathymetry: turquoise shallow continental shelf -> dark abyssal plain -> subduction trench
        vec3 abyssCol  = vec3(0.03, 0.11, 0.24);
        vec3 shelfCol  = vec3(0.09, 0.36, 0.46);
        vec3 sandCol   = vec3(0.78, 0.72, 0.54);
        vec3 desertCol = vec3(0.75, 0.64, 0.42);
        vec3 steppeCol = vec3(0.54, 0.54, 0.31);
        vec3 grassCol  = vec3(0.28, 0.48, 0.20);
        vec3 forestCol = vec3(0.15, 0.35, 0.14);
        vec3 tropicCol = vec3(0.09, 0.29, 0.12);
        vec3 borealCol = vec3(0.18, 0.27, 0.17);
        vec3 tundraCol = vec3(0.42, 0.43, 0.34);
        vec3 rockWarm  = vec3(0.46, 0.40, 0.34);
        vec3 rockCold  = vec3(0.40, 0.42, 0.46);
        vec3 snowCol   = vec3(0.92, 0.95, 0.99);

        float effM = clamp(moist + (nMacro - 0.5) * 0.16, 0.0, 1.0);
        vec3 aridBiome = mix(desertCol, steppeCol, smoothstep(0.16, 0.36, effM));
        vec3 warmBiome = mix(aridBiome, grassCol, smoothstep(0.26, 0.50, effM));
        vec3 wetBiome  = mix(forestCol, tropicCol, smoothstep(16.0, 25.0, tempC) * smoothstep(0.48, 0.84, effM));
        vec3 landCol   = mix(warmBiome, wetBiome, smoothstep(0.32, 0.66, effM));

        // High-latitude boreal & tundra transition
        landCol = mix(landCol, borealCol, (1.0 - smoothstep(2.0, 10.0, tempC)) * 0.68);
        landCol = mix(landCol, tundraCol, (1.0 - smoothstep(-1.5, 4.0, tempC)) * 0.76);
        landCol *= (0.86 + 0.28 * (nMacro * 0.6 + nFine * 0.4));

        // Bathymetry shelf -> narrow sandy coast (0..3m only!) -> lush inland biome
        vec3 seabed = mix(abyssCol, shelfCol, smoothstep(-95.0, -2.0, vElev));
        vec3 albedo = mix(seabed, sandCol, smoothstep(-0.3, 0.4, vElev));
        albedo = mix(albedo, landCol, smoothstep(1.2, 4.8, vElev));

        // Orogenic mountain cordillera rock + cliffs
        vec3 rockCol = mix(rockWarm, rockCold, 1.0 - smoothstep(0.0, 16.0, tempC));
        rockCol *= (0.78 + 0.38 * nFine);
        float rockMask = clamp(
          smoothstep(0.055, 0.19, slope) +
          smoothstep(0.56, 0.85, hNorm) * 0.65 +
          orogeny * smoothstep(0.30, 0.65, hNorm) * 0.35,
          0.0, 1.0
        );
        albedo = mix(albedo, rockCol, rockMask * smoothstep(2.0, 12.0, vElev));

        // Freezing snowline & polar glaciers (only where land/peaks are actually below freezing!)
        float freeze = 0.6 + (nMacro - 0.5) * 2.4;
        float snowMask = (1.0 - smoothstep(freeze - 2.6, freeze, tempC)) * smoothstep(0.5, 4.0, vElev);
        snowMask *= (1.0 - smoothstep(0.24, 0.48, slope));
        albedo = mix(albedo, snowCol, snowMask);

        // Seamless spherical rivers & alpine lakes
        if (riverWet > 0.18 && vElev > 0.5) {
          vec3 rivCol = mix(vec3(0.16, 0.46, 0.66), vec3(0.08, 0.32, 0.54), nFine);
          albedo = mix(albedo, rivCol, smoothstep(0.18, 0.62, riverWet) * 0.78);
        }
        if (lake > 0.42 && vElev > 0.5) {
          vec3 lkCol = mix(vec3(0.10, 0.44, 0.60), vec3(0.05, 0.26, 0.46), nMacro);
          albedo = mix(albedo, lkCol, smoothstep(0.42, 0.76, lake) * 0.88);
        }

        // Optional Tectonic Plate Boundary Overlay (Red-Orange = Subduction/Collision, Cyan = Divergent Rift!)
        if (uShowTectonics > 0.5) {
          float convGlow = smoothstep(0.16, 0.75, tectBound);
          float divGlow  = smoothstep(0.16, 0.75, -tectBound);
          albedo = mix(albedo, vec3(1.0, 0.34, 0.12), convGlow * 0.78);
          albedo = mix(albedo, vec3(0.18, 0.90, 1.0), divGlow * 0.78);
        }

        // Planetary lighting + subtle atmospheric limb scattering
        float ndl = dot(N, uSunDir);
        float sunDiff = smoothstep(-0.14, 0.18, ndl) * max(ndl, 0.0);
        float wrap = max(dot(U, uSunDir) * 0.5 + 0.5, 0.0);
        vec3 ambient = mix(vec3(0.07, 0.10, 0.18), vec3(0.36, 0.46, 0.58), wrap);
        vec3 color = albedo * (ambient + vec3(1.22, 1.15, 1.05) * sunDiff);

        vec3 V = normalize(cameraPosition - vWorldPos);
        float rim = pow(1.0 - max(dot(U, V), 0.0), 3.0);
        color = mix(color, vec3(0.45, 0.72, 0.98), rim * 0.34 * wrap);

        gl_FragColor = vec4(color, 1.0);
      }
    `,
  });
}

function createSphericalOcean(radius) {
  const geo = new THREE.SphereGeometry(radius, 96, 64);
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: {
      uSunDir: { value: sunDir.clone() },
      uTime: { value: 0.0 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vNormalW;
      varying vec3 vWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform float uTime;
      varying vec3 vNormalW;
      varying vec3 vWorldPos;
      void main() {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);
        float ndl = max(dot(N, uSunDir), 0.0);
        float wrap = dot(N, uSunDir) * 0.5 + 0.5;
        vec3 deepOcean = vec3(0.03, 0.14, 0.30);
        vec3 shallowTint = vec3(0.08, 0.32, 0.50);
        vec3 col = mix(deepOcean, shallowTint, wrap * 0.42) * (0.36 + 0.94 * ndl);

        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.2);
        col = mix(col, vec3(0.40, 0.68, 0.95), fresnel * 0.55 * wrap);
        vec3 H = normalize(uSunDir + V);
        float spec = pow(max(dot(N, H), 0.0), 80.0) * smoothstep(0.0, 0.15, ndl);
        col += vec3(1.0, 0.96, 0.88) * spec * 0.62;

        // Deep oceanic color with subtle translucent shallow shelf glow
        gl_FragColor = vec4(col, 0.85);
      }
    `,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 1;
  return mesh;
}

function createAtmosphereHalo(radius) {
  const geo = new THREE.SphereGeometry(radius * 1.072, 64, 48);
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    uniforms: { uSunDir: { value: sunDir.clone() } },
    vertexShader: /* glsl */ `
      varying vec3 vNormalW;
      varying vec3 vWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      varying vec3 vNormalW;
      varying vec3 vWorldPos;
      void main() {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);
        float rim = pow(clamp(0.65 - dot(N, V), 0.0, 1.0), 2.4);
        float daySide = smoothstep(-0.35, 0.45, dot(N, uSunDir));
        vec3 col = mix(vec3(0.20, 0.48, 0.92), vec3(0.55, 0.82, 1.0), rim);
        gl_FragColor = vec4(col * rim * (0.28 + 0.95 * daySide), rim * 0.74);
      }
    `,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 2;
  return mesh;
}

// ---------- Build Seamless Stitched Cube-Sphere Planet Mesh ----------
function disposePlanet() {
  if (planetGroup) {
    planetGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (o.material.uniforms?.uMasks?.value) o.material.uniforms.uMasks.value.dispose();
        if (o.material.uniforms?.uClimate?.value) o.material.uniforms.uClimate.value.dispose();
        o.material.dispose();
      }
    });
    scene.remove(planetGroup);
    planetGroup = null;
  }
  faceMeshes = [];
  oceanMesh = null;
  atmoMesh = null;
  poleGroup = null;
}

function buildPlanetMesh() {
  disposePlanet();
  const { radius: R, faceSize: N, maxElev } = currentPlanet.meta;
  const segs = N - 1; // Exact 1:1 vertex match with tectonic grid -> zero sampling lag & seamless edges!
  const vertsPerSide = segs + 1;

  planetGroup = new THREE.Group();
  const tmpDir = new THREE.Vector3();
  const tU = new THREE.Vector3();
  const tV = new THREE.Vector3();
  const dPlusU = new THREE.Vector3();
  const dMinusU = new THREE.Vector3();
  const dPlusV = new THREE.Vector3();
  const dMinusV = new THREE.Vector3();
  const normalVec = new THREE.Vector3();
  const eps = 1.6 / N; // Small angular step for stitched cross-edge spherical normals!

  for (let fIdx = 0; fIdx < 6; fIdx++) {
    const face = currentPlanet.faces[fIdx];
    const heights = face.heights;
    const masks = face.masks;
    const climate = face.climate;

    const positions = new Float32Array(vertsPerSide * vertsPerSide * 3);
    const normals = new Float32Array(vertsPerSide * vertsPerSide * 3);
    const uvs = new Float32Array(vertsPerSide * vertsPerSide * 2);
    const elevAttr = new Float32Array(vertsPerSide * vertsPerSide);
    const indices = [];

    for (let j = 0; j < vertsPerSide; j++) {
      const fv = j / segs;
      const v = fv * 2 - 1;
      for (let i = 0; i < vertsPerSide; i++) {
        const fu = i / segs;
        const u = fu * 2 - 1;
        const k = j * vertsPerSide + i;

        const hRaw = heights[k];
        const elev = hRaw > 0 ? hRaw * genState.reliefScale : hRaw;
        faceUVToUnitDir(fIdx, u, v, tmpDir);
        // Keep underwater geometry within -26m of ocean surface so globe limb is a razor-smooth sphere
        // while preserving full bathymetry depth in elevAttr for shelf/trench shading!
        const rad = R + Math.max(elev, -26.0);
        positions[k * 3] = tmpDir.x * rad;
        positions[k * 3 + 1] = tmpDir.y * rad;
        positions[k * 3 + 2] = tmpDir.z * rad;
        uvs[k * 2] = fu;
        uvs[k * 2 + 1] = fv;
        elevAttr[k] = elev;

        // Compute exact stitched spherical finite-difference normal across all 6 faces -> ZERO edge creases!
        const ref = Math.abs(tmpDir.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
        tU.crossVectors(ref, tmpDir).normalize();
        tV.crossVectors(tmpDir, tU).normalize();

        dPlusU.copy(tmpDir).addScaledVector(tU, eps).normalize();
        dMinusU.copy(tmpDir).addScaledVector(tU, -eps).normalize();
        dPlusV.copy(tmpDir).addScaledVector(tV, eps).normalize();
        dMinusV.copy(tmpDir).addScaledVector(tV, -eps).normalize();

        const hPU = sampleSphereSurface(dPlusU).elev;
        const hMU = sampleSphereSurface(dMinusU).elev;
        const hPV = sampleSphereSurface(dPlusV).elev;
        const hMV = sampleSphereSurface(dMinusV).elev;

        const arcStep = 2.0 * eps * R;
        const dhdu = (hPU - hMU) / arcStep;
        const dhdv = (hPV - hMV) / arcStep;
        // Amplify normal relief slightly so coastal cordilleras & island arcs pop crisply without spiky geometry
        const slopeBoost = 2.6;
        normalVec
          .copy(tmpDir)
          .addScaledVector(tU, -dhdu * slopeBoost)
          .addScaledVector(tV, -dhdv * slopeBoost)
          .normalize();

        normals[k * 3] = normalVec.x;
        normals[k * 3 + 1] = normalVec.y;
        normals[k * 3 + 2] = normalVec.z;
      }
    }

    for (let j = 0; j < segs; j++) {
      for (let i = 0; i < segs; i++) {
        const a = j * vertsPerSide + i;
        const b = a + 1;
        const c = a + vertsPerSide;
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('aElev', new THREE.BufferAttribute(elevAttr, 1));
    geo.setIndex(indices);

    const maskTex = new THREE.DataTexture(masks, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
    maskTex.flipY = false;
    maskTex.wrapS = THREE.ClampToEdgeWrapping;
    maskTex.wrapT = THREE.ClampToEdgeWrapping;
    maskTex.minFilter = THREE.LinearFilter;
    maskTex.magFilter = THREE.LinearFilter;
    maskTex.needsUpdate = true;

    const climTex = new THREE.DataTexture(climate, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
    climTex.flipY = false;
    climTex.wrapS = THREE.ClampToEdgeWrapping;
    climTex.wrapT = THREE.ClampToEdgeWrapping;
    climTex.minFilter = THREE.LinearFilter;
    climTex.magFilter = THREE.LinearFilter;
    climTex.needsUpdate = true;

    const mat = createPlanetFaceMaterial(maskTex, climTex, maxElev * genState.reliefScale);
    const mesh = new THREE.Mesh(geo, mat);
    planetGroup.add(mesh);
    faceMeshes.push(mesh);
  }

  oceanMesh = createSphericalOcean(R);
  planetGroup.add(oceanMesh);
  atmoMesh = createAtmosphereHalo(R);
  planetGroup.add(atmoMesh);

  planetGroup.add(buildSphericalForest(R, maxElev));

  poleGroup = buildPoleMarkers(R);
  poleGroup.visible = document.getElementById('chk-poles').checked;
  planetGroup.add(poleGroup);

  scene.add(planetGroup);
  rebuildTargetPatchOverlay();
}

// ---------- Stage 1 -> Stage 2 Target Patch Overlay & Conditioning Extractor ----------
function tangentBasisForDir(centerDir) {
  const c = centerDir.clone().normalize();
  const ref = Math.abs(c.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const east = new THREE.Vector3().crossVectors(ref, c).normalize();
  const north = new THREE.Vector3().crossVectors(c, east).normalize();
  return { c, east, north };
}

function projectTangentToUnitDir(c, east, north, tx, ty, out = new THREE.Vector3()) {
  const ang = Math.hypot(tx, ty);
  if (ang < 1e-6) return out.copy(c);
  const ca = Math.cos(ang);
  const sa = Math.sin(ang) / ang;
  return out
    .set(
      c.x * ca + (east.x * tx + north.x * ty) * sa,
      c.y * ca + (east.y * tx + north.y * ty) * sa,
      c.z * ca + (east.z * tx + north.z * ty) * sa
    )
    .normalize();
}

function rebuildTargetPatchOverlay() {
  if (targetPatchGroup) {
    scene.remove(targetPatchGroup);
    targetPatchGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    targetPatchGroup = null;
  }
  if (!currentPlanet || !stage2.targetDir) return;

  const R = currentPlanet.meta.radius;
  const { c, east, north } = tangentBasisForDir(stage2.targetDir);
  const halfRad = THREE.MathUtils.degToRad(stage2.spanDeg * 0.5);
  const g = new THREE.Group();
  const tmp = new THREE.Vector3();

  // Build glowing curved square outline hugging the sphere surface
  const segs = 28;
  const borderPts = [];
  const edges = [
    [-1, -1, 1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, 1],
    [-1, 1, -1, -1],
  ];
  for (const [sx0, sy0, sx1, sy1] of edges) {
    for (let i = 0; i < segs; i++) {
      const f = i / segs;
      const tx = (sx0 + (sx1 - sx0) * f) * halfRad;
      const ty = (sy0 + (sy1 - sy0) * f) * halfRad;
      projectTangentToUnitDir(c, east, north, tx, ty, tmp);
      const s = sampleSphereSurface(tmp);
      const r = R + Math.max(s.elev, 0) + 22;
      borderPts.push(tmp.clone().multiplyScalar(r));
    }
  }
  borderPts.push(borderPts[0].clone());

  const outerLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(borderPts),
    new THREE.LineBasicMaterial({ color: '#38f2d8', transparent: true, opacity: 0.95 })
  );
  g.add(outerLine);

  // Subtle translucent center crosshair + beacon pin
  const crossLen = halfRad * 0.16;
  const crossH = [];
  const crossV = [];
  for (let i = -4; i <= 4; i++) {
    const u = (i / 4) * crossLen;
    projectTangentToUnitDir(c, east, north, u, 0, tmp);
    crossH.push(tmp.clone().multiplyScalar(R + Math.max(sampleSphereSurface(tmp).elev, 0) + 26));
    projectTangentToUnitDir(c, east, north, 0, u, tmp);
    crossV.push(tmp.clone().multiplyScalar(R + Math.max(sampleSphereSurface(tmp).elev, 0) + 26));
  }
  const crossMat = new THREE.LineBasicMaterial({ color: '#ffd166', transparent: true, opacity: 0.9 });
  g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(crossH), crossMat));
  g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(crossV), crossMat));

  const cSample = sampleSphereSurface(c);
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(22, 12, 10),
    new THREE.MeshBasicMaterial({ color: '#ffd166' })
  );
  dot.position.copy(c).multiplyScalar(R + Math.max(cSample.elev, 0) + 26);
  g.add(dot);

  targetPatchGroup = g;
  scene.add(targetPatchGroup);
}

function updateStage2TargetInfo() {
  const el = document.getElementById('stage2-target-info');
  if (!el || !currentPlanet) return;
  const { c, east, north } = tangentBasisForDir(stage2.targetDir);
  const halfRad = THREE.MathUtils.degToRad(stage2.spanDeg * 0.5);
  const tmp = new THREE.Vector3();

  let landCnt = 0;
  let totalCnt = 0;
  let peakH = 0;
  let sumTemp = 0;
  let landTempCnt = 0;
  for (let r = 0; r < 9; r++) {
    const ty = (0.5 - r / 8) * 2 * halfRad;
    for (let col = 0; col < 9; col++) {
      const tx = (col / 8 - 0.5) * 2 * halfRad;
      projectTangentToUnitDir(c, east, north, tx, ty, tmp);
      const s = sampleSphereSurface(tmp);
      totalCnt++;
      if (s.elev > 0) {
        landCnt++;
        if (s.elev > peakH) peakH = s.elev;
        sumTemp += s.tempC;
        landTempCnt++;
      }
    }
  }
  const latDeg = THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(c.y, -1, 1)));
  const lonDeg = THREE.MathUtils.radToDeg(Math.atan2(c.x, c.z));
  const latStr = `${Math.abs(latDeg).toFixed(1)}°${latDeg >= 0 ? 'N' : 'S'}`;
  const lonStr = `${Math.abs(lonDeg).toFixed(1)}°${lonDeg >= 0 ? 'E' : 'W'}`;
  const landPct = Math.round((landCnt / totalCnt) * 100);
  const avgTemp = landTempCnt > 0 ? Math.round(sumTemp / landTempCnt) : Math.round(sampleSphereSurface(c).tempC);

  let note = '';
  if (landPct < 8) {
    note = '<br/><span style="color:#fbbf24;">Mostly open ocean — click a coast or continent for richer terrain!</span>';
  } else if (landPct > 94) {
    note = '<br/><span style="color:#86efac;">Interior mountain / plateau basin</span>';
  } else {
    note = '<br/><span style="color:#7dd3fc;">Coastline + mountain range (ideal for 30m AI detail)</span>';
  }
  el.innerHTML =
    `<strong>Target: ${latStr} ${lonStr}</strong> (${stage2.spanDeg}° box)<br/>` +
    `Land <strong>${landPct}%</strong> · Peak <strong>${Math.round(peakH)} m</strong> · Avg <strong>${avgTemp}°C</strong>` +
    note;
}

function setStage2Target(unitDir) {
  stage2.targetDir.copy(unitDir).normalize();
  rebuildTargetPatchOverlay();
  updateStage2TargetInfo();
}

function float32ToBase64(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function extractStage2Payload() {
  const coarseSize = 8;
  const padding = 64;
  const fullCoarse = coarseSize + 2 * padding; // 136x136
  const refSize = 128;

  const { c, east, north } = tangentBasisForDir(stage2.targetDir);
  const cellStepRad = THREE.MathUtils.degToRad(stage2.spanDeg) / coarseSize;
  const tmp = new THREE.Vector3();

  const sampleGrid = (gridSize, totalSpanRad, withStd) => {
    const elev = new Float32Array(gridSize * gridSize);
    const temp = new Float32Array(gridSize * gridSize);
    const precip = new Float32Array(gridSize * gridSize);
    const tempStd = withStd ? new Float32Array(gridSize * gridSize) : null;
    const precipCv = withStd ? new Float32Array(gridSize * gridSize) : null;

    for (let r = 0; r < gridSize; r++) {
      const ty = ((gridSize - 1) * 0.5 - r) * (totalSpanRad / gridSize);
      for (let col = 0; col < gridSize; col++) {
        const tx = (col - (gridSize - 1) * 0.5) * (totalSpanRad / gridSize);
        projectTangentToUnitDir(c, east, north, tx, ty, tmp);
        const s = sampleSphereSurface(tmp);
        const idx = r * gridSize + col;
        // Map planet elevation [0..240] -> Earth DEM meters [15..3315], ocean -> [-2860..-60]
        const maxRef = Math.max(240.0 * genState.reliefScale, 120.0);
        if (s.elev > 0) {
          const norm = Math.min(1.25, s.elev / maxRef);
          elev[idx] = 15.0 + Math.pow(norm, 1.15) * 3300.0;
        } else {
          const dNorm = Math.min(1.0, -s.elev / 260.0);
          elev[idx] = -(60.0 + dNorm * 2800.0);
        }
        temp[idx] = Math.max(-18.0, Math.min(32.0, s.tempC));
        precip[idx] = 120.0 + Math.pow(s.moisture, 1.25) * 2800.0;
        if (withStd) {
          tempStd[idx] = (2.5 + Math.abs(tmp.y) * 11.5) * 100.0;
          precipCv[idx] = 22.0 + (1.0 - s.moisture) * 48.0;
        }
      }
    }
    return { elev, temp, precip, tempStd, precipCv };
  };

  const coarse = sampleGrid(fullCoarse, fullCoarse * cellStepRad, true);
  const ref = sampleGrid(refSize, coarseSize * cellStepRad, false);
  const latDeg = THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(c.y, -1, 1)));
  const lonDeg = THREE.MathUtils.radToDeg(Math.atan2(c.x, c.z));

  return {
    seed: genState.seed,
    archetype: genState.archetype,
    seaLevelBias: genState.seaLevelBias,
    latDeg: Number(latDeg.toFixed(2)),
    lonDeg: Number(lonDeg.toFixed(2)),
    spanDeg: stage2.spanDeg,
    coarseSize,
    padding,
    refSize,
    condElev: float32ToBase64(coarse.elev),
    condTemp: float32ToBase64(coarse.temp),
    condTempStd: float32ToBase64(coarse.tempStd),
    condPrecip: float32ToBase64(coarse.precip),
    condPrecipCv: float32ToBase64(coarse.precipCv),
    refElev: float32ToBase64(ref.elev),
    refTemp: float32ToBase64(ref.temp),
    refPrecip: float32ToBase64(ref.precip),
  };
}

async function pollStage2Health() {
  const pill = document.getElementById('stage2-pill');
  if (!pill) return;
  try {
    const resp = await fetch('/api/health', { cache: 'no-store' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    stage2.serverReady = !!data.ready;
    stage2.serverDevice = (data.device || 'mps').toUpperCase();
    if (data.ready) {
      pill.textContent = `● ${stage2.serverDevice} Ready`;
      pill.className = 'status-pill ready';
    } else if (data.error) {
      pill.textContent = '✕ Load Error';
      pill.className = 'status-pill offline';
    } else {
      pill.textContent = `◐ Warming ${stage2.serverDevice}…`;
      pill.className = 'status-pill warming';
    }
  } catch {
    stage2.serverReady = false;
    pill.textContent = '○ Server Offline';
    pill.className = 'status-pill offline';
  }
}

async function synthesizeStage2Target() {
  if (stage2.synthesizing || !currentPlanet) return;
  const btn = document.getElementById('btn-synthesize');
  const progWrap = document.getElementById('synth-progress-wrap');
  const progFill = document.getElementById('synth-progress-fill');
  const progText = document.getElementById('synth-status-text');
  const resBox = document.getElementById('synth-result');
  const resMetrics = document.getElementById('synth-result-metrics');
  const openBtn = document.getElementById('btn-open-3d');

  stage2.synthesizing = true;
  btn.disabled = true;
  btn.textContent = 'Synthesizing 30m AI Terrain…';
  resBox.style.display = 'none';
  progWrap.style.display = 'block';
  progFill.style.width = '6%';
  progText.textContent = 'Extracting Stage 1 tectonic conditioning fields…';

  // Poll live server phase while MPS runs
  const pollId = window.setInterval(async () => {
    try {
      const r = await fetch('/api/status', { cache: 'no-store' });
      if (!r.ok) return;
      const st = await r.json();
      if (st.message) progText.textContent = st.message;
      if (typeof st.progress === 'number') {
        progFill.style.width = `${Math.max(8, Math.round(st.progress * 100))}%`;
      }
    } catch {
      // ignore transient poll errors
    }
  }, 600);

  try {
    const payload = extractStage2Payload();
    progFill.style.width = '14%';
    progText.textContent = `Sending patch to ${stage2.serverDevice} Terrain Diffusion 30m…`;

    const resp = await fetch('/api/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || `Server error (${resp.status})`);
    }

    stage2.lastResult = data;
    progFill.style.width = '100%';
    progText.textContent = `Done in ${data.totalSec}s (MPS ${data.mpsSec}s · Hydro ${data.hydroSec}s)`;
    resMetrics.innerHTML =
      `<strong>✓ ${data.coordLabel} Synthesized!</strong> (${data.totalSec}s)<br/>` +
      `Coastline IoU <strong>${data.iouPct}%</strong> · Elev r=<strong>${data.elevCorr}</strong><br/>` +
      `<strong>${data.lakeCount}</strong> alpine lakes · <strong>${data.riverCount}</strong> rivers (0 dead-ends)`;
    openBtn.href = data.openUrl;
    resBox.style.display = 'block';
  } catch (err) {
    progFill.style.width = '100%';
    progText.innerHTML =
      `<span style="color:#fca5a5;">${err.message || 'Stage 2 server unreachable.'} ` +
      `(Run <code>just stage2-server</code>)</span>`;
  } finally {
    window.clearInterval(pollId);
    stage2.synthesizing = false;
    btn.disabled = false;
    btn.textContent = '✨ Synthesize 30m AI Terrain Here';
    pollStage2Health();
  }
}

function buildSphericalForest(R, maxElev) {
  const maxTrees = 4200;
  const coneGeo = new THREE.ConeGeometry(4.4, 15.5, 6);
  coneGeo.translate(0, 7.75, 0);
  const mat = new THREE.MeshLambertMaterial({ color: '#ffffff' });
  const inst = new THREE.InstancedMesh(coneGeo, mat, maxTrees);

  const dummy = new THREE.Object3D();
  const upY = new THREE.Vector3(0, 1, 0);
  const dir = new THREE.Vector3();
  const color = new THREE.Color();

  let s = 9001 + genState.seed * 3571;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);

  let placed = 0;
  let tries = 0;
  while (placed < maxTrees && tries < maxTrees * 18) {
    tries++;
    const z = rnd() * 2 - 1;
    const theta = rnd() * Math.PI * 2;
    const rad = Math.sqrt(Math.max(0, 1 - z * z));
    dir.set(rad * Math.cos(theta), z, rad * Math.sin(theta));
    const sample = sampleSphereSurface(dir);
    if (sample.elev < 8.0 || sample.elev > maxElev * genState.reliefScale * 0.56) continue;
    if (sample.lake > 0.35 || sample.river > 0.35) continue;
    if (sample.tempC < 2.0 || sample.moisture < 0.32) continue;

    const r = R + sample.elev - 1.2;
    dummy.position.copy(dir).multiplyScalar(r);
    dummy.quaternion.setFromUnitVectors(upY, dir);
    dummy.rotateY(rnd() * Math.PI * 2);
    const sc = 0.65 + rnd() * 0.72;
    dummy.scale.set(sc, sc * (0.85 + rnd() * 0.4), sc);
    dummy.updateMatrix();
    inst.setMatrixAt(placed, dummy.matrix);

    if (sample.tempC > 18) color.setRGB(0.10 + rnd() * 0.06, 0.32 + rnd() * 0.10, 0.13);
    else if (sample.tempC > 8) color.setRGB(0.15 + rnd() * 0.07, 0.37 + rnd() * 0.10, 0.15);
    else color.setRGB(0.13 + rnd() * 0.05, 0.25 + rnd() * 0.07, 0.16);
    inst.setColorAt(placed, color);
    placed++;
  }
  inst.count = placed;
  inst.instanceMatrix.needsUpdate = true;
  if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
  return inst;
}

function buildPoleMarkers(R) {
  const g = new THREE.Group();
  const magN = genState.magNorth.clone();
  const magS = magN.clone().negate();

  const axisGeo = new THREE.BufferGeometry().setFromPoints([
    magS.clone().multiplyScalar(R * 1.24),
    magN.clone().multiplyScalar(R * 1.24),
  ]);
  const axisMat = new THREE.LineBasicMaterial({ color: '#ff6666', transparent: true, opacity: 0.65 });
  g.add(new THREE.Line(axisGeo, axisMat));

  const nPin = new THREE.Mesh(
    new THREE.SphereGeometry(52, 16, 12),
    new THREE.MeshBasicMaterial({ color: '#ff5252' })
  );
  nPin.position.copy(magN).multiplyScalar(R * 1.24);
  g.add(nPin);

  const sPin = new THREE.Mesh(
    new THREE.SphereGeometry(46, 16, 12),
    new THREE.MeshBasicMaterial({ color: '#63b3ff' })
  );
  sPin.position.copy(magS).multiplyScalar(R * 1.24);
  g.add(sPin);

  const spinGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, -R * 1.18, 0),
    new THREE.Vector3(0, R * 1.18, 0),
  ]);
  g.add(new THREE.Line(spinGeo, new THREE.LineBasicMaterial({ color: '#c4dcff', transparent: true, opacity: 0.32 })));

  return g;
}

// ---------- Explorer Character & Home Beacon ----------
function buildExplorerAndBeacon() {
  explorerGroup = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(1.1, 2.2, 6, 10),
    new THREE.MeshStandardMaterial({ color: '#2b5ea8', roughness: 0.55 })
  );
  body.position.y = 2.0;
  explorerGroup.add(body);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.95, 12, 10),
    new THREE.MeshStandardMaterial({ color: '#f3cfb0', roughness: 0.6 })
  );
  head.position.y = 4.2;
  explorerGroup.add(head);
  const hat = new THREE.Mesh(
    new THREE.ConeGeometry(1.1, 1.8, 10),
    new THREE.MeshStandardMaterial({ color: '#d93829', roughness: 0.5 })
  );
  hat.position.y = 5.3;
  explorerGroup.add(hat);
  explorerGroup.visible = false;
  scene.add(explorerGroup);

  homeBeacon = new THREE.Group();
  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.6, 0.8, 42, 8),
    new THREE.MeshBasicMaterial({ color: '#ffd166' })
  );
  pole.position.y = 21;
  homeBeacon.add(pole);
  const orb = new THREE.Mesh(
    new THREE.SphereGeometry(3.2, 12, 10),
    new THREE.MeshBasicMaterial({ color: '#ff5e5b' })
  );
  orb.position.y = 43;
  homeBeacon.add(orb);
  homeBeacon.visible = false;
  scene.add(homeBeacon);
}

// ---------- Landing, Mode Switching & Dynamic Continent Buttons ----------
function landAtDirection(unitDir) {
  nav.surfDir.copy(unitDir).normalize();
  const ref = Math.abs(nav.surfDir.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  nav.surfForward.copy(ref).addScaledVector(nav.surfDir, -ref.dot(nav.surfDir)).normalize();
  nav.homeDir = nav.surfDir.clone();

  const R = currentPlanet.meta.radius;
  const rightVec = new THREE.Vector3().crossVectors(nav.surfForward, nav.surfDir).normalize();
  const beaconDir = nav.homeDir.clone().addScaledVector(rightVec, 18.0 / R).normalize();
  const s = sampleSphereSurface(beaconDir);
  homeBeacon.position.copy(beaconDir).multiplyScalar(R + Math.max(s.elev, 0));
  homeBeacon.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), beaconDir);
  homeBeacon.visible = true;

  setStage2Target(nav.surfDir);
  setMode('surface');
}

function landOnContinent(idx = 0) {
  if (!currentPlanet) return;
  const list = currentPlanet.meta.landmasses;
  if (list && list.length > 0) {
    const pick = list[Math.min(idx, list.length - 1)];
    landAtDirection(new THREE.Vector3(...pick.dir));
    return;
  }
  landAtDirection(new THREE.Vector3(0, 0, 1));
}

function refreshContinentButtons() {
  const container = document.getElementById('continent-buttons');
  if (!container || !currentPlanet) return;
  container.innerHTML = '';
  const masses = currentPlanet.meta.landmasses.slice(0, 6);
  masses.forEach((lm, i) => {
    const btn = document.createElement('button');
    const kind = lm.fracPct >= 1.6 ? `Continent ${i + 1}` : `Island ${i + 1}`;
    btn.textContent = `${kind} (${lm.fracPct.toFixed(1)}%)`;
    btn.addEventListener('click', () => landOnContinent(i));
    container.appendChild(btn);
  });
}

function setMode(mode) {
  nav.mode = mode;
  document.getElementById('btn-mode-orbit').classList.toggle('active', mode === 'orbit');
  document.getElementById('btn-mode-surface').classList.toggle('active', mode === 'surface');
  document.getElementById('orbit-hint').innerHTML =
    mode === 'orbit'
      ? '<strong>Orbit Mode:</strong> Drag to rotate globe · Scroll to zoom · <strong>Double-click any continent or island arc</strong> to land and walk around the sphere!'
      : '<strong>Surface Mode:</strong> Hold <strong>W</strong> to walk a true great circle around the planet back to your golden Start Beacon! Watch the <strong>Magnetic Compass</strong> near the poles.';
  if (explorerGroup) explorerGroup.visible = mode === 'surface';
  if (mode === 'surface' && !nav.homeDir) {
    landOnContinent(0);
  }
}

// ---------- True Live Spherical Plate-Tectonics Planet Generation ----------
function generatePlanetFromState(customSeed = null) {
  if (customSeed !== null) genState.seed = customSeed;
  const loading = document.getElementById('loading');
  loading.style.display = '';
  loading.textContent = `Simulating spherical plate tectonics (Seed #${genState.seed})…`;

  // Yield a frame so loading text shows if needed
  currentPlanet = generateTectonicPlanet({
    seed: genState.seed,
    archetype: genState.archetype,
    faceSize: 128,
    radius: 3000,
    seaLevelBias: genState.seaLevelBias,
    maxElev: 240,
  });

  genState.magNorth.set(...currentPlanet.meta.magNorth).normalize();
  buildPlanetMesh();
  refreshContinentButtons();

  // Aim orbit camera + Stage 2 target reticle toward the planet's largest continent so every roll opens on dramatic geography!
  if (currentPlanet.meta.landmasses.length > 0) {
    const d = currentPlanet.meta.landmasses[0].dir;
    nav.orbitYaw = Math.atan2(d[0], d[2]);
    nav.orbitPitch = THREE.MathUtils.clamp(Math.asin(Math.max(-0.8, Math.min(0.8, d[1]))), -0.75, 0.75);
    setStage2Target(new THREE.Vector3(...d));
  } else {
    setStage2Target(new THREE.Vector3(0, 0, 1));
  }
  const resBox = document.getElementById('synth-result');
  if (resBox) resBox.style.display = 'none';

  const m = currentPlanet.meta;
  document.getElementById('planet-subtitle').textContent =
    `Planet #${genState.planetNumber} (seed ${m.seed}) · ${m.plateCount} plates · ` +
    `${m.majorContinents} continents, ${m.islands} islands · ${m.landPct}% land (${m.genTimeMs} ms)`;

  loading.style.display = 'none';

  if (nav.mode === 'surface') {
    landOnContinent(0);
  }
}

function generateNewPlanet(explicitSeed = null) {
  genState.planetNumber++;
  const nextSeed = explicitSeed !== null ? explicitSeed : Math.floor(1 + Math.random() * 99999);
  generatePlanetFromState(nextSeed);
}

// ---------- UI Bindings ----------
document.getElementById('btn-reroll').addEventListener('click', () => generateNewPlanet());
document.getElementById('sel-archetype').addEventListener('change', (e) => {
  genState.archetype = e.target.value;
  generateNewPlanet();
});
document.getElementById('btn-mode-orbit').addEventListener('click', () => setMode('orbit'));
document.getElementById('btn-mode-surface').addEventListener('click', () => {
  if (nav.mode !== 'surface') landOnContinent(0);
});

document.getElementById('sea-slider').addEventListener('input', (e) => {
  genState.seaLevelBias = Number(e.target.value);
  const v = genState.seaLevelBias;
  document.getElementById('sea-val').textContent =
    Math.abs(v) < 0.01 ? 'Normal' : v > 0 ? `+${Math.round(v * 100)}% Ocean` : `${Math.round(v * 100)}% Ocean`;
  generatePlanetFromState(genState.seed);
});

document.getElementById('relief-slider').addEventListener('input', (e) => {
  genState.reliefScale = Number(e.target.value);
  document.getElementById('relief-val').textContent = `${genState.reliefScale.toFixed(1)}×`;
  if (currentPlanet) buildPlanetMesh();
});

document.getElementById('speed-slider').addEventListener('input', (e) => {
  nav.speedMult = Number(e.target.value);
  const labels = { 1: 'Stroll', 2: 'Walk / Sprint', 4: 'Fast', 7: 'Rover', 12: 'Jet' };
  document.getElementById('speed-val').textContent = labels[nav.speedMult] || `${nav.speedMult}×`;
});

document.getElementById('chk-tectonics').addEventListener('change', (e) => {
  genState.showTectonics = e.target.checked;
  faceMeshes.forEach((m) => {
    if (m.material?.uniforms?.uShowTectonics) {
      m.material.uniforms.uShowTectonics.value = genState.showTectonics ? 1.0 : 0.0;
    }
  });
});

document.getElementById('chk-poles').addEventListener('change', (e) => {
  if (poleGroup) poleGroup.visible = e.target.checked;
});

document.getElementById('sel-stage2-span').addEventListener('change', (e) => {
  stage2.spanDeg = Number(e.target.value) || 46;
  rebuildTargetPatchOverlay();
  updateStage2TargetInfo();
});

document.getElementById('btn-target-explorer').addEventListener('click', () => {
  if (!currentPlanet) return;
  const dir = nav.mode === 'surface' ? nav.surfDir.clone() : camera.position.clone().normalize();
  setStage2Target(dir);
});

document.getElementById('btn-synthesize').addEventListener('click', () => {
  synthesizeStage2Target();
});

// ---------- HUD & Live Magnetic Compass ----------
const elMode = document.getElementById('stat-mode');
const elLatLon = document.getElementById('stat-latlon');
const elElev = document.getElementById('stat-elev');
const elTemp = document.getElementById('stat-temp');
const elBiome = document.getElementById('stat-biome');
const elHome = document.getElementById('stat-home');
const elNeedle = document.getElementById('compass-needle');

const clock = new THREE.Clock();

function updateNavigation(dt) {
  if (!currentPlanet) return;
  const R = currentPlanet.meta.radius;

  if (nav.mode === 'orbit') {
    if (!pointerDown) nav.orbitYaw += dt * 0.025;
    const cp = Math.cos(nav.orbitPitch);
    camera.position.set(
      Math.sin(nav.orbitYaw) * cp * nav.orbitDist,
      Math.sin(nav.orbitPitch) * nav.orbitDist,
      Math.cos(nav.orbitYaw) * cp * nav.orbitDist
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);

    // Keep dramatic 3/4 studio sunlight keyed to the viewer so whichever hemisphere you inspect is brightly lit!
    const sunYaw = nav.orbitYaw + 0.58;
    const sunPitch = THREE.MathUtils.clamp(nav.orbitPitch + 0.28, -0.7, 0.85);
    const scp = Math.cos(sunPitch);
    sunDir.set(Math.sin(sunYaw) * scp, Math.sin(sunPitch), Math.cos(sunYaw) * scp).normalize();
    sunLight.position.copy(sunDir).multiplyScalar(15000);
    faceMeshes.forEach((m) => m.material?.uniforms?.uSunDir?.value.copy(sunDir));
    if (oceanMesh?.material?.uniforms?.uSunDir) oceanMesh.material.uniforms.uSunDir.value.copy(sunDir);
    if (atmoMesh?.material?.uniforms?.uSunDir) atmoMesh.material.uniforms.uSunDir.value.copy(sunDir);

    const subDir = camera.position.clone().normalize();
    updateHUD(subDir, null);
  } else {
    const up = nav.surfDir;
    nav.surfForward.addScaledVector(up, -nav.surfForward.dot(up)).normalize();
    const right = new THREE.Vector3().crossVectors(nav.surfForward, up).normalize();

    const move = new THREE.Vector3();
    if (keys.has('w') || keys.has('arrowup')) move.add(nav.surfForward);
    if (keys.has('s') || keys.has('arrowdown')) move.sub(nav.surfForward);
    if (keys.has('d') || keys.has('arrowright')) move.add(right);
    if (keys.has('a') || keys.has('arrowleft')) move.sub(right);

    if (move.lengthSq() > 0.001) {
      move.normalize();
      const boost = keys.has('shift') ? 2.6 : 1.0;
      const metersPerSec = 28.0 * nav.speedMult * boost;
      const dAngle = (metersPerSec * dt) / R;
      const rotAxis = new THREE.Vector3().crossVectors(up, move).normalize();
      const qStep = new THREE.Quaternion().setFromAxisAngle(rotAxis, dAngle);
      nav.surfDir.applyQuaternion(qStep).normalize();
      nav.surfForward.applyQuaternion(qStep).normalize();
    }

    const s = sampleSphereSurface(nav.surfDir);
    const groundElev = Math.max(s.elev, 0.0);
    const playerPos = nav.surfDir.clone().multiplyScalar(R + groundElev);

    const mBasis = new THREE.Matrix4().makeBasis(
      new THREE.Vector3().crossVectors(nav.surfForward, nav.surfDir).normalize(),
      nav.surfDir,
      nav.surfForward.clone().negate()
    );
    explorerGroup.position.copy(playerPos);
    explorerGroup.quaternion.setFromRotationMatrix(mBasis);

    const camOffsetBack = nav.surfForward.clone().multiplyScalar(-Math.cos(nav.surfPitch) * nav.surfCamDist * 1.45);
    const camOffsetUp = nav.surfDir.clone().multiplyScalar(Math.sin(nav.surfPitch) * nav.surfCamDist * 1.35 + 14.0);
    const camPos = playerPos.clone().add(camOffsetBack).add(camOffsetUp);

    const camDir = camPos.clone().normalize();
    const camFloor = R + Math.max(sampleSphereSurface(camDir).elev, 0.0) + 5.0;
    if (camPos.length() < camFloor) camPos.copy(camDir).multiplyScalar(camFloor);

    const lookTarget = playerPos.clone().addScaledVector(nav.surfDir, 5.5).addScaledVector(nav.surfForward, 36.0);
    camera.up.copy(nav.surfDir);
    camera.position.copy(camPos);
    camera.lookAt(lookTarget);

    updateHUD(nav.surfDir, nav.surfForward);
  }
}

function updateHUD(dir, forward) {
  const R = currentPlanet.meta.radius;
  const s = sampleSphereSurface(dir);
  const latDeg = THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1)));
  const lonDeg = THREE.MathUtils.radToDeg(Math.atan2(dir.x, dir.z));

  elMode.textContent = nav.mode === 'orbit' ? 'ORBIT' : s.elev <= 0.3 ? 'SAILING' : 'WALKING';
  elMode.className = `v badge ${nav.mode === 'orbit' ? 'diorama' : s.elev <= 0.3 ? 'wading' : ''}`;
  elLatLon.textContent = `${Math.abs(latDeg).toFixed(1)}°${latDeg >= 0 ? 'N' : 'S'} ${Math.abs(lonDeg).toFixed(1)}°${lonDeg >= 0 ? 'E' : 'W'}`;
  elElev.textContent = `${Math.round(Math.max(s.elev, 0))} m`;
  elTemp.textContent = `${Math.round(s.tempC)}°C`;

  let biome = 'Temperate Forest';
  if (s.elev <= 0.4) biome = 'Deep Ocean';
  else if (s.lake > 0.42) biome = 'Alpine Lake';
  else if (s.river > 0.25) biome = 'River Valley';
  else if (s.tempC < 0.0) biome = 'Polar Glacier / Snow';
  else if (s.tempC < 4.5) biome = 'Boreal Tundra';
  else if (s.moisture < 0.24 && s.tempC > 12) biome = 'Rain-Shadow Desert';
  else if (s.moisture < 0.36) biome = 'Dry Steppe / Savanna';
  else if (s.tempC > 19 && s.moisture > 0.56) biome = 'Tropical Rainforest';
  elBiome.textContent = biome;

  if (nav.homeDir) {
    const gcDistKm = (Math.acos(THREE.MathUtils.clamp(dir.dot(nav.homeDir), -1, 1)) * R) / 1000.0;
    elHome.textContent = `${gcDistKm.toFixed(2)} km`;
  } else {
    elHome.textContent = '—';
  }

  // Live Magnetic Compass Needle
  const magN = genState.magNorth;
  if (forward) {
    const toMag = magN.clone().addScaledVector(dir, -magN.dot(dir));
    if (toMag.lengthSq() > 1e-6) {
      toMag.normalize();
      const right = new THREE.Vector3().crossVectors(forward, dir).normalize();
      const bearing = Math.atan2(toMag.dot(right), toMag.dot(forward));
      elNeedle.style.transform = `rotate(${THREE.MathUtils.radToDeg(bearing).toFixed(1)}deg)`;
    }
  } else {
    const camRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    const camUp = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
    const bearing = Math.atan2(magN.dot(camRight), magN.dot(camUp));
    elNeedle.style.transform = `rotate(${THREE.MathUtils.radToDeg(bearing).toFixed(1)}deg)`;
  }
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  updateNavigation(dt);
  if (oceanMesh?.material?.uniforms?.uTime) {
    oceanMesh.material.uniforms.uTime.value += dt;
  }
  if (nav.mode === 'surface') {
    const day = THREE.MathUtils.clamp(nav.surfDir.dot(sunDir) * 0.5 + 0.55, 0.15, 1.0);
    scene.background.setRGB(0.06 + 0.36 * day, 0.12 + 0.52 * day, 0.22 + 0.62 * day);
  } else {
    scene.background.set('#04070f');
  }
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Expose debug/testing hooks
window.__terragrokPlanet = {
  generateNewPlanet,
  setArchetype(arch, seed = null) {
    genState.archetype = arch;
    document.getElementById('sel-archetype').value = arch;
    generateNewPlanet(seed);
  },
  setShowTectonics(on) {
    genState.showTectonics = !!on;
    document.getElementById('chk-tectonics').checked = !!on;
    faceMeshes.forEach((m) => {
      if (m.material?.uniforms?.uShowTectonics) {
        m.material.uniforms.uShowTectonics.value = genState.showTectonics ? 1.0 : 0.0;
      }
    });
  },
  setMode,
  landOnContinent,
  setOrbit(yaw, pitch, dist = 8200) {
    setMode('orbit');
    nav.orbitYaw = yaw;
    nav.orbitPitch = pitch;
    nav.orbitDist = dist;
  },
  getMeta: () => currentPlanet?.meta ?? null,
  setStage2Target(x, y, z) {
    setStage2Target(new THREE.Vector3(x, y, z));
  },
  synthesizeStage2Target,
  getStage2Result: () => stage2.lastResult,
};

// Boot!
document.getElementById('sel-archetype').value = genState.archetype;
buildExplorerAndBeacon();
generatePlanetFromState(genState.seed);
pollStage2Health();
window.setInterval(pollStage2Health, 4000);
animate();
