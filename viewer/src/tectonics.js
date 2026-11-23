/**
 * TerraGrok — True Seamless Spherical Plate Tectonics & Continental Drift Engine
 *
 * Generates geologically diverse planets natively on the unit sphere S^2:
 *  - Zero cube-face bias and zero edge seams (continents, rifts, mountain belts,
 *    and rivers freely cross all 6 cube-sphere faces and both poles).
 *  - Power-law weighted tectonic plates with 3D Euler rotation poles (v = w x p).
 *  - Rigid spherical continental drift & rifting (splitting supercontinents along
 *    divergent boundaries to leave matching jigsaw margins + mid-ocean ridges).
 *  - Continent-continent Himalayan collisions & narrow land bridges / isthmuses.
 *  - Ocean-continent Andean coastal mountain cordilleras + offshore trenches.
 *  - Ocean-ocean curved volcanic island arcs (Japan / New Zealand / Indonesia style).
 *  - Stationary mantle-plume hotspot volcanic chains (Hawaii style).
 *  - Prevailing-wind orographic rain shadows + latitude/lapse climate.
 *  - Exact stitched-sphere Barnes Priority-Flood hydrology (0 dead-end rivers,
 *    lakes & rivers flow seamlessly across cube-face borders).
 */

export const CUBE_FACES = [
  { name: 'px', n: [1, 0, 0], u: [0, 0, -1], v: [0, -1, 0] },
  { name: 'nx', n: [-1, 0, 0], u: [0, 0, 1], v: [0, -1, 0] },
  { name: 'py', n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, 1] },
  { name: 'ny', n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, -1] },
  { name: 'pz', n: [0, 0, 1], u: [1, 0, 0], v: [0, -1, 0] },
  { name: 'nz', n: [0, 0, -1], u: [-1, 0, 0], v: [0, -1, 0] },
];

// ---------- Deterministic PRNG (Mulberry32) ----------
function makeRng(seed) {
  let a = (seed >>> 0) || 1;
  return function rnd() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- Fast Seamless 3D Value/Gradient Noise on S^2 ----------
class Noise3D {
  constructor(seed) {
    const rnd = makeRng(seed);
    this.perm = new Uint8Array(512);
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const tmp = p[i];
      p[i] = p[j];
      p[j] = tmp;
    }
    for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255];
  }

  // Standard Perlin 3D gradient noise in roughly [-1, 1]
  n3(x, y, z) {
    const X = Math.floor(x) & 255;
    const Y = Math.floor(y) & 255;
    const Z = Math.floor(z) & 255;
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);
    const zf = z - Math.floor(z);
    const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10);
    const v = yf * yf * yf * (yf * (yf * 6 - 15) + 10);
    const w = zf * zf * zf * (zf * (zf * 6 - 15) + 10);

    const p = this.perm;
    const A = p[X] + Y;
    const AA = p[A] + Z;
    const AB = p[A + 1] + Z;
    const B = p[X + 1] + Y;
    const BA = p[B] + Z;
    const BB = p[B + 1] + Z;

    const g000 = grad3(p[AA], xf, yf, zf);
    const g100 = grad3(p[BA], xf - 1, yf, zf);
    const g010 = grad3(p[AB], xf, yf - 1, zf);
    const g110 = grad3(p[BB], xf - 1, yf - 1, zf);
    const g001 = grad3(p[AA + 1], xf, yf, zf - 1);
    const g101 = grad3(p[BA + 1], xf - 1, yf, zf - 1);
    const g011 = grad3(p[AB + 1], xf, yf - 1, zf - 1);
    const g111 = grad3(p[BB + 1], xf - 1, yf - 1, zf - 1);

    const x00 = g000 + u * (g100 - g000);
    const x10 = g010 + u * (g110 - g010);
    const x01 = g001 + u * (g101 - g001);
    const x11 = g011 + u * (g111 - g011);
    const y0 = x00 + v * (x10 - x00);
    const y1 = x01 + v * (x11 - x01);
    return y0 + w * (y1 - y0);
  }

  fbm(x, y, z, octaves = 5, lac = 2.05, gain = 0.5) {
    let sum = 0;
    let amp = 1.0;
    let freq = 1.0;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += amp * this.n3(x * freq, y * freq, z * freq);
      norm += amp;
      amp *= gain;
      freq *= lac;
    }
    return sum / norm;
  }

  ridged(x, y, z, octaves = 4, lac = 2.1, gain = 0.52) {
    let sum = 0;
    let amp = 1.0;
    let freq = 1.0;
    let weight = 1.0;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      let n = 1.0 - Math.abs(this.n3(x * freq, y * freq, z * freq));
      n = n * n * weight;
      weight = Math.max(0.15, Math.min(1.0, n * 1.6));
      sum += n * amp;
      norm += amp;
      amp *= gain;
      freq *= lac;
    }
    return sum / norm;
  }
}

function grad3(hash, x, y, z) {
  const h = hash & 15;
  const u = h < 8 ? x : y;
  const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
  return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
}

// ---------- Vector & Rodrigues Spherical Rotation Helpers ----------
function randomUnitVec(rnd) {
  const z = rnd() * 2 - 1;
  const t = rnd() * Math.PI * 2;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(t), z, r * Math.sin(t)];
}

function normalize3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Build a 3x3 Rodrigues rotation matrix (flat 9-element Float64Array) around unit axis k by angle theta. */
function makeRotationMatrix(k, theta) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const t = 1 - c;
  const [x, y, z] = k;
  return new Float64Array([
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ]);
}

function applyMat3(m, x, y, z, out) {
  out[0] = m[0] * x + m[1] * y + m[2] * z;
  out[1] = m[3] * x + m[4] * y + m[5] * z;
  out[2] = m[6] * x + m[7] * y + m[8] * z;
  return out;
}

export function faceUVToUnitDir(faceIdx, u, v, out = [0, 0, 0]) {
  const f = CUBE_FACES[faceIdx];
  const x = f.n[0] + u * f.u[0] + v * f.v[0];
  const y = f.n[1] + u * f.u[1] + v * f.v[1];
  const z = f.n[2] + u * f.u[2] + v * f.v[2];
  const inv = 1 / (Math.hypot(x, y, z) || 1);
  out[0] = x * inv;
  out[1] = y * inv;
  out[2] = z * inv;
  return out;
}

export function unitDirToFaceUV(x, y, z) {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  let faceIdx = 0;
  let u = 0;
  let v = 0;
  if (ax >= ay && ax >= az) {
    if (x >= 0) { faceIdx = 0; u = -z / ax; v = -y / ax; }
    else { faceIdx = 1; u = z / ax; v = -y / ax; }
  } else if (ay >= ax && ay >= az) {
    if (y >= 0) { faceIdx = 2; u = x / ay; v = z / ay; }
    else { faceIdx = 3; u = x / ay; v = -z / ay; }
  } else {
    if (z >= 0) { faceIdx = 4; u = x / az; v = -y / az; }
    else { faceIdx = 5; u = -x / az; v = -y / az; }
  }
  return { faceIdx, u: Math.max(-1, Math.min(1, u)), v: Math.max(-1, Math.min(1, v)) };
}

function smoothstep(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

// ---------- Archetype Parameter Presets ----------
const ARCHETYPES = {
  earth: {
    label: 'Earth-Like (Varied Continents)',
    plateRange: [10, 14],
    superClusters: [4, 6],
    minorTerranes: [2, 4],
    driftRange: [0.22, 0.38],
    oceanFrac: [0.68, 0.74],
    islandArcStrength: 1.05,
    hotspotCount: [2, 4],
    lobeScale: 1.04,
  },
  supercontinent: {
    label: 'Supercontinent (1–2 Giant Landmasses)',
    plateRange: [7, 11],
    superClusters: [1, 2],
    minorTerranes: [1, 3],
    driftRange: [0.08, 0.19],
    oceanFrac: [0.64, 0.72],
    islandArcStrength: 0.90,
    hotspotCount: [2, 3],
    lobeScale: 1.26,
  },
  rifting: {
    label: 'Rifting Breakup (Split Jigsaw Continents)',
    plateRange: [9, 13],
    superClusters: [2, 3],
    minorTerranes: [2, 4],
    driftRange: [0.30, 0.48],
    oceanFrac: [0.68, 0.74],
    islandArcStrength: 0.90,
    hotspotCount: [2, 4],
    lobeScale: 1.14,
  },
  archipelago: {
    label: 'Ring-of-Fire Archipelagos & Island Arcs',
    plateRange: [12, 16],
    superClusters: [3, 5],
    minorTerranes: [5, 9],
    driftRange: [0.24, 0.44],
    oceanFrac: [0.78, 0.85],
    islandArcStrength: 1.42,
    hotspotCount: [5, 8],
    lobeScale: 0.74,
  },
};

function pickArchetype(archetype, rnd) {
  if (archetype && ARCHETYPES[archetype]) return archetype;
  const r = rnd();
  if (r < 0.38) return 'earth';
  if (r < 0.60) return 'rifting';
  if (r < 0.80) return 'supercontinent';
  return 'archipelago';
}

function randRange(rnd, a, b) {
  return a + rnd() * (b - a);
}
function randInt(rnd, a, b) {
  return a + Math.floor(rnd() * (b - a + 1));
}

// ---------- Tectonic Model Builder (Pure S^2) ----------
export function createTectonicModel({
  seed = 42,
  archetype = 'auto',
  plateCount = null,
  driftScale = 1.0,
  seaLevelBias = 0.0,
} = {}) {
  const rnd = makeRng(seed);
  const chosenArch = pickArchetype(archetype, rnd);
  const cfg = ARCHETYPES[chosenArch];

  const nPlates = plateCount
    ? Math.max(4, Math.min(20, Math.round(plateCount)))
    : randInt(rnd, cfg.plateRange[0], cfg.plateRange[1]);

  const driftAngleBase = randRange(rnd, cfg.driftRange[0], cfg.driftRange[1]) * driftScale;
  const targetOceanFrac = Math.max(
    0.45,
    Math.min(0.88, randRange(rnd, cfg.oceanFrac[0], cfg.oceanFrac[1]) + seaLevelBias)
  );

  const nWarp = new Noise3D(seed * 7919 + 17);
  const nCrust = new Noise3D(seed * 6271 + 101);
  const nMountain = new Noise3D(seed * 3571 + 313);
  const nDetail = new Noise3D(seed * 104729 + 557);

  // 1. Build power-law weighted tectonic plates (large Pacific/Eurasian + medium + small microplates)
  // Place seeds with mild repulsion so no two seeds sit right on top of each other,
  // while power-law weights guarantee wildly varied plate sizes!
  const plates = [];
  for (let i = 0; i < nPlates; i++) {
    let bestC = randomUnitVec(rnd);
    let bestMinD = -1;
    const tries = i === 0 ? 1 : 12;
    for (let t = 0; t < tries; t++) {
      const cand = randomUnitVec(rnd);
      let minD = 99;
      for (let j = 0; j < plates.length; j++) {
        const d = Math.acos(Math.max(-1, Math.min(1, dot3(cand, plates[j].center))));
        if (d < minD) minD = d;
      }
      if (minD > bestMinD) {
        bestMinD = minD;
        bestC = cand;
      }
    }

    // Power-law size bias: first ~25% are huge major plates, last ~25% are small microplates
    const rank = i / Math.max(1, nPlates - 1);
    let sizeBias = 0;
    if (rank < 0.25) sizeBias = randRange(rnd, 0.14, 0.28);
    else if (rank < 0.65) sizeBias = randRange(rnd, -0.05, 0.10);
    else sizeBias = randRange(rnd, -0.22, -0.08);

    // Random Euler rotation pole and speed (rad per epoch)
    const axis = randomUnitVec(rnd);
    const speed = randRange(rnd, 0.35, 1.35) * (rnd() < 0.5 ? 1 : -1);
    const totalAngle = driftAngleBase * speed;
    const invRot = makeRotationMatrix(axis, -totalAngle);
    const fwdRot = makeRotationMatrix(axis, totalAngle);

    // Plate buoyancy / continental affinity: 1 giant Pacific-style oceanic plate + ~22% other oceanic plates
    // so continents naturally disperse across BOTH hemispheres instead of crowding into one half!
    const isOceanicPlate = (i === 0) || (i > 2 && rnd() < 0.24);
    const ageBias = randRange(rnd, -0.15, 0.15);

    plates.push({
      id: i,
      center: bestC,
      sizeBias,
      axis,
      omega: speed,
      totalAngle,
      invRot,
      fwdRot,
      isOceanicPlate,
      ageBias,
    });
  }

  // 2. Build asymmetric proto-continents (anisotropic tapered craton lobes + smaller terranes)
  // Anchor major continental clusters on continental plates (pre-drift) so seeds NEVER lose their
  // continents to oceanic-plate subtraction and turn the ocean floor into scattered island confetti!
  const lobes = [];
  const nClusters = randInt(rnd, cfg.superClusters[0], cfg.superClusters[1]);
  const clusterCenters = [];
  for (let c = 0; c < nClusters; c++) {
    let bestC = randomUnitVec(rnd);
    let bestScore = -1e9;
    const tries = chosenArch === 'supercontinent' ? 8 : 16;
    for (let t = 0; t < tries; t++) {
      const cand = randomUnitVec(rnd);
      // Prefer placing continental cratons on continental plates (pre-drift rotated back!)
      const pInfo = findPlate(cand[0], cand[1], cand[2], plates, nWarp);
      const pl = plates[pInfo.p1];
      let minD = 1.6;
      for (let j = 0; j < clusterCenters.length; j++) {
        const d = Math.acos(Math.max(-1, Math.min(1, dot3(cand, clusterCenters[j]))));
        if (d < minD) minD = d;
      }
      const plateBonus = pl.isOceanicPlate ? -1.4 : 0.8;
      // For supercontinent, keep clusters close; for earth/rifting, allow organic varied spacing
      const spaceScore = chosenArch === 'supercontinent' ? -minD * 0.4 : Math.min(minD, 1.35);
      const score = plateBonus + spaceScore + rnd() * 0.35;
      if (score > bestScore) {
        bestScore = score;
        // Transform candidate by plate's inverse rotation so after drift it lands cleanly on this plate!
        bestC = applyMat3(pl.invRot, cand[0], cand[1], cand[2], [0, 0, 0]);
      }
    }
    clusterCenters.push(normalize3(bestC));
  }

  // Helper to construct an oriented tangent basis on S^2 for an elongated/tapered lobe
  function addLobe(center, rx, ry, taper, strength, isBridge = false) {
    const ref = Math.abs(center[1]) < 0.85 ? [0, 1, 0] : [1, 0, 0];
    const t1 = normalize3(cross3(ref, center));
    const t2 = normalize3(cross3(center, t1));
    const ang = rnd() * Math.PI * 2;
    const ca = Math.cos(ang);
    const sa = Math.sin(ang);
    const major = [t1[0] * ca + t2[0] * sa, t1[1] * ca + t2[1] * sa, t1[2] * ca + t2[2] * sa];
    const minor = [-t1[0] * sa + t2[0] * ca, -t1[1] * sa + t2[1] * ca, -t1[2] * sa + t2[2] * ca];
    lobes.push({ center, major, minor, rx, ry, taper, strength, isBridge });
  }

  // Major supercraton lobes with power-law continent sizing (huge -> medium -> compact!)
  for (let c = 0; c < nClusters; c++) {
    const base = clusterCenters[c];
    // First cluster is largest (Eurasia/Pangaea), subsequent clusters step down organically!
    const sizeTier = c === 0 ? 1.18 : c === 1 ? 0.94 : c === 2 ? 0.80 : randRange(rnd, 0.56, 0.78);
    const lobesPerCluster = chosenArch === 'supercontinent'
      ? randInt(rnd, 6, 9)
      : chosenArch === 'archipelago'
        ? randInt(rnd, 2, 3)
        : Math.max(2, Math.round(randInt(rnd, 3, 6) * sizeTier));

    let cursor = [...base];
    for (let l = 0; l < lobesPerCluster; l++) {
      const stepDir = randomUnitVec(rnd);
      const stepRad = l === 0 ? 0.0 : randRange(rnd, 0.14, 0.40) * cfg.lobeScale * sizeTier;
      const next = normalize3([
        cursor[0] + stepDir[0] * stepRad,
        cursor[1] + stepDir[1] * stepRad,
        cursor[2] + stepDir[2] * stepRad,
      ]);
      const mix = chosenArch === 'supercontinent' ? 0.38 : 0.45;
      cursor = normalize3([
        next[0] * (1 - mix) + base[0] * mix,
        next[1] * (1 - mix) + base[1] * mix,
        next[2] * (1 - mix) + base[2] * mix,
      ]);

      const rx = randRange(rnd, 0.22, 0.50) * cfg.lobeScale * sizeTier;
      // Anisotropic aspect ratio (1.45x to 3.4x) -> tapered South America / India / Japan / Italy forms!
      const ry = rx / randRange(rnd, 1.45, 3.35);
      const taper = randRange(rnd, -0.62, 0.62);
      const strength = randRange(rnd, 0.68, 1.12);
      addLobe(cursor, rx, ry, taper, strength, false);
    }
  }

  // Occasional narrow tectonic land bridge / isthmus (Panama / Sinai) linking two nearby clusters
  if (nClusters >= 2 && rnd() < 0.65) {
    // Pick the closest pair of clusters so the isthmus is a realistic narrow bridge!
    let bestI = 0, bestJ = 1, bestDot = -2;
    for (let i = 0; i < nClusters; i++) {
      for (let j = i + 1; j < nClusters; j++) {
        const d = dot3(clusterCenters[i], clusterCenters[j]);
        if (d > bestDot) { bestDot = d; bestI = i; bestJ = j; }
      }
    }
    if (bestDot > -0.1) {
      const mid = normalize3([
        clusterCenters[bestI][0] + clusterCenters[bestJ][0],
        clusterCenters[bestI][1] + clusterCenters[bestJ][1],
        clusterCenters[bestI][2] + clusterCenters[bestJ][2],
      ]);
      addLobe(mid, randRange(rnd, 0.24, 0.44), randRange(rnd, 0.048, 0.082), randRange(rnd, -0.35, 0.35), 0.80, true);
    }
  }

  // Smaller independent microcontinents / islands (Greenland, Madagascar, New Zealand, Great Britain scale)
  const nMinor = randInt(rnd, cfg.minorTerranes[0], cfg.minorTerranes[1]);
  for (let m = 0; m < nMinor; m++) {
    const pos = randomUnitVec(rnd);
    const rx = randRange(rnd, 0.09, 0.22);
    const ry = rx / randRange(rnd, 1.5, 3.6);
    addLobe(pos, rx, ry, randRange(rnd, -0.55, 0.55), randRange(rnd, 0.64, 0.98), false);
  }

  // 3. Mantle Plume Hotspots (Hawaii / Galapagos / Azores curved island trails)
  const nHotspots = randInt(rnd, cfg.hotspotCount[0], cfg.hotspotCount[1]);
  const hotspots = [];
  for (let h = 0; h < nHotspots; h++) {
    const pos = randomUnitVec(rnd);
    // Find which plate currently sits over this hotspot to trace its curved trail!
    const pInfo = findPlate(pos[0], pos[1], pos[2], plates, nWarp);
    const pl = plates[pInfo.p1];
    const nIslands = randInt(rnd, 4, 8);
    const spacingAngle = randRange(rnd, 0.032, 0.056);
    const chain = [];
    for (let k = 0; k < nIslands; k++) {
      // Older islands were carried forward by the plate's Euler rotation!
      const rotK = makeRotationMatrix(pl.axis, pl.omega * k * spacingAngle * 0.9);
      const ip = applyMat3(rotK, pos[0], pos[1], pos[2], [0, 0, 0]);
      const radius = randRange(rnd, 0.015, 0.027) * Math.pow(0.84, k);
      const height = randRange(rnd, 0.50, 0.88) * Math.pow(0.85, k);
      chain.push({ p: normalize3(ip), r: radius, h: height });
    }
    hotspots.push({ origin: pos, chain });
  }

  // Tilted Magnetic Pole (7° to 21° from geographic north)
  const magTilt = (7 + rnd() * 14) * (Math.PI / 180);
  const magAz = rnd() * Math.PI * 2;
  const magNorth = normalize3([
    Math.sin(magTilt) * Math.cos(magAz),
    Math.cos(magTilt),
    Math.sin(magTilt) * Math.sin(magAz),
  ]);

  return {
    seed,
    archetype: chosenArch,
    archetypeLabel: cfg.label,
    plates,
    lobes,
    hotspots,
    nWarp,
    nCrust,
    nMountain,
    nDetail,
    driftAngleBase,
    targetOceanFrac,
    islandArcStrength: cfg.islandArcStrength,
    magNorth,
  };
}

// Fast plate lookup with 3D domain-warped coordinates (returns 1st & 2nd closest weighted plates)
function findPlate(x, y, z, plates, nWarp) {
  // Multi-octave 3D warp makes plate boundaries organic & sinuous rather than straight arcs
  const w1 = 0.24;
  const wx = x + w1 * nWarp.fbm(x * 2.1 + 3.1, y * 2.1 - 1.7, z * 2.1 + 4.3, 3);
  const wy = y + w1 * nWarp.fbm(x * 2.1 - 5.2, y * 2.1 + 2.9, z * 2.1 - 1.1, 3);
  const wz = z + w1 * nWarp.fbm(x * 2.1 + 1.4, y * 2.1 + 6.3, z * 2.1 - 3.8, 3);
  const inv = 1 / (Math.hypot(wx, wy, wz) || 1);
  const px = wx * inv;
  const py = wy * inv;
  const pz = wz * inv;

  let best1 = -1e9;
  let best2 = -1e9;
  let p1 = 0;
  let p2 = 1;
  for (let i = 0; i < plates.length; i++) {
    const pl = plates[i];
    const c = pl.center;
    const score = px * c[0] + py * c[1] + pz * c[2] + pl.sizeBias;
    if (score > best1) {
      best2 = best1;
      p2 = p1;
      best1 = score;
      p1 = i;
    } else if (score > best2) {
      best2 = score;
      p2 = i;
    }
  }
  return { p1, p2, margin: best1 - best2, wx: px, wy: py, wz: pz };
}

// Evaluate pre-drift proto-craton crustal thickness field at unit vector (x, y, z)
function evalProtoCrust(x, y, z, lobes, nCrust) {
  // 3D domain warp gives realistic peninsulas, gulfs, inland seas, and fractal margins
  const wa = 0.22;
  const qx = x + wa * nCrust.fbm(x * 2.8 + 11.3, y * 2.8 - 4.1, z * 2.8 + 2.7, 3);
  const qy = y + wa * nCrust.fbm(x * 2.8 - 6.7, y * 2.8 + 8.4, z * 2.8 - 5.2, 3);
  const qz = z + wa * nCrust.fbm(x * 2.8 + 3.9, y * 2.8 + 1.2, z * 2.8 + 9.6, 3);
  const inv = 1 / (Math.hypot(qx, qy, qz) || 1);
  const ux = qx * inv;
  const uy = qy * inv;
  const uz = qz * inv;

  let best = -1.0;
  for (let i = 0; i < lobes.length; i++) {
    const L = lobes[i];
    const dCenter = ux * L.center[0] + uy * L.center[1] + uz * L.center[2];
    if (dCenter < -0.15) continue; // back hemisphere cull

    // Project onto local tangent plane (major/minor) for anisotropic tapered teardrop shape
    const a = ux * L.major[0] + uy * L.major[1] + uz * L.major[2];
    const b = ux * L.minor[0] + uy * L.minor[1] + uz * L.minor[2];
    const uNorm = a / L.rx;
    // Taper width linearly along major axis -> South America / India / Africa wedge!
    const widthScale = Math.max(0.28, 1.0 + L.taper * Math.max(-1, Math.min(1, uNorm)));
    const vNorm = b / (L.ry * widthScale);
    const r2 = uNorm * uNorm + vNorm * vNorm + Math.max(0, 1 - dCenter) * 0.35;
    if (r2 < 2.6) {
      const val = L.strength * Math.exp(-r2 * 1.18) - 0.14;
      if (val > best) best = val;
    }
  }

  // Medium-scale fractal bays, capes, and coastal islands — strictly masked to continental margins
  // so deep ocean basins stay vast and clean instead of sprouting random FBM island confetti!
  const coastFbm = nCrust.fbm(ux * 5.2 + 2.1, uy * 5.2 - 3.4, uz * 5.2 + 1.8, 4, 2.15, 0.52);
  const marginMask = smoothstep(-0.36, 0.08, best);
  return best + coastFbm * (0.03 + 0.24 * marginMask);
}

// ---------- Stitched Cube-Sphere Barnes Priority-Flood Hydrology ----------
class MinHeap {
  constructor(capacity) {
    this.keys = new Float32Array(capacity);
    this.vals = new Int32Array(capacity);
    this.size = 0;
  }
  push(key, val) {
    let i = this.size++;
    const keys = this.keys;
    const vals = this.vals;
    keys[i] = key;
    vals[i] = val;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (keys[p] <= keys[i]) break;
      const tk = keys[p]; keys[p] = keys[i]; keys[i] = tk;
      const tv = vals[p]; vals[p] = vals[i]; vals[i] = tv;
      i = p;
    }
  }
  popVal() {
    const out = this.vals[0];
    const n = --this.size;
    if (n > 0) {
      const keys = this.keys;
      const vals = this.vals;
      keys[0] = keys[n];
      vals[0] = vals[n];
      let i = 0;
      while (true) {
        let s = i;
        const l = (i << 1) + 1;
        const r = l + 1;
        if (l < n && keys[l] < keys[s]) s = l;
        if (r < n && keys[r] < keys[s]) s = r;
        if (s === i) break;
        const tk = keys[s]; keys[s] = keys[i]; keys[i] = tk;
        const tv = vals[s]; vals[s] = vals[i]; vals[i] = tv;
        i = s;
      }
    }
    return out;
  }
}

/**
 * Compute exact 4-connected + diagonal stitched neighbors across all 6 faces of the cube-sphere,
 * then run Barnes Priority-Flood to fill depressions (alpine lakes) and trace pit-free
 * precipitation-weighted rivers seamlessly across cube-face borders!
 */
function runSphericalHydrology(N, heights, precip, masksOut) {
  const N2 = N * N;
  const total = 6 * N2;
  const visited = new Uint8Array(total);
  const filled = new Float32Array(heights);
  const receiver = new Int32Array(total);
  receiver.fill(-1);

  const heap = new MinHeap(total + 64);

  // Seed priority flood from all ocean cells (elev <= 0)
  for (let i = 0; i < total; i++) {
    if (heights[i] <= 0) {
      visited[i] = 1;
      filled[i] = heights[i];
      heap.push(heights[i], i);
    }
  }

  // Cache boundary neighbor mapping via 3D direction projection so edges stitch with zero bugs
  const edgeCache = new Map();
  const tmpDir = [0, 0, 0];
  function getNeighbor(face, r, c, dr, dc) {
    const nr = r + dr;
    const nc = c + dc;
    if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
      return face * N2 + nr * N + nc;
    }
    const key = ((face * N + r) * N + c) * 9 + (dr + 1) * 3 + (dc + 1);
    let hit = edgeCache.get(key);
    if (hit !== undefined) return hit;
    const u = (nc / (N - 1)) * 2 - 1;
    const v = (nr / (N - 1)) * 2 - 1;
    faceUVToUnitDir(face, u, v, tmpDir);
    const mapped = unitDirToFaceUV(tmpDir[0], tmpDir[1], tmpDir[2]);
    const r2 = Math.max(0, Math.min(N - 1, Math.round((mapped.v + 1) * 0.5 * (N - 1))));
    const c2 = Math.max(0, Math.min(N - 1, Math.round((mapped.u + 1) * 0.5 * (N - 1))));
    hit = mapped.faceIdx * N2 + r2 * N + c2;
    edgeCache.set(key, hit);
    return hit;
  }

  const dirs = [
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-1, -1], [-1, 1], [1, -1], [1, 1],
  ];

  const order = new Int32Array(total);
  let orderLen = 0;

  while (heap.size > 0) {
    const cur = heap.popVal();
    order[orderLen++] = cur;
    const curElev = filled[cur];
    const face = Math.floor(cur / N2);
    const rem = cur - face * N2;
    const r = Math.floor(rem / N);
    const c = rem - r * N;

    for (let d = 0; d < 8; d++) {
      const nb = getNeighbor(face, r, c, dirs[d][0], dirs[d][1]);
      if (visited[nb]) continue;
      visited[nb] = 1;
      // Downstream receiver points strictly toward the already-drained ocean-connected cell!
      receiver[nb] = cur;
      const nbElev = heights[nb];
      const nextElev = nbElev <= curElev ? curElev + 0.002 : nbElev;
      filled[nb] = nextElev;
      heap.push(nextElev, nb);
    }
  }

  // Accumulate climate-gated liquid runoff in exact upstream->downstream topological order!
  // Dry deserts and frozen glaciers contribute 0.0 local runoff so they stay river-free
  // (exotic rivers can still traverse desert edges if fed by wet highland headwaters).
  const flowAcc = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    if (heights[i] > 0) {
      flowAcc[i] = precip[i] * 2.0;
    }
  }
  for (let k = orderLen - 1; k >= 0; k--) {
    const idx = order[k];
    const rec = receiver[idx];
    if (rec >= 0 && heights[idx] > 0) {
      flowAcc[rec] += flowAcc[idx];
    }
  }

  // Write river & alpine lake masks into RGBA masksOut (R=river, G=lake, B=tectonicBoundary, A=255)
  let lakeCells = 0;
  let riverCells = 0;
  const riverThresh = Math.max(28, (N * N) * 0.0026);
  for (let i = 0; i < total; i++) {
    const h = heights[i];
    const lakeDepth = h > 0 ? filled[i] - h : 0;
    let isLake = 0;
    if (h > 48.0 && lakeDepth > 38.0 && flowAcc[i] > 14.0 && precip[i] > 0.08) {
      isLake = Math.min(255, Math.round(150 + Math.min(105, (lakeDepth - 30) * 2.2)));
      // Raise lake surface slightly to flat spillway elevation
      heights[i] = filled[i] - 0.1;
      lakeCells++;
    }
    let riverVal = 0;
    if (h > 0 && flowAcc[i] > riverThresh) {
      const logF = Math.log10(flowAcc[i] / riverThresh + 1);
      riverVal = Math.min(255, Math.round(95 + logF * 135));
      riverCells++;
    }
    masksOut[i * 4 + 0] = riverVal;
    masksOut[i * 4 + 1] = isLake;
    masksOut[i * 4 + 3] = 255;
  }

  return { lakeCells, riverCells };
}

// ---------- Connected-Component Landmass Counter (True Stitched Sphere!) ----------
function analyzeContinents(N, heights) {
  const N2 = N * N;
  const total = 6 * N2;
  const seen = new Uint8Array(total);
  const queue = new Int32Array(total);
  const sizes = [];
  const centroids = [];
  const tmpDir = [0, 0, 0];

  function get4Nb(face, r, c, dr, dc) {
    const nr = r + dr;
    const nc = c + dc;
    if (nr >= 0 && nr < N && nc >= 0 && nc < N) return face * N2 + nr * N + nc;
    const u = (nc / (N - 1)) * 2 - 1;
    const v = (nr / (N - 1)) * 2 - 1;
    faceUVToUnitDir(face, u, v, tmpDir);
    const m = unitDirToFaceUV(tmpDir[0], tmpDir[1], tmpDir[2]);
    const r2 = Math.max(0, Math.min(N - 1, Math.round((m.v + 1) * 0.5 * (N - 1))));
    const c2 = Math.max(0, Math.min(N - 1, Math.round((m.u + 1) * 0.5 * (N - 1))));
    return m.faceIdx * N2 + r2 * N + c2;
  }

  const d4 = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  let totalLand = 0;
  for (let i = 0; i < total; i++) {
    if (heights[i] <= 0 || seen[i]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = i;
    seen[i] = 1;
    let cx = 0;
    let cy = 0;
    let cz = 0;
    let maxH = 0;
    let bestLandDir = null;

    while (head < tail) {
      const cur = queue[head++];
      totalLand++;
      const f = Math.floor(cur / N2);
      const rem = cur - f * N2;
      const r = Math.floor(rem / N);
      const c = rem - r * N;
      const u = (c / (N - 1)) * 2 - 1;
      const v = (r / (N - 1)) * 2 - 1;
      faceUVToUnitDir(f, u, v, tmpDir);
      cx += tmpDir[0];
      cy += tmpDir[1];
      cz += tmpDir[2];
      if (heights[cur] > maxH && heights[cur] < 520) {
        maxH = heights[cur];
        bestLandDir = [tmpDir[0], tmpDir[1], tmpDir[2]];
      }
      for (let d = 0; d < 4; d++) {
        const nb = get4Nb(f, r, c, d4[d][0], d4[d][1]);
        if (!seen[nb] && heights[nb] > 0) {
          seen[nb] = 1;
          queue[tail++] = nb;
        }
      }
    }
    sizes.push(tail);
    centroids.push({
      cells: tail,
      fracPct: (tail / total) * 100,
      dir: bestLandDir || normalize3([cx, cy, cz]),
    });
  }

  centroids.sort((a, b) => b.cells - a.cells);
  const majorContinents = centroids.filter((c) => c.fracPct >= 1.6).length;
  const islands = centroids.filter((c) => c.fracPct >= 0.04 && c.fracPct < 1.6).length;
  return {
    landPct: (totalLand / total) * 100,
    majorContinents,
    islands,
    landmasses: centroids.slice(0, 8),
  };
}

// ---------- Full Cube-Sphere Tectonic Planet Generator ----------
/**
 * Generates all 6 seamless cube-sphere faces of a brand-new tectonic planet!
 * Returns:
 *  - meta: radius, seed, archetype, continent stats, magNorth, plates, etc.
 *  - faces: array of 6 { heights: Float32Array, masks: Uint8Array, climate: Uint8Array, plateIds: Uint8Array }
 */
export function generateTectonicPlanet({
  seed = 42,
  archetype = 'auto',
  climateMode = 'earth',
  faceSize = 128,
  radius = 3000,
  plateCount = null,
  driftScale = 1.0,
  seaLevelBias = 0.0,
  maxElev = 680,
} = {}) {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const model = createTectonicModel({ seed, archetype, plateCount, driftScale, seaLevelBias });
  const { plates, lobes, hotspots, nWarp, nCrust, nMountain, nDetail } = model;

  const N = faceSize;
  const N2 = N * N;
  const total = 6 * N2;

  const rawHeight = new Float32Array(total);
  const plateIds = new Uint8Array(total);
  const boundaryStrength = new Float32Array(total); // >0 convergent, <0 divergent
  const orogenyField = new Float32Array(total);
  const unitX = new Float32Array(total);
  const unitY = new Float32Array(total);
  const unitZ = new Float32Array(total);

  const dir = [0, 0, 0];
  const p0A = [0, 0, 0];
  const p0B = [0, 0, 0];

  // Pass 1: Evaluate true seamless spherical tectonics at every (f, r, c) -> unit vector on S^2
  for (let f = 0; f < 6; f++) {
    const faceBase = f * N2;
    for (let r = 0; r < N; r++) {
      const v = (r / (N - 1)) * 2 - 1;
      for (let c = 0; c < N; c++) {
        const u = (c / (N - 1)) * 2 - 1;
        faceUVToUnitDir(f, u, v, dir);
        const x = dir[0];
        const y = dir[1];
        const z = dir[2];
        const idx = faceBase + r * N + c;
        unitX[idx] = x;
        unitY[idx] = y;
        unitZ[idx] = z;

        // 1. Plate lookup (3D warped weighted Voronoi on S^2)
        const plInfo = findPlate(x, y, z, plates, nWarp);
        const pl1 = plates[plInfo.p1];
        const pl2 = plates[plInfo.p2];
        plateIds[idx] = pl1.id;

        // 2. Inverse Euler rotation to reconstruct pre-drift crust (gives real rifting & matching coasts!)
        applyMat3(pl1.invRot, x, y, z, p0A);
        let crust1 = evalProtoCrust(p0A[0], p0A[1], p0A[2], lobes, nCrust);
        if (pl1.isOceanicPlate) crust1 -= 0.42;

        applyMat3(pl2.invRot, x, y, z, p0B);
        let crust2 = evalProtoCrust(p0B[0], p0B[1], p0B[2], lobes, nCrust);
        if (pl2.isOceanicPlate) crust2 -= 0.42;

        // 3. Relative plate velocity across boundary normal (from plate 1 -> plate 2)
        // Plate surface velocity v_i = omega_i * (axis_i x p)
        const v1x = pl1.omega * (pl1.axis[1] * z - pl1.axis[2] * y);
        const v1y = pl1.omega * (pl1.axis[2] * x - pl1.axis[0] * z);
        const v1z = pl1.omega * (pl1.axis[0] * y - pl1.axis[1] * x);

        const v2x = pl2.omega * (pl2.axis[1] * z - pl2.axis[2] * y);
        const v2y = pl2.omega * (pl2.axis[2] * x - pl2.axis[0] * z);
        const v2z = pl2.omega * (pl2.axis[0] * y - pl2.axis[1] * x);

        // Approximate boundary normal pointing from plate 1 toward plate 2
        let nx = pl2.center[0] - pl1.center[0];
        let ny = pl2.center[1] - pl1.center[1];
        let nz = pl2.center[2] - pl1.center[2];
        // Make tangent to sphere at p
        const ndp = nx * x + ny * y + nz * z;
        nx -= ndp * x; ny -= ndp * y; nz -= ndp * z;
        const nLen = Math.hypot(nx, ny, nz) || 1;
        nx /= nLen; ny /= nLen; nz /= nLen;

        // Convergence > 0 means plates collide; < 0 means plates diverge (rift!)
        const vRelX = v1x - v2x;
        const vRelY = v1y - v2y;
        const vRelZ = v1z - v2z;
        const vConv = vRelX * nx + vRelY * ny + vRelZ * nz;

        // Boundary distance proxy in radians (~margin / 1.2)
        const bDist = plInfo.margin;

        // 4. Rifting / Divergence vs Convergence / Subduction / Island Arcs!
        // Smoothly blend raw advected crust across passive/convergent boundaries so we never get
        // artificial hard Voronoi jump cliffs, while preserving crisp jagged rift oceans!
        let h = crust1;
        if (bDist < 0.055 && vConv >= -0.04) {
          const blendT = 0.5 * (1.0 - smoothstep(0.0, 0.055, bDist));
          h = crust1 * (1.0 - blendT) + crust2 * blendT;
        }
        let mountainBoost = 0;

        if (vConv < -0.04) {
          // DIVERGING BOUNDARY (RIFTING!)
          const divRate = Math.min(1.6, -vConv);
          // Jagged rift coastline modulation so split margins look fractal, not ruler-straight
          const riftJitter = nDetail.fbm(x * 7.2 + 2.1, y * 7.2 - 4.7, z * 7.2 + 1.3, 4) * 0.036;
          const bRift = Math.max(0, bDist + riftJitter);
          const riftWidth = 0.028 + divRate * model.driftAngleBase * 0.26;
          const riftMask = Math.exp(-(bRift * bRift) / (2 * riftWidth * riftWidth));
          h -= riftMask * (0.52 + divRate * 0.28);
          // Subtle underwater Mid-Ocean Ridge right along the center of the rift
          const mor = Math.exp(-(bRift * bRift) / (2 * 0.010 * 0.010));
          h += mor * 0.14;
          boundaryStrength[idx] = -riftMask;
        } else if (vConv > 0.04) {
          // CONVERGING BOUNDARY (COLLISION / SUBDUCTION / ISLAND ARCS!)
          const convRate = Math.min(1.8, vConv);
          const cont1 = crust1 > 0.02;
          const cont2 = crust2 > 0.02;

          // Anisotropic directional mountain spine along plate boundary!
          const ridge = nMountain.ridged(x * 7.5 + 1.3, y * 7.5 - 2.4, z * 7.5 + 5.1, 4, 2.15, 0.54);
          const ridgeFine = nDetail.ridged(x * 16.0 - 4.2, y * 16.0 + 3.1, z * 16.0 - 1.9, 3);

          if (cont1 && cont2) {
            // HIMALAYAN CONTINENT-CONTINENT COLLISION + SUTURE / LAND BRIDGE
            const belt = Math.exp(-(bDist * bDist) / (2 * 0.085 * 0.085));
            const uplift = belt * (0.42 + convRate * 0.44) * (0.45 + 0.85 * ridge);
            h += uplift + belt * 0.12;
            mountainBoost += uplift * 1.4;
            boundaryStrength[idx] = belt;
          } else if (cont1 && !cont2) {
            // ANDEAN COASTAL CORDILLERA (continental overriding edge)
            const cordillera = Math.exp(-((bDist - 0.026) * (bDist - 0.026)) / (2 * 0.046 * 0.046));
            const uplift = cordillera * (0.34 + convRate * 0.42) * (0.40 + 0.90 * ridge + 0.25 * ridgeFine);
            h += uplift;
            mountainBoost += uplift * 1.3;
            boundaryStrength[idx] = cordillera;
          } else if (!cont1 && cont2) {
            // OCEANIC SIDE OF SUBDUCTION: DEEP OFFSHORE TRENCH!
            const trench = Math.exp(-((bDist - 0.016) * (bDist - 0.016)) / (2 * 0.018 * 0.018));
            h -= trench * (0.42 + convRate * 0.25);
            boundaryStrength[idx] = trench * 0.7;
          } else {
            // OCEAN-OCEAN SUBDUCTION -> CURVED VOLCANIC ISLAND ARC (Japan / NZ / Indonesia / Aleutians!)
            // Volcanic arcs rise from the abyssal floor to pierce sea level (like real Mariana/Tonga/Japan arcs!)
            const isOverriding = (pl1.ageBias > pl2.ageBias) || (pl1.ageBias === pl2.ageBias && pl1.id > pl2.id);
            if (isOverriding && convRate > 0.12) {
              const arcBand = Math.exp(-((bDist - 0.036) * (bDist - 0.036)) / (2 * 0.018 * 0.018));
              const arcChain = nMountain.fbm(x * 11.5 + 7.1, y * 11.5 - 3.3, z * 11.5 + 2.2, 3);
              const arcPeaks = smoothstep(-0.08, 0.42, arcChain) * (0.55 + 0.65 * ridge);
              const arcBoost = arcBand * arcPeaks * (0.62 + convRate * 0.34) * model.islandArcStrength;
              if (arcBoost > 0.08) {
                const arcElev = -0.34 + arcBoost * 0.92;
                if (arcElev > h) h = arcElev;
                mountainBoost += arcBoost * 1.1;
              }
              boundaryStrength[idx] = arcBand;
            } else if (!isOverriding) {
              // Deep Marianas-style trench beside the island arc
              const trench = Math.exp(-((bDist - 0.014) * (bDist - 0.014)) / (2 * 0.016 * 0.016));
              h -= trench * 0.38;
              boundaryStrength[idx] = trench * 0.6;
            }
          }
        }

        // 5. Stationary Mantle Plume Hotspot Chains (Hawaii / Galapagos / Azores)
        // Shield volcanoes rise straight from the abyssal floor to break sea level in open oceans!
        for (let hi = 0; hi < hotspots.length; hi++) {
          const chain = hotspots[hi].chain;
          for (let k = 0; k < chain.length; k++) {
            const isl = chain[k];
            const dDot = x * isl.p[0] + y * isl.p[1] + z * isl.p[2];
            if (dDot < 0.96) continue;
            const angDist = Math.acos(Math.min(1, dDot));
            if (angDist < isl.r * 2.5) {
              const cone = Math.exp(-(angDist * angDist) / (2 * isl.r * isl.r));
              const rough = 0.45 + 0.72 * nDetail.ridged(x * 18.0 + 1.1, y * 18.0 - 2.3, z * 18.0 + 0.7, 3);
              const boost = cone * isl.h * 0.72 * rough;
              const hsElev = -0.38 + boost * 1.05;
              if (hsElev > h) h = hsElev;
              mountainBoost += boost * 0.75;
            }
          }
        }

        // 6. Fine 3D Fractal Coastlines, Fjords & Interior Terrain Relief (100% seamless on S^2!)
        const fineCoast = nDetail.fbm(x * 13.5 - 2.3, y * 13.5 + 5.7, z * 13.5 - 4.1, 4, 2.1, 0.5);
        const microCrag = nDetail.ridged(x * 24.0 + 3.1, y * 24.0 - 1.9, z * 24.0 + 6.4, 3);
        // Stronger fractal detail right along coastlines for fjords/bays, solid continental craton interiors
        const coastWeight = 0.052 + 0.045 * Math.exp(-h * h * 6.0);
        h += fineCoast * coastWeight + (h > 0.02 ? microCrag * 0.085 : 0);

        // Gently blend minor plate discontinuities ONLY inside deep ocean so ocean floors stay smooth
        if (bDist < 0.02 && h < -0.05 && crust1 < 0 && crust2 < 0) {
          h += (nDetail.n3(x * 8, y * 8, z * 8) * 0.03);
        }

        rawHeight[idx] = h;
        orogenyField[idx] = mountainBoost;
      }
    }
  }

  // Pass 2: Calibrate sea level to target Earth-like ocean fraction (bimodal hypsography!)
  const sampleCount = 12000;
  const stride = Math.max(1, Math.floor(total / sampleCount));
  const samples = [];
  for (let i = 0; i < total; i += stride) samples.push(rawHeight[i]);
  samples.sort((a, b) => a - b);
  const quantileSea = samples[Math.min(samples.length - 1, Math.floor(samples.length * model.targetOceanFrac))];
  // Guardrail: never let quantile sea-level drop down into the -0.85 abyssal ocean floor!
  const seaThreshold = Math.max(-0.27, quantileSea);

  // Convert normalized raw tectonic height -> realistic meters with bimodal continental shelf + alpine peaks
  const heights = new Float32Array(total);
  let maxRawAbove = 0.001;
  for (let i = 0; i < total; i++) {
    const d = rawHeight[i] - seaThreshold;
    if (d > maxRawAbove) maxRawAbove = d;
  }

  for (let i = 0; i < total; i++) {
    const d = rawHeight[i] - seaThreshold;
    if (d >= 0) {
      // Nonlinear Earth land hypsography: clean coastal plains (>4m) + dramatic tectonic mountain spines!
      const t = d / maxRawAbove;
      const mountainWeight = Math.min(1.0, orogenyField[i] * 1.45);
      const elev = 3.5 + (Math.pow(t, 1.15) * 0.65 + Math.pow(t, 2.0) * 0.35 + mountainWeight * t * 0.25) * (maxElev - 3.5);
      heights[i] = Math.min(maxElev * 1.08, elev);
    } else {
      // Sharp continental shelf drop-off into deep ocean basins
      const s = Math.min(1.0, -d / 0.55);
      const depth = -(3.0 + Math.pow(s, 0.75) * 230.0 + s * 60.0);
      heights[i] = depth;
    }
  }

  // Pass 3: Spherical Climate with Latitude + Lapse Rate + Prevailing-Wind Orographic Rain Shadows!
  const tempC = new Float32Array(total);
  const precip = new Float32Array(total);
  const climateAll = new Uint8Array(total * 4);
  const masksAll = new Uint8Array(total * 4);

  // Helper to sample height at arbitrary unit direction (for upwind rain shadow lookup!)
  function sampleHeightAtDir(ux, uy, uz) {
    const m = unitDirToFaceUV(ux, uy, uz);
    const r = Math.max(0, Math.min(N - 1, Math.round((m.v + 1) * 0.5 * (N - 1))));
    const c = Math.max(0, Math.min(N - 1, Math.round((m.u + 1) * 0.5 * (N - 1))));
    return heights[m.faceIdx * N2 + r * N + c];
  }

  // Climate preset offsets: Earth Varied | Arid Dune World | Frozen Ice World | Lush Tropical
  const climTempBias = climateMode === 'arid' ? 7.5 : climateMode === 'frozen' ? -15.5 : climateMode === 'lush' ? 4.0 : 0.0;
  const climMoistBias = climateMode === 'arid' ? -0.32 : climateMode === 'frozen' ? -0.10 : climateMode === 'lush' ? 0.24 : 0.0;
  const liquidRunoff = new Float32Array(total);

  for (let i = 0; i < total; i++) {
    const x = unitX[i];
    const y = unitY[i];
    const z = unitZ[i];
    const h = heights[i];
    const absLat = Math.abs(y); // sin(|lat|) in [0, 1]
    const latDeg = Math.asin(Math.min(1, absLat)) * (180 / Math.PI);

    // Latitude + elevation lapse rate temperature (°C)
    const regionalTempNoise = nCrust.fbm(x * 2.5 + 14.2, y * 2.5 - 3.1, z * 2.5 + 7.4, 2) * 4.2;
    const seaTemp = 29.0 - Math.pow(absLat, 1.32) * 47.0 + regionalTempNoise + climTempBias;
    const altDrop = (Math.max(0, h) / Math.max(1, maxElev)) * 21.0; // ~21°C drop on tallest alpine ranges -> snowcapped peaks!
    const tC = seaTemp - altDrop;
    tempC[i] = tC;

    // Hadley Cell baseline moisture warped by planetary Rossby waves so deserts NEVER look like horizontal stripes!
    const rossbyWarp = nCrust.fbm(x * 2.1 + 9.4, y * 2.1 - 5.2, z * 2.1 + 3.7, 3) * 14.0;
    const effLatDeg = Math.max(0, Math.min(90, latDeg + rossbyWarp));
    const tropWet = Math.exp(-Math.pow(effLatDeg / 19.0, 2)) * 0.48;
    const subtropDry = Math.exp(-Math.pow((effLatDeg - 27.0) / 13.0, 2)) * 0.44;
    const midLatWet = Math.exp(-Math.pow((effLatDeg - 54.0) / 15.0, 2)) * 0.40;
    const moistureNoise = nCrust.fbm(x * 3.1 - 7.1, y * 3.1 + 4.3, z * 3.1 - 2.6, 3) * 0.34;

    // Prevailing wind direction: Easterlies (Trades) in tropics (<32°), Westerlies in mid-lats (32°-68°)
    const windSign = latDeg < 32 ? -1 : 1; // +1 = blowing eastward (upwind is west)
    const eqLen = Math.hypot(x, z) || 1;
    const upX = -windSign * (-z / eqLen);
    const upZ = -windSign * (x / eqLen);

    // Sample upwind 3 steps to detect ocean breeze vs continental interior aridity & mountain rain shadows!
    let rainShadow = 0;
    let coastalHumid = 0;
    if (h > 0) {
      const s1 = 0.045;
      const s2 = 0.11;
      const s3 = 0.22;
      const u1 = normalize3([x + upX * s1, y, z + upZ * s1]);
      const u2 = normalize3([x + upX * s2, y, z + upZ * s2]);
      const u3 = normalize3([x + upX * s3, y, z + upZ * s3]);
      const hUp1 = sampleHeightAtDir(u1[0], u1[1], u1[2]);
      const hUp2 = sampleHeightAtDir(u2[0], u2[1], u2[2]);
      const hUp3 = sampleHeightAtDir(u3[0], u3[1], u3[2]);

      // Upwind ocean -> lush moist eastern/western windward coasts!
      if (hUp1 <= 0) coastalHumid += 0.28;
      else if (hUp2 <= 0) coastalHumid += 0.16;
      else if (hUp3 <= 0) coastalHumid += 0.05;
      else coastalHumid -= 0.22; // Deep continental interior continentality (Gobi/Sahara/Outback!)

      // Upwind high mountain barrier -> strong leeward rain shadow desert (Atacama/Patagonia!)
      const barrier = Math.max(hUp1, hUp2) - h;
      if (barrier > 60) rainShadow -= Math.min(0.52, (barrier - 60) / 340);
      // Windward slope rising into wind -> orographic rainfall!
      if (h - hUp1 > 50 && hUp2 < h) rainShadow += Math.min(0.28, (h - hUp1) / 320);
    }

    const moist = Math.max(0.02, Math.min(1.0, 0.31 + tropWet - subtropDry + midLatWet + coastalHumid + rainShadow + moistureNoise + climMoistBias));
    precip[i] = moist;
    // Physical liquid runoff for spherical rivers: 0 on frozen glaciers (<-3.5°C) and bone-dry deserts (<0.18)!
    const freezeFrac = Math.max(0, Math.min(1, (tC + 3.5) / 6.0));
    const aridFrac = Math.max(0, Math.min(1, (moist - 0.18) / 0.72));
    liquidRunoff[i] = freezeFrac * aridFrac;

    // Pack climate RGBA: R=tempNorm [-20..40]->[0..255], G=moisture [0..255], B=orogeny [0..255], A=255
    climateAll[i * 4 + 0] = Math.max(0, Math.min(255, Math.round(((tC + 20) / 60) * 255)));
    climateAll[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(moist * 255)));
    climateAll[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(Math.min(1, orogenyField[i] * 1.3) * 255)));
    climateAll[i * 4 + 3] = 255;

    // Pack tectonic boundary indicator in masks B channel (128=neutral, >128 convergent, <128 divergent)
    masksAll[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(128 + boundaryStrength[i] * 120)));
  }

  // Pass 4: Exact Seamless Stitched-Sphere Priority-Flood Rivers & Alpine Lakes (climate-gated!)
  const hydro = runSphericalHydrology(N, heights, liquidRunoff, masksAll);

  // Pass 5: Measure real continent count, island count, and largest landmass centroids!
  const stats = analyzeContinents(N, heights);

  // Slice into 6 per-face objects matching viewer/src/planet.js
  const faces = [];
  for (let f = 0; f < 6; f++) {
    const start = f * N2;
    faces.push({
      name: CUBE_FACES[f].name,
      heights: heights.subarray(start, start + N2),
      masks: masksAll.subarray(start * 4, (start + N2) * 4),
      climate: climateAll.subarray(start * 4, (start + N2) * 4),
      plateIds: plateIds.subarray(start, start + N2),
    });
  }

  const elapsedMs = Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0);

  return {
    meta: {
      seed,
      archetype: model.archetype,
      archetypeLabel: model.archetypeLabel,
      radius,
      diameterKm: Number(((radius * 2) / 1000).toFixed(2)),
      circumferenceKm: Number(((2 * Math.PI * radius) / 1000).toFixed(2)),
      faceSize: N,
      maxElev,
      plateCount: plates.length,
      magNorth: model.magNorth,
      landPct: Number(stats.landPct.toFixed(1)),
      majorContinents: stats.majorContinents,
      islands: stats.islands,
      landmasses: stats.landmasses,
      lakeCells: hydro.lakeCells,
      riverCells: hydro.riverCells,
      genTimeMs: elapsedMs,
    },
    faces,
  };
}
