import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { generateTectonicPlanet, unitDirToFaceUV } from '../viewer/src/tectonics.js';

const CASES = [
  { seed: 101, archetype: 'earth', title: 'Seed 101 · Earth-Like' },
  { seed: 204, archetype: 'rifting', title: 'Seed 204 · Rifting Breakup' },
  { seed: 309, archetype: 'supercontinent', title: 'Seed 309 · Supercontinent' },
  { seed: 417, archetype: 'archipelago', title: 'Seed 417 · Island Arcs' },
  { seed: 523, archetype: 'earth', title: 'Seed 523 · Earth-Like #2' },
  { seed: 888, archetype: 'auto', title: 'Seed 888 · Auto Tectonic' },
];

const W = 540;
const H = 270;
const outDir = '/Users/noam/Workspace/terragrok/out/planet';
fs.mkdirSync(outDir, { recursive: true });

function shadePixel(h, tempC, moist, river, lake, bound) {
  if (h <= 0) {
    // Bathymetry: turquoise continental shelf -> deep ocean -> dark subduction trench
    const d = Math.min(1, -h / 260);
    let r = Math.round(28 * (1 - d) + 9 * d);
    let g = Math.round(92 * (1 - d) + 28 * d);
    let b = Math.round(138 * (1 - d) + 64 * d);
    if (h > -28) {
      const sh = 1 - -h / 28;
      r = Math.round(r + sh * 30);
      g = Math.round(g + sh * 65);
      b = Math.round(b + sh * 50);
    }
    return [r, g, b];
  }
  if (lake > 110) return [32, 118, 178];
  if (river > 115) return [42, 128, 186];

  // Biome shading
  let r, g, b;
  if (tempC < -2.0) {
    r = 232; g = 239; b = 247; // snow/glacier
  } else if (h > 420) {
    r = 148; g = 138; b = 130; // alpine rock
    if (tempC < 1.5) { r = 225; g = 232; b = 240; }
  } else if (moist < 0.24 && tempC > 10) {
    r = 204; g = 178; b = 122; // hot desert (Atacama/Sahara)
  } else if (moist < 0.36) {
    r = 162; g = 156; b = 96; // steppe / savanna
  } else if (tempC > 19 && moist > 0.52) {
    r = 34; g = 98; b = 44; // tropical rainforest
  } else if (tempC < 4) {
    r = 104; g = 116; b = 92; // boreal tundra
  } else {
    r = 68; g = 126; b = 58; // temperate forest / grassland
  }

  // Subtle hillshade boost from elevation
  const eNorm = Math.min(1, h / 650);
  const boost = 0.85 + eNorm * 0.32;
  return [
    Math.min(255, Math.round(r * boost)),
    Math.min(255, Math.round(g * boost)),
    Math.min(255, Math.round(b * boost)),
  ];
}

const manifest = [];
for (const c of CASES) {
  const planet = generateTectonicPlanet({ seed: c.seed, archetype: c.archetype, faceSize: 128 });
  const m = planet.meta;
  console.log(
    `${c.title.padEnd(28)} | arch=${m.archetype.padEnd(14)} | plates=${String(m.plateCount).padStart(2)} | ` +
    `continents=${m.majorContinents} islands=${String(m.islands).padStart(2)} | land=${m.landPct}% | ` +
    `time=${m.genTimeMs}ms`
  );

  const N = m.faceSize;
  const rgb = new Uint8Array(W * H * 3);
  for (let py = 0; py < H; py++) {
    const lat = (0.5 - py / (H - 1)) * Math.PI;
    const cl = Math.cos(lat);
    const sl = Math.sin(lat);
    for (let px = 0; px < W; px++) {
      const lon = (px / (W - 1) - 0.5) * Math.PI * 2;
      const ux = cl * Math.sin(lon);
      const uy = sl;
      const uz = cl * Math.cos(lon);
      const mapped = unitDirToFaceUV(ux, uy, uz);
      const face = planet.faces[mapped.faceIdx];
      const r = Math.max(0, Math.min(N - 1, Math.round((mapped.v + 1) * 0.5 * (N - 1))));
      const col = Math.max(0, Math.min(N - 1, Math.round((mapped.u + 1) * 0.5 * (N - 1))));
      const idx = r * N + col;
      const elev = face.heights[idx];
      const tempC = (face.climate[idx * 4 + 0] / 255) * 60 - 20;
      const moist = face.climate[idx * 4 + 1] / 255;
      const river = face.masks[idx * 4 + 0];
      const lake = face.masks[idx * 4 + 1];
      const bound = face.masks[idx * 4 + 2];
      const [R, G, B] = shadePixel(elev, tempC, moist, river, lake, bound);
      const o = (py * W + px) * 3;
      rgb[o] = R;
      rgb[o + 1] = G;
      rgb[o + 2] = B;
    }
  }
  const rawPath = `${outDir}/tec_${c.seed}.rgb`;
  fs.writeFileSync(rawPath, rgb);
  manifest.push({
    rawPath,
    label: `${c.title} (${m.majorContinents} cont, ${m.islands} isl, ${m.landPct}% land, ${m.genTimeMs}ms)`,
  });
}

fs.writeFileSync(`${outDir}/tec_manifest.json`, JSON.stringify({ W, H, items: manifest }, null, 2));
console.log('Raw gallery exported, compositing PNG...');
