/**
 * Stage 2 (Option C) Step 1:
 * Extract a tangent-plane tectonic conditioning patch (elevation, temperature, precipitation)
 * from any location on a Stage 1 Spherical Tectonic Planet (viewer/src/tectonics.js)
 * for direct conditioning of Terrain Diffusion 30m!
 */
import fs from 'node:fs';
import { generateTectonicPlanet, unitDirToFaceUV } from '../viewer/src/tectonics.js';

const args = process.argv.slice(2);
function getArg(name, def) {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : def;
}

const seed = Number(getArg('seed', '101'));
const archetype = getArg('archetype', 'earth');
const coarseSize = Number(getArg('coarse', '8')); // 8x8 coarse cells -> 1024x1024 at pool=2 or 2048x2048 at pool=1
const spanDeg = Number(getArg('span', '46')); // angular width of region on sphere (degrees)
const outDir = '/Users/noam/Workspace/terragrok/out/tectonics_td30m';
fs.mkdirSync(outDir, { recursive: true });

console.log(`Generating Stage 1 tectonic planet (seed=${seed}, archetype=${archetype})...`);
const planet = generateTectonicPlanet({ seed, archetype, faceSize: 160, maxElev: 240 });
const N = planet.meta.faceSize;

function sampleDir(ux, uy, uz) {
  const m = unitDirToFaceUV(ux, uy, uz);
  const f = planet.faces[m.faceIdx];
  const gx = (m.u * 0.5 + 0.5) * (N - 1);
  const gy = (m.v * 0.5 + 0.5) * (N - 1);
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

  const h = (f.heights[i00] * (1 - tx) + f.heights[i10] * tx) * (1 - ty) +
            (f.heights[i01] * (1 - tx) + f.heights[i11] * tx) * ty;
  const tempNorm = f.climate[i00 * 4] / 255;
  const moist = f.climate[i00 * 4 + 1] / 255;
  const orog = f.climate[i00 * 4 + 2] / 255;
  const tempC = tempNorm * 60 - 20;
  return { h, tempC, moist, orog };
}

function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

// Auto-discover the most dramatic scenic coastal mountain + bay + peninsula patch on the planet!
let bestCenter = null;
let bestScore = -1e9;
const halfRad = (spanDeg * 0.5) * (Math.PI / 180);

for (let latDeg = -55; latDeg <= 55; latDeg += 10) {
  const lat = latDeg * (Math.PI / 180);
  for (let lonDeg = -180; lonDeg < 180; lonDeg += 12) {
    const lon = lonDeg * (Math.PI / 180);
    const center = [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)];
    const ref = Math.abs(center[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const east = normalize(cross(ref, center));
    const north = normalize(cross(center, east));

    let landCnt = 0;
    let totalCnt = 0;
    let maxH = 0;
    let sumOrog = 0;
    let sumTemp = 0;
    let landTempCnt = 0;
    for (let r = 0; r < 9; r++) {
      const ty = (0.5 - r / 8) * 2 * halfRad;
      for (let c = 0; c < 9; c++) {
        const tx = (c / 8 - 0.5) * 2 * halfRad;
        const d = normalize([
          center[0] + east[0] * tx + north[0] * ty,
          center[1] + east[1] * tx + north[1] * ty,
          center[2] + east[2] * tx + north[2] * ty,
        ]);
        const s = sampleDir(d[0], d[1], d[2]);
        totalCnt++;
        if (s.h > 0) {
          landCnt++;
          if (s.h > maxH) maxH = s.h;
          sumOrog += s.orog;
          sumTemp += s.tempC;
          landTempCnt++;
        }
      }
    }
    const landFrac = landCnt / totalCnt;
    // Ideal landing showcase: 42%..65% land (bold coastline + ocean bay!), strong mountains, temperate/varied temp!
    if (landFrac < 0.35 || landFrac > 0.72) continue;
    const avgTemp = landTempCnt > 0 ? sumTemp / landTempCnt : 0;
    const coastBalance = 1.0 - Math.abs(landFrac - 0.54) * 3.0;
    const tempBonus = avgTemp > 3 && avgTemp < 22 ? 1.0 : 0.2;
    const score = coastBalance * 2.2 + (maxH / 240) * 2.5 + (sumOrog / totalCnt) * 3.0 + tempBonus;
    if (score > bestScore) {
      bestScore = score;
      bestCenter = { center, east, north, latDeg, lonDeg, landFrac, maxH, avgTemp };
    }
  }
}

console.log(
  `Selected scenic tectonic region: lat=${bestCenter.latDeg}°, lon=${bestCenter.lonDeg}° | ` +
  `land=${(bestCenter.landFrac * 100).toFixed(1)}% | maxH=${bestCenter.maxH.toFixed(0)}m | avgTemp=${bestCenter.avgTemp.toFixed(1)}°C`
);

// Export padded coarse conditioning arrays (PADDING=64 on each side so Terrain Diffusion has smooth real planetary context!)
const PADDING = 64;
const fullCoarse = coarseSize + 2 * PADDING;
const elevCoarse = new Float32Array(fullCoarse * fullCoarse);
const tempCoarse = new Float32Array(fullCoarse * fullCoarse);
const tempStdCoarse = new Float32Array(fullCoarse * fullCoarse);
const precipCoarse = new Float32Array(fullCoarse * fullCoarse);
const precipCvCoarse = new Float32Array(fullCoarse * fullCoarse);

const cellStepRad = (spanDeg * (Math.PI / 180)) / coarseSize;

function samplePatchGrid(gridSize, totalSpanRad, outElev, outTemp, outPrecip, outTempStd, outPrecipCv) {
  for (let r = 0; r < gridSize; r++) {
    // Row 0 = north (+north), row gridSize-1 = south (-north)
    const ty = ((gridSize - 1) * 0.5 - r) * (totalSpanRad / gridSize);
    for (let c = 0; c < gridSize; c++) {
      const tx = (c - (gridSize - 1) * 0.5) * (totalSpanRad / gridSize);
      // Gnomonic/exponential tangent-plane projection onto sphere (smooth even far from center!)
      const ang = Math.hypot(tx, ty);
      let d;
      if (ang < 1e-6) {
        d = bestCenter.center;
      } else {
        const ca = Math.cos(ang);
        const sa = Math.sin(ang) / ang;
        d = normalize([
          bestCenter.center[0] * ca + (bestCenter.east[0] * tx + bestCenter.north[0] * ty) * sa,
          bestCenter.center[1] * ca + (bestCenter.east[1] * tx + bestCenter.north[1] * ty) * sa,
          bestCenter.center[2] * ca + (bestCenter.east[2] * tx + bestCenter.north[2] * ty) * sa,
        ]);
      }
      const s = sampleDir(d[0], d[1], d[2]);
      const idx = r * gridSize + c;
      // Convert planet-scale meters [0..240] -> real Earth DEM meters [0..3400] for Terrain Diffusion 30m!
      // Ocean floor [-290..0] -> real bathymetry [-3200..-80]
      if (s.h > 0) {
        const norm = s.h / 240.0;
        outElev[idx] = 15.0 + Math.pow(norm, 1.15) * 3300.0;
      } else {
        const dNorm = Math.min(1.0, -s.h / 260.0);
        outElev[idx] = -(60.0 + dNorm * 2800.0);
      }
      outTemp[idx] = Math.max(-18, Math.min(32, s.tempC));
      // WorldClim annual precipitation (mm/yr): 120mm desert -> 2900mm rainforest
      outPrecip[idx] = 120.0 + Math.pow(s.moist, 1.25) * 2800.0;
      if (outTempStd) {
        // WorldClim bio4 temperature seasonality (internal units °C * 100!)
        const absLat = Math.abs(d[1]);
        outTempStd[idx] = (2.5 + absLat * 11.5) * 100.0;
      }
      if (outPrecipCv) {
        // WorldClim bio15 precipitation seasonality CV (%)
        outPrecipCv[idx] = 22.0 + (1.0 - s.moist) * 48.0;
      }
    }
  }
}

// 1. Padded coarse conditioning grid (exact planetary context out to 64 cells!)
samplePatchGrid(
  fullCoarse,
  fullCoarse * cellStepRad,
  elevCoarse,
  tempCoarse,
  precipCoarse,
  tempStdCoarse,
  precipCvCoarse
);

// 2. High-res 256x256 reference view of the EXACT inner tectonic region (for BEFORE vs AFTER visual comparison!)
const refSize = 256;
const refElev = new Float32Array(refSize * refSize);
const refTemp = new Float32Array(refSize * refSize);
const refPrecip = new Float32Array(refSize * refSize);
samplePatchGrid(refSize, coarseSize * cellStepRad, refElev, refTemp, refPrecip, null, null);

// 3. Also export a 540x270 equirectangular preview of the full planet with the target box coordinates!
const W = 540, H = 270;
const globeElev = new Float32Array(W * H);
for (let py = 0; py < H; py++) {
  const lat = (0.5 - py / (H - 1)) * Math.PI;
  const cl = Math.cos(lat);
  const sl = Math.sin(lat);
  for (let px = 0; px < W; px++) {
    const lon = (px / (W - 1) - 0.5) * Math.PI * 2;
    const s = sampleDir(cl * Math.sin(lon), sl, cl * Math.cos(lon));
    globeElev[py * W + px] = s.h;
  }
}

fs.writeFileSync(`${outDir}/cond_elev.f32`, Buffer.from(elevCoarse.buffer));
fs.writeFileSync(`${outDir}/cond_temp.f32`, Buffer.from(tempCoarse.buffer));
fs.writeFileSync(`${outDir}/cond_temp_std.f32`, Buffer.from(tempStdCoarse.buffer));
fs.writeFileSync(`${outDir}/cond_precip.f32`, Buffer.from(precipCoarse.buffer));
fs.writeFileSync(`${outDir}/cond_precip_cv.f32`, Buffer.from(precipCvCoarse.buffer));
fs.writeFileSync(`${outDir}/ref_elev.f32`, Buffer.from(refElev.buffer));
fs.writeFileSync(`${outDir}/ref_temp.f32`, Buffer.from(refTemp.buffer));
fs.writeFileSync(`${outDir}/ref_precip.f32`, Buffer.from(refPrecip.buffer));
fs.writeFileSync(`${outDir}/globe_elev.f32`, Buffer.from(globeElev.buffer));

fs.writeFileSync(
  `${outDir}/patch_meta.json`,
  JSON.stringify(
    {
      seed,
      archetype,
      coarseSize,
      padding: PADDING,
      fullCoarse,
      refSize,
      spanDeg,
      latDeg: bestCenter.latDeg,
      lonDeg: bestCenter.lonDeg,
      landFrac: Number(bestCenter.landFrac.toFixed(3)),
      planetMeta: {
        plateCount: planet.meta.plateCount,
        majorContinents: planet.meta.majorContinents,
        islands: planet.meta.islands,
        landPct: planet.meta.landPct,
      },
      globeW: W,
      globeH: H,
    },
    null,
    2
  )
);

console.log(`Exported Stage 1 tectonic conditioning patch (${coarseSize}x${coarseSize}, padded ${fullCoarse}x${fullCoarse}) to ${outDir}`);
