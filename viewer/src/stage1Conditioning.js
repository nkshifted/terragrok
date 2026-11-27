/**
 * Stage 1 -> Stage 2 conditioning extraction (pure JS, no three.js, runs in browser AND node).
 *
 * A "chart" is the flat world a home patch lives in: an azimuthal-equidistant projection of
 * the planet centred on the home patch. Chart tile (qi, qj) is the square `spanDeg` wide whose
 * centre sits qi tiles east and qj tiles SOUTH of home (qj grows with viewer +Z / grid rows).
 * Every tile of a chart samples the same continuous planet field in the same frame, which is
 * what lets Stage 2 synthesize neighbours as shifted windows of one seamless diffusion world.
 *
 * planet.js uses this for the home patch (qi = qj = 0); experiments/stage1_patch.mjs uses it
 * server-side for streamed neighbours. Keep both on this one code path.
 */
import { VOLCANO_KINDS, generateTectonicPlanet, unitDirToFaceUV } from './tectonics.js';

/** Planet generation settings that must match between the globe page and the server. */
export const STAGE1_PLANET_DEFAULTS = { faceSize: 128, radius: 3000, maxElev: 240 };
export const CONDITIONING_PADDING = 64;

const DEG = Math.PI / 180;

export function generateStage1Planet({ seed, archetype = 'earth', climateMode = 'earth', seaLevelBias = 0, geoTimeMyr = 0 }) {
  return generateTectonicPlanet({
    seed,
    archetype,
    climateMode,
    seaLevelBias,
    geoTimeMyr,
    ...STAGE1_PLANET_DEFAULTS,
  });
}

export function coarseSizeForResolution(resolution) {
  const r = String(resolution || '2k').toLowerCase();
  return r === '4k' ? 32 : r === '1k' ? 8 : 16;
}

/** Bilinear elevation (m) + nearest climate at a unit direction. */
export function sampleSphere(planet, x, y, z, reliefScale = 1.0) {
  const N = planet.meta.faceSize;
  const { faceIdx, u, v } = unitDirToFaceUV(x, y, z);
  const f = planet.faces[faceIdx];
  const gx = (u * 0.5 + 0.5) * (N - 1);
  const gy = (v * 0.5 + 0.5) * (N - 1);
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const x1 = Math.min(x0 + 1, N - 1);
  const y1 = Math.min(y0 + 1, N - 1);
  const tx = gx - x0;
  const ty = gy - y0;
  const i00 = y0 * N + x0;
  const h =
    (f.heights[i00] * (1 - tx) + f.heights[y0 * N + x1] * tx) * (1 - ty) +
    (f.heights[y1 * N + x0] * (1 - tx) + f.heights[y1 * N + x1] * tx) * ty;
  const c00 = i00 * 4;
  return {
    elev: h > 0 ? h * reliefScale : h,
    tempC: (f.climate[c00] / 255.0) * 60.0 - 20.0,
    moisture: f.climate[c00 + 1] / 255.0,
  };
}

/** Bilinear bedrock weights (granite, volcanic, limestone, metamorphic; 0..255) at a unit direction. */
export function sampleRock(planet, x, y, z, out, o = 0) {
  const N = planet.meta.faceSize;
  const { faceIdx, u, v } = unitDirToFaceUV(x, y, z);
  const rock = planet.faces[faceIdx].rock;
  const gx = (u * 0.5 + 0.5) * (N - 1);
  const gy = (v * 0.5 + 0.5) * (N - 1);
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const x1 = Math.min(x0 + 1, N - 1);
  const y1 = Math.min(y0 + 1, N - 1);
  const tx = gx - x0;
  const ty = gy - y0;
  for (let c = 0; c < 4; c++) {
    out[o + c] =
      (rock[(y0 * N + x0) * 4 + c] * (1 - tx) + rock[(y0 * N + x1) * 4 + c] * tx) * (1 - ty) +
      (rock[(y1 * N + x0) * 4 + c] * (1 - tx) + rock[(y1 * N + x1) * 4 + c] * tx) * ty;
  }
  return out;
}

/** Unit direction -> chart plane (tx east, ty north; radians of arc from home). Inverse of chartToUnitDir. */
export function unitDirToChart(basis, x, y, z) {
  const { c, east, north } = basis;
  const ca = Math.max(-1, Math.min(1, x * c[0] + y * c[1] + z * c[2]));
  const ang = Math.acos(ca);
  const e = x * east[0] + y * east[1] + z * east[2];
  const n = x * north[0] + y * north[1] + z * north[2];
  const s = Math.hypot(e, n);
  if (s < 1e-9) return [0, 0];
  return [(e / s) * ang, (n / s) * ang];
}

/**
 * The planet's volcanoes that reach into chart tile (qi, qj) or the strip around it, in tile
 * units: (x, z) from the tile's centre (east, south; the tile spans -0.5..0.5) and `radius`
 * as a share of the tile's width. `baseDem` is the planet's own height at the site, in the same
 * DEM metres as the conditioning, so every tile that shares a cone builds it from one base.
 */
export function chartTileVolcanoes(planet, { reliefScale = 1.0, homeLatDeg, homeLonDeg, spanDeg = 46, qi = 0, qj = 0, reach = 0.2 }) {
  const basis = chartBasis(homeLatDeg, homeLonDeg);
  const spanRad = spanDeg * DEG;
  const out = [];
  for (const v of planet.volcanoes || []) {
    // The chart is only trusted near home; skip sites on the far side of the planet
    if (v.dir[0] * basis.c[0] + v.dir[1] * basis.c[1] + v.dir[2] * basis.c[2] < 0) continue;
    const [tx, ty] = unitDirToChart(basis, v.dir[0], v.dir[1], v.dir[2]);
    const x = tx / spanRad - qi;
    const z = -ty / spanRad - qj;
    const radius = v.radius / spanRad;
    if (Math.abs(x) > 0.5 + reach + radius || Math.abs(z) > 0.5 + reach + radius) continue;
    const k = VOLCANO_KINDS[v.kind];
    const baseElev = sampleSphere(planet, v.dir[0], v.dir[1], v.dir[2], reliefScale).elev;
    if (baseElev <= 0) continue; // stands in the sea at this resolution: nothing to build on
    out.push({
      id: v.id,
      kind: v.kind,
      x: Number(x.toFixed(5)),
      z: Number(z.toFixed(5)),
      radius: Number(radius.toFixed(5)),
      aspect: Number(v.aspect.toFixed(4)),
      crater: k.crater,
      craterDepth: k.craterDepth,
      active: v.active,
      ...(v.hotSpring ? { hotSpring: true } : {}),
      baseDem: Number(stage1ElevToDem(baseElev, reliefScale).toFixed(1)),
    });
  }
  return out;
}

/** (centre, east, north) unit vectors of the chart frame at lat/lon (Y-up globe). */
export function chartBasis(latDeg, lonDeg) {
  const lat = latDeg * DEG;
  const lon = lonDeg * DEG;
  const cp = Math.cos(lat);
  const c = [Math.sin(lon) * cp, Math.sin(lat), Math.cos(lon) * cp];
  const ref = Math.abs(c[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  let e = [ref[1] * c[2] - ref[2] * c[1], ref[2] * c[0] - ref[0] * c[2], ref[0] * c[1] - ref[1] * c[0]];
  const el = Math.hypot(e[0], e[1], e[2]) || 1;
  e = [e[0] / el, e[1] / el, e[2] / el];
  let n = [c[1] * e[2] - c[2] * e[1], c[2] * e[0] - c[0] * e[2], c[0] * e[1] - c[1] * e[0]];
  const nl = Math.hypot(n[0], n[1], n[2]) || 1;
  n = [n[0] / nl, n[1] / nl, n[2] / nl];
  return { c, east: e, north: n };
}

/** Chart plane (tx east, ty north; radians of arc from home) -> unit direction. */
export function chartToUnitDir(basis, tx, ty, out = [0, 0, 0]) {
  const { c, east, north } = basis;
  const ang = Math.hypot(tx, ty);
  if (ang < 1e-6) {
    out[0] = c[0];
    out[1] = c[1];
    out[2] = c[2];
    return out;
  }
  const ca = Math.cos(ang);
  const sa = Math.sin(ang) / ang;
  const x = c[0] * ca + (east[0] * tx + north[0] * ty) * sa;
  const y = c[1] * ca + (east[1] * tx + north[1] * ty) * sa;
  const z = c[2] * ca + (east[2] * tx + north[2] * ty) * sa;
  const l = Math.hypot(x, y, z) || 1;
  out[0] = x / l;
  out[1] = y / l;
  out[2] = z / l;
  return out;
}

/** Lat/lon (deg) of the centre of chart tile (qi, qj). */
export function chartTileLatLon(homeLatDeg, homeLonDeg, spanDeg, qi, qj) {
  const d = chartToUnitDir(chartBasis(homeLatDeg, homeLonDeg), qi * spanDeg * DEG, -qj * spanDeg * DEG);
  return {
    latDeg: Math.asin(Math.max(-1, Math.min(1, d[1]))) / DEG,
    lonDeg: Math.atan2(d[0], d[2]) / DEG,
  };
}

/** Stage 1 metres -> the Earth-DEM-like metres Terrain Diffusion is conditioned on. */
export function stage1ElevToDem(elev, reliefScale = 1.0) {
  if (elev > 0) {
    const maxRef = Math.max(240.0 * reliefScale, 120.0);
    const norm = Math.min(1.25, elev / maxRef);
    return 15.0 + Math.pow(norm, 1.15) * 3300.0;
  }
  const dNorm = Math.min(1.0, -elev / 260.0);
  return -(60.0 + dNorm * 2800.0);
}

/**
 * Sample a gridSize² raster whose centre is chart tile (qi, qj) and whose cells are
 * `cellRad` apart. Row 0 is the north edge, column 0 the west edge.
 */
function sampleChartGrid(planet, basis, reliefScale, gridSize, cellRad, offEastRad, offSouthRad, withStd, withRock = false) {
  const n = gridSize * gridSize;
  const elev = new Float32Array(n);
  const temp = new Float32Array(n);
  const precip = new Float32Array(n);
  const tempStd = withStd ? new Float32Array(n) : null;
  const precipCv = withStd ? new Float32Array(n) : null;
  const rock = withRock ? new Uint8Array(n * 4) : null;
  const rk = [0, 0, 0, 0];
  const d = [0, 0, 0];
  const half = (gridSize - 1) * 0.5;
  for (let r = 0; r < gridSize; r++) {
    const ty = (half - r) * cellRad - offSouthRad;
    for (let col = 0; col < gridSize; col++) {
      const tx = (col - half) * cellRad + offEastRad;
      chartToUnitDir(basis, tx, ty, d);
      const s = sampleSphere(planet, d[0], d[1], d[2], reliefScale);
      const idx = r * gridSize + col;
      elev[idx] = stage1ElevToDem(s.elev, reliefScale);
      temp[idx] = Math.max(-18.0, Math.min(32.0, s.tempC));
      precip[idx] = 35.0 + Math.pow(s.moisture, 1.35) * 2500.0;
      if (withStd) {
        tempStd[idx] = (2.5 + Math.abs(d[1]) * 11.5) * 100.0;
        precipCv[idx] = 22.0 + (1.0 - s.moisture) * 48.0;
      }
      if (withRock) {
        sampleRock(planet, d[0], d[1], d[2], rk);
        for (let ch = 0; ch < 4; ch++) rock[idx * 4 + ch] = Math.round(rk[ch]);
      }
    }
  }
  return { elev, temp, precip, tempStd, precipCv, rock };
}

/**
 * Conditioning + reference rasters for chart tile (qi, qj).
 *
 *  coarse.*  (coarseSize + 2·padding)² model conditioning, centred on the tile
 *  ref.*     Stage 1 reference (coastline lock, climate): refSize² over the tile, plus
 *            refPad extra cells on every side when `margin` is set. The server synthesizes
 *            that margin too and crops it after its local filters, so tile edges match.
 */
export const SYNTH_MARGIN_PX = 64;

export function extractChartTile(
  planet,
  { reliefScale = 1.0, homeLatDeg, homeLonDeg, spanDeg = 46, qi = 0, qj = 0, coarseSize = 16, padding = CONDITIONING_PADDING, refSize = null, margin = false }
) {
  const basis = chartBasis(homeLatDeg, homeLonDeg);
  const spanRad = spanDeg * DEG;
  const fullCoarse = coarseSize + 2 * padding;
  const ref = refSize || (coarseSize >= 32 ? 512 : 256);
  const offE = qi * spanRad;
  const offS = qj * spanRad;
  // Output pixels per tile = coarseSize * 128 (Stage 2 runs the model at coarse_pooling = 2)
  const refPad = margin ? Math.round((SYNTH_MARGIN_PX * ref) / (coarseSize * 128)) : 0;
  return {
    coarseSize,
    padding,
    refSize: ref,
    refPad,
    coarse: sampleChartGrid(planet, basis, reliefScale, fullCoarse, spanRad / coarseSize, offE, offS, true),
    ref: sampleChartGrid(planet, basis, reliefScale, ref + 2 * refPad, spanRad / ref, offE, offS, false, true),
    volcanoes: chartTileVolcanoes(planet, { reliefScale, homeLatDeg, homeLonDeg, spanDeg, qi, qj }),
  };
}

/** Low-res Stage 1 elevation (DEM metres) for a not-yet-synthesized "ghost" tile. */
export function extractGhostElev(planet, { reliefScale = 1.0, homeLatDeg, homeLonDeg, spanDeg = 46, qi = 0, qj = 0, size = 256 }) {
  const basis = chartBasis(homeLatDeg, homeLonDeg);
  const spanRad = spanDeg * DEG;
  return sampleChartGrid(planet, basis, reliefScale, size, spanRad / size, qi * spanRad, qj * spanRad, false).elev;
}

export function float32ToBase64(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** The conditioning fields of a Stage 2 request body, base64-encoded. */
export function encodeChartTilePayload(tile) {
  return {
    coarseSize: tile.coarseSize,
    padding: tile.padding,
    refSize: tile.refSize,
    refPad: tile.refPad,
    condElev: float32ToBase64(tile.coarse.elev),
    condTemp: float32ToBase64(tile.coarse.temp),
    condTempStd: float32ToBase64(tile.coarse.tempStd),
    condPrecip: float32ToBase64(tile.coarse.precip),
    condPrecipCv: float32ToBase64(tile.coarse.precipCv),
    refElev: float32ToBase64(tile.ref.elev),
    refTemp: float32ToBase64(tile.ref.temp),
    refPrecip: float32ToBase64(tile.ref.precip),
    // Bedrock weights (RGBA bytes, same grid as the other ref rasters) and the cones to build
    refRock: float32ToBase64(tile.ref.rock),
    volcanoes: tile.volcanoes,
  };
}
