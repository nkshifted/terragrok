/**
 * Seam atlas: one continuous height / mask / climate surface for the 3×3 tiles around the player.
 *
 * Pure JS (no three.js) so it is unit-testable under node (`viewer/tests/seamAtlas.test.mjs`).
 *
 * Conventions
 *  - Tile (qi, qj) covers world X in [(qi-0.5)·span, (qi+0.5)·span] and Z likewise; qj grows south.
 *  - A tile's native raster is node-registered: sample (r, c) sits at ((c/(N-1)) - 0.5)·span, so
 *    its first / last rows and columns lie exactly on the tile's edges.
 *  - The atlas is cell-registered: texel K sits at ((K + 0.5)/W - 0.5)·atlasSpan, so every slot
 *    covers exactly one tile and features land where the tile's own river / lake data says.
 *
 * Seam correction
 *  Tiles of one chart agree closely at shared edges, but not exactly (1K vs 2K tiles, a ghost
 *  beside real terrain, per-tile hydrology carving). Each tile therefore gets an additive
 *  correction that makes both sides of every shared edge equal to a common target:
 *    - the smooth part of the mismatch is spread over a wide band (no cliffs, no ramps you notice)
 *    - the remaining fine detail is cross-faded over a narrow one
 *  Real terrain outweighs ghosts, so a ghost bends to meet real terrain rather than warping it.
 *  The same correction is exposed through `seamOffsetAt` so rivers, lakes and props move with
 *  the ground they sit on.
 */

export const ATLAS_TILES = 3;
const HALF = (ATLAS_TILES - 1) / 2;
const GHOST_WEIGHT = 1e-3;
const EMPTY_HEIGHT = -30;

export function tileKey(qi, qj) {
  return `${qi},${qj}`;
}

function smooth01(t) {
  const x = t <= 0 ? 0 : t >= 1 ? 1 : t;
  return x * x * (3 - 2 * x);
}

/** 1 at the edge (t = 0) falling smoothly to 0 at t >= 1. */
function falloff(t) {
  return 1 - smooth01(t);
}

/** Bilinear sample of a node-registered size² raster at (u, v) in [0, 1]². */
function sampleNode(src, size, u, v) {
  const gx = Math.min(size - 1, Math.max(0, u * (size - 1)));
  const gy = Math.min(size - 1, Math.max(0, v * (size - 1)));
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const x1 = Math.min(size - 1, x0 + 1);
  const y1 = Math.min(size - 1, y0 + 1);
  const tx = gx - x0;
  const ty = gy - y0;
  const a = src[y0 * size + x0] * (1 - tx) + src[y0 * size + x1] * tx;
  const b = src[y1 * size + x0] * (1 - tx) + src[y1 * size + x1] * tx;
  return a * (1 - ty) + b * ty;
}

/**
 * Resample a tile's native rasters onto its cell-registered atlas slot. Done once per tile
 * (cache the result) so re-stitching after a neighbour lands does not touch unchanged tiles.
 *
 * @param {{heights: Float32Array, size: number, masks?: Uint8Array, climate?: Uint8Array, climateSize?: number, rock?: Uint8Array}} tile
 *        `rock` (bedrock weights, see experiments/volcano.py) is on the climate raster's grid.
 */
export function prepareTileSlot(tile, tileN = 1024, climN = 512) {
  const { heights, size } = tile;
  const grid = new Float32Array(tileN * tileN);
  for (let r = 0; r < tileN; r++) {
    const v = (r + 0.5) / tileN;
    for (let c = 0; c < tileN; c++) {
      grid[r * tileN + c] = sampleNode(heights, size, (c + 0.5) / tileN, v);
    }
  }

  let masks = null;
  if (tile.masks) {
    masks = new Uint8Array(tileN * tileN * 4);
    for (let r = 0; r < tileN; r++) {
      const sr = Math.min(size - 1, Math.round(((r + 0.5) / tileN) * (size - 1)));
      for (let c = 0; c < tileN; c++) {
        const sc = Math.min(size - 1, Math.round(((c + 0.5) / tileN) * (size - 1)));
        const si = (sr * size + sc) * 4;
        const di = (r * tileN + c) * 4;
        masks[di] = tile.masks[si];
        masks[di + 1] = tile.masks[si + 1];
        masks[di + 2] = tile.masks[si + 2];
        masks[di + 3] = tile.masks[si + 3];
      }
    }
  }

  let climate = null;
  if (tile.climate) {
    const cs = tile.climateSize || size;
    climate = new Uint8Array(climN * climN * 4);
    for (let r = 0; r < climN; r++) {
      const sr = Math.min(cs - 1, Math.round(((r + 0.5) / climN) * (cs - 1)));
      for (let c = 0; c < climN; c++) {
        const sc = Math.min(cs - 1, Math.round(((c + 0.5) / climN) * (cs - 1)));
        const si = (sr * cs + sc) * 4;
        const di = (r * climN + c) * 4;
        climate[di] = tile.climate[si];
        climate[di + 1] = tile.climate[si + 1];
        climate[di + 2] = tile.climate[si + 2];
        climate[di + 3] = 255;
      }
    }
  }

  let rock = null;
  if (tile.rock) {
    const cs = tile.climateSize || size;
    rock = new Uint8Array(climN * climN * 4);
    for (let r = 0; r < climN; r++) {
      const sr = Math.min(cs - 1, Math.round(((r + 0.5) / climN) * (cs - 1)));
      for (let c = 0; c < climN; c++) {
        const sc = Math.min(cs - 1, Math.round(((c + 0.5) / climN) * (cs - 1)));
        const si = (sr * cs + sc) * 4;
        rock.set(tile.rock.subarray(si, si + 4), (r * climN + c) * 4);
      }
    }
  }

  // Exact edge profiles from the native raster, sampled at the slot's texel centres
  const edges = { W: new Float32Array(tileN), E: new Float32Array(tileN), N: new Float32Array(tileN), S: new Float32Array(tileN) };
  for (let k = 0; k < tileN; k++) {
    const t = (k + 0.5) / tileN;
    edges.W[k] = sampleNode(heights, size, 0, t);
    edges.E[k] = sampleNode(heights, size, 1, t);
    edges.N[k] = sampleNode(heights, size, t, 0);
    edges.S[k] = sampleNode(heights, size, t, 1);
  }
  const last = size - 1;
  const corners = {
    NW: heights[0],
    NE: heights[last],
    SW: heights[last * size],
    SE: heights[last * size + last],
  };
  return { grid, masks, climate, rock, edges, corners, tileN, climN };
}

/** Box-blur `src` (edge-clamped) with the given radius, in two passes (≈ triangle filter). */
function blur1d(src, radius) {
  const n = src.length;
  if (radius < 1) return Float32Array.from(src);
  let cur = Float32Array.from(src);
  for (let pass = 0; pass < 2; pass++) {
    const out = new Float32Array(n);
    let acc = 0;
    for (let i = -radius; i <= radius; i++) acc += cur[Math.min(n - 1, Math.max(0, i))];
    const w = 2 * radius + 1;
    for (let i = 0; i < n; i++) {
      out[i] = acc / w;
      acc += cur[Math.min(n - 1, i + radius + 1)] - cur[Math.max(0, i - radius)];
    }
    cur = out;
  }
  return cur;
}

/**
 * Build the offset profile for one edge of one tile.
 * @param {Float32Array|null} raw   target − own along the edge (null when nothing adjoins it)
 * @param {number} c0, c1           required offsets at the edge's two corners
 * @returns {{low: Float32Array, high: Float32Array}} smooth part + fine residual (0 at both ends)
 */
function edgeProfile(raw, c0, c1, n, cornerPx, smoothPx) {
  const exact = new Float32Array(n);
  const r0 = raw ? raw[0] : 0;
  const r1 = raw ? raw[n - 1] : 0;
  for (let k = 0; k < n; k++) {
    const g0 = falloff(k / cornerPx);
    const g1 = falloff((n - 1 - k) / cornerPx);
    exact[k] = (raw ? raw[k] : 0) + (c0 - r0) * g0 + (c1 - r1) * g1;
  }
  const low = blur1d(exact, smoothPx);
  // Re-pin the smooth part to the corner offsets so the residual vanishes at both ends
  const pin = Math.max(2, smoothPx * 2);
  const d0 = exact[0] - low[0];
  const d1 = exact[n - 1] - low[n - 1];
  const high = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    low[k] += d0 * falloff(k / pin) + d1 * falloff((n - 1 - k) / pin);
    high[k] = exact[k] - low[k];
  }
  return { low, high };
}

function tileWeight(t) {
  return t.ghost ? GHOST_WEIGHT : 1;
}

/**
 * @param {object} opts
 * @param {number} opts.centerQi, opts.centerQj  centre tile of the 3×3 window
 * @param {Map<string, {qi:number, qj:number, ghost:boolean, slot:object}>} opts.tiles
 *        tiles keyed by tileKey(); `slot` from prepareTileSlot(); missing tiles are open sea
 * @param {number[]} [opts.fallbackClimate]  RGB used where a tile has no climate raster
 */
export function buildSeamAtlas({
  centerQi = 0,
  centerQj = 0,
  tiles,
  tileN = 1024,
  climN = 512,
  worldSpan = 4800,
  bandM = 1300,
  detailM = 56,
  cornerM = 1300,
  smoothM = 70,
  fallbackClimate = [128, 128, 128],
}) {
  const W = ATLAS_TILES * tileN;
  const CW = ATLAS_TILES * climN;
  const heights = new Float32Array(W * W);
  const masks = new Uint8Array(W * W * 4);
  const climate = new Uint8Array(CW * CW * 4);
  // Bedrock weights; all zero (plain sediment) where a tile has no rock raster
  const rock = new Uint8Array(CW * CW * 4);
  let hasRock = false;

  const pxPerM = tileN / worldSpan;
  const bandPx = Math.min(tileN * 0.5, Math.max(2, bandM * pxPerM));
  const detailPx = Math.max(1, detailM * pxPerM);
  const cornerPx = Math.max(2, cornerM * pxPerM);
  const smoothPx = Math.max(1, Math.round(smoothM * pxPerM));

  const get = (qi, qj) => tiles.get(tileKey(qi, qj)) || null;

  // Shared corner target: weighted mean of every tile touching that grid corner.
  // Corner (ci, cj) is the NW corner of tile (ci, cj).
  const cornerTarget = (ci, cj) => {
    let sw = 0;
    let sh = 0;
    const touch = [
      [get(ci, cj), 'NW'],
      [get(ci - 1, cj), 'NE'],
      [get(ci, cj - 1), 'SW'],
      [get(ci - 1, cj - 1), 'SE'],
    ];
    let count = 0;
    for (const [t, which] of touch) {
      if (!t) continue;
      const w = tileWeight(t);
      sw += w;
      sh += w * t.slot.corners[which];
      count++;
    }
    return count >= 2 ? sh / sw : null;
  };

  const slots = new Map();
  const fW = new Float32Array(tileN);
  const fE = new Float32Array(tileN);
  const dW = new Float32Array(tileN);
  const dE = new Float32Array(tileN);
  for (let k = 0; k < tileN; k++) {
    fW[k] = falloff((k + 0.5) / bandPx);
    fE[k] = falloff((tileN - k - 0.5) / bandPx);
    dW[k] = falloff((k + 0.5) / detailPx);
    dE[k] = falloff((tileN - k - 0.5) / detailPx);
  }

  for (let sj = 0; sj < ATLAS_TILES; sj++) {
    for (let si = 0; si < ATLAS_TILES; si++) {
      const qi = centerQi + si - HALF;
      const qj = centerQj + sj - HALF;
      const t = get(qi, qj);
      const c0 = si * tileN;
      const r0 = sj * tileN;

      if (!t) {
        for (let r = 0; r < tileN; r++) heights.fill(EMPTY_HEIGHT, (r0 + r) * W + c0, (r0 + r) * W + c0 + tileN);
        fillMasks(masks, W, c0, r0, tileN, null);
        fillClimate(climate, CW, si * climN, sj * climN, climN, null, fallbackClimate, 0);
        continue;
      }

      const own = t.slot;
      const wOwn = tileWeight(t);
      // Corner offsets (0 where nothing else touches the corner)
      const cOff = {};
      for (const [name, ci, cj] of [
        ['NW', qi, qj],
        ['NE', qi + 1, qj],
        ['SW', qi, qj + 1],
        ['SE', qi + 1, qj + 1],
      ]) {
        const target = cornerTarget(ci, cj);
        cOff[name] = target == null ? 0 : target - own.corners[name];
      }
      // Raw edge offsets toward the weighted target shared with each neighbour
      const rawEdge = (nb, ownEdge, nbEdge) => {
        if (!nb) return null;
        const wNb = tileWeight(nb);
        const k = wNb / (wOwn + wNb);
        const out = new Float32Array(tileN);
        const a = own.edges[ownEdge];
        const b = nb.slot.edges[nbEdge];
        for (let i = 0; i < tileN; i++) out[i] = (b[i] - a[i]) * k;
        return out;
      };
      const pW = edgeProfile(rawEdge(get(qi - 1, qj), 'W', 'E'), cOff.NW, cOff.SW, tileN, cornerPx, smoothPx);
      const pE = edgeProfile(rawEdge(get(qi + 1, qj), 'E', 'W'), cOff.NE, cOff.SE, tileN, cornerPx, smoothPx);
      const pN = edgeProfile(rawEdge(get(qi, qj - 1), 'N', 'S'), cOff.NW, cOff.NE, tileN, cornerPx, smoothPx);
      const pS = edgeProfile(rawEdge(get(qi, qj + 1), 'S', 'N'), cOff.SW, cOff.SE, tileN, cornerPx, smoothPx);

      const slot = { qi, qj, ghost: !!t.ghost, pW, pE, pN, pS, cOff };
      slots.set(tileKey(qi, qj), slot);

      const g = own.grid;
      for (let r = 0; r < tileN; r++) {
        const fN = fW[r];
        const fS = fE[r];
        const dN = dW[r];
        const dS = dE[r];
        const lowWr = pW.low[r];
        const lowEr = pE.low[r];
        const highWr = pW.high[r];
        const highEr = pE.high[r];
        const cornerN = fN * cOff.NW;
        const cornerNE = fN * cOff.NE;
        const cornerS = fS * cOff.SW;
        const cornerSE = fS * cOff.SE;
        let di = (r0 + r) * W + c0;
        let si2 = r * tileN;
        for (let c = 0; c < tileN; c++, di++, si2++) {
          const a = fW[c];
          const b = fE[c];
          const corr =
            a * lowWr +
            b * lowEr +
            fN * pN.low[c] +
            fS * pS.low[c] -
            a * (cornerN + cornerS) -
            b * (cornerNE + cornerSE) +
            dW[c] * highWr +
            dE[c] * highEr +
            dN * pN.high[c] +
            dS * pS.high[c];
          heights[di] = g[si2] + corr;
        }
      }
      fillMasks(masks, W, c0, r0, tileN, own.masks);
      fillClimate(climate, CW, si * climN, sj * climN, climN, own.climate, fallbackClimate, t.ghost ? 0 : 255);
      if (own.rock) {
        hasRock = true;
        for (let r = 0; r < climN; r++) {
          rock.set(own.rock.subarray(r * climN * 4, (r + 1) * climN * 4), ((sj * climN + r) * CW + si * climN) * 4);
        }
      }
    }
  }

  const atlasSpan = ATLAS_TILES * worldSpan;
  const meta = {
    tileN,
    climN,
    worldSpan,
    centerQi,
    centerQj,
    size: W,
    climateSize: CW,
    atlasSpan,
    centerX: centerQi * worldSpan,
    centerZ: centerQj * worldSpan,
    bandPx,
    detailPx,
  };
  return { heights, masks, climate, rock, hasRock, meta, slots };
}

function fillMasks(masks, W, c0, r0, tileN, src) {
  if (src) {
    for (let r = 0; r < tileN; r++) {
      masks.set(src.subarray(r * tileN * 4, (r + 1) * tileN * 4), ((r0 + r) * W + c0) * 4);
    }
    return;
  }
  // No hydrology: dry, neutral curvature, middling moisture
  for (let r = 0; r < tileN; r++) {
    let di = ((r0 + r) * W + c0) * 4;
    for (let c = 0; c < tileN; c++, di += 4) {
      masks[di] = 0;
      masks[di + 1] = 0;
      masks[di + 2] = 128;
      masks[di + 3] = 110;
    }
  }
}

function fillClimate(climate, CW, c0, r0, climN, src, fallback, alpha) {
  for (let r = 0; r < climN; r++) {
    let di = ((r0 + r) * CW + c0) * 4;
    let si = r * climN * 4;
    for (let c = 0; c < climN; c++, di += 4, si += 4) {
      if (src) {
        climate[di] = src[si];
        climate[di + 1] = src[si + 1];
        climate[di + 2] = src[si + 2];
      } else {
        climate[di] = fallback[0];
        climate[di + 1] = fallback[1];
        climate[di + 2] = fallback[2];
      }
      // Alpha marks real terrain (255) vs ghost / empty (0); the terrain shader hazes ghosts
      climate[di + 3] = alpha;
    }
  }
}

function atlasCoords(meta, x, z) {
  const W = meta.size;
  return {
    fx: ((x - meta.centerX) / meta.atlasSpan + 0.5) * W - 0.5,
    fz: ((z - meta.centerZ) / meta.atlasSpan + 0.5) * W - 0.5,
  };
}

/** Bilinear stitched height (raw metres) at world XZ; null outside the atlas. */
export function sampleAtlasHeight(atlas, x, z) {
  if (!atlas) return null;
  const { meta, heights } = atlas;
  const W = meta.size;
  const { fx, fz } = atlasCoords(meta, x, z);
  if (fx < -0.5 || fz < -0.5 || fx > W - 0.5 || fz > W - 0.5) return null;
  const cx = Math.min(W - 1, Math.max(0, fx));
  const cz = Math.min(W - 1, Math.max(0, fz));
  const x0 = Math.floor(cx);
  const z0 = Math.floor(cz);
  const x1 = Math.min(W - 1, x0 + 1);
  const z1 = Math.min(W - 1, z0 + 1);
  const tx = cx - x0;
  const tz = cz - z0;
  const a = heights[z0 * W + x0] * (1 - tx) + heights[z0 * W + x1] * tx;
  const b = heights[z1 * W + x0] * (1 - tx) + heights[z1 * W + x1] * tx;
  return a * (1 - tz) + b * tz;
}

/** Nearest masks [river, lake, curvature, moisture] (0..255) at world XZ; null outside. */
export function sampleAtlasMasks(atlas, x, z) {
  if (!atlas) return null;
  const { meta, masks } = atlas;
  const W = meta.size;
  const { fx, fz } = atlasCoords(meta, x, z);
  const cx = Math.round(fx);
  const cz = Math.round(fz);
  if (cx < 0 || cz < 0 || cx >= W || cz >= W) return null;
  const i = (cz * W + cx) * 4;
  return [masks[i], masks[i + 1], masks[i + 2], masks[i + 3]];
}

/** Nearest climate RGBA (0..255) at world XZ; null outside. */
export function sampleAtlasClimate(atlas, x, z) {
  if (!atlas) return null;
  const { meta, climate } = atlas;
  const CW = meta.climateSize;
  const cx = Math.floor(((x - meta.centerX) / meta.atlasSpan + 0.5) * CW);
  const cz = Math.floor(((z - meta.centerZ) / meta.atlasSpan + 0.5) * CW);
  if (cx < 0 || cz < 0 || cx >= CW || cz >= CW) return null;
  const i = (cz * CW + cx) * 4;
  return [climate[i], climate[i + 1], climate[i + 2], climate[i + 3]];
}

/** Tile index containing world XZ. */
export function tileAt(worldSpan, x, z) {
  return { qi: Math.floor(x / worldSpan + 0.5), qj: Math.floor(z / worldSpan + 0.5) };
}

/**
 * Seam correction (raw metres) the atlas applied at world XZ: stitched height − the owning
 * tile's own height. Add it to anything positioned from that tile's data (river points, lake
 * levels) so it stays glued to the stitched ground. 0 outside the atlas.
 */
export function seamOffsetAt(atlas, x, z) {
  if (!atlas) return 0;
  const { meta, slots } = atlas;
  const { qi, qj } = tileAt(meta.worldSpan, x, z);
  const s = slots.get(tileKey(qi, qj));
  if (!s) return 0;
  const n = meta.tileN;
  const u = (x / meta.worldSpan + 0.5 - qi) * n; // texel units across the tile, 0..n
  const v = (z / meta.worldSpan + 0.5 - qj) * n;
  const lerp = (arr, f) => {
    const p = Math.min(n - 1, Math.max(0, f - 0.5));
    const i0 = Math.floor(p);
    const i1 = Math.min(n - 1, i0 + 1);
    return arr[i0] + (arr[i1] - arr[i0]) * (p - i0);
  };
  const a = falloff(u / meta.bandPx);
  const b = falloff((n - u) / meta.bandPx);
  const fN = falloff(v / meta.bandPx);
  const fS = falloff((n - v) / meta.bandPx);
  return (
    a * lerp(s.pW.low, v) +
    b * lerp(s.pE.low, v) +
    fN * lerp(s.pN.low, u) +
    fS * lerp(s.pS.low, u) -
    a * (fN * s.cOff.NW + fS * s.cOff.SW) -
    b * (fN * s.cOff.NE + fS * s.cOff.SE) +
    falloff(u / meta.detailPx) * lerp(s.pW.high, v) +
    falloff((n - u) / meta.detailPx) * lerp(s.pE.high, v) +
    falloff(v / meta.detailPx) * lerp(s.pN.high, u) +
    falloff((n - v) / meta.detailPx) * lerp(s.pS.high, u)
  );
}
