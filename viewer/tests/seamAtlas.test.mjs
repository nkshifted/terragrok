import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ATLAS_TILES,
  buildSeamAtlas,
  prepareTileSlot,
  sampleAtlasHeight,
  seamOffsetAt,
  tileAt,
  tileKey,
} from '../src/seamAtlas.js';

const N = 64; // atlas texels per tile in tests
const SPAN = 4800;
const OPTS = { tileN: N, climN: 16, worldSpan: SPAN };

/** Tile whose native raster is f(worldX, worldZ) sampled node-registered at `size`². */
function makeTile(qi, qj, f, { size = 33, ghost = false } = {}) {
  const heights = new Float32Array(size * size);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const x = (qi + c / (size - 1) - 0.5) * SPAN;
      const z = (qj + r / (size - 1) - 0.5) * SPAN;
      heights[r * size + c] = f(x, z);
    }
  }
  return { qi, qj, ghost, slot: prepareTileSlot({ heights, size }, N, 16) };
}

function tileMap(...tiles) {
  return new Map(tiles.map((t) => [tileKey(t.qi, t.qj), t]));
}

/** Largest height step between horizontally / vertically adjacent texels inside data slots. */
function maxStep(atlas, tiles) {
  const W = atlas.meta.size;
  const has = (K, R) => {
    const qi = atlas.meta.centerQi + Math.floor(K / N) - 1;
    const qj = atlas.meta.centerQj + Math.floor(R / N) - 1;
    return tiles.has(tileKey(qi, qj));
  };
  let m = 0;
  for (let r = 0; r < W; r++) {
    for (let c = 0; c < W; c++) {
      if (!has(c, r)) continue;
      const h = atlas.heights[r * W + c];
      if (c + 1 < W && has(c + 1, r)) m = Math.max(m, Math.abs(atlas.heights[r * W + c + 1] - h));
      if (r + 1 < W && has(c, r + 1)) m = Math.max(m, Math.abs(atlas.heights[(r + 1) * W + c] - h));
    }
  }
  return m;
}

test('a lone tile is pasted unchanged and the rest is open sea', () => {
  const f = (x, z) => 50 + x * 0.01 - z * 0.02;
  const tiles = tileMap(makeTile(0, 0, f));
  const atlas = buildSeamAtlas({ ...OPTS, tiles });
  assert.equal(atlas.meta.size, ATLAS_TILES * N);
  for (const [x, z] of [[0, 0], [-2300, 2300], [1000, -1700], [2300, 2300]]) {
    assert.ok(Math.abs(sampleAtlasHeight(atlas, x, z) - f(x, z)) < 0.05, `at ${x},${z}`);
    assert.ok(Math.abs(seamOffsetAt(atlas, x, z)) < 1e-9);
  }
  assert.ok(sampleAtlasHeight(atlas, 4800, 0) < 0, 'missing neighbour reads as sea');
  assert.equal(sampleAtlasHeight(atlas, 99999, 0), null);
});

test('tiles that already agree are left alone', () => {
  const f = (x, z) => 80 + 30 * Math.sin(x / 900) + 20 * Math.cos(z / 700);
  const tiles = tileMap(makeTile(0, 0, f, { size: 129 }), makeTile(1, 0, f, { size: 129 }), makeTile(0, 1, f, { size: 129 }));
  const atlas = buildSeamAtlas({ ...OPTS, tiles });
  for (const [x, z] of [[2390, 100], [2410, 100], [100, 2395], [100, 2405], [2000, 2000]]) {
    assert.ok(Math.abs(sampleAtlasHeight(atlas, x, z) - f(x, z)) < 0.6, `at ${x},${z}`);
  }
});

test('a height step between two tiles becomes a wide ramp, not a cliff', () => {
  const tiles = tileMap(makeTile(0, 0, () => 0), makeTile(1, 0, () => 100));
  const atlas = buildSeamAtlas({ ...OPTS, tiles });
  // 100 m spread over ~2 x 1300 m: far gentler than the raw 100 m step
  assert.ok(maxStep(atlas, tiles) < 8, `max step ${maxStep(atlas, tiles)}`);
  assert.ok(Math.abs(sampleAtlasHeight(atlas, 2400, 0) - 50) < 3, 'meets in the middle at the seam');
  assert.ok(Math.abs(sampleAtlasHeight(atlas, 0, 0)) < 1e-6, 'interior of the left tile untouched');
  assert.ok(Math.abs(sampleAtlasHeight(atlas, 4800, 0) - 100) < 1e-6, 'interior of the right tile untouched');
});

test('four tiles meeting at a corner stay continuous', () => {
  const tiles = tileMap(
    makeTile(0, 0, () => 0),
    makeTile(1, 0, () => 60),
    makeTile(0, 1, () => 120),
    makeTile(1, 1, () => -20)
  );
  const atlas = buildSeamAtlas({ ...OPTS, tiles });
  assert.ok(maxStep(atlas, tiles) < 12, `max step ${maxStep(atlas, tiles)}`);
  const corner = [0.25, -0.25].flatMap((dx) => [0.25, -0.25].map((dz) => sampleAtlasHeight(atlas, 2400 + dx, 2400 + dz)));
  assert.ok(Math.max(...corner) - Math.min(...corner) < 3, `corner spread ${corner}`);
});

test('an L-shaped set (missing diagonal) is still continuous on its two seams', () => {
  const tiles = tileMap(makeTile(0, 0, () => 0), makeTile(1, 0, () => 90), makeTile(0, 1, () => -40));
  const atlas = buildSeamAtlas({ ...OPTS, tiles });
  assert.ok(maxStep(atlas, tiles) < 12, `max step ${maxStep(atlas, tiles)}`);
});

test('a ghost bends to meet real terrain; real terrain does not move', () => {
  const tiles = tileMap(makeTile(0, 0, () => 10), makeTile(1, 0, () => 200, { ghost: true }));
  const atlas = buildSeamAtlas({ ...OPTS, tiles });
  for (const x of [-2000, 0, 2000, 2390]) {
    assert.ok(Math.abs(sampleAtlasHeight(atlas, x, 0) - 10) < 0.5, `real tile moved at x=${x}`);
  }
  assert.ok(Math.abs(sampleAtlasHeight(atlas, 2440, 0) - 10) < 8, 'ghost edge meets the real tile');
  assert.ok(Math.abs(sampleAtlasHeight(atlas, 4800, 0) - 200) < 1e-6, 'ghost interior untouched');
});

test('seamOffsetAt reports exactly what the atlas added', () => {
  const fa = (x, z) => 40 + 0.01 * x + 15 * Math.sin(z / 300);
  const fb = (x, z) => 95 - 0.02 * z + 10 * Math.cos(x / 250);
  const tiles = tileMap(makeTile(0, 0, fa, { size: 257 }), makeTile(1, 0, fb, { size: 257 }), makeTile(0, -1, fb, { size: 257 }));
  const atlas = buildSeamAtlas({ ...OPTS, tiles });
  const W = atlas.meta.size;
  // Compare at texel centres, where the atlas value is raw + correction with no interpolation
  for (const [K, R] of [[N + 3, N + 10], [2 * N - 1, N + 30], [2 * N, N + 30], [N + 20, N], [N + 40, N - 1]]) {
    const x = ((K + 0.5) / W - 0.5) * atlas.meta.atlasSpan;
    const z = ((R + 0.5) / W - 0.5) * atlas.meta.atlasSpan;
    const { qi, qj } = tileAt(SPAN, x, z);
    const t = tiles.get(tileKey(qi, qj));
    const raw = t.slot.grid[(R - (qj + 1) * N) * N + (K - (qi + 1) * N)];
    assert.ok(Math.abs(atlas.heights[R * W + K] - raw - seamOffsetAt(atlas, x, z)) < 1e-3, `texel ${K},${R}`);
  }
});

test('the window follows its centre tile', () => {
  const tiles = tileMap(makeTile(2, -1, () => 77), makeTile(3, -1, () => 77));
  const atlas = buildSeamAtlas({ ...OPTS, tiles, centerQi: 2, centerQj: -1 });
  assert.equal(atlas.meta.centerX, 2 * SPAN);
  assert.ok(Math.abs(sampleAtlasHeight(atlas, 2 * SPAN, -SPAN) - 77) < 1e-6);
  assert.ok(Math.abs(sampleAtlasHeight(atlas, 3 * SPAN, -SPAN) - 77) < 1e-6);
  assert.equal(sampleAtlasHeight(atlas, SPAN, -SPAN), -30, 'no tile west of centre yet, so open sea');
  assert.equal(sampleAtlasHeight(atlas, 0, 0), null, 'home is outside this window');
});
