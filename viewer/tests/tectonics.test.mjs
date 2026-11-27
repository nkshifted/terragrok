import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { GEO_TIME_RANGE_MYR, generateTectonicPlanet, geoTimeToTau } from '../src/tectonics.js';

function fingerprint(planet) {
  const h = createHash('sha1');
  for (const f of planet.faces) {
    h.update(new Uint8Array(f.heights.buffer, f.heights.byteOffset, f.heights.byteLength));
    h.update(f.masks);
    h.update(f.climate);
    h.update(f.plateIds);
  }
  return h.digest('hex').slice(0, 16);
}

const at = (geoTimeMyr, extra = {}) => generateTectonicPlanet({ seed: 7, archetype: 'rifting', faceSize: 64, maxElev: 240, geoTimeMyr, ...extra });

function stats(planet) {
  let peak = -Infinity;
  let high = 0;
  let nan = 0;
  for (const f of planet.faces) {
    for (const h of f.heights) {
      if (!Number.isFinite(h)) nan++;
      if (h > peak) peak = h;
      if (h > 150) high++;
    }
  }
  return { peak, high, nan };
}

// Synthesized worlds regenerate the planet from its seed, so "today" may never change: a
// different fingerprint here means every existing world would grow seams against new tiles.
test('today is exactly the planet the generator always made', () => {
  const cases = [
    [{ seed: 7, archetype: 'rifting', faceSize: 64 }, '947477f32c2b7cfd'],
    [{ seed: 4242, archetype: 'supercontinent', faceSize: 64 }, 'fec1329a391be057'],
    [{ seed: 9, archetype: 'archipelago', faceSize: 64 }, '0b898cd41e804035'],
  ];
  for (const [opts, want] of cases) {
    assert.equal(fingerprint(generateTectonicPlanet({ ...opts, maxElev: 240 })), want);
    assert.equal(fingerprint(generateTectonicPlanet({ ...opts, maxElev: 240, geoTimeMyr: 0 })), want);
  }
});

test('a moment in time is reproducible and differs from today', () => {
  assert.equal(fingerprint(at(60)), fingerprint(at(60)));
  assert.notEqual(fingerprint(at(60)), fingerprint(at(0)));
  assert.notEqual(fingerprint(at(-60)), fingerprint(at(0)));
  assert.equal(at(60).meta.geoTimeMyr, 60);
});

test('every moment in range gives a finite planet with land and sea', () => {
  for (const t of [GEO_TIME_RANGE_MYR[0], -75, -5, 5, 100, GEO_TIME_RANGE_MYR[1]]) {
    const p = at(t);
    assert.equal(stats(p).nan, 0, `NaN at ${t}`);
    assert.ok(p.meta.landPct > 8 && p.meta.landPct < 60, `land ${p.meta.landPct}% at ${t}`);
  }
});

test('time is clamped to the supported range', () => {
  assert.equal(geoTimeToTau(0), 0);
  assert.equal(geoTimeToTau(1e9), geoTimeToTau(GEO_TIME_RANGE_MYR[1]));
  assert.equal(fingerprint(at(-1e9)), fingerprint(at(GEO_TIME_RANGE_MYR[0])));
});

test('a small step in time is a small change', () => {
  const a = at(0);
  const b = at(5);
  let diff = 0;
  let n = 0;
  for (let f = 0; f < 6; f++) {
    for (let i = 0; i < a.faces[f].heights.length; i++) {
      diff += Math.abs(a.faces[f].heights[i] - b.faces[f].heights[i]);
      n++;
    }
  }
  assert.ok(diff / n < 12, `mean change ${(diff / n).toFixed(1)} m over 5 Myr`);
});

test('mountains are absent at the start and build up going forward', () => {
  const start = stats(at(GEO_TIME_RANGE_MYR[0]));
  const today = stats(at(0));
  const later = stats(at(150));
  assert.ok(start.high < today.high * 0.6, `high ground ${start.high} at the start vs ${today.high} today`);
  assert.ok(later.high > today.high, `high ground ${later.high} later vs ${today.high} today`);
});

// ---------- Rock types and volcanoes ----------
import {
  chartBasis,
  chartTileVolcanoes,
  chartToUnitDir,
  encodeChartTilePayload,
  extractChartTile,
  generateStage1Planet,
  unitDirToChart,
} from '../src/stage1Conditioning.js';

test('every face carries a rock map whose weights never exceed the regional cap', () => {
  const p = at(0);
  const seen = [0, 0, 0, 0];
  for (const f of p.faces) {
    assert.equal(f.rock.length, f.heights.length * 4);
    for (let i = 0; i < f.heights.length; i++) {
      let sum = 0;
      for (let c = 0; c < 4; c++) {
        sum += f.rock[i * 4 + c];
        if (f.rock[i * 4 + c] > 60) seen[c]++;
      }
      assert.ok(sum <= 0.8 * 255 + 4, `rock weights sum to ${sum}`);
    }
  }
  // All four rock types occur somewhere on the planet
  assert.ok(seen.every((n) => n > 20), `granite/volcanic/limestone/metamorphic cells: ${seen}`);
});

test('volcanoes stand on land, apart from each other, and are the same every time', () => {
  const p = generateStage1Planet({ seed: 101, archetype: 'earth' });
  const q = generateStage1Planet({ seed: 101, archetype: 'earth' });
  assert.deepEqual(p.volcanoes, q.volcanoes);
  // A small planet has a handful of obvious volcanoes, not a rash of them
  assert.ok(p.volcanoes.length >= 3 && p.volcanoes.length <= 8, `${p.volcanoes.length} volcanoes`);
  assert.equal(p.meta.volcanoCount, p.volcanoes.length);
  const kinds = new Set(p.volcanoes.map((v) => v.kind));
  assert.ok(kinds.has('strato'));
  for (let i = 0; i < p.volcanoes.length; i++) {
    for (let j = i + 1; j < p.volcanoes.length; j++) {
      const a = p.volcanoes[i];
      const b = p.volcanoes[j];
      const ang = Math.acos(Math.min(1, a.dir[0] * b.dir[0] + a.dir[1] * b.dir[1] + a.dir[2] * b.dir[2]));
      assert.ok(ang > 0.4, `${a.kind} and ${b.kind} only ${ang.toFixed(3)} rad apart`);
    }
  }
});

test('the chart inverse undoes chartToUnitDir', () => {
  const basis = chartBasis(31.5, -47.25);
  for (const [tx, ty] of [[0.3, -0.2], [-0.7, 0.45], [0.001, 0.0], [1.1, 0.9]]) {
    const d = chartToUnitDir(basis, tx, ty);
    const [bx, by] = unitDirToChart(basis, d[0], d[1], d[2]);
    assert.ok(Math.abs(bx - tx) < 1e-6 && Math.abs(by - ty) < 1e-6, `${tx},${ty} -> ${bx},${by}`);
  }
});

test('one big volcano per planet is a hot spring and does not erupt', () => {
  for (const [seed, archetype] of [[7, 'rifting'], [4242, 'supercontinent'], [9, 'archipelago']]) {
    const p = generateTectonicPlanet({ seed, archetype, faceSize: 64 });
    const springs = p.volcanoes.filter((v) => v.hotSpring);
    assert.equal(springs.length, 1, `seed ${seed}`);
    assert.ok(springs[0].kind !== 'cinder' && springs[0].active === false);
  }
});

test('neighbouring tiles agree on where a shared volcano stands', () => {
  const p = generateStage1Planet({ seed: 101, archetype: 'earth' });
  const v = p.volcanoes.find((x) => x.kind === 'strato');
  const lat = (Math.asin(v.dir[1]) * 180) / Math.PI;
  const lon = (Math.atan2(v.dir[0], v.dir[2]) * 180) / Math.PI;
  // Home placed so the volcano sits on the seam between tile (0,0) and tile (1,0)
  const home = chartBasis(lat, lon);
  const west = chartToUnitDir(home, (-23 * Math.PI) / 180, 0);
  const chart = {
    homeLatDeg: (Math.asin(west[1]) * 180) / Math.PI,
    homeLonDeg: (Math.atan2(west[0], west[2]) * 180) / Math.PI,
    spanDeg: 46,
  };
  const a = chartTileVolcanoes(p, { ...chart, qi: 0, qj: 0 }).find((x) => x.id === v.id);
  const b = chartTileVolcanoes(p, { ...chart, qi: 1, qj: 0 }).find((x) => x.id === v.id);
  assert.ok(a && b, 'both tiles list it');
  assert.ok(Math.abs(a.x - 0.5) < 0.02, `on the east edge of the home tile: x=${a.x}`);
  assert.ok(Math.abs(a.x - b.x - 1) < 1e-4 && Math.abs(a.z - b.z) < 1e-4);
  assert.equal(a.baseDem, b.baseDem);
  assert.equal(a.radius, b.radius);
});

test('a tile payload carries its rock raster and volcano list', () => {
  const p = generateStage1Planet({ seed: 101, archetype: 'earth' });
  const site = p.volcanoes.find((x) => x.kind === 'strato').dir;
  const homeLatDeg = Number(((Math.asin(site[1]) * 180) / Math.PI).toFixed(2));
  const homeLonDeg = Number(((Math.atan2(site[0], site[2]) * 180) / Math.PI).toFixed(2));
  const tile = extractChartTile(p, { homeLatDeg, homeLonDeg, spanDeg: 46, coarseSize: 8, margin: true });
  const full = tile.refSize + 2 * tile.refPad;
  assert.equal(tile.ref.rock.length, full * full * 4);
  const payload = encodeChartTilePayload(tile);
  assert.equal(Buffer.from(payload.refRock, 'base64').length, full * full * 4);
  assert.ok(Array.isArray(payload.volcanoes) && payload.volcanoes.length > 0);
  for (const v of payload.volcanoes) assert.ok(v.baseDem > 0 && v.radius > 0);
});
