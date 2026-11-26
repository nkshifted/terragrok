import { prepareTileSlot, tileAt, tileKey } from './seamAtlas.js';

const HEARTBEAT_MS = 2000;
const ATLAS_TILE_N = 1024;
const ATLAS_CLIM_N = 512;
/** Recentre the 3×3 window once the player is this far (in tiles) past the centre tile's middle. */
const RECENTRE_AT = 0.62;
/** Keep loaded tiles this many rings around the window centre; drop the rest. */
const KEEP_RING = 2;
/**
 * Beyond the stitched 3×3, tiles stay on screen as cheap low-detail "far" copies out to this
 * many rings, so terrain you walked through (and ghosts of what lies ahead) does not vanish.
 * A full tile costs ~9 MB here plus its share of the atlas; a far copy costs ~0.6 MB.
 */
const FAR_RING = 3;
const FAR_N = 256;
const FAR_CLIM_N = 128;

/** Downsample a prepared slot (see seamAtlas.prepareTileSlot) into a far copy's rasters. */
function makeFarCopy(entry) {
  const { slot } = entry;
  const n = slot.tileN;
  const f = n / FAR_N;
  const heights = new Float32Array(FAR_N * FAR_N);
  for (let r = 0; r < FAR_N; r++) {
    for (let c = 0; c < FAR_N; c++) {
      let acc = 0;
      for (let dr = 0; dr < f; dr++) {
        const row = (r * f + dr) * n + c * f;
        for (let dc = 0; dc < f; dc++) acc += slot.grid[row + dc];
      }
      heights[r * FAR_N + c] = acc / (f * f);
    }
  }
  let masks = null;
  if (slot.masks) {
    masks = new Uint8Array(FAR_N * FAR_N * 4);
    const half = Math.floor(f / 2);
    for (let r = 0; r < FAR_N; r++) {
      for (let c = 0; c < FAR_N; c++) {
        const si = ((r * f + half) * n + c * f + half) * 4;
        masks.set(slot.masks.subarray(si, si + 4), (r * FAR_N + c) * 4);
      }
    }
  }
  let climate = null;
  if (slot.climate) {
    const cn = slot.climN;
    const cf = cn / FAR_CLIM_N;
    climate = new Uint8Array(FAR_CLIM_N * FAR_CLIM_N * 4);
    for (let r = 0; r < FAR_CLIM_N; r++) {
      for (let c = 0; c < FAR_CLIM_N; c++) {
        const si = (r * cf * cn + c * cf) * 4;
        climate.set(slot.climate.subarray(si, si + 4), (r * FAR_CLIM_N + c) * 4);
      }
    }
  }
  return {
    key: entry.key,
    qi: entry.qi,
    qj: entry.qj,
    tileId: entry.tileId,
    rev: entry.rev,
    ghost: entry.ghost,
    walkable: entry.walkable,
    meta: entry.meta,
    lakes: entry.lakes,
    version: entry.version,
    size: FAR_N,
    climateSize: FAR_CLIM_N,
    heights,
    masks,
    climate,
  };
}

/**
 * Tile stream client.
 *
 * Posts the player's position to the Stage 2 server every couple of seconds. The server keeps
 * synthesizing the tiles nearest the player (experiments/tile_stream.py) and answers with what
 * exists; this module loads those tiles and maintains the 3×3 window the explorer stitches.
 *
 * @param {{ onChange: () => void, getPlayerPos: () => {x: number, z: number} }} deps
 */
export function createTileStreamController({ onChange, getPlayerPos }) {
  let homeWorldId = null;
  let worldSpan = 4800;
  let timer = null;
  let busy = false;
  let generation = 0;
  let live = false;
  let centre = { qi: 0, qj: 0 };
  let lastStatus = null;
  let version = 0;
  /** @type {Map<string, object>} full tiles near the window (stitched into the atlas) */
  const tiles = new Map();
  /** @type {Map<string, object>} low-detail copies of every tile seen, drawn outside the window */
  const far = new Map();

  function ringFromCentre(qi, qj) {
    return Math.max(Math.abs(qi - centre.qi), Math.abs(qj - centre.qj));
  }

  function remember(entry) {
    tiles.set(entry.key, entry);
    far.set(entry.key, makeFarCopy(entry));
  }

  async function fetchBytes(url, Type, expected) {
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
    const arr = new Type(await resp.arrayBuffer());
    if (expected != null && arr.length !== expected) {
      throw new Error(`${url}: expected ${expected} values, got ${arr.length}`);
    }
    return arr;
  }

  async function loadTile(st) {
    const base = `/worlds/${encodeURIComponent(st.tileId)}`;
    const meta = await fetch(`${base}.json`, { cache: 'no-store' }).then((r) => {
      if (!r.ok) throw new Error(`${base}.json: HTTP ${r.status}`);
      return r.json();
    });
    const ghost = meta.quality === 'ghost';
    const size = meta.size;
    const entry = {
      key: tileKey(st.qi, st.qj),
      qi: st.qi,
      qj: st.qj,
      tileId: st.tileId,
      rev: st.rev,
      quality: ghost ? 'ghost' : st.quality,
      ghost,
      walkable: !ghost,
      meta,
      rivers: [],
      lakes: [],
      version: ++version,
    };
    if (ghost) {
      const heights = await fetchBytes(`${base}.ghost.height.f32`, Float32Array, size * size);
      entry.slot = prepareTileSlot({ heights, size }, ATLAS_TILE_N, ATLAS_CLIM_N);
      return entry;
    }
    const climateSize = meta.climateSize || size;
    const [heights, masks, climate, rivers, lakes] = await Promise.all([
      fetchBytes(`${base}.height.f32`, Float32Array, size * size),
      fetchBytes(`${base}.masks.u8`, Uint8Array, size * size * 4),
      meta.hasClimate ? fetchBytes(`${base}.climate.u8`, Uint8Array, climateSize * climateSize * 4) : null,
      fetch(`${base}.rivers.json`, { cache: 'no-store' }).then((r) => (r.ok ? r.json() : [])),
      fetch(`${base}.lakes.json`, { cache: 'no-store' }).then((r) => (r.ok ? r.json() : [])),
    ]);
    entry.slot = prepareTileSlot({ heights, size, masks, climate, climateSize }, ATLAS_TILE_N, ATLAS_CLIM_N);
    entry.rivers = rivers;
    entry.lakes = lakes;
    return entry;
  }

  function recentre(pos) {
    const fx = pos.x / worldSpan - centre.qi;
    const fz = pos.z / worldSpan - centre.qj;
    const next = { ...centre };
    if (Math.abs(fx) > RECENTRE_AT) next.qi = Math.round(pos.x / worldSpan);
    if (Math.abs(fz) > RECENTRE_AT) next.qj = Math.round(pos.z / worldSpan);
    if (next.qi === centre.qi && next.qj === centre.qj) return false;
    centre = next;
    return true;
  }

  async function beat() {
    if (!homeWorldId || busy) return;
    busy = true;
    const gen = generation;
    let changed = false;
    try {
      const pos = getPlayerPos();
      const resp = await fetch('/api/tiles/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ homeWorldId, x: pos.x, z: pos.z }),
      });
      const st = await resp.json().catch(() => null);
      if (gen !== generation) return;
      if (!resp.ok || !st?.ok) {
        // 400 / 404: this world cannot be tiled (not planet-synthesized, or deleted) - stop asking.
        // Anything else (server restarting, dev proxy 5xx) is transient: keep the heartbeat going.
        if (resp.status === 400 || resp.status === 404) stopTimer();
        return;
      }
      lastStatus = st;
      if (!live) {
        live = true;
        changed = true;
      }
      if (recentre(pos)) {
        // Show the shifted window at once with what is already in memory; loads follow below
        changed = true;
        onChange();
      }

      // Window tiles first (they are walkable terrain), nearest first
      const wanted = (st.tiles || []).filter((t) => !(t.qi === 0 && t.qj === 0));
      for (const t of wanted) {
        if (ringFromCentre(t.qi, t.qj) > 1) continue;
        const have = tiles.get(tileKey(t.qi, t.qj));
        if (have && have.tileId === t.tileId && have.rev === t.rev) continue;
        try {
          const entry = await loadTile(t);
          if (gen !== generation) return;
          remember(entry);
          changed = true;
        } catch (err) {
          // Usually a tile that is mid-write; the next heartbeat retries
          console.warn('[tile-stream] load failed', t.tileId, err.message);
        }
      }

      // Then page in far copies of tiles outside the window: a few per heartbeat, nearest
      // first, each loaded once, downsampled and released
      const missingFar = wanted
        .filter((t) => {
          const ring = ringFromCentre(t.qi, t.qj);
          if (ring <= 1 || ring > FAR_RING) return false;
          const have = far.get(tileKey(t.qi, t.qj));
          return !(have && have.tileId === t.tileId && have.rev === t.rev);
        })
        .sort((a, b) => a.dist - b.dist)
        .slice(0, 3);
      for (const t of missingFar) {
        try {
          const entry = await loadTile(t);
          if (gen !== generation) return;
          far.set(entry.key, makeFarCopy(entry));
          changed = true;
        } catch (err) {
          console.warn('[tile-stream] far load failed', t.tileId, err.message);
        }
      }

      for (const [key, e] of tiles) {
        if (key === tileKey(0, 0)) continue;
        if (ringFromCentre(e.qi, e.qj) > KEEP_RING) tiles.delete(key);
      }
      for (const [key, e] of far) {
        // Home is not in the server's tile list, so its copy could never be paged back in
        if (key === tileKey(0, 0)) continue;
        if (ringFromCentre(e.qi, e.qj) > FAR_RING + 1) far.delete(key);
      }
    } catch {
      // Stage 2 server offline: the explorer keeps working on the home tile alone
    } finally {
      busy = false;
      if (changed && gen === generation) onChange();
    }
  }

  function stopTimer() {
    if (timer) window.clearInterval(timer);
    timer = null;
  }

  /**
   * @param {object} homeMeta  the home world's meta JSON
   * @param {{heights: Float32Array, masks: Uint8Array, climate: Uint8Array|null, rivers: object[], lakes: object[]}} home
   */
  function start(homeMeta, home) {
    stop();
    if (!homeMeta?.planetOrigin || homeMeta.drillOrigin || homeMeta.tileOrigin) return;
    homeWorldId = homeMeta.id;
    worldSpan = homeMeta.worldSpan || 4800;
    const size = homeMeta.size;
    remember({
      key: tileKey(0, 0),
      qi: 0,
      qj: 0,
      tileId: homeMeta.id,
      rev: 0,
      quality: String(homeMeta.resolution || '2k').toLowerCase(),
      ghost: false,
      walkable: true,
      meta: homeMeta,
      rivers: home.rivers,
      lakes: home.lakes,
      version: ++version,
      slot: prepareTileSlot(
        { heights: home.heights, size, masks: home.masks, climate: home.climate, climateSize: homeMeta.climateSize || size },
        ATLAS_TILE_N,
        ATLAS_CLIM_N
      ),
    });
    beat();
    timer = window.setInterval(beat, HEARTBEAT_MS);
  }

  function stop() {
    stopTimer();
    generation++;
    homeWorldId = null;
    live = false;
    busy = false;
    centre = { qi: 0, qj: 0 };
    lastStatus = null;
    tiles.clear();
    far.clear();
  }

  /** True once the server has confirmed a stream for this world. */
  function isLive() {
    return live;
  }

  /** The 3×3 window to stitch: centre tile + every loaded tile within one ring of it. */
  function getWindow() {
    const win = new Map();
    for (const [key, e] of tiles) {
      if (Math.max(Math.abs(e.qi - centre.qi), Math.abs(e.qj - centre.qj)) <= 1) win.set(key, e);
    }
    return { centerQi: centre.qi, centerQj: centre.qj, tiles: win };
  }

  /** Far copies of every known tile that is not (yet) part of the stitched window. */
  function getFarTiles() {
    const out = [];
    for (const e of far.values()) {
      // ...plus window tiles whose full data is still loading (just after a teleport)
      if (ringFromCentre(e.qi, e.qj) > 1 || !tiles.has(e.key)) out.push(e);
    }
    return out;
  }

  /** Real (synthesized) terrain here? True for window tiles and for far tiles alike. */
  function isWalkable(x, z) {
    const { qi, qj } = tileAt(worldSpan, x, z);
    const key = tileKey(qi, qj);
    if (ringFromCentre(qi, qj) <= 1) return Boolean(tiles.get(key)?.walkable);
    return Boolean(far.get(key)?.walkable);
  }

  /** Low-detail height (raw metres) from a far copy; null where nothing is known. */
  function sampleFarHeight(x, z) {
    const { qi, qj } = tileAt(worldSpan, x, z);
    const e = far.get(tileKey(qi, qj));
    if (!e) return null;
    const n = e.size;
    const fx = Math.min(n - 1, Math.max(0, (x / worldSpan + 0.5 - qi) * n - 0.5));
    const fz = Math.min(n - 1, Math.max(0, (z / worldSpan + 0.5 - qj) * n - 0.5));
    const x0 = Math.floor(fx);
    const z0 = Math.floor(fz);
    const x1 = Math.min(n - 1, x0 + 1);
    const z1 = Math.min(n - 1, z0 + 1);
    const tx = fx - x0;
    const tz = fz - z0;
    const h = e.heights;
    const a = h[z0 * n + x0] * (1 - tx) + h[z0 * n + x1] * tx;
    const b = h[z1 * n + x0] * (1 - tx) + h[z1 * n + x1] * tx;
    return a * (1 - tz) + b * tz;
  }

  /** Run a heartbeat now (after a teleport) instead of waiting for the next tick. */
  function poke() {
    beat();
  }

  /** One-line progress for the HUD, or '' when there is nothing to say. */
  function describe() {
    if (!live || !lastStatus) return '';
    const pos = getPlayerPos();
    const p = tileAt(worldSpan, pos.x, pos.z);
    let real = 0;
    let total = 0;
    for (let dj = -1; dj <= 1; dj++) {
      for (let di = -1; di <= 1; di++) {
        total++;
        if (tiles.get(tileKey(p.qi + di, p.qj + dj))?.walkable) real++;
      }
    }
    const a = lastStatus.active;
    const where = p.qi === 0 && p.qj === 0 ? 'home tile' : `tile ${p.qi},${p.qj}`;
    const work = a ? ` · synthesizing ${a.qi},${a.qj}` : real < total ? ' · queued' : '';
    return `${where} · ${real}/${total} nearby tiles ready${work}`;
  }

  return {
    start,
    stop,
    poke,
    isLive,
    getWindow,
    getFarTiles,
    isWalkable,
    sampleFarHeight,
    describe,
    tileN: ATLAS_TILE_N,
    climN: ATLAS_CLIM_N,
    farRing: FAR_RING,
  };
}
