import { prepareTileSlot, tileAt, tileKey } from './seamAtlas.js';

const HEARTBEAT_MS = 2000;
const ATLAS_TILE_N = 1024;
const ATLAS_CLIM_N = 512;
/** Recentre the 3×3 window once the player is this far (in tiles) past the centre tile's middle. */
const RECENTRE_AT = 0.62;
/** Keep loaded tiles this many rings around the window centre; drop the rest. */
const KEEP_RING = 2;

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
  /** @type {Map<string, object>} */
  const tiles = new Map();

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
      if (recentre(pos)) changed = true;

      for (const t of st.tiles || []) {
        if (t.qi === 0 && t.qj === 0) continue;
        if (Math.max(Math.abs(t.qi - centre.qi), Math.abs(t.qj - centre.qj)) > 1) continue;
        const have = tiles.get(tileKey(t.qi, t.qj));
        if (have && have.tileId === t.tileId && have.rev === t.rev) continue;
        try {
          const entry = await loadTile(t);
          if (gen !== generation) return;
          tiles.set(entry.key, entry);
          changed = true;
        } catch (err) {
          // Usually a tile that is mid-write; the next heartbeat retries
          console.warn('[tile-stream] load failed', t.tileId, err.message);
        }
      }
      for (const [key, e] of tiles) {
        if (key === tileKey(0, 0)) continue;
        if (Math.max(Math.abs(e.qi - centre.qi), Math.abs(e.qj - centre.qj)) > KEEP_RING) {
          tiles.delete(key);
        }
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
    tiles.set(tileKey(0, 0), {
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

  function isWalkable(x, z) {
    const { qi, qj } = tileAt(worldSpan, x, z);
    if (Math.max(Math.abs(qi - centre.qi), Math.abs(qj - centre.qj)) > 1) return false;
    return Boolean(tiles.get(tileKey(qi, qj))?.walkable);
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

  return { start, stop, isLive, getWindow, isWalkable, describe, tileN: ATLAS_TILE_N, climN: ATLAS_CLIM_N };
}
