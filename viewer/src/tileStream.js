import {
  buildGhostTileMesh,
  createGhostTileGroup,
  disposeGhostTileGroup,
} from './tileAtlas.js';

/**
 * Poll tile-stream status and attach ghost neighbor meshes around the home world.
 */
export function createTileStreamController({ scene, getVScale }) {
  let group = null;
  let pollId = null;
  let homeWorldId = null;
  let worldSpan = 4800;
  const loaded = new Set();

  async function fetchGhostHeights(tileId, size) {
    const url = `/worlds/${encodeURIComponent(tileId)}.ghost.height.f32`;
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`ghost height HTTP ${resp.status}`);
    const buf = await resp.arrayBuffer();
    const heights = new Float32Array(buf);
    if (heights.length !== size * size) {
      throw new Error(`ghost size mismatch ${heights.length} vs ${size * size}`);
    }
    return heights;
  }

  async function ensureGhostMesh(tile) {
    const tid = tile.tileId;
    if (!tid || loaded.has(tid)) return;
    if (tile.qi == null || tile.qj == null) return;
    // Skip home cell
    if (tile.qi === 0 && tile.qj === 0) return;

    const metaResp = await fetch(`/worlds/${encodeURIComponent(tid)}.json`, { cache: 'no-store' });
    if (!metaResp.ok) return;
    const meta = await metaResp.json();
    if (meta.quality && meta.quality !== 'ghost') {
      // Phase 2+: real tiles; for now still skip non-ghost in this controller
      return;
    }
    const size = meta.size || 256;
    const span = meta.worldSpan || worldSpan;
    const heights = await fetchGhostHeights(tid, size);
    if (!group) {
      group = createGhostTileGroup();
      scene.add(group);
    }
    const mesh = buildGhostTileMesh(heights, size, span, tile.qi, tile.qj, getVScale());
    mesh.name = `ghost:${tid}`;
    group.add(mesh);
    loaded.add(tid);
  }

  async function pollOnce() {
    if (!homeWorldId) return;
    try {
      const resp = await fetch(`/api/tiles/status?homeWorldId=${encodeURIComponent(homeWorldId)}`, {
        cache: 'no-store',
      });
      if (!resp.ok) return;
      const st = await resp.json();
      for (const tile of st.done || []) {
        if ((tile.quality || 'ghost') === 'ghost') {
          await ensureGhostMesh(tile);
        }
      }
    } catch {
      // Stage 2 offline — ignore
    }
  }

  function start(meta) {
    stop();
    if (!meta?.planetOrigin || !meta.id) return;
    homeWorldId = meta.id;
    worldSpan = meta.worldSpan || 4800;
    const o = meta.planetOrigin;
    // Re-announce stream so status has planned neighbors even after server restart
    fetch('/api/tiles/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        homeWorldId,
        seed: o.seed,
        latDeg: o.latDeg,
        lonDeg: o.lonDeg,
        spanDeg: o.spanDeg || 46,
        maxRing: 1,
      }),
    }).catch(() => {});
    pollOnce();
    pollId = window.setInterval(pollOnce, 2500);
  }

  function stop() {
    if (pollId) {
      window.clearInterval(pollId);
      pollId = null;
    }
    homeWorldId = null;
    loaded.clear();
    if (group) {
      scene.remove(group);
      disposeGhostTileGroup(group);
      group = null;
    }
  }

  function setVScale(_v) {
    // Force remesh so ghost heights pick up the new vertical scale
    loaded.clear();
    if (group) {
      scene.remove(group);
      disposeGhostTileGroup(group);
      group = null;
    }
    pollOnce();
  }

  return { start, stop, setVScale, pollOnce };
}
