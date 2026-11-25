import {
  buildGhostTileMesh,
  createGhostTileGroup,
  disposeGhostTileGroup,
} from './tileAtlas.js';

/**
 * Poll tile-stream status: show ghosts, upgrade to 1K walkable neighbor heightfields.
 */
export function createTileStreamController({ scene, getVScale, onWalkableChange }) {
  let group = null;
  let pollId = null;
  let homeWorldId = null;
  let worldSpan = 4800;
  /** @type {Map<string, { qi:number, qj:number, quality:string, heights:Float32Array, size:number, mesh?:THREE.Mesh }>} */
  const tiles = new Map();

  function tileKey(qi, qj) {
    return `${qi},${qj}`;
  }

  async function fetchHeightF32(tileId, size, ghost) {
    const url = ghost
      ? `/worlds/${encodeURIComponent(tileId)}.ghost.height.f32`
      : `/worlds/${encodeURIComponent(tileId)}.height.f32`;
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`height HTTP ${resp.status}`);
    const buf = await resp.arrayBuffer();
    const heights = new Float32Array(buf);
    if (heights.length !== size * size) {
      throw new Error(`height size mismatch ${heights.length} vs ${size * size}`);
    }
    return heights;
  }

  function removeMesh(entry) {
    if (entry?.mesh && group) {
      group.remove(entry.mesh);
      entry.mesh.geometry?.dispose();
      entry.mesh.material?.dispose();
      entry.mesh = null;
    }
  }

  async function upsertTile(tile) {
    const tid = tile.tileId;
    if (!tid || tile.qi == null || tile.qj == null) return;
    if (tile.qi === 0 && tile.qj === 0) return;

    const quality = String(tile.quality || 'ghost').toLowerCase();
    const walkable = quality === '1k' || quality === '2k' || quality === '4k' || tile.walkable;
    const key = tileKey(tile.qi, tile.qj);
    const prev = tiles.get(key);
    if (prev && prev.quality === quality && prev.tileId === tid) return;

    const metaResp = await fetch(`/worlds/${encodeURIComponent(tid)}.json`, { cache: 'no-store' });
    if (!metaResp.ok) return;
    const meta = await metaResp.json();
    const size = meta.size || (walkable ? 1024 : 256);
    const span = meta.worldSpan || worldSpan;

    let heights;
    try {
      heights = await fetchHeightF32(tid, size, !walkable && (meta.quality === 'ghost' || quality === 'ghost'));
    } catch {
      if (walkable) {
        // 1K may still be writing; keep ghost if present
        return;
      }
      throw new Error('ghost missing');
    }

    if (!group) {
      group = createGhostTileGroup();
      scene.add(group);
    }
    removeMesh(prev);

    const mesh = buildGhostTileMesh(heights, size, span, tile.qi, tile.qj, getVScale());
    mesh.name = `${walkable ? 'walk' : 'ghost'}:${tid}`;
    if (walkable) {
      mesh.material.color.set('#7a9a6a');
      mesh.material.opacity = 0.92;
      mesh.material.flatShading = false;
    }
    group.add(mesh);

    tiles.set(key, {
      tileId: tid,
      qi: tile.qi,
      qj: tile.qj,
      quality: walkable ? quality : 'ghost',
      walkable: !!walkable,
      heights,
      size,
      worldSpan: span,
      mesh,
    });
    onWalkableChange?.(getWalkableTiles());
  }

  function getWalkableTiles() {
    return [...tiles.values()].filter((t) => t.walkable);
  }

  /** Sample neighbor height in home-local XZ (viewer meters * before vScale). */
  function sampleNeighborRaw(x, z) {
    const span = worldSpan;
    const qi = Math.floor((x + span * 0.5) / span);
    const qj = Math.floor((z + span * 0.5) / span);
    if (qi === 0 && qj === 0) return null;
    const entry = tiles.get(tileKey(qi, qj));
    if (!entry?.walkable || !entry.heights) return null;
    const localX = x - qi * span;
    const localZ = z - qj * span;
    const size = entry.size;
    const u = (localX / span + 0.5) * (size - 1);
    const v = (localZ / span + 0.5) * (size - 1);
    const x0 = Math.min(size - 1, Math.max(0, Math.floor(u)));
    const z0 = Math.min(size - 1, Math.max(0, Math.floor(v)));
    const x1 = Math.min(size - 1, x0 + 1);
    const z1 = Math.min(size - 1, z0 + 1);
    const tx = u - x0;
    const tz = v - z0;
    const h = entry.heights;
    const a = h[z0 * size + x0] * (1 - tx) + h[z0 * size + x1] * tx;
    const b = h[z1 * size + x0] * (1 - tx) + h[z1 * size + x1] * tx;
    return a * (1 - tz) + b * tz;
  }

  function walkBounds() {
    const span = worldSpan;
    let minX = -span * 0.5;
    let maxX = span * 0.5;
    let minZ = -span * 0.5;
    let maxZ = span * 0.5;
    for (const t of getWalkableTiles()) {
      minX = Math.min(minX, (t.qi - 0.5) * span);
      maxX = Math.max(maxX, (t.qi + 0.5) * span);
      minZ = Math.min(minZ, (t.qj - 0.5) * span);
      maxZ = Math.max(maxZ, (t.qj + 0.5) * span);
    }
    const pad = 1;
    return { minX: minX + pad, maxX: maxX - pad, minZ: minZ + pad, maxZ: maxZ - pad };
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
        await upsertTile(tile);
      }
    } catch {
      // Stage 2 offline
    }
  }

  function start(meta) {
    stop();
    if (!meta?.planetOrigin || !meta.id) return;
    homeWorldId = meta.id;
    worldSpan = meta.worldSpan || 4800;
    const o = meta.planetOrigin;
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
    tiles.clear();
    if (group) {
      scene.remove(group);
      disposeGhostTileGroup(group);
      group = null;
    }
    onWalkableChange?.([]);
  }

  function setVScale(_v) {
    tiles.clear();
    if (group) {
      scene.remove(group);
      disposeGhostTileGroup(group);
      group = null;
    }
    pollOnce();
  }

  return {
    start,
    stop,
    setVScale,
    pollOnce,
    sampleNeighborRaw,
    walkBounds,
    getWalkableTiles,
  };
}
