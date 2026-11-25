# Planet Tile Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Click “Land here” on the planet, explore a home patch immediately, and grow a continuous walkable world as neighbors fill in (ghost → 1K → 2K) until the planet is traversable.

**Architecture:** Invisible rectangular tiles (~46° / 4.8 km) form a tangent grid around the landing point. Stage 2’s single MPS worker drains a priority queue. The explorer hosts a multi-tile height atlas; walkable bounds expand as tiles reach ≥1K. Cyan square UI is removed only after streaming works.

**Tech stack:** Existing Stage 2 server (`experiments/stage2_server.py`), planet conditioning (`viewer/src/planet.js`), explorer clipmap (`viewer/src/main.js`, `clipmap.js`, `terrainShader.js`), synthesized world cache (`experiments/world_cache.py`).

---

## File map

| File | Responsibility |
|------|----------------|
| `experiments/tile_grid.py` | Lat/lon ↔ tile indices, neighbor rings, tile ids, great-circle offsets |
| `experiments/stage2_server.py` | Job queue endpoints: enqueue rings, ghost bake, 1K/2K synth, status |
| `viewer/src/tileStream.js` | Client scheduler: poll status, fetch tile assets, notify atlas |
| `viewer/src/tileAtlas.js` | Multi-tile height/mask textures, world XZ ↔ tile UV, seam blend |
| `viewer/src/main.js` | Growing walk bounds, sampleHeight via atlas, horizon mesh hooks |
| `viewer/src/planet.js` / `planet.html` | Land-here CTA (Phase 4); keep cyan until then |
| `tests/test_tile_grid.py` | Ring order, adjacency, no duplicate tiles |

---

## Phase 1 — Ghost horizon + background queue

*Deliverable: after entering a synthesized world, low-res Stage-1 ghost neighbors appear beyond the home rim while MPS quietly queues real tiles (not walkable yet).*

### Task 1: Tile grid math

**Files:** `experiments/tile_grid.py`, `tests/test_tile_grid.py`

- [x] Add `tile_id(seed, lat, lon, span_deg)` stable string id
- [x] Add `neighbor_centers(lat, lon, span_deg, ring)` → list of `(lat, lon, ring, qi, qj)` in ring-major order (ring1: 8-neigh, ring2: 16, …)
- [x] Use local EN tangent basis at home center; step = `span_deg` in degrees along east/north (good enough for ≤2 rings; document polar caveat)
- [x] Tests: ring0 empty neighbors; ring1 length 8; ids unique; opposite neighbors ≈ `span_deg` apart

### Task 2: Ghost height bake (no MPS)

**Files:** `experiments/stage2_server.py` (or small `experiments/tile_ghost.py`), reuse planet conditioning sample path

- [x] `POST /api/tiles/ghost` accepts client-sampled `elev` (Stage-1 lives in browser) + tile meta
- [x] Write `viewer/public/worlds/<tileId>.ghost.height.f32` + JSON meta (`quality: "ghost"`)
- [x] Register in synth catalog with `tileOrigin: { homeWorldId, qi, qj, ring, quality: "ghost" }`
- [x] Smoke: ghost API + unit test persist height/catalog

### Task 3: Background queue API

**Files:** `experiments/stage2_server.py`

- [x] `POST /api/tiles/stream` plans ring neighbors via `tile_grid` (ghost upload is client-driven in Phase 1)
- [x] `GET /api/tiles/status?homeWorldId=` → `{ pending, active, done }`
- [x] Phase 2: MPS 1K neighbor queue via `/api/tiles/enqueue` (priority 10)
- [x] Ghost path does not use MPS — interactive synthesize unaffected

### Task 4: Explorer ghost rim

**Files:** `viewer/src/tileAtlas.js`, `viewer/src/tileStream.js`, `viewer/src/main.js`

- [x] On `loadWorld`, if `meta.planetOrigin`, call `startTileStream(homeMeta)`
- [x] Poll `/api/tiles/status`; ghost meshes at `(qi * worldSpan, 0, qj * worldSpan)`
- [x] Style ghosts: flat grey/blue Lambert, semi-transparent; walk still home-only
- [x] Planet synth kicks off ring-1 ghost uploads in background
- [ ] Manual check: synthesize → Enter → look past rim for grey ghost continents

**Phase 1 exit criteria:** Horizon no longer drops to void on at least one side; walk still stops at home rim; Stage 2 console shows ghost tile writes.

---

## Phase 2 — 1K walkable merge

*Deliverable: finished 1K neighbors join the walkable atlas; you can cross the old edge.*

### Task 5: Atlas sampler

**Files:** `viewer/src/tileAtlas.js`, `viewer/src/terrainShader.js` (or CPU path first)

- [ ] Atlas maps world XZ → home-relative tile `(qi,qj)` + local uv
- [ ] `sampleHeight(x,z)` checks home then neighbors; blend 32–64 m overlap strip (linear height mix)
- [ ] `walkHalfSpan()` becomes axis-aligned bounds of all tiles with `quality ∈ {1k,2k}` (padding 1 m)
- [ ] Clipmap: either (a) CPU-stitch into one Float32 texture when neighbor commits, or (b) shader multi-texture — prefer **(a) stitch** for v1 (rebuild texture on commit, max ~3×3 @ 1K initially)

### Task 6: Auto-synth 1K neighbors

**Files:** `experiments/stage2_server.py`, planet conditioning extraction shared with `planet.js`

- [ ] Server-side (or client-posted) conditioning for neighbor lat/lon — extract Stage-1 fields without UI
- [ ] Prefer storing a compact planet cache on first Land/synth so neighbors don’t need the browser tab
- [ ] On 1K complete: write normal world assets; client stitches and expands walk

**Phase 2 exit criteria:** Walk from home into a neighbor without falling into void; seam blend acceptable at walking speed.

---

## Phase 3 — 2K upgrade + player priority

### Task 7: Quality ladder

- [ ] Queue policy per tile: ghost → 1k → 2k (skip if user chose 1K-only planet)
- [ ] Re-stitch atlas on 2K replace (same qi,qj slot)
- [ ] Priority = `ring * 100 + distToPlayerTiles` (update every ~2 s from client `POST /api/tiles/priority`)

---

## Phase 4 — Land here (kill cyan UX)

### Task 8: Planet UI

**Files:** `viewer/planet.html`, `viewer/src/planet.js`

- [ ] Replace synthesize/cyan chrome with **Land here** on click
- [ ] Flow: ghost home optional → blocking 1K/2K home synth → open explorer with `?world=&stream=1`
- [ ] Hide cyan target box; footprints optional as subtle “explored” tint
- [ ] Cache manager still lists tiles/worlds

---

## Testing

- Unit: `tile_grid` rings/ids (`just test`)
- Manual: Land → see ghosts → wait for 1K → walk across seam
- Perf: 3×3 @ 1K stitch < 100 ms; MPS queue never blocks UI status polls

## Out of scope (later)

- True spherical walking / polar tile singularities
- Cross-tile hydrology continuity
- Full-planet 2K resident in VRAM
- Circle masks (square tiles only)

---

## Execution note

Implement **Phase 1 Tasks 1–4** first; stop and demo before Phase 2. Do not remove cyan UI until Phase 2 works.
