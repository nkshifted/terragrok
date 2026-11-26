# Tile Seam Atlas Implementation Plan

> **Status (2026-11-26):** superseded. Streaming now synthesizes every tile as a window of one
> shared diffusion world and stitches a player-centred 3×3; see "How it works" in the README.
> Kept as a record of the earlier design.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stitch home + ring‑1 walkable neighbors into one blended height/climate atlas so walk and look are continuous across tile edges.

**Architecture:** CPU bake of a 1024‑per‑tile atlas with a 64 m linear blend on shared edges; explorer binds one terrain material to that atlas and samples walk height from it. Neighbor walkable meshes are hidden once the atlas is active.

**Tech Stack:** Existing viewer (`tileStream.js`, `tileAtlas.js`, `main.js`, `terrainShader.js`, `clipmap.js`); Float32/Uint8 DataTextures; no Stage2 changes required for this slice.

**Spec:** `docs/superpowers/specs/2026-11-24-tile-seam-atlas-design.md`

## Global Constraints

- Max window: 3×3 tiles @ 1024 cells/tile
- Blend strip: 64 m (worldSpan 4800 → ~13.65 px at 1024)
- Downsample higher-res tiles with bilinear zoom-equivalent
- Do not block explore on full ring completion
- Keep ghost preview meshes; do not put ghosts in atlas v1

## Review Focus

- Walking exactly on a shared edge uses blended height (not a one-sided cliff)
- Climate snow line does not hard-cut at the joint after stitch
- Atlas rebuild after 1K→2K upgrade still continuous
- Home-only (no neighbors) keeps prior single-tile path (no regression)
- Walk bounds still expand with walkable tiles

---

### Task 1: Pure stitch helpers + tests

**Files:**
- Create: `experiments/seam_atlas.py`
- Create: `tests/test_seam_atlas.py`

- [x] Step 1: Write failing tests for: single tile identity; two tiles with height step → blend strip monotonic; atlas dims `tilesX*N × tilesY*N`
- [x] Step 2: Implement `stitch_heights(tiles, tile_n, world_span_m, blend_m)` and `stitch_climate(...)`
- [x] Step 3: `just test` passes

### Task 2: Viewer seamAtlas module + tileStream integration

**Files:**
- Create: `viewer/src/seamAtlas.js`
- Modify: `viewer/src/tileStream.js`

- [x] Step 1: Port stitch to JS; `buildSeamAtlas(...)`
- [x] Step 2: tileStream keeps `rawHeights` (+ climate/masks); on change notify rebuild
- [x] Step 3: Hide walkable neighbor meshes (ghosts OK)

### Task 3: Bind atlas to explorer terrain + walk

**Files:**
- Modify: `viewer/src/main.js`
- Modify: `viewer/src/terrainShader.js` (`uAtlasSpan` / `uAtlasGrid`)

- [x] Step 1: On atlas rebuild, upload height/climate/masks; expand clipmap
- [x] Step 2: `sampleHeight` uses atlas when present
- [x] Step 3: Home-only: leave existing path (atlas null)

### Task 4: Manual verify checklist

- [ ] Enter streamed world; wait for neighbor 1K; walk across rim — no cliff
- [ ] Diorama over junction — no dark crack / four plates
- [ ] Confirm Stage2 still warm; no server change required

---

## Later (not this plan)

- Spherical cube-face tile chart for whole planet
- Prebake catalog of ~100 planets
- Rivers across seams
