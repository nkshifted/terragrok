# Tile Seam Atlas (Approach B) — Design

> **Status (2026-11-26):** superseded. Streaming now synthesizes every tile as a window of one
> shared diffusion world and stitches a player-centred 3×3; see "How it works" in the README.
> Kept as a record of the earlier design.

**Intent:** Walking and looking across tile edges must not cliff, crack, or knife climate — starting with today’s ring‑1 (3×3), before whole‑planet tiling.

**Why:** Progressive synth (ghost → 1K → 2K) already fills neighbors; separate meshes show hard seams. Fix look + walk first; underlying per‑tile disagreement is OK if the presented surface is continuous.

## Goals

- One continuous height (+ climate cues) surface for all walkable tiles in the loaded window
- Walk sampling uses the same surface (no invisible walls / height pops at borders)
- Rebuild on each tile commit / quality upgrade in &lt;100 ms for 3×3 @ 1K
- Works as soon as a neighbor reaches 1K; 2K upgrades re-stitch the same slot

## Non-goals (this slice)

- Whole-planet spherical chart / &gt;1 ring
- Cross-tile hydrology / rivers spanning seams
- Holding full-planet 2K in VRAM
- Approach A per-mesh feathering (superseded here)

## Anti-goals

- Visible tile grid in diorama or walk mode after neighbors are walkable
- Blocking explore until the whole ring is done

## Approaches considered

| | Idea | Tradeoff |
|---|------|----------|
| A | Soften joints on separate meshes | Faster, mushy; thrown away if we do B |
| **B (chosen)** | CPU-stitch one atlas + one terrain draw | Clean seams; rebuild hitch on commit |
| C | Constrain Stage2 borders | Real data match; slow; doesn’t help ghosts |

## Design (builder detail)

1. **Tile slots:** Home `(0,0)` + walkable neighbors from tile stream. Ghosts stay as optional separate preview (non-walkable); not required in atlas for v1.
2. **Atlas grid:** Axis-aligned union of walkable `(qi,qj)`. Each slot contributes a **1024²** height cell (downsample 2K/4K, upsample ghost/low if ever included). Max window **3×3 @ 1K** this slice.
3. **Edge blend:** On bake, for each atlas texel within `blendMeters` (64 m default) of a shared edge, linear-mix height (and climate RGBA) with the neighbor slot’s sample. Interior of a tile stays that tile’s data.
4. **Render:** One clipmap/terrain material bound to atlas textures; world origin = atlas center (or home). Span = `tilesX * worldSpan`. Hide/remove per-neighbor walkable meshes once atlas is active.
5. **Walk:** `sampleHeight` / walk bounds read atlas (raw × vScale). Same blend as visuals.
6. **Update:** On `onWalkableChange` / quality upgrade → rebuild atlas → upload textures → expand clipmap if AABB grew.

## Calls (ours)

- Stitch resolution fixed at 1024/tile for v1 even if home is 2K (continuity &gt; native home res when multi-tile).
- Climate blended in the same strip as height.
- Masks (rivers/lakes) nearest-tile, not blended (avoids ghost rivers in the strip); optional later.

## Success

Enter a streamed world, wait for ≥1 neighbor 1K, walk across the old rim: no cliff, no dark crack, snow/rock line softens across the joint. Diorama over the 2×2 junction looks like one landscape, not four plates.
