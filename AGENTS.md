# Notes for agents working in this repo

Read [README.md](README.md) first for what the project is and how the pieces fit. This file is
the part that is easy to get wrong.

## Run and verify

```bash
just test
```

Runs pytest and `node --test` in a few seconds; neither loads the model. Run it before saying a
change works.

```bash
just dev
```

Starts the Stage 2 server (:8765) and the viewer (:5173). `.claude/launch.json` defines the
viewer for preview tooling. To exercise streaming end to end you need a synthesized home world,
which means the model: Apple Silicon, ~30 s per 2K tile. `window.__terragrok` in the explorer
exposes `relocateExplorer(x, z)`, `focusDiorama(x, z, dist)`, `setCameraMode(mode)`,
`summonVehicle(kind)` and `getVehicle()` for driving it from a console.

A browser tab that is hidden has its timers throttled, so the explorer's heartbeat slows to
about one a minute. If tiles stop synthesizing while you test headlessly, post the heartbeat
yourself:

```bash
curl -s -X POST localhost:8765/api/tiles/stream -H 'Content-Type: application/json' -d '{"homeWorldId":"<id>","x":0,"z":0}'
```

## Conventions that are not obvious from one file

- **Tile indices.** Tile `(qi, qj)` is `qi` tiles east and `qj` tiles **south** of home. `qj`
  follows viewer +Z and heightfield rows, not latitude.
- **Raster registration.** A tile's own rasters are node-registered (first and last samples sit
  on the tile edges). The seam atlas is cell-registered. `seamAtlas.js` documents both; the
  shaders switch with `uCellReg`. Mixing them up shifts rivers a few metres off their channels.
  The terrain mesh puts its vertices on the samples (a half-cell shift when `uCellReg` is set):
  without it the drawn ground is a blurred copy and the player floats above sharp ridges.
- **Vertical units.** Heights on disk are "viewer metres" (DEM metres × the chart's `vScale`,
  soft-kneed above 330 m). Shaders and meshes multiply by the UI's vertical-scale slider.
- **One MPS thread.** All model work runs on the server's single worker thread; Metal crashes
  when called from several. Do not call the pipeline from request handlers.
- **Catalog writes** go through `CATALOG_LOCK` in `world_cache.py`; the worker, HTTP and ghost
  threads all touch `synthesized_catalog.json`.

## What must stay identical across the tiles of a chart

Seams are invisible only because every tile of a chart is the same world. Changing any of
these per tile brings the cliffs back:

- the projection origin and the planet settings (`chart_from_home_meta`)
- the diffusion seed (`chart_seed`, from the **home** lat/lon, not the tile's)
- resolution, `quality` and `fidelity` (1K and 2K are different model scales, not a ladder)
- `vScale` (`dem_to_viewer`)
- world-anchored procedural detail (`origin_px` in the refinement functions)
- climate: `blend_climate` and `pack_climate` in `chart.py` are per-pixel with fixed ranges.
  Nothing may be centred or scaled by a statistic of the tile (a mean, a percentile); that
  put a straight climate line on every seam. If the rasters change meaning, bump
  `CLIMATE_VERSION`: older tiles are then recomputed by `_refresh_tile_climate` on first use

Post-processing that looks at a neighbourhood (blurs, relaxation) runs on a 64-pixel margin
that is cropped afterwards. New filters belong before that crop in `_synthesize_on_worker`.

## Shared code

- `viewer/src/stage1Conditioning.js` and `tectonics.js` run in the browser **and** under node
  (`experiments/stage1_patch.mjs`). Keep them free of three.js and DOM APIs.
- `viewer/src/seamAtlas.js` is pure for the same reason: it is unit-tested under node.
- `viewer/src/vehiclePhysics.js` likewise: keep meshes and anything three.js in `vehicles.js`.

## Housekeeping

- `viewer/public/worlds/planet_*`, `tile_*` and `synthesized_catalog.json` are generated and
  gitignored. Do not commit them. The five demo worlds and `catalog.json` are tracked.
- `.vendor/` and `out/` are gitignored. Use `paths.py` (`VENDOR_TD`, `REPO_ROOT`, `OUT_DIR`)
  rather than building paths by hand or hard-coding absolute ones; worktrees depend on it.
- `docs/superpowers/` holds earlier plans and specs. They record decisions at the time and
  are not kept in sync with the code; the README is.
