# TerraGrok

A testbed for procedural worlds you can walk around in. A tectonic planet is simulated in the
browser; any patch of it can be turned into a 4.8 km world of diffusion-synthesized terrain with
physically solved rivers and lakes, and that world keeps growing in every direction as you walk.

## Quick start

Requires [mise](https://mise.jdx.dev) and, for terrain synthesis, an Apple Silicon Mac (the model
runs on MPS; CUDA and CPU also work, CPU slowly).

```bash
mise trust && just setup
```

```bash
just dev
```

`just setup` installs the pinned toolchain, Python and viewer dependencies, and clones the
[Terrain Diffusion](https://github.com/xandergos/terrain-diffusion) model code into `.vendor/`.
`just dev` starts the synthesis server and the viewer; the first run downloads the
`xandergos/terrain-diffusion-30m` weights from Hugging Face.

Then open <http://127.0.0.1:5173/planet.html>:

1. Click the globe to place a target square and synthesize it (15 s to a minute, depending
   on the resolution you pick).
2. Open the finished patch in the explorer and walk (WASD, Shift to sprint).
3. Neighbouring terrain appears hazy at first (a low-detail "ghost" of the planet) and sharpens
   as each tile is synthesized, roughly 30 s per tile. Keep walking: tiles are always
   generated around wherever you are. The subtitle line shows progress.

The viewer alone (`just viewer`) works without the server, on the five bundled demo worlds.

## How it works

**Stage 1: the planet.** [`viewer/src/tectonics.js`](viewer/src/tectonics.js) simulates plate
tectonics, climate and coarse hydrology directly on the sphere. It is pure JavaScript, so the
same code runs in the globe page and, under node, on the server.

**Stage 2: a patch.** The globe page samples elevation and climate around the target and posts
them to [`experiments/stage2_server.py`](experiments/stage2_server.py). The server conditions the
diffusion model on that patch, locks coastlines to the planet, relaxes cliffs, adds biome detail
and solves hydrology with Landlab's priority-flood router, so every river ends at the sea, a lake,
a confluence or the tile edge. The result is written to `viewer/public/worlds/` (gitignored).

**Streaming: the patch grows.** The home patch defines a *chart*: a flat projection of the planet
centred on it. Each further tile is the same chart shifted by whole tiles, synthesized as a
shifted window of the *same* diffusion world with the same seed, settings and vertical scale.
Because the model seeds its noise by world coordinate, adjacent tiles agree at their shared edge
to within about a metre on gentle ground and a few metres in mountains. Rivers that leave a tile
through an edge are injected as inflow into the tile beyond when it is synthesized. Climate is blended
per pixel from the planet's and the model's, so biomes and the snow line run straight through
seams; tiles from before that was true have their climate recomputed (about 2 s each, no
re-synthesis) the first time their world is opened.

- [`experiments/tile_stream.py`](experiments/tile_stream.py) plans what to synthesize: the
  explorer posts the player's position every two seconds and the server works on the nearest
  missing tile in the 3×3 around the player. Nothing is queued ahead, and synthesis pauses when
  no explorer tab has checked in for a couple of minutes.
- [`viewer/src/seamAtlas.js`](viewer/src/seamAtlas.js) stitches the 3×3 tiles around the player
  into one height / mask / climate atlas and removes what mismatch remains: the smooth part is
  spread over a ~1.3 km band, fine detail over ~50 m. Rivers, lakes, trees and rocks are built per
  tile on that stitched surface.
- Tiles outside that 3×3 are not dropped from view. Each one seen stays on screen as a 256²
  "far" copy (terrain, lakes and a thinned scatter of trees) out to three rings from the player,
  so the isometric view shows where you have been. Stepping or teleporting onto a far tile
  recentres the 3×3 and swaps in its full data.

**The explorer.** [`viewer/src/main.js`](viewer/src/main.js) renders with a geometry clipmap, a
custom terrain shader, river ribbons, lake surfaces, instanced grass and props.

**The explorer's body.** [`viewer/src/character.js`](viewer/src/character.js) loads a rigged
glTF character and plays walk / jog / sprint / swim / jump clips to match what the player is
doing; `C` (or the Explorer buttons) switches between four presets. Each preset is an outfit
(a headless clothed body) with a head, skin tone and hair attached to its skeleton. The
models and animations are Quaternius's CC0 kits, packed into `viewer/public/characters/` by
`experiments/pack_characters.py`.

**Vehicles.** `B`, `K`, `R`, `J` and `G` summon a boat, canoe, horse, jeep or glider and put the
explorer on it; `E` gets off, or back on when standing next to it. W/S is speed; steer with A/D,
or hold W and turn the camera, as on foot.
A boat stays on lakes and the sea (`B` again switches between speedboat and sailboat), while
the canoe also follows rivers: it is slim, turns with the bends by itself and scrapes over
small bars and reeds. The horse climbs slopes up to about 60° and the jeep up to 50°,
both losing speed as it steepens; Space makes either jump, and both leave the ground by
themselves over a crest taken at speed; one that comes down in deep water is lost and leaves
its rider swimming. A glider launches with a kick, sinks slowly, dives with
W, flares with S and climbs while Space is held.
[`viewer/src/vehiclePhysics.js`](viewer/src/vehiclePhysics.js) holds those rules (pure, unit
tested); [`viewer/src/vehicles.js`](viewer/src/vehicles.js) holds the meshes. The horse
(Quaternius, with walk and gallop clips), the SUV that stands in for a jeep and the boats
(Kenney) are CC0 models packed into `viewer/public/vehicles/` by
`experiments/pack_vehicles.py`; the glider and the canoe are built in code, as are the
stand-ins shown while a model loads.

## Layout

| Path | What is there |
| --- | --- |
| `experiments/stage2_server.py` | Synthesis server (HTTP on :8765), single MPS worker thread |
| `experiments/chart.py`, `tile_stream.py`, `tile_grid.py`, `tile_ghost.py` | Chart parameters, stream planner, tile ids, ghost tiles |
| `experiments/stage1_patch.mjs` | Node entry point that samples the planet for the server |
| `experiments/export_viewer_worlds.py` | Hydrology solver; packs the demo worlds |
| `experiments/world_cache.py`, `paths.py` | Synthesized-world catalog; repo paths |
| `experiments/*.py` (others) | The standalone experiments behind `just geogenesis` … `just planet` |
| `viewer/src/` | Vite + three.js app: `planet.html` (globe) and `index.html` (explorer) |
| `viewer/public/worlds/`, `viewer/public/planet/` | Bundled demo assets; synthesized worlds land here too |
| `tests/`, `viewer/tests/` | pytest and `node --test` suites; neither needs the model |
| `docs/superpowers/` | Design notes and plans, kept as history |

## Commands

| Command | Does |
| --- | --- |
| `just dev` | Server + viewer together; Ctrl+C stops both |
| `just stage2-server` / `just viewer` | Either half on its own |
| `just test` | Python and JavaScript tests (a few seconds) |
| `just vendor` | Clone the pinned model code if it is missing |
| `just all`, `just terrain-diffusion`, `just planet`, `just export-viewer` | The original offline experiments; outputs go to `out/` |

## Working in a git worktree

`.vendor/` and `out/` are gitignored, so a fresh worktree has neither. The model code is found
automatically: `experiments/paths.py` falls back to the main checkout's `.vendor/` (or
`TERRAGROK_VENDOR_DIR`). Run `cd viewer && npm ci` once per worktree. Synthesized worlds are
per-checkout: a worktree starts with only the demo worlds.

## Known limits

- **Stitched terrain is 4.7 m per sample.** The atlas holds each tile at 1024², so a 2K home
  loses its extra sharpness once streaming starts.
- **A stutter when a tile lands.** Re-stitching takes 0.2–0.3 s.
- **Far tiles are coarse.** Outside the 3×3 around the player, terrain is 19 m per sample with
  no river ribbons, and it is not seam-corrected, so hairline gaps can show between far tiles.
- **Compass north is the map's north.** It matches the planet at home and drifts from true
  planet north the further you walk, because the chart is flat.
- **Rivers only continue downstream into newer tiles.** A river flowing into a tile that was
  synthesized earlier stops at that edge. Each tile also treats its edge as a place water can
  leave, so two tiles sometimes both drain to the same low point on their seam.
- **The chart is flat; the planet is not.** A chart repeats every 360° of arc (about 7.8 tiles)
  from home, and geography far from home is increasingly distorted. Walking does not wrap
  around the globe.
- **Home worlds synthesized before streaming existed** did not record their vertical scale or
  climate settings. The scale is estimated on first use and the defaults are assumed, so their
  seams are a little rougher than those of new worlds. Tiles from the earlier streaming scheme
  (`tile_s<seed>_…`) are ignored and can be deleted.
- Drill-down sub-regions and the demo worlds are single tiles; they do not stream.
