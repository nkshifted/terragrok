set shell := ["mise", "exec", "--", "bash", "-euo", "pipefail", "-c"]

# List available terraforming experiments
default:
    @just --list

# Install mise tools, Python + viewer dependencies, and the Terrain Diffusion model code
setup:
    mise trust
    mise install
    uv sync
    cd viewer && npm ci
    just vendor

# Clone the pinned Terrain Diffusion code into .vendor/ (git worktrees reuse the main checkout's copy)
vendor rev="e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230":
    python experiments/paths.py --check || (git clone https://github.com/xandergos/terrain-diffusion.git .vendor/terrain-diffusion && git -C .vendor/terrain-diffusion checkout --quiet {{rev}})

# Run all 4 terraforming experiments
all: geogenesis terrain-gen fastscape whitebox

# Experiment 1: GeoGenesis planetary tectonics, climate, erosion & biomes
geogenesis seed="42" grid="512":
    mkdir -p out
    geogenesis generate --preset earthlike --seed {{seed}} --grid {{grid}} --run --out out/geogenesis-earthlike
    geogenesis generate --preset archipelago --seed 108 --grid {{grid}} --run --out out/geogenesis-archipelago

# Experiment 2: terrain-generator fractal + Beyer hydraulic erosion + hydrology
terrain-gen seed="42" size="1024":
    mkdir -p out/terrain-gen
    terrain-generator -W {{size}} -H {{size}} -s {{seed}} --droplets 1048576 --breach-basins 0.01 --layers shaded,height,flow,rivers,lakes,ocean --prefix island_ -o out/terrain-gen
    terrain-generator -W {{size}} -H {{size}} -s 7 --archipelago 7 --droplets 1048576 --rivers 0.012 --layers shaded,height,rivers,lakes,ocean --prefix arch_ -o out/terrain-gen

# Experiment 3: Programmable FastNoiseLite + Fastscapelib geomorphology pipeline
fastscape seed="42" size="512" steps="80":
    python experiments/fastscape_pipeline.py --seed {{seed}} --size {{size}} --steps {{steps}} --out out/fastscape

# Experiment 4: Whitebox Workflows geomorphometry & landform classification (chained on Experiment 2)
whitebox:
    python experiments/whitebox_geomorphometry.py --heightmap out/terrain-gen/island_42_height.png --out out/whitebox

# Experiment 5: Terrain Diffusion 30m (AI Earth DEM + WorldClim climate on Apple Silicon MPS/CPU)
terrain-diffusion seed="42":
    uv run --extra diffusion python experiments/terrain_diffusion_test.py --seed {{seed}}

# Experiment 6: Zero-Seam Cube-Sphere Planet (6 Terrain Diffusion 30m continents + Landlab hydrology)
planet seed="42" radius="3000":
    uv run --extra diffusion python experiments/generate_planet.py --seed {{seed}} --radius {{radius}}

# Pack generated worlds into binary assets for the 3D viewer
export-viewer:
    python experiments/export_viewer_worlds.py

# Launch persistent warm Stage 2 Terrain Diffusion 30m MPS server (http://127.0.0.1:8765)
stage2-server:
    uv run --extra diffusion python experiments/stage2_server.py --port 8765

# Launch the interactive 3D world explorer at http://127.0.0.1:5173
viewer:
    cd viewer && npm run dev

# One-command local stack: Stage 2 MPS backend + Vite viewer (Ctrl+C stops both)
dev:
    bash scripts/dev.sh

# Fast invariant / regression tests (no MPS / no diffusion model required)
test:
    uv run --with pytest pytest tests/ -q
    cd viewer && node --test "tests/*.test.mjs"

# Remove generated outputs
clean:
    rm -rf out
