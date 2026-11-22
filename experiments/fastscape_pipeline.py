#!/usr/bin/env python3
"""
Experiment 3: Programmable Geomorphology Pipeline (FastNoiseLite + Fastscapelib)

Demonstrates how to build a custom, deterministic, game-oriented terraforming
pipeline by composing:
  1. pyfastnoiselite  -> Macro continental mask + tectonic uplift field (OpenSimplex2S FBM + Ridged)
  2. fastscapelib     -> Physically based landscape evolution over geological time:
                         - Priority-flood D8 flow routing (Barnes 2014)
                         - Stream Power Law (SPL) fluvial river incision (Braun & Willett 2013)
                         - Linear hillslope diffusion (ADI solver)
  3. Orographic climate -> Prevailing westerlies + rain shadow + lapse-rate temperature
  4. Game-ready exports -> 16-bit PNG heightmap, raw float32 elevation, and diagnostic atlas
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fastscapelib as fs
import matplotlib
import numpy as np
from PIL import Image
from pyfastnoiselite.pyfastnoiselite import (
    FastNoiseLite,
    FractalType,
    NoiseType,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap


def sample_noise_grid(
    size: int,
    seed: int,
    frequency: float,
    octaves: int,
    fractal_type: FractalType,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Sample a 2D normalized [-1, 1] noise field using FastNoiseLite."""
    noise = FastNoiseLite(seed=seed)
    noise.noise_type = NoiseType.NoiseType_OpenSimplex2S
    noise.fractal_type = fractal_type
    noise.fractal_octaves = octaves
    noise.fractal_lacunarity = lacunarity
    noise.fractal_gain = gain
    noise.frequency = frequency

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    coords = np.vstack([x.ravel(), y.ravel()])
    field = np.asarray(noise.gen_from_coords(coords), dtype=np.float64).reshape(size, size)
    return field


def build_tectonic_fields(size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build initial elevation (m), tectonic uplift rate (m/yr), and land mask."""
    # Macro continent shape: radial falloff warped by low-frequency FBM
    yy, xx = np.mgrid[0:size, 0:size]
    ny = (yy / (size - 1)) * 2.0 - 1.0
    nx = (xx / (size - 1)) * 2.0 - 1.0
    radial = np.sqrt((nx * 1.05) ** 2 + (ny * 1.15) ** 2)

    macro_fbm = sample_noise_grid(
        size=size,
        seed=seed,
        frequency=2.2 / size,
        octaves=5,
        fractal_type=FractalType.FractalType_FBm,
    )
    ridge_belt = sample_noise_grid(
        size=size,
        seed=seed + 101,
        frequency=3.0 / size,
        octaves=6,
        fractal_type=FractalType.FractalType_Ridged,
        gain=0.52,
    )
    detail_fbm = sample_noise_grid(
        size=size,
        seed=seed + 202,
        frequency=10.0 / size,
        octaves=4,
        fractal_type=FractalType.FractalType_FBm,
    )

    # Continent mask smoothly dropping below sea level at borders (~40-45% land)
    continent_signal = 0.55 * macro_fbm + 0.58 - 0.85 * (radial**1.75)
    land_strength = np.clip(continent_signal * 1.9, 0.0, 1.0)

    # Asymmetric tectonic collision zone (stronger uplift along western/central arc)
    collision_arc = np.exp(-(((nx + 0.15 - 0.35 * macro_fbm) / 0.38) ** 2))
    ridge_norm = np.clip((ridge_belt + 1.0) * 0.5, 0.0, 1.0)

    # Uplift rate in m/yr (active orogen: up to ~3.2 mm/yr -> alpine peaks ~3000m)
    uplift_rate = (
        land_strength
        * (0.00035 + 0.0028 * collision_arc * (0.35 + 0.65 * ridge_norm))
    )
    # Keep boundaries pinned at sea level
    uplift_rate[0, :] = 0.0
    uplift_rate[-1, :] = 0.0
    uplift_rate[:, 0] = 0.0
    uplift_rate[:, -1] = 0.0

    # Initial low-relief proto-continent with gentle perturbation to seed drainage
    init_elev = (
        land_strength * (120.0 + 180.0 * ridge_norm + 25.0 * detail_fbm)
        - (1.0 - land_strength) * (450.0 + 180.0 * np.clip(-continent_signal, 0.0, 1.5))
    )
    init_elev[0, :] = -450.0
    init_elev[-1, :] = -450.0
    init_elev[:, 0] = -450.0
    init_elev[:, -1] = -450.0

    return init_elev, uplift_rate, land_strength


def evolve_landscape(
    init_elev: np.ndarray,
    uplift_rate: np.ndarray,
    world_km: float = 200.0,
    n_steps: int = 80,
    dt_yr: float = 25_000.0,
    k_spl: float = 1.6e-5,
    k_diff: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Fastscapelib Stream Power Law + ADI Hillslope Diffusion over geological time."""
    ny, nx = init_elev.shape
    length_m = world_km * 1000.0

    grid = fs.RasterGrid.from_length(
        shape=(ny, nx),
        length=(length_m, length_m),
        bounds_status=fs.NodeStatus.FIXED_VALUE,
    )
    flow_graph = fs.FlowGraph(
        grid,
        [fs.SingleFlowRouter(), fs.PFloodSinkResolver()],
    )
    spl_eroder = fs.SPLEroder(
        flow_graph=flow_graph,
        k_coef=k_spl,
        area_exp=0.45,
        slope_exp=1.0,
    )
    diff_eroder = fs.DiffusionADIEroder(grid=grid, k_coef=k_diff)

    # Evolve subaerial topography (clamp ocean floor separately to preserve shelf)
    elev = np.maximum(init_elev, 0.0).copy()
    subaerial_mask = init_elev > -60.0

    for _ in range(n_steps):
        elev += uplift_rate * dt_yr
        elev = np.where(subaerial_mask, np.maximum(elev, 0.0), 0.0)
        flow_graph.update_routes(elev)
        drainage_area = flow_graph.accumulate(1.0)
        elev -= spl_eroder.erode(elev, drainage_area, dt_yr)
        elev -= diff_eroder.erode(elev, dt_yr)
        elev[0, :] = 0.0
        elev[-1, :] = 0.0
        elev[:, 0] = 0.0
        elev[:, -1] = 0.0

    # Final drainage accumulation
    flow_graph.update_routes(elev)
    drainage_area = flow_graph.accumulate(1.0)

    # Merge evolved land with bathymetry below sea level
    final_elev = np.where(elev > 1.0, elev, np.minimum(init_elev, -1.0))
    return final_elev, drainage_area


def compute_climate_and_biomes(
    elev: np.ndarray, drainage_area: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple orographic rain-shadow precipitation + lapse-rate temperature + biomes."""
    ny, nx = elev.shape
    lat = np.linspace(55.0, 15.0, ny)[:, None]  # North (cool) to South (warm)
    land = elev > 0.0

    # Temperature (°C): latitude gradient - 6.5°C per km elevation
    temp_c = 28.0 - 0.55 * (lat - 15.0) - 6.5 * np.maximum(elev, 0.0) / 1000.0

    # Prevailing westerly winds (left -> right) with orographic rain-out & rain shadow
    moisture = np.ones(ny, dtype=np.float64) * 1.0
    precip_mm = np.zeros((ny, nx), dtype=np.float64)
    prev_h = np.zeros(ny, dtype=np.float64)

    for col in range(nx):
        h = np.maximum(elev[:, col], 0.0)
        is_ocean = ~land[:, col]
        # Recharge moisture over ocean
        moisture = np.where(is_ocean, np.minimum(moisture + 0.05, 1.25), moisture)
        # Orographic lift increases rain-out on windward slopes
        dh_up = np.maximum(h - prev_h, 0.0)
        rain_frac = np.where(
            is_ocean,
            0.006,
            0.014 + 0.0045 * (h / 1000.0) + 0.008 * (dh_up / 100.0),
        )
        rain = moisture * np.clip(rain_frac, 0.006, 0.12)
        precip_mm[:, col] = rain * 78_000.0
        moisture = np.maximum(moisture - rain * 0.52, 0.09)
        prev_h = h

    # Add riparian moisture boost near major rivers
    river_boost = np.clip(np.log10(np.maximum(drainage_area, 1.0)) - 6.5, 0.0, 2.5) * 180.0
    precip_mm = np.clip(precip_mm + river_boost, 50.0, 3800.0)

    # Biome IDs:
    # 0=Ocean, 1=Alpine Snow/Ice, 2=Tundra/Rock, 3=Boreal Forest,
    # 4=Desert/Scrub, 5=Grassland/Steppe, 6=Temperate Forest, 7=Rainforest
    biomes = np.zeros((ny, nx), dtype=np.uint8)
    biomes[land & (temp_c < -1.0)] = 1
    biomes[land & (temp_c >= -1.0) & (temp_c < 4.0)] = 2
    biomes[land & (temp_c >= 4.0) & (temp_c < 10.0) & (precip_mm >= 450)] = 3
    biomes[land & (temp_c >= 4.0) & (precip_mm < 450)] = 4
    biomes[land & (temp_c >= 10.0) & (precip_mm >= 450) & (precip_mm < 1050)] = 5
    biomes[land & (temp_c >= 10.0) & (precip_mm >= 1050) & (precip_mm < 1950)] = 6
    biomes[land & (temp_c >= 10.0) & (precip_mm >= 1950)] = 7
    return temp_c, precip_mm, biomes


def render_outputs(
    out_dir: Path,
    elev: np.ndarray,
    uplift_rate: np.ndarray,
    drainage_area: np.ndarray,
    precip_mm: np.ndarray,
    biomes: np.ndarray,
    world_km: float,
    seed: int,
    elapsed_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    land = elev > 0.0

    # 1. Export raw float32 + 16-bit PNG heightmap (standard game engine input)
    elev.astype(np.float32).tofile(out_dir / "elevation.f32")
    emin, emax = float(elev.min()), float(elev.max())
    norm16 = np.clip((elev - emin) / max(emax - emin, 1e-6) * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(norm16).save(out_dir / "height16.png")

    # 2. High-res shaded topographic relief + river network
    dx_m = (world_km * 1000.0) / elev.shape[1]
    ls = LightSource(azdeg=315, altdeg=35)
    land_elev = np.maximum(elev, 0.0)
    rgb_land = ls.shade(
        land_elev,
        cmap=plt.cm.terrain,
        vert_exag=3.5,
        dx=dx_m,
        dy=dx_m,
        blend_mode="soft",
        vmin=-600.0,
        vmax=max(float(elev.max()), 1500.0),
    )

    # Ocean bathymetry shading
    depth_norm = np.clip(-elev / max(float(-elev.min()), 1.0), 0.0, 1.0)
    ocean_rgb = np.stack(
        [
            0.05 + 0.14 * (1.0 - depth_norm) ** 2,
            0.16 + 0.28 * (1.0 - depth_norm) ** 2,
            0.34 + 0.34 * (1.0 - depth_norm) ** 2,
        ],
        axis=-1,
    )
    relief = np.where(land[..., None], rgb_land[..., :3], ocean_rgb)

    # Overlay rivers where drainage area exceeds threshold
    river_thresh = np.percentile(drainage_area[land], 97.5) if land.any() else 1e9
    rivers = land & (drainage_area >= river_thresh)
    relief[rivers] = [0.10, 0.42, 0.78]

    Image.fromarray((np.clip(relief, 0, 1) * 255).astype(np.uint8)).save(
        out_dir / "relief_rivers.png"
    )

    # 3. 2x2 Diagnostic Atlas
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), dpi=140)
    fig.patch.set_facecolor("#111418")

    # Panel 1: Shaded Relief + Rivers
    axes[0, 0].imshow(relief)
    axes[0, 0].set_title(
        f"Fastscapelib Shaded Relief + Rivers\n(peak {elev.max():.0f} m, {world_km:.0f}×{world_km:.0f} km)",
        color="white",
        fontsize=11,
        pad=8,
    )

    # Panel 2: Tectonic Uplift Rate (input forcing)
    im_up = axes[0, 1].imshow(uplift_rate * 1000.0, cmap="inferno")
    axes[0, 1].set_title("Tectonic Uplift Forcing (mm/yr)", color="white", fontsize=11, pad=8)
    cb2 = fig.colorbar(im_up, ax=axes[0, 1], fraction=0.046, pad=0.03)
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.get_yticklabels(), color="white", fontsize=8)

    # Panel 3: Log10 Drainage Area (watersheds & fluvial network)
    log_da = np.where(land, np.log10(np.maximum(drainage_area, 1.0)), np.nan)
    im_da = axes[1, 0].imshow(log_da, cmap="viridis")
    axes[1, 0].set_facecolor("#0b1c33")
    axes[1, 0].set_title("Log10 Drainage Area (m²) — Dendritic Basins", color="white", fontsize=11, pad=8)
    cb3 = fig.colorbar(im_da, ax=axes[1, 0], fraction=0.046, pad=0.03)
    cb3.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb3.ax.get_yticklabels(), color="white", fontsize=8)

    # Panel 4: Emergent Climate & Biomes (with hillshade)
    biome_colors = [
        "#12355b",  # 0 Ocean
        "#eef4f8",  # 1 Alpine Snow
        "#9ba29f",  # 2 Tundra/Rock
        "#2d5a4c",  # 3 Boreal Forest
        "#d8be85",  # 4 Desert/Scrub
        "#8ab058",  # 5 Grassland/Steppe
        "#3c7c3e",  # 6 Temperate Forest
        "#1c542d",  # 7 Rainforest
    ]
    biome_cmap = ListedColormap(biome_colors)
    hillshade = ls.hillshade(land_elev, vert_exag=2.8, dx=dx_m, dy=dx_m)
    biome_rgb = biome_cmap(biomes)[..., :3]
    shade_factor = np.where(land, 0.55 + 0.55 * hillshade, 1.0)[..., None]
    shaded_biomes = np.clip(biome_rgb * shade_factor, 0.0, 1.0)
    shaded_biomes[rivers] = [0.16, 0.54, 0.88]

    axes[1, 1].imshow(shaded_biomes)
    axes[1, 1].set_title(
        f"Orographic Biomes (Westerlies → Rain Shadow)\n(mean land precip {precip_mm[land].mean():.0f} mm/yr)",
        color="white",
        fontsize=11,
        pad=8,
    )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2c3440")

    fig.suptitle(
        f"Experiment 3: FastNoiseLite + Fastscapelib Geomorphology Pipeline (seed={seed}, {elev.shape[0]}²)",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "summary.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    stats = {
        "seed": seed,
        "grid": list(elev.shape),
        "world_km": world_km,
        "elapsed_seconds": round(elapsed_s, 3),
        "land_fraction": round(float(land.mean()), 4),
        "min_elevation_m": round(emin, 1),
        "max_elevation_m": round(emax, 1),
        "mean_land_elevation_m": round(float(elev[land].mean()), 1) if land.any() else 0.0,
        "river_cells": int(rivers.sum()),
        "mean_land_precip_mm": round(float(precip_mm[land].mean()), 1) if land.any() else 0.0,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="FastNoiseLite + Fastscapelib terraforming test")
    parser.add_argument("--size", type=int, default=512, help="Grid resolution (NxN)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--steps", type=int, default=80, help="Geological time steps")
    parser.add_argument("--world-km", type=float, default=200.0, help="Domain width in km")
    parser.add_argument("--out", type=Path, default=Path("out/fastscape"), help="Output directory")
    args = parser.parse_args()

    t0 = time.perf_counter()
    init_elev, uplift_rate, _ = build_tectonic_fields(args.size, args.seed)
    elev, drainage_area = evolve_landscape(
        init_elev=init_elev,
        uplift_rate=uplift_rate,
        world_km=args.world_km,
        n_steps=args.steps,
    )
    _, precip_mm, biomes = compute_climate_and_biomes(elev, drainage_area)
    elapsed = time.perf_counter() - t0
    render_outputs(
        out_dir=args.out,
        elev=elev,
        uplift_rate=uplift_rate,
        drainage_area=drainage_area,
        precip_mm=precip_mm,
        biomes=biomes,
        world_km=args.world_km,
        seed=args.seed,
        elapsed_s=elapsed,
    )


if __name__ == "__main__":
    main()
