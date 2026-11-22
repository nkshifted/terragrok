#!/usr/bin/env python3
"""
Experiment 4: Whitebox Workflows Geomorphometry & Playable Landform Extraction

Demonstrates tool-chaining: takes a 16-bit heightmap produced by another generator
(e.g. terrain-generator or fastscapelib) and extracts game-ready geographic intelligence
using Whitebox Workflows (open tier):
  - Least-cost depression breaching & D8 flow accumulation
  - Multidirectional hillshade
  - Slope & terrain traversability mask
  - Geomorphons 10-class landform classification (Flat, Peak, Ridge, Shoulder,
    Spur, Slope, Hollow, Footslope, Valley, Pit) -> ideal for procedural POI placement!
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
import whitebox_workflows as wb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


GEOMORPHON_NAMES = {
    1: ("Flat", "#d9e2c4"),
    2: ("Peak / Summit", "#ffffff"),
    3: ("Ridge", "#c85a32"),
    4: ("Shoulder", "#e69f56"),
    5: ("Spur", "#d4c36a"),
    6: ("Slope", "#78a662"),
    7: ("Hollow", "#4d9182"),
    8: ("Footslope", "#4979a6"),
    9: ("Valley", "#264d80"),
    10: ("Pit / Basin", "#1a2442"),
}


def load_heightmap(path: Path, max_elev_m: float, sea_level_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Load 16-bit PNG heightmap into elevation meters and land mask."""
    img = Image.open(path)
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() > 255:
        norm = arr / 65535.0
    else:
        norm = arr / 255.0
    elev_m = (norm - sea_level_frac) / (1.0 - sea_level_frac) * max_elev_m
    land_mask = norm > sea_level_frac
    return elev_m, land_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Whitebox Workflows geomorphometry pipeline")
    parser.add_argument(
        "--heightmap",
        type=Path,
        default=Path("out/terrain-gen/island_42_height.png"),
        help="Input 16-bit PNG heightmap",
    )
    parser.add_argument("--cell-size-m", type=float, default=20.0, help="Meters per pixel")
    parser.add_argument("--max-elev-m", type=float, default=1800.0, help="Peak elevation in meters")
    parser.add_argument("--sea-level", type=float, default=0.08, help="Normalized sea level threshold")
    parser.add_argument("--out", type=Path, default=Path("out/whitebox"), help="Output directory")
    args = parser.parse_args()

    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)

    elev_m, land = load_heightmap(args.heightmap, args.max_elev_m, args.sea_level)
    ny, nx = elev_m.shape

    # Write Esri ASCII grid DEM so Whitebox preserves exact cell size & coordinates
    wbe = wb.WbEnvironment()
    wbe.verbose = False

    # Clamp ocean floor to 0m for terrestrial geomorphometry
    dem_arr = np.where(land, np.maximum(elev_m, 0.0), 0.0).astype(np.float64)
    asc_path = args.out / "input_dem.asc"
    header = (
        f"ncols {nx}\n"
        f"nrows {ny}\n"
        f"xllcorner 0.0\n"
        f"yllcorner 0.0\n"
        f"cellsize {args.cell_size_m}\n"
        f"NODATA_value -32768.0"
    )
    np.savetxt(asc_path, dem_arr, fmt="%.3f", header=header, comments="")
    dem = wbe.read_raster(str(asc_path))

    # 1. Multidirectional hillshade
    hillshade = wbe.terrain.multidirectional_hillshade(input=dem, z_factor=1.5).to_numpy()

    # 2. Slope in degrees (traversability / pathfinding cost)
    slope_deg = wbe.terrain.slope(input=dem, units="degrees").to_numpy()

    # 3. Geomorphons 10-class landform classification
    geomorph = wbe.terrain.geomorphons(
        input=dem,
        search_distance=20,
        flatness_threshold=1.5,
        flatness_distance=0,
        skip_distance=0,
        output_forms=True,
    ).to_numpy()

    # 4. Least-cost depression breaching + D8 flow accumulation
    breached = wbe.hydrology.breach_depressions_least_cost(dem=dem, max_dist=40)
    flow_accum = wbe.hydrology.d8_flow_accum(
        input=breached, out_type="cells", log_transform=True
    ).to_numpy()

    elapsed = time.perf_counter() - t0

    # Build 2x2 diagnostic figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), dpi=140)
    fig.patch.set_facecolor("#111418")

    # Panel 1: Multidirectional Hillshade
    hs_vis = np.where(land, hillshade, np.nan)
    axes[0, 0].imshow(hs_vis, cmap="gray")
    axes[0, 0].set_facecolor("#0c223d")
    axes[0, 0].set_title("Whitebox Multidirectional Hillshade", color="white", fontsize=11, pad=8)

    # Panel 2: Geomorphons Landform Classification
    gm_masked = np.where(land, np.clip(np.rint(geomorph), 1, 10), 0).astype(np.int32)
    gm_colors = ["#0c223d"] + [GEOMORPHON_NAMES[i][1] for i in range(1, 11)]
    gm_cmap = ListedColormap(gm_colors)
    axes[0, 1].imshow(gm_masked, cmap=gm_cmap, vmin=0, vmax=10, interpolation="nearest")
    axes[0, 1].set_title(
        "Geomorphons Landforms (POI & Biome Placement)",
        color="white",
        fontsize=11,
        pad=8,
    )
    legend_handles = [
        Patch(facecolor=GEOMORPHON_NAMES[i][1], label=GEOMORPHON_NAMES[i][0])
        for i in [2, 3, 5, 6, 7, 8, 9, 1]
    ]
    axes[0, 1].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7.5,
        framealpha=0.85,
        facecolor="#161b22",
        labelcolor="white",
        edgecolor="#30363d",
    )

    # Panel 3: Slope (Traversability / Movement Cost)
    slope_vis = np.where(land, np.clip(slope_deg, 0, 50), np.nan)
    im_sl = axes[1, 0].imshow(slope_vis, cmap="RdYlGn_r", vmin=0, vmax=45)
    axes[1, 0].set_facecolor("#0c223d")
    axes[1, 0].set_title("Slope (°) — Traversability & Cliff Mask", color="white", fontsize=11, pad=8)
    cb3 = fig.colorbar(im_sl, ax=axes[1, 0], fraction=0.046, pad=0.03)
    cb3.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb3.ax.get_yticklabels(), color="white", fontsize=8)

    # Panel 4: Least-Cost Breached Log D8 Flow Accumulation
    fa_vis = np.where(land, flow_accum, np.nan)
    im_fa = axes[1, 1].imshow(fa_vis, cmap="PuBu", vmin=0, vmax=np.nanpercentile(fa_vis, 99.5))
    axes[1, 1].set_facecolor("#0c223d")
    axes[1, 1].set_title("Breached D8 Log Flow Accumulation (Watersheds)", color="white", fontsize=11, pad=8)
    cb4 = fig.colorbar(im_fa, ax=axes[1, 1], fraction=0.046, pad=0.03)
    cb4.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb4.ax.get_yticklabels(), color="white", fontsize=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2c3440")

    fig.suptitle(
        f"Experiment 4: Whitebox Workflows Geomorphometry on {args.heightmap.name} ({nx}×{ny}, {elapsed:.2f}s)",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out / "landforms_analysis.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Compute landform distribution summary on land
    land_gm = gm_masked[land]
    counts = {
        GEOMORPHON_NAMES[i][0]: round(float((land_gm == i).mean() * 100.0), 2)
        for i in range(1, 11)
    }
    stats = {
        "input": str(args.heightmap),
        "grid": [ny, nx],
        "cell_size_m": args.cell_size_m,
        "elapsed_seconds": round(elapsed, 3),
        "mean_slope_deg": round(float(slope_deg[land].mean()), 2),
        "steep_cliffs_gt30deg_pct": round(float((slope_deg[land] > 30.0).mean() * 100.0), 2),
        "landform_pct": counts,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
