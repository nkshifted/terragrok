#!/usr/bin/env python3
"""
Apple Silicon MPS/CPU generator for Terrain Diffusion (30m) with macro pacing control.

Supports `coarse_pooling` (compresses macro planetary scale so a single playable world
spans diverse coastlines, lowlands, valleys, plateaus, and alpine peaks) and `histogram_raw`
(biases landforms toward dramatic relief), saving a high-res DEM + WorldClim climate layers
to `out/terrain-diffusion/` for micro-detail refinement, Landlab hydrology, and the 3D Explorer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add vendored terrain-diffusion repo to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from paths import VENDOR_TD, add_vendor_to_syspath  # noqa: E402

add_vendor_to_syspath()

from terrain_diffusion.inference.world_pipeline import WorldPipeline  # noqa: E402


OUT_DIR = REPO_ROOT / "out" / "terrain-diffusion"


def pick_diverse_coastal_region(
    coarse_elev_m: np.ndarray,
    coarse_temp_c: np.ndarray,
    coarse_precip_mm: np.ndarray,
    ci0: int,
    cj0: int,
    patch_coarse_cells: int,
) -> tuple[int, int, dict]:
    """
    Scan the coarse planetary map to find a `patch_coarse_cells x patch_coarse_cells` window
    packing maximum landform & climate diversity:
      - Real coastline / bays / peninsulas (~20-45% ocean)
      - Lowland valleys & coastal plains (0..250m)
      - Highland plateaus & ridges (300..1000m)
      - Tall alpine peaks (>1200m)
      - Broad elevation std-dev + temperature/precipitation contrast
    """
    h, w = coarse_elev_m.shape
    span = max(patch_coarse_cells, 2)
    best_score = -1e9
    best_pos = (ci0 + h // 2, cj0 + w // 2)
    best_info: dict = {}

    for r in range(2, h - span - 2):
        for c in range(2, w - span - 2):
            patch = coarse_elev_m[r : r + span, c : c + span]
            p_temp = coarse_temp_c[r : r + span, c : c + span]
            p_prec = coarse_precip_mm[r : r + span, c : c + span]

            ocean_frac = float((patch <= 0.0).mean())
            if ocean_frac < 0.12 or ocean_frac > 0.58:
                continue

            land = patch[patch > 0.0]
            if land.size < 4:
                continue

            p_max = float(land.max())
            if p_max < 450.0:
                continue

            # Reward having ALL elevation bands present: lowlands + foothills + alpine peaks!
            low_frac = float(((patch > 0.0) & (patch < 220.0)).mean())
            mid_frac = float(((patch >= 220.0) & (patch < 900.0)).mean())
            high_frac = float((patch >= 900.0).mean())
            band_diversity = (
                min(low_frac, 0.25) * 4.0
                + min(mid_frac, 0.25) * 4.0
                + min(high_frac, 0.20) * 5.0
            )  # max ~3.0

            # Reward coastline complexity (perimeter transitions between land & sea)
            is_sea = patch <= 0.0
            coast_edges = int(np.sum(is_sea[:, 1:] != is_sea[:, :-1]) + np.sum(is_sea[1:, :] != is_sea[:-1, :]))

            # Reward climate variety (temp & precip spread across the region)
            temp_span = float(p_temp.max() - p_temp.min())
            prec_span = float(p_prec.max() - p_prec.min())

            # Target ~28% ocean so ~72% is diverse walkable land
            ocean_score = 1.0 - abs(ocean_frac - 0.28) * 2.5
            relief_score = min(p_max, 2600.0) / 1000.0 + float(land.std()) / 500.0

            score = (
                band_diversity * 900.0
                + relief_score * 550.0
                + ocean_score * 650.0
                + min(coast_edges, span * 3) * 45.0
                + min(temp_span, 16.0) * 35.0
                + min(prec_span, 1200.0) * 0.25
            )

            if score > best_score:
                best_score = score
                best_pos = (ci0 + r, cj0 + c)
                best_info = {
                    "ocean_frac": round(ocean_frac, 2),
                    "land_max_m": round(p_max, 1),
                    "land_std_m": round(float(land.std()), 1),
                    "coast_edges": coast_edges,
                    "temp_span_c": round(temp_span, 1),
                    "prec_span_mm": round(prec_span, 0),
                    "score": round(score, 1),
                }

    return best_pos[0], best_pos[1], best_info


def run_generation(
    model_id: str,
    seed: int,
    device: str,
    tile_size: int = 1024,
    coarse_pooling: int = 2,
    histogram_raw: list[float] | None = None,
    coarse_radius: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load Terrain Diffusion, find a maximally diverse coastal-to-alpine region, and synthesize it."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    hist = histogram_raw or [0.0, 0.0, 0.0, 1.0, 1.8]
    print(
        f"Loading {model_id} on device={device} (seed={seed}, size={tile_size}, "
        f"coarse_pooling={coarse_pooling}, histogram={hist})..."
    )

    orig_cwd = os.getcwd()
    if VENDOR_TD.exists():
        os.chdir(VENDOR_TD)
    try:
        world = WorldPipeline.from_pretrained(
            model_id,
            seed=seed,
            caching_strategy="direct",
            torch_compile=False,
            latents_batch_size=4,
            coarse_pooling=coarse_pooling,
            histogram_raw=hist,
            log_mode="info",
        )
        world.to(device)
        world.bind()
    finally:
        os.chdir(orig_cwd)

    timings["load_s"] = round(time.perf_counter() - t0, 2)
    print(f"  Loaded & bound in {timings['load_s']}s")

    with world:
        # 1. Scan coarse planetary map (already pooled by `coarse_pooling` inside world.coarse!)
        t1 = time.perf_counter()
        ci0, cj0 = -coarse_radius, -coarse_radius
        coarse_raw = world.coarse[:, ci0 : ci0 + 2 * coarse_radius, cj0 : cj0 + 2 * coarse_radius]
        coarse_map = (coarse_raw[:-1] / coarse_raw[-1:]).detach().cpu().numpy()
        c0 = coarse_map[0]
        coarse_elev_m = np.sign(c0) * (c0 ** 2)
        coarse_temp_c = coarse_map[2]
        coarse_precip_mm = coarse_map[4]
        timings["coarse_scan_s"] = round(time.perf_counter() - t1, 2)

        # Each coarse pixel corresponds to 256 fine pixels (latent_compression * 32)
        scale = int(world.latent_compression) * 32
        patch_coarse_cells = max(tile_size // scale, 2)

        best_ci, best_cj, info = pick_diverse_coastal_region(
            coarse_elev_m,
            coarse_temp_c,
            coarse_precip_mm,
            ci0,
            cj0,
            patch_coarse_cells=patch_coarse_cells,
        )
        print(
            f"  Coarse scan ({2*coarse_radius}x{2*coarse_radius}, patch={patch_coarse_cells}x{patch_coarse_cells} coarse) "
            f"took {timings['coarse_scan_s']}s -> selected ({best_ci}, {best_cj}) {info}"
        )

        # 2. Synthesize high-res fine tile + WorldClim climate channels
        i1 = best_ci * scale
        j1 = best_cj * scale
        i2 = i1 + tile_size
        j2 = j1 + tile_size

        t2 = time.perf_counter()
        out = world.get(i1, j1, i2, j2, with_climate=True)
        if device == "mps":
            torch.mps.synchronize()
        timings["tile_gen_s"] = round(time.perf_counter() - t2, 2)

        elev = out["elev"].detach().cpu().numpy().astype(np.float32)
        climate = out["climate"].detach().cpu().numpy().astype(np.float32)
        print(
            f"  Generated {tile_size}x{tile_size} tile in {timings['tile_gen_s']}s! "
            f"elev=[{elev.min():.1f}m .. {elev.max():.1f}m], "
            f"ocean={(elev <= 0).mean()*100:.1f}%, "
            f"temp=[{climate[0].min():.1f}..{climate[0].max():.1f}C], "
            f"precip=[{climate[2].min():.0f}..{climate[2].max():.0f}mm]"
        )

        meta = {
            "model_id": model_id,
            "seed": seed,
            "device": device,
            "tile_size": tile_size,
            "coarse_pooling": coarse_pooling,
            "histogram_raw": hist,
            "coarse_ij": [best_ci, best_cj],
            "fine_bbox": [i1, j1, i2, j2],
            "timings": timings,
            "coarse_info": info,
        }
        return elev, climate, meta


def save_overview_plot(elev: np.ndarray, climate: np.ndarray, meta: dict, out_png: Path) -> None:
    """Save 4-panel visual diagnostic of Terrain Diffusion DEM & climate."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=135)
    fig.patch.set_facecolor("#0f141c")

    gy, gx = np.gradient(elev, 30.0)
    sun = np.array([0.65, -0.45, 0.62])
    sun /= np.linalg.norm(sun)
    nz = np.ones_like(gx)
    norm = np.sqrt(gx**2 + gy**2 + nz**2)
    shade = np.clip((-gx * sun[0] - gy * sun[1] + nz * sun[2]) / norm, 0.15, 1.0)

    rgb = np.zeros((*elev.shape, 3), dtype=np.float32)
    ocean = elev <= 0.0
    p98 = float(np.percentile(elev[~ocean], 98)) if (~ocean).any() else 1000.0
    t_land = np.clip(elev / max(p98, 1.0), 0.0, 1.0)
    rgb[~ocean, 0] = np.interp(t_land[~ocean], [0.0, 0.22, 0.60, 0.86, 1.0], [0.24, 0.30, 0.50, 0.72, 0.96])
    rgb[~ocean, 1] = np.interp(t_land[~ocean], [0.0, 0.22, 0.60, 0.86, 1.0], [0.48, 0.50, 0.48, 0.72, 0.97])
    rgb[~ocean, 2] = np.interp(t_land[~ocean], [0.0, 0.22, 0.60, 0.86, 1.0], [0.22, 0.24, 0.44, 0.74, 0.99])
    rgb[~ocean] *= (0.36 + 0.74 * shade[~ocean, None])

    t_sea = np.clip(-elev[ocean] / 500.0, 0.0, 1.0)
    rgb[ocean, 0] = 0.05 + 0.10 * (1.0 - t_sea)
    rgb[ocean, 1] = 0.18 + 0.28 * (1.0 - t_sea)
    rgb[ocean, 2] = 0.34 + 0.28 * (1.0 - t_sea)

    axes[0, 0].imshow(np.clip(rgb, 0, 1))
    axes[0, 0].set_title(
        f"Terrain Diffusion 30m ({meta['tile_size']}px, pool={meta['coarse_pooling']}, seed={meta['seed']})\n"
        f"elev {elev.min():.0f}m .. {elev.max():.0f}m | gen {meta['timings']['tile_gen_s']}s ({meta['device']})",
        color="white", fontsize=10,
    )
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(elev, cmap="terrain")
    axes[0, 1].set_title("Elevation (meters)", color="white", fontsize=10)
    axes[0, 1].axis("off")
    cb1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb1.ax.get_yticklabels(), color="white", fontsize=8)

    im2 = axes[1, 0].imshow(climate[0], cmap="coolwarm")
    axes[1, 0].set_title(
        f"Elevation-Adjusted Temp ({climate[0].min():.1f}..{climate[0].max():.1f}°C)",
        color="white", fontsize=10,
    )
    axes[1, 0].axis("off")
    cb2 = fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.get_yticklabels(), color="white", fontsize=8)

    im3 = axes[1, 1].imshow(climate[2], cmap="YlGnBu")
    axes[1, 1].set_title(
        f"Annual Precipitation ({climate[2].min():.0f}..{climate[2].max():.0f} mm/yr)",
        color="white", fontsize=10,
    )
    axes[1, 1].axis("off")
    cb3 = fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cb3.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb3.ax.get_yticklabels(), color="white", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large diverse Terrain Diffusion 30m world on Apple Silicon")
    parser.add_argument("--model", default="xandergos/terrain-diffusion-30m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--coarse-pooling", type=int, default=2)
    parser.add_argument("--histogram", default="0.0,0.0,0.0,1.0,1.8")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist = [float(x.strip()) for x in args.histogram.split(",")]

    if args.device == "auto":
        candidates = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    else:
        candidates = [args.device]

    elev = climate = meta = None
    for dev in candidates:
        try:
            elev, climate, meta = run_generation(
                args.model,
                args.seed,
                dev,
                tile_size=args.size,
                coarse_pooling=args.coarse_pooling,
                histogram_raw=hist,
            )
            break
        except Exception as exc:
            print(f"Device {dev!r} failed: {exc!r}")
            if dev == candidates[-1]:
                raise
            print("Falling back to next device...")

    assert elev is not None and climate is not None and meta is not None
    elev.tofile(OUT_DIR / f"elev_{args.size}.f32")
    # Also write canonical elev_latest.f32 / climate_latest.npz for export_viewer_worlds.py
    elev.tofile(OUT_DIR / "elev_latest.f32")
    np.savez_compressed(
        OUT_DIR / "climate_latest.npz",
        temp=climate[0],
        temp_season=climate[1],
        precip=climate[2],
        precip_cv=climate[3],
    )
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    save_overview_plot(elev, climate, meta, OUT_DIR / "terrain_diffusion_overview.png")
    print(f"Saved {args.size}x{args.size} outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
