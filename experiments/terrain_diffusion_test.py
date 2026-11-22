#!/usr/bin/env python3
"""
Apple Silicon benchmark & scenic coastal-mountain exporter for Terrain Diffusion (30m).

Tests xandergos/terrain-diffusion-30m on macOS ARM64 (trying PyTorch MPS first with
automatic CPU fallback), scans the coarse planetary map to find a dramatic
mountain-to-sea coastal fjord/valley region, generates a 512x512 DEM + WorldClim
climate maps, and saves artifacts to `out/terrain-diffusion/` for our Landlab
hydrology pipeline and Three.js 3D Explorer.
"""

from __future__ import annotations

import argparse
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
VENDOR_TD = REPO_ROOT / ".vendor" / "terrain-diffusion"
if VENDOR_TD.exists():
    sys.path.insert(0, str(VENDOR_TD))

from terrain_diffusion.inference.world_pipeline import WorldPipeline  # noqa: E402


OUT_DIR = REPO_ROOT / "out" / "terrain-diffusion"


def pick_scenic_coastal_tile(coarse_elev_m: np.ndarray, ci0: int, cj0: int) -> tuple[int, int, dict]:
    """
    Scan coarse 7.7km/px elevation grid to locate a dramatic 2x2 coarse tile
    (512x512 fine pixels) containing BOTH ocean coastline AND tall alpine mountains!
    """
    h, w = coarse_elev_m.shape
    best_score = -1e9
    best_pos = (ci0 + h // 2, cj0 + w // 2)
    best_info = {}

    for r in range(2, h - 3):
        for c in range(2, w - 3):
            patch = coarse_elev_m[r : r + 2, c : c + 2]
            nbr = coarse_elev_m[r - 1 : r + 3, c - 1 : c + 3]
            p_min = float(patch.min())
            p_max = float(patch.max())
            n_min = float(nbr.min())
            n_max = float(nbr.max())

            # Want real coastline inside or right along the 2x2 patch + high mountain relief!
            has_coast = (p_min < 20.0 and p_max > 250.0) or (n_min < -50.0 and p_max > 350.0)
            if not has_coast:
                continue

            # Favor ~20-65% ocean in 4x4 neighborhood + steep relief (500m-2500m peaks)
            ocean_frac = float((nbr <= 0.0).mean())
            coast_bonus = 1.0 - abs(ocean_frac - 0.35) * 2.0
            relief = min(p_max, 2400.0) - max(p_min, -400.0)
            score = relief + coast_bonus * 900.0
            if p_min < 0.0 < p_max:
                score += 600.0  # Direct shoreline inside the 512x512 patch!

            if score > best_score:
                best_score = score
                best_pos = (ci0 + r, cj0 + c)
                best_info = {
                    "patch_min_m": round(p_min, 1),
                    "patch_max_m": round(p_max, 1),
                    "nbr_ocean_frac": round(ocean_frac, 2),
                    "score": round(score, 1),
                }

    return best_pos[0], best_pos[1], best_info


def run_generation(
    model_id: str,
    seed: int,
    device: str,
    tile_size: int = 512,
    coarse_radius: int = 16,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load Terrain Diffusion, find a scenic coastal mountain tile, and synthesize it."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    print(f"Loading {model_id} on device={device} (seed={seed})...")

    # terrain-diffusion uses relative paths like `data/global/etopo_10m.tif`
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
            log_mode="info",
        )
        world.to(device)
        world.bind()
    finally:
        os.chdir(orig_cwd)
    timings["load_s"] = round(time.perf_counter() - t0, 2)
    print(f"  Loaded & bound in {timings['load_s']}s")

    with world:
        # 1. Scan coarse planetary map (7.7 km/px) to pick a scenic coastal mountain region
        t1 = time.perf_counter()
        ci0, cj0 = -coarse_radius, -coarse_radius
        coarse_raw = world.coarse[:, ci0 : ci0 + 2 * coarse_radius, cj0 : cj0 + 2 * coarse_radius]
        coarse_map = (coarse_raw[:-1] / coarse_raw[-1:]).detach().cpu().numpy()
        # Channel 0 is signed-sqrt mean elevation -> invert to meters
        c0 = coarse_map[0]
        coarse_elev_m = np.sign(c0) * (c0 ** 2)
        timings["coarse_scan_s"] = round(time.perf_counter() - t1, 2)

        best_ci, best_cj, info = pick_scenic_coastal_tile(coarse_elev_m, ci0, cj0)
        print(
            f"  Coarse scan ({2*coarse_radius}x{2*coarse_radius}) took {timings['coarse_scan_s']}s -> "
            f"selected coarse ({best_ci}, {best_cj}) {info}"
        )

        # 2. Synthesize high-res fine tile (30m/px) + WorldClim climate channels
        scale = int(world.latent_compression) * 32  # 256 fine pixels per coarse pixel
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
            "coarse_ij": [best_ci, best_cj],
            "fine_bbox": [i1, j1, i2, j2],
            "timings": timings,
            "coarse_info": info,
        }
        return elev, climate, meta


def save_overview_plot(elev: np.ndarray, climate: np.ndarray, meta: dict, out_png: Path) -> None:
    """Save 4-panel visual diagnostic of Terrain Diffusion DEM & climate."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=130)
    fig.patch.set_facecolor("#0f141c")

    # 1. Shaded relief DEM
    gy, gx = np.gradient(elev, 30.0)
    sun = np.array([0.65, -0.45, 0.62])
    sun /= np.linalg.norm(sun)
    nz = np.ones_like(gx)
    norm = np.sqrt(gx**2 + gy**2 + nz**2)
    shade = np.clip((-gx * sun[0] - gy * sun[1] + nz * sun[2]) / norm, 0.15, 1.0)

    rgb = np.zeros((*elev.shape, 3), dtype=np.float32)
    ocean = elev <= 0.0
    t_land = np.clip(elev / max(float(np.percentile(elev[~ocean], 98)) if (~ocean).any() else 1000.0, 1.0), 0.0, 1.0)
    # Hypsometric ramp: emerald coast -> forest -> alpine rock -> snow
    rgb[~ocean, 0] = np.interp(t_land[~ocean], [0.0, 0.25, 0.65, 0.88, 1.0], [0.24, 0.30, 0.50, 0.72, 0.96])
    rgb[~ocean, 1] = np.interp(t_land[~ocean], [0.0, 0.25, 0.65, 0.88, 1.0], [0.48, 0.50, 0.48, 0.72, 0.97])
    rgb[~ocean, 2] = np.interp(t_land[~ocean], [0.0, 0.25, 0.65, 0.88, 1.0], [0.22, 0.24, 0.44, 0.74, 0.99])
    rgb[~ocean] *= (0.38 + 0.72 * shade[~ocean, None])
    # Deep-to-shallow ocean
    t_sea = np.clip(-elev[ocean] / 500.0, 0.0, 1.0)
    rgb[ocean, 0] = 0.05 + 0.10 * (1.0 - t_sea)
    rgb[ocean, 1] = 0.18 + 0.28 * (1.0 - t_sea)
    rgb[ocean, 2] = 0.34 + 0.28 * (1.0 - t_sea)

    axes[0, 0].imshow(np.clip(rgb, 0, 1))
    axes[0, 0].set_title(
        f"Terrain Diffusion 30m DEM (seed={meta['seed']}, {meta['device']})\n"
        f"elev {elev.min():.0f}m .. {elev.max():.0f}m | gen {meta['timings']['tile_gen_s']}s",
        color="white", fontsize=10,
    )
    axes[0, 0].axis("off")

    # 2. Raw elevation heatmap
    im1 = axes[0, 1].imshow(elev, cmap="terrain")
    axes[0, 1].set_title("Elevation (meters)", color="white", fontsize=10)
    axes[0, 1].axis("off")
    cb1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb1.ax.get_yticklabels(), color="white", fontsize=8)

    # 3. Annual Mean Temperature (elevation-adjusted!)
    im2 = axes[1, 0].imshow(climate[0], cmap="coolwarm")
    axes[1, 0].set_title("Elevation-Adjusted Mean Temp (°C)", color="white", fontsize=10)
    axes[1, 0].axis("off")
    cb2 = fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.get_yticklabels(), color="white", fontsize=8)

    # 4. Annual Precipitation (mm/yr)
    im3 = axes[1, 1].imshow(climate[2], cmap="YlGnBu")
    axes[1, 1].set_title("Annual Precipitation (mm/yr)", color="white", fontsize=10)
    axes[1, 1].axis("off")
    cb3 = fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cb3.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb3.ax.get_yticklabels(), color="white", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Terrain Diffusion 30m on Apple Silicon")
    parser.add_argument("--model", default="xandergos/terrain-diffusion-30m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        candidates = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    else:
        candidates = [args.device]

    elev = climate = meta = None
    for dev in candidates:
        try:
            elev, climate, meta = run_generation(args.model, args.seed, dev, tile_size=args.size)
            break
        except Exception as exc:
            print(f"Device {dev!r} failed: {exc!r}")
            if dev == candidates[-1]:
                raise
            print("Falling back to next device...")

    assert elev is not None and climate is not None and meta is not None
    elev.tofile(OUT_DIR / "elev_512.f32")
    np.savez_compressed(
        OUT_DIR / "climate_512.npz",
        temp=climate[0],
        temp_season=climate[1],
        precip=climate[2],
        precip_cv=climate[3],
    )
    import json
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    save_overview_plot(elev, climate, meta, OUT_DIR / "terrain_diffusion_overview.png")
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
