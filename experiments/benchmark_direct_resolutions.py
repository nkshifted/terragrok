#!/usr/bin/env python3
"""
Benchmark direct high-resolution Terrain Diffusion 30m synthesis on Apple Silicon MPS:
  1. 1x Baseline:     1024 x 1024 (4.69 m/px over 4.8 km) — High (2-step) & Fast (1-step)
  2. 2x Direct:       2048 x 2048 (2.34 m/px over 4.8 km) — High (2-step) & Fast (1-step)
  3. 4x Direct:       4096 x 4096 (1.17 m/px over 4.8 km) — Fast (1-step) & High (2-step)
  4. 4x 2-Stage Drill: 1024 x 1024 on center 1.2 km x 1.2 km (1.17 m/px)

Also benchmarks CPU hydrology & micro-detail cost (native vs 1024-hydro + high-res carve)
and outputs a 4-panel hillshade visual comparison of the exact same center 1.2 km valley:
  experiments/benchmark_resolutions_compare.png
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, zoom
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from paths import VENDOR_TD, add_vendor_to_syspath  # noqa: E402

add_vendor_to_syspath()

from terrain_diffusion.inference.world_pipeline import WorldPipeline
from export_viewer_worlds import apply_micro_detail_refinement, solve_physical_hydrology
from stage2_server import fuse_laplacian_coastline, _relax_angle_of_repose


def make_coastal_mountain_field(grid_n: int, extent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic tectonic peninsula + fjord + mountain ridge conditioning field."""
    lin = np.linspace(-extent, extent, grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)

    # Diagonal peninsula + dramatic bay in center + alpine ridge
    r_main = np.hypot((xx + 0.12) * 0.9, (yy - 0.08) * 1.15)
    peninsula = 1.15 - r_main * 1.05
    # Fjord bay carving into the lower-right of the center window
    bay = 0.72 * np.exp(-((xx - 0.18) ** 2 / 0.045 + (yy - 0.15) ** 2 / 0.032))
    # Secondary mountain spine running across the center window
    ridge = 0.85 * np.exp(-((xx * 0.75 + yy * 0.65 + 0.06) ** 2) / 0.045) * np.exp(-(r_main**2) / 1.4)
    # Multi-frequency tectonic undulation
    undul = (
        0.24 * np.sin(xx * 4.5 + 0.4) * np.cos(yy * 3.8 - 0.2)
        + 0.14 * np.sin((xx - yy) * 9.0)
        + 0.08 * np.cos((xx + 1.4 * yy) * 15.0)
    )
    raw = peninsula - bay + ridge + undul

    elev = np.where(raw > 0.0, 40.0 + np.power(np.clip(raw, 0.0, 2.5), 1.15) * 1450.0, np.maximum(raw * 1800.0 - 80.0, -2600.0)).astype(np.float32)
    temp = np.clip(14.0 - np.maximum(elev, 0.0) * 0.0055 - yy * 3.0, -10.0, 28.0).astype(np.float32)
    temp_std = np.full_like(temp, 520.0)
    precip = np.clip(980.0 + np.maximum(elev, 0.0) * 0.25 - xx * 180.0, 350.0, 2200.0).astype(np.float32)
    precip_cv = np.full_like(precip, 32.0)
    return elev, temp, temp_std, precip, precip_cv


def run_mps_synth(
    world: WorldPipeline,
    coarse_size: int,
    padding: int,
    onestep: bool,
    seed: int = 777123,
    custom_fields: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    full_coarse = coarse_size + 2 * padding
    if custom_fields is None:
        # Extent = full_coarse / coarse_size so the inner coarse_size cells always cover [-1, +1]
        extent = full_coarse / float(coarse_size)
        cond_elev, cond_temp, cond_tstd, cond_prec, cond_pcv = make_coastal_mountain_field(full_coarse, extent)
    else:
        cond_elev, cond_temp, cond_tstd, cond_prec, cond_pcv = custom_fields

    px_per_raw_cell = 256 // 2  # coarse_pooling=2 -> 128 px/cell
    pi1 = padding * px_per_raw_cell
    pi2 = (padding + coarse_size) * px_per_raw_cell

    world.onestep_latent = onestep
    prev_cwd = os.getcwd()
    os.chdir(VENDOR_TD)
    try:
        world.set_custom_conditioning_import(0, cond_elev, 0, 0, default_value=-1500.0, rebuild=False)
        world.set_custom_conditioning_import(1, cond_temp, 0, 0, default_value=12.0, rebuild=False)
        world.set_custom_conditioning_import(2, cond_tstd, 0, 0, default_value=600.0, rebuild=False)
        world.set_custom_conditioning_import(3, cond_prec, 0, 0, default_value=900.0, rebuild=False)
        world.set_custom_conditioning_import(4, cond_pcv, 0, 0, default_value=35.0, rebuild=False)
        world.seed = seed
        world.rebuild()

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            res = world.get(pi1, pi1, pi2, pi2, with_climate=False)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        elev_raw = res["elev"].detach().cpu().numpy().astype(np.float32)
        del res
        world.empty_cache()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    finally:
        os.chdir(prev_cwd)

    # Also build exact [-1, +1] reference at 256x256 for Laplacian coastline lock
    ref_elev, _, _, _, _ = make_coastal_mountain_field(256, 1.0)
    return elev_raw, ref_elev, dt


def make_hillshade(h: np.ndarray, cell_m: float, label: str) -> Image.Image:
    gy, gx = np.gradient(h, cell_m)
    # Normalize surface normal
    nz = np.ones_like(gx)
    norm = np.sqrt(gx * gx + gy * gy + nz * nz) + 1e-6
    nx, ny, nz = -gx / norm, -gy / norm, nz / norm
    # Sun from NW at 32 deg elevation
    lx, ly, lz = -0.58, -0.58, 0.57
    l_len = np.sqrt(lx * lx + ly * ly + lz * lz)
    shade = np.clip((nx * lx + ny * ly + nz * lz) / l_len, 0.08, 1.0)

    # Simple hypsometric coloring (ocean blue, green lowland, warm rock highland, snow peaks)
    n = h.shape[0]
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    land = h > 0.0
    h_norm = np.clip(h / max(float(np.percentile(h[land], 99.5)) if land.any() else 1000.0, 200.0), 0.0, 1.0)
    slope = np.hypot(gx, gy)

    # Ocean
    rgb[~land] = np.array([28.0, 74.0, 118.0])
    # Land palette
    r = 66.0 + 110.0 * h_norm + 45.0 * np.clip(slope * 0.9, 0.0, 1.0)
    g = 112.0 + 62.0 * h_norm - 18.0 * np.clip(slope * 0.9, 0.0, 1.0)
    b = 58.0 + 78.0 * h_norm + 24.0 * np.clip(slope * 0.9, 0.0, 1.0)
    # High peaks snow tint
    snow = np.clip((h_norm - 0.72) / 0.25, 0.0, 1.0) * np.clip(1.0 - slope * 0.6, 0.0, 1.0)
    r = r * (1.0 - snow) + 235.0 * snow
    g = g * (1.0 - snow) + 240.0 * snow
    b = b * (1.0 - snow) + 248.0 * snow

    rgb[land, 0] = np.clip(r[land] * (0.35 + 0.95 * shade[land]), 0, 255)
    rgb[land, 1] = np.clip(g[land] * (0.35 + 0.95 * shade[land]), 0, 255)
    rgb[land, 2] = np.clip(b[land] * (0.35 + 0.95 * shade[land]), 0, 255)

    img = Image.fromarray(rgb.astype(np.uint8))
    if img.size != (768, 768):
        img = img.resize((768, 768), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 758, 52], fill=(10, 16, 28))
    draw.text((20, 22), label, fill=(238, 246, 255))
    return img


def roughness_metric(crop: np.ndarray, cell_m: float) -> tuple[float, float]:
    """Measure true fine-scale terrain relief energy (high-pass std + p95 slope)."""
    hp = crop - gaussian_filter(crop, sigma=max(1.5, 18.0 / cell_m))
    gy, gx = np.gradient(crop, cell_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    land = crop > 0.0
    if not land.any():
        return 0.0, 0.0
    return float(np.std(hp[land])), float(np.percentile(slope_deg[land], 95))


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== TerraGrok Direct Resolution Benchmark ({device.upper()}, BF16) ===", flush=True)
    t_load0 = time.perf_counter()
    world = WorldPipeline.from_pretrained(
        "xandergos/terrain-diffusion-30m",
        seed=42,
        coarse_pooling=2,
        histogram_raw=[0.0, 0.0, 0.0, 1.0, 1.8],
        torch_compile=False,
        caching_strategy="direct",
        dtype="bf16" if device == "mps" else None,
        decoder_tile_size=512,
        decoder_tile_stride=448,
        T=1,
    )
    world.to(device)
    world.set_cond_snr([0.03, 0.03, 0.03, 0.03, 0.03])
    prev_cwd = os.getcwd()
    os.chdir(VENDOR_TD)
    try:
        world.bind(hdf5_file=None)
    finally:
        os.chdir(prev_cwd)
    print(f"Model loaded in {time.perf_counter() - t_load0:.2f}s", flush=True)

    # Warmup small run so MPS shaders are compiled before timing
    print("Warming up MPS kernels (1024² fast)...", flush=True)
    _, _, dt_warm = run_mps_synth(world, coarse_size=8, padding=64, onestep=True, seed=111)
    print(f"Warmup done in {dt_warm:.2f}s\n", flush=True)

    results: dict[str, dict] = {}
    configs = [
        ("1x_high", 8, 64, False),
        ("2x_high", 16, 64, False),
        ("4x_fast", 32, 64, True),
    ]

    crops: dict[str, tuple[np.ndarray, float, str]] = {}
    base_1x_fused = None

    for name, csize, pad, onestep in configs:
        out_n = csize * 128
        m_px = 4800.0 / out_n
        print(f"Running {name:8s} ({out_n}x{out_n}, {m_px:.2f} m/px, onestep={onestep})...", flush=True)
        elev_raw, ref_elev, t_mps = run_mps_synth(world, coarse_size=csize, padding=pad, onestep=onestep)
        t_fuse0 = time.perf_counter()
        ref_up = zoom(ref_elev, out_n / ref_elev.shape[0], order=3).astype(np.float32)
        elev_fused = fuse_laplacian_coastline(elev_raw, ref_up, fidelity="exact")
        t_fuse = time.perf_counter() - t_fuse0

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        # Extract center 1/4 x 1/4 (= 1.2 km x 1.2 km valley)
        c0 = (out_n * 3) // 8
        c1 = (out_n * 5) // 8
        crop = elev_fused[c0:c1, c0:c1].copy()
        hp_std, p95_slope = roughness_metric(crop, m_px)
        results[name] = {
            "out_n": out_n,
            "m_px": m_px,
            "t_mps": t_mps,
            "t_fuse": t_fuse,
            "rss_mb": rss_mb,
            "hp_std": hp_std,
            "p95_slope": p95_slope,
        }
        print(
            f"  -> MPS={t_mps:5.2f}s | Laplacian={t_fuse:4.2f}s | HP-std={hp_std:5.1f}m | p95-slope={p95_slope:4.1f}° | RSS={rss_mb:.0f}MB",
            flush=True,
        )
        if name == "1x_high":
            base_1x_fused = elev_fused
            crops["1x_high"] = (crop, m_px, f"A. Current 1x (1024², 4.69 m/px) — MPS {t_mps:.1f}s")
        elif name == "2x_high":
            crops["2x_high"] = (crop, m_px, f"B. Direct 2x (2048², 2.34 m/px) — MPS {t_mps:.1f}s")
        elif name == "4x_high":
            crops["4x_high"] = (crop, m_px, f"C. Direct 4x (4096², 1.17 m/px) — MPS {t_mps:.1f}s")
        elif name == "4x_fast" and "4x_high" not in crops:
            crops["4x_fast"] = (crop, m_px, f"C. Direct 4x Fast (4096², 1.17 m/px) — MPS {t_mps:.1f}s")

        del elev_raw, ref_up
        if name != "1x_high":
            del elev_fused
        gc.collect()

    # Also run current 2-stage 4x drill-down on the exact center 1.2 km x 1.2 km crop of base_1x_fused!
    print("\nRunning 2-Stage 4x Drill-Down (1024² on center 1.2km x 1.2km)...", flush=True)
    assert base_1x_fused is not None
    # Build 136x136 conditioning around center 256x256 crop of 1024x1024 parent (exact stage2_server.py drill logic)
    pad = 64
    csize = 8
    full_c = csize + 2 * pad
    # Center 1/4 of [-1, 1] is [-0.25, +0.25], so extent = (full_c / csize) * 0.25
    drill_extent = (full_c / float(csize)) * 0.25
    d_fields = make_coastal_mountain_field(full_c, drill_extent)
    # Replace inner 8x8 with exact downsampled parent 1x center crop!
    parent_crop = base_1x_fused[384:640, 384:640]
    d_elev = d_fields[0].copy()
    d_elev[pad : pad + csize, pad : pad + csize] = zoom(parent_crop, csize / parent_crop.shape[0], order=1)
    drill_raw, _, t_mps_drill = run_mps_synth(
        world, coarse_size=8, padding=64, onestep=False, seed=777124, custom_fields=(d_elev, *d_fields[1:])
    )
    parent_up = zoom(parent_crop, 1024 / parent_crop.shape[0], order=3).astype(np.float32)
    drill_fused = fuse_laplacian_coastline(drill_raw, parent_up, fidelity="exact")
    hp_d, p95_d = roughness_metric(drill_fused, 1.171875)
    print(f"  -> 2-Stage 4x Drill: MPS={t_mps_drill:5.2f}s | HP-std={hp_d:5.1f}m | p95-slope={p95_d:4.1f}°", flush=True)
    crops["drill_4x"] = (drill_fused, 1.171875, f"D. 2-Stage 4x Drill (1024², 1.17 m/px) — MPS {t_mps_drill:.1f}s")

    # Benchmark CPU post-processing (hydrology + micro-detail) at 1024² vs 2048²!
    print("\nBenchmarking CPU Post-Processing (Hydrology + Micro-Detail)...", flush=True)
    h1024 = np.where(base_1x_fused > 0, base_1x_fused * 0.22, np.maximum(base_1x_fused * 0.05, -50.0)).astype(np.float32)
    runoff1024 = np.ones_like(h1024)

    t0 = time.perf_counter()
    h1024_r = _relax_angle_of_repose(h1024, cell_size_m=4800.0 / 1024, max_slope_deg=52.0, iterations=14)
    h1024_m = apply_micro_detail_refinement(h1024_r, world_span_m=4800.0, seed=42)
    t_micro_1024 = time.perf_counter() - t0
    t0 = time.perf_counter()
    solve_physical_hydrology(
        elev=h1024_m,
        world_span_m=4800.0,
        min_lake_cells=48,
        min_lake_depth_m=0.55,
        max_spring_rivers=14,
        runoff_map=runoff1024,
        max_lakes=12,
        min_lake_runoff=0.18,
    )
    t_hydro_1024 = time.perf_counter() - t0
    print(f"  1024² CPU: micro+repose={t_micro_1024:.2f}s | hydrology={t_hydro_1024:.2f}s | total={t_micro_1024 + t_hydro_1024:.2f}s", flush=True)

    h2048 = zoom(h1024, 2.0, order=3).astype(np.float32)
    runoff2048 = np.ones_like(h2048)
    t0 = time.perf_counter()
    h2048_r = _relax_angle_of_repose(h2048, cell_size_m=4800.0 / 2048, max_slope_deg=52.0, iterations=14)
    h2048_m = apply_micro_detail_refinement(h2048_r, world_span_m=4800.0, seed=42)
    t_micro_2048 = time.perf_counter() - t0
    t0 = time.perf_counter()
    solve_physical_hydrology(
        elev=h2048_m,
        world_span_m=4800.0,
        min_lake_cells=192,
        min_lake_depth_m=0.55,
        max_spring_rivers=14,
        runoff_map=runoff2048,
        max_lakes=12,
        min_lake_runoff=0.18,
    )
    t_hydro_2048 = time.perf_counter() - t0
    print(f"  2048² CPU (native): micro+repose={t_micro_2048:.2f}s | hydrology={t_hydro_2048:.2f}s | total={t_micro_2048 + t_hydro_2048:.2f}s", flush=True)

    # Render 2x2 visual comparison of the exact center 1.2 km x 1.2 km valley!
    panel_a = make_hillshade(*crops["1x_high"])
    panel_b = make_hillshade(*crops["2x_high"])
    panel_c = make_hillshade(*(crops["4x_high"] if "4x_high" in crops else crops["4x_fast"]))
    panel_d = make_hillshade(*crops["drill_4x"])

    grid = Image.new("RGB", (768 * 2 + 12, 768 * 2 + 12), (12, 16, 24))
    grid.paste(panel_a, (0, 0))
    grid.paste(panel_b, (768 + 12, 0))
    grid.paste(panel_c, (0, 768 + 12))
    grid.paste(panel_d, (768 + 12, 768 + 12))
    out_png = REPO_ROOT / "experiments" / "benchmark_resolutions_compare.png"
    grid.save(out_png)
    print(f"\nSaved visual comparison to {out_png}", flush=True)


if __name__ == "__main__":
    main()
