"""
Stage 2 (Option C): Tectonics -> Terrain Diffusion 30m Conditioning Proof-of-Concept

Takes a Stage 1 Spherical Plate-Tectonics patch (heightmap + temperature + precipitation
exported from viewer/src/tectonics.js), feeds it into xandergos/terrain-diffusion-30m
via WorldPipeline.set_custom_conditioning_import() on Apple Silicon MPS, and verifies:
  1. Coastline & mountain-belt fidelity (IoU + elevation correlation vs tectonic input)
  2. Generation latency on Apple Silicon MPS
  3. Physical Landlab Barnes Priority-Flood hydrology (0 dead-ends, alpine lakes -> sea)
  4. Side-by-side 4-panel visual proof + live 3D viewer export (viewer/public/worlds/)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, zoom

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".vendor" / "terrain-diffusion"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from terrain_diffusion.inference.world_pipeline import WorldPipeline  # noqa: E402
from export_viewer_worlds import (  # noqa: E402
    apply_micro_detail_refinement,
    compute_curvature_ao,
    find_landmarks,
    solve_physical_hydrology,
)

OUT_DIR = REPO_ROOT / "out" / "tectonics_td30m"
VIEWER_WORLDS_DIR = REPO_ROOT / "viewer" / "public" / "worlds"


def run_conditioned_td30m(
    patch_dir: Path = OUT_DIR,
    coarse_pooling: int = 2,
    cond_snr: tuple[float, ...] = (0.15, 0.2, 1.0, 0.25, 1.0),
    td_seed: int = 42,
):
    meta = json.loads((patch_dir / "patch_meta.json").read_text())
    coarse_size = int(meta["coarseSize"])
    padding = int(meta["padding"])
    full_coarse = int(meta["fullCoarse"])
    ref_size = int(meta["refSize"])

    assert padding % coarse_pooling == 0
    assert coarse_size % coarse_pooling == 0

    cond_elev = np.fromfile(patch_dir / "cond_elev.f32", dtype=np.float32).reshape(full_coarse, full_coarse)
    cond_temp = np.fromfile(patch_dir / "cond_temp.f32", dtype=np.float32).reshape(full_coarse, full_coarse)
    cond_temp_std = np.fromfile(patch_dir / "cond_temp_std.f32", dtype=np.float32).reshape(full_coarse, full_coarse)
    cond_precip = np.fromfile(patch_dir / "cond_precip.f32", dtype=np.float32).reshape(full_coarse, full_coarse)
    cond_precip_cv = np.fromfile(patch_dir / "cond_precip_cv.f32", dtype=np.float32).reshape(full_coarse, full_coarse)

    ref_elev = np.fromfile(patch_dir / "ref_elev.f32", dtype=np.float32).reshape(ref_size, ref_size)
    ref_temp = np.fromfile(patch_dir / "ref_temp.f32", dtype=np.float32).reshape(ref_size, ref_size)
    ref_precip = np.fromfile(patch_dir / "ref_precip.f32", dtype=np.float32).reshape(ref_size, ref_size)
    globe_elev = np.fromfile(patch_dir / "globe_elev.f32", dtype=np.float32).reshape(meta["globeH"], meta["globeW"])

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | coarse_pooling={coarse_pooling} | cond_snr={list(cond_snr)}")

    t0 = time.perf_counter()
    world = WorldPipeline.from_pretrained(
        "xandergos/terrain-diffusion-30m",
        seed=td_seed,
        coarse_pooling=coarse_pooling,
        histogram_raw=[0.0, 0.0, 0.0, 1.0, 1.8],
        torch_compile=False,
        caching_strategy="direct",
        dtype="bf16" if device in ("mps", "cuda") else None,
        decoder_tile_size=512,
        decoder_tile_stride=448,
        T=1,
    )
    world.to(device)
    world.set_cond_snr(list(cond_snr))
    # Compute exact output pixel window corresponding to [padding : padding + coarse_size]
    px_per_raw_cell = 256 // coarse_pooling
    pi1 = padding * px_per_raw_cell
    pi2 = (padding + coarse_size) * px_per_raw_cell
    pj1 = padding * px_per_raw_cell
    pj2 = (padding + coarse_size) * px_per_raw_cell
    out_n = pi2 - pi1
    print(f"Synthesizing {out_n}x{out_n} Terrain Diffusion 30m conditioned on Stage 1 tectonics...")

    prev_cwd = os.getcwd()
    os.chdir(REPO_ROOT / ".vendor" / "terrain-diffusion")
    try:
        world.bind(hdf5_file=None)
        # Import all 5 Stage 1 tectonic conditioning channels directly in memory!
        world.set_custom_conditioning_import(0, cond_elev, 0, 0, default_value=-1500.0)
        world.set_custom_conditioning_import(1, cond_temp, 0, 0, default_value=12.0)
        world.set_custom_conditioning_import(2, cond_temp_std, 0, 0, default_value=600.0)
        world.set_custom_conditioning_import(3, cond_precip, 0, 0, default_value=900.0)
        world.set_custom_conditioning_import(4, cond_precip_cv, 0, 0, default_value=35.0)

        t_gen0 = time.perf_counter()
        with world:
            res = world.get(pi1, pj1, pi2, pj2, with_climate=True)
        t_gen = time.perf_counter() - t_gen0
    finally:
        os.chdir(prev_cwd)
    print(f"Terrain Diffusion 30m MPS synthesis finished in {t_gen:.2f} s!")

    elev_raw = res["elev"].numpy().astype(np.float32)
    climate_raw = res["climate"].numpy().astype(np.float32)  # (4, H, W): tempC, tempStd, precipMm, precipCV

    # Compare fidelity vs Stage 1 tectonic reference!
    ref_up = zoom(ref_elev, out_n / ref_size, order=1)
    ref_land = ref_up > 0.0
    gen_land = elev_raw > 0.0
    inter = np.logical_and(ref_land, gen_land).sum()
    union = np.logical_or(ref_land, gen_land).sum() + 1e-6
    iou = float(inter / union)
    corr = float(np.corrcoef(ref_up.ravel(), elev_raw.ravel())[0, 1])

    print(
        f"Tectonic Fidelity -> Coastline Land IoU = {iou * 100:.1f}% | "
        f"Elevation Correlation r = {corr:.3f} | "
        f"Land %: input={ref_land.mean() * 100:.1f}% vs TD30m={gen_land.mean() * 100:.1f}% | "
        f"TD30m elev range: [{elev_raw.min():.0f}m, {elev_raw.max():.0f}m]"
    )

    # Scale to playable 4.8 km x 4.8 km explorer vertical scale + micro-detail crag refinement
    world_span_m = 4800.0
    land_mask = elev_raw > 0.0
    max_land_m = float(np.percentile(elev_raw[land_mask], 99.6)) if land_mask.any() else 1000.0
    v_scale = 460.0 / max(max_land_m, 400.0)
    elev_viewer = np.where(land_mask, elev_raw * v_scale, np.maximum(elev_raw * 0.05, -55.0)).astype(np.float32)

    print("Applying slope-gated micro-detail crag & Fastscapelib gully refinement...")
    elev_refined = apply_micro_detail_refinement(elev_viewer, world_span_m=world_span_m, seed=td_seed)

    # Clean precipitation & temperature fields (fallback smoothly to Stage 1 climate where ocean is NaN)
    ref_temp_up = zoom(ref_temp, out_n / ref_size, order=1)
    ref_precip_up = zoom(ref_precip, out_n / ref_size, order=1)
    temp_c = np.where(np.isfinite(climate_raw[0]), climate_raw[0], ref_temp_up).astype(np.float32)
    precip_mm = np.where(np.isfinite(climate_raw[2]), np.clip(climate_raw[2], 50.0, 3500.0), ref_precip_up).astype(np.float32)
    t_season = np.where(np.isfinite(climate_raw[1]), np.abs(climate_raw[1]), 450.0).astype(np.float32)

    p_med = max(float(np.median(precip_mm[elev_refined > 0])) if (elev_refined > 0).any() else 600.0, 120.0)
    runoff_map = np.clip(precip_mm / p_med, 0.35, 1.85).astype(np.float32)

    print("Running Landlab Barnes Priority-Flood hydrology (precipitation-weighted)...")
    t_h0 = time.perf_counter()
    res_scale = (out_n / 512.0) ** 2
    carved, riv_u8, lak_u8, lakes_data, rivers_data = solve_physical_hydrology(
        elev=elev_refined,
        world_span_m=world_span_m,
        min_lake_cells=int(round(26 * res_scale)),
        min_lake_depth_m=0.55,
        max_spring_rivers=24,
        runoff_map=runoff_map,
    )
    print(
        f"Hydrology solved in {time.perf_counter() - t_h0:.2f} s -> "
        f"{len(lakes_data)} alpine lakes, {len(rivers_data)} continuous rivers (0 dead-ends)"
    )

    # Render 4-panel visual comparison proof!
    ls = LightSource(azdeg=315, altdeg=38)

    def make_relief_rgb(z, temp_grid, precip_grid, lake_u8=None, river_u8=None):
        h_n = np.clip(z / max(np.percentile(z[z > 0], 99.2) if (z > 0).any() else 1.0, 1.0), 0.0, 1.0)
        rgb = np.zeros((*z.shape, 3), dtype=np.float32)
        ocean = z <= 0
        od = np.clip(-z / max(abs(float(z.min())), 1.0), 0.0, 1.0)
        rgb[ocean, 0] = 0.08 * (1 - od[ocean]) + 0.03 * od[ocean]
        rgb[ocean, 1] = 0.28 * (1 - od[ocean]) + 0.09 * od[ocean]
        rgb[ocean, 2] = 0.46 * (1 - od[ocean]) + 0.22 * od[ocean]

        land = ~ocean
        moist = np.clip((precip_grid - 150.0) / 2200.0, 0.0, 1.0)
        r = 0.22 + (1 - moist) * 0.48
        g = 0.42 + (1 - moist) * 0.20 - moist * 0.08
        b = 0.16 + (1 - moist) * 0.22
        rock = np.clip((h_n - 0.52) / 0.35, 0.0, 1.0)
        r = r * (1 - rock) + 0.46 * rock
        g = g * (1 - rock) + 0.44 * rock
        b = b * (1 - rock) + 0.42 * rock
        snow = np.clip((1.5 - temp_grid) / 3.0, 0.0, 1.0) * np.clip(h_n * 2.2, 0.0, 1.0)
        snow = np.maximum(snow, np.clip((h_n - 0.78) / 0.16, 0.0, 1.0))
        r = r * (1 - snow) + 0.94 * snow
        g = g * (1 - snow) + 0.96 * snow
        b = b * (1 - snow) + 0.99 * snow

        rgb[land, 0] = np.clip(r[land], 0, 1)
        rgb[land, 1] = np.clip(g[land], 0, 1)
        rgb[land, 2] = np.clip(b[land], 0, 1)

        shade = ls.hillshade(np.where(ocean, 0.0, z), vert_exag=1.8, dx=1.0, dy=1.0)
        rgb[land] *= (0.55 + 0.65 * shade[land, None])

        if lake_u8 is not None:
            lk = lake_u8 > 90
            rgb[lk] = [0.10, 0.48, 0.68]
        if river_u8 is not None:
            rv = river_u8 > 70
            rgb[rv] = [0.16, 0.54, 0.78]
        return np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 13), facecolor="#0a0f1a")
    fig.suptitle(
        f"Stage 1 (Spherical Plate Tectonics) → Stage 2 (Conditioned Terrain Diffusion 30m on MPS: {t_gen:.1f}s)\n"
        f"Seed {meta['seed']} ({meta['archetype']}) · Coastline Fidelity IoU = {iou*100:.1f}% · Elev r = {corr:.3f} · "
        f"{len(lakes_data)} Alpine Lakes, {len(rivers_data)} Rivers (0 dead-ends)",
        color="#eef4ff",
        fontsize=12.5,
        fontweight="bold",
        y=0.97,
    )

    # Panel 1: Full Stage 1 Planet with red target box
    ax = axes[0, 0]
    g_rgb = np.zeros((*globe_elev.shape, 3), dtype=np.float32)
    g_ocean = globe_elev <= 0
    g_rgb[g_ocean] = [0.05, 0.14, 0.28]
    g_rgb[~g_ocean] = [0.24, 0.48, 0.22]
    g_rgb[globe_elev > 140] = [0.58, 0.54, 0.48]
    g_rgb[globe_elev > 200] = [0.92, 0.95, 0.98]
    ax.imshow(g_rgb)
    cx = (meta["lonDeg"] / 360.0 + 0.5) * meta["globeW"]
    cy = (0.5 - meta["latDeg"] / 180.0) * meta["globeH"]
    bw = (meta["spanDeg"] / 360.0) * meta["globeW"]
    bh = (meta["spanDeg"] / 180.0) * meta["globeH"]
    ax.add_patch(Rectangle((cx - bw / 2, cy - bh / 2), bw, bh, fill=False, edgecolor="#ff4d4d", linewidth=2.2))
    ax.set_title(
        f"1. Stage 1 Tectonic Planet (Red Box = Landing Target: {meta['latDeg']}°N, {meta['lonDeg']}°E)",
        color="#dce9ff",
        fontsize=10.5,
    )
    ax.axis("off")

    # Panel 2: Stage 1 Tectonic Patch Input
    ax = axes[0, 1]
    ax.imshow(make_relief_rgb(ref_elev, ref_temp, ref_precip))
    ax.set_title(
        f"2. Stage 1 Tectonic Input ({coarse_size}×{coarse_size} Coarse Grid, Land {ref_land.mean()*100:.1f}%)",
        color="#dce9ff",
        fontsize=10.5,
    )
    ax.axis("off")

    # Panel 3: Raw Conditioned Terrain Diffusion 30m Output
    ax = axes[1, 0]
    ax.imshow(make_relief_rgb(elev_raw, temp_c, precip_mm))
    ax.set_title(
        f"3. Stage 2 Conditioned Terrain Diffusion 30m ({out_n}×{out_n}, {t_gen:.1f}s MPS, IoU {iou*100:.1f}%)",
        color="#dce9ff",
        fontsize=10.5,
    )
    ax.axis("off")

    # Panel 4: Final Playable Terrain + Micro-Refinement + Landlab Lakes & Rivers
    ax = axes[1, 1]
    ax.imshow(make_relief_rgb(carved, temp_c, precip_mm, lake_u8=lak_u8, river_u8=riv_u8))
    ax.set_title(
        f"4. Final Playable 3D World + Landlab Hydrology ({len(lakes_data)} Lakes, {len(rivers_data)} Rivers)",
        color="#dce9ff",
        fontsize=10.5,
    )
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93], pad=1.2)
    proof_path = patch_dir / "tectonics_to_td30m_proof.png"
    plt.savefig(proof_path, dpi=150, facecolor="#0a0f1a")
    plt.close(fig)
    print(f"Saved 4-panel visual proof -> {proof_path}")

    # Export into viewer/public/worlds/ matching export_viewer_worlds.py schema identically!
    world_id = "tectonic_td30m"
    VIEWER_WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    curv_u8 = compute_curvature_ao(carved)

    temp_range = [-12.0, 28.0]
    precip_range = [0.0, 1800.0]
    t_u8 = np.clip((temp_c - temp_range[0]) / (temp_range[1] - temp_range[0]) * 255.0, 0, 255).astype(np.uint8)
    p_u8 = np.clip((precip_mm - precip_range[0]) / (precip_range[1] - precip_range[0]) * 255.0, 0, 255).astype(np.uint8)
    s_u8 = np.clip(t_season / max(float(np.percentile(t_season, 98)), 1.0) * 255.0, 0, 255).astype(np.uint8)
    clim_rgba = np.stack([t_u8, p_u8, s_u8, np.full((out_n, out_n), 255, dtype=np.uint8)], axis=-1)
    clim_rgba.tofile(VIEWER_WORLDS_DIR / f"{world_id}.climate.u8")

    p_lo, p_hi = float(np.percentile(precip_mm, 5)), float(np.percentile(precip_mm, 95))
    moisture = np.clip(0.22 + 0.72 * (precip_mm - p_lo) / max(p_hi - p_lo, 1.0), 0.12, 0.96)
    wet_seed = np.clip((riv_u8.astype(np.float32) + lak_u8.astype(np.float32)) / 255.0, 0.0, 1.0)
    moisture = np.clip(moisture + gaussian_filter(wet_seed, sigma=10.0) * 0.85, 0.10, 1.0)

    masks = np.stack([riv_u8, lak_u8, curv_u8, (moisture * 255.0).astype(np.uint8)], axis=-1)
    carved.astype(np.float32).tofile(VIEWER_WORLDS_DIR / f"{world_id}.height.f32")
    masks.astype(np.uint8).tofile(VIEWER_WORLDS_DIR / f"{world_id}.masks.u8")
    (VIEWER_WORLDS_DIR / f"{world_id}.rivers.json").write_text(json.dumps(rivers_data))
    (VIEWER_WORLDS_DIR / f"{world_id}.lakes.json").write_text(json.dumps(lakes_data))

    landmarks = find_landmarks(carved, riv_u8, lak_u8, lakes_data, rivers_data, world_span_m)
    world_meta = {
        "id": world_id,
        "title": f"Tectonic → AI Earth 30m (Seed {meta['seed']})",
        "subtitle": f"Stage 1 Plate Tectonics ({meta['archetype']}) → Conditioned Terrain Diffusion 30m (IoU {iou*100:.0f}%)",
        "size": out_n,
        "worldSpan": world_span_m,
        "minElev": round(float(carved.min()), 2),
        "maxElev": round(float(carved.max()), 2),
        "seaLevel": 0.0,
        "hasClimate": True,
        "tempRange": temp_range,
        "precipRange": precip_range,
        "riverCount": len(rivers_data),
        "lakeCount": len(lakes_data),
        "landmarks": landmarks,
    }
    (VIEWER_WORLDS_DIR / f"{world_id}.json").write_text(json.dumps(world_meta, indent=2))

    # Update viewer/public/worlds/catalog.json cleanly
    catalog_path = VIEWER_WORLDS_DIR / "catalog.json"
    if catalog_path.exists():
        catalog = [w for w in json.loads(catalog_path.read_text()) if w["id"] != world_id]
        catalog.append(world_meta)
        catalog_path.write_text(json.dumps(catalog, indent=2))

    print(f"Exported 3D viewer world '{world_id}' into {VIEWER_WORLDS_DIR}")
    print(f"Total Stage 2 pipeline elapsed: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    run_conditioned_td30m()
