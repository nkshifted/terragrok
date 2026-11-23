#!/usr/bin/env python3
"""
Zero-Seam Spherical Planet Generator (Approach 1: Terrain Diffusion 30m Continent Stamping
onto a 6-Face Cube-Sphere + Landlab Barnes Lake-Spillway-to-Sea Hydrology).

How zero-seam stitching works:
  1. Synthesize 6 distinct Earth-grade 512x512 regions with `xandergos/terrain-diffusion-30m`
     (coarse_pooling=2, high-relief beauty histogram) on Apple Silicon MPS / CPU.
  2. Assign each region to one face of a Cube-Sphere (+X, -X, +Y North Pole, -Y South Pole,
     +Z, -Z) and apply an organic FBM-warped continental-shelf falloff near each face's outer
     rim so every continent/archipelago drops cleanly below sea level before the face edge ->
     MATHEMATICALLY ZERO SEAMS across all 12 cube edges and ZERO polar pinching at the poles!
  3. Run Landlab Barnes PriorityFloodFlowRouter on each continent to solve real flat alpine
     lakes and continuous mountain-to-sea rivers (0 dead-ends!), then project every river
     point and lake cell directly onto the 3D sphere.
  4. Combine planetary latitude (|y| on the unit sphere) with elevation lapse rate and
     Terrain Diffusion precipitation to produce authentic equatorial, temperate, boreal,
     and polar ice-cap biomes, plus a tilted magnetic pole axis for real compass navigation!
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, FractalType, NoiseType
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_TD = REPO_ROOT / ".vendor" / "terrain-diffusion"
if VENDOR_TD.exists():
    sys.path.insert(0, str(VENDOR_TD))

# Reuse our verified micro-detail + Landlab Barnes lake-to-sea solver!
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from export_viewer_worlds import (  # noqa: E402
    compute_curvature_ao,
    resample_array,
    solve_physical_hydrology,
)
from terrain_diffusion.inference.world_pipeline import WorldPipeline  # noqa: E402

OUT_DEBUG = REPO_ROOT / "out" / "planet"
OUT_VIEWER = REPO_ROOT / "viewer" / "public" / "planet"

# Standard right-handed Cube-Sphere face basis: (name, normal, right_u, down_v)
# Matches Three.js cube-sphere construction identically!
CUBE_FACES = [
    ("px", (1, 0, 0), (0, 0, -1), (0, -1, 0)),   # Equatorial East
    ("nx", (-1, 0, 0), (0, 0, 1), (0, -1, 0)),   # Equatorial West
    ("py", (0, 1, 0), (1, 0, 0), (0, 0, 1)),     # North Pole
    ("ny", (0, -1, 0), (1, 0, 0), (0, 0, -1)),   # South Pole
    ("pz", (0, 0, 1), (1, 0, 0), (0, -1, 0)),    # Equatorial Front
    ("nz", (0, 0, -1), (-1, 0, 0), (0, -1, 0)),  # Equatorial Back
]


def face_uv_to_unit_sphere(face_idx: int, u_grid: np.ndarray, v_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map face-local coordinates u, v in [-1, 1] to unit-sphere (sx, sy, sz)."""
    _, n, ru, dv = CUBE_FACES[face_idx]
    cx = n[0] + u_grid * ru[0] + v_grid * dv[0]
    cy = n[1] + u_grid * ru[1] + v_grid * dv[1]
    cz = n[2] + u_grid * ru[2] + v_grid * dv[2]
    inv_len = 1.0 / np.sqrt(cx * cx + cy * cy + cz * cz)
    return cx * inv_len, cy * inv_len, cz * inv_len


def pick_six_distinct_land_regions(
    coarse_elev_m: np.ndarray,
    ci0: int,
    cj0: int,
    min_sep: int = 4,
) -> list[tuple[int, int, float]]:
    """
    Pick 6 well-separated 2x2 coarse tiles (each -> 512x512 fine Terrain Diffusion tile)
    with strong land relief so every Cube-Sphere face gets a unique Earth-like continent!
    """
    h, w = coarse_elev_m.shape
    candidates: list[tuple[float, int, int]] = []
    for r in range(2, h - 4):
        for c in range(2, w - 4):
            patch = coarse_elev_m[r : r + 2, c : c + 2]
            p_max = float(patch.max())
            p_mean = float(patch.mean())
            if p_max < 220.0 or p_mean < -120.0:
                continue
            # Prefer high relief & strong land presence
            score = p_max * 0.9 + max(p_mean, 0.0) * 1.4 + float(patch.std()) * 1.2
            candidates.append((score, ci0 + r, cj0 + c))

    candidates.sort(key=lambda x: -x[0])
    chosen: list[tuple[int, int, float]] = []
    for score, ci, cj in candidates:
        if all(max(abs(ci - oci), abs(cj - ocj)) >= min_sep for oci, ocj, _ in chosen):
            chosen.append((ci, cj, round(score, 1)))
            if len(chosen) == 6:
                break

    # Fallback if coarse map is sparse
    while len(chosen) < 6:
        idx = len(chosen)
        chosen.append((ci0 + 4 + idx * 3, cj0 + 4 + (idx % 3) * 4, 0.0))
    return chosen


def shape_continent_mask(size: int, seed: int, is_polar: bool = False) -> np.ndarray:
    """
    Build an organic, FBM-warped continent envelope in [0..1] that equals 1.0 over the
    interior (~65-75% of the face) and smoothly drops to 0.0 well before the face border.
    Guarantees every face border sits cleanly underwater -> ZERO cube-edge seams!
    """
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)

    # Organic domain warp so continents have natural peninsulas, bays, and archipelagos
    fn = FastNoiseLite(seed=seed)
    fn.noise_type = NoiseType.NoiseType_OpenSimplex2
    fn.fractal_type = FractalType.FractalType_FBm
    fn.fractal_octaves = 4
    fn.frequency = 1.65

    coords = np.vstack([xx.ravel(), yy.ravel()])
    warp = fn.gen_from_coords(coords).reshape(size, size)

    # Slightly elliptical/organic shape per face, wider for polar ice/fjords
    r_euclid = np.hypot(xx, yy)
    r_cheb = np.maximum(np.abs(xx), np.abs(yy))
    r_blend = 0.65 * r_euclid + 0.35 * r_cheb

    # Perturb radius with organic fractal bays/headlands, then force hard drop to 0 near edge (>0.88)
    inner = 0.42 if is_polar else 0.40
    outer = 0.86 if is_polar else 0.84
    r_warped = r_blend + warp * 0.22
    mask = 1.0 - np.clip((r_warped - inner) / (outer - inner), 0.0, 1.0)
    # Smoothstep
    mask = mask * mask * (3.0 - 2.0 * mask)

    # Strict safety ocean moat on the outermost 5% rim so cube edges are 100% seamless ocean!
    edge_fade = 1.0 - np.clip((r_cheb - 0.84) / 0.11, 0.0, 1.0)
    edge_fade = edge_fade * edge_fade * (3.0 - 2.0 * edge_fade)
    return (mask * edge_fade).astype(np.float32)


def generate_planet(
    seed: int = 42,
    planet_radius_m: float = 3000.0,
    face_export_size: int = 256,
    device: str = "auto",
) -> None:
    """Generate a full 6-face zero-seam Earth-like sphere using Terrain Diffusion 30m."""
    OUT_DEBUG.mkdir(parents=True, exist_ok=True)
    OUT_VIEWER.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = device

    t0 = time.perf_counter()
    print(f"Generating 6-face Terrain Diffusion 30m Planet (seed={seed}, R={planet_radius_m:.0f}m, dev={dev})...")

    orig_cwd = os.getcwd()
    if VENDOR_TD.exists():
        os.chdir(VENDOR_TD)
    try:
        world = WorldPipeline.from_pretrained(
            "xandergos/terrain-diffusion-30m",
            seed=seed,
            caching_strategy="direct",
            torch_compile=False,
            latents_batch_size=4,
            coarse_pooling=2,
            histogram_raw=[0.0, 0.0, 0.0, 1.0, 1.8],
            log_mode="info",
        )
        world.to(dev)
        world.bind()
    finally:
        os.chdir(orig_cwd)
    print(f"  Model loaded in {time.perf_counter() - t0:.2f}s")

    # Face arc length on sphere: pi/2 * R (~4,712 m per face for R=3000 m!)
    face_span_m = (math.pi * 0.5) * planet_radius_m
    max_elev_m = planet_radius_m * 0.072  # ~216 m dramatic mountains on a 6 km planet!

    raw_tiles: list[tuple[np.ndarray, np.ndarray]] = []
    with world:
        t1 = time.perf_counter()
        coarse_r = 22
        ci0, cj0 = -coarse_r, -coarse_r
        coarse_raw = world.coarse[:, ci0 : ci0 + 2 * coarse_r, cj0 : cj0 + 2 * coarse_r]
        coarse_map = (coarse_raw[:-1] / coarse_raw[-1:]).detach().cpu().numpy()
        c0 = coarse_map[0]
        coarse_elev_m = np.sign(c0) * (c0 ** 2)

        picks = pick_six_distinct_land_regions(coarse_elev_m, ci0, cj0, min_sep=4)
        print(f"  Coarse planetary scan ({time.perf_counter() - t1:.2f}s) -> 6 continent seeds: {picks}")

        scale = int(world.latent_compression) * 32  # 256 fine pixels per coarse pixel
        t2 = time.perf_counter()
        for f_idx, (ci, cj, _) in enumerate(picks):
            i1, j1 = ci * scale, cj * scale
            out = world.get(i1, j1, i1 + 512, j1 + 512, with_climate=True)
            if dev == "mps":
                torch.mps.synchronize()
            elev = out["elev"].detach().cpu().numpy().astype(np.float32)
            clim = out["climate"].detach().cpu().numpy().astype(np.float32)
            raw_tiles.append((elev, clim))
            print(
                f"    Face {CUBE_FACES[f_idx][0]}: TD elev=[{elev.min():.0f}..{elev.max():.0f}m], "
                f"precip=[{clim[2].min():.0f}..{clim[2].max():.0f}mm]"
            )
        print(f"  Synthesized all 6 Terrain Diffusion continents in {time.perf_counter() - t2:.2f}s!")

    # Process each of the 6 Cube-Sphere faces cleanly -> ZERO edge seams + physical rivers & lakes!
    faces_meta = []
    all_rivers_3d = []
    all_lakes_3d = []
    atlas_images = []

    lin_f = np.linspace(-1.0, 1.0, face_export_size, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(lin_f, lin_f)

    for f_idx, (fname, _, _, _) in enumerate(CUBE_FACES):
        is_polar = fname in ("py", "ny")
        raw_elev, raw_clim = raw_tiles[f_idx]

        # Resample to face export size (256x256 per face -> 393k total pixels across 6 faces!)
        e_res = resample_array(raw_elev, face_export_size, is_mask=False).astype(np.float32)
        p_res = np.maximum(resample_array(raw_clim[2], face_export_size, is_mask=False).astype(np.float32), 0.0)

        # Normalize positive land relief of this AI continent to [0 .. max_elev_m]
        # Shift up slightly if mostly coastal so each face forms a rich, proud continent/archipelago
        p35 = float(np.percentile(e_res, 35))
        e_shifted = e_res - min(p35, 0.0) * 0.65
        land_top = max(float(np.percentile(e_shifted, 99.2)), 150.0)
        e_norm = np.clip(e_shifted / land_top, -0.35, 1.15) * max_elev_m

        # Apply organic zero-seam coastal shelf envelope!
        cont_mask = shape_continent_mask(face_export_size, seed=seed * 31 + f_idx * 97 + 13, is_polar=is_polar)
        sea_floor = -32.0
        # Pull outer rim smoothly into deep ocean (-32m) while preserving 100% of interior AI landforms!
        elev_face = e_norm * cont_mask + sea_floor * (1.0 - cont_mask)
        # Ensure strict negative ocean on the exact border pixels (guarantees ZERO seam across cube edges!)
        elev_face[0, :] = np.minimum(elev_face[0, :], -28.0)
        elev_face[-1, :] = np.minimum(elev_face[-1, :], -28.0)
        elev_face[:, 0] = np.minimum(elev_face[:, 0], -28.0)
        elev_face[:, -1] = np.minimum(elev_face[:, -1], -28.0)

        # Compute real planetary latitude |sy| on the unit sphere for every pixel on this face!
        sx, sy, sz = face_uv_to_unit_sphere(f_idx, u_grid, v_grid)
        abs_lat = np.abs(sy)  # 0.0 = equator, 1.0 = exact North/South Pole!

        # Planetary temperature model:
        #   Equator sea-level ~ +27 C -> mid-latitudes ~ +13 C -> Poles ~ -10 C
        #   Plus alpine lapse rate (-18 C at summit peak!)
        sea_temp = 27.0 - 37.0 * (abs_lat ** 1.25)
        lapse = np.maximum(elev_face, 0.0) / max(max_elev_m, 1.0) * 20.0
        temp_c = (sea_temp - lapse).astype(np.float32)

        # Combine Terrain Diffusion's local rainfall pattern with planetary Hadley/temperate moisture bands
        p_norm_local = (p_res - float(np.percentile(p_res, 5))) / max(float(np.percentile(p_res, 95) - np.percentile(p_res, 5)), 1.0)
        p_norm_local = np.clip(p_norm_local, 0.0, 1.0)
        # Dry subtropical belt around lat ~ 0.35..0.55 (20..35 deg), wet tropics (0.0) & wet mid-lats (0.65)
        hadley = 0.72 - 0.32 * np.exp(-((abs_lat - 0.40) / 0.16) ** 2) - 0.35 * np.clip((abs_lat - 0.80) / 0.20, 0.0, 1.0)
        precip_mm = np.clip((0.45 * hadley + 0.55 * p_norm_local) * 1100.0, 40.0, 1200.0).astype(np.float32)

        # Solve exact Landlab Barnes Lake-Spillway-to-Sea hydrology on this continent!
        p_med = max(float(np.median(precip_mm[elev_face > 0])) if (elev_face > 0).any() else 500.0, 120.0)
        runoff = np.clip(precip_mm / p_med, 0.4, 1.8).astype(np.float32)

        print(f"  Solving Landlab hydrology for face {fname} (lat {abs_lat.min():.2f}..{abs_lat.max():.2f})...")
        carved, riv_u8, lak_u8, lakes_data, rivers_data = solve_physical_hydrology(
            elev=elev_face,
            world_span_m=face_span_m,
            min_lake_cells=14,
            min_lake_depth_m=0.45,
            max_spring_rivers=10,
            runoff_map=runoff,
        )
        curv_u8 = compute_curvature_ao(carved)

        # Pack moisture (precip + riparian wetness)
        wet_seed = np.clip((riv_u8.astype(np.float32) + lak_u8.astype(np.float32)) / 255.0, 0.0, 1.0)
        moisture = np.clip((precip_mm / 1050.0) + gaussian_filter(wet_seed, sigma=4.0) * 0.7, 0.08, 1.0)

        # Pack face masks (RGBA8: R=riverWet, G=lake, B=curvAO, A=moisture)
        masks_rgba = np.stack([riv_u8, lak_u8, curv_u8, (moisture * 255.0).astype(np.uint8)], axis=-1)

        # Pack face climate (RGBA8: R=temp [-25..30 C], G=precip [0..1200 mm], B=abs_lat, A=255)
        t_u8 = np.clip((temp_c - (-25.0)) / 55.0 * 255.0, 0, 255).astype(np.uint8)
        p_u8 = np.clip(precip_mm / 1200.0 * 255.0, 0, 255).astype(np.uint8)
        l_u8 = np.clip(abs_lat * 255.0, 0, 255).astype(np.uint8)
        clim_rgba = np.stack([t_u8, p_u8, l_u8, np.full((face_export_size, face_export_size), 255, dtype=np.uint8)], axis=-1)

        # Save face binary files
        carved.astype(np.float32).tofile(OUT_VIEWER / f"face_{fname}.height.f32")
        masks_rgba.astype(np.uint8).tofile(OUT_VIEWER / f"face_{fname}.masks.u8")
        clim_rgba.astype(np.uint8).tofile(OUT_VIEWER / f"face_{fname}.climate.u8")

        # Project this face's rivers into 3D unit-sphere coordinates [ux, uy, uz, elev, width, slope]
        for rv in rivers_data:
            pts_3d = []
            for wx, wz, wy, w_m, sl in rv["points"]:
                u_val = float(np.clip((wx / face_span_m) * 2.0, -1.0, 1.0))
                v_val = float(np.clip((wz / face_span_m) * 2.0, -1.0, 1.0))
                px, py, pz = face_uv_to_unit_sphere(f_idx, np.array(u_val), np.array(v_val))
                # Slightly lift above spherical chord sag on curved mesh
                pts_3d.append([
                    round(float(px), 6),
                    round(float(py), 6),
                    round(float(pz), 6),
                    round(float(wy), 2),
                    round(float(w_m), 2),
                    round(float(sl), 3),
                ])
            if len(pts_3d) >= 4:
                all_rivers_3d.append({"face": fname, "source": rv["source"], "dest": rv["dest"], "points": pts_3d})

        # Project lakes onto the face for 3D curved lake rendering
        for lk in lakes_data:
            all_lakes_3d.append({
                "face": fname,
                "faceIndex": f_idx,
                "level": lk["level"],
                "cells": lk["cells"],
            })

        # Find a scenic walkable landing point on this face
        land_mask = (carved > 8.0) & (carved < max_elev_m * 0.45) & (lak_u8 < 128)
        if land_mask.any():
            pts = np.argwhere(land_mask)
            mid = pts[len(pts) // 2]
            lu = (mid[1] / (face_export_size - 1)) * 2.0 - 1.0
            lv = (mid[0] / (face_export_size - 1)) * 2.0 - 1.0
        else:
            lu, lv = 0.0, 0.0
        lx, ly, lz = face_uv_to_unit_sphere(f_idx, np.array(lu), np.array(lv))

        faces_meta.append({
            "name": fname,
            "index": f_idx,
            "landFrac": round(float((carved > 0.0).mean()), 3),
            "maxElev": round(float(carved.max()), 2),
            "riverCount": len(rivers_data),
            "lakeCount": len(lakes_data),
            "landingDir": [round(float(lx), 5), round(float(ly), 5), round(float(lz), 5)],
        })

        # Keep visual preview for atlas
        atlas_images.append((fname, carved, temp_c, riv_u8, lak_u8))

    # Tilt magnetic north pole by ~11.5 degrees from geographic north (+Y) for realistic compass navigation!
    mag_tilt = math.radians(11.5)
    mag_azim = math.radians(38.0)
    mag_north = [
        round(math.sin(mag_tilt) * math.cos(mag_azim), 5),
        round(math.cos(mag_tilt), 5),
        round(math.sin(mag_tilt) * math.sin(mag_azim), 5),
    ]

    planet_meta = {
        "seed": seed,
        "radius": planet_radius_m,
        "diameterKm": round((planet_radius_m * 2.0) / 1000.0, 2),
        "circumferenceKm": round((2.0 * math.pi * planet_radius_m) / 1000.0, 2),
        "faceSize": face_export_size,
        "maxElev": round(max_elev_m, 2),
        "tempRange": [-25.0, 30.0],
        "precipRange": [0.0, 1200.0],
        "magneticNorth": mag_north,
        "totalRivers": len(all_rivers_3d),
        "totalLakes": len(all_lakes_3d),
        "faces": faces_meta,
    }

    (OUT_VIEWER / "planet.json").write_text(json.dumps(planet_meta, indent=2))
    (OUT_VIEWER / "rivers3d.json").write_text(json.dumps(all_rivers_3d))
    (OUT_VIEWER / "lakes3d.json").write_text(json.dumps(all_lakes_3d))

    # Render 6-face planet atlas preview
    save_atlas_plot(atlas_images, planet_meta, OUT_DEBUG / "planet_atlas.png")
    print(
        f"Planet complete in {time.perf_counter() - t0:.1f}s! "
        f"D={planet_meta['diameterKm']}km (circ={planet_meta['circumferenceKm']}km) | "
        f"{len(all_lakes_3d)} lakes, {len(all_rivers_3d)} rivers -> {OUT_VIEWER}"
    )


def save_atlas_plot(atlas_images: list, meta: dict, out_png: Path) -> None:
    """Save a 2x3 visual atlas of all 6 Cube-Sphere continents with rivers & polar ice."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=130)
    fig.patch.set_facecolor("#0b1018")
    max_e = meta["maxElev"]

    for ax, (fname, elev, temp_c, riv_u8, lak_u8) in zip(axes.ravel(), atlas_images):
        gy, gx = np.gradient(elev, 18.0)
        shade = np.clip((-gx * 0.55 - gy * 0.45 + 1.0) / np.sqrt(gx**2 + gy**2 + 1.0), 0.2, 1.0)
        rgb = np.zeros((*elev.shape, 3), dtype=np.float32)
        sea = elev <= 0.0
        t = np.clip(elev / max(max_e, 1.0), 0.0, 1.0)
        rgb[~sea, 0] = np.interp(t[~sea], [0.0, 0.3, 0.7, 1.0], [0.28, 0.30, 0.50, 0.94])
        rgb[~sea, 1] = np.interp(t[~sea], [0.0, 0.3, 0.7, 1.0], [0.50, 0.49, 0.47, 0.96])
        rgb[~sea, 2] = np.interp(t[~sea], [0.0, 0.3, 0.7, 1.0], [0.22, 0.24, 0.44, 0.99])
        # Snow where temp < 1C
        snow = (~sea) & (temp_c < 1.0)
        rgb[snow] = [0.92, 0.96, 1.0]
        rgb[~sea] *= (0.4 + 0.65 * shade[~sea, None])
        # Ocean
        rgb[sea] = [0.06, 0.22, 0.40]
        # Rivers & Lakes
        wat = (~sea) & ((riv_u8 > 80) | (lak_u8 > 100))
        rgb[wat] = [0.22, 0.62, 0.85]

        ax.imshow(np.clip(rgb, 0, 1))
        label = {"py": "North Pole (+Y)", "ny": "South Pole (-Y)"}.get(fname, f"Equatorial ({fname.upper()})")
        ax.set_title(f"{label} | T={temp_c.min():.0f}..{temp_c.max():.0f}°C", color="white", fontsize=9.5)
        ax.axis("off")

    fig.suptitle(
        f"TerraGrok Zero-Seam Cube-Sphere Planet (seed={meta['seed']}, D={meta['diameterKm']} km) — 6 Terrain Diffusion 30m Continents",
        color="white",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a zero-seam walkable Terrain Diffusion 30m Cube-Sphere planet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=float, default=3000.0, help="Planet radius in meters (default 3000m -> 6km diameter)")
    parser.add_argument("--face-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()
    generate_planet(seed=args.seed, planet_radius_m=args.radius, face_export_size=args.face_size, device=args.device)


if __name__ == "__main__":
    main()
