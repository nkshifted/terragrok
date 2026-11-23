"""
Persistent Warm Stage 2 Synthesis Server (Stage 1 Planet -> Terrain Diffusion 30m on MPS).

Keeps xandergos/terrain-diffusion-30m loaded in Apple Silicon MPS memory so clicking ANY
location on the Stage 1 spherical planet (viewer/planet.html) synthesizes a real 1024x1024
30m Earth-detail DEM + climate + Landlab physical lake/river watershed in ~22s (no cold start),
writes it into viewer/public/worlds/, updates catalog.json, and opens directly in index.html.

Run via:
    uv run python experiments/stage2_server.py --port 8765
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import queue
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
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

VIEWER_WORLDS_DIR = REPO_ROOT / "viewer" / "public" / "worlds"


def _decode_f32(b64_str: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)


def _format_coord_slug(lat_deg: float, lon_deg: float) -> str:
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"{abs(round(lat_deg))}{ns}_{abs(round(lon_deg))}{ew}"


def _signed_sqrt(z: np.ndarray) -> np.ndarray:
    return np.sign(z) * np.sqrt(np.abs(z))


def _inv_signed_sqrt(s: np.ndarray) -> np.ndarray:
    return np.sign(s) * (s * s)


def _compute_biome_hydrology_profile(
    elev: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    res_scale: float,
) -> dict:
    """
    Derive physical liquid runoff, lake/river budgets, absolute moisture, and biome label
    directly from absolute temperature (°C) and precipitation (mm/yr):
      - Polar Glaciers (mean T < -3.5°C): 0 liquid rivers, 0 unfrozen lakes
      - Hot/Cold Arid Deserts (< 240 mm/yr): 0 rivers, 0 lakes (wind-sculpted sand & rock)
      - Tundra / Dry Steppe: sparse 1-5 rivers, 0-2 lakes
      - Temperate / Rainforest: full rich alpine lake & river watersheds
    """
    land = elev > 0.5
    if land.any():
        mean_t = float(np.mean(temp_c[land]))
        mean_p = float(np.mean(precip_mm[land]))
    else:
        mean_t = float(np.mean(temp_c))
        mean_p = float(np.mean(precip_mm))

    # Liquid fraction: frozen solid below -3°C, fully liquid above +2.5°C
    freeze_factor = np.clip((temp_c + 3.0) / 5.5, 0.0, 1.0)
    # Effective runoff after evaporation: bone-dry deserts (< 210 mm/yr) produce ~0 surface runoff;
    # hotter deserts evaporate even more rainfall before rivers can form!
    evap_floor = 180.0 + np.clip((temp_c - 10.0) * 8.0, 0.0, 130.0)
    aridity_factor = np.clip((precip_mm - evap_floor) / 560.0, 0.0, 1.65)
    runoff_map = (freeze_factor * aridity_factor).astype(np.float32)

    wetness = float(np.mean(runoff_map[land])) if land.any() else float(np.mean(runoff_map))

    if mean_t < -3.5:
        biome = "Polar Glacier"
        max_lakes = 0
        max_rivers = 0
        min_lake_runoff = 0.35
    elif wetness < 0.14 or (mean_p < 300.0 and mean_t > 12.0):
        biome = "Arid Desert"
        max_lakes = 0
        max_rivers = 0
        min_lake_runoff = 0.35
    elif mean_t < 2.5 and wetness < 0.45:
        biome = "Arctic Tundra"
        max_lakes = max(0, int(round(wetness * 6)))
        max_rivers = max(1, int(round(wetness * 10)))
        min_lake_runoff = 0.16
    elif wetness < 0.36:
        biome = "Dry Steppe"
        max_lakes = max(0, int(round((wetness - 0.14) * 5)))
        max_rivers = max(1, int(round((wetness - 0.08) * 10)))
        min_lake_runoff = 0.22
    elif mean_t > 18.0 and mean_p > 1050.0:
        biome = "Tropical Rainforest"
        max_lakes = 10
        max_rivers = 22
        min_lake_runoff = 0.15
    else:
        biome = "Temperate Highlands"
        t_norm = float(np.clip((wetness - 0.30) / 0.65, 0.0, 1.0))
        max_lakes = int(round(3 + t_norm * 7))
        max_rivers = int(round(6 + t_norm * 14))
        min_lake_runoff = 0.15

    # Absolute moisture (NOT stretched per patch!): deserts stay dry (<0.12), glaciers stay 0.0!
    abs_moist = np.clip((precip_mm - 140.0) / 920.0, 0.02, 0.96) * np.clip((temp_c + 3.5) / 7.0, 0.0, 1.0)

    return {
        "biome": biome,
        "mean_t": mean_t,
        "mean_p": mean_p,
        "wetness": wetness,
        "runoff_map": runoff_map,
        "abs_moist": abs_moist.astype(np.float32),
        "max_lakes": max_lakes,
        "max_spring_rivers": max_rivers,
        "min_lake_runoff": min_lake_runoff,
        "min_lake_cells": int(round(28 * res_scale)),
    }


def _apply_biome_surface_refinement(
    elev_viewer: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    world_span_m: float,
    seed: int,
) -> np.ndarray:
    """
    Biome-aware micro-terrain refinement:
      - Hot arid deserts: smooth wind-sculpted sand dune ripples instead of rocky alpine chatter
      - Polar glaciers: sweeping smooth glacial snowfields & crevasse swells
      - Temperate / alpine / coastal rock: full crag & gully micro-detail
    """
    craggy = apply_micro_detail_refinement(elev_viewer, world_span_m=world_span_m, seed=seed)

    desert_w = np.clip((temp_c - 12.0) / 8.0, 0.0, 1.0) * np.clip((340.0 - precip_mm) / 180.0, 0.0, 1.0)
    glacier_w = np.clip((-2.5 - temp_c) / 4.5, 0.0, 1.0)
    land = (elev_viewer > 2.0).astype(np.float32)

    out = craggy.copy()
    if float(desert_w.max()) > 0.05:
        ny, nx = elev_viewer.shape
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        # Wind-sculpted barchan/transverse dune waves at 2 natural scales
        ang = (seed % 360) * np.pi / 180.0
        u = (xx * np.cos(ang) + yy * np.sin(ang)) * (world_span_m / max(nx, 1))
        v = (-xx * np.sin(ang) + yy * np.cos(ang)) * (world_span_m / max(nx, 1))
        warp = gaussian_filter(
            np.sin(u * 0.011 + np.sin(v * 0.007) * 2.2) * 3.8
            + np.sin(u * 0.032 + np.cos(v * 0.019) * 1.4) * 1.4,
            sigma=1.5,
        )
        # Only place dunes on gentle-to-moderate desert floors (preserve rocky mesas on steep cliffs!)
        gy, gx = np.gradient(elev_viewer, world_span_m / max(nx - 1, 1))
        flat_desert = np.clip(1.0 - np.hypot(gx, gy) / 0.42, 0.0, 1.0)
        sand_surface = gaussian_filter(elev_viewer, sigma=2.5, mode="nearest") + warp * flat_desert
        blend = desert_w * flat_desert * land * 0.85
        out = out * (1.0 - blend) + sand_surface * blend

    if float(glacier_w.max()) > 0.05:
        smooth_ice = gaussian_filter(elev_viewer, sigma=2.2, mode="nearest")
        blend = glacier_w * land * 0.72
        out = out * (1.0 - blend) + smooth_ice * blend

    return out.astype(np.float32)


def _relax_angle_of_repose(
    elev: np.ndarray,
    cell_size_m: float,
    max_slope_deg: float = 52.0,
    iterations: int = 12,
) -> np.ndarray:
    """
    Fast vectorized talus/thermal angle-of-repose relaxation on land pixels.
    Prevents narrow Stage 1 islets from becoming sheer 80° vertical pillars
    while preserving natural crags, peaks, and exact coastlines (ocean stays <= 0).
    """
    out = elev.astype(np.float32, copy=True)
    max_diff = float(np.tan(np.deg2rad(max_slope_deg)) * cell_size_m)
    max_diag = max_diff * 1.4142135
    for _ in range(iterations):
        # 4-connected + 4-diagonal neighbor max height cap
        up = np.empty_like(out)
        dn = np.empty_like(out)
        lt = np.empty_like(out)
        rt = np.empty_like(out)
        up[1:, :] = out[:-1, :]
        up[0, :] = out[0, :]
        dn[:-1, :] = out[1:, :]
        dn[-1, :] = out[-1, :]
        lt[:, 1:] = out[:, :-1]
        lt[:, 0] = out[:, 0]
        rt[:, :-1] = out[:, 1:]
        rt[:, -1] = out[:, -1]

        nbr_min = np.minimum(np.minimum(up, dn), np.minimum(lt, rt))
        cap = np.maximum(nbr_min + max_diff, 0.0)
        excess = np.maximum(out - cap, 0.0)
        # Relax excess gently downward on land so cliffs smooth into steep natural ridges
        land = out > 0.0
        if not np.any(excess[land] > 0.25):
            break
        out = np.where(land, out - excess * 0.42, out)

        # Also check diagonals every other pass
        d1 = np.roll(np.roll(out, 1, axis=0), 1, axis=1)
        d2 = np.roll(np.roll(out, 1, axis=0), -1, axis=1)
        d3 = np.roll(np.roll(out, -1, axis=0), 1, axis=1)
        d4 = np.roll(np.roll(out, -1, axis=0), -1, axis=1)
        d_min = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))
        d_cap = np.maximum(d_min + max_diag, 0.0)
        d_excess = np.maximum(out - d_cap, 0.0)
        d_excess[0, :] = d_excess[-1, :] = d_excess[:, 0] = d_excess[:, -1] = 0.0
        out = np.where(land, out - d_excess * 0.28, out)
    return out


def fuse_laplacian_coastline(
    elev_td: np.ndarray,
    ref_up: np.ndarray,
    fidelity: str = "exact",
) -> np.ndarray:
    """
    Laplacian frequency-band fusion in Terrain Diffusion's native signed-sqrt space:
      - Macro shape (sigma=20px): blended toward exact 256x256 Stage 1 globe coastlines/islands
      - High-frequency 30m geology (peaks, ridges, cirques, valleys): 100% Terrain Diffusion 30m!
    Preserves every small island, bay, and strait from the globe square while keeping 30m AI realism
    and natural coastal mountain slopes (no sheer cliff walls).
    """
    alpha_map = {"exact": 0.88, "balanced": 0.58, "creative": 0.0}
    alpha = alpha_map.get(fidelity, 0.88)
    if alpha <= 0.0:
        return elev_td

    s_td = _signed_sqrt(elev_td.astype(np.float32))
    # Clip deep abyssal bathymetry (-2800m -> -320m) before signed-sqrt so ocean trenches
    # don't create an extreme negative step that pulls down narrow island shorelines!
    ref_clipped = np.maximum(ref_up.astype(np.float32), -320.0)
    s_ref = _signed_sqrt(ref_clipped)

    # Split Terrain Diffusion into macro shape + 30m AI alpine ridges/valleys
    s_td_low = gaussian_filter(s_td, sigma=20.0, mode="nearest")
    s_td_high = s_td - s_td_low

    # Two-scale smoothing of Stage 1 reference:
    #   - sigma=7.5px builds smooth, natural mountain slopes (no vertical 80° cookie-cutter walls!)
    #   - sigma=2.5px preserves exact crisp shoreline contours of tiny islands & coves
    s_ref_broad = gaussian_filter(s_ref, sigma=7.5, mode="nearest")
    s_ref_crisp = gaussian_filter(s_ref, sigma=2.5, mode="nearest")
    # Near the coastline (|s_ref_crisp| < 8), lean toward crisp shoreline; inland, lean toward smooth slopes
    shore_band = np.exp(-((s_ref_crisp / 6.5) ** 2))
    s_ref_low = s_ref_broad * (1.0 - 0.65 * shore_band) + s_ref_crisp * (0.65 * shore_band)

    # Rescued small islands: where the globe has land (ref_up > 0) that 4x4 diffusion missed (elev_td <= 0),
    # transfer genuine Terrain Diffusion neural ridge texture from landed areas so rescued islands look
    # every bit as craggy and realistic as native TD30m peaks!
    td_land = elev_td > 15.0
    rescued = (ref_up > 0.0) & (~td_land)
    if np.any(rescued) and np.any(td_land):
        rescued_mask = gaussian_filter(rescued.astype(np.float32), sigma=3.5, mode="nearest")
        donor_high = np.rot90(s_td_high, 2)
        land_std = float(np.std(s_td_high[td_land]))
        donor_std = float(np.std(donor_high[rescued])) if rescued.sum() > 20 else land_std
        if donor_std < land_std * 0.45:
            donor_high = 0.6 * np.rot90(s_td_high, 1) + 0.6 * np.rot90(s_td_high, 3)
        # Scale rescued island ridge amplitude proportionally to island elevation so tiny atolls stay realistic
        elev_weight = np.clip(s_ref_low / 26.0, 0.22, 0.90)
        s_td_high = s_td_high * (1.0 - rescued_mask * 0.55) + donor_high * (rescued_mask * 0.68 * elev_weight)

    s_fused = (1.0 - alpha) * s_td_low + alpha * s_ref_low + s_td_high
    return _inv_signed_sqrt(s_fused).astype(np.float32)


class WarmStage2Engine:
    def __init__(
        self,
        coarse_pooling: int = 2,
        cond_snr: tuple[float, ...] = (0.45, 0.25, 1.0, 0.25, 1.0),
    ):
        self.coarse_pooling = coarse_pooling
        self.cond_snr = list(cond_snr)
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.world: WorldPipeline | None = None
        self.ready = False
        self.load_error: str | None = None
        self.lock = threading.Lock()
        # Dedicated single worker thread so PyTorch MPS / Metal ALWAYS runs on the exact same OS thread!
        self._job_queue: queue.Queue = queue.Queue()
        self.status = {
            "busy": False,
            "phase": "initializing",
            "message": f"Loading Terrain Diffusion 30m onto {self.device.upper()}…",
            "progress": 0.0,
            "startedAt": None,
            "lastWorldId": None,
        }

    def start_worker(self):
        threading.Thread(target=self._worker_loop, daemon=True, name="stage2-mps-worker").start()

    def _worker_loop(self):
        self.load_model()
        while True:
            kind, payload, done_evt, holder = self._job_queue.get()
            try:
                if kind == "drilldown":
                    holder["result"] = self._drilldown_on_worker(payload)
                else:
                    holder["result"] = self._synthesize_on_worker(payload)
            except Exception as exc:
                holder["error"] = exc
            finally:
                done_evt.set()
                self._job_queue.task_done()

    def synthesize(self, payload: dict) -> dict:
        if not self.ready or self.world is None:
            raise RuntimeError(self.load_error or "Terrain Diffusion 30m model is still warming up.")
        done_evt = threading.Event()
        holder: dict = {}
        self._job_queue.put(("synthesize", payload, done_evt, holder))
        done_evt.wait()
        if "error" in holder:
            raise holder["error"]
        return holder["result"]

    def drilldown(self, payload: dict) -> dict:
        if not self.ready or self.world is None:
            raise RuntimeError(self.load_error or "Terrain Diffusion 30m model is still warming up.")
        done_evt = threading.Event()
        holder: dict = {}
        self._job_queue.put(("drilldown", payload, done_evt, holder))
        done_evt.wait()
        if "error" in holder:
            raise holder["error"]
        return holder["result"]

    def load_model(self):
        t0 = time.perf_counter()
        try:
            use_dtype = "bf16" if self.device in ("mps", "cuda") else None
            print(
                f"[Stage2] Loading xandergos/terrain-diffusion-30m onto {self.device} "
                f"(pool={self.coarse_pooling}, dtype={use_dtype}, stride=448, T=1)…",
                flush=True,
            )
            world = WorldPipeline.from_pretrained(
                "xandergos/terrain-diffusion-30m",
                seed=42,
                coarse_pooling=self.coarse_pooling,
                histogram_raw=[0.0, 0.0, 0.0, 1.0, 1.8],
                torch_compile=False,
                caching_strategy="direct",
                dtype=use_dtype,
                decoder_tile_size=512,
                decoder_tile_stride=448,
                T=1,
            )
            world.to(self.device)
            world.set_cond_snr(self.cond_snr)

            prev_cwd = os.getcwd()
            os.chdir(REPO_ROOT / ".vendor" / "terrain-diffusion")
            try:
                world.bind(hdf5_file=None)
            finally:
                os.chdir(prev_cwd)

            self.world = world
            self.ready = True
            elapsed = time.perf_counter() - t0
            self.status.update(
                {
                    "busy": False,
                    "phase": "idle",
                    "message": f"Warm on {self.device.upper()} ({elapsed:.1f}s load)",
                    "progress": 1.0,
                }
            )
            print(f"[Stage2] Model warm & ready on {self.device.upper()} in {elapsed:.1f}s!", flush=True)
        except Exception as exc:
            self.load_error = str(exc)
            self.status.update({"busy": False, "phase": "error", "message": f"Load failed: {exc}"})
            print(f"[Stage2] ERROR loading model: {exc}", flush=True)
            raise

    def _synthesize_on_worker(self, payload: dict) -> dict:
        if not self.ready or self.world is None:
            raise RuntimeError(self.load_error or "Terrain Diffusion 30m model is still warming up.")

        with self.lock:
            t0 = time.perf_counter()
            seed = int(payload.get("seed", 101))
            archetype = str(payload.get("archetype", "earth"))
            lat_deg = float(payload.get("latDeg", 0.0))
            lon_deg = float(payload.get("lonDeg", 0.0))
            span_deg = float(payload.get("spanDeg", 46.0))
            coarse_size = int(payload.get("coarseSize", 8))
            padding = int(payload.get("padding", 64))
            ref_size = int(payload.get("refSize", 128))
            full_coarse = coarse_size + 2 * padding

            if padding % self.coarse_pooling != 0 or coarse_size % self.coarse_pooling != 0:
                raise ValueError(f"coarseSize ({coarse_size}) and padding ({padding}) must be multiples of coarse_pooling ({self.coarse_pooling})")

            self.status.update(
                {
                    "busy": True,
                    "phase": "decoding",
                    "message": "Decoding Stage 1 tectonic conditioning fields…",
                    "progress": 0.08,
                    "startedAt": time.time(),
                }
            )

            cond_elev = _decode_f32(payload["condElev"], (full_coarse, full_coarse))
            cond_temp = _decode_f32(payload["condTemp"], (full_coarse, full_coarse))
            cond_temp_std = _decode_f32(payload["condTempStd"], (full_coarse, full_coarse))
            cond_precip = _decode_f32(payload["condPrecip"], (full_coarse, full_coarse))
            cond_precip_cv = _decode_f32(payload["condPrecipCv"], (full_coarse, full_coarse))
            ref_elev = _decode_f32(payload["refElev"], (ref_size, ref_size))
            ref_temp = _decode_f32(payload["refTemp"], (ref_size, ref_size))
            ref_precip = _decode_f32(payload["refPrecip"], (ref_size, ref_size))

            # Deterministic unique diffusion seed per (planet seed, lat, lon)
            td_seed = (
                (seed * 73856093)
                ^ (int(round((lat_deg + 90.0) * 10)) * 19349663)
                ^ (int(round((lon_deg + 180.0) * 10)) * 83492791)
            ) & 0x7FFFFFFF

            px_per_raw_cell = 256 // self.coarse_pooling
            pi1 = padding * px_per_raw_cell
            pi2 = (padding + coarse_size) * px_per_raw_cell
            pj1 = padding * px_per_raw_cell
            pj2 = (padding + coarse_size) * px_per_raw_cell
            out_n = pi2 - pi1

            self.status.update(
                {
                    "phase": "synthesizing_mps",
                    "message": f"Synthesizing {out_n}×{out_n} 30m Earth terrain on {self.device.upper()}…",
                    "progress": 0.22,
                }
            )
            print(
                f"[Stage2] Synthesizing planet seed={seed} ({archetype}) @ {lat_deg:.1f}°, {lon_deg:.1f}° "
                f"(span={span_deg:.0f}°, td_seed={td_seed})…",
                flush=True,
            )

            quality = str(payload.get("quality", "high")).lower()
            self.world.onestep_latent = quality == "fast"

            prev_cwd = os.getcwd()
            os.chdir(REPO_ROOT / ".vendor" / "terrain-diffusion")
            try:
                # Batch-update all 5 tectonic conditioning channels & reset tile cache cleanly!
                self.world.set_custom_conditioning_import(0, cond_elev, 0, 0, default_value=-1500.0, rebuild=False)
                self.world.set_custom_conditioning_import(1, cond_temp, 0, 0, default_value=12.0, rebuild=False)
                self.world.set_custom_conditioning_import(2, cond_temp_std, 0, 0, default_value=600.0, rebuild=False)
                self.world.set_custom_conditioning_import(3, cond_precip, 0, 0, default_value=900.0, rebuild=False)
                self.world.set_custom_conditioning_import(4, cond_precip_cv, 0, 0, default_value=35.0, rebuild=False)
                self.world.seed = td_seed
                self.world.rebuild()

                t_gen0 = time.perf_counter()
                with torch.no_grad():
                    res = self.world.get(pi1, pj1, pi2, pj2, with_climate=True)
                if self.device == "mps":
                    torch.mps.synchronize()
                t_gen = time.perf_counter() - t_gen0
                elev_raw = res["elev"].detach().cpu().numpy().astype(np.float32)
                climate_raw = res["climate"].detach().cpu().numpy().astype(np.float32)
                del res
                self.world.empty_cache()
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
            finally:
                os.chdir(prev_cwd)

            # Fidelity metrics & optional Laplacian Coastline Lock vs Stage 1 tectonic reference
            fidelity = str(payload.get("fidelity", "exact")).lower()
            ref_up = zoom(ref_elev, out_n / ref_size, order=3).astype(np.float32)
            elev_fused = fuse_laplacian_coastline(elev_raw, ref_up, fidelity=fidelity)

            ref_land = ref_up > 0.0
            gen_land = elev_fused > 0.0
            inter = np.logical_and(ref_land, gen_land).sum()
            union = np.logical_or(ref_land, gen_land).sum() + 1e-6
            iou = float(inter / union)
            std_ref = float(np.std(ref_up))
            std_gen = float(np.std(elev_fused))
            corr = (
                float(np.corrcoef(ref_up.ravel(), elev_fused.ravel())[0, 1])
                if std_ref > 1e-3 and std_gen > 1e-3
                else 0.0
            )

            self.status.update(
                {
                    "phase": "refining_micro_detail",
                    "message": f"Diffusion done ({t_gen:.1f}s, IoU {iou*100:.0f}%). Refining crags & gullies…",
                    "progress": 0.70,
                }
            )

            world_span_m = 4800.0
            land_mask = elev_fused > 0.0
            max_land_m = float(np.percentile(elev_fused[land_mask], 99.6)) if land_mask.any() else 1000.0
            # True physical relief scaling: flat deserts & tundra plains stay gentle open plains,
            # while towering alpine ranges & volcanic arcs still rise dramatically to 340-430m!
            v_scale = min(420.0 / max(max_land_m, 1550.0), 0.28)
            elev_viewer = np.where(land_mask, elev_fused * v_scale, np.maximum(elev_fused * 0.05, -55.0)).astype(np.float32)
            # Soft-knee extreme outlier peaks above 330m so tiny islets don't tower as vertical spires
            elev_viewer = np.where(
                elev_viewer > 330.0,
                330.0 + 115.0 * np.tanh((elev_viewer - 330.0) / 150.0),
                elev_viewer,
            ).astype(np.float32)
            # Talus / angle-of-repose relaxation (~52° max macro cliff) before micro-crag refinement
            elev_viewer = _relax_angle_of_repose(
                elev_viewer,
                cell_size_m=world_span_m / out_n,
                max_slope_deg=52.0,
                iterations=14,
            )

            # Preserve Stage 1 planetary climate mean EXACTLY, adding ONLY TD30m's zero-mean local spatial
            # anomaly so Sahara deserts stay bone-dry (<220mm) and polar glaciers stay genuinely frozen (<-5°C)!
            ref_temp_up = zoom(ref_temp, out_n / ref_size, order=1).astype(np.float32)
            ref_precip_up = zoom(ref_precip, out_n / ref_size, order=1).astype(np.float32)
            valid_t = np.isfinite(climate_raw[0])
            td_t_mean = float(np.mean(climate_raw[0][valid_t])) if valid_t.any() else 10.0
            td_t_anom = np.where(valid_t, (climate_raw[0] - td_t_mean) * 0.28, 0.0)
            temp_c = np.clip(ref_temp_up + td_t_anom, -22.0, 38.0).astype(np.float32)

            valid_p = np.isfinite(climate_raw[2]) & (climate_raw[2] > 1.0)
            td_p_mean = float(np.mean(climate_raw[2][valid_p])) if valid_p.any() else 600.0
            td_p_ratio = np.where(valid_p, np.clip(climate_raw[2] / max(td_p_mean, 50.0), 0.72, 1.32), 1.0)
            precip_mm = np.clip(ref_precip_up * td_p_ratio, 35.0, 3500.0).astype(np.float32)
            t_season = np.where(np.isfinite(climate_raw[1]), np.abs(climate_raw[1]), 450.0).astype(np.float32)

            elev_refined = _apply_biome_surface_refinement(
                elev_viewer, temp_c, precip_mm, world_span_m=world_span_m, seed=td_seed
            )

            res_scale = (out_n / 512.0) ** 2
            hydro_prof = _compute_biome_hydrology_profile(elev_refined, temp_c, precip_mm, res_scale)
            biome_label = hydro_prof["biome"]

            self.status.update(
                {
                    "phase": "solving_hydrology",
                    "message": f"Biome: {biome_label} ({hydro_prof['mean_t']:.0f}°C, {hydro_prof['mean_p']:.0f}mm) — solving hydrology…",
                    "progress": 0.82,
                }
            )

            t_h0 = time.perf_counter()
            carved, riv_u8, lak_u8, lakes_data, rivers_data = solve_physical_hydrology(
                elev=elev_refined,
                world_span_m=world_span_m,
                min_lake_cells=hydro_prof["min_lake_cells"],
                min_lake_depth_m=0.55,
                max_spring_rivers=hydro_prof["max_spring_rivers"],
                runoff_map=hydro_prof["runoff_map"],
                max_lakes=hydro_prof["max_lakes"],
                min_lake_runoff=hydro_prof["min_lake_runoff"],
            )
            t_hydro = time.perf_counter() - t_h0

            self.status.update(
                {
                    "phase": "exporting",
                    "message": "Writing 3D viewer world & updating catalog…",
                    "progress": 0.94,
                }
            )

            coord_slug = _format_coord_slug(lat_deg, lon_deg)
            coord_label = coord_slug.replace("_", " ").replace("N", "°N").replace("S", "°S").replace("E", "°E").replace("W", "°W")
            safe_arch = re.sub(r"[^a-z0-9]", "", archetype.lower())[:8] or "earth"
            world_id = f"planet_s{seed}_{safe_arch}_{coord_slug.lower()}"

            VIEWER_WORLDS_DIR.mkdir(parents=True, exist_ok=True)
            curv_u8 = compute_curvature_ao(carved)

            temp_range = [-16.0, 32.0]
            precip_range = [0.0, 1800.0]
            t_u8 = np.clip((temp_c - temp_range[0]) / (temp_range[1] - temp_range[0]) * 255.0, 0, 255).astype(np.uint8)
            p_u8 = np.clip((precip_mm - precip_range[0]) / (precip_range[1] - precip_range[0]) * 255.0, 0, 255).astype(np.uint8)
            s_u8 = np.clip(t_season / max(float(np.percentile(t_season, 98)), 1.0) * 255.0, 0, 255).astype(np.uint8)
            clim_rgba = np.stack([t_u8, p_u8, s_u8, np.full((out_n, out_n), 255, dtype=np.uint8)], axis=-1)
            clim_rgba.tofile(VIEWER_WORLDS_DIR / f"{world_id}.climate.u8")

            # Absolute biome moisture + riparian wetness ONLY where real rivers/lakes exist!
            wet_seed = np.clip((riv_u8.astype(np.float32) + lak_u8.astype(np.float32)) / 255.0, 0.0, 1.0)
            moisture = np.clip(hydro_prof["abs_moist"] + gaussian_filter(wet_seed, sigma=10.0) * 0.75, 0.0, 1.0)

            masks = np.stack([riv_u8, lak_u8, curv_u8, (moisture * 255.0).astype(np.uint8)], axis=-1)
            carved.astype(np.float32).tofile(VIEWER_WORLDS_DIR / f"{world_id}.height.f32")
            masks.astype(np.uint8).tofile(VIEWER_WORLDS_DIR / f"{world_id}.masks.u8")
            (VIEWER_WORLDS_DIR / f"{world_id}.rivers.json").write_text(json.dumps(rivers_data))
            (VIEWER_WORLDS_DIR / f"{world_id}.lakes.json").write_text(json.dumps(lakes_data))

            landmarks = find_landmarks(carved, riv_u8, lak_u8, lakes_data, rivers_data, world_span_m)
            world_meta = {
                "id": world_id,
                "title": f"Planet #{seed} · {coord_label} ({biome_label})",
                "subtitle": f"{biome_label} ({hydro_prof['mean_t']:.0f}°C, {hydro_prof['mean_p']:.0f}mm) · IoU {iou*100:.0f}%",
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
                "planetOrigin": {
                    "seed": seed,
                    "archetype": archetype,
                    "latDeg": round(lat_deg, 2),
                    "lonDeg": round(lon_deg, 2),
                    "spanDeg": round(span_deg, 1),
                },
                "landmarks": landmarks,
            }
            (VIEWER_WORLDS_DIR / f"{world_id}.json").write_text(json.dumps(world_meta, indent=2))

            # Write dynamic Stage 2 regions into a gitignored catalog overlay so tracked catalog.json stays clean
            synth_catalog_path = VIEWER_WORLDS_DIR / "synthesized_catalog.json"
            synth_catalog = json.loads(synth_catalog_path.read_text()) if synth_catalog_path.exists() else []
            synth_catalog = [w for w in synth_catalog if w["id"] != world_id]
            synth_catalog.append(world_meta)
            synth_catalog_path.write_text(json.dumps(synth_catalog, indent=2))

            total_s = time.perf_counter() - t0
            self.status.update(
                {
                    "busy": False,
                    "phase": "done",
                    "message": f"Synthesized {coord_label} in {total_s:.1f}s!",
                    "progress": 1.0,
                    "lastWorldId": world_id,
                }
            )
            print(
                f"[Stage2] DONE {world_id} in {total_s:.1f}s (MPS={t_gen:.1f}s, hydro={t_hydro:.1f}s) | "
                f"IoU={iou*100:.1f}%, r={corr:.3f} | {len(lakes_data)} lakes, {len(rivers_data)} rivers",
                flush=True,
            )

            return {
                "ok": True,
                "worldId": world_id,
                "title": world_meta["title"],
                "coordLabel": coord_label,
                "iouPct": round(iou * 100.0, 1),
                "elevCorr": round(corr, 3),
                "landPct": round(float(gen_land.mean() * 100.0), 1),
                "lakeCount": len(lakes_data),
                "riverCount": len(rivers_data),
                "mpsSec": round(t_gen, 2),
                "hydroSec": round(t_hydro, 2),
                "totalSec": round(total_s, 2),
                "openUrl": f"/index.html?world={world_id}",
            }

    def _drilldown_on_worker(self, payload: dict) -> dict:
        if not self.ready or self.world is None:
            raise RuntimeError(self.load_error or "Terrain Diffusion 30m model is still warming up.")

        with self.lock:
            t0 = time.perf_counter()
            parent_id = str(payload.get("parentWorldId", "")).strip()
            if not re.match(r"^[a-zA-Z0-9_-]+$", parent_id):
                raise ValueError(f"Invalid parentWorldId: {parent_id!r}")

            parent_json_path = VIEWER_WORLDS_DIR / f"{parent_id}.json"
            parent_height_path = VIEWER_WORLDS_DIR / f"{parent_id}.height.f32"
            if not parent_json_path.exists() or not parent_height_path.exists():
                raise FileNotFoundError(f"Parent world not found on disk: {parent_id}")

            parent_meta = json.loads(parent_json_path.read_text())
            parent_size = int(parent_meta["size"])
            parent_span = float(parent_meta["worldSpan"])
            parent_h = np.fromfile(parent_height_path, dtype=np.float32).reshape((parent_size, parent_size))

            # Optional parent climate
            parent_clim_path = VIEWER_WORLDS_DIR / f"{parent_id}.climate.u8"
            parent_clim = (
                np.fromfile(parent_clim_path, dtype=np.uint8).reshape((parent_size, parent_size, 4))
                if parent_clim_path.exists()
                else None
            )
            t_range = parent_meta.get("tempRange", [-12.0, 28.0])
            p_range = parent_meta.get("precipRange", [0.0, 1800.0])

            zoom_factor = int(payload.get("zoomFactor", 4))
            zoom_factor = 2 if zoom_factor <= 2 else 4
            half = 0.5 / zoom_factor
            cu = float(np.clip(float(payload.get("centerU", 0.5)), half, 1.0 - half))
            cv = float(np.clip(float(payload.get("centerV", 0.5)), half, 1.0 - half))
            u0, u1 = cu - half, cu + half
            v0, v1 = cv - half, cv + half

            child_span = parent_span / zoom_factor
            out_n = 1024
            pixel_m = child_span / out_n
            depth = int(parent_meta.get("drillOrigin", {}).get("depth", 0)) + 1

            self.status.update(
                {
                    "busy": True,
                    "phase": "preparing_drilldown",
                    "message": f"Extracting {zoom_factor}× sub-region ({child_span/1000:.2f} km, {pixel_m:.2f} m/px)…",
                    "progress": 0.10,
                    "startedAt": time.time(),
                }
            )

            # Extract exact sub-region from parent heightfield -> cubic upsample to 1024x1024
            r0 = int(np.clip(round(v0 * (parent_size - 1)), 0, parent_size - 2))
            r1 = int(np.clip(round(v1 * (parent_size - 1)), r0 + 2, parent_size))
            c0 = int(np.clip(round(u0 * (parent_size - 1)), 0, parent_size - 2))
            c1 = int(np.clip(round(u1 * (parent_size - 1)), c0 + 2, parent_size))
            crop_h = parent_h[r0:r1, c0:c1]
            ref_viewer_up = zoom(crop_h, (out_n / crop_h.shape[0], out_n / crop_h.shape[1]), order=3).astype(np.float32)

            # Convert parent viewer meters into Terrain Diffusion's native Earth DEM elevation range
            parent_land = parent_h > 0.0
            parent_peak = float(np.percentile(parent_h[parent_land], 99.5)) if parent_land.any() else 260.0
            dem_scale = float(np.clip(1450.0 / max(parent_peak, 240.0), 2.2, 6.0))
            ref_dem = np.where(ref_viewer_up > 0.0, ref_viewer_up * dem_scale, np.maximum(ref_viewer_up * 12.0, -360.0)).astype(np.float32)

            # Build padded 136x136 conditioning grids centered on the cropped sub-region using real parent context!
            coarse_size = 8
            padding = 64
            full_coarse = coarse_size + 2 * padding
            step_u = (u1 - u0) / coarse_size
            step_v = (v1 - v0) / coarse_size
            grid_i = np.arange(full_coarse, dtype=np.float32) - (full_coarse - 1) * 0.5
            sample_u = np.clip(cu + grid_i * step_u, 0.0, 1.0)
            sample_v = np.clip(cv + grid_i * step_v, 0.0, 1.0)
            su_idx = np.clip(np.round(sample_u * (parent_size - 1)).astype(np.int32), 0, parent_size - 1)
            sv_idx = np.clip(np.round(sample_v * (parent_size - 1)).astype(np.int32), 0, parent_size - 1)
            cond_h_parent = parent_h[np.ix_(sv_idx, su_idx)]
            cond_elev = np.where(cond_h_parent > 0.0, cond_h_parent * dem_scale, np.maximum(cond_h_parent * 12.0, -360.0)).astype(np.float32)

            if parent_clim is not None:
                c_sub = parent_clim[np.ix_(sv_idx, su_idx)]
                cond_temp = (t_range[0] + (c_sub[:, :, 0].astype(np.float32) / 255.0) * (t_range[1] - t_range[0])).astype(np.float32)
                cond_precip = np.clip(
                    p_range[0] + (c_sub[:, :, 1].astype(np.float32) / 255.0) * (p_range[1] - p_range[0]),
                    80.0,
                    3000.0,
                ).astype(np.float32)
                clim_crop = parent_clim[r0:r1, c0:c1]
                crop_t = t_range[0] + (clim_crop[:, :, 0].astype(np.float32) / 255.0) * (t_range[1] - t_range[0])
                crop_p = p_range[0] + (clim_crop[:, :, 1].astype(np.float32) / 255.0) * (p_range[1] - p_range[0])
                ref_temp_up = zoom(crop_t, (out_n / crop_t.shape[0], out_n / crop_t.shape[1]), order=1).astype(np.float32)
                ref_precip_up = zoom(crop_p, (out_n / crop_p.shape[0], out_n / crop_p.shape[1]), order=1).astype(np.float32)
            else:
                cond_temp = np.full((full_coarse, full_coarse), 10.0, dtype=np.float32)
                cond_precip = np.full((full_coarse, full_coarse), 900.0, dtype=np.float32)
                ref_temp_up = np.full((out_n, out_n), 10.0, dtype=np.float32)
                ref_precip_up = np.full((out_n, out_n), 900.0, dtype=np.float32)

            cond_temp_std = np.full((full_coarse, full_coarse), 550.0, dtype=np.float32)
            cond_precip_cv = np.full((full_coarse, full_coarse), 36.0, dtype=np.float32)

            # Deterministic unique diffusion seed per (parent, center, zoom)
            h_str = f"{parent_id}:{zoom_factor}:{round(cu*1000)}:{round(cv*1000)}"
            td_seed = 2166136261
            for ch in h_str.encode("utf-8"):
                td_seed = ((td_seed ^ ch) * 16777619) & 0x7FFFFFFF

            px_per_raw_cell = 256 // self.coarse_pooling
            pi1 = padding * px_per_raw_cell
            pi2 = (padding + coarse_size) * px_per_raw_cell
            pj1 = padding * px_per_raw_cell
            pj2 = (padding + coarse_size) * px_per_raw_cell

            quality = str(payload.get("quality", "high")).lower()
            self.world.onestep_latent = quality == "fast"

            self.status.update(
                {
                    "phase": "synthesizing_mps",
                    "message": f"Synthesizing {zoom_factor}× micro-terrain ({pixel_m:.2f} m/px) on {self.device.upper()}…",
                    "progress": 0.24,
                }
            )
            print(
                f"[Stage2] Drill-Down {zoom_factor}× on {parent_id} @ ({cu:.2f}, {cv:.2f}) -> "
                f"{child_span:.0f}m ({pixel_m:.2f}m/px, td_seed={td_seed})…",
                flush=True,
            )

            prev_cwd = os.getcwd()
            os.chdir(REPO_ROOT / ".vendor" / "terrain-diffusion")
            try:
                self.world.set_custom_conditioning_import(0, cond_elev, 0, 0, default_value=-360.0, rebuild=False)
                self.world.set_custom_conditioning_import(1, cond_temp, 0, 0, default_value=10.0, rebuild=False)
                self.world.set_custom_conditioning_import(2, cond_temp_std, 0, 0, default_value=550.0, rebuild=False)
                self.world.set_custom_conditioning_import(3, cond_precip, 0, 0, default_value=900.0, rebuild=False)
                self.world.set_custom_conditioning_import(4, cond_precip_cv, 0, 0, default_value=36.0, rebuild=False)
                self.world.seed = td_seed
                self.world.rebuild()

                t_gen0 = time.perf_counter()
                with torch.no_grad():
                    res = self.world.get(pi1, pj1, pi2, pj2, with_climate=True)
                if self.device == "mps":
                    torch.mps.synchronize()
                t_gen = time.perf_counter() - t_gen0
                elev_raw = res["elev"].detach().cpu().numpy().astype(np.float32)
                climate_raw = res["climate"].detach().cpu().numpy().astype(np.float32)
                del res
                self.world.empty_cache()
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
            finally:
                os.chdir(prev_cwd)

            # High-fidelity Laplacian Super-Resolution:
            # Lock macro silhouette/valleys/coastlines tightly to the parent crop, while injecting
            # genuine Terrain Diffusion 30m high-frequency crags, rock spurs, and ledges!
            fidelity = str(payload.get("fidelity", "exact")).lower()
            alpha = 0.90 if fidelity == "exact" else 0.72
            s_td = _signed_sqrt(elev_raw)
            s_ref = _signed_sqrt(ref_dem)
            s_td_low = gaussian_filter(s_td, sigma=18.0, mode="nearest")
            s_td_high = s_td - s_td_low
            s_ref_low = gaussian_filter(s_ref, sigma=2.2, mode="nearest")

            # Taper injected neural crag amplitude smoothly to 0 in open water so sea stays calm
            land_blend = gaussian_filter((ref_viewer_up > -2.0).astype(np.float32), sigma=4.0, mode="nearest")
            # Moderate high-freq gain proportionally to zoom so 4x detail looks crisp, not jagged
            detail_gain = 0.72 if zoom_factor == 2 else 0.56
            s_fused = (1.0 - alpha) * s_td_low + alpha * s_ref_low + s_td_high * (detail_gain * land_blend)
            elev_fused = _inv_signed_sqrt(s_fused).astype(np.float32)

            ref_land = ref_viewer_up > 0.0
            gen_land = elev_fused > 0.0
            iou = float(np.logical_and(ref_land, gen_land).sum() / (np.logical_or(ref_land, gen_land).sum() + 1e-6))
            corr = (
                float(np.corrcoef(ref_dem.ravel(), elev_fused.ravel())[0, 1])
                if float(np.std(ref_dem)) > 1e-3 and float(np.std(elev_fused)) > 1e-3
                else 1.0
            )

            self.status.update(
                {
                    "phase": "refining_micro_detail",
                    "message": f"Diffusion done ({t_gen:.1f}s, IoU {iou*100:.0f}%). Sculpting {pixel_m:.1f}m crags…",
                    "progress": 0.70,
                }
            )

            # Preserve exact parent elevation scale in meters (no artificial vertical squashing!)
            elev_viewer = np.where(
                elev_fused > 0.0,
                elev_fused / dem_scale,
                np.maximum(elev_fused / 12.0, -55.0),
            ).astype(np.float32)
            elev_viewer = _relax_angle_of_repose(
                elev_viewer,
                cell_size_m=pixel_m,
                max_slope_deg=58.0,
                iterations=8,
            )
            # Preserve parent climate mean EXACTLY + subtle zero-mean TD30m micro-climate variation
            valid_t = np.isfinite(climate_raw[0])
            td_t_mean = float(np.mean(climate_raw[0][valid_t])) if valid_t.any() else 10.0
            td_t_anom = np.where(valid_t, (climate_raw[0] - td_t_mean) * 0.20, 0.0)
            temp_c = np.clip(ref_temp_up + td_t_anom, -22.0, 38.0).astype(np.float32)

            valid_p = np.isfinite(climate_raw[2]) & (climate_raw[2] > 1.0)
            td_p_mean = float(np.mean(climate_raw[2][valid_p])) if valid_p.any() else 600.0
            td_p_ratio = np.where(valid_p, np.clip(climate_raw[2] / max(td_p_mean, 50.0), 0.78, 1.24), 1.0)
            precip_mm = np.clip(ref_precip_up * td_p_ratio, 35.0, 3500.0).astype(np.float32)
            t_season = np.where(np.isfinite(climate_raw[1]), np.abs(climate_raw[1]), 450.0).astype(np.float32)

            elev_refined = _apply_biome_surface_refinement(
                elev_viewer, temp_c, precip_mm, world_span_m=child_span, seed=td_seed
            )

            res_scale = (out_n / 512.0) ** 2
            hydro_prof = _compute_biome_hydrology_profile(elev_refined, temp_c, precip_mm, res_scale)
            biome_label = hydro_prof["biome"]

            self.status.update(
                {
                    "phase": "solving_hydrology",
                    "message": f"{biome_label} ({pixel_m:.1f}m/px) — solving climate hydrology…",
                    "progress": 0.82,
                }
            )

            t_h0 = time.perf_counter()
            carved, riv_u8, lak_u8, lakes_data, rivers_data = solve_physical_hydrology(
                elev=elev_refined,
                world_span_m=child_span,
                min_lake_cells=max(int(round(22 * res_scale)), 16),
                min_lake_depth_m=0.45,
                max_spring_rivers=hydro_prof["max_spring_rivers"],
                runoff_map=hydro_prof["runoff_map"],
                max_lakes=hydro_prof["max_lakes"],
                min_lake_runoff=hydro_prof["min_lake_runoff"],
            )
            t_hydro = time.perf_counter() - t_h0

            self.status.update(
                {
                    "phase": "exporting",
                    "message": "Saving high-resolution sub-region…",
                    "progress": 0.94,
                }
            )

            # Keep child world ID compact and deterministic
            base_parent = parent_id[:36]
            child_id = f"{base_parent}_z{zoom_factor}_{int(round(cu * 100)):02d}_{int(round(cv * 100)):02d}"
            VIEWER_WORLDS_DIR.mkdir(parents=True, exist_ok=True)
            curv_u8 = compute_curvature_ao(carved)

            temp_range = [-16.0, 32.0]
            precip_range = [0.0, 1800.0]
            t_u8 = np.clip((temp_c - temp_range[0]) / (temp_range[1] - temp_range[0]) * 255.0, 0, 255).astype(np.uint8)
            p_u8 = np.clip((precip_mm - precip_range[0]) / (precip_range[1] - precip_range[0]) * 255.0, 0, 255).astype(np.uint8)
            s_u8 = np.clip(t_season / max(float(np.percentile(t_season, 98)), 1.0) * 255.0, 0, 255).astype(np.uint8)
            clim_rgba = np.stack([t_u8, p_u8, s_u8, np.full((out_n, out_n), 255, dtype=np.uint8)], axis=-1)
            clim_rgba.tofile(VIEWER_WORLDS_DIR / f"{child_id}.climate.u8")

            wet_seed = np.clip((riv_u8.astype(np.float32) + lak_u8.astype(np.float32)) / 255.0, 0.0, 1.0)
            moisture = np.clip(hydro_prof["abs_moist"] + gaussian_filter(wet_seed, sigma=10.0) * 0.75, 0.0, 1.0)

            masks = np.stack([riv_u8, lak_u8, curv_u8, (moisture * 255.0).astype(np.uint8)], axis=-1)
            carved.astype(np.float32).tofile(VIEWER_WORLDS_DIR / f"{child_id}.height.f32")
            masks.astype(np.uint8).tofile(VIEWER_WORLDS_DIR / f"{child_id}.masks.u8")
            (VIEWER_WORLDS_DIR / f"{child_id}.rivers.json").write_text(json.dumps(rivers_data))
            (VIEWER_WORLDS_DIR / f"{child_id}.lakes.json").write_text(json.dumps(lakes_data))

            landmarks = find_landmarks(carved, riv_u8, lak_u8, lakes_data, rivers_data, child_span)
            parent_short = parent_meta.get("title", parent_id).replace(" (30m AI)", "")
            world_meta = {
                "id": child_id,
                "title": f"{parent_short} · {zoom_factor}× Zoom ({pixel_m:.1f}m/px)",
                "subtitle": f"{zoom_factor}× Drill-Down ({child_span/1000:.2f} km · {pixel_m:.2f} m/px · IoU {iou*100:.0f}%)",
                "size": out_n,
                "worldSpan": round(child_span, 2),
                "minElev": round(float(carved.min()), 2),
                "maxElev": round(float(carved.max()), 2),
                "seaLevel": 0.0,
                "hasClimate": True,
                "tempRange": temp_range,
                "precipRange": precip_range,
                "riverCount": len(rivers_data),
                "lakeCount": len(lakes_data),
                "drillOrigin": {
                    "parentWorldId": parent_id,
                    "parentTitle": parent_meta.get("title", parent_id),
                    "centerU": round(cu, 4),
                    "centerV": round(cv, 4),
                    "zoomFactor": zoom_factor,
                    "spanM": round(child_span, 2),
                    "pixelM": round(pixel_m, 2),
                    "depth": depth,
                },
                "landmarks": landmarks,
            }
            if "planetOrigin" in parent_meta:
                world_meta["planetOrigin"] = parent_meta["planetOrigin"]

            (VIEWER_WORLDS_DIR / f"{child_id}.json").write_text(json.dumps(world_meta, indent=2))

            synth_catalog_path = VIEWER_WORLDS_DIR / "synthesized_catalog.json"
            synth_catalog = json.loads(synth_catalog_path.read_text()) if synth_catalog_path.exists() else []
            synth_catalog = [w for w in synth_catalog if w["id"] != child_id]
            synth_catalog.append(world_meta)
            synth_catalog_path.write_text(json.dumps(synth_catalog, indent=2))

            total_s = time.perf_counter() - t0
            self.status.update(
                {
                    "busy": False,
                    "phase": "done",
                    "message": f"Drilled {zoom_factor}× ({pixel_m:.2f}m/px) in {total_s:.1f}s!",
                    "progress": 1.0,
                    "lastWorldId": child_id,
                }
            )
            print(
                f"[Stage2] DRILL DONE {child_id} in {total_s:.1f}s (MPS={t_gen:.1f}s, hydro={t_hydro:.1f}s) | "
                f"IoU={iou*100:.1f}%, r={corr:.3f} | {len(lakes_data)} lakes, {len(rivers_data)} rivers",
                flush=True,
            )

            return {
                "ok": True,
                "worldId": child_id,
                "title": world_meta["title"],
                "zoomFactor": zoom_factor,
                "spanM": round(child_span, 1),
                "pixelM": round(pixel_m, 2),
                "iouPct": round(iou * 100.0, 1),
                "elevCorr": round(corr, 3),
                "lakeCount": len(lakes_data),
                "riverCount": len(rivers_data),
                "mpsSec": round(t_gen, 2),
                "hydroSec": round(t_hydro, 2),
                "totalSec": round(total_s, 2),
                "openUrl": f"/index.html?world={child_id}",
            }


def make_handler(engine: WarmStage2Engine):
    class Stage2Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Quiet polling logs for /api/status and /api/health
            if self.path in ("/api/status", "/api/health"):
                return
            super().log_message(format, *args)

        def _send_json(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path == "/api/health":
                self._send_json(
                    200,
                    {
                        "ready": engine.ready,
                        "device": engine.device,
                        "busy": engine.status["busy"],
                        "phase": engine.status["phase"],
                        "message": engine.status["message"],
                        "error": engine.load_error,
                    },
                )
            elif self.path == "/api/status":
                self._send_json(200, {"ready": engine.ready, "device": engine.device, **engine.status})
            else:
                self._send_json(404, {"error": f"Unknown endpoint {self.path}"})

        def do_POST(self):
            if self.path not in ("/api/synthesize", "/api/drilldown"):
                self._send_json(404, {"error": f"Unknown endpoint {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if self.path == "/api/drilldown":
                    result = engine.drilldown(payload)
                else:
                    result = engine.synthesize(payload)
                self._send_json(200, result)
            except Exception as exc:
                engine.status.update({"busy": False, "phase": "error", "message": str(exc)})
                print(f"[Stage2] Synthesis error: {exc}", flush=True)
                self._send_json(500, {"ok": False, "error": str(exc)})

    return Stage2Handler


def main():
    parser = argparse.ArgumentParser(description="TerraGrok Stage 2 Warm Terrain Diffusion 30m Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pool", type=int, default=2)
    args = parser.parse_args()

    engine = WarmStage2Engine(coarse_pooling=args.pool)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"[Stage2] Listening on http://{args.host}:{args.port} (warming Terrain Diffusion 30m in background…)", flush=True)

    # Warm the model & run ALL MPS synthesis on one dedicated thread (prevents Metal multi-thread SIGBUS!)
    engine.start_worker()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Stage2] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
