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


class WarmStage2Engine:
    def __init__(
        self,
        coarse_pooling: int = 2,
        cond_snr: tuple[float, ...] = (0.15, 0.2, 1.0, 0.25, 1.0),
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
            payload, done_evt, holder = self._job_queue.get()
            try:
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
        self._job_queue.put((payload, done_evt, holder))
        done_evt.wait()
        if "error" in holder:
            raise holder["error"]
        return holder["result"]

    def load_model(self):
        t0 = time.perf_counter()
        try:
            print(f"[Stage2] Loading xandergos/terrain-diffusion-30m onto {self.device} (pool={self.coarse_pooling})…", flush=True)
            world = WorldPipeline.from_pretrained(
                "xandergos/terrain-diffusion-30m",
                seed=42,
                coarse_pooling=self.coarse_pooling,
                histogram_raw=[0.0, 0.0, 0.0, 1.0, 1.8],
                torch_compile=False,
                caching_strategy="direct",
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

            # Fidelity metrics vs Stage 1 tectonic reference
            ref_up = zoom(ref_elev, out_n / ref_size, order=1)
            ref_land = ref_up > 0.0
            gen_land = elev_raw > 0.0
            inter = np.logical_and(ref_land, gen_land).sum()
            union = np.logical_or(ref_land, gen_land).sum() + 1e-6
            iou = float(inter / union)
            std_ref = float(np.std(ref_up))
            std_gen = float(np.std(elev_raw))
            corr = (
                float(np.corrcoef(ref_up.ravel(), elev_raw.ravel())[0, 1])
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
            land_mask = elev_raw > 0.0
            max_land_m = float(np.percentile(elev_raw[land_mask], 99.6)) if land_mask.any() else 1000.0
            v_scale = 460.0 / max(max_land_m, 400.0)
            elev_viewer = np.where(land_mask, elev_raw * v_scale, np.maximum(elev_raw * 0.05, -55.0)).astype(np.float32)
            elev_refined = apply_micro_detail_refinement(elev_viewer, world_span_m=world_span_m, seed=td_seed)

            ref_temp_up = zoom(ref_temp, out_n / ref_size, order=1)
            ref_precip_up = zoom(ref_precip, out_n / ref_size, order=1)
            temp_c = np.where(np.isfinite(climate_raw[0]), climate_raw[0], ref_temp_up).astype(np.float32)
            precip_mm = np.where(np.isfinite(climate_raw[2]), np.clip(climate_raw[2], 50.0, 3500.0), ref_precip_up).astype(np.float32)
            t_season = np.where(np.isfinite(climate_raw[1]), np.abs(climate_raw[1]), 450.0).astype(np.float32)

            p_med = max(float(np.median(precip_mm[elev_refined > 0])) if (elev_refined > 0).any() else 600.0, 120.0)
            runoff_map = np.clip(precip_mm / p_med, 0.35, 1.85).astype(np.float32)

            self.status.update(
                {
                    "phase": "solving_hydrology",
                    "message": "Solving Landlab lakes & physics-widened valley rivers…",
                    "progress": 0.82,
                }
            )

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
                "title": f"Planet #{seed} · {coord_label} (30m AI)",
                "subtitle": f"Tectonic {archetype} @ {coord_label} → TD30m (IoU {iou*100:.0f}%)",
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
            if self.path != "/api/synthesize":
                self._send_json(404, {"error": f"Unknown endpoint {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
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
