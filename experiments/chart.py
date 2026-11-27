"""Chart helpers shared by Stage 2 synthesis and the tile stream (numpy only, no torch).

A chart is the flat world a home patch lives in. Everything that must agree across tile
edges is fixed per chart here: the planet settings, the projection origin, the diffusion
seed, the synthesis settings and the vertical scale.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from paths import EXPERIMENTS_DIR, REPO_ROOT

STAGE1_PATCH_SCRIPT = EXPERIMENTS_DIR / "stage1_patch.mjs"
WORLD_SPAN_M = 4800.0
GHOST_SIZE = 256
V_SCALE_MIN = 420.0 / 4500.0
V_SCALE_MAX = 0.28


def decode_f32(b64_str: str, shape: tuple[int, int]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float32).copy().reshape(shape)


def coarse_size_for(resolution: str) -> int:
    r = str(resolution).lower()
    return 32 if "4" in r else (8 if "1" in r else 16)


def chart_seed(seed: int, home_lat_deg: float, home_lon_deg: float) -> int:
    """Diffusion seed shared by every tile of a chart (same formula the home patch always used)."""
    return (
        (int(seed) * 73856093)
        ^ (int(round((home_lat_deg + 90.0) * 10)) * 19349663)
        ^ (int(round((home_lon_deg + 180.0) * 10)) * 83492791)
    ) & 0x7FFFFFFF


def relief_v_scale(elev_dem: np.ndarray) -> float:
    """DEM metres -> viewer metres factor for a patch (flat regions stay flat, ranges reach ~420 m)."""
    land = elev_dem > 0.0
    max_land_m = float(np.percentile(elev_dem[land], 99.6)) if land.any() else 1000.0
    return min(420.0 / max(max_land_m, 1550.0), V_SCALE_MAX)


def dem_to_viewer(elev_dem: np.ndarray, v_scale: float) -> np.ndarray:
    """Vertical mapping used by every tile of a chart: scaled land, shallow sea, soft-kneed peaks."""
    land = elev_dem > 0.0
    out = np.where(land, elev_dem * v_scale, np.maximum(elev_dem * 0.05, -55.0)).astype(np.float32)
    return np.where(out > 330.0, 330.0 + 115.0 * np.tanh((out - 330.0) / 150.0), out).astype(np.float32)


def viewer_to_dem(elev_viewer: np.ndarray, v_scale: float) -> np.ndarray:
    """Inverse of `dem_to_viewer` on land (sea comes back as 0: nothing below it is needed)."""
    h = np.asarray(elev_viewer, dtype=np.float32)
    knee = 330.0 + 150.0 * np.arctanh(np.clip((h - 330.0) / 115.0, 0.0, 0.999))
    return (np.maximum(np.where(h > 330.0, knee, h), 0.0) / max(float(v_scale), 1e-6)).astype(np.float32)


# Bump when the climate rasters on disk change meaning; older tiles are refreshed on first use
CLIMATE_VERSION = 2
CLIMATE_TEMP_RANGE = [-16.0, 32.0]
CLIMATE_PRECIP_RANGE = [0.0, 1800.0]
CLIMATE_SEASON_MAX = 1500.0
_TD_CLIMATE_WEIGHT = 0.28


def blend_climate(raw_temp, raw_precip, ref_temp, ref_precip) -> tuple[np.ndarray, np.ndarray]:
    """
    Final temperature (deg C) and rainfall (mm/yr): the planet's climate pulled 28% of the way
    towards the diffusion model's, which adds lapse-rate and rain-shadow detail.

    Strictly per-pixel on purpose. Both inputs are continuous across the tiles of a chart, so the
    result is too; anything computed over a tile (a mean to centre on, a percentile to scale by)
    differs from tile to tile and shows up as a straight climate line along every seam.
    """
    ref_temp = np.asarray(ref_temp, dtype=np.float32)
    ref_precip = np.asarray(ref_precip, dtype=np.float32)
    raw_temp = np.asarray(raw_temp, dtype=np.float32)
    raw_precip = np.asarray(raw_precip, dtype=np.float32)
    d_t = np.where(np.isfinite(raw_temp), raw_temp - ref_temp, 0.0)
    temp_c = np.clip(ref_temp + _TD_CLIMATE_WEIGHT * d_t, -22.0, 38.0)
    ok = np.isfinite(raw_precip) & (raw_precip > 1.0)
    ratio = np.where(ok, np.maximum(raw_precip, 1.0) / np.maximum(ref_precip, 1.0), 1.0)
    precip_mm = np.clip(ref_precip * np.clip(ratio**_TD_CLIMATE_WEIGHT, 0.72, 1.32), 35.0, 3500.0)
    return temp_c.astype(np.float32), precip_mm.astype(np.float32)


def pack_climate(temp_c: np.ndarray, precip_mm: np.ndarray, t_season: np.ndarray, climate_n: int) -> np.ndarray:
    """RGBA8 climate raster (temp, rainfall, seasonality, 255) at climate_n², fixed ranges."""
    from scipy.ndimage import zoom

    def fit(a):
        a = np.asarray(a, dtype=np.float32)
        return a if a.shape[0] == climate_n else zoom(a, climate_n / float(a.shape[0]), order=1)

    def u8(a, lo, hi):
        return np.clip((fit(a) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    return np.stack(
        [
            u8(temp_c, *CLIMATE_TEMP_RANGE),
            u8(precip_mm, *CLIMATE_PRECIP_RANGE),
            u8(t_season, 0.0, CLIMATE_SEASON_MAX),
            np.full((climate_n, climate_n), 255, dtype=np.uint8),
        ],
        axis=-1,
    )


# Bump when Stage 1 picks different volcano sites: a chart only gets cones in new tiles if its
# home tile was built from the same set (see viewer/src/tectonics.js pickVolcanoes)
VOLCANO_VERSION = 2


def chart_from_home_meta(meta: dict) -> dict:
    """Chart parameters for a synthesized home world. Raises if it has no planet origin."""
    po = meta.get("planetOrigin")
    if not po or meta.get("drillOrigin") or meta.get("tileOrigin"):
        raise ValueError(f"{meta.get('id')} is not a planet-synthesized home world; it has no chart")
    home_res = str(meta.get("resolution") or "2k").lower()
    return {
        "homeWorldId": meta["id"],
        "seed": int(po["seed"]),
        "archetype": po.get("archetype") or "earth",
        # Older home worlds did not record these; they were synthesized with the defaults
        "climateMode": po.get("climateMode") or "earth",
        "seaLevelBias": float(po.get("seaLevelBias") or 0.0),
        # Geological time of the planet this chart was cut from (0 = the planet as first generated)
        "geoTimeMyr": float(po.get("geoTimeMyr") or 0.0),
        # Cones are built into the terrain only for charts whose home tile has them; adding
        # them to the later tiles of an older chart would put a cliff where a cone meets its seam
        "volcanoes": int(po.get("volcanoVersion") or 0) == VOLCANO_VERSION,
        "reliefScale": float(po.get("reliefScale") or 1.0),
        "homeLatDeg": float(po["latDeg"]),
        "homeLonDeg": float(po["lonDeg"]),
        "spanDeg": float(po.get("spanDeg") or 46.0),
        "quality": str(po.get("quality") or "high"),
        "fidelity": str(po.get("fidelity") or "exact"),
        "tileResolution": "1k" if "1" in home_res else "2k",
        "worldSpan": float(meta.get("worldSpan") or WORLD_SPAN_M),
        "vScale": float(meta["vScale"]) if meta.get("vScale") else None,
    }


def run_stage1_patch(chart: dict, tiles: list[dict], timeout_s: float = 60.0) -> list[dict]:
    """Sample Stage 1 for chart tiles via node (see experiments/stage1_patch.mjs)."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node not found on PATH (start the server through `just`/mise so it is)")
    req = {k: chart[k] for k in ("seed", "archetype", "climateMode", "seaLevelBias", "reliefScale", "homeLatDeg", "homeLonDeg", "spanDeg")}
    req["geoTimeMyr"] = chart.get("geoTimeMyr", 0.0)
    req["tiles"] = tiles
    proc = subprocess.run(
        [node, str(STAGE1_PATCH_SCRIPT)],
        input=json.dumps(req).encode("utf-8"),
        capture_output=True,
        cwd=str(REPO_ROOT),
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"stage1_patch.mjs failed: {proc.stderr.decode('utf-8', 'replace')[-400:]}")
    return json.loads(proc.stdout.decode("utf-8"))["tiles"]


def estimate_home_v_scale(home_heights: np.ndarray, ref_dem: np.ndarray) -> float:
    """
    Recover the vertical scale of a home world synthesized before `vScale` was recorded, by
    regressing its heights on the Stage 1 reference it was coastline-locked to.
    """
    n = home_heights.shape[0]
    r = ref_dem.shape[0]
    if n != r:
        idx = np.clip(((np.arange(n) + 0.5) * r / n).astype(int), 0, r - 1)
        ref = ref_dem[np.ix_(idx, idx)]
    else:
        ref = ref_dem
    use = (ref > 60.0) & (home_heights > 2.0) & (home_heights < 300.0)
    if int(use.sum()) < 500:
        return float(np.clip(relief_v_scale(ref_dem), V_SCALE_MIN, V_SCALE_MAX))
    v = float((home_heights[use] * ref[use]).sum() / (ref[use] ** 2).sum())
    return float(np.clip(v, V_SCALE_MIN, V_SCALE_MAX))


def collect_edge_inflows(
    home_world_id: str,
    qi: int,
    qj: int,
    catalog: list[dict],
    worlds_dir: Path,
    world_span_m: float = WORLD_SPAN_M,
) -> list[dict]:
    """
    Rivers of already-synthesized edge neighbours that leave through the edge they share with
    chart tile (qi, qj). Returned as {edge, t, q}: `edge` is OUR edge (W/E/N/S), `t` the position
    along it (0 = north or west end), `q` the river's discharge there (None if not recorded).
    """
    by_q: dict[tuple[int, int], str] = {(0, 0): home_world_id}
    for e in catalog:
        to = e.get("tileOrigin") or {}
        if to.get("homeWorldId") == home_world_id and to.get("scheme") == "chart" and e.get("quality") != "ghost":
            by_q[(int(to["qi"]), int(to["qj"]))] = e["id"]

    half = world_span_m * 0.5
    tol = world_span_m * 0.006  # a few cells: river ends are smoothed slightly off the exact edge
    # (neighbour offset, our edge, neighbour's axis index that must sit at its far side, sign)
    sides = [((-1, 0), "W", 0, +1), ((1, 0), "E", 0, -1), ((0, -1), "N", 1, +1), ((0, 1), "S", 1, -1)]
    out: list[dict] = []
    for (di, dj), edge, axis, sign in sides:
        wid = by_q.get((qi + di, qj + dj))
        if not wid:
            continue
        path = worlds_dir / f"{wid}.rivers.json"
        if not path.exists():
            continue
        try:
            rivers = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for r in rivers:
            if not isinstance(r, dict) or r.get("dest") != "edge" or not r.get("points"):
                continue
            end = r["points"][-1]
            if abs(end[axis] - sign * half) > tol:
                continue  # leaves through a different edge of the neighbour
            along = end[1 - axis]
            out.append({"edge": edge, "t": round(min(1.0, max(0.0, along / world_span_m + 0.5)), 5), "q": r.get("q")})
    return out
