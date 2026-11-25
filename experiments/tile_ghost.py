"""Persist Stage-1 ghost height tiles for planet tile streaming (no torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tile_grid import tile_id
from world_cache import SYNTH_CATALOG_PATH, VIEWER_WORLDS_DIR, load_synth_catalog, save_synth_catalog

GHOST_SUFFIX = ".ghost.height.f32"
DEFAULT_WORLD_SPAN_M = 4800.0


def ghost_paths(tile: str, worlds_dir: Path | None = None) -> tuple[Path, Path]:
    root = worlds_dir or VIEWER_WORLDS_DIR
    return root / f"{tile}.json", root / f"{tile}{GHOST_SUFFIX}"


def write_ghost_tile(
    elev: np.ndarray,
    *,
    seed: int,
    lat_deg: float,
    lon_deg: float,
    span_deg: float = 46.0,
    home_world_id: str,
    qi: int,
    qj: int,
    ring: int,
    world_span_m: float = DEFAULT_WORLD_SPAN_M,
    worlds_dir: Path | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Write ghost height + meta and upsert synthesized_catalog entry."""
    if elev.dtype != np.float32:
        elev = elev.astype(np.float32)
    if elev.ndim != 2 or elev.shape[0] != elev.shape[1]:
        raise ValueError("elev must be a square float32 array")
    size = int(elev.shape[0])
    tid = tile_id(seed, lat_deg, lon_deg, span_deg)
    root = worlds_dir or VIEWER_WORLDS_DIR
    root.mkdir(parents=True, exist_ok=True)
    meta_path, height_path = ghost_paths(tid, worlds_dir=root)

    meta = {
        "id": tid,
        "title": f"Ghost · {lat_deg:.0f}°,{lon_deg:.0f}° (ring {ring})",
        "subtitle": f"Stage-1 ghost {size}² · awaiting AI synth",
        "size": size,
        "worldSpan": float(world_span_m),
        "minElev": round(float(elev.min()), 2),
        "maxElev": round(float(elev.max()), 2),
        "seaLevel": 0.0,
        "quality": "ghost",
        "hasClimate": False,
        "riverCount": 0,
        "lakeCount": 0,
        "planetOrigin": {
            "seed": int(seed),
            "latDeg": round(float(lat_deg), 2),
            "lonDeg": round(float(lon_deg), 2),
            "spanDeg": float(span_deg),
        },
        "tileOrigin": {
            "homeWorldId": home_world_id,
            "qi": int(qi),
            "qj": int(qj),
            "ring": int(ring),
            "quality": "ghost",
        },
        "landmarks": {"plains": [0.5, 0.5]},
    }
    height_path.write_bytes(elev.tobytes())
    meta_path.write_text(json.dumps(meta, indent=2))

    cat_path = catalog_path or SYNTH_CATALOG_PATH
    catalog = load_synth_catalog(cat_path)
    catalog = [e for e in catalog if e.get("id") != tid]
    catalog.append(meta)
    save_synth_catalog(catalog, catalog_path=cat_path, worlds_dir=root)
    return {"ok": True, "tileId": tid, "world": meta}


def list_tiles_for_home(
    home_world_id: str,
    *,
    catalog_path: Path | None = None,
) -> list[dict]:
    catalog = load_synth_catalog(catalog_path)
    return [
        e
        for e in catalog
        if (e.get("tileOrigin") or {}).get("homeWorldId") == home_world_id
        or e.get("id") == home_world_id
    ]
