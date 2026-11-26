"""Tests for ghost tile persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from tile_ghost import list_tiles_for_home, write_ghost_tile  # noqa: E402


def test_write_ghost_tile_persists_height_and_catalog(tmp_path: Path):
    elev = np.linspace(-20, 80, 64 * 64, dtype=np.float32).reshape(64, 64)
    cat = tmp_path / "synthesized_catalog.json"
    out = write_ghost_tile(
        elev,
        tile_id="tile_planet_home_e1_s0",
        seed=42,
        lat_deg=10.0,
        lon_deg=-20.0,
        span_deg=46.0,
        home_world_id="planet_home",
        qi=1,
        qj=0,
        ring=1,
        worlds_dir=tmp_path,
        catalog_path=cat,
    )
    assert out["ok"]
    tid = out["tileId"]
    assert (tmp_path / f"{tid}.json").exists()
    raw = np.fromfile(tmp_path / f"{tid}.ghost.height.f32", dtype=np.float32)
    assert raw.shape == (64 * 64,)
    listed = list_tiles_for_home("planet_home", catalog_path=cat)
    assert any(e["id"] == tid for e in listed)
    meta = json.loads((tmp_path / f"{tid}.json").read_text())
    assert meta["quality"] == "ghost"
    assert meta["tileOrigin"]["qi"] == 1
    assert meta["tileOrigin"]["scheme"] == "chart"


def test_ghost_never_overwrites_a_synthesized_tile(tmp_path: Path):
    cat = tmp_path / "synthesized_catalog.json"
    tid = "tile_planet_home_e1_s0"
    cat.write_text(json.dumps([{"id": tid, "quality": "2k", "tileOrigin": {"homeWorldId": "planet_home"}}]))
    out = write_ghost_tile(
        np.zeros((8, 8), dtype=np.float32),
        tile_id=tid,
        seed=42,
        lat_deg=0.0,
        lon_deg=0.0,
        home_world_id="planet_home",
        qi=1,
        qj=0,
        ring=1,
        worlds_dir=tmp_path,
        catalog_path=cat,
    )
    assert out["skipped"] is True
    assert not (tmp_path / f"{tid}.ghost.height.f32").exists()
    assert json.loads(cat.read_text())[0]["quality"] == "2k"
