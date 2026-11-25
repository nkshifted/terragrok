"""Biome hydrology profile + physical hydrology invariants (no MPS required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from biome_profile import compute_biome_hydrology_profile  # noqa: E402
from export_viewer_worlds import find_landmarks, solve_physical_hydrology  # noqa: E402


def _bowl_island(n: int = 96, span_m: float = 2400.0) -> np.ndarray:
    """Central island with a closed alpine depression that should become a lake."""
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cz = (n - 1) * 0.5
    r = np.hypot(xx - cz, yy - cy) / (n * 0.42)
    elev = 90.0 * (1.0 - r)
    # Inland depression near center-left
    bowl = np.hypot(xx - (cz - n * 0.08), yy - cy) / (n * 0.08)
    elev = elev - 28.0 * np.clip(1.0 - bowl, 0.0, 1.0)
    elev = np.where(r > 1.05, -40.0, elev)
    return elev.astype(np.float32)


def test_arid_desert_profile_zero_hydro_budget():
    n = 64
    elev = np.full((n, n), 40.0, dtype=np.float32)
    elev[:4, :] = -20.0
    temp = np.full((n, n), 28.0, dtype=np.float32)
    precip = np.full((n, n), 180.0, dtype=np.float32)
    prof = compute_biome_hydrology_profile(elev, temp, precip, res_scale=1.0)
    assert prof["biome"] == "Arid Desert"
    assert prof["max_lakes"] == 0
    assert prof["max_spring_rivers"] == 0
    assert float(np.mean(prof["abs_moist"][elev > 0])) < 0.12


def test_polar_glacier_profile_zero_hydro_budget():
    n = 64
    elev = np.full((n, n), 50.0, dtype=np.float32)
    elev[:4, :] = -20.0
    temp = np.full((n, n), -12.0, dtype=np.float32)
    precip = np.full((n, n), 400.0, dtype=np.float32)
    prof = compute_biome_hydrology_profile(elev, temp, precip, res_scale=1.0)
    assert prof["biome"] == "Polar Glacier"
    assert prof["max_lakes"] == 0
    assert prof["max_spring_rivers"] == 0
    assert float(np.max(prof["abs_moist"])) < 0.05


def test_arid_fast_path_returns_zero_rivers_and_lakes():
    elev = _bowl_island()
    carved, riv, lak, lakes, rivers = solve_physical_hydrology(
        elev=elev,
        world_span_m=2400.0,
        max_spring_rivers=0,
        max_lakes=0,
    )
    assert lakes == []
    assert rivers == []
    assert int(riv.max()) == 0
    assert int(lak.max()) == 0
    assert carved.shape == elev.shape


def test_temperate_hydrology_has_zero_dead_end_rivers():
    elev = _bowl_island(n=128, span_m=3200.0)
    runoff = np.ones_like(elev, dtype=np.float32)
    carved, riv, lak, lakes, rivers = solve_physical_hydrology(
        elev=elev,
        world_span_m=3200.0,
        min_lake_cells=20,
        max_spring_rivers=10,
        max_lakes=6,
        runoff_map=runoff,
        min_lake_runoff=0.0,
    )
    assert len(rivers) >= 1
    for rv in rivers:
        assert rv.get("dest") != "dead_end"
        assert len(rv.get("points") or []) >= 2


def test_plains_landmark_prefers_map_center():
    n = 128
    elev = np.full((n, n), 25.0, dtype=np.float32)
    # Peak shoved into the corner (old spawn bug case)
    elev[2:8, n - 8 : n - 2] = 220.0
    elev[:3, :] = -30.0
    elev[-3:, :] = -30.0
    elev[:, :3] = -30.0
    elev[:, -3:] = -30.0
    rivers = np.zeros((n, n), dtype=np.uint8)
    lakes = np.zeros((n, n), dtype=np.uint8)
    lm = find_landmarks(elev, rivers, lakes, [], [], world_span_m=2400.0)
    px, pz = lm["plains"]
    # Should be near center, not near the corner peak
    assert 0.35 < px < 0.65
    assert 0.35 < pz < 0.65
