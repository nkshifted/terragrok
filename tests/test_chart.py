"""Chart helpers: what must be identical across every tile of a chart."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from chart import (  # noqa: E402
    chart_from_home_meta,
    chart_seed,
    coarse_size_for,
    collect_edge_inflows,
    dem_to_viewer,
    estimate_home_v_scale,
    relief_v_scale,
)
from paths import main_checkout_root, vendor_terrain_diffusion  # noqa: E402


def test_chart_seed_depends_only_on_planet_and_home():
    a = chart_seed(523, -12.91, -45.18)
    assert a == chart_seed(523, -12.91, -45.18)
    assert a != chart_seed(524, -12.91, -45.18)
    assert a != chart_seed(523, -12.91, -45.3)
    assert 0 <= a <= 0x7FFFFFFF


def test_coarse_size():
    assert [coarse_size_for(r) for r in ("1k", "2K", "4k")] == [8, 16, 32]


def test_dem_to_viewer_is_monotonic_and_continuous_at_the_coast_and_knee():
    dem = np.linspace(-3000.0, 4500.0, 20001, dtype=np.float32)
    out = dem_to_viewer(dem, 0.2)
    assert np.all(np.diff(out) >= -1e-4)
    assert out.min() >= -55.0
    assert float(np.abs(np.diff(out)).max()) < 0.2  # no jump at sea level or at the 330 m knee
    assert out.max() < 330.0 + 115.0 + 1e-3


def test_relief_scale_is_clamped():
    flat = np.full((32, 32), 200.0, dtype=np.float32)
    alpine = np.full((32, 32), 4200.0, dtype=np.float32)
    assert relief_v_scale(flat) == pytest.approx(420.0 / 1550.0)
    assert relief_v_scale(alpine) == pytest.approx(0.1)


def test_estimate_home_v_scale_recovers_the_scale_across_resolutions():
    rng = np.random.default_rng(0)
    ref = (rng.random((64, 64), dtype=np.float32) * 1200.0 + 80.0).astype(np.float32)
    home = np.repeat(np.repeat(ref, 4, axis=0), 4, axis=1) * 0.21
    assert estimate_home_v_scale(home.astype(np.float32), ref) == pytest.approx(0.21, abs=0.005)


def test_chart_from_home_meta_defaults_and_rejections():
    meta = {
        "id": "planet_s1_earth_10n_20e_2k",
        "resolution": "2K",
        "worldSpan": 4800.0,
        "planetOrigin": {"seed": 1, "archetype": "earth", "latDeg": 10.0, "lonDeg": 20.0, "spanDeg": 46.0},
    }
    chart = chart_from_home_meta(meta)
    assert chart["tileResolution"] == "2k"
    assert (chart["climateMode"], chart["seaLevelBias"], chart["reliefScale"]) == ("earth", 0.0, 1.0)
    assert (chart["quality"], chart["fidelity"]) == ("high", "exact")
    # Worlds made before the time slider are the planet's "today"; later ones keep their moment
    assert chart["geoTimeMyr"] == 0.0
    # Cones only for charts whose home tile was built from the current set of volcano sites
    assert chart["volcanoes"] is False
    for version, expect in ((1, False), (2, True)):
        po = {**meta["planetOrigin"], "volcanoVersion": version}
        assert chart_from_home_meta({**meta, "planetOrigin": po})["volcanoes"] is expect
    aged = {**meta, "planetOrigin": {**meta["planetOrigin"], "geoTimeMyr": 60.0}}
    assert chart_from_home_meta(aged)["geoTimeMyr"] == 60.0
    assert chart["vScale"] is None
    assert chart_from_home_meta({**meta, "resolution": "1K", "vScale": 0.25})["tileResolution"] == "1k"
    with pytest.raises(ValueError):
        chart_from_home_meta({"id": "island"})
    # Drill-down children and streamed tiles carry a planetOrigin but are not chart homes
    with pytest.raises(ValueError):
        chart_from_home_meta({**meta, "drillOrigin": {"parentWorldId": "x"}})
    with pytest.raises(ValueError):
        chart_from_home_meta({**meta, "tileOrigin": {"homeWorldId": "x"}})


def test_vendor_dir_falls_back_to_the_main_checkout_from_a_worktree(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TERRAGROK_VENDOR_DIR", raising=False)
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".vendor" / "terrain-diffusion" / "terrain_diffusion").mkdir(parents=True)
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")

    assert main_checkout_root(wt) == main
    assert vendor_terrain_diffusion(wt) == main / ".vendor" / "terrain-diffusion"
    assert main_checkout_root(main) is None
    # A local clone wins over the main checkout's
    (wt / ".vendor" / "terrain-diffusion" / "terrain_diffusion").mkdir(parents=True)
    assert vendor_terrain_diffusion(wt) == wt / ".vendor" / "terrain-diffusion"


def test_edge_inflows_come_only_from_rivers_leaving_through_the_shared_edge(tmp_path: Path):
    import json

    home = "planet_home_2k"
    west = "tile_planet_home_2k_w1_s0"
    catalog = [
        {"id": home},
        {"id": west, "quality": "2k", "tileOrigin": {"homeWorldId": home, "qi": -1, "qj": 0, "scheme": "chart"}},
        {"id": "ghost", "quality": "ghost", "tileOrigin": {"homeWorldId": home, "qi": -2, "qj": 1, "scheme": "chart"}},
    ]
    # Home: one river out of its WEST edge at z=+1200, one out of its SOUTH edge, one to the sea
    (tmp_path / f"{home}.rivers.json").write_text(
        json.dumps(
            [
                {"dest": "edge", "points": [[0, 0, 50, 4, 0.1], [-2398.0, 1200.0, 30, 9, 0.02]]},
                {"dest": "edge", "q": 777.0, "points": [[0, 0, 50, 4, 0.1], [600.0, 2400.0, 20, 9, 0.02]]},
                {"dest": "sea", "points": [[0, 0, 50, 4, 0.1], [2400.0, 0.0, 0, 9, 0.02]]},
            ]
        )
    )
    (tmp_path / f"{west}.rivers.json").write_text(
        json.dumps([{"dest": "edge", "q": 55.5, "points": [[0, 0, 9, 4, 0.1], [-600.0, 2400.0, 5, 6, 0.02]]}])
    )

    # West tile (-1, 0): receives home's west-edge river through ITS east edge
    assert collect_edge_inflows(home, -1, 0, catalog, tmp_path) == [{"edge": "E", "t": 0.75, "q": None}]
    # South tile (0, 1): receives home's south-edge river through its north edge
    assert collect_edge_inflows(home, 0, 1, catalog, tmp_path) == [{"edge": "N", "t": 0.625, "q": 777.0}]
    # South-west tile (-1, 1): the west tile's south-edge river; the ghost beside it contributes nothing
    assert collect_edge_inflows(home, -1, 1, catalog, tmp_path) == [{"edge": "N", "t": 0.375, "q": 55.5}]
    assert collect_edge_inflows(home, 1, 0, catalog, tmp_path) == []


def _smooth_field(rows: int, cols: int, col0: int, scale: float, offset: float) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, col0 : col0 + cols].astype(np.float32)
    return (offset + scale * np.sin(xx / 37.0) * np.cos(yy / 23.0) + scale * xx / 90.0).astype(np.float32)


def test_climate_blend_is_continuous_across_a_seam():
    """Two tiles cut from one world must get the climate the whole world would have got."""
    from chart import blend_climate

    n = 64
    raw_t = _smooth_field(n, 2 * n, 0, 9.0, 4.0)
    ref_t = _smooth_field(n, 2 * n, 5, 8.0, 5.0)
    raw_p = 900.0 + 60.0 * _smooth_field(n, 2 * n, 11, 5.0, 0.0)
    ref_p = 700.0 + 50.0 * _smooth_field(n, 2 * n, 3, 5.0, 0.0)
    whole_t, whole_p = blend_climate(raw_t, raw_p, ref_t, ref_p)
    west_t, west_p = blend_climate(raw_t[:, :n], raw_p[:, :n], ref_t[:, :n], ref_p[:, :n])
    east_t, east_p = blend_climate(raw_t[:, n:], raw_p[:, n:], ref_t[:, n:], ref_p[:, n:])
    # The two halves have very different means: a per-tile mean would show up here
    assert abs(float(raw_t[:, :n].mean() - raw_t[:, n:].mean())) > 3.0
    np.testing.assert_array_equal(np.hstack([west_t, east_t]), whole_t)
    np.testing.assert_array_equal(np.hstack([west_p, east_p]), whole_p)


def test_climate_blend_stays_near_the_planet_climate():
    from chart import blend_climate

    ref_t = np.full((8, 8), 10.0, np.float32)
    ref_p = np.full((8, 8), 100.0, np.float32)
    t, p = blend_climate(ref_t + 10.0, ref_p * 50.0, ref_t, ref_p)
    assert np.allclose(t, 12.8, atol=1e-4)
    assert np.allclose(p, 132.0, atol=1e-2)  # a desert stays a desert
    t, p = blend_climate(np.full((8, 8), np.nan, np.float32), np.zeros((8, 8), np.float32), ref_t, ref_p)
    assert np.allclose(t, 10.0) and np.allclose(p, 100.0)


def test_climate_packing_does_not_depend_on_the_tile():
    from chart import pack_climate

    t = np.full((16, 16), 8.0, np.float32)
    p = np.full((16, 16), 900.0, np.float32)
    calm = pack_climate(t, p, np.full((16, 16), 300.0, np.float32), 8)
    wild = pack_climate(t, p, np.linspace(300.0, 1400.0, 256, dtype=np.float32).reshape(16, 16), 8)
    assert calm.shape == (8, 8, 4) and calm.dtype == np.uint8
    assert calm[0, 0, 2] == wild[0, 0, 2] == 51  # same seasonality, same byte, whatever else is in the tile
    assert calm[0, 0, 0] == 127 and calm[0, 0, 1] == 127 and calm[0, 0, 3] == 255


def test_viewer_to_dem_inverts_the_vertical_mapping_on_land():
    from chart import viewer_to_dem

    dem = np.array([0.5, 40.0, 900.0, 2000.0, 2600.0], dtype=np.float32)
    back = viewer_to_dem(dem_to_viewer(dem, 0.18), 0.18)
    np.testing.assert_allclose(back, dem, rtol=2e-3)
    assert viewer_to_dem(dem_to_viewer(np.array([-800.0], np.float32), 0.18), 0.18)[0] == 0.0
