"""Chart tile-grid helpers: ids and player-centred windows."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from tile_grid import chart_tile_id, tile_of, window_tiles  # noqa: E402


def test_tile_ids_are_stable_distinct_and_gitignored():
    ids = {chart_tile_id("planet_home_2k", qi, qj) for qi in range(-2, 3) for qj in range(-2, 3)}
    assert len(ids) == 25
    assert chart_tile_id("planet_home_2k", 1, -2) == "tile_planet_home_2k_e1_n2"
    assert chart_tile_id("planet_home_2k", -3, 4) == "tile_planet_home_2k_w3_s4"
    # .gitignore keeps synthesized tiles out of the repo by the `tile_` prefix
    assert all(i.startswith("tile_") for i in ids)


def test_tile_of_uses_tile_centres():
    assert tile_of(0.0, 0.0) == (0, 0)
    assert tile_of(0.49, -0.49) == (0, 0)
    assert tile_of(0.51, -0.51) == (1, -1)
    assert tile_of(-1.2, 2.4) == (-1, 2)


def test_window_is_centred_on_the_players_tile_nearest_first():
    win = window_tiles(0.0, 0.0, ring=1)
    assert len(win) == 9 and len(set(win)) == 9
    assert win[0] == (0, 0)
    # The four edge-sharing neighbours come before the four diagonals
    assert set(win[1:5]) == {(1, 0), (-1, 0), (0, 1), (0, -1)}


def test_window_follows_the_player_and_prefers_the_side_they_are_near():
    win = window_tiles(2.4, -1.0, ring=1)
    assert win[0] == (2, -1)
    assert {t[0] for t in win} == {1, 2, 3} and {t[1] for t in win} == {-2, -1, 0}
    # Standing east of the tile centre: the east neighbour outranks the west one
    assert win.index((3, -1)) < win.index((1, -1))


def test_wider_ring():
    assert len(window_tiles(0.0, 0.0, ring=2)) == 25
