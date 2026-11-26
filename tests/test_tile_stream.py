"""Tile stream planner: synthesis follows the player; ghosts never replace real terrain."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from tile_grid import chart_tile_id  # noqa: E402
from tile_stream import (  # noqa: E402
    MAX_SYNTH_FAILURES,
    SESSION_TTL_S,
    TileStreamRegistry,
    normalize_quality,
    tile_resolution_for_home,
)

HOME = "planet_home_2k"
CHART = {"seed": 1, "tileResolution": "2k", "worldSpan": 4800.0, "vScale": 0.2}


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _registry() -> tuple[TileStreamRegistry, Clock]:
    clock = Clock()
    reg = TileStreamRegistry(clock=clock)
    reg.open_session(HOME, CHART)
    return reg, clock


def _drain(reg: TileStreamRegistry) -> list[tuple[int, int]]:
    """Run the planner to exhaustion, marking every job done; return the order."""
    order = []
    while (job := reg.next_synth_job()) is not None:
        order.append((job["qi"], job["qj"]))
        reg.mark_tile(HOME, job["qi"], job["qj"], job["tileId"], job["resolution"])
        assert len(order) < 100
    return order


def test_quality_labels():
    assert normalize_quality("2K") == "2k"
    assert normalize_quality("1k") == "1k"
    assert normalize_quality("ghost") == "ghost"
    assert tile_resolution_for_home("1K") == "1k"
    assert tile_resolution_for_home("2K") == "2k"
    assert tile_resolution_for_home("4K") == "2k"


def test_ring_one_around_home_edge_neighbours_first_never_home():
    reg, _ = _registry()
    order = _drain(reg)
    assert len(order) == 8 and (0, 0) not in order
    assert set(order[:4]) == {(1, 0), (-1, 0), (0, 1), (0, -1)}


def test_synthesis_follows_the_player_past_ring_one():
    reg, _ = _registry()
    _drain(reg)
    assert reg.next_synth_job() is None
    reg.touch(HOME, 1.0, 0.0)  # walked into the east neighbour
    order = _drain(reg)
    assert set(order) == {(2, -1), (2, 0), (2, 1)}
    assert order[0] == (2, 0)
    reg.touch(HOME, 2.0, 1.0)
    assert set(_drain(reg)) == {(3, 0), (3, 1), (3, 2), (1, 2), (2, 2)}


def test_job_carries_chart_and_stable_tile_id():
    reg, _ = _registry()
    job = reg.next_synth_job()
    assert job["homeWorldId"] == HOME
    assert job["tileId"] == chart_tile_id(HOME, job["qi"], job["qj"])
    assert job["resolution"] == "2k"
    assert job["chart"]["vScale"] == 0.2


def test_idle_session_stops_consuming_synthesis():
    reg, clock = _registry()
    assert reg.next_synth_job() is not None
    clock.t += SESSION_TTL_S + 1
    assert reg.next_synth_job() is None
    reg.touch(HOME, 0.0, 0.0)
    assert reg.next_synth_job() is not None


def test_most_recently_active_world_wins():
    reg, clock = _registry()
    reg.open_session("planet_other_1k", {**CHART, "tileResolution": "1k"})
    clock.t += 1
    reg.touch("planet_other_1k", 0.0, 0.0)
    job = reg.next_synth_job()
    assert job["homeWorldId"] == "planet_other_1k" and job["resolution"] == "1k"


def test_ghost_batch_is_single_flight_and_covers_ring_two():
    reg, _ = _registry()
    batch = reg.claim_ghost_batch(HOME)
    assert len(batch) == 24 and (0, 0) not in batch
    assert reg.claim_ghost_batch(HOME) == []  # already baking
    for qi, qj in batch:
        reg.mark_tile(HOME, qi, qj, chart_tile_id(HOME, qi, qj), "ghost")
    reg.release_ghost_batch(HOME)
    assert reg.claim_ghost_batch(HOME) == []  # nothing missing any more
    reg.touch(HOME, 1.0, 0.0)
    assert {t[0] for t in reg.claim_ghost_batch(HOME)} == {3}


def test_a_late_ghost_never_replaces_real_terrain():
    reg, _ = _registry()
    tid = chart_tile_id(HOME, 1, 0)
    reg.mark_tile(HOME, 1, 0, tid, "2K")
    reg.mark_tile(HOME, 1, 0, tid, "ghost")
    tile = next(t for t in reg.status(HOME)["tiles"] if (t["qi"], t["qj"]) == (1, 0))
    assert tile["quality"] == "2k" and tile["walkable"] is True


def test_upgrade_from_ghost_bumps_revision():
    reg, _ = _registry()
    tid = chart_tile_id(HOME, 0, 1)
    reg.mark_tile(HOME, 0, 1, tid, "ghost")
    rev_ghost = next(t for t in reg.status(HOME)["tiles"] if t["tileId"] == tid)["rev"]
    reg.mark_tile(HOME, 0, 1, tid, "2k")
    tile = next(t for t in reg.status(HOME)["tiles"] if t["tileId"] == tid)
    assert tile["rev"] > rev_ghost and tile["walkable"] is True


def test_repeatedly_failing_tile_is_skipped():
    reg, _ = _registry()
    first = reg.next_synth_job()
    key = (first["qi"], first["qj"])
    for _ in range(MAX_SYNTH_FAILURES):
        reg.mark_failed(HOME, *key)
    assert key not in _drain(reg)


def test_existing_catalog_tiles_are_adopted_only_from_this_chart_scheme():
    reg = TileStreamRegistry(clock=Clock())
    reg.open_session(
        HOME,
        CHART,
        existing_tiles=[
            {"id": "tile_new", "quality": "2k", "tileOrigin": {"homeWorldId": HOME, "qi": 1, "qj": 0, "scheme": "chart"}},
            # Tiles from the earlier per-tile scheme do not line up with a chart; ignore them
            {"id": "tile_old", "quality": "1k", "tileOrigin": {"homeWorldId": HOME, "qi": -1, "qj": 0}},
        ],
    )
    tiles = {(t["qi"], t["qj"]): t for t in reg.status(HOME)["tiles"]}
    assert tiles[(1, 0)]["tileId"] == "tile_new"
    assert (-1, 0) not in tiles
    order = _drain(reg)
    assert (1, 0) not in order and (-1, 0) in order


def test_status_for_unknown_world():
    assert TileStreamRegistry().status("nope")["ok"] is False
