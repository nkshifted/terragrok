"""Unit tests for planet tile-grid neighbor rings."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from tile_grid import (  # noqa: E402
    great_circle_deg,
    neighbor_centers,
    tile_id,
)


def test_tile_id_stable_and_distinct():
    a = tile_id(101, 12.5, -40.0, 46.0)
    b = tile_id(101, 12.5, -40.0, 46.0)
    c = tile_id(101, 12.6, -40.0, 46.0)
    assert a == b
    assert a != c
    assert "s101" in a


def test_ring0_has_no_neighbors():
    assert neighbor_centers(10.0, 20.0, 46.0, max_ring=0) == []


def test_ring1_has_eight_unique_neighbors():
    home_lat, home_lon = 15.0, -30.0
    span = 46.0
    neigh = neighbor_centers(home_lat, home_lon, span, max_ring=1)
    assert len(neigh) == 8
    keys = {(n["qi"], n["qj"]) for n in neigh}
    assert keys == {
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    }
    ids = {tile_id(1, n["latDeg"], n["lonDeg"], span) for n in neigh}
    assert len(ids) == 8
    assert all(n["ring"] == 1 for n in neigh)


def test_cardinal_neighbors_approx_span_apart():
    home_lat, home_lon = 0.0, 0.0
    span = 46.0
    neigh = neighbor_centers(home_lat, home_lon, span, max_ring=1)
    by_q = {(n["qi"], n["qj"]): n for n in neigh}
    east = by_q[(1, 0)]
    north = by_q[(0, 1)]
    # At equator, east step ≈ span; north step ≈ span
    assert abs(great_circle_deg(home_lat, home_lon, east["latDeg"], east["lonDeg"]) - span) < 1.5
    assert abs(great_circle_deg(home_lat, home_lon, north["latDeg"], north["lonDeg"]) - span) < 1.5


def test_ring2_adds_sixteen_more():
    neigh = neighbor_centers(5.0, 10.0, 46.0, max_ring=2)
    assert len(neigh) == 8 + 16
    assert sum(1 for n in neigh if n["ring"] == 2) == 16
    # Ring-major: all ring1 before any ring2
    rings = [n["ring"] for n in neigh]
    assert rings == sorted(rings)
