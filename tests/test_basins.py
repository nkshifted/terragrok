"""Landlocked basins below sea level become dry pans, never patches of sea."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from basins import PAN_LEVEL_M, dry_landlocked_basins  # noqa: E402
from volcano import pack_rock  # noqa: E402


def island_with_hollow():
    """Sea all round, land in the middle, and a hollow below sea level inside the land."""
    elev = np.full((80, 80), -20.0, dtype=np.float32)
    elev[10:70, 10:70] = 30.0
    elev[30:50, 30:50] = -6.0
    return elev


def test_inland_hollow_is_raised_and_the_ocean_is_left_alone():
    elev = island_with_hollow()
    out, salt = dry_landlocked_basins(elev)
    assert (out[30:50, 30:50] == PAN_LEVEL_M).all()
    assert salt[35:45, 35:45].all() and not salt[:28].any()
    assert (out[:10] == -20.0).all() and (out[10:28, 10:70] == 30.0).all()
    # Nothing below sea level is left that the sea cannot reach from the edge
    again, _ = dry_landlocked_basins(out)
    assert again is out


def test_hollow_open_to_the_edge_stays_sea():
    elev = island_with_hollow()
    elev[40, :35] = -3.0  # a channel from the hollow to the coast
    out, salt = dry_landlocked_basins(elev)
    assert out is elev and not salt.any()


def test_a_lake_in_the_hollow_keeps_its_bed():
    elev = island_with_hollow()
    lake = np.zeros(elev.shape, dtype=np.uint8)
    lake[30:50, 30:50] = 255
    out, salt = dry_landlocked_basins(elev, lake)
    assert (out[30:50, 30:50] == -6.0).all() and not salt.any()


def test_stray_pits_are_raised_but_not_marked_as_salt():
    elev = island_with_hollow()
    elev[30:50, 30:50] = 30.0
    elev[20, 20] = -1.0
    out, salt = dry_landlocked_basins(elev)
    assert out[20, 20] == PAN_LEVEL_M and not salt.any()


def test_salt_takes_the_limestone_channel_above_the_regional_cap():
    salt = np.zeros((64, 64), dtype=bool)
    salt[:32] = True
    rock = pack_rock(None, None, 16, salt)
    assert tuple(rock[2, 2]) == (0, 0, 255, 0)
    assert tuple(rock[13, 13]) == (0, 0, 0, 0)
