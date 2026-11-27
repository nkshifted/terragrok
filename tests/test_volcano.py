"""Volcano cones and rock maps: pointwise, so shared cones match across tile seams."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from volcano import pack_rock, stamp_volcanoes

TILE = 256
MARGIN = 16
SPAN = 4800.0


def cone(**over):
    return {"id": 7, "kind": "strato", "x": 0.0, "z": 0.0, "radius": 0.12, "aspect": 0.42, "crater": 0.1, "craterDepth": 0.14, "baseDem": 40.0, **over}


def stamp(elev, volcanoes, margin=MARGIN):
    return stamp_volcanoes(elev, volcanoes, tile_px=TILE, margin_px=margin, world_span_m=SPAN, base_to_viewer=lambda d: d)


def flat(level=40.0, margin=MARGIN):
    return np.full((TILE + 2 * margin, TILE + 2 * margin), level, dtype=np.float32)


def test_cone_rises_to_its_height_with_a_crater_and_leaves_the_rest_alone():
    out, lava, placed = stamp(flat(), [cone()])
    c = MARGIN + TILE // 2
    height = 0.42 * 0.12 * SPAN
    assert 40.0 + 0.6 * height < out.max() <= 40.0 + height  # the rim sits below the cone's ideal tip
    assert out[c, c] < out.max() - 0.05 * height  # crater floor below the rim
    assert out[4, 4] == 40.0 and lava[4, 4] == 0.0
    assert lava[c, c] == 1.0
    assert placed and placed[0]["id"] == 7


def test_stamped_terrain_is_finite_for_every_kind():
    for kind in ("strato", "shield", "cinder"):
        out, lava, _ = stamp(flat(), [cone(kind=kind, id=k) for k in range(1, 6)])
        assert np.isfinite(out).all() and np.isfinite(lava).all(), kind


def test_cone_never_digs_into_higher_ground():
    ground = flat(400.0)
    out, _, _ = stamp(ground, [cone(craterDepth=0.0)])
    assert (out >= ground - 1e-3).all()


def test_two_tiles_build_the_same_cone_across_their_seam():
    # A cone on the shared edge, seen from the west tile (x = +0.5) and the east tile (x = -0.5)
    rng = np.random.default_rng(3)
    world = rng.random((TILE + 2 * MARGIN, 2 * TILE + 2 * MARGIN)).astype(np.float32) * 30.0 + 20.0
    west, _, _ = stamp(world[:, : TILE + 2 * MARGIN].copy(), [cone(x=0.5)])
    east, _, _ = stamp(world[:, TILE:].copy(), [cone(x=-0.5)])
    # Columns the two padded windows share
    np.testing.assert_allclose(west[:, TILE:], east[:, : 2 * MARGIN], atol=1e-3)
    assert west[:, TILE:].max() > 100.0


def test_only_cones_centred_in_the_tile_are_reported():
    _, _, placed = stamp(flat(), [cone(x=0.52), cone(id=9, x=-0.2)])
    assert [p["id"] for p in placed] == [9]


def test_pack_rock_resamples_and_lava_overrides():
    ref = np.zeros((32, 32, 4), dtype=np.uint8)
    ref[..., 0] = 204  # granite everywhere
    lava = np.zeros((64, 64), dtype=np.float32)
    lava[:32] = 1.0
    rock = pack_rock(ref, lava, 16)
    assert rock.shape == (16, 16, 4) and rock.dtype == np.uint8
    assert tuple(rock[2, 2]) == (0, 255, 0, 0)
    assert tuple(rock[13, 13]) == (204, 0, 0, 0)
    assert pack_rock(None, None, 8).sum() == 0


# ---------- Craters and water ----------

from volcano import crater_plugs, fill_hot_springs  # noqa: E402


def plugs(volcanoes, margin=MARGIN):
    n = TILE + 2 * margin
    return crater_plugs(volcanoes, (n, n), tile_px=TILE, margin_px=margin, world_span_m=SPAN, base_to_viewer=lambda d: d)


def test_only_erupting_and_hot_spring_craters_are_plugged():
    assert not plugs([cone()]).any()  # dormant: the solver may give it a lake
    for flag in ("active", "hotSpring"):
        v = cone(**{flag: True})
        out, _, _ = stamp(flat(), [v])
        plug = plugs([v])
        c = MARGIN + TILE // 2
        assert plug[c, c] > 1.0 and plug[4, 4] == 0.0
        # Filled, the crater is no longer a hollow: its floor stands level with its rim
        r = int(0.1 * 0.12 * TILE)
        filled = (out + plug)[c - r : c + r + 1, c - r : c + r + 1]
        assert filled[r, r] > filled.max() - 0.05 * plug[c, c]


def test_hot_spring_fills_its_crater_below_the_rim():
    v = cone(hotSpring=True, radius=0.4, aspect=0.2)
    out, _, _ = stamp(flat(margin=0), [v], margin=0)
    lak = np.zeros(out.shape, dtype=np.uint8)
    lakes = [{"id": 3, "level": 1.0, "cells": []}]
    before = out.copy()
    sinter, levels = fill_hot_springs(out, lak, lakes, [v], world_span_m=SPAN, lake_grid=TILE // 2)
    assert np.array_equal(out, before)  # the ground is never touched
    c = TILE // 2
    pool = lakes[-1]
    assert pool["hotSpring"] and pool["id"] == 4 and levels == {7: pool["level"]}
    assert out[c, c] < pool["level"] < out.max()
    assert lak[c, c] == 255 and lak[4, 4] == 0
    assert all(0 <= r < TILE // 2 and 0 <= col < TILE // 2 and d > 0 for r, col, d in pool["cells"])
    assert [c // 2, c // 2] in [cell[:2] for cell in pool["cells"]]
    # A crust around the pool, shown in the granite channel above the regional cap
    assert sinter[c, c] == 1.0 and sinter[4, 4] == 0.0
    rock = pack_rock(None, None, 64, None, sinter)
    assert rock[32, 32, 0] > 214 and rock[2, 2, 0] == 0


def test_no_pool_without_a_hot_spring():
    out, _, _ = stamp(flat(margin=0), [cone(active=True)], margin=0)
    lak = np.zeros(out.shape, dtype=np.uint8)
    lakes: list[dict] = []
    sinter, levels = fill_hot_springs(out, lak, lakes, [cone(active=True)], world_span_m=SPAN, lake_grid=TILE)
    assert not lakes and not levels and not lak.any() and not sinter.any()
