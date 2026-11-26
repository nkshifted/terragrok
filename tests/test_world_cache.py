"""Unit tests for synthesized-world disk cache helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from world_cache import (  # noqa: E402
    delete_cached_world,
    list_cached_worlds,
    patch_cached_world,
    prune_unbookmarked_worlds,
    save_synth_catalog,
)


def _touch_world(root: Path, world_id: str, meta: dict) -> None:
    meta = {**meta, "id": world_id}
    (root / f"{world_id}.json").write_text(json.dumps(meta, indent=2))
    (root / f"{world_id}.height.f32").write_bytes(b"\x00" * 16)
    (root / f"{world_id}.masks.u8").write_bytes(b"\x00" * 16)


def test_delete_cascades_drill_children(tmp_path: Path):
    cat = tmp_path / "synthesized_catalog.json"
    parent = {
        "id": "parent_a",
        "title": "Parent",
        "planetOrigin": {"seed": 1, "archetype": "earth"},
    }
    child = {
        "id": "child_b",
        "title": "Child",
        "planetOrigin": {"seed": 1, "archetype": "earth"},
        "drillOrigin": {"parentWorldId": "parent_a", "depth": 1},
    }
    _touch_world(tmp_path, "parent_a", parent)
    _touch_world(tmp_path, "child_b", child)
    save_synth_catalog([parent, child], catalog_path=cat, worlds_dir=tmp_path)

    result = delete_cached_world("parent_a", cascade=True, catalog_path=cat, worlds_dir=tmp_path)
    assert set(result["deletedIds"]) == {"child_b", "parent_a"}
    assert not (tmp_path / "parent_a.json").exists()
    assert not (tmp_path / "child_b.json").exists()
    listed = list_cached_worlds(catalog_path=cat, worlds_dir=tmp_path)
    assert listed["count"] == 0


def test_patch_rename_and_bookmark(tmp_path: Path):
    cat = tmp_path / "synthesized_catalog.json"
    meta = {
        "id": "w1",
        "title": "Old Title",
        "planetOrigin": {"seed": 7, "archetype": "earth"},
    }
    _touch_world(tmp_path, "w1", meta)
    save_synth_catalog([meta], catalog_path=cat, worlds_dir=tmp_path)

    out = patch_cached_world(
        "w1",
        {"title": "Favorite Fjord", "bookmarked": True},
        catalog_path=cat,
        worlds_dir=tmp_path,
    )
    assert out["world"]["title"] == "Favorite Fjord"
    assert out["world"]["bookmarked"] is True
    disk = json.loads((tmp_path / "w1.json").read_text())
    assert disk["title"] == "Favorite Fjord"
    assert disk["bookmarked"] is True


def test_prune_keeps_bookmarked(tmp_path: Path):
    cat = tmp_path / "synthesized_catalog.json"
    keep = {
        "id": "keep_me",
        "title": "Keep",
        "bookmarked": True,
        "planetOrigin": {"seed": 9, "archetype": "earth"},
    }
    drop = {
        "id": "drop_me",
        "title": "Drop",
        "bookmarked": False,
        "planetOrigin": {"seed": 9, "archetype": "earth"},
    }
    _touch_world(tmp_path, "keep_me", keep)
    _touch_world(tmp_path, "drop_me", drop)
    save_synth_catalog([keep, drop], catalog_path=cat, worlds_dir=tmp_path)

    result = prune_unbookmarked_worlds(
        seed=9,
        archetype="earth",
        catalog_path=cat,
        worlds_dir=tmp_path,
    )
    assert "drop_me" in result["deletedIds"]
    assert "keep_me" not in result["deletedIds"]
    assert (tmp_path / "keep_me.json").exists()
    assert not (tmp_path / "drop_me.json").exists()


def test_delete_cascades_streamed_tiles_and_ghost_rasters(tmp_path: Path):
    cat = tmp_path / "synthesized_catalog.json"
    home = {"id": "planet_home_2k", "title": "Home"}
    tile = {"id": "tile_planet_home_2k_e1_s0", "tileOrigin": {"homeWorldId": "planet_home_2k", "qi": 1, "qj": 0}}
    ghost = {"id": "tile_planet_home_2k_e2_s0", "quality": "ghost", "tileOrigin": {"homeWorldId": "planet_home_2k"}}
    other = {"id": "planet_other_2k", "title": "Other"}
    for m in (home, tile, other):
        _touch_world(tmp_path, m["id"], m)
    (tmp_path / f"{ghost['id']}.json").write_text(json.dumps(ghost))
    (tmp_path / f"{ghost['id']}.ghost.height.f32").write_bytes(b"\x00" * 16)
    save_synth_catalog([home, tile, ghost, other], catalog_path=cat, worlds_dir=tmp_path)

    result = delete_cached_world("planet_home_2k", catalog_path=cat, worlds_dir=tmp_path)

    assert set(result["deletedIds"]) == {home["id"], tile["id"], ghost["id"]}
    assert sorted(p.name for p in tmp_path.glob("tile_*")) == []
    assert (tmp_path / "planet_other_2k.json").exists()


def test_listing_folds_streamed_tiles_into_their_home(tmp_path: Path):
    cat = tmp_path / "synthesized_catalog.json"
    home = {"id": "planet_home_2k", "title": "Home", "planetOrigin": {"seed": 7}}
    tile = {
        "id": "tile_planet_home_2k_e1_s0",
        "quality": "2k",
        "planetOrigin": {"seed": 7},
        "tileOrigin": {"homeWorldId": "planet_home_2k", "qi": 1, "qj": 0},
    }
    ghost = {**tile, "id": "tile_planet_home_2k_e2_s0", "quality": "ghost"}
    for m in (home, tile):
        _touch_world(tmp_path, m["id"], m)
    save_synth_catalog([home, tile, ghost], catalog_path=cat, worlds_dir=tmp_path)

    listed = list_cached_worlds(seed=7, catalog_path=cat, worlds_dir=tmp_path)

    assert [w["id"] for w in listed["worlds"]] == ["planet_home_2k"]
    assert listed["worlds"][0]["tileCount"] == 1
    one_world = sum((tmp_path / f"planet_home_2k{s}").stat().st_size for s in (".json", ".height.f32", ".masks.u8"))
    assert listed["worlds"][0]["diskBytes"] > one_world
