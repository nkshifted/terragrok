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
