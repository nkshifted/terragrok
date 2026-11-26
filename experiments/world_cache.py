"""Synthesized-world disk cache helpers (no torch / MPS dependency)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER_WORLDS_DIR = REPO_ROOT / "viewer" / "public" / "worlds"
SYNTH_CATALOG_PATH = VIEWER_WORLDS_DIR / "synthesized_catalog.json"
# Serializes read-modify-write of the catalog across the server's worker / HTTP / ghost threads
CATALOG_LOCK = threading.RLock()
WORLD_FILE_SUFFIXES = (
    ".json",
    ".height.f32",
    ".masks.u8",
    ".climate.u8",
    ".rivers.json",
    ".lakes.json",
    ".ghost.height.f32",
)


def load_synth_catalog(catalog_path: Path | None = None) -> list[dict]:
    path = catalog_path or SYNTH_CATALOG_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_synth_catalog(entries: list[dict], catalog_path: Path | None = None, worlds_dir: Path | None = None) -> None:
    path = catalog_path or SYNTH_CATALOG_PATH
    root = worlds_dir or VIEWER_WORLDS_DIR
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))


def world_disk_bytes(world_id: str, worlds_dir: Path | None = None) -> int:
    root = worlds_dir or VIEWER_WORLDS_DIR
    total = 0
    for suf in WORLD_FILE_SUFFIXES:
        p = root / f"{world_id}{suf}"
        if p.exists():
            total += p.stat().st_size
    return total


def enrich_world_entry(entry: dict, worlds_dir: Path | None = None) -> dict:
    wid = entry.get("id")
    out = dict(entry)
    if wid:
        nbytes = world_disk_bytes(str(wid), worlds_dir=worlds_dir)
        out["diskBytes"] = nbytes
        out["diskMB"] = round(nbytes / (1024 * 1024), 2)
    return out


def list_cached_worlds(
    seed: int | None = None,
    archetype: str | None = None,
    *,
    catalog_path: Path | None = None,
    worlds_dir: Path | None = None,
) -> dict:
    catalog = [e for e in load_synth_catalog(catalog_path) if e.get("id")]
    # Streamed chart tiles are part of their home world: fold their disk use into it
    # instead of listing dozens of tiles and ghosts as worlds of their own.
    tiles_by_home: dict[str, list[dict]] = {}
    for e in catalog:
        home = (e.get("tileOrigin") or {}).get("homeWorldId")
        if home:
            tiles_by_home.setdefault(home, []).append(e)
    entries = []
    for e in catalog:
        if e.get("tileOrigin"):
            continue
        out = enrich_world_entry(e, worlds_dir=worlds_dir)
        tiles = tiles_by_home.get(e["id"], [])
        if tiles:
            extra = sum(world_disk_bytes(str(t["id"]), worlds_dir=worlds_dir) for t in tiles)
            out["tileCount"] = sum(1 for t in tiles if t.get("quality") != "ghost")
            out["diskBytes"] += extra
            out["diskMB"] = round(out["diskBytes"] / (1024 * 1024), 2)
        entries.append(out)
    if seed is not None:
        entries = [
            e
            for e in entries
            if (e.get("planetOrigin") or {}).get("seed") == seed
            or str((e.get("planetOrigin") or {}).get("seed")) == str(seed)
        ]
    if archetype:
        entries = [
            e
            for e in entries
            if (e.get("planetOrigin") or {}).get("archetype") == archetype
        ]
    total_bytes = sum(int(e.get("diskBytes") or 0) for e in entries)
    return {
        "ok": True,
        "worlds": entries,
        "count": len(entries),
        "totalBytes": total_bytes,
        "totalMB": round(total_bytes / (1024 * 1024), 2),
    }


def delete_cached_world(
    world_id: str,
    cascade: bool = True,
    *,
    catalog_path: Path | None = None,
    worlds_dir: Path | None = None,
) -> dict:
    """Delete a synthesized world and optionally its drill-down children and streamed tiles."""
    root = worlds_dir or VIEWER_WORLDS_DIR
    cat_path = catalog_path or SYNTH_CATALOG_PATH
    catalog = load_synth_catalog(cat_path)
    by_id = {e.get("id"): e for e in catalog if e.get("id")}
    if world_id not in by_id and not (root / f"{world_id}.json").exists():
        raise FileNotFoundError(f"World not found: {world_id}")

    to_delete = {world_id}
    if cascade:
        changed = True
        while changed:
            changed = False
            for e in catalog:
                wid = e.get("id")
                parent = (e.get("drillOrigin") or {}).get("parentWorldId")
                home = (e.get("tileOrigin") or {}).get("homeWorldId")
                if wid and (parent in to_delete or home in to_delete) and wid not in to_delete:
                    to_delete.add(wid)
                    changed = True

    removed_files = []
    for wid in sorted(to_delete):
        for suf in WORLD_FILE_SUFFIXES:
            p = root / f"{wid}{suf}"
            if p.exists():
                p.unlink()
                removed_files.append(p.name)

    new_catalog = [e for e in catalog if e.get("id") not in to_delete]
    save_synth_catalog(new_catalog, catalog_path=cat_path, worlds_dir=root)
    return {
        "ok": True,
        "deletedIds": sorted(to_delete),
        "removedFiles": removed_files,
        "remaining": len(new_catalog),
    }


def patch_cached_world(
    world_id: str,
    patch: dict,
    *,
    catalog_path: Path | None = None,
    worlds_dir: Path | None = None,
) -> dict:
    root = worlds_dir or VIEWER_WORLDS_DIR
    cat_path = catalog_path or SYNTH_CATALOG_PATH
    catalog = load_synth_catalog(cat_path)
    entry = next((e for e in catalog if e.get("id") == world_id), None)
    meta_path = root / f"{world_id}.json"
    if entry is None and not meta_path.exists():
        raise FileNotFoundError(f"World not found: {world_id}")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(entry or {})
    if "title" in patch and patch["title"] is not None:
        title = str(patch["title"]).strip()
        if not title:
            raise ValueError("title cannot be empty")
        if len(title) > 120:
            raise ValueError("title too long (max 120 chars)")
        meta["title"] = title
    if "bookmarked" in patch:
        meta["bookmarked"] = bool(patch["bookmarked"])

    meta_path.write_text(json.dumps(meta, indent=2))
    if entry is None:
        catalog.append(meta)
    else:
        for i, e in enumerate(catalog):
            if e.get("id") == world_id:
                catalog[i] = {**e, **meta}
                break
    save_synth_catalog(catalog, catalog_path=cat_path, worlds_dir=root)
    return {"ok": True, "world": enrich_world_entry(meta, worlds_dir=root)}


def prune_unbookmarked_worlds(
    seed: int | None = None,
    archetype: str | None = None,
    *,
    catalog_path: Path | None = None,
    worlds_dir: Path | None = None,
) -> dict:
    """Delete all non-bookmarked synthesized worlds (optionally filtered by planet)."""
    listed = list_cached_worlds(
        seed=seed,
        archetype=archetype,
        catalog_path=catalog_path,
        worlds_dir=worlds_dir,
    )["worlds"]
    victims = [e["id"] for e in listed if e.get("id") and not e.get("bookmarked")]
    deleted = []
    for wid in victims:
        try:
            delete_cached_world(
                wid,
                cascade=True,
                catalog_path=catalog_path,
                worlds_dir=worlds_dir,
            )
            deleted.append(wid)
        except FileNotFoundError:
            continue
    remaining = list_cached_worlds(
        seed=seed,
        archetype=archetype,
        catalog_path=catalog_path,
        worlds_dir=worlds_dir,
    )
    return {
        "ok": True,
        "deletedIds": deleted,
        "deletedCount": len(deleted),
        "remaining": remaining["count"],
        "totalMB": remaining["totalMB"],
    }
