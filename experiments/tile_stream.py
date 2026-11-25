"""In-memory tile stream sessions for expanding-ring ghost (Phase 1)."""

from __future__ import annotations

import threading
from typing import Any

from tile_ghost import list_tiles_for_home
from tile_grid import neighbor_centers, tile_id


class TileStreamRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def start_stream(
        self,
        *,
        home_world_id: str,
        seed: int,
        lat_deg: float,
        lon_deg: float,
        span_deg: float = 46.0,
        max_ring: int = 1,
    ) -> dict[str, Any]:
        neighbors = neighbor_centers(lat_deg, lon_deg, span_deg, max_ring=max_ring)
        planned = []
        for n in neighbors:
            tid = tile_id(seed, n["latDeg"], n["lonDeg"], span_deg)
            planned.append({**n, "tileId": tid})

        existing = {
            e["id"]: e
            for e in list_tiles_for_home(home_world_id)
            if e.get("id") and e.get("id") != home_world_id
        }
        done = []
        pending = []
        for p in planned:
            if p["tileId"] in existing and (existing[p["tileId"]].get("quality") or existing[p["tileId"]].get("tileOrigin", {}).get("quality")) == "ghost":
                done.append({**p, "quality": "ghost"})
            elif p["tileId"] in existing:
                # Real synth world reused as neighbor — treat as done at its quality
                q = existing[p["tileId"]].get("quality") or existing[p["tileId"]].get("resolution") or "1k"
                done.append({**p, "quality": str(q).lower()})
            else:
                pending.append(p)

        session = {
            "homeWorldId": home_world_id,
            "seed": seed,
            "latDeg": lat_deg,
            "lonDeg": lon_deg,
            "spanDeg": span_deg,
            "maxRing": max_ring,
            "planned": planned,
            "pendingGhost": [p["tileId"] for p in pending],
            "done": done,
            "active": None,
        }
        with self._lock:
            self._sessions[home_world_id] = session
        return {
            "ok": True,
            "homeWorldId": home_world_id,
            "neighbors": planned,
            "pendingGhost": session["pendingGhost"],
            "done": done,
        }

    def mark_ghost_done(self, home_world_id: str, tile_id_str: str, meta: dict) -> None:
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if not sess:
                return
            sess["pendingGhost"] = [t for t in sess["pendingGhost"] if t != tile_id_str]
            sess["done"] = [d for d in sess["done"] if d.get("tileId") != tile_id_str]
            n = next((p for p in sess["planned"] if p["tileId"] == tile_id_str), None)
            entry = {
                "tileId": tile_id_str,
                "quality": "ghost",
                "qi": (n or {}).get("qi", (meta.get("tileOrigin") or {}).get("qi")),
                "qj": (n or {}).get("qj", (meta.get("tileOrigin") or {}).get("qj")),
                "ring": (n or {}).get("ring", (meta.get("tileOrigin") or {}).get("ring")),
                "latDeg": (meta.get("planetOrigin") or {}).get("latDeg"),
                "lonDeg": (meta.get("planetOrigin") or {}).get("lonDeg"),
            }
            sess["done"].append(entry)

    def mark_synth_done(self, home_world_id: str, tile_id_str: str, info: dict) -> None:
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if not sess:
                # Still record via a lightweight session so status/fromCatalog works mid-flight
                sess = {
                    "homeWorldId": home_world_id,
                    "planned": [],
                    "pendingGhost": [],
                    "done": [],
                    "active": None,
                }
                self._sessions[home_world_id] = sess
            sess["done"] = [d for d in sess["done"] if d.get("tileId") != tile_id_str]
            sess["done"].append(
                {
                    "tileId": tile_id_str,
                    "quality": str(info.get("quality") or "1k").lower(),
                    "qi": info.get("qi"),
                    "qj": info.get("qj"),
                    "ring": info.get("ring"),
                    "latDeg": info.get("latDeg"),
                    "lonDeg": info.get("lonDeg"),
                    "walkable": True,
                }
            )
            sess["active"] = None
            print(
                f"[TileStream] {tile_id_str} → {info.get('quality')} "
                f"(qi={info.get('qi')}, qj={info.get('qj')}) for {home_world_id}",
                flush=True,
            )

    def status(self, home_world_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if sess:
                return {
                    "ok": True,
                    "homeWorldId": home_world_id,
                    "pending": list(sess["pendingGhost"]),
                    "active": sess.get("active"),
                    "done": list(sess["done"]),
                    "plannedCount": len(sess["planned"]),
                }
        # Cold status from catalog only
        tiles = [e for e in list_tiles_for_home(home_world_id) if e.get("id") != home_world_id]
        done = []
        for e in tiles:
            to = e.get("tileOrigin") or {}
            done.append(
                {
                    "tileId": e["id"],
                    "quality": e.get("quality") or to.get("quality") or "ghost",
                    "qi": to.get("qi"),
                    "qj": to.get("qj"),
                    "ring": to.get("ring"),
                    "latDeg": (e.get("planetOrigin") or {}).get("latDeg"),
                    "lonDeg": (e.get("planetOrigin") or {}).get("lonDeg"),
                }
            )
        return {
            "ok": True,
            "homeWorldId": home_world_id,
            "pending": [],
            "active": None,
            "done": done,
            "plannedCount": len(done),
            "fromCatalog": True,
        }


TILE_STREAMS = TileStreamRegistry()
