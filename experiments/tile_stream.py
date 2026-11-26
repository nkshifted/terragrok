"""Tile stream planner: which chart tiles to ghost / synthesize next (no numpy / torch).

The explorer posts the player's position every couple of seconds. Each post keeps the
session alive and re-centres the wanted window, so synthesis follows the player instead
of stopping at the first ring around home. The Stage 2 worker asks `next_synth_job()`
whenever it is idle; nothing is pre-queued, so there is nothing to cancel or re-sort.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from tile_grid import chart_tile_id, tile_distance, tile_of, window_tiles

# Synthesize real terrain this far around the player's tile (1 -> the 3x3 the viewer draws)
SYNTH_RING = 1
# Bake cheap Stage-1 ghosts one ring further so newly entered windows are never empty
GHOST_RING = 2
# A session with no heartbeat for this long stops consuming MPS time. Browsers throttle timers
# in background tabs to about one a minute, so this must comfortably exceed that.
SESSION_TTL_S = 150.0
MAX_SYNTH_FAILURES = 2

_QUALITY_RANK = {"ghost": 0, "1k": 1, "2k": 2, "4k": 3}


def quality_rank(q: str | None) -> int:
    return _QUALITY_RANK.get(str(q or "").lower(), -1)


def normalize_quality(q: str | None) -> str:
    """Map '2K' / '2k' / 2048-ish labels to 'ghost' | '1k' | '2k' | '4k'."""
    s = str(q or "1k").lower()
    if s == "ghost":
        return "ghost"
    if "4" in s:
        return "4k"
    if "2" in s:
        return "2k"
    return "1k"


def tile_resolution_for_home(home_resolution: str | None) -> str:
    """
    Neighbours are synthesized at the home's resolution (capped at 2K): 1K and 2K sample the
    planet at different model scales, so only same-resolution tiles line up feature for feature.
    """
    return "1k" if normalize_quality(home_resolution or "2k") == "1k" else "2k"


class TileStreamRegistry:
    def __init__(self, clock=time.monotonic) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._sessions: dict[str, dict[str, Any]] = {}

    # ---- sessions -------------------------------------------------------------------
    def has_session(self, home_world_id: str) -> bool:
        with self._lock:
            return home_world_id in self._sessions

    def open_session(self, home_world_id: str, chart: dict, existing_tiles: list[dict] | None = None) -> None:
        """Create (or refresh the chart of) a session. `existing_tiles`: catalog entries."""
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if sess is None:
                sess = {
                    "homeWorldId": home_world_id,
                    "chart": dict(chart),
                    "x": 0.0,
                    "z": 0.0,
                    "lastSeen": self._clock(),
                    "tiles": {},
                    "active": None,
                    "ghostBusy": False,
                }
                self._sessions[home_world_id] = sess
            else:
                sess["chart"] = dict(chart)
            for e in existing_tiles or []:
                to = e.get("tileOrigin") or {}
                if to.get("scheme") != "chart" or to.get("qi") is None or to.get("qj") is None:
                    continue
                key = (int(to["qi"]), int(to["qj"]))
                q = normalize_quality(e.get("quality") or to.get("quality") or "ghost")
                cur = sess["tiles"].get(key)
                if cur is None or quality_rank(q) > quality_rank(cur["quality"]):
                    sess["tiles"][key] = {"tileId": e["id"], "quality": q, "rev": 1, "failures": 0}

    def chart(self, home_world_id: str) -> dict | None:
        with self._lock:
            sess = self._sessions.get(home_world_id)
            return dict(sess["chart"]) if sess else None

    def touch(self, home_world_id: str, x_tiles: float, z_tiles: float) -> None:
        """Heartbeat: player position in tile units relative to the home centre."""
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if not sess:
                return
            sess["x"] = float(x_tiles)
            sess["z"] = float(z_tiles)
            sess["lastSeen"] = self._clock()

    def _live_session(self) -> dict | None:
        now = self._clock()
        live = [s for s in self._sessions.values() if now - s["lastSeen"] <= SESSION_TTL_S]
        return max(live, key=lambda s: s["lastSeen"]) if live else None

    # ---- planning -------------------------------------------------------------------
    def next_synth_job(self) -> dict | None:
        """Nearest not-yet-synthesized tile around the most recently active player."""
        with self._lock:
            sess = self._live_session()
            if not sess:
                return None
            target = sess["chart"].get("tileResolution", "2k")
            for qi, qj in window_tiles(sess["x"], sess["z"], SYNTH_RING):
                if qi == 0 and qj == 0:
                    continue
                t = sess["tiles"].get((qi, qj))
                if t and quality_rank(t["quality"]) >= 1:
                    continue
                if t and t.get("failures", 0) >= MAX_SYNTH_FAILURES:
                    continue
                sess["active"] = {"qi": qi, "qj": qj, "startedAt": time.time()}
                return {
                    "homeWorldId": sess["homeWorldId"],
                    "qi": qi,
                    "qj": qj,
                    "tileId": chart_tile_id(sess["homeWorldId"], qi, qj),
                    "resolution": target,
                    "chart": dict(sess["chart"]),
                }
            sess["active"] = None
            return None

    def claim_ghost_batch(self, home_world_id: str) -> list[tuple[int, int]]:
        """Tiles near the player with no data at all. Single-flight per session."""
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if not sess or sess["ghostBusy"]:
                return []
            missing = [
                (qi, qj)
                for qi, qj in window_tiles(sess["x"], sess["z"], GHOST_RING)
                if (qi, qj) != (0, 0) and (qi, qj) not in sess["tiles"]
            ]
            if missing:
                sess["ghostBusy"] = True
            return missing

    def release_ghost_batch(self, home_world_id: str) -> None:
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if sess:
                sess["ghostBusy"] = False

    # ---- results --------------------------------------------------------------------
    def mark_tile(self, home_world_id: str, qi: int, qj: int, tile_id: str, quality: str) -> None:
        q = normalize_quality(quality)
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if not sess:
                return
            key = (int(qi), int(qj))
            cur = sess["tiles"].get(key)
            if cur and quality_rank(cur["quality"]) > quality_rank(q):
                return  # never let a late ghost replace real terrain
            sess["tiles"][key] = {
                "tileId": tile_id,
                "quality": q,
                "rev": (cur["rev"] + 1) if cur else 1,
                "failures": 0,
            }
            if sess["active"] and (sess["active"]["qi"], sess["active"]["qj"]) == key and q != "ghost":
                sess["active"] = None

    def mark_failed(self, home_world_id: str, qi: int, qj: int) -> None:
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if not sess:
                return
            key = (int(qi), int(qj))
            cur = sess["tiles"].setdefault(
                key, {"tileId": chart_tile_id(home_world_id, qi, qj), "quality": None, "rev": 0, "failures": 0}
            )
            cur["failures"] = cur.get("failures", 0) + 1
            sess["active"] = None

    def forget(self, home_world_id: str) -> None:
        with self._lock:
            self._sessions.pop(home_world_id, None)

    def status(self, home_world_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(home_world_id)
            if not sess:
                return {"ok": False, "homeWorldId": home_world_id, "error": "no tile stream for this world"}
            pqi, pqj = tile_of(sess["x"], sess["z"])
            tiles = [
                {
                    "tileId": t["tileId"],
                    "qi": qi,
                    "qj": qj,
                    "quality": t["quality"],
                    "walkable": quality_rank(t["quality"]) >= 1,
                    "rev": t["rev"],
                    "dist": round(tile_distance(sess["x"], sess["z"], qi, qj), 3),
                }
                for (qi, qj), t in sess["tiles"].items()
                if t["quality"]
            ]
            return {
                "ok": True,
                "homeWorldId": home_world_id,
                "playerQi": pqi,
                "playerQj": pqj,
                "tileResolution": sess["chart"].get("tileResolution"),
                "worldSpan": sess["chart"].get("worldSpan"),
                "active": dict(sess["active"]) if sess["active"] else None,
                "tiles": tiles,
            }


TILE_STREAMS = TileStreamRegistry()
