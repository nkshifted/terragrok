"""Chart tile-grid helpers: stable ids and player-centred windows (no numpy / torch).

A home patch defines a flat "chart" (see viewer/src/stage1Conditioning.js). Chart tile
(qi, qj) lies qi tiles east and qj tiles SOUTH of home, matching viewer +X / +Z and
heightfield columns / rows. Home is (0, 0).
"""

from __future__ import annotations

import math


def chart_tile_id(home_world_id: str, qi: int, qj: int) -> str:
    """Stable world id for chart tile (qi, qj) of a home world (gitignored `tile_*`)."""
    ew = "e" if qi >= 0 else "w"
    sn = "s" if qj >= 0 else "n"
    return f"tile_{home_world_id}_{ew}{abs(int(qi))}_{sn}{abs(int(qj))}"


def tile_of(x_tiles: float, z_tiles: float) -> tuple[int, int]:
    """Tile index containing a position given in tile units (home centre = 0, 0)."""
    return int(math.floor(x_tiles + 0.5)), int(math.floor(z_tiles + 0.5))


def tile_distance(x_tiles: float, z_tiles: float, qi: int, qj: int) -> float:
    """Euclidean distance (tile units) from a position to a tile's centre."""
    return math.hypot(qi - x_tiles, qj - z_tiles)


def window_tiles(x_tiles: float, z_tiles: float, ring: int = 1) -> list[tuple[int, int]]:
    """
    Tiles within `ring` (Chebyshev) of the tile under the player, nearest centre first.
    Includes the player's own tile; ties break deterministically on (qj, qi).
    """
    cqi, cqj = tile_of(x_tiles, z_tiles)
    out = [
        (qi, qj)
        for qj in range(cqj - ring, cqj + ring + 1)
        for qi in range(cqi - ring, cqi + ring + 1)
    ]
    out.sort(key=lambda t: (tile_distance(x_tiles, z_tiles, t[0], t[1]), t[1], t[0]))
    return out
