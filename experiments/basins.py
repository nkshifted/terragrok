"""
Landlocked basins below sea level.

The explorer draws the sea wherever the ground is below zero. A hollow that is below sea level
but cut off from the ocean, and that the hydrology solver did not turn into a lake (too dry,
too small), would therefore show a patch of sea with surf in the middle of the land. Such a
basin is a dry pan in nature (Death Valley, the Dead Sea shore): its floor is raised to just
above sea level here and reported, so the rock map can mark it as salt flat.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening, label

PAN_LEVEL_M = 0.7
# A lake mask above this (of 255) is water the explorer draws itself; it hides the sea there
LAKE_MASK_MIN = 13


def dry_landlocked_basins(elev: np.ndarray, lake_u8: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (elevation with dry basins raised to PAN_LEVEL_M, bool mask of the salt flats).

    "Ocean" is below-sea ground connected to the tile's edge, as in solve_physical_hydrology,
    so a basin that crosses a tile edge counts as ocean in both tiles and is left alone: the
    two sides of a seam never disagree. Stray below-sea pixels are raised too, but only pans
    wide enough to read as a surface are marked as salt.
    """
    below = elev <= 0.0
    lbl, _ = label(below, structure=np.ones((3, 3), dtype=int))
    border = set(np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])))
    border.discard(0)
    dry = below & ~np.isin(lbl, list(border))
    if lake_u8 is not None:
        dry &= lake_u8 < LAKE_MASK_MIN
    if not dry.any():
        return elev, dry
    out = elev.astype(np.float32, copy=True)
    out[dry] = PAN_LEVEL_M
    return out, binary_opening(dry, iterations=2)
