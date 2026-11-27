"""
Volcano cones and bedrock maps for synthesized tiles.

The diffusion model knows nothing about volcanoes: Stage 1 says where they stand
(`chartTileVolcanoes` in viewer/src/stage1Conditioning.js) and this module builds them into the
synthesized terrain. Everything here is a function of a pixel's own position only, so two
tiles that share a cone build the same cone on both sides of their seam. Do not add anything
that looks at a neighbourhood of the tile (a blur, a local mean).
"""

from __future__ import annotations

import base64

import numpy as np
from scipy.ndimage import zoom


def decode_rock(b64_str: str, size: int) -> np.ndarray:
    """A base64 (size, size, 4) u8 rock raster as sent by the Stage 1 sampler."""
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8).copy().reshape(size, size, 4)

ROCK_VERSION = 1
# Channel order of a rock raster; the remainder to 255 is sandstone / sediment
ROCK_CHANNELS = ("granite", "volcanic", "limestone", "metamorphic")
# Cones may not rise above this (viewer metres): the explorer's fog and camera assume it
MAX_SUMMIT_M = 520.0


def _hash01(n: int, salt: int) -> float:
    h = (int(n) * 2654435761 + salt * 40503) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return h / 4294967296.0


def _profile(kind: str, s: np.ndarray) -> np.ndarray:
    """Height of a cone, 1 at the centre and 0 at s = 1 (s = distance / radius)."""
    t = np.clip(1.0 - s, 0.0, 1.0)
    if kind == "shield":
        return np.clip(1.0 - s * s, 0.0, 1.0) ** 1.5  # broad dome
    if kind == "cinder":
        return t  # straight scree slope
    return t**1.7  # stratovolcano: steepening towards the summit


def stamp_volcanoes(
    elev: np.ndarray,
    volcanoes: list[dict],
    *,
    tile_px: int,
    margin_px: int,
    world_span_m: float,
    base_to_viewer,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build cones into `elev` (viewer metres; the tile plus `margin_px` on every side).

    `volcanoes` are in tile units from the tile centre (see chartTileVolcanoes); `base_to_viewer`
    converts a site's `baseDem` to viewer metres. Returns the new elevation, a 0..1 lava
    weight for the rock map, and the cones whose summit is inside the tile (for landmarks).
    """
    out = elev.astype(np.float32, copy=True)
    lava = np.zeros(elev.shape, dtype=np.float32)
    placed: list[dict] = []
    n = elev.shape[0]
    px_m = world_span_m / tile_px
    for v in volcanoes or []:
        kind = str(v.get("kind") or "strato")
        radius_px = float(v["radius"]) * tile_px
        if radius_px < 3.0:
            continue
        cj = margin_px + (float(v["x"]) + 0.5) * tile_px
        ci = margin_px + (float(v["z"]) + 0.5) * tile_px
        reach = radius_px * 1.25
        i0, i1 = int(np.floor(ci - reach)), int(np.ceil(ci + reach)) + 1
        j0, j1 = int(np.floor(cj - reach)), int(np.ceil(cj + reach)) + 1
        i0c, i1c, j0c, j1c = max(i0, 0), min(i1, n), max(j0, 0), min(j1, n)
        if i0c >= i1c or j0c >= j1c:
            continue
        vid = int(v.get("id") or 0)
        ii, jj = np.mgrid[i0c:i1c, j0c:j1c].astype(np.float32)
        di, dj = ii - ci, jj - cj
        theta = np.arctan2(di, dj)
        # A cone is never a perfect circle: its outline wobbles and one flank runs longer
        wobble = (
            1.0
            + 0.10 * np.sin(2.0 * theta + 6.283 * _hash01(vid, 1))
            + 0.06 * np.sin(3.0 * theta + 6.283 * _hash01(vid, 2))
            + 0.03 * np.sin(7.0 * theta + 6.283 * _hash01(vid, 3))
        )
        s = np.hypot(di, dj) / (radius_px * wobble)

        height_m = float(v.get("aspect", 0.4)) * radius_px * px_m
        base_m = max(float(base_to_viewer(float(v.get("baseDem", 0.0)))), 1.0)
        height_m = min(height_m, max(MAX_SUMMIT_M - base_m, height_m * 0.35))
        # Erosion gullies down the flanks, deepest mid-slope
        # (several uneven sets that twist with distance, so they do not read as a pinwheel)
        gullies = 0.0
        if kind != "shield":
            for m in range(4):
                k = 5 + int(_hash01(vid, 10 + m) * 14)
                twist = (_hash01(vid, 20 + m) - 0.5) * 5.0
                amp = 0.012 + 0.02 * _hash01(vid, 30 + m)
                gullies = gullies + amp * np.sin(k * theta + twist * s + 6.283 * _hash01(vid, 40 + m))
            gullies = gullies * np.clip(np.sin(np.pi * np.clip(s, 0.0, 1.0)), 0.0, 1.0) ** 1.5
        # The summit is cut off at the crater rim and hollowed out inside it
        crater_r = float(v.get("crater", 0.1))
        bowl = np.clip(1.0 - (s / crater_r) ** 2, 0.0, 1.0)
        if kind == "shield":
            bowl = np.clip(bowl * 3.0, 0.0, 1.0)  # flat-floored caldera
        shape = _profile(kind, np.maximum(s, crater_r)) - float(v.get("craterDepth", 0.15)) * bowl
        cone = base_m + height_m * np.clip(shape + gullies, 0.0, None)

        sub = out[i0c:i1c, j0c:j1c]
        # Pull the ground towards the cone, fully near the summit and not at all at the foot,
        # but never dig: where the land already stands higher the cone merges into it
        pull = np.clip(1.12 - s, 0.0, 1.0) ** 0.75 * 0.86
        built = np.maximum(sub, sub + (cone - sub) * pull)
        out[i0c:i1c, j0c:j1c] = built

        # Lava and ash: solid on the cone, breaking into tongues that run out past its foot
        tongues = (
            0.5
            + 0.3 * np.sin((3 + int(_hash01(vid, 6) * 3)) * theta + 6.283 * _hash01(vid, 7))
            + 0.2 * np.sin((7 + int(_hash01(vid, 8) * 6)) * theta + 6.283 * _hash01(vid, 9))
        )
        cover = np.clip((0.95 + 0.4 * tongues - s) / 0.3, 0.0, 1.0)
        lava[i0c:i1c, j0c:j1c] = np.maximum(lava[i0c:i1c, j0c:j1c], cover)

        if margin_px <= ci < n - margin_px and margin_px <= cj < n - margin_px:
            placed.append({**v, "summitM": round(float(built.max()), 1), "heightM": round(height_m, 1)})
    return out, lava, placed


def pack_rock(ref_rock: np.ndarray | None, lava: np.ndarray | None, rock_n: int) -> np.ndarray:
    """
    The tile's bedrock raster, (rock_n, rock_n, 4) u8 in ROCK_CHANNELS order.

    `ref_rock` is Stage 1's regional bedrock over the tile (any square size, at most 0.8 per
    channel); `lava` the tile's 0..1 cone cover from `stamp_volcanoes`. Fresh lava takes the
    volcanic channel above 0.8, which the explorer draws as a young cone.
    """
    if ref_rock is None:
        rock = np.zeros((rock_n, rock_n, 4), dtype=np.float32)
    else:
        f = rock_n / ref_rock.shape[0]
        rock = zoom(ref_rock.astype(np.float32) / 255.0, (f, f, 1), order=1)
    if lava is not None and lava.max() > 0.0:
        cover = np.clip(zoom(lava, rock_n / lava.shape[0], order=1), 0.0, 1.0)[..., None]
        fresh = np.zeros(4, dtype=np.float32)
        fresh[1] = 1.0
        rock = rock * (1.0 - cover) + fresh * cover
    return np.clip(np.rint(rock * 255.0), 0, 255).astype(np.uint8)
