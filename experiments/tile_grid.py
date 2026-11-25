"""Planet tile-grid helpers: stable ids, tangent-plane neighbor rings (no torch)."""

from __future__ import annotations

import math
from typing import Any


def tile_id(seed: int, lat_deg: float, lon_deg: float, span_deg: float = 46.0) -> str:
    """Stable id for a square patch centered at (lat, lon) on planet seed."""
    lat_i = int(round(lat_deg * 10))
    lon_i = int(round(lon_deg * 10))
    span_i = int(round(span_deg))
    ns = "n" if lat_i >= 0 else "s"
    ew = "e" if lon_i >= 0 else "w"
    return f"tile_s{int(seed)}_{abs(lat_i)}{ns}_{abs(lon_i)}{ew}_{span_i}"


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


def latlon_to_unit(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    cp = math.cos(lat)
    return (math.sin(lon) * cp, math.sin(lat), math.cos(lon) * cp)


def unit_to_latlon(x: float, y: float, z: float) -> tuple[float, float]:
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    lat = _rad2deg(math.asin(max(-1.0, min(1.0, y))))
    lon = _rad2deg(math.atan2(x, z))
    return lat, lon


def great_circle_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    a = latlon_to_unit(lat1, lon1)
    b = latlon_to_unit(lat2, lon2)
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return _rad2deg(math.acos(dot))


def _tangent_basis(lat_deg: float, lon_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return (center, east, north) unit vectors at lat/lon (Y-up globe)."""
    c = latlon_to_unit(lat_deg, lon_deg)
    # East = normalize(worldY × c); near poles fall back to lon-based east.
    east = (c[2], 0.0, -c[0])
    en = math.sqrt(east[0] ** 2 + east[2] ** 2)
    if en < 1e-8:
        lon = _deg2rad(lon_deg)
        east = (-math.sin(lon), 0.0, math.cos(lon))
        en = math.sqrt(east[0] ** 2 + east[2] ** 2) or 1.0
    east = (east[0] / en, 0.0, east[2] / en)
    # north = east × c  (points toward +lat)
    north = (
        east[1] * c[2] - east[2] * c[1],
        east[2] * c[0] - east[0] * c[2],
        east[0] * c[1] - east[1] * c[0],
    )
    nn = math.sqrt(north[0] ** 2 + north[1] ** 2 + north[2] ** 2) or 1.0
    north = (north[0] / nn, north[1] / nn, north[2] / nn)
    return c, east, north


def offset_latlon(lat_deg: float, lon_deg: float, east_deg: float, north_deg: float) -> tuple[float, float]:
    """Move along local east/north by angular degrees (small-step geodesic on sphere)."""
    c, east, north = _tangent_basis(lat_deg, lon_deg)
    # Rodrigues-style: rotate center toward (east*te + north*tn)
    te = _deg2rad(east_deg)
    tn = _deg2rad(north_deg)
    ang = math.hypot(te, tn)
    if ang < 1e-12:
        return lat_deg, lon_deg
    # Direction in tangent plane
    dx = (east[0] * te + north[0] * tn) / ang
    dy = (east[1] * te + north[1] * tn) / ang
    dz = (east[2] * te + north[2] * tn) / ang
    ca, sa = math.cos(ang), math.sin(ang)
    x = c[0] * ca + dx * sa
    y = c[1] * ca + dy * sa
    z = c[2] * ca + dz * sa
    return unit_to_latlon(x, y, z)


def neighbor_centers(
    lat_deg: float,
    lon_deg: float,
    span_deg: float = 46.0,
    max_ring: int = 1,
) -> list[dict[str, Any]]:
    """
    Tangent-plane square grid neighbors around (lat, lon), ring-major order.

    qi = east index, qj = north index. Home (0,0) is excluded.
    Polar caveat: basis degenerates near ±90° — fine for ≤2 rings away from poles.
    """
    if max_ring < 1:
        return []
    out: list[dict[str, Any]] = []
    for ring in range(1, max_ring + 1):
        for qj in range(-ring, ring + 1):
            for qi in range(-ring, ring + 1):
                if max(abs(qi), abs(qj)) != ring:
                    continue
                nlat, nlon = offset_latlon(lat_deg, lon_deg, qi * span_deg, qj * span_deg)
                out.append(
                    {
                        "qi": qi,
                        "qj": qj,
                        "ring": ring,
                        "latDeg": round(nlat, 4),
                        "lonDeg": round(nlon, 4),
                        "spanDeg": float(span_deg),
                    }
                )
    return out
