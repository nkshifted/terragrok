"""Climate → biome / liquid-runoff / lake-river budgets (torch-free)."""

from __future__ import annotations

import numpy as np


def compute_biome_hydrology_profile(
    elev: np.ndarray,
    temp_c: np.ndarray,
    precip_mm: np.ndarray,
    res_scale: float,
) -> dict:
    """
    Derive physical liquid runoff, lake/river budgets, absolute moisture, and biome label
    directly from absolute temperature (°C) and precipitation (mm/yr):
      - Polar Glaciers (mean T < -3.5°C): 0 liquid rivers, 0 unfrozen lakes
      - Hot/Cold Arid Deserts (< 240 mm/yr): 0 rivers, 0 lakes (wind-sculpted sand & rock)
      - Tundra / Dry Steppe: sparse 1-5 rivers, 0-2 lakes
      - Temperate / Rainforest: full rich alpine lake & river watersheds
    """
    land = elev > 0.5
    if land.any():
        mean_t = float(np.mean(temp_c[land]))
        mean_p = float(np.mean(precip_mm[land]))
    else:
        mean_t = float(np.mean(temp_c))
        mean_p = float(np.mean(precip_mm))

    freeze_factor = np.clip((temp_c + 3.0) / 5.5, 0.0, 1.0)
    evap_floor = 180.0 + np.clip((temp_c - 10.0) * 8.0, 0.0, 130.0)
    aridity_factor = np.clip((precip_mm - evap_floor) / 560.0, 0.0, 1.65)
    runoff_map = (freeze_factor * aridity_factor).astype(np.float32)

    wetness = float(np.mean(runoff_map[land])) if land.any() else float(np.mean(runoff_map))

    if mean_t < -3.5:
        biome = "Polar Glacier"
        max_lakes = 0
        max_rivers = 0
        min_lake_runoff = 0.35
    elif wetness < 0.14 or (mean_p < 300.0 and mean_t > 12.0):
        biome = "Arid Desert"
        max_lakes = 0
        max_rivers = 0
        min_lake_runoff = 0.35
    elif mean_t < 2.5 and wetness < 0.45:
        biome = "Arctic Tundra"
        max_lakes = max(0, int(round(wetness * 6)))
        max_rivers = max(1, int(round(wetness * 10)))
        min_lake_runoff = 0.16
    elif wetness < 0.36:
        biome = "Dry Steppe"
        max_lakes = max(0, int(round((wetness - 0.14) * 5)))
        max_rivers = max(1, int(round((wetness - 0.08) * 10)))
        min_lake_runoff = 0.22
    elif mean_t > 18.0 and mean_p > 1050.0:
        biome = "Tropical Rainforest"
        max_lakes = 10
        max_rivers = 22
        min_lake_runoff = 0.15
    else:
        biome = "Temperate Highlands"
        t_norm = float(np.clip((wetness - 0.30) / 0.65, 0.0, 1.0))
        max_lakes = int(round(3 + t_norm * 7))
        max_rivers = int(round(6 + t_norm * 14))
        min_lake_runoff = 0.15

    abs_moist = np.clip((precip_mm - 140.0) / 920.0, 0.02, 0.96) * np.clip((temp_c + 3.5) / 7.0, 0.0, 1.0)

    return {
        "biome": biome,
        "mean_t": mean_t,
        "mean_p": mean_p,
        "wetness": wetness,
        "runoff_map": runoff_map,
        "abs_moist": abs_moist.astype(np.float32),
        "max_lakes": max_lakes,
        "max_spring_rivers": max_rivers,
        "min_lake_runoff": min_lake_runoff,
        "min_lake_cells": int(round(28 * res_scale)),
    }
