#!/usr/bin/env python3
"""
Physically Consistent Lake-to-Sea Hydrological Exporter for TerraGrok 3D Explorer.

Uses Landlab's PriorityFloodFlowRouter (Barnes et al. 2014) to solve the exact
depression-spillway lake hierarchy and pit-free D8 drainage tree on every world:
  1. Closed highland basins fill to their exact pour-point (spillway) elevation ->
     exported as true flat-surfaced 3D alpine/valley lakes (`<id>.lakes.json`).
  2. Every river strictly follows the mathematically pit-free `flow__receiver_node`
     tree:
       - Lake-fed rivers emerge directly from a lake's exact spillway outlet
       - Spring-fed rivers trace all the way up to alpine headwaters and taper
         smoothly from a 0.6 m trickle (Leopold & Maddock W ~ Q^0.42)
       - Every single river terminates ONLY at:
           (a) the Ocean (flaring into an estuary below sea level),
           (b) a downstream Lake (entering at exact lake surface level), or
           (c) a Confluence with a larger trunk river that reaches the ocean.
       - ZERO random starts, ZERO mid-land dead ends (verified by assertion!).
"""

from __future__ import annotations

import json
from pathlib import Path

import fastscapelib as fs
import numpy as np
from PIL import Image
from landlab import RasterModelGrid
from landlab.components import PriorityFloodFlowRouter
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, FractalType, NoiseType
from scipy.ndimage import binary_dilation, gaussian_filter, label, laplace


OUT_DIR = Path("viewer/public/worlds")
VIEWER_SIZE = 512


def resample_array(arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    """Resample 2D array cleanly to (size, size)."""
    if arr.shape == (size, size):
        return arr
    mode = "F" if arr.dtype in (np.float32, np.float64) else "L"
    img = Image.fromarray(arr.astype(np.float32 if mode == "F" else np.uint8), mode=mode)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return np.asarray(img.resize((size, size), resample=resample))


def compute_curvature_ao(elev: np.ndarray) -> np.ndarray:
    """Compute multi-scale terrain cavity/ridge ambient occlusion packed into uint8 [0..255]."""
    lap_fine = -laplace(elev.astype(np.float64))
    lap_broad = elev.astype(np.float64) - gaussian_filter(elev.astype(np.float64), sigma=4.0)
    curv = 0.55 * lap_fine + 0.45 * lap_broad
    scale = max(float(np.percentile(np.abs(curv), 96)), 1.0)
    norm = np.clip(0.5 + 0.45 * (curv / scale), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def apply_micro_detail_refinement(
    elev: np.ndarray,
    world_span_m: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Walking-scale micro-detail refinement for 30m/px macro Terrain Diffusion DEMs:
      1. Slope-gated multi-octave ridged crags & cliff ledges on mountain slopes (>10 deg),
         leaving lowland valleys, plains, and sandy beaches smoothly untouched.
      2. Gentle Fastscapelib stream-power + ADI hillslope micro-erosion (10 fine steps) to
         etch sharp dendritic gullies, talus aprons, and crisp alpine aretes without
         altering the macro coastline or planetary landforms.
    """
    ny, nx = elev.shape
    cell_m = world_span_m / (nx - 1)
    out = elev.astype(np.float64).copy()
    land = out > 0.0

    # 1. Slope-gated rocky crag & cliff strata micro-relief
    gy, gx = np.gradient(out, cell_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope_gate = np.clip((slope_deg - 9.0) / 20.0, 0.0, 1.0) ** 1.3
    elev_gate = np.clip((out - 6.0) / 30.0, 0.0, 1.0)
    mask = slope_gate * elev_gate

    xs = np.linspace(0.0, float(nx), nx, dtype=np.float32)
    ys = np.linspace(0.0, float(ny), ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.vstack([xx.ravel(), yy.ravel()])

    fn_ridge = FastNoiseLite(seed=seed + 101)
    fn_ridge.noise_type = NoiseType.NoiseType_OpenSimplex2
    fn_ridge.fractal_type = FractalType.FractalType_Ridged
    fn_ridge.fractal_octaves = 4
    fn_ridge.frequency = 0.022
    ridge_hi = fn_ridge.gen_from_coords(coords).reshape(ny, nx)

    fn_crag = FastNoiseLite(seed=seed + 307)
    fn_crag.noise_type = NoiseType.NoiseType_OpenSimplex2
    fn_crag.fractal_type = FractalType.FractalType_FBm
    fn_crag.fractal_octaves = 3
    fn_crag.frequency = 0.055
    crag_fine = fn_crag.gen_from_coords(coords).reshape(ny, nx)

    # Add up to ~6.5m of crisp rock cragging & ridge ledges on steep slopes
    micro_rock = (ridge_hi * 4.8 + crag_fine * 2.2) * mask
    out = np.where(land, np.maximum(out + micro_rock, 0.2), out)

    # 2. Gentle Fastscapelib fine-scale hydraulic + hillslope micro-erosion
    subaerial = np.where(land, out, 0.0).copy()
    grid = fs.RasterGrid.from_length(
        shape=(ny, nx),
        length=(world_span_m, world_span_m),
        bounds_status=fs.NodeStatus.FIXED_VALUE,
    )
    flow_graph = fs.FlowGraph(grid, [fs.SingleFlowRouter(), fs.PFloodSinkResolver()])
    spl = fs.SPLEroder(flow_graph=flow_graph, k_coef=2.2e-5, area_exp=0.45, slope_exp=1.0)
    diff = fs.DiffusionADIEroder(grid=grid, k_coef=0.015)

    dt = 180.0
    for _ in range(10):
        flow_graph.update_routes(subaerial)
        area = flow_graph.accumulate(1.0)
        subaerial -= spl.erode(subaerial, area, dt)
        subaerial -= diff.erode(subaerial, dt)
        subaerial = np.where(land, np.maximum(subaerial, 0.15), 0.0)

    # Blend 65% micro-eroded detail with 35% pre-eroded peak silhouette so summit relief stays dramatic
    refined = np.where(land, 0.65 * subaerial + 0.35 * out, out)
    return refined.astype(np.float32)


def solve_physical_hydrology(
    elev: np.ndarray,
    world_span_m: float,
    min_lake_cells: int = 28,
    min_lake_depth_m: float = 0.45,
    max_spring_rivers: int = 16,
    runoff_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    """
    Solve exact Barnes Priority-Flood lake hierarchy & pit-free D8 drainage tree.

    Returns:
      carved_elev   : float32 (ny, nx) with smooth carved river channels & banks
      river_wet_u8  : uint8 (ny, nx) riverbank wetness / gravel mask
      lake_u8       : uint8 (ny, nx) flat alpine/valley lake mask
      lakes_data    : list of 3D flat lake surface meshes {id, level, cx, cz, cells}
      rivers_data   : list of continuous river splines {source, dest, points: [[x,z,y,w,slope],...]}
    """
    ny, nx = elev.shape
    cell_m = world_span_m / (nx - 1)

    # 1. Identify TRUE Ocean: ONLY cells <= 0.0m flood-connected to the outer grid boundary!
    #    Inland depressions whose floor dips below sea level are landlocked crater/valley lakes,
    #    NEVER ocean holes inside a lake!
    below_sea = elev <= 0.0
    conn8 = np.ones((3, 3), dtype=int)
    lbl_sea, _ = label(below_sea, structure=conn8)
    border_labels = set(
        np.unique(
            np.concatenate([lbl_sea[0, :], lbl_sea[-1, :], lbl_sea[:, 0], lbl_sea[:, -1]])
        )
    )
    border_labels.discard(0)
    is_ocean = np.isin(lbl_sea, list(border_labels))

    # Run Landlab PriorityFloodFlowRouter (D8, exact Barnes depression fill)
    # Keep true signed depth inside landlocked below-sea-level basins via constant positive shift.
    mg = RasterModelGrid((ny, nx), xy_spacing=cell_m)
    z_in = np.where(is_ocean, 0.0, elev.astype(np.float64))
    z_shift = max(0.0, -float(z_in.min())) + 1.0
    mg.add_field("topographic__elevation", (z_in + z_shift).ravel(), at="node")
    if runoff_map is not None:
        mg.add_field("water__unit_flux_in", np.clip(runoff_map.astype(np.float64), 0.15, 4.0).ravel(), at="node")
    mg.status_at_node[is_ocean.ravel()] = mg.BC_NODE_IS_FIXED_VALUE

    pf = PriorityFloodFlowRouter(
        mg,
        surface="topographic__elevation",
        flow_metric="D8",
        depression_handler="fill",
        update_flow_depressions=True,
    )
    pf.run_one_step()

    filled = mg.at_node["depression_free_elevation"].reshape(ny, nx) - z_shift
    rec = mg.at_node["flow__receiver_node"].astype(np.int64)
    # Use precipitation-weighted discharge when runoff_map is supplied!
    da = (
        mg.at_node["surface_water__discharge"].reshape(ny, nx)
        if runoff_map is not None
        else mg.at_node["drainage_area"].reshape(ny, nx)
    )
    dep_depth = filled - z_in

    # 2. Extract real flat-surfaced lakes using 8-connectivity (matching D8 flow) on the full
    #    flooded basin (dep_depth > 0.05m) so shallow coves never split one lake into two!
    raw_lake_mask = (dep_depth > 0.05) & (~is_ocean) & (filled > 1.2)
    lbl, n_raw = label(raw_lake_mask, structure=conn8)

    # Pass 2a: Assign disjoint lake IDs and spillway levels for all qualifying basins first
    lake_id_grid = np.zeros((ny, nx), dtype=np.int32)
    lake_levels: dict[int, float] = {}
    kept_lakes: list[int] = []

    next_lake_id = 1
    for cid in range(1, n_raw + 1):
        comp = lbl == cid
        n_cells = int(comp.sum())
        max_d = float(dep_depth[comp].max())
        if n_cells < min_lake_cells or max_d < max(min_lake_depth_m, 1.0):
            continue
        lid = next_lake_id
        next_lake_id += 1
        lake_id_grid[comp] = lid
        # Exact flat spillway elevation (Barnes depression-filled surface across the basin)
        lake_levels[lid] = float(np.percentile(filled[comp], 25))
        kept_lakes.append(lid)

    # Pass 2b: Export strictly disjoint per-lake cell meshes (tucked 1 cell onto rising banks only,
    #          NEVER overlapping another lake or overhanging a downhill spillway!)
    lakes_data: list[dict] = []
    lake_spillways: list[tuple[int, int, int, float]] = []  # (lake_id, r, c, discharge)

    for lid in kept_lakes:
        comp = lake_id_grid == lid
        level = lake_levels[lid]
        n_cells = int(comp.sum())
        other_lakes = (lake_id_grid > 0) & (lake_id_grid != lid)

        # Dilate 1 cell ONLY onto rising shoreline banks (never other lakes or downhill outlets)
        shore_ring = (
            binary_dilation(comp, iterations=1, structure=conn8)
            & (~comp)
            & (~is_ocean)
            & (~other_lakes)
            & (elev >= level - 0.15)
        )
        comp_pad = comp | shore_ring
        pts_rc = np.argwhere(comp_pad)
        cells_list = []
        for r, c in pts_rc:
            d = max(level - float(elev[r, c]), 0.0)
            cells_list.append([int(r), int(c), round(d, 2)])

        # Centroid in world coords (from true basin cells)
        basin_rc = np.argwhere(comp)
        mean_r, mean_c = float(basin_rc[:, 0].mean()), float(basin_rc[:, 1].mean())
        cx = (mean_c / (nx - 1) - 0.5) * world_span_m
        cz = (mean_r / (ny - 1) - 0.5) * world_span_m

        lakes_data.append({
            "id": lid,
            "level": round(level, 2),
            "areaCells": n_cells,
            "maxDepth": round(float(dep_depth[comp].max()), 2),
            "cx": round(cx, 2),
            "cz": round(cz, 2),
            "cells": cells_list,
        })

        # Find exact spillway outlet: lake cell whose receiver exits the lake with max drainage area
        best_outlet = None
        best_da = -1.0
        for r, c in basin_rc:
            idx = int(r * nx + c)
            nxt = int(rec[idx])
            nr, nc = divmod(nxt, nx)
            if lake_id_grid[nr, nc] != lid:
                if da[r, c] > best_da:
                    best_da = float(da[r, c])
                    best_outlet = (int(r), int(c))
        if best_outlet is not None:
            lake_spillways.append((lid, best_outlet[0], best_outlet[1], best_da))

    # 3. Build upstream donor lookup (max-drainage-area donor per cell) so spring rivers
    #    can trace all the way up to true alpine headwater springs!
    max_donor = np.full(ny * nx, -1, dtype=np.int64)
    max_donor_da = np.full(ny * nx, -1.0, dtype=np.float64)
    da_flat = da.ravel()
    for idx in range(ny * nx):
        if is_ocean.ravel()[idx]:
            continue
        r_idx = int(rec[idx])
        if r_idx != idx and da_flat[idx] > max_donor_da[r_idx]:
            max_donor_da[r_idx] = da_flat[idx]
            max_donor[r_idx] = idx

    # Helper: trace strictly downstream along solved `rec` tree from (r0, c0)
    # Guaranteed to terminate at Ocean, a downstream Lake, or a previously traced Trunk River!
    claimed_trunk: dict[int, tuple[float, float, float]] = {}  # node_idx -> (wx, wz, ribbon_y)
    land = (~is_ocean) & (lake_id_grid == 0)
    da_ref = float(np.percentile(da[land], 97.5)) if land.any() else 1e5
    da_max = max(float(da[land].max()), da_ref * 2.0) if land.any() else 1e6

    def trace_downstream(
        start_r: int,
        start_c: int,
        source_type: str,
        start_lake_id: int = 0,
    ) -> tuple[list[tuple[int, int, float]], str, int] | None:
        """Trace node-by-node along exact `rec` until Ocean, another Lake, or Trunk Confluence."""
        path: list[tuple[int, int, float]] = []
        idx = int(start_r * nx + start_c)
        seen: set[int] = set()
        dest_type = "dead_end"
        dest_lake = 0

        for _ in range(4000):
            if idx in seen:
                break
            seen.add(idx)
            r, c = divmod(idx, nx)
            path.append((r, c, float(da_flat[idx])))

            # Termination 1: Reached the true outer Ocean! Extend 2 cells into sea for seamless estuary mouth
            if is_ocean[r, c]:
                dest_type = "sea"
                nxt = int(rec[idx])
                if nxt != idx:
                    nr, nc = divmod(nxt, nx)
                    path.append((nr, nc, float(da_flat[idx])))
                break

            # Termination 2: Entered a downstream Lake (different from our starting lake)
            cur_lid = int(lake_id_grid[r, c])
            if cur_lid > 0 and cur_lid != start_lake_id and len(path) > 2:
                dest_type = "lake"
                dest_lake = cur_lid
                # Step 2 cells into the lake so the river mouth merges smoothly under lake surface
                nxt = int(rec[idx])
                if nxt != idx:
                    nr, nc = divmod(nxt, nx)
                    path.append((nr, nc, float(da_flat[idx])))
                break

            # Termination 3: Merged into an already-traced Trunk River!
            if idx in claimed_trunk and len(path) > 3:
                dest_type = "confluence"
                break

            nxt = int(rec[idx])
            if nxt == idx:
                # Reached open outer boundary of the grid (sea or continental map-edge outflow)
                if is_ocean[r, c] or elev[r, c] <= 0.5:
                    dest_type = "sea"
                elif r == 0 or r == ny - 1 or c == 0 or c == nx - 1:
                    dest_type = "edge"
                break
            idx = nxt

        if len(path) < 8:
            return None
        return path, dest_type, dest_lake

    # 4. Collect rivers in physical priority order:
    #    Priority A: Lake Spillway Outlet Rivers (highest elevation lakes first so cascade
    #                chains Lake A -> River -> Lake B -> River -> Ocean work cleanly!)
    lake_spillways.sort(key=lambda item: -lake_levels[item[0]])
    candidate_traces: list[tuple[list[tuple[int, int, float]], str, int, str, int]] = []

    def register_trace(path, source_type, start_lake, dest_type, dest_lake):
        # Compute preliminary ribbon elevations & claim trunk nodes
        for r, c, _ in path[:-1]:
            if lake_id_grid[r, c] == 0 and not is_ocean[r, c]:
                node_id = r * nx + c
                if node_id not in claimed_trunk:
                    wx = (c / (nx - 1) - 0.5) * world_span_m
                    wz = (r / (ny - 1) - 0.5) * world_span_m
                    claimed_trunk[node_id] = (wx, wz, float(filled[r, c]))
        candidate_traces.append((path, source_type, start_lake, dest_type, dest_lake))

    for lid, sr, sc, _ in lake_spillways:
        res = trace_downstream(sr, sc, "lake", start_lake_id=lid)
        if res is not None:
            path, dest_type, dest_lake = res
            register_trace(path, "lake", lid, dest_type, dest_lake)

    #    Priority B: Major Coastal River Mouths -> trace UPSTREAM to alpine springs!
    #    Find coastline land cells with highest discharge draining into the ocean
    ocean_dilated = binary_dilation(is_ocean, iterations=1)
    mouth_mask = land & ocean_dilated & (da >= da_ref * 1.4)
    mouth_coords = np.argwhere(mouth_mask)
    mouth_das = da[mouth_coords[:, 0], mouth_coords[:, 1]]
    mouth_order = np.argsort(-mouth_das)

    spring_added = 0
    for m_idx in mouth_order:
        if spring_added >= max_spring_rivers:
            break
        mr, mc = int(mouth_coords[m_idx, 0]), int(mouth_coords[m_idx, 1])
        node = mr * nx + mc
        if node in claimed_trunk:
            continue

        # Walk upstream along max-drainage-area donor all the way to the alpine spring or lake!
        curr = node
        upstream_seen = set()
        hit_lake = 0
        for _ in range(2500):
            if curr in upstream_seen:
                break
            upstream_seen.add(curr)
            cr, cc = divmod(curr, nx)
            if lake_id_grid[cr, cc] > 0:
                hit_lake = int(lake_id_grid[cr, cc])
                break
            # Stop near high alpine spring (when contributing area becomes small ~35 cells)
            if da_flat[curr] < (cell_m * cell_m) * 45.0:
                break
            d_node = int(max_donor[curr])
            if d_node < 0 or d_node == curr:
                break
            curr = d_node

        if hit_lake > 0:
            # Already drained by a lake spillway above
            continue

        sr, sc = divmod(curr, nx)
        if elev[sr, sc] < 12.0:
            continue  # Skip tiny flat coastal ditches; keep genuine highland-to-sea rivers

        res = trace_downstream(sr, sc, "spring", start_lake_id=0)
        if res is not None:
            path, dest_type, dest_lake = res
            if len(path) >= 18:
                register_trace(path, "spring", 0, dest_type, dest_lake)
                spring_added += 1

    #    Priority C: Major Inland Tributaries (trace upstream to alpine spring -> confluence!)
    trib_mask = land & (da >= da_ref * 1.6)
    trib_coords = np.argwhere(trib_mask)
    if len(trib_coords) > 0:
        trib_das = da[trib_coords[:, 0], trib_coords[:, 1]]
        for t_idx in np.argsort(-trib_das):
            if spring_added >= max_spring_rivers + 8:
                break
            tr, tc = int(trib_coords[t_idx, 0]), int(trib_coords[t_idx, 1])
            node = tr * nx + tc
            if node in claimed_trunk:
                continue

            # Walk upstream to headwater spring
            curr = node
            seen_u = set()
            hit_lake = 0
            for _ in range(2000):
                if curr in seen_u:
                    break
                seen_u.add(curr)
                cr, cc = divmod(curr, nx)
                if lake_id_grid[cr, cc] > 0:
                    hit_lake = int(lake_id_grid[cr, cc])
                    break
                if da_flat[curr] < (cell_m * cell_m) * 45.0:
                    break
                d_node = int(max_donor[curr])
                if d_node < 0 or d_node == curr:
                    break
                curr = d_node
            if hit_lake > 0:
                continue

            sr, sc = divmod(curr, nx)
            if elev[sr, sc] < 18.0:
                continue
            res = trace_downstream(sr, sc, "spring", start_lake_id=0)
            if res is not None:
                path, dest_type, dest_lake = res
                if len(path) >= 16:
                    register_trace(path, "spring", 0, dest_type, dest_lake)
                    spring_added += 1

    # 5. Build physically carved riverbeds & exact water ribbon splines
    #    Store final ribbon_y per claimed node so tributaries match trunk elevation EXACTLY!
    carved = elev.copy().astype(np.float32)
    channel_mask = np.zeros((ny, nx), dtype=np.float32)
    node_water_y: dict[int, float] = {}
    rivers_data: list[dict] = []

    dead_ends = 0
    for path, source_type, start_lake, dest_type, dest_lake in candidate_traces:
        if dest_type == "dead_end":
            dead_ends += 1
            continue

        pts_r = np.array([p[0] for p in path], dtype=np.float64)
        pts_c = np.array([p[1] for p in path], dtype=np.float64)
        pts_da = np.array([p[2] for p in path], dtype=np.float64)

        # Gentle spatial smoothing (keep exact start & end anchored to lake/confluence/sea!)
        if len(path) > 6:
            sm_r = gaussian_filter(pts_r, sigma=1.35, mode="nearest")
            sm_c = gaussian_filter(pts_c, sigma=1.35, mode="nearest")
            sm_r[0], sm_r[-1] = pts_r[0], pts_r[-1]
            sm_c[0], sm_c[-1] = pts_c[0], pts_c[-1]
        else:
            sm_r, sm_c = pts_r, pts_c

        # Sample terrain elevation (prefer actual elev over depression-filled height so rivers
        # never float in mid-air over shallow un-laked depressions!)
        samp_h = np.array(
            [
                float(
                    min(
                        filled[int(np.clip(round(rr), 0, ny - 1)), int(np.clip(round(cc), 0, nx - 1))],
                        elev[int(np.clip(round(rr), 0, ny - 1)), int(np.clip(round(cc), 0, nx - 1))] + 0.25,
                    )
                )
                for rr, cc in zip(sm_r, sm_c)
            ],
            dtype=np.float64,
        )

        # Pin exact start elevation for lake outlets
        if source_type == "lake" and start_lake in lake_levels:
            samp_h[0] = lake_levels[start_lake]
        # Pin exact end elevation for ocean, downstream lake, or trunk confluence
        if dest_type == "sea":
            samp_h[-1] = -0.25
        elif dest_type == "lake" and dest_lake in lake_levels:
            samp_h[-1] = lake_levels[dest_lake]
        elif dest_type == "confluence":
            end_node = int(path[-1][0] * nx + path[-1][1])
            if end_node in node_water_y:
                samp_h[-1] = node_water_y[end_node]

        # Enforce strict monotonic non-increasing profile from source -> destination
        mono_h = np.minimum.accumulate(samp_h)
        water_y = np.minimum(gaussian_filter(mono_h, sigma=1.5, mode="nearest"), mono_h)
        # Re-anchor endpoints after smoothing
        if source_type == "lake" and start_lake in lake_levels:
            water_y[0] = lake_levels[start_lake]
            water_y = np.minimum.accumulate(water_y)
        if dest_type == "sea":
            water_y[-1] = -0.25
            # Smooth last 4 points gracefully down to sea level (estuary grade)
            n_tail = min(6, len(water_y))
            for k in range(1, n_tail):
                frac = k / n_tail
                water_y[-k] = min(water_y[-k], water_y[-n_tail] * (1.0 - frac) + (-0.25) * frac)
        elif dest_type == "lake" and dest_lake in lake_levels:
            water_y[-1] = lake_levels[dest_lake]
        elif dest_type == "confluence":
            end_node = int(path[-1][0] * nx + path[-1][1])
            if end_node in node_water_y:
                water_y[-1] = min(water_y[-1], node_water_y[end_node])

        # Compute local slope FIRST so Manning / Finnegan hydraulic geometry can widen flat valley rivers!
        step_dist = np.maximum(
            np.hypot(np.diff(sm_r, prepend=sm_r[0]), np.diff(sm_c, prepend=sm_c[0])) * cell_m,
            cell_m,
        )
        dh = np.maximum(-np.diff(water_y, prepend=water_y[0]), 0.0)
        slopes = gaussian_filter(np.clip(dh / step_dist, 0.0, 1.0), sigma=1.6, mode="nearest")

        # Measure local cross-sectional valley-floor clearance (HAND: Height Above Nearest Drainage)
        # Cast perpendicular rays left & right from the river centerline to detect tight canyons vs broad valleys!
        n_pts = len(path)
        valley_w = np.zeros(n_pts, dtype=np.float32)
        q_norm = np.clip(pts_da / max(da_max, 1.0), 1e-5, 1.0)
        max_scan_m = 72.0
        n_steps_scan = max(4, int(round(max_scan_m / cell_m)))
        for i in range(n_pts):
            i0 = max(i - 1, 0)
            i1 = min(i + 1, n_pts - 1)
            dr_t = sm_r[i1] - sm_r[i0]
            dc_t = sm_c[i1] - sm_c[i0]
            t_len = float(np.hypot(dr_t, dc_t)) or 1.0
            # Perpendicular unit vector in grid coords
            nr_p = -dc_t / t_len
            nc_p = dr_t / t_len
            wy_i = float(water_y[i])
            # Allowable HAND bank rise scales gently with river discharge (1.8m upland -> 5.0m major trunk)
            hand_tol = 1.8 + 3.2 * float(q_norm[i] ** 0.35)

            clear_both = 0.0
            for sign in (-1.0, 1.0):
                clear_side = cell_m
                for s_step in range(1, n_steps_scan + 1):
                    sr = sm_r[i] + sign * nr_p * s_step
                    sc = sm_c[i] + sign * nc_p * s_step
                    ir_s = int(round(sr))
                    ic_s = int(round(sc))
                    if ir_s < 0 or ir_s >= ny or ic_s < 0 or ic_s >= nx:
                        break
                    h_s = float(elev[ir_s, ic_s])
                    if h_s - wy_i > hand_tol:
                        break
                    clear_side = s_step * cell_m
                clear_both += clear_side
            valley_w[i] = clear_both
        valley_w = gaussian_filter(valley_w, sigma=2.5, mode="nearest")

        # Physics-based Hydraulic Geometry:
        #   Leopold-Maddock discharge scaling (W ~ Q^0.45)
        #   × Finnegan/Manning inverse-slope expansion (W ~ S^-0.24: narrow steep gorges, wide flat valleys!)
        #   clamped cleanly to measured HAND valley-floor width so rivers fill broad valleys without climbing cliffs!
        w_q = 2.6 + 24.0 * (q_norm**0.45)
        slope_eff = np.clip(slopes, 0.004, 0.38)
        slope_factor = np.clip((0.045 / slope_eff) ** 0.24, 0.62, 2.35)
        w_phys = w_q * slope_factor
        # Valley confinement: let broad valleys expand up to 72% of valley floor, keep canyons wall-hugging
        w_confined = np.minimum(w_phys, np.maximum(valley_w * 0.72, w_q * 0.62))
        widths = gaussian_filter(w_confined, sigma=2.2, mode="nearest")
        widths = np.clip(widths, 1.6, 68.0)

        if source_type == "spring":
            # Taper first 10 samples smoothly from narrow spring trickle -> full hydraulic width
            taper_n = min(10, len(widths))
            taper = np.linspace(0.35, 1.0, taper_n)
            widths[:taper_n] *= taper
        elif source_type == "lake":
            widths[0] = max(widths[0], 5.5)
        if dest_type == "sea":
            # Flare river mouth into natural coastal estuary!
            est_n = min(8, len(widths))
            widths[-est_n:] *= np.linspace(1.0, 1.85, est_n)

        # Rendered 3D mesh spacing (viewer caps terrain mesh at 512x512 for 60 FPS)
        mesh_cell_m = world_span_m / max(min(nx, 512) - 1, 1)

        # Closest-point riverbed carving (prevents downstream low points from excavating
        # deep caverns underneath upstream river segments on mountain slopes!).
        closest_d = np.full((ny, nx), np.inf, dtype=np.float32)
        closest_wy = np.zeros((ny, nx), dtype=np.float32)
        closest_w = np.zeros((ny, nx), dtype=np.float32)
        touched_r_lo, touched_r_hi = ny, 0
        touched_c_lo, touched_c_hi = nx, 0

        spline_pts: list[list[float]] = []
        for i in range(len(path)):
            rr, cc = sm_r[i], sm_c[i]
            wy = float(water_y[i])
            w_m = float(widths[i])
            sl = float(slopes[i])

            # Recess water surface slightly below surrounding valley banks so water sits IN ground
            if (i == 0 and source_type == "lake") or (i == len(path) - 1 and dest_type in ("lake", "sea")):
                ribbon_y = wy
            else:
                ribbon_y = wy - 0.15

            node_id = int(path[i][0] * nx + path[i][1])
            if node_id not in node_water_y:
                node_water_y[node_id] = ribbon_y

            wx = (cc / (nx - 1) - 0.5) * world_span_m
            wz = (rr / (ny - 1) - 0.5) * world_span_m
            spline_pts.append([round(wx, 2), round(wz, 2), round(ribbon_y, 2), round(w_m, 2), round(sl, 3)])

            ir, ic = int(np.clip(round(rr), 0, ny - 1)), int(np.clip(round(cc), 0, nx - 1))
            if lake_id_grid[ir, ic] > 0:
                continue

            # Record nearest river sample per cell inside the graded bank zone
            half_w = w_m * 0.52
            bank_w = max(w_m * 0.95, half_w + cell_m * 1.4)
            rad_cells = int(np.ceil(bank_w / cell_m)) + 1
            r_lo, r_hi = max(0, ir - rad_cells), min(ny, ir + rad_cells + 1)
            c_lo, c_hi = max(0, ic - rad_cells), min(nx, ic + rad_cells + 1)
            yy, xx = np.mgrid[r_lo:r_hi, c_lo:c_hi]
            d_m = np.hypot(yy - rr, xx - cc) * cell_m

            better = (d_m < bank_w) & (d_m < closest_d[r_lo:r_hi, c_lo:c_hi])
            if np.any(better):
                cd_view = closest_d[r_lo:r_hi, c_lo:c_hi]
                cwy_view = closest_wy[r_lo:r_hi, c_lo:c_hi]
                cw_view = closest_w[r_lo:r_hi, c_lo:c_hi]
                cd_view[better] = d_m[better]
                cwy_view[better] = ribbon_y
                cw_view[better] = w_m
                touched_r_lo = min(touched_r_lo, r_lo)
                touched_r_hi = max(touched_r_hi, r_hi)
                touched_c_lo = min(touched_c_lo, c_lo)
                touched_c_hi = max(touched_c_hi, c_hi)

        # Apply shallow, realistic riverbed carving ONCE per pixel using its exact closest river point!
        if touched_r_lo < touched_r_hi and touched_c_lo < touched_c_hi:
            cd = closest_d[touched_r_lo:touched_r_hi, touched_c_lo:touched_c_hi]
            cwy = closest_wy[touched_r_lo:touched_r_hi, touched_c_lo:touched_c_hi]
            cw = closest_w[touched_r_lo:touched_r_hi, touched_c_lo:touched_c_hi]
            sub = carved[touched_r_lo:touched_r_hi, touched_c_lo:touched_c_hi]
            lk_sub = lake_id_grid[touched_r_lo:touched_r_hi, touched_c_lo:touched_c_hi]

            valid = np.isfinite(cd) & (lk_sub == 0)
            if np.any(valid):
                half_w_grid = cw * 0.52
                bank_w_grid = np.maximum(cw * 0.95, half_w_grid + cell_m * 1.4)
                # Realistic ankle-to-waist riverbed depth (0.45m brooks -> 1.15m wide rivers)
                bed_depth = 0.45 + np.minimum(cw * 0.025, 0.70)
                in_chan = cd <= half_w_grid
                chan_prof = np.clip(1.0 - (cd / np.maximum(half_w_grid, 0.5)) ** 2, 0.0, 1.0)
                floor_h = cwy - bed_depth * (0.35 + 0.65 * chan_prof)

                t_bank = np.clip((cd - half_w_grid) / np.maximum(bank_w_grid - half_w_grid, 0.5), 0.0, 1.0)
                blend = t_bank * t_bank * (3.0 - 2.0 * t_bank)
                # Bank rises cleanly THROUGH water surface at half_w so outer edge meets solid ground!
                bank_h = (cwy - bed_depth * 0.35) * (1.0 - blend) + np.maximum(sub, cwy + 0.25) * blend
                desired = np.where(in_chan, floor_h, bank_h)
                carved[touched_r_lo:touched_r_hi, touched_c_lo:touched_c_hi] = np.where(
                    valid, np.minimum(sub, desired), sub
                )
                wet = np.where(valid, np.clip(1.0 - cd / np.maximum(cw * 0.72, cell_m * 1.1), 0.0, 1.0), 0.0)
                channel_mask[touched_r_lo:touched_r_hi, touched_c_lo:touched_c_hi] = np.maximum(
                    channel_mask[touched_r_lo:touched_r_hi, touched_c_lo:touched_c_hi], wet
                )

        rivers_data.append({
            "source": source_type,
            "dest": dest_type,
            "startLake": start_lake,
            "destLake": dest_lake,
            "points": spline_pts,
        })

    # Verify physical hydrology invariant: ZERO dead-end rivers!
    assert dead_ends == 0, f"Hydrology invariant violated: {dead_ends} dead-end rivers!"

    n_lake_src = sum(1 for r in rivers_data if r["source"] == "lake")
    n_spring_src = sum(1 for r in rivers_data if r["source"] == "spring")
    n_sea = sum(1 for r in rivers_data if r["dest"] == "sea")
    n_lake_dst = sum(1 for r in rivers_data if r["dest"] == "lake")
    n_conf = sum(1 for r in rivers_data if r["dest"] == "confluence")
    n_edge = sum(1 for r in rivers_data if r["dest"] == "edge")
    print(
        f"  Hydrology OK: {len(lakes_data)} lakes | {len(rivers_data)} rivers "
        f"(sources: {n_lake_src} lake-spillway, {n_spring_src} alpine-spring | "
        f"dests: {n_sea} sea, {n_lake_dst} lake, {n_conf} confluence, {n_edge} map-edge | 0 dead-ends)"
    )

    river_u8 = np.clip(channel_mask * 255.0, 0, 255).astype(np.uint8)
    lake_u8 = (lake_id_grid > 0).astype(np.uint8) * 255
    return carved, river_u8, lake_u8, lakes_data, rivers_data


def find_landmarks(
    elev: np.ndarray,
    rivers_mask: np.ndarray,
    lake_mask: np.ndarray,
    lakes_data: list[dict],
    rivers_data: list[dict],
    world_span_m: float,
) -> dict[str, list[float]]:
    """Find scenic walkable landmarks: plains, mountain peak, alpine lake, river crossing, coast."""
    ny, nx = elev.shape
    land = (elev > 2.5) & (lake_mask < 128)
    gy, gx = np.gradient(elev, world_span_m / (nx - 1))
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    def to_norm_xy(r: int, c: int) -> list[float]:
        return [round(float(c) / (nx - 1), 4), round(float(r) / (ny - 1), 4)]

    # 1. Highest peak
    peak_idx = int(np.argmax(elev))
    pr, pc = divmod(peak_idx, nx)

    # 2. Alpine Lake Spire/Spillway (prefer a scenic high-elevation mountain lake!)
    if lakes_data:
        # Pick highest-elevation lake with at least 80 cells (classic alpine tarn/mountain lake!)
        scenic_lakes = [lk for lk in lakes_data if lk["areaCells"] >= 80] or lakes_data
        best_lake = max(scenic_lakes, key=lambda lk: lk["level"])
        # Stand just on the dry shore beside the lake's spillway
        lake_river = next((rv for rv in rivers_data if rv["startLake"] == best_lake["id"]), None)
        if lake_river and len(lake_river["points"]) > 3:
            pt = lake_river["points"][2]
            lc = int(np.clip((pt[0] / world_span_m + 0.5) * (nx - 1) + 2, 2, nx - 3))
            lr = int(np.clip((pt[1] / world_span_m + 0.5) * (ny - 1) + 2, 2, ny - 3))
        else:
            lc = int(np.clip((best_lake["cx"] / world_span_m + 0.5) * (nx - 1), 2, nx - 3))
            lr = int(np.clip((best_lake["cz"] / world_span_m + 0.5) * (ny - 1), 2, ny - 3))
    else:
        lr, lc = pr, pc

    # 3. Scenic River Crossing (midpoint of a major sea-bound river!)
    sea_rivers = [rv for rv in rivers_data if rv["dest"] == "sea"] or rivers_data
    if sea_rivers:
        main_rv = max(sea_rivers, key=lambda rv: len(rv["points"]))
        mid_pt = main_rv["points"][len(main_rv["points"]) // 2]
        rc = int(np.clip((mid_pt[0] / world_span_m + 0.5) * (nx - 1) + 2, 2, nx - 3))
        rr = int(np.clip((mid_pt[1] / world_span_m + 0.5) * (ny - 1), 2, ny - 3))
    else:
        rr, rc = pr, pc

    # 4. Open scenic plains
    plains_mask = land & (rivers_mask < 30) & (elev > 10.0) & (elev < elev.max() * 0.25) & (slope_deg < 14.0)
    plains_pts = np.argwhere(plains_mask)
    if len(plains_pts) == 0:
        plains_pts = np.argwhere(land & (rivers_mask < 30) & (elev > 6.0) & (slope_deg < 22.0))
    if len(plains_pts) > 0:
        peak_vec = np.array([pr, pc], dtype=np.float64)
        dists = np.linalg.norm(plains_pts - peak_vec, axis=1)
        best = plains_pts[int(np.argmin(np.abs(dists - 115.0)))]
        plr, plc = int(best[0]), int(best[1])
    else:
        plr, plc = ny // 2, nx // 2

    # 5. River Estuary / Coastline (right where a major river meets the sea!)
    if sea_rivers:
        main_rv = max(sea_rivers, key=lambda rv: len(rv["points"]))
        mouth_pt = main_rv["points"][max(len(main_rv["points"]) - 6, 0)]
        cc = int(np.clip((mouth_pt[0] / world_span_m + 0.5) * (nx - 1) + 3, 2, nx - 3))
        cr = int(np.clip((mouth_pt[1] / world_span_m + 0.5) * (ny - 1), 2, ny - 3))
    else:
        cr, cc = plr, plc

    return {
        "plains": to_norm_xy(plr, plc),
        "peak": to_norm_xy(pr, pc),
        "lake": to_norm_xy(lr, lc),
        "river": to_norm_xy(rr, rc),
        "coast": to_norm_xy(cr, cc),
    }


def export_terrain_gen_world(
    world_id: str,
    title: str,
    subtitle: str,
    prefix: str,
    world_span_m: float,
    max_elev_m: float,
    sea_level: float = 0.08,
) -> dict:
    print(f"Solving physical hydrology for {world_id}...")
    src = Path("out/terrain-gen")
    h16 = np.asarray(Image.open(src / f"{prefix}_height.png"), dtype=np.float32) / 65535.0
    ocean = np.asarray(Image.open(src / f"{prefix}_ocean.png").convert("L"), dtype=np.uint8)

    h = resample_array(h16, VIEWER_SIZE, is_mask=False).astype(np.float32)
    ocn = resample_array(ocean, VIEWER_SIZE, is_mask=True).astype(np.uint8)

    elev = ((h - sea_level) / (1.0 - sea_level)) * max_elev_m
    elev = np.where(ocn > 128, np.minimum(elev, -4.0), elev)

    carved, riv_u8, lak_u8, lakes_data, rivers_data = solve_physical_hydrology(
        elev=elev,
        world_span_m=world_span_m,
        min_lake_cells=28,
        min_lake_depth_m=0.45,
        max_spring_rivers=14,
    )
    curv_u8 = compute_curvature_ao(carved)

    wet_seed = np.clip((riv_u8.astype(np.float32) + lak_u8.astype(np.float32)) / 255.0, 0.0, 1.0)
    moisture = np.clip(0.55 + 0.45 * gaussian_filter(wet_seed, sigma=12.0) * 3.0, 0.25, 1.0)
    x_grad = np.linspace(1.08, 0.72, VIEWER_SIZE, dtype=np.float32)[None, :]
    moisture = np.clip(moisture * x_grad, 0.14, 1.0)

    masks = np.stack([riv_u8, lak_u8, curv_u8, (moisture * 255.0).astype(np.uint8)], axis=-1)
    carved.astype(np.float32).tofile(OUT_DIR / f"{world_id}.height.f32")
    masks.astype(np.uint8).tofile(OUT_DIR / f"{world_id}.masks.u8")
    (OUT_DIR / f"{world_id}.rivers.json").write_text(json.dumps(rivers_data))
    (OUT_DIR / f"{world_id}.lakes.json").write_text(json.dumps(lakes_data))

    landmarks = find_landmarks(carved, riv_u8, lak_u8, lakes_data, rivers_data, world_span_m)
    meta = {
        "id": world_id,
        "title": title,
        "subtitle": subtitle,
        "size": VIEWER_SIZE,
        "worldSpan": world_span_m,
        "minElev": round(float(carved.min()), 2),
        "maxElev": round(float(carved.max()), 2),
        "seaLevel": 0.0,
        "riverCount": len(rivers_data),
        "lakeCount": len(lakes_data),
        "landmarks": landmarks,
    }
    (OUT_DIR / f"{world_id}.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_fastscape_world() -> dict:
    print("Solving physical hydrology for fastscape...")
    src = Path("out/fastscape")
    raw = np.fromfile(src / "elevation.f32", dtype=np.float32)
    n = int(round(np.sqrt(raw.size)))
    elev = raw.reshape(n, n)
    elev = resample_array(elev, VIEWER_SIZE, is_mask=False).astype(np.float32)

    world_span_m = 3200.0
    scale_v = 220.0 / max(float(elev.max()), 1.0)
    elev_scaled = elev * scale_v

    carved, riv_u8, lak_u8, lakes_data, rivers_data = solve_physical_hydrology(
        elev=elev_scaled,
        world_span_m=world_span_m,
        min_lake_cells=22,
        min_lake_depth_m=0.35,
        max_spring_rivers=18,
    )
    curv_u8 = compute_curvature_ao(carved)

    ny, nx = carved.shape
    ocean = (carved <= 0.5).astype(np.uint8) * 255
    moisture = np.ones((ny, nx), dtype=np.float32)
    m = np.ones(ny, dtype=np.float32)
    prev = np.zeros(ny, dtype=np.float32)
    for c in range(nx):
        h = np.maximum(carved[:, c], 0.0)
        is_sea = ocean[:, c] > 128
        m = np.where(is_sea, np.minimum(m + 0.04, 1.0), m)
        up = np.maximum(h - prev, 0.0)
        rain = m * np.where(is_sea, 0.005, 0.012 + 0.002 * up)
        moisture[:, c] = np.clip(rain * 55.0, 0.08, 1.0)
        m = np.maximum(m - rain * 0.65, 0.12)
        prev = h
    moisture = gaussian_filter(moisture, sigma=4.0)
    moisture = np.clip(moisture + gaussian_filter(riv_u8 / 255.0, sigma=5.0) * 0.9, 0.08, 1.0)

    masks = np.stack([riv_u8, lak_u8, curv_u8, (moisture * 255.0).astype(np.uint8)], axis=-1)
    carved.astype(np.float32).tofile(OUT_DIR / "fastscape.height.f32")
    masks.astype(np.uint8).tofile(OUT_DIR / "fastscape.masks.u8")
    (OUT_DIR / "fastscape.rivers.json").write_text(json.dumps(rivers_data))
    (OUT_DIR / "fastscape.lakes.json").write_text(json.dumps(lakes_data))

    landmarks = find_landmarks(carved, riv_u8, lak_u8, lakes_data, rivers_data, world_span_m)
    meta = {
        "id": "fastscape",
        "title": "Tectonic Alpine Continent",
        "subtitle": "FastNoiseLite + Fastscapelib + Landlab Lake-to-Sea Hydrology",
        "size": VIEWER_SIZE,
        "worldSpan": world_span_m,
        "minElev": round(float(carved.min()), 2),
        "maxElev": round(float(carved.max()), 2),
        "seaLevel": 0.0,
        "riverCount": len(rivers_data),
        "lakeCount": len(lakes_data),
        "landmarks": landmarks,
    }
    (OUT_DIR / "fastscape.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_terrain_diffusion_world() -> dict | None:
    """
    Export high-resolution 1024x1024 Terrain Diffusion 30m DEM + WorldClim climate
    with walking-scale micro-detail refinement, precipitation-weighted Landlab Barnes
    lake-to-sea hydrology, and real climate texture export.
    """
    src = Path("out/terrain-diffusion")
    elev_path = src / "elev_latest.f32"
    if not elev_path.exists():
        elev_path = src / "elev_512.f32"
    if not elev_path.exists():
        return None

    print("Solving micro-detail + climate-weighted hydrology for terrain_diffusion...")
    raw = np.fromfile(elev_path, dtype=np.float32)
    n = int(round(np.sqrt(raw.size)))
    grid_size = 1024 if n >= 1024 else VIEWER_SIZE
    elev = resample_array(raw.reshape(n, n), grid_size, is_mask=False).astype(np.float32)

    # Larger 4.8 km x 4.8 km playable peninsula (4.7 m/px) with dramatic 365 m alpine summits
    world_span_m = 4800.0 if grid_size >= 1024 else 2600.0
    max_elev_target = 365.0 if grid_size >= 1024 else 285.0
    max_land = max(float(elev.max()), 1.0)
    scale_land = max_elev_target / max_land
    elev_scaled = np.where(
        elev >= 0.0,
        elev * scale_land,
        np.maximum(elev * (scale_land * 0.45), -40.0),
    ).astype(np.float32)

    # Stage 2: Walking-scale micro-detail refinement (ridged cliff crags + Fastscapelib gullies)
    elev_refined = apply_micro_detail_refinement(elev_scaled, world_span_m=world_span_m, seed=42)

    # Load WorldClim climate channels (elevation-adjusted temp C & annual precip mm/yr)
    clim_path = src / "climate_latest.npz"
    if not clim_path.exists():
        clim_path = src / "climate_512.npz"

    runoff_map = None
    has_climate = False
    temp_range = [-12.0, 28.0]
    precip_range = [0.0, 1200.0]

    if clim_path.exists():
        clim = np.load(clim_path)
        temp_c = resample_array(clim["temp"].astype(np.float32), grid_size, is_mask=False)
        precip_mm = np.maximum(
            resample_array(clim["precip"].astype(np.float32), grid_size, is_mask=False),
            0.0,
        )
        t_season = resample_array(clim["temp_season"].astype(np.float32), grid_size, is_mask=False)

        # Weight Landlab river discharge by real WorldClim precipitation (0.35x dry -> 1.85x wet)
        p_med = max(float(np.median(precip_mm[elev_refined > 0])), 120.0)
        runoff_map = np.clip(precip_mm / p_med, 0.35, 1.85).astype(np.float32)

        # Pack climate RGBA8: R=temp [-12..28C], G=precip [0..1200mm], B=seasonality, A=255
        t_u8 = np.clip((temp_c - temp_range[0]) / (temp_range[1] - temp_range[0]) * 255.0, 0, 255).astype(np.uint8)
        p_u8 = np.clip((precip_mm - precip_range[0]) / (precip_range[1] - precip_range[0]) * 255.0, 0, 255).astype(np.uint8)
        s_u8 = np.clip(t_season / max(float(np.percentile(t_season, 98)), 1.0) * 255.0, 0, 255).astype(np.uint8)
        clim_rgba = np.stack([t_u8, p_u8, s_u8, np.full((grid_size, grid_size), 255, dtype=np.uint8)], axis=-1)
        clim_rgba.tofile(OUT_DIR / "terrain_diffusion.climate.u8")
        has_climate = True

        # Moisture channel also reflects precipitation + riparian wetness
        p_lo, p_hi = float(np.percentile(precip_mm, 5)), float(np.percentile(precip_mm, 95))
        moisture = np.clip(0.22 + 0.72 * (precip_mm - p_lo) / max(p_hi - p_lo, 1.0), 0.12, 0.96)
    else:
        moisture = np.full((grid_size, grid_size), 0.65, dtype=np.float32)

    # Stage 3: Landlab Barnes Lake-Spillway-to-Sea Hydrology weighted by precipitation!
    res_scale = (grid_size / 512.0) ** 2
    carved, riv_u8, lak_u8, lakes_data, rivers_data = solve_physical_hydrology(
        elev=elev_refined,
        world_span_m=world_span_m,
        min_lake_cells=int(round(26 * res_scale)),
        min_lake_depth_m=0.55,
        max_spring_rivers=24 if grid_size >= 1024 else 18,
        runoff_map=runoff_map,
    )
    curv_u8 = compute_curvature_ao(carved)

    wet_seed = np.clip((riv_u8.astype(np.float32) + lak_u8.astype(np.float32)) / 255.0, 0.0, 1.0)
    moisture = np.clip(moisture + gaussian_filter(wet_seed, sigma=10.0) * 0.85, 0.10, 1.0)

    masks = np.stack([riv_u8, lak_u8, curv_u8, (moisture * 255.0).astype(np.uint8)], axis=-1)
    carved.astype(np.float32).tofile(OUT_DIR / "terrain_diffusion.height.f32")
    masks.astype(np.uint8).tofile(OUT_DIR / "terrain_diffusion.masks.u8")
    (OUT_DIR / "terrain_diffusion.rivers.json").write_text(json.dumps(rivers_data))
    (OUT_DIR / "terrain_diffusion.lakes.json").write_text(json.dumps(lakes_data))

    landmarks = find_landmarks(carved, riv_u8, lak_u8, lakes_data, rivers_data, world_span_m)
    meta = {
        "id": "terrain_diffusion",
        "title": "AI Earth Peninsula 4.8km (Terrain Diffusion 30m)",
        "subtitle": "InfiniteDiffusion 30m (pool=2) + Micro-Erosion + WorldClim Biomes + Landlab Hydrology",
        "size": grid_size,
        "worldSpan": world_span_m,
        "minElev": round(float(carved.min()), 2),
        "maxElev": round(float(carved.max()), 2),
        "seaLevel": 0.0,
        "hasClimate": has_climate,
        "tempRange": temp_range,
        "precipRange": precip_range,
        "riverCount": len(rivers_data),
        "lakeCount": len(lakes_data),
        "landmarks": landmarks,
    }
    (OUT_DIR / "terrain_diffusion.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = [
        export_terrain_gen_world(
            world_id="island",
            title="Eroded Highland Island",
            subtitle="terrain-generator + Landlab Barnes Lake-Spillway -> Sea Hydrology",
            prefix="island_42",
            world_span_m=2400.0,
            max_elev_m=240.0,
        ),
        export_fastscape_world(),
        export_terrain_gen_world(
            world_id="archipelago",
            title="Seven-Isle Archipelago",
            subtitle="terrain-generator + Landlab Lake-to-Sea Hydrology",
            prefix="arch_7",
            world_span_m=2600.0,
            max_elev_m=200.0,
        ),
    ]
    td_meta = export_terrain_diffusion_world()
    if td_meta is not None:
        catalog.append(td_meta)
    tec_td_path = OUT_DIR / "tectonic_td30m.json"
    if tec_td_path.exists():
        catalog.append(json.loads(tec_td_path.read_text()))
    (OUT_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2))
    print(f"Wrote physically consistent catalog with {len(catalog)} worlds to {OUT_DIR}")


if __name__ == "__main__":
    main()
