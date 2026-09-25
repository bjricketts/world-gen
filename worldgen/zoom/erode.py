"""Local erosion and drainage on a zoomed region (milestone 7).

The whole requested region is eroded as one grid, so its rivers and valleys are
continuous with no internal seams. Water leaves at the sea within the region and
across the region's outer edge, whose elevation is pinned to the inherited
surface so neighbouring regions still meet there. Rivers that cross into the
region from the global network are injected at the edge and carried downstream,
so a river entering the frame arrives at its real size. The global tectonic and
climate simulations are not re-run; only the sub-grid relief is eroded.

Rivers work first. Glaciers then work the ground at the snowline of the coldest
epoch the planet remembers (the present one without a history record), cutting
U-shaped valleys and basins that hold lakes once the ice is gone. The present
ice is laid on last: no river runs or cuts under it, and its meltwater leaves
at its margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .. import heuristics as h
from ..hydrology import build_drainage, erode
from ..hydrology.drainage import _strahler
from ..hydrology.graph import accumulate, priority_breach
from ..hydrology.rainfall import Rainfall, fu_runoff
from .detail import DetailedRegion
from .glacier import glacial_cooling_k, glaciate, ice_cover
from .tiles import RegionGrid


@dataclass
class ErodedRegion:
    """A zoomed region after local erosion, with its drainage network."""

    grid: RegionGrid
    radius_m: float
    seed: int
    elevation: np.ndarray                # eroded detailed elevation, sea level = 0
    ocean: np.ndarray
    receiver: np.ndarray                 # cell each cell drains to
    discharge_m3_s: np.ndarray           # local runoff plus rivers entering from the global network
    river_order: np.ndarray              # Strahler order of the channel network, 0 off it
    basin: np.ndarray
    lake: np.ndarray
    lake_level_m: np.ndarray
    endorheic: np.ndarray
    drainage_area_km2: np.ndarray        # upstream catchment area of each cell
    inflow_m3_s: np.ndarray              # discharge injected where global rivers cross the edge
    eroded_m: np.ndarray
    deposited_m: np.ndarray
    fabric: np.ndarray
    temperature_k: np.ndarray
    precipitation_m: Optional[np.ndarray] = None
    runoff_m: Optional[np.ndarray] = None
    categorical: Optional[dict] = None
    exported_m3: float = 0.0             # sediment carried out across the region's edge
    perennial: Optional[np.ndarray] = None   # channels with enough flow to carry water all year; the rest are dry
    channel_head_km2: Optional[np.ndarray] = None   # catchment each node needs to start a channel
    ice_thickness_m: Optional[np.ndarray] = None   # present ice above the bed (glaciers and ice sheets)
    glacial_m: Optional[np.ndarray] = None   # depth cut by glaciers
    sea_ice: Optional[np.ndarray] = None


def _edge_distance(region: RegionGrid) -> np.ndarray:
    """Return each node's distance (in lattice steps) from the region's outer edge."""
    rows, cols = region.shape
    r, c = np.divmod(np.arange(region.size), cols)
    return np.minimum.reduce([r, rows - 1 - r, c, cols - 1 - c]).astype(float)


def _open_edges(region: RegionGrid, height: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Return heights for water routing, with edge nodes dropped below any lower inside neighbour.

    The pinned edge can stand above the eroded ground just inside it; the land
    continues beyond the frame, so water there should leave rather than pond
    against the edge.
    """
    adj = region.neighbours
    rows = np.repeat(np.arange(region.size), np.diff(adj.indptr))
    cols = adj.indices
    inside = ~edge[cols]
    lowest = np.full(region.size, np.inf)
    np.minimum.at(lowest, rows[inside], height[cols[inside]])
    out = height.copy()
    out[edge] = np.minimum(height[edge], lowest[edge] - 1e-3)
    return out


def channel_heads_km2(region: RegionGrid, height: np.ndarray, radius_m: float) -> np.ndarray:
    """Return the catchment (km²) each node needs to start a channel, from the ground slope (A·S² = const)."""
    rows, cols = region.shape
    step = region.mean_spacing(radius_m)
    d_row, d_col = np.gradient(height.reshape(rows, cols), step)
    slope = np.hypot(d_row, d_col).ravel()
    area = h.ZOOM_RIVER_MIN_AREA_KM2 * (h.ZOOM_CHANNEL_HEAD_REF_SLOPE / np.maximum(slope, 1e-9)) ** 2
    smallest = h.ZOOM_CHANNEL_HEAD_MIN_CELLS * region.cell_area_m2(radius_m) / 1e6
    return np.clip(area, smallest, h.ZOOM_CHANNEL_HEAD_MAX_KM2)


def _global_inflow(world, region: RegionGrid, interior: np.ndarray) -> np.ndarray:
    """Return per-node discharge (m³/s) where global rivers flow into the region."""
    surface = world.surface
    inflow = np.zeros(region.size)
    if "discharge" not in surface or "flow_to" not in surface:
        return inflow
    discharge = surface["discharge"].values
    flow_to = surface["flow_to"].values.astype(np.int64)
    order = surface["river_order"].values if "river_order" in surface else discharge
    inside = region.contains(world.grid.points)
    # A river cell outside the region whose water flows to a cell inside it enters across the edge.
    entering = (np.asarray(order) > 0) & (~inside) & inside[flow_to]
    cells = np.flatnonzero(entering)
    if cells.size == 0:
        return inflow
    _, nearest = cKDTree(region.points[interior]).query(world.grid.points[cells])
    np.add.at(inflow, interior[nearest], discharge[cells])
    return inflow


def erode_region(world, detailed: DetailedRegion, duration_myr: float = h.ZOOM_EROSION_MYR) -> ErodedRegion:
    """Return the zoomed region after erosion, with rivers continued from the global network."""
    region = detailed.grid
    radius = detailed.radius_m
    relief = float(world.surface.attrs.get("relief_factor", 1.0))
    cell_area = region.cell_area_m2(radius)
    gravity = float(world.state.bulk.surface_gravity_m_s2)
    boundary = region.boundary()
    interior = np.flatnonzero(~boundary)
    outlet = detailed.ocean | boundary                       # water leaves at the sea and the outer edge

    height = detailed.elevation.astype(float).copy()
    runoff = (fu_runoff(detailed.precipitation_m, detailed.evaporation_m)
              if detailed.precipitation_m is not None and detailed.evaporation_m is not None else None)

    eroded_m = deposited_m = glacial_m = np.zeros(region.size)
    exported = 0.0
    sheet = detailed.ice_sheet
    fade = np.clip(_edge_distance(region) / h.ZOOM_EDGE_BLEND_CELLS, 0.0, 1.0)
    if runoff is not None:
        adj = region.neighbours
        # No river forms under an ice sheet.
        river_runoff = runoff if sheet is None else runoff * (1.0 - sheet)
        # Cut outlets through the hollows the detail noise leaves, except where the cut would be deeper
        # than a river plausibly incises (those stay lakes); then let water leave across the frame edge.
        breached = priority_breach(height, adj.indptr, adj.indices, outlet, h.ZOOM_BREACH_MAX_M, 1e-3)
        flow = _open_edges(region, breached, boundary)
        result = erode(region, radius, flow, detailed.ocean, river_runoff, duration_myr, relief,
                       h.ZOOM_EROSION_STEP_MYR, cell_area_m2=cell_area, creep_m2_per_myr=h.ZOOM_CREEP_M2_PER_MYR,
                       edge=boundary)
        # The outer edge stays pinned to the inherited surface, and erosion fades in over a few cells
        # inside it, so the frame neither steps nor disagrees with a neighbouring region.
        height = breached + fade * (result.height_m - flow)
        # Creep partly refills the narrow outlets the breach cut, rebuilding low sills; cut them again.
        height = priority_breach(height, adj.indptr, adj.indices, outlet, h.ZOOM_BREACH_MAX_M, 1e-3)
        eroded_m, deposited_m, exported = result.eroded_m, result.deposited_m, result.exported_m3

    snowline = detailed.snowline_m
    if snowline is not None and detailed.precipitation_m is not None:
        # Glaciers of the coldest remembered epoch, whose snowline stood lower by that cooling.
        lowered = snowline - glacial_cooling_k(world) / h.LAPSE_RATE_K_PER_M
        carved, _ = glaciate(region, radius, height, lowered, detailed.precipitation_m, detailed.ocean, boundary,
                             cell_area, sheet=sheet, sheet_top_m=detailed.ice_top_m, gravity_m_s2=gravity)
        glacial_m = fade * (height - carved)
        height = height - glacial_m
        eroded_m = eroded_m + glacial_m
        exported += float(glacial_m.sum() * cell_area)       # glacial sediment leaves; moraines are not built

    ice = ice_cover(region, radius, height, snowline if snowline is not None else np.full(region.size, np.inf),
                    detailed.precipitation_m, detailed.ocean, boundary, cell_area, sheet, detailed.ice_top_m, gravity)
    covered = ice.covered & ~detailed.ocean

    receiver = np.arange(region.size)
    discharge = np.zeros(region.size)
    river_order = np.zeros(region.size, dtype=np.int8)
    basin = np.zeros(region.size, dtype=np.int32)
    lake = np.zeros(region.size, dtype=bool)
    lake_level = np.full(region.size, np.nan)
    endorheic = np.zeros(region.size, dtype=bool)
    drainage_area = np.zeros(region.size)
    inflow = np.zeros(region.size)
    perennial = np.zeros(region.size, dtype=bool)
    heads = channel_heads_km2(region, height, radius)

    if runoff is not None:
        rain = Rainfall(precipitation_m=detailed.precipitation_m, evaporation_m=detailed.evaporation_m,
                        runoff_m=runoff, temperature_k=detailed.temperature_k)
        drainage = build_drainage(region, radius, _open_edges(region, height, boundary), outlet, rain,
                                  cell_area_m2=cell_area)
        inflow = _global_inflow(world, region, interior)
        discharge = drainage.discharge_m3_s + accumulate(drainage.order, drainage.receiver, inflow)
        # A channel starts where the catchment reaches the slope's channel-head area and continues downstream;
        # whether it carries water all year is keyed to its flow, so the same channel pattern is wet in a
        # humid climate and dry in a desert.
        drainage_area = accumulate(drainage.order, drainage.receiver, np.full(region.size, cell_area)) / 1e6
        # Ice that does not fill a lake's basin above the water leaves the lake a lake.
        depth = np.nan_to_num(drainage.lake_level_m - height, nan=0.0)
        covered &= ~(drainage.lake & (ice.thickness_m <= depth))
        open_land = (~outlet) & ~drainage.lake & ~covered          # no channel is mapped on ice
        perennial = open_land & (discharge >= h.ZOOM_STREAM_MIN_DISCHARGE_M3_S)
        started = (drainage_area >= heads).astype(float)
        channel = open_land & (accumulate(drainage.order, drainage.receiver, started) > 0.0)
        river_order = _strahler(drainage.order, drainage.receiver, channel | perennial)
        receiver, basin, lake = drainage.receiver, drainage.basin, drainage.lake & ~covered
        lake_level, endorheic = drainage.lake_level_m, drainage.endorheic

    return ErodedRegion(
        grid=region, radius_m=radius, seed=detailed.seed, elevation=height, ocean=detailed.ocean,
        receiver=receiver, discharge_m3_s=discharge, river_order=river_order, basin=basin, lake=lake,
        lake_level_m=lake_level, endorheic=endorheic, drainage_area_km2=drainage_area, inflow_m3_s=inflow,
        eroded_m=eroded_m, deposited_m=deposited_m, fabric=detailed.fabric, temperature_k=detailed.temperature_k,
        precipitation_m=detailed.precipitation_m, runoff_m=runoff, categorical=detailed.categorical,
        exported_m3=exported, perennial=perennial, channel_head_km2=heads,
        ice_thickness_m=np.where(covered, ice.thickness_m, 0.0),
        glacial_m=glacial_m, sea_ice=detailed.sea_ice)
