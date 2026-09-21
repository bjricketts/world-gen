"""Local erosion and drainage on a zoomed region (milestone 7).

The whole requested region is eroded as one grid, so its rivers and valleys are
continuous with no internal seams. Water leaves at the sea within the region and
across the region's outer edge, whose elevation is pinned to the inherited
surface so neighbouring regions still meet there. Rivers that cross into the
region from the global network are injected at the edge and carried downstream,
so a river entering the frame arrives at its real size. The global tectonic and
climate simulations are not re-run; only the sub-grid relief is eroded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .. import heuristics as h
from ..hydrology import build_drainage, erode
from ..hydrology.drainage import _strahler
from ..hydrology.graph import accumulate
from ..hydrology.rainfall import Rainfall, fu_runoff
from .detail import DetailedRegion
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
    river_order: np.ndarray              # Strahler order, 0 where there is no river
    basin: np.ndarray
    lake: np.ndarray
    lake_level_m: np.ndarray
    endorheic: np.ndarray
    inflow_m3_s: np.ndarray              # discharge injected where global rivers cross the edge
    eroded_m: np.ndarray
    deposited_m: np.ndarray
    fabric: np.ndarray
    temperature_k: np.ndarray
    precipitation_m: Optional[np.ndarray] = None
    runoff_m: Optional[np.ndarray] = None
    categorical: Optional[dict] = None


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
    boundary = region.boundary()
    interior = np.flatnonzero(~boundary)
    outlet = detailed.ocean | boundary                       # water leaves at the sea and the outer edge

    height = detailed.elevation.astype(float).copy()
    runoff = (fu_runoff(detailed.precipitation_m, detailed.evaporation_m)
              if detailed.precipitation_m is not None and detailed.evaporation_m is not None else None)

    eroded_m = deposited_m = np.zeros(region.size)
    if runoff is not None:
        result = erode(region, radius, height, outlet, runoff, duration_myr, relief,
                       h.ZOOM_EROSION_STEP_MYR, cell_area_m2=cell_area)
        height = result.height_m.copy()
        height[boundary] = detailed.elevation[boundary]      # pin the outer edge to the inherited surface
        eroded_m, deposited_m = result.eroded_m, result.deposited_m

    receiver = np.arange(region.size)
    discharge = np.zeros(region.size)
    river_order = np.zeros(region.size, dtype=np.int8)
    basin = np.zeros(region.size, dtype=np.int32)
    lake = np.zeros(region.size, dtype=bool)
    lake_level = np.full(region.size, np.nan)
    endorheic = np.zeros(region.size, dtype=bool)
    inflow = np.zeros(region.size)

    if runoff is not None:
        rain = Rainfall(precipitation_m=detailed.precipitation_m, evaporation_m=detailed.evaporation_m,
                        runoff_m=runoff, temperature_k=detailed.temperature_k)
        drainage = build_drainage(region, radius, height, outlet, rain, cell_area_m2=cell_area)
        inflow = _global_inflow(world, region, interior)
        discharge = drainage.discharge_m3_s + accumulate(drainage.order, drainage.receiver, inflow)
        # Local streams are keyed to upstream area (catchments are small at zoom); big rivers enter by discharge.
        upstream_km2 = accumulate(drainage.order, drainage.receiver, np.full(region.size, cell_area)) / 1e6
        river = (~outlet) & ~drainage.lake & ((upstream_km2 >= h.ZOOM_RIVER_MIN_AREA_KM2)
                                              | (discharge >= h.RIVER_MIN_DISCHARGE_M3_S))
        river_order = _strahler(drainage.order, drainage.receiver, river)
        receiver, basin, lake = drainage.receiver, drainage.basin, drainage.lake
        lake_level, endorheic = drainage.lake_level_m, drainage.endorheic

    return ErodedRegion(
        grid=region, radius_m=radius, seed=detailed.seed, elevation=height, ocean=detailed.ocean,
        receiver=receiver, discharge_m3_s=discharge, river_order=river_order, basin=basin, lake=lake,
        lake_level_m=lake_level, endorheic=endorheic, inflow_m3_s=inflow, eroded_m=eroded_m,
        deposited_m=deposited_m, fabric=detailed.fabric, temperature_k=detailed.temperature_k,
        precipitation_m=detailed.precipitation_m, runoff_m=runoff, categorical=detailed.categorical)
