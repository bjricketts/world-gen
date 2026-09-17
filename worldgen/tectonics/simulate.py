"""Time loop of the tectonic simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..grid import SphereGrid
from ..surface.landforms import random_unit_vectors
from ..util import named_rng
from ..hydrology import WaterSetting, erode_surface
from ..surface.sealevel import ocean_fill
from .crust import Crust, Plates, elevation, typical_speed_m_myr
from .initial import initial_state
from .processes import StepContext, StepStats, age_and_erode, hotspots, interact, merge_colliding, move, \
    update_motion, uplift
from .remap import absorb_small_plates, balance_continents, initiate_subduction, remap, rift, tidy_plates


@dataclass
class Snapshot:
    """Grid fields at one moment of the simulation."""

    time_myr: float            # relative to the end of the simulation (negative = past)
    plate: np.ndarray
    continental: np.ndarray
    cont_elev: np.ndarray
    ocean_age: np.ndarray
    ocean_extra: np.ndarray
    sediment: np.ndarray
    elevation: np.ndarray | None = None   # filled in when the snapshot is converted to a surface


@dataclass
class SimulationResult:
    """Final crust on the grid, final plates, counters and optional snapshots."""

    crust: Crust
    plates: Plates
    stats: StepStats
    duration_myr: float
    start: str
    hotspots: np.ndarray        # (m, 3) fixed mantle hotspot positions
    snapshots: list[Snapshot] = field(default_factory=list)


def _snapshot(crust: Crust, time_myr: float) -> Snapshot:
    """Return the grid fields needed to draw the crust at this moment."""
    return Snapshot(time_myr=time_myr, plate=crust.plate.astype(np.int16), continental=crust.continental.copy(),
                    cont_elev=crust.cont_elev.astype(np.float32), ocean_age=crust.ocean_age.astype(np.float32),
                    ocean_extra=crust.ocean_extra.astype(np.float32), sediment=crust.sediment.astype(np.float32))


def _resize(history: np.ndarray, size: int) -> np.ndarray:
    """Return the collision history padded with zeros for newly added plates."""
    if history.shape[0] == size:
        return history
    out = np.zeros((size, size))
    out[:history.shape[0], :history.shape[1]] = history
    return out


def erode_crust(crust: Crust, grid: SphereGrid, water: WaterSetting, duration_myr: float) -> None:
    """Apply river erosion and sedimentation to crust that lies on the grid cells."""
    raw = elevation(crust, water.relief)
    fill = ocean_fill(grid, raw, water.radius_m)
    level = fill.level_for_volume(water.ocean_volume_m3)
    height = raw - level
    ocean = fill.pass_level < level
    result = erode_surface(grid, water, height, ocean, duration_myr, h.SIM_EROSION_STEP_MYR)
    change = result.height_m - height
    # Isostasy: the crust rises as material is removed and sinks under the weight of sediment.
    change = np.where(change < 0.0, change * h.EROSION_ISOSTATIC_FACTOR, change * h.SEDIMENT_ISOSTATIC_FACTOR)
    cont = crust.continental
    crust.cont_elev[cont] += change[cont]
    crust.sediment[~cont] += change[~cont]


def run(grid: SphereGrid, radius_m: float, activity: float, continental_fraction: float, relief: float,
        start: str, duration_myr: float, seed: int, snapshot_interval_myr: float | None = None,
        water: WaterSetting | None = None) -> SimulationResult:
    """Simulate plate tectonics for ``duration_myr`` and return the final state on the grid.

    With ``water`` (liquid surface water), rivers erode the land at each
    re-map; otherwise continental relief decays uniformly.

    Snapshots are taken when crust is re-mapped onto the grid, so the
    snapshot interval is rounded to a multiple of the re-map interval.
    """
    ctx = StepContext(radius_m=radius_m, edge_rad=grid.mean_spacing(), relief=relief,
                      typical_speed=typical_speed_m_myr(activity), dt=h.TECTONIC_STEP_MYR)
    crust, plates = initial_state(grid, radius_m, activity, continental_fraction, start, seed)
    stats = StepStats()
    target_count = plates.count
    continental_fraction = float(crust.continental.mean())
    history = np.zeros((len(plates.active), len(plates.active)))

    rng = named_rng(seed, "tectonics.hotspots")
    expected = h.HOTSPOTS_EARTH * max(activity, 0.05) ** 0.5 * (radius_m / c.R_EARTH) ** 2
    hotspot_positions = random_unit_vectors(rng, min(int(rng.poisson(expected)), h.HOTSPOTS_MAX))

    steps = max(int(round(duration_myr / ctx.dt)), 1)
    remap_myr = h.TECTONIC_REMAP_STEPS * ctx.dt
    if snapshot_interval_myr:
        snapshot_interval_myr = max(round(snapshot_interval_myr / remap_myr), 1) * remap_myr
    snapshots: list[Snapshot] = []
    next_snapshot = 0.0
    if snapshot_interval_myr:
        snapshots.append(_snapshot(crust, -steps * ctx.dt))
        next_snapshot = snapshot_interval_myr

    since_remap = 0
    for step in range(1, steps + 1):
        move(crust, plates, ctx.dt)
        alive_before = crust.alive.copy()
        pairs = interact(crust, plates, ctx, stats)
        uplift(crust, ctx)
        hotspots(crust, hotspot_positions, ctx)
        age_and_erode(crust, ctx, rivers=water is not None)
        update_motion(crust, plates, pairs, alive_before, ctx)
        history *= np.exp(-ctx.dt / h.COLLISION_MEMORY_MYR)
        history += pairs
        merge_colliding(crust, plates, history, target_count, stats)
        since_remap += 1

        if since_remap == h.TECTONIC_REMAP_STEPS or step == steps:
            crust = remap(crust, plates, grid, since_remap * ctx.dt, ctx)
            tidy_plates(crust, plates, grid)
            balance_continents(crust, grid, continental_fraction)
            if water is not None:
                erode_crust(crust, grid, water, since_remap * ctx.dt)
            absorb_small_plates(crust, plates, grid)
            if step < steps:
                rift(crust, plates, grid, since_remap * ctx.dt, ctx, stats, target_count, seed, step)
                initiate_subduction(crust, plates, grid, since_remap * ctx.dt, ctx, stats, target_count, seed, step)
                history = _resize(history, len(plates.active))
            since_remap = 0
            elapsed = step * ctx.dt
            if snapshot_interval_myr and (elapsed >= next_snapshot - 1e-9 or step == steps):
                snapshots.append(_snapshot(crust, elapsed - steps * ctx.dt))
                next_snapshot += snapshot_interval_myr

    return SimulationResult(crust=crust, plates=plates, stats=stats, duration_myr=steps * ctx.dt,
                            start=start, hotspots=hotspot_positions, snapshots=snapshots)
