"""Starting configurations for the tectonic simulation."""

from __future__ import annotations

import numpy as np

from .. import heuristics as h
from ..grid import SphereGrid
from ..noise import fbm
from ..surface import plates as heuristic_plates
from ..surface.distance import distance_from
from ..surface.fields import Boundary, SurfaceFields
from ..surface.landforms import random_unit_vectors
from ..util import named_rng
from .crust import Crust, Plates, typical_speed_m_myr

STARTS = ("supercontinent", "cratons")
INITIAL_MAX_OCEAN_AGE_MYR = 180.0


def continent_mask(grid: SphereGrid, start: str, area_fraction: float, seed: int) -> np.ndarray:
    """Return the initial continental crust: one supercontinent or several scattered cratons."""
    rng = named_rng(seed, "tectonics.start")
    pts = grid.points
    detail = fbm(pts, seed, "tectonics.continents", frequency=1.5, octaves=5)
    if start == "supercontinent":
        centre = random_unit_vectors(rng, 1)[0]
        potential = pts @ centre + 1.6 * detail
    elif start == "cratons":
        lo, hi = h.CRATON_COUNT
        centres = random_unit_vectors(rng, int(rng.integers(lo, hi + 1)))
        potential = (pts @ centres.T).max(axis=1) + 1.1 * detail
    else:
        raise ValueError(f"unknown tectonic start '{start}'; choose from {', '.join(STARTS)}")
    return potential >= np.quantile(potential, 1.0 - area_fraction)


def initial_state(grid: SphereGrid, radius_m: float, activity: float, continental_fraction: float,
                  start: str, seed: int) -> tuple[Crust, Plates]:
    """Return the crust and plates at the start of the simulation."""
    fields = SurfaceFields(grid=grid, radius_m=radius_m)
    layout = heuristic_plates.make_plates(fields, activity, seed)
    speed = typical_speed_m_myr(activity)
    omega = layout.euler_poles * (speed / radius_m * named_rng(seed, "tectonics.speeds")
                                  .uniform(0.3, 1.0, size=layout.count))[:, None]

    crust = Crust.empty(grid.points)
    crust.plate[:] = layout.cell_plate
    crust.continental[:] = continent_mask(grid, start, continental_fraction, seed)

    # Oceanic crust starts with ages set by distance from the initial spreading ridges.
    velocity = heuristic_plates.cell_velocity(fields, layout)
    i, j, kind, _ = heuristic_plates.classify_boundaries(fields, layout, velocity)
    ridges = np.concatenate([i[kind == Boundary.DIVERGENT], j[kind == Boundary.DIVERGENT]])
    dist_km, _ = distance_from(grid, ridges, radius_m / 1e3)
    half_rate_km_myr = speed / 2e3
    crust.ocean_age[:] = np.minimum(np.nan_to_num(dist_km / half_rate_km_myr, posinf=INITIAL_MAX_OCEAN_AGE_MYR),
                                    INITIAL_MAX_OCEAN_AGE_MYR)

    variation = fbm(grid.points, seed, "tectonics.continent.relief", frequency=2.0, octaves=4)
    crust.cont_elev[:] = h.CONTINENT_BASE_M + h.CONTINENT_INITIAL_RELIEF_M * variation
    plates = Plates(omega=omega, active=np.ones(layout.count, dtype=bool))
    return crust, plates
