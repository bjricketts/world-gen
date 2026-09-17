"""Tier 0 climate during the history integration, read from a lazily filled table.

The zonal model is solved once per table node, a grid in instellation,
greenhouse optical depth, surface pressure and land fraction, on a warm and
a cold (frozen) branch. Values between nodes are interpolated multilinearly,
so the right-hand side of the ODE system stays smooth and cheap. Dry planets
and steam atmospheres use the grey-atmosphere estimate instead.
"""

from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from dataclasses import dataclass

from .. import heuristics as h
from ..atmosphere import default_albedo, surface_temperature
from ..climate import COLD_START_K, WARM_START_K, climate_setting, solve_zonal
from ..orbit import equilibrium_temperature

CLIMATE_FIELDS = ("mean_k", "albedo", "open_water", "open_ocean", "ice_line_deg")
GREY_LIMIT_K = 360.0     # above this the grey estimate replaces the zonal model


@dataclass
class ClimatePoint:
    """Global climate at one instant."""

    mean_k: float
    albedo: float
    open_water: float        # share of the whole surface
    open_ocean: float        # share of the ocean
    ice_line_deg: float


@dataclass(frozen=True)
class OrbitClimate:
    """Orbit and spin properties the climate needs, fixed during a run (apart from locking)."""

    eccentricity: float
    obliquity_rad: float
    periapsis_longitude_rad: float
    rotation_period_s: float
    orbital_period_s: float
    gravity: float
    planet_class: str


class ClimateTable:
    """Warm- and cold-branch Tier 0 solutions on a lazily evaluated grid."""

    def __init__(self, orbit: OrbitClimate, land_albedo: float | None = None):
        self.orbit = orbit
        self.land_albedo = land_albedo
        self.nodes: dict[tuple, tuple[float, ...]] = {}
        self.solves = 0

    def _grey(self, s: float, tau: float, pressure_bar: float) -> ClimatePoint:
        """Return the grey-atmosphere climate without ice."""
        albedo = default_albedo(True, pressure_bar * 1e5, self.orbit.planet_class)
        t = surface_temperature(equilibrium_temperature(s, albedo), tau)
        return ClimatePoint(t, albedo, 1.0, 1.0, 90.0)

    def _node(self, key: tuple) -> tuple[float, ...]:
        """Return the solved climate at a node, solving it on first use."""
        if key in self.nodes:
            return self.nodes[key]
        i, j, k, m, synchronous, branch = key
        s = math.exp(i * h.HISTORY_TABLE_LOG_S_STEP)
        tau = math.expm1(j * h.HISTORY_TABLE_TAU_STEP)
        pressure = math.exp(k * h.HISTORY_TABLE_LOG_P_STEP)
        land = min(max(m * h.HISTORY_TABLE_LAND_STEP, 0.0), 1.0)
        grey = self._grey(s, tau, pressure)
        if grey.mean_k > GREY_LIMIT_K and branch == "warm":
            values = (grey.mean_k, grey.albedo, 1.0 - land, 1.0, 90.0)
        else:
            o = self.orbit
            albedo = default_albedo(True, pressure * 1e5, o.planet_class)
            reference = min(grey.mean_k, GREY_LIMIT_K)
            setting = climate_setting(s, o.eccentricity, o.obliquity_rad, o.periapsis_longitude_rad, synchronous,
                                      pressure * 1e5, o.gravity, "n2_co2", tau, o.rotation_period_s,
                                      o.orbital_period_s, albedo, False, True, reference)
            start = WARM_START_K if branch == "warm" else COLD_START_K
            z = solve_zonal(setting, land, start_k=start, bands=h.HISTORY_CLIMATE_BANDS,
                            land_albedo=self.land_albedo, tolerance_k=h.HISTORY_CLIMATE_TOLERANCE_K)
            self.solves += 1
            values = (z.mean_k, z.albedo, z.open_water_fraction, z.open_ocean_fraction, z.ice_line_deg)
        self.nodes[key] = values
        return values

    def lookup(self, s: float, tau: float, pressure_bar: float, land: float, synchronous: bool,
               branch: str) -> ClimatePoint:
        """Return the interpolated climate on the given branch ('warm' or 'cold')."""
        coords = (math.log(max(s, 1e-6)) / h.HISTORY_TABLE_LOG_S_STEP,
                  math.log1p(max(tau, 0.0)) / h.HISTORY_TABLE_TAU_STEP,
                  math.log(max(pressure_bar, 1e-4)) / h.HISTORY_TABLE_LOG_P_STEP,
                  min(max(land, 0.0), 1.0) / h.HISTORY_TABLE_LAND_STEP)
        base = [math.floor(x) for x in coords]
        frac = [x - b for x, b in zip(coords, base)]
        total = [0.0] * len(CLIMATE_FIELDS)
        for corner in itertools.product((0, 1), repeat=4):
            weight = 1.0
            for f, bit in zip(frac, corner):
                weight *= f if bit else 1.0 - f
            if weight <= 0.0:
                continue
            key = tuple(b + bit for b, bit in zip(base, corner)) + (synchronous, branch)
            values = self._node(key)
            for n, value in enumerate(values):
                total[n] += weight * value
        return ClimatePoint(*total)


_TABLES: "OrderedDict[tuple, ClimateTable]" = OrderedDict()


def shared_table(orbit: OrbitClimate, land_albedo: float | None = None) -> ClimateTable:
    """Return a climate table for this orbit, reusing one built by an earlier run in this process."""
    key = (orbit, None if land_albedo is None else round(land_albedo, 6))
    table = _TABLES.pop(key, None) or ClimateTable(orbit, land_albedo)
    _TABLES[key] = table
    while len(_TABLES) > h.HISTORY_TABLE_CACHE:
        _TABLES.popitem(last=False)
    return table


def grey_climate(s: float, tau: float, pressure_bar: float, present: bool, planet_class: str) -> ClimatePoint:
    """Return the climate of a planet without surface water (or with a steam atmosphere)."""
    albedo = default_albedo(present, pressure_bar * 1e5, planet_class)
    t = surface_temperature(equilibrium_temperature(s, albedo), tau)
    return ClimatePoint(t, albedo, 0.0, 0.0, 0.0)
