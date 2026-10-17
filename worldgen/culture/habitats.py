"""Habitats of colonists: what each cell needs to be lived in, and whether a space elevator can stand.

A cell is open when its air is breathable, its pressure and temperature are
comfortable and the air above shields from cosmic rays (Hassler et al. 2014);
domed when shielded but lacking one of the others; underground when the air
above is too thin to shield; and not habitable when its annual mean
temperature is outside the survivable band. A space elevator needs the
synchronous orbit inside the Hill sphere and a taper ratio a strong material
can meet (Pearson 1975).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..surface.fields import Terrain


class Habitat(IntEnum):
    """Habitat class of a cell for colonists."""

    NONE = 0
    OPEN = 1
    DOME = 2
    UNDERGROUND = 3


def air_column_g_cm2(pressure_pa: np.ndarray, gravity_m_s2: float) -> np.ndarray:
    """Return the mass of air above each point (g/cm²) from its pressure."""
    return pressure_pa / gravity_m_s2 / 10.0


def classify_habitats(state, elevation_m: np.ndarray, temperature_k: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Return the habitat class of points given their elevation, annual mean temperature and whether they are land."""
    atm, g = state.atmosphere, state.bulk.surface_gravity_m_s2
    z = np.maximum(np.asarray(elevation_m, dtype=float), 0.0)
    pressure = atm.surface_pressure_pa * np.exp(-z / atm.scale_height_m) if atm.present else np.zeros(z.size)
    oxygen = state.biosphere.oxygen_fraction if state.biosphere is not None else 0.0
    o2_bar = oxygen * pressure / c.BAR
    lo, hi = h.OXYGEN_BREATHABLE_BAR
    p_lo, p_hi = h.HABITAT_OPEN_PRESSURE_BAR
    t_lo, t_hi = h.HABITAT_OPEN_TEMPERATURE_K
    s_lo, s_hi = h.HABITAT_SURVIVABLE_TEMPERATURE_K
    shielded = air_column_g_cm2(pressure, g) >= h.RADIATION_SHIELD_G_CM2
    comfortable = ((o2_bar >= lo) & (o2_bar <= hi) & (pressure >= p_lo * c.BAR) & (pressure <= p_hi * c.BAR)
                   & (temperature_k >= t_lo) & (temperature_k <= t_hi))
    livable = land & (temperature_k >= s_lo) & (temperature_k <= s_hi)
    out = np.full(z.size, Habitat.NONE, dtype=np.int8)
    out[livable & ~shielded] = Habitat.UNDERGROUND
    out[livable & shielded] = Habitat.DOME
    out[livable & shielded & comfortable] = Habitat.OPEN
    return out


def habitat_classes(world) -> np.ndarray:
    """Return the habitat class of every cell of a world's surface (NONE at sea)."""
    s = world.surface
    return classify_habitats(world.state, s["elevation"].values, s["air_temperature"].values.astype(float),
                             ~s["ocean"].values.astype(bool))


def habitat_capacity(world, classes: np.ndarray) -> np.ndarray:
    """Return the share of a cell's capacity a sealed habitat keeps (0 where none), with the lava-tube bonus."""
    share = np.zeros(classes.size)
    for name, value in h.HABITAT_CAPACITY.items():
        share[classes == Habitat[name.upper()]] = value
    g = world.state.bulk.surface_gravity_m_s2 / c.G_EARTH
    if g <= h.LAVA_TUBE_MAX_GRAVITY_EARTH and "terrain" in world.surface:
        volcanic = np.isin(world.surface["terrain"].values, [Terrain.VOLCANO, Terrain.VOLCANIC_PLAIN])
        share[volcanic & (classes == Habitat.UNDERGROUND)] *= h.LAVA_TUBE_BONUS
    return share


def elevator_taper(state) -> float:
    """Return the taper ratio of a balanced space elevator (inf where the synchronous orbit is outside the Hill sphere)."""
    orbit, bulk = state.orbit, state.bulk
    gm = c.G * bulk.mass_kg
    r0 = bulk.radius_m
    rs = (gm * orbit.rotation_period_s**2 / (4 * np.pi**2)) ** (1 / 3)
    if not np.isfinite(rs) or rs >= orbit.hill_radius_m or rs <= r0:
        return float("inf")
    height = h.ELEVATOR_CHARACTERISTIC_HEIGHT_KM * 1e3 * c.G_EARTH / bulk.surface_gravity_m_s2
    return float(np.exp(r0 / height * (1 + r0**3 / (2 * rs**3) - 3 * r0 / (2 * rs))))


def elevator_feasible(state) -> bool:
    """Return whether a space elevator can stand on the planet."""
    return elevator_taper(state) <= h.ELEVATOR_MAX_TAPER
