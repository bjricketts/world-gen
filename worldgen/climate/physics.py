"""Shared climate physics: radiation, heat capacity, transport and albedo for the energy-balance models.

Outgoing longwave radiation is the grey-atmosphere flux linearised about a
reference temperature, with a water-vapour feedback that lowers its slope
(Earth's all-sky value is ~2.1 W m⁻² K⁻¹; North et al. 1981). Heat is
transported by diffusing moist static energy (Siler et al. 2018), with a
diffusivity that scales with pressure, heat capacity, molecular weight and
rotation rate (Williams & Kasting 1997). Planetary albedo is a zonal
cloud and zenith-angle pattern (North et al. 1981) plus the surface albedo
seen through the clouds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import constants as c
from .. import heuristics as h

LATENT_HEAT_J_KG = 2.5e6
WATER_VAPOUR_GAS_CONSTANT = 461.5
FREEZING_K = 273.15
CP_AIR = {"n2_o2": 1004.0, "n2_co2": 1004.0, "co2": 850.0, "n2_ch4": 1040.0, "h2_he": 12000.0, "none": 1004.0}
EARTH_ROTATION_S = 86_164.0


def saturation_vapour_pressure(t_k: np.ndarray) -> np.ndarray:
    """Return the saturation vapour pressure of water (Pa), Clausius–Clapeyron about 273.16 K."""
    t = np.clip(t_k, 150.0, 400.0)
    return 611.657 * np.exp(LATENT_HEAT_J_KG / WATER_VAPOUR_GAS_CONSTANT * (1.0 / 273.16 - 1.0 / t))


def saturation_humidity(t_k: np.ndarray, pressure_pa: float) -> np.ndarray:
    """Return the saturation specific humidity (kg/kg), limited to 0.2 in very warm, thin air."""
    return np.minimum(0.622 * saturation_vapour_pressure(t_k) / max(pressure_pa, 1.0), 0.2)


def moist_slope(t_k: np.ndarray, pressure_pa: float, cp: float, humidity: float) -> np.ndarray:
    """Return d(h / c_p)/dT, the moist static energy change per kelvin relative to the dry value."""
    q = saturation_humidity(t_k, pressure_pa)
    dq = q * LATENT_HEAT_J_KG / (WATER_VAPOUR_GAS_CONSTANT * np.clip(t_k, 150.0, 400.0) ** 2)
    return 1.0 + LATENT_HEAT_J_KG * humidity * dq / cp


@dataclass
class ClimateSetting:
    """Planet properties the energy-balance models need."""

    flux_w_m2: float                 # stellar flux at the semi-major axis
    eccentricity: float
    obliquity_rad: float
    periapsis_longitude_rad: float
    synchronous: bool
    pressure_pa: float
    gravity_m_s2: float
    composition: str
    optical_depth: float
    rotation_period_s: float
    year_s: float                    # orbital period
    albedo: float                    # ice-free planetary albedo, or the user's global value
    albedo_fixed: bool               # the global mean albedo must equal ``albedo``
    has_water: bool                  # surface water present (liquid or frozen)
    reference_k: float               # temperature the radiation is linearised about

    @property
    def cp(self) -> float:
        """Return the specific heat of the air (J/kg/K)."""
        return CP_AIR.get(self.composition, 1004.0)

    @property
    def humidity(self) -> float:
        """Return the near-surface relative humidity used for moist transport (0 on dry planets)."""
        return h.CLIMATE_RELATIVE_HUMIDITY if self.has_water and self.pressure_pa > 611.657 else 0.0

    def outgoing(self, reference_k: float | None = None) -> tuple[float, float]:
        """Return (a, b) with outgoing longwave flux a + b·T (W/m², T in K) about the reference temperature."""
        t0 = self.reference_k if reference_k is None else reference_k
        grey = c.SIGMA_SB * t0**4 / (1.0 + 0.75 * self.optical_depth)
        b = 4.0 * grey / t0
        if self.humidity > 0.0:
            b *= water_vapour_factor(t0, self.pressure_pa)
        return grey - b * t0, b

    def air_heat_capacity(self) -> float:
        """Return the heat capacity of the atmospheric column (J/m²/K)."""
        return self.cp * self.pressure_pa / self.gravity_m_s2

    def heat_capacity(self, land_fraction: np.ndarray) -> np.ndarray:
        """Return the effective surface heat capacity (J/m²/K) of cells with the given land fraction."""
        air = self.air_heat_capacity()
        ocean = h.OCEAN_MIXED_LAYER_M * 4.0e6 + air if self.has_water else h.LAND_HEAT_CAPACITY + air
        land = h.LAND_HEAT_CAPACITY + air
        return land_fraction * land + (1.0 - land_fraction) * ocean

    def diffusivity(self) -> float:
        """Return the moist-static-energy diffusivity (W/m²/K on the unit sphere, per unit d(h/c_p))."""
        if self.pressure_pa <= 0.0:
            return 0.0
        mu = h.MEAN_MOLECULAR_WEIGHT.get(self.composition, 29.0) or 29.0
        scale = (self.pressure_pa / c.P_EARTH) * (self.cp / 1004.0) * (29.0 / mu) ** 2
        if self.synchronous:
            # Day-to-night overturning is not limited by rotation.
            scale *= h.SYNCHRONOUS_TRANSPORT_FACTOR
        else:
            omega_ratio = EARTH_ROTATION_S / max(self.rotation_period_s, 1.0)
            scale *= omega_ratio ** -h.TRANSPORT_ROTATION_EXPONENT
        _, b = self.outgoing()
        return float(min(h.MOIST_DIFFUSIVITY_EARTH * scale, h.TRANSPORT_MAX_RATIO * b))

    def cloud_masking(self) -> float:
        """Return how much of a surface albedo change shows in the planetary albedo."""
        return (1.0 - self.albedo) ** 2 if not self.albedo_fixed else (1.0 - h.ALBEDO_TEMPERATE) ** 2


def water_vapour_factor(t_k: float, pressure_pa: float) -> float:
    """Return the factor by which water vapour lowers the slope of outgoing radiation."""
    q = float(saturation_humidity(np.array(t_k), pressure_pa))
    q_earth = float(saturation_humidity(np.array(c.T_SURFACE_EARTH), c.P_EARTH))
    return max(1.0 - h.WATER_VAPOUR_FEEDBACK * min(q / q_earth, 2.0), h.WATER_VAPOUR_FACTOR_MIN)


def zonal_albedo_pattern(sin_lat: np.ndarray) -> np.ndarray:
    """Return the ice-free planetary albedo anomaly from clouds and sun angle (North et al. 1981)."""
    return h.ALBEDO_ZONAL_P2 * 0.5 * (3.0 * sin_lat**2 - 1.0)


def substellar_albedo_pattern(cos_angle: np.ndarray) -> np.ndarray:
    """Return the ice-free albedo anomaly of a synchronous planet: bright clouds over the substellar region."""
    day = np.maximum(cos_angle, 0.0)
    return h.SUBSTELLAR_CLOUD_ALBEDO * (day - 0.25)


def freeze_fraction(t_k: np.ndarray, threshold_k: float) -> np.ndarray:
    """Return the ice or snow cover (0–1) against temperature, a smooth step below ``threshold_k``."""
    return 1.0 / (1.0 + np.exp((t_k - threshold_k) / h.ICE_TRANSITION_K))


def optical_depth_for_outgoing(grey_w_m2: float, temperature_k: float) -> float:
    """Return the grey optical depth whose outgoing flux at ``temperature_k`` is ``grey_w_m2``."""
    return (c.SIGMA_SB * temperature_k**4 / grey_w_m2 - 1.0) / 0.75 if grey_w_m2 > 0 else math.inf


def outgoing_constant(optical_depth: float, reference_k: float, b: float) -> float:
    """Return the constant a of outgoing flux a + b·T for another optical depth, keeping the slope b."""
    return c.SIGMA_SB * reference_k**4 / (1.0 + 0.75 * optical_depth) - b * reference_k


def surface_albedo(land_fraction: np.ndarray, t_k: np.ndarray, has_water: bool,
                   land_albedo: np.ndarray | float, glacier: np.ndarray | float = 0.0) -> np.ndarray:
    """Return the surface albedo of cells with the given land fraction and temperature.

    Oceans freeze below ``SEA_ICE_AIR_THRESHOLD_K``; land is snow-covered
    below 0 °C when the planet has water. ``glacier`` marks land under ice
    sheets.
    """
    ocean = h.SURFACE_ALBEDO_OCEAN
    land = land_albedo
    if has_water:
        ocean = ocean + (h.SURFACE_ALBEDO_ICE - ocean) * freeze_fraction(t_k, h.SEA_ICE_AIR_THRESHOLD_K)
        snow = np.maximum(freeze_fraction(t_k, FREEZING_K) * h.SNOW_COVER_MAX, glacier)
        land = land + (h.SURFACE_ALBEDO_ICE - land) * snow
    return land_fraction * land + (1.0 - land_fraction) * ocean


def hadley_edge_deg(rotation_period_h: float) -> float:
    """Return the poleward edge of the Hadley cells: wider on slowly rotating planets."""
    return float(np.clip(h.HADLEY_EDGE_EARTH_DEG * (rotation_period_h / 24.0) ** h.HADLEY_ROTATION_EXPONENT,
                         15.0, 60.0))


def hadley_width_sin(rotation_period_h: float) -> float:
    """Return the width (in sine of latitude) of the Hadley weighting function, scaled from Earth's by the cell edge."""
    return h.HADLEY_WEIGHT_WIDTH * math.sin(math.radians(hadley_edge_deg(rotation_period_h))) / 0.5
