"""Tier 1 climate, thermal part: a two-dimensional energy-balance model on the climate grid.

Each cell mixes land and ocean heat capacities by its land fraction and
exchanges moist static energy with its neighbours. Temperatures are solved
at sea level; the lapse rate gives the actual surface temperature, which
sets outgoing radiation, snow and ice. The periodic seasonal state is found
directly as an annual mean plus harmonics (North et al. 1983), iterating on
albedo. With ``seasonal=False`` only the annual mean is solved, for use
inside the tectonic simulation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .. import heuristics as h
from .insolation import harmonics, seasonal_insolation, substellar_insolation, synthesise
from .mesh import climate_grid, cotangent_weights, edge_mean, laplacian
from .physics import (
    FREEZING_K,
    ClimateSetting,
    freeze_fraction,
    moist_slope,
    optical_depth_for_outgoing,
    outgoing_constant,
    substellar_albedo_pattern,
    surface_albedo,
    zonal_albedo_pattern,
)
from .zonal import WARM_START_K, ZonalClimate, convergence


@dataclass
class SurfaceState:
    """Surface properties of the climate cells."""

    land_fraction: np.ndarray        # 0–1
    height_m: np.ndarray             # mean land height above sea level (0 over ocean)
    land_albedo: np.ndarray          # snow-free land albedo
    glacier: np.ndarray              # share of the cell under ice sheets


@dataclass
class TemperatureControl:
    """A mean surface temperature to hold by adjusting the greenhouse, and the optical depths allowed."""

    target_k: float
    lowest_optical_depth: float
    highest_optical_depth: float


@dataclass
class ThermalClimate:
    """Temperatures and ice on the climate grid through the year (samples × cells)."""

    sea_level_k: np.ndarray          # temperature reduced to sea level
    surface_k: np.ndarray            # temperature at the mean cell height
    sea_ice: np.ndarray              # sea-ice cover of the ocean part
    albedo: np.ndarray               # planetary albedo
    insolation: np.ndarray           # W/m²
    conductance: np.ndarray          # moist diffusivity per stored edge
    mean_k: float
    planetary_albedo: float
    converged: bool
    optical_depth: float             # greenhouse optical depth the solution used (adjusted under temperature control)
    ocean_warming_k: np.ndarray | float = 0.0   # annual warming of each cell by ocean heat transport

    @property
    def samples(self) -> int:
        """Return the number of time samples through the year."""
        return self.sea_level_k.shape[0]


def cell_coordinate(setting: ClimateSetting) -> np.ndarray:
    """Return the coordinate the zonal model uses at each climate cell: sine of latitude or cosine of substellar angle."""
    points = climate_grid().points
    return points[:, 0] if setting.synchronous else points[:, 2]


def solve_thermal(setting: ClimateSetting, surface: SurfaceState, reference: ZonalClimate,
                  seasonal: bool = True, initial: Optional[ThermalClimate] = None,
                  control: Optional[TemperatureControl] = None) -> ThermalClimate:
    """Return the periodic temperature field of the climate grid.

    ``reference`` is the Tier 0 solution for the same planet; its annual
    temperature profile sets the moist diffusivity and, unless a nearby
    ``initial`` solution is given, the starting state. With ``control`` the
    greenhouse is adjusted, within its limits, to hold the mean surface
    temperature at the target.
    """
    grid = climate_grid()
    n = grid.size
    x = cell_coordinate(setting)
    weights = cotangent_weights(n)
    f_land = np.clip(surface.land_fraction, 0.0, 1.0) if setting.has_water else np.ones(n)
    f_ocean = 1.0 - f_land

    if setting.synchronous:
        insolation = substellar_insolation(x, setting.flux_w_m2)[None, :]
        pattern = substellar_albedo_pattern(x)
        count = 1
    else:
        samples = h.CLIMATE_TIME_SAMPLES
        insolation = seasonal_insolation(x, setting.flux_w_m2, setting.obliquity_rad, setting.eccentricity,
                                         setting.periapsis_longitude_rad, samples)
        pattern = zonal_albedo_pattern(x)
        count = h.CLIMATE_HARMONICS
        if not seasonal:
            insolation = insolation.mean(axis=0, keepdims=True)
            count = 1
    samples = insolation.shape[0]
    q_mean = float(insolation.mean())

    start = _reference_profile(reference, x)
    lapse = h.LAPSE_RATE_K_PER_M * np.maximum(surface.height_m, 0.0)
    ocean_heating = _ocean_heating(reference, x, f_ocean)
    conductance = setting.diffusivity() * moist_slope(edge_mean(weights, start), setting.pressure_pa, setting.cp,
                                                     setting.humidity)
    operator = laplacian(weights, conductance)
    heat = setting.heat_capacity(f_land)
    omega = 2.0 * np.pi / setting.year_s
    mask = setting.cloud_masking()

    t_ref = reference.mean_k
    a, b = setting.outgoing(t_ref)
    if control is not None:
        a_low = outgoing_constant(control.highest_optical_depth, t_ref, b)
        a_high = outgoing_constant(control.lowest_optical_depth, t_ref, b)
        if initial is not None:
            a = min(max(outgoing_constant(initial.optical_depth, t_ref, b), a_low), a_high)
    factors = _factors(operator, heat, b, omega, count)

    if initial is not None and initial.sea_level_k.shape == (samples, n):
        temp = initial.sea_level_k.copy()
    else:
        temp = np.tile(start, (samples, 1))
    offset = 0.0
    converged = False
    for it in range(h.CLIMATE_MAX_ITERATIONS):
        actual = temp - lapse
        albedo = np.clip(setting.albedo + offset + pattern
                         + mask * (surface_albedo(f_land, actual, setting.has_water, surface.land_albedo,
                                                  surface.glacier) - h.SURFACE_ALBEDO_OCEAN), 0.0, 0.95)
        if setting.albedo_fixed and q_mean > 0:
            offset += setting.albedo - float((insolation * albedo).mean()) / q_mean
        forcing = insolation * (1.0 - albedo) - a + b * lapse + ocean_heating
        coeffs = harmonics(forcing, count)
        solved = np.stack([factors[k].solve(coeffs[k] if k else coeffs[k].real) for k in range(count)])
        new = synthesise(solved.astype(complex), samples)
        change = float(np.abs(new - temp).max())
        if control is not None:
            a = min(max(a - b * (control.target_k - float((new - lapse).mean())), a_low), a_high)
        relax = h.CLIMATE_RELAXATION if it < 40 else 0.5 * h.CLIMATE_RELAXATION
        temp = temp + relax * (new - temp)
        if change < h.SURFACE_CLIMATE_TOLERANCE_K:
            converged = True
            break

    actual = temp - lapse
    sea_ice = freeze_fraction(temp, h.SEA_ICE_AIR_THRESHOLD_K) if setting.has_water else np.zeros_like(temp)
    planetary = float((insolation * albedo).mean() / q_mean) if q_mean > 0 else float(albedo.mean())
    return ThermalClimate(sea_level_k=temp, surface_k=actual, sea_ice=sea_ice * (f_ocean > 0), albedo=albedo,
                          insolation=insolation, conductance=conductance, mean_k=float(actual.mean()),
                          planetary_albedo=planetary, converged=converged,
                          optical_depth=optical_depth_for_outgoing(a + b * t_ref, t_ref),
                          ocean_warming_k=factors[0].solve(ocean_heating) if ocean_heating.any() else 0.0)


_FACTOR_CACHE: dict = {}


def _factors(operator, heat: np.ndarray, b: float, omega: float, count: int) -> list:
    """Return LU factors of the harmonic systems, reusing those of an identical recent system."""
    key = (hashlib.blake2b(operator.data.tobytes() + heat.tobytes(), digest_size=16).hexdigest(), b, omega, count)
    if key not in _FACTOR_CACHE:
        n = heat.size
        # Seasonal anomalies mix less efficiently than the annual mean (heuristic).
        _FACTOR_CACHE.clear()
        _FACTOR_CACHE[key] = [
            splu((diags(1j * k * omega * heat + b) - h.SEASONAL_TRANSPORT_FACTOR * operator).tocsc()) if k else
            splu((diags(np.full(n, b)) - operator).tocsc()) for k in range(count)]
    return _FACTOR_CACHE[key]


def _ocean_heating(reference: ZonalClimate, x: np.ndarray, f_ocean: np.ndarray) -> np.ndarray:
    """Return the annual heating (W/m²) of each cell by the Tier 0 ocean heat transport.

    Each band's heating is shared among its cells in proportion to their
    ocean fraction, so the band total is kept.
    """
    flux = reference.ocean_transport
    if flux is None or not np.any(flux):
        return np.zeros(x.size)
    n = reference.coordinate.size
    band = np.clip(((x + 1.0) * 0.5 * n).astype(int), 0, n - 1)
    share = np.bincount(band, weights=f_ocean, minlength=n) / np.maximum(np.bincount(band, minlength=n), 1)
    per_ocean = np.where(share > 0, convergence(flux, n) / np.maximum(share, 1e-9), 0.0)
    return per_ocean[band] * f_ocean


def _reference_profile(reference: ZonalClimate, x: np.ndarray) -> np.ndarray:
    """Return the Tier 0 annual temperature at each climate cell's coordinate."""
    f = reference.land_fraction
    annual = (1.0 - f) * reference.ocean_k.mean(axis=0) + f * reference.land_k.mean(axis=0)
    return np.interp(x, reference.coordinate, annual)


def snow_cover(surface_k: np.ndarray) -> np.ndarray:
    """Return the snow cover of land (0–1) against surface temperature."""
    return freeze_fraction(surface_k, FREEZING_K)


__all__ = ["SurfaceState", "TemperatureControl", "ThermalClimate", "solve_thermal", "snow_cover", "WARM_START_K"]
