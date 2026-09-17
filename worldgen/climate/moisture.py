"""Tier 1 climate, moisture part: evaporation minus precipitation from moist energy transport.

The energy flux between neighbouring climate cells is split into a Hadley
part, weighted by exp(−(x − x_ITCZ)²/σ²) with x the sine of latitude, and an
eddy part (Siler et al. 2018). Eddies carry latent heat down the humidity
gradient. The Hadley cell's lower branch carries moisture against its energy
flux, at a mass flux equal to the energy flux divided by the gross moist
stability. The divergence of latent heat transport, spread over a few
cells to stand in for convective organisation, gives E − P. On
synchronously rotating planets the overturning cell is centred on the
substellar point.

Evaporation follows the saturation humidity pattern, scaled so the global
mean rises by a few per cent per kelvin (Held & Soden 2006) and never uses
more than a fixed share of the absorbed sunlight. The resulting
ocean-equivalent precipitation, E − (E − P), is what an ocean surface would
receive; land precipitation is derived from it on the surface grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constants as c
from .. import heuristics as h
from .ebm import ThermalClimate, cell_coordinate
from .insolation import harmonics, synthesise
from .mesh import climate_grid, cotangent_weights
from .physics import LATENT_HEAT_J_KG, ClimateSetting, hadley_edge_deg, hadley_width_sin, saturation_humidity

WATER_DENSITY = 1000.0


@dataclass
class MoistClimate:
    """Water fluxes on the climate grid (samples × cells, m/yr)."""

    evaporation_m: np.ndarray        # ocean-equivalent evaporation
    net_export_m: np.ndarray         # E − P from moisture transport
    precipitation_m: np.ndarray      # ocean-equivalent precipitation
    itcz_sin: np.ndarray             # (samples,) position of the rising branch (sine of latitude)
    global_evaporation_m: float


def rising_branch(setting: ClimateSetting, sea_level_k: np.ndarray) -> np.ndarray:
    """Return the sine of latitude of the warmest zonal band at each time sample, within the Hadley cells."""
    if setting.synchronous:
        return np.ones(sea_level_k.shape[0])
    z = climate_grid().points[:, 2]
    bands = h.ZONAL_BANDS
    idx = np.minimum(((z + 1.0) * 0.5 * bands).astype(int), bands - 1)
    counts = np.bincount(idx, minlength=bands)
    centres = -1.0 + (np.arange(bands) + 0.5) * 2.0 / bands
    limit = np.sin(np.radians(max(hadley_edge_deg(setting.rotation_period_s / c.SECONDS_PER_HOUR) - 5.0, 5.0)))
    out = np.zeros(sea_level_k.shape[0])
    for t, temp in enumerate(sea_level_k):
        zonal = np.bincount(idx, weights=temp, minlength=bands) / np.maximum(counts, 1)
        zonal = np.convolve(zonal, [0.25, 0.5, 0.25], mode="same")
        zonal[np.abs(centres) > limit] = -np.inf
        out[t] = centres[int(np.argmax(zonal))]
    return out


def hadley_weight(setting: ClimateSetting, edge_x: np.ndarray, rising: float) -> np.ndarray:
    """Return the Hadley share of the energy flux on each edge."""
    if setting.synchronous:
        return np.exp(-(((1.0 - edge_x) / h.SUBSTELLAR_CELL_WIDTH) ** 2))
    width = hadley_width_sin(setting.rotation_period_s / c.SECONDS_PER_HOUR)
    return np.exp(-(((edge_x - rising) / width) ** 2))


def global_evaporation(setting: ClimateSetting, thermal: ThermalClimate) -> float:
    """Return the global mean evaporation (m/yr) from the mean temperature and the absorbed sunlight."""
    if setting.humidity <= 0.0:
        return 0.0
    rate = h.EVAPORATION_EARTH_M * np.exp(h.HYDROLOGY_SENSITIVITY * (thermal.mean_k - c.T_SURFACE_EARTH))
    absorbed = float((thermal.insolation * (1.0 - thermal.albedo)).mean())
    cap = h.LATENT_SHARE_MAX * absorbed * c.SECONDS_PER_YEAR / (LATENT_HEAT_J_KG * WATER_DENSITY)
    return float(min(rate, cap))


def solve_moisture(setting: ClimateSetting, thermal: ThermalClimate, land_fraction: np.ndarray) -> MoistClimate:
    """Return evaporation, moisture divergence and ocean-equivalent precipitation on the climate grid.

    Moisture transport is computed from the temperatures without the ocean's
    heating, so it follows the total (atmospheric and oceanic) poleward
    transport the scheme was calibrated to; the Hadley cell's surface winds
    also drive the tropical ocean transport (Held 2001).
    """
    grid = climate_grid()
    weights = cotangent_weights(grid.size)
    rows = np.repeat(np.arange(grid.size), np.diff(weights.indptr))
    cols = weights.indices
    x = cell_coordinate(setting)
    edge_x = 0.5 * (x[rows] + x[cols])
    base = weights.data * thermal.conductance / setting.cp
    rh = setting.humidity
    temp = thermal.sea_level_k - thermal.ocean_warming_k
    samples = temp.shape[0]
    rising = rising_branch(setting, temp)

    # Transport of the seasonal anomalies is weaker than that of the annual mean, as in the thermal model.
    mean_t = temp.mean(axis=0)
    to_year = c.SECONDS_PER_YEAR / (LATENT_HEAT_J_KG * WATER_DENSITY)
    export = np.zeros_like(temp)
    if rh > 0.0:
        q_mean = rh * saturation_humidity(mean_t, setting.pressure_pa)
        h_mean = setting.cp * mean_t + LATENT_HEAT_J_KG * q_mean
        for t in range(samples):
            q = rh * saturation_humidity(temp[t], setting.pressure_pa)
            mse = setting.cp * temp[t] + LATENT_HEAT_J_KG * q
            mse_eff = h_mean + h.SEASONAL_TRANSPORT_FACTOR * (mse - h_mean)
            lq_eff = LATENT_HEAT_J_KG * (q_mean + h.SEASONAL_TRANSPORT_FACTOR * (q - q_mean))
            energy = base * (mse_eff[rows] - mse_eff[cols])
            w = hadley_weight(setting, edge_x, rising[t])
            top = h.GROSS_MOIST_STABILITY_FACTOR * float(np.max(mse))
            stability = np.maximum(top - 0.5 * (mse[rows] + mse[cols]), h.GROSS_MOIST_STABILITY_MIN * top)
            q_edge = 0.5 * (q[rows] + q[cols])
            latent = (-(w * energy) / stability * LATENT_HEAT_J_KG * q_edge
                      + (1.0 - w) * base * (lq_eff[rows] - lq_eff[cols]))
            export[t] = np.bincount(rows, weights=latent, minlength=grid.size) * to_year

    if h.MOISTURE_SPREAD_STEPS > 0:
        from ..surface.distance import smooth   # imported here: the surface package builds on the climate package

        export = np.stack([smooth(grid, row, h.MOISTURE_SPREAD_STEPS) for row in export])
    total = global_evaporation(setting, thermal)
    qsat = saturation_humidity(thermal.sea_level_k, setting.pressure_pa)
    ocean_share = 1.0 - np.clip(land_fraction, 0.0, 1.0)
    pattern = qsat * (1.0 - h.SEA_ICE_EVAPORATION_CUT * thermal.sea_ice)
    scale = float((pattern * ocean_share).mean() / max(ocean_share.mean(), 1e-6)) if ocean_share.any() else 1.0
    evaporation = total * pattern / max(scale, 1e-12)
    precipitation = np.maximum(evaporation - export, h.RAIN_MINIMUM_M * (total > 0))
    return MoistClimate(evaporation_m=evaporation, net_export_m=export, precipitation_m=precipitation,
                        itcz_sin=rising, global_evaporation_m=total)


def monthly(values: np.ndarray, months: int = 12) -> np.ndarray:
    """Return a periodic series (axis 0) resampled to monthly means."""
    samples = values.shape[0]
    if samples == 1:
        return np.repeat(values, months, axis=0)
    fine = synthesise(harmonics(values, min(samples // 2, 6)), months * 4)
    return fine.reshape(months, 4, *values.shape[1:]).mean(axis=1)
