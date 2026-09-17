"""Top-of-atmosphere insolation through the year.

Daily-mean insolation follows the standard spherical-astronomy expression
(Berger 1978) with the orbit's eccentricity and obliquity. The year is
sampled uniformly in time, starting at the northern winter solstice, so
sample 0 is roughly Earth's early January. Synchronously rotating planets
get a fixed pattern centred on the substellar point (longitude 0 on the
equator).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def solar_longitude(samples: int, eccentricity: float, periapsis_longitude_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the solar longitude (rad) and the distance relative to the semi-major axis at evenly spaced times.

    Time starts when the solar longitude is 270° (northern winter solstice).
    ``periapsis_longitude_rad`` is the solar longitude at periapsis (Earth ~283°).
    """
    e = float(np.clip(eccentricity, 0.0, 0.95))
    # Mean anomaly at the start, from the true anomaly of the solstice.
    nu0 = 1.5 * np.pi - periapsis_longitude_rad
    e_anom0 = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2), np.sqrt(1 + e) * np.cos(nu0 / 2))
    m0 = e_anom0 - e * np.sin(e_anom0)
    mean = m0 + 2.0 * np.pi * np.arange(samples) / samples
    ecc = mean.copy()
    for _ in range(30):
        ecc = ecc - (ecc - e * np.sin(ecc) - mean) / (1.0 - e * np.cos(ecc))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(ecc / 2), np.sqrt(1 - e) * np.cos(ecc / 2))
    distance = 1.0 - e * np.cos(ecc)
    return np.mod(nu + periapsis_longitude_rad, 2 * np.pi), distance


def seasonal_insolation(sin_lat: np.ndarray, flux_w_m2: float, obliquity_rad: float, eccentricity: float,
                        periapsis_longitude_rad: float, samples: int) -> np.ndarray:
    """Return daily-mean insolation (W/m², shape samples × points) at the given sines of latitude.

    ``flux_w_m2`` is the flux at the semi-major axis. The orbit-averaged
    global mean is flux / (4 √(1 − e²)).
    """
    lon, dist = solar_longitude(samples, eccentricity, periapsis_longitude_rad)
    dec = np.arcsin(np.sin(obliquity_rad) * np.sin(lon))[:, None]
    lat = np.arcsin(np.clip(sin_lat, -1, 1))[None, :]
    cos_h0 = np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0)
    h0 = np.arccos(cos_h0)
    daily = (h0 * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(h0)) / np.pi
    return flux_w_m2 / dist[:, None] ** 2 * np.maximum(daily, 0.0)


def substellar_insolation(cos_angle: np.ndarray, flux_w_m2: float) -> np.ndarray:
    """Return the fixed insolation (W/m²) of a synchronously rotating planet against the cosine of the substellar angle."""
    return flux_w_m2 * np.maximum(cos_angle, 0.0)


def harmonics(values: np.ndarray, count: int) -> np.ndarray:
    """Return complex Fourier coefficients k = 0 … count−1 of values sampled evenly over one period (axis 0).

    The series is reconstructed as Re Σ c_k e^{i k t} with c_0 the mean and
    c_k (k ≥ 1) twice the discrete Fourier coefficient.
    """
    n = values.shape[0]
    if n == 1:
        return values.astype(complex)
    coeffs = np.tensordot(_analysis(count, n), values, axes=(1, 0))
    return coeffs


def synthesise(coeffs: np.ndarray, samples: int) -> np.ndarray:
    """Return the real series with the given harmonics at evenly spaced times (axis 0)."""
    if samples == 1 and coeffs.shape[0] == 1:
        return coeffs.real.copy()
    return np.tensordot(_synthesis(coeffs.shape[0], samples), coeffs, axes=(1, 0)).real


@lru_cache(maxsize=16)
def _analysis(count: int, n: int) -> np.ndarray:
    """Return the matrix that maps n even samples to harmonics 0 … count−1."""
    phase = np.exp(-1j * np.outer(np.arange(count), 2 * np.pi * np.arange(n) / n)) / n
    phase[1:] *= 2.0
    return phase


@lru_cache(maxsize=16)
def _synthesis(count: int, samples: int) -> np.ndarray:
    """Return the matrix that maps harmonics 0 … count−1 to even samples."""
    return np.exp(1j * np.outer(2 * np.pi * np.arange(samples) / samples, np.arange(count)))
