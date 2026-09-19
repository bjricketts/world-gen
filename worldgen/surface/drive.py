"""What the integrated history tells the surface about the planet's tectonics.

History mode knows the plate creation rate and the melt production through
time, which the surface would otherwise have to guess from a single activity
index. Plate speed follows the creation rate (in the thermal model a plate's
heat flow goes as the square root of its speed, so speed goes as the square of
the flux, which is what ``spreading`` already measures), and the volcanic
vigour follows the melt.

The simulated span is set so the plates cover about as much ground as Earth's
do in ``TECTONIC_DURATION_MYR``: faster plates need less time, slower ones
more, within a range that keeps the run affordable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import heuristics as h


@dataclass(frozen=True)
class TectonicDrive:
    """Plate creation and melting over the simulated window, relative to present Earth.

    ``time_myr`` counts back from the target epoch (0 at the end, negative in
    the past); the other arrays are sampled at those times.
    """

    time_myr: np.ndarray
    spreading: np.ndarray
    melt: np.ndarray

    @classmethod
    def from_series(cls, series: dict, age_gyr: float) -> "TectonicDrive":
        """Return the drive read from a history timeline's sampled series."""
        time = (np.asarray(series["time_gyr"], dtype=float) - age_gyr) * 1e3
        spreading = np.asarray(series.get("spreading", series["melt"]), dtype=float)
        return cls(time_myr=time, spreading=spreading, melt=np.asarray(series["melt"], dtype=float))

    def at(self, time_myr: float) -> tuple[float, float]:
        """Return the spreading rate and melt production at a time before the target epoch."""
        return (float(np.interp(time_myr, self.time_myr, self.spreading)),
                float(np.interp(time_myr, self.time_myr, self.melt)))

    @property
    def spreading_now(self) -> float:
        """Return the plate creation rate at the target epoch."""
        return float(self.spreading[-1])

    @property
    def melt_now(self) -> float:
        """Return the melt production at the target epoch."""
        return float(self.melt[-1])

    def speed_m_myr(self, time_myr: float = 0.0) -> float:
        """Return the typical plate speed at a time before the target epoch."""
        return plate_speed_m_myr(self.at(time_myr)[0])

    def duration_myr(self, default: float = h.TECTONIC_DURATION_MYR) -> float:
        """Return how long to simulate so the plates travel about as far as Earth's do in ``default``."""
        lo, hi = h.TECTONIC_DURATION_RANGE_MYR
        rate = max(self.spreading_now, 1e-3)
        return float(np.clip(default / rate, lo, hi))

    def turnover_myr(self) -> float:
        """Return how long the planet takes to replace its sea floor."""
        return float(h.SEAFLOOR_TURNOVER_EARTH_MYR / max(self.spreading_now, 1e-3))


def plate_speed_m_myr(spreading: float) -> float:
    """Return the typical plate speed (m/Myr) for a plate creation rate relative to Earth's."""
    lo, hi = h.PLATE_SPEED_SPREADING_RANGE
    return h.PLATE_SPEED_EARTH_CM_YR / 100.0 * 1e6 * float(np.clip(spreading, lo, hi))
