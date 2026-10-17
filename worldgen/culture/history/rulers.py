"""Rulers, successions and provinces of the states in an organic history.

- *Rulers.* Each state has a ruler whose reign ends at random with a mean
  of RULER_REIGN_YEARS, or RULER_REIGN_PRIMOGENITURE_YEARS once the state has
  adopted primogeniture (Kokkonen & Sundell 2012: 11.4 and 20.9 years in
  European monarchies 1000–1800).
- *Primogeniture* is adopted at PRIMOGENITURE_ADOPT_RATE per year once the
  capital has writing (a heuristic rate).
- *Provinces* are the factions of a state: its land split around seeds spread
  as far apart as possible from the capital outward, one per PROVINCE_KM2.
"""

from __future__ import annotations

import numpy as np

from ... import heuristics as h


def reign_years(primogeniture: bool) -> float:
    """Return the mean reign under a state's succession rule."""
    return h.RULER_REIGN_PRIMOGENITURE_YEARS if primogeniture else h.RULER_REIGN_YEARS


def succession_times(rng: np.random.Generator, mean_reign: float, years: float) -> np.ndarray:
    """Return the times (years into a step) at which rulers die or are deposed, for exponential reigns."""
    times, t = [], 0.0
    while True:
        t += rng.exponential(mean_reign)
        if t > years:
            return np.array(times)
        times.append(t)


def province_count(cells: int, cell_area_km2: float) -> int:
    """Return the number of provinces of a state of ``cells`` cells (at least one, at most one per cell)."""
    return int(min(cells, max(1, round(cells * cell_area_km2 / h.PROVINCE_KM2))))


def provinces(points: np.ndarray, cells: np.ndarray, capital: int, count: int) -> np.ndarray:
    """Return the province (0 = the capital's) of each of a state's cells.

    Seeds are chosen by farthest-point sampling from the capital; each cell
    joins its nearest seed.
    """
    pts = points[cells]
    seeds = [int(np.argmax(pts @ points[capital]))]          # the capital, or the cell nearest to it
    far = 1.0 - pts @ pts[seeds[0]]
    for _ in range(count - 1):
        s = int(np.argmax(far))
        seeds.append(s)
        far = np.minimum(far, 1.0 - pts @ pts[s])
    return np.argmax(pts @ pts[seeds].T, axis=1)
