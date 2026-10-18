"""Turn a culture spec into resolved cultures: ranges drawn, development filled, defaults applied.

Values the spec leaves open that depend on the planet (population from the
carrying capacity, polity counts from the land area) stay ``None`` here and are
derived when the civilisation is placed.
"""

from __future__ import annotations

import math
from typing import Optional

from .. import heuristics as h
from ..util import named_rng
from .development import resolve_development
from .model import Culture
from .spec import CultureEntry, CultureSpec

# Fields whose [low, high] ranges are drawn log-uniformly.
LOG_SCALE = frozenset({"population", "largest_population", "founded_years_ago"})


def culture_seed(spec: CultureSpec, planet_seed: int) -> int:
    """Return the seed of the culture layer: the spec's own seed, else the planet seed."""
    return planet_seed if spec.seed is None else spec.seed


def _draw(value, seed: int, name: str, log: bool) -> Optional[float]:
    """Return a fixed value as given, or a draw from a [low, high] range on its own random stream."""
    if not isinstance(value, tuple):
        return None if value is None else float(value)
    lo, hi = value
    rng = named_rng(seed, name)
    if log and lo > 0:
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def resolve_culture(entry: CultureEntry, index: int, seed: int) -> Culture:
    """Return one resolved culture from its spec entry."""
    key = f"culture.{index}"

    def draw(value, name):
        return _draw(value, seed, f"{key}.{name}", name in LOG_SCALE)

    default = h.CULTURE_DEFAULT_DEVELOPMENT[entry.inhabitants]
    development = resolve_development(entry.development.model_dump(), default)
    organic = entry.mode == "organic"
    if organic and entry.inhabitants == "native":
        development = resolve_development(dict(h.HISTORY_START_DEVELOPMENT), default)
    count = draw(entry.polities.count, "polity_count")
    colony = entry.colony
    return Culture(
        id=index,
        name=entry.name,
        inhabitants=entry.inhabitants,
        development=development,
        population=None if organic else draw(entry.population, "population"),
        homeland=entry.homeland.model_dump() if entry.homeland else None,
        zipf_exponent=None if organic else draw(entry.settlements.zipf_exponent, "zipf_exponent"),
        largest_population=None if organic else draw(entry.settlements.largest_population, "largest_population"),
        polity_count=None if count is None or organic else max(1, len(entry.polities.capitals), int(round(count))),
        polity_type=None if organic else entry.polities.type,
        capitals=[] if organic else [p.model_dump(exclude_none=True) for p in entry.polities.capitals],
        territories=[] if organic else [[list(v) for v in polygon] for polygon in entry.polities.territories],
        founded_years_ago=draw(colony.founded_years_ago, "founded_years_ago") if colony else None,
        origin=colony.origin if colony else None,
        landing_sites=[p.model_dump(exclude_none=True) for p in colony.landing_sites] if colony else [],
        immigrants_per_year=colony.immigrants_per_year if colony else 0.0,
        immigration_years=colony.immigration_years if colony else None,
        independence_years=colony.independence_years if colony else None,
        crowd_diseases=bool(colony.crowd_diseases) if colony else False,
        mode=entry.mode,
    )


def resolve_cultures(spec: CultureSpec, planet_seed: int) -> list[Culture]:
    """Return the resolved cultures of a culture spec, in spec order."""
    seed = culture_seed(spec, planet_seed)
    return [resolve_culture(entry, i, seed) for i, entry in enumerate(spec.cultures)]
