"""The culture section of a spec: which peoples inhabit the planet, how developed they are and how they are organised.

The section can sit in the planet spec (``culture:``) or in its own YAML file
applied to a saved world. Numeric fields follow the spec convention: a number
(fixed), a ``[low, high]`` range (drawn) or omitted (drawn or derived).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import warnings

import yaml
from pydantic import Field, field_validator, model_validator

from .. import heuristics as h
from ..spec_base import Param, Section
from .development import CommunicationLevel, EnergyLevel, TransportLevel

Inhabitants = Literal["native", "colonist"]
CultureMode = Literal["scoped", "organic"]
PolityType = Literal["band", "tribe", "chiefdom", "city_state", "kingdom", "empire", "federation", "republic",
                     "corporate", "colony"]


class Place(Section):
    """A point on the planet, with an optional name."""

    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=360)
    name: Optional[str] = None


class Homeland(Section):
    """The region a culture occupies; without one the culture takes the land best suited to it."""

    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=360)
    radius_km: float = Field(gt=0)


class DevelopmentSpec(Section):
    """Level on each development axis. Axes left out take the mean level of those given."""

    energy: Optional[EnergyLevel] = None
    transport: Optional[TransportLevel] = None
    communication: Optional[CommunicationLevel] = None


class SettlementSpec(Section):
    """Settlement-size distribution: rank-size exponent and the population of the largest settlement."""

    zipf_exponent: Param = None
    largest_population: Param = None


class PolitySpec(Section):
    """Political structure: how many polities, of which type, with optional capitals and territories."""

    count: Param = None
    type: Optional[PolityType] = None
    capitals: list[Place] = []
    territories: list[list[tuple[float, float]]] = []   # one polygon of (lat, lon) vertices per polity

    @model_validator(mode="after")
    def _enough_polities(self):
        """Reject a fixed polity count smaller than the number of capitals or territories given."""
        if isinstance(self.count, (int, float)) and not isinstance(self.count, bool):
            if self.count < 1:
                raise ValueError("polity count must be at least 1")
            if max(len(self.capitals), len(self.territories)) > self.count:
                raise ValueError("more capitals or territories than polities")
        return self


class ColonySpec(Section):
    """Founding of a colonist culture."""

    founded_years_ago: Param = None
    origin: Optional[str] = None
    landing_sites: list[Place] = []
    immigrants_per_year: float = Field(default=0.0, ge=0)          # organic colonies: arrivals after landing
    immigration_years: Optional[float] = Field(default=None, gt=0)   # how long they keep arriving (None: to the present)
    independence_years: Optional[float] = Field(default=None, gt=0)  # organic colonies: years tied to the home power


EventKind = Literal["eruption", "drought", "pandemic", "impact", "sea_level"]


class HistoryEvent(Section):
    """An event placed in an organic history by hand.

    ``size`` is the eruption's stratospheric sulfur (Tg S), the drought's
    share of rain lost (0–1), the pandemic's deaths per year as a share of the
    world's people (‰), the impact's energy (Mt TNT) or the sea's rise (m).
    Left out, it is drawn as for the planet's own events. A drought needs a
    place; an eruption, pandemic or impact without one is placed as the
    planet's own events are. A sea-level rise floods land up to its height.
    """

    kind: EventKind
    years_ago: float = Field(ge=0)
    lat_deg: Optional[float] = Field(default=None, ge=-90, le=90)
    lon_deg: Optional[float] = Field(default=None, ge=-180, le=360)
    size: Optional[float] = Field(default=None, gt=0)
    radius_km: Optional[float] = Field(default=None, gt=0)          # drought
    duration_years: Optional[float] = Field(default=None, gt=0)     # drought, pandemic

    @model_validator(mode="after")
    def _placed(self):
        """Require a place for droughts and both coordinates or neither; a sea level needs its rise."""
        if (self.lat_deg is None) != (self.lon_deg is None):
            raise ValueError("an event takes both lat_deg and lon_deg, or neither")
        if self.kind == "drought" and self.lat_deg is None:
            raise ValueError("a drought needs lat_deg and lon_deg")
        if self.kind == "sea_level" and self.size is None:
            raise ValueError("a sea_level event needs size (the rise in m)")
        if self.kind == "drought" and self.size is not None and self.size > 1:
            raise ValueError("a drought's size is the share of rain lost, at most 1")
        return self


class HistorySpec(Section):
    """The span of an organic history and the epochs it keeps (years before the present)."""

    start_years_ago: float = Field(default=h.HISTORY_START_YEARS_AGO, gt=0)
    epochs: list[float] = []
    step_years: Optional[float] = Field(default=None, gt=0)
    late_step_years: Optional[float] = Field(default=None, gt=0)
    pace: float = Field(default=1.0, gt=0)          # multiplies the rate at which knowledge grows
    planet_events: bool = True                      # eruptions, droughts, pandemics and impacts drawn at their rates
    events: list[HistoryEvent] = []                 # events placed by hand

    @model_validator(mode="after")
    def _epochs_in_span(self):
        """Reject epochs outside the span; keep them unique and oldest first."""
        for e in self.epochs:
            if not 0 <= e <= self.start_years_ago:
                raise ValueError(f"epoch {e} years ago lies outside the history (0 to {self.start_years_ago})")
        self.epochs = sorted({float(e) for e in self.epochs}, reverse=True)
        for e in self.events:
            if e.years_ago > self.start_years_ago:
                raise ValueError(f"a {e.kind} {e.years_ago} years ago lies outside the history")
        return self


def _given(value) -> bool:
    """Return whether a spec value carries anything beyond its defaults."""
    if isinstance(value, Section):
        return any(_given(getattr(value, f)) for f in type(value).model_fields)
    return value not in (None, [], ())


# Fields an organic culture does not use: its development, population and organisation emerge from the history.
ORGANIC_IGNORED = ("development", "population", "settlements", "polities")


class CultureEntry(Section):
    """One culture: its inhabitants, development, population and organisation."""

    name: Optional[str] = None
    mode: CultureMode = "scoped"
    inhabitants: Inhabitants = "native"
    development: DevelopmentSpec = DevelopmentSpec()
    population: Param = None
    homeland: Optional[Homeland] = None
    settlements: SettlementSpec = SettlementSpec()
    polities: PolitySpec = PolitySpec()
    colony: Optional[ColonySpec] = None

    @model_validator(mode="after")
    def _colony_only_for_colonists(self):
        """Reject colony details on a native culture."""
        if self.colony is not None and self.inhabitants != "colonist":
            raise ValueError("colony details are only valid for inhabitants: colonist")
        return self

    @model_validator(mode="after")
    def _organic_ignores_outcomes(self):
        """Warn that an organic culture's given development, population and organisation are ignored.

        A colonist keeps its development as the level it arrives with. Immigration
        after landing applies to organic colonies only.
        """
        if self.mode != "organic":
            if self.colony is not None and (self.colony.immigrants_per_year > 0
                                            or self.colony.independence_years is not None):
                warnings.warn(f"culture {self.name or ''}: immigrants_per_year and independence_years apply to "
                              "organic colonies only; ignored", stacklevel=2)
            return self
        ignored = [f for f in ORGANIC_IGNORED if _given(getattr(self, f))
                   and not (f == "development" and self.inhabitants == "colonist")]
        if ignored:
            warnings.warn(f"organic culture {self.name or ''}: {', '.join(ignored)} emerge from the history; "
                          "the given values are ignored", stacklevel=2)
        return self

    @field_validator("population")
    @classmethod
    def _positive_population(cls, v):
        """Reject non-positive populations."""
        low = v[0] if isinstance(v, tuple) else v
        if low is not None and low <= 0:
            raise ValueError("population must be positive")
        return v


class CultureSpec(Section):
    """The cultures of a planet. ``seed`` redraws the culture alone; by default the planet seed is used."""

    seed: Optional[int] = None
    history: Optional[HistorySpec] = None
    cultures: list[CultureEntry] = Field(min_length=1)

    @property
    def organic(self) -> bool:
        """Return whether any culture is organic."""
        return any(c.mode == "organic" for c in self.cultures)

    def history_or_default(self) -> HistorySpec:
        """Return the history section, or the default one when the spec has none."""
        return self.history if self.history is not None else HistorySpec()

    @model_validator(mode="after")
    def _colonies_within_history(self):
        """Reject organic colonies founded before the history starts."""
        start = self.history_or_default().start_years_ago
        for c in self.cultures:
            if c.mode == "organic" and c.colony is not None:
                f = c.colony.founded_years_ago
                high = max(f) if isinstance(f, tuple) else f
                if high is not None and high > start:
                    raise ValueError(f"organic colony founded {high} years ago, before the history starts ({start})")
        return self


def load_culture_spec(path: str | Path) -> CultureSpec:
    """Read a culture spec from a YAML file, either the section itself or a file with a top-level ``culture`` key."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "culture" in data and "cultures" not in data:
        data = data["culture"]
    return CultureSpec.model_validate(data)


def save_culture_spec(spec: CultureSpec, path: str | Path) -> None:
    """Write a culture spec to a YAML file, omitting unset fields."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"culture": spec.model_dump(mode="json", exclude_none=True, exclude_defaults=True)}, f,
                       sort_keys=False, allow_unicode=True)
