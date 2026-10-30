"""The civilisation placed on a planet: cultures, polities, settlements, roads, per-cell layers and a chronicle.

Every object has a stable id, a founding time (years before the civilisation's
epoch) and a parent, so the organic history can record where
each came from. User-scoped mode fills the founding times it knows (colonies)
and leaves the rest unknown (NaN or None).

Settlements and roads are held as column tables (NumPy arrays) because a
populated world has thousands of them. Road paths are sequences of global grid
cells, stored flat with an offset per road.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import xarray as xr

from .development import Development
from .spec import CultureSpec

# Settlement forms, stored as small integer codes.
SETTLEMENT_FORMS = ("camp", "hamlet", "village", "town", "city", "metropolis", "outpost", "dome", "underground",
                    "orbital_anchor")
# Road classes, stored as small integer codes.
ROAD_KINDS = ("track", "road", "trunk", "highway", "sea_route")


@dataclass
class Culture:
    """One culture with its resolved inputs. Drawn values are fixed here; ``None`` means derived later."""

    id: int
    name: Optional[str]
    inhabitants: str
    development: Development
    population: Optional[float] = None
    homeland: Optional[dict] = None                     # lat_deg, lon_deg, radius_km
    zipf_exponent: Optional[float] = None
    largest_population: Optional[float] = None
    polity_count: Optional[int] = None
    polity_type: Optional[str] = None
    capitals: list[dict] = field(default_factory=list)
    territories: list[list[list[float]]] = field(default_factory=list)
    founded_years_ago: Optional[float] = None
    origin: Optional[str] = None
    landing_sites: list[dict] = field(default_factory=list)
    immigrants_per_year: float = 0.0                    # organic colonies: arrivals per year after landing
    immigration_years: Optional[float] = None           # for how long (None: to the present)
    independence_years: Optional[float] = None          # organic colonies: years tied to the home power (None: by turmoil)
    crowd_diseases: bool = False                        # organic colonies from off the planet: carry crowd diseases
    home_culture: Optional[int] = None                  # organic colonies founded from this planet: the home culture's id
    creole_of: list[int] = field(default_factory=list)  # a creole culture: the ids of the two cultures it formed from
    phonology: Optional[dict] = None
    parent: int = -1
    mode: str = "scoped"                                # "scoped" (placed from the spec) or "organic" (emerges from a history)

    def to_plain(self) -> dict:
        """Return the culture as built-in types for YAML."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["development"] = self.development.to_names()
        return out

    @classmethod
    def from_plain(cls, data: dict) -> "Culture":
        """Return a culture from its YAML form."""
        data = dict(data)
        data["development"] = Development.from_names(data["development"])
        return cls(**data)


@dataclass
class Polity:
    """A political unit belonging to one culture."""

    id: int
    culture: int
    type: str
    name: Optional[str] = None
    capital: int = -1                                   # settlement id
    parent: int = -1                                    # polity it split from or belongs to
    founded_years_ago: Optional[float] = None
    population: Optional[float] = None
    area_km2: Optional[float] = None
    rulers: Optional[int] = None                        # organic states: rulers so far
    primogeniture: Optional[bool] = None                # organic states: succession by primogeniture

    def to_plain(self) -> dict:
        """Return the polity as built-in types for YAML."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class ChronicleEntry:
    """One event in a civilisation's history (filled by the organic mode)."""

    years_ago: float
    kind: str
    text: str
    refs: dict = field(default_factory=dict)            # ids of the cultures, polities or settlements involved


class _Table:
    """Column table of equal-length NumPy arrays; subclasses list the columns and their dtypes."""

    COLUMNS: dict[str, str] = {}
    DIM = "row"

    def __init__(self, **columns):
        """Store the given columns, filling missing ones with their empty defaults."""
        n = len(next(iter(columns.values()))) if columns else 0
        for name, dtype in self.COLUMNS.items():
            if name in columns:
                values = np.asarray(columns[name], dtype=dtype)
            else:
                values = np.full(n, self._fill(dtype), dtype=dtype)
            if len(values) != n:
                raise ValueError(f"column {name} has {len(values)} rows, expected {n}")
            setattr(self, name, values)
        unknown = set(columns) - set(self.COLUMNS)
        if unknown:
            raise ValueError(f"unknown columns: {sorted(unknown)}")

    @staticmethod
    def _fill(dtype: str):
        """Return the value that marks an unknown entry in a column of this dtype."""
        if dtype.startswith("f"):
            return np.nan
        if dtype.startswith("U") or dtype == "str" or dtype == "object":
            return ""
        return -1

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(getattr(self, next(iter(self.COLUMNS))))

    @classmethod
    def empty(cls):
        """Return a table with no rows."""
        return cls(**{name: np.empty(0, dtype=dtype) for name, dtype in cls.COLUMNS.items()})

    @classmethod
    def variable(cls, name: str) -> str:
        """Return the dataset variable name of a column (prefixed with the table's dimension)."""
        return f"{cls.DIM}_{name}"

    def to_dataset(self) -> xr.Dataset:
        """Return the table as dataset variables along its own dimension."""
        return xr.Dataset({self.variable(name): (self.DIM, getattr(self, name)) for name in self.COLUMNS})

    @classmethod
    def _columns_from(cls, ds: xr.Dataset) -> dict:
        """Return the table's columns present in a dataset."""
        return {name: ds[cls.variable(name)].values for name in cls.COLUMNS if cls.variable(name) in ds}

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        """Return a table from dataset variables along its dimension (absent columns get their fill)."""
        present = cls._columns_from(ds)
        if not present:
            return cls.empty()
        return cls(**present)


class SettlementTable(_Table):
    """Settlements: position, population, form, culture, settlement region (with rank in it) and polity."""

    DIM = "settlement"
    COLUMNS = {
        "id": "i8", "culture": "i4", "polity": "i4", "cell": "i8", "lat_deg": "f8", "lon_deg": "f8",
        "population": "f8", "form": "i1", "founded_years_ago": "f8", "parent": "i8", "name": "U40",
        "region": "i4", "rank": "i8",
    }


class TradeTable(_Table):
    """Trade links: the gravity pairs a road network is built from, between settlements (id, or −1 for a rural
    node) at two cells, with their gravity flow and the length (km) and travel time (hours) of their route."""

    DIM = "trade"
    COLUMNS = {"id": "i8", "a": "i8", "b": "i8", "a_cell": "i8", "b_cell": "i8", "culture": "i4", "flow": "f8",
               "km": "f8", "hours": "f8"}


class SeriesTable(_Table):
    """An organic history over time: per culture at the end of each step (years ago), its people, the area of the
    cells it lives in, its living states and the largest one's area, and its people-weighted mean levels on the three
    axes with the highest energy level in force."""

    DIM = "series"
    COLUMNS = {"years_ago": "f8", "culture": "i8", "people": "f8", "area_km2": "f8", "states": "i4",
               "largest_state_km2": "f8", "energy": "f4", "transport": "f4", "communication": "f4",
               "top_energy": "i1"}

    def until(self, years_ago: float) -> "SeriesTable":
        """Return the rows at or before a date (``years_ago`` or earlier)."""
        keep = self.years_ago >= years_ago - 1e-9
        return SeriesTable(**{name: getattr(self, name)[keep] for name in self.COLUMNS})


class RoadTable(_Table):
    """Road segments between settlements or junctions (−1), with their paths as global grid cells."""

    DIM = "road"
    COLUMNS = {"id": "i8", "a": "i8", "b": "i8", "kind": "i1", "length_km": "f8", "founded_years_ago": "f8",
               "culture": "i4", "flow": "f8"}

    def __init__(self, path_offsets=None, path_cells=None, **columns):
        """Store the columns and the flat path cells; road i follows cells[offsets[i]:offsets[i + 1]]."""
        super().__init__(**columns)
        n = len(self)
        self.path_offsets = np.zeros(n + 1, dtype="i8") if path_offsets is None else np.asarray(path_offsets, "i8")
        self.path_cells = np.empty(0, dtype="i8") if path_cells is None else np.asarray(path_cells, "i8")
        if len(self.path_offsets) != n + 1 or self.path_offsets[-1] != len(self.path_cells):
            raise ValueError("path offsets do not match the roads and path cells")

    def path(self, i: int) -> np.ndarray:
        """Return the grid cells along road ``i``."""
        return self.path_cells[self.path_offsets[i]:self.path_offsets[i + 1]]

    def to_dataset(self) -> xr.Dataset:
        """Return the roads and their paths as dataset variables."""
        ds = super().to_dataset()
        return ds.assign(road_path_offsets=("road_offset", self.path_offsets),
                         road_path_cells=("road_vertex", self.path_cells))

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "RoadTable":
        """Return roads and their paths from dataset variables."""
        columns = cls._columns_from(ds)
        if not columns:
            return cls.empty()
        return cls(path_offsets=ds["road_path_offsets"].values, path_cells=ds["road_path_cells"].values, **columns)


@dataclass
class Civilisation:
    """Everything culture adds to a world at one epoch (``epoch_years_ago`` = 0 is the world's present)."""

    spec: CultureSpec
    seed: int
    cultures: list[Culture]
    polities: list[Polity] = field(default_factory=list)
    settlements: SettlementTable = field(default_factory=SettlementTable.empty)
    roads: RoadTable = field(default_factory=RoadTable.empty)
    trade: TradeTable = field(default_factory=TradeTable.empty)
    series: SeriesTable = field(default_factory=SeriesTable.empty)     # organic history over time
    layers: xr.Dataset = field(default_factory=xr.Dataset)   # per-cell fields on the world grid
    chronicle: list[ChronicleEntry] = field(default_factory=list)
    epoch_years_ago: float = 0.0
    epochs: list["Civilisation"] = field(default_factory=list)   # earlier states kept by an organic history, oldest first
    previous_epoch_years_ago: Optional[float] = None   # set by ``at_epoch`` on the epoch it returns (not saved)

    def at_epoch(self, years_ago: float) -> "Civilisation":
        """Return the civilisation as it stood at a kept epoch (0: the present)."""
        dates = [c.epoch_years_ago for c in self.epochs]
        for k, civ in enumerate([*self.epochs, self]):
            if abs(civ.epoch_years_ago - years_ago) < 1e-6:
                if civ is not self:
                    civ.previous_epoch_years_ago = dates[k - 1] if k > 0 else None
                return civ
        kept = ", ".join(f"{c.epoch_years_ago:g}" for c in [*self.epochs, self])
        raise KeyError(f"no epoch {years_ago:g} years ago (kept: {kept})")

    def event_window(self) -> tuple[float, float]:
        """Return (since, until) years ago: from the previous kept epoch (or the start of the history) to this one."""
        if self.previous_epoch_years_ago is not None:
            since = self.previous_epoch_years_ago
        elif self.epochs:
            since = self.epochs[-1].epoch_years_ago
        else:
            since = float(self.spec.history_or_default().start_years_ago)
        return float(since), float(self.epoch_years_ago)

    def culture(self, culture_id: int) -> Culture:
        """Return the culture with the given id."""
        for c in self.cultures:
            if c.id == culture_id:
                return c
        raise KeyError(culture_id)
