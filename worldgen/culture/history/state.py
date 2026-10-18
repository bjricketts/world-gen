"""The state an organic history carries from step to step, and the chronicle it writes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..development import AXES, Development
from ..model import ChronicleEntry

AXIS_NAMES = tuple(AXES)


LIVESTOCK_NONE, LIVESTOCK_KEPT, LIVESTOCK_BROUGHT = 0, 1, 2      # codes of HistoryState.livestock


def in_force(level: np.ndarray) -> np.ndarray:
    """Return the level in force for real-valued levels: level k holds from k − 0.5 to k + 0.5."""
    return np.floor(np.asarray(level, dtype=float) + 0.5).astype(int)


@dataclass
class HistoryPolity:
    """A state polity of the history: capital cell, founding and end (years ago), parent, cohesion, peak area and rulers."""

    id: int
    culture: int
    capital: int
    founded_years_ago: float
    parent: int = -1
    cohesion: float = 0.5
    ended_years_ago: float | None = None
    peak_km2: float = 0.0                    # largest area it held
    primogeniture: bool = False              # succession rule adopted
    reign_started_years_ago: float | None = None   # the present ruler's accession (None: at the founding)
    rulers: int = 1                          # rulers so far, the present one included

    @property
    def alive(self) -> bool:
        """Return whether the polity still exists."""
        return self.ended_years_ago is None


@dataclass
class HistoryState:
    """Per-cell people and development of the organic cultures at one date.

    ``population`` is people per cell (culture × cell). ``development`` holds
    the level on each axis per cell (culture × axis × cell) as a real number.
    Level k is in force from k − 0.5 to k + 0.5 (``in_force``): the anchors of
    each level (Earth's population in that era) sit at its middle, and the
    capacity of a cell between two anchors is interpolated.
    """

    years_ago: float
    step: int
    culture_ids: list[int]
    population: np.ndarray
    development: np.ndarray
    active: np.ndarray                       # culture present yet (colonists land during the history)
    owner: np.ndarray = None                 # state polity id per cell, −1 where none (one state per cell)
    polities: list = field(default_factory=list)   # HistoryPolity records, alive and ended
    livestock: np.ndarray = None             # per cell: 0 no herds, 1 herds kept (LIVESTOCK_KEPT), 2 brought by settlers

    def __post_init__(self):
        """Start with no states and no herds."""
        if self.owner is None:
            self.owner = np.full(self.population.shape[1], -1, dtype=np.int64)
        if self.livestock is None:
            self.livestock = np.zeros(self.population.shape[1], dtype=np.int8)

    def copy(self) -> "HistoryState":
        """Return an independent copy."""
        return replace(self, culture_ids=list(self.culture_ids), population=self.population.copy(),
                       development=self.development.copy(), active=self.active.copy(), owner=self.owner.copy(),
                       polities=[replace(p) for p in self.polities], livestock=self.livestock.copy())

    def index(self, culture_id: int) -> int:
        """Return the row of a culture in the state arrays."""
        return self.culture_ids.index(culture_id)

    def culture_development(self, row: int) -> Development:
        """Return a culture's development: on each axis the level reached by at least half its people."""
        people = self.population[row]
        total = people.sum()
        levels = {}
        for a, axis in enumerate(AXIS_NAMES):
            level = in_force(self.development[row, a])
            if total <= 0:
                levels[axis] = int(level.max(initial=0))
                continue
            order = np.argsort(-level, kind="stable")
            reached = np.cumsum(people[order]) >= 0.5 * total
            levels[axis] = int(level[order][np.argmax(reached)])
        return Development(**levels)

    def cell_development(self, row: int, cell: int) -> Development:
        """Return the development in force in one cell of a culture."""
        return Development(**{axis: int(in_force(self.development[row, a, cell])) for a, axis in enumerate(AXIS_NAMES)})

    def top_energy(self) -> int:
        """Return the highest energy level reached in any occupied cell of any active culture."""
        occupied = (self.population > 0) & self.active[:, None]
        if not occupied.any():
            return 0
        return int(in_force(self.development[:, 0][occupied]).max())


@dataclass
class Chronicle:
    """The events of a history, in the order they happened."""

    entries: list[ChronicleEntry] = field(default_factory=list)

    def add(self, years_ago: float, kind: str, text: str, **refs) -> None:
        """Record one event."""
        self.entries.append(ChronicleEntry(years_ago=float(years_ago), kind=kind, text=text,
                                           refs={k: int(v) for k, v in refs.items()}))

    def until(self, years_ago: float) -> list[ChronicleEntry]:
        """Return the events up to (and including) a date."""
        return [e for e in self.entries if e.years_ago >= years_ago - 1e-9]
