"""Time series of an organic history: per culture, at the end of every step, its people, land, states and levels."""

from __future__ import annotations

import numpy as np

from ..model import SeriesTable
from .state import in_force


class SeriesRecord:
    """Rows of the series table, gathered while the history runs."""

    def __init__(self):
        """Start with no rows."""
        self.rows: dict[str, list] = {name: [] for name in SeriesTable.COLUMNS}

    def update(self, ctx, state, when: float) -> None:
        """Record every culture with people as it stands ``when`` years ago."""
        cell_km2 = float(np.mean(ctx.cov.cell_area_km2))
        owner = state.owner
        n = len(state.polities)
        ruled = owner >= 0
        area = np.bincount(owner[ruled], minlength=n) * cell_km2
        culture_of = np.fromiter((p.culture for p in state.polities), dtype=np.int64, count=n)
        alive = np.fromiter((p.alive for p in state.polities), dtype=bool, count=n) & (area > 0)
        for row in np.flatnonzero(state.active):
            people = state.population[row]
            total = float(people.sum())
            if total <= 0:
                continue
            cid = state.culture_ids[row]
            mine = alive & (culture_of == cid)
            levels = np.average(state.development[row], axis=1, weights=people)
            top = in_force(state.development[row, 0][people > 0]).max()
            for name, value in (("years_ago", when), ("culture", cid), ("people", total),
                                ("area_km2", float((people > 0).sum() * cell_km2)), ("states", int(mine.sum())),
                                ("largest_state_km2", float(area[mine].max()) if mine.any() else 0.0),
                                ("energy", levels[0]), ("transport", levels[1]), ("communication", levels[2]),
                                ("top_energy", int(top))):
                self.rows[name].append(value)

    def table(self) -> SeriesTable:
        """Return the rows as a table."""
        return SeriesTable(**self.rows)


def series_step(ctx, state, years: float) -> None:
    """Process: record the series at the end of the step."""
    ctx.cache.setdefault("series", SeriesRecord()).update(ctx, state, state.years_ago - years)
