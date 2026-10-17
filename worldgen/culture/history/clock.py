"""The time steps of an organic history: one generation at a time, shorter once a culture burns fossil fuel.

Steps end exactly on every kept epoch and on the present, so the snapshots
fall on the requested dates.
"""

from __future__ import annotations

from ... import heuristics as h
from ..spec import HistorySpec


def step_years(history: HistorySpec, energy_level: int) -> float:
    """Return the step length for the highest energy level any organic culture has reached."""
    if energy_level >= h.HISTORY_LATE_FROM_ENERGY:
        return history.late_step_years or h.HISTORY_LATE_STEP_YEARS
    return history.step_years or h.HISTORY_STEP_YEARS


def stops(history: HistorySpec, extra: tuple[float, ...] = ()) -> list[float]:
    """Return the dates the steps must land on (epochs, the present and any extra dates), newest last."""
    dates = {float(e) for e in history.epochs} | {0.0} | {float(e) for e in extra if 0 <= e < history.start_years_ago}
    return sorted(dates, reverse=True)


def next_date(now: float, length: float, dates: list[float]) -> float:
    """Return the date one step after ``now`` (years ago), cut short at the next date that must be hit."""
    target = now - length
    upcoming = [d for d in dates if d < now]
    if upcoming:
        target = max(target, upcoming[0])
    return max(target, 0.0)
