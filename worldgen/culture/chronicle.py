"""Reading an organic history's chronicle: major entries in full, minor ones counted per period.

An entry is major when its kind always is (origins, firsts, colonies,
independence, creoles, contact epidemics, eruptions, droughts, sea-level
rises, knowledge loss) or when it is large: a war or break-up of
CHRONICLE_MAJOR_KM2 or more, a war of CHRONICLE_MAJOR_WAR_STATES states or
more, a pandemic or impact killing CHRONICLE_MAJOR_DEATHS, an overseas colony
that founds a new colonist culture. Secessions, conquests and the rest
are minor. Periods are CHRONICLE_PERIOD_YEARS long, shorter towards the present.
State and culture numbers in the text are replaced by their names where any
kept epoch named them.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .. import heuristics as h

MAJOR_KINDS = {"origin", "farming_origin", "first_energy", "first_transport", "first_communication", "first_state",
               "great_power", "landfall", "landing", "colony_state", "independence", "creole", "supremacy",
               "epidemic", "eruption", "drought", "sea_level", "knowledge_loss"}
DEADLY_KINDS = {"pandemic", "impact"}
SIZED_KINDS = {"war", "break_up"}
PLURALS = {"war": "wars", "break_up": "break-ups", "secession": "secessions", "conquest": "conquests",
           "overseas_colony": "overseas colonies"}

_STATE = re.compile(r"\b[Ss]tate (\d+)")
_CULTURE = re.compile(r"\bCulture (\d+)")


def is_major(entry) -> bool:
    """Return whether a chronicle entry is written out in full."""
    if entry.kind in MAJOR_KINDS:
        return True
    refs = entry.refs or {}
    if entry.kind in SIZED_KINDS:
        return (refs.get("area_km2", 0) >= h.CHRONICLE_MAJOR_KM2
                or refs.get("states", 0) >= h.CHRONICLE_MAJOR_WAR_STATES)
    if entry.kind in DEADLY_KINDS:
        return refs.get("killed", 0) >= h.CHRONICLE_MAJOR_DEATHS
    if entry.kind == "overseas_colony":
        return bool(refs.get("new", False))
    return False


def period_years(years_ago: float) -> float:
    """Return the length of the chronicle period an entry so many years ago falls in."""
    for older_than, length in h.CHRONICLE_PERIOD_YEARS:
        if years_ago >= older_than:
            return length
    return h.CHRONICLE_PERIOD_YEARS[-1][1]


def period_of(years_ago: float) -> tuple[float, float]:
    """Return the (start, end) years ago of the period holding a date."""
    length = period_years(years_ago)
    start = (years_ago // length + (years_ago % length > 0)) * length
    return start, start - length


def names(civ) -> tuple[dict[int, str], dict[int, str]]:
    """Return the names of states and cultures by id, from the civilisation and its kept epochs."""
    states, cultures = {}, {}
    for c in [*civ.epochs, civ]:
        for p in c.polities:
            if p.name:
                states.setdefault(p.id, p.name)
        for k in c.cultures:
            if k.name:
                cultures[k.id] = k.name
    return states, cultures


def entries(civ, since: Optional[float] = None, until: Optional[float] = None, kinds: Optional[Iterable[str]] = None,
            culture: Optional[int] = None, minor: bool = True) -> list:
    """Return the chronicle entries between ``since`` and ``until`` years ago (inclusive), of the given kinds,
    involving a culture, oldest first; ``minor=False`` keeps only major entries."""
    kinds = set(kinds) if kinds else None
    out = []
    for e in civ.chronicle:
        if since is not None and e.years_ago > since or until is not None and e.years_ago < until:
            continue
        if kinds is not None and e.kind not in kinds:
            continue
        if culture is not None and culture not in (e.refs.get("culture"), e.refs.get("subject")):
            continue
        if not minor and not is_major(e):
            continue
        out.append(e)
    return sorted(out, key=lambda e: -e.years_ago)


def named_text(entry, states: dict, cultures: dict) -> str:
    """Return an entry's text with state and culture numbers replaced by names where known."""
    text = _STATE.sub(lambda m: states.get(int(m.group(1)), m.group(0)), entry.text)
    return _CULTURE.sub(lambda m: cultures.get(int(m.group(1)), m.group(0)), text)


def _when(years_ago: float) -> str:
    """Return a date as text."""
    return "present" if years_ago <= 0 else f"{years_ago:,.0f} years ago"


def format_chronicle(civ, since: Optional[float] = None, until: Optional[float] = None,
                     kinds: Optional[Iterable[str]] = None, culture: Optional[int] = None,
                     minor: str = "count") -> str:
    """Return the chronicle as text by period: major entries in full, minor ones counted (``minor="count"``),
    listed (``"list"``) or left out (``"hide"``)."""
    states, cultures = names(civ)
    chosen = entries(civ, since, until, kinds, culture)
    if not chosen:
        return "No chronicle entries."
    lines, current, counts = [], None, {}

    def close():
        if counts and minor == "count":
            lines.append("    also " + ", ".join(f"{n} {PLURALS.get(k, k.replace('_', ' ')) if n > 1 else k.replace('_', ' ')}"
                                                 for k, n in sorted(counts.items(), key=lambda kv: -kv[1])))

    for e in chosen:
        show = is_major(e) or minor == "list"
        if not show and minor == "hide":
            continue
        period = period_of(e.years_ago)
        if period != current:
            close()
            counts = {}
            current = period
            lines += ["", f"{_when(period[0])} to {_when(max(period[1], 0.0))}"]
        if show:
            lines.append(f"  {e.years_ago:>8,.0f}  {named_text(e, states, cultures)}")
        else:
            counts[e.kind] = counts.get(e.kind, 0) + 1
    close()
    return "\n".join(lines).lstrip("\n")
