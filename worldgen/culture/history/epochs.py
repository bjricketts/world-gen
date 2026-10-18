"""A civilisation from an organic history: the milestone 8 placement run on the history's people at each kept epoch.

At every kept epoch the organic cultures take the development their people
have reached and the density the history gives them; scoped cultures present
at that date are placed as in user-scoped mode. Settlements, roads, trade and
names then follow the milestone 8 steps, each epoch reusing the previous
epoch's roads, and settlements and roads get stable ids and founding dates
(``continuity.py``); organic cultures keep the history's states
and borders, with their ids, founding dates and parents. The present epoch is the
civilisation itself, the earlier ones are stored in ``epochs``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import xarray as xr

from ... import heuristics as h
from ..build import finish_civilisation, geology_layers
from ..density import population_density
from ..model import Civilisation, Culture
from ..polities import GivenStates
from ..names import make_phonology
from ..spec import CultureSpec
from ..suitability import covariates, density_terms, log_capacity
from .run import HistoryResult, default_processes, run_history
from .suitability import build_suitability
from .continuity import carry_through
from .polities import subpolity_counts
from .state import HistoryState


def scoped_at(culture: Culture, years_ago: float) -> Culture | None:
    """Return a scoped culture as it stood at a date, or ``None`` if it was not there yet.

    Colonies are there from their founding and younger by the date; other
    scoped cultures exist only at the present.
    """
    if culture.inhabitants == "colonist":
        founded = culture.founded_years_ago if culture.founded_years_ago is not None else h.COLONY_DEFAULT_AGE_YEARS
        if founded < years_ago:
            return None
        return replace(culture, founded_years_ago=founded - years_ago)
    return culture if years_ago == 0 else None


def epoch_people(world, cov, cultures: list[Culture], state: HistoryState, seed: int):
    """Return the cultures present at a snapshot's date and their suitability and density layers."""
    area = cov.cell_area_km2
    present, density = [], []
    scoped = [s for s in (scoped_at(c, state.years_ago) for c in cultures if c.mode != "organic") if s is not None]
    scoped_density = {}
    if scoped:
        d, _, populations, sites = population_density(world.grid.points, cov, scoped, seed)
        for k, c in enumerate(scoped):
            scoped_density[c.id] = (d[k], populations[k], sites[k])
    for c in cultures:
        if c.mode == "organic":
            if c.id not in state.culture_ids:                    # arose later in the history
                continue
            row = state.index(c.id)
            people = state.population[row]
            if not state.active[row] or people.sum() <= 0:
                continue
            present.append(replace(c, development=state.culture_development(row), population=float(people.sum())))
            density.append(people / area)
        elif c.id in scoped_density:
            d, population, sites = scoped_density[c.id]
            at = next(s for s in scoped if s.id == c.id)
            present.append(replace(at, population=population,
                                   landing_sites=sites if c.inhabitants == "colonist" else c.landing_sites))
            density.append(d)
    if not present:
        return [], None
    suit = []
    for c in present:
        ln = log_capacity(cov, density_terms(c))
        top = np.max(np.where(np.isfinite(ln), ln, -np.inf))
        suit.append(np.where(np.isfinite(ln), np.exp(ln - top), 0.0))
    coords = {"culture": [c.id for c in present]}
    layers = xr.Dataset({
        "suitability": xr.DataArray(np.array(suit), dims=("culture", "cell"), coords=coords, attrs={
            "long_name": "carrying capacity relative to the best cell for the culture"}),
        "population_density": xr.DataArray(np.array(density), dims=("culture", "cell"), coords=coords, attrs={
            "long_name": "population density", "units": "people/km2"}),
    })
    return present, layers


def history_layers(state: HistoryState, cultures: list[Culture], cell_area_km2: float) -> xr.Dataset:
    """Return the history's own per-cell fields at a snapshot: development level on each axis and sub-cell polity
    counts by organic culture id, and the cells' livestock."""
    ids = [c.id for c in cultures if c.mode == "organic" and c.id in state.culture_ids]
    if not ids:
        return xr.Dataset()
    rows = [state.index(i) for i in ids]
    counts = np.array([subpolity_counts(state.population[r], state.owner, state.development[r, 0], cell_area_km2)
                       for r in rows], dtype="float32")
    return xr.Dataset({
        "development_level": xr.DataArray(
            state.development[rows], dims=("organic", "axis", "cell"),
            coords={"organic": ids, "axis": ["energy", "transport", "communication"]},
            attrs={"long_name": "development level reached (integer part: level in force)"}),
        "subpolity_count": xr.DataArray(counts, dims=("organic", "cell"), coords={"organic": ids}, attrs={
            "long_name": "bands, tribes or chiefdoms in cells without a state"}),
        "livestock": xr.DataArray(state.livestock.astype("int8"), dims=("cell",), attrs={
            "long_name": "herd animals kept (0 none, 1 kept, 2 brought by settlers)"})})


def state_development(state: HistoryState, row: int, cell: int):
    """Return the development at a state's capital, with communication at least writing (every state has it)."""
    dev = state.cell_development(row, cell)
    return replace(dev, communication=max(dev.communication, h.POLITY_FORM_COMMUNICATION))


def place_epoch(world, spec: CultureSpec, seed: int, cultures: list[Culture], state: HistoryState, cov,
                geo: xr.Dataset, result: HistoryResult, old_roads=None) -> Civilisation:
    """Return the civilisation at one snapshot of the history (``old_roads``: the previous kept epoch's)."""
    present, people = epoch_people(world, cov, cultures, state, seed)
    layers = geo.merge(history_layers(state, cultures, cov.cell_area_km2))
    if not present:
        civ = Civilisation(spec=spec, seed=seed, cultures=[], layers=layers)
    else:
        records = [p for p in state.polities if p.alive]
        rows = {p.culture: state.index(p.culture) for p in records}
        given = GivenStates(owner=state.owner, records=records,
                            development={p.id: state_development(state, rows[p.culture], p.capital) for p in records},
                            cultures={c.id for c in present if c.mode == "organic"})
        civ = finish_civilisation(world, spec, seed, present, layers, people, given,
                                  len(result.snapshots[-1].polities), old_roads)
    civ.epoch_years_ago = state.years_ago
    civ.chronicle = result.chronicle.until(state.years_ago)
    return civ


def organic_civilisation(world, spec: CultureSpec, seed: int, cultures: list[Culture]) -> Civilisation:
    """Return the civilisation an organic history leaves at the present, with the earlier kept epochs."""
    if world.surface is None:
        raise ValueError("an organic history needs the planet's surface")
    geo = geology_layers(world)
    cov = covariates(world, geo.get("resource_potential"))
    cultures = [replace(c, phonology=c.phonology or make_phonology(seed, f"culture.{c.id}.phonology"))
                for c in cultures]
    history = spec.history_or_default()
    organic = [c for c in cultures if c.mode == "organic"]
    suit = build_suitability(world, cov, geo.get("resource_potential"))
    geo = geo.assign(suitability_component=suit.to_dataarray().astype("float32"))
    result = run_history(world, organic, history, seed, cov, default_processes(), suit,
                         geo.get("resource_potential"), [c.id for c in cultures])
    landed = {c.id: c for c in result.cultures}
    known = {c.id for c in cultures}
    cultures = [landed.get(c.id, c) for c in cultures] + [c for c in result.cultures if c.id not in known]
    cultures = [replace(c, phonology=c.phonology or make_phonology(seed, f"culture.{c.id}.phonology")) for c in cultures]
    civs = []
    for snapshot in result.snapshots:
        civs.append(place_epoch(world, spec, seed, cultures, snapshot, cov, geo, result,
                                civs[-1].roads if civs else None))
    if result.cells is not None:
        carry_through(civs, result.cells, {c.id for c in result.cultures}, history.start_years_ago)
    if result.series is not None:
        table = result.series.table()
        for civ in civs:
            civ.series = table.until(civ.epoch_years_ago)
    present = civs[-1]
    present.epochs = civs[:-1]
    return present

