"""Place a civilisation on a world from a culture spec (user-scoped mode)."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import xarray as xr

from .density import population_density
from .geology import Setting, geology
from .. import heuristics as h
from .habitats import Habitat, elevator_feasible, habitat_classes
from .model import SETTLEMENT_FORMS, Civilisation
from .names import name_everything
from .polities import build_polities
from .resources import resource_potential
from .roads import build_network
from .settlements import place_settlements
from .suitability import covariates
from .resolve import culture_seed, resolve_cultures
from .spec import CultureSpec


def build_civilisation(world, spec: Optional[CultureSpec] = None) -> Civilisation:
    """Return the civilisation a culture spec places on a world; ``spec`` defaults to the world spec's culture section.

    With organic cultures in the spec the civilisation comes from a simulated
    history and carries the kept epochs.
    """
    spec = spec if spec is not None else world.spec.culture
    if spec is None:
        raise ValueError("no culture spec given and the world spec has no culture section")
    seed = culture_seed(spec, world.spec.seed)
    cultures = resolve_cultures(spec, world.spec.seed)
    if spec.organic:
        from .history.epochs import organic_civilisation
        return organic_civilisation(world, spec, seed, cultures)
    layers = geology_layers(world)
    if world.surface is None:
        return Civilisation(spec=spec, seed=seed, cultures=cultures, layers=layers)
    cultures, people = population_layers(world, cultures, seed, layers.get("resource_potential"))
    return finish_civilisation(world, spec, seed, cultures, layers, people)


def finish_civilisation(world, spec: CultureSpec, seed: int, cultures, layers: xr.Dataset,
                        people: xr.Dataset, given_polities=None, id_offset: int = 0,
                        old_roads=None) -> Civilisation:
    """Return the civilisation from placed people: settlements, roads, trade, polities and names on top of the given layers.

    ``given_polities`` (``GivenStates``) and ``id_offset`` pass an organic history's states to
    the polity step (``build_polities``); ``old_roads`` are an earlier epoch's roads, cheaper to reuse.
    """
    cultures, settlements, placed = settlement_layers(world, cultures, people, seed)
    layers = layers.merge(people).merge(placed)
    civ = Civilisation(spec=spec, seed=seed, cultures=cultures, layers=layers)
    civ.settlements = settlements
    people_cells = layers["population_density"].values * _cell_area(world)
    civ.roads, civ.trade = build_network(world, settlements, cultures, people_cells, old_roads)
    polities, settlement_polity, owner = build_polities(world, cultures, settlements, people_cells, seed,
                                                               given_polities, id_offset)
    civ.polities = polities
    civ.settlements.polity = settlement_polity
    civ.layers["polity"] = xr.DataArray(owner, dims=("culture", "cell"), coords={"culture": [c.id for c in cultures]},
                                        attrs={"long_name": "polity (-1: none)"})
    civ.cultures, civ.polities, civ.settlements.name = name_everything(world, civ.cultures, civ.settlements,
                                                                       civ.polities, seed)
    return civ


def _cell_area(world) -> float:
    """Return the area of one grid cell (km²)."""
    radius_km = float(world.surface.attrs["radius_m"]) / 1e3
    return 4 * np.pi * radius_km**2 / world.grid.size


def geology_layers(world) -> xr.Dataset:
    """Return the per-cell layers culture builds on: tectonic setting and resource potential."""
    if world.surface is None:
        return xr.Dataset()
    geo = geology(world)
    setting = xr.DataArray(geo.setting, dims="cell", attrs={
        "long_name": "tectonic setting", "flag_values": " ".join(str(int(s)) for s in Setting),
        "flag_meanings": " ".join(s.name.lower() for s in Setting)})
    return xr.Dataset({"tectonic_setting": setting, "resource_potential": resource_potential(world, geo)})


def population_layers(world, cultures, seed: int, resources=None):
    """Return the cultures with populations and landing sites filled, and their suitability and density layers."""
    cov = covariates(world, resources)
    density, capacity, populations, sites = population_density(world.grid.points, cov, cultures, seed)
    filled = [replace(c, population=p, landing_sites=s if c.inhabitants == "colonist" else c.landing_sites)
              for c, p, s in zip(cultures, populations, sites)]
    coords = {"culture": [c.id for c in cultures]}
    layers = xr.Dataset({
        "suitability": xr.DataArray(capacity, dims=("culture", "cell"), coords=coords, attrs={
            "long_name": "carrying capacity relative to the best cell for the culture"}),
        "population_density": xr.DataArray(density, dims=("culture", "cell"), coords=coords, attrs={
            "long_name": "population density", "units": "people/km2"}),
    })
    return filled, layers


def settlement_layers(world, cultures, people: xr.Dataset, seed: int):
    """Return the cultures with settlement parameters filled, the listed settlements, and the dispersed and region layers."""
    radius_km = float(world.surface.attrs["radius_m"]) / 1e3
    grid = world.grid
    habitat = habitat_classes(world) if world.state is not None else None
    table, dispersed, regions, filled = place_settlements(
        grid.points, people["population_density"].values, cultures, 4 * np.pi * radius_km**2 / grid.size,
        grid.mean_spacing(radius_km), radius_km, seed, habitat)
    if world.state is not None:
        anchor_elevators(world, table, filled)
    coords = {"culture": [c.id for c in cultures]}
    layers = xr.Dataset({
        "dispersed_density": xr.DataArray(dispersed, dims=("culture", "cell"), coords=coords, attrs={
            "long_name": "density of people outside the listed settlements", "units": "people/km2"}),
        "region": xr.DataArray(regions, dims=("culture", "cell"), coords=coords, attrs={
            "long_name": "settlement region (-1: unoccupied)"}),
    })
    if habitat is not None:
        layers["habitat"] = xr.DataArray(habitat, dims="cell", attrs={
            "long_name": "habitat colonists need", "flag_values": " ".join(str(int(v)) for v in Habitat),
            "flag_meanings": " ".join(v.name.lower() for v in Habitat)})
    return filled, table, layers


def anchor_elevators(world, settlements, cultures) -> None:
    """Make the largest settlement near the equator of each spacefaring colony an orbital anchor, where an elevator can stand."""
    if not elevator_feasible(world.state):
        return
    lat = world.grid.lat
    for c in cultures:
        if c.inhabitants != "colonist" or c.development.transport < 5:
            continue
        mine = np.flatnonzero((settlements.culture == c.id)
                              & (np.abs(lat[settlements.cell]) <= h.ELEVATOR_MAX_LATITUDE_DEG))
        if mine.size:
            top = mine[np.argmax(settlements.population[mine])]
            settlements.form[top] = SETTLEMENT_FORMS.index("orbital_anchor")


def world_at_epoch(world, years_ago: float):
    """Return the world with its civilisation as it stood at a kept epoch of its organic history (0: the present)."""
    from dataclasses import replace

    if world.culture is None:
        raise ValueError(f"{world.state.name} has no culture")
    return replace(world, culture=world.culture.at_epoch(years_ago))


def apply_culture(world, spec: Optional[CultureSpec] = None):
    """Return a copy of the world carrying the civilisation built from ``spec`` (or its own culture section)."""
    return replace(world, culture=build_civilisation(world, spec))
