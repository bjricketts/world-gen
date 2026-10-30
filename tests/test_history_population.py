"""Growth and spread of people in the organic history."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, build_civilisation, resolve_cultures
from worldgen.culture.development import ENERGY_LEVELS
from worldgen.culture.history.population import (SpreadLinks, advance_front, capacity_by_level, cell_capacity,
                                                 diffuse, front_speed_km_yr, grow, landmasses, row_normalise,
                                                 spread_share)
from worldgen.culture.suitability import covariates, productive_area_km2
from worldgen.grid import build_grid


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_front_speed_follows_fort_and_mendez_with_pinhasis_parameters():
    a, m, t = h.SPREAD_GROWTH_RATE, h.SPREAD_MOBILITY_KM2, h.SPREAD_GENERATION_YEARS
    assert front_speed_km_yr() == pytest.approx(math.sqrt(a * m / t) / (1 + a * t / 2))
    assert 0.6 <= front_speed_km_yr() <= 1.1                    # Pinhasi et al. (2005), demic model
    # lattice diffusion share·Δx²/(4Δt) equals m/(4T)
    assert spread_share(25.0, 100.0) * 100.0**2 / (4 * 25.0) == pytest.approx(m / (4 * t))


def test_logistic_growth_is_exact_and_shared():
    K = np.array([[100.0, 100.0, 0.0]])
    P = np.array([[10.0, 100.0, 5.0]])
    out = grow(P, K, 0.03, 50.0)
    e = math.exp(0.03 * 50.0)
    assert out[0, 0] == pytest.approx(100 * 10 * e / (100 + 10 * (e - 1)))
    assert out[0, 1] == pytest.approx(100.0) and out[0, 2] == 0
    two = grow(np.array([[50.0], [50.0]]), np.array([[100.0], [100.0]]), 0.03, 1000.0)
    assert two.sum() == pytest.approx(100.0, rel=1e-6)          # cultures share a cell's capacity


def test_diffusion_conserves_people_and_stays_in_settled_cells():
    weight = sparse.csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=float))
    P = np.array([100.0, 0.0, 50.0, 0.0])
    out = diffuse(P, weight, P > 0, 0.8)                         # above the limit: sub-steps
    assert out.sum() == pytest.approx(P.sum())
    assert out[1] == 0 and out[3] == 0 and out[2] > 50


def _chain_links(n, km):
    """Links along a chain of cells ``km`` apart (both directions)."""
    class Links:
        pass
    links = Links()
    links.rows = np.r_[np.arange(n - 1), np.arange(1, n)]
    links.cols = np.r_[np.arange(1, n), np.arange(n - 1)]
    links.flat_km = np.full(links.rows.size, km)
    return links


def test_front_crosses_links_at_the_front_speed_and_conserves_people():
    n, km, years = 40, 100.0, 25.0
    links = _chain_links(n, km)
    K = np.full(n, 1000.0)
    P = np.zeros(n); P[0] = 1000.0
    progress, credit = np.zeros(n), np.zeros(n)
    settled_at = {}
    total = P.sum()
    for step in range(400):
        P, progress, credit, new, _ = advance_front(P, K, progress, credit, links, years)
        P = grow(P[None], K[None], h.SPREAD_GROWTH_RATE, years)[0]
        for c in new:
            settled_at[int(c)] = (step + 1) * years
        assert np.all(P >= 0)
    cells = sorted(settled_at)
    speed = (cells[-1] - cells[0]) * km / (settled_at[cells[-1]] - settled_at[cells[0]])
    assert speed == pytest.approx(front_speed_km_yr(), rel=0.1)
    P0 = np.zeros(n); P0[0] = 1000.0
    P1, *_ = advance_front(P0, K, np.full(n, 0.99), np.zeros(n), links, years)
    assert P1.sum() == pytest.approx(total)                      # founders move, nobody is created


def test_front_speed_is_the_same_on_coarse_and_fine_grids():
    """On a planet-wide uniform lattice the front runs at the target speed on every grid and step."""

    R = 6371.0
    speeds = []
    for n, dt in ((2500, 25.0), (10000, 25.0), (10000, 5.0)):
        g = build_grid(n)
        coo = g.neighbours.tocoo()

        class Links:
            rows, cols, flat_km = coo.row, coo.col, coo.data * R
        area = 4 * np.pi * R**2 / n
        K = np.full(n, 3.0 * area)
        P = np.zeros(n); P[0] = 0.5 * K[0]
        progress, credit = np.zeros(n), np.zeros(n)
        arrival = np.full(n, np.nan)
        dist = R * np.arccos(np.clip(g.points @ g.points[0], -1, 1))
        for step in range(int(3000 / dt)):
            P = grow(P[None], K[None], h.SPREAD_GROWTH_RATE, dt)[0]
            P, progress, credit, new, _ = advance_front(P, K, progress, credit, Links, dt)
            arrival[new] = (step + 1) * dt
        ok = np.isfinite(arrival) & (dist > 400)
        speeds.append(np.polyfit(arrival[ok], dist[ok], 1)[0])
    assert np.allclose(speeds, front_speed_km_yr(), rtol=0.12)


def test_capacity_holds_earths_population_per_productive_land(earth):
    cov = covariates(earth)
    culture = resolve_cultures(CultureSpec.model_validate({"cultures": [{"mode": "organic"}]}), 0)[0]
    levels = capacity_by_level(culture, cov)
    productive = productive_area_km2(cov).sum()
    for level, name in enumerate(ENERGY_LEVELS):
        assert levels[level].sum() == pytest.approx(h.POPULATION_ANCHOR[name] / h.EARTH_PRODUCTIVE_LAND_KM2 * productive)
    energy = np.where(np.arange(cov.land.size) % 2 == 0, 0.5, 1.7)
    K = cell_capacity(levels, energy)
    assert K[0] == levels[0, 0] and K[1] == levels[1, 1]


def test_sea_routes_open_with_boats(earth):
    # wheel_and_sail: on the ~220 km preview grid a strait is a full cell of sea, beyond pack_and_boat's reach
    foot, boat = SpreadLinks(earth, 0), SpreadLinks(earth, 2)
    ocean = earth.surface["ocean"].values.astype(bool)
    assert not ocean[foot.rows].any() and not ocean[foot.cols].any()
    assert not ocean[boat.cols].any() and boat.rows.size > foot.rows.size
    mass = landmasses(earth)
    assert (mass[boat.rows] != mass[boat.cols]).any() and (mass[foot.rows] == mass[foot.cols]).all()
    assert np.all(boat.flat_km > 0) and row_normalise(boat.weight).sum(axis=1).max() == pytest.approx(1.0)


def test_landmasses_are_labelled_largest_first(earth):
    mass = landmasses(earth)
    ocean = earth.surface["ocean"].values.astype(bool)
    assert np.all(mass[ocean] == -1) and np.all(mass[~ocean] >= 0)
    sizes = np.bincount(mass[~ocean])
    assert np.all(np.diff(sizes) <= 0)


def test_a_homeland_culture_spreads_and_stays_within_capacity(earth):
    spec = CultureSpec.model_validate({"history": {"start_years_ago": 2000, "epochs": [1900]}, "cultures": [
        {"name": "Oru", "mode": "organic", "homeland": {"lat_deg": 40, "lon_deg": 20, "radius_km": 500}}]})
    civ = build_civilisation(earth, spec)
    early = civ.epochs[0].layers["population_density"].values[0]
    late = civ.layers["population_density"].values[0]
    assert (late > 0).sum() > (early > 0).sum()                  # the front has moved out
    area = covariates(earth).cell_area_km2
    culture = civ.cultures[0]
    levels = capacity_by_level(culture, covariates(earth))
    assert np.all(late * area <= levels[0] * 1.1 + 1e-6)       # moves into full cells overshoot a little
    again = build_civilisation(earth, spec)
    assert np.array_equal(again.layers["population_density"].values[0], late)
