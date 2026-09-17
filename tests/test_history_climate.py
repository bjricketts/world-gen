"""Tier 0 climate in history mode: the table, the warm and frozen branches, and the event log."""

import math
from dataclasses import replace

import pytest

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.atmosphere import default_albedo, surface_temperature
from worldgen.climate import COLD_START_K, WARM_START_K, climate_setting, solve_zonal
from worldgen.evolve import evolve
from worldgen.evolve.snapshot import SnapshotEvolver
from worldgen.history.climate import GREY_LIMIT_K, ClimateTable, OrbitClimate
from worldgen.history.integrate import EventLog, HistoryModel, initial_modes, initial_state, open_margin
from worldgen.history.params import build_params
from worldgen.orbit import equilibrium_temperature
from worldgen.priors import resolve
from worldgen.spec import load_spec
from worldgen.surface import relief_factor

from .conftest import EXAMPLES

EARTH_ORBIT = OrbitClimate(eccentricity=0.0167, obliquity_rad=math.radians(23.44),
                           periapsis_longitude_rad=math.radians(283), rotation_period_s=23.934 * 3600.0,
                           orbital_period_s=365.25 * 86400.0, gravity=9.81, planet_class="rocky")


def direct_climate(s, tau, pressure_bar, land, branch="warm", orbit=EARTH_ORBIT):
    """Return the Tier 0 solution at exactly these conditions, bypassing the table."""
    albedo = default_albedo(True, pressure_bar * 1e5, orbit.planet_class)
    grey = surface_temperature(equilibrium_temperature(s, albedo), tau)
    setting = climate_setting(s, orbit.eccentricity, orbit.obliquity_rad, orbit.periapsis_longitude_rad, False,
                              pressure_bar * 1e5, orbit.gravity, "n2_co2", tau, orbit.rotation_period_s,
                              orbit.orbital_period_s, albedo, False, True, min(grey, GREY_LIMIT_K))
    return solve_zonal(setting, land, start_k=WARM_START_K if branch == "warm" else COLD_START_K,
                       bands=h.HISTORY_CLIMATE_BANDS, tolerance_k=h.HISTORY_CLIMATE_TOLERANCE_K)


def model_of(name="earth_history.yaml"):
    """Return a history model for an example planet, without integrating it."""
    spec = load_spec(EXAMPLES / name)
    base = SnapshotEvolver().build(spec.model_copy(update={"mode": "snapshot"}), spec.seed)
    inputs = resolve(spec, spec.seed)
    params = build_params(inputs.get, base.bulk, base.orbit, base.star.mass_kg / c.M_SUN, 0.0)
    o = base.orbit
    orbit = OrbitClimate(o.eccentricity, o.obliquity_rad, o.periapsis_longitude_rad, o.rotation_period_s,
                         o.orbital_period_s, base.bulk.surface_gravity_m_s2, base.bulk.planet_class)
    return HistoryModel(params, ClimateTable(orbit), relief_factor(base.bulk.surface_gravity_m_s2),
                        base.bulk.planet_class, eccentricity=o.eccentricity)


def test_climate_table_matches_a_direct_solve():
    """Interpolated table values stay within a fraction of a kelvin of the zonal model itself."""
    table = ClimateTable(EARTH_ORBIT)
    for s, tau, pressure, land in ((1.0, 0.84, 1.0, 0.29), (0.95, 1.4, 1.2, 0.4), (1.1, 0.5, 0.6, 0.15)):
        got = table.lookup(s, tau, pressure, land, False, "warm")
        want = direct_climate(s, tau, pressure, land)
        assert got.mean_k == pytest.approx(want.mean_k, abs=2.0)
        assert got.open_water == pytest.approx(want.open_water_fraction, abs=0.03)
        assert got.albedo == pytest.approx(want.albedo, abs=0.02)


def test_both_branches_are_stable_between_the_tipping_points():
    """Between the tipping points a frozen planet stays frozen and a warm one stays warm, so history picks."""
    table = ClimateTable(EARTH_ORBIT)
    warm = table.lookup(0.8, 0.84, 1.0, 0.29, False, "warm")
    cold = table.lookup(0.8, 0.84, 1.0, 0.29, False, "cold")
    assert warm.mean_k - cold.mean_k > 10.0                          # the two branches are far apart
    assert open_margin(warm) > 0.0                                   # open ocean: no freeze
    assert open_margin(cold) < 0.0                                   # ice-covered: no thaw
    exit_margin = min(open_margin(cold, 1.5), open_margin(warm))
    assert exit_margin < 0.0                                         # leaving the snowball needs more than this

    bright_warm = table.lookup(1.0, 0.84, 1.0, 0.29, False, "warm")
    bright_cold = table.lookup(1.0, 0.84, 1.0, 0.29, False, "cold")
    assert bright_cold.mean_k == pytest.approx(bright_warm.mean_k, abs=1.0)   # one solution above the tipping point
    assert min(open_margin(bright_cold, 1.5), open_margin(bright_warm)) > 0.0

    dim = table.lookup(0.6, 0.84, 1.0, 0.29, False, "warm")
    assert open_margin(dim) < 0.0                                    # below the tipping point only ice is stable


def test_initial_branch_follows_the_instellation():
    """Earth starts warm with its clock running, Venus forms in a runaway and Mars forms frozen."""
    branches = {}
    for name in ("earth_history.yaml", "venus_history.yaml", "mars_history.yaml"):
        model = model_of(name)
        y = initial_state(model.p)
        branches[name] = initial_modes(model, h.HISTORY_START_GYR, y, "mobile_lid")
    assert branches["earth_history.yaml"].climate == "warm"
    assert branches["earth_history.yaml"].clock_start == h.HISTORY_START_GYR    # the origin-of-life clock runs
    assert branches["venus_history.yaml"].climate == "runaway"
    assert branches["mars_history.yaml"].climate == "snowball"
    assert branches["mars_history.yaml"].clock_start is None


def test_freezing_and_thawing_move_life_and_are_logged():
    """A snowball drives life under the ice and a thaw brings it back; a runaway ends it."""
    model = model_of()
    y = initial_state(model.p)
    t = h.HISTORY_START_GYR
    living = replace(initial_modes(model, t, y, "mobile_lid"), life="surface", surface_life=True, life_origin=t)
    kinds = []

    def log(kind, detail, flagged):
        """Collect the logged event kinds."""
        kinds.append(kind)

    frozen = model.transition("snowball_onset", t, y, living, log)
    assert frozen.climate == "snowball" and frozen.life == "subsurface"
    assert kinds == ["snowball_onset", "life_retreat"]
    thawed = model.transition("snowball_exit", t, y, frozen, log)
    assert thawed.climate == "warm" and thawed.life in ("surface", "ocean")
    assert kinds[-1] == "life_emerges"

    boiled = model.transition("runaway_onset", t, y, thawed, log)
    assert boiled.climate == "runaway" and boiled.life == "none"
    dried = model.transition("ocean_loss", t, y, thawed, log)
    assert dried.climate == "dry" and dried.regime == "stagnant_lid"   # plate tectonics needs water
    assert "extinction" in kinds and "regime_change" in kinds


def test_event_log_collapses_a_fast_cycle():
    """Freezes in quick succession become one limit-cycle entry, but other events still get through."""
    log = EventLog()
    for i in range(6):
        t = 1.0 + 0.1 * i
        log.add(t, "snowball_onset", f"freeze {i}")
        if i == 3:
            log.add(t + 0.02, "dynamo_shutdown", "the core dynamo stops")
        log.add(t + 0.05, "snowball_exit", f"thaw {i}")
    kinds = [e.kind for e in log.events]
    assert kinds.count("limit_cycle") == 1
    assert kinds.count("snowball_onset") == h.CYCLE_COLLAPSE_AFTER - 1   # the first few are still reported in full
    assert kinds.count("snowball_exit") == h.CYCLE_COLLAPSE_AFTER - 1
    assert "dynamo_shutdown" in kinds                                    # unrelated events are never folded in
    summary = next(e for e in log.events if e.kind == "limit_cycle")
    assert "6 freezes" in summary.detail and "every 100 Myr" in summary.detail
    assert len(log.freezes) == 6
    assert log.cycling(1.6) and not log.cycling(3.0)


def test_event_log_keeps_separate_glaciations_apart():
    """Glaciations spread over a history stay individual events."""
    log = EventLog()
    for t in (1.0, 2.0, 3.0, 4.0):
        log.add(t, "snowball_onset", "the oceans freeze over")
        log.add(t + 0.2, "snowball_exit", "the ice melts back")
    kinds = [e.kind for e in log.events]
    assert kinds.count("snowball_onset") == 4
    assert "limit_cycle" not in kinds
    assert not log.cycling(4.6)


@pytest.mark.slow
def test_history_temperature_comes_from_the_carbon_cycle():
    """Earth's climate follows the weathering flux balance, not the drawn weathering target."""
    spec = load_spec(EXAMPLES / "earth_history.yaml")
    state, timeline = evolve(spec)
    assert not state.atmosphere.weathering_regulated
    drawn = state.atmosphere.weathering_target_k
    assert abs(drawn - state.atmosphere.surface_temperature_k) > 5.0   # the drawn target is not what is reached
    weathering = timeline.series["weathering"][-1]
    assert weathering == pytest.approx(timeline.series["outgassing"][-1], rel=0.05)
    assert weathering == pytest.approx(h.WEATHERING_EARTH, rel=0.3)


@pytest.mark.slow
def test_a_user_weathering_target_overrides_the_flux_balance():
    """A held weathering target regulates the final climate and is flagged as an override."""
    spec = load_spec(EXAMPLES / "earth_history.yaml")
    spec = spec.model_copy(update={"atmosphere": spec.atmosphere.model_copy(
        update={"weathering_target_k": 295.0})})
    state, _ = evolve(spec)
    assert state.atmosphere.weathering_regulated
    assert state.atmosphere.surface_temperature_k == pytest.approx(295.0, abs=h.CLIMATE_TARGET_MISS_K)
    assert "atmosphere.weathering_target_k" in [o.field for o in state.overrides]
