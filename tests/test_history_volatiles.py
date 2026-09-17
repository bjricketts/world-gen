"""Volatile cycles in history mode: the air, the greenhouse, weathering, the water cycle and escape."""

import pytest

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.atmosphere import surface_temperature
from worldgen.evolve import evolve
from worldgen.evolve.snapshot import SnapshotEvolver
from worldgen.history import volatiles as vol
from worldgen.history.params import build_params
from worldgen.orbit import equilibrium_temperature
from worldgen.priors import resolve
from worldgen.spec import load_spec

from .conftest import EXAMPLES


def params_of(name="earth_history.yaml"):
    """Return the history constants of an example planet."""
    spec = load_spec(EXAMPLES / name).model_copy(update={"mode": "snapshot"})
    state = SnapshotEvolver().build(spec, spec.seed)
    inputs = resolve(spec, spec.seed)
    return build_params(inputs.get, state.bulk, state.orbit, state.star.mass_kg / c.M_SUN, 0.0)


def air_at(pressure_bar, co2_bar, wet=True):
    """Return an air record with the given total and CO₂ pressure."""
    return vol.Air(pressure_bar=pressure_bar, co2_bar=co2_bar, n2_bar=pressure_bar - co2_bar, o2_bar=0.0,
                   co2_mass=0.0, o2_fraction=0.0, co2_fraction=co2_bar / pressure_bar)


def test_air_partition_conserves_carbon():
    """Carbon splits between air and ocean; the ocean holds most of it on an Earth-like planet."""
    p = params_of()
    air = vol.partition_air(p, h.CARBON_SURFACE_EARTH, 1.0, h.NITROGEN_EARTH, 0.0)
    assert air.co2_bar == pytest.approx(h.CO2_EARTH_BAR, rel=0.15)      # pre-industrial 280 ppm
    assert air.pressure_bar == pytest.approx(0.78, rel=0.05)            # N₂ alone, before O₂
    assert air.co2_mass < 0.1 * h.CARBON_SURFACE_EARTH                  # most dissolved carbon is in the ocean
    dry = vol.partition_air(p, h.CARBON_SURFACE_EARTH, 0.0, h.NITROGEN_EARTH, 0.0)
    assert dry.co2_mass == pytest.approx(h.CARBON_SURFACE_EARTH)        # without an ocean it is all in the air
    assert vol.partition_air(p, 10.0, 1.0, 0.0, 0.0).co2_bar > air.co2_bar


def test_greenhouse_calibration_points():
    """τ(CO₂, P) matches Earth, the CO₂ forcing of doubling, Mars and Venus."""
    earth = vol.optical_depth(air_at(1.013, h.CO2_EARTH_BAR), wet=True)
    doubled = vol.optical_depth(air_at(1.013, 2 * h.CO2_EARTH_BAR), wet=True)
    assert earth == pytest.approx(h.TAU0["n2_o2"], rel=0.02)
    forcing = c.SIGMA_SB * 288.0**4 * (1.0 / (1 + 0.75 * earth) - 1.0 / (1 + 0.75 * doubled))
    assert forcing == pytest.approx(3.7, abs=0.4)                       # W/m² per doubling
    mars = vol.optical_depth(air_at(0.006, 0.0059), wet=False)
    assert surface_temperature(equilibrium_temperature(0.43, 0.25), mars) == pytest.approx(215, abs=8)
    venus = vol.optical_depth(air_at(92.0, 92.0), wet=False)
    assert surface_temperature(equilibrium_temperature(1.91, 0.76), venus) == pytest.approx(737, abs=20)
    archean = vol.optical_depth(air_at(1.1, 0.1), wet=True)
    assert 280 < surface_temperature(equilibrium_temperature(0.75, 0.3), archean) < 310


def _fluxes(p, **kwargs):
    """Return carbon fluxes for an Earth-like planet with the given conditions."""
    settings = dict(crust=h.CARBON_CRUST_EARTH, mantle=h.CARBON_MANTLE_EARTH, co2_bar=h.CO2_EARTH_BAR,
                    surface_k=288.0, land=0.29, ocean=True, liquid=1.0, regime="mobile_lid", melt=1.0,
                    spreading=1.0, heat_ratio=1.0, land_life=1.0)
    settings.update(kwargs)
    return vol.carbon_fluxes(p, **settings)


def test_earth_carbon_fluxes_balance():
    """At present Earth the weathering sink matches the volcanic and arc sources."""
    p = params_of()
    f = _fluxes(p)
    assert f.weathering == pytest.approx(h.WEATHERING_EARTH, rel=0.1)
    assert f.volcanic + f.arc == pytest.approx(h.WEATHERING_EARTH, rel=0.15)
    assert f.seafloor == pytest.approx(h.SEAFLOOR_WEATHERING_SHARE * h.WEATHERING_EARTH, rel=0.15)


def test_weathering_responds_to_climate_and_life():
    """Weathering speeds up when it is warm or CO₂-rich, stops without liquid water, and is capped by supply."""
    p = params_of()
    base = _fluxes(p).weathering
    assert _fluxes(p, surface_k=298.0).weathering > base
    assert _fluxes(p, co2_bar=10 * h.CO2_EARTH_BAR).weathering > base
    assert _fluxes(p, liquid=0.0).weathering == 0.0                     # a snowball stops weathering
    assert _fluxes(p, ocean=False).weathering == 0.0
    assert _fluxes(p, land_life=0.0).continental < _fluxes(p).continental
    hot = _fluxes(p, surface_k=360.0).continental
    assert hot < h.WEATHERING_SUPPLY_RATIO * h.WEATHERING_EARTH         # the rock supply limits it
    assert _fluxes(p, regime="stagnant_lid").seafloor == 0.0            # no new sea floor to carbonate


def test_water_cycle_matches_the_steady_state():
    """Regassing and degassing balance near Earth's split, and a stagnant lid only degasses."""
    p = params_of()
    degas, regas = vol.water_fluxes(p, 1.0, h.MANTLE_WATER_EARTH_OCEANS, "mobile_lid", 1.0, 1.0, True)
    assert degas == pytest.approx(h.WATER_DEGASSING_EARTH, rel=0.1)
    assert regas == pytest.approx(degas, rel=0.15)
    wetter = vol.water_fluxes(p, 2.0, h.MANTLE_WATER_EARTH_OCEANS, "mobile_lid", 1.0, 1.0, True)[1]
    assert wetter > regas                                               # deeper oceans push water into the mantle
    lid = vol.water_fluxes(p, 1.0, h.MANTLE_WATER_EARTH_OCEANS, "stagnant_lid", 1.0, 0.0, True)
    assert lid[1] == 0.0 and lid[0] > 0.0


def test_escape_limits():
    """Hydrogen escape is diffusion-limited on a temperate planet and energy-limited in a steam atmosphere."""
    p = params_of()
    flux = vol.xuv_flux(c.L_SUN * h.XUV_SATURATED, 1.0, p)              # saturated young Sun at 1 AU
    temperate = vol.water_escape(p, flux, 288.0, steam=False)
    moist = vol.water_escape(p, flux, 340.0, steam=False)
    steam = vol.water_escape(p, flux, 600.0, steam=True)
    assert temperate < 0.05                                            # oceans per Gyr: negligible
    assert moist > 10 * temperate
    assert steam > moist
    assert 4.57 / steam < 0.2                                          # a steam ocean is gone within 200 Myr
    assert vol.energy_limited(p, 2 * flux) == pytest.approx(2 * vol.energy_limited(p, flux))


def test_bulk_escape_follows_the_shoreline():
    """Mars loses heavy gases at the observed rate today; Earth and Venus keep theirs."""
    rates = {}
    for name, instellation in (("earth_history.yaml", 1.0), ("mars_history.yaml", 0.431),
                               ("venus_history.yaml", 1.91)):
        p = params_of(name)
        flux = vol.xuv_flux(c.L_SUN * 6.73e-6, 1.0, p)                 # present solar XUV
        rates[name] = vol.bulk_escape(p, flux, instellation) * 1e18 / c.SECONDS_PER_GYR   # kg/s
    assert 0.5 < rates["mars_history.yaml"] < 20.0                     # MAVEN: a few kg/s
    assert rates["earth_history.yaml"] < 0.1
    assert rates["venus_history.yaml"] < rates["mars_history.yaml"]


@pytest.mark.slow
def test_history_conserves_carbon_and_water():
    """Nothing is created or destroyed: carbon stays put and lost water leaves as escape."""
    spec = load_spec(EXAMPLES / "earth_history.yaml")
    state, timeline = evolve(spec)
    carbon = sum(state.inputs[k] for k in ("history.surface_carbon", "history.crust_carbon",
                                           "history.mantle_carbon"))
    start = h.CARBON_SURFACE_EARTH + h.CARBON_CRUST_EARTH + h.CARBON_MANTLE_EARTH
    assert carbon == pytest.approx(start, rel=0.01)
    water = (timeline.series["surface_water_oceans"][-1] + timeline.series["mantle_water_oceans"][-1])
    initial = spec.history.initial_water_mass_fraction * state.bulk.mass_kg / c.EARTH_OCEAN_MASS
    assert water == pytest.approx(initial, rel=0.02)                   # Earth loses almost no water


@pytest.mark.slow
def test_reference_planets_end_close_to_the_real_ones():
    """Earth, Mars and Venus end with the right air, water and temperature."""
    earth = evolve(load_spec(EXAMPLES / "earth_history.yaml"))[0]
    mars = evolve(load_spec(EXAMPLES / "mars_history.yaml"))[0]
    venus = evolve(load_spec(EXAMPLES / "venus_history.yaml"))[0]
    assert earth.atmosphere.surface_temperature_k == pytest.approx(288, abs=6)
    assert earth.atmosphere.surface_pressure_pa / c.BAR == pytest.approx(1.0, rel=0.2)
    assert earth.atmosphere.surface_water == "liquid"
    assert earth.water.surface_mass_kg / c.EARTH_OCEAN_MASS == pytest.approx(1.0, rel=0.3)
    assert 1e-4 < earth.inputs["history.co2_bar"] < 2e-3               # a few hundred ppm of CO₂

    assert mars.atmosphere.surface_water == "ice"
    assert mars.atmosphere.surface_pressure_pa / c.BAR < 0.1           # most of the air is gone
    assert not mars.interior.magnetic_field

    assert venus.atmosphere.surface_water == "none"                    # the ocean is lost in the runaway
    assert venus.atmosphere.surface_temperature_k > 650
    assert 40 < venus.atmosphere.surface_pressure_pa / c.BAR < 150
    assert any(e.kind == "runaway_onset" for e in venus.events)
    assert any(e.kind == "ocean_loss" for e in venus.events)
