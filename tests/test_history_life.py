"""The biosphere in time: productivity, the oxygen budget, biotic methane and biotic weathering."""

import pytest

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.evolve import evolve
from worldgen.evolve.snapshot import SnapshotEvolver
from worldgen.history import life as life_model
from worldgen.history import volatiles as vol
from worldgen.history.params import build_params
from worldgen.priors import resolve
from worldgen.spec import load_spec

from .conftest import EXAMPLES

EARTH_MELT_AT_1_GYR = 3.5      # melt production relative to present Earth in the integrated history


def params_of(name="earth_history.yaml"):
    """Return the history constants of an example planet."""
    spec = load_spec(EXAMPLES / name).model_copy(update={"mode": "snapshot"})
    state = SnapshotEvolver().build(spec, spec.seed)
    inputs = resolve(spec, spec.seed)
    return build_params(inputs.get, state.bulk, state.orbit, state.star.mass_kg / c.M_SUN, 0.0)


@pytest.fixture(scope="module")
def earth_history():
    """Earth integrated from formation, shared by the tests that read its history."""
    return evolve(load_spec(EXAMPLES / "earth_history.yaml"))


def test_productivity_follows_the_life_the_climate_and_the_land():
    """Ocean life alone reaches about half of Earth's burial; land life makes up the rest once it spreads."""
    p = params_of()
    assert life_model.productivity(p, "none", 0.0, 288.0, 0.29, 0.9) == (0.0, 0.0)
    assert life_model.productivity(p, "subsurface", 1.0, 288.0, 0.29, 0.9) == (0.0, 0.0)
    ocean = life_model.productivity(p, "ocean", 1.0, 288.0, 0.29, 0.9)
    assert ocean == (pytest.approx(h.OCEAN_PRODUCTIVITY_SHARE, rel=0.1), 0.0)   # no land biosphere in the ocean
    assert life_model.productivity(p, "surface", 0.0, 288.0, 0.29, 0.9)[0] == pytest.approx(ocean[0])
    total, land = life_model.productivity(p, "surface", 1.0, 288.0, 0.29, 0.9)
    assert total == pytest.approx(1.0, rel=0.05) and land == pytest.approx(1.0, rel=0.05)
    assert life_model.productivity(p, "surface", 1.0, 330.0, 0.29, 0.9)[1] < 0.3 * land   # too hot for land life
    assert life_model.productivity(p, "ocean", 1.0, 288.0, 0.01, 0.95)[0] > ocean[0]      # an ocean world
    young = life_model.productivity(p, "surface", 1.0, 288.0, 0.29, 0.9, established=0.5)
    assert young[0] == pytest.approx(0.5 * total)                                         # a biosphere grows in


def test_oxygen_rises_only_once_the_reductant_flux_falls():
    """A young mantle's reductants swamp burial; the same biosphere oxygenates the air later."""
    p = params_of()
    early = life_model.oxygen_rate(p, 1.0, 1e-6, 0.62, EARTH_MELT_AT_1_GYR, 288.0, 0.2, 1.0, 1.0)
    late = life_model.oxygen_rate(p, 4.0, 1e-6, 0.62, 1.2, 288.0, 0.2, 1.0, 1.0)
    assert early < 0.5 * late                          # the same life buries the same carbon, the sink is smaller
    dead = life_model.oxygen_rate(p, 4.0, 1e-6, 0.0, 1.2, 288.0, 0.2, 1.0, 1.0)
    assert dead < 0.0                                  # without life there is no source at all
    present = life_model.oxygen_rate(p, c.AGE_SUN_GYR, h.OXYGEN_MASS_EARTH, 1.0, 1.0, 288.0, 0.29, 1.0, 1.0)
    assert abs(present) < 0.1 * h.OXYGEN_BURIAL_EARTH   # present Earth is near its steady state


def test_oxygen_sinks_grow_with_rock_heat_and_sea_floor():
    """Exposed land, fresh sea floor and a hot surface all take up O₂."""
    p = params_of()
    def rate(**kwargs):
        """Return dO₂/dt for present Earth with some conditions changed."""
        settings = dict(t_gyr=c.AGE_SUN_GYR, oxygen=h.OXYGEN_MASS_EARTH, total_productivity=1.0, melt=1.0,
                        surface_k=288.0, land=0.29, heat_ratio=1.0, spreading=1.0)
        settings.update(kwargs)
        return life_model.oxygen_rate(p, **settings)

    assert rate(land=0.8) < rate()                            # more rock to weather
    assert rate(heat_ratio=3.0) < rate()                      # faster resurfacing exposes more of it
    assert rate(land=0.0, spreading=1.0) < rate(land=0.0, spreading=0.0)    # an ocean world still has a sink
    # A Venus-like crust takes up the oxygen left behind by escaping hydrogen on OXYGEN_CRUST_SINK_GYR.
    dead = rate(total_productivity=0.0)
    assert rate(surface_k=750.0, total_productivity=0.0) - dead == pytest.approx(
        -h.OXYGEN_MASS_EARTH / h.OXYGEN_CRUST_SINK_GYR, rel=0.1)
    assert dead < -h.OXYGEN_BURIAL_EARTH                       # and without life nothing replaces it


def test_methane_tracks_the_oxygen_record():
    """CH₄ runs from 10³ ppm in an anoxic air down to about 1 ppm at present O₂."""
    p = params_of()
    assert life_model.methane_fraction(p, "none", 0.0) == 0.0
    assert life_model.methane_fraction(p, "ocean", 0.0) == pytest.approx(h.METHANE_BIOTIC)      # Archean
    proterozoic = life_model.methane_fraction(p, "ocean", 0.01)
    assert 3e-6 < proterozoic < 3e-5                                                           # order 10 ppm
    modern = life_model.methane_fraction(p, "surface", 0.21)
    assert modern == pytest.approx(7e-7, rel=0.3)                                              # pre-industrial
    assert life_model.methane_fraction(p, "ocean", 0.0, established=0.25) == pytest.approx(h.METHANE_BIOTIC / 4)


def test_land_life_speeds_continental_weathering():
    """Land biota multiply the weathering of exposed rock, up to the drawn biotic factor."""
    p = params_of()
    settings = dict(crust=h.CARBON_CRUST_EARTH, mantle=h.CARBON_MANTLE_EARTH, co2_bar=h.CO2_EARTH_BAR,
                    surface_k=288.0, land=0.29, ocean=True, liquid=1.0, regime="mobile_lid", melt=1.0,
                    spreading=1.0, heat_ratio=1.0)
    full = vol.carbon_fluxes(p, land_life=1.0, **settings).continental
    bare = vol.carbon_fluxes(p, land_life=0.0, **settings).continental
    assert full == pytest.approx(h.WEATHERING_EARTH * (1.0 - h.SEAFLOOR_WEATHERING_SHARE)
                                 * h.CONTINENTAL_WEATHERING_SCALE, rel=0.15)
    assert 0.6 * p.biotic_weathering < full / bare <= p.biotic_weathering


@pytest.mark.slow
def test_earth_oxygenates_in_two_steps(earth_history):
    """Anoxic Archean, a first rise in the Proterozoic, a second with land life, 21% O₂ today."""
    state, timeline = earth_history
    kinds = [(e.kind, e.time_s / c.SECONDS_PER_GYR) for e in timeline.events]
    origin = next(t for kind, t in kinds if kind == "origin_of_life")
    rises = [t for kind, t in kinds if kind == "oxygenation"]
    colonised = next(t for kind, t in kinds if kind == "land_colonisation")
    assert origin < 1.0
    assert len(rises) == 2
    assert 2.0 < rises[0] < 3.2                        # the Great Oxidation, 2.4 Ga in the record
    assert rises[1] >= colonised                       # the second rise follows life onto the land

    s = timeline.series
    archean = s["time_gyr"].index(min(s["time_gyr"], key=lambda t: abs(t - 1.0)))
    assert s["o2_fraction"][archean] < 1e-5            # Archean upper limits
    assert s["ch4_fraction"][archean] == pytest.approx(h.METHANE_BIOTIC, rel=0.1)
    assert s["o2_fraction"][-1] == pytest.approx(0.21, abs=0.03)
    assert 3e-7 < s["ch4_fraction"][-1] < 3e-6
    assert state.atmosphere.composition == "n2_o2"


@pytest.mark.slow
def test_history_oxygen_replaces_the_age_rule(earth_history):
    """The biosphere's oxygen comes from the integrated air, not from the snapshot rule for its age."""
    state, timeline = earth_history
    assert state.biosphere.oxygen_fraction == pytest.approx(timeline.series["o2_fraction"][-1])
    assert state.provenance["biosphere.oxygen_fraction"] == "derived"
    assert state.provenance["biosphere.age_gyr"] == "derived"
    held = load_spec(EXAMPLES / "earth_history.yaml")
    held = held.model_copy(update={"biosphere": held.biosphere.model_copy(update={"oxygen_fraction": 0.05})})
    other, _ = evolve(held)
    assert other.biosphere.oxygen_fraction == pytest.approx(0.05)
    assert "biosphere.oxygen_fraction" in [o.field for o in other.overrides]


@pytest.mark.slow
def test_a_slower_reductant_decline_delays_the_oxidation():
    """Oxygenation waits on the mantle: stronger early reductants push it hundreds of Myr later."""
    spec = load_spec(EXAMPLES / "earth_history.yaml")
    spec = spec.model_copy(update={"history": spec.history.model_copy(update={"reductant_decay_gyr": 1.5})})
    state, timeline = evolve(spec)
    first = next(e.time_s / c.SECONDS_PER_GYR for e in timeline.events if e.kind == "oxygenation")
    assert first > 3.0
    assert timeline.series["o2_fraction"][-1] == pytest.approx(0.21, abs=0.03)   # it still ends oxygen-rich
