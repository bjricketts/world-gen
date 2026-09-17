import numpy as np
import pytest

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen import interior
from worldgen.util import named_rng

from .conftest import EXAMPLES, rocky_body

# (mass M_E, radius R_E, core mass fraction, surface T, surface water, tidal W)
EARTH = (1.0, 1.0, 0.325, 288, True, 0.0)
VENUS = (0.815, 0.95, 0.30, 737, False, 0.0)
MARS = (0.107, 0.532, 0.24, 215, False, 0.0)
MOON = (0.0123, 0.273, 0.02, 250, False, 0.0)
MERCURY = (0.0553, 0.383, 0.70, 440, False, 0.0)
IO = (0.015, 0.286, 0.20, 110, False, 1e14)


def _interior(body_params, regime=None):
    m, r, cmf, t, water, tidal = body_params
    body = rocky_body(m, r, cmf)
    state, issues, _ = interior.build_interior(
        body, 4.57, 1.0, tidal, t, water, regime, None, named_rng(0, "r"), named_rng(0, "d"))
    return state, issues


def test_earth_heat_budget():
    state, _ = _interior(EARTH)
    assert state.radiogenic_power_w / 1e12 == pytest.approx(20, rel=0.15)
    assert state.surface_heat_flux_w_m2 == pytest.approx(0.087, rel=0.15)
    assert state.activity_index == pytest.approx(1.0, rel=0.05)


def _most_likely(params):
    state, _ = _interior(params)
    return max(state.regime_probabilities, key=state.regime_probabilities.get)


@pytest.mark.parametrize("body, expected", [
    (EARTH, "mobile_lid"), (MARS, "stagnant_lid"), (MOON, "inactive"),
    (MERCURY, "inactive"), (IO, "heat_pipe"),
])
def test_solar_system_regimes(body, expected):
    assert _most_likely(body) == expected


def test_venus_does_not_favour_plates():
    state, _ = _interior(VENUS)
    p = state.regime_probabilities
    assert p["mobile_lid"] < p["stagnant_lid"]
    assert p["mobile_lid"] < p["episodic"]


def test_probabilities_normalised():
    state, _ = _interior(EARTH)
    assert sum(state.regime_probabilities.values()) == pytest.approx(1.0)


def test_unlikely_override_warns():
    _, issues = _interior(MOON, regime="mobile_lid")
    assert any(i.kind == "conflict" for i in issues)


def test_radiogenic_heat_decays():
    assert interior.radiogenic_heat_per_kg(0.0) > 2 * interior.radiogenic_heat_per_kg(4.57)


# --- History mode: thermal evolution, dynamo and melting ---------------------------------------

def _run_thermal(spec_file, regime, t_end=c.AGE_SUN_GYR):
    """Integrate only the thermal equations of an example planet and return samples of the rates."""
    from scipy.integrate import solve_ivp
    from worldgen.evolve.snapshot import SnapshotEvolver
    from worldgen.history import thermal
    from worldgen.history.params import build_params
    from worldgen.priors import resolve
    from worldgen.spec import load_spec

    spec = load_spec(EXAMPLES / spec_file).model_copy(update={"mode": "snapshot"})
    state = SnapshotEvolver().build(spec, spec.seed)
    inputs = resolve(spec, spec.seed)
    params = build_params(inputs.get, state.bulk, state.orbit, state.star.mass_kg / c.M_SUN, 0.0)
    y0 = [params.mantle_temperature_k, thermal.initial_core_k(params, params.mantle_temperature_k)]
    solution = solve_ivp(
        lambda t, y: thermal.temperature_rates(params, thermal.thermal_rates(params, t, y[0], y[1], regime, 288.0)),
        (h.HISTORY_START_GYR, t_end), y0, method="BDF", dense_output=True, rtol=1e-6)
    times = np.linspace(h.HISTORY_START_GYR, t_end, 200)
    rates = [thermal.thermal_rates(params, t, *solution.sol(t), regime, 288.0) for t in times]
    return times, solution, rates


def test_earth_thermal_history():
    """Earth cools to its present heat flow, keeps its dynamo and grows an inner core late."""
    times, solution, rates = _run_thermal("earth_history.yaml", "mobile_lid")
    mantle, core = solution.sol(c.AGE_SUN_GYR)
    assert mantle == pytest.approx(1630, abs=80)                 # present potential temperature ~1620 K
    assert rates[-1].heat_flux_w_m2 == pytest.approx(0.09, rel=0.15)
    assert rates[-1].surface_w / 1e12 == pytest.approx(46, rel=0.15)
    assert all(r.dynamo for r in rates)                          # the geodynamo never stops
    inner_core_start = times[np.argmax([r.inner_core > 0 for r in rates])]
    assert 2.5 < inner_core_start < 4.5                          # the inner core is younger than the planet
    assert max(r.melt for r in rates) > rates[-1].melt           # melt production peaks early
    assert rates[-1].melt == pytest.approx(1.0, abs=0.3)         # Earth's present melt production is the reference
    assert mantle < max(solution.sol(t)[0] for t in times)       # the mantle warms first, then cools


def test_mars_loses_its_dynamo():
    """A small stagnant-lid planet cools its core early and has no inner core."""
    times, solution, rates = _run_thermal("mars_history.yaml", "stagnant_lid")
    dynamo = [r.dynamo for r in rates]
    assert dynamo[0] and not dynamo[-1]
    shutdown = times[np.argmax(~np.array(dynamo))]
    assert 0.1 < shutdown < 1.0                                  # Mars's field is gone by ~4 Ga
    assert rates[-1].inner_core == 0.0
    assert rates[-1].heat_flux_w_m2 * 1e3 == pytest.approx(20, abs=8)   # InSight: ~20 mW/m²
    assert rates[-1].melt < 0.1                                  # little volcanism today


def test_regime_and_mass_scaling():
    """Stagnant lids run hotter and outgas less than plates; bigger planets stay more active."""
    from worldgen.history import thermal
    from worldgen.history.params import build_params
    from worldgen.priors import resolve
    from worldgen.evolve.snapshot import SnapshotEvolver
    from worldgen.spec import load_spec

    spec = load_spec(EXAMPLES / "earth_history.yaml").model_copy(update={"mode": "snapshot"})
    state = SnapshotEvolver().build(spec, spec.seed)
    inputs = resolve(spec, spec.seed)
    params = build_params(inputs.get, state.bulk, state.orbit, 1.0, 0.0)
    plates = thermal.thermal_rates(params, 4.57, 1650.0, 4100.0, "mobile_lid", 288.0)
    lid = thermal.thermal_rates(params, 4.57, 1650.0, 4100.0, "stagnant_lid", 288.0)
    dead = thermal.thermal_rates(params, 4.57, 1650.0, 4100.0, "inactive", 288.0)
    assert lid.surface_w < plates.surface_w
    assert lid.melt < plates.melt and lid.spreading == 0.0
    assert dead.melt == 0.0 and dead.activity < plates.activity
    hot = thermal.thermal_rates(params, 4.57, 1900.0, 4100.0, "mobile_lid", 288.0)
    assert hot.surface_w > plates.surface_w and hot.melt > plates.melt


@pytest.mark.slow
def test_regime_rules_over_time():
    """The regime follows the heat budget: a dead moon goes inactive, a strongly tidally heated body heat-pipes."""
    from worldgen.evolve import evolve
    from worldgen.spec import PlanetSpec, load_spec

    base = load_spec(EXAMPLES / "mars_history.yaml").model_dump()

    def run(name, **body):
        """Return the final state of a Mars-like history with different bulk properties."""
        data = dict(base, name=name, interior={}, body={**base["body"], **body})
        return evolve(PlanetSpec.model_validate(data))[0]

    moon = run("moon", mass_mearth=0.0123, radius_rearth=0.273)
    io = run("io", mass_mearth=0.015, radius_rearth=0.286, tidal_heating_w=1e14)
    assert moon.interior.tectonic_regime == "inactive"
    assert not moon.interior.magnetic_field
    assert io.interior.tectonic_regime == "heat_pipe"
    assert io.interior.surface_heat_flux_w_m2 > 10 * c.EARTH_HEAT_FLUX
