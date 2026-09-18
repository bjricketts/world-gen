"""History validation: prior quantiles, the reference-planet checks and the sensitivity sweep."""

import importlib.util
import sys
from pathlib import Path

import pytest

from worldgen import constants as c
from worldgen.priors.archetypes import BASE_PRIORS
from worldgen.priors.distributions import Choice, Fixed, LogUniform, TruncNormal, Uniform

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def script(name):
    """Import one of the scripts as a module."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_priors_report_their_quantiles():
    """Every distribution can name the value at a quantile, which is what the sweep varies over."""
    assert Uniform(10.0, 40.0).quantile(0.0) == 10.0
    assert Uniform(10.0, 40.0).quantile(0.25) == 17.5
    assert LogUniform(0.5, 2.0).quantile(0.5) == pytest.approx(1.0)      # the geometric mean
    assert LogUniform(0.5, 2.0).quantile(0.1) < LogUniform(0.5, 2.0).quantile(0.9)
    normal = TruncNormal(1.0, 0.2, 0.5, 2.0)
    assert normal.quantile(0.5) == pytest.approx(1.0, abs=0.01)
    assert normal.quantile(0.16) == pytest.approx(0.8, abs=0.02)         # one standard deviation below
    assert normal.quantile(0.0) == 0.5 and normal.quantile(1.0) == 2.0   # truncated at the ends
    assert Choice(("a", "b")).quantile(0.1) == "a" and Choice(("a", "b")).quantile(0.9) == "b"
    assert Fixed(3.0).quantile(0.9) == 3.0
    for name, prior in BASE_PRIORS.items():
        assert prior.quantile(0.5) is not None, name


def test_the_sweep_covers_every_history_parameter():
    """Each poorly constrained history parameter has a prior to sweep and a way into the spec."""
    from worldgen.spec import PlanetSpec

    sensitivity = script("history_sensitivity")
    drawn = [n for n in BASE_PRIORS if n.startswith("history.")]
    assert set(drawn) <= set(sensitivity.PARAMETERS)
    assert "history.carbon_inventory" in sensitivity.PARAMETERS      # has no prior of its own
    spec = PlanetSpec(name="sweep")
    for name in sensitivity.PARAMETERS:
        value = sensitivity.prior_of(name).quantile(0.9)
        changed = sensitivity.with_parameter(spec, name, value)
        section, field = name.split(".", 1)
        assert getattr(getattr(changed, section), field) == value


def test_influence_ranks_the_largest_relative_spread():
    """A parameter's influence is its largest move measured against what counts as a large move."""
    sensitivity = script("history_sensitivity")
    row = sensitivity.Row(planet="earth", parameter="history.outgassing_efficiency", values=[0.5, 1.0, 2.0])
    row.spreads = {o.label: 0.0 for o in sensitivity.OUTCOMES}
    assert row.influence == 0.0
    row.spreads["T [K]"] = 20.0                                      # two "large moves" of 10 K
    assert row.influence == pytest.approx(2.0)
    row.spreads["O₂"] = 0.5                                          # ten large moves of 0.05
    assert row.influence == pytest.approx(10.0)


def test_logarithmic_outcomes_have_a_floor():
    """An outcome that can reach zero is floored, so an empty planet does not dominate the ranking."""
    sensitivity = script("history_sensitivity")
    pressure = next(o for o in sensitivity.OUTCOMES if o.label == "P [dex]")

    class Air:
        surface_pressure_pa = 0.0

    class Empty:
        atmosphere = Air()

    assert pressure.value(Empty()) == pytest.approx(-6.0)


def test_reference_checks_compare_against_the_real_planets():
    """The check helpers accept a value only inside their tolerance, and the tables cover each planet."""
    validate = script("validate_history")
    assert validate.within(0.25)(1.2, 1.0) and not validate.within(0.25)(1.3, 1.0)
    assert validate.near(6.0)(283.0, 288.0) and not validate.near(6.0)(281.0, 288.0)
    assert validate.between(0.1, 5.0)(0.7, 0.7) and not validate.between(0.1, 5.0)(9.0, 0.7)
    assert validate.before(1.5)(0.4, None) and not validate.before(1.5)(None, None)
    assert set(validate.PLANETS) == {"earth", "mars", "venus"}
    for name, (spec_name, checks) in validate.PLANETS.items():
        assert (validate.EXAMPLES / spec_name).exists()
        assert any("temperature" in check.label for check in checks)


@pytest.mark.slow
def test_the_reference_planets_pass_their_checks():
    """Earth, Mars and Venus meet every observational check the validation script makes."""
    from worldgen.evolve import evolve
    from worldgen.spec import load_spec

    validate = script("validate_history")
    missed = []
    for name, (spec_name, checks) in validate.PLANETS.items():
        state, timeline = evolve(load_spec(validate.EXAMPLES / spec_name))
        missed += [f"{name}: {check.label}" for check in checks if not check.run(state, timeline)[2]]
        if name == "earth":
            assert state.epoch_s / c.SECONDS_PER_GYR == pytest.approx(4.57, abs=0.01)
            assert all(ok for _, _, _, ok in validate.compare_with_snapshot(state))
    assert missed == []
