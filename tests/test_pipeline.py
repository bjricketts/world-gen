import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from worldgen import PlanetSpec, candidates, format_report, generate
from worldgen import constants as c
from worldgen.cli import app
from worldgen.priors import ARCHETYPES, resolve
from worldgen.spec import field_role

from .conftest import EXAMPLES


def test_earth_end_to_end(earth_spec):
    state, timeline = generate(earth_spec)
    assert state.atmosphere.surface_temperature_k == pytest.approx(288, abs=2)
    assert state.atmosphere.surface_water == "liquid"
    assert state.bulk.radius_m / c.R_EARTH == pytest.approx(1.0, abs=0.01)
    assert state.occupiability.habitable_unaided
    assert not [i for i in state.issues if i.level != "info"]
    assert len(timeline.states) == 1
    assert "Earth" in format_report(state)


def test_user_values_are_kept(earth_spec):
    state, _ = generate(earth_spec)
    assert state.provenance["body.mass_mearth"] == "user"
    assert state.inputs["orbit.obliquity_deg"] == 23.44


def test_same_seed_same_planet():
    spec = PlanetSpec(seed=5)
    a, _ = generate(spec)
    b, _ = generate(spec)
    assert a.to_dict() == b.to_dict()


def test_fixing_one_value_does_not_change_other_draws():
    free = resolve(PlanetSpec(seed=3, priors={"archetype": "temperate"}))
    fixed = resolve(PlanetSpec(seed=3, priors={"archetype": "temperate"}, star={"mass_msun": 0.8}))
    assert free.values["body.mass_mearth"] == fixed.values["body.mass_mearth"]
    assert free.values["orbit.obliquity_deg"] == fixed.values["orbit.obliquity_deg"]


def test_ranges_are_respected():
    spec = PlanetSpec(seed=1, body={"mass_mearth": (2.0, 3.0)})
    state, _ = generate(spec)
    assert 2.0 <= state.bulk.mass_kg / c.M_EARTH <= 3.0
    assert state.provenance["body.mass_mearth"] == "user_range"


@pytest.mark.parametrize("name", list(ARCHETYPES))
def test_every_archetype_generates(name):
    for seed in range(5):
        state, _ = generate(PlanetSpec(seed=seed, priors={"archetype": name}))
        assert state.archetype == name
        assert 0.0 <= state.occupiability.score <= 1.0


def test_giant_has_no_surface():
    state, _ = generate(PlanetSpec(seed=0, priors={"archetype": "giant"}, body={"mass_mearth": 300.0}))
    assert state.bulk.planet_class == "gas giant"
    assert state.interior.tectonic_regime == "fluid"
    assert state.atmosphere.composition == "h2_he"


def test_red_dwarf_example_is_locked():
    from worldgen.spec import load_spec
    state, _ = generate(load_spec(EXAMPLES / "temperate_red_dwarf.yaml"))
    assert state.orbit.spin_state == "synchronous"


def test_history_mode_not_available_yet():
    with pytest.raises(NotImplementedError):
        generate(PlanetSpec(mode="history"))


def test_candidates_sorted():
    ranked = candidates(PlanetSpec(seed=10), 6)
    scores = [s.occupiability.score for s in ranked if not any(i.level == "error" for i in s.issues)]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("bad", [
    {"star": {"mass_msun": (2.0, 1.0)}},
    {"orbit": {"semi_major_axis_au": 1.0, "instellation_earth": 1.0}},
    {"body": {"mas_mearth": 1.0}},
    {"atmosphere": {"composition": "argon"}},
])
def test_invalid_specs_rejected(bad):
    with pytest.raises(ValidationError):
        PlanetSpec.model_validate(bad)


def test_field_roles():
    assert field_role("atmosphere.surface_pressure_bar", "snapshot") == "input"
    assert field_role("atmosphere.surface_pressure_bar", "history") == "target"
    assert field_role("body.mass_mearth", "history") == "input"


def test_cli(tmp_path):
    runner = CliRunner()
    assert "temperate" in runner.invoke(app, ["archetypes"]).output
    template = tmp_path / "planet.yaml"
    assert runner.invoke(app, ["template", str(template)]).exit_code == 0
    out = tmp_path / "state.yaml"
    result = runner.invoke(app, ["generate", str(template), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    result = runner.invoke(app, ["random", "--candidates", "3", "--seed", "4"])
    assert result.exit_code == 0 and "Best candidate" in result.output
    assert runner.invoke(app, ["generate", str(template), "--mode", "history"]).exit_code == 1


SUN = {"mass_msun": 1.0, "age_gyr": 4.57, "metallicity_feh": 0.0}
VENUS = PlanetSpec(
    name="Venus", star=SUN,
    orbit={"semi_major_axis_au": 0.723, "eccentricity": 0.007, "obliquity_deg": 2.6, "rotation_period_h": 5832.5},
    body={"mass_mearth": 0.815, "radius_rearth": 0.949, "water_mass_fraction": 2e-9},
    atmosphere={"surface_pressure_bar": 92.0, "bond_albedo": 0.76},
)
MARS = PlanetSpec(
    name="Mars", star=SUN,
    orbit={"semi_major_axis_au": 1.524, "eccentricity": 0.093, "obliquity_deg": 25.2, "rotation_period_h": 24.62},
    body={"mass_mearth": 0.107, "radius_rearth": 0.532, "water_mass_fraction": 2e-6},
    atmosphere={"surface_pressure_bar": 0.006, "bond_albedo": 0.25},
)


def test_venus_end_to_end():
    state, _ = generate(VENUS)
    assert state.archetype is None
    assert state.atmosphere.composition == "co2"
    assert state.atmosphere.surface_temperature_k == pytest.approx(737, rel=0.05)
    p = state.interior.regime_probabilities
    assert p["mobile_lid"] < p["stagnant_lid"]
    assert not state.occupiability.habitable_unaided


@pytest.mark.parametrize("seed", range(10))
def test_mars_end_to_end(seed):
    state, _ = generate(MARS.model_copy(update={"seed": seed}))
    assert 200 < state.atmosphere.surface_temperature_k < 225
    assert state.atmosphere.surface_water in ("ice", "none")


def test_archetype_mix_respects_fixed_values():
    from worldgen.priors.sampling import compatible_archetypes
    user = {"body.mass_mearth": 300.0}
    names = compatible_archetypes(user)
    assert "giant" in names and "temperate" not in names
    for seed in range(10):
        assert resolve(PlanetSpec(seed=seed, body={"mass_mearth": 300.0})).archetype == "giant"


def test_every_archetype_bounds_mass():
    assert all("body.mass_mearth" in a.priors for a in ARCHETYPES.values())
