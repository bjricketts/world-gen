"""Giant appearance: cloud class, albedo, colour and jets (task 11.2)."""

import pytest

from worldgen import generate, generate_world, heuristics as h, load_world, save_world
from worldgen.giants.appearance import class_albedo, sudarsky_class
from worldgen.report import format_report
from worldgen.spec import PlanetSpec

# Semi-major axis (AU), mass (M_E), radius (R_E), envelope fraction, rotation (h)
GIANTS = {
    "Jupiter": (5.2, 317.8, 11.21, 0.9, 9.93),
    "Saturn": (9.58, 95.2, 9.45, 0.85, 10.56),
    "Uranus": (19.2, 14.5, 4.01, 0.15, 17.24),
    "Neptune": (30.1, 17.1, 3.88, 0.15, 16.11),
}


def giant(name="Giant", a=5.2, mass=317.8, radius=11.21, envelope=0.9, rotation=9.93, **extra):
    """Return the state of a giant around a Sun-like star."""
    orbit = {"semi_major_axis_au": a}
    if rotation is not None:
        orbit["rotation_period_h"] = rotation
    spec = {"name": name, "seed": 1, "star": {"mass_msun": 1.0, "age_gyr": 4.6}, "orbit": orbit,
            "body": {"mass_mearth": mass, "radius_rearth": radius, "envelope_mass_fraction": envelope}, **extra}
    return generate(PlanetSpec.model_validate(spec))[0]


@pytest.fixture(scope="module")
def solar():
    """Return the four Solar System giants."""
    return {n: giant(n, *v) for n, v in GIANTS.items()}


def test_solar_system_giants_are_class_one(solar):
    """All four giants have ammonia or methane cloud decks (class I) with the measured Bond albedos.

    Jupiter 0.503 (Li et al. 2018) and Saturn 0.41 (Wang et al. 2024) share the class value 0.46.
    """
    for name, state in solar.items():
        assert state.appearance.sudarsky_class == "I", name
    assert solar["Jupiter"].appearance.bond_albedo == pytest.approx(0.503, abs=0.05)
    assert solar["Saturn"].appearance.bond_albedo == pytest.approx(0.41, abs=0.06)
    assert solar["Neptune"].appearance.bond_albedo == pytest.approx(0.30, abs=0.01)


def test_internal_heat_matches_jupiter_and_saturn(solar):
    """Internal temperatures follow the Cassini-era fluxes, 7.485 W/m² (107 K) and 2.84 W/m² (84 K).

    Li et al. (2018) and Wang et al. (2024); T_eff then comes out near 124.7 and 96.3 K.
    """
    assert solar["Jupiter"].appearance.internal_temperature_k == pytest.approx(107, rel=0.03)
    assert solar["Saturn"].appearance.internal_temperature_k == pytest.approx(84, rel=0.03)
    assert solar["Jupiter"].appearance.effective_temperature_k == pytest.approx(124.7, abs=2.0)
    assert solar["Saturn"].appearance.effective_temperature_k == pytest.approx(96.3, abs=2.0)


def test_band_counts_and_equatorial_jets(solar):
    """Gas giants have ~20 jets and a prograde equator; ice giants ~3 and a retrograde one (Lian & Showman 2010)."""
    for name in ("Jupiter", "Saturn"):
        ap = solar[name].appearance
        assert 10 <= ap.jet_count <= 40 and ap.equatorial_jet == "prograde", name
    for name in ("Uranus", "Neptune"):
        ap = solar[name].appearance
        assert 1 <= ap.jet_count <= 6 and ap.equatorial_jet == "retrograde", name


def test_colours(solar):
    """Jupiter is warmer in colour than its clear model; the ice giants are blue."""
    def rgb(hex_colour):
        """Return the channels of a hex colour."""
        return [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    r, _, b = rgb(solar["Jupiter"].appearance.colour)
    assert r > b
    r, _, b = rgb(solar["Neptune"].appearance.colour)
    assert b > r
    assert rgb(solar["Uranus"].appearance.colour) == pytest.approx(rgb(solar["Neptune"].appearance.colour), abs=20)


@pytest.mark.parametrize("a_au, expected", [(5.2, "I"), (1.5, "II"), (0.3, "III"), (0.05, "IV"), (0.015, "V")])
def test_class_sequence_with_distance(a_au, expected):
    """A Jupiter moved inward runs through the classes I to V."""
    assert giant(a=a_au, rotation=None).appearance.sudarsky_class == expected


def test_low_gravity_hot_giant_is_class_five():
    """A low-gravity giant above 900 K has silicate clouds (class V)."""
    assert sudarsky_class(1000, 8.0) == "V" and sudarsky_class(1000, 20.0) == "IV"


def test_albedo_blends_across_boundaries():
    """The Bond albedo moves continuously from class I to class II."""
    values = [class_albedo(t, 25.0, False) for t in range(130, 175)]
    assert max(abs(b - a) for a, b in zip(values, values[1:])) < 0.05
    assert values[0] == pytest.approx(h.GIANT_ALBEDO["I gas"]) and values[-1] == pytest.approx(0.81)


def test_hot_locked_giant_has_one_broad_jet():
    """A tidally locked hot Jupiter has one superrotating jet as wide as tens of degrees and an eastward hot spot."""
    ap = giant(a=0.047, mass=220, radius=15.1, rotation=None).appearance
    assert ap.regime == "locked" and ap.jet_count == 1 and ap.equatorial_jet == "prograde"
    assert 20 <= ap.jet_half_width_deg <= 60 and ap.hot_spot_offset_deg > 0 and ap.polar_vortices


def test_user_albedo_is_kept():
    """A Bond albedo in the spec overrides the class value."""
    state = giant(atmosphere={"bond_albedo": 0.5})
    assert state.appearance.bond_albedo == 0.5 and state.atmosphere.bond_albedo == 0.5


def test_report_and_round_trip(tmp_path, solar):
    """Giants report an appearance section and no water section; the appearance saves and loads."""
    text = format_report(solar["Jupiter"])
    assert "Appearance" in text and "\nWater\n" not in text
    world = generate_world(PlanetSpec.model_validate({"name": "J", "seed": 1, "body": {"mass_mearth": 317.8},
                                                      "orbit": {"semi_major_axis_au": 5.2}}), "preview")
    save_world(world, tmp_path / "j")
    assert load_world(tmp_path / "j").state.appearance == world.state.appearance
