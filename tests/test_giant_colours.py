"""Giant colours from atmospheric composition (task 11.11)."""

import math

import numpy as np
import pytest

from worldgen import constants as c
from worldgen import generate
from worldgen import heuristics as h
from worldgen.giants import colours as gc
from worldgen.spec import PlanetSpec
from worldgen.state import GiantAppearance

# Disc colours of the Payne et al. (2025) geometric albedo spectra under a 5772 K star.
OBSERVED = {"Jupiter": "#c1c1b2", "Saturn": "#c4b99b", "Uranus": "#a0c3ca", "Neptune": "#8db7ca"}
CH4_SOLAR = gc.mole_fractions(1.0)["CH4"]
# gravity (m/s²), a (AU), Bond albedo in use, internal flux (W/m²), metallicity: Jupiter and Saturn
# as the model builds them, Uranus and Neptune with their observed internal heat and deep CH4
# (3% and 7%, Irwin et al. 2022)
REFERENCE = {"Jupiter": (24.79, 5.20, 0.46, 7.06, 2.3), "Saturn": (10.44, 9.54, 0.46, 2.61, 9.2),
             "Uranus": (8.87, 19.19, 0.30, 0.04, 0.03 / CH4_SOLAR),
             "Neptune": (11.15, 30.07, 0.30, 0.43, 0.07 / CH4_SOLAR)}


def _rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], float)


def _column(name):
    g, a, albedo, internal, z = REFERENCE[name]
    return gc.cloud_column(g, (1 - albedo) * c.S_EARTH / a**2 / 4, internal, z, 1, scatter=False)


@pytest.fixture(scope="module")
def reference():
    out = {}
    for name in REFERENCE:
        column, t_of_p = _column(name)
        out[name] = column, t_of_p, gc.layered_colours(column, t_of_p, 5772.0)
    return out


def test_saturation_pressures():
    # Ackerman & Marley (2001) quote NH3 saturation 1.09e-5 bar at 129 K, where Jupiter's NH3 saturates
    assert gc.saturation_bar("NH3", 129.0) == pytest.approx(1.09e-5, rel=0.01)
    # Grundy et al. (2023) Table 1: CH4 I at 0.43e-9 bar at 37 K (their constant-latent-heat fit)
    assert gc.saturation_bar("CH4", 37.0) == pytest.approx(0.43e-9, rel=0.15)
    assert math.isinf(gc.saturation_bar("CH4", 95.0))
    assert gc.saturation_bar("H2O", 273.16) == pytest.approx(6.11e-3, rel=0.01)


def test_mass_trend_metallicity():
    assert gc.atmospheric_metallicity(c.M_JUPITER, 0.0, 1) == pytest.approx(2.3, rel=0.7)
    jupiter = [gc.atmospheric_metallicity(c.M_JUPITER, 0.0, s) for s in range(200)]
    neptune = [gc.atmospheric_metallicity(0.054 * c.M_JUPITER, 0.0, s) for s in range(200)]
    assert np.median(jupiter) == pytest.approx(10 ** (8.84 - 8.47), rel=0.15)
    assert 40 < np.median(neptune) < 90                       # Lothringer et al. 2025 fit: ~63 at Neptune's mass
    assert gc.atmospheric_metallicity(c.M_JUPITER, 0.3, 4) == pytest.approx(
        2 * gc.atmospheric_metallicity(c.M_JUPITER, 0.0, 4), rel=0.01)


def test_one_bar_temperatures_and_decks(reference):
    # observed 165, 135, 76, 72 K; the model's ammonia cloud at ~0.6 and ~2 bar, methane at 1–3 bar
    for name, target, tol in (("Jupiter", 165, 5), ("Saturn", 135, 5), ("Uranus", 76, 5), ("Neptune", 72, 7)):
        assert reference[name][0].t_1bar_k == pytest.approx(target, abs=tol), name
    assert reference["Jupiter"][0].deck == reference["Saturn"][0].deck == "NH3"
    assert reference["Uranus"][0].deck == reference["Neptune"][0].deck == "CH4"
    assert 0.4 < reference["Jupiter"][0].deck_bar < 0.8 and 1.5 < reference["Saturn"][0].deck_bar < 2.7
    assert all(1.0 < reference[n][0].deck_bar < 3.0 for n in ("Uranus", "Neptune"))


def test_solar_system_colours(reference):
    for name, observed in OBSERVED.items():
        assert np.linalg.norm(_rgb(reference[name][2].disc) - _rgb(observed)) < 12, name
    distance = {pair: np.linalg.norm(_rgb(reference[pair[0]][2].disc) - _rgb(reference[pair[1]][2].disc))
                for pair in (("Jupiter", "Saturn"), ("Uranus", "Neptune"))}
    assert all(d > 6 for d in distance.values()), distance
    jupiter, saturn = (_rgb(reference[n][2].disc) for n in ("Jupiter", "Saturn"))
    assert saturn[0] - saturn[2] > jupiter[0] - jupiter[2]       # Saturn is the yellower
    uranus, neptune = (_rgb(reference[n][2].disc) for n in ("Uranus", "Neptune"))
    assert neptune[2] - neptune[0] > uranus[2] - uranus[0]       # Neptune is the bluer


def test_bands_and_storms(reference):
    jup, sat = reference["Jupiter"][2], reference["Saturn"][2]
    contrast = {n: np.linalg.norm(_rgb(look.zone) - _rgb(look.belt)) for n, look in (("J", jup), ("S", sat))}
    assert contrast["J"] > 2 * contrast["S"] > 0              # Saturn's haze mutes its belts
    red = _rgb(jup.vortex)
    assert red[0] - red[2] > _rgb(jup.belt)[0] - _rgb(jup.belt)[2]
    assert reference["Neptune"][2].vortex == ""


def test_metal_rich_giants_are_darker_and_cloud_deeper():
    g, a, albedo, internal, _ = REFERENCE["Jupiter"]
    absorbed = (1 - albedo) * c.S_EARTH / a**2 / 4
    poor, rich = (gc.cloud_column(g, absorbed, internal, z, 1, scatter=False)[0] for z in (1.0, 10.0))
    assert rich.deck_bar > poor.deck_bar and rich.haze_tau > poor.haze_tau
    dark = gc.hot_darkening(10.0)
    assert np.all(dark < 1) and np.all(np.diff(dark) < 0)      # red and near-infrared darken most
    assert np.allclose(gc.hot_darkening(1.0), 1.0)


def test_generated_giants_and_override():
    base = {"name": "g", "seed": 3, "star": {"mass_msun": 1.0, "age_gyr": 4.6, "metallicity_feh": 0.0},
            "orbit": {"semi_major_axis_au": 5.2}, "body": {"mass_mearth": 317.8, "envelope_mass_fraction": 0.9}}
    state = generate(PlanetSpec.model_validate(base))[0]
    assert state.appearance.cloud_deck == "NH3" and state.appearance.vortex_colour
    assert state.provenance["body.atmospheric_metallicity"] == "heuristic"
    base["body"]["atmospheric_metallicity"] = 30.0
    rich = generate(PlanetSpec.model_validate(base))[0]
    assert rich.appearance.atmospheric_metallicity == 30.0
    assert rich.provenance["body.atmospheric_metallicity"] == "user"
    assert rich.appearance.cloud_deck_bar > state.appearance.cloud_deck_bar
    hot = dict(base, orbit={"semi_major_axis_au": 0.3})
    assert generate(PlanetSpec.model_validate(hot))[0].appearance.cloud_deck == ""


def test_old_appearance_loads():
    fields = dict(sudarsky_class="I", effective_temperature_k=120.0, internal_temperature_k=100.0,
                  internal_flux_w_m2=5.0, bond_albedo=0.4, colour="#c0c0b0", zone_colour="#c8c8c0",
                  belt_colour="#b0a090", visible_albedo=0.5, regime="banded", jet_speed_m_s=40.0,
                  rhines_length_m=1e7, jet_count=20, equatorial_jet="prograde", jet_half_width_deg=8.0,
                  hot_spot_offset_deg=0.0, polar_vortices=False)
    old = GiantAppearance(**fields)
    assert old.vortex_colour == "" and old.atmospheric_metallicity == 1.0


def test_optics_table():
    o = gc.optics()
    n = len(o["wavelength_nm"])
    assert o["wavelength_nm"][0] == 380 and o["wavelength_nm"][-1] == 1000
    assert o["ch4_cross_section_m2"].shape == (n,) and o["ch4_cross_section_m2"].max() > 5e-28
    for kind in ("haze_upper", "haze_methane", "haze_troposphere", "chromophore", "cloud"):
        assert np.all((o[f"{kind}_albedo"] > 0) & (o[f"{kind}_albedo"] <= 1))
        assert o[f"{kind}_extinction"][np.argmin(abs(o["wavelength_nm"] - 800))] == pytest.approx(1.0)
    absorption = o["chromophore_extinction"] * (1 - o["chromophore_albedo"])
    assert absorption[0] > 3 * absorption[-1]                          # the chromophore absorbs blue
    assert h.GIANT_CLOUD_ALBEDO < 1
