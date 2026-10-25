"""Cloud tops of giant planets: winds, bands, vortices, poles, locked giants, and their maps and exports (task 11.3)."""

import math

import numpy as np
import pytest

from worldgen import generate, generate_world, heuristics as h, load_world, save_world
from worldgen.giants.clouds import (CloudModel, _polar_banding, build_cloud_model, cloud_model, deformation_radius,
                                    evaluate, wind_nodes)
from worldgen.render.raster import lonlat_vectors
from worldgen.spec import PlanetSpec


def giant_spec(a=5.2, mass=317.8, radius=11.21, envelope=0.9, rotation=9.93, seed=1):
    """Return the spec of a giant around a Sun-like star."""
    orbit = {"semi_major_axis_au": a}
    if rotation is not None:
        orbit["rotation_period_h"] = rotation
    return PlanetSpec.model_validate({"name": "Giant", "seed": seed, "star": {"mass_msun": 1.0, "age_gyr": 4.6},
                                      "orbit": orbit, "body": {"mass_mearth": mass, "radius_rearth": radius,
                                                               "envelope_mass_fraction": envelope}})


@pytest.fixture(scope="module")
def jupiter():
    """Return Jupiter's state and cloud model."""
    state, _ = generate(giant_spec())
    return state, build_cloud_model(state)


@pytest.fixture(scope="module")
def neptune():
    """Return Neptune's state and cloud model."""
    state, _ = generate(giant_spec(30.1, 17.1, 3.88, 0.15, 16.11))
    return state, build_cloud_model(state)


def ring(lat_deg, n=720):
    """Return unit vectors along a circle of latitude."""
    lat, lon = math.radians(lat_deg), np.radians(np.linspace(-180, 180, n, endpoint=False))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.full(n, math.sin(lat))])


def test_deformation_radius_matches_jupiter(jupiter):
    """The calibrated deformation radius is the ~1800 km Sankar et al. (2024) derive for Jupiter."""
    state, _ = jupiter
    assert deformation_radius(state) / 1e3 == pytest.approx(1800, rel=0.1)


def test_wind_profiles(jupiter, neptune):
    """Jets alternate, Jupiter's equator peaks near 100 m/s with a dip, Neptune's equator runs at −400 m/s."""
    _, jm = jupiter
    _, nm = neptune
    u = np.asarray(jm.wind_m_s)
    assert max(u) == pytest.approx(h.WIND_EQUATOR_JUPITER_M_S)
    eq = jm.wind_lat_deg.index(0.0)
    assert u[eq] < u[eq + 1]                                       # double-peaked equatorial jet
    off = u[eq + 2:-1]
    assert np.all(np.sign(off[:-1]) != np.sign(off[1:]))          # alternating jets
    assert min(nm.wind_m_s) == pytest.approx(h.WIND_EQUATOR_ICE_M_S)
    assert max(nm.wind_m_s) == pytest.approx(h.WIND_HIGH_LATITUDE_ICE_M_S)


def test_wind_nodes_single_peak():
    """Without a double peak the equatorial node carries the full equatorial speed."""
    lat, u = wind_nodes(np.radians([0.0, 10.0, 20.0]), -300.0, 200.0, double_peak=False)
    assert u[list(lat).index(0.0)] == -300.0
    assert list(lat) == sorted(lat)


def test_zones_lie_equatorward_of_prograde_jets(jupiter):
    """Just equatorward of each northern prograde jet the band phase is a zone (+), just poleward a belt (−)."""
    _, m = jupiter
    lat, u = np.asarray(m.wind_lat_deg), np.asarray(m.wind_m_s)
    checked = 0
    for k in range(len(lat)):
        if 5 < lat[k] < 60 and u[k] > 0:
            below = evaluate(m, ring(lat[k] - 1.0), ("band",))["band"].mean()
            above = evaluate(m, ring(lat[k] + 1.0), ("band",))["band"].mean()
            assert below > 0 > above
            checked += 1
    assert checked >= 2


def test_temperature_and_relief_follow_bands(jupiter):
    """Zones are colder and higher than belts."""
    _, m = jupiter
    _, _, vec = lonlat_vectors(180, 360)
    f = evaluate(m, vec, ("band", "temperature", "relief"))
    zone, belt = f["band"] > 0.5, f["band"] < -0.5
    assert f["temperature"][zone].mean() < f["temperature"][belt].mean()
    assert f["relief"][zone].mean() > f["relief"][belt].mean()


def test_vortex_statistics(jupiter):
    """Sizes and aspect ratios follow the log-normal catalogue values; anticyclones sit in zones."""
    state, m = jupiter
    v = m.vortices
    kinds = np.array(v["kind"])
    ordinary = np.isin(kinds, ["white", "red", "brown", "dark"])
    assert ordinary.sum() > 50
    aspect = np.asarray(v["a_m"])[ordinary] / np.asarray(v["b_m"])[ordinary]
    assert np.median(aspect) == pytest.approx(h.VORTEX_ASPECT_MEDIAN, rel=0.2)
    size = np.asarray(v["a_m"])[ordinary] / m.deformation_radius_m
    assert 0.2 < np.median(size) < 1.5
    centres = np.column_stack([np.cos(np.radians(v["lat_deg"])) * np.cos(np.radians(v["lon_deg"])),
                               np.cos(np.radians(v["lat_deg"])) * np.sin(np.radians(v["lon_deg"])),
                               np.sin(np.radians(v["lat_deg"]))])
    lat_c = np.asarray(v["lat_deg"])
    band = evaluate(m, centres, ("band",))["band"]
    white = (kinds == "white") & (np.abs(lat_c) < 60)
    brown = (kinds == "brown") & (np.abs(lat_c) < 60)
    assert np.mean(band[white] > 0) > 0.7 and np.mean(band[brown] < 0) > 0.7


def test_great_oval_fills_a_zone():
    """A great oval sits in a tropical zone, no wider than the zone, with an eccentricity in the GRS range."""
    for seed in range(1, 12):
        state, _ = generate(giant_spec(seed=seed))
        m = build_cloud_model(state)
        kinds = m.vortices["kind"]
        if "great" in kinds:
            i = kinds.index("great")
            a, b, lat = m.vortices["a_m"][i], m.vortices["b_m"][i], m.vortices["lat_deg"][i]
            e = math.sqrt(1 - (b / a) ** 2)
            assert h.GREAT_OVAL_ECCENTRICITY[0] - 1e-9 <= e <= h.GREAT_OVAL_ECCENTRICITY[1] + 1e-9
            nodes = np.asarray(m.wind_lat_deg)
            k = np.searchsorted(nodes, lat) - 1
            width = math.radians(nodes[k + 1] - nodes[k]) * m.radius_m
            assert 2 * b <= width
            return
    pytest.fail("no great oval in eleven seeds")


def test_polar_cyclone_ring_count():
    """Circumpolar rings hold as many cyclones as fit 5.87 radii of maximum wind apart (Gavriel & Kaspi 2021)."""
    for seed in range(1, 12):
        state, _ = generate(giant_spec(seed=seed))
        m = build_cloud_model(state)
        ring_lat = [x for x, k in zip(m.vortices["lat_deg"], m.vortices["kind"])
                    if k == "polar" and abs(x) < 89]
        if ring_lat:
            r_max = h.POLAR_CYCLONE_RMAX_LD * m.deformation_radius_m
            colat = math.radians(90 - h.POLAR_RING_LATITUDE_DEG)
            expected = int(2 * math.pi * m.radius_m * math.sin(colat) / (h.POLAR_RING_SPACING * r_max))
            north = sum(1 for x in ring_lat if x > 0)
            assert north in (0, expected) and 4 <= expected <= 14
            return
    pytest.fail("no circumpolar ring in eleven seeds")


def test_polar_banding_fades_except_to_a_hexagon():
    """Bands fade into the chaotic polar region above ~70°, but continue to a hexagon's jet."""
    m = CloudModel(seed=1, radius_m=6e7, regime="banded", ice_like=False, effective_temperature_k=100,
                   scale_height_m=2e4, deformation_radius_m=2e6)
    lat = np.radians([50.0, 80.0, -80.0])
    assert list(_polar_banding(m, lat)) == [1.0, 0.0, 0.0]
    m.hexagons = [{"lat_deg": 78.0, "rotation_deg": 0.0}]
    assert _polar_banding(m, lat)[1] == 1.0 and _polar_banding(m, lat)[2] == 0.0


def test_hexagon_shapes_the_polar_jet():
    """With a hexagon, the band phase along the jet's latitude varies with a six-fold period."""
    m = None
    for seed in range(1, 30):
        state, _ = generate(giant_spec(9.58, 95.2, 9.45, 0.85, 10.56, seed=seed))
        cand = build_cloud_model(state)
        if cand.hexagons:
            m = cand
            break
    assert m is not None
    hx = m.hexagons[0]
    sign = 1 if hx["lat_deg"] > 0 else -1
    phase = evaluate(m, ring(hx["lat_deg"] - sign * 3.0, 3600), ("band",))["band"]
    spectrum = np.abs(np.fft.rfft(phase - phase.mean()))
    assert np.argmax(spectrum[1:20]) + 1 in (6, 12, 18)


def test_locked_hot_jupiter():
    """A locked hot Jupiter is hottest east of the substellar point with a day–night contrast set by T_eq."""
    state, _ = generate(giant_spec(0.047, 220, 15.1, rotation=None))
    m = build_cloud_model(state)
    assert m.regime == "locked"
    temps = evaluate(m, ring(0.0, 3600), ("temperature",))["temperature"]
    lon = np.linspace(-180, 180, 3600, endpoint=False)
    smooth = np.convolve(temps, np.ones(201) / 201, mode="same")
    assert 0 < lon[np.argmax(smooth[200:-200]) + 200] < 60
    assert temps.max() - temps.min() == pytest.approx(m.day_night_k, rel=0.4)
    wind = evaluate(m, ring(0.0, 10), ("zonal_wind",))["zonal_wind"]
    assert np.all(wind > 0.9 * h.WIND_LOCKED_M_S)


def test_evaluation_is_deterministic_and_chunked(jupiter):
    """Evaluation gives the same values in one call and in chunks."""
    _, m = jupiter
    _, _, vec = lonlat_vectors(64, 128)
    whole = evaluate(m, vec)
    import worldgen.giants.clouds as clouds
    old, clouds.CHUNK = clouds.CHUNK, 1000
    try:
        parts = evaluate(m, vec)
    finally:
        clouds.CHUNK = old
    for key in whole:
        np.testing.assert_allclose(whole[key], parts[key], atol=1e-6)


def test_giant_world_maps_exports_and_round_trip(tmp_path):
    """A giant's world has cloud tops that map, export at full detail, save as format 8 and load back."""
    import matplotlib
    matplotlib.use("Agg")
    from worldgen.export import global_colours, global_raster, globe_meshes
    from worldgen.render import plot_map
    from worldgen.render.explore import available_fields

    world = generate_world(giant_spec(), "preview")
    assert world.cloud_tops and not world.has_ground
    assert available_fields(world)[0] == "clouds"
    plot_map(world, "elevation", "mollweide", width=300)
    plot_map(world, "winds", "orthographic", width=200)
    winds = global_raster(world, "winds", 256)
    assert winds.data.shape == (128, 256)
    assert winds.data.max() == pytest.approx(h.WIND_EQUATOR_JUPITER_M_S, rel=0.05)
    assert global_colours(world, "clouds", 128).data.shape == (64, 128, 3)
    meshes = globe_meshes(world, 0)
    assert len(meshes) == 6 and all(mm.layer == "terrain" for mm in meshes)
    with pytest.raises(ValueError):
        plot_map(world, "rainfall")
    with pytest.raises(ValueError):
        global_raster(world, "rivers", 64)
    save_world(world, tmp_path / "j")
    loaded = load_world(tmp_path / "j")
    assert cloud_model(loaded.surface) == cloud_model(world.surface)
    assert loaded.state.appearance == world.state.appearance


def test_zoom_and_culture_refuse_giants():
    """A giant has no ground to zoom into or to settle."""
    from worldgen.zoom import zoom_region

    world = generate_world(giant_spec(), "preview")
    with pytest.raises(ValueError, match="no solid surface"):
        zoom_region(world, 0.0, 0.0)


def test_cloud_surface_grid_fields(jupiter):
    """The grid fields match the model at the cells, and the elevation is the cloud-top relief."""
    world = generate_world(giant_spec(), "preview")
    ds = world.surface
    m = cloud_model(ds)
    f = evaluate(m, world.grid.points[:500], ("relief", "zonal_wind"))
    np.testing.assert_allclose(ds["elevation"].values[:500], f["relief"], rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(ds["zonal_wind"].values[:500], f["zonal_wind"], rtol=1e-4, atol=1e-3)
    assert not ds["ocean"].values.any()
    assert abs(float(ds["elevation"].max())) < 10 * m.scale_height_m
