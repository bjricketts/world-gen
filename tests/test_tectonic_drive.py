"""The tectonic simulation driven by the integrated interior history (milestone 6)."""

import numpy as np
import pytest

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.evolve import evolve
from worldgen.grid import build_grid
from worldgen.spec import PlanetSpec, load_spec
from worldgen.surface.drive import TectonicDrive, plate_speed_m_myr

from .conftest import EXAMPLES


def drive_of(spreading, melt=None, span_gyr=1.0, age_gyr=4.57):
    """Return a drive whose spreading and melt run over the last ``span_gyr`` of a planet's life."""
    spreading = np.asarray(spreading, dtype=float)
    melt = spreading if melt is None else np.asarray(melt, dtype=float)
    times = np.linspace(age_gyr - span_gyr, age_gyr, spreading.size)
    return TectonicDrive.from_series({"time_gyr": times, "spreading": spreading, "melt": melt}, age_gyr)


def test_plate_speed_follows_the_creation_rate():
    """Earth's rate gives Earth's plate speed; the range is bounded at both ends."""
    earth = plate_speed_m_myr(1.0)
    assert earth == pytest.approx(h.PLATE_SPEED_EARTH_CM_YR / 100.0 * 1e6)      # 5 cm/yr = 50 km/Myr
    assert plate_speed_m_myr(3.0) == pytest.approx(3.0 * earth)                 # speed ∝ q², which is the rate
    lo, hi = h.PLATE_SPEED_SPREADING_RANGE
    assert plate_speed_m_myr(0.0) == pytest.approx(lo * earth)
    assert plate_speed_m_myr(100.0) == pytest.approx(hi * earth)


def test_the_drive_reads_the_history_through_the_window():
    """The drive interpolates the creation rate and melt over the simulated window."""
    drive = drive_of(spreading=[2.0, 1.5, 1.0], melt=[4.0, 2.0, 1.0])
    assert drive.spreading_now == 1.0 and drive.melt_now == 1.0
    assert drive.at(0.0) == (1.0, 1.0)
    assert drive.at(-1000.0) == (2.0, 4.0)                                      # the start of the window
    assert drive.at(-500.0)[0] == pytest.approx(1.5)                            # halfway
    assert drive.at(-9999.0) == (2.0, 4.0)                                      # before the history: held
    assert drive.speed_m_myr(-1000.0) == pytest.approx(2.0 * drive.speed_m_myr(0.0))


def test_the_simulated_span_follows_the_sea_floor_turnover():
    """Fast plates need less time to cover the same ground, slow ones more, within the range."""
    lo, hi = h.TECTONIC_DURATION_RANGE_MYR
    assert drive_of([1.0, 1.0]).duration_myr() == pytest.approx(h.TECTONIC_DURATION_MYR)
    assert drive_of([1.0, 2.0]).duration_myr() == pytest.approx(h.TECTONIC_DURATION_MYR / 2.0)
    assert drive_of([1.0, 10.0]).duration_myr() == lo                           # clamped for a fast planet
    assert drive_of([1.0, 0.05]).duration_myr() == hi                           # and for a slow one
    assert drive_of([1.0, 1.0]).turnover_myr() == pytest.approx(h.SEAFLOOR_TURNOVER_EARTH_MYR)
    assert drive_of([1.0, 4.0]).turnover_myr() == pytest.approx(h.SEAFLOOR_TURNOVER_EARTH_MYR / 4.0)


@pytest.mark.slow
def test_a_hot_interior_drives_faster_plates_and_more_volcanism():
    """The same planet run with a hot and a cool history differs in speed, sea-floor age and hotspots."""
    from worldgen.tectonics import run

    grid = build_grid(4000)
    settings = dict(radius_m=c.R_EARTH, activity=1.0, continental_fraction=0.3, relief=1.0,
                    start="cratons", duration_myr=200.0, seed=5)
    hot = run(grid, drive=drive_of([3.0, 3.0], melt=[9.0, 9.0]), **settings)
    cool = run(grid, drive=drive_of([0.4, 0.4], melt=[0.2, 0.2]), **settings)
    plain = run(grid, **settings)

    assert np.median(hot.crust.ocean_age) < np.median(cool.crust.ocean_age)     # fast plates keep the floor young
    assert len(hot.hotspots) > len(cool.hotspots)
    assert np.median(plain.crust.ocean_age) == pytest.approx(np.median(
        run(grid, **settings).crust.ocean_age))                                 # no drive: unchanged and repeatable


@pytest.mark.slow
def test_earth_is_driven_at_its_own_rates():
    """Earth's history asks for Earth's plate speed, melting and simulated span."""
    _, timeline = evolve(load_spec(EXAMPLES / "earth_history.yaml"))
    drive = TectonicDrive.from_series(timeline.series, c.AGE_SUN_GYR)
    assert drive.spreading_now == pytest.approx(1.0, abs=0.15)
    assert drive.melt_now == pytest.approx(1.0, abs=0.15)
    assert drive.speed_m_myr() / 1e3 == pytest.approx(50.0, abs=8.0)            # km/Myr, Earth's ~5 cm/yr
    assert drive.duration_myr() == pytest.approx(h.TECTONIC_DURATION_MYR, rel=0.2)
    assert drive.turnover_myr() == pytest.approx(h.SEAFLOOR_TURNOVER_EARTH_MYR, rel=0.2)


@pytest.mark.slow
def test_a_history_world_records_what_drove_its_tectonics(world_cache):
    """A history-mode surface reports the plate speed it used and says so in the report."""
    spec = load_spec(EXAMPLES / "earth_history.yaml")
    world = world_cache(spec, "preview")
    features = world.state.surface.features
    assert features["plate_speed_km_myr"] == pytest.approx(50.0, abs=10.0)
    assert 200.0 <= features["simulated_myr"] <= 800.0
    assert any("from the planet's history" in i.message for i in world.state.issues)
    assert world.state.provenance["surface.tectonics_duration_myr"] == "derived"


@pytest.mark.slow
def test_a_snapshot_world_keeps_the_fixed_span(world_cache):
    """Without a history the simulation keeps the activity index and the default span."""
    spec = PlanetSpec(name="Snapshot world", seed=2, priors={"archetype": "temperate"})
    world = world_cache(spec, "preview")
    features = world.state.surface.features
    assert features["simulated_myr"] == pytest.approx(h.TECTONIC_DURATION_MYR, abs=h.TECTONIC_STEP_MYR)
    assert world.state.provenance["surface.tectonics_duration_myr"] != "derived"
    assert not any("from the planet's history" in i.message for i in world.state.issues)
