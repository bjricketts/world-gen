"""Measures and rows of scripts/validate_organic.py."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_organic.py"


@pytest.fixture(scope="module")
def vo():
    spec = importlib.util.spec_from_file_location("validate_organic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_war_exponent_recovers_a_power_law(vo):
    rng = np.random.default_rng(0)
    x = (1 - rng.random(20000)) ** (-1 / 0.53)                 # pdf exponent 1.53
    conflicts = [SimpleNamespace(severity=v) for v in x]
    assert vo.war_exponent(conflicts) == pytest.approx(1.53, abs=0.03)
    assert np.isnan(vo.war_exponent(conflicts[:5]))


def test_state_measures_give_the_largest_share_and_effective_number(vo):
    snap = SimpleNamespace(owner=np.array([0, 0, 1, 1, -1, 2, 2, 2, 2, -1]))
    largest, effective = vo.state_measures(snap, 10.0, 100.0)
    assert largest == pytest.approx(0.4) and effective == pytest.approx(8 ** 2 / (2 ** 2 + 2 ** 2 + 4 ** 2))
    assert vo.state_measures(SimpleNamespace(owner=np.full(3, -1)), 10.0, 100.0) == (0.0, 0.0)


def test_rows_aggregate_with_verdicts_and_known_gaps(vo):
    m = {"hyde": [1.0] * 8, "origins": 12, "front_km_yr": 0.9, "arable": 0.62, "first_water_wind": 3000.0,
         "first_fossil": 200.0, "industrial_now": 1.0, "largest": [0.002, 0.04, 0.05, 0.18, 0.09, 0.23],
         "effective": [500, 200, 50, 30, 20], "lifetime_1e6": 220.0, "war_alpha": 1.5, "wars_per_century": 50.0,
         "tambora_per_century": 0.25, "flu_1600_1945": 1, "flu_expected": 1.2, "seconds": 60.0,
         "native_kept_100y": 0.1, "colonist_share_now": 0.3, "colony_people_now": 1e6, "independence_after": 200.0,
         "steppe_belt_km": 8000.0, "steppe_empire": 0.18, "herd_origins": 0.4, "contact_struck": 1e6,
         "colonists_on_struck": 0.4}
    table = vo.rows([m, dict(m, first_fossil=np.nan, steppe_belt_km=3000.0, steppe_empire=np.nan,
                             largest=[0.002, 0.04, 0.05, 0.18, 0.09, np.nan])])
    verdicts = {label: v for label, _, _, v in table}
    assert verdicts["farming origins"] == "pass" and verdicts["population / HYDE, 1 CE"] == "ok"
    assert verdicts["natives kept 100 yr after contact"] == ""                  # off-planet colonies: reported only
    fossil = next(r for r in table if r[0].startswith("first fossil"))
    assert "[1/2]" in fossil[2] and fossil[3] == "ok"
    assert "FAIL" not in verdicts.values()
    steppe = next(r for r in table if r[0].startswith("largest steppe empire"))
    assert "[1/2]" in steppe[2] and steppe[3] == "ok"                       # only worlds with a steppe belt count
    late = next(r for r in table if r[0] == "largest state / land, 1900")
    assert "[1/2]" in late[2]                                                 # only worlds at the era's energy level


def test_worlds_enter_state_rows_at_earths_energy_level(vo):
    from types import SimpleNamespace

    development = np.zeros((1, 3, 3))
    development[0, 0] = [3.0, 1.6, 0.2]                                         # energy per cell
    snap = SimpleNamespace(population=np.array([[0.0, 5.0, 1.0]]), development=development)
    assert vo.leading_level(snap) == 2                                          # the empty cell's level does not count
    assert vo.TAAGEPERA_LEVEL[115] == 3 and vo.TAAGEPERA_LEVEL[715] == 2
