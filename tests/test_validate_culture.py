"""Verdict logic of scripts/validate_culture.py."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_culture.py"


@pytest.fixture(scope="module")
def vc():
    spec = importlib.util.spec_from_file_location("validate_culture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tolerance_tests(vc):
    assert vc.near(0.05)(0.38, 0.335) and not vc.near(0.05)(0.40, 0.335)
    assert vc.factor(1.5)(1.4, 1.0) and vc.factor(1.5)(0.7, 1.0) and not vc.factor(1.5)(0.6, 1.0)
    assert vc.between(11, 15)(12, 13) and not vc.between(11, 15)(16, 13)


def test_verdicts_count_checks_but_not_gaps_or_reports(vc):
    ok, bad = vc.near(0.1), vc.near(0.0)
    assert vc.verdict(ok, "check", 1.0, 1.05) == "pass"
    assert vc.verdict(bad, "check", 1.0, 1.05) == "FAIL"
    assert vc.verdict(bad, "gap", 1.0, 1.05) == "gap"
    assert vc.verdict(ok, "gap", 1.0, 1.05) == "ok"
    assert vc.verdict(None, "report", 1.0, None) == ""


def test_every_row_has_a_known_kind(vc):
    for label, earth, fmt, test, kind in vc.COLUMNS:
        assert kind in ("check", "gap", "report")
        assert (test is None) == (kind == "report") and (earth is None) <= (kind == "report")
        fmt.format(1.0)
