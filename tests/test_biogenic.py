"""Sediment the sea makes itself: platforms, shelf and slope carbonate, pelagic rain."""

import numpy as np

from worldgen import heuristics as h
from worldgen.tectonics.biogenic import biogenic_thickness

TOP = h.CARBONATE_PLATFORM_TOP_M
FACTOR = h.SEDIMENT_ISOSTATIC_FACTOR


def _one(depth, before=None, warm=True, continental=True, duration=20.0, strength=1.0):
    depth = np.array([float(depth)])
    before = depth if before is None else np.array([float(before)])
    return float(biogenic_thickness(depth, before, np.array([warm]), np.array([continental]), duration, strength)[0])


def test_warm_lit_platforms_fill_their_space_to_just_below_sea_level():
    thickness = _one(40.0)
    assert thickness * FACTOR == np.float64(40.0 - TOP)


def test_platforms_keep_pace_with_subsidence_but_drown_when_they_start_too_deep():
    kept = _one(300.0, before=30.0)                 # sank 270 m this interval, but started in the lit zone
    assert kept * FACTOR == 300.0 - TOP
    drowned = _one(300.0, before=280.0)             # already below the lit zone: shelf/slope rate only
    assert drowned == h.SLOPE_CARBONATE_M_MYR * 20.0


def test_cool_shelves_accumulate_slowly_and_never_above_the_platform_top():
    assert _one(150.0, warm=False) == h.SHELF_CARBONATE_M_MYR * 20.0
    assert _one(10.0, warm=False) * FACTOR <= 10.0 - TOP + 1e-9


def test_slope_rates_are_for_continental_margins_and_ridges_get_pelagic_rain():
    assert _one(2500.0, warm=False) == h.SLOPE_CARBONATE_M_MYR * 20.0
    assert _one(2500.0, warm=False, continental=False) == h.PELAGIC_ABOVE_CCD_M_MYR * 20.0


def test_pelagic_rain_slows_below_the_compensation_depth():
    above = _one(h.CARBONATE_COMPENSATION_DEPTH_M - 100, warm=False, continental=False)
    below = _one(h.CARBONATE_COMPENSATION_DEPTH_M + 100, warm=False, continental=False)
    assert above > below > 0


def test_no_sea_life_or_dry_land_means_no_biogenic_sediment():
    assert _one(40.0, strength=0.0) == 0.0
    assert _one(-100.0) == 0.0
    assert _one(40.0, strength=0.5) == 0.5 * _one(40.0)
