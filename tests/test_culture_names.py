"""Sound systems, words and place names."""

import random
from pathlib import Path

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture.build import build_civilisation
from worldgen.culture.names import Namer, WordMaker, compose, make_phonology
from worldgen.culture.spec import CultureSpec


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_inventories_follow_wals_and_hold_the_core():
    phons = [make_phonology(0, f"c{i}") for i in range(600)]
    nc = np.array([len(p["consonants"]) for p in phons])
    nv = np.array([len(p["vowels"]) for p in phons])
    assert 19 <= np.median(nc) <= 23 and nc.min() >= 6
    assert 0.40 < np.mean((nv >= 5) & (nv <= 6)) < 0.62            # WALS 287 of 564
    syl = [p["syllables"] for p in phons]
    assert 0.45 < syl.count("moderate") / len(syl) < 0.68           # WALS 274 of 486
    for p in phons:
        assert set(h.CORE_CONSONANTS[:min(5, len(p["consonants"]))]) <= set(p["consonants"])
        assert set(h.CORE_VOWELS[:len(p["vowels"])]) <= set(p["vowels"])
        spellings = [p["spelling"][s] for s in p["consonants"] + p["vowels"]]
        assert len(set(spellings)) == len(spellings)                 # no two segments spelled alike
        assert len(set(p["elements"].values())) == len(h.PLACE_ELEMENTS)


def test_common_segments_are_common_across_cultures():
    phons = [make_phonology(1, f"c{i}") for i in range(600)]
    share = {c: np.mean([c in p["consonants"] for p in phons]) for c in ("ŋ", "f", "ʔ", "q", "θ")}
    assert share["ŋ"] > share["f"] > share["q"] > share["θ"]
    assert share["ʔ"] > share["q"]


@pytest.mark.parametrize("cls", ["simple", "moderate", "complex"])
def test_words_keep_to_their_syllable_class(cls):
    p = make_phonology(2, "x")
    p = dict(p, syllables=cls)
    maker = WordMaker(p, random.Random(0))
    vowels = set(p["vowels"])
    for _ in range(300):
        segs = [s.rstrip("ː") for s in maker.segments(3)]
        run, longest = 0, 0
        for s in segs:
            run = 0 if s in vowels else run + 1
            longest = max(longest, run)
        limit = {"simple": 1, "moderate": 3, "complex": 5}[cls]
        assert longest <= limit
        if cls == "simple":
            assert segs[-1] in vowels


def test_namer_gives_unique_readable_names():
    p = make_phonology(3, "y")
    namer = Namer(p, 3, "names")
    names = [namer.place(("river", "harbour")) for _ in range(3000)]
    assert len(set(names)) == len(names)
    assert all(h.NAME_MIN_LETTERS <= len(n) <= h.NAME_MAX_LETTERS + 6 for n in names)
    with_element = [n for n in names if p["elements"]["river"] in n.lower() or p["elements"]["harbour"] in n.lower()]
    assert 0.2 < len(with_element) / len(names) < 0.6
    assert compose("Abo", "ret", "suffix") == "Aboret" and compose("Abo", "ret", "prefix") == "Retabo"


def test_world_names_are_unique_per_culture_and_saved(earth, tmp_path):
    from worldgen.culture.io import load_civilisation, save_civilisation

    spec = CultureSpec(cultures=[{"development": {"energy": "fossil"}}, {"development": {"energy": "farming"},
                                  "name": "Given"}])
    civ = build_civilisation(earth, spec)
    assert civ.cultures[1].name == "Given" and civ.cultures[0].name
    st = civ.settlements
    for c in civ.cultures:
        names = st.name[st.culture == c.id]
        assert len(set(names)) == len(names) and all(len(n) >= h.NAME_MIN_LETTERS for n in names)
        assert c.phonology and c.phonology["consonants"]
    assert all(p.name for p in civ.polities)
    again = build_civilisation(earth, spec)
    np.testing.assert_array_equal(st.name, again.settlements.name)
    loaded = load_civilisation(save_civilisation(civ, tmp_path / "culture"))
    np.testing.assert_array_equal(loaded.settlements.name, st.name)
    assert loaded.cultures[0].phonology == civ.cultures[0].phonology
