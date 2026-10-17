"""Names: a sound system per culture, words built from its syllables, and place names with elements.

Each culture draws a consonant and vowel inventory (sizes from WALS; the
near-universal core first, the rest weighted by PHOIBLE frequencies), a
syllable class (simple, moderately complex or complex) and spellings in Latin
letters. Words are built syllable by syllable within the class, so every name
can be pronounced by its own culture. Place names add words the culture coins
for river, hill, harbour and the like when the place has that feature.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np

from .. import heuristics as h
from ..util import named_rng


def _draw_class(rng: random.Random, table):
    """Return one class from a table of (class, count) pairs, weighted by count."""
    return rng.choices([c for c, _ in table], weights=[n for _, n in table])[0]


def _draw_inventory(rng: random.Random, offered: dict, core, size: int) -> list[str]:
    """Return an inventory of ``size`` segments: the core, then others drawn by frequency without replacement."""
    chosen = [s for s in core if s in offered][:size]
    rest = [s for s in offered if s not in chosen]
    while len(chosen) < size and rest:
        s = rng.choices(rest, weights=[offered[x][0] for x in rest])[0]
        chosen.append(s)
        rest.remove(s)
    return chosen


def _spell(segments: list[str], offered: dict) -> dict[str, str]:
    """Return a spelling per segment, most frequent first, each the first of its spellings not yet taken."""
    taken, out = set(), {}
    for s in sorted(segments, key=lambda x: -offered[x][0]):
        options = offered[s][1]
        pick = next((o for o in options if o not in taken), None)
        while pick is None or pick in taken:
            pick = (pick or options[0]) + "h"
        out[s] = pick
        taken.add(pick)
    return out


def make_phonology(seed: int, name: str) -> dict:
    """Return a culture's sound system: inventories, spellings, syllable class and place-name elements."""
    rng = random.Random(int(named_rng(seed, name).integers(2**62)))
    lo, hi = _draw_class(rng, h.CONSONANT_CLASSES)
    consonants = _draw_inventory(rng, h.CONSONANTS, h.CORE_CONSONANTS, min(rng.randint(lo, hi), len(h.CONSONANTS)))
    lo, hi = _draw_class(rng, h.VOWEL_CLASSES)
    vowels = _draw_inventory(rng, h.VOWELS, h.CORE_VOWELS, min(rng.randint(lo, hi), len(h.VOWELS)))
    spelling = _spell(consonants + vowels, {**h.CONSONANTS, **h.VOWELS})
    phonology = {
        "consonants": consonants, "vowels": vowels, "spelling": spelling,
        "syllables": _draw_class(rng, h.SYLLABLE_CLASSES),
        "long_vowels": rng.random() < h.LONG_VOWEL_SHARE,
        "element_position": "suffix" if rng.random() < h.PLACE_ELEMENT_SUFFIX_SHARE else "prefix",
    }
    gen = WordMaker(phonology, rng)
    phonology["elements"] = {}
    for e in h.PLACE_ELEMENTS:                  # distinct words with an initial consonant
        for _ in range(50):
            w = gen.word(syllables=1 if phonology["syllables"] != "simple" else 2, onset=True).lower()
            if w not in phonology["elements"].values():
                break
        phonology["elements"][e] = w
    return phonology


class WordMaker:
    """Builds words from a culture's syllables and spells them."""

    def __init__(self, phonology: dict, rng: random.Random):
        """Prepare the weighted segment lists of a sound system."""
        self.p = phonology
        self.rng = rng
        self.cons = phonology["consonants"]
        self.cw = [h.CONSONANTS[c][0] for c in self.cons]
        self.vows = phonology["vowels"]
        self.vw = [h.VOWELS[v][0] for v in self.vows]
        self.second = [c for c in self.cons if c in h.CLUSTER_SECOND and c not in ("w", "j")] or \
                      [c for c in self.cons if c in h.CLUSTER_SECOND]
        self.shape = h.SYLLABLE_SHAPE[phonology["syllables"]]
        codas = self.cons if phonology["syllables"] == "complex" else [c for c in self.cons if c in h.CODA_CONSONANTS]
        self.codas = codas or self.cons
        self.codaw = [h.CONSONANTS[c][0] for c in self.codas]

    def syllable(self, first: bool, last: bool, onset: bool = False) -> list[str]:
        """Return the segments of one syllable within the culture's syllable class (``onset`` forces a consonant)."""
        r, (no_onset_first, no_onset, two, three, coda, two_coda) = self.rng, self.shape
        segs = []
        if onset or r.random() >= (no_onset_first if first else no_onset):
            c = r.choices(self.cons, weights=self.cw)[0]
            if first and c == "ʔ":
                c = self.cons[0]
            if self.second and r.random() < two:
                if r.random() < three and "s" in self.cons:
                    segs.append("s")
                segs += [c, r.choice(self.second)] if c not in self.second else [c]
            else:
                segs.append(c)
        v = r.choices(self.vows, weights=self.vw)[0]
        segs.append(v + ("ː" if self.p["long_vowels"] and r.random() < 0.15 else ""))
        if r.random() < coda:
            segs.append(r.choices(self.codas, weights=self.codaw)[0])
            if r.random() < two_coda and last:
                segs.append(r.choices(self.codas, weights=self.codaw)[0])
        return segs

    def segments(self, syllables: int, onset: bool = False) -> list[str]:
        """Return the segments of a word of the given number of syllables."""
        out = []
        for i in range(syllables):
            out += self.syllable(i == 0, i == syllables - 1, onset and i == 0)
        return out

    def spell(self, segments: list[str]) -> str:
        """Return a word's spelling, with long vowels doubled and no letter tripled."""
        sp = self.p["spelling"]
        text = "".join(sp[s[:-1]] * 2 if s.endswith("ː") else sp[s] for s in segments)
        for ch in set(text):
            while ch * 3 in text:
                text = text.replace(ch * 3, ch * 2)
        return text

    def word(self, syllables: int | None = None, onset: bool = False) -> str:
        """Return a capitalised new word of at least NAME_MIN_LETTERS letters."""
        counts, weights = h.SYLLABLES_PER_NAME[self.p["syllables"]]
        for _ in range(20):
            n = syllables if syllables is not None else self.rng.choices(counts, weights=weights)[0]
            text = self.spell(self.segments(n, onset))
            if len(text) >= h.NAME_MIN_LETTERS:
                break
        return text[:1].upper() + text[1:]


def compose(root: str, element: str, position: str) -> str:
    """Return a place name from a root word and an element, in the culture's order."""
    if position == "suffix":
        return root + element
    return element[:1].upper() + element[1:] + root.lower()


class Namer:
    """Hands out names for one culture, unique within it."""

    def __init__(self, phonology: dict, seed: int, name: str):
        """Start a name stream for a culture."""
        self.p = phonology
        self.words = WordMaker(phonology, random.Random(int(named_rng(seed, name).integers(2**62))))
        self.used: set[str] = set()

    def root(self) -> str:
        """Return a new word not used before in this culture."""
        for _ in range(50):
            w = self.words.word()
            if w not in self.used and len(w) <= h.NAME_MAX_LETTERS:
                self.used.add(w)
                return w
        w = self.words.word(syllables=4)
        self.used.add(w)
        return w

    def place(self, features: tuple[str, ...] = ()) -> str:
        """Return a place name, taking one of the place's features as an element now and then."""
        root = self.root()
        rng = self.words.rng
        options = [f for f in features if f in self.p["elements"]]
        if options and rng.random() < h.PLACE_ELEMENT_SHARE:
            name = compose(root, self.p["elements"][rng.choice(options)], self.p["element_position"])
            if name not in self.used and len(name) <= h.NAME_MAX_LETTERS + 6:
                self.used.add(name)
                return name
        return root


def place_features(world, cells: np.ndarray) -> list[tuple[str, ...]]:
    """Return the name-giving features of each cell: river, hill, harbour, lake, field."""
    s, grid = world.surface, world.grid
    ocean = s["ocean"].values.astype(bool)
    lake = s["lake"].values.astype(bool) if "lake" in s else np.zeros(grid.size, bool)
    adj = grid.neighbours.tocsr()
    q = s["discharge"].values if "discharge" in s else np.zeros(grid.size)
    terrain = s["terrain"].values
    out = []
    for c in cells:
        nb = adj.indices[adj.indptr[c]:adj.indptr[c + 1]]
        f = []
        if q[c] > h.RIVER_CROSSING_M3S:
            f.append("river")
        if ocean[nb].any():
            f.append("harbour")
        if lake[nb].any() or lake[c]:
            f.append("lake")
        if terrain[c] in (1, 2, 3):
            f.append("hill")
        elif terrain[c] == 0:
            f.append("field")
        out.append(tuple(f))
    return out


def name_everything(world, cultures, settlements, polities, seed: int):
    """Return cultures with phonologies and names, named polities, and settlement names (in table order)."""
    names = np.array(settlements.name, dtype=object)
    named_cultures, named_polities = [], list(polities)
    features = place_features(world, settlements.cell) if len(settlements) else []
    for i, c in enumerate(cultures):
        phon = c.phonology or make_phonology(seed, f"culture.{i}.phonology")
        namer = Namer(phon, seed, f"culture.{i}.names")
        culture_name = c.name or namer.root()
        mine = np.flatnonzero(settlements.culture == c.id)
        for k in mine[np.argsort(-settlements.population[mine], kind="stable")]:
            extra = ("great",) if settlements.rank[k] == 1 and settlements.population[k] >= 1e5 else ()
            new = ("new",) if c.inhabitants == "colonist" else ()
            names[k] = namer.place(features[k] + extra + new)
        for j, p in enumerate(named_polities):
            if p.culture != c.id:
                continue
            if namer.words.rng.random() < h.POLITY_NAMED_AFTER_CAPITAL and p.capital >= 0:
                base = names[p.capital]
            else:
                base = namer.root()
            name = base
            if namer.words.rng.random() < h.POLITY_LAND_ELEMENT_SHARE:
                name = compose(base, phon["elements"]["land"], phon["element_position"])
            named_polities[j] = replace(p, name=name if len(name) <= 40 else base)
        named_cultures.append(replace(c, phonology=phon, name=culture_name))
    return named_cultures, named_polities, names.astype("U40")
