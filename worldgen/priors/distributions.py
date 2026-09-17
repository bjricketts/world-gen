"""Simple probability distributions for drawing unset parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np


@dataclass(frozen=True)
class Fixed:
    """Always returns the same value."""

    value: Union[float, str, bool]

    def sample(self, rng: np.random.Generator):
        """Return the fixed value."""
        return self.value

    def allows(self, low, high) -> bool:
        """Return whether a user value or range is compatible with this distribution."""
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            return low <= self.value <= high
        return low == self.value


@dataclass(frozen=True)
class Uniform:
    """Uniform between low and high."""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a value."""
        return float(rng.uniform(self.low, self.high))

    def allows(self, low, high) -> bool:
        """Return whether a user value or range overlaps this distribution's support."""
        return low <= self.high and high >= self.low


@dataclass(frozen=True)
class LogUniform:
    """Uniform in log between low and high (both > 0)."""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a value."""
        return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))

    def allows(self, low, high) -> bool:
        """Return whether a user value or range overlaps this distribution's support."""
        return low <= self.high and high >= self.low


@dataclass(frozen=True)
class TruncNormal:
    """Normal distribution truncated to [low, high]."""

    mean: float
    sd: float
    low: float = -math.inf
    high: float = math.inf

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a value."""
        for _ in range(1000):
            x = rng.normal(self.mean, self.sd)
            if self.low <= x <= self.high:
                return float(x)
        return float(min(max(self.mean, self.low), self.high))

    def allows(self, low, high) -> bool:
        """Return whether a user value or range overlaps this distribution's support."""
        return low <= self.high and high >= self.low


@dataclass(frozen=True)
class Choice:
    """One of several options with optional weights."""

    options: Sequence[Union[str, float, bool]]
    weights: Sequence[float] | None = None

    def sample(self, rng: np.random.Generator):
        """Draw an option."""
        p = None
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            p = w / w.sum()
        return self.options[int(rng.choice(len(self.options), p=p))]

    def allows(self, low, high) -> bool:
        """Return whether a user value is one of the options."""
        return any(low == o or (isinstance(o, (int, float)) and low <= o <= high) for o in self.options)


@dataclass(frozen=True)
class HZFraction:
    """Instellation given as a position across the conservative habitable zone.

    0 is the runaway-greenhouse limit and 1 the maximum-greenhouse limit (by
    orbital distance); values outside 0–1 lie inside or beyond the zone. The
    sampler converts the drawn position to instellation once the star and
    planet mass are known.
    """

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a habitable-zone position (not an instellation)."""
        return float(rng.uniform(self.low, self.high))

    def allows(self, low, high) -> bool:
        """Return True: compatibility with an absolute instellation depends on the star."""
        return True


Distribution = Union[Fixed, Uniform, LogUniform, TruncNormal, Choice, HZFraction]
