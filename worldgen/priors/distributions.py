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

    def quantile(self, q: float):
        """Return the fixed value at any quantile."""
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

    def quantile(self, q: float) -> float:
        """Return the value below which a fraction ``q`` of the distribution lies."""
        return float(self.low + q * (self.high - self.low))

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

    def quantile(self, q: float) -> float:
        """Return the value below which a fraction ``q`` of the distribution lies."""
        return float(math.exp(math.log(self.low) + q * (math.log(self.high) - math.log(self.low))))

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

    def quantile(self, q: float) -> float:
        """Return the value below which a fraction ``q`` of the truncated distribution lies."""
        if q <= 0.0 or q >= 1.0:
            return float(self.low if q <= 0.0 else self.high)
        edges = [0.5 * (1.0 + math.erf((x - self.mean) / (self.sd * math.sqrt(2.0))))
                 for x in (self.low, self.high)]
        inner = edges[0] + min(max(q, 0.0), 1.0) * (edges[1] - edges[0])
        return float(min(max(self.mean + self.sd * math.sqrt(2.0) * _erf_inverse(2.0 * inner - 1.0),
                             self.low), self.high))

    def allows(self, low, high) -> bool:
        """Return whether a user value or range overlaps this distribution's support."""
        return low <= self.high and high >= self.low


@dataclass(frozen=True)
class Choice:
    """One of several options with optional weights."""

    options: Sequence[Union[str, float, bool]]
    weights: Sequence[float] | None = None

    def quantile(self, q: float):
        """Return the option at quantile ``q`` of the (weighted) options."""
        w = np.ones(len(self.options)) if self.weights is None else np.asarray(self.weights, dtype=float)
        cumulative = np.cumsum(w / w.sum())
        return self.options[int(np.searchsorted(cumulative, min(max(q, 0.0), 1.0 - 1e-12)))]

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

    def quantile(self, q: float) -> float:
        """Return the habitable-zone position at quantile ``q``."""
        return float(self.low + q * (self.high - self.low))

    def allows(self, low, high) -> bool:
        """Return True: compatibility with an absolute instellation depends on the star."""
        return True


def _erf_inverse(y: float) -> float:
    """Return the inverse error function (Winitzki's approximation, refined by one Newton step)."""
    y = min(max(y, -1.0 + 1e-12), 1.0 - 1e-12)
    a = 0.147
    term = 2.0 / (math.pi * a) + math.log(1.0 - y * y) / 2.0
    x = math.copysign(math.sqrt(math.sqrt(term * term - math.log(1.0 - y * y) / a) - term), y)
    return x - (math.erf(x) - y) / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))


Distribution = Union[Fixed, Uniform, LogUniform, TruncNormal, Choice, HZFraction]
