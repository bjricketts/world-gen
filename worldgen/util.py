"""Small shared helpers: reproducible random streams and smooth thresholds."""

import hashlib
import math

import numpy as np


def stable_hash(*parts: object) -> int:
    """Return a 64-bit integer hash of the parts that is identical across runs and platforms."""
    text = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def named_rng(seed: int, name: str) -> np.random.Generator:
    """Return a random generator unique to (seed, name).

    Each quantity draws from its own stream, so adding or removing one
    random draw does not change any other draw for the same seed.
    """
    return np.random.default_rng(stable_hash("worldgen", seed, name))


def logistic(x: float) -> float:
    """Return the logistic function 1 / (1 + e^-x), safe for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def log_threshold(value: float, threshold: float, width: float) -> float:
    """Return a smooth 0→1 step as ``value`` rises through ``threshold`` (width in natural-log units)."""
    if value <= 0:
        return 0.0
    return logistic((math.log(value) - math.log(threshold)) / width)


def choose(rng: np.random.Generator, probabilities: dict[str, float]) -> str:
    """Pick a key from a dict of probabilities."""
    keys = list(probabilities)
    p = np.array([probabilities[k] for k in keys], dtype=float)
    p = p / p.sum()
    return str(rng.choice(keys, p=p))
