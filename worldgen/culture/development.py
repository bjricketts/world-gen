"""Development of a culture on three ordinal axes: energy, transport and communication.

Each axis has six named levels, 0 to 5. Energy sets how many people the land
carries and what habitats can be sustained; transport sets road density, reach
and sea routes; communication sets how large and cohesive a polity can be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ENERGY_LEVELS = ("foraging", "farming", "water_wind", "fossil", "electric", "fusion")
TRANSPORT_LEVELS = ("foot", "pack_and_boat", "wheel_and_sail", "rail_and_steam", "motor_and_air", "spaceflight")
COMMUNICATION_LEVELS = ("oral", "writing", "print", "telegraph", "digital", "interplanetary")

AXES = {"energy": ENERGY_LEVELS, "transport": TRANSPORT_LEVELS, "communication": COMMUNICATION_LEVELS}

EnergyLevel = Literal["foraging", "farming", "water_wind", "fossil", "electric", "fusion"]
TransportLevel = Literal["foot", "pack_and_boat", "wheel_and_sail", "rail_and_steam", "motor_and_air", "spaceflight"]
CommunicationLevel = Literal["oral", "writing", "print", "telegraph", "digital", "interplanetary"]


@dataclass(frozen=True)
class Development:
    """A culture's level on each axis, 0 (lowest) to 5 (highest)."""

    energy: int
    transport: int
    communication: int

    def __post_init__(self):
        """Reject levels outside 0–5."""
        for axis in AXES:
            level = getattr(self, axis)
            if not 0 <= level < len(AXES[axis]):
                raise ValueError(f"{axis} level {level} outside 0–{len(AXES[axis]) - 1}")

    def name(self, axis: str) -> str:
        """Return the level name on one axis."""
        return AXES[axis][getattr(self, axis)]

    def to_names(self) -> dict[str, str]:
        """Return the level names on all three axes."""
        return {axis: self.name(axis) for axis in AXES}

    @classmethod
    def from_names(cls, names: dict[str, str]) -> "Development":
        """Return the development given level names for all three axes."""
        return cls(**{axis: AXES[axis].index(names[axis]) for axis in AXES})


def resolve_development(given: dict[str, Optional[str]], default: dict[str, str]) -> Development:
    """Return the development from the axes the user gave.

    With no axis given, ``default`` is used. Axes left out take the rounded
    mean level of those given, so a culture set only to 'fossil' energy gets
    transport and communication of the same era.
    """
    levels = {axis: AXES[axis].index(v) for axis, v in given.items() if v is not None}
    if not levels:
        return Development.from_names(default)
    fill = int(sum(levels.values()) / len(levels) + 0.5)
    return Development(**{axis: levels.get(axis, fill) for axis in AXES})
