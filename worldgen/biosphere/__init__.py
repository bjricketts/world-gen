"""Life: whether a planet has it, its effect on the air, vegetation, biomes and coupling with the climate."""

from .biomes import KOPPEN_CODES, Biome, classify_biomes, koppen
from .coupling import CoupledSurface, couple_surface
from .life import LifeInputs, build_biosphere
from .vegetation import Vegetation, build_vegetation, productive_share

__all__ = ["KOPPEN_CODES", "Biome", "CoupledSurface", "LifeInputs", "Vegetation", "build_biosphere",
           "build_vegetation", "classify_biomes", "couple_surface", "koppen", "productive_share"]
