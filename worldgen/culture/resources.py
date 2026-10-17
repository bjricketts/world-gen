"""Resource potential of every cell, from the tectonic setting, sediment, climate, life and the planet's past.

Each class has a potential from 0 (absent) to 1 (as favourable as the class's
best geological setting allows). Values are not normalised per world, so a
planet without life has no coal or oil and a planet whose oceans were never
anoxic has no iron formations. Individual deposits are drawn from these
potentials when a region is zoomed; the best cells mark notable deposits.

Settings and their sources:

- copper: porphyry systems in magmatic arcs above subduction zones, preserved
  only while young because they form at shallow depth (Sillitoe 2010).
- gold: orogenic gold in collisional and accretionary belts, also in very old
  cratons (Groves et al. 1998; Goldfarb et al. 2001).
- tin_tungsten: granites melted from enriched crust in collision belts
  (Romer & Kroner 2016).
- iron: iron formations on old stable crust, laid down while the oceans were
  liquid and without free oxygen (Bekker et al. 2010).
- lead_zinc: sediment-hosted deposits in rift and passive-margin basins, from
  oxidised basin brines (Leach et al. 2010).
- hydrocarbons: thick basin sediment with marine source rocks, weighted by
  setting (Klemme & Ulmishek 1991), calibrated to the giant fields of
  Mann et al. (2003).
- coal: plant matter buried in wet, subsiding basins once land plants exist
  (Nelsen et al. 2016).
- evaporites: arid closed and rift basins (Warren 2010).
- laterite: bauxite and nickel laterites on tropical, humid, low-relief,
  long-stable ground (Freyssinet et al. 2005).
- timber: forest cover.
- uranium: sandstone and unconformity deposits in basins on old crust once
  surface water is oxidised, and granite-related deposits in collision belts
  (Cuney 2009).
- chromite_pge: chromitites in ophiolites along sutures (González-Jiménez et
  al. 2014) and magmatic PGE and Ni–Cu in large intrusions on continents
  (Maier & Groves 2011).
- rare_earths: carbonatites and alkaline rocks in continental rifts, and
  ion-adsorption clays on weathered granites in the tropics (Smith et al. 2016).
- geothermal: high heat flow near volcanoes, rifts and arcs (Bertani 2016).
- placer_gold: gold carried down rivers from eroding gold belts (heuristic).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from .. import heuristics as h
from ..biosphere.biomes import Biome
from ..hydrology.graph import upstream_order
from ..noise import fbm
from ..surface.fields import OrogenyKind, Terrain
from .geology import Geology, Setting, geology

RESOURCES = ("copper", "gold", "tin_tungsten", "iron", "lead_zinc", "hydrocarbons", "coal", "evaporites",
             "laterite", "timber", "uranium", "chromite_pge", "rare_earths", "geothermal", "placer_gold")

FOREST_WEIGHT = {Biome.BOREAL_FOREST: 1.0, Biome.TEMPERATE_FOREST: 1.0, Biome.TEMPERATE_RAINFOREST: 1.0,
                 Biome.TROPICAL_SEASONAL_FOREST: 1.0, Biome.TROPICAL_RAINFOREST: 1.0,
                 Biome.WOODLAND_SHRUBLAND: 0.5, Biome.SAVANNA: 0.3}


def ramp(x, low: float, high: float) -> np.ndarray:
    """Return 0 below ``low``, 1 above ``high``, linear between."""
    return np.clip((np.asarray(x, dtype=float) - low) / (high - low), 0.0, 1.0)


def _field(world, name: str, fill: float = 0.0) -> np.ndarray:
    """Return a surface field as floats, or a constant where the surface lacks it."""
    s = world.surface
    return s[name].values.astype(float) if name in s else np.full(s.sizes["cell"], fill)


class _Context:
    """Surface fields and geology shared by the resource rules."""

    def __init__(self, world, geo: Geology):
        self.geo = geo
        self.t = geo.deep_time
        self.setting = geo.setting
        self.land = geo.land
        self.cont = geo.continental
        self.d = geo.distance_km
        self.elevation = _field(world, "elevation")
        self.sediment = _field(world, "sediment")
        self.kind = world.surface["orogeny_kind"].values if "orogeny_kind" in world.surface else \
            np.zeros(self.elevation.size, dtype=np.int8)
        # A belt with a kind but no age is a present-day belt of the heuristic plate layout.
        age = _field(world, "orogeny_age", np.nan)
        age = np.where(np.isnan(age) & (self.kind > 0), 0.0, age)
        self.orogeny_age = np.where(np.isfinite(age), age, np.inf)
        self.terrain = world.surface["terrain"].values
        self.temperature_c = _field(world, "air_temperature", 273.15) - 273.15
        self.precip_mm = _field(world, "precipitation") * 1e3
        self.pet_mm = _field(world, "potential_evaporation") * 1e3
        self.ice = _field(world, "ice_thickness") > 0
        self.vegetation = _field(world, "vegetation_cover")
        self.biome = world.surface["biome"].values if "biome" in world.surface else np.zeros(self.land.size, int)
        self.endorheic = _field(world, "endorheic") > 0
        self.lake = _field(world, "lake") > 0
        self.flow_to = world.surface["flow_to"].values.astype(np.int64) if "flow_to" in world.surface else None
        self.discharge = _field(world, "discharge")
        self.nearest = geo.nearest
        # Ground a deposit can sit in: land, continental shelves and slopes, and island arcs.
        self.crustal = (self.land | (self.cont & (self.elevation > h.RESOURCE_MIN_ELEVATION_M))) & ~self.ice

    def belt(self, name: str, weight_by_kind: dict, preservation_myr: float, width_km: float) -> np.ndarray:
        """Return a belt potential: the kind and age of the nearest belt cell, fading with distance from it."""
        idx = self.nearest.get(name)
        if idx is None:
            return np.zeros(self.land.size)
        ok = idx >= 0
        value = np.zeros(self.land.size)
        src = idx[ok]
        weight = np.zeros(src.size)
        for kind, w in weight_by_kind.items():
            weight[self.kind[src] == kind] = w
        age = self.orogeny_age[src]
        preserved = np.where(np.isfinite(age), np.exp(-age / preservation_myr), h.OLD_BELT_PRESERVATION)
        value[ok] = weight * preserved * np.exp(-self.d[name][ok] / width_km)
        return value


def _copper(x: _Context) -> np.ndarray:
    kinds = {OrogenyKind.ANDEAN: 1.0, OrogenyKind.ARC: 1.0}
    arc = x.belt("arc_belt", kinds, h.PORPHYRY_PRESERVATION_MYR, h.BELT_WIDTH_KM)
    active = np.exp(-x.d["subduction"] / h.ACTIVE_ARC_WIDTH_KM)
    return np.maximum(arc, active)


def _gold(x: _Context) -> np.ndarray:
    kinds = {OrogenyKind.COLLISION: 1.0, OrogenyKind.ANDEAN: h.GOLD_ACCRETIONARY_WEIGHT,
             OrogenyKind.ARC: h.GOLD_ARC_WEIGHT}
    belts = np.maximum(x.belt("collision_belt", kinds, h.OROGENIC_GOLD_PRESERVATION_MYR, h.BELT_WIDTH_KM),
                       x.belt("arc_belt", kinds, h.OROGENIC_GOLD_PRESERVATION_MYR, h.BELT_WIDTH_KM))
    greenstone = h.GREENSTONE_GOLD * ramp(x.t.age_gyr, *h.GREENSTONE_PLANET_AGE_GYR) * (x.setting == Setting.CRATON)
    return np.maximum(belts, greenstone)


def _tin_tungsten(x: _Context) -> np.ndarray:
    kinds = {OrogenyKind.COLLISION: 1.0, OrogenyKind.ANDEAN: h.TIN_ANDEAN_WEIGHT}
    return np.maximum(x.belt("collision_belt", kinds, h.GRANITE_PRESERVATION_MYR, h.GRANITE_BELT_WIDTH_KM),
                      x.belt("arc_belt", kinds, h.GRANITE_PRESERVATION_MYR, h.GRANITE_BELT_WIDTH_KM))


def _stable(x: _Context) -> np.ndarray:
    """Return stable continental ground: cratons and sag basins without recent mountain building."""
    return x.cont & np.isin(x.setting, (Setting.CRATON, Setting.SAG_BASIN, Setting.PASSIVE_MARGIN)) & \
        (x.orogeny_age > h.STABLE_CRUST_MYR)


def _iron(x: _Context) -> np.ndarray:
    return _stable(x) * ramp(x.t.anoxic_ocean_gyr, *h.IRON_FORMATION_ANOXIC_GYR)


def _lead_zinc(x: _Context) -> np.ndarray:
    basins = np.isin(x.setting, (Setting.RIFT, Setting.PASSIVE_MARGIN, Setting.SAG_BASIN))
    return basins * ramp(x.sediment, *h.LEAD_ZINC_SEDIMENT_M) * ramp(x.t.oxygenated_gyr, 0.0, h.OXIDISED_GYR)


def _hydrocarbons(x: _Context) -> np.ndarray:
    factor = np.zeros(len(Setting))
    for name, value in h.HYDROCARBON_SETTING_FACTOR.items():
        factor[Setting[name]] = value
    source = ramp(x.t.life_gyr, *h.HYDROCARBON_LIFE_GYR)
    return x.cont * factor[x.setting] * ramp(x.sediment, *h.HYDROCARBON_BURIAL_M) * source


def _coal(x: _Context) -> np.ndarray:
    factor = np.zeros(len(Setting))
    for name, value in h.COAL_SETTING_FACTOR.items():
        factor[Setting[name]] = value
    wet = h.COAL_WET_FLOOR + (1 - h.COAL_WET_FLOOR) * ramp(x.precip_mm, *h.COAL_WET_MM)
    plants = ramp(x.t.land_life_gyr, *h.COAL_LAND_LIFE_GYR)
    return x.cont * factor[x.setting] * ramp(x.sediment, *h.COAL_SEDIMENT_M) * wet * plants


def _aridity(x: _Context) -> np.ndarray:
    """Return 1 on dry ground (precipitation well below potential evaporation), 0 on wet ground."""
    ratio = x.precip_mm / np.maximum(x.pet_mm, 1.0)
    return 1.0 - ramp(ratio, *h.EVAPORITE_ARIDITY_RATIO)


def _evaporites(x: _Context) -> np.ndarray:
    basin = np.maximum.reduce([x.endorheic * 1.0, x.lake * 1.0, h.EVAPORITE_RIFT * (x.setting == Setting.RIFT),
                               h.EVAPORITE_MARGIN * (x.setting == Setting.PASSIVE_MARGIN)])
    return basin * _aridity(x) * x.land


def _tropical(x: _Context) -> np.ndarray:
    """Return how tropical and humid the climate is (0–1)."""
    return ramp(x.temperature_c, *h.LATERITE_TEMPERATURE_C) * ramp(x.precip_mm, *h.LATERITE_PRECIPITATION_MM)


def _laterite(x: _Context) -> np.ndarray:
    low = ~np.isin(x.terrain, (Terrain.MOUNTAIN, Terrain.VOLCANO))
    stable = x.orogeny_age > h.LATERITE_STABLE_MYR
    return _tropical(x) * low * stable * x.land


def _timber(x: _Context) -> np.ndarray:
    weight = np.zeros(max(int(x.biome.max()) + 1, len(Biome)))
    for biome, w in FOREST_WEIGHT.items():
        weight[biome] = w
    return x.vegetation * weight[x.biome] * x.land


def _uranium(x: _Context) -> np.ndarray:
    oxidised = ramp(x.t.oxygenated_gyr, 0.0, h.OXIDISED_GYR)
    basins = (np.isin(x.setting, (Setting.SAG_BASIN, Setting.CRATON))
              * ramp(x.sediment, *h.URANIUM_SEDIMENT_M) * (x.orogeny_age > h.STABLE_CRUST_MYR))
    granites = h.URANIUM_GRANITE * x.belt("collision_belt", {OrogenyKind.COLLISION: 1.0},
                                          h.GRANITE_PRESERVATION_MYR, h.GRANITE_BELT_WIDTH_KM)
    return oxidised * np.maximum(basins, granites) * x.cont


def _chromite_pge(x: _Context) -> np.ndarray:
    sutures = h.OPHIOLITE_WEIGHT * x.belt("collision_belt", {OrogenyKind.COLLISION: 1.0},
                                          h.OPHIOLITE_PRESERVATION_MYR, h.OPHIOLITE_WIDTH_KM)
    intrusions = (x.cont & (x.setting == Setting.VOLCANIC)
                  & (x.d["subduction"] > h.INTRAPLATE_MIN_ARC_DISTANCE_KM)) * 1.0
    return np.maximum(sutures, intrusions)


def _rare_earths(x: _Context) -> np.ndarray:
    carbonatite = x.cont * np.exp(-x.d["rift"] / h.RIFT_ALKALINE_WIDTH_KM)
    granite = np.maximum(x.belt("arc_belt", {OrogenyKind.ANDEAN: 1.0}, h.GRANITE_PRESERVATION_MYR,
                                h.GRANITE_BELT_WIDTH_KM),
                         x.belt("collision_belt", {OrogenyKind.COLLISION: 1.0}, h.GRANITE_PRESERVATION_MYR,
                                h.GRANITE_BELT_WIDTH_KM))
    clays = h.ION_CLAY_WEIGHT * granite * _tropical(x)
    return np.maximum(carbonatite, clays)


def _geothermal(x: _Context) -> np.ndarray:
    heat = np.maximum.reduce([np.exp(-x.d["volcanic"] / h.GEOTHERMAL_VOLCANIC_KM),
                              np.exp(-x.d["rift"] / h.GEOTHERMAL_RIFT_KM),
                              h.GEOTHERMAL_ARC * np.exp(-x.d["subduction"] / h.GEOTHERMAL_ARC_KM)])
    scale = np.clip(h.GEOTHERMAL_BASE + (1 - h.GEOTHERMAL_BASE) * x.t.heat_flux_ratio, 0.0, 1.0)
    return heat * scale * x.land


def _placer_gold(x: _Context, gold: np.ndarray, spacing_km: float) -> np.ndarray:
    if x.flow_to is None:
        return np.zeros(x.land.size)
    keep = np.exp(-spacing_km / h.PLACER_TRANSPORT_KM)
    carried = gold * x.land
    order = upstream_order(x.flow_to)
    for i in order[::-1]:                      # sources first
        r = x.flow_to[i]
        if r != i and carried[i] * keep > carried[r]:
            carried[r] = carried[i] * keep
    river = x.discharge >= h.PLACER_MIN_DISCHARGE_M3_S
    return carried * river * x.land


# Classes whose deposits cluster into metallogenic districts inside their favourable settings.
DISTRICT_CLASSES = ("copper", "gold", "tin_tungsten", "iron", "lead_zinc", "uranium", "chromite_pge", "rare_earths")


def districts(world, name: str) -> np.ndarray:
    """Return a 0–1 field that concentrates a class into districts of ~``DISTRICT_SCALE_KM`` across."""
    radius_km = float(world.surface.attrs["radius_m"]) / 1e3
    noise = fbm(world.grid.points, int(world.surface.attrs.get("seed", 0)), f"district.{name}",
                frequency=radius_km / h.DISTRICT_SCALE_KM, octaves=3)
    return ramp(noise, *h.DISTRICT_NOISE_RANGE)


def resource_potential(world, geo: Optional[Geology] = None) -> xr.DataArray:
    """Return the potential (0–1) of every resource class on every cell, as a (resource, cell) array."""
    geo = geo if geo is not None else geology(world)
    x = _Context(world, geo)
    values = {
        "copper": _copper(x), "gold": _gold(x), "tin_tungsten": _tin_tungsten(x), "iron": _iron(x),
        "lead_zinc": _lead_zinc(x), "hydrocarbons": _hydrocarbons(x), "coal": _coal(x),
        "evaporites": _evaporites(x), "laterite": _laterite(x), "timber": _timber(x), "uranium": _uranium(x),
        "chromite_pge": _chromite_pge(x), "rare_earths": _rare_earths(x), "geothermal": _geothermal(x),
    }
    for name in DISTRICT_CLASSES:
        values[name] = values[name] * districts(world, name)
    spacing_km = float(world.surface.attrs.get("spacing_km", world.grid.mean_spacing(
        float(world.surface.attrs["radius_m"]) / 1e3)))
    values["placer_gold"] = _placer_gold(x, values["gold"], spacing_km)
    data = np.stack([np.where(x.crustal | (name in ("timber", "geothermal", "evaporites", "laterite", "placer_gold")),
                              values[name], 0.0) for name in RESOURCES]).astype(np.float32)
    return xr.DataArray(np.clip(data, 0.0, 1.0), dims=("resource", "cell"),
                        coords={"resource": list(RESOURCES)}, name="resource_potential",
                        attrs={"long_name": "resource potential (0 absent, 1 most favourable setting)"})
