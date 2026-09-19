"""Global surface generation: elevation, landforms, crust and sea level on the spherical grid."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np
import xarray as xr

from .. import constants as c
from .. import heuristics as h
from .. import water as water_model
from ..grid import build_grid, resolve_resolution
from ..state import Issue, PlanetState, SurfaceSummary, Timeline
from . import plates, regimes
from ..biosphere import CoupledSurface, classify_biomes, couple_surface, koppen, productive_share
from ..atmosphere import refine_climate
from ..climate import SurfaceClimate, band_land_fraction, planet_climate, planet_control
from ..climate.ice import ICE_DENSITY
from ..priors.occupiability import score_planet
from ..hydrology import Drainage, WaterSetting, build_drainage, erode_surface
from .drive import TectonicDrive
from .fabric import structural_fabric
from .fields import Boundary, Crust, SurfaceFields, Terrain
from .relicts import Relict, paint as paint_relicts, read_history as read_relict_history
from .sealevel import OceanFill, apply_sea_level, ocean_fill

DEFAULT_LAND_FRACTION = 0.3

__all__ = ["build_surface", "relief_factor", "Terrain", "Crust", "Boundary"]


def relief_factor(surface_gravity_m_s2: float) -> float:
    """Return the scale applied to mountain and volcano heights, 1 at Earth gravity."""
    lo, hi = h.RELIEF_FACTOR_RANGE
    return float(np.clip((c.G_EARTH / surface_gravity_m_s2) ** h.RELIEF_GRAVITY_EXPONENT, lo, hi))


def build_surface(state: PlanetState, resolution: int | str = "standard",
                  snapshot_interval_myr: Optional[float] = None,
                  timeline: Optional[Timeline] = None) -> Optional[xr.Dataset]:
    """Return the global surface of a planet, or ``None`` for planets without a solid surface.

    The planet state gains a surface summary, and notes on the land-fraction
    target are added to its report. For simulated plate tectonics,
    ``snapshot_interval_myr`` adds the elevation, plates and crust type at
    that interval through the simulation, and a history-mode ``timeline``
    drives the plate speed, the volcanism and the simulated span from the
    planet's own interior history.
    """
    regime = state.interior.tectonic_regime
    if regime == "fluid":
        state.issues.append(Issue("info", "note", "surface", "no solid surface: no terrain generated"))
        return None

    n = resolve_resolution(resolution)
    grid = build_grid(n)
    fields = SurfaceFields(grid=grid, radius_m=state.bulk.radius_m)
    seed = state.draw_seed
    relief = relief_factor(state.bulk.surface_gravity_m_s2)
    age_gyr = state.star.age_s / c.SECONDS_PER_GYR
    activity = state.interior.activity_index
    water = state.atmosphere.surface_water
    has_ocean = water in ("liquid", "ice")
    erosion = h.CRATER_WATER_EROSION if water == "liquid" else 1.0

    liquid = water == "liquid"
    target = state.inputs.get("surface.land_fraction")
    if target is not None and not has_ocean and target < 1.0:
        state.issues.append(Issue("warning", "conflict", "surface",
                                  f"land fraction {target:.2f} requested but there is no surface water"))
    # Continental crust covers the expected land plus a shelf; dry planets keep a default highland extent.
    if not has_ocean:
        crust_target = DEFAULT_LAND_FRACTION
    elif target is not None:
        crust_target = target
    else:
        crust_target = state.water.land_fraction_estimate

    climate_setting, zonal = planet_climate(state)
    # Frozen planets get snowline erosion on the finished surface; liquid water adds rivers, also in the simulation.
    setting = _water_setting(state, relief, target, climate_setting, zonal, liquid) if has_ocean else None
    sim_setting = setting if liquid else None

    drive = None
    if timeline is not None and timeline.series:
        drive = TectonicDrive.from_series(timeline.series, age_gyr)

    snapshots = []
    mode = state.inputs.get("surface.tectonics", "simulated")
    if regime == "mobile_lid" and mode == "simulated":
        from .. import tectonics    # imported here: the simulation builds on this package

        start = state.inputs.get("surface.tectonics_start", "supercontinent")
        duration = _tectonic_duration(state, drive)
        features, result = tectonics.build_simulated_terrain(
            fields, activity, crust_target, relief, age_gyr, seed, start, duration, snapshot_interval_myr,
            sim_setting, drive)
        snapshots = result.snapshots
    elif regime == "mobile_lid":
        features = plates.build_plate_terrain(fields, activity, crust_target, relief, age_gyr, seed)
    elif regime == "stagnant_lid":
        features = regimes.build_stagnant_lid(fields, activity, relief, age_gyr, erosion, seed)
    elif regime == "episodic":
        features = regimes.build_episodic(fields, activity, relief, erosion, seed)
    elif regime == "heat_pipe":
        features = regimes.build_heat_pipe(fields, relief, seed)
    else:
        features = regimes.build_inactive(fields, relief, age_gyr, erosion, seed)

    fill, level = _sea_level(state, fields, has_ocean, target, report=setting is None)
    drainage = rain = None
    if setting is not None:
        # Final erosion of the detailed surface, then sea level again for the same water.
        setting.ocean_volume_m3 = state.water.ocean_volume_m3
        height = fields.elevation - level
        eroded = erode_surface(grid, setting, height, fill.pass_level < level, h.FINAL_EROSION_MYR,
                               h.EROSION_STEP_MYR)
        fields.elevation[:] = eroded.height_m + level
        fill, level = _sea_level(state, fields, has_ocean, target, report=True)
    ocean = apply_sea_level(fields, fill, level)
    snapshot_volume = state.water.ocean_volume_m3 if has_ocean else 0.0
    if has_ocean and state.climate is not None:
        # The global climate again, with the land where the surface put it.
        bands = band_land_fraction(grid, ocean, climate_setting.synchronous)
        _refine_global_climate(state, bands)
        climate_setting, zonal = planet_climate(state, bands)
    control = planet_control(state, zonal)

    def couple(ocean_mask, initial=None):
        """Return the coupled climate, ice and vegetation of the current surface."""
        return couple_surface(grid, fields.radius_m, fields.elevation, ocean_mask, climate_setting, zonal,
                              state.bulk.surface_gravity_m_s2, state.biosphere, state.orbit.instellation_earth,
                              control=control, initial=initial)

    coupled = couple(ocean)
    if has_ocean:
        coupled, ocean, ice_features = _store_ice(state, fields, coupled, ocean, target, couple)
        features.update(ice_features)
    climate = coupled.climate
    features["climate_optical_depth"] = climate.thermal.optical_depth
    features.update(_transport_features(zonal, fields.radius_m, climate_setting.synchronous))
    if control is not None and abs(climate.thermal.optical_depth - state.atmosphere.greenhouse_optical_depth) > \
            h.CLIMATE_OPTICAL_DEPTH_NOTE * max(state.atmosphere.greenhouse_optical_depth, 0.1):
        state.issues.append(Issue(
            "info", "note", "climate",
            f"the surface climate needs a greenhouse optical depth of {climate.thermal.optical_depth:.2f} to hold "
            f"{control.target_k:.0f} K (global estimate {state.atmosphere.greenhouse_optical_depth:.2f})"))
    features.update(_climate_features(climate, ocean))
    features.update(_life_features(coupled, ocean))
    state.issues.append(Issue("info", "heuristic", "climate",
                              "surface climate from a two-dimensional energy-balance model with moisture transport, "
                              "iterated with ice sheets and vegetation"))
    if not climate.converged:
        state.issues.append(Issue("warning", "validity", "climate", "the surface climate model did not converge"))
    if state.climate is not None and abs(climate.mean_k - state.climate.mean_temperature_k) > h.CLIMATE_TIER_MISMATCH_K:
        state.issues.append(Issue(
            "warning", "validity", "climate",
            f"the surface climate settles at {climate.mean_k:.0f} K, far from the global estimate of "
            f"{state.climate.mean_temperature_k:.0f} K (the planet is near an ice-cover tipping point)"))
    if not coupled.converged:
        state.issues.append(Issue("info", "note", "climate",
                                  f"climate, ice and vegetation still changed after {coupled.passes} passes"))
    if state.biosphere is not None and state.biosphere.life == "surface":
        _refine_occupiability(state, productive_share(coupled.vegetation, ocean))
    if liquid:
        rain = climate.rainfall
        drainage = build_drainage(grid, fields.radius_m, fields.elevation, ocean, rain)
        fields.terrain[drainage.lake] = Terrain.LAKE
        features.update(_hydrology_features(drainage, ocean, fields))

    relict = None
    past = read_relict_history(state, timeline)
    if past is not None:
        relict, relict_features = paint_relicts(
            grid, state, past, fields.elevation, ocean,
            ocean_fill(grid, fields.elevation, fields.radius_m) if has_ocean else None,
            climate.rainfall.temperature_k, coupled.ice.glaciated if has_ocean else None, relief, seed)
        features.update(relict_features)
        if relict_features:
            kinds = ", ".join(f"{Relict(k).name.lower().replace('_', ' ')} {(relict == k).mean():.0%}"
                              for k in np.unique(relict) if k != Relict.NONE)
            state.issues.append(Issue("info", "heuristic", "surface",
                                      f"relict features from the planet's history ({kinds}), preserved for "
                                      f"{past.memory_myr:.0f} Myr against this surface's erosion"))
    if regime == "mobile_lid" and mode == "simulated":
        driven = "" if drive is None else (f", with plates at {features['plate_speed_km_myr']:.0f} km/Myr and "
                                           f"{drive.melt_now:.2g} × Earth's melting from the planet's history")
        state.issues.append(Issue("info", "note", "surface",
                                  f"plate tectonics simulated for {features['simulated_myr']:.0f} Myr "
                                  f"from a {start} start{driven}"))
    else:
        state.issues.append(Issue("info", "heuristic", "surface",
                                  f"terrain for the '{regime}' regime uses heuristic landform rules"))

    summary = SurfaceSummary(
        grid_size=n,
        spacing_km=fields.spacing_km,
        land_fraction=float(1.0 - ocean.mean()),
        has_ocean=has_ocean,
        frozen_ocean=water == "ice",
        relief_factor=relief,
        min_elevation_m=float(fields.elevation.min()),
        max_elevation_m=float(fields.elevation.max()),
        features={k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in features.items()},
    )
    state.surface = summary

    attrs = {
        "planet": state.name,
        "seed": state.seed,
        "draw_seed": seed,
        "regime": regime,
        "relief_factor": relief,
        "has_ocean": int(has_ocean),
        "frozen_ocean": int(water == "ice"),
        "land_fraction": summary.land_fraction,
        "spacing_km": summary.spacing_km,
        "degrees_per_cell": math.degrees(summary.spacing_km / (state.bulk.radius_m / 1e3)),
        "tectonics": mode if regime == "mobile_lid" else "none",
    }
    fields.fabric[:] = structural_fabric(grid, fields.elevation, fields.crust, fields.crust_age_myr,
                                         fields.orogeny_age_myr, relief)
    dataset = fields.to_dataset(ocean, attrs)
    if relict is not None:
        dataset = dataset.merge(xr.Dataset(data_vars={
            "relict": ("cell", relict, {"long_name": "feature left by the planet's past "
                                                     "(worldgen.surface.relicts.Relict)",
                                        "flag_values": " ".join(str(int(r)) for r in Relict),
                                        "flag_meanings": " ".join(r.name.lower() for r in Relict)})}))
    dataset = dataset.merge(_climate_dataset(climate, has_ocean, climate_setting.humidity > 0.0))
    dataset = dataset.merge(_life_dataset(coupled, climate, ocean, state))
    if drainage is not None:
        dataset = dataset.merge(_hydrology_dataset(drainage, climate))
    if snapshots:
        dataset = dataset.merge(_snapshot_dataset(snapshots, grid, state, fill, snapshot_volume))
    return dataset


def _tectonic_duration(state: PlanetState, drive: Optional[TectonicDrive]) -> float:
    """Return how long to simulate: the value in the spec, or one sea-floor turnover from the history."""
    value = state.inputs.get("surface.tectonics_duration_myr", h.TECTONIC_DURATION_MYR)
    if drive is None or state.provenance.get("surface.tectonics_duration_myr") in ("user", "user_range"):
        return value
    duration = drive.duration_myr(value)
    state.provenance["surface.tectonics_duration_myr"] = "derived"
    state.inputs["surface.tectonics_duration_myr"] = duration
    return duration


def _water_setting(state: PlanetState, relief: float, target: Optional[float], climate_setting,
                   zonal, rivers: bool) -> WaterSetting:
    """Return the planet properties that the climate, drainage and erosion steps need."""
    radius = state.bulk.radius_m
    volume = (state.water.ocean_volume_m3 if target is None
              else water_model.volume_for_land_fraction(target, radius, relief, state.interior.tectonic_regime))
    return WaterSetting(radius_m=radius, ocean_volume_m3=volume, climate=climate_setting, zonal=zonal, relief=relief,
                        rivers=rivers, control=planet_control(state, zonal))


def _life_features(coupled: CoupledSurface, ocean: np.ndarray) -> dict:
    """Return summary numbers for ice sheets and vegetation."""
    land = ~ocean
    out = {"climate_coupling_passes": coupled.passes}
    if land.any():
        out["ice_sheet_fraction_of_land"] = float(coupled.ice.glaciated[land].mean())
        out["max_ice_thickness_m"] = float(coupled.ice.thickness_m.max())
        out["vegetated_fraction_of_land"] = float(coupled.vegetation.cover[land].mean())
        out["productive_fraction_of_land"] = productive_share(coupled.vegetation, ocean)
    return out


def _refine_occupiability(state: PlanetState, productive: float) -> None:
    """Update the occupiability score with the productive land measured on the surface."""
    s = state
    s.occupiability = score_planet(s.star, s.bulk, s.interior, s.atmosphere, s.occupiability.weights, s.biosphere,
                                   productivity=productive / h.PRODUCTIVE_LAND_EARTH)


def _life_dataset(coupled: CoupledSurface, climate: SurfaceClimate, ocean: np.ndarray,
                  state: PlanetState) -> xr.Dataset:
    """Return ice sheets, vegetation, biomes and climate classes as surface variables."""
    rain = climate.rainfall
    ice = coupled.ice
    veg = coupled.vegetation
    glaciated = ice.glaciated
    lat_north = np.asarray(build_grid(ocean.size).points[:, 2] > 0.0)
    biome = classify_biomes(rain.temperature_k, rain.precipitation_m, rain.monthly_temperature_k, ocean, glaciated,
                            state.biosphere)
    data = {
        "ice_thickness": ("cell", ice.thickness_m.astype(np.float32), {"units": "m", "long_name": "ice-sheet thickness"}),
        "ice_surface": ("cell", ice.surface_rise_m.astype(np.float32),
                        {"units": "m", "long_name": "height of the ice surface above the bedrock"}),
        "snowline": ("cell", np.where(np.isfinite(ice.ela_m), ice.ela_m, np.nan).astype(np.float32),
                     {"units": "m", "long_name": "equilibrium-line altitude above sea level"}),
        "biome": ("cell", biome, {"long_name": "biome class (worldgen.biosphere.Biome)"}),
        "koppen": ("cell", np.where(ocean, 0, koppen(rain.monthly_temperature_k, rain.monthly_precipitation_m
                                                     if rain.monthly_precipitation_m is not None
                                                     else np.zeros_like(rain.monthly_temperature_k),
                                                     lat_north, glaciated)).astype(np.int8),
                   {"long_name": "Köppen–Geiger class (index into worldgen.biosphere.KOPPEN_CODES)"}),
        "land_albedo": ("cell", np.where(ocean, np.nan, veg.land_albedo).astype(np.float32),
                        {"long_name": "snow-free land albedo"}),
    }
    if state.biosphere is not None and state.biosphere.life == "surface":
        data["productivity"] = ("cell", veg.productivity.astype(np.float32),
                                {"units": "g/m2/yr", "long_name": "net primary productivity"})
        data["vegetation_cover"] = ("cell", veg.cover.astype(np.float32), {"long_name": "vegetation cover (0–1)"})
    return xr.Dataset(data_vars=data)


def _climate_features(climate: SurfaceClimate, ocean: np.ndarray) -> dict:
    """Return summary numbers for the surface climate."""
    rain = climate.rainfall
    out = {
        "climate_mean_temperature_k": climate.mean_k,
        "climate_planetary_albedo": climate.planetary_albedo,
    }
    if ocean.any():
        out["sea_ice_fraction_of_ocean"] = float(1.0 - climate.open_ocean_fraction)
        out["ocean_precipitation_m"] = float(rain.precipitation_m[ocean].mean())
    if (~ocean).any():
        out["land_precipitation_m"] = float(rain.precipitation_m[~ocean].mean())
        out["land_temperature_k"] = float(rain.temperature_k[~ocean].mean())
    return out


def _climate_dataset(climate: SurfaceClimate, has_ocean: bool, wet: bool) -> xr.Dataset:
    """Return temperature, precipitation and sea ice as surface variables."""
    rain = climate.rainfall
    month = ("month", np.arange(1, 13, dtype=np.int8),
             {"long_name": "month; month 1 begins at the northern winter solstice"})
    data = {
        "air_temperature": ("cell", rain.temperature_k.astype(np.float32),
                            {"units": "K", "long_name": "annual mean near-surface air temperature"}),
        "monthly_temperature": (("month", "cell"), rain.monthly_temperature_k.astype(np.float32),
                                {"units": "K", "long_name": "monthly mean near-surface air temperature"}),
    }
    if wet:
        data["precipitation"] = ("cell", rain.precipitation_m.astype(np.float32),
                                 {"units": "m/yr", "long_name": "annual precipitation"})
        data["monthly_precipitation"] = (("month", "cell"), rain.monthly_precipitation_m.astype(np.float32),
                                         {"units": "m/yr", "long_name": "monthly precipitation as an annual rate"})
        data["potential_evaporation"] = ("cell", rain.evaporation_m.astype(np.float32),
                                         {"units": "m/yr", "long_name": "potential evaporation"})
    if has_ocean:
        data["sea_ice"] = ("cell", climate.sea_ice.astype(np.float32),
                           {"long_name": "annual mean sea-ice cover (0–1)"})
    return xr.Dataset(data_vars=data, coords={"month": month})


def _hydrology_features(drainage: Drainage, ocean: np.ndarray, fields: SurfaceFields) -> dict:
    """Return summary numbers for lakes and rivers."""
    land = ~ocean
    if not land.any():
        return {}
    area_km2 = 4 * np.pi * (fields.radius_m / 1e3) ** 2 / fields.grid.size
    mouths = land & ocean[drainage.receiver]
    basins = np.bincount(drainage.basin[land])
    return {
        "lake_fraction_of_land": float(drainage.lake[land].mean()),
        "endorheic_fraction_of_land": float(drainage.endorheic[land].mean()),
        "largest_basin_km2": float(basins[1:].max() * area_km2) if basins.size > 1 else 0.0,
        "largest_river_m3_s": float(drainage.discharge_m3_s[mouths].max()) if mouths.any() else 0.0,
        "river_outflow_km3_yr": float(drainage.discharge_m3_s[mouths].sum() * c.SECONDS_PER_YEAR / 1e9),
    }


def _hydrology_dataset(drainage: Drainage, climate: SurfaceClimate) -> xr.Dataset:
    """Return runoff, lakes and rivers as surface variables."""
    return xr.Dataset(data_vars={
        "runoff": ("cell", climate.rainfall.runoff_m.astype(np.float32), {"units": "m/yr", "long_name": "runoff"}),
        "discharge": ("cell", drainage.discharge_m3_s.astype(np.float32),
                      {"units": "m3/s", "long_name": "mean flow leaving the cell"}),
        "flow_to": ("cell", drainage.receiver.astype(np.int32),
                    {"long_name": "cell the water flows to (itself at the ocean and closed lakes)"}),
        "drainage_basin": ("cell", drainage.basin, {"long_name": "drainage basin, 1 = largest, 0 = ocean"}),
        "endorheic": ("cell", drainage.endorheic, {"long_name": "drains to a lake without an outlet"}),
        "lake": ("cell", drainage.lake, {"long_name": "covered by a lake"}),
        "lake_level": ("cell", drainage.lake_level_m.astype(np.float32),
                       {"units": "m", "long_name": "lake surface elevation"}),
        "river_order": ("cell", drainage.river_order, {"long_name": "Strahler order of rivers, 0 = no river"}),
    })


def _sea_level(state: PlanetState, fields: SurfaceFields, has_ocean: bool,
               target: Optional[float], report: bool = True) -> tuple[Optional[OceanFill], float]:
    """Return the ocean fill and sea level, updating the water inventory when a land fraction is requested.

    Without surface water the level is the mean elevation. ``report`` adds
    notes on the water implied by a requested land fraction.
    """
    if not has_ocean:
        state.provenance["surface.land_fraction"] = "derived"
        return None, float(fields.elevation.mean())
    fill = ocean_fill(fields.grid, fields.elevation, fields.radius_m)
    water = state.water
    if target is None:
        state.provenance["surface.land_fraction"] = "derived"
        return fill, fill.level_for_volume(water.ocean_volume_m3)

    # The requested land fraction sets the amount of surface water.
    level = fill.level_for_land_fraction(target)
    volume = fill.volume(level)
    bulk = state.bulk
    surface = volume * h.SEAWATER_DENSITY / bulk.mass_kg
    total = water_model.total_water_for_surface(surface, water_model.gravity_ratio(bulk), water.plate_cycling)
    if not report:
        pass
    elif state.provenance.get("body.water_mass_fraction") in ("user", "user_range"):
        state.issues.append(Issue(
            "warning", "conflict", "surface",
            f"land fraction {target:.2f} needs a total water mass fraction of {total:.3g}, "
            f"not the {water.total_mass_fraction:.3g} given; the land fraction is used"))
    else:
        state.issues.append(Issue("info", "note", "surface",
                                  f"land fraction {target:.2f} sets the total water mass fraction to {total:.3g}"))
    if report and total > h.WATER_FRACTION_PLAUSIBLE_MAX:
        state.issues.append(Issue(
            "warning", "validity", "surface",
            f"a water mass fraction of {total:.3g} would normally drown the continents (Cowan & Abbot 2014)"))
    _set_surface_water(state, volume, volume)
    state.provenance["water.total_mass_fraction"] = "derived"
    return fill, level


def _set_surface_water(state: PlanetState, surface_volume_m3: float, ocean_volume_m3: float) -> float:
    """Set the water inventory from the surface water (as seawater) and the part in the ocean; return the total."""
    bulk = state.bulk
    surface = surface_volume_m3 * h.SEAWATER_DENSITY / bulk.mass_kg
    total = water_model.total_water_for_surface(surface, water_model.gravity_ratio(bulk), state.water.plate_cycling)
    state.water = replace(
        state.water,
        total_mass_fraction=total,
        mantle_mass_fraction=total - surface,
        surface_mass_kg=surface * bulk.mass_kg,
        ocean_volume_m3=ocean_volume_m3,
        seafloor_pressure_ratio=water_model.gravity_ratio(bulk)**2 * surface / h.SURFACE_WATER_EARTH,
    )
    return total


def _refine_global_climate(state: PlanetState, land_by_band: np.ndarray) -> None:
    """Replace the global atmosphere and climate with those for the built surface's land distribution."""
    atmosphere, climate, _, issues = refine_climate(state, land_by_band)
    before = state.climate.mean_temperature_k
    state.atmosphere, state.climate = atmosphere, climate
    state.issues = [i for i in state.issues if i.subsystem not in ("atmosphere", "climate")] + issues
    s = state
    s.occupiability = score_planet(s.star, s.bulk, s.interior, s.atmosphere, s.occupiability.weights, s.biosphere)
    if abs(climate.mean_temperature_k - before) > h.CLIMATE_TARGET_MISS_K:
        state.issues.append(Issue("info", "note", "climate",
                                  f"with the built surface's land distribution the global mean temperature is "
                                  f"{climate.mean_temperature_k:.0f} K (estimate before the surface: {before:.0f} K)"))


def _transport_features(zonal, radius_m: float, synchronous: bool) -> dict:
    """Return the Tier 0 atmospheric and oceanic poleward heat transport at 35° N (PW), for rotating planets."""
    if synchronous or zonal.atmosphere_transport is None:
        return {}
    n = zonal.coordinate.size
    edges = -1.0 + 2.0 / n * np.arange(1, n)
    x = math.sin(math.radians(35.0))
    scale = radius_m**2 / 1e15
    return {"atmosphere_heat_transport_35n_pw": float(np.interp(x, edges, zonal.atmosphere_transport)) * scale,
            "ocean_heat_transport_35n_pw": float(np.interp(x, edges, zonal.ocean_transport)) * scale}


def _store_ice(state: PlanetState, fields: SurfaceFields, coupled: CoupledSurface, ocean: np.ndarray,
               target: Optional[float], couple) -> tuple[CoupledSurface, np.ndarray, dict]:
    """Take the water in ice sheets from the ocean, lowering sea level, and return the re-coupled surface.

    With a requested land fraction the sea level stays and the planet's
    water rises by the ice instead. Ice sheets holding more than
    ``ICE_MAX_WATER_SHARE`` of the surface water are thinned to that share.
    """
    grid = fields.grid
    cell = 4 * np.pi * fields.radius_m**2 / grid.size
    as_water = ICE_DENSITY / h.SEAWATER_DENSITY
    ice_volume = float(coupled.ice.thickness_m.sum()) * cell * as_water
    if ice_volume <= 0.0:
        return coupled, ocean, {}
    ocean_volume = state.water.ocean_volume_m3
    limit = h.ICE_MAX_WATER_SHARE * (ocean_volume if target is None else ocean_volume / (1 - h.ICE_MAX_WATER_SHARE))
    thinned = ice_volume > limit
    if thinned:
        scale = limit / ice_volume
        ice = coupled.ice
        coupled = replace(coupled, ice=replace(ice, thickness_m=ice.thickness_m * scale,
                                               surface_rise_m=ice.surface_rise_m * scale))
        ice_volume = limit
        state.issues.append(Issue("info", "heuristic", "climate",
                                  f"ice sheets thinned to hold {h.ICE_MAX_WATER_SHARE:.0%} of the surface water "
                                  "(their profile would need more water than the planet has)"))
    drop = 0.0
    if target is None:
        fill = ocean_fill(grid, fields.elevation, fields.radius_m)
        level = fill.level_for_volume(ocean_volume - ice_volume)
        drop = -level
        ocean = apply_sea_level(fields, fill, level)
        state.water = replace(state.water, ocean_volume_m3=ocean_volume - ice_volume)
    else:
        total = _set_surface_water(state, ocean_volume + ice_volume, ocean_volume)
        state.issues.append(Issue("info", "note", "surface",
                                  f"with ice sheets the total water mass fraction is {total:.3g}"))
    if drop > h.SEA_LEVEL_RECOUPLE_M or thinned:
        coupled = couple(ocean, coupled)
    ocean_area = max(ocean.mean(), 1e-9) * 4 * np.pi * fields.radius_m**2
    return coupled, ocean, {"ice_sheet_volume_km3": ice_volume / as_water / 1e9,
                            "ice_sea_level_equivalent_m": ice_volume / ocean_area,
                            "glacial_sea_level_drop_m": drop}


def _snapshot_dataset(snapshots, grid, state: PlanetState, fill: Optional[OceanFill],
                      volume_m3: float) -> xr.Dataset:
    """Return simulation snapshots on (time, cell), with the final surface's water (ice included) as ocean."""
    elevation = np.stack([s.elevation for s in snapshots]).astype(np.float64)
    for row in elevation:
        if fill is None:
            row -= row.mean()
        else:
            row -= ocean_fill(grid, row, state.bulk.radius_m).level_for_volume(volume_m3)
    elevation = elevation.astype(np.float32)
    return xr.Dataset(
        data_vars={
            "snapshot_elevation": (("time", "cell"), elevation,
                                   {"units": "m", "long_name": "simulated elevation, without surface detail"}),
            "snapshot_plate": (("time", "cell"), np.stack([s.plate for s in snapshots]),
                               {"long_name": "plate index"}),
            "snapshot_continental": (("time", "cell"), np.stack([s.continental for s in snapshots]),
                                     {"long_name": "continental crust"}),
        },
        coords={"time": ("time", np.array([s.time_myr for s in snapshots], dtype=np.float32),
                         {"units": "Myr", "long_name": "time relative to the present surface"})},
    )
