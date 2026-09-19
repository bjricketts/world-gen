"""Plate-tectonic simulation: plates move, collide, subduct, rift and merge over geological time.

``build_simulated_terrain`` runs the simulation and fills the surface fields
of a mobile-lid planet from its final state.
"""

from __future__ import annotations

import numpy as np

from .. import heuristics as h
from ..noise import fbm, ridged
from ..surface import plates as heuristic_plates
from ..surface.drive import TectonicDrive
from ..surface.distance import distance_from, smooth
from ..surface.fields import Boundary, Crust as CrustType, SurfaceFields, Terrain
from ..hydrology import WaterSetting
from .crust import OROGENY_ARC, Crust, Plates, ocean_depth
from .initial import STARTS
from .simulate import SimulationResult, Snapshot, run

__all__ = ["STARTS", "SimulationResult", "Snapshot", "TectonicDrive", "build_simulated_terrain", "run"]

RIDGE_AGE_CELLS = 1.5        # oceanic cells within this many grid spacings of a ridge are marked as ridge
MOUNTAIN_UPLIFT_M = 1500.0   # uplift above the continental base that counts as a mountain (Earth gravity)
ARC_ISLAND_M = 1000.0        # arc relief above the sea floor that counts as a mountain (Earth gravity)
VOLCANO_RELIEF_M = 1000.0    # hotspot relief that counts as a volcano (Earth gravity)
RIDGED_MEDIAN = 0.52         # median of the ridged noise field


def _compact_plate_ids(plate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return plate ids renumbered 0..k-1 and the original id of each new one."""
    original, compact = np.unique(plate, return_inverse=True)
    return compact.astype(np.int16), original


def _boundaries(fields: SurfaceFields, plates: Plates, original: np.ndarray):
    """Return the boundary edges between plates with their type, using the plates' final motion."""
    omega = plates.omega[original]
    speed = np.linalg.norm(omega, axis=1)
    layout = heuristic_plates.Plates(
        count=len(original),
        seeds=np.zeros((len(original), 3)),
        euler_poles=omega / np.maximum(speed, 1e-30)[:, None],
        angular_speed=speed / 1e6,
        cell_plate=fields.plate,
    )
    velocity = heuristic_plates.cell_velocity(fields, layout)
    return heuristic_plates.classify_boundaries(fields, layout, velocity)


def _elevation(fields: SurfaceFields, continental: np.ndarray, cont_elev: np.ndarray, ocean_age: np.ndarray,
               ocean_extra: np.ndarray, sediment: np.ndarray, relief: float, seed: int, detail: bool = True
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return the elevation built from crust properties on the grid, and the mountain uplift on continents.

    Continental margins are lowered into shelves and coastlines smoothed.
    With ``detail``, small-scale noise and mountain crests are added.
    """
    pts = fields.grid.points
    cont = continental
    raw = np.where(cont, cont_elev, ocean_depth(ocean_age, relief) + ocean_extra + sediment)
    uplift = np.where(cont, np.maximum(cont_elev - h.CONTINENT_BASE_M, 0.0), 0.0)
    flat = np.where(cont, np.minimum(cont_elev, h.CONTINENT_BASE_M), raw)
    coast_km, _ = distance_from(fields.grid, np.flatnonzero(~cont), fields.radius_m / 1e3,
                                limit_km=5 * h.SIM_MARGIN_WIDTH_KM)
    flat -= np.where(cont, h.SIM_MARGIN_DROP_M * np.exp(-coast_km / h.SIM_MARGIN_WIDTH_KM), 0.0)
    flat = smooth(fields.grid, flat, h.MARGIN_SMOOTHING_STEPS)
    if not detail:
        return flat + uplift, uplift

    crests = ridged(pts, seed, "mountains", 6.0, 5)
    rough = (crests / RIDGED_MEDIAN) ** h.SIM_MOUNTAIN_SHARPNESS
    noise = np.where(cont, h.SIM_CONTINENT_NOISE_M * fbm(pts, seed, "continent.detail", 3.0, 6),
                     h.OCEAN_NOISE_M * fbm(pts, seed, "ocean.detail", 4.0, 5))
    arc = np.where(~cont, np.maximum(ocean_extra, 0.0), 0.0)
    mountains = uplift * rough
    out = flat + noise + mountains + arc * (rough - 1.0)

    cap = h.MAX_ELEVATION_EARTH_M * relief
    soft = 0.6 * cap
    return np.where(out > soft, soft + 0.4 * cap * np.tanh((out - soft) / (0.4 * cap)), out), mountains


def _terrain(fields: SurfaceFields, crust: Crust, mountains: np.ndarray, hotspots: np.ndarray,
             relief: float, half_rate_km_myr: float) -> None:
    """Classify each cell's landform from the crust state."""
    cont = crust.continental
    terrain = fields.terrain
    recent = np.nan_to_num(crust.orogeny_age, nan=np.inf) < h.MOUNTAIN_OROGENY_MAX_MYR

    terrain[:] = Terrain.PLAIN
    terrain[cont & (fields.elevation > 1500)] = Terrain.HIGHLAND
    ridge_age = RIDGE_AGE_CELLS * fields.spacing_km / max(half_rate_km_myr, 1e-6)
    terrain[~cont & (crust.ocean_age < ridge_age)] = Terrain.RIDGE
    terrain[cont & recent & (mountains > MOUNTAIN_UPLIFT_M * relief)] = Terrain.MOUNTAIN
    arc = ~cont & (crust.orogeny_kind == OROGENY_ARC) & (crust.ocean_extra > ARC_ISLAND_M * relief)
    terrain[arc] = Terrain.MOUNTAIN
    at_boundary = smooth(fields.grid, (fields.boundary != Boundary.NONE).astype(float), 2) > 0
    terrain[~cont & at_boundary & (crust.ocean_extra < -0.4 * h.TRENCH_DEPTH_M)] = Terrain.TRENCH

    # Continental cells next to young sea floor are rifts.
    i, j = fields.grid.edges()
    young = ~cont & (crust.ocean_age < h.YOUNG_RIFT_MYR)
    rift = np.zeros(len(cont), dtype=bool)
    rift[i[cont[i] & young[j]]] = True
    rift[j[cont[j] & young[i]]] = True
    terrain[rift] = Terrain.RIFT

    # Volcanoes: active hotspots, and the island chains they left on the sea floor.
    islands = ~cont & crust.volcanic & (crust.ocean_extra > VOLCANO_RELIEF_M * relief)
    terrain[islands] = Terrain.VOLCANO
    if len(hotspots):
        radius = max(h.HOTSPOT_RADIUS_KM, 0.8 * fields.spacing_km) / (fields.radius_m / 1e3)
        near = np.concatenate([np.asarray(c, dtype=np.int64)
                               for c in fields.grid.tree.query_ball_point(hotspots, radius)])
        terrain[near] = Terrain.VOLCANO


def build_simulated_terrain(fields: SurfaceFields, activity: float, land_fraction: float, relief: float,
                            age_gyr: float, seed: int, start: str, duration_myr: float,
                            snapshot_interval_myr: float | None = None,
                            water: WaterSetting | None = None,
                            drive: TectonicDrive | None = None) -> tuple[dict, SimulationResult]:
    """Simulate plate tectonics, fill the surface fields and return summary attributes and the full result.

    The simulated time is limited to the planet's age. With ``drive`` the
    plate speed and the volcanism follow the planet's integrated history.
    """
    duration = min(duration_myr, age_gyr * 1e3)
    continental_target = float(np.clip(land_fraction + h.CONTINENTAL_SHELF_EXTRA, 0.02, h.CONTINENTAL_MAX_FRACTION))
    result = run(fields.grid, fields.radius_m, activity, continental_target, relief, start, duration, seed,
                 snapshot_interval_myr, water, drive)
    crust = result.crust
    cont = crust.continental

    fields.plate[:], original = _compact_plate_ids(crust.plate)
    fields.crust[:] = np.where(cont, CrustType.CONTINENTAL, CrustType.OCEANIC)
    fields.crust_age_myr[:] = np.where(cont, np.nan, crust.ocean_age)
    fields.orogeny_age_myr[:] = crust.orogeny_age
    fields.elevation[:], mountains = _elevation(fields, cont, crust.cont_elev, crust.ocean_age, crust.ocean_extra,
                                                crust.sediment, relief, seed)
    # Snapshots omit the surface detail, which is fixed in place and would not move with the plates.
    for snap in result.snapshots:
        snap.elevation = _elevation(fields, snap.continental, snap.cont_elev, snap.ocean_age, snap.ocean_extra,
                                    snap.sediment, relief, seed, detail=False)[0].astype(np.float32)

    i, j, kind, _ = _boundaries(fields, result.plates, original)
    for code in (Boundary.TRANSFORM, Boundary.DIVERGENT, Boundary.CONVERGENT):
        sel = kind == code
        fields.boundary[i[sel]] = code
        fields.boundary[j[sel]] = code

    speed_km_myr = (h.PLATE_SPEED_EARTH_CM_YR * heuristic_plates.KM_PER_MYR_PER_CM_YR * max(activity, 0.05) ** 0.5
                    if drive is None else drive.speed_m_myr() / 1e3)
    half_rate = speed_km_myr / 2
    _terrain(fields, crust, mountains, result.hotspots, relief, half_rate)

    ocean_age = crust.ocean_age[~cont]
    stats = result.stats
    features = {
        "plates": len(original),
        "hotspots": len(result.hotspots),
        "continental_crust_fraction": float(cont.mean()),
        "simulated_myr": result.duration_myr,
        "subducted_points": stats.subducted,
        "plate_merges": stats.merges,
        "rifts": stats.rifts,
        "subduction_initiations": stats.initiations,
        "median_ocean_age_myr": float(np.median(ocean_age)) if ocean_age.size else 0.0,
        "plate_speed_km_myr": float(speed_km_myr),
    }
    return features, result
