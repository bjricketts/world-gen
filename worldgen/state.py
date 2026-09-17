"""Data structures describing a generated planet at one epoch.

All values are SI unless the field name says otherwise. Each state object is a
plain dataclass so it can be printed, compared and serialised easily.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

Provenance = Literal["user", "user_range", "archetype", "prior", "derived", "heuristic", "default"]
IssueLevel = Literal["info", "warning", "error"]
IssueKind = Literal["heuristic", "conflict", "validity", "note"]


@dataclass
class Issue:
    """A note for the consistency report."""

    level: IssueLevel
    kind: IssueKind
    subsystem: str
    message: str


@dataclass
class StarState:
    """Host star at the planet's epoch."""

    mass_kg: float
    age_s: float
    metallicity_feh: float
    luminosity_w: float
    radius_m: float
    effective_temperature_k: float
    main_sequence_lifetime_s: float
    spectral_type: str
    xuv_fraction: float            # L_XUV / L_bol
    xuv_fraction_relative_sun: float


@dataclass
class HabitableZone:
    """Habitable-zone boundaries as instellation (Earth units) and distance (m)."""

    recent_venus_s: float
    runaway_greenhouse_s: float
    maximum_greenhouse_s: float
    early_mars_s: float
    recent_venus_m: float
    runaway_greenhouse_m: float
    maximum_greenhouse_m: float
    early_mars_m: float
    position: str
    conservative_fraction: float   # 0 at the runaway limit, 1 at the maximum-greenhouse limit (by distance)


@dataclass
class OrbitState:
    """Orbit and spin of the planet."""

    semi_major_axis_m: float
    eccentricity: float
    obliquity_rad: float
    orbital_period_s: float
    rotation_period_s: float
    instellation_earth: float      # orbit-averaged, Earth = 1
    tidal_lock_time_s: float
    tidal_q: float
    spin_state: str                # "free", "synchronous", "3:2 resonance"
    hill_radius_m: float
    habitable_zone: HabitableZone
    periapsis_longitude_rad: float = 0.0   # solar longitude at periapsis


@dataclass
class BulkState:
    """Bulk properties of the planet."""

    mass_kg: float
    radius_m: float
    core_mass_fraction: float
    water_mass_fraction: float
    envelope_mass_fraction: float
    planet_class: str
    density_kg_m3: float
    surface_gravity_m_s2: float
    escape_velocity_m_s: float
    moment_of_inertia_factor: float


@dataclass
class InteriorState:
    """Interior heat budget and geological regime."""

    radiogenic_power_w: float
    tidal_power_w: float
    surface_heat_flux_w_m2: float
    activity_index: float
    tectonic_regime: str
    regime_probabilities: dict[str, float]
    magnetic_field: bool
    dynamo_probability: float


@dataclass
class AtmosphereState:
    """Atmosphere and global mean surface conditions."""

    present: bool
    retention_probability: float
    shoreline_ratio: float         # < 1 means inside the cosmic shoreline
    surface_pressure_pa: float
    composition: str
    mean_molecular_weight: float
    bond_albedo: float
    equilibrium_temperature_k: float
    greenhouse_optical_depth: float
    surface_temperature_k: float
    scale_height_m: float
    water_boiling_point_k: float
    surface_water: str             # "liquid", "ice", "vapour", "none"
    weathering_regulated: bool
    weathering_target_k: float
    runaway_greenhouse: bool


@dataclass
class ClimateSummary:
    """Global climate from the zonal energy-balance model (Tier 0)."""

    mean_temperature_k: float
    planetary_albedo: float
    ice_free_albedo: float         # albedo the model starts from (the user's global value if albedo_fixed)
    albedo_fixed: bool
    open_water_fraction: float     # share of the whole surface that is open water (annual mean)
    open_ocean_fraction: float     # share of the ocean free of ice
    ice_line_deg: float            # latitude (substellar angle if synchronous) where permanent sea ice begins
    dayside_temperature_k: Optional[float]
    nightside_temperature_k: Optional[float]
    snowball_stable: bool          # a fully frozen state is also stable
    converged: bool


@dataclass
class WaterState:
    """Water inventory and its split between mantle and surface."""

    total_mass_fraction: float     # all water / planet mass
    mantle_mass_fraction: float    # water held in the mantle / planet mass
    surface_mass_kg: float
    ocean_volume_m3: float         # surface water as liquid seawater
    seafloor_pressure_ratio: float  # relative to Earth's
    plate_cycling: bool            # exchange with the mantle through subduction
    land_fraction_estimate: float  # from a reference hypsometry, before the surface is built


@dataclass
class BiosphereState:
    """Life on the planet and its effect on the air."""

    life: str                      # "surface", "ocean", "subsurface", "none"
    biochemistry: str              # "oxygenic", "methanogenic", "alien"
    product: str                   # gas released: "o2", "ch4", "none"
    age_gyr: float
    oxygen_fraction: float         # O2 mole fraction of the air
    methane_fraction: float        # CH4 mole fraction of the air
    optimum_temperature_k: float
    temperature_tolerance_k: float
    pigment_absorption_nm: float
    pigment_colour: str            # "#rrggbb", vegetation seen under the star's light
    vegetation_albedo: float
    productivity: float            # expected land productivity relative to Earth's (global state estimate)
    weathering_cooling_k: float    # lowering of the weathering target by biota


@dataclass
class Occupiability:
    """How suitable the planet is for people, unaided or with technology."""

    score: float
    factors: dict[str, float]
    habitable_unaided: bool
    weights: Optional[dict[str, float]] = None   # user weights the score was computed with


@dataclass
class SurfaceSummary:
    """Headline numbers for a generated global surface."""

    grid_size: int
    spacing_km: float
    land_fraction: float
    has_ocean: bool
    frozen_ocean: bool
    relief_factor: float
    min_elevation_m: float
    max_elevation_m: float
    features: dict[str, float]


@dataclass
class TimelineEvent:
    """An event in a planet's history (history mode)."""

    time_s: float                  # planet age
    kind: str                      # e.g. 'ocean_loss', 'runaway_onset', 'dynamo_shutdown', 'snowball_onset'
    detail: str
    flagged: bool = False          # physics overrode a value the user asked to hold


@dataclass
class Override:
    """A user value applied at the end of a history run in place of the integrated value."""

    field: str                     # spec path
    history_value: object
    user_value: object


@dataclass
class PlanetState:
    """Everything known about a generated planet at one epoch."""

    name: str
    seed: int
    draw_seed: int                 # seed of the accepted draw (differs from seed after redraws)
    attempts: int                  # draws needed to satisfy the archetype's constraints
    constraints_met: bool
    mode: str
    archetype: Optional[str]
    epoch_s: float
    star: StarState
    orbit: OrbitState
    bulk: BulkState
    interior: InteriorState
    atmosphere: AtmosphereState
    occupiability: Occupiability
    inputs: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    water: Optional[WaterState] = None
    surface: Optional[SurfaceSummary] = None
    climate: Optional[ClimateSummary] = None
    biosphere: Optional[BiosphereState] = None
    events: list[TimelineEvent] = field(default_factory=list)       # history mode: what happened, oldest first
    overrides: list[Override] = field(default_factory=list)         # history mode: user values applied at the end

    def to_dict(self) -> dict:
        """Return the state as nested plain Python types."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PlanetState":
        """Return a state rebuilt from ``to_dict`` output (e.g. a saved YAML file)."""
        d = dict(data)
        orbit = dict(d["orbit"])
        orbit["habitable_zone"] = HabitableZone(**orbit["habitable_zone"])
        d["star"] = StarState(**d["star"])
        d["orbit"] = OrbitState(**orbit)
        d["bulk"] = BulkState(**d["bulk"])
        d["interior"] = InteriorState(**d["interior"])
        d["atmosphere"] = AtmosphereState(**d["atmosphere"])
        d["occupiability"] = Occupiability(**d["occupiability"])
        d["issues"] = [Issue(**i) for i in d.get("issues", [])]
        if d.get("water") is not None:
            d["water"] = WaterState(**d["water"])
        if d.get("biosphere") is not None:
            d["biosphere"] = BiosphereState(**d["biosphere"])
        if d.get("climate") is not None:
            d["climate"] = ClimateSummary(**d["climate"])
        if d.get("surface") is not None:
            d["surface"] = SurfaceSummary(**d["surface"])
        d["events"] = [TimelineEvent(**e) for e in d.get("events", [])]
        d["overrides"] = [Override(**o) for o in d.get("overrides", [])]
        return cls(**d)


@dataclass
class Timeline:
    """Global states over time. Snapshot mode has a single point.

    ``states`` are full planet states at ``times_s`` (chosen epochs and the
    target epoch); ``series`` holds sampled history values on
    ``series['time_gyr']`` for figures; ``events`` lists what happened.
    """

    times_s: list[float] = field(default_factory=list)
    states: list[PlanetState] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)
    series: dict[str, list] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the timeline as nested plain Python types (states included)."""
        return {"times_s": list(self.times_s), "states": [s.to_dict() for s in self.states],
                "events": [asdict(e) for e in self.events], "series": dict(self.series)}

    @classmethod
    def from_dict(cls, data: dict) -> "Timeline":
        """Return a timeline rebuilt from ``to_dict`` output."""
        return cls(times_s=list(data.get("times_s", [])),
                   states=[PlanetState.from_dict(s) for s in data.get("states", [])],
                   events=[TimelineEvent(**e) for e in data.get("events", [])],
                   series={k: list(v) for k, v in data.get("series", {}).items()})
