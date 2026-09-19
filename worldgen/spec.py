"""User-facing planet specification: what the user fixes, constrains or leaves open.

Each numeric parameter can be:
  * a number: fixed value,
  * a two-element list ``[low, high]``: drawn from that range,
  * omitted / ``null``: drawn from the archetype prior or derived from physics.

Units in the spec are user-friendly (solar masses, AU, Earth masses, bar,
hours, degrees); they are converted to SI when the planet is built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

Range = tuple[float, float]
Param = Optional[Union[float, Range]]

TectonicRegime = Literal["mobile_lid", "stagnant_lid", "episodic", "heat_pipe", "inactive", "fluid"]
Composition = Literal["n2_o2", "n2_co2", "co2", "n2_ch4", "h2_he", "none"]
Mode = Literal["snapshot", "history"]
TectonicsMode = Literal["simulated", "heuristic"]
TectonicStart = Literal["supercontinent", "cratons"]
LifeMode = Literal["surface", "ocean", "subsurface", "none"]
Biochemistry = Literal["oxygenic", "methanogenic", "alien"]
AlienProduct = Literal["o2", "ch4", "none"]


class _Section(BaseModel):
    """Base for spec sections: rejects unknown keys and checks ranges."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _check_range(cls, v):
        """Reject ranges whose low value exceeds the high value."""
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            if v[0] > v[1]:
                raise ValueError(f"range low value {v[0]} exceeds high value {v[1]}")
        return v


class StarSpec(_Section):
    """Host star parameters."""

    mass_msun: Param = None
    age_gyr: Param = None
    metallicity_feh: Param = None
    activity_percentile: Param = None    # rotation percentile at birth (0 slow, 1 fast); sets the XUV history


class OrbitSpec(_Section):
    """Orbit and spin. Give either semi-major axis or instellation, not both."""

    semi_major_axis_au: Param = None
    instellation_earth: Param = None
    eccentricity: Param = None
    obliquity_deg: Param = None
    rotation_period_h: Param = None
    periapsis_longitude_deg: Param = None   # season at periapsis, as solar longitude (Earth ~283)

    @model_validator(mode="after")
    def _one_distance(self):
        """Reject specs that give both a semi-major axis and an instellation."""
        if self.semi_major_axis_au is not None and self.instellation_earth is not None:
            raise ValueError("give either semi_major_axis_au or instellation_earth, not both")
        return self


class BodySpec(_Section):
    """Bulk properties of the planet. Any two of mass, radius and composition fix the third."""

    mass_mearth: Param = None
    radius_rearth: Param = None
    core_mass_fraction: Param = None
    water_mass_fraction: Param = None    # all water, including what the mantle holds (Earth ~0.0006)
    envelope_mass_fraction: Param = None
    radiogenic_abundance: Param = None   # heat-producing elements relative to Earth
    tidal_heating_w: Param = None
    tidal_q: Param = None                # tidal dissipation quality factor


class InteriorSpec(_Section):
    """Overrides for interior outcomes that are otherwise drawn from heuristics."""

    tectonic_regime: Optional[TectonicRegime] = None
    magnetic_field: Optional[bool] = None


class AtmosphereSpec(_Section):
    """Atmosphere parameters and overrides."""

    present: Optional[bool] = None
    surface_pressure_bar: Param = None
    composition: Optional[Composition] = None
    bond_albedo: Param = None
    surface_temperature_k: Param = None
    volatile_richness: Param = None      # volatile inventory relative to Earth
    weathering_target_k: Param = None    # temperature the carbonate-silicate cycle settles at


class SurfaceSpec(_Section):
    """Surface targets used when building maps."""

    land_fraction: Param = None          # target land fraction; sets the surface water when given
    tectonics: Optional[TectonicsMode] = None      # plate simulation, or the fast heuristic layout
    tectonics_start: Optional[TectonicStart] = None  # initial continents of the simulation
    tectonics_duration_myr: Param = None           # simulated time (Myr), limited by the planet's age


class BiosphereSpec(_Section):
    """Life on the planet. Unset values follow rules from the planet's water, temperature and age."""

    life: Optional[LifeMode] = None               # surface (land and sea), ocean only, subsurface, or none
    biochemistry: Optional[Biochemistry] = None   # oxygenic photosynthesis (default), methanogenic, or alien
    alien_product: Optional[AlienProduct] = None  # gas an alien biosphere releases (default none)
    age_gyr: Param = None                         # time since life began (default: planet age − 0.7 Gyr)
    optimum_temperature_k: Param = None           # growth optimum (Earth-like 298 K)
    temperature_tolerance_k: Param = None         # growth stops this far above the optimum (Earth-like 25 K)
    pigment_absorption_nm: Param = None           # peak absorption of the main pigment (default from the star)
    oxygen_fraction: Param = None                 # O2 mole fraction of the air (default from biosphere age)


class HistorySpec(_Section):
    """Initial conditions and poorly constrained rates for history mode. Unset values are drawn."""

    epochs_gyr: Optional[list[float]] = None      # extra epochs at which full states are stored
    initial_water_mass_fraction: Param = None     # all water at formation (default: body.water_mass_fraction)
    carbon_inventory: Param = None                # total carbon relative to Earth's, per unit planet mass
    nitrogen_inventory: Param = None              # atmospheric nitrogen relative to Earth's, per unit planet mass
    initial_mantle_temperature_k: Param = None    # mantle potential temperature at formation
    outgassing_efficiency: Param = None           # volcanic degassing relative to Earth's at the same melt rate
    weathering_efficiency: Param = None           # silicate weathering relative to Earth's at the same climate
    weathering_temperature_scale_k: Param = None  # e-folding temperature of weathering kinetics
    weathering_co2_exponent: Param = None         # weathering ∝ pCO₂^exponent
    biotic_weathering_factor: Param = None        # speed-up of land weathering by a full land biosphere
    escape_efficiency: Param = None               # heating efficiency of XUV-driven escape
    mantle_activation_energy_kj: Param = None     # activation energy of mantle viscosity
    core_adiabatic_heat_flow: Param = None        # heat the core conducts along its adiabat, relative to Earth's
    life_origin_delay_gyr: Param = None           # time with habitable conditions before life appears
    oxygen_burial_efficiency: Param = None        # organic burial (O₂ source) relative to Earth's
    reductant_decay_gyr: Param = None             # e-folding time of the volcanic reductant flux (mantle oxidation)
    land_colonisation_delay_gyr: Param = None     # time from the origin of surface life to life on land

    @field_validator("epochs_gyr")
    @classmethod
    def _positive_epochs(cls, v):
        """Reject negative epochs."""
        if v is not None and any(t < 0 for t in v):
            raise ValueError("epochs must be non-negative")
        return v


class PriorSpec(_Section):
    """How unset parameters are drawn."""

    archetype: Optional[str] = None
    occupiability_weights: Optional[dict[str, float]] = None
    enforce_constraints: bool = True     # redraw until the archetype's required outcomes hold
    max_attempts: int = 200

    @field_validator("max_attempts")
    @classmethod
    def _positive(cls, v: int) -> int:
        """Reject a maximum attempt count below one."""
        if v < 1:
            raise ValueError("max_attempts must be at least 1")
        return v


class PlanetSpec(BaseModel):
    """Complete specification of a planet to generate."""

    model_config = ConfigDict(extra="forbid")

    name: str = "Unnamed"
    seed: int = 0
    mode: Mode = "snapshot"
    target_epoch_gyr: Optional[float] = None
    star: StarSpec = StarSpec()
    orbit: OrbitSpec = OrbitSpec()
    body: BodySpec = BodySpec()
    interior: InteriorSpec = InteriorSpec()
    atmosphere: AtmosphereSpec = AtmosphereSpec()
    surface: SurfaceSpec = SurfaceSpec()
    biosphere: BiosphereSpec = BiosphereSpec()
    history: HistorySpec = HistorySpec()
    priors: PriorSpec = PriorSpec()


# Fields describing the planet's present state rather than its initial
# conditions. In history mode, user values for these are targets/overrides
# applied at the end of the evolution; in snapshot mode they are inputs.
STATE_FIELDS = frozenset(
    {
        "orbit.rotation_period_h",
        "body.water_mass_fraction",
        "interior.tectonic_regime",
        "interior.magnetic_field",
        "atmosphere.present",
        "atmosphere.surface_pressure_bar",
        "atmosphere.composition",
        "atmosphere.bond_albedo",
        "atmosphere.surface_temperature_k",
        "atmosphere.weathering_target_k",
        "surface.land_fraction",
        "biosphere.life",
        "biosphere.oxygen_fraction",
    }
)


def field_role(path: str, mode: Mode) -> str:
    """Return whether a spec field acts as an 'input' or a 'target' in the given mode."""
    if mode == "history" and path in STATE_FIELDS:
        return "target"
    return "input"


def load_spec(path: str | Path) -> PlanetSpec:
    """Read a planet specification from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return PlanetSpec.model_validate(data)


def save_spec(spec: PlanetSpec, path: str | Path) -> None:
    """Write a planet specification to a YAML file, omitting unset fields."""
    data = spec.model_dump(mode="json", exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


SPEC_TEMPLATE = """\
# World generator planet specification.
# Each numeric value may be a number (fixed), a [low, high] range (drawn),
# or omitted (drawn from the archetype prior or derived from physics).

name: New World
seed: 42
mode: snapshot            # snapshot | history (integrated from formation)

priors:
  archetype: temperate    # see `worldgen archetypes`; omit for a weighted mix
  # occupiability_weights: {temperature: 1, gravity: 1, pressure: 1, water: 1, radiation: 1}
  # enforce_constraints: true     # redraw until the archetype's required outcomes hold
  # max_attempts: 200

star:
  mass_msun: 0.9          # solar masses
  # age_gyr: [3, 6]
  # metallicity_feh: 0.0
  # activity_percentile: 0.5      # birth rotation percentile: 0 slow, 1 fast (XUV history)

orbit:
  # semi_major_axis_au: 1.0   # or give instellation_earth instead
  instellation_earth: [0.8, 1.1]
  # eccentricity: 0.02
  # obliquity_deg: 23
  # rotation_period_h: 24
  # periapsis_longitude_deg: 283   # season of closest approach (solar longitude; 270 = northern winter solstice)

body:
  mass_mearth: 1.3
  # radius_rearth: 1.1
  # core_mass_fraction: 0.32
  # water_mass_fraction: 0.0006   # total water; Earth keeps ~1/3 at the surface
  # envelope_mass_fraction: 0.0
  # radiogenic_abundance: 1.0
  # tidal_heating_w: 0
  # tidal_q: 100                  # tidal dissipation; lower means faster tidal locking

interior: {}
  # tectonic_regime: mobile_lid   # mobile_lid | stagnant_lid | episodic | heat_pipe | inactive
  # magnetic_field: true

surface: {}
  # land_fraction: 0.3            # overrides the water: sea level is set to leave this much land
  # tectonics: simulated          # simulated | heuristic (fast, no history)
  # tectonics_start: cratons      # supercontinent | cratons
  # tectonics_duration_myr: 400   # simulated time, limited by the star's age

biosphere: {}
  # life: surface                 # surface | ocean | subsurface | none (default from water, temperature and age)
  # biochemistry: oxygenic        # oxygenic | methanogenic | alien
  # alien_product: none           # o2 | ch4 | none (alien biochemistry)
  # age_gyr: 3.8                  # time since life began
  # optimum_temperature_k: 298
  # temperature_tolerance_k: 25
  # pigment_absorption_nm: 680    # default: where the star's light peaks at the surface
  # oxygen_fraction: 0.21

history: {}                # used in history mode; unset values are drawn
  # epochs_gyr: [0.5, 1, 2, 3]    # states stored at these planet ages
  # initial_water_mass_fraction: 0.0007
  # carbon_inventory: 1.0         # relative to Earth, per unit planet mass
  # nitrogen_inventory: 1.0
  # initial_mantle_temperature_k: 2000
  # outgassing_efficiency: 1.0
  # weathering_efficiency: 1.0
  # weathering_temperature_scale_k: 20
  # weathering_co2_exponent: 0.45
  # biotic_weathering_factor: 4
  # escape_efficiency: 0.15
  # mantle_activation_energy_kj: 300
  # core_adiabatic_heat_flow: 1.0
  # life_origin_delay_gyr: 0.4
  # oxygen_burial_efficiency: 1.0
  # reductant_decay_gyr: 3.0
  # land_colonisation_delay_gyr: 3.4

atmosphere: {}
  # present: true
  # surface_pressure_bar: 1.2
  # composition: n2_o2            # n2_o2 | n2_co2 | co2 | n2_ch4 | h2_he | none
  # bond_albedo: 0.3
  # surface_temperature_k: 290
  # volatile_richness: 1.0
  # weathering_target_k: 288      # temperature CO2 regulation settles at (plate tectonics + water)
"""
