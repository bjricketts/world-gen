# Spec file reference

A planet is described by a YAML spec. Every field is optional: what the spec leaves out is drawn from the archetype prior or derived from physics, and the report records where each value came from (user, range, archetype, prior, derived, heuristic or default). `worldgen template planet.yaml` writes an annotated example; complete specs are in `examples/`.

```yaml
spec_version: 1
name: Ember
seed: 7
mode: snapshot
priors:
  archetype: tidally_locked
star:
  mass_msun: [0.15, 0.3]      # a range: drawn from it
body:
  mass_mearth: 1.4            # a number: fixed
```

## Conventions

| Topic | Rule |
|---|---|
| Units | Spec files use practical units named in the field: `mass_msun` (M☉), `semi_major_axis_au` (AU), `mass_mearth` (M⊕), `surface_pressure_bar` (bar), `rotation_period_h` (hours), `_deg` (degrees), `_k` (kelvin), `_gyr` (Gyr). Saved states use SI units. |
| Numbers and ranges | A field typed "number or [low, high]" takes a fixed value, a range drawn from, or nothing. The low value must not exceed the high value. |
| Unknown keys | Rejected in every section. |
| Version | `spec_version` is 1. A file without it is read as version 1; any other value is rejected. |
| Draws | The same spec and seed give the same planet, and fixing one value leaves the draws of the other values unchanged. |
| Culture file | The `culture` section can also be a file of its own, with the section under a top-level `culture:` key or as the section itself, and applied to a saved world with `worldgen culture`. |
| Python | `load_spec` and `save_spec` read and write spec files; `PlanetSpec` validates the same fields in code. |

Sections are listed below in the order they appear in the file. `unset` in the Default column means the value is drawn or derived.

## Sections

### Top level

Complete specification of a planet to generate.

| Field | Type | Default | Description |
|---|---|---|---|
| `spec_version` | integer | `1` | Version of the spec format; this release reads 1 |
| `name` | text | `Unnamed` | Name of the planet |
| `seed` | integer | `0` | Random seed; the same spec and seed give the same planet |
| `mode` | `snapshot` \| `history` | `snapshot` | `snapshot` builds the present state from the inputs; `history` integrates from formation |
| `target_epoch_gyr` | number | unset | Epoch to build (Gyr); the star's age when unset |
| `star` | section | (section defaults) | Host star |
| `orbit` | section | (section defaults) | Orbit and spin |
| `body` | section | (section defaults) | Bulk properties |
| `interior` | section | (section defaults) | Interior overrides |
| `atmosphere` | section | (section defaults) | Atmosphere inputs and overrides |
| `surface` | section | (section defaults) | Surface targets |
| `biosphere` | section | (section defaults) | Life |
| `history` | section | (section defaults) | History-mode initial conditions and rates |
| `priors` | section | (section defaults) | How unset values are drawn |
| `culture` | section | unset | Inhabitants placed after the planet is built |
| `system` | section | (section defaults) | Moons and rings |
| `parent` | section | unset | Present when the body is a moon |

### `star`

Host star parameters.

| Field | Type | Default | Description |
|---|---|---|---|
| `mass_msun` | number or [low, high] | unset | Stellar mass (M☉) |
| `age_gyr` | number or [low, high] | unset | Stellar age (Gyr) |
| `metallicity_feh` | number or [low, high] | unset | Stellar [Fe/H] (dex) |
| `activity_percentile` | number or [low, high] | unset | Rotation percentile at birth (0 slow, 1 fast); sets the XUV history |

### `orbit`

Orbit and spin. Give either semi-major axis or instellation, not both.

| Field | Type | Default | Description |
|---|---|---|---|
| `semi_major_axis_au` | number or [low, high] | unset | Semi-major axis (AU); not with `instellation_earth` |
| `instellation_earth` | number or [low, high] | unset | Stellar flux at the orbit relative to Earth's; not with `semi_major_axis_au` |
| `eccentricity` | number or [low, high] | unset | Orbital eccentricity |
| `obliquity_deg` | number or [low, high] | unset | Axial tilt (degrees) |
| `rotation_period_h` | number or [low, high] | unset | Rotation period (hours) |
| `periapsis_longitude_deg` | number or [low, high] | unset | Season at periapsis, as solar longitude (Earth ~283) |

### `body`

Bulk properties of the planet. Any two of mass, radius and composition fix the third.

| Field | Type | Default | Description |
|---|---|---|---|
| `mass_mearth` | number or [low, high] | unset | Mass (M⊕) |
| `radius_rearth` | number or [low, high] | unset | Radius (R⊕) |
| `core_mass_fraction` | number or [low, high] | unset | Iron core share of the mass |
| `water_mass_fraction` | number or [low, high] | unset | All water, including what the mantle holds (Earth ~0.0006) |
| `envelope_mass_fraction` | number or [low, high] | unset | Share of the mass in a hydrogen–helium envelope |
| `radiogenic_abundance` | number or [low, high] | unset | Heat-producing elements relative to Earth |
| `tidal_heating_w` | number or [low, high] | unset | Tidal heating of the interior (W) |
| `tidal_q` | number or [low, high] | unset | Tidal dissipation quality factor |
| `atmospheric_metallicity` | number or [low, high] | unset | Giants: C/H of the envelope relative to protosolar |

### `interior`

Overrides for interior outcomes that are otherwise drawn from heuristics.

| Field | Type | Default | Description |
|---|---|---|---|
| `tectonic_regime` | `mobile_lid` \| `stagnant_lid` \| `episodic` \| `heat_pipe` \| `inactive` \| `fluid` | unset | Sets the tectonic regime instead of drawing it |
| `magnetic_field` | boolean | unset | Sets whether a magnetic dynamo is active instead of drawing it |

### `atmosphere`

Atmosphere parameters and overrides.

| Field | Type | Default | Description |
|---|---|---|---|
| `present` | boolean | unset | Whether the planet keeps an atmosphere |
| `surface_pressure_bar` | number or [low, high] | unset | Surface pressure (bar) |
| `composition` | `n2_o2` \| `n2_co2` \| `co2` \| `n2_ch4` \| `h2_he` \| `none` | unset | Main composition class |
| `bond_albedo` | number or [low, high] | unset | Bond albedo |
| `surface_temperature_k` | number or [low, high] | unset | Fixes the mean surface temperature (K) instead of computing it |
| `volatile_richness` | number or [low, high] | unset | Volatile inventory relative to Earth |
| `weathering_target_k` | number or [low, high] | unset | Temperature the carbonate-silicate cycle settles at |

### `surface`

Surface targets used when building maps.

| Field | Type | Default | Description |
|---|---|---|---|
| `land_fraction` | number or [low, high] | unset | Target land fraction; sets the surface water when given |
| `tectonics` | `simulated` \| `heuristic` | unset | Plate simulation, or the fast heuristic layout |
| `tectonics_start` | `supercontinent` \| `cratons` | unset | Initial continents of the simulation |
| `tectonics_duration_myr` | number or [low, high] | unset | Simulated time (Myr), limited by the planet's age |

### `biosphere`

Life on the planet. Unset values follow rules from the planet's water, temperature and age.

| Field | Type | Default | Description |
|---|---|---|---|
| `life` | `surface` \| `ocean` \| `subsurface` \| `none` | unset | Surface (land and sea), ocean only, subsurface, or none |
| `biochemistry` | `oxygenic` \| `methanogenic` \| `alien` | unset | Oxygenic photosynthesis (default), methanogenic, or alien |
| `alien_product` | `o2` \| `ch4` \| `none` | unset | Gas an alien biosphere releases (default none) |
| `age_gyr` | number or [low, high] | unset | Time since life began (default: planet age − 0.7 Gyr) |
| `optimum_temperature_k` | number or [low, high] | unset | Growth optimum (Earth-like 298 K) |
| `temperature_tolerance_k` | number or [low, high] | unset | Growth stops this far above the optimum (Earth-like 25 K) |
| `pigment_absorption_nm` | number or [low, high] | unset | Peak absorption of the main pigment (default from the star) |
| `oxygen_fraction` | number or [low, high] | unset | O2 mole fraction of the air (default from biosphere age) |

### `history`

Initial conditions and poorly constrained rates for history mode. Unset values are drawn.

| Field | Type | Default | Description |
|---|---|---|---|
| `epochs_gyr` | list of number | unset | Extra epochs at which full states are stored |
| `initial_water_mass_fraction` | number or [low, high] | unset | All water at formation (default: body.water_mass_fraction) |
| `carbon_inventory` | number or [low, high] | unset | Total carbon relative to Earth's, per unit planet mass |
| `nitrogen_inventory` | number or [low, high] | unset | Atmospheric nitrogen relative to Earth's, per unit planet mass |
| `initial_mantle_temperature_k` | number or [low, high] | unset | Mantle potential temperature at formation |
| `outgassing_efficiency` | number or [low, high] | unset | Volcanic degassing relative to Earth's at the same melt rate |
| `weathering_efficiency` | number or [low, high] | unset | Silicate weathering relative to Earth's at the same climate |
| `weathering_temperature_scale_k` | number or [low, high] | unset | E-folding temperature of weathering kinetics |
| `weathering_co2_exponent` | number or [low, high] | unset | Weathering ∝ pCO₂^exponent |
| `biotic_weathering_factor` | number or [low, high] | unset | Speed-up of land weathering by a full land biosphere |
| `escape_efficiency` | number or [low, high] | unset | Heating efficiency of XUV-driven escape |
| `mantle_activation_energy_kj` | number or [low, high] | unset | Activation energy of mantle viscosity |
| `core_adiabatic_heat_flow` | number or [low, high] | unset | Heat the core conducts along its adiabat, relative to Earth's |
| `life_origin_delay_gyr` | number or [low, high] | unset | Time with habitable conditions before life appears |
| `oxygen_burial_efficiency` | number or [low, high] | unset | Organic burial (O₂ source) relative to Earth's |
| `reductant_decay_gyr` | number or [low, high] | unset | E-folding time of the volcanic reductant flux (mantle oxidation) |
| `land_colonisation_delay_gyr` | number or [low, high] | unset | Time from the origin of surface life to life on land |

### `priors`

How unset parameters are drawn.

| Field | Type | Default | Description |
|---|---|---|---|
| `archetype` | text | unset | Archetype to draw from (see `worldgen archetypes`) |
| `occupiability_weights` | mapping | unset | Weights of the occupiability score terms (`temperature`, `gravity`, `pressure`, `water`, `radiation`), used to rank candidate planets |
| `enforce_constraints` | boolean | `true` | Redraw until the archetype's required outcomes hold |
| `max_attempts` | integer | `200` | Redraws allowed while enforcing constraints; at least 1 |

### `system`

A planet's moons and rings. Moons left unset are drawn (an empty list means none); rings follow the regime or a list.

| Field | Type | Default | Description |
|---|---|---|---|
| `moons` | list of moons | unset | Listed moons; unset draws them, an empty list means none |
| `rings` | `auto` \| `none` \| `broad` \| `narrow` \| `dusty warm` \| `dusty cold` or a list of rings | `auto` | `auto` draws rings by the temperature the ring material would reach, `none` gives none, the other regimes force one, a list gives the rings |

### `system.moons[]`

A moon listed by the user in a planet's system.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | text | unset | Name of the moon |
| `mass_mearth` | number | required | Mass (M⊕) |
| `radius_rearth` | number | unset | Radius (R⊕); derived from mass and composition when unset |
| `water_mass_fraction` | number | `0.0` | Water share of the mass |
| `semi_major_axis_rp` | number | required | Semi-major axis (planet radii) |
| `eccentricity` | number | `0.0` | Orbital eccentricity |
| `inclination_deg` | number | `0.0` | Inclination to the planet's equator (degrees) |
| `retrograde` | boolean | `false` | Whether the orbit is retrograde |

### `system.rings[]`

A ring listed by the user in a planet's system.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | text | unset | Name of the ring |
| `inner_radius_rp` | number | required | Inner edge (planet radii); greater than 1 and less than the outer edge |
| `outer_radius_rp` | number | required | Outer edge (planet radii) |
| `optical_depth` | number | `0.5` | Normal optical depth |
| `bond_albedo` | number | `0.5` | Bond albedo of the ring material |
| `kind` | `icy` \| `polluted` \| `dark` \| `dusty` | `icy` | Ring material |

### `parent`

The planet a moon orbits, and the moon's orbit about it. Present only in a moon's spec.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | text | unset | Name of the parent planet |
| `mass_mearth` | number or [low, high] | unset | The parent's mass (M⊕) |
| `radius_rearth` | number or [low, high] | unset | Default: Chen & Kipping mean radius for the mass |
| `rotation_period_h` | number or [low, high] | unset | The parent's rotation period (hours) |
| `obliquity_deg` | number or [low, high] | unset | The parent's; the moon orbits in its equatorial plane |
| `bond_albedo` | number or [low, high] | unset | The parent's Bond albedo |
| `tidal_q` | number or [low, high] | unset | The parent's tidal quality factor |
| `love_number_k2` | number or [low, high] | unset | The parent's tidal Love number k₂ |
| `semi_major_axis_rp` | number or [low, high] | unset | The moon's orbit, in parent radii |
| `eccentricity` | number or [low, high] | unset | The moon's orbital eccentricity about the parent |
| `inclination_deg` | number or [low, high] | unset | To the parent's equator |
| `effective_temperature_k` | number or [low, high] | unset | The parent's glow; default: its equilibrium temperature |
| `magnetic_moment_am2` | number or [low, high] | unset | The parent's dipole; default: by mass from the Solar System planets |
| `moon_origin` | `regular` \| `captured` \| `impact` | unset | How the moon came to orbit it |

### `culture`

The cultures of a planet. ``seed`` redraws the culture alone; by default the planet seed is used.

| Field | Type | Default | Description |
|---|---|---|---|
| `seed` | integer | unset | Redraws the culture alone; the planet seed is used when unset |
| `history` | section | unset | Span and epochs of an organic history |
| `cultures` | list of cultures | required | The cultures; at least one |

### `culture.history`

The span of an organic history and the epochs it keeps (years before the present).

| Field | Type | Default | Description |
|---|---|---|---|
| `start_years_ago` | number | `12000.0` | Start of the history (years before the present) (> 0) |
| `epochs` | list of number | `[]` | Epochs to keep besides the present (years before the present, inside the span) |
| `step_years` | number | unset | Time step (years); default `HISTORY_STEP_YEARS` (> 0) |
| `late_step_years` | number | unset | Shorter step used once an organic culture reaches fossil energy (years); default `HISTORY_LATE_STEP_YEARS` (> 0) |
| `pace` | number | `1.0` | Multiplies the rate at which knowledge grows (> 0) |
| `planet_events` | boolean | `true` | Eruptions, droughts, pandemics and impacts drawn at their rates |
| `events` | list of events | `[]` | Events placed by hand |

### `culture.history.events[]`

An event placed in an organic history by hand.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `eruption` \| `drought` \| `pandemic` \| `impact` \| `sea_level` | required | Kind of event |
| `years_ago` | number | required | Date (years before the present) (≥ 0) |
| `lat_deg` | number | unset | Latitude (degrees); give both coordinates or neither; required for a drought (≥ -90, ≤ 90) |
| `lon_deg` | number | unset | Longitude (degrees) (≥ -180, ≤ 360) |
| `size` | number | unset | Eruption: stratospheric sulfur (Tg S). Drought: share of rain lost (0–1). Pandemic: deaths per year as a share of the world's people (‰). Impact: energy (Mt TNT). Sea level: rise (m), required. Drawn when unset (> 0) |
| `radius_km` | number | unset | Radius of a drought (km) (> 0) |
| `duration_years` | number | unset | Duration of a drought or pandemic (years) (> 0) |

### `culture.cultures[]`

One culture: its inhabitants, development, population and organisation.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | text | unset | Name of the culture; drawn when unset |
| `mode` | `scoped` \| `organic` | `scoped` | `scoped` places the culture to match the spec; `organic` grows it from a simulated history |
| `inhabitants` | `native` \| `colonist` | `native` | `native` to the planet, or `colonist` from elsewhere |
| `development` | section | (section defaults) | Development levels |
| `population` | number or [low, high] | unset | Total population; derived from carrying capacity when unset |
| `homeland` | section | unset | Region occupied; without one the culture takes the land best suited to it |
| `settlements` | section | (section defaults) | Settlement-size distribution |
| `polities` | section | (section defaults) | Political structure |
| `colony` | section | unset | Colony details; only for colonists |

### `development`

Level on each development axis. Axes left out take the mean level of those given.

| Field | Type | Default | Description |
|---|---|---|---|
| `energy` | `foraging` \| `farming` \| `water_wind` \| `fossil` \| `electric` \| `fusion` | unset | Energy level; axes left out take the mean level of those given |
| `transport` | `foot` \| `pack_and_boat` \| `wheel_and_sail` \| `rail_and_steam` \| `motor_and_air` \| `spaceflight` | unset | Transport level |
| `communication` | `oral` \| `writing` \| `print` \| `telegraph` \| `digital` \| `interplanetary` | unset | Communication level |

### `homeland`

The region a culture occupies; without one the culture takes the land best suited to it.

| Field | Type | Default | Description |
|---|---|---|---|
| `lat_deg` | number | required | Latitude of the centre (degrees, −90 to 90) |
| `lon_deg` | number | required | Longitude of the centre (degrees, −180 to 360) (≥ -180, ≤ 360) |
| `radius_km` | number | required | Radius of the region (km); greater than 0 |

### `settlements`

Settlement-size distribution: rank-size exponent and the population of the largest settlement.

| Field | Type | Default | Description |
|---|---|---|---|
| `zipf_exponent` | number or [low, high] | unset | Rank-size exponent of settlement sizes |
| `largest_population` | number or [low, high] | unset | Population of the largest settlement |

### `polities`

Political structure: how many polities, of which type, with optional capitals and territories.

| Field | Type | Default | Description |
|---|---|---|---|
| `count` | number or [low, high] | unset | Number of polities; at least 1 and at least the number of capitals or territories given |
| `type` | `band` \| `tribe` \| `chiefdom` \| `city_state` \| `kingdom` \| `empire` \| `federation` \| `republic` \| `corporate` \| `colony` | unset | Kind of polity |
| `capitals` | list of places | `[]` | Capitals named in the spec; the remaining capitals are chosen |
| `territories` | list of polygons, each a list of [lat, lon] | `[]` | One polygon of (lat, lon) vertices per polity |

### `capitals[]` and `landing_sites[]`

A point on the planet, with an optional name.

| Field | Type | Default | Description |
|---|---|---|---|
| `lat_deg` | number | required | Latitude (degrees, −90 to 90) |
| `lon_deg` | number | required | Longitude (degrees, −180 to 360) (≥ -180, ≤ 360) |
| `name` | text | unset | Name of the place |

### `colony`

Founding of a colonist culture.

| Field | Type | Default | Description |
|---|---|---|---|
| `founded_years_ago` | number or [low, high] | unset | Landing date (years before the present) |
| `origin` | text | unset | Origin of the colonists, as text; in an organic history the name of the culture they come from |
| `landing_sites` | list of places | `[]` | Places where the colonists land |
| `immigrants_per_year` | number | `0.0` | Organic colonies: arrivals after landing (≥ 0) |
| `immigration_years` | number | unset | How long they keep arriving (None: to the present) (> 0) |
| `independence_years` | number | unset | Organic colonies: years tied to the home power (> 0) |
| `crowd_diseases` | boolean | `false` | Organic colonies from off the planet: carry crowd diseases from herd animals |

## Checks applied when a spec is read

| Check | Section |
|---|---|
| Unknown key | All |
| Range with low above high | All numeric fields |
| Both `semi_major_axis_au` and `instellation_earth` given | `orbit` |
| `max_attempts` below 1 | `priors` |
| A spec with `parent` (a moon) that lists moons of its own | Top level |
| Ring with `1 < inner_radius_rp < outer_radius_rp` not met | `system.rings[]` |
| Fixed polity count below 1, or below the number of capitals or territories given | `culture.cultures[].polities` |
| Colony details on a native culture | `culture.cultures[]` |
| Epoch or event outside the history span; an event with only one coordinate; a drought without a place; a sea-level event without a rise; a drought `size` above 1 | `culture.history` |
| Organic colony founded before the history starts | `culture` |
| Non-positive population | `culture.cultures[]` |

Two cases warn instead of failing. An organic culture ignores the development, population, settlements and polities it is given, because they emerge from the history (a colonist keeps its development as the level it arrives with). `immigrants_per_year` and `independence_years` apply to organic colonies only.
