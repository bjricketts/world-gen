# worldgen

Physically motivated planet generation for world building. Give it as much or as little as you like
(star, orbit, mass, atmosphere...) and it fills in the rest from physics and from archetype priors,
then reports where every value came from and anything that doesn't add up.

Implemented so far (see [DESIGN.md](DESIGN.md)): the global physics core in snapshot mode (milestones 1
and 1.1), global surface maps with non-plate landforms (milestone 2), a plate-tectonic simulation
for mobile-lid planets (milestone 3), water: sea level from the planet's water, rivers, lakes and river
erosion (milestone 4), and climate, ice sheets, life and biomes (milestone 5). Known gaps and
planned fixes are in [OPEN_POINTS.md](OPEN_POINTS.md).

## Install

```bash
pip install -e ".[dev,notebooks]"
```

Requires Python 3.10+. Maps use Cartopy, which installs from wheels on Windows, macOS and Linux.

## Command line

```bash
worldgen archetypes                       # list world types
worldgen template planet.yaml             # write an annotated spec to edit
worldgen generate planet.yaml             # build the planet and print its report
worldgen generate planet.yaml -o out.yaml # also save the full state (SI units)
worldgen generate planet.yaml -n 10       # draw 10 variants, show the most occupiable
worldgen random --archetype ocean --seed 3

# Maps and saved worlds
worldgen random -a temperate --seed 3 --map overview.png --save kestrel
worldgen generate planet.yaml -r high -m globe.png -f elevation -p orthographic --lon 120 --lat 10
worldgen map kestrel terrain.png -f terrain -p robinson
worldgen report kestrel

# Plate tectonics
worldgen generate planet.yaml --start cratons --duration 600 --snapshots 20 --save kestrel
worldgen drift kestrel drift.gif          # animation; a .png gives a panel figure
worldgen generate planet.yaml --tectonics heuristic   # fast layout without simulation

# History mode: the planet integrated from formation
worldgen generate planet.yaml --mode history --epochs "0.5,1,2" --save kestrel
worldgen history kestrel life.png -f life   # or -f climate (default) or -f interior
worldgen map kestrel relicts.png -f relicts # shorelines, dry valleys and scoured ground of its past
worldgen report kestrel --timeline         # every event, and the planet at each epoch
```

Map fields: `elevation`, `terrain`, `plates`, `crust_age`, `orogeny_age`, `temperature`, `rainfall`, `basins`,
`biomes`, `koppen`, `ice`, `relicts` (history mode), or `overview` and `climate` (multi-panel). Rivers are drawn on elevation, terrain, rainfall and basin maps.
Projections: `equirectangular`, `mollweide`, `robinson`, `orthographic`, `north_polar`, `south_polar`.
Resolutions: `preview` (10k cells), `standard` (40k), `high` (160k), or a cell count.

## Python

```python
from worldgen import PlanetSpec, generate, format_report, format_timeline

spec = PlanetSpec(
    name="Ember",
    seed=7,
    priors={"archetype": "tidally_locked"},
    star={"mass_msun": (0.15, 0.3)},       # a range is drawn from
    body={"mass_mearth": 1.4},             # a number is fixed
)                                          # anything omitted is drawn or derived
state, timeline = generate(spec)
print(format_report(state))
print(state.atmosphere.surface_temperature_k)

# With a surface
from worldgen import generate_world, save_world, load_world
from worldgen.render import plot_overview, plot_map

world = generate_world(spec, resolution="standard")
plot_overview(world)
plot_map(world, "elevation", "orthographic", central_longitude=60)
world.surface            # xarray Dataset: elevation, ocean, terrain, crust, plate, boundary, crust_age, orogeny_age,
                         # air_temperature, monthly_temperature, precipitation, sea_ice, ice_thickness, biome,
                         # koppen, vegetation_cover, and with liquid water: runoff, discharge, lake, river_order, ...
world.state.water        # total, mantle and surface water, ocean volume
world.state.climate      # zonal climate: mean temperature, albedo, open water, ice line, day/night side
world.state.biosphere    # life, O2 and CH4, pigment colour, expected productivity
plot_climate(world)      # seasonal temperature and rain, biomes, Köppen classes
save_world(world, "ember")

# Continental drift (mobile-lid planets)
from worldgen.render import plot_history, animate_history

world = generate_world(spec, snapshot_interval_myr=20)
plot_history(world)
animate_history(world, "drift.gif")

# History mode: everything integrated from formation
from worldgen.render import plot_climate_history, plot_interior_history, plot_life_history

state, timeline = generate(spec.model_copy(update={"mode": "history"}))
timeline.events          # origin of life, oxygenation, snowball onset, dynamo shutdown, ...
timeline.states          # full planet states at the epochs the spec asked for
timeline.series          # sampled values through the run, for figures
print(format_timeline(timeline, state))
plot_climate_history(timeline)
plot_life_history(timeline)
```

A guided tour is in `notebooks/01_quickstart.ipynb`.

## Surfaces

The global surface lives on a Fibonacci grid of nearly uniform cells. How it is built depends on the
tectonic regime:

| Regime | Surface |
|---|---|
| mobile lid | Simulated plate tectonics (below), or with `surface.tectonics: heuristic` a single plate layout with landforms placed from boundary types |
| stagnant lid | Hemispheric dichotomy, a volcanic rise with giant shield volcanoes, rift systems, craters |
| episodic | Young volcanic plains with rough highland blocks, scattered volcanoes, few craters |
| heat pipe | Smooth lava plains, many volcanic centres, isolated massifs, no craters |
| inactive | Cratered highlands and flooded low maria |

Mountain and volcano heights, and ocean basin depths, scale inversely with surface gravity.

### Water and sea level

`body.water_mass_fraction` is the planet's total water (Earth: about 0.0006). On plate-tectonic
planets the mantle holds part of it, more at higher gravity (Cowan & Abbot 2014); Earth keeps about
1.7 oceans in the mantle and 1 at the surface. Ice sheets hold part of the surface water, and the
rest fills the ocean basins; closed basins covering at least 0.5% of the planet hold enclosed seas at
the same level. The land fraction follows. Setting `surface.land_fraction` places sea level directly and changes the water to
match (within about 0.01); the report shows the implied water. Planets without surface water have no ocean, and
elevations are measured from the mean surface.

### Climate

Climate comes in two tiers that share their physics (`worldgen/climate/`):

- **Tier 0** runs in the global state: a zonal energy-balance model with seasons, land and ocean
  columns, moist heat transport and ice albedo. It sets the planetary albedo, the mean temperature,
  the ice line and how much of the ocean stays open, which decides the water phase: a 265 K world
  with an open tropical ocean has liquid water. On tidally locked planets the bands run from the
  substellar point, and the report gives day- and night-side temperatures. Where a fully frozen
  state is also stable, the partly open one is used and the report says so.
- **Tier 1** runs on the built surface: the same energy balance on a fixed 10k-cell sphere grid,
  solved directly for its seasonal cycle, with precipitation from moist energy transport split
  into Hadley and eddy parts (after Siler et al. 2018). Rain reaches land along seasonal winds,
  with monsoon flow toward warm continents and rain shadows behind mountains. Temperatures are
  scaled to each cell's height. Once the surface exists, Tier 0 is solved again with the real land
  in each band and the global state is updated, so `generate_world` can report a slightly different
  climate from `generate`. Where weathering holds the temperature (or you set it), Tier 1 adjusts
  the greenhouse to keep it and the report gives the optical depth it needed.
- **Ocean heat transport** is a poleward flux of fixed shape, largest in the subtropics and reaching
  past the sea-ice edge, scaled to the atmospheric transport and each latitude's ocean width. It
  carries about a fifth of the poleward heat at 35° N on the Earth example, as on Earth.

Month 1 begins at the northern winter solstice; `orbit.periapsis_longitude_deg` places the
closest approach in the year (Earth: 283).

### Ice sheets and life

- **Ice sheets** grow where snowfall beats degree-day melt; their thickness follows a plastic profile
  from the ice margin, and the ice surface feeds back into the climate. Mountains well above the
  snowline are worn down toward it, in the tectonic simulation and on the finished surface.
- **Life** (`biosphere` spec section) follows rules unless set: planets older than 1 Gyr with liquid
  water have surface life (ocean-only on water worlds), frozen water worlds have subsurface life.
  Oxygenic life raises O₂ with biosphere age (21% after ~4 Gyr), methanogens add CH₄ and warm the
  planet, and land biota lower the weathering temperature by a few kelvin. Vegetation colour follows
  the star's light.
- **Vegetation** comes from Miami-model productivity; it darkens land and carries moisture further
  inland. Climate, ice and vegetation are iterated on the surface until they agree (usually 3–6
  passes).
- **Biomes** (Whittaker-type) and **Köppen–Geiger classes** are stored for every cell.
- **Occupiability** includes a biosphere factor from breathable O₂ and food productivity, refined
  with the productive land once the surface is built.

### Rivers and lakes

Planets with liquid surface water get:
- Drainage: every land cell drains to the sea or to a lake. Lakes grow until evaporation balances
  inflow; full lakes overflow. Rivers are marked with discharge and stream order.
- River erosion and sediment: rivers cut valleys, sediment fills basins and builds shelves. With
  simulated tectonics this runs throughout the simulation, which keeps continents near sea level
  and old mountain ranges worn down.

The report summarises the largest river and basin, total outflow, lakes and inland drainage.
Frozen and dry planets get no rivers.

A saved world is a folder with `spec.yaml`, `state.yaml` and `surface.zarr`; a history-mode world also
carries `timeline.yaml` (its events and sampled series) and one state file per epoch under `epochs/`.
Worlds saved before milestone 6 use an older format and need to be generated again.

### Plate-tectonic simulation

Mobile-lid planets run a plate simulation in 2 Myr steps, 400 Myr by default
(`surface.tectonics_duration_myr`, limited by the star's age). Crust is carried as points that move
with their plate:

- Oceanic crust sinks beneath continents and beneath younger oceanic crust, building Andean mountain
  belts, island arcs and trenches. Continents that meet collide, raise mountains and eventually fuse.
- Plates turn toward their subduction zones, rift apart (mostly those carrying large continents), and
  old sea floor breaks away from its continent and starts to sink.
- Every 20 Myr the crust is re-mapped onto the grid; gaps become new sea floor whose age grows away from
  the ridge, and sea-floor depth follows its age.
- Hotspots at fixed mantle positions leave volcanic island chains. Mountains erode over about 100 Myr.

The start is one supercontinent or scattered cratons (`surface.tectonics_start`, drawn per seed if
unset). `orogeny_age` records when each region last saw mountain building; old orogens appear as
highlands. With `--snapshots N` (or `snapshot_interval_myr`) the saved world also holds elevation,
plates and crust type every N Myr.

Typical run times (400 Myr, including climate, rivers, ice and erosion): see DESIGN.md section 5.2;
standard resolution takes about 35 s, heuristic mode about 10 s.
The first run also compiles the Numba kernels (a few seconds, cached afterwards).

## History mode

`mode: history` integrates the planet from formation to its present age instead of evaluating
relations at one epoch. One coupled system carries the star's brightening and XUV, the mantle and
core cooling with their dynamo and melting, outgassing and atmospheric escape, the water traded
with the mantle, carbon between air, ocean, crust and mantle under a Tier 0 climate, and the
biosphere: origin of life, the rise of O₂, biotic methane and biotic weathering. Events (snowball
onset and exit, runaway, ocean loss, dynamo shutdown, regime change, origin of life, oxygenation)
restart the integration and are logged, and the CO₂ of the air comes from the weathering flux
balance rather than the drawn target.

`generate` returns a `Timeline` as well as the state: its events, full planet states at the epochs
the spec asks for (`history.epochs_gyr`, or `--epochs`), and a sampled series behind the figures.
Present-state fields in the spec become overrides applied at the end and flagged in the report.

The history also reaches the surface: plate speed follows the plate creation rate the thermal model
produces, volcanism follows the melt, the simulated span follows the sea-floor turnover, and the
planet keeps relicts of its past — a terrace at a lost sea level, valley networks cut when it still
had rain, ground the ice has left, plains buried by older volcanism — for as long as its own
erosion preserves them (~100 Myr on Earth, billions of years on a dry, quiet world).

Earth, Mars and Venus have example specs (`examples/*_history.yaml`); `scripts/validate_history.py`
checks them against the real planets and `scripts/history_sensitivity.py` ranks which poorly
constrained parameters move the outcome.

## Spec files

YAML, in user-friendly units (solar masses, AU, Earth masses, bar, hours, degrees). Each numeric
field may be a number, a `[low, high]` range, or omitted. See `examples/` and `worldgen template`.

Resolution order for each value:

1. **user**: fixed in the spec
2. **range**: drawn from a user range
3. **archetype**: drawn from the chosen archetype's prior (or, if none is named, from an archetype
   compatible with the fixed values, weighted toward habitable and occupiable worlds). Some archetypes
   give instellation as a position across the star's habitable zone rather than an absolute value.
4. **prior**: drawn from broad base priors
5. **derived / heuristic / default**: computed by the physics modules

## Archetypes and constraints

Each archetype declares outcomes its planets must have, e.g. `temperate` requires free rotation, liquid
surface water with at least half the ocean free of ice, and no runaway greenhouse; `ice` requires a frozen
surface; `tidally_locked` requires synchronous rotation and a day side at 250–330 K. If a draw misses them, unset and range-drawn values are redrawn with derived seeds
(up to `priors.max_attempts`, default 200). Fixed values never change. The report shows the accepted
draw and its seed, or a warning naming the constraints that could not be met. Set
`priors.enforce_constraints: false` to accept the first draw.

`atmosphere.weathering_target_k` (the temperature CO₂ regulation settles at, drawn from 260–320 K) and
`body.tidal_q` (tidal dissipation, drawn log-uniformly from 10–500 for rocky planets) can be fixed in
the spec like any other value.

## Package layout

| Module | Role |
|---|---|
| `spec.py` | Spec schema, YAML load/save, template |
| `priors/` | Archetypes, distributions, sampling, outcome constraints, occupiability score |
| `star.py` | Luminosity, radius, temperature, lifetime, XUV activity |
| `orbit.py` | Period, instellation, habitable zone, tidal locking, spin |
| `bulk.py` | Mass–radius–composition, planet class, gravity |
| `interior.py` | Radiogenic heat, tectonic regime, magnetic dynamo |
| `atmosphere.py` | Retention, pressure, greenhouse, global climate (Tier 0), water phase |
| `grid/` | Fibonacci sphere grid, triangulation, neighbours |
| `noise.py` | Seamless 3D fractal noise on the sphere |
| `water.py` | Water inventory: mantle and surface water, land fraction estimates |
| `surface/` | Heuristic plates, landforms, non-plate regimes, sea level, surface dataset, tectonic drive and relict features from the history |
| `climate/` | Insolation, Tier 0 and Tier 1 energy balance, moisture and precipitation, ice sheets and snowline erosion |
| `biosphere/` | Life and its gases, pigment colour, vegetation, biomes and Köppen classes, climate–ice–vegetation coupling |
| `hydrology/` | Water balance, drainage, lakes, rivers, river erosion and sediment |
| `tectonics/` | Plate simulation: crust points, Numba kernels, processes, re-mapping, conversion to a surface |
| `render/` | Cartopy map projections, colour schemes, hillshading, smoothed rivers, drift figures and animations, history figures |
| `world.py` | World container, save and load |
| `history/` | The history ODE system: per-planet constants, thermal evolution, volatiles, Tier 0 climate table, biosphere, integration, states at epochs |
| `evolve/` | Snapshot evolver and history evolver behind one interface |
| `heuristics.py` | Every tunable calibration constant, in one place |
| `report.py` | Text report and state export |
| `cli.py` | Command-line interface |
| `constants.py`, `units.py` | SI constants and unit conversions |

Internally all values are SI floats; names carry units where needed (`mass_kg`, `period_s`).

## Heuristics

Several quantities have no standard formula: tectonic regime, dynamo, greenhouse strength versus
pressure, atmospheric retention probability, heat transport on other planets, the presence of life,
and occupiability. These use simple rules calibrated
against Solar System bodies, with all constants in `heuristics.py`. The report lists every heuristic
used for a planet.

## Calibration scripts

Run from the `scripts/` folder:

| Script | Purpose |
|---|---|
| `validate_earth.py` | Earth example against Earth over several seeds (hypsometry, ocean, plates, rivers, lakes, climate, ice, Köppen shares); `--set NAME=VALUE` overrides a heuristic for a sweep |
| `archetype_fidelity.py` | Archetype outcomes over many draws: temperatures, water phase, open ocean, estimated land, life, redraws |
| `validate_history.py` | History-mode Earth, Mars and Venus against the real planets, and Earth against its snapshot build; exits non-zero if a check misses |
| `history_sensitivity.py` | Ranks the poorly constrained history parameters by how far they move the outcome, one at a time over each prior |
| `hypsometry_table.py` | Reference land fraction against ocean volume, for `worldgen/water.py` |
| `render_examples.py` | The example figures in "Claude outputs" |
| `planet_gallery.py` | Several planets with their reports, saved worlds and every applicable map and figure, one folder each |

## Tests

```bash
pytest                   # full suite, ~4-5 min on two cores
pytest -n auto           # in parallel (needs pytest-xdist, included in the dev extras)
pytest -m "not slow"     # skip tests that generate worlds, ~27 s
```

The suite checks each relation against published values, runs Earth, Venus and Mars through the full
pipeline, checks that every archetype meets its constraints, and covers grids, terrain for every regime,
the tectonic simulation's rules and outputs, the water inventory, drainage, lakes and erosion,
the climate tiers (insolation, energy balance, moisture, snowball bistability, day–night contrast), ice
sheets, life, vegetation, biomes and Köppen classes, saving and loading, and rendering in every projection.
