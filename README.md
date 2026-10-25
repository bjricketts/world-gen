# worldgen

Physically motivated planet generation for world building. Give it as much or as little as you like
(star, orbit, mass, atmosphere...) and it fills in the rest from physics and from archetype priors,
then reports where every value came from and anything that doesn't add up.

Implemented so far (see [DESIGN.md](DESIGN.md)): the global physics core in snapshot mode (milestones 1
and 1.1), global surface maps with non-plate landforms (milestone 2), a plate-tectonic simulation
for mobile-lid planets (milestone 3), water: sea level from the planet's water, rivers, lakes and river
erosion (milestone 4), climate, ice sheets, life and biomes (milestone 5), history mode (milestone 6),
and zoomed local maps of any region (milestone 7), and culture placed from a user spec (milestone 8):
every cell has a tectonic setting and a potential for fifteen resource classes, and each culture gets a
population density map, settlements, roads, polities and names, carried down to villages, local roads
and refined borders on zoomed maps. Milestone 9 (organic cultures that emerge from a simulated history)
is under way: the spec, the history's time steps, the kept epochs (task 9.1), the land's suitability (task 9.2) the growth and spread of people (task 9.3), farming and development (task 9.4), states that form, expand and break apart (task 9.5), rulers, wars between states and alliances (task 9.6), settlements, roads and trade through time (task 9.7), and colonists among natives (task 9.8) are in place.
Milestone 10 (export and a 3D globe) is done: any map field can be written as a global GeoTIFF, 16-bit PNG
or RAW raster, as values or as its map colours (task 10.1), and zoomed regions can be exported with their culture
as rasters and GeoJSON, or as Unity and Unreal heightmaps (task 10.2); a tile pyramid carries the heights from
the globe down to local maps with matching seams (task 10.3); the globe, pyramid tiles and regions export as
GLB/glTF, PLY or OBJ meshes with per-vertex colours from any map field (task 10.4); a 3D globe shows any map
field at any kept epoch with zoomed regions draped over it (task 10.5); `worldgen export` and `worldgen globe`
run all of it from the command line, and `scripts/validate_export.py` checks it (task 10.6).
Milestone 11 (giants and moons) is done: giants have a cloud class, albedo, colours, jets and cloud-top maps,
rings and moon systems are drawn from their formation, moons are built as worlds with tides, eclipses,
planetshine and subsurface oceans, a `habitable_moon` archetype draws habitable moons with their parents,
`scripts/validate_solar_system.py` checks the Solar System's giants and moons, and `worldgen system` and
`worldgen moon` draw a system as a diagram or a 3D scene and build its moons.
Known gaps and planned fixes are in
[OPEN_POINTS.md](OPEN_POINTS.md).

## Install

```bash
pip install -e ".[dev,notebooks]"
```

The history explorer notebook also needs `pip install -e ".[explore]"` (ipywidgets and ipympl). GeoTIFF export
and mesh export need `pip install -e ".[export]"` (rasterio, trimesh); PNG and RAW export work without it. The 3D
globe and the 3D system scene need `pip install -e ".[globe]"` (PyVista with its Jupyter backend, ipywidgets,
and imageio with ffmpeg for movies).

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

# Zoom: the local map of a region
worldgen zoom kestrel local.png --lat=48.5 --lon=12        # ~0.75 km lattice over ~150 km or more
worldgen zoom kestrel close.png --lat=48.5 --lon=12 -r 150 --half-width 10 -f biome --cache
worldgen zoom kestrel_settled town.png --lat=48.5 --lon=12  # with villages, roads and borders (--no-culture: without)

# Culture: people, settlements, roads, polities and names on a saved world
worldgen culture kestrel culture.yaml --save kestrel_settled --map culture.png   # 2x2 culture overview
worldgen map kestrel_settled political.png -f political -p orthographic --lon 20 --lat 40
worldgen map kestrel_settled copper.png -f resources:copper
worldgen report kestrel_settled            # planet report plus the culture section

# Organic history: a culture section with mode: organic and history.epochs
worldgen culture kestrel organic.yaml --save kestrel_history
worldgen chronicle kestrel_history --since 3000 --until 1000   # major entries, minor ones counted per period
worldgen chronicle kestrel_history --kind independence --culture Corin
worldgen epochs kestrel_history epochs.png -f political         # a field at the kept epochs; .gif animates
worldgen epochs kestrel_history timeline.png --timeline         # people, levels and events against time
worldgen map kestrel_history old.png -f political --epoch 2000  # --epoch also works for report and zoom
worldgen map kestrel_history events.png -f events --epoch 2000  # planet events since the previous kept epoch
worldgen map kestrel_history wars.png -f political --events      # the same events over any field

# Export (pip install -e ".[export]" for GeoTIFF and meshes)
worldgen export fields kestrel                                   # what the world can export, with units
worldgen export raster kestrel elevation.tif                     # format from the suffix: .tif, .png, .r16/.raw
worldgen export raster kestrel texture.png -f biomes --colour --hillshade
worldgen export raster kestrel_history states.tif -f political --epoch 2000
worldgen export region kestrel_settled alps/ --lat=46.5 --lon=9 --crs laea   # fields, colours, culture, GeoJSON
worldgen export engine kestrel terrain.r16 --lat=46.5 --lon=9 --engine unity --size 1025
worldgen export pyramid kestrel tiles/ --place 46.5,9,9 --meshes  # levels 0-3 (~6.5 min) and a chain to level 9
worldgen export mesh kestrel globe.glb --level 2 -f biomes --exaggeration 20
worldgen export mesh kestrel alps.glb --lat=46.5 --lon=9          # a region in local metres

# The 3D globe (pip install -e ".[globe]")
worldgen globe kestrel_history                                    # a window: keys n/p for fields, an epoch slider
worldgen globe kestrel_history globe.html -f political -e 2000    # an interactive page; .png for an image
worldgen globe kestrel_history views/ -f elevation,political --all-epochs --region 46.5,9

# Giants and moons
worldgen generate jupiter.yaml --save jupiter                    # a giant: cloud class, jets, rings, moons
worldgen map jupiter clouds.png -f clouds -p robinson             # also temperature, winds, bands, vortices, relief
worldgen moon jupiter                                             # list the moons
worldgen moon jupiter europa -r preview --map europa.png          # build Europa as a world, kept in jupiter/moons/
worldgen system jupiter system.png --view edge                    # moons, rings and limits; .svg or .pdf too
worldgen system jupiter scene.png --3d --time 2                   # the 3D scene with shadows; .html, .mp4 or .gif
worldgen system jupiter/moons/01_europa europa.png                # a moon's folder draws its parent, moon marked
```

Map fields: `elevation`, `terrain`, `plates`, `crust_age`, `orogeny_age`, `temperature`, `rainfall`, `basins`,
`biomes`, `koppen`, `ice`, `relicts` (history mode), the culture layers `political`, `population`, `cultures`,
`polities`, `settlements`, `roads`, `habitat`, `resources:<name>` (or `resources:best`) and `suitability[:<culture>]`, an organic history's
`energy`, `transport` and `communication[:<culture>]`, `bands`, `livestock`, `land:<component>` (fresh_water, arable, access,
resources, climate, defensibility) and `events`, or `overview`, `climate` and `culture`
(multi-panel). Rivers are drawn on elevation, terrain, rainfall and basin maps and under the political,
settlement and road layers.
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

# Zoom: the local map of the region around a place
from worldgen.zoom import zoom_region
from worldgen.render import plot_local_map

region = zoom_region(world, 48.5, 12.0)                     # or resolution_m=150, half_width_km=10
plot_local_map(world, region)                               # mode="biome" for biome colours
region.elevation, region.river_order, region.ice_thickness_m, region.lake   # fields on the region lattice

# Culture on a zoomed region of a world that has one
from worldgen.culture.local import zoom_culture

local = zoom_culture(world, region)                         # villages, footprints, local roads, borders
plot_local_map(world, region, culture=local)

# Export: global rasters of any map field (format from the suffix: .tif, .png, .r16/.raw)
from worldgen.export import export_global, global_raster

export_global(world, "elevation.tif", "elevation", width=4096)       # float metres, planet-sphere CRS
export_global(world, "heightmap.r16", "elevation")                   # 16-bit codes; scale in heightmap.r16.json
export_global(world, "biomes.png", "biomes")                         # category codes; labels in the sidecar
export_global(world, "texture.png", "elevation", colour=True, hillshade=True)   # the map's colours
export_global(world, "states.tif", "political", epoch=2000)           # a culture layer at a kept epoch
raster = global_raster(world, "rainfall:7")                          # values in memory (month 7)

# Export: zoomed regions on their native lattice (GCP-referenced; warp with a thin-plate spline) or north-up LAEA
from worldgen.export import export_region, export_region_bundle, engine_region, export_engine_heightmap

export_region(world, region, "region.tif")                           # heights in metres
export_region(world, region, "region_laea.tif", crs="laea")          # resampled north up
export_region_bundle(world, region, "region/", culture=local)        # fields, colours, culture rasters, GeoJSON
terrain = engine_region(world, 48.5, 12.0, "unity", 513)             # built on a 129-node lattice: 513² exactly
export_engine_heightmap(world, terrain, "terrain.r16", "unity")      # import settings in terrain.r16.json

# Export: a tile pyramid (levels 0-3 for the globe, ~6.5 min; finer levels around chosen places)
from worldgen.export import build_pyramid, export_pyramid

pyramid = build_pyramid(world, places=[(48.5, 12.0, 9)])             # a chain down to level 9 (~0.14 km)
export_pyramid(pyramid, world, "tiles/")                             # L<level>/<face>/<x>_<y>_elevation.png + tileset.json

# Export: meshes (RTIN with locked borders; terrain plus a sea-surface layer; GLB nodes in planet metres)
from worldgen.export import export_globe_mesh, export_region_mesh, export_pyramid_meshes

export_globe_mesh(world, "globe.glb", level=2, field="biomes", exaggeration=20)   # from the global grid
export_globe_mesh(world, "globe3.glb", level=3, pyramid=pyramid)                 # from pyramid tiles
export_region_mesh(world, terrain, "terrain.obj", max_error_m=20)                # local east-north-up metres

# The 3D globe (pip install -e ".[globe]")
from worldgen.render import GlobeViewer

globe = GlobeViewer(world, level=2)              # heights x20; pyramid=... for pyramid tiles
globe.add_region(region)                         # draped over the globe
globe.show("political", 2000)                    # window: keys n/p step fields, slider picks the kept epoch
globe.widget()                                   # in Jupyter (trame backend)
globe.screenshot("globe.png", "biomes", lat_deg=45, lon_deg=10)
globe.save_html("globe.html", "political", 2000)            # one interactive page
globe.export_views("views/", ("political", "energy"))   # an HTML page per field and kept epoch
```

A guided tour is in `notebooks/01_quickstart.ipynb`; `notebooks/02_organic_history.ipynb` runs an organic history
and reads its chronicle, series and epochs; `notebooks/03_history_explorer.ipynb` browses a history with a widget:
field and epoch controls, planet events, the values of the cell under the cursor and a timeline
(`HistoryExplorer(world).widget()`); `notebooks/04_export_and_globe.ipynb` exports global rasters, zoomed regions, engine heightmaps, a tile pyramid and meshes, and shows the 3D globe (milestone 10); `notebooks/05_giants_and_moons.ipynb` builds Jupiter with the Galilean moons, Saturn with drawn rings and moons, Europa and a habitable moon as worlds, and draws their systems (milestone 11).

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

- **Ice sheets** grow where snowfall beats degree-day melt (the year's snow melts first, at about half
  the rate of ice); their thickness follows a plastic profile
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
Zoomed regions kept with `--cache` (or `cache_dir`) live under `regions/`. A world with inhabitants keeps
its civilisation under `culture/`: `culture.yaml` (spec, cultures, polities, chronicle) and `culture.zarr`
(per-cell layers, settlement and road tables), with the earlier epochs of an organic history saved the
same way under `culture/epochs/<years ago>/` (culture format 2; format 1 still loads). `state.yaml` holds the
planet's system summary (moons, rings) or, for a moon, its parent, and moon worlds built from a planet are kept
under `moons/`. A giant's `surface.zarr` holds its cloud tops, with the cloud model in its attributes. This is
format 8. Worlds saved before task 11.3 (format 7) load as they were; those saved by milestone 10 (format 6) load without a system summary; those saved by milestone 7 (format 5) load
without a civilisation, and those saved by milestone 6 (format 4) also load, with the structural fabric the
zoom needs computed from their surface. Older worlds need to be generated again.

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
- On planets with rivers, continents subside where they were stretched at rifting (McKenzie 1978) and
  beside mountain belts (foreland flexure, DeCelles & Giles 1996), and river sediment fills the space.
  On planets with sea life, the sea adds its own sediment: carbonate platforms in warm shallow water,
  carbonate on shelves and slopes, and pelagic rain on the deep sea floor.

The start is one supercontinent or scattered cratons (`surface.tectonics_start`, drawn per seed if
unset). `orogeny_age` records when each region last saw mountain building and `orogeny_kind` what kind
(Andean, arc or collision); old orogens appear as highlands. `sediment` holds the sediment thickness. With `--snapshots N` (or `snapshot_interval_myr`) the saved world also holds elevation,
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

## Zoom and local maps

`zoom_region(world, lat, lon)` (or `worldgen zoom`) builds a local map of the region around a place:

- **Tiles.** The planet is covered by a cube-sphere quadtree. A region is a block of tiles on one cube
  face with a regular lattice (128 nodes per tile edge); the level is chosen from the requested spacing
  (`resolution_m`, 750 m by default), giving a spacing between half the request and the request, and the
  block covers `half_width_km` (75 km by default) around the place. Each tile has its own seed from the planet's seed, so a region is the same whenever it is built,
  and neighbouring regions agree exactly where they meet.
- **Detail.** The global surface is interpolated on the planet grid's triangles, and the relief finer
  than the global grid is added: fractal noise that continues the planet's relief spectrum, stretched
  along the structural grain (fold belts, sea-floor fabric) and scaled by how rugged the ground is.
  `ruggedness` is `earth` (default), `dramatic` or `gentle`. The climate is downscaled to the detail
  (lapse rate, local rain shadow).
- **Erosion.** Rivers erode the region for 15 Myr with its edge pinned to the inherited surface; rivers
  from the global network enter across the edge at their real size. Glaciers then work the ground at the
  snowline of the coldest epoch the planet remembers (the present one without a history record), cutting
  U-shaped valleys and basins that hold lakes once the ice has gone. The present ice (inherited ice sheets
  and local glaciers) is laid on last.
- **Streams.** A channel starts where its catchment reaches a slope-dependent size (steep ground
  needs less) and is perennial where its mean flow reaches 0.1 m³/s; drier channels are dry washes.
- **Map.** `plot_local_map` draws relief or biome tints on the world's colour scale, hillshading, the
  coast, lakes, ice, perennial streams (width by discharge) and dashed dry washes. How much of the
  network is drawn follows the map scale, so zooming in shows smaller streams.

A default region (~146k nodes) takes 10–15 s to build; with `--cache` it is stored in the world
folder and loads in about a second next time. The cache is keyed to the world and the request; delete
`regions/` after changing heuristics or upgrading.

## Spec files

YAML, in user-friendly units (solar masses, AU, Earth masses, bar, hours, degrees). Each numeric
field may be a number, a `[low, high]` range, or omitted. See `examples/` and `worldgen template`.

Resolution order for each value:

1. **user**: fixed in the spec
2. **range**: drawn from a user range
3. **archetype**: drawn from the chosen archetype's prior (or, if none is named, from an archetype
   compatible with the fixed values, weighted toward habitable and occupiable worlds). Some archetypes
   give instellation as a position across the star's habitable zone rather than an absolute value.
   `habitable_moon` makes a temperate moon of a giant in the habitable zone. The moon formed in the
   giant's disc (3 × 10⁻⁵–10⁻⁴ of a 3–13 M_J planet's mass, 8–30 planet radii out) or was captured
   (0.1–1 M⊕ around a 0.3–3 M_J planet, 6–60 planet radii out, half of them retrograde);
   `parent.moon_origin` picks one (`impact` is refused: impacts on giants make moons too small). The archetype is
   used for any spec with a `parent` section and no archetype, and for about 3% of specs that fix nothing
   about the body.
4. **prior**: drawn from broad base priors
5. **derived / heuristic / default**: computed by the physics modules

### Giant planets

Ice giants, gas giants and brown dwarfs get an appearance, reported under "Appearance": the cloud
class of Sudarsky et al. (2000, I ammonia to V silicate clouds) from the effective temperature, which
adds internal heat from the planet's mass and age to the starlight; the Bond albedo that class gives
(used for the equilibrium temperature unless the spec sets `atmosphere.bond_albedo`); the colour of
the disc, its zones and belts under the star's light; and the jets: their number from the Rhines
scale and the direction of the equatorial jet. Tidally locked giants have one broad superrotating
equatorial jet, a hot spot east of the substellar point and polar vortices instead of bands. Jupiter
comes out class I with 23 jets, Uranus and Neptune class I with 3 and retrograde equators.

Cold giants (classes I and II) are coloured by their atmospheres. The envelope's heavy elements follow
the Solar System's mass trend (Jupiter ~2×, Saturn ~9×, Uranus and Neptune ~60–80× protosolar carbon),
shifted by the star's [Fe/H]; `body.atmospheric_metallicity` sets it. A temperature profile decides where
ammonia, water or methane condense; above the uppermost cloud, hydrogen scattering, methane absorption,
hazes and Jupiter's red-brown chromophore set the spectrum. Jupiter comes out cream with brown belts,
Saturn butterscotch with faint bands under a thicker haze, and ice giants cyan to azure, paler with less
internal heat (Uranus). The report's "heavy elements" and "visible cloud" rows give the metallicity, the
cloud and its haze. Hotter classes keep model spectra darkened by metallicity.

A giant's world has cloud tops in place of a surface: zonal winds that alternate between the jets,
zones and belts, a population of vortices after the JunoCam catalogue (white and red anticyclones in
zones, brown and dark cyclones in belts, sometimes a great oval), polar cyclones with a ring of
circumpolar cyclones or a hexagon, and on ice giants dark spots and bright clouds. Locked giants show
their day–night temperatures and hot spot. The model behind them is kept with the world and evaluated
wherever it is drawn, so maps, rasters and the globe show vortices smaller than a grid cell:

```python
from worldgen import generate_world, load_spec
from worldgen.export import export_global
from worldgen.render import GlobeViewer, plot_map

world = generate_world(load_spec("jupiter.yaml"), "standard")
plot_map(world, "clouds", "orthographic")          # also temperature, winds, bands, vortices, relief
export_global(world, "jupiter_winds.tif", "winds")
GlobeViewer(world, level=2).show()
```

Zoomed regions, culture and the ground maps (terrain, rainfall, biomes and so on) need solid ground and
refuse giants.

Giants get rings by the temperature their ring material would reach (dense rings form near 70 K, after
Hedman 2015): broad bright icy rings like Saturn's, narrow dark ringlets like Uranus', or tenuous dusty
rings like Jupiter's and Neptune's (with arcs). Hot giants lose their rings to Poynting–Robertson drag, and
rocky planets have rings only when listed. `system.rings` in the spec is `auto` (default), `none`, one of
`broad`, `narrow`, `dusty warm`, `dusty cold`, or a list of rings in planet radii. The report lists them
under "Moons and rings", and the 3D globe draws them.

### Moons and parent planets

A planet's `system.moons` lists moons by mass (Earth masses) and semi-major axis (planet radii); left
out, moons are drawn, and `moons: []` means none. Gas giants that open a gap in their star's nebula get
a Jovian system (4–5 similar moons, the inner ones in 2:1 resonances, rocky inside and icy outside),
others a Saturnian one (one dominant icy moon), after Sasaki et al. (2010); giants with rings also get
small moons spread out of the rings (Crida & Charnoz 2012). Half of ice giants get a Uranian system
from a giant impact (five moons, ~10⁻⁴ of the planet's mass, after Woo et al. 2021), the rest
ring-born moons. Every giant also meets one binary planetesimal, which leaves a captured moon (often
retrograde, like Triton) about a quarter of the time around ice giants; a surviving capture clears the
regular moons. Rocky planets get a moon from a giant impact about two times in five (a large one about
one in ten, after Elser et al. 2011). Every planet's state carries its system summary: the moons,
the stable region (0.36 of the Hill radius prograde, 0.5 retrograde), the ring–moon boundary for icy
and rocky material, the synchronous orbit and the largest moon that survives tidal migration. The
report lists them under "Moons and rings".

A spec with a `parent` section is a moon: `orbit` is then the parent's orbit about the star, and
`parent` gives the parent's mass (and optionally radius, rotation, obliquity, Bond albedo, Q, k₂) and
the moon's orbit about it (`semi_major_axis_rp`, `eccentricity`, `inclination_deg`, and optionally
the parent's `effective_temperature_k` and `magnetic_moment_am2`). The moon locks to its parent, takes
the parent's obliquity, and is checked against the Roche limit and the parent's Hill sphere. Its
climate sees the starlight left after eclipses by the parent plus the light the parent reflects, and
absorbs the parent's glow and the tidal heat the parent raises in it. Rocky moons use fixed-Q tides
with their own Q, raised to Io's k₂/Q of 0.015 when tides would keep the interior partially molten.
Frozen water-rich moons dissipate in their ice shell, between Europa's k₂/Q of ~10⁻³ and
Enceladus's 0.015. A drawn eccentricity is damped by the moon's tides over its age; moons in a
resonance, and moons whose spec gives `eccentricity`, keep theirs. The tidal heat also drives the
interior. The report gives the parent's magnetopause and how much of the moon's orbit lies inside
it. Frozen bodies with surface water get an ice shell over any subsurface ocean, from conduction of
their internal heat (Tjoa et al. 2020) below a porous lid (solid-ice conduction from 135 K). High-
pressure ice (III to VI) lies below the ocean where the IAPWS melting curves put it: Europa gets a
20 km shell, Ganymede 120 km of ice over a 150 km ocean. Cold water-rich worlds hold more nitrogen (Titan's
1.5 bar), thick nitrogen–methane air is hazy (albedo 0.26), and below 63 K the air sits at the N₂
vapour pressure (Triton's microbars). From Python, a planet's moon becomes a world on request:

```python
from worldgen import generate_world, load_world
from worldgen.giants import moon_spec, moon_world

jupiter = load_world("jupiter")
europa = moon_world(jupiter, 1, "preview", folder="jupiter")   # kept under jupiter/moons/01_europa
spec = moon_spec(jupiter, 1)                                   # or edit the spec first
```

A planet's system can be drawn as a diagram, seen from above or edge-on, with the moons' orbits on a
logarithmic radial axis, the rings, the Roche, synchronous, stable and Hill limits and a table of the
moons, or as a 3D scene lit by the star, with the planet's shadow on the rings, the rings' shadow on the
planet and moon shadows and eclipses. The scene compresses distances beyond the rings by default and
draws kept moons with their own surfaces:

```python
from worldgen.render.system import plot_system
from worldgen.render.system_scene import system_scene

plot_system(jupiter, view="edge", path="jupiter_system.png")   # PNG or SVG; highlight="Europa" marks a moon
scene = system_scene(jupiter, folder="jupiter")                # scale="true" for true proportions
scene.screenshot("jupiter_scene.png", t_s=86400)               # one day on
scene.save_html("jupiter_scene.html")                          # an interactive page (vtk.js)
scene.save_movie("jupiter_week.mp4", duration_s=7 * 86400)     # or .gif; scene.show() opens a window with a time slider
```

### Culture section

A spec may carry a `culture:` section (or the same section in its own file, applied to a saved world
with `apply_culture`). It lists one or more cultures, each with:

- `inhabitants`: `native` or `colonist` (colonists add `colony: {founded_years_ago, origin, landing_sites}`; organic colonies also take `immigrants_per_year`, `immigration_years`, `independence_years` and `crowd_diseases`, whether a colony from off the planet brings herd animals' diseases, default false);
- `development` on three axes, each with six levels. Axes left out take the mean level of those given:
  - energy: `foraging`, `farming`, `water_wind`, `fossil`, `electric`, `fusion`;
  - transport: `foot`, `pack_and_boat`, `wheel_and_sail`, `rail_and_steam`, `motor_and_air`, `spaceflight`;
  - communication: `oral`, `writing`, `print`, `telegraph`, `digital`, `interplanetary`;
- `population`, an optional `homeland` (centre and radius), `settlements: {zipf_exponent, largest_population}`
  and `polities: {count, type, capitals, territories}`.

Numeric fields follow the spec convention (number, range or omitted). `culture.seed` redraws the culture
alone. `worldgen template` shows a commented example.

Each culture has a `mode`: `scoped` (the default, placed from its entry as described here) or `organic`
(milestone 9, in progress). An organic culture emerges from a simulated history: its development,
population, settlements and polities are not given (values given are ignored with a warning); a native
one starts as foragers in its `homeland` or on the land it takes, and an organic colony keeps its
`development` as the level it lands with. The optional `history: {start_years_ago, epochs}` block sets the
span (12 000 years by default) and the dates, in years before the present, whose state is kept besides
the present. The history steps one generation (25 years) at a time, and 5 years once any organic culture
burns fossil fuel. Scoped cultures can sit beside organic ones; a scoped colony appears at the epochs
after its founding, other scoped cultures only at the present. People grow up to the land's capacity at their
level and spread into empty land at the speed of the Neolithic farming front (~0.8 km/yr on flat ground,
after Pinhasi et al. 2005), slower in rough country, and across the sea once they have boats. Farming begins
where many foragers hold arable land and spreads as a front; knowledge grows with the number of people in
contact (after Kremer 1993), so large connected populations advance first, and industry needs coal or oil
within reach. `history: {pace: 2}` doubles the pace of knowledge. States form where literate people live
densely and take neighbouring land (after Turchin et al. 2013), the further from their capital the harder
unless their communication reaches far; before states, each cell holds a count of bands, tribes or chiefdoms
(`subpolity_count`). A state may rule people of several cultures. Rulers reign about 11 years, 21 once a
state adopts primogeniture (Kokkonen & Sundell 2012), and at a succession a state may break into its
provinces. States go to war when one is three times stronger at their border, wars spread to neighbours and
allies (after Cederman 2003), and taking a capital takes the state. States keep their ids, founding dates and
parents across epochs and record their rulers; the largest states, wars and break-ups enter the chronicle.
Settlements keep their ids, names and founding dates from one kept epoch to the next (a cell's largest
settlement dates from the cell's first farmers), roads of an earlier epoch are reused and dated, and each
epoch stores its trade links (`civ.trade`: settlement pairs, flow, route length and travel time), with trade
falling off with distance faster before rail than after. Where cultures share land, the one the land holds more
of displaces the other. An organic colony lands with 10⁴ founders and a colonial state, can receive
`immigrants_per_year`, is backed by its home power until independence (`independence_years`, or else at the
decolonisation rate of Strang 1991, five times faster while the home state is at war and at once if it is
conquered; a home state that breaks up passes the colony to its heir), spreads at the speed of its transport, and,
if its home keeps herd animals, brings crowd diseases that kill 90% of the people without livestock of each
landmass it reaches within a century (after the Americas, Koch et al. 2019; Wolfe et al. 2007), who then do not
grow for 250 years. Each farming origin draws whether it has herd animals, more often amid open grassland, and the
herds spread with farming. Colonists die at a settler mortality that rises
with malaria suitability (Acemoglu et al. 2001), so they fill temperate land. States with ocean-going sail and
print found colonies of their own on other landmasses. Peoples ruled by a more advanced culture take its
technology with the adoption lags of Comin & Hobijn (2010) unless the rulers hold them apart (supremacy, drawn
per ruling and likelier the further ahead the rulers are); people shift between cultures sharing land as
languages do (Abrams & Strogatz 2003); under rulers who are not supremacists some become bilingual (Mira &
Paredes 2005), and a pair's bilinguals become a new creole culture; regions of a state where another literate
culture holds two thirds of the people break away. The planet's events press on the history: eruptions
at the planet's volcanic activity cool a hemisphere for five to ten years (after the ice-core record of Toohey
& Sigl 2017 and Sigl et al. 2015), droughts dry regions for decades to centuries, and both cut the land's
capacity, raise war and break-up (14% per standard deviation, Hsiang et al. 2013) and push people to move;
pandemics arise at the rate and size of Marani et al. (2021) and follow the travel network; small impacts
strike at the rate of Brown et al. (2002); and peoples too few or too isolated lose their levels (Henrich
2004). `history: {planet_events: false}` turns these off, and `history.events` places events by hand:

```yaml
history:
  events:
    - {kind: eruption, years_ago: 1500, size: 50}           # Tg of stratospheric sulfur
    - {kind: drought, years_ago: 4200, lat_deg: 35, lon_deg: 40, size: 0.4, radius_km: 1500, duration_years: 250}
    - {kind: pandemic, years_ago: 700, size: 20}             # ‰ of the world's people dying a year
    - {kind: impact, years_ago: 300, lat_deg: 61, lon_deg: 102, size: 10}   # Mt
    - {kind: sea_level, years_ago: 8000, size: 20}           # metres of rise; floods low land for good
```

The civilisation of an organic history carries its kept epochs (`civ.at_epoch(years)`, or
`world_at_epoch(world, years)` for maps and reports), its chronicle and a series table (`civ.series`: per culture
and step, people, settled area, states, the largest state and mean levels). `format_chronicle(civ)` reads the
chronicle by period: origins, firsts, colonies, independence, creoles, planet events and the largest wars and
break-ups in full, other entries counted. `plot_epochs`, `animate_epochs` and `plot_chronicle` draw the history; a
state or culture keeps its colour across epochs. The report ends with a table of the epochs.
The history rates every land cell for fresh water, arable land (crop season, moisture, slope and permafrost,
after Zabel et al. 2014), access by coast and river, resources, climate and defensibility (rugged ground,
coasts and large rivers); each epoch stores these as `suitability_component`.

A civilisation carries two layers from the planet: `tectonic_setting` (craton, sag basin, passive
margin, rift, foreland, strike-slip or subduction margin, orogen, volcanic) and `resource_potential`, a
0–1 potential per cell for copper, gold, tin–tungsten, iron, lead–zinc, hydrocarbons, coal,
evaporites, laterite, timber, uranium, chromite and platinum metals, rare earths, geothermal heat and
placer gold. Potentials follow the geological setting, the sediment, the climate and the planet's past
(a lifeless planet has no coal or oil).

Each culture then gets `suitability` (its carrying capacity relative to its best cell) and
`population_density` (people/km²). Capacity follows productivity, river water, the coast, elevation and
heat, weighted by the culture's energy level; colonists ignore productivity and seek resources instead,
and natives need land life. People gather where others already live. A homeland confines a culture, a
colony spreads from its landing sites with age, and natives without a homeland divide the remaining land.
A population left open follows Earth's population at the culture's energy level (4.4 million foragers
to 7.3 billion at `electric`), scaled by the productive land the culture holds; a colony grows from
10 000 founders. On the Earth example a modern culture puts 36% of its people below 100 m and half within
200 km of the coast, as on Earth (`scripts/validate_culture.py`).
`scripts/validate_organic.py` checks organic histories on six Earth worlds against HYDE 3.2, Taagepera,
Arbesman, Clauset, the ice-core eruption record and Koch et al.; development there runs ahead of Earth's
(known gaps, DESIGN.md 5.2).

Settlements come in regional rank-size systems: each culture's land is split into regions (larger with
better transport), and each region's people live in settlements whose sizes fall as 1/rank^(1/ζ) from
50 people up (`zipf_exponent`, default 1). The largest settlement follows from the region's population,
or the spec's `largest_population` sets the number of regions. Settlements down to about one per
occupied cell are listed with position, population, form (hamlet to metropolis), region and rank;
larger ones keep central-place distances from each other. The rest stay as each cell's
`dispersed_density`. Foragers live dispersed, in camps that zoomed maps draw.

Roads join each culture's settlements and every other cell where at least 500 of its people live.
Every node links to its strongest gravity partners nearby, and the links are routed by travel time
(Tobler's hiking function, slowed by slope, rivers and ice; boats from `pack_and_boat` on). Busy
links are routed first and later ones reuse them, so trunks form. Each road segment has a path of
grid cells, a length, the flow it carries and a class (track, road, trunk, highway or sea route) that
depends on the flow and the transport level. The global grid holds the main roads; minor roads belong
to zoomed maps.

Polities divide each culture's land. Capitals are the largest settlements (or those the spec names),
and each cell joins the capital with the most influence: larger capitals reach further, and travel
time counts against them, so borders fall along mountains, wide rivers and straits. Without a
`polities.count`, the number of polities follows Earth's political fragmentation at the culture's
communication level. Each polity has a capital, type, population and area; territories are the
`polity` layer, and spec `territories` override them.

Zoomed regions carry the culture down to villages. Each cell's dispersed people become villages and
hamlets, sized so the rank-size law continues below the listed settlements and placed on flat ground
near water; every settlement gets a built-up footprint that grows with its population. Local tracks
and roads link neighbouring settlements by the same gravity-and-reuse routing on the region's real
slopes, the global main roads run through the towns they pass, and polity borders are redrawn by
travel time, so they follow large rivers and ridges. `worldgen zoom` draws all this when the world has
a culture.

Every culture gets its own sound system: a consonant and vowel inventory, a syllable shape and a
spelling, drawn so that across many cultures they match the world's languages (WALS and PHOIBLE).
Settlements, polities and the culture itself (unless the spec names it) are named from it, and places
by rivers, coasts, lakes or hills sometimes carry the culture's word for that feature. Names are unique
within a culture, and the sound system is saved with it.

Colonists live where the planet lets them. Each land cell is open (breathable air, comfortable pressure
and temperature, enough air overhead to stop cosmic rays), domed, underground (air too thin to shield)
or uninhabitable (too hot or cold), stored as the `habitat` layer. Colonies land on low, flat,
low-latitude ground, favour open cells, and name their settlements outposts, domes or underground
habitats accordingly; on low-gravity volcanic ground lava tubes help. Where a space elevator could
stand, the spacefaring colony's largest settlement near the equator becomes its anchor. A planet with no
survivable surface, like Venus, cannot take colonists.

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
| `spec.py`, `spec_base.py` | Spec schema, YAML load/save, template; shared section base |
| `priors/` | Archetypes, distributions, sampling, outcome constraints, occupiability score |
| `star.py` | Luminosity, radius, temperature, lifetime, XUV activity |
| `orbit.py` | Period, instellation, habitable zone, tidal locking (to the star, or a moon's to its parent), spin |
| `giants/` | Giants and moons (milestone 11): giant appearance (cloud class, albedo, colour, jets), moon and ring limits, the system summary, parent planets, `moon_spec` and `moon_world` |
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
| `tectonics/` | Plate simulation: crust points, Numba kernels, processes, re-mapping, subsidence, biogenic sediment, conversion to a surface |
| `render/` | Cartopy map projections, colour schemes, hillshading, smoothed rivers, drift figures and animations, history figures, local maps of zoomed regions, the system diagram and 3D system scene |
| `zoom/` | Cube-sphere quadtree tiles and region lattices, inherited surface and downscaled climate, sub-grid detail, local erosion, glaciers and drainage, the `zoom_region` request and saved regions |
| `culture/` | Culture spec, development axes, resolved cultures, tectonic settings and resource potentials, suitability, population density, settlements, roads, polities and names, the civilisation data model (cultures, polities, settlement and road tables, per-cell layers, chronicle) and its storage; `culture/history/` the organic mode (history spec, steps, state, run, kept epochs) |
| `export/` | Export (milestone 10): fields with units and labels, planet-sphere and LAEA CRS, barycentric samplers, GeoTIFF/PNG/RAW writers with JSON sidecars, global rasters, region rasters and bundles, GeoJSON culture vectors, engine heightmaps, the tile pyramid, RTIN meshes |
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
| `validate_culture.py` | Culture on Earth worlds against Earth (population, settlements, roads, polities), sound systems against WALS and PHOIBLE, zoomed regions, and colonies on hostile worlds (`--hostile`); known gaps are shown but not counted; exits non-zero if a check misses |
| `validate_solar_system.py` | The giants (T_eff, albedo, jets, rings), drawn moon systems over many draws, the Earth–Moon impact statistics, and Io, Europa, Ganymede, Callisto, Enceladus, Titan, Triton and the Moon built from their real orbits, each against a cited value; exits non-zero if a check misses |
| `validate_export.py` | Every export of a world against its source (file round trips, georeferencing against GDAL's thin-plate warp, seams between faces, levels and mesh tiles, the mesh tolerance) with timings; `--pyramid` adds the full pyramid; exits non-zero if a check misses |
| `history_sensitivity.py` | Ranks the poorly constrained history parameters by how far they move the outcome, one at a time over each prior |
| `hypsometry_table.py` | Reference land fraction against ocean volume, for `worldgen/water.py` |
| `render_examples.py` | The example figures in "Claude outputs" |
| `planet_gallery.py` | Several planets with their reports, saved worlds and every applicable map and figure, one folder each |

## Tests

```bash
pytest                   # full suite, ~9 min on one core (~5.5 min with -n 2)
pytest -n auto           # in parallel (needs pytest-xdist, included in the dev extras)
pytest -m "not slow"     # skip tests that generate worlds, ~25 s
pytest -m zoom           # one category (physics, history, surface, climate, hydrology, zoom,
                         # culture, organic, render, export, cli)
python scripts/select_tests.py -n 4            # only the tests relevant to uncommitted changes
python scripts/select_tests.py --base HEAD~1   # ... to changes since a commit; --list prints the selection
python scripts/select_tests.py --tests-only    # only the changed and new test files
pytest tests/test_system.py::test_moon_locks_to_parent   # a single test
```

`tests/categories.py` sorts the tests into categories by the part of the package they cover and lists
which categories build on which (a surface change also runs hydrology, zoom and export). The selection
script maps changed files to categories, runs changed test files directly, maps a change in
`heuristics.py` to the modules that use the changed constants, skips documentation and notebooks, and
falls back to the full suite for files no category covers (such as `conftest.py`). The end-to-end
pipeline test and the check of the tables always run. New test files must be added to a category.

The suite checks each relation against published values, runs Earth, Venus and Mars through the full
pipeline, checks that every archetype meets its constraints, and covers grids, terrain for every regime,
the tectonic simulation's rules and outputs, the water inventory, drainage, lakes and erosion,
the climate tiers (insolation, energy balance, moisture, snowball bistability, day–night contrast), ice
sheets, life, vegetation, biomes and Köppen classes, saving and loading, rendering in every projection,
and the zoom: tile addressing and seams, inheritance, detail, local erosion, glaciers, streams, saved and
cached regions, and local maps; subsidence and biogenic sediment; and the culture spec, its resolution,
tectonic settings, resource potentials, carrying capacity, clustering, culture ranges, default
populations, settlement regions and placement, travel costs, routing and road classes, polity partitions, counts and types,
sound systems and names, colonist habitats and elevators, the saved civilisation, and the organic
history's spec, steps, kept epochs and their storage, moon and ring limits against Sicardy et al. (2024) and Barnes & O'Brien (2002), parents and moon worlds, its suitability components, the growth and spread of people, and farming and development.
