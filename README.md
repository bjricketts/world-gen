# worldgen

worldgen generates planets for world building from physical inputs. Give it as much or as little as you like (star, orbit, mass, atmosphere) and it fills in the rest from physics and archetype priors, then reports where every value came from and anything that does not add up. The planet gets a surface, climate, biomes, rivers and, optionally, moons, rings and cultures. Any region can be zoomed into for a local map.

![Overview of the Earth example: elevation, temperature, rainfall and biomes](docs/images/earth_overview.png)

## Modelled behaviour

| Area | What is computed |
|---|---|
| Star, orbit, bulk | Luminosity, habitable zone, tidal locking, mass–radius–composition, gravity |
| Interior | Radiogenic heat, tectonic regime, magnetic dynamo |
| Surface | Plate-tectonic simulation or heuristic landforms, sea level from the water inventory, rivers, lakes, erosion |
| Atmosphere and climate | Retention, greenhouse, energy-balance climate with seasons, ice sheets, vegetation, biomes, Köppen classes |
| History | The planet integrated from formation to a chosen epoch |
| Giants and moons | Giant cloud class and colours, rings, moon systems, moons built as worlds |
| Zoom | Local maps of any region with sub-grid relief, erosion, glaciers and drainage |
| Culture | Settlements, roads, polities and languages, placed from a spec or grown from a simulated history |
| Export | GeoTIFF, PNG and RAW rasters, region bundles, engine heightmaps, tile pyramids, meshes, a 3D globe |

The model is described in [docs/design.md](docs/design.md). What is not modelled, with the measured size of each gap, is in [docs/open-points.md](docs/open-points.md).

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/bjricketts/world-gen
cd world-gen
pip install -e .
```

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `export` | rasterio, trimesh | GeoTIFF and mesh export (PNG and RAW export need nothing) |
| `globe` | PyVista, ipywidgets, imageio | The 3D globe, the 3D system scene and movies |
| `notebooks`, `explore` | JupyterLab, ipywidgets, ipympl | Running the notebooks and the history explorer |
| `dev` | pytest, pytest-xdist, ruff, nbclient | Tests and linting |
| `data` | miepython | Rebuilding `giant_optics.npz` with `scripts/build_giant_optics.py` |

## Basic usage

From a spec file:

```bash
worldgen template planet.yaml                      # write an annotated spec to edit
worldgen generate planet.yaml --save mypl          # build the planet, print its report, keep the world
worldgen map mypl terrain.png -f terrain -p robinson
worldgen zoom mypl coast.png --lat 35 --lon 12     # local map around a place
```

Or from an archetype:

```bash
worldgen archetypes                                # list world types
worldgen random -a temperate --seed 3 --map overview.png
```

From Python:

```python
from worldgen import PlanetSpec, generate, generate_world, format_report

spec = PlanetSpec(
    name="Ember",
    seed=7,
    priors={"archetype": "tidally_locked"},
    star={"mass_msun": (0.15, 0.3)},       # a range is drawn from
    body={"mass_mearth": 1.4},             # a number is fixed
)                                          # anything omitted is drawn or derived
state, timeline = generate(spec)
print(format_report(state))

world = generate_world(spec, resolution="standard")   # adds the surface, climate and biomes
world.surface                                          # an xarray Dataset of all map fields
```

Resolutions are `preview` (10,000 cells), `standard` (40,000), `high` (160,000) or a cell count.

## Command-line interface

`worldgen <command> --help` lists the options of each command. You can save a world to a folder with `--save`.

| Command | Purpose |
|---|---|
| `archetypes` | List the world types |
| `template` | Write an annotated spec file to edit |
| `generate` | Build a planet from a spec, print its report, and optionally map, save or export it |
| `random` | Build a random planet, optionally from an archetype |
| `map` | Draw a map of a saved world in any field and projection |
| `report` | Print the report of a saved world |
| `drift` | Draw the simulated continental drift as an animation or panel figure |
| `history` | Draw the integrated history (climate, interior or biosphere against time) |
| `zoom` | Draw the local map of a region of a saved world |
| `culture` | Place a culture on a saved world, print its report, and optionally map it |
| `chronicle` | Print the chronicle of an organic culture history |
| `epochs` | Draw a culture field at the kept epochs of an organic history, or its timeline |
| `system` | Draw a planet's moons, rings and limits as a diagram, or show the system in 3D |
| `moon` | Build a planet's moon as a world |
| `globe` | Show a saved world as a 3D globe, or write images or interactive pages |
| `export` | Export heightfields, region bundles, engine heightmaps, tile pyramids and meshes |

## Planet specifications

A spec is a YAML file in practical units (solar masses, AU, Earth masses, bar, hours, degrees). Each numeric field is a number (fixed), a `[low, high]` range (drawn) or omitted (drawn from the archetype prior or derived from physics). Each value resolves in this order, and the report records whether it was calculated, supplied or assumed:

| Source | Meaning |
|---|---|
| user | Fixed in the spec |
| range | Drawn from a range in the spec |
| archetype | Drawn from the prior of the named archetype, or of one compatible with the fixed values |
| prior | Drawn from a broad base prior |
| derived, heuristic, default | Computed by the physics modules |

The archetypes are `temperate`, `arid`, `ocean`, `ice`, `tidally_locked`, `hot_volcanic`, `super_earth`, `small_world`, `habitable_moon` and `giant`. The same seed gives the same planet, and fixing one value does not change the seeded values of the other parameters. [docs/spec.md](docs/spec.md) lists every section and field. Examples are in `examples/`.

## Python interface

The functions and classes in `worldgen.__all__` (`PlanetSpec`, `load_spec`, `save_spec`, `generate`, `generate_world`, `candidates`, `format_report`, `format_timeline`, `save_state`, `World`, `save_world`, `load_world`), the command-line interface and the file formats below form the public interface and follow semantic versioning. The submodules used in the notebooks (`worldgen.render`, `worldgen.zoom`, `worldgen.culture`, `worldgen.giants`, `worldgen.export`) are documented in [docs/design.md](docs/design.md) and may change between minor releases.

## File formats

| File | Version | Readable |
|---|---|---|
| Spec (YAML) | `spec_version` 1 | Files without the field are version 1 |
| Saved world (zarr and YAML) | 8 | 4 to 8 |
| Culture | 3 | 1 to 3 |
| Zoomed region | 1 | 1 |

## Documentation

| Document | Content |
|---|---|
| [docs/design.md](docs/design.md) | The model as built: equations, sources, constants, validation numbers |
| [docs/spec.md](docs/spec.md) | The spec file: sections, fields, units and defaults |
| [docs/open-points.md](docs/open-points.md) | Open items by subsystem, with measured impact |
| [docs/references.md](docs/references.md) | The sources cited in the documents and in the code |
| [CHANGELOG.md](CHANGELOG.md) | Changes by release |

## Notebooks

Run them from the `notebooks/` folder. They contain saved outputs, so they can be read without running.

| Notebook | Content | Extras |
|---|---|---|
| `01_quickstart` | Generate, report, map, zoom, place a culture, save and reload | `dev`, `notebooks` |
| `02_organic_history` | An organic culture history on a generated planet | `dev`, `notebooks` |
| `03_history_explorer` | Browse an organic history by field and epoch | `explore` |
| `04_export_and_globe` | Rasters, region bundles, tile pyramids, meshes and the 3D globe | `export`, `globe` |
| `05_giants_and_moons` | Giant appearance, rings, moon systems, moon worlds | `globe` |

## Validation and tests

The model is calibrated against Earth, Mars, Venus and the giants and moons of the Solar System. The scripts below rerun those comparisons from the repository root (`python scripts/<name>.py`). Those marked "exit" return a non-zero status when a check fails.

| Script | Purpose |
|---|---|
| `validate_earth.py` | The Earth example against Earth over several seeds: hypsometry, ocean, plates, rivers, lakes, climate, ice, Köppen shares. `--set NAME=VALUE` overrides a heuristic for a sweep |
| `archetype_fidelity.py` | Archetype outcomes over many draws: temperatures, water phase, open ocean, land, life |
| `validate_history.py` | History-mode Earth, Mars and Venus against the real planets, and Earth against its snapshot build (exit) |
| `validate_culture.py` | Culture on Earth worlds against Earth, sound systems against WALS and PHOIBLE, zoomed regions, colonies on hostile worlds with `--hostile` (exit) |
| `validate_organic.py` | Organic histories on the Earth example against Earth's Holocene: population, farming, development, states, wars, planet events |
| `validate_solar_system.py` | The giants, drawn moon systems, and Io, Europa, Ganymede, Callisto, Enceladus, Titan, Triton and the Moon against cited values (exit) |
| `validate_export.py` | Every export against its source; `--pyramid` adds the full tile pyramid, about 8 minutes (exit) |
| `history_sensitivity.py` | Ranks the poorly constrained history parameters by their effect on the outcome |
| `hypsometry_table.py` | Reference land fraction against ocean volume |
| `build_giant_optics.py` | Rebuilds `worldgen/data/giant_optics.npz` (needs the `data` extra) |
| `render_examples.py`, `planet_gallery.py` | Example figures, and several planets with every applicable map |

Tests:

```bash
pytest                   # full suite, about 7 minutes with two workers
pytest -n auto           # in parallel (pytest-xdist is in the dev extra)
pytest -m "not slow"     # skip tests that generate worlds, about a minute
pytest -m zoom           # one category: physics, history, surface, climate, hydrology, zoom,
                         # culture, organic, render, export, cli
python scripts/select_tests.py -n 4   # only the tests relevant to uncommitted changes
```

## Reproducibility

The same spec and seed give the same planet on one set of library versions. Different versions of numpy, scipy and the other dependencies can give a different planet from the same seed (see [docs/open-points.md](docs/open-points.md)). State the library versions with any published world if you have any issues.

## Citation

See [CITATION.cff](CITATION.cff). GitHub also has a "Cite this repository" on this repository.

## Licence and data

MIT licence (see [LICENSE](LICENSE)). [NOTICE](NOTICE) lists the third-party data and sources used; the cited works that make this project possible are in [docs/references.md](docs/references.md).
