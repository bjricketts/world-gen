"""Command-line interface: ``worldgen --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .constants import M_EARTH
from .generate import candidates, generate, generate_world
from .priors import ARCHETYPES
from .report import format_culture, format_report, format_timeline, save_state
from .spec import SPEC_TEMPLATE, PlanetSpec, PriorSpec, load_spec
from .world import World, load_world, save_world

app = typer.Typer(help="Generate physically motivated planets for world building.", no_args_is_help=True)

RESOLUTION_HELP = "Grid resolution: preview (10k cells), standard (40k), high (160k), or a cell count."
MAP_HELP = "Write a map image to this file (PNG, JPG, PDF or SVG)."
PROJECTION_HELP = "Map projection: equirectangular, mollweide, robinson, orthographic, north_polar, south_polar."
FIELD_HELP = ("Map field: elevation, terrain, plates, crust_age, orogeny_age, temperature, rainfall, basins, "
              "biomes, koppen, ice; culture layers political, population, cultures, polities, settlements, roads, "
              "habitat, resources:<name> or resources:best, suitability[:<culture>]; from an organic history energy, "
              "transport, communication[:<culture>], bands, livestock, land:<component>, events; or 'overview' / "
              "'climate' / 'culture' for a multi-panel figure.")
TECTONICS_HELP = "Plate tectonics: simulated (default) or heuristic (fast, no history)."
START_HELP = "Initial continents of the simulation: supercontinent or cratons (default: drawn per seed)."
DURATION_HELP = "Simulated tectonic time in Myr (default 400)."
SNAPSHOTS_HELP = ("Store the tectonic history every N Myr (rounded to 20 Myr steps) in the saved world; "
                  "simulated tectonics only.")
EPOCHS_HELP = "History mode: ages in Gyr to keep full states for, e.g. '0.5,1,2' (default: the spec's epochs)."
TOPIC_HELP = "History figure: climate, interior or life."
EPOCH_HELP = "Show the culture as it stood at this kept epoch of its organic history (years ago; 0 is the present)."


def _at_epoch(world: World, epoch: Optional[float]) -> World:
    """Return the world with its culture at a kept epoch, or unchanged when no epoch is asked for."""
    if epoch is None:
        return world
    from .culture import world_at_epoch

    try:
        return world_at_epoch(world, epoch)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e.args[0] if e.args else e}", err=True)
        raise typer.Exit(code=1)


def _resolution(value: str) -> int | str:
    """Return a named resolution unchanged, or a numeric one as an integer."""
    return int(value) if value.isdigit() else value


def _epochs(value: str) -> list[float]:
    """Return the epoch ages parsed from a comma-separated option."""
    try:
        return sorted(float(part) for part in value.replace(" ", "").split(",") if part)
    except ValueError:
        raise typer.BadParameter(f"could not read '{value}' as a list of ages in Gyr")


def _spec_with_overrides(spec: PlanetSpec, seed: Optional[int] = None, mode: Optional[str] = None,
                         tectonics: Optional[str] = None, start: Optional[str] = None,
                         duration: Optional[float] = None, epochs: Optional[str] = None) -> PlanetSpec:
    """Return the spec with command-line options applied."""
    update = {}
    if seed is not None:
        update["seed"] = seed
    if mode is not None:
        update["mode"] = mode
    if epochs is not None:
        update["history"] = spec.history.model_copy(update={"epochs_gyr": _epochs(epochs)})
    surface = {k: v for k, v in (("tectonics", tectonics), ("tectonics_start", start),
                                 ("tectonics_duration_myr", duration)) if v is not None}
    if surface:
        try:
            update["surface"] = spec.surface.model_validate({**spec.surface.model_dump(), **surface})
        except ValueError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(code=1)
    return spec.model_copy(update=update)


def _write_map(world: World, path: Path, field: str, projection_name: str,
               central_longitude: float, central_latitude: float, events: bool = False) -> None:
    """Draw the requested map of a world and write it to an image file."""
    import matplotlib

    matplotlib.use("Agg")
    from .render import plot_climate, plot_culture, plot_map, plot_overview, save_figure

    if world.surface is None:
        typer.echo(f"no map: {world.state.name} has no solid surface", err=True)
        return
    if world.cloud_tops and field in ("overview", "climate", "culture"):
        typer.echo(f"no {field} map: {world.state.name} is a giant; map its cloud tops "
                   "(clouds, temperature, winds, bands, vortices, relief)", err=True)
        return
    if field == "overview":
        fig = plot_overview(world)
    elif field == "climate":
        fig = plot_climate(world)
    elif field == "culture":
        fig = plot_culture(world)
    else:
        fig = plot_map(world, field, projection_name, central_longitude=central_longitude,
                       central_latitude=central_latitude, events=events or None).figure
    save_figure(fig, path)
    typer.echo(f"map written to {path}")


def _run(spec: PlanetSpec, epoch: Optional[float], out: Optional[Path], n: int, resolution: str,
         map_path: Optional[Path], field: str, projection_name: str, lon: float, lat: float,
         save: Optional[Path], snapshots: Optional[float] = None) -> None:
    """Generate the planet, print its report, and write any requested outputs."""
    try:
        if n > 1:
            ranked = candidates(spec, n)
            typer.echo(f"{'rank':<5}{'seed':<8}{'archetype':<16}{'class':<18}{'T_s [K]':>8}{'score':>8}")
            for rank, s in enumerate(ranked, 1):
                typer.echo(f"{rank:<5}{s.seed:<8}{s.archetype or '-':<16}{s.bulk.planet_class:<18}"
                           f"{s.atmosphere.surface_temperature_k:>8.0f}{s.occupiability.score:>8.2f}")
            spec = spec.model_copy(update={"seed": ranked[0].seed, "name": ranked[0].name})
            typer.echo("\nBest candidate:\n")
        if map_path is not None or save is not None:
            world = generate_world(spec, _resolution(resolution), epoch, snapshots)
            state = world.state
        else:
            world = None
            state, _ = generate(spec, epoch)
    except NotImplementedError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(format_report(state, [] if save is not None and state.system is not None else None))
    if world is not None and world.culture is not None:
        typer.echo(format_culture(world.culture, world))
    if out is not None:
        save_state(state, out)
        typer.echo(f"\nstate written to {out}")
    if save is not None:
        save_world(world, save)
        typer.echo(f"world saved to {save}")
    if map_path is not None:
        _write_map(world, map_path, field, projection_name, lon, lat)


@app.command("generate")
def generate_cmd(
    spec_file: Path = typer.Argument(..., exists=True, help="YAML planet specification."),
    seed: Optional[int] = typer.Option(None, help="Override the spec's seed."),
    mode: Optional[str] = typer.Option(None, help="Override the mode: snapshot or history."),
    epoch: Optional[float] = typer.Option(None, help="Target epoch in Gyr (default: the star's age)."),
    candidates_n: int = typer.Option(1, "--candidates", "-n", help="Generate N variants and keep the best."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the global state to this YAML file."),
    resolution: str = typer.Option("standard", "--resolution", "-r", help=RESOLUTION_HELP),
    map_path: Optional[Path] = typer.Option(None, "--map", "-m", help=MAP_HELP),
    field: str = typer.Option("overview", "--field", "-f", help=FIELD_HELP),
    projection_name: str = typer.Option("mollweide", "--projection", "-p", help=PROJECTION_HELP),
    lon: float = typer.Option(0.0, help="Central longitude of the map (degrees)."),
    lat: float = typer.Option(20.0, help="Central latitude for globe views (degrees)."),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save the world (state and surface) to this folder."),
    tectonics: Optional[str] = typer.Option(None, "--tectonics", help=TECTONICS_HELP),
    start: Optional[str] = typer.Option(None, "--start", help=START_HELP),
    duration: Optional[float] = typer.Option(None, "--duration", help=DURATION_HELP),
    snapshots: Optional[float] = typer.Option(None, "--snapshots", help=SNAPSHOTS_HELP),
    epochs: Optional[str] = typer.Option(None, "--epochs", help=EPOCHS_HELP),
) -> None:
    """Generate a planet from a spec file, print its report, and optionally map or save it."""
    spec = _spec_with_overrides(load_spec(spec_file), seed, mode, tectonics, start, duration, epochs)
    _run(spec, epoch, out, candidates_n, resolution, map_path, field, projection_name, lon, lat, save, snapshots)


@app.command("random")
def random_cmd(
    archetype: Optional[str] = typer.Option(None, "--archetype", "-a", help="Archetype name (default: weighted mix)."),
    seed: int = typer.Option(0, help="Random seed."),
    candidates_n: int = typer.Option(1, "--candidates", "-n", help="Generate N variants and keep the best."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the global state to this YAML file."),
    resolution: str = typer.Option("standard", "--resolution", "-r", help=RESOLUTION_HELP),
    map_path: Optional[Path] = typer.Option(None, "--map", "-m", help=MAP_HELP),
    field: str = typer.Option("overview", "--field", "-f", help=FIELD_HELP),
    projection_name: str = typer.Option("mollweide", "--projection", "-p", help=PROJECTION_HELP),
    lon: float = typer.Option(0.0, help="Central longitude of the map (degrees)."),
    lat: float = typer.Option(20.0, help="Central latitude for globe views (degrees)."),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save the world (state and surface) to this folder."),
    tectonics: Optional[str] = typer.Option(None, "--tectonics", help=TECTONICS_HELP),
    start: Optional[str] = typer.Option(None, "--start", help=START_HELP),
    duration: Optional[float] = typer.Option(None, "--duration", help=DURATION_HELP),
    snapshots: Optional[float] = typer.Option(None, "--snapshots", help=SNAPSHOTS_HELP),
    mode: Optional[str] = typer.Option(None, help="Evolution mode: snapshot or history."),
    epochs: Optional[str] = typer.Option(None, "--epochs", help=EPOCHS_HELP),
) -> None:
    """Generate a random planet, optionally from an archetype."""
    if archetype is not None and archetype not in ARCHETYPES:
        typer.echo(f"unknown archetype '{archetype}'; see `worldgen archetypes`", err=True)
        raise typer.Exit(code=1)
    spec = PlanetSpec(name="Random world", seed=seed, priors=PriorSpec(archetype=archetype))
    spec = _spec_with_overrides(spec, mode=mode, tectonics=tectonics, start=start, duration=duration,
                                epochs=epochs)
    _run(spec, None, out, candidates_n, resolution, map_path, field, projection_name, lon, lat, save, snapshots)


@app.command("map")
def map_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    map_path: Path = typer.Argument(..., help=MAP_HELP),
    field: str = typer.Option("overview", "--field", "-f", help=FIELD_HELP),
    projection_name: str = typer.Option("mollweide", "--projection", "-p", help=PROJECTION_HELP),
    lon: float = typer.Option(0.0, help="Central longitude of the map (degrees)."),
    lat: float = typer.Option(20.0, help="Central latitude for globe views (degrees)."),
    epoch: Optional[float] = typer.Option(None, "--epoch", "-e", help=EPOCH_HELP),
    events: bool = typer.Option(False, "--events", help="Draw the planet events since the previous kept epoch "
                                                          "(organic histories)."),
) -> None:
    """Draw a map of a saved world."""
    world = _at_epoch(load_world(world_dir), epoch)
    if (events or field == "events") and world.culture is None:
        typer.echo("error: events need a world with an organic history", err=True)
        raise typer.Exit(code=1)
    _write_map(world, map_path, field, projection_name, lon, lat, events)


@app.command("drift")
def drift_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save --snapshots."),
    path: Path = typer.Argument(..., help="Output file: a GIF gives an animation, other image types a panel figure."),
    fps: int = typer.Option(4, help="Frames per second of the animation."),
    panels: int = typer.Option(6, help="Number of panels in the figure."),
    projection_name: str = typer.Option("mollweide", "--projection", "-p", help=PROJECTION_HELP),
    lon: float = typer.Option(0.0, help="Central longitude of the map (degrees)."),
) -> None:
    """Draw the simulated continental drift of a saved world as an animation or a panel figure."""
    import matplotlib

    matplotlib.use("Agg")
    from .render import animate_history, has_history, plot_history, save_figure

    world = load_world(world_dir)
    if not has_history(world):
        typer.echo("no tectonic history in this world; generate it with --snapshots", err=True)
        raise typer.Exit(code=1)
    if path.suffix.lower() == ".gif":
        animate_history(world, path, fps=fps, projection_name=projection_name, central_longitude=lon)
    else:
        save_figure(plot_history(world, panels, projection_name, lon), path)
    typer.echo(f"drift written to {path}")


@app.command("history")
def history_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    path: Path = typer.Argument(..., help="Output image file (PNG, JPG, PDF or SVG)."),
    topic: str = typer.Option("climate", "--field", "-f", help=TOPIC_HELP),
) -> None:
    """Draw the integrated history of a saved world: its climate, interior or biosphere against time."""
    import matplotlib

    matplotlib.use("Agg")
    from .render import has_timeline, plot_timeline, save_figure

    world = load_world(world_dir)
    if not has_timeline(world):
        typer.echo("no integrated history in this world; generate it with mode: history", err=True)
        raise typer.Exit(code=1)
    try:
        figure = plot_timeline(world, topic)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    save_figure(figure, path)
    typer.echo(f"history written to {path}")


@app.command("zoom")
def zoom_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    path: Path = typer.Argument(..., help="Output image file (PNG, JPG, PDF or SVG)."),
    lat: float = typer.Option(..., help="Latitude of the region's centre (degrees)."),
    lon: float = typer.Option(..., help="Longitude of the region's centre (degrees)."),
    resolution: Optional[float] = typer.Option(None, "--resolution", "-r",
                                               help="Lattice spacing in metres (default 750)."),
    level: Optional[int] = typer.Option(None, help="Quadtree level instead of a resolution."),
    half_width: Optional[float] = typer.Option(None, "--half-width",
                                               help="Half-width of the region in km (default 75)."),
    ruggedness: str = typer.Option("earth", help="Sub-grid relief: earth, dramatic or gentle."),
    field: str = typer.Option("relief", "--field", "-f", help="Map tint: relief or biome."),
    cache: bool = typer.Option(False, "--cache", help="Keep the region in the world folder under regions/ and "
                                                       "reuse it next time."),
    culture: bool = typer.Option(True, "--culture/--no-culture", help="Draw settlements, roads and borders when "
                                                                      "the world has a culture."),
    epoch: Optional[float] = typer.Option(None, "--epoch", "-e", help=EPOCH_HELP),
) -> None:
    """Zoom into a saved world and draw the local map of the region around a place."""
    import time

    import matplotlib

    matplotlib.use("Agg")
    from . import heuristics as h
    from .render import save_local_map
    from .world import REGIONS_FOLDER
    from .zoom import zoom_region

    if field not in ("relief", "biome"):
        typer.echo(f"error: unknown field '{field}'; choose relief or biome", err=True)
        raise typer.Exit(code=1)
    world = _at_epoch(load_world(world_dir), epoch)
    start = time.perf_counter()
    try:
        region = zoom_region(world, lat, lon, resolution_m=resolution, level=level,
                             half_width_km=half_width if half_width is not None else h.ZOOM_DEFAULT_HALF_WIDTH_KM,
                             ruggedness=ruggedness, cache_dir=world_dir / REGIONS_FOLDER if cache else None)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    local = None
    if culture and world.culture is not None and "population_density" in world.culture.layers:
        from .culture.local import zoom_culture

        local = zoom_culture(world, region)
    save_local_map(world, region, path, mode=field, title=f"{world.state.name}: {lat:.2f}°, {lon:.2f}°",
                   culture=local)
    if local is not None:
        typer.echo(f"{len(local):,} settlements ({local.population.sum():,.0f} people), "
                   f"{len(local.roads):,} road segments")
    rows, cols = region.grid.shape
    spacing = region.grid.mean_spacing(region.radius_m)
    typer.echo(f"region {cols * spacing / 1e3:.0f} × {rows * spacing / 1e3:.0f} km at {spacing:.0f} m "
               f"(level {region.grid.level}, {region.grid.size:,} nodes) in {time.perf_counter() - start:.1f} s; "
               f"map written to {path}")


@app.command("report")
def report_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    timeline: bool = typer.Option(False, "--timeline", "-t", help="Print the full event log and epoch table instead."),
    epoch: Optional[float] = typer.Option(None, "--epoch", "-e", help=EPOCH_HELP),
) -> None:
    """Print the report of a saved world."""
    world = _at_epoch(load_world(world_dir), epoch)
    if not timeline:
        from .giants import kept_moons

        kept = list(kept_moons(world_dir, world)) if world.state.system is not None else None
        typer.echo(format_report(world.state, kept))
        if world.culture is not None:
            typer.echo(format_culture(world.culture, world))
        return
    if world.timeline is None or not world.timeline.events:
        typer.echo("no integrated history in this world; generate it with mode: history", err=True)
        raise typer.Exit(code=1)
    typer.echo(format_timeline(world.timeline, world.state))


@app.command("culture")
def culture_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    culture_spec: Path = typer.Argument(..., exists=True, dir_okay=False,
                                        help="YAML file with a culture section (or the section itself)."),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save the world with its culture to this folder."),
    map_path: Optional[Path] = typer.Option(None, "--map", "-m", help=MAP_HELP),
    field: str = typer.Option("culture", "--field", "-f", help=FIELD_HELP),
    projection_name: str = typer.Option("mollweide", "--projection", "-p", help=PROJECTION_HELP),
    lon: float = typer.Option(0.0, help="Central longitude of the map (degrees)."),
    lat: float = typer.Option(20.0, help="Central latitude for globe views (degrees)."),
    epoch: Optional[float] = typer.Option(None, "--epoch", "-e", help="Report and map this kept epoch of an organic "
                                                                       "history (years ago); the save keeps them all."),
) -> None:
    """Place a culture on a saved world, print its report, and optionally save the world and draw a map."""
    from .culture import apply_culture, load_culture_spec

    world = load_world(world_dir)
    if not world.has_ground:
        typer.echo(f"error: {world.state.name} has no solid surface for a culture", err=True)
        raise typer.Exit(code=1)
    try:
        world = apply_culture(world, load_culture_spec(culture_spec))
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    shown = _at_epoch(world, epoch)
    typer.echo(format_culture(shown.culture, shown))
    if save is not None:
        save_world(world, save)
        typer.echo(f"\nworld saved to {save}")
    if map_path is not None:
        _write_map(shown, map_path, field, projection_name, lon, lat)


@app.command("chronicle")
def chronicle_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    since: Optional[float] = typer.Option(None, help="Start this many years ago (default: the history's start)."),
    until: Optional[float] = typer.Option(None, help="End this many years ago (default: the present)."),
    kind: Optional[list[str]] = typer.Option(None, "--kind", "-k", help="Only entries of this kind (repeatable), "
                                                                         "e.g. war, pandemic, independence."),
    culture: Optional[str] = typer.Option(None, "--culture", "-c", help="Only entries involving this culture "
                                                                         "(name or id)."),
    minor: str = typer.Option("count", help="Minor entries: count (per period), list or hide."),
) -> None:
    """Print the chronicle of a saved world's organic history: major entries in full, minor ones counted."""
    from .culture import format_chronicle

    world = load_world(world_dir)
    civ = world.culture
    if civ is None or not civ.chronicle:
        typer.echo("no chronicle in this world; give it a culture with organic cultures", err=True)
        raise typer.Exit(code=1)
    if minor not in ("count", "list", "hide"):
        typer.echo(f"error: unknown --minor '{minor}'; choose count, list or hide", err=True)
        raise typer.Exit(code=1)
    cid = None
    if culture is not None:
        known = {c.name.lower(): c.id for e in [*civ.epochs, civ] for c in e.cultures if c.name}
        cid = int(culture) if culture.isdigit() else known.get(culture.lower())
        if cid is None:
            typer.echo(f"error: no culture named '{culture}'", err=True)
            raise typer.Exit(code=1)
    typer.echo(format_chronicle(civ, since, until, kind or None, cid, minor))


@app.command("epochs")
def epochs_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    path: Path = typer.Argument(..., help="Output file: a GIF gives an animation, other image types a panel figure."),
    field: str = typer.Option("political", "--field", "-f", help="Culture field: political, polities, cultures, "
                                                                  "population, roads or settlements."),
    timeline: bool = typer.Option(False, "--timeline", "-t", help="Draw the history's people, levels and major "
                                                                   "chronicle entries against time instead."),
    since: Optional[float] = typer.Option(None, help="Timeline start, years ago (default: the history's start)."),
    fps: float = typer.Option(1.0, help="Frames per second of the animation."),
    panels: int = typer.Option(9, help="Largest number of panels in the figure."),
    projection_name: str = typer.Option("mollweide", "--projection", "-p", help=PROJECTION_HELP),
    lon: float = typer.Option(0.0, help="Central longitude of the maps (degrees)."),
) -> None:
    """Draw a saved world's organic history: a culture field at its kept epochs (panels or GIF), or a timeline."""
    import matplotlib

    matplotlib.use("Agg")
    from .render import animate_epochs, plot_chronicle, plot_epochs, save_figure

    world = load_world(world_dir)
    try:
        if timeline:
            save_figure(plot_chronicle(world, since=since), path)
        elif path.suffix.lower() == ".gif":
            animate_epochs(world, path, field, fps=fps, projection_name=projection_name, central_longitude=lon)
        else:
            save_figure(plot_epochs(world, field, panels=panels, projection_name=projection_name,
                                    central_longitude=lon), path)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"history written to {path}")


export_app = typer.Typer(help="Export heightfields, region bundles, engine heightmaps, tile pyramids and meshes.",
                         no_args_is_help=True)
app.add_typer(export_app, name="export")

RASTER_SUFFIX_HELP = "File type: .tif (GeoTIFF), .png (16-bit) or .raw (16-bit little-endian)."
REGION_RESOLUTION_HELP = "Lattice spacing in metres (default 750)."
RUGGEDNESS_HELP = "Sub-grid relief: earth, dramatic or gentle."
CACHE_HELP = "Keep zoomed regions in the world folder under regions/ and reuse them next time."
EXAGGERATION_HELP = "Vertical exaggeration of the mesh."
MAX_ERROR_HELP = "Simplification tolerance in metres (default: 5% of the node spacing)."


def _fail(error: Exception) -> None:
    """Report an error on stderr and stop with exit code 1."""
    typer.echo(f"error: {error.args[0] if error.args else error}", err=True)
    raise typer.Exit(code=1)


def _fields(value: str) -> list[str]:
    """Return the field names of a comma-separated option."""
    return [part for part in value.replace(" ", "").split(",") if part]


def _region(world: World, world_dir: Path, lat: float, lon: float, resolution: Optional[float],
            level: Optional[int], half_width: Optional[float], ruggedness: str, cache: bool):
    """Zoom into the region around a place of a saved world."""
    from . import heuristics as h
    from .world import REGIONS_FOLDER
    from .zoom import zoom_region

    try:
        return zoom_region(world, lat, lon, resolution_m=resolution, level=level,
                           half_width_km=half_width if half_width is not None else h.ZOOM_DEFAULT_HALF_WIDTH_KM,
                           ruggedness=ruggedness, cache_dir=world_dir / REGIONS_FOLDER if cache else None)
    except ValueError as e:
        _fail(e)


def _local_culture(world: World, region):
    """Return the settlements, roads and borders of a region, or None when the world has no culture."""
    if world.culture is None or "population_density" not in world.culture.layers:
        return None
    from .culture.local import zoom_culture

    return zoom_culture(world, region)


def _describe(region) -> str:
    """Return the size, spacing and level of a region."""
    rows, cols = region.grid.shape
    spacing = region.grid.mean_spacing(region.radius_m)
    return (f"{cols * spacing / 1e3:.0f} × {rows * spacing / 1e3:.0f} km at {spacing:.0f} m "
            f"(level {region.grid.level}, {cols} × {rows} nodes)")


@export_app.command("raster")
def export_raster_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    path: Path = typer.Argument(..., help=RASTER_SUFFIX_HELP),
    field: str = typer.Option("elevation", "--field", "-f", help="Field to export; see `worldgen export fields`."),
    width: int = typer.Option(2048, help="Width in pixels of the equirectangular raster (height is half)."),
    colour: bool = typer.Option(False, "--colour", help="Write the map colours (RGB) instead of values."),
    hillshade: bool = typer.Option(False, "--hillshade", help="Shade the colours by relief."),
    epoch: Optional[float] = typer.Option(None, "--epoch", "-e", help=EPOCH_HELP),
) -> None:
    """Write a global field of a saved world as an equirectangular raster with a JSON sidecar."""
    from .export import export_global

    try:
        out = export_global(load_world(world_dir), path, field, width, colour, hillshade, epoch)
    except (KeyError, ValueError) as e:
        _fail(e)
    typer.echo(f"{field} written to {out}")


@export_app.command("fields")
def export_fields_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
) -> None:
    """List the global fields a saved world can export, with their units and descriptions."""
    from .export import EXPORT_FIELDS, export_field

    world = load_world(world_dir)
    rows = []
    for name in EXPORT_FIELDS:
        try:
            f = export_field(world, name)
        except (KeyError, ValueError):
            continue
        rows.append((name, f.units or "-", f.description))
    widths = [max(len(r[i]) for r in rows) + 2 for i in (0, 1)]
    for name, units, description in rows:
        typer.echo(f"{name:<{widths[0]}}{units:<{widths[1]}}{description}")


@export_app.command("region")
def export_region_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    folder: Path = typer.Argument(..., file_okay=False, help="Output folder for the bundle."),
    lat: float = typer.Option(..., help="Latitude of the region's centre (degrees)."),
    lon: float = typer.Option(..., help="Longitude of the region's centre (degrees)."),
    resolution: Optional[float] = typer.Option(None, "--resolution", "-r", help=REGION_RESOLUTION_HELP),
    level: Optional[int] = typer.Option(None, help="Quadtree level instead of a resolution."),
    half_width: Optional[float] = typer.Option(None, "--half-width", help="Half-width of the region in km "
                                                                          "(default 75)."),
    ruggedness: str = typer.Option("earth", help=RUGGEDNESS_HELP),
    fields: str = typer.Option("elevation,ocean,lake,rivers,biomes", help="Comma-separated region fields."),
    suffix: str = typer.Option(".tif", "--format", help=RASTER_SUFFIX_HELP),
    crs: str = typer.Option("native", help="native (the zoom lattice, georeferenced by control points) or laea "
                                           "(resampled onto a Lambert azimuthal equal-area grid)."),
    culture: bool = typer.Option(True, "--culture/--no-culture", help="Add culture layers and GeoJSON vectors "
                                                                      "when the world has a culture."),
    cache: bool = typer.Option(False, "--cache", help=CACHE_HELP),
    epoch: Optional[float] = typer.Option(None, "--epoch", "-e", help=EPOCH_HELP),
) -> None:
    """Zoom into a saved world and write the region's fields, colours and vectors into a folder."""
    from .export import export_region_bundle

    world = _at_epoch(load_world(world_dir), epoch)
    region = _region(world, world_dir, lat, lon, resolution, level, half_width, ruggedness, cache)
    local = _local_culture(world, region) if culture else None
    try:
        export_region_bundle(world, region, folder, _fields(fields), suffix, local, crs)
    except (KeyError, ValueError) as e:
        _fail(e)
    typer.echo(f"region {_describe(region)} written to {folder}")


@export_app.command("engine")
def export_engine_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    path: Path = typer.Argument(..., help="Heightmap file: .r16 or .raw (16-bit little-endian) or .png (16-bit)."),
    lat: float = typer.Option(..., help="Latitude of the heightmap's centre (degrees)."),
    lon: float = typer.Option(..., help="Longitude of the heightmap's centre (degrees)."),
    engine: str = typer.Option("unity", help="Game engine: unity (2ⁿ + 1 sizes) or unreal (k·126 + 1 sizes)."),
    size: int = typer.Option(513, help="Heightmap size in nodes; see the engine's sizes in the error message."),
    resolution: Optional[float] = typer.Option(None, "--resolution", "-r", help=REGION_RESOLUTION_HELP),
    level: Optional[int] = typer.Option(None, help="Quadtree level instead of a resolution."),
    ruggedness: str = typer.Option("earth", help=RUGGEDNESS_HELP),
    cache: bool = typer.Option(False, "--cache", help=CACHE_HELP),
) -> None:
    """Write a square heightmap sized for a game engine, with its import settings in a JSON sidecar."""
    from .export import engine_region, export_engine_heightmap
    from .world import REGIONS_FOLDER

    world = load_world(world_dir)
    try:
        region = engine_region(world, lat, lon, engine, size, resolution, level, ruggedness,
                               world_dir / REGIONS_FOLDER if cache else None)
        out = export_engine_heightmap(world, region, path, engine)
    except (KeyError, ValueError) as e:
        _fail(e)
    typer.echo(f"{engine} heightmap {_describe(region)} written to {out}")


def _places(values: list[str]) -> list[tuple[float, float, int]]:
    """Return the (latitude, longitude, finest level) of each --place option."""
    out = []
    for value in values:
        try:
            lat, lon, level = value.replace(" ", "").split(",")
            out.append((float(lat), float(lon), int(level)))
        except ValueError:
            raise typer.BadParameter(f"could not read '{value}' as latitude,longitude,level")
    return out


@export_app.command("pyramid")
def export_pyramid_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    folder: Path = typer.Argument(..., file_okay=False, help="Output folder for the tiles and tileset.json."),
    top_level: int = typer.Option(3, "--top-level", help="Finest global level, built one block per cube face "
                                                         "(level 3 takes ~1 min per face)."),
    place: list[str] = typer.Option([], "--place", help="Refine around a place down to a level: "
                                                        "'latitude,longitude,level'. Repeatable."),
    fields: str = typer.Option("elevation", help="Comma-separated tile fields: elevation, ocean, lake, rivers, "
                                                 "ice."),
    suffix: str = typer.Option(".png", "--format", help=RASTER_SUFFIX_HELP),
    meshes: bool = typer.Option(False, "--meshes", help="Also write a GLB mesh per tile under meshes/."),
    exaggeration: float = typer.Option(1.0, help=EXAGGERATION_HELP),
    ruggedness: str = typer.Option("earth", help=RUGGEDNESS_HELP),
    cache: bool = typer.Option(False, "--cache", help=CACHE_HELP),
) -> None:
    """Build the tile pyramid of a saved world (global levels and refined chains) and write its tiles."""
    import time

    from .export import build_pyramid, export_pyramid, export_pyramid_meshes
    from .world import REGIONS_FOLDER

    world = load_world(world_dir)
    start = time.perf_counter()
    try:
        pyramid = build_pyramid(world, _places(place), top_level, ruggedness,
                                world_dir / REGIONS_FOLDER if cache else None)
        export_pyramid(pyramid, world, folder, _fields(fields), suffix)
        if meshes:
            export_pyramid_meshes(pyramid, world, folder / "meshes", exaggeration=exaggeration)
    except (KeyError, ValueError) as e:
        _fail(e)
    tiles = sum(len(b.tiles) for blocks in pyramid.levels.values() for b in blocks)
    typer.echo(f"{tiles:,} tiles on levels {min(pyramid.levels)}–{max(pyramid.levels)} written to {folder} "
               f"in {time.perf_counter() - start:.0f} s")


@export_app.command("mesh")
def export_mesh_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    path: Path = typer.Argument(..., help="Mesh file: .glb, .gltf, .ply or .obj."),
    lat: Optional[float] = typer.Option(None, help="Latitude of a region to mesh (default: the whole globe)."),
    lon: Optional[float] = typer.Option(None, help="Longitude of a region to mesh."),
    level: Optional[int] = typer.Option(None, help="Quadtree level: of the globe's tiles (default 2) or of the "
                                                   "region."),
    field: Optional[str] = typer.Option(None, "--field", "-f", help="Vertex colours: a global field for the globe "
                                                                    "(default elevation); relief (default) or "
                                                                    "biome for a region."),
    exaggeration: float = typer.Option(1.0, help=EXAGGERATION_HELP),
    max_error: Optional[float] = typer.Option(None, "--max-error", help=MAX_ERROR_HELP),
    sea: bool = typer.Option(True, "--sea/--no-sea", help="Add a sea-level layer over the ocean."),
    resolution: Optional[float] = typer.Option(None, "--resolution", "-r", help=REGION_RESOLUTION_HELP),
    half_width: Optional[float] = typer.Option(None, "--half-width", help="Half-width of the region in km "
                                                                          "(default 75)."),
    ruggedness: str = typer.Option("earth", help=RUGGEDNESS_HELP),
    cache: bool = typer.Option(False, "--cache", help=CACHE_HELP),
    epoch: Optional[float] = typer.Option(None, "--epoch", "-e", help=EPOCH_HELP),
) -> None:
    """Write a mesh of the globe or of the region around a place, with vertex colours."""
    from .export import export_globe_mesh, export_region_mesh

    if (lat is None) != (lon is None):
        _fail(ValueError("give both --lat and --lon for a region, or neither for the globe"))
    world = _at_epoch(load_world(world_dir), epoch)
    try:
        if lat is None:
            out = export_globe_mesh(world, path, 2 if level is None else level, field or "elevation", max_error,
                                    exaggeration, sea)
            what = f"globe at level {2 if level is None else level}"
        else:
            if field not in (None, "relief", "biome"):
                _fail(ValueError(f"unknown region colouring '{field}'; choose relief or biome"))
            region = _region(world, world_dir, lat, lon, resolution, level, half_width, ruggedness, cache)
            out = export_region_mesh(world, region, path, field or "relief", max_error, exaggeration, sea)
            what = f"region {_describe(region)}"
    except (KeyError, ValueError) as e:
        _fail(e)
    typer.echo(f"{what} written to {out}")


@app.command("globe")
def globe_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    out: Optional[Path] = typer.Argument(None, help="An image (.png), an interactive page (.html), or a folder for "
                                                    "every field and epoch asked for; without it a window opens."),
    field: str = typer.Option("elevation", "--field", "-f", help="Field to show; comma-separated for a folder."),
    epoch: Optional[float] = typer.Option(None, "--epoch", "-e", help=EPOCH_HELP),
    all_epochs: bool = typer.Option(False, "--all-epochs", help="Write every kept epoch into the folder."),
    level: int = typer.Option(1, help="Quadtree level of the globe's tiles (1 is ~50 km at the Earth's size)."),
    exaggeration: Optional[float] = typer.Option(None, help="Vertical exaggeration (default 20; 3 for a giant's "
                                                            "cloud tops)."),
    lat: float = typer.Option(20.0, help="Latitude the camera looks down on (degrees)."),
    lon: float = typer.Option(0.0, help="Longitude the camera looks down on (degrees)."),
    region: list[str] = typer.Option([], "--region", help="Drape a zoomed region over the globe: "
                                                          "'latitude,longitude'. Repeatable."),
    fmt: str = typer.Option("png", "--format", help="File type in a folder: png or html."),
) -> None:
    """Show a saved world as a 3D globe in a window, or write images or interactive pages of it."""
    from .render import GlobeViewer

    world = load_world(world_dir)
    try:
        viewer = GlobeViewer(world, level, exaggeration=exaggeration)
        for value in region:
            try:
                rlat, rlon = (float(v) for v in value.replace(" ", "").split(","))
            except ValueError:
                raise typer.BadParameter(f"could not read '{value}' as latitude,longitude")
            viewer.add_region(_region(world, world_dir, rlat, rlon, None, None, None, "earth", False))
        fields = ["clouds" if world.cloud_tops and f == "elevation" else f for f in _fields(field)]
        missing = [f for f in fields if f not in viewer.fields]
        if missing:
            _fail(ValueError(f"'{missing[0]}' is not available; choose from {', '.join(viewer.fields)}"))
        if out is None:
            viewer.show(fields[0], epoch, lat, lon)
            return
        suffix = out.suffix.lower()
        if suffix == ".png":
            viewer.screenshot(out, fields[0], epoch, lat, lon)
        elif suffix == ".html":
            viewer.save_html(out, fields[0], epoch, lat, lon)
        elif suffix:
            _fail(ValueError(f"unknown output type '{suffix}'; use .png, .html or a folder"))
        else:
            if fmt not in ("png", "html"):
                _fail(ValueError(f"unknown format '{fmt}'; use png or html"))
            epochs = None if all_epochs else [viewer.epoch if epoch is None else epoch]
            files = viewer.export_views(out, fields, epochs, fmt, lat, lon)
            typer.echo(f"{len(files)} views written to {out}")
            return
    except (KeyError, ValueError, ImportError) as e:
        _fail(e)
    typer.echo(f"globe written to {out}")


def _system_world(world: World, world_dir: Path, three_d: bool):
    """Return the world whose system to draw, its folder, the moon to highlight and given moon worlds.

    A planet draws its own system. A moon kept in a planet's folder draws its
    parent's; a standalone moon draws its parent rebuilt from its parent
    section with only this moon.
    """
    from .giants import parent_folder, rebuilt_parent_spec

    state = world.state
    if state.system is not None:
        return world, world_dir, None, {}
    if state.parent is None:
        raise ValueError(f"{state.name} has no moons, rings or parent planet to draw")
    planet_dir = parent_folder(world_dir)
    if planet_dir is not None:
        return load_world(planet_dir), planet_dir, state.name, {state.name: world}
    typer.echo(f"{state.name} has no saved parent: drawing {state.parent.name or 'its parent'} rebuilt from its "
               "parent section, without rings or other moons")
    spec = rebuilt_parent_spec(state, world.spec)
    parent = generate_world(spec, "preview") if three_d else generate(spec)[0]
    return parent, None, state.name, {state.name: world}


@app.command("system")
def system_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save (a planet "
                                                                             "or a moon)."),
    out: Optional[Path] = typer.Argument(None, help="Diagram: .png, .svg or .pdf; with --3d: .png, .html, .mp4 or "
                                                    ".gif. Without it a window opens."),
    three_d: bool = typer.Option(False, "--3d", help="The 3D scene (planet, rings and moons lit by the star, with "
                                                     "shadows) instead of the diagram."),
    view: str = typer.Option("top", help="Diagram view: top (down the spin axis) or edge (edge-on)."),
    linear: bool = typer.Option(False, help="Diagram: linear distances instead of the logarithmic axis."),
    highlight: Optional[str] = typer.Option(None, help="Diagram: a moon to mark (a moon's folder marks itself)."),
    scale: str = typer.Option("compressed", help="3D: compressed (true out past the rings, logarithmic beyond, "
                                                 "moons enlarged) or true."),
    season: float = typer.Option(60.0, help="3D: the star's angle along the planet's orbit (degrees; 0 and 180 put "
                                            "both poles equally lit)."),
    time: float = typer.Option(0.0, "--time", help="Time in days: where the moons are."),
    duration: Optional[float] = typer.Option(None, help="3D movie: span in days (default one orbit of the "
                                                        "outermost moon)."),
    frames: int = typer.Option(120, help="3D movie: number of frames."),
    level: int = typer.Option(1, help="3D: quadtree level of the planet's globe."),
) -> None:
    """Draw a planet's moons, rings and limits as a diagram, or show its system in 3D."""
    import math

    world = load_world(world_dir)
    try:
        planet, folder, marked, moon_worlds = _system_world(world, world_dir, three_d)
        highlight = highlight or marked
        if not three_d:
            import matplotlib

            if out is not None:
                matplotlib.use("Agg")
            from .render.system import plot_system

            if view not in ("top", "edge"):
                raise ValueError(f"unknown view '{view}'; use top or edge")
            if out is not None and out.suffix.lower() not in (".png", ".svg", ".pdf"):
                raise ValueError(f"unknown output type '{out.suffix}'; use .png, .svg or .pdf")
            plot_system(planet, view, "linear" if linear else "log", highlight, moon_worlds, path=out,
                        time_s=time * 86400.0)
            if out is None:
                import matplotlib.pyplot as plt

                plt.show()
                return
            typer.echo(f"system diagram written to {out}")
            return
        from .render.system_scene import system_scene

        scene = system_scene(planet, scale, math.radians(season), level, moon_worlds, folder)
        if out is None:
            scene.show(time * 86400.0)
            return
        suffix = out.suffix.lower()
        if suffix == ".png":
            scene.screenshot(out, time * 86400.0)
        elif suffix == ".html":
            scene.save_html(out, time * 86400.0)
        elif suffix in (".mp4", ".gif"):
            scene.save_movie(out, None if duration is None else duration * 86400.0, frames)
        else:
            raise ValueError(f"unknown output type '{suffix}'; use .png, .html, .mp4 or .gif")
        typer.echo(f"system scene written to {out}")
    except (KeyError, ValueError, ImportError) as e:
        _fail(e)


@app.command("moon")
def moon_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="A planet's folder written by --save."),
    moon: Optional[str] = typer.Argument(None, help="The moon's name or number (1 is the innermost); without it the "
                                                    "moons are listed."),
    resolution: str = typer.Option("standard", "--resolution", "-r", help=RESOLUTION_HELP),
    map_path: Optional[Path] = typer.Option(None, "--map", "-m", help=MAP_HELP),
    field: str = typer.Option("overview", "--field", "-f", help=FIELD_HELP),
    projection_name: str = typer.Option("robinson", "--projection", "-p", help=PROJECTION_HELP),
    keep: bool = typer.Option(True, help="Keep the moon's world in the planet's folder (moons/NN_name), where the "
                                         "report, the 3D scene and later runs find it."),
) -> None:
    """Build a planet's moon as a world (or load it if kept), print its report, and keep it with the planet."""
    from .giants import kept_moons, moon_folder, moon_index, moon_world

    world = load_world(world_dir)
    sy = world.state.system
    if sy is None or not sy.moons:
        _fail(ValueError(f"{world.state.name} has no moons"))
    if moon is None:
        kept = kept_moons(world_dir, world)
        for k, m in enumerate(sy.moons, 1):
            typer.echo(f"{k:>3}  {m.name:<16}{m.mass_kg / M_EARTH:>10.3g} M⊕  "
                       f"{m.semi_major_axis_m / world.state.bulk.radius_m:>7.1f} R_p  {m.origin:<9}"
                       + ("  kept" if m.name in kept else ""))
        return
    try:
        index = moon_index(world, moon)
        built = moon_world(world, index, _resolution(resolution), world_dir if keep else None)
    except (KeyError, IndexError, ValueError, NotImplementedError) as e:
        _fail(e)
    typer.echo(format_report(built.state))
    if keep:
        typer.echo(f"\nmoon kept in {moon_folder(world_dir, world, index)}")
    if map_path is not None:
        _write_map(built, map_path, field, projection_name, 0.0, 0.0)



@app.command("archetypes")
def archetypes_cmd() -> None:
    """List the available archetypes."""
    for a in ARCHETYPES.values():
        typer.echo(f"{a.name:<16}{a.description}")


@app.command("template")
def template_cmd(
    path: Path = typer.Argument(Path("planet.yaml"), help="Where to write the annotated template."),
    force: bool = typer.Option(False, help="Overwrite an existing file."),
) -> None:
    """Write an annotated example spec file."""
    if path.exists() and not force:
        typer.echo(f"{path} exists; use --force to overwrite", err=True)
        raise typer.Exit(code=1)
    path.write_text(SPEC_TEMPLATE, encoding="utf-8")
    typer.echo(f"template written to {path}")


if __name__ == "__main__":
    app()
