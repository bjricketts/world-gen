"""Command-line interface: ``worldgen --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .generate import candidates, generate, generate_world
from .priors import ARCHETYPES
from .report import format_report, format_timeline, save_state
from .spec import SPEC_TEMPLATE, PlanetSpec, PriorSpec, load_spec
from .world import World, load_world, save_world

app = typer.Typer(help="Generate physically motivated planets for world building.", no_args_is_help=True)

RESOLUTION_HELP = "Grid resolution: preview (10k cells), standard (40k), high (160k), or a cell count."
MAP_HELP = "Write a map image to this file (PNG, JPG, PDF or SVG)."
PROJECTION_HELP = "Map projection: equirectangular, mollweide, robinson, orthographic, north_polar, south_polar."
FIELD_HELP = ("Map field: elevation, terrain, plates, crust_age, orogeny_age, temperature, rainfall, basins, "
              "biomes, koppen, ice, or 'overview' / 'climate' for a multi-panel figure.")
TECTONICS_HELP = "Plate tectonics: simulated (default) or heuristic (fast, no history)."
START_HELP = "Initial continents of the simulation: supercontinent or cratons (default: drawn per seed)."
DURATION_HELP = "Simulated tectonic time in Myr (default 400)."
SNAPSHOTS_HELP = ("Store the tectonic history every N Myr (rounded to 20 Myr steps) in the saved world; "
                  "simulated tectonics only.")
EPOCHS_HELP = "History mode: ages in Gyr to keep full states for, e.g. '0.5,1,2' (default: the spec's epochs)."
TOPIC_HELP = "History figure: climate, interior or life."


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
               central_longitude: float, central_latitude: float) -> None:
    """Draw the requested map of a world and write it to an image file."""
    import matplotlib

    matplotlib.use("Agg")
    from .render import plot_climate, plot_map, plot_overview, save_figure

    if world.surface is None:
        typer.echo(f"no map: {world.state.name} has no solid surface", err=True)
        return
    if field == "overview":
        fig = plot_overview(world)
    elif field == "climate":
        fig = plot_climate(world)
    else:
        fig = plot_map(world, field, projection_name, central_longitude=central_longitude,
                       central_latitude=central_latitude).figure
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
    typer.echo(format_report(state))
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
) -> None:
    """Draw a map of a saved world."""
    _write_map(load_world(world_dir), map_path, field, projection_name, lon, lat)


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


@app.command("report")
def report_cmd(
    world_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder written by --save."),
    timeline: bool = typer.Option(False, "--timeline", "-t", help="Print the full event log and epoch table instead."),
) -> None:
    """Print the report of a saved world."""
    world = load_world(world_dir)
    if not timeline:
        typer.echo(format_report(world.state))
        return
    if world.timeline is None or not world.timeline.events:
        typer.echo("no integrated history in this world; generate it with mode: history", err=True)
        raise typer.Exit(code=1)
    typer.echo(format_timeline(world.timeline, world.state))


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
