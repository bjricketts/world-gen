# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/) from 1.0.0:
the public interface is the command-line interface, the functions in `worldgen.__all__`, the spec file format
(`spec_version`) and the save formats (world format 8, culture format 3, region format 1). Older save formats stay
readable (worlds 4–8, cultures 1–3).

## [1.0.0] - 2026-10-31

First public release.

### Model

- Planet specifications from stellar, orbital and bulk inputs, with archetype priors and a record of where each
  value came from.
- Interior and tectonic regime, plate-tectonic simulation, hydrology and erosion, oceans, atmosphere and
  energy-balance climate with ice, biomes and relicts.
- Snapshot mode and history mode (integration from formation).
- Giants (appearance, colours from composition, clouds), rings and moons, including moon environments and
  habitable moons.
- Zoom into any region for local maps, with tiled detail, glaciers and local erosion.
- Culture: user-scoped civilisations (settlements, roads, polities, languages) and organic histories that emerge from
  a simulated past.
- Export: heightfields, rasters, region packages, meshes, and an interactive globe viewer.

### Interface

- 17 command-line commands under `worldgen`; the Python API in `worldgen.__all__`.
- Spec files are YAML with an optional `spec_version` (1). Files without it read as version 1.
- Documentation in `docs/`: `design.md`, `spec.md`, `open-points.md`, `references.md`.

### Known limits

Open items, with their measured impact, are in `docs/open-points.md`. Generated worlds depend on the versions of
numpy, scipy and other libraries; state the versions with any published world.
