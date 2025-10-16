# Building DNA Synthetic Dataset Generator

This tool generates a synthetic dataset of building "static & fabric" data: geometry, construction materials, and systems, aligned with a dynamic digital twin framework for retrofit optimization.

## Contents
- Generator CLI: `python -m building_dna.generate`
- Output: `datasets/building_dna/output/<building_id>/...`
- Schemas: JSON Schema files for validation
- Templates: SVG/OBJ templates used for geometry

## Generated Artifacts per Building
- metadata.json
- geometry/
  - floorplan_level_<n>.svg
  - mass_model.obj
  - site_context.geojson
- lidar/
  - point_cloud.ply
  - point_cloud.csv
- fabric/
  - walls.json
  - roofs.json
  - floors.json
  - windows_doors.json
  - airtightness.json
- systems/
  - hvac.json
  - dhw.json
  - lighting.json
  - renewables.json

## Usage
```
python -m building_dna.generate --count 3 --seed 42
```
