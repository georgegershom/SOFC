# SOFC Microstructure-Informed Dataset (v1.0.0)

Synthetic dataset inspired by the study
"A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation".
The bundle links material properties, microstructure, morphology, and FEM-ready artefacts derived from one coherent stochastic microstructure.

## Layout

- `dataset_description.json`: dataset metadata including seed, voxel spacing, array shapes.
- `initial_state/`
  - `microCT_volume_unfired.npz`: multi-field volume (solid, porosity, grain ids, scalar contrast).
  - `SEM_cross_sections/*.png`: slice images mimicking SEM cross-sections.
- `post_mortem/`
  - `microCT_volume_sintered.npz`: post-sintering volume with volumetric strain & warp fields.
  - `SEM_EDS/`: pseudo SEM/EDS outputs and element map (`.npz`).
  - `dic_surface_displacement.npz`, `dic_displacement_magnitude.png`: synthetic surface DIC field.
- `material_properties/`
  - `thermo_mechanical_properties.csv`: temperature-dependent thermo-mechanical property table.
  - `constitutive_model_params.json`: Norton-Bailey creep, Kelvin-Voigt, elastic constants.
- `fem_ready/`
  - `microstructure_porosity.vtr`: porosity field (VTK Rectilinear Grid) for FEM mapping.
  - `initial_conditions_fields.npz`: temperature, residual stress, creep strain fields.
  - `thermal_creep_schedule.csv`: processing schedule with targeted creep activation.
  - `boundary_conditions.json`: symmetry, assembly load, reference DIC data.
- `analysis/`
  - `porosity_statistics.csv`: porosity comparison pre/post sintering.
  - `layer_thickness_map.csv`: layer thickness statistics inferred from SEM slices.
  - `validation_metrics.json`: warp, porosity reduction, peak creep strain, DIC metrics.
- `metadata/image_assets.json`: image metadata (capture mode, pixel size).

## Usage

1. **Material modelling**
   - Interpolate `thermo_mechanical_properties.csv` across operating temperatures.
   - Map `constitutive_model_params.json` directly into FEM material cards.
2. **Geometry & initial conditions**
   - Import `microstructure_porosity.vtr` as a microstructure-informed field or mesh generator input.
   - Apply `initial_conditions_fields.npz` for temperature, residual stress, and creep initialisation.
3. **Experimental comparison**
   - Use `SEM_cross_sections` and `SEM_EDS` for defect recognition or interface analysis benchmarks.
   - `dic_surface_displacement.npz` provides a DIC target for closed-loop calibration.
4. **Sintering workflow evaluation**
   - Follow `thermal_creep_schedule.csv` when simulating targeted creep activation strategies.

## Regeneration

Run `python3 scripts/generate_microstructure_dataset.py [--seed <int>]` to regenerate all artefacts. The script cleans the target folder before writing fresh outputs.

## Archive

The generator also produces `datasets/sofc_microstructure_informed_v1.zip` for easy distribution.

## License & attribution

This is synthetic data; you may reuse it freely for research or engineering validation. Please attribute the dataset name and generator script when sharing results.
