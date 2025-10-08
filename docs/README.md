## Multi-Scale FEM Validation & Residual Analysis Dataset

This repository contains a fabricated, but structurally realistic, dataset for validating a thermo-mechanical FEM model and performing residual analysis. Data are organized across macro-, meso-, micro-scales and simulation outputs.

### Directory Layout

- `/workspace/data/macro/processed`
  - `material_properties.json`: bulk properties (E, ν, CTE, density, k, Cp)
  - `cell_dimensions.json`: sample dimensions and coordinate frame
  - `sintering_profile.csv`: time-temperature profile (ramp–soak–cool)
  - `residual_stress_surface_XRD.csv`: surface residual stresses (XRD/Raman-like)
  - `residual_stress_bulk_synchrotron.csv`: sub-surface/bulk stresses (synchrotron-like)
- `/workspace/data/meso/processed`
  - `microcrack_locations_SEM.csv`: micro-crack positions and features from cross-section SEM
  - `fracture_tests.csv`: loading/thermal test matrix with crack occurrence
  - `critical_thresholds.json`: fabricated critical temperature/load for cracking
  - `grain_size_distribution.csv`: RVE grain size log-normal distribution
  - `porosity_map.csv`: spatial porosity fraction map
- `/workspace/data/micro/processed`
  - `ebsd_map.csv`: per-grain positions and Euler angles (φ1, Φ, φ2)
  - `grain_boundary_properties.csv`: per-grain GB strength/energy/diffusivity
- `/workspace/data/simulation/processed`
  - `full_field.csv`: reference full-field solution (T, U, σ, ε) at nodes
  - `collocation_subset.csv`: strategically sampled points near edges/surfaces
- `/workspace/data/manifest.json`: list of all produced files and seed

### Schemas

- `sintering_profile.csv`
  - `time_min, temperature_C`
- `residual_stress_surface_XRD.csv`
  - `x_mm, y_mm, z_mm, method, sigma_xx_MPa, sigma_yy_MPa, sigma_xy_MPa, uncertainty_MPa`
- `residual_stress_bulk_synchrotron.csv`
  - `x_mm, y_mm, z_mm, method, sigma_xx_MPa, sigma_yy_MPa, sigma_zz_MPa, sigma_xy_MPa, sigma_yz_MPa, sigma_xz_MPa`
- `microcrack_locations_SEM.csv`
  - `crack_id, x_um, y_um, feature, orientation_deg, length_um`
- `fracture_tests.csv`
  - `test_id, max_temperature_C, applied_load_MPa, heating_rate_C_per_min, cracked, crack_temperature_C`
- `grain_size_distribution.csv`
  - `grain_id, equivalent_diameter_um`
- `porosity_map.csv`
  - `x_um, y_um, porosity_fraction`
- `ebsd_map.csv`
  - `grain_id, x_um, y_um, phi1_deg, Phi_deg, phi2_deg`
- `grain_boundary_properties.csv`
  - `grain_id, gb_strength_MPa, gb_energy_J_per_m2, gb_diffusivity_m2_per_s`
- `full_field.csv` and `collocation_subset.csv`
  - `node_id, (feature_tag for collocation), x_mm, y_mm, z_mm, T_C, Ux_um, Uy_um, Uz_um, sigma_xx_MPa, sigma_yy_MPa, sigma_zz_MPa, sigma_xy_MPa, sigma_yz_MPa, sigma_xz_MPa, epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xy, epsilon_yz, epsilon_xz`

### Usage

Generate datasets (reproducible with a seed):

```bash
python3 /workspace/scripts/generate_dataset.py --seed 123
```

### Notes and Assumptions

- Values are fabricated but constrained to plausible ranges for ceramic electrolytes.
- Residual stresses feature compressive centers and relax toward edges; bulk data introduce through-thickness decay.
- Simulation "full_field" is a synthetic reference solution with thermal gradients and isotropic elasticity approximations.
- Collocation points prefer free edges and free surfaces to capture stress concentrations.
- Microstructural statistics (grain size, porosity) follow smooth fields with noise; EBSD orientations are random.

