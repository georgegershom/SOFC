## Synthetic Optimization and Validation Datasets

This folder contains four fabricated datasets to support inverse modeling and PSO-based defect identification workflows. All data are synthetic and safe for sharing.

### 1) `fem_vs_experimental_profiles.csv`
- **Purpose**: Compare FEM-predicted vs experimental stress/strain profiles along the gauge.
- **Schema**:
  - `specimen_id` (str): Sample identifier
  - `position_mm` (float): Position along gauge length in mm
  - `fem_stress_MPa` (float)
  - `fem_strain` (float)
  - `exp_stress_MPa` (float)
  - `exp_strain` (float)
  - `temperature_C` (int)
  - `loading_rate_MPa_per_s` (float)

### 2) `crack_depth_xrd_vs_model.csv`
- **Purpose**: Validate crack depth estimates from synchrotron XRD vs model predictions.
- **Schema**:
  - `sample_id` (str)
  - `region` (str): e.g., `notch_root`, `midspan`, `edge`, `weld_zone`
  - `xrd_crack_depth_um` (float)
  - `model_crack_depth_um` (float)
  - `xrd_confidence_0to1` (float)
  - `beam_energy_keV` (float)
  - `exposure_ms` (int)

### 3) `sintering_parameters_optima.csv`
- **Purpose**: Explore optimal sintering parameters; includes cooling rate constrained to 1–2 °C/min.
- **Schema**:
  - `experiment_id` (str)
  - `binder_ratio_pct` (float)
  - `peak_temp_C` (float)
  - `dwell_time_min` (float)
  - `cooling_rate_C_per_min` (float)
  - `atmosphere` (str): `air`, `argon`, `hydrogen`
  - `density_pct` (float)
  - `porosity_pct` (float)
  - `fracture_toughness_MPa_m05` (float)
  - `grain_size_um` (float)
  - `objective_score` (float): Composite score in [0,1]
  - `is_pso_selected` (int): 1 if selected by PSO criterion

### 4) `geometric_design_variations.csv`
- **Purpose**: Compare design geometries (bow-shaped vs rectangular vs serpentine) and flow/structural figures.
- **Schema**:
  - `design_id` (str)
  - `geometry_type` (str)
  - `channel_width_mm` (float)
  - `channel_height_mm` (float)
  - `curvature_1_per_mm` (float)
  - `length_mm` (float)
  - `pred_pressure_drop_Pa` (float)
  - `pred_stress_MPa` (float)
  - `meas_pressure_drop_Pa` (float)
  - `meas_stress_MPa` (float)

### Reproducibility
- Generator script: `/workspace/scripts/generate_synthetic_data.py`
- To regenerate:
```bash
/usr/bin/python3 /workspace/scripts/generate_synthetic_data.py
```

