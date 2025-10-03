
# Optimization & Validation Dataset (Synthetic)

This directory contains four synthetic CSV datasets for inverse modeling and PSO-based defect identification workflows. Values are generated with realistic trends and controlled noise for benchmarking and method development.

## Files
- fem_vs_experimental_stress_strain.csv
- crack_depth_xrd_vs_model.csv
- sintering_parameters_optimization.csv
- geometric_design_variations.csv

## 1) fem_vs_experimental_stress_strain.csv
Columns:
- specimen_id: identifier of the tested sample (e.g., AL-A)
- material: Al2O3 or ZrO2-3Y
- temperature_C: test temperature (25 C)
- strain_percent: engineering strain (%) from 0.0 to 2.0
- strain_rate_per_s: fixed nominal rate (1e-3 1/s)
- stress_exp_MPa: experimental stress with noise and specimen bias
- stress_fem_MPa: FEM-predicted stress with slight nonlinearity
- error_MPa: stress_fem_MPa - stress_exp_MPa
- exp_uncertainty_MPa: estimated experimental sigma used in synthesis

## 2) crack_depth_xrd_vs_model.csv
Columns:
- sample_id: S01..S20
- region: edge or midspan
- x_mm, y_mm: scan coordinates along the surface (y fixed 0 here)
- crack_depth_xrd_um: synchrotron XRD-inferred crack depth (um)
- crack_depth_model_um: model-predicted crack depth (um)
- residual_MPa: residual stress trend used to modulate features

## 3) sintering_parameters_optimization.csv
Experimental design space and responses. True optimum shaped near cooling_rate ≈ 1.5 C/min, hold_time ≈ 2 h, peak_temp ≈ 1450 C.
Columns:
- cooling_rate_C_per_min
- hold_time_hr
- peak_temp_C
- porosity_percent (lower is better)
- relative_density_percent
- flexural_strength_MPa (higher is better)

## 4) geometric_design_variations.csv
Three design archetypes: bow_shaped, rectangular, tapered. At multiple flow rates, we provide pressure drop, stress, and displacement.
Columns:
- design (bow_shaped | rectangular | tapered)
- channel_height_mm, channel_width_mm, curvature_radius_mm
- pressure_drop_kPa
- max_principal_stress_MPa
- disp_peak_um

Notes:
- All datasets are synthetic. Use for algorithm development, screening, plotting, and PSO objective demonstrations.
- Random seeds fixed per file for reproducibility.
