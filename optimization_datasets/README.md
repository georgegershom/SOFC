# 📐 Optimization and Validation Dataset

**Generated for Inverse Modeling and PSO-Based Defect Identification**

---

## 📋 Overview

This comprehensive dataset collection was fabricated for research in optimization and validation of ceramic sintering processes, defect identification, and geometric design evaluation. The data supports:

- **FEM vs Experimental Validation**: Stress/strain profile comparisons
- **Defect Detection**: Synchrotron XRD measurements vs PSO-based inverse modeling
- **Process Optimization**: Sintering parameter identification
- **Design Optimization**: Geometric channel design variations

---

## 📊 Dataset Contents

### 1. **Stress/Strain Profiles** (`stress_strain_profiles.csv`)
**5,000 data points across 100 samples**

Compares Finite Element Method (FEM) predictions with experimental synchrotron X-ray diffraction (XRD) measurements.

#### Key Columns:
- `sample_id`: Unique sample identifier
- `position_mm`: Measurement position along sample (0-100 mm)
- `fem_stress_MPa`: FEM-predicted stress
- `experimental_stress_MPa`: XRD-measured stress
- `fem_strain_microstrain`: FEM-predicted strain
- `experimental_strain_microstrain`: XRD-measured strain
- `stress_residual_MPa`: Absolute difference (FEM vs Experimental)
- `strain_residual_microstrain`: Strain residual
- `youngs_modulus_GPa`: Young's modulus (180-220 GPa)
- `poisson_ratio`: Poisson's ratio (0.28-0.32)
- `measurement_method`: Always "synchrotron_XRD"
- `fem_mesh_size_mm`: FEM mesh resolution (0.5, 1.0, or 2.0 mm)
- `convergence_achieved`: Boolean indicating FEM convergence

#### Statistics:
- Average stress residual: **6.57 MPa**
- Stress range: -100 to +100 MPa
- Realistic sinusoidal residual stress patterns with measurement noise

---

### 2. **Crack Depth Estimates** (`crack_depth_estimates.csv`)
**200 samples with crack measurements**

Compares synchrotron XRD crack depth measurements with PSO inverse model predictions.

#### Key Columns:
- `sample_id`: Unique sample identifier
- `xrd_crack_depth_mm`: XRD measured crack depth
- `pso_predicted_depth_mm`: PSO model prediction
- `true_crack_depth_mm`: Ground truth (for validation)
- `xrd_measurement_error_mm`: XRD measurement error
- `pso_prediction_error_mm`: PSO prediction error
- `xrd_pso_difference_mm`: Difference between XRD and PSO
- `pso_iterations`: Number of PSO iterations (50-200)
- `pso_convergence_rate`: Convergence rate (0.85-0.99)
- `fitness_value`: PSO fitness function value (lower = better)
- `crack_type`: "surface", "subsurface", or "through-thickness"
- `crack_orientation_deg`: Crack angle (0-90°)
- `crack_width_mm`: Crack opening (0.01-0.5 mm)
- `x_position_mm`, `y_position_mm`: Crack location
- `stress_intensity_factor_MPa_sqrtm`: Stress intensity factor
- `detection_confidence`: Detection confidence (0.7-0.99)
- `material`: Al2O3, ZrO2, Si3N4, or SiC
- `sintering_temperature_C`: Sintering temperature (1400-1700°C)

#### Statistics:
- Average XRD measurement error: **~0.15 mm**
- Average PSO prediction error: **0.070 mm**
- Crack depth range: 0.1-5.0 mm
- PSO demonstrates better accuracy than direct XRD measurement

---

### 3. **Sintering Parameters** (`sintering_parameters.csv`)
**150 experimental runs**

Optimal sintering parameter identification with emphasis on **1-2°C/min cooling rate**.

#### Key Columns:
- `experiment_id`: Unique experiment identifier
- `cooling_rate_C_per_min`: **Critical parameter** (0.5-5.0°C/min)
- `heating_rate_C_per_min`: Ramp-up rate (2.0-10.0°C/min)
- `hold_temperature_C`: Sintering temperature (1400-1700°C)
- `hold_time_hours`: Dwell time (1-8 hours)
- `atmosphere`: "air", "argon", "vacuum", or "nitrogen"
- `pressure_kPa`: Processing pressure (0.001-101.325 kPa)
- `density_percent_theoretical`: Resulting density (% theoretical)
- `grain_size_um`: Average grain size (μm)
- `porosity_percent`: Residual porosity (%)
- `flexural_strength_MPa`: Three-point bend strength
- `hardness_HV`: Vickers hardness
- `fracture_toughness_MPa_sqrtm`: K_IC value
- `crack_density_per_cm2`: Defect density
- `residual_stress_MPa`: Internal stress
- `warpage_mm`: Geometric distortion
- `total_cycle_time_hours`: Processing duration
- `energy_cost_relative`: Energy consumption (relative units)
- `quality_score`: Overall quality metric (0-100)
- `optimal_range_cooling`: Boolean flag for 1-2°C/min range
- `material`: Ceramic material type
- `green_density_percent`: Pre-sintered density (50-65%)

#### Key Insights:
- **Optimal cooling rate: 1-2°C/min** (37 out of 150 experiments)
- Optimal cooling minimizes crack density and residual stress
- Higher quality scores correlate with optimal cooling rates
- Faster cooling (>3°C/min) increases defect density

---

### 4. **Geometric Designs** (`geometric_designs.csv`)
**100 design variations**

Comparison of different channel geometries: bow-shaped, rectangular, trapezoidal, and circular.

#### Key Columns:
- `design_id`: Unique design identifier
- `design_type`: "bow_shaped", "rectangular", "trapezoidal", or "circular"
- `length_mm`, `width_mm`, `height_mm`: Basic dimensions
- `wall_thickness_mm`: Wall thickness (1-5 mm)
- `channel_count`: Number of channels
- `design_specific_parameter`: 
  - Bow: curvature_radius
  - Rectangular: aspect_ratio
  - Trapezoidal: taper_angle
  - Circular: diameter
- `volume_mm3`: Total volume
- `surface_area_mm2`: Surface area
- `hydraulic_diameter_mm`: Hydraulic diameter
- `flow_rate_L_per_min`: Flow rate (0.1-10.0 L/min)
- `pressure_drop_kPa`: Pressure drop (0.1-5.0 kPa)
- `stress_concentration_factor`: Stress concentration (1.0-3.5)
- `max_von_mises_stress_MPa`: Maximum stress
- `max_displacement_mm`: Maximum deformation
- `max_strain_microstrain`: Maximum strain
- `thermal_performance_score`: Heat transfer performance (0-100)
- `max_temperature_C`: Peak temperature (100-600°C)
- `thermal_gradient_C_per_mm`: Temperature gradient
- `heat_transfer_coefficient_W_m2K`: Heat transfer coefficient
- `manufacturing_difficulty_score`: Fabrication complexity (1-10)
- `manufacturing_cost_relative`: Relative cost
- `warpage_risk_mm`: Distortion susceptibility
- `crack_risk_score`: Cracking susceptibility
- `weight_kg`: Component weight
- `efficiency_score`: Overall performance metric
- `material`: Ceramic material
- `sintering_temperature_C`: Processing temperature

#### Key Insights:
- **Circular designs**: Best thermal performance, lowest stress concentration
- **Bow-shaped designs**: Good balance of performance and stress reduction
- **Rectangular designs**: Highest stress concentration (corners), easier manufacturing
- **Trapezoidal designs**: Moderate stress, good flow characteristics

---

### 5. **PSO Optimization History** (`pso_optimization_history.csv`)
**1,260 data points from 50 optimization runs**

Particle Swarm Optimization convergence history for inverse defect identification.

#### Key Columns:
- `run_id`: Unique optimization run identifier
- `iteration`: Iteration number (0 to max_iterations, step=10)
- `n_particles`: Swarm size (20, 30, 50, or 100)
- `inertia_weight`: PSO inertia (w = 0.4-0.9)
- `cognitive_param_c1`: Personal best influence (1.5-2.5)
- `social_param_c2`: Global best influence (1.5-2.5)
- `best_fitness`: Best fitness in swarm (lower = better)
- `average_fitness`: Mean fitness across particles
- `worst_fitness`: Worst fitness in swarm
- `swarm_diversity`: Population diversity metric (0-1)
- `best_crack_depth_mm`: Best crack depth estimate
- `best_crack_width_mm`: Best crack width estimate
- `best_crack_angle_deg`: Best crack orientation estimate
- `convergence_achieved`: Boolean flag (fitness < 0.05)
- `computation_time_sec`: Iteration computation time

#### Key Insights:
- Exponential convergence with rate 0.85-0.99
- Swarm diversity decreases over iterations (exploration → exploitation)
- Larger swarms (100 particles) show better final fitness
- Typical convergence within 100-200 iterations

---

## 🎯 Recommended Use Cases

### 1. **Inverse Modeling Validation**
Use `stress_strain_profiles.csv` and `crack_depth_estimates.csv` to:
- Train inverse FEM models
- Validate PSO-based parameter identification
- Assess measurement uncertainty

### 2. **Process Optimization**
Use `sintering_parameters.csv` to:
- Identify optimal sintering windows
- Correlate processing parameters with quality metrics
- Minimize defects and residual stress
- **Target: 1-2°C/min cooling rate**

### 3. **Design Optimization**
Use `geometric_designs.csv` to:
- Compare channel geometries
- Multi-objective optimization (stress, thermal, manufacturing)
- Pareto frontier analysis
- Trade-off studies

### 4. **Algorithm Development**
Use `pso_optimization_history.csv` to:
- Benchmark PSO variants
- Test convergence criteria
- Optimize hyperparameters (w, c1, c2)
- Compare with other metaheuristics

---

## 📈 Quick Statistics

| Dataset | Rows | Samples | Key Metric |
|---------|------|---------|------------|
| Stress/Strain | 5,000 | 100 | 6.57 MPa avg residual |
| Crack Depth | 200 | 200 | 0.070 mm PSO error |
| Sintering | 150 | 150 | 37 optimal experiments |
| Geometric | 100 | 100 | Circular = best efficiency |
| PSO History | 1,260 | 50 runs | ~150 iterations to converge |

---

## 🔬 Data Generation Methodology

All data was synthesized using:
- **Physics-based models**: Realistic stress/strain distributions
- **Measurement noise**: Gaussian noise simulating experimental uncertainty
- **Domain constraints**: Materials science limits (E, ν, temperature, etc.)
- **Optimization dynamics**: Exponential PSO convergence patterns
- **Random seed**: 42 (for reproducibility)

### Validation Characteristics:
- XRD measurement error: σ ≈ 0.15 mm
- Stress measurement noise: σ ≈ 8 MPa
- Strain measurement noise: σ ≈ 50 μɛ
- PSO convergence: 85-99% per iteration decay

---

## 🛠️ Usage Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
stress_strain = pd.read_csv('stress_strain_profiles.csv')
crack_depth = pd.read_csv('crack_depth_estimates.csv')
sintering = pd.read_csv('sintering_parameters.csv')
geometric = pd.read_csv('geometric_designs.csv')
pso_history = pd.read_csv('pso_optimization_history.csv')

# Example 1: Find optimal sintering conditions
optimal = sintering[sintering['optimal_range_cooling'] == True]
print(f"Average quality score (optimal): {optimal['quality_score'].mean():.2f}")

# Example 2: Compare XRD vs PSO accuracy
import numpy as np
xrd_rmse = np.sqrt((crack_depth['xrd_measurement_error_mm']**2).mean())
pso_rmse = np.sqrt((crack_depth['pso_prediction_error_mm']**2).mean())
print(f"XRD RMSE: {xrd_rmse:.3f} mm")
print(f"PSO RMSE: {pso_rmse:.3f} mm")

# Example 3: Best geometric design
best_design = geometric.loc[geometric['efficiency_score'].idxmax()]
print(f"Best design: {best_design['design_type']}")
```

---

## 📊 Visualizations

Run `visualize_datasets.py` to generate:
1. `stress_strain_analysis.png` - FEM vs Experimental comparison
2. `crack_depth_analysis.png` - XRD vs PSO accuracy
3. `sintering_optimization.png` - Parameter optimization results
4. `geometric_designs.png` - Design performance comparison
5. `pso_convergence.png` - Optimization convergence curves

---

## 📝 Citation

If you use this dataset, please cite:
```
Optimization and Validation Dataset for Inverse Modeling and PSO-Based 
Defect Identification in Ceramic Sintering Processes. 
Generated: 2025-10-03
```

---

## 🔗 Files Included

```
optimization_datasets/
├── stress_strain_profiles.csv       (5000 rows)
├── crack_depth_estimates.csv        (200 rows)
├── sintering_parameters.csv         (150 rows)
├── geometric_designs.csv            (100 rows)
├── pso_optimization_history.csv     (1260 rows)
├── dataset_summary.json             (metadata)
├── stress_strain_analysis.png       (visualization)
├── crack_depth_analysis.png         (visualization)
├── sintering_optimization.png       (visualization)
├── geometric_designs.png            (visualization)
├── pso_convergence.png              (visualization)
└── README.md                        (this file)
```

---

## 🎓 Key Findings Summary

### ✅ Validation Results
- **FEM vs Experimental**: Average stress residual **6.57 MPa** (good agreement)
- **PSO Accuracy**: Outperforms direct XRD measurement (0.070 vs 0.15 mm error)
- **Convergence**: PSO typically converges in **100-200 iterations**

### ✅ Optimization Results
- **Optimal Cooling Rate**: **1-2°C/min** minimizes defects
- **Quality Impact**: 15-20% improvement within optimal range
- **Crack Density**: Reduced by ~60% at optimal cooling rates

### ✅ Design Results
- **Best Geometry**: Circular channels (lowest stress concentration)
- **Bow-shaped**: 20% better than rectangular for stress reduction
- **Trade-off**: Circular = high performance, rectangular = easy manufacturing

---

## 📧 Contact & Support

For questions about this dataset or custom data generation, please refer to the generation scripts:
- `optimization_validation_dataset.py` - Main data generator
- `visualize_datasets.py` - Visualization tools

**Dataset Version**: 1.0  
**Last Updated**: October 3, 2025
