# Multi-Scale FEM Validation Dataset for SOFC Electrolytes

## Overview

This comprehensive dataset provides experimental measurements, high-fidelity FEM simulation outputs, and multi-scale material characterization data for validating Finite Element Method models of Solid Oxide Fuel Cell (SOFC) electrolytes. The dataset is specifically designed for:

1. **FEM Model Validation** - Compare simulation predictions against experimental ground truth
2. **Residual Analysis** - Identify regions where models deviate from measurements
3. **Surrogate Modeling** - Train data-driven models (Neural Networks, Gaussian Processes, etc.)
4. **Multi-Scale Analysis** - Connect behavior across macro, meso, and micro scales

## Dataset Structure

```
fem_validation_dataset/
├── experimental_data/
│   ├── residual_stress/
│   │   ├── xrd_surface_residual_stress.csv         (42 measurements)
│   │   ├── raman_spectroscopy_stress.csv           (28 measurements)
│   │   └── synchrotron_xrd_subsurface.csv          (28 measurements)
│   └── crack_analysis/
│       ├── crack_initiation_data.csv               (19 cracks)
│       ├── crack_propagation_data.csv              (7 sequences, 34 time steps)
│       └── sem_fractography_observations.csv       (19 observations)
├── simulation_output/
│   ├── full_field/
│   │   ├── fem_full_field_solution.csv             (41 nodes, T/U/σ/ε)
│   │   └── fem_element_data.csv                    (38 elements)
│   └── collocation_points/
│       ├── collocation_point_data.csv              (30 strategic points)
│       └── collocation_point_metadata.json
├── multi_scale_data/
│   ├── macro_scale/
│   │   ├── bulk_material_properties.csv            (Temperature-dependent)
│   │   ├── cell_geometry.csv                       (6 cell configurations)
│   │   └── sintering_profile.csv                   (20 time steps)
│   ├── meso_scale/
│   │   ├── microstructure_characterization.csv     (15 ROIs)
│   │   └── rve_geometry_data.csv                   (8 RVEs)
│   └── micro_scale/
│       ├── grain_boundary_properties.csv           (12 GBs)
│       └── local_crystallographic_orientation.csv  (25 points)
└── metadata/
    └── dataset_summary.json
```

## Data Categories

### 1. Experimental Data - Residual Stress (Ground Truth)

This is your **validation target** - the experimental measurements against which your FEM predictions will be compared.

#### a) X-ray Diffraction (XRD) Surface Measurements
- **File**: `experimental_data/residual_stress/xrd_surface_residual_stress.csv`
- **Points**: 42 surface measurements
- **Coverage**: Grid pattern across sample surface (0-10 mm × 0-5 mm)
- **Measured**: σ_xx, σ_yy, σ_xy (in-plane stress components)
- **Uncertainty**: ±5-9 MPa
- **Use**: Macro-scale surface validation

**Key Columns**:
- `x_position_mm`, `y_position_mm`, `z_position_mm`: Measurement location
- `sigma_xx_MPa`, `sigma_yy_MPa`, `sigma_xy_MPa`: Stress tensor components
- `measurement_error_MPa`: Experimental uncertainty
- `diffraction_peak_2theta_deg`: Peak position (quality indicator)

#### b) Raman Spectroscopy Stress
- **File**: `experimental_data/residual_stress/raman_spectroscopy_stress.csv`
- **Points**: 28 measurements (higher spatial resolution than XRD)
- **Method**: Stress calculated from Raman peak shift
- **Uncertainty**: ±8-19 MPa (higher than XRD due to peak fitting)
- **Use**: Cross-validation of XRD, higher spatial resolution

**Key Columns**:
- `raman_shift_cm-1`: Peak position shift
- `calculated_stress_MPa`: Derived stress value
- `stress_uncertainty_MPa`: Measurement uncertainty

#### c) Synchrotron X-ray Diffraction (Subsurface)
- **File**: `experimental_data/residual_stress/synchrotron_xrd_subsurface.csv`
- **Points**: 28 depth-resolved measurements
- **Unique Feature**: Full 3D stress tensor (σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz)
- **Depth Range**: 0-1 mm
- **Use**: Volume validation, not just surface

**Key Columns**:
- `depth_mm`: Measurement depth below surface
- `sigma_xx_MPa` through `sigma_yz_MPa`: Full stress tensor
- `von_mises_stress_MPa`: Equivalent stress
- `hydrostatic_stress_MPa`: Mean stress

### 2. Experimental Data - Crack Analysis

#### a) Crack Initiation Data
- **File**: `experimental_data/crack_analysis/crack_initiation_data.csv`
- **Cracks**: 19 identified cracks across 5 samples
- **Use**: Validate stress-based failure criteria

**Critical Information**:
- `crack_length_um`, `crack_width_nm`: Crack geometry
- `critical_load_N`, `critical_temperature_K`: Failure conditions
- `stress_intensity_factor_MPa_sqrt_m`: Fracture mechanics parameter
- `grain_boundary_crack`: TRUE if intergranular
- `pore_associated`: TRUE if nucleated at pore

**Key Insight**: 15/19 cracks are grain-boundary cracks, 12/19 are pore-associated → validates stress concentration predictions

#### b) Crack Propagation Data
- **File**: `experimental_data/crack_analysis/crack_propagation_data.csv`
- **Sequences**: 7 in-situ crack growth sequences
- **Time Steps**: 4-7 observations per crack
- **Use**: Dynamic validation, crack growth rate models

**Key Columns**:
- `elapsed_time_s`: Time since initial observation
- `crack_length_um`: Current crack length
- `crack_growth_rate_m_s`: Instantaneous growth rate
- `J_integral_J_m2`: Energy release rate

### 3. Simulation Output - Full-Field Data

This is your **FEM model output** - the "reference solution" from your high-fidelity simulation.

#### a) Nodal Solution
- **File**: `simulation_output/full_field/fem_full_field_solution.csv`
- **Nodes**: 41 nodes (representative subset; full model has 100K+ nodes)
- **Fields**: Temperature (T), Displacement (U), Stress (σ), Strain (ε)

**All Fields Available**:
- `temperature_K`: Thermal field
- `displacement_x/y/z_um`: Displacement vector
- `stress_xx/yy/zz/xy/xz/yz_MPa`: Full stress tensor
- `strain_xx/yy/zz/xy/xz/yz`: Full strain tensor
- `von_mises_MPa`: Equivalent stress
- `hydrostatic_MPa`: Mean stress

**How to Use**:
1. Compare FEM `stress_xx` at (x,y,z) with experimental XRD `sigma_xx_MPa`
2. Calculate residuals: R = |FEM - Experiment| / Uncertainty
3. Identify regions with high residuals → model needs refinement

#### b) Element Data
- **File**: `simulation_output/full_field/fem_element_data.csv`
- **Elements**: 38 elements
- **Additional Info**: Element type, volume, strain energy, microstructure flags

**Key Columns**:
- `has_pore`: TRUE if element contains pore
- `pore_volume_fraction`: Porosity in element
- `grain_boundary_length_mm`: GB length in element
- `total_strain_energy_mJ`: Energy stored in element

### 4. Collocation Points (Strategic Subset)

**This is the core of your methodology** - a carefully selected subset of the full-field data.

#### File: `simulation_output/collocation_points/collocation_point_data.csv`
- **Points**: 30 strategically selected locations
- **Purpose**: "As-if-measured" data for surrogate training and residual analysis

**Selection Strategy**:
1. **Free Surface Points (10)**: Where thermal gradients are highest
2. **Near-Pore Points (7)**: Within 10 μm of pores (stress concentrators)
3. **Grain Boundary Points (6)**: Within 1 μm of GBs (crack initiation sites)
4. **Interior Points (7)**: Bulk behavior away from defects

**Key Columns**:
- `location_type`: free_surface / interior / grain_boundary
- `distance_to_pore_um`: Distance to nearest pore
- `distance_to_grain_boundary_um`: Distance to nearest GB
- `selection_criterion`: Why this point was chosen

**Use Cases**:
1. **Surrogate Model Training**: Use these 30 points to train a Neural Network, then predict the full field
2. **Residual Analysis**: Compare collocation point predictions vs. full-field "truth" to find model errors
3. **Sensor Placement**: Optimal locations for experimental measurements

### 5. Multi-Scale Data

#### a) Macro-Scale
- **Material Properties**: Temperature-dependent E, ν, CTE, thermal conductivity
- **Cell Geometry**: 6 different cell configurations
- **Sintering Profile**: Complete thermal history (298K → 1673K → 298K)

**Use**: Input to macro-scale FEM model

#### b) Meso-Scale
- **Microstructure Characterization**: Grain size, porosity, pore size distributions
- **RVE Geometry**: Representative Volume Elements for homogenization

**Critical Data**:
- `grain_size_mean_um`: Average grain diameter (8-13 μm)
- `porosity_pct`: Volume fraction of pores (5-10%)
- `pore_size_mean_um`: Average pore diameter (2.6-6.8 μm)
- `grain_boundary_density_mm-1`: GB length per unit volume

**Use**: Generate microstructural FEM models, homogenize properties

#### c) Micro-Scale
- **Grain Boundary Properties**: Energy, mobility, diffusivity, misorientation
- **Crystallographic Orientation**: Euler angles, Schmid factors, Taylor factors

**Use**: Crystal plasticity models, GB fracture criteria

## How to Use This Dataset

### Workflow 1: FEM Model Validation

```python
import pandas as pd
import numpy as np

# Load experimental ground truth
xrd_data = pd.read_csv('experimental_data/residual_stress/xrd_surface_residual_stress.csv')

# Load FEM predictions (at matching locations)
fem_data = pd.read_csv('simulation_output/full_field/fem_full_field_solution.csv')

# Calculate residuals
residuals = fem_data['stress_xx_MPa'] - xrd_data['sigma_xx_MPa']
relative_error = residuals / xrd_data['measurement_error_MPa']

# Identify regions with high error (|relative_error| > 2)
problematic_regions = fem_data[np.abs(relative_error) > 2]
print(f"Model needs refinement at {len(problematic_regions)} locations")
```

### Workflow 2: Surrogate Model Training

```python
# Load collocation points (sparse "measurements")
colloc = pd.read_csv('simulation_output/collocation_points/collocation_point_data.csv')

# Features: spatial coordinates
X = colloc[['x_coord_mm', 'y_coord_mm', 'z_coord_mm']].values

# Target: stress field
y = colloc['von_mises_MPa'].values

# Train surrogate model (e.g., Gaussian Process)
from sklearn.gaussian_process import GaussianProcessRegressor
gp = GaussianProcessRegressor()
gp.fit(X, y)

# Predict at new locations
fem_full = pd.read_csv('simulation_output/full_field/fem_full_field_solution.csv')
X_new = fem_full[['x_coord_mm', 'y_coord_mm', 'z_coord_mm']].values
y_pred = gp.predict(X_new)

# Compare surrogate predictions vs. full FEM
error = np.abs(y_pred - fem_full['von_mises_MPa'].values)
print(f"Mean surrogate error: {error.mean():.2f} MPa")
```

### Workflow 3: Crack Prediction Model

```python
# Load crack initiation data
cracks = pd.read_csv('experimental_data/crack_analysis/crack_initiation_data.csv')

# Load stress state at crack locations
stress_at_cracks = fem_data.merge(cracks, on=['x_position_mm', 'y_position_mm'])

# Correlation analysis
import matplotlib.pyplot as plt
plt.scatter(stress_at_cracks['von_mises_MPa'], 
            stress_at_cracks['stress_intensity_factor_MPa_sqrt_m'])
plt.xlabel('Von Mises Stress (MPa)')
plt.ylabel('Stress Intensity Factor (MPa√m)')
plt.title('Stress vs. Fracture Toughness')
plt.show()

# Build failure criterion
threshold = stress_at_cracks['von_mises_MPa'].min()
print(f"Crack initiation threshold: {threshold:.1f} MPa")
```

## Important Notes

### Data Alignment
- FEM nodes and experimental measurement locations do **not** perfectly overlap
- Use spatial interpolation (nearest neighbor, kriging, or RBF) to compare
- Collocation points are intentionally placed at experimental measurement locations

### Coordinate System
- Origin (0,0,0) is at the bottom-left corner of the sample
- X-axis: Length direction
- Y-axis: Width direction  
- Z-axis: Thickness direction (depth into sample)
- All coordinates in millimeters unless otherwise specified

### Units
- Length: mm (millimeters) or μm (micrometers)
- Stress: MPa (megapascals)
- Temperature: K (kelvin)
- Time: s (seconds) or min (minutes)
- Energy: mJ (millijoules) or J/m² (joules per square meter)

### Stress Convention
- **Negative values = compressive stress** (common in ceramics after cooling)
- Von Mises stress is always positive (equivalent stress)
- Hydrostatic stress: negative = compression, positive = tension

## Key Insights from Dataset

1. **Stress Distribution**:
   - Surface residual stresses: -40 to -180 MPa (compressive)
   - Peak stresses at (7.5, 2.5) region → likely due to pore/GB concentration
   - Stress increases with depth in some regions

2. **Crack Behavior**:
   - 78-87% of fracture is intergranular (grain boundary cracking)
   - 63% of cracks associated with pores
   - Critical von Mises stress for crack initiation: ~150-170 MPa
   - Crack growth accelerates above ~100 J/m² J-integral

3. **Microstructure Effects**:
   - Grain size: 8-13 μm (larger grains in high-stress regions)
   - Porosity: 5-10% (higher near crack sites)
   - High-angle grain boundaries (>50°) show higher energy and slower diffusivity

## Recommended Analysis Tools

### Python
```bash
pip install pandas numpy scipy matplotlib scikit-learn
pip install pyvista  # For 3D visualization
```

### MATLAB
- Use `readtable()` for CSV import
- `scatteredInterpolant()` for spatial interpolation
- Statistics and Machine Learning Toolbox for surrogate models

### ParaView
- Convert CSV to VTK format for 3D visualization
- Use "Table to Points" filter

## Citation

If you use this dataset in your research, please cite:

```
@dataset{fem_validation_sofc_2025,
  title={Multi-Scale FEM Validation Dataset for SOFC Electrolytes},
  author={Research Team},
  year={2025},
  publisher={Institution},
  version={1.0.0}
}
```

## License

This dataset is released under **CC-BY-4.0** (Creative Commons Attribution 4.0 International).

You are free to:
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit

## Contact

For questions, issues, or suggestions:
- Email: contact@institution.edu
- Institution: Materials Science & Engineering Department

## Changelog

### Version 1.0.0 (2025-10-08)
- Initial release
- 98 experimental stress measurements
- 45 crack analysis data points
- 41 full-field FEM nodes + 38 elements
- 30 collocation points
- Complete multi-scale characterization

---

**Dataset Generated**: 2025-10-08  
**Format Version**: 1.0.0  
**Total Files**: 17 data files + 2 metadata files