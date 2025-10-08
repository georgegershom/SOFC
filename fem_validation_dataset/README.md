# FEM Validation & Analysis Dataset

## Overview
This comprehensive dataset provides multi-scale experimental and simulation data for validating Finite Element Method (FEM) models of SOFC electrolyte materials, specifically focusing on residual stress analysis and crack initiation/propagation.

## Dataset Structure

```
fem_validation_dataset/
├── residual_stress/
│   └── experimental/
│       ├── xrd_surface_stress.json         # X-ray Diffraction surface measurements
│       ├── raman_spectroscopy.json         # Raman spectroscopy stress data
│       └── synchrotron_xrd.json           # Synchrotron subsurface stress profiles
├── crack_analysis/
│   └── experimental/
│       ├── sem_crack_analysis.json        # SEM micro-crack observations
│       └── fracture_mechanics.json        # Fracture toughness and critical loads
├── simulation_output/
│   ├── fem_full_field_data.csv           # Complete FEM nodal results
│   ├── element_data.json                 # Element-level stress/strain data
│   └── collocation_points.json           # Strategic sampling points for surrogate modeling
├── multi_scale_data/
│   ├── macro_scale_cell_data.json        # Cell-level bulk properties
│   ├── meso_scale_microstructure.json    # RVE grain and pore structure
│   └── micro_scale_grain_boundary.json   # Grain boundary and crystallographic data
└── analysis_scripts/
    ├── data_visualization.py             # Comprehensive plotting and visualization
    └── residual_analysis.py              # Residual calculation and validation metrics
```

## Data Categories

### 1. Residual Stress State (Experimental)
- **Surface Measurements**: XRD and Raman spectroscopy data at multiple locations
- **Subsurface Profiles**: Synchrotron XRD depth-dependent stress measurements
- **Temperature Dependence**: Stress evolution from 25°C to 800°C

### 2. Crack Initiation & Propagation
- **Micro-crack Characteristics**: Length, width, orientation, type (inter/transgranular)
- **Initiation Sites**: Pores, grain boundaries, triple junctions
- **Critical Conditions**: Temperature, mechanical load, thermal cycles
- **Fracture Mechanics**: K_IC, Paris law parameters, Weibull statistics

### 3. FEM Simulation Output
- **Full-field Data**: Temperature, displacement, stress, strain at all nodes
- **Element Data**: Integration point stresses, damage parameters
- **Stress Concentrations**: Identified critical regions with SCF values

### 4. Collocation Points
- **Strategic Sampling**: 150 points selected at critical microstructural features
- **Location Types**: Near pores, grain boundaries, triple junctions, free surfaces
- **Complete State**: Full stress/strain tensors, temperature, microstructural metrics

### 5. Multi-Scale Data

#### Macro-Scale (Cell Level)
- Bulk material properties (E = 210 GPa, ν = 0.31, CTE = 10.5×10⁻⁶/K)
- Cell dimensions and operating conditions
- Global FEM results and critical regions

#### Meso-Scale (Grain & Pore Level)
- Grain size distribution (mean: 2.5 μm, log-normal)
- Porosity data (7.2% total, 5.8% open)
- RVE size: 50×50×50 μm³
- Microstructural stress analysis

#### Micro-Scale (Grain Boundary)
- GB types and properties
- Crystallographic orientation (EBSD data)
- Local chemistry and segregation
- Crack initiation criteria

## Key Features

### Material Properties
- **Material**: 8YSZ (8 mol% Yttria-Stabilized Zirconia)
- **Young's Modulus**: 210 GPa
- **Poisson's Ratio**: 0.31
- **CTE**: 10.5×10⁻⁶/K
- **Fracture Toughness**: 2.8 MPa√m
- **Density**: 5.9 g/cm³ (96.7% theoretical)

### Stress State
- **Average Surface Stress**: σ_xx = -289.1 MPa, σ_yy = -302.8 MPa
- **Maximum Von Mises**: 406.8 MPa (at collocation points)
- **Stress Gradient**: Up to 1250 MPa/mm in critical regions

### Microstructure
- **Grain Size**: 2.5 ± 0.8 μm (log-normal distribution)
- **Porosity**: 7.2% (5.8% open, 1.4% closed)
- **Pore Size**: 0.8 ± 0.4 μm
- **Crack Density**: 3.8 per mm²

## Analysis Scripts

### 1. Data Visualization (`data_visualization.py`)
- Residual stress depth profiles
- Crack statistics and distributions
- 3D collocation point visualization
- Multi-scale comparison plots
- Automated summary report generation

### 2. Residual Analysis (`residual_analysis.py`)
- Residual calculation between experimental and simulation
- Statistical error metrics (RMSE, MAE, R²)
- Gaussian Process surrogate modeling
- Critical region identification
- Uncertainty quantification

## Usage

### Running Visualization
```bash
python analysis_scripts/data_visualization.py
```
This generates:
- Comprehensive plots in `figures/` directory
- Summary report in `summary_report.txt`

### Running Residual Analysis
```bash
python analysis_scripts/residual_analysis.py
```
This produces:
- Residual analysis plots
- Statistical metrics in `residual_analysis_results/`
- Detailed validation report

## Data Format

### JSON Files
All JSON files follow a consistent structure:
- `metadata`: Measurement/simulation details
- `data`: Main measurements/results
- `statistics`: Summary statistics where applicable

### CSV Files
- Header row with descriptive column names
- Consistent units (MPa for stress, mm for length, °C for temperature)

## Validation Metrics

The dataset includes built-in validation metrics:
- **Porosity Error**: 4.0%
- **Grain Size Error**: 3.8%
- **Stress Prediction R²**: >0.9 (target)
- **Normality of Residuals**: Shapiro-Wilk test included

## Applications

This dataset is designed for:
1. **FEM Model Validation**: Compare simulation results with experimental data
2. **Surrogate Model Training**: Use collocation points for ML/GP models
3. **Residual Analysis**: Identify model weaknesses and refinement areas
4. **Multi-scale Modeling**: Link macro, meso, and micro-scale phenomena
5. **Crack Prediction**: Validate crack initiation and propagation models

## Notes

- All stress values are in MPa
- Negative stresses indicate compression
- Coordinates are in mm unless otherwise specified
- Temperature-dependent properties included where relevant
- Statistical distributions provided for stochastic analysis

## Contact & Citation

This is a fabricated dataset for demonstration purposes.
Generated: October 8, 2025

For questions or additional data requirements, please refer to the analysis scripts for data generation methodology.