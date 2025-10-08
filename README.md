# FEM Model Validation Dataset

## Overview
This repository contains a comprehensive synthetic dataset for validating Finite Element Method (FEM) models and performing residual analysis on YSZ (Yttria-Stabilized Zirconia) SOFC electrolyte materials.

## Dataset Contents

### 1. Macro-Scale Data (Cell Level)
- **Bulk Material Properties**: Elastic modulus, Poisson ratio, coefficient of thermal expansion, density
- **Surface Residual Stress**: X-ray diffraction (XRD) measurements across sample surface
- **Raman Spectroscopy**: Stress calibration data and local stress measurements
- **Cell Geometry**: Dimensions, active area, porosity measurements
- **Sintering Profile**: Temperature-time history during processing

### 2. Meso-Scale Data (Grain & Pore Level)
- **Microstructure Characterization**: Grain centers, sizes, pore locations from SEM/CT scans
- **Grain Size Distribution**: Statistical analysis of grain sizes with log-normal fitting
- **Porosity Analysis**: Spatial distribution of porosity across the sample
- **Representative Volume Element (RVE)**: Properties for microstructural modeling

### 3. Crack Initiation & Propagation Data
- **Crack Locations**: Positions, lengths, orientations from SEM analysis of fractured cross-sections
- **Critical Conditions**: Temperature, stress, and strain thresholds for crack initiation
- **Crack Propagation**: Growth rates and Paris law parameters for fatigue analysis

### 4. FEM Simulation Output Data
- **Full-Field Data**: Complete temperature, stress, displacement, and strain fields from high-fidelity FEM simulation
- **Collocation Points**: Strategic measurement points for validation and residual analysis
- **Mesh Information**: Node/element counts and coordinate systems

### 5. Micro-Scale Data (Grain Boundary Level)
- **Grain Boundary Properties**: Energy, diffusivity, and mechanical properties
- **EBSD Data**: Crystallographic orientations and misorientation angles
- **Local Stress Concentrations**: Stress gradients and concentration factors

## Files

### Data Files
- `validation_dataset.json` (13 MB) - Complete dataset in human-readable JSON format
- `validation_dataset.h5` (120 KB) - Compressed binary format for efficient storage

### Scripts
- `validation_dataset.py` - Main dataset generator script
- `data_visualization.py` - Visualization and analysis scripts
- `dataset_summary.md` - Detailed dataset documentation

### Documentation
- `README.md` - This file
- `dataset_summary.md` - Comprehensive dataset documentation

## Quick Start

### Loading the Dataset
```python
import json
import h5py

# Load JSON format (recommended for exploration)
with open('validation_dataset.json', 'r') as f:
    dataset = json.load(f)

# Load HDF5 format (recommended for large-scale analysis)
with h5py.File('validation_dataset.h5', 'r') as f:
    macro_data = f['macro_scale']
    meso_data = f['meso_scale']
    # ... access other groups
```

### Running Visualizations
```bash
python3 data_visualization.py
```

This will generate:
- Residual stress analysis plots
- Microstructure visualizations
- Grain size distribution analysis
- Crack analysis plots
- FEM simulation result plots
- Collocation point analysis

## Dataset Structure

```
validation_dataset/
├── metadata/                 # Dataset information
├── macro_scale/            # Cell-level experimental data
│   ├── bulk_properties/    # Material properties
│   ├── xrd_measurements/   # Surface stress data
│   ├── raman_data/         # Local stress measurements
│   └── sintering_profile/  # Processing conditions
├── meso_scale/             # Grain & pore level data
│   ├── microstructure/     # SEM/CT scan data
│   ├── grain_sizes/        # Grain size distribution
│   ├── porosity/          # Porosity analysis
│   └── rve_data/          # Representative volume element
├── crack_data/            # Crack analysis data
│   ├── crack_locations/   # SEM crack analysis
│   ├── critical_conditions/ # Failure thresholds
│   └── crack_propagation/ # Growth rate data
├── fem_simulation/        # FEM simulation output
│   ├── full_field/        # Complete field data
│   └── collocation_points/ # Strategic measurement points
└── micro_scale/           # Grain boundary data
    ├── grain_boundary_properties/ # GB properties
    ├── ebsd_data/         # Crystallographic data
    └── local_stress/      # Local stress concentrations
```

## Usage Examples

### Residual Stress Validation
```python
# Extract XRD measurements
xrd_data = dataset['macro_scale']['xrd_measurements']
positions = np.array(xrd_data['positions'])
stresses = np.array(xrd_data['stress_values'])

# Compare with FEM predictions
fem_stress = dataset['fem_simulation']['full_field']['stress']['xx']
# ... perform validation analysis
```

### Collocation Point Analysis
```python
# Extract collocation points
collocation_data = dataset['fem_simulation']['collocation_points']
points = collocation_data['points']

# Analyze sparse measurements
for point in points:
    coords = point['coordinates']
    temperature = point['temperature']
    stress = point['stress']
    # ... perform residual analysis
```

### Crack Analysis
```python
# Extract crack data
crack_data = dataset['crack_data']['crack_locations']
positions = np.array(crack_data['positions'])
lengths = np.array(crack_data['lengths'])

# Analyze crack patterns
# ... perform crack analysis
```

## Data Quality and Validation

### Synthetic Nature
- All data is synthetically generated for demonstration purposes
- Based on realistic material properties and experimental conditions
- Includes appropriate measurement uncertainties and noise

### Validation Strategy
1. **Macro-scale**: Use bulk properties for global model validation
2. **Meso-scale**: Use microstructure data for local stress concentration analysis
3. **Micro-scale**: Use grain boundary data for crack initiation prediction
4. **Collocation**: Use sparse measurements for model refinement

### Recommended Analysis Workflow
1. Load full-field FEM simulation data
2. Extract collocation point measurements
3. Compare with experimental residual stress data
4. Perform residual analysis to identify model inaccuracies
5. Refine model in critical regions
6. Validate against crack initiation data

## Requirements

### Python Dependencies
- numpy
- pandas
- scipy
- matplotlib
- h5py
- seaborn (for visualization)

### Installation
```bash
pip install numpy pandas scipy matplotlib h5py seaborn
```

## Citation

If you use this dataset in your research, please cite:

```
FEM Model Validation Dataset for YSZ SOFC Electrolyte Materials
Generated for residual analysis and model validation studies
```

## License

This dataset is provided for research and educational purposes. Please ensure appropriate attribution when using this data.

## Contact

For questions about the dataset structure or usage, please refer to the source code in `validation_dataset.py` or the detailed documentation in `dataset_summary.md`.