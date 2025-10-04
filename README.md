# Synthetic Synchrotron X-ray Data for SOFC Creep Analysis

This repository contains tools for generating synthetic synchrotron X-ray experimental data for validating creep deformation models in Solid Oxide Fuel Cell (SOFC) metallic interconnects.

## Overview

The generated dataset simulates realistic synchrotron X-ray experiments including:
- **4D Tomography**: 3D microstructure evolution over time
- **X-ray Diffraction**: Phase identification and residual stress/strain mapping
- **Comprehensive Metadata**: Material properties, test conditions, and calibration data

## Quick Start

### Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### Generate All Data

```bash
# Run the main generation script
python generate_all_data.py
```

This will create approximately 500 MB of synthetic data in the `synchrotron_data/` directory.

### Generate Individual Components

```bash
# Generate only tomography data
python synchrotron_data/scripts/generate_tomography_data.py

# Generate only diffraction data  
python synchrotron_data/scripts/generate_diffraction_data.py

# Generate only metadata
python synchrotron_data/scripts/generate_metadata.py

# Create visualizations (requires existing data)
python synchrotron_data/scripts/visualization_tools.py
```

## Data Structure

```
synchrotron_data/
├── tomography/
│   ├── initial/
│   │   ├── initial_microstructure.h5    # Initial 3D microstructure
│   │   └── grain_map.h5                 # Grain boundary information
│   └── time_series/
│       ├── creep_T600_S150.h5          # Time evolution at 600°C, 150 MPa
│       ├── creep_T700_S100.h5          # Time evolution at 700°C, 100 MPa
│       └── creep_T800_S75.h5           # Time evolution at 800°C, 75 MPa
├── diffraction/
│   ├── initial/
│   │   ├── initial_pattern.h5           # Initial XRD pattern
│   │   └── initial_strain_map.h5        # Initial strain distribution
│   └── time_series/
│       └── insitu_T*_S*.h5             # In-situ diffraction during creep
├── metadata/
│   ├── experiment_metadata.json         # Complete experimental setup
│   ├── experiment_metadata.yaml         # Human-readable format
│   ├── TEST*_metadata.json             # Individual test conditions
│   └── test_summary.csv                # Quick reference table
└── visualizations/
    ├── tomography_slices.png            # 3D volume visualization
    ├── creep_evolution.png              # Time evolution plots
    ├── diffraction_pattern.png          # XRD pattern
    └── strain_stress_maps.png           # Strain/stress fields
```

## Data Format

### HDF5 Structure

All primary data is stored in HDF5 format for efficient storage and access:

#### Tomography Files
- **Volume Data**: 3D arrays (256×256×256 voxels)
- **Attributes**: voxel_size, temperature, stress, time_hours
- **Phase Labels**: 
  - 0: Pores
  - 1: Matrix
  - 2: Grain boundaries
  - 3: Oxides
  - 4: Creep cavities

#### Diffraction Files
- **Patterns**: 2theta and intensity arrays
- **Strain Maps**: 2D arrays for each strain/stress component
- **Phase Evolution**: Time-dependent phase fractions

## Key Features

### 1. Realistic Microstructure Generation
- Voronoi tessellation for grain structure
- Proper grain size distribution
- Initial defects (pores, inclusions)
- Grain boundary networks

### 2. Creep Evolution Simulation
- Cavity nucleation at grain boundaries
- Cavity growth and coalescence
- Crack propagation paths
- Time-dependent damage accumulation

### 3. Diffraction Data
- Multiple crystal structures (BCC, FCC, hexagonal, orthorhombic)
- Peak broadening and shifts from strain
- Texture effects
- Phase transformations

### 4. Comprehensive Metadata
- Material specifications (Crofer 22 APU)
- Test conditions (temperature, stress, atmosphere)
- Synchrotron parameters (beam energy, detector specs)
- Calibration information

## Usage Examples

### Loading Tomography Data

```python
import h5py
import numpy as np

# Load initial microstructure
with h5py.File('synchrotron_data/tomography/initial/initial_microstructure.h5', 'r') as f:
    volume = f['volume'][:]
    voxel_size = f['volume'].attrs['voxel_size']
    print(f"Volume shape: {volume.shape}")
    print(f"Voxel size: {voxel_size*1e6:.2f} µm")

# Load time series
with h5py.File('synchrotron_data/tomography/time_series/creep_T700_S100.h5', 'r') as f:
    # Get specific time step
    time_5 = f['time_005'][:]
    time_hours = f['time_005'].attrs['time_hours']
    cavity_volume = f['time_005'].attrs['cavity_volume']
    print(f"At t={time_hours:.1f}h: {cavity_volume} cavity voxels")
```

### Loading Diffraction Data

```python
# Load diffraction pattern
with h5py.File('synchrotron_data/diffraction/time_series/insitu_T700_S100.h5', 'r') as f:
    # Get pattern at time step 5
    two_theta = f['patterns/pattern_005/2theta'][:]
    intensity = f['patterns/pattern_005/intensity'][:]
    
    # Get strain map
    strain_xx = f['strain_maps/map_005/strain_xx'][:]
    von_mises = f['strain_maps/map_005/von_mises_stress'][:]
```

### Accessing Metadata

```python
import json

# Load experimental metadata
with open('synchrotron_data/metadata/experiment_metadata.json', 'r') as f:
    metadata = json.load(f)
    
# Access material properties
material = metadata['material_specifications']['interconnect_alloy']
print(f"Material: {material['designation']}")
print(f"Cr content: {material['composition']['Cr']}%")
print(f"Young's modulus: {material['mechanical_properties']['youngs_modulus']} GPa")
```

## Customization

You can modify the generation parameters in each script:

### Tomography Parameters
- `volume_size`: Change resolution (default: 256×256×256)
- `voxel_size`: Physical voxel size (default: 0.65 µm)
- `num_grains`: Number of grains in volume
- `porosity`: Initial porosity fraction

### Diffraction Parameters
- `beam_energy`: X-ray energy (default: 80 keV)
- `two_theta_range`: Angular range for patterns
- `phase_composition`: Initial phase fractions

### Test Conditions
Modify the `test_conditions` lists in each script to simulate different temperatures, stresses, and durations.

## Physical Basis

The synthetic data is based on established creep mechanisms:

1. **Norton Power Law**: ε̇ = A·σⁿ·exp(-Q/RT)
2. **Cavity Nucleation**: Preferential at grain boundaries
3. **Diffusion-Controlled Growth**: Cavity growth rate ∝ stress/temperature
4. **Crack Coalescence**: Linking of nearby cavities

## Validation Recommendations

When using this data for model validation:

1. **Start with Initial State**: Validate your model can reproduce the initial microstructure statistics
2. **Check Phase Fractions**: Ensure phase evolution matches expected oxidation kinetics
3. **Verify Strain Fields**: Compare predicted and "measured" strain distributions
4. **Track Damage Evolution**: Monitor cavity volume and crack length progression
5. **Cross-Validate**: Use multiple test conditions for robust validation

## Limitations

This is synthetic data with simplified physics:
- Creep laws are approximated
- Oxidation kinetics are simplified
- Grain boundary sliding is not fully modeled
- Phase transformations are basic

For actual validation, real synchrotron data should be used when available.

## Citation

If you use this data generation tool in your research, please cite:
```
Synthetic Synchrotron X-ray Data Generator for SOFC Creep Analysis
[Your Institution], 2024
https://github.com/[your-repo]
```

## License

This project is provided as-is for research purposes.

## Contact

For questions or improvements, please open an issue or contact the maintainers.