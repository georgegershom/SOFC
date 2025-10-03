# 💻 Numerical Simulation Dataset - Complete Documentation

## Overview

This is a comprehensive synthetic numerical simulation dataset that mimics multi-physics FEM (Finite Element Method) outputs from software like COMSOL and ABAQUS. The dataset includes coupled thermal-mechanical-electrical simulations for battery cell analysis.

## Dataset Contents

### 📥 Input Parameters

#### 1. **Mesh Data** (`input_parameters/mesh_data/`)
- Element size (0.25-2.0 mm)
- Element types: hex8, hex20, tet4, tet10
- Interface refinement factors (2-5x)
- Number of elements and nodes
- Format: JSON files

#### 2. **Boundary Conditions** (`input_parameters/boundary_conditions/`)
- Mechanical: displacement, force, or mixed conditions
- Thermal: convection coefficients, ambient temperature, heat flux
- Electrical: applied voltage (3.0-4.2V), current density, charge rate
- Format: JSON files

#### 3. **Material Models**
- Elastic properties: Young's modulus (70 GPa), Poisson's ratio (0.33)
- Plastic parameters: yield strength (250 MPa), hardening modulus
- Creep: Norton-Bailey law with temperature dependency
- Thermal: conductivity (200 W/m·K), expansion coefficient
- Electrochemical: electrical conductivity, diffusivity

#### 4. **Thermal Profiles** (`input_parameters/thermal_profiles/`)
- Transient heating/cooling profiles
- Heating rates: 1-10°C/min
- Maximum temperature: 60-85°C
- Hold times and cooling rates
- Format: NPZ files with time-temperature arrays

### 📤 Output Data

#### 1. **Stress Fields** (`output_data/stress_fields/`)
- Von Mises stress
- Principal stresses (σ₁, σ₂, σ₃)
- Interfacial shear stresses (τxy, τxz, τyz)
- 3D spatial distribution over time
- Format: HDF5 files with compression

#### 2. **Strain Fields** (`output_data/strain_fields/`)
- Elastic strains (3 components)
- Plastic strains
- Creep strains (time-dependent)
- Thermal strains
- Total strain magnitude
- Format: HDF5 files

#### 3. **Damage Evolution** (`output_data/damage_evolution/`)
- Damage variable D (0-1 scale)
- Damage rate evolution
- Critical element tracking (D > 0.9)
- Multiple damage models:
  - Lemaitre ductile damage
  - Cohesive zone (interface)
  - Fatigue accumulation
- Format: HDF5 files

#### 4. **Temperature Distributions** (`output_data/temperature_distributions/`)
- 3D temperature fields
- Heat flux vectors (x, y, z components)
- Joule heating from electrical current
- Convection and conduction effects
- Format: HDF5 files

#### 5. **Voltage Distributions** (`output_data/voltage_distributions/`)
- Electric potential fields
- Current density vectors
- Joule heating distribution
- Total current calculations
- Format: HDF5 files

#### 6. **Failure Predictions** (`output_data/failure_predictions/`)
- Delamination risk maps (0-1 scale)
- Crack initiation probability
- Crack propagation angles
- Failure index combining multiple criteria
- Format: HDF5 files

## Physical Models Implemented

### 1. **Thermo-Mechanical Coupling**
- Temperature-dependent material properties
- Thermal expansion/contraction stresses
- Heat generation from mechanical work

### 2. **Electrochemical-Mechanical Interactions**
- Voltage-induced stresses
- Current density effects
- Joule heating from electrical resistance

### 3. **Damage and Failure Models**
- **Lemaitre Model**: Ductile damage accumulation
- **Cohesive Zone Model**: Interface delamination
- **Fatigue Model**: Cyclic loading damage

### 4. **Material Nonlinearities**
- J2 plasticity with isotropic hardening
- Norton-Bailey creep law
- Temperature-dependent elastic modulus

## Data Format and Access

### HDF5 Structure
```
simulation.h5
├── t_0/                    # Time step 0
│   ├── von_mises          # Von Mises stress array
│   ├── sigma_1            # Principal stress 1
│   ├── coordinates/       # Spatial coordinates
│   │   ├── x
│   │   ├── y
│   │   └── z
│   └── [other fields]
├── t_1/                    # Time step 1
│   └── ...
└── ...
```

### Python Access Example
```python
import h5py
import numpy as np

# Load stress data
with h5py.File('output_data/stress_fields/sim_0000_stress.h5', 'r') as f:
    # Get von Mises stress at time step 10
    stress = f['t_10']['von_mises'][:]
    
    # Get coordinates
    x = f['t_10']['coordinates']['x'][:]
    y = f['t_10']['coordinates']['y'][:]
    z = f['t_10']['coordinates']['z'][:]
```

### MATLAB Access Example
```matlab
% Load stress data
stress = h5read('sim_0000_stress.h5', '/t_10/von_mises');
x = h5read('sim_0000_stress.h5', '/t_10/coordinates/x');
```

## Units

| Quantity | Unit |
|----------|------|
| Stress | MPa |
| Strain | mm/mm (dimensionless) |
| Temperature | °C |
| Voltage | V |
| Current Density | A/m² |
| Time | seconds |
| Length | mm |
| Damage | 0-1 (dimensionless) |
| Energy | J |
| Power | W |

## Dataset Statistics

- Grid resolution: 20×20×10 to 50×50×20 elements
- Time steps: 20-100 per simulation
- Total data points per simulation: ~8,000-100,000
- File sizes: 5-50 MB per simulation (compressed)

## Applications

This dataset is suitable for:

1. **Machine Learning**
   - Training surrogate models for FEM simulations
   - Failure prediction algorithms
   - Time-series forecasting of damage evolution

2. **Model Validation**
   - Benchmarking numerical methods
   - Verifying multi-physics coupling implementations

3. **Research Studies**
   - Parameter sensitivity analysis
   - Design optimization
   - Reliability assessment

4. **Educational Purposes**
   - Teaching finite element concepts
   - Demonstrating multi-physics phenomena
   - Visualization techniques

## Generation Parameters

The dataset can be regenerated with custom parameters:

```bash
python scripts/generate_dataset.py \
    --num_simulations 100 \
    --grid_size 30 30 15 \
    --time_steps 50 \
    --parameter_sweep
```

## Visualization Tools

The package includes comprehensive visualization capabilities:
- 2D/3D contour plots
- Time evolution animations
- Parameter sweep analysis
- Statistical summaries

## Citation

If you use this dataset, please acknowledge it as:
```
Numerical Simulation Dataset for Multi-Physics FEM Analysis
Generated using synthetic data generator mimicking COMSOL/ABAQUS outputs
Version 1.0, 2025
```

## License

This dataset is provided for research and educational purposes.

## Contact

For questions or issues with the dataset, please refer to the documentation and example scripts provided.