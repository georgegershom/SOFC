# Numerical Simulation Dataset for Multi-Physics FEM Models

## Overview
This dataset contains comprehensive numerical simulation data generated for multi-physics Finite Element Method (FEM) models, suitable for COMSOL and ABAQUS simulations. The dataset includes both input parameters and output results for 100 simulation cases.

## Dataset Structure

### Input Parameters

#### 1. Mesh Data (`mesh_data.csv`)
- **Element sizes**: 0.1-2.0 mm
- **Element types**: Tetrahedral, Hexahedral, Triangular, Quadrilateral, Wedge, Pyramid, Mixed
- **Interface refinement levels**: 1-4
- **Total elements**: 10,000-1,000,000
- **Mesh quality**: 0.6-1.0
- **Aspect ratios**: 1.0-10.0

#### 2. Boundary Conditions (`boundary_conditions.csv`)
- **Temperature conditions**: Ambient temp (20-25°C), max temp (100-300°C), gradients, convection coefficients
- **Displacement conditions**: Fixed displacements, applied forces, pressure loads, constraint types
- **Voltage conditions**: Applied voltage (0-1000V), current density, electrical conductivity, dielectric constant

#### 3. Material Models (`material_models.csv`)
- **Elastic properties**: Young's modulus (1-500 GPa), Poisson's ratio (0.1-0.5), shear modulus, bulk modulus
- **Plastic properties**: Yield strength (50-2000 MPa), hardening modulus, strain hardening exponent
- **Creep properties**: Creep coefficient, creep exponent, activation energy, reference stress
- **Thermal properties**: Thermal conductivity (0.1-400 W/mK), specific heat, density, thermal expansion
- **Electrochemical properties**: Electrical conductivity, ionic conductivity, diffusion coefficient, electrochemical potential

#### 4. Thermal Profiles (`numerical_simulation_dataset.json`)
- **Heating/cooling rates**: 1-10°C/min
- **Temperature cycles**: Duration, max/min temperatures, ramp times
- **Thermal history**: Time-temperature profiles for each simulation

### Output Data

#### 1. Stress Distributions (`output_stress_distributions.csv`)
- Von Mises stress (0-1000 MPa)
- Principal stresses (σ₁, σ₂, σ₃)
- Interfacial shear stress

#### 2. Strain Fields (`output_strain_fields.csv`)
- Elastic strain (0-0.01)
- Plastic strain (0-0.1)
- Creep strain (0-0.05)
- Thermal strain (-0.01 to 0.01)
- Total strain (0-0.15)

#### 3. Damage Evolution (`output_damage_evolution.csv`)
- Damage variable D (0-1)
- Damage rate (0-1e-3 1/s)
- Crack length (0-10 mm)
- Crack density (0-100 cracks/mm²)
- Fatigue life (100-1e6 cycles)

#### 4. Temperature Distributions (`output_temperature_distributions.csv`)
- Max/min temperatures (50-500°C / -50-50°C)
- Temperature gradients (0.1-100°C/mm)
- Effective thermal conductivity

#### 5. Voltage Distributions (`output_voltage_distributions.csv`)
- Max/min voltage (0-1000V / -1000-0V)
- Voltage gradients (0-1000 V/mm)
- Current density (0-1000 A/m²)
- Electric field strength (0-1e6 V/m)

#### 6. Failure Predictions (`output_failure_predictions.csv`)
- Delamination probability (0-1)
- Crack initiation time (0-10000 hours)
- Failure modes: Brittle fracture, Ductile failure, Fatigue, Creep rupture, Thermal shock, Electrical breakdown
- Safety factors (0.5-5.0)

## Usage

### Loading the Dataset
```python
import json
import pandas as pd

# Load the complete dataset
with open('numerical_simulation_dataset.json', 'r') as f:
    dataset = json.load(f)

# Load specific CSV files
mesh_data = pd.read_csv('simulation_data/mesh_data.csv')
stress_data = pd.read_csv('simulation_data/output_stress_distributions.csv')
```

### Key Statistics
- **Dataset size**: 100 simulations
- **Input parameter categories**: 4 (mesh, boundary conditions, materials, thermal profiles)
- **Output data categories**: 6 (stress, strain, damage, temperature, voltage, failure predictions)

### Sample Statistics
- **Stress (von Mises)**: Mean = 448.8 MPa, Std = 283.8 MPa
- **Strain (total)**: Mean = 0.074, Std = 0.046
- **Damage variable**: Mean = 0.512, Std = 0.275
- **Temperature**: Mean = 275.4°C, Std = 122.8°C

## File Structure
```
/workspace/
├── numerical_simulation_dataset.py    # Dataset generation script
├── numerical_simulation_dataset.json # Complete dataset (JSON format)
├── simulation_data/                  # CSV files for easy analysis
│   ├── mesh_data.csv
│   ├── boundary_conditions.csv
│   ├── material_models.csv
│   ├── output_stress_distributions.csv
│   ├── output_strain_fields.csv
│   ├── output_damage_evolution.csv
│   ├── output_temperature_distributions.csv
│   ├── output_voltage_distributions.csv
│   └── output_failure_predictions.csv
└── README.md                        # This documentation
```

## Applications
This dataset is suitable for:
- Machine learning model training for FEM simulation prediction
- Statistical analysis of multi-physics simulation results
- Validation of simulation methodologies
- Research in computational mechanics and materials science
- Development of surrogate models for expensive FEM simulations

## Data Generation
The dataset was generated using realistic parameter ranges based on typical FEM simulation practices. All values are within physically meaningful ranges for engineering materials and simulation conditions.