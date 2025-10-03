
# FEM Simulation Dataset Summary
Generated on: 2025-10-03 06:58:01

## Dataset Overview
This synthetic dataset simulates multi-physics FEM analysis results typical of COMSOL/ABAQUS simulations.

## Input Parameters

### Mesh Data
- Total nodes: 8000
- Total elements: 5000
- Element types: TETRA10, HEX20, WEDGE6, HEX8, TETRA4
- Element size range: 5.59e-01 - 4.56e+00

### Boundary Conditions
- Temperature BC nodes: 800
- Displacement BC nodes: 400
- Voltage BC nodes: 640
- Heat flux BC nodes: 960
- Simulation time: 3600.0 seconds

### Material Models
- Number of materials: 5
- Materials: aluminum, copper, polymer, ceramic, interface

### Thermal Profiles
- Number of profiles: 10
- Heating rates: 1-10°C/min
- Cooling rates: 1-10°C/min

## Output Data

### Stress Distributions
- Von Mises stress range: 5287.5 - 2251375472.3 MPa
- Principal stresses: σ₁, σ₂, σ₃
- Interface elements with shear stress: 750

### Strain Fields
- Elastic strain: All elements
- Plastic strain: All elements (accumulated)
- Creep strain: 1000 polymer elements
- Thermal strain: All elements

### Damage Evolution
- Elements with damage: 4962
- Failed elements: 3577
- Maximum damage: 1.000

### Field Distributions
- Temperature range: 3.8 - 66.8 °C
- Voltage range: 2.43 - 4.72 V

### Failure Predictions
- Delamination initiated: 750 interface elements
- Crack initiation: 500 elements

## File Structure
```
fem_dataset/
├── nodes.csv                    # Node coordinates
├── elements.csv                 # Element connectivity and properties
├── mesh_quality.json           # Mesh quality metrics
├── boundary_conditions.json    # All boundary conditions
├── material_models.json        # Material property definitions
├── thermal_profiles.json       # Transient thermal loading
├── stress_distributions.json   # Stress field results
├── strain_fields.json          # Strain field results
├── damage_evolution.json       # Damage variable evolution
├── field_distributions.json    # Temperature and voltage fields
├── failure_predictions.json    # Delamination and crack predictions
└── summary_report.md           # This summary
```

## Usage Notes
- All stress values are in Pa (Pascals) unless otherwise specified
- Time arrays are in seconds
- Temperature values are in Celsius
- Voltage values are in Volts
- Damage variable ranges from 0 (no damage) to 1 (complete failure)
- Element and node IDs start from 1

## Data Validation
This synthetic dataset includes realistic:
- Material property ranges
- Stress-strain relationships
- Thermal coupling effects
- Damage evolution patterns
- Failure mode interactions

The data can be used for:
- Machine learning model training
- Algorithm validation
- Visualization development
- Educational purposes
