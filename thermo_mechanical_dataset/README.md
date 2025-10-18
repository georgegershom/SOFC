# Thermo-Mechanical Modeling Dataset for Fire-Resistant Structural Elements
## High-Performance Rubberized Concrete

### Dataset Overview

This comprehensive numerical modeling dataset provides complete, structured, and physically consistent input parameters for finite element model development of fire-resistant structural elements utilizing high-performance rubberized concrete. The dataset enables multi-physics modeling incorporating thermal, mechanical, and poro-mechanical coupling with temperature-dependent degradation.

### Research Context

**Title:** "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete"

**Purpose:** Enable rigorous model verification through independent validation datasets while providing complete material property data for multi-physics finite element analysis.

### Dataset Structure

#### Material Mixes
- **C**: Control concrete (0% rubber content)
- **R5S**: 5% shredded rubber by volume
- **R10S**: 10% shredded rubber by volume  
- **R15S**: 15% shredded rubber by volume
- **R20S**: 20% shredded rubber by volume
- **R10L**: 10% large rubber particles by volume

#### Temperature Range
- **Range**: 20°C to 800°C
- **Increment**: 5°C steps (157 temperature points)
- **Coverage**: Complete fire exposure scenario

#### Data Types
- **Calibration**: ~70% of data for model parameter fitting
- **Validation**: ~30% of data for independent model verification

### Dataset Components

#### 1. Thermal Properties (`thermal_properties.csv`)
- **Thermal Conductivity** (W/m·K): Temperature-dependent thermal transport
- **Specific Heat** (J/kg·K): Heat capacity evolution with temperature
- **Thermal Expansion Coefficient** (1/K): Dimensional changes with temperature
- **Density** (kg/m³): Mass density including moisture loss effects

#### 2. Mechanical Properties (`mechanical_properties.csv`)
- **Compressive Strength** (MPa): Temperature-dependent strength degradation
- **Elastic Modulus** (GPa): Stiffness evolution with temperature
- **Tensile Strength** (MPa): Tensile capacity degradation
- **Poisson's Ratio**: Lateral strain behavior
- **Fracture Energy** (N/m): Crack propagation resistance

#### 3. Transport Properties (`transport_properties.csv`)
- **Porosity**: Void fraction evolution with temperature
- **Permeability** (m²): Fluid transport capacity
- **Water Diffusivity** (m²/s): Moisture transport
- **Vapor Diffusivity** (m²/s): Vapor phase transport

#### 4. Deformation Properties (`deformation_properties.csv`)
- **Creep Coefficient**: Time-dependent deformation
- **Shrinkage Strain**: Drying and thermal shrinkage
- **Thermal Strain**: Temperature-induced deformation
- **Creep Modulus** (GPa): Time-dependent stiffness

#### 5. Stress-Strain Curves (`stress_strain_curves.json`)
- Complete stress-strain relationships for all mixes
- Temperature-dependent curve parameters
- Hognestad model implementation
- 100 strain points per curve (0 to 0.01)

### Statistical Properties

All material properties include:
- **Mean Values**: Primary property values
- **Standard Deviations**: Statistical uncertainty bounds
- **Coefficient of Variation**: Ranges from 3% to 30% depending on property
- **Physical Consistency**: Interdependent properties maintain thermodynamic consistency

### FEA Software Compatibility

#### ABAQUS Input Files (`abaqus_inputs/`)
- Material property definitions
- Temperature-dependent property tables
- Damage initiation and evolution parameters
- Creep and thermal expansion definitions

#### ANSYS Input Files (`ansys_inputs/`)
- Material property commands
- Temperature-dependent data tables
- Multi-physics coupling parameters
- Structural and thermal analysis setup

#### COMSOL Input Files (`comsol_inputs/`)
- Material property expressions
- Temperature-dependent functions
- Multi-physics module parameters
- Coupled thermal-structural analysis

### Usage Guidelines

#### 1. Data Loading
```python
import pandas as pd
import json

# Load property data
thermal_df = pd.read_csv('thermal_properties.csv')
mechanical_df = pd.read_csv('mechanical_properties.csv')
transport_df = pd.read_csv('transport_properties.csv')
deformation_df = pd.read_csv('deformation_properties.csv')

# Load stress-strain curves
with open('stress_strain_curves.json', 'r') as f:
    stress_strain_curves = json.load(f)
```

#### 2. Property Interpolation
```python
import numpy as np
from scipy.interpolate import interp1d

# Create interpolation functions
def get_property_at_temp(df, mix_id, property_name, temperature):
    mix_data = df[df['Mix_ID'] == mix_id]
    f = interp1d(mix_data['Temperature_C'], mix_data[property_name])
    return f(temperature)
```

#### 3. Statistical Bounds
```python
# Get property with uncertainty bounds
def get_property_with_bounds(df, mix_id, property_name, temperature, confidence=0.95):
    mix_data = df[df['Mix_ID'] == mix_id]
    f_mean = interp1d(mix_data['Temperature_C'], mix_data[property_name])
    f_std = interp1d(mix_data['Temperature_C'], mix_data[property_name + '_Std'])
    
    mean_val = f_mean(temperature)
    std_val = f_std(temperature)
    
    # Calculate confidence bounds
    z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%
    lower_bound = mean_val - z_score * std_val
    upper_bound = mean_val + z_score * std_val
    
    return mean_val, lower_bound, upper_bound
```

### Physical Model Implementation

#### Temperature-Dependent Functions
All properties follow physically-based degradation functions:

1. **Thermal Properties**: Exponential decay with temperature
2. **Mechanical Properties**: Linear degradation with temperature thresholds
3. **Transport Properties**: Power-law relationships with temperature
4. **Deformation Properties**: Time-temperature superposition principles

#### Multi-Physics Coupling
- **Thermal-Mechanical**: Thermal expansion and temperature-dependent stiffness
- **Thermal-Transport**: Temperature-dependent permeability and diffusivity
- **Mechanical-Transport**: Stress-dependent porosity and permeability
- **Coupled Deformation**: Creep, shrinkage, and thermal strain interactions

### Validation and Calibration

#### Calibration Dataset
- Used for model parameter identification
- Covers full temperature range
- Includes all material mixes
- Statistical representation of material variability

#### Validation Dataset
- Independent verification data
- Separate from calibration process
- Used for model performance assessment
- Rigorous model validation protocol

### Quality Assurance

#### Physical Consistency Checks
- Thermodynamic compatibility between properties
- Energy conservation in coupled models
- Mass conservation in transport equations
- Stress-strain relationship validity

#### Statistical Validation
- Normal distribution assumptions verified
- Coefficient of variation within expected ranges
- Temperature dependency smoothness
- Mix-to-mix property relationships

### File Organization

```
thermo_mechanical_dataset/
├── README.md                           # This documentation
├── complete_dataset.xlsx              # All data in Excel format
├── thermal_properties.csv             # Thermal property data
├── mechanical_properties.csv          # Mechanical property data
├── transport_properties.csv           # Transport property data
├── deformation_properties.csv         # Deformation property data
├── stress_strain_curves.json          # Stress-strain relationships
├── abaqus_inputs/                     # ABAQUS material files
│   ├── material_C.inp
│   ├── material_R5S.inp
│   └── ...
├── ansys_inputs/                      # ANSYS material files
│   ├── material_C.txt
│   ├── material_R5S.txt
│   └── ...
└── comsol_inputs/                     # COMSOL material files
    ├── material_C.txt
    ├── material_R5S.txt
    └── ...
```

### Citation

If you use this dataset in your research, please cite:

```
Thermo-Mechanical Modeling Dataset for Fire-Resistant Structural Elements
Utilizing High-Performance Rubberized Concrete
[Your Institution], [Year]
DOI: [To be assigned]
```

### Contact

For questions about this dataset or its usage, please contact:
- Research Team: [Contact Information]
- Technical Support: [Support Information]

### License

This dataset is provided for research purposes. Please refer to the license agreement for usage terms and conditions.

---

**Dataset Version**: 1.0  
**Last Updated**: [Current Date]  
**Total Data Points**: 3,768  
**Temperature Range**: 20°C - 800°C  
**Material Mixes**: 6 (C, R5S, R10S, R15S, R20S, R10L)  
**Property Categories**: 4 (Thermal, Mechanical, Transport, Deformation)