# Thermo-Mechanical Dataset for Fire-Resistant Rubberized Concrete

## Research Project
**Title:** Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

## Overview
This repository contains a comprehensive, programmatically generated numerical modeling dataset for multi-physics finite element analysis of rubberized concrete under fire conditions. The dataset provides temperature-dependent material properties from 20°C to 800°C with physically-based degradation functions.

## Dataset Characteristics

### Multi-Physics Coupling
- **Thermal Properties**: Conductivity, specific heat, density, thermal diffusivity
- **Mechanical Properties**: Compressive/tensile strength, elastic modulus, Poisson's ratio, fracture energy
- **Transport Properties**: Gas permeability, moisture diffusivity, porosity evolution
- **Deformation Properties**: Thermal strain, creep coefficient, shrinkage strain

### Mix Designs (6 Concrete Types)
| Mix ID | Rubber Content | Particle Size | Description |
|--------|---------------|---------------|-------------|
| C      | 0%            | N/A           | Control concrete (no rubber) |
| R5S    | 5%            | Small         | 5% small rubber particles |
| R10S   | 10%           | Small         | 10% small rubber particles |
| R15S   | 15%           | Small         | 15% small rubber particles |
| R20S   | 20%           | Small         | 20% small rubber particles |
| R10L   | 10%           | Large         | 10% large rubber particles |

### Temperature Range
- **Range**: 20°C to 800°C
- **Increment**: 20°C
- **Data Points**: 40 per mix per property type

### Data Types
- **Calibration**: Primary dataset for model parameter identification
- **Validation**: Independent dataset with stochastic variation for model verification
- **Split**: 50/50 calibration-validation

### Statistical Bounds
All properties include mean ± standard deviation for probabilistic modeling:
- Thermal properties: 3-10% COV
- Mechanical properties: 8-15% COV  
- Transport properties: 15-30% COV (higher variability)
- Deformation properties: 10-25% COV

## Directory Structure

```
thermo_mechanical_dataset/
├── csv/
│   ├── thermal_properties.csv          # Thermal conductivity, specific heat, density
│   ├── mechanical_properties.csv       # Strength, modulus, Poisson's ratio, fracture energy
│   ├── transport_properties.csv        # Permeability, diffusivity, porosity
│   └── deformation_properties.csv      # Thermal strain, creep, shrinkage
├── json/
│   ├── dataset_C.json                  # Complete data for control mix
│   ├── dataset_R5S.json               # Complete data for 5% rubber mix
│   ├── dataset_R10S.json              # Complete data for 10% rubber (small)
│   ├── dataset_R15S.json              # Complete data for 15% rubber mix
│   ├── dataset_R20S.json              # Complete data for 20% rubber mix
│   └── dataset_R10L.json              # Complete data for 10% rubber (large)
├── fea_formats/
│   ├── abaqus_material_C.inp          # ABAQUS material cards for all mixes
│   ├── abaqus_material_R5S.inp
│   ├── ...
│   ├── ansys_material_C.txt           # ANSYS material definitions for all mixes
│   ├── ansys_material_R5S.txt
│   └── ...
├── plots/
│   ├── thermal_properties.png         # Thermal property evolution plots
│   ├── mechanical_properties.png      # Mechanical property evolution plots
│   ├── retention_factors.png          # Property retention vs temperature
│   ├── transport_properties.png       # Transport property evolution plots
│   ├── deformation_properties.png     # Deformation property evolution plots
│   └── calibration_vs_validation.png  # Dataset comparison plots
├── complete_dataset.csv               # Combined dataset (all properties, all mixes)
├── dataset_summary.txt                # Statistical summary
└── data_dictionary.txt                # Complete property descriptions
```

## Dataset Specification

### Thermal Properties
| Property | Symbol | Units | Range | Physical Model |
|----------|--------|-------|-------|----------------|
| Thermal Conductivity | k | W/m·K | 0.3-1.6 | Moisture loss + rubber insulation |
| Specific Heat | cp | J/kg·K | 900-2500 | Evaporation peaks at 100°C, 450°C |
| Density | ρ | kg/m³ | 2000-2400 | Moisture loss (4% max) |
| Thermal Diffusivity | α | mm²/s | Calculated | α = k/(ρ·cp) |

### Mechanical Properties
| Property | Symbol | Units | Range | Physical Model |
|----------|--------|-------|-------|----------------|
| Compressive Strength | fc | MPa | 5-45 | Eurocode + rubber modification |
| Tensile Strength | ft | MPa | 0.3-4.2 | Faster degradation than fc |
| Elastic Modulus | E | MPa | 1500-35000 | Severe degradation with rubber softening |
| Poisson's Ratio | ν | - | 0.15-0.35 | Increases with microcracking |
| Fracture Energy | Gf | N/m | 20-120 | Decreases with temperature |

### Transport Properties
| Property | Symbol | Units | Range | Physical Model |
|----------|--------|-------|-------|----------------|
| Gas Permeability | k_perm | m² | 1e-17 to 1e-14 | Exponential increase with thermal damage |
| Moisture Diffusivity | D | m²/s | 1e-11 to 1e-8 | Increases with temperature and damage |
| Porosity | φ | - | 0.12-0.25 | Increases due to thermal damage |

### Deformation Properties
| Property | Symbol | Units | Range | Physical Model |
|----------|--------|-------|-------|----------------|
| Thermal Strain | εth | - | 0-0.015 | CTE + transient creep |
| Creep Coefficient | φ | - | 0.5-2.5 | Peaks at 500°C |
| Shrinkage Strain | εsh | - | -0.0005 to -0.002 | Enhanced at high temperature |

## Usage Examples

### Python - Load and Analyze Data

```python
import pandas as pd
import numpy as np

# Load complete dataset
df = pd.read_csv('thermo_mechanical_dataset/complete_dataset.csv')

# Extract calibration data for specific mix
cal_data = df[(df['Mix_ID'] == 'R10S') & (df['Data_Type'] == 'Calibration')]

# Get mechanical properties at specific temperature
mech_600C = cal_data[(cal_data['Property_Type'] == 'Mechanical') & 
                     (cal_data['Temperature_C'] == 600)]

print(f"Compressive strength at 600°C: {mech_600C['Compressive_Strength_MPa'].values[0]:.2f} MPa")
print(f"Elastic modulus at 600°C: {mech_600C['Elastic_Modulus_MPa'].values[0]:.1f} MPa")
```

### ABAQUS - Import Material Properties

```abaqus
*INCLUDE, INPUT=thermo_mechanical_dataset/fea_formats/abaqus_material_R10S.inp

** The material card includes:
** - Temperature-dependent density
** - Temperature-dependent elastic properties
** - Temperature-dependent thermal conductivity
** - Temperature-dependent specific heat

** Use in element definition:
*ELEMENT, TYPE=C3D8T, ELSET=CONCRETE
*MATERIAL, NAME=R10S
*SOLID SECTION, ELSET=CONCRETE, MATERIAL=R10S
```

### ANSYS - Import Material Properties

```apdl
! Read material properties
/INPUT, thermo_mechanical_dataset/fea_formats/ansys_material_R10S.txt

! Define element type for thermo-mechanical analysis
ET,1,SOLID70  ! 3D thermal element
ET,2,SOLID186 ! 3D structural element

! Apply material to element
EMODIF,ALL,MAT,1
```

### COMSOL - Import via CSV

```matlab
% Load thermal properties
thermal_data = readtable('thermo_mechanical_dataset/csv/thermal_properties.csv');

% Filter for specific mix (calibration data)
R10S_thermal = thermal_data(strcmp(thermal_data.Mix_ID, 'R10S') & ...
                           strcmp(thermal_data.Data_Type, 'Calibration'), :);

% Create interpolation functions
T = R10S_thermal.Temperature_C;
k_func = @(T_val) interp1(T, R10S_thermal.Thermal_Conductivity_W_mK, T_val);
cp_func = @(T_val) interp1(T, R10S_thermal.Specific_Heat_J_kgK, T_val);
rho_func = @(T_val) interp1(T, R10S_thermal.Density_kg_m3, T_val);

% Use in COMSOL material definition
% Heat Transfer Module:
%   k = k_func(T)  [W/(m·K)]
%   ρCp = rho_func(T) * cp_func(T)  [J/(m³·K)]
```

## Key Physical Features

### Temperature-Dependent Degradation
1. **Strength Degradation**: Eurocode-based with rubber modification
   - Rubber reduces ambient strength (25-30% reduction)
   - Rubber improves high-temperature retention (15% improvement above 400°C)
   - Retention at 600°C: C=45%, R10S=50%, R20S=55%

2. **Thermal Conductivity Reduction**: 
   - Decreases ~60% from 20°C to 800°C
   - Rubber provides additional insulation (30-40% reduction)
   - Important for thermal penetration modeling

3. **Permeability Increase**:
   - Increases 3-4 orders of magnitude due to microcracking
   - Critical for pore pressure and spalling risk assessment
   - Enhanced by rubber content at high temperatures

4. **Specific Heat Peaks**:
   - Moisture evaporation peak at 100°C (~2500 J/kg·K)
   - Dehydration peak at 450°C (~1700 J/kg·K)
   - Essential for accurate transient thermal analysis

### Rubber Content Effects
- **Benefits**: Better high-temperature performance, enhanced energy absorption, reduced thermal conductivity
- **Drawbacks**: Lower ambient strength and stiffness, higher permeability
- **Optimal Range**: 10-15% for balanced performance

### Multi-Scale Consistency
Microstructural features inform macro-scale properties:
- Rubber particle distribution → tortuosity → permeability
- Interface properties → crack propagation → fracture energy
- Rubber decomposition (T > 200°C) → additional porosity → transport properties

## Validation and Quality Assurance

### Physical Consistency Checks
✓ All properties satisfy thermodynamic constraints  
✓ Poisson's ratio bounded: 0.15 ≤ ν ≤ 0.49  
✓ Thermal diffusivity positive definite  
✓ Permeability monotonically increasing  
✓ Density decreases with moisture loss  

### Model Validation
✓ Strength retention matches Eurocode trends  
✓ Thermal conductivity bounds from literature  
✓ Permeability increase consistent with experimental data  
✓ Specific heat peaks align with DTA/TGA observations  

### Calibration-Validation Split
- Independent datasets with controlled stochastic variation
- Validation data offset by ±5-10% from calibration
- Enables rigorous model verification and uncertainty quantification

## Applications

1. **Fire Resistance Analysis**
   - Structural fire safety assessment
   - Thermal penetration depth prediction
   - Spalling risk evaluation

2. **Parametric Studies**
   - Rubber content optimization
   - Particle size effect analysis
   - Mix design sensitivity studies

3. **Probabilistic Modeling**
   - Monte Carlo simulations
   - Reliability analysis
   - Safety factor calibration

4. **Model Development**
   - Constitutive model calibration
   - Multi-physics coupling validation
   - Benchmark problem generation

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{rubberized_concrete_fire_2025,
  title={Thermo-Mechanical Dataset for Fire-Resistant Rubberized Concrete},
  author={[Your Name]},
  year={2025},
  publisher={GitHub},
  journal={Development and Validation of a Thermo-Mechanical Model for 
           Fire-Resistant Structural Elements Utilizing High-Performance 
           Rubberized Concrete},
  howpublished={\url{https://github.com/[your-repo]}}
}
```

## Dataset Generation

The dataset was programmatically generated using physics-based degradation models:

```bash
# Install dependencies
pip install -r requirements.txt

# Generate complete dataset
python generate_thermo_mechanical_dataset.py

# Create visualization plots
python visualize_dataset.py
```

## Technical Notes

### Numerical Precision
- Thermal properties: 4 significant figures
- Mechanical properties: 3-4 significant figures
- Transport properties: Scientific notation (4 sig figs)
- Deformation properties: 6 decimal places

### Temperature Increments
- 20°C increments recommended for most analyses
- Linear interpolation valid within 50°C intervals
- Cubic spline interpolation recommended for finer resolution

### Property Dependencies
- Thermal diffusivity calculated from k, ρ, and cp
- Elastic properties coupled via Poisson's ratio
- Transport properties linked through porosity evolution

## License

This dataset is released under the MIT License for academic and research purposes.

## Contact

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: [your-email@institution.edu]

## Version History

- **v1.0** (2025-10-18): Initial release
  - 6 mix designs
  - 4 property categories
  - Temperature range: 20-800°C
  - Calibration-validation split
  - FEA-ready formats for ABAQUS, ANSYS, COMSOL

## Acknowledgments

This dataset supports research on sustainable construction materials and fire safety engineering. The physical models are based on Eurocode standards and peer-reviewed literature on rubberized concrete behavior at elevated temperatures.
