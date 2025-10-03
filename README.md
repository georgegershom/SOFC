# SOFC Materials Properties Dataset

## 📊 Overview
This repository contains comprehensive fabricated datasets for Solid Oxide Fuel Cell (SOFC) materials properties, including thermo-physical, mechanical, and electrochemical characteristics of key components.

## 🔩 Materials Covered
- **Ni-YSZ Anode** (Nickel - Yttria-Stabilized Zirconia)
- **8YSZ Electrolyte** (8mol% Yttria-Stabilized Zirconia)
- **LSM Cathode** (Lanthanum Strontium Manganite)
- **LSCF Cathode** (Lanthanum Strontium Cobalt Ferrite)
- **Crofer22APU Interconnect** (Ferritic Stainless Steel)
- **GDC Barrier Layer** (Gadolinium-Doped Ceria)

## 📁 Dataset Files

### 1. `sofc_materials_thermal_mechanical.json`
Contains comprehensive thermal and mechanical properties:
- Thermal Expansion Coefficient (TEC) with temperature dependence
- Young's Modulus and Poisson's Ratio
- Density and porosity data
- Thermal conductivity and specific heat capacity
- Fracture strength and Weibull modulus

### 2. `sofc_creep_parameters.json`
Time-dependent deformation parameters:
- Norton-Bailey creep model parameters (B, n, m, Q)
- Primary and steady-state creep data
- Diffusion and dislocation creep mechanisms
- Environmental factors (oxidation, reduction effects)
- Temperature-dependent creep rates

### 3. `sofc_plasticity_parameters.json`
Plasticity model parameters for structural analysis:
- Johnson-Cook model parameters (A, B, n, C, m)
- Ramberg-Osgood parameters
- Chaboche kinematic/isotropic hardening parameters
- Damage parameters and fracture criteria
- Cyclic loading behavior

### 4. `sofc_electrochemical_properties.json`
Electrochemical performance data:
- Ionic and electronic conductivity
- Exchange current density for H₂ oxidation and O₂ reduction
- Activation energies and overpotentials
- Charge transfer resistance
- Triple phase boundary characteristics
- Degradation parameters

### 5. `sofc_temperature_dependent_properties.csv`
Tabulated temperature-dependent properties from 300K to 1373K:
- All key properties at specific temperature points
- Easy-to-use format for interpolation
- Direct input for FEM simulations

## 🔬 Property Ranges

### Temperature Range
- Operating: 973-1273 K (700-1000°C)
- Data provided: 300-1373 K

### Key Parameters at 1073K (800°C)
| Material | TEC (×10⁻⁶/K) | E (GPa) | σ_ionic (S/cm) | σ_electronic (S/cm) |
|----------|----------------|---------|----------------|---------------------|
| Ni-YSZ | 12.8 | 75 | 0.05 | 3400 |
| 8YSZ | 10.6 | 195 | 0.056 | ~10⁻⁸ |
| LSM | 12.0 | 66 | ~10⁻⁷ | 130 |
| LSCF | 15.7 | 62 | 0.16 | 315 |

## 📈 Data Visualization

Run the included Python script to generate comprehensive visualizations:

```bash
pip install -r requirements.txt
python sofc_data_visualization.py
```

This will generate:
- Thermal expansion comparison plots
- Arrhenius conductivity plots
- Mechanical properties comparison
- Electrochemical performance charts
- Summary report

## 🎯 Applications
- Finite Element Analysis (FEA) of SOFC stacks
- Multiphysics simulations (thermal-mechanical-electrochemical)
- Material selection and optimization
- Degradation modeling
- Stack design and performance prediction

## ⚠️ Important Notes
1. **Data Nature**: This is fabricated/synthetic data based on typical literature values
2. **Validation**: Always validate against experimental data for critical applications
3. **Porosity Effects**: Properties are porosity-dependent; use provided correction factors
4. **Temperature Dependence**: Most properties show strong temperature dependence
5. **Microstructure**: Actual properties depend on processing and microstructure

## 🔧 Usage Example

### Python
```python
import json
import pandas as pd

# Load thermal-mechanical properties
with open('sofc_materials_thermal_mechanical.json', 'r') as f:
    data = json.load(f)

# Access Ni-YSZ Young's modulus at 1073K
E_NiYSZ = data['materials']['Ni-YSZ_Anode']['mechanical_properties']['youngs_modulus']['temperature_dependent']['1073']
print(f"Ni-YSZ Young's Modulus at 1073K: {E_NiYSZ/1000:.0f} GPa")

# Load temperature-dependent data
temp_data = pd.read_csv('sofc_temperature_dependent_properties.csv')
```

## 📚 References
Data values are based on typical ranges found in SOFC literature:
- Journal of Power Sources
- Journal of The Electrochemical Society
- Solid State Ionics
- International Journal of Hydrogen Energy

## 📝 License
This dataset is provided for research and educational purposes. Please cite appropriately if used in publications.

## 🤝 Contributing
For corrections or additions to the dataset, please submit a pull request or open an issue.

---
*Generated: October 3, 2025*
*Version: 1.0*