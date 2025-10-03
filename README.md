# SOFC Material Properties Dataset 🔩

A comprehensive dataset of thermo-physical, mechanical, and electrochemical properties for Solid Oxide Fuel Cell (SOFC) components.

## 📋 Dataset Overview

This dataset includes material properties for 6 key SOFC materials:

1. **Ni-YSZ Anode** - Nickel-Yttria Stabilized Zirconia composite
2. **8YSZ Electrolyte** - 8 mol% Yttria Stabilized Zirconia
3. **LSM Cathode** - Lanthanum Strontium Manganite
4. **Crofer22 APU Interconnect** - Ferritic stainless steel
5. **LSCF Cathode** - Lanthanum Strontium Cobalt Ferrite (alternative cathode)
6. **CGO Electrolyte** - Ceria-Gadolinia (alternative electrolyte)

## 🔬 Properties Included

### Thermal Properties
- **Thermal Expansion Coefficient (TEC)** - Temperature-dependent expansion (1/K)
- **Thermal Conductivity** - Heat transfer capability (W/m·K)
- **Specific Heat Capacity** - Energy storage per unit mass (J/kg·K)
- **Operating Temperature Range** - Valid temperature range (K)

### Mechanical Properties
- **Young's Modulus** - Elastic stiffness (GPa)
- **Poisson's Ratio** - Lateral strain response (dimensionless)
- **Density** - Mass per unit volume (kg/m³)
- **Ultimate Tensile Strength** - Maximum stress before failure (MPa)
- **Yield Strength** - Onset of plastic deformation (MPa)

### Creep Parameters (Norton-Bailey Model)
The creep strain rate follows: **ε̇ = B × σⁿ × exp(-Q/RT)**

- **B** - Pre-exponential factor (1/Pa^n·s)
- **n** - Stress exponent (dimensionless)
- **Q** - Activation energy (J/mol)

### Plasticity Parameters (Johnson-Cook Model)
For metallic materials (Ni-YSZ, Crofer22):

- **A** - Initial yield stress (MPa)
- **B** - Hardening constant (MPa)
- **n** - Hardening exponent
- **C** - Strain rate constant
- **m** - Thermal softening exponent

### Porosity Characteristics
- **Porosity** - Volume fraction of pores (0-1)
- **Mean Pore Size** - Average pore diameter (μm)
- **Tortuosity** - Path complexity factor (dimensionless)

### Electrochemical Properties
- **Ionic Conductivity** - Ion transport capability (S/m)
- **Electronic Conductivity** - Electron transport capability (S/m)
- **Activation Overpotential** - Energy barrier for reactions (V)
- **Exchange Current Density** - Reaction kinetics parameter (A/m²)
- **Activation Energies** - Temperature dependence of conductivities (J/mol)

## 📁 File Formats

The dataset is available in multiple formats:

- **`sofc_materials_dataset.csv`** - Tabular format for spreadsheet applications
- **`sofc_materials_dataset.json`** - Structured format for programming applications
- **`sofc_material_properties.py`** - Python generator with full data structure

## 🚀 Usage Examples

### Python Usage

```python
import pandas as pd
import json

# Load CSV data
df = pd.read_csv('sofc_materials_dataset.csv')

# Display Ni-YSZ properties
ni_ysz = df[df['Material_ID'] == 'ni_ysz_anode']
print(ni_ysz[['Name', 'TEC_1/K', 'Youngs_Modulus_GPa', 'Porosity']])

# Load JSON data for programmatic access
with open('sofc_materials_dataset.json', 'r') as f:
    materials = json.load(f)

# Access specific material properties
anode = materials['ni_ysz_anode']
print(f"Anode TEC: {anode['thermal']['thermal_expansion_coefficient']} 1/K")
print(f"Anode Porosity: {anode['porosity']['porosity']}")
```

### Using the Python Generator

```python
from sofc_material_properties import SOFCDatasetGenerator

# Create generator instance
generator = SOFCDatasetGenerator()

# Get specific material
ni_ysz = generator.get_material('ni_ysz_anode')
print(f"Material: {ni_ysz.name}")
print(f"Young's Modulus: {ni_ysz.mechanical.youngs_modulus} GPa")

# List all materials
materials = generator.list_materials()
print(f"Available materials: {materials}")

# Export to custom format
df = generator.to_dataframe()
df.to_excel('sofc_materials.xlsx', index=False)
```

## 📊 Key Material Characteristics

| Material | TEC (×10⁻⁶ K⁻¹) | Young's Modulus (GPa) | Porosity (%) | Primary Function |
|----------|------------------|----------------------|--------------|------------------|
| Ni-YSZ | 12.5 | 45 | 35 | Anode (fuel oxidation) |
| 8YSZ | 10.8 | 200 | 5 | Electrolyte (ion transport) |
| LSM | 11.2 | 120 | 30 | Cathode (oxygen reduction) |
| Crofer22 | 11.8 | 220 | 2 | Interconnect (electrical connection) |
| LSCF | 15.8 | 95 | 32 | Alternative cathode |
| CGO | 12.8 | 180 | 4 | Alternative electrolyte |

## 🔧 Applications

This dataset is suitable for:

- **Finite Element Analysis (FEA)** - Structural and thermal simulations
- **Computational Fluid Dynamics (CFD)** - Mass and heat transfer modeling
- **Electrochemical Modeling** - Performance prediction and optimization
- **Materials Research** - Property correlation and design studies
- **SOFC Stack Design** - Component selection and optimization

## 📚 Data Sources and Validation

The properties are based on:
- Peer-reviewed literature values
- Experimental measurements from SOFC research
- Industry standard material specifications
- Validated computational models

**Note**: Properties may vary with temperature, microstructure, and processing conditions. Always validate critical properties for specific applications.

## 🔄 Data Updates

The dataset can be easily extended or modified using the Python generator:

```python
# Add new material
new_material = SOFCMaterial(
    name="Custom Material",
    composition="Custom Composition",
    # ... define all properties
)

generator.materials['custom_material'] = new_material
```

## 📖 References

Key literature sources for material properties:
1. Fuel Cell Handbook (EG&G Technical Services, 2004)
2. High Temperature Solid Oxide Fuel Cells (Singhal & Kendall, 2003)
3. Journal of Power Sources - SOFC materials research
4. Solid State Ionics - Electrochemical properties
5. Materials Science and Engineering reports

## 📄 License

This dataset is provided for research and educational purposes. Please cite appropriately when used in publications.

---

**Generated by**: SOFC Material Properties Dataset Generator  
**Version**: 1.0  
**Date**: October 2025