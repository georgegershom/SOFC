# SOFC Material Properties Dataset

## Overview
This dataset contains comprehensive thermo-physical, mechanical, and electrochemical properties for Solid Oxide Fuel Cell (SOFC) components. The data is based on typical literature values and industry standards for SOFC materials operating at high temperatures (800-1000°C).

## Dataset Contents

### 🔩 1. Material Properties Included

#### **Anode Materials**
- **Ni-YSZ (Nickel - Yttria Stabilized Zirconia composite)**
  - Most common SOFC anode material
  - 30-40% porosity for gas diffusion
  - Includes complete mechanical and electrochemical characterization

#### **Electrolyte Materials**
- **8YSZ (8 mol% Yttria-Stabilized Zirconia)**
  - Standard high-temperature electrolyte
  - Dense, gas-tight structure (<5% porosity)
  - Pure ionic conductor
  
- **CGO (Gadolinium-Doped Ceria)**
  - Intermediate temperature electrolyte
  - Higher ionic conductivity than YSZ at lower temperatures

#### **Cathode Materials**
- **LSM (Lanthanum Strontium Manganite - La₀.₈Sr₀.₂MnO₃)**
  - Traditional high-temperature cathode
  - Electronic conductor, requires YSZ for TPB
  
- **LSM-YSZ Composite**
  - Enhanced triple-phase boundary (TPB)
  - Improved electrochemical performance
  
- **LSCF (Lanthanum Strontium Cobalt Ferrite)**
  - Mixed ionic-electronic conductor (MIEC)
  - For intermediate temperature operation

#### **Interconnect Materials**
- **Crofer 22 APU**
  - Ferritic stainless steel
  - Thermal expansion matched to ceramics

---

## Properties Included

### 1. **Thermo-Physical Properties**
| Property | Description | Units |
|----------|-------------|-------|
| Thermal Expansion Coefficient (TEC) | Linear thermal expansion | 10⁻⁶ K⁻¹ |
| Thermal Conductivity | Heat conduction capability | W/(m·K) |
| Specific Heat Capacity | Heat storage capacity | J/(kg·K) |
| Density | Mass per unit volume | kg/m³ |
| Porosity | Void fraction | dimensionless |

### 2. **Mechanical Properties**
| Property | Description | Units |
|----------|-------------|-------|
| Young's Modulus | Elastic stiffness | GPa |
| Poisson's Ratio | Lateral strain ratio | dimensionless |
| **Norton-Bailey Creep Parameters** | | |
| B | Pre-exponential factor | Pa⁻ⁿ s⁻¹ |
| n | Stress exponent | dimensionless |
| Q | Activation energy | kJ/mol |
| **Johnson-Cook Plasticity** (Ni-YSZ) | | |
| A | Initial yield stress | MPa |
| B | Hardening modulus | MPa |
| n | Hardening exponent | dimensionless |
| C | Strain rate sensitivity | dimensionless |
| m | Thermal softening | dimensionless |

### 3. **Electrochemical Properties**
| Property | Description | Units |
|----------|-------------|-------|
| Electronic Conductivity | Electron transport | S/m |
| Ionic Conductivity | Ion (O²⁻) transport | S/m |
| Exchange Current Density | Reaction kinetics | A/m² |
| Activation Overpotential Coefficient | Charge transfer coefficient (α) | dimensionless |
| Activation Energy | Temperature dependence | kJ/mol |

---

## File Formats

### 1. **CSV Format** (`sofc_material_properties.csv`)
- Flat table format
- Easy to import into Excel, MATLAB, or other tools
- Column headers: Component, Layer, Property, Value, Unit, Temperature_Range_K, Notes

### 2. **JSON Format** (`sofc_material_properties.json`)
- Hierarchical structure
- Nested by component → material → property category
- Ideal for Python/JavaScript applications
- Includes metadata and notes

### 3. **Python Module** (`sofc_material_properties.py`)
- Object-oriented database class
- Built-in calculation methods:
  - Temperature-dependent conductivity (Arrhenius equation)
  - Creep rate calculation (Norton-Bailey law)
  - Easy property lookup
- Data classes for type safety
- Export to pandas DataFrame

---

## Usage Examples

### Python Usage

```python
from sofc_material_properties import SOFCMaterialDatabase

# Initialize database
db = SOFCMaterialDatabase()

# Get material properties
ni_ysz = db.get_material_properties('anode', 'Ni-YSZ')
print(f"TEC: {ni_ysz['thermo_physical'].thermal_expansion_coefficient}")

# Calculate temperature-dependent ionic conductivity
sigma = db.get_ionic_conductivity('electrolyte', '8YSZ', temperature=1073)
print(f"Ionic conductivity at 1073K: {sigma:.3f} S/m")

# Calculate creep strain rate
creep_rate = db.calculate_creep_rate('anode', 'Ni-YSZ', 
                                     stress=50e6,  # 50 MPa
                                     temperature=1073)
print(f"Creep rate: {creep_rate:.3e} 1/s")

# Export to pandas DataFrame
df = db.export_to_dataframe()
df.to_csv('my_export.csv')
```

### MATLAB/Simulink Usage

```matlab
% Load CSV data
data = readtable('sofc_material_properties.csv');

% Filter for specific material
ni_ysz_data = data(strcmp(data.Layer, 'Ni-YSZ'), :);

% Extract specific property
tec = data.Value(strcmp(data.Property, 'Thermal_Expansion_Coefficient') & ...
                 strcmp(data.Layer, 'Ni-YSZ'));
```

### COMSOL Multiphysics

1. Import CSV directly into COMSOL material database
2. Use property values in physics modules:
   - Heat Transfer: thermal conductivity, specific heat, density
   - Solid Mechanics: Young's modulus, Poisson's ratio, creep parameters
   - Electrochemistry: ionic/electronic conductivity

### ANSYS

Import material properties for:
- Structural analysis (thermal-mechanical coupling)
- Creep analysis using Norton-Bailey law
- Thermal expansion mismatch studies

---

## Temperature Dependencies

Many properties are temperature-dependent. The dataset includes:

### Ionic Conductivity (Arrhenius Equation)
```
σ(T) = σ₀ × exp(-Eₐ / (R × T))
```
where:
- σ₀ = reference conductivity (at reference temperature)
- Eₐ = activation energy
- R = gas constant (8.314 J/(mol·K))
- T = absolute temperature (K)

### Creep Strain Rate (Norton-Bailey Law)
```
ε̇ = B × σⁿ × exp(-Q / (R × T))
```
where:
- B = pre-exponential factor
- σ = applied stress
- n = stress exponent
- Q = activation energy for creep
- T = absolute temperature

---

## Important Notes

### Typical Operating Conditions
- **High-Temperature SOFC**: 800-1000°C (1073-1273 K)
- **Intermediate Temperature SOFC**: 600-800°C (873-1073 K)

### Material Matching Considerations
1. **Thermal Expansion Coefficient (TEC) Matching**
   - Critical to prevent delamination and cracking
   - Ni-YSZ (12.5) ≈ Crofer (12.0) ≈ LSM (11.8) ≈ 8YSZ (10.5)
   - LSCF (14.5) has higher TEC → requires careful integration

2. **Porosity Requirements**
   - Anode: 30-40% for H₂ diffusion
   - Cathode: 30-35% for O₂ diffusion
   - Electrolyte: <5% for gas impermeability

3. **Triple-Phase Boundary (TPB)**
   - Critical for electrochemical reactions
   - Composite materials (LSM-YSZ) enhance TPB length

### Data Sources & Validation
This dataset is fabricated based on:
- Literature values from peer-reviewed publications
- Typical manufacturer specifications
- Standard SOFC material databases
- Values represent room temperature to 1273K range

**⚠️ Important**: For critical applications, validate these values with:
- Experimental measurements
- Manufacturer datasheets
- Application-specific testing

---

## Applications

This dataset is suitable for:

✅ **Computational Modeling**
- Finite Element Analysis (FEA)
- Computational Fluid Dynamics (CFD)
- Multiphysics simulations

✅ **Design & Optimization**
- Stack design
- Material selection
- Thermal management

✅ **Research & Development**
- Baseline comparison studies
- Parametric analysis
- Proof-of-concept modeling

✅ **Educational Purposes**
- Teaching material
- Course projects
- Academic research

---

## Citation

If you use this dataset in your research, please cite as:

```
SOFC Material Properties Dataset (2025)
Comprehensive thermo-physical, mechanical, and electrochemical properties
for Solid Oxide Fuel Cell components
```

---

## File Structure

```
.
├── README.md                          # This file
├── sofc_material_properties.csv      # Flat CSV format
├── sofc_material_properties.json     # Hierarchical JSON format
├── sofc_material_properties.py       # Python database module
└── sofc_materials_summary.csv        # Generated summary table
```

---

## Updates & Maintenance

**Version**: 1.0  
**Date**: October 2025  
**Status**: Fabricated dataset for research purposes

For questions, corrections, or additions, please refer to the latest SOFC literature or contact material manufacturers directly.

---

## License

This dataset is provided as-is for research and educational purposes. No warranty is provided for the accuracy or applicability of the data for any specific use case.

---

## References

Key material systems referenced:
- Ni-YSZ: Standard SOFC anode cermet
- 8YSZ: 8 mol% Y₂O₃-ZrO₂ electrolyte
- LSM: La₀.₈Sr₀.₂MnO₃ cathode
- LSCF: La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃ MIEC cathode
- CGO: Ce₀.₉Gd₀.₁O₂ intermediate temp electrolyte
- Crofer 22 APU: Commercial ferritic steel interconnect

For detailed references, consult recent SOFC review articles and handbooks.
