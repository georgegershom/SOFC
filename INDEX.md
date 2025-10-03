# 🔋 SOFC Material Properties Dataset - Complete Index

**Version:** 1.0  
**Date:** October 3, 2025  
**Status:** Research/Educational Dataset

---

## 📁 Dataset Files

### Data Files (Core Dataset)

| File | Format | Description | Use Case |
|------|--------|-------------|----------|
| `sofc_material_properties.csv` | CSV | Flat table format with all properties | Excel, MATLAB, general import |
| `sofc_material_properties.json` | JSON | Hierarchical nested structure | Python, JavaScript, web apps |
| `sofc_material_properties.py` | Python Module | Object-oriented database with methods | Python simulations, analysis |
| `sofc_materials_summary.csv` | CSV | Condensed summary table | Quick reference |

### Documentation Files

| File | Description |
|------|-------------|
| `README.md` | Main documentation with dataset overview |
| `EQUATIONS_AND_MODELS.md` | Mathematical models and equations |
| `INDEX.md` | This file - complete project index |

### Visualization Scripts

| File | Description | Output |
|------|-------------|--------|
| `visualize_properties.py` | Python script to generate plots | 6 PNG images |

### Generated Visualizations

| File | Description |
|------|-------------|
| `tec_comparison.png` | Thermal expansion coefficient comparison |
| `conductivity_vs_temperature.png` | Temperature-dependent ionic conductivity |
| `mechanical_properties.png` | Young's modulus and Poisson's ratio |
| `creep_rates.png` | Creep strain rates vs stress |
| `porosity_comparison.png` | Porosity requirements by component |
| `electrochemical_properties.png` | Exchange current density and conductivities |

---

## 🧪 Materials Included

### Anode
- **Ni-YSZ** (Nickel - Yttria Stabilized Zirconia)
  - Porosity: 35%
  - TEC: 12.5 × 10⁻⁶ K⁻¹
  - Complete mechanical and electrochemical data

### Electrolyte
- **8YSZ** (8 mol% Y₂O₃-ZrO₂)
  - High-temperature SOFC standard
  - Dense structure (2% porosity)
  - Ionic conductivity: 3.2 S/m at 1073K

- **CGO** (Ce₀.₉Gd₀.₁O₂)
  - Intermediate temperature electrolyte
  - Higher conductivity at lower temperatures
  - Ionic conductivity: 8.5 S/m at 873K

### Cathode
- **LSM** (La₀.₈Sr₀.₂MnO₃)
  - Traditional high-temperature cathode
  - Electronic conductor
  - TEC: 11.8 × 10⁻⁶ K⁻¹

- **LSM-YSZ Composite**
  - Enhanced triple-phase boundary
  - Mixed ionic-electronic properties
  - Exchange current: 3500 A/m²

- **LSCF** (La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃)
  - Intermediate temperature cathode
  - MIEC (Mixed Ionic-Electronic Conductor)
  - Highest exchange current: 5200 A/m²

### Interconnect
- **Crofer 22 APU**
  - Ferritic stainless steel
  - TEC-matched to ceramics
  - High electronic conductivity: 1.2 × 10⁶ S/m

---

## 📊 Properties Covered

### 1. Thermo-Physical Properties
- ✅ Thermal Expansion Coefficient (TEC)
- ✅ Thermal Conductivity
- ✅ Specific Heat Capacity
- ✅ Density
- ✅ Porosity

### 2. Mechanical Properties
- ✅ Young's Modulus
- ✅ Poisson's Ratio
- ✅ **Norton-Bailey Creep Parameters** (B, n, Q)
- ✅ **Johnson-Cook Plasticity** (A, B, n, C, m) - for Ni-YSZ

### 3. Electrochemical Properties
- ✅ Electronic Conductivity
- ✅ Ionic Conductivity
- ✅ Exchange Current Density
- ✅ Activation Overpotential Coefficient
- ✅ Activation Energy (for temperature dependence)

---

## 🚀 Quick Start Guide

### Option 1: Python Analysis

```python
from sofc_material_properties import SOFCMaterialDatabase

# Initialize
db = SOFCMaterialDatabase()

# Get properties
props = db.get_material_properties('anode', 'Ni-YSZ')
print(f"TEC: {props['thermo_physical'].thermal_expansion_coefficient}")

# Calculate temperature-dependent conductivity
sigma = db.get_ionic_conductivity('electrolyte', '8YSZ', temperature=1073)

# Calculate creep rate
creep_rate = db.calculate_creep_rate('anode', 'Ni-YSZ', 
                                     stress=50e6, temperature=1073)
```

### Option 2: MATLAB/CSV Import

```matlab
% Load data
data = readtable('sofc_material_properties.csv');

% Filter specific material
ni_ysz = data(strcmp(data.Layer, 'Ni-YSZ'), :);

% Extract property
tec = data.Value(strcmp(data.Property, 'Thermal_Expansion_Coefficient') & ...
                 strcmp(data.Layer, 'Ni-YSZ'));
```

### Option 3: JSON for Web/JavaScript

```javascript
// Load JSON
fetch('sofc_material_properties.json')
  .then(response => response.json())
  .then(data => {
    const niYsz = data.sofc_material_properties.anode['Ni-YSZ'];
    console.log('TEC:', niYsz.thermo_physical.thermal_expansion_coefficient);
  });
```

---

## 📐 Mathematical Models Available

### Temperature Dependencies
- **Arrhenius Conductivity**: σ(T) = σ₀ × exp(-Eₐ/RT)
- **Nernst Equation**: E = E⁰ + (RT/nF) × ln(Q)

### Mechanical Models
- **Norton-Bailey Creep**: ε̇ = B × σⁿ × exp(-Q/RT)
- **Johnson-Cook Plasticity**: σ_y = [A + Bεⁿ][1 + C ln(ε̇*)][1 - T*ᵐ]

### Electrochemical Models
- **Butler-Volmer Kinetics**: i = i₀ × [exp(αFη/RT) - exp(-αFη/RT)]
- **Ohmic Resistance**: R = t / (σ × A)

*See `EQUATIONS_AND_MODELS.md` for complete derivations and examples.*

---

## 🎯 Use Cases

### ✅ Computational Modeling
- Finite Element Analysis (FEA)
- COMSOL Multiphysics
- ANSYS simulations
- CFD analysis

### ✅ Design & Optimization
- Stack design
- Material selection studies
- Thermal management
- Stress analysis

### ✅ Research & Development
- Parametric studies
- Sensitivity analysis
- Model validation
- Literature comparison

### ✅ Education
- Course material
- Student projects
- Tutorial examples
- Teaching demonstrations

---

## 📈 Visualization Examples

Run the visualization script to generate plots:

```bash
python3 visualize_properties.py
```

**Generated plots:**
1. **TEC Comparison** - Bar chart of thermal expansion coefficients
2. **Conductivity vs Temperature** - Arrhenius behavior curves
3. **Mechanical Properties** - Young's modulus and Poisson's ratio
4. **Creep Rates** - Norton-Bailey power law behavior
5. **Porosity Requirements** - By component type
6. **Electrochemical Properties** - Exchange currents and conductivities

---

## ⚙️ Software Compatibility

### Verified Compatible With:
- ✅ Python 3.7+
- ✅ MATLAB R2018a+
- ✅ Microsoft Excel
- ✅ COMSOL Multiphysics 5.x+
- ✅ ANSYS Mechanical/Fluent
- ✅ Pandas/NumPy/SciPy
- ✅ JavaScript (Node.js/Browser)

### Required Python Packages:
```bash
pip install numpy pandas matplotlib
```

---

## 📚 Key References

### Material Systems
- **Ni-YSZ**: Standard SOFC anode cermet (30-40% Ni by volume)
- **8YSZ**: 8 mol% Y₂O₃-stabilized ZrO₂ electrolyte
- **LSM**: La₀.₈Sr₀.₂MnO₃ perovskite cathode
- **LSCF**: La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃ MIEC cathode
- **CGO**: Ce₀.₉Gd₀.₁O₂ fluorite electrolyte
- **Crofer 22 APU**: Fe-Cr ferritic steel interconnect

### Operating Conditions
- **High-Temperature SOFC**: 800-1000°C (1073-1273 K)
- **Intermediate-Temperature SOFC**: 600-800°C (873-1073 K)
- **Typical Stack Pressure**: 1-3 bar
- **Typical Current Density**: 0.3-1.0 A/cm²

---

## ⚠️ Important Disclaimers

1. **Fabricated Data**: This dataset is generated based on typical literature values and is intended for educational and preliminary research purposes.

2. **Validation Required**: For critical applications (commercial products, safety-critical systems), validate all properties through:
   - Experimental measurements
   - Manufacturer datasheets
   - Application-specific testing

3. **Temperature Ranges**: Properties are valid within specified temperature ranges (typically 298-1273 K).

4. **Microstructure Dependency**: Many properties (especially mechanical and electrochemical) depend strongly on:
   - Manufacturing process
   - Sintering conditions
   - Particle size distribution
   - Microstructural features

5. **No Warranty**: Data provided as-is without warranty for any specific application.

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 2025 | Initial release with 7 materials, full property set |

---

## 📞 Support & Contributions

### Questions?
- Consult `README.md` for detailed documentation
- See `EQUATIONS_AND_MODELS.md` for mathematical formulations
- Review example code in `sofc_material_properties.py`

### Found an Error?
- Cross-reference with peer-reviewed literature
- Validate experimentally when possible
- Document discrepancies for your specific application

### Want to Extend?
The Python module is designed for easy extension:
- Add new materials to `_initialize_database()`
- Add new properties to dataclasses
- Implement new calculation methods

---

## 📄 License

This dataset is provided for research and educational purposes. No commercial warranty is implied or provided.

---

## 🎓 Citation Suggestion

```
SOFC Material Properties Dataset (2025)
Comprehensive thermo-physical, mechanical, and electrochemical properties
for Solid Oxide Fuel Cell components
Version 1.0
```

---

## ✨ Summary Statistics

- **Materials**: 7 (across 4 component types)
- **Properties**: 20+ per material
- **Temperature Range**: 298-1273 K
- **Data Points**: 70+ unique property values
- **Models**: 4 mathematical models with parameters
- **Visualizations**: 6 publication-ready plots

---

**Last Updated:** October 3, 2025  
**Maintained by:** Research Dataset Initiative  
**Status:** ✅ Complete and Ready for Use

---

*For the latest information and updates, refer to individual documentation files.*
