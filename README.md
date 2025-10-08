# YSZ Material Properties Dataset for FEM Thermomechanical Analysis

## Overview

This dataset provides comprehensive, temperature-dependent material properties for **Yttria-Stabilized Zirconia (YSZ)**, specifically designed for Finite Element Method (FEM) thermomechanical simulations of Solid Oxide Fuel Cell (SOFC) electrolytes.

## ⚠️ Important Disclaimer

**This is a FABRICATED dataset** synthesized from typical literature values and engineering correlations. While based on realistic trends for 8YSZ (8 mol% Y₂O₃-ZrO₂), **these values should NOT be used for production design or critical engineering decisions without experimental validation**.

For actual FEM modeling, you should:
1. Obtain certified material data from your material supplier
2. Conduct experimental characterization (tensile tests, dilatometry, thermal analysis)
3. Verify data against peer-reviewed literature
4. Use commercial databases (e.g., Granta MI, MPDB)

## Dataset Contents

### 1. `ysz_material_properties.csv`
Main temperature-dependent properties from 25°C to 1500°C:

| Property | Symbol | Units | Description |
|----------|--------|-------|-------------|
| **Young's Modulus** | E | GPa | Elastic stiffness (decreases with T) |
| **Poisson's Ratio** | ν | - | Lateral/axial strain ratio (~0.31-0.37) |
| **Coefficient of Thermal Expansion** | α | 10⁻⁶ K⁻¹ | Thermal strain per degree (~10-13) |
| **Density** | ρ | kg/m³ | Mass density (~6050 at RT) |
| **Thermal Conductivity** | k | W/m·K | Heat conduction ability (~2.7 at RT) |
| **Fracture Toughness** | K_IC | MPa·m⁰·⁵ | Crack resistance (~1.2 at RT) |
| **Creep Exponent** | n | - | Stress exponent for creep law |
| **Creep Activation Energy** | Q | kJ/mol | Thermal activation barrier for creep |

**Temperature Range:** 25°C to 1500°C (16 data points)

### 2. `weibull_parameters.csv`
Statistical strength parameters for probabilistic failure analysis:

| Property | Symbol | Units | Description |
|----------|--------|-------|-------------|
| **Weibull Modulus** | m | - | Shape parameter (~10.5 at RT, decreases with T) |
| **Characteristic Strength** | σ₀ | MPa | Scale parameter (~420 MPa at RT) |
| **Mean Strength** | σ_mean | MPa | Average tensile strength |
| **Standard Deviation** | σ_std | MPa | Strength variability |

**Temperature Range:** 25°C to 1400°C (8 data points)

### 3. `creep_model_parameters.csv`
Power-law creep model parameters:

```
ε̇ = A · σⁿ · d⁻ᵐ · exp(-Q/RT)
```

Where:
- `ε̇` = creep strain rate (s⁻¹)
- `σ` = applied stress (Pa)
- `d` = grain size (μm)
- `T` = absolute temperature (K)
- `A` = pre-exponential factor (2.5×10⁻¹⁵ Pa⁻ⁿ·s⁻¹)
- `n` = stress exponent (2.5, varies with T)
- `m` = grain size exponent (2.0)
- `Q` = activation energy (380 kJ/mol)

## Usage

### Basic Python Usage

```python
from material_properties_loader import YSZMaterialProperties

# Initialize the loader
ysz = YSZMaterialProperties()

# Get property at specific temperature
E_800C = ysz.get_property('Youngs_Modulus_GPa', 800)
print(f"Young's Modulus at 800°C: {E_800C:.2f} GPa")

# Get all properties at once
props_600C = ysz.get_all_properties(600)

# Calculate creep rate
creep_rate = ysz.get_creep_rate(
    stress_mpa=50, 
    temperature_c=1000, 
    grain_size_um=2.0
)

# Generate plots
ysz.plot_properties(save_path='properties.png')

# Export for FEM
ysz.export_for_fem('fem_input.csv')
```

### Installation Requirements

```bash
pip install pandas numpy scipy matplotlib
```

### Running the Example

```bash
python material_properties_loader.py
```

This will:
1. Load all datasets
2. Print property summaries at 25°C and 800°C
3. Generate visualization plots
4. Export FEM-ready CSV file

## Key Features of the Data

### 1. **Temperature Dependencies**

| Property | Trend | Significance |
|----------|-------|--------------|
| Young's Modulus | ↓ 205 → 40 GPa | Material softens at high T → reduced thermal stress |
| CTE | ↑ 10.2 → 13.2 (10⁻⁶/K) | Increased expansion → higher thermal strain |
| Thermal Conductivity | ↓ 2.7 → 1.9 W/m·K | Reduced heat transfer at high T |
| Fracture Toughness | ↓ 1.2 → 0.56 MPa·m⁰·⁵ | More brittle at elevated T |
| Weibull Modulus | ↓ 10.5 → 5.5 | Greater strength scatter at high T |

### 2. **Interpolation**
The Python loader uses **cubic spline interpolation** for smooth property curves between data points. Extrapolation is enabled but should be used cautiously.

### 3. **Creep Modeling**
High-temperature creep becomes significant above ~800°C and is critical for:
- Stress relaxation during sintering
- Long-term dimensional stability
- Crack tip stress redistribution

## FEM Implementation Tips

### For ANSYS Users
1. Define material using `MPTEMP` and `MPDATA` commands
2. Use `TB,CREEP` for creep parameters
3. Define `TB,PRONY` for viscoelastic behavior if needed

### For COMSOL Users
1. Import CSV directly into Material Library
2. Use "Interpolation Function" for temperature dependency
3. Enable "Creep" physics module for viscoplastic analysis

### For Abaqus Users
1. Use `*MATERIAL` and `*ELASTIC, TYPE=ISOTROPIC`
2. Define `*EXPANSION` for CTE
3. Use `*CREEP, LAW=POWER` with rate-dependent data

## Data Validation Checklist

Before using this data in production:

- [ ] Verify Young's modulus matches your specific YSZ composition (3YSZ, 8YSZ, 10YSZ differ)
- [ ] Confirm CTE with dilatometry data from your supplier
- [ ] Check thermal conductivity against sintered density (porosity reduces k)
- [ ] Validate fracture toughness with your microstructure (grain size dependent)
- [ ] Obtain Weibull parameters from flexural strength tests (minimum 30 samples)
- [ ] Characterize creep with constant-load tensile tests at operating temperatures
- [ ] Account for atmosphere effects (O₂ partial pressure affects defect chemistry)

## References & Data Sources

This synthesized dataset is inspired by typical values from:

1. **Mechanical Properties:**
   - Atkinson & Selçuk (2000), *J. Euro. Ceram. Soc.* - Elastic modulus of YSZ
   - Tsoga & Nikolopoulos (1996) - Thermal expansion of YSZ
   
2. **Fracture & Statistics:**
   - Weil & Kraft (1988) - Weibull statistics for ceramics
   - Kendall et al. (1986) - Fracture toughness of zirconia

3. **Creep Behavior:**
   - Jiménez-Melendo et al. (1998), *J. Am. Ceram. Soc.* - Creep of YSZ
   - Lakki et al. (2000) - High-temperature mechanical relaxation

**Note:** Actual citations are illustrative. Real implementation requires thorough literature review.

## License

This dataset is provided for **educational and demonstration purposes only**. 

MIT License - Use at your own risk. No warranty provided.

## Contributing

If you have experimentally validated data for YSZ that you can share:
1. Fork this repository
2. Add data with proper citations
3. Submit a pull request with validation documentation

## Contact

For questions about FEM implementation or data validation:
- Open an issue in this repository
- Provide context about your specific SOFC application

---

**Last Updated:** October 2025  
**Dataset Version:** 1.0  
**Material:** 8YSZ (8 mol% Y₂O₃-stabilized ZrO₂)