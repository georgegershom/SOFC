# YSZ Material Properties Dataset for FEM Analysis

## Overview
This dataset contains comprehensive material properties for **8 mol% Yttria-Stabilized Zirconia (8YSZ)** used in Solid Oxide Fuel Cell (SOFC) electrolytes. The data is specifically formatted for thermomechanical finite element modeling.

## Material Composition
- **Material**: ZrO₂-8Y₂O₃ (8 mol% Yttria content)
- **Application**: SOFC electrolyte
- **Temperature Range**: 25°C to 1500°C (RT to sintering temperature)

## Key Properties Included

### 1. Mechanical Properties
- **Young's Modulus (E)**: 210 GPa (RT) → 170 GPa (1500°C)
- **Poisson's Ratio (ν)**: 0.31 (assumed constant)
- **Density (ρ)**: 6.08 g/cm³ (RT) → 5.92 g/cm³ (1500°C)

### 2. Thermal Properties
- **Coefficient of Thermal Expansion (CTE)**: 10.5×10⁻⁶/K (RT) → 13.5×10⁻⁶/K (1500°C)
- **Thermal Conductivity (k)**: 2.2 W/m·K (RT) → 1.4 W/m·K (1500°C)
- **Specific Heat**: 480 J/kg·K (assumed constant)

### 3. Fracture Properties
- **Fracture Toughness (K_IC)**: 2.5 MPa·m^0.5 (RT) → 0.9 MPa·m^0.5 (1500°C)
- **Weibull Modulus (m)**: 12 (RT), 8 (high temperature)
- **Characteristic Strength (σ₀)**: 400 MPa (RT), 200 MPa (high temperature)

### 4. Creep Properties
- **Activation Energy (Q)**: 450 kJ/mol
- **Creep Exponent (n)**: 2.5
- **Pre-exponential Factor (A)**: 1.2×10⁻⁸ s⁻¹·MPa⁻ⁿ

## File Formats Available

1. **YSZ_Material_Properties_Dataset.csv** - Comprehensive CSV format with all properties
2. **YSZ_Material_Properties_ANSYS.txt** - ANSYS APDL format
3. **YSZ_Material_Properties_Abaqus.inp** - Abaqus input format

## Usage Guidelines

### For Thermomechanical Analysis
- Use temperature-dependent properties for accurate stress calculations
- Consider creep effects for sintering and high-temperature operation
- Apply appropriate failure criteria based on Weibull statistics

### Temperature Ranges
- **Operating**: 600-1000°C (typical SOFC operation)
- **Sintering**: 1400-1600°C (manufacturing process)
- **Cooling**: 2-5°C/min (recommended cooling rate)

### Critical Considerations
- **Thermal Stress**: CTE variation is crucial for thermal stress calculations
- **Creep**: Essential for sintering simulation and stress relaxation
- **Fracture**: Use Weibull parameters for probabilistic failure analysis
- **Phase Transitions**: Consider tetragonal-cubic transition at 2370°C

## Data Sources
- Literature values from materials science journals
- Typical values for 8YSZ composition
- Representative of commercial SOFC materials

## Validation Notes
- Properties are based on typical literature values
- Temperature dependencies follow expected trends
- Suitable for preliminary FEM analysis
- For critical applications, verify with experimental data

## Software Compatibility
- ANSYS Mechanical APDL
- Abaqus/Standard
- COMSOL Multiphysics
- Any FEM software accepting CSV input

## Contact
For questions about this dataset or custom material property requirements, refer to the original literature sources or conduct experimental validation.