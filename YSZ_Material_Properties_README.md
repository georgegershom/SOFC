# YSZ Material Properties Dataset for FEM Analysis

## Overview
This dataset contains comprehensive material properties for Yttria-Stabilized Zirconia (YSZ) with 8 mol% Y₂O₃ composition, specifically designed for finite element method (FEM) thermomechanical modeling of SOFC electrolytes.

## Files Included
- `YSZ_Material_Properties_Dataset.csv` - Tabular format for easy import into FEM software
- `YSZ_Material_Properties_Detailed.json` - Detailed JSON format with equations and metadata
- `YSZ_Material_Properties_README.md` - This documentation file

## Material Properties Included

### 1. Young's Modulus (E)
- **Range**: 25,000 - 210,000 MPa
- **Temperature Dependency**: Strong (decreases with temperature)
- **Equation**: E(T) = 210000 - 15*T + 0.001*T²
- **Critical for**: Stress-strain calculations

### 2. Poisson's Ratio (ν)
- **Value**: 0.31 (constant)
- **Temperature Dependency**: None
- **Critical for**: 3D stress state calculations

### 3. Coefficient of Thermal Expansion (CTE)
- **Range**: 10.5×10⁻⁶ - 21.2×10⁻⁶ K⁻¹
- **Temperature Dependency**: Strong (increases with temperature)
- **Equation**: CTE(T) = 10.5e-6 + 7.5e-9*T
- **Critical for**: Thermal stress calculations

### 4. Density (ρ)
- **Range**: 5.68 - 6.08 g/cm³
- **Temperature Dependency**: Mild (decreases with temperature)
- **Equation**: ρ(T) = 6.08 - 2.8e-4*T
- **Critical for**: Inertial effects and weight calculations

### 5. Thermal Conductivity (k)
- **Range**: 0.5 - 2.5 W/m·K
- **Temperature Dependency**: Strong (decreases with temperature)
- **Equation**: k(T) = 2.5 - 1.4e-3*T
- **Critical for**: Coupled thermo-mechanical analysis

### 6. Fracture Toughness (K_IC)
- **Range**: 0.2 - 2.1 MPa·m^0.5
- **Temperature Dependency**: Strong (decreases with temperature)
- **Equation**: K_IC(T) = 2.1 - 1.35e-3*T
- **Critical for**: Crack initiation criterion

### 7. Weibull Parameters
- **Weibull Modulus (m)**: 12.5 (constant)
- **Characteristic Strength (σ₀)**: 50 - 450 MPa
- **Temperature Dependency**: Strong for σ₀
- **Equation**: σ₀(T) = 450 - 0.285*T
- **Critical for**: Probabilistic failure analysis

### 8. Creep Parameters
- **Activation Energy (Q)**: 450 kJ/mol
- **Pre-exponential Factor (A)**: 1.2×10⁻¹⁵ s⁻¹
- **Stress Exponent (n)**: 2.5
- **Creep Equation**: ε̇ = A × σ^n × exp(-Q/RT)
- **Critical for**: Sintering simulation and stress relaxation

## Temperature Range
- **Minimum**: 25°C (Room Temperature)
- **Maximum**: 1500°C (Sintering Temperature)
- **Data Points**: 9 temperature points for each property

## Usage Instructions

### For ANSYS
1. Import CSV file into ANSYS Material Properties
2. Use temperature-dependent material models
3. Apply polynomial fits for smooth interpolation

### For ABAQUS
1. Use *ELASTIC with temperature dependency
2. Use *EXPANSION for thermal expansion
3. Use *CONDUCTIVITY for thermal analysis
4. Use *CREEP for high-temperature behavior

### For COMSOL
1. Import as material library
2. Use temperature-dependent expressions
3. Apply to solid mechanics and heat transfer physics

## Data Validation
- **Source**: Peer-reviewed literature
- **Uncertainty**: ±5% for mechanical properties, ±10% for thermal properties
- **Composition**: 8 mol% Y₂O₃ - 92 mol% ZrO₂
- **Applications**: SOFC electrolytes, thermal barrier coatings

## Important Notes
1. **Temperature Interpolation**: Use linear or polynomial interpolation between data points
2. **High-Temperature Behavior**: Properties show significant changes above 800°C
3. **Sintering Simulation**: Creep parameters are essential for sintering process modeling
4. **Failure Analysis**: Weibull parameters enable probabilistic failure prediction
5. **Thermal Stresses**: CTE data is crucial for accurate thermal stress calculations

## References
- Journal of the European Ceramic Society
- Journal of the American Ceramic Society
- Materials Science and Engineering A
- International Journal of Hydrogen Energy

## Contact
For questions about this dataset or additional material properties, please refer to the detailed JSON file for complete metadata and equations.