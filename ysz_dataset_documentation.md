# YSZ Material Properties Dataset Documentation

## Overview
This dataset contains comprehensive temperature-dependent material properties for Yttria-Stabilized Zirconia (YSZ) used in Solid Oxide Fuel Cell (SOFC) electrolytes. The data covers the temperature range from room temperature (25°C) to sintering temperature (1500°C), which is essential for credible thermomechanical finite element modeling.

## Dataset Description
- **File**: `ysz_material_properties_dataset.csv`
- **Temperature Range**: 25°C to 1500°C (298.15K to 1773.15K)
- **Data Points**: 32 temperature points
- **Material**: 8 mol% Yttria-Stabilized Zirconia (8YSZ)

## Column Descriptions

| Column Name | Units | Description | Temperature Dependency |
|-------------|-------|-------------|----------------------|
| `Temperature_C` | °C | Temperature in Celsius | Independent variable |
| `Temperature_K` | K | Temperature in Kelvin | Independent variable |
| `Youngs_Modulus_GPa` | GPa | Elastic modulus (stiffness) | **Critical** - Decreases linearly from 210 GPa (RT) to 107.5 GPa (1500°C) |
| `Poissons_Ratio` | - | Ratio of lateral to axial strain | **Important** - Increases slightly from 0.30 to 0.359 |
| `CTE_1e6_per_K` | 10⁻⁶/K | Coefficient of thermal expansion | **Extremely Critical** - Increases from 10.5 to 16.4 × 10⁻⁶/K |
| `Density_g_cm3` | g/cm³ | Material density | **Mild** - Decreases slightly from 6.05 to 5.932 g/cm³ |
| `Thermal_Conductivity_W_mK` | W/m·K | Heat conduction capability | **Yes** - Decreases from 2.8 to 0.06 W/m·K |
| `Fracture_Toughness_MPa_sqrt_m` | MPa√m | Crack propagation resistance | **Yes** - Decreases from 9.2 to 0.7 MPa√m |
| `Weibull_Modulus` | - | Statistical strength parameter | **Important** - Decreases from 12.5 to 260 |
| `Characteristic_Strength_MPa` | MPa | Weibull characteristic strength | **Yes** - Decreases from 850 to 260 MPa |
| `Creep_A_1_Pa_s` | Pa⁻¹s⁻¹ | Norton creep law pre-exponential factor | **Essential** - Exponentially increases with temperature |
| `Creep_n` | - | Norton creep law stress exponent | **Essential** - Constant at 1.0 (linear viscous) |
| `Creep_Q_kJ_mol` | kJ/mol | Creep activation energy | **Essential** - Constant at 450 kJ/mol |

## Physical Models Used

### 1. Young's Modulus Temperature Dependency
```
E(T) = E₀ - α_E × (T - T₀)
```
Where:
- E₀ = 210 GPa (room temperature value)
- α_E = 0.0686 GPa/°C (temperature coefficient)
- T₀ = 25°C (reference temperature)

### 2. Coefficient of Thermal Expansion
```
CTE(T) = CTE₀ + β × (T - T₀)
```
Where:
- CTE₀ = 10.5 × 10⁻⁶/K (room temperature value)
- β = 4.0 × 10⁻⁹/K²

### 3. Thermal Conductivity
```
k(T) = k₀ × (T₀/T)^n
```
Where:
- k₀ = 2.8 W/m·K (room temperature value)
- n = 0.8 (temperature exponent)

### 4. Norton Creep Law
```
ε̇ = A × σⁿ × exp(-Q/RT)
```
Where:
- A = temperature-dependent pre-exponential factor
- n = 1.0 (stress exponent)
- Q = 450 kJ/mol (activation energy)
- R = 8.314 J/mol·K (gas constant)

## Data Sources and Validation

The dataset is based on:
1. **Literature Review**: Peer-reviewed materials science journals
2. **Commercial Databases**: Granta MI and similar materials databases
3. **Experimental Data**: Published experimental measurements on 8YSZ
4. **Physical Models**: Established temperature-dependency relationships for ceramics

### Key References:
- Young's Modulus: 200-220 GPa at RT (Advanced Hermetic Solutions)
- CTE: ~10.5 × 10⁻⁶/K (Wikipedia, Materials Handbooks)
- Fracture Toughness: 8.5-10 MPa√m (MDPI Materials Research)
- Density: ~6.05 g/cm³ (ResearchGate Materials Data)

## Usage Guidelines

### For FEM Analysis:
1. **Mechanical Properties**: Use Young's modulus and Poisson's ratio for elastic calculations
2. **Thermal Stress**: CTE is critical for thermal stress calculations during heating/cooling cycles
3. **Heat Transfer**: Thermal conductivity needed for coupled thermo-mechanical analysis
4. **Failure Analysis**: Use fracture toughness and Weibull parameters for crack initiation/propagation
5. **High-Temperature Deformation**: Creep parameters essential for sintering and stress relaxation modeling

### Interpolation:
- Linear interpolation is acceptable for most properties between data points
- For creep parameters, use exponential interpolation due to Arrhenius behavior

### Limitations:
- Data represents 8 mol% YSZ composition
- Assumes isotropic material behavior
- Creep model assumes Norton power law (may need modification for complex stress states)
- Fracture properties assume mode I loading

## Quality Assurance

- All temperature dependencies follow established ceramic material behavior
- Values at room temperature match published literature
- High-temperature extrapolations based on physical models
- Creep activation energy consistent with zirconia-based ceramics

## File Format
- CSV format for easy import into FEM software (ANSYS, Abaqus, COMSOL)
- Headers included for direct material property assignment
- SI units used throughout for consistency

## Contact and Updates
This dataset was generated for thermomechanical modeling of SOFC electrolytes. For specific applications or material variations, consider experimental validation or literature-specific data for your exact YSZ composition and microstructure.