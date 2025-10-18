# Data Dictionary - Thermo-Mechanical Modeling Dataset

## Overview

This data dictionary provides comprehensive descriptions of all variables, units, and data structures in the thermo-mechanical modeling dataset for fire-resistant structural elements utilizing high-performance rubberized concrete.

## General Data Structure

### Common Fields

All datasets contain the following common fields:

| Field | Type | Description | Units | Range/Values |
|-------|------|-------------|-------|--------------|
| `Mix_ID` | String | Concrete mix identifier | - | C, R5S, R10S, R15S, R20S, R10L |
| `Temperature_C` | Integer | Temperature | °C | 20 to 800 |
| `Data_Type` | String | Data usage type | - | Calibration, Validation |
| `Property_Type` | String | Property category | - | Thermal, Mechanical, Transport, Deformation |
| `Rubber_Content_Percent` | Float | Rubber content by volume | % | 0 to 20 |
| `Particle_Size` | String | Rubber particle size | - | small, large |

### Mix Type Definitions

| Mix ID | Description | Rubber Content | Particle Size |
|--------|-------------|----------------|---------------|
| C | Control concrete | 0% | - |
| R5S | Rubberized concrete | 5% | Small |
| R10S | Rubberized concrete | 10% | Small |
| R15S | Rubberized concrete | 15% | Small |
| R20S | Rubberized concrete | 20% | Small |
| R10L | Rubberized concrete | 10% | Large |

## Thermal Properties

### thermal_conductivity.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Thermal_Conductivity_Mean_W_m_K` | Float | Mean thermal conductivity | W/m·K | 0.8 - 2.0 |
| `Thermal_Conductivity_Std_W_m_K` | Float | Standard deviation | W/m·K | 0.08 - 0.20 |

**Physical Basis:** Temperature-dependent degradation due to phonon scattering and microcracking. Rubber content reduces conductivity due to lower intrinsic conductivity of rubber particles.

### specific_heat.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Specific_Heat_Mean_J_kg_K` | Float | Mean specific heat capacity | J/kg·K | 880 - 1400 |
| `Specific_Heat_Std_J_kg_K` | Float | Standard deviation | J/kg·K | 70 - 112 |

**Physical Basis:** Increases with temperature due to lattice vibrations. Rubber content slightly increases specific heat.

### thermal_diffusivity.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Thermal_Diffusivity_Mean_mm2_s` | Float | Mean thermal diffusivity | mm²/s | 0.3 - 0.8 |
| `Thermal_Diffusivity_Std_mm2_s` | Float | Standard deviation | mm²/s | 0.03 - 0.08 |
| `Density_kg_m3` | Float | Material density | kg/m³ | 1560 - 2400 |

**Physical Basis:** Calculated as α = k/(ρ·cp). Decreases with rubber content due to lower density and conductivity.

### thermal_expansion.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Thermal_Expansion_Coeff_Mean_per_K` | String (Scientific) | Mean expansion coefficient | K⁻¹ | 1.2e-5 - 2.0e-5 |
| `Thermal_Expansion_Coeff_Std_per_K` | String (Scientific) | Standard deviation | K⁻¹ | 1.4e-6 - 2.4e-6 |

**Physical Basis:** Increases with temperature and rubber content. Rubber has higher thermal expansion than concrete matrix.

## Mechanical Properties

### compressive_strength.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Compressive_Strength_Mean_MPa` | Float | Mean compressive strength | MPa | 15 - 52 |
| `Compressive_Strength_Std_MPa` | Float | Standard deviation | MPa | 1.8 - 6.2 |

**Physical Basis:** Initially increases slightly up to 200°C, then decreases due to dehydration and microcracking. Rubber content reduces strength.

### tensile_strength.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Tensile_Strength_Mean_MPa` | Float | Mean tensile strength | MPa | 1.2 - 4.4 |
| `Tensile_Strength_Std_MPa` | Float | Standard deviation | MPa | 0.18 - 0.66 |

**Physical Basis:** Degrades more rapidly than compressive strength with temperature. Rubber content reduces tensile strength.

### elastic_modulus.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Elastic_Modulus_Mean_MPa` | Float | Mean elastic modulus | MPa | 2400 - 32000 |
| `Elastic_Modulus_Std_MPa` | Float | Standard deviation | MPa | 240 - 3200 |

**Physical Basis:** Decreases almost linearly with temperature. Rubber content significantly reduces modulus due to lower stiffness of rubber.

### poissons_ratio.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Poissons_Ratio_Mean` | Float | Mean Poisson's ratio | - | 0.18 - 0.28 |
| `Poissons_Ratio_Std` | Float | Standard deviation | - | 0.014 - 0.022 |

**Physical Basis:** Increases slightly with temperature. Rubber content increases Poisson's ratio due to higher compressibility.

### fracture_properties.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Fracture_Toughness_Mean_MPa_m05` | Float | Mean fracture toughness | MPa·m^0.5 | 1.0 - 1.8 |
| `Fracture_Toughness_Std_MPa_m05` | Float | Standard deviation | MPa·m^0.5 | 0.18 - 0.32 |
| `Fracture_Energy_Mean_J_m2` | Float | Mean fracture energy | J/m² | 100 - 220 |
| `Fracture_Energy_Std_J_m2` | Float | Standard deviation | J/m² | 20 - 44 |

**Physical Basis:** Rubber improves fracture toughness through crack bridging and energy dissipation mechanisms.

## Transport Properties

### permeability.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Intrinsic_Permeability_Mean_m2` | String (Scientific) | Mean intrinsic permeability | m² | 1.2e-18 - 8.5e-17 |
| `Intrinsic_Permeability_Std_m2` | String (Scientific) | Standard deviation | m² | 3.0e-19 - 2.1e-17 |
| `Relative_Permeability` | Float | Relative to water | - | 1.2 - 85 |

**Physical Basis:** Increases with temperature due to microcracking and dehydration. Rubber content increases permeability through interfacial zones.

### porosity.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Total_Porosity_Mean` | Float | Mean total porosity | - | 0.12 - 0.35 |
| `Total_Porosity_Std` | Float | Standard deviation | - | 0.018 - 0.053 |
| `Effective_Porosity_Mean` | Float | Mean effective porosity | - | 0.08 - 0.28 |
| `Effective_Porosity_Std` | Float | Standard deviation | - | 0.014 - 0.050 |
| `Connectivity_Factor` | Float | Effective/Total porosity | - | 0.67 - 0.80 |

**Physical Basis:** Increases with temperature due to dehydration and thermal decomposition. Rubber creates additional porosity.

### moisture_transport.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Moisture_Diffusivity_Mean_m2_s` | String (Scientific) | Mean moisture diffusivity | m²/s | 2.5e-12 - 3.8e-10 |
| `Moisture_Diffusivity_Std_m2_s` | String (Scientific) | Standard deviation | m²/s | 5.5e-13 - 8.4e-11 |
| `Sorption_Capacity_Mean_kg_kg` | Float | Mean sorption capacity | kg/kg | 0.008 - 0.055 |
| `Sorption_Capacity_Std_kg_kg` | Float | Standard deviation | kg/kg | 0.0014 - 0.0099 |
| `Moisture_Permeability_kg_m_s_Pa` | String (Scientific) | Combined transport parameter | kg/m/s/Pa | 2.0e-14 - 2.1e-11 |

**Physical Basis:** Diffusivity follows Arrhenius relationship. Sorption decreases with temperature due to dehydration.

### gas_transport.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `O2_Diffusivity_Mean_m2_s` | String (Scientific) | Mean oxygen diffusivity | m²/s | 1.8e-6 - 2.8e-4 |
| `O2_Diffusivity_Std_m2_s` | String (Scientific) | Standard deviation | m²/s | 3.6e-7 - 5.6e-5 |
| `CO2_Diffusivity_Mean_m2_s` | String (Scientific) | Mean CO2 diffusivity | m²/s | 1.2e-6 - 1.9e-4 |
| `CO2_Diffusivity_Std_m2_s` | String (Scientific) | Standard deviation | m²/s | 2.4e-7 - 3.8e-5 |
| `O2_Permeability_m2_s_Pa` | String (Scientific) | Oxygen permeability | m²/s/Pa | 2.5e-9 - 3.9e-7 |
| `CO2_Permeability_m2_s_Pa` | String (Scientific) | CO2 permeability | m²/s/Pa | 4.0e-8 - 6.3e-6 |

**Physical Basis:** Gas diffusion increases with temperature and porosity. CO2 has higher solubility than O2.

## Deformation Properties

### creep_parameters.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Creep_Coefficient_A_Mean` | String (Scientific) | Mean creep coefficient | 1/(MPa^n·s^(1-m)) | 2.5e-12 - 8.2e-10 |
| `Creep_Coefficient_A_Std` | String (Scientific) | Standard deviation | 1/(MPa^n·s^(1-m)) | 7.5e-13 - 2.5e-10 |
| `Stress_Exponent_n_Mean` | Float | Mean stress exponent | - | 1.20 - 1.32 |
| `Stress_Exponent_n_Std` | Float | Standard deviation | - | 0.18 - 0.20 |
| `Time_Exponent_m_Mean` | Float | Mean time exponent | - | 0.16 - 0.19 |
| `Time_Exponent_m_Std` | Float | Standard deviation | - | 0.019 - 0.023 |
| `Activation_Energy_Mean_J_mol` | Float | Mean activation energy | J/mol | 39600 - 45000 |
| `Activation_Energy_Std_J_mol` | Float | Standard deviation | J/mol | 3960 - 4500 |

**Physical Basis:** Norton-Bailey creep law: ε̇ = A·σ^n·t^m·exp(-Q/RT). Rubber increases creep due to viscoelastic behavior.

### shrinkage_parameters.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Ultimate_Autogenous_Shrinkage_Mean` | String (Scientific) | Mean ultimate autogenous shrinkage | strain | 7.2e-5 - 6.8e-4 |
| `Ultimate_Autogenous_Shrinkage_Std` | String (Scientific) | Standard deviation | strain | 1.8e-5 - 1.7e-4 |
| `Ultimate_Drying_Shrinkage_Mean` | String (Scientific) | Mean ultimate drying shrinkage | strain | 2.9e-4 - 1.4e-3 |
| `Ultimate_Drying_Shrinkage_Std` | String (Scientific) | Standard deviation | strain | 5.8e-5 - 2.8e-4 |
| `Shrinkage_Time_Constant_Mean_days` | Float | Mean time constant | days | 22 - 45 |
| `Shrinkage_Time_Constant_Std_days` | Float | Standard deviation | days | 4.0 - 8.1 |
| `Humidity_Factor_Mean` | Float | Mean humidity factor | - | 0.68 - 0.85 |
| `Humidity_Factor_Std` | Float | Standard deviation | - | 0.068 - 0.085 |

**Physical Basis:** Shrinkage decreases with rubber content due to reduced cement paste content. Time constant increases with rubber.

### thermal_strain.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Thermal_Expansion_Coeff_Mean_per_K` | String (Scientific) | Mean expansion coefficient | K⁻¹ | 1.2e-5 - 2.0e-5 |
| `Thermal_Expansion_Coeff_Std_per_K` | String (Scientific) | Standard deviation | K⁻¹ | 1.4e-6 - 2.4e-6 |
| `Instantaneous_Thermal_Strain_Mean` | String (Scientific) | Mean instantaneous strain | strain | 0 - 1.6e-2 |
| `Instantaneous_Thermal_Strain_Std` | String (Scientific) | Standard deviation | strain | 0 - 1.6e-3 |
| `Time_Dependent_Thermal_Strain_Mean` | String (Scientific) | Mean time-dependent strain | strain | 0 - 2.4e-3 |
| `Time_Dependent_Thermal_Strain_Std` | String (Scientific) | Standard deviation | strain | 0 - 6.0e-4 |
| `Total_Thermal_Strain_Mean` | String (Scientific) | Mean total thermal strain | strain | 0 - 1.8e-2 |
| `Total_Thermal_Strain_Std` | String (Scientific) | Standard deviation | strain | 0 - 2.2e-3 |

**Physical Basis:** Total strain includes instantaneous thermal expansion and time-dependent microstructural changes.

### damage_evolution.csv

| Field | Type | Description | Units | Typical Range |
|-------|------|-------------|-------|---------------|
| `Damage_Level_Mean` | Float | Mean damage parameter | - | 0 - 0.86 |
| `Damage_Level_Std` | Float | Standard deviation | - | 0 - 0.17 |
| `Damage_Rate_Mean_per_C` | String (Scientific) | Mean damage rate | 1/°C | 0 - 1.2e-3 |
| `Damage_Rate_Std_per_C` | String (Scientific) | Standard deviation | 1/°C | 0 - 3.0e-4 |
| `Residual_Strength_Factor_Mean` | Float | Mean residual strength factor | - | 0.14 - 1.0 |
| `Residual_Strength_Factor_Std` | Float | Standard deviation | - | 0.011 - 0.080 |
| `Residual_Stiffness_Factor_Mean` | Float | Mean residual stiffness factor | - | 0.05 - 1.0 |
| `Residual_Stiffness_Factor_Std` | Float | Standard deviation | - | 0.005 - 0.10 |
| `Damage_Threshold_Temp_C` | Float | Damage threshold temperature | °C | 276 - 300 |

**Physical Basis:** Damage evolution: D = D_max·(1 - exp(-k·(T - T_threshold))). Rubber reduces damage threshold temperature.

## Data Quality and Validation

### Statistical Bounds
- All mean values represent physically consistent properties
- Standard deviations typically range from 8% to 30% of mean values
- Higher variability for transport properties reflects measurement challenges

### Temperature Dependencies
- All properties show smooth, physically realistic temperature dependencies
- No discontinuities or unphysical jumps in property values
- Degradation functions based on established material science principles

### Multi-Physics Consistency
- Thermal diffusivity calculated consistently from conductivity, specific heat, and density
- Mechanical property relationships maintained (e.g., strength-modulus correlations)
- Transport properties linked through porosity and microstructure

### Calibration/Validation Split
- Approximately 70% calibration, 30% validation data
- Random split maintains representative sampling across temperature and mix ranges
- Independent validation data enables robust model verification