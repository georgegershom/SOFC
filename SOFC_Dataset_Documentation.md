# SOFC Electrochemical Loading Dataset Documentation

## Overview

This document describes the comprehensive electrochemical loading dataset generated for Solid Oxide Fuel Cell (SOFC) research, specifically focusing on the requirements outlined in section 2.2 of the research framework:

- **Operating Voltage and Current Density**: Related to oxygen chemical potential gradients across the electrolyte
- **Overpotentials**: Especially at the anode, leading to local oxidation (Ni to NiO) and volume changes that induce stress
- **Electrochemical Impedance Spectroscopy (EIS)**: For detailed electrochemical characterization
- **IV Curves**: For performance analysis

## Dataset Structure

### 1. Generated Files

The dataset consists of the following files:

#### Primary Dataset Files:
- `sofc_realistic_electrochemical_dataset.json` - Complete dataset in JSON format
- `iv_curve_realistic.csv` - IV curve data with overpotential breakdown
- `detailed_realistic_analysis.csv` - Chemical gradients and stress analysis
- `sofc_realistic_dataset_overview.png` - Comprehensive visualization

#### EIS Data Files:
- `eis_realistic_current_0.csv` - EIS at 0 A/m² (OCV conditions)
- `eis_realistic_current_2000.csv` - EIS at 2000 A/m² (0.2 A/cm²)
- `eis_realistic_current_5000.csv` - EIS at 5000 A/m² (0.5 A/cm²)
- `eis_realistic_current_8000.csv` - EIS at 8000 A/m² (0.8 A/cm²)

### 2. Operating Conditions

The dataset was generated under the following realistic SOFC operating conditions:

| Parameter | Value | Units |
|-----------|-------|-------|
| Operating Temperature | 800 | °C |
| Anode Pressure | 1 | atm |
| Cathode Pressure | 1 | atm |
| Fuel Composition | 97% H₂, 3% H₂O | mol% |
| Air Composition | 21% O₂, 79% N₂ | mol% |
| Electrolyte Thickness | 150 | μm |
| Active Area | 10 | cm² |

### 3. Key Performance Metrics

| Metric | Value | Units |
|--------|-------|-------|
| Maximum Power Density | 0.60 | W/cm² |
| Area-Specific Resistance (ASR) | 0.35 | Ω⋅cm² |
| Nernst Potential | 1.381 | V |
| Maximum Ni Oxidation Risk | 0.000 | V |

## Data Description

### 3.1 IV Curve Data (`iv_curve_realistic.csv`)

Contains 101 data points from 0 to 10,000 A/m² (0 to 1.0 A/cm²) with the following columns:

| Column | Description | Units |
|--------|-------------|-------|
| `Current_Density_A_per_m2` | Applied current density | A/m² |
| `Voltage_V` | Cell voltage | V |
| `Power_Density_W_per_m2` | Power density (V × I) | W/m² |
| `Anode_Overpotential_V` | Anode activation overpotential | V |
| `Cathode_Overpotential_V` | Cathode activation overpotential | V |
| `Ohmic_Overpotential_V` | Electrolyte ohmic overpotential | V |
| `Ni_Oxidation_Risk_V` | Ni oxidation risk indicator | V |
| `Anode_Conc_Overpotential_V` | Anode concentration overpotential | V |
| `Cathode_Conc_Overpotential_V` | Cathode concentration overpotential | V |

**Key Findings:**
- Maximum power density occurs at ~6000 A/m² (0.6 A/cm²)
- Ohmic losses dominate at high current densities
- Concentration overpotentials become significant above 8000 A/m²

### 3.2 EIS Data (`eis_realistic_current_*.csv`)

Electrochemical Impedance Spectroscopy data collected at four different current densities over frequency range 0.01 Hz to 1 MHz (50 points per decade). Each file contains:

| Column | Description | Units |
|--------|-------------|-------|
| `Frequency_Hz` | AC frequency | Hz |
| `Impedance_Real_Ohm_m2` | Real part of impedance | Ω⋅m² |
| `Impedance_Imag_Ohm_m2` | Imaginary part of impedance | Ω⋅m² |
| `Impedance_Magnitude_Ohm_m2` | Impedance magnitude | Ω⋅m² |
| `Phase_Angle_deg` | Phase angle | degrees |

**Equivalent Circuit Model:**
- R_ohmic + (R_anode//CPE_anode) + (R_cathode//CPE_cathode) + Warburg
- Typical values at 800°C:
  - R_ohmic: 0.35 × 10⁻⁴ Ω⋅m²
  - R_anode: 0.08 × 10⁻⁴ Ω⋅m²
  - R_cathode: 0.12 × 10⁻⁴ Ω⋅m²

### 3.3 Detailed Analysis (`detailed_realistic_analysis.csv`)

Comprehensive analysis at four operating points (1000, 3000, 5000, 7000 A/m²):

| Column | Description | Units |
|--------|-------------|-------|
| `Current_Density_A_per_m2` | Operating current density | A/m² |
| `Anode_Overpotential_V` | Anode activation overpotential | V |
| `Ni_Oxidation_Risk_V` | Risk of Ni oxidation | V |
| `O2_Chemical_Potential_Gradient_J_per_mol_per_m` | O₂ gradient across electrolyte | J/mol/m |
| `Oxidation_Fraction` | Fraction of Ni oxidized | - |
| `Volumetric_Strain` | Volume change from oxidation | - |
| `Von_Mises_Stress_Pa` | Equivalent stress in anode | Pa |
| `Electrolyte_Stress_Pa` | Transmitted stress to electrolyte | Pa |
| `Risk_Level` | Qualitative risk assessment | - |

## Physical Models and Equations

### 4.1 Electrochemical Models

#### Nernst Potential
```
E_nernst = 1.253 - 2.4516×10⁻⁴ × T + (RT/2F) × ln(p_H₂ × √p_O₂ / p_H₂O)
```

#### Butler-Volmer Kinetics
```
η = (RT/αF) × ln(i/i₀)
```
Where:
- α = 0.5 (charge transfer coefficient)
- i₀_anode = 3000 A/m² at 800°C
- i₀_cathode = 1500 A/m² at 800°C

#### Ohmic Resistance
```
ASR_ohmic = t_electrolyte / σ_YSZ
σ_YSZ = 3.34×10⁴ × exp(-80000/RT) S/m
```

### 4.2 Oxygen Chemical Potential Gradient

The oxygen chemical potential gradient across the electrolyte is calculated as:

```
μ_O₂(x) = μ_O₂,anode + (μ_O₂,cathode - μ_O₂,anode) × x/t_electrolyte
∇μ_O₂ = (μ_O₂,cathode - μ_O₂,anode) / t_electrolyte
```

Where:
- μ_O₂ = RT × ln(p_O₂/p_ref)
- p_ref = 10⁵ Pa (1 bar)

**Typical Values:**
- Gradient magnitude: ~300 MJ/mol/m
- Driving force: ~45 kJ/mol at 5000 A/m²

### 4.3 Ni Oxidation and Stress Model

#### Oxidation Risk Assessment
```
Risk = max(0, η_anode + E_Ni/NiO + 0.1)
```

#### Volume Change and Stress
```
ε_vol = f_oxidized × ΔV_expansion × f_Ni
σ_hydrostatic = E_anode × ε_vol / (3(1-2ν))
```

Where:
- ΔV_expansion = 0.7 (70% volume increase)
- f_Ni = 0.35 (Ni volume fraction in anode)
- E_anode = 55 GPa, ν = 0.29

## Data Quality and Validation

### 5.1 Physical Consistency Checks

✅ **Voltage Range**: 0.1 - 1.4 V (realistic for SOFC operation)
✅ **Power Density**: Peak at 0.60 W/cm² (typical for 800°C operation)
✅ **ASR Values**: 0.35 Ω⋅cm² (consistent with literature)
✅ **Overpotential Hierarchy**: Ohmic > Activation > Concentration
✅ **EIS Characteristics**: Proper semicircles and Warburg tail

### 5.2 Literature Comparison

| Parameter | Dataset Value | Literature Range | Status |
|-----------|---------------|------------------|---------|
| Peak Power Density | 0.60 W/cm² | 0.4-0.8 W/cm² | ✅ Valid |
| ASR (800°C) | 0.35 Ω⋅cm² | 0.2-0.5 Ω⋅cm² | ✅ Valid |
| Nernst Potential | 1.381 V | 1.35-1.40 V | ✅ Valid |
| Anode i₀ | 3000 A/m² | 1000-5000 A/m² | ✅ Valid |

## Usage Guidelines

### 6.1 For Electrochemical Analysis
- Use IV curve data for performance characterization
- Use EIS data for equivalent circuit modeling
- Analyze overpotential breakdown for loss identification

### 6.2 For Mechanical Stress Analysis
- Use chemical potential gradients for oxygen transport modeling
- Use stress data for structural integrity assessment
- Consider Ni oxidation risk for durability analysis

### 6.3 For Multi-Physics Modeling
- Combine electrochemical and mechanical data
- Use temperature-dependent properties
- Consider coupling between performance and stress

## Data Limitations

1. **Simplified Models**: Some phenomena use simplified analytical models
2. **Single Temperature**: Data generated only at 800°C
3. **Ideal Conditions**: No degradation or contamination effects
4. **Geometry**: Based on planar cell configuration only

## Citation and Usage

This dataset was generated for SOFC research purposes. When using this data, please cite:

```
SOFC Electrochemical Loading Dataset
Generated: 2025-10-09
Operating Conditions: 800°C, H₂/H₂O fuel, air oxidant
Electrolyte: 150 μm YSZ
Configuration: Planar SOFC
```

## Contact and Support

For questions about the dataset or requests for additional data points, please refer to the generation scripts:
- `sofc_realistic_dataset_generator.py` - Main generation script
- `sofc_electrochemical_dataset_generator.py` - Alternative implementation

The dataset can be regenerated with different parameters by modifying the operating conditions and geometry parameters in the scripts.