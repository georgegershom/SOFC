# SOFC Electrochemical Loading Dataset

## Overview

This dataset contains comprehensive electrochemical loading data for Solid Oxide Fuel Cells (SOFCs), specifically designed to support research on electrolyte fracture risk prediction and mechanical durability analysis. The dataset includes operating voltage, current density, overpotentials, and impedance spectroscopy data with particular focus on anode oxidation effects (Ni to NiO conversion) and their impact on mechanical stress.

## Dataset Structure

The dataset consists of four main components:

### 1. I-V Curves (`iv_curves.csv`)
- **Records**: 5,000 data points
- **Description**: Current-voltage characteristics at different operating temperatures
- **Key Parameters**:
  - `temperature_c`: Operating temperature (700-850°C)
  - `current_density_acm2`: Current density (0-1.2 A/cm²)
  - `voltage_v`: Cell voltage
  - `power_density_wcm2`: Power density
  - `eta_ohmic_v`, `eta_activation_v`, `eta_concentration_v`: Individual overpotential components

### 2. Electrochemical Impedance Spectroscopy (`eis_spectra.csv`)
- **Records**: 1,500 data points
- **Description**: EIS spectra at various operating conditions
- **Key Parameters**:
  - `frequency_hz`: Frequency range (0.1-100,000 Hz)
  - `z_real_ohm_cm2`, `z_imag_ohm_cm2`: Real and imaginary impedance components
  - `phase_angle_deg`: Phase angle
  - `r_ohmic_ohm_cm2`, `r_ct_ohm_cm2`: Ohmic and charge transfer resistances

### 3. Overpotential Analysis (`overpotential_analysis.csv`)
- **Records**: 200 measurements
- **Description**: Detailed overpotential analysis with focus on anode oxidation effects
- **Key Parameters**:
  - `ni_oxidation_factor`: Factor representing Ni to NiO oxidation extent
  - `eta_anode_activation_v`, `eta_anode_concentration_v`: Anode overpotentials
  - `ni_volume_change_percent`: Volume change due to Ni oxidation
  - `induced_stress_mpa`: Mechanical stress induced by volume changes
  - `oxygen_chemical_potential_gradient`: Related to oxygen chemical potential gradient

### 4. Operating Conditions (`operating_conditions.csv`)
- **Records**: 100 conditions
- **Description**: Various operating conditions and their performance metrics
- **Key Parameters**:
  - `fuel_utilization`, `air_utilization`: Gas utilization rates
  - `efficiency`: Cell efficiency
  - `oxygen_chemical_potential_gradient`: Oxygen chemical potential gradient

## Key Features

### Anode Oxidation Effects
The dataset specifically addresses the critical issue of Ni to NiO oxidation in SOFC anodes:
- **Volume Changes**: Ni to NiO conversion causes ~69% volume expansion
- **Stress Induction**: Volume changes generate mechanical stresses (up to 50 MPa)
- **Overpotential Increase**: Ni oxidation increases anode activation and concentration overpotentials
- **Time Dependence**: Oxidation effects increase with operating time and temperature

### Electrochemical-Mechanical Coupling
- **Oxygen Chemical Potential Gradient**: Directly related to overpotentials and stress generation
- **Stress-Strain Relationships**: Volume changes from oxidation affect mechanical integrity
- **Temperature Dependence**: All effects are temperature-dependent with Arrhenius behavior

## Data Formats

The dataset is provided in multiple formats:
- **CSV**: Individual datasets for easy analysis
- **JSON**: Complete dataset with metadata
- **HDF5**: Efficient binary format for large-scale analysis
- **Metadata**: Detailed information about data generation and parameters

## Usage Examples

### Python Analysis
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load I-V curve data
iv_data = pd.read_csv('iv_curves.csv')

# Plot I-V curves at different temperatures
for temp in [700, 750, 800, 850]:
    temp_data = iv_data[iv_data['temperature_c'] == temp]
    plt.plot(temp_data['current_density_acm2'], temp_data['voltage_v'], 
             label=f'{temp}°C')

plt.xlabel('Current Density (A/cm²)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.title('SOFC I-V Characteristics')
plt.show()
```

### Overpotential Analysis
```python
# Load overpotential data
overpotential_data = pd.read_csv('overpotential_analysis.csv')

# Analyze Ni oxidation effects
plt.scatter(overpotential_data['ni_oxidation_factor'], 
           overpotential_data['induced_stress_mpa'])
plt.xlabel('Ni Oxidation Factor')
plt.ylabel('Induced Stress (MPa)')
plt.title('Ni Oxidation vs. Mechanical Stress')
plt.show()
```

## Physical Background

### SOFC Operation
Solid Oxide Fuel Cells operate at high temperatures (600-1000°C) and convert chemical energy to electrical energy through electrochemical reactions:
- **Anode**: H₂ + O²⁻ → H₂O + 2e⁻
- **Cathode**: ½O₂ + 2e⁻ → O²⁻
- **Electrolyte**: O²⁻ ion transport (8YSZ)

### Mechanical Challenges
- **Thermal Expansion Mismatch**: Different CTEs cause residual stresses
- **Anode Oxidation**: Ni to NiO conversion induces volume changes and stress
- **Brittle Electrolyte**: 8YSZ is susceptible to fracture under tensile stress
- **Thermal Cycling**: Startup/shutdown cycles create fatigue loading

### Electrochemical-Mechanical Coupling
- **Overpotentials**: Create oxygen chemical potential gradients
- **Volume Changes**: Ni oxidation causes ~69% volume expansion
- **Stress Generation**: Volume changes and thermal effects create mechanical stress
- **Fracture Risk**: High stress concentrations can cause electrolyte fracture

## Validation and Limitations

### Data Characteristics
- **Synthetic Data**: Generated using validated physical models
- **Realistic Parameters**: Based on literature values and experimental data
- **Temperature Range**: 650-900°C (typical SOFC operating range)
- **Current Density**: 0-1.2 A/cm² (realistic operating range)

### Limitations
- **Simplified Models**: Some complex phenomena are simplified
- **Idealized Conditions**: Perfect gas distribution and uniform properties assumed
- **Single Cell**: Stack effects not included
- **Steady State**: Transient effects limited

## Applications

This dataset is designed for:
1. **Fracture Risk Analysis**: Predicting electrolyte fracture under electrochemical loading
2. **Constitutive Model Validation**: Comparing elastic vs. viscoelastic models
3. **Design Optimization**: Optimizing cell geometry and operating conditions
4. **Durability Studies**: Understanding long-term degradation mechanisms
5. **Machine Learning**: Training models for SOFC performance prediction

## References

- Selimovic et al. (2005). "Modeling of solid oxide fuel cell applied to the analysis of SOFC-based energy systems"
- Nakajo et al. (2012). "Modeling of thermal stresses and probability of survival of tubular SOFC"
- Boccaccini et al. (2016). "Creep behavior of yttria-stabilized zirconia at high temperatures"

## Contact

For questions about this dataset or requests for additional data, please refer to the research article: "A Comparative Analysis of Constitutive Models for Predicting the Electrolyte's Fracture Risk in Planar SOFCs"

---
*Generated on: 2024*
*Dataset Version: 1.0*
*Total Size: ~1.5 MB*