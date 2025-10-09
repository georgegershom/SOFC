# SOFC Electrochemical Loading Dataset

## Overview

This dataset contains comprehensive electrochemical performance data for Solid Oxide Fuel Cell (SOFC) testing, with a focus on the coupling between electrochemical loading and mechanical stress in the electrolyte. The dataset is designed to support research on SOFC durability, particularly the analysis of thermo-mechanical stresses induced by electrochemical phenomena.

**Generation Date**: October 9, 2025  
**Temperature Range**: 650°C - 850°C  
**Current Density Range**: 0 - 1.5 A/cm²

## Dataset Components

### 1. IV Curve Data (`sofc_iv_curve_800C.csv`)

Current-voltage (IV) characteristics at 800°C operating temperature.

**Columns**:
- `Current_Density_A_cm2`: Operating current density (A/cm²)
- `Voltage_V`: Cell voltage (V)
- `Overpotential_Anode_V`: Anode activation overpotential (V)
- `Overpotential_Cathode_V`: Cathode activation overpotential (V)
- `Overpotential_Ohmic_V`: Ohmic overpotential through electrolyte (V)
- `Overpotential_Concentration_V`: Concentration/diffusion overpotential (V)
- `Power_Density_W_cm2`: Power output density (W/cm²)
- `Temperature_C`: Operating temperature (°C)

**Key Results**:
- Peak Power Density: **0.272 W/cm²**
- Maximum Anode Overpotential: **0.256 V**
- Maximum Cathode Overpotential: **0.513 V**

**Physical Significance**:
- IV curves show the relationship between operating voltage and current density
- Overpotentials indicate the losses in each cell component
- Total overpotential = V_OCV - V_operating = η_anode + η_cathode + η_ohmic + η_concentration

### 2. Electrochemical Impedance Spectroscopy Data (`sofc_eis_data.csv`)

EIS measurements at multiple current densities (0.0, 0.2, 0.4, 0.6, 0.8, 1.0 A/cm²).

**Columns**:
- `Current_Density_A_cm2`: Operating current density during measurement
- `Frequency_Hz`: AC frequency (0.01 Hz - 100 kHz)
- `Z_Real_Ohm_cm2`: Real part of impedance (Ω·cm²)
- `Z_Imag_Ohm_cm2`: Imaginary part of impedance (Ω·cm²)
- `Z_Magnitude_Ohm_cm2`: Impedance magnitude (Ω·cm²)
- `Z_Phase_deg`: Phase angle (degrees)
- `Temperature_C`: Operating temperature (°C)

**Physical Significance**:
- EIS reveals the contribution of different processes to cell resistance
- High-frequency intercept: Ohmic resistance
- Mid-frequency arc: Charge transfer resistance (anode and cathode)
- Low-frequency arc: Gas diffusion/concentration polarization
- Nyquist plots (Z_Real vs Z_Imag) show characteristic RC semicircles

### 3. Overpotential-Stress Coupling Data (`sofc_overpotential_stress_data.csv`)

**Critical dataset linking electrochemical overpotentials to mechanical stress through Ni oxidation.**

**Columns**:
- `Current_Density_A_cm2`: Operating current density
- `Voltage_V`: Cell voltage
- `Overpotential_Anode_V`: Anode overpotential
- `O2_Partial_Pressure_Anode_Pa`: Local oxygen partial pressure at anode (Pa)
- `O2_Partial_Pressure_Cathode_Pa`: Oxygen partial pressure at cathode (Pa, ~21000 for air)
- `O2_Chemical_Potential_Gradient_J_mol`: Chemical potential gradient across electrolyte (J/mol)
- `ln_PO2_Ratio_Cathode_Anode`: Natural log of oxygen partial pressure ratio
- `Oxidation_Risk_Factor`: Risk factor for Ni to NiO conversion (>1.0 indicates risk)
- `Ni_Fraction_Oxidized`: Fraction of Ni converted to NiO
- `Stress_Induced_MPa`: Mechanical stress induced by volume expansion (MPa)
- `Temperature_C`: Operating temperature

**Physical Significance**:
- **Oxygen Chemical Potential Gradient**: Δμ_O2 = -4F × i × R_ohmic
  - Drives ionic current through electrolyte
  - Higher current → larger gradient → higher oxygen activity at anode
  
- **Anode Oxidation Risk**: 
  - High overpotentials increase local oxygen partial pressure
  - When P_O2 > P_O2_critical (~10⁻¹⁵ Pa at 800°C), Ni can oxidize to NiO
  - Oxidation Risk Factor = P_O2_anode / P_O2_critical
  
- **Volume Change and Stress**:
  - NiO has 68% larger molar volume than Ni
  - Constrained expansion in YSZ matrix induces stress
  - σ = E/(1-2ν) × ε_vol, where ε_vol = (V_NiO - V_Ni)/V_Ni
  - This stress couples back to mechanical fracture risk in electrolyte

**Key Results**:
- Maximum O₂ Chemical Potential Gradient: **868.4 kJ/mol**
- Maximum Oxidation Risk Factor: **0.66** (below critical threshold)
- Demonstrates the coupling between electrochemical and mechanical phenomena

### 4. Multi-Temperature IV Curves (`sofc_multi_temperature_iv_curves.csv`)

IV characteristics at five temperatures: 650°C, 700°C, 750°C, 800°C, 850°C.

**Columns**: Same as IV curve data, with varying `Temperature_C`

**Physical Significance**:
- Shows temperature dependence of electrochemical performance
- Higher temperatures → lower activation overpotentials
- Higher temperatures → better ionic conductivity
- Trade-off between performance and mechanical degradation

### 5. Degradation Time-Series Data (`sofc_degradation_time_series.csv`)

Performance degradation over 5000 hours at constant current (0.5 A/cm²).

**Columns**:
- `Time_hours`: Operating time (hours)
- `Current_Density_A_cm2`: Constant current density
- `Voltage_V`: Cell voltage (degrades over time)
- `Overpotential_Anode_V`: Anode overpotential (increases with degradation)
- `Overpotential_Cathode_V`: Cathode overpotential (increases with degradation)
- `Overpotential_Ohmic_V`: Ohmic overpotential (increases with degradation)
- `R_Ohmic_Ohm_cm2`: Ohmic resistance (increases over time)
- `Power_Density_W_cm2`: Power density (decreases over time)
- `Degradation_Rate_mV_per_kh`: Voltage degradation rate (mV per 1000 hours)
- `Temperature_C`: Operating temperature

**Physical Significance**:
- Simulates realistic SOFC degradation mechanisms
- Anode degradation: Ni coarsening, poisoning (rate: 5×10⁻⁵ h⁻¹)
- Cathode degradation: Sr segregation, delamination (rate: 8×10⁻⁵ h⁻¹)
- Ohmic degradation: Interface resistance increase (rate: 3×10⁻⁵ h⁻¹)
- Total voltage loss over 5000h: **110.2 mV**

## Metadata Files

### `sofc_dataset_metadata.json`
Contains dataset description, parameters, and file information.

### `sofc_dataset_summary.json`
Summary statistics including:
- Peak power density
- Maximum overpotentials
- Stress coupling parameters
- Degradation rates

## Physical Background

### Relation to Research Article (Section 2.2)

This dataset directly addresses the requirements specified in Section 2.2 of the research article on SOFC electrolyte fracture risk:

#### Operating Voltage and Current Density
- IV curves provide voltage-current relationships across operational range
- Related to **oxygen chemical potential gradient** through:
  ```
  Δμ_O2 = -nF × V_ohmic = -4F × i × R_ohmic
  ```
- Gradient drives oxygen ion flux through electrolyte
- Higher current density → larger chemical potential gradient

#### Overpotentials and Mechanical Stress
- **Anode overpotentials** create local electrochemical conditions that can lead to:
  1. Increased local oxygen partial pressure
  2. Ni to NiO oxidation when P_O2 exceeds critical threshold
  3. Volume expansion (68% for NiO vs Ni)
  4. Induced stress in constrained YSZ electrolyte
  
- Coupling equation:
  ```
  P_O2,anode = P_O2,ref × exp(η_anode × 4F / RT)
  σ_induced = E_YSZ / (1 - 2ν) × ε_vol
  ```

### Key Electrochemical-Mechanical Coupling

The dataset demonstrates three critical coupling mechanisms:

1. **Chemical Potential → Ionic Current**
   - Nernst equation relates voltage to oxygen partial pressure ratio
   - Electrochemical driving force for ion transport

2. **Overpotential → Oxidation State**
   - High overpotentials shift local equilibrium
   - Can cause redox cycling of Ni/NiO at anode

3. **Phase Change → Mechanical Stress**
   - NiO formation involves 68% volume expansion
   - Constrained expansion induces stress in YSZ electrolyte
   - Stress couples to fracture risk analysis (see Sections 2.5 and 3.4 of article)

## Usage Examples

### Python Example: Load and Analyze IV Curve

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load IV curve data
iv_data = pd.read_csv('sofc_iv_curve_800C.csv')

# Plot IV and power curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# IV curve
ax1.plot(iv_data['Current_Density_A_cm2'], iv_data['Voltage_V'])
ax1.set_xlabel('Current Density (A/cm²)')
ax1.set_ylabel('Voltage (V)')
ax1.set_title('IV Curve at 800°C')
ax1.grid(True)

# Power curve
ax2.plot(iv_data['Current_Density_A_cm2'], iv_data['Power_Density_W_cm2'])
ax2.set_xlabel('Current Density (A/cm²)')
ax2.set_ylabel('Power Density (W/cm²)')
ax2.set_title('Power Curve at 800°C')
ax2.grid(True)

plt.tight_layout()
plt.savefig('iv_power_curves.png', dpi=300)
```

### Python Example: Analyze Overpotential-Stress Coupling

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load overpotential-stress data
stress_data = pd.read_csv('sofc_overpotential_stress_data.csv')

# Identify oxidation risk region
risk_data = stress_data[stress_data['Oxidation_Risk_Factor'] > 0.5]

# Plot overpotential vs stress
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(stress_data['Overpotential_Anode_V'], 
                     stress_data['Stress_Induced_MPa'],
                     c=stress_data['Oxidation_Risk_Factor'],
                     cmap='RdYlBu_r', s=50)
ax.set_xlabel('Anode Overpotential (V)')
ax.set_ylabel('Induced Stress (MPa)')
ax.set_title('Electrochemical-Mechanical Coupling')
plt.colorbar(scatter, label='Oxidation Risk Factor')
ax.grid(True, alpha=0.3)
plt.savefig('overpotential_stress_coupling.png', dpi=300)
```

### Python Example: EIS Nyquist Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load EIS data
eis_data = pd.read_csv('sofc_eis_data.csv')

# Plot Nyquist plots for different current densities
fig, ax = plt.subplots(figsize=(10, 8))

for i_cd in [0.0, 0.3, 0.5, 0.8, 1.0]:
    data = eis_data[eis_data['Current_Density_A_cm2'] == i_cd]
    ax.plot(data['Z_Real_Ohm_cm2'], -data['Z_Imag_Ohm_cm2'], 
            'o-', label=f'{i_cd} A/cm²', markersize=3)

ax.set_xlabel('Z_Real (Ω·cm²)')
ax.set_ylabel('-Z_Imag (Ω·cm²)')
ax.set_title('EIS Nyquist Plots at Various Current Densities')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.savefig('eis_nyquist.png', dpi=300)
```

## Material Parameters Used

| Parameter | Value | Unit |
|-----------|-------|------|
| Operating Temperature | 800 | °C |
| Open Circuit Voltage (E_OCV) | 1.05 | V |
| Ohmic Resistance (R_ohmic) | 0.15 | Ω·cm² |
| Anode Exchange Current Density | 5000 | A/m² |
| Cathode Exchange Current Density | 1000 | A/m² |
| Charge Transfer Coefficient | 0.5 | - |
| YSZ Young's Modulus (800°C) | 170 | GPa |
| YSZ Poisson's Ratio | 0.23 | - |
| NiO Volume Expansion | 68 | % |

## Integration with FEA Models

This electrochemical dataset is designed to be integrated with finite element analysis (FEA) models for coupled multi-physics simulations:

1. **Thermal Loading**:
   - Use Power_Density_W_cm2 as heat generation source term
   - Account for spatial variation in current distribution

2. **Mechanical Loading**:
   - Import Stress_Induced_MPa as body force or initial stress
   - Couple with thermal expansion stresses
   - Evaluate combined stress state for fracture risk

3. **Time-Dependent Analysis**:
   - Use degradation data for long-term simulations
   - Update material properties based on operating hours
   - Assess cumulative damage and fatigue

## References

This dataset supports the analysis presented in:

**"A Comparative Analysis of Constitutive Models for Predicting the Electrolyte's Fracture Risk in Planar SOFCs"**

Particularly relevant to:
- Section 2.2: Electrochemical Loading Data
- Section 2.3: Boundary Conditions and Load Cases
- Section 3.4: Comparative Fracture Risk Assessment

## Citation

If you use this dataset in your research, please cite:

```
SOFC Electrochemical Loading Dataset
Generated: October 9, 2025
Source: Synthetic data based on validated SOFC electrochemical models
Material parameters from experimental characterization of 8YSZ electrolyte systems
```

## Data Quality and Validation

This dataset is generated using validated electrochemical models based on:
- Butler-Volmer kinetics for activation overpotentials
- Ohm's law for ohmic losses
- Fick's law for concentration polarization
- Equivalent circuit models for EIS
- Thermodynamic equilibrium for O₂ partial pressures

Model parameters are derived from experimental literature on YSZ-based SOFCs operating at 800°C.

## Contact and Support

For questions about this dataset or requests for additional data:
- Check the metadata files for generation parameters
- Refer to the Python generator script for implementation details
- Consult the research article for theoretical background

## License

This dataset is provided for research and educational purposes. Please acknowledge the source in any publications or presentations.
