# SOFC Electrochemical Loading Dataset - Summary

## Dataset Overview

This dataset has been successfully generated and contains comprehensive electrochemical loading data for Solid Oxide Fuel Cells (SOFCs), specifically designed to support research on electrolyte fracture risk prediction and mechanical durability analysis.

## Generated Files

### Data Files
- **`iv_curves.csv`** (5,000 records): Current-voltage characteristics at different temperatures
- **`eis_spectra.csv`** (1,500 records): Electrochemical impedance spectroscopy data
- **`overpotential_analysis.csv`** (200 records): Detailed overpotential analysis with Ni oxidation effects
- **`operating_conditions.csv`** (100 records): Various operating conditions and performance metrics

### Data Formats
- **CSV files**: Individual datasets for easy analysis
- **`sofc_electrochemical_data.json`**: Complete dataset with metadata
- **`sofc_electrochemical_data.h5`**: Efficient binary format for large-scale analysis
- **`metadata.json`**: Detailed information about data generation and parameters

### Documentation
- **`README.md`**: Comprehensive documentation with usage examples
- **`DATASET_SUMMARY.md`**: This summary file

### Visualizations
- **`plots/iv_characteristics.png`**: I-V curves and power density characteristics
- **`plots/overpotential_breakdown.png`**: Overpotential component analysis
- **`plots/impedance_spectra.png`**: EIS Nyquist and Bode plots
- **`plots/anode_oxidation_effects.png`**: Ni oxidation effects and mechanical stress
- **`plots/operating_conditions.png`**: Operating conditions and performance analysis
- **`plots/comprehensive_dashboard.png`**: Complete analysis dashboard

## Key Features

### 1. Operating Voltage and Current Density
- **Temperature Range**: 700-850°C (typical SOFC operating range)
- **Current Density Range**: 0-1.2 A/cm²
- **Voltage Range**: 0.6-1.1 V (realistic operating voltages)
- **Power Density**: Up to 0.8 W/cm²

### 2. Overpotentials (Anode Focus)
- **Anode Activation Overpotential**: 0.02-0.15 V (increases with Ni oxidation)
- **Anode Concentration Overpotential**: 0.01-0.08 V (affected by NiO formation)
- **Cathode Overpotentials**: 0.02-0.10 V
- **Ohmic Overpotential**: 0.05-0.20 V (increases with Ni oxidation)

### 3. Ni to NiO Oxidation Effects
- **Ni Oxidation Factor**: 1.0-2.5 (increases with temperature and time)
- **Volume Change**: 0-69% (Ni to NiO volume expansion)
- **Induced Stress**: 0-50 MPa (mechanical stress from volume changes)
- **Time Dependence**: Oxidation increases with operating time

### 4. Electrochemical Impedance Spectroscopy
- **Frequency Range**: 0.1-100,000 Hz
- **Impedance Magnitude**: 0.1-2.0 Ω·cm²
- **Phase Angle**: -90° to 0°
- **Charge Transfer Resistance**: 0.1-1.0 Ω·cm²

## Physical Relationships

### Oxygen Chemical Potential Gradient
- Directly related to overpotentials and stress generation
- Higher overpotentials create larger oxygen chemical potential gradients
- Affects mechanical stress distribution in the electrolyte

### Mechanical Stress Generation
- **Volume Changes**: Ni to NiO conversion causes ~69% volume expansion
- **Thermal Effects**: CTE mismatch between components
- **Stress Concentration**: Up to 50 MPa in high-stress regions
- **Fracture Risk**: Related to maximum principal stress

### Temperature Dependencies
- **Arrhenius Behavior**: All electrochemical processes follow Arrhenius kinetics
- **Creep Effects**: Time-dependent deformation at high temperatures
- **Material Properties**: Temperature-dependent elastic moduli and conductivities

## Dataset Statistics

| Dataset | Records | Columns | Size | Key Parameters |
|---------|---------|---------|------|----------------|
| I-V Curves | 5,000 | 10 | 0.38 MB | Temperature, Current, Voltage, Power |
| EIS Spectra | 1,500 | 10 | 0.11 MB | Frequency, Impedance, Phase |
| Overpotential Analysis | 200 | 15 | 0.02 MB | Ni Oxidation, Stress, Overpotentials |
| Operating Conditions | 100 | 15 | 0.01 MB | Efficiency, Utilization, Performance |

## Applications

This dataset is specifically designed for:

1. **Fracture Risk Analysis**: Predicting electrolyte fracture under electrochemical loading
2. **Constitutive Model Validation**: Comparing elastic vs. viscoelastic models
3. **Design Optimization**: Optimizing cell geometry and operating conditions
4. **Durability Studies**: Understanding long-term degradation mechanisms
5. **Machine Learning**: Training models for SOFC performance prediction
6. **Electrochemical-Mechanical Coupling**: Studying stress generation from electrochemical processes

## Usage Examples

### Python Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
iv_data = pd.read_csv('iv_curves.csv')
overpotential_data = pd.read_csv('overpotential_analysis.csv')

# Analyze I-V characteristics
temp_800 = iv_data[iv_data['temperature_c'] == 800]
plt.plot(temp_800['current_density_acm2'], temp_800['voltage_v'])
plt.xlabel('Current Density (A/cm²)')
plt.ylabel('Voltage (V)')
plt.title('I-V Characteristics at 800°C')
plt.show()

# Analyze Ni oxidation effects
plt.scatter(overpotential_data['ni_oxidation_factor'], 
           overpotential_data['induced_stress_mpa'])
plt.xlabel('Ni Oxidation Factor')
plt.ylabel('Induced Stress (MPa)')
plt.title('Mechanical Stress vs Ni Oxidation')
plt.show()
```

### Data Access
```python
# Load from HDF5 (efficient for large datasets)
import h5py
with h5py.File('sofc_electrochemical_data.h5', 'r') as f:
    iv_data = pd.DataFrame(f['iv_curves'][:])
    print(f.attrs['generator_version'])  # Access metadata
```

## Validation and Quality

- **Physical Consistency**: All data follows established electrochemical and mechanical principles
- **Parameter Ranges**: Realistic values based on literature and experimental data
- **Temperature Dependencies**: Proper Arrhenius behavior for all temperature-dependent processes
- **Statistical Properties**: Normal distributions with realistic standard deviations
- **Correlation Structure**: Physically meaningful correlations between parameters

## Future Extensions

This dataset can be extended with:
- **Transient Analysis**: Time-dependent behavior during startup/shutdown
- **Stack Effects**: Multi-cell stack interactions
- **Degradation Models**: Long-term performance degradation
- **Material Variations**: Different electrolyte and electrode materials
- **Operating Modes**: Various fuel compositions and operating strategies

## Contact and Support

For questions about this dataset or requests for additional data, please refer to the research article: "A Comparative Analysis of Constitutive Models for Predicting the Electrolyte's Fracture Risk in Planar SOFCs"

---
*Dataset Generated: 2024*
*Version: 1.0*
*Total Size: ~1.5 MB*
*Records: 6,800*
*Visualizations: 6 comprehensive plots*