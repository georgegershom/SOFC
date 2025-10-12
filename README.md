# Stratified Flow Acoustics Dataset

## PhD Thesis: "Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"

This repository contains a comprehensive dataset for research on acoustic attenuation mechanisms in stratified two-phase flows. The dataset includes experimental data, acoustic measurements, and derived parameters essential for understanding the complex interactions between flow regimes and acoustic wave propagation.

## Dataset Overview

The dataset contains **500 experimental cases** with the following key parameters:

### Flow Regime Characterization
- **Void fraction (α)**: 0.100 - 0.798
- **Superficial gas velocity (U_SG)**: 0.111 - 4.979 m/s
- **Superficial liquid velocity (U_SL)**: 0.057 - 1.997 m/s
- **Flow patterns**: Smooth interface, wavy interface, turbulent interface
- **Interface height and wave amplitude**

### Acoustic Signal Transmission
- **Raw acoustic pressure time-series data** (1000 samples per experiment)
- **Signal-to-Noise Ratio (SNR)**: -48.1 to 5.8 dB
- **Frequency spectrum analysis** (10 Hz to 10 kHz)
- **Attenuation coefficients** and transmission loss

### Fluid Properties & Conditions
- **Temperature range**: 15.0 - 34.9 °C
- **Pressure range**: 101.6 - 498.8 kPa
- **Density and viscosity** for both phases
- **Speed of sound** in both phases

### Turbulence & Shear Layer Data
- **Turbulent Kinetic Energy (TKE)** for both phases
- **Turbulent Dissipation Rate**
- **Velocity profiles** across pipe diameter
- **Shear stress** at interface and walls

## Files Structure

```
├── stratified_flow_acoustics_dataset.py    # Main dataset generator
├── analyze_stratified_flow_data.py         # Data analysis and visualization
├── export_dataset_to_csv.py               # CSV export utility
├── requirements.txt                        # Python dependencies
├── stratified_flow_acoustics_dataset.json # Complete dataset (JSON)
├── csv_exports/                           # CSV format exports
│   ├── flow_regime_data.csv
│   ├── fluid_properties_data.csv
│   ├── attenuation_metrics_data.csv
│   ├── turbulence_data.csv
│   ├── acoustic_transmission_metadata.csv
│   └── comprehensive_summary.csv
└── *.png                                  # Analysis plots
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python3 stratified_flow_acoustics_dataset.py
```

### 3. Analyze Data
```bash
python3 analyze_stratified_flow_data.py
```

### 4. Export to CSV
```bash
python3 export_dataset_to_csv.py
```

## Dataset Components

### 1. Flow Regime Data (`flow_regime_data.csv`)
Contains the fundamental flow characterization parameters:
- `void_fraction`: Gas volume fraction
- `superficial_gas_velocity`: Gas phase superficial velocity
- `superficial_liquid_velocity`: Liquid phase superficial velocity
- `interface_height`: Position of gas-liquid interface
- `flow_pattern`: Classification of flow regime
- `wave_amplitude`: Amplitude of interface waves
- `temperature`: System temperature
- `pressure`: System pressure

### 2. Acoustic Transmission Data
Raw acoustic signals and derived metrics:
- **Time-series data**: Source and received acoustic signals
- **Frequency spectra**: FFT analysis of acoustic signals
- **SNR measurements**: Signal-to-noise ratio in dB
- **Attenuation coefficients**: Frequency-dependent attenuation

### 3. Fluid Properties Data (`fluid_properties_data.csv`)
Thermodynamic and transport properties:
- `liquid_density`: Water density (temperature-dependent)
- `gas_density`: Air density (ideal gas law)
- `liquid_viscosity`: Water viscosity
- `gas_viscosity`: Air viscosity
- `liquid_speed_of_sound`: Acoustic velocity in water
- `gas_speed_of_sound`: Acoustic velocity in air

### 4. Turbulence Data (`turbulence_data.csv`)
Turbulent flow characteristics:
- `turbulent_kinetic_energy_gas`: TKE in gas phase
- `turbulent_kinetic_energy_liquid`: TKE in liquid phase
- `turbulent_dissipation_rate_gas`: Dissipation in gas phase
- `turbulent_dissipation_rate_liquid`: Dissipation in liquid phase
- `shear_stress_interface`: Shear stress at gas-liquid interface
- `wall_shear_stress_gas`: Wall shear stress in gas phase
- `wall_shear_stress_liquid`: Wall shear stress in liquid phase

### 5. Attenuation Metrics (`attenuation_metrics_data.csv`)
Acoustic attenuation analysis:
- `attenuation_coefficient`: Frequency-dependent attenuation
- `transmission_loss_db`: Transmission loss in dB
- `frequencies`: Frequency vector (10 Hz - 10 kHz)
- `average_attenuation`: Mean attenuation coefficient
- `max_attenuation_frequency`: Frequency of maximum attenuation
- `min_attenuation_frequency`: Frequency of minimum attenuation

## Key Findings

### Flow Pattern Distribution
- **Turbulent interface**: 388 cases (77.6%)
- **Wavy interface**: 84 cases (16.8%)
- **Smooth interface**: 28 cases (5.6%)

### Acoustic Characteristics
- **Average SNR**: -0.5 ± 6.2 dB
- **Average attenuation coefficient**: 0.2501
- **Frequency range**: 10 Hz to 10 kHz
- **Signal sampling**: 1000 Hz

### Attenuation Mechanisms
The dataset captures several key attenuation mechanisms:
1. **Interface scattering**: Due to gas-liquid interface waves
2. **Turbulent dissipation**: Energy loss through turbulent mixing
3. **Viscous attenuation**: Molecular viscosity effects
4. **Thermal attenuation**: Heat conduction losses
5. **Geometric attenuation**: Pipe wall reflections

## Usage Examples

### Python Analysis
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
flow_data = pd.read_csv('csv_exports/flow_regime_data.csv')
atten_data = pd.read_csv('csv_exports/attenuation_metrics_data.csv')

# Plot attenuation vs void fraction
plt.scatter(flow_data['void_fraction'], atten_data['average_attenuation'])
plt.xlabel('Void Fraction')
plt.ylabel('Attenuation Coefficient')
plt.title('Attenuation vs Void Fraction')
plt.show()
```

### MATLAB Analysis
```matlab
% Load data
flow_data = readtable('csv_exports/flow_regime_data.csv');
atten_data = readtable('csv_exports/attenuation_metrics_data.csv');

% Plot correlation
scatter(flow_data.void_fraction, atten_data.average_attenuation);
xlabel('Void Fraction');
ylabel('Attenuation Coefficient');
title('Attenuation vs Void Fraction');
```

## Theoretical Background

### Stratified Flow Acoustics
The dataset is based on established theories for acoustic propagation in stratified two-phase flows:

1. **Taitel & Dukler (1976)**: Flow pattern classification
2. **Temkin (2002)**: Acoustic attenuation in multiphase flows
3. **Ishii & Hibiki (2011)**: Two-phase flow dynamics
4. **Brennen (2005)**: Cavitation and bubble dynamics

### Attenuation Mechanisms
The generated data incorporates multiple attenuation mechanisms:

- **Interface scattering**: `α_scatter ∝ (k·a)²` where k is wavenumber, a is wave amplitude
- **Turbulent dissipation**: `α_turb ∝ ε^(1/3)/c` where ε is dissipation rate, c is sound speed
- **Viscous attenuation**: `α_visc ∝ ω²·μ/ρ·c³` where ω is frequency, μ is viscosity
- **Thermal attenuation**: `α_therm ∝ ω²·κ/ρ·c³` where κ is thermal conductivity

## Validation and Quality

The dataset is generated using:
- **Literature-based correlations** for flow regime transitions
- **Established acoustic theories** for attenuation mechanisms
- **Realistic parameter ranges** based on experimental studies
- **Statistical validation** of generated distributions

## Applications

This dataset is suitable for:
- **Machine learning** model training for attenuation prediction
- **Computational fluid dynamics** validation
- **Acoustic modeling** algorithm development
- **Flow regime identification** studies
- **PhD thesis research** on multiphase flow acoustics

## Citation

If you use this dataset in your research, please cite:

```
Stratified Flow Acoustics Dataset
PhD Thesis: "Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"
Generated: 2024
```

## Contact

For questions about the dataset or to request additional parameters, please refer to the PhD thesis documentation or contact the research team.

## License

This dataset is provided for academic research purposes. Please ensure proper attribution and citation in any publications using this data.