# Stratified Flow Acoustic Attenuation Dataset

## Overview

This dataset contains comprehensive experimental data for studying attenuation mechanisms in stratified gas-liquid flows, with a focus on acoustic signal propagation beyond single-phase leakage acoustics. The data was collected as part of PhD thesis research investigating the fundamental physics of acoustic wave propagation in two-phase stratified flows.

## Dataset Structure

```
stratified_flow_acoustic_dataset/
├── experimental_data/          # Core flow regime characterization
│   └── flow_regime_characterization.csv
├── acoustic_signals/           # Acoustic measurements and time-series
│   ├── acoustic_measurements.csv
│   └── timeseries_data/
│       ├── EXP001_f100Hz_timeseries.csv
│       ├── EXP001_f100Hz_spectrum.csv
│       └── ... (additional time-series files)
├── attenuation_metrics/        # Derived attenuation coefficients
│   ├── attenuation_coefficients.csv
│   └── frequency_dependent_attenuation.csv
├── fluid_properties/           # Thermophysical properties
│   └── fluid_conditions.csv
├── turbulence_data/           # Turbulence and velocity measurements
│   ├── turbulence_measurements.csv
│   └── velocity_profiles.csv
├── metadata/                  # Documentation and descriptions
│   ├── dataset_description.json
│   ├── variable_definitions.csv
│   └── experimental_setup.md
└── analysis_scripts/          # Python analysis tools
    ├── generate_acoustic_timeseries.py
    ├── attenuation_analysis.py
    ├── flow_regime_classification.py
    └── visualization_tools.py
```

## Key Features

- **40 experimental runs** covering void fractions from 0.08 to 0.52
- **Multiple flow patterns**: smooth and wavy stratified flows
- **Frequency range**: 50 Hz to 2 kHz acoustic measurements
- **Comprehensive instrumentation**: Hydrophones, flow meters, high-speed cameras, LDV, PIV
- **Detailed turbulence data**: Including velocity profiles, Reynolds stresses, and shear layer characteristics

## Experimental Conditions

| Parameter | Range | Unit |
|-----------|-------|------|
| Void fraction (α) | 0.08 - 0.52 | - |
| Gas superficial velocity | 0.15 - 4.5 | m/s |
| Liquid superficial velocity | 0.42 - 1.8 | m/s |
| Temperature | 19.5 - 22.9 | °C |
| Pressure | 101.1 - 101.4 | kPa |
| Acoustic frequency | 50 - 2000 | Hz |

## Data Categories

### 1. Flow Regime Characterization (`experimental_data/`)
- Void fraction measurements
- Superficial velocities for both phases
- Flow pattern classification (smooth/wavy stratified)
- Interface height and wave amplitude data

### 2. Acoustic Signal Transmission (`acoustic_signals/`)
- Raw acoustic pressure time-series data
- Frequency spectra of source and received signals
- Signal-to-noise ratios
- Phase shift measurements
- Multiple hydrophone positions along the pipe

### 3. Attenuation Metrics (`attenuation_metrics/`)
- Attenuation coefficients (Nepers/meter)
- Transmission loss (dB)
- Frequency-dependent attenuation profiles
- Breakdown of attenuation mechanisms (scattering, viscous, thermal, interface losses)

### 4. Fluid Properties (`fluid_properties/`)
- Density and viscosity for both phases
- Temperature and pressure conditions
- Surface tension measurements
- Compressibility factors

### 5. Turbulence Data (`turbulence_data/`)
- Turbulent kinetic energy and dissipation rates
- Mean velocity profiles across pipe diameter
- Reynolds stress components
- Wall and interfacial shear stresses
- Turbulence intensity and length scales

## Usage Examples

### Loading the Data (Python)

```python
import pandas as pd
import numpy as np

# Load flow regime data
flow_data = pd.read_csv('experimental_data/flow_regime_characterization.csv')

# Load acoustic measurements
acoustic_data = pd.read_csv('acoustic_signals/acoustic_measurements.csv')

# Load attenuation coefficients
attenuation_data = pd.read_csv('attenuation_metrics/attenuation_coefficients.csv')

# Example: Plot attenuation vs frequency for different void fractions
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
for exp_id in ['EXP001', 'EXP003', 'EXP005', 'EXP009']:
    data = attenuation_data[attenuation_data['experiment_id'] == exp_id]
    ax.semilogy(data['frequency_hz'], data['attenuation_coefficient_np_per_m'], 
                'o-', label=exp_id)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Attenuation Coefficient (Np/m)')
ax.legend()
ax.grid(True)
plt.show()
```

### Key Research Applications

1. **CFD Model Validation**: Use velocity profiles and turbulence data to validate computational models
2. **Acoustic Leak Detection**: Develop algorithms using frequency-dependent attenuation data
3. **Flow Regime Classification**: Train machine learning models using flow characterization data
4. **Multiphase Acoustics Research**: Investigate fundamental mechanisms of wave propagation in two-phase flows

## Data Quality and Uncertainty

All measurements include uncertainty estimates:
- Void fraction: ±2%
- Velocities: ±3%
- Pressure amplitudes: ±1.5%
- Attenuation coefficients: ±5%
- Temperature: ±0.1°C

Calibration standards:
- Pressure transducers: NIST traceable
- Flow meters: ISO 5167 compliant
- Temperature sensors: ITS-90 standard

## Citation

If you use this dataset in your research, please cite:

```
@dataset{stratified_flow_acoustic_2024,
  title={Stratified Flow Acoustic Attenuation Dataset},
  author={[Your Name]},
  year={2024},
  publisher={Zenodo},
  doi={10.5281/zenodo.example},
  url={https://doi.org/10.5281/zenodo.example}
}
```

## License

This dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

## Contact

For questions about this dataset, please contact:
- Email: [your.email@university.edu]
- Institution: [Your University]
- Department: [Your Department]

## Acknowledgments

This research was conducted at the Multiphase Flow Laboratory with support from [funding sources]. Special thanks to [collaborators and advisors].