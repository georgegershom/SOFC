# Stratified Flow Acoustic Attenuation Dataset

## Overview

This dataset has been generated for the PhD research topic:  
**"Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"**

The dataset provides comprehensive synthetic experimental data for gas-liquid stratified flows with detailed acoustic attenuation measurements, suitable for validation of theoretical models, algorithm development, and machine learning applications.

## Dataset Description

### Physical System
- **Flow Configuration**: Horizontal gas-liquid stratified flow
- **Working Fluids**: Air-Water system
- **Pipe Dimensions**: 
  - Diameter: 50 mm (typical laboratory scale)
  - Length: 5 meters
- **Number of Experiments**: 100 unique operating conditions

### Data Generation Methodology

The dataset is based on established physical models and empirical correlations from multiphase flow literature:

1. **Flow Regime Classification**: Taitel-Dukler flow map
2. **Void Fraction Calculation**: Drift flux model with slip ratio corrections
3. **Acoustic Attenuation**: Multiple mechanisms modeled:
   - Classical viscous absorption
   - Scattering from interface waves
   - Turbulence-induced attenuation
   - Two-phase mixture effects (Wood's equation)
4. **Turbulence Modeling**: k-ε model with wall functions
5. **Velocity Profiles**: Power law distribution

## Dataset Files

### 1. `flow_regime_characterization.csv`
Flow regime parameters for each experiment.

**Key Parameters**:
- `experiment_id`: Unique identifier for each test
- `timestamp`: Time of experiment
- `U_SG`: Superficial gas velocity (m/s)
- `U_SL`: Superficial liquid velocity (m/s)
- `void_fraction`: Gas volume fraction (dimensionless, 0-1)
- `flow_pattern`: Classification (smooth_stratified or wavy_stratified)
- `interface_height`: Normalized interface height (h/D)
- `wave_amplitude`: Interface wave amplitude (m)
- `wave_frequency`: Interface wave frequency (Hz)
- `temperature`: Operating temperature (°C)
- `pressure`: System pressure (bar, absolute)
- `rho_gas`, `rho_liquid`: Phase densities (kg/m³)
- `mu_gas`, `mu_liquid`: Phase dynamic viscosities (Pa·s)

### 2. `acoustic_attenuation_data.csv`
Acoustic transmission and attenuation measurements.

**Key Parameters**:
- `experiment_id`: Links to flow regime data
- `frequency`: Acoustic frequency tested (Hz)
- `attenuation_coefficient`: Total attenuation coefficient (Np/m)
- `transmission_loss_dB`: Transmission loss across pipe (dB)
- `source_amplitude`: Source signal amplitude (normalized)
- `received_amplitude`: Received signal amplitude after attenuation
- `SNR_dB`: Signal-to-Noise Ratio (dB)
- `sound_speed_gas`, `sound_speed_liquid`, `sound_speed_mixture`: Phase and mixture sound speeds (m/s)
- `viscous_atten_contribution`: Contribution from viscous absorption
- `scattering_atten_contribution`: Contribution from scattering
- `turbulence_atten_contribution`: Contribution from turbulence

**Frequencies Tested**: 100, 500, 1000, 2000, 5000, 10000 Hz

### 3. `acoustic_timeseries_data.json`
Raw time-series acoustic pressure data for detailed analysis.

**Structure** (JSON format):
```json
[
  {
    "experiment_id": 1,
    "time": [array of time values],
    "source_signal": [array of source pressure values],
    "received_signal": [array of received pressure values],
    "sampling_rate": 51200
  }
]
```

**Details**:
- Available for first 10 experiments
- Sampling rate: 51.2 kHz
- Duration: 1 second
- Includes flow-induced noise and measurement noise

### 4. `turbulence_shear_data.csv`
Turbulence statistics and shear stress measurements.

**Key Parameters**:
- `Re_gas`, `Re_liquid`: Reynolds numbers for each phase
- `friction_factor_gas`, `friction_factor_liquid`: Phase friction factors
- `tau_wall_gas`, `tau_wall_liquid`: Wall shear stress (Pa)
- `tau_interface`: Interfacial shear stress (Pa)
- `TKE_gas`, `TKE_liquid`: Turbulent kinetic energy (m²/s²)
- `dissipation_rate_gas`, `dissipation_rate_liquid`: Turbulent dissipation rate (m²/s³)
- `friction_velocity_gas`, `friction_velocity_liquid`: Friction velocity (m/s)

### 5. `velocity_profiles.csv`
Detailed velocity profiles across the pipe diameter.

**Key Parameters**:
- `experiment_id`: Links to other data
- `phase`: 'gas' or 'liquid'
- `radial_position`: Distance from wall (m)
- `normalized_position`: Normalized radial position (dimensionless)
- `velocity`: Local velocity (m/s)
- `normalized_velocity`: Normalized by superficial velocity

### 6. `dataset_summary.json`
Statistical summary of all measurements.

### 7. `metadata.json`
Comprehensive metadata including:
- Dataset description and version
- Experimental setup details
- Physical models used
- Data file descriptions
- Measurement ranges
- Usage recommendations and limitations
- References

## Operating Conditions Range

| Parameter | Minimum | Maximum | Unit |
|-----------|---------|---------|------|
| Void Fraction (α) | 0.05 | 0.95 | - |
| Gas Superficial Velocity (U_SG) | 0.5 | 15.0 | m/s |
| Liquid Superficial Velocity (U_SL) | 0.01 | 0.5 | m/s |
| Temperature | 15 | 30 | °C |
| Pressure | 1.0 | 3.0 | bar |
| Acoustic Frequency | 100 | 10,000 | Hz |

## Data Quality and Uncertainty

- **Nature**: Synthetic data generated from physical models
- **Uncertainty**: Random variations (~10%) added to simulate measurement noise
- **Physical Basis**: Based on established correlations from literature
- **Validation**: Recommended to validate against experimental data when available

## Usage Examples

### Python - Loading and Analyzing Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load flow regime data
flow_data = pd.read_csv('stratified_flow_dataset/flow_regime_characterization.csv')

# Load attenuation data
attenuation_data = pd.read_csv('stratified_flow_dataset/acoustic_attenuation_data.csv')

# Example 1: Plot attenuation vs frequency for different void fractions
import matplotlib.pyplot as plt

# Merge datasets
merged = attenuation_data.merge(flow_data[['experiment_id', 'void_fraction']], on='experiment_id')

# Plot for experiment 1
exp1_data = merged[merged['experiment_id'] == 1]
plt.figure(figsize=(10, 6))
plt.semilogy(exp1_data['frequency'], exp1_data['attenuation_coefficient'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation Coefficient (Np/m)')
plt.title('Acoustic Attenuation vs Frequency')
plt.grid(True)
plt.show()

# Example 2: Analyze attenuation mechanisms
mechanisms = ['viscous_atten_contribution', 
              'scattering_atten_contribution', 
              'turbulence_atten_contribution']

for mech in mechanisms:
    print(f"{mech}: {attenuation_data[mech].mean():.6f} Np/m")
```

### Loading Time-Series Data

```python
import json

# Load acoustic time series
with open('stratified_flow_dataset/acoustic_timeseries_data.json', 'r') as f:
    timeseries = json.load(f)

# Access data for experiment 1
exp1_timeseries = timeseries[0]
time = np.array(exp1_timeseries['time'])
source = np.array(exp1_timeseries['source_signal'])
received = np.array(exp1_timeseries['received_signal'])

# Perform FFT analysis
from scipy.fft import fft, fftfreq

N = len(source)
sampling_rate = exp1_timeseries['sampling_rate']
yf = fft(received)
xf = fftfreq(N, 1/sampling_rate)

# Plot spectrum
plt.plot(xf[:N//2], np.abs(yf[:N//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Received Signal Spectrum')
plt.show()
```

## Applications

This dataset is suitable for:

1. **Model Validation**: Compare theoretical attenuation models against synthetic experimental data
2. **Algorithm Development**: Develop signal processing algorithms for acoustic leak detection
3. **Machine Learning**: Train ML models for:
   - Flow regime classification
   - Attenuation prediction
   - Parameter estimation from acoustic signals
4. **Sensitivity Analysis**: Study effects of various parameters on attenuation
5. **Educational Purposes**: Teaching multiphase flow and acoustic phenomena
6. **Experimental Planning**: Design real experiments based on synthetic results

## Limitations and Recommendations

### Limitations
- Synthetic data may not capture all physical complexities of real systems
- Simplified models used for some phenomena (e.g., interfacial waves)
- Does not include:
  - Slug flow transitions
  - Complex pipe geometries
  - Temperature gradients
  - Chemical reactions or phase changes

### Recommendations
- **Validate** against experimental data when available
- Use as **starting point** for understanding system behavior
- **Complement** with targeted experiments for critical parameters
- Consider **uncertainty** when making design decisions

## Physical Models and References

### Flow Regime
- Taitel, Y., & Dukler, A. E. (1976). "A model for predicting flow regime transitions in horizontal and near horizontal gas-liquid flow." AIChE Journal, 22(1), 47-55.

### Acoustic Properties
- Wood, A. B. (1930). "A Textbook of Sound." G. Bell and Sons, London.
- Brennen, C. E. (2005). "Fundamentals of Multiphase Flow." Cambridge University Press.

### Two-Phase Flow
- Prosperetti, A. (2015). "Linear pressure waves in bubbly liquids: Comparison between theory and experiments." The Journal of the Acoustical Society of America, 85(2), 409-417.

### Turbulence Modeling
- Launder, B. E., & Spalding, D. B. (1974). "The numerical computation of turbulent flows." Computer Methods in Applied Mechanics and Engineering, 3(2), 269-289.

## Installation and Setup

### Requirements
```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

### Generating the Dataset

```bash
python generate_stratified_flow_dataset.py
```

This will create a `stratified_flow_dataset/` directory with all data files.

## Dataset Structure

```
stratified_flow_dataset/
├── flow_regime_characterization.csv      # Flow conditions and properties
├── acoustic_attenuation_data.csv         # Attenuation measurements
├── acoustic_timeseries_data.json         # Raw acoustic signals
├── turbulence_shear_data.csv            # Turbulence statistics
├── velocity_profiles.csv                # Velocity profiles
├── dataset_summary.json                 # Statistical summary
└── metadata.json                        # Complete metadata
```

## Citation

If you use this dataset in your research, please cite:

```
Stratified Flow Acoustic Attenuation Dataset (2025)
PhD Research: Study on the Attenuation Mechanisms in Stratified Flows: 
Beyond Single Phase Leakage Acoustics
Generated: October 2025
```

## Contact and Support

For questions, issues, or contributions related to this dataset, please refer to the research documentation or contact the research team.

## License

This dataset is provided for research and educational purposes. Please check with your institution regarding appropriate use and citation requirements.

## Version History

- **v1.0** (2025-10-12): Initial dataset release
  - 100 experimental conditions
  - 6 acoustic frequencies per experiment
  - Complete turbulence and flow characterization data
  - Time-series acoustic data for 10 experiments

## Acknowledgments

This dataset was generated using established physical models and correlations from the multiphase flow and acoustics literature. The methodology is based on decades of research in two-phase flow dynamics and acoustic wave propagation.
