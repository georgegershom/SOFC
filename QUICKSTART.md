# Quick Start Guide

## Stratified Flow Acoustic Attenuation Dataset

This guide will help you quickly get started with using the dataset.

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate the Dataset

```bash
python generate_stratified_flow_dataset.py
```

This creates a `stratified_flow_dataset/` directory with all data files.

### Step 3: Analyze and Visualize

```bash
python analyze_dataset.py
```

This generates comprehensive visualization plots.

---

## 📁 Dataset Structure

```
stratified_flow_dataset/
├── flow_regime_characterization.csv      # 100 experiments × 17 parameters
├── acoustic_attenuation_data.csv         # 600 measurements (6 frequencies × 100 exp)
├── acoustic_timeseries_data.json         # Time-series for 10 experiments
├── turbulence_shear_data.csv            # Turbulence data for 100 experiments
├── velocity_profiles.csv                 # 4000 velocity points
├── dataset_summary.json                  # Statistical summary
└── metadata.json                         # Complete metadata
```

---

## 💻 Basic Usage Examples

### Example 1: Load and Explore Data

```python
import pandas as pd
import numpy as np

# Load flow regime data
flow_data = pd.read_csv('stratified_flow_dataset/flow_regime_characterization.csv')
print(flow_data.head())

# Load attenuation data
attenuation = pd.read_csv('stratified_flow_dataset/acoustic_attenuation_data.csv')
print(attenuation.head())

# Basic statistics
print("\nFlow Patterns Distribution:")
print(flow_data['flow_pattern'].value_counts())

print("\nVoid Fraction Statistics:")
print(flow_data['void_fraction'].describe())
```

### Example 2: Analyze Attenuation vs Frequency

```python
import matplotlib.pyplot as plt

# Select experiment 1
exp1 = attenuation[attenuation['experiment_id'] == 1]

# Plot attenuation vs frequency
plt.figure(figsize=(10, 6))
plt.loglog(exp1['frequency'], exp1['attenuation_coefficient'], 'o-', linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation Coefficient (Np/m)')
plt.title('Acoustic Attenuation vs Frequency')
plt.grid(True, which='both', alpha=0.3)
plt.show()
```

### Example 3: Compare Flow Patterns

```python
import seaborn as sns

# Merge datasets
merged = attenuation.merge(flow_data[['experiment_id', 'flow_pattern', 'void_fraction']], 
                           on='experiment_id')

# Compare attenuation for different flow patterns at 1000 Hz
data_1kHz = merged[merged['frequency'] == 1000]

plt.figure(figsize=(10, 6))
sns.boxplot(x='flow_pattern', y='attenuation_coefficient', data=data_1kHz)
plt.ylabel('Attenuation Coefficient (Np/m)')
plt.title('Attenuation Comparison: Smooth vs Wavy Stratified (1000 Hz)')
plt.show()
```

### Example 4: Analyze Acoustic Time Series

```python
import json
from scipy.fft import fft, fftfreq

# Load time-series data
with open('stratified_flow_dataset/acoustic_timeseries_data.json', 'r') as f:
    timeseries = json.load(f)

# Get experiment 1 data
exp1_ts = timeseries[0]
time = np.array(exp1_ts['time'])
received = np.array(exp1_ts['received_signal'])

# Compute FFT
N = len(received)
sr = exp1_ts['sampling_rate']
yf = fft(received)
xf = fftfreq(N, 1/sr)

# Plot spectrum
plt.figure(figsize=(12, 6))
plt.semilogy(xf[:N//2], np.abs(yf[:N//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Received Signal Frequency Spectrum')
plt.xlim(0, 5000)
plt.grid(True)
plt.show()
```

### Example 5: Turbulence Analysis

```python
# Load turbulence data
turbulence = pd.read_csv('stratified_flow_dataset/turbulence_shear_data.csv')

# Merge with flow data
merged = turbulence.merge(flow_data[['experiment_id', 'U_SG', 'void_fraction']], 
                         on='experiment_id')

# Plot TKE vs gas velocity
plt.figure(figsize=(10, 6))
scatter = plt.scatter(merged['U_SG'], merged['TKE_gas'], 
                     c=merged['void_fraction'], cmap='viridis', 
                     s=80, alpha=0.7, edgecolors='k')
plt.xlabel('Gas Superficial Velocity (m/s)')
plt.ylabel('Turbulent Kinetic Energy (m²/s²)')
plt.title('TKE vs Gas Velocity (colored by void fraction)')
plt.colorbar(scatter, label='Void Fraction')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🔍 Key Parameters Reference

### Flow Regime Parameters
- **U_SG**: Gas superficial velocity (m/s)
- **U_SL**: Liquid superficial velocity (m/s)
- **void_fraction**: Gas volume fraction (0-1)
- **flow_pattern**: smooth_stratified or wavy_stratified
- **interface_height**: Normalized interface position (0-1)
- **wave_amplitude**: Interface wave amplitude (m)

### Acoustic Parameters
- **frequency**: Acoustic frequency tested (Hz)
- **attenuation_coefficient**: Total attenuation (Np/m)
- **transmission_loss_dB**: Loss across pipe (dB)
- **SNR_dB**: Signal-to-noise ratio (dB)
- **sound_speed_mixture**: Effective sound speed in two-phase flow (m/s)

### Turbulence Parameters
- **Re_gas, Re_liquid**: Reynolds numbers
- **TKE_gas, TKE_liquid**: Turbulent kinetic energy (m²/s²)
- **tau_interface**: Interfacial shear stress (Pa)
- **dissipation_rate**: Turbulent dissipation (m²/s³)

---

## 📊 Research Applications

### 1. Attenuation Modeling
Study how different mechanisms contribute to acoustic attenuation:
- Viscous absorption
- Scattering from interface waves
- Turbulence effects

### 2. Flow Pattern Classification
Use acoustic signatures to classify flow regimes:
- Train ML models on acoustic features
- Predict flow patterns from attenuation spectra

### 3. Parameter Estimation
Develop inverse methods to estimate:
- Void fraction from acoustic measurements
- Interface wave characteristics
- Flow velocities

### 4. Sensitivity Analysis
Investigate effects of:
- Gas/liquid velocity ratios
- Void fraction on attenuation
- Frequency-dependent behavior

---

## 🎯 Next Steps

1. **Explore the data**: Use `analyze_dataset.py` to generate visualizations
2. **Read the documentation**: Check `README_DATASET.md` for detailed information
3. **Experiment**: Modify parameters in `generate_stratified_flow_dataset.py`
4. **Validate**: Compare results with experimental data if available
5. **Extend**: Add your own analysis scripts and models

---

## 📚 Additional Resources

- **Dataset Documentation**: `README_DATASET.md`
- **Metadata**: `stratified_flow_dataset/metadata.json`
- **Summary Statistics**: `stratified_flow_dataset/dataset_summary.json`

---

## ⚠️ Important Notes

- This is **synthetic data** based on physical models
- **Validate** against real experiments when possible
- Use for **algorithm development**, **preliminary analysis**, and **education**
- Random variations (~10%) simulate measurement uncertainty

---

## 🐛 Troubleshooting

### Issue: Module not found
```bash
pip install numpy pandas scipy matplotlib seaborn
```

### Issue: Permission denied
```bash
chmod +x generate_stratified_flow_dataset.py
```

### Issue: Dataset not found
Make sure to run the generator first:
```bash
python generate_stratified_flow_dataset.py
```

---

## 📧 Questions?

Refer to the comprehensive `README_DATASET.md` or check the metadata file for detailed information about the dataset structure and physical models used.

---

**Happy Analyzing! 🚀**
