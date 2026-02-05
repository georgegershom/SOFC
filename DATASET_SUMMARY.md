# Acoustic Pressure Dataset - Generation Summary

## Dataset Generated Successfully! ✅

### Overview
This repository contains a comprehensive synthetic dataset for studying acoustic wave propagation and attenuation mechanisms in stratified flows during leak events.

**Research Topic:** Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics

---

## 📦 Dataset Contents

### Main Files
- **`generate_acoustic_data.py`**
  - Complete Python script for generating the full dataset
  - Fully documented with physical models
  - Can be customized for different scenarios

- **`create_combined_files.py`**
  - Helper script to create combined CSV files and ZIP archive
  - Reconstructs large files from per-second data
  - Run after cloning: `python3 create_combined_files.py`

**Note:** Large files (complete CSV files >50MB and ZIP archive) are excluded from git 
to comply with GitHub size limits. Use `create_combined_files.py` to regenerate them 
from the individual per-second CSV files included in the repository.

### Directory Structure
```
acoustic_pressure_dataset/
├── experiment_metadata.json               (Experiment configuration)
├── README.md                               (Detailed documentation)
├── figures/                                (5 visualization figures)
│   ├── Fig1_Multisensor_Timeseries.png
│   ├── Fig2_Attenuation_Analysis.png
│   ├── Fig3_Frequency_Analysis.png
│   ├── Fig4_Group_Comparison.png
│   └── Fig5_Wave_Propagation.png
├── Group_01/                               (30 per-second CSV files, Leak @ A, 17.06 kHz)
│   ├── Group_01_second_00_A.csv
│   ├── Group_01_second_01_A.csv
│   └── ... (through second_29)
├── Group_02/                               (30 per-second CSV files, Leak @ D, 10 kHz)
├── Group_03/                               (30 per-second CSV files, Leak @ E, 10 kHz)
└── Group_04/                               (30 per-second CSV files, Leak @ A, 10 kHz)

After running create_combined_files.py:
├── Group_01/Group_01_complete_A.csv       (100 MB - generated)
├── Group_02/Group_02_complete_D.csv       (59 MB - generated)
├── Group_03/Group_03_complete_E.csv       (59 MB - generated)
├── Group_04/Group_04_complete_A.csv       (59 MB - generated)
└── acoustic_pressure_data_all_groups.zip  (228 MB - generated)
```

---

## 📊 Dataset Specifications

### Test Groups Configuration

| Group | Leak Location | Sampling Rate | Sensor Config | Duration |
|-------|---------------|---------------|---------------|----------|
| Group_01 | Position A (5.0m) | 17,060 Hz | A,B | 30s |
| Group_02 | Position D (15.0m) | 10,000 Hz | A,B,C,D | 30s |
| Group_03 | Position E (22.0m) | 10,000 Hz | A,B | 30s |
| Group_04 | Position A (5.0m) | 10,000 Hz | A,B,C,D | 30s |

### Sensor Array (14 Sensors)
- **Sensors:** PG01 through PG14
- **Positions:** 0.5m to 27.0m along test section
- **Coverage:** Strategic placement before/after valves and leak points

### Experimental Timeline
- **0-5 seconds:** Baseline (normal operation)
- **5-20 seconds:** Active leak (valve open)
- **20-30 seconds:** Recovery (post-leak)

---

## 🔬 Physical Parameters & Models

### Acoustic Properties
- **Speed of Sound:** 1,500 m/s (water/stratified medium)
- **Attenuation Coefficient:** 0.05 dB/m
- **Base Pressure:** 101,325 Pa (atmospheric)
- **Leak Amplitude:** ~5,000 Pa peak pressure change

### Signal Components
1. **Baseline Noise:**
   - White noise (σ = 50 Pa)
   - Low-frequency drift (0.5 Hz, 30 Pa)
   - Electrical noise (50 Hz, 20 Pa)

2. **Leak Signature:**
   - Fundamental frequency: ~100 Hz
   - Harmonic components (2f, 3f)
   - Turbulence effects
   - Rise time envelope (~0.5s)

3. **Attenuation Effects:**
   - Geometric spreading (1/(1+distance))
   - Material absorption (exp(-α·distance))
   - Stratification interface reflections

4. **Wave Propagation:**
   - Time delays based on speed of sound
   - Distance-dependent amplitude decay
   - Post-leak exponential decay

---

## 📈 Visualization Figures

### Figure 1: Multi-Sensor Time Series
Shows synchronized pressure measurements from 8 sensors during the complete leak event cycle.

### Figure 2: Attenuation Analysis
Spatial attenuation characteristics comparing measured vs. theoretical models across all test groups.

### Figure 3: Frequency Domain Analysis
Power spectral density (PSD) analysis showing frequency content during baseline, active leak, and post-leak periods.

### Figure 4: Group Comparison
Inter-group comparison of leak signatures demonstrating effects of different leak locations and sampling rates.

### Figure 5: Wave Propagation
Space-time visualization showing acoustic wave propagation and speed validation.

---

## 💾 Data Format

### CSV File Structure
Each CSV file contains time-series data with columns:
```
Time_s, PG01, PG02, PG03, PG04, ..., PG14
```

**Example Data:**
```csv
Time_s,PG01,PG02,PG03,...,PG14
0.000000,101325.523,101330.234,101328.456,...
0.000059,101324.891,101329.567,101327.892,...
0.000117,101326.234,101331.123,101329.345,...
```

### File Naming Convention
- **Per-second files:** `Group_XX_second_YY_Z.csv`
  - XX = Group number (01-04)
  - YY = Second number (00-29)
  - Z = Leak location (A, D, or E)

- **Complete files:** `Group_XX_complete_Z.csv`
  - Full 30-second dataset in single file

---

## 🚀 Quick Start Usage

### Step 1: Generate Combined Files (First Time)
```bash
# After cloning the repository, run:
python3 create_combined_files.py
```

This creates the complete CSV files and ZIP archive from the per-second data files.

### Step 2: Load and Visualize Data (Python)
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load Group 01 complete dataset (after running create_combined_files.py)
df = pd.read_csv('acoustic_pressure_dataset/Group_01/Group_01_complete_A.csv')

# Or load individual per-second file directly
df_second = pd.read_csv('acoustic_pressure_dataset/Group_01/Group_01_second_05_A.csv')

# Plot sensor PG05
plt.figure(figsize=(14, 6))
plt.plot(df['Time_s'], df['PG05']/1000, linewidth=0.8)
plt.axvline(5, color='r', linestyle='--', label='Valve Open')
plt.axvline(20, color='g', linestyle='--', label='Valve Close')
plt.axvspan(5, 20, alpha=0.1, color='red')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (kPa)')
plt.title('Sensor PG05 - Acoustic Pressure During Leak Event')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('my_analysis.png', dpi=300)
plt.show()
```

### Frequency Analysis
```python
import numpy as np

# Load data
df = pd.read_csv('extracted_data/Group_01/Group_01_complete_A.csv')
sampling_rate = 17060  # Hz for Group 01

# Extract leak period (5-20 seconds)
leak_start = int(5 * sampling_rate)
leak_end = int(20 * sampling_rate)
leak_signal = df['PG05'].values[leak_start:leak_end]

# FFT
fft_vals = np.fft.fft(leak_signal)
fft_freq = np.fft.fftfreq(len(leak_signal), 1/sampling_rate)
psd = np.abs(fft_vals)**2 / len(leak_signal)

# Plot frequency spectrum
plt.figure(figsize=(12, 6))
positive_freq = fft_freq > 0
plt.semilogy(fft_freq[positive_freq], psd[positive_freq])
plt.xlim([0, 500])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Leak Signature Frequency Content')
plt.grid(True, alpha=0.3)
plt.show()
```

### Calculate Attenuation
```python
# Sensor positions (meters from origin)
sensor_positions = {
    'PG01': 0.5, 'PG02': 1.2, 'PG03': 2.0, 'PG04': 3.5,
    'PG05': 5.0, 'PG06': 6.8, 'PG07': 8.5, 'PG08': 10.2,
    'PG09': 12.0, 'PG10': 14.5, 'PG11': 17.0, 'PG12': 20.0,
    'PG13': 23.5, 'PG14': 27.0
}

leak_position = 5.0  # meters (Position A for Group 01)

# Calculate peak amplitude during leak for each sensor
amplitudes = {}
for sensor in ['PG01', 'PG02', 'PG03', 'PG04', 'PG05', 'PG06']:
    signal = df[sensor].values[leak_start:leak_end]
    baseline = df[sensor].values[:leak_start].mean()
    peak_amplitude = np.max(np.abs(signal - baseline))
    distance = abs(sensor_positions[sensor] - leak_position)
    amplitudes[sensor] = (distance, peak_amplitude)

# Plot attenuation curve
distances = [v[0] for v in amplitudes.values()]
peaks = [v[1] for v in amplitudes.values()]

plt.figure(figsize=(10, 6))
plt.scatter(distances, peaks, s=100, c='blue', edgecolors='black')
plt.xlabel('Distance from Leak (m)')
plt.ylabel('Peak Pressure Amplitude (Pa)')
plt.title('Acoustic Attenuation vs Distance')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📚 Research Applications

This dataset is suitable for:

1. **Leak Detection Algorithm Development**
   - Machine learning model training
   - Signal processing technique validation
   - Detection threshold optimization

2. **Attenuation Mechanism Studies**
   - Geometric spreading analysis
   - Material absorption characterization
   - Stratification interface effects

3. **Wave Propagation Research**
   - Speed of sound validation
   - Time-of-flight analysis
   - Multi-path propagation studies

4. **Sensor Network Optimization**
   - Placement strategy evaluation
   - Coverage analysis
   - Redundancy studies

5. **Frequency Domain Analysis**
   - Leak signature characterization
   - Harmonic content analysis
   - Noise filtering techniques

---

## 📖 Documentation

Detailed documentation is available in:
- **`acoustic_pressure_dataset/README.md`** - Complete dataset documentation
- **`acoustic_pressure_dataset/experiment_metadata.json`** - Technical specifications
- **`generate_acoustic_data.py`** - Inline code documentation

---

## 🔄 Reproducibility

The entire dataset can be regenerated with:
```bash
python3 generate_acoustic_data.py
```

This ensures:
- Complete reproducibility (fixed random seed)
- Ability to modify parameters
- Generation of custom scenarios
- Verification of results

---

## 📊 Data Statistics

### File Counts (in Git)
- **Per-second CSV files:** 120 (4 groups × 30 seconds)
- **Visualization figures:** 5
- **Metadata files:** 2 (JSON + README)
- **Scripts:** 2 (generator + combiner)

### Additional Files (Generated Locally)
- **Complete time series files:** 4 (one per group, via create_combined_files.py)
- **ZIP archive:** 1 (all files, via create_combined_files.py)

### Data Volume
- **Git repository size:** ~542 MB (per-second CSV files + figures)
- **Total uncompressed:** ~800 MB (with complete CSV files)
- **Compressed (ZIP):** 228 MB (generated via create_combined_files.py)
- **Compression ratio:** ~3.5:1

### Data Points
- **Group_01:** 511,800 samples/sensor × 14 sensors = 7,165,200 data points
- **Groups_02-04:** 300,000 samples/sensor × 14 sensors = 4,200,000 data points each
- **Total:** ~19.8 million pressure measurements

---

## 🎯 Key Features

✅ **Realistic acoustic signatures** with proper wave physics
✅ **Multiple test configurations** for comprehensive analysis
✅ **High-frequency sampling** (10-17 kHz) capturing wave details
✅ **Spatial sensor array** enabling propagation studies
✅ **Multiple leak locations** for spatial analysis
✅ **Stratified flow effects** including interface phenomena
✅ **Complete metadata** for full experimental context
✅ **Professional visualizations** ready for publication
✅ **Easy-to-use format** (standard CSV files)
✅ **Reproducible generation** with documented code

---

## 📞 Support

For questions, issues, or additional requirements, please refer to:
- Dataset README: `acoustic_pressure_dataset/README.md`
- Generation script: `generate_acoustic_data.py` (fully documented)
- Metadata file: `acoustic_pressure_dataset/experiment_metadata.json`

---

## 📄 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{acoustic_pressure_stratified_flow_2026,
  title={Synthetic Acoustic Pressure Dataset for Stratified Flow Leakage Studies},
  author={},
  year={2026},
  note={Study: Attenuation Mechanisms in Stratified Flows - Beyond Single Phase Leakage Acoustics},
  url={https://github.com/...}
}
```

---

**Generated:** February 5, 2026  
**Format:** CSV (compressed) + PNG figures  
**License:** Research and educational use  
**Version:** 1.0
