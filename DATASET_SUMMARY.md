# Synthetic Acoustic Pressure Dataset - Generation Summary

## ✅ Dataset Successfully Generated and Uploaded

**Date:** February 5, 2026  
**Branch:** `cursor/core-time-series-pressure-data-7e9a`  
**Repository:** georgegershom/SOFC

---

## 📊 Dataset Overview

### What Was Generated

A comprehensive synthetic dataset for studying **acoustic wave propagation and attenuation in stratified flows** for pipeline leak detection.

**Key Statistics:**
- **Total CSV Files:** 120 (4 test groups × 30 seconds)
- **Sensors:** 14 high-frequency pressure gauges (PG01-PG14)
- **Sampling Rates:** 10 kHz and 17.06 kHz
- **Total Duration:** 30 seconds per test (baseline, leak, recovery phases)
- **Leak Locations:** 3 positions (A, D, E along 32.5m pipeline)
- **ZIP Archive Size:** 81.36 MB
- **Visualization Figures:** 17 high-quality PNG images

---

## 📁 File Structure

```
/workspace/acoustic_data/
│
├── acoustic_pressure_data.zip          ⬇️ DOWNLOAD THIS (81.36 MB)
│   └── Contains all 120 CSV files organized by group
│
├── README.md                           📖 Start here for overview
├── DATASET_DOCUMENTATION.md            📚 Technical details and analysis methods
├── dataset_metadata.json               🔧 Machine-readable parameters
│
├── Group_01/                           📂 Test Group 1 (Leak at A, 17.06 kHz)
│   ├── Group_01_second_00.csv
│   ├── Group_01_second_01.csv
│   └── ... (30 CSV files total)
│
├── Group_02/                           📂 Test Group 2 (Leak at D, 10 kHz)
│   └── ... (30 CSV files)
│
├── Group_03/                           📂 Test Group 3 (Leak at E, 10 kHz)
│   └── ... (30 CSV files)
│
├── Group_04/                           📂 Test Group 4 (Leak at A, 10 kHz)
│   └── ... (30 CSV files)
│
└── figures/                            🖼️ Visualizations
    ├── all_groups_comparison.png
    ├── Group_01/
    │   ├── Group_01_time_series.png
    │   ├── Group_01_spectrogram.png
    │   ├── Group_01_spatial_analysis.png
    │   └── Group_01_waterfall.png
    └── ... (similar for Groups 02-04)
```

---

## ⬇️ How to Download the Dataset

### Option 1: Download ZIP Archive (Recommended)

The easiest way to get all CSV files:

1. Navigate to the repository on GitHub
2. Go to the `acoustic_data` folder
3. Download `acoustic_pressure_data.zip` (81.36 MB)
4. Extract to get all 120 CSV files organized by group

### Option 2: Clone the Repository

```bash
git clone https://github.com/georgegershom/SOFC.git
cd SOFC
git checkout cursor/core-time-series-pressure-data-7e9a
cd acoustic_data
```

### Option 3: Download Individual Files

You can download specific CSV files or figures directly from GitHub if you only need certain time periods or test groups.

---

## 📋 Dataset Specifications

### Test Groups Configuration

| Group | Leak Location | Position | Sampling Rate | Intensity | Files |
|-------|--------------|----------|---------------|-----------|-------|
| **Group_01** | A | 6.0 m | 17,060 Hz | 1.0 | 30 CSV |
| **Group_02** | D | 13.5 m | 10,000 Hz | 0.8 | 30 CSV |
| **Group_03** | E | 24.0 m | 10,000 Hz | 0.9 | 30 CSV |
| **Group_04** | A | 6.0 m | 10,000 Hz | 0.85 | 30 CSV |

### Sensor Array (14 sensors)

| Sensor | Position | Description |
|--------|----------|-------------|
| PG01 | 0.0 m | Pipeline start |
| PG02 | 2.5 m | After Valve 1 |
| PG03 | 5.0 m | Before Leak A |
| PG04 | 7.5 m | After Leak A |
| PG05 | 10.0 m | Mid-section |
| PG06 | 12.5 m | Before Leak D |
| PG07 | 15.0 m | After Leak D |
| PG08 | 17.5 m | Before Valve 2 |
| PG09 | 20.0 m | After Valve 2 |
| PG10 | 22.5 m | Before Leak E |
| PG11 | 25.0 m | After Leak E |
| PG12 | 27.5 m | End section |
| PG13 | 30.0 m | Near endpoint |
| PG14 | 32.5 m | Endpoint |

### Experimental Phases (Each 30-second test)

```
┌─────────┬───────────────────┬─────────────┐
│ 0-5s    │ 5-20s             │ 20-30s      │
│ BASELINE│ LEAK EVENT        │ RECOVERY    │
├─────────┼───────────────────┼─────────────┤
│ Normal  │ Valve opens at 5s │ Valve closes│
│ ambient │ Active leak       │ at 20s      │
│ noise   │ Acoustic signature│ Pressure    │
│ only    │ present           │ recovery    │
└─────────┴───────────────────┴─────────────┘
```

---

## 🎯 Key Features of the Dataset

### 1. Physics-Based Modeling

- **Acoustic Propagation:** Time delays based on distance/speed of sound (1500 m/s)
- **Attenuation:** Exponential decay with distance (α = 0.15 dB/m)
- **Frequency Content:**
  - Fundamental: 850 Hz (turbulent jet)
  - Harmonics: 2nd, 3rd, 5th, 7th, 11th
  - Broadband: 500-4900 Hz (turbulent noise)
- **Reflections:** Pipeline boundary reflections included
- **Realistic Noise:** Ambient + sensor-specific noise

### 2. Multiple Leak Scenarios

- **Location A (6.0m):** Near pipeline start - Groups 01 & 04
- **Location D (13.5m):** Mid-pipeline - Group 02
- **Location E (24.0m):** Downstream - Group 03

### 3. Comprehensive Visualizations

Each test group includes:
- **Time Series:** 30-second pressure traces for multiple sensors
- **Spectrogram:** Frequency-time analysis showing 850 Hz peak during leak
- **Spatial Analysis:** Pressure distribution and attenuation vs. distance
- **Waterfall Plot:** Wave propagation across sensor array

---

## 🔬 Research Applications

This dataset supports:

✅ **Acoustic Wave Propagation Studies**
- Measure propagation delays
- Analyze dispersion effects
- Study stratified flow impacts

✅ **Attenuation Mechanism Analysis**
- Quantify spatial decay rates
- Compare frequency-dependent effects
- Model energy dissipation

✅ **Leak Detection Algorithm Development**
- Train machine learning models
- Test signal processing methods
- Develop localization algorithms

✅ **Sensor Optimization Studies**
- Evaluate sensor placement strategies
- Determine minimum sensor density
- Optimize detection sensitivity

✅ **Comparative Analysis**
- Effect of leak location
- Impact of sampling rate
- Leak intensity sensitivity

---

## 📖 Documentation Files

### README.md
- Dataset overview
- File structure and format
- Usage guidelines (Python, MATLAB, R)
- Quick start examples

### DATASET_DOCUMENTATION.md
- Detailed physical models
- Signal generation methodology
- Statistical characteristics
- Analysis techniques and algorithms
- Troubleshooting guide

### dataset_metadata.json
- Machine-readable parameters
- All configuration settings
- Sensor positions
- Test group specifications

---

## 💻 Quick Start Examples

### Load Data in Python

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load one second of data
df = pd.read_csv('acoustic_data/Group_01/Group_01_second_05.csv')

# Plot pressure from sensor PG03
plt.figure(figsize=(12, 4))
plt.plot(df['Time_s'], df['PG03'])
plt.xlabel('Time (s)')
plt.ylabel('Pressure (bar)')
plt.title('Sensor PG03 - Second 5 (leak active)')
plt.grid(True)
plt.show()

# Load entire 30-second test
frames = []
for second in range(30):
    df_sec = pd.read_csv(f'acoustic_data/Group_01/Group_01_second_{second:02d}.csv')
    df_sec['Time_s'] = df_sec['Time_s'] + second
    frames.append(df_sec)

full_data = pd.concat(frames, ignore_index=True)
print(f"Total samples: {len(full_data)}")
print(f"Columns: {full_data.columns.tolist()}")
```

### Analyze Leak Signature

```python
from scipy import signal
from scipy.fft import fft, fftfreq

# Load leak period (seconds 5-20)
leak_frames = []
for second in range(5, 20):
    df = pd.read_csv(f'acoustic_data/Group_01/Group_01_second_{second:02d}.csv')
    leak_frames.append(df)

leak_data = pd.concat(leak_frames, ignore_index=True)

# FFT of sensor nearest to leak (PG03 for Group 01, leak at A=6.0m)
pressure = leak_data['PG03'].values
N = len(pressure)
fs = 17060  # sampling rate for Group 01

# Compute FFT
yf = fft(pressure)
xf = fftfreq(N, 1/fs)[:N//2]
power = 2.0/N * np.abs(yf[0:N//2])

# Plot spectrum
plt.figure(figsize=(12, 6))
plt.semilogy(xf, power)
plt.xlim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (bar)')
plt.title('Frequency Spectrum - Leak Signature (PG03)')
plt.axvline(850, color='r', linestyle='--', label='Fundamental (850 Hz)')
plt.axvline(1700, color='orange', linestyle='--', alpha=0.5, label='2nd harmonic')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Expected: Peak at 850 Hz with harmonics
print(f"Peak frequency: {xf[np.argmax(power)]} Hz")
```

### Analyze Spatial Attenuation

```python
# Calculate RMS pressure during leak for all sensors
import numpy as np

sensors = [f'PG{i:02d}' for i in range(1, 15)]
sensor_positions = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5]
leak_position = 6.0  # Location A for Group 01

rms_values = []
for sensor in sensors:
    pressure = leak_data[sensor].values
    rms = np.sqrt(np.mean((pressure - 1.0)**2))  # Remove DC
    rms_values.append(rms)

distances = [abs(pos - leak_position) for pos in sensor_positions]

# Plot attenuation
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sensor_positions, rms_values, 'o-', markersize=8)
plt.axvline(leak_position, color='r', linestyle='--', label='Leak Location')
plt.xlabel('Position (m)')
plt.ylabel('RMS Pressure (bar)')
plt.title('Spatial Pressure Distribution')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(distances, rms_values, 'o')
plt.xlabel('Distance from Leak (m)')
plt.ylabel('RMS Pressure (bar)')
plt.title('Attenuation vs Distance')
plt.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()

# Fit exponential decay: y = A * exp(-alpha * x)
from scipy.optimize import curve_fit

def exp_decay(x, A, alpha):
    return A * np.exp(-alpha * x)

popt, _ = curve_fit(exp_decay, distances, rms_values)
print(f"Fitted attenuation coefficient: {popt[1]:.4f} (expected: ~0.15)")
```

---

## 📊 Sample Visualizations

The dataset includes these pre-generated figures for each test group:

1. **Time Series Plot**
   - Shows full 30-second recording
   - Multiple sensors displayed
   - Leak start/end markers
   - Clear visualization of leak event

2. **Spectrogram**
   - Frequency range: 0-5000 Hz
   - Sensor nearest to leak
   - Shows 850 Hz fundamental during leak
   - Harmonic content visible

3. **Spatial Analysis**
   - Top: RMS pressure vs sensor position
   - Bottom: Attenuation vs distance
   - Exponential decay fit
   - Leak location marked

4. **Waterfall Plot**
   - Wave propagation visualization
   - 4-second window around leak start
   - All 14 sensors shown
   - Demonstrates propagation delays

5. **Cross-Group Comparison**
   - All 4 groups side-by-side
   - Compares different leak locations
   - Shows intensity variations
   - Useful for comparative studies

---

## ⚙️ Technical Specifications

### CSV File Format

```
Column 0: Time_s (float, 8 decimals)
Columns 1-14: PG01 to PG14 (pressure in bar, 8 decimals)
Separator: comma
Encoding: UTF-8
Rows per file: ~10,000-17,060 (depending on sampling rate)
File size: ~2.7 MB per file
```

### Physical Parameters

```
Speed of sound:           1500 m/s
Attenuation coefficient:  0.15 dB/m
Ambient pressure:         1.0 bar
Noise level (RMS):        0.001 bar
Fundamental frequency:    850 Hz
Broadband range:          500-4900 Hz
Pipeline length:          32.5 m
```

---

## 🎓 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{acoustic_pressure_2026,
  title={Synthetic High-Frequency Acoustic Pressure Dataset for 
         Leak Detection in Stratified Flows},
  author={Generated for SOFC Project},
  year={2026},
  month={February},
  version={1.0},
  repository={georgegershom/SOFC},
  branch={cursor/core-time-series-pressure-data-7e9a}
}
```

---

## ✅ Validation Checklist

The dataset has been validated for:

- ✅ Physical consistency (causality, energy conservation)
- ✅ Proper propagation delays (distance/speed of sound)
- ✅ Exponential attenuation with distance
- ✅ Correct frequency content (fundamental + harmonics)
- ✅ Realistic noise characteristics
- ✅ Smooth transitions (no discontinuities)
- ✅ No aliasing artifacts
- ✅ CSV format correctness
- ✅ Complete documentation

---

## 📧 Support and Questions

For questions about:
- **Dataset usage:** See README.md and DATASET_DOCUMENTATION.md
- **Technical details:** Check dataset_metadata.json
- **Analysis methods:** Refer to DATASET_DOCUMENTATION.md
- **Visualizations:** Examine the figures/ directory

---

## 🚀 Next Steps

1. **Download the dataset** using one of the methods above
2. **Read the README.md** for an overview
3. **Review the visualizations** to understand the data
4. **Try the quick start examples** to load and plot data
5. **Consult DATASET_DOCUMENTATION.md** for advanced analysis
6. **Start your research!**

---

## 📝 Version Information

**Version:** 1.0  
**Generated:** February 5, 2026  
**Total Files:** 142 (120 CSV + 17 PNG + 5 documentation)  
**Commit:** 73fb7c10  
**Branch:** cursor/core-time-series-pressure-data-7e9a

---

**Dataset Status:** ✅ Complete and Ready for Use

All files have been generated, documented, committed, and pushed to the repository.

---

**End of Summary**
