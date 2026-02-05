# Acoustic Pressure Dataset: Stratified Flow Leakage Study

## Overview
This synthetic dataset simulates high-frequency acoustic pressure measurements for studying 
leak detection and wave attenuation mechanisms in stratified (multi-phase) flows.

**Research Topic:** Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics

## Important Note About File Sizes

**Large files excluded from Git:** To comply with GitHub's file size limits, the complete 
CSV files (>50MB) and ZIP archive (228MB) are not stored in the repository. 

**To generate these files:** Run the helper script after cloning:
```bash
python3 ../create_combined_files.py
```

This will create:
- Complete CSV files for each group (combining all per-second files)
- ZIP archive with all CSV files and metadata

The per-second CSV files are fully included in the repository and contain all the data.

---

## Dataset Structure

### Test Groups
The dataset contains 4 experimental groups with varying configurations:

- **Group_01**: Leak at Position A, Sensors A-B active, 17,060 Hz sampling
- **Group_02**: Leak at Position D, All sensors active, 10,000 Hz sampling
- **Group_03**: Leak at Position E, Sensors A-B active, 10,000 Hz sampling
- **Group_04**: Leak at Position A, All sensors active, 10,000 Hz sampling

### Sensors
- **Total Sensors:** 14 (PG01 through PG14)
- **Positions:** Strategically placed from 0.5m to 27.0m along the test section
- **Purpose:** Capture spatial attenuation and wave propagation characteristics

### Experimental Protocol
- **Duration:** 30 seconds per test
- **Baseline Period:** 0-5 seconds (normal operation)
- **Leak Event:** 5-20 seconds (valve open)
- **Recovery Period:** 20-30 seconds (post-leak)

## Data Files

### CSV Files
Each test group includes:
- **Per-second files** (included in git): `Group_XX_second_YY_Z.csv`
  - 30 files per group (one per second)
  - Each ~2-3 MB in size
- **Complete files** (generated locally): `Group_XX_complete_Z.csv`
  - Created by running `create_combined_files.py`
  - Combines all 30 seconds into one file
  - Group_01: ~100 MB, Others: ~59 MB each

**CSV Format:**
```
Time_s, PG01, PG02, PG03, ..., PG14
0.0000, 101325.5, 101330.2, ...
0.0001, 101324.8, 101329.5, ...
```

### Figures
1. **Fig1_Multisensor_Timeseries.png** - Multi-sensor pressure time series
2. **Fig2_Attenuation_Analysis.png** - Spatial attenuation characteristics
3. **Fig3_Frequency_Analysis.png** - FFT/PSD frequency domain analysis
4. **Fig4_Group_Comparison.png** - Inter-group leak signature comparison
5. **Fig5_Wave_Propagation.png** - Space-time wave propagation visualization

### Metadata
- **experiment_metadata.json** - Complete experimental configuration and parameters

## Physical Parameters

- **Speed of Sound:** 1500 m/s (water/stratified medium)
- **Attenuation Coefficient:** 0.05 dB/m
- **Base Pressure:** 101,325 Pa (atmospheric)
- **Leak Amplitude:** ~5,000 Pa (peak pressure change)
- **Leak Frequency:** ~100 Hz fundamental with harmonics

## Key Features

### Acoustic Signatures
- **Baseline Noise:** System noise, flow fluctuations, electrical interference
- **Leak Signal:** Multi-frequency components (fundamental + harmonics)
- **Attenuation:** Distance-dependent geometric and material damping
- **Stratification Effects:** Interface reflections and mode conversion
- **Wave Propagation:** Time delays based on speed of sound

### Analysis Capabilities
This dataset enables:
- Leak detection algorithm development
- Attenuation mechanism characterization
- Wave propagation speed validation
- Sensor placement optimization
- Frequency-domain signature analysis
- Machine learning model training

## Usage Example (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load complete dataset for Group 01
df = pd.read_csv('Group_01/Group_01_complete_A.csv')

# Plot sensor PG05
plt.figure(figsize=(12, 6))
plt.plot(df['Time_s'], df['PG05'])
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Sensor PG05 - Acoustic Pressure')
plt.axvline(5, color='r', linestyle='--', label='Valve Open')
plt.axvline(20, color='g', linestyle='--', label='Valve Close')
plt.legend()
plt.grid(True)
plt.show()
```

## Citation
If you use this dataset, please cite:
```
Synthetic Acoustic Pressure Dataset for Stratified Flow Leakage Studies
Generated: 2026
Study: Attenuation Mechanisms in Stratified Flows - Beyond Single Phase Leakage Acoustics
```

## Contact
For questions or additional information about this dataset, please refer to the 
experiment_metadata.json file for complete technical specifications.

---
**Dataset Generated:** 2026-02-05 14:05:07
**Format:** CSV (compressed in ZIP) + Figures (PNG)
**License:** Research and educational use
