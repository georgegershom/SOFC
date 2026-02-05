# 🎯 Quick Access Guide - Acoustic Pressure Dataset

## ⬇️ Download the Dataset

### Main ZIP Archive (All CSV Files)
📦 **File:** `acoustic_data/acoustic_pressure_data.zip`  
📏 **Size:** 81.36 MB  
📋 **Contains:** 120 CSV files (4 groups × 30 seconds)

**Direct Path in Repository:**
```
/workspace/acoustic_data/acoustic_pressure_data.zip
```

---

## 📚 Documentation Files

| File | Purpose | Location |
|------|---------|----------|
| **DATASET_SUMMARY.md** | Quick overview & getting started | `/workspace/DATASET_SUMMARY.md` |
| **README.md** | Complete dataset documentation | `/workspace/acoustic_data/README.md` |
| **DATASET_DOCUMENTATION.md** | Technical details & analysis | `/workspace/acoustic_data/DATASET_DOCUMENTATION.md` |
| **dataset_metadata.json** | Machine-readable parameters | `/workspace/acoustic_data/dataset_metadata.json` |

---

## 📂 CSV Data Files

### Group 01 (30 files) - Leak at Location A, 17.06 kHz
```
/workspace/acoustic_data/Group_01/Group_01_second_00.csv
/workspace/acoustic_data/Group_01/Group_01_second_01.csv
...
/workspace/acoustic_data/Group_01/Group_01_second_29.csv
```

### Group 02 (30 files) - Leak at Location D, 10 kHz
```
/workspace/acoustic_data/Group_02/Group_02_second_00.csv
...
```

### Group 03 (30 files) - Leak at Location E, 10 kHz
```
/workspace/acoustic_data/Group_03/Group_03_second_00.csv
...
```

### Group 04 (30 files) - Leak at Location A, 10 kHz
```
/workspace/acoustic_data/Group_04/Group_04_second_00.csv
...
```

---

## 🖼️ Visualization Figures (17 PNG files)

### Comparison Figure
```
/workspace/acoustic_data/figures/all_groups_comparison.png
```

### Group-Specific Figures (4 per group)
```
/workspace/acoustic_data/figures/Group_01/
├── Group_01_time_series.png
├── Group_01_spectrogram.png
├── Group_01_spatial_analysis.png
└── Group_01_waterfall.png

(Similar structure for Groups 02, 03, 04)
```

---

## 🔧 Generator Script
```
/workspace/generate_acoustic_data.py
```
Python script used to generate all data (can be re-run with modifications)

---

## 📊 Key Dataset Features

### 4 Test Groups
- **Group_01:** Leak A (6m), 17.06 kHz, intensity 1.0
- **Group_02:** Leak D (13.5m), 10 kHz, intensity 0.8
- **Group_03:** Leak E (24m), 10 kHz, intensity 0.9
- **Group_04:** Leak A (6m), 10 kHz, intensity 0.85

### 14 Sensors (PG01-PG14)
Positioned every 2.5m along 32.5m pipeline

### 3 Phases per Test
- 0-5s: Baseline (normal operation)
- 5-20s: Leak event (valve open)
- 20-30s: Recovery (valve closed)

### Physical Model
- Speed of sound: 1500 m/s
- Attenuation: 0.15 dB/m
- Fundamental freq: 850 Hz + harmonics
- Broadband: 500-4900 Hz

---

## 💻 Quick Load Examples

### Python
```python
import pandas as pd
df = pd.read_csv('acoustic_data/Group_01/Group_01_second_05.csv')
print(df.head())
```

### MATLAB
```matlab
data = readtable('acoustic_data/Group_01/Group_01_second_05.csv');
```

### R
```r
library(readr)
df <- read_csv('acoustic_data/Group_01/Group_01_second_05.csv')
```

---

## ✅ Status

**All tasks completed successfully:**
- ✅ 120 CSV files generated
- ✅ 17 visualization figures created
- ✅ 4 documentation files written
- ✅ ZIP archive created (81.36 MB)
- ✅ All files committed to git
- ✅ Pushed to branch: `cursor/core-time-series-pressure-data-7e9a`

---

## 🚀 Start Here

1. Download `acoustic_data/acoustic_pressure_data.zip`
2. Read `DATASET_SUMMARY.md` for overview
3. Check `acoustic_data/figures/` for visualizations
4. Load CSV files using examples above
5. Refer to `acoustic_data/DATASET_DOCUMENTATION.md` for analysis methods

---

**Repository:** georgegershom/SOFC  
**Branch:** cursor/core-time-series-pressure-data-7e9a  
**Generated:** February 5, 2026
