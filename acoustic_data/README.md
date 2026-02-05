# High-Frequency Acoustic Pressure Dataset
## Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics

---

## Dataset Overview

This dataset contains synthetic high-frequency acoustic pressure data for analyzing acoustic wave propagation and attenuation in pipeline leak detection experiments. The data is designed to support research on leak detection in stratified flows with multiple sensor configurations and leak locations.

**Generated:** February 5, 2026  
**Version:** 1.0  
**Total Size:** 81.36 MB (compressed)

---

## Dataset Description

### Experimental Configuration

This dataset simulates a controlled leak detection experiment on a pipeline system with the following characteristics:

- **Total Duration:** 30 seconds per test
- **Sampling Rates:** 
  - Group 01: 17,060 Hz (high-resolution)
  - Groups 02-04: 10,000 Hz (standard)
- **Sensors:** 14 high-frequency pressure gauges (PG01-PG14)
- **Pipeline Length:** 32.5 meters
- **Speed of Sound:** 1500 m/s (water medium)
- **Attenuation Coefficient:** 0.15 dB/m @ 1 kHz

### Experimental Phases

Each test recording contains three distinct phases:

1. **Baseline Phase (0-5s)**
   - Normal operation conditions
   - Ambient pressure: 1.0 bar
   - Background noise: ~0.001 bar RMS
   - Low-frequency drift and white noise

2. **Transient/Leak Phase (5-20s)**
   - Valve opens at t=5s
   - Active leak with acoustic signature
   - Valve closes at t=20s
   - Total leak duration: 15 seconds

3. **Recovery Phase (20-30s)**
   - Post-leak pressure recovery
   - Decay of acoustic signature
   - Return to baseline conditions

---

## Sensor Configuration

### Sensor Array Layout

| Sensor | Position (m) | Location Description |
|--------|-------------|---------------------|
| PG01 | 0.0 | Before Valve 1 (Reference) |
| PG02 | 2.5 | After Valve 1 |
| PG03 | 5.0 | Before Leak Point A |
| PG04 | 7.5 | After Leak Point A |
| PG05 | 10.0 | Mid-section |
| PG06 | 12.5 | Before Leak Point D |
| PG07 | 15.0 | After Leak Point D |
| PG08 | 17.5 | Before Valve 2 |
| PG09 | 20.0 | After Valve 2 |
| PG10 | 22.5 | Before Leak Point E |
| PG11 | 25.0 | After Leak Point E |
| PG12 | 27.5 | End section |
| PG13 | 30.0 | Near endpoint |
| PG14 | 32.5 | Endpoint |

### Leak Locations

- **Location A:** 6.0 m (between PG03 and PG04)
- **Location D:** 13.5 m (between PG06 and PG07)
- **Location E:** 24.0 m (between PG10 and PG11)

---

## Test Groups

### Group 01
- **Leak Location:** A (6.0 m)
- **Sampling Rate:** 17,060 Hz
- **Leak Intensity:** 1.0 (maximum)
- **Active Sensor Config:** A, B, C, D (all configurations)
- **Purpose:** High-resolution reference data with maximum leak intensity

### Group 02
- **Leak Location:** D (13.5 m)
- **Sampling Rate:** 10,000 Hz
- **Leak Intensity:** 0.8
- **Active Sensor Config:** A, B
- **Purpose:** Mid-pipeline leak with reduced sensor configuration

### Group 03
- **Leak Location:** E (24.0 m)
- **Sampling Rate:** 10,000 Hz
- **Leak Intensity:** 0.9
- **Active Sensor Config:** A, B, C, D
- **Purpose:** Downstream leak with full sensor array

### Group 04
- **Leak Location:** A (6.0 m)
- **Sampling Rate:** 10,000 Hz
- **Leak Intensity:** 0.85
- **Active Sensor Config:** A, B
- **Purpose:** Comparison with Group 01 at different sampling rate and intensity

---

## Acoustic Signature Characteristics

### Frequency Components

1. **Fundamental Frequency:** ~850 Hz
   - Represents turbulent jet frequency from leak
   - Primary acoustic signature

2. **Harmonic Series:** 2nd, 3rd, 5th, 7th, 11th harmonics
   - Decreasing amplitude with harmonic order
   - Characteristic of turbulent flow

3. **Broadband Turbulence:** 500-4900 Hz
   - Band-pass filtered white noise
   - Simulates turbulent jet noise

4. **Pipeline Reflections:**
   - Boundary reflections with ~50ms delay
   - Amplitude: ~15% of primary signal

### Spatial Characteristics

- **Propagation Delay:** Distance / Speed of Sound
- **Attenuation:** Exponential decay with distance
- **Distance-dependent amplitude:** A(d) = A₀ × exp(-α × d)
  - Where α ≈ 0.15 dB/m

---

## File Structure

```
acoustic_data/
├── acoustic_pressure_data.zip          # All CSV files (81.36 MB)
├── dataset_metadata.json               # Machine-readable metadata
├── README.md                           # This file
├── DATASET_DOCUMENTATION.md           # Detailed technical documentation
├── Group_01/                          # 30 CSV files (one per second)
│   ├── Group_01_second_00.csv
│   ├── Group_01_second_01.csv
│   └── ... (28 more files)
├── Group_02/                          # 30 CSV files
│   ├── Group_02_second_00.csv
│   └── ...
├── Group_03/                          # 30 CSV files
│   ├── Group_03_second_00.csv
│   └── ...
├── Group_04/                          # 30 CSV files
│   ├── Group_04_second_00.csv
│   └── ...
└── figures/                           # Visualization files
    ├── all_groups_comparison.png
    ├── Group_01/
    │   ├── Group_01_time_series.png
    │   ├── Group_01_spectrogram.png
    │   ├── Group_01_spatial_analysis.png
    │   └── Group_01_waterfall.png
    ├── Group_02/
    │   └── ... (4 figures)
    ├── Group_03/
    │   └── ... (4 figures)
    └── Group_04/
        └── ... (4 figures)
```

---

## CSV File Format

Each CSV file contains 1 second of data with the following columns:

- **Time_s:** Time in seconds (0.0 to 1.0 within each file)
- **PG01 to PG14:** Pressure measurements in bar (14 sensor columns)

**Format Specifications:**
- Delimiter: Comma (`,`)
- Decimal Precision: 8 digits
- Header Row: Yes
- Missing Values: None
- Encoding: UTF-8

**Example:**
```csv
Time_s,PG01,PG02,PG03,PG04,PG05,PG06,PG07,PG08,PG09,PG10,PG11,PG12,PG13,PG14
0.00000000,1.00012345,1.00023456,1.00034567,...
0.00005862,1.00013456,1.00024567,1.00035678,...
...
```

---

## Visualization Files

### Time Series Plots
- **Filename:** `{group}_time_series.png`
- **Content:** Full 30-second pressure traces for 4 selected sensors
- **Purpose:** Overview of leak event and temporal dynamics

### Spectrograms
- **Filename:** `{group}_spectrogram.png`
- **Content:** Frequency-time analysis of sensor nearest to leak
- **Frequency Range:** 0-5000 Hz
- **Purpose:** Identify acoustic frequency components and their evolution

### Spatial Analysis
- **Filename:** `{group}_spatial_analysis.png`
- **Content:** 
  - Top panel: RMS pressure vs. sensor position
  - Bottom panel: Attenuation vs. distance from leak
- **Purpose:** Analyze spatial pressure distribution and attenuation characteristics

### Waterfall Plots
- **Filename:** `{group}_waterfall.png`
- **Content:** Wave propagation visualization (4-second window around leak start)
- **Purpose:** Visualize acoustic wave arrival at different sensor positions

### Comparison Plot
- **Filename:** `all_groups_comparison.png`
- **Content:** Side-by-side comparison of leak signatures across all four groups
- **Purpose:** Compare different leak locations and intensities

---

## Data Usage Guidelines

### Loading Data in Python

```python
import pandas as pd
import numpy as np

# Load one second of data
df = pd.read_csv('Group_01/Group_01_second_05.csv')

# Access time and pressure data
time = df['Time_s'].values
pressure_pg03 = df['PG03'].values

# Load entire test (30 seconds)
frames = []
for second in range(30):
    df = pd.read_csv(f'Group_01/Group_01_second_{second:02d}.csv')
    df['Time_s'] = df['Time_s'] + second  # Adjust time to global scale
    frames.append(df)
full_data = pd.concat(frames, ignore_index=True)
```

### Loading Data in MATLAB

```matlab
% Load one second of data
data = readtable('Group_01/Group_01_second_05.csv');
time = data.Time_s;
pressure_PG03 = data.PG03;

% Load entire test
full_data = [];
for second = 0:29
    filename = sprintf('Group_01/Group_01_second_%02d.csv', second);
    temp = readtable(filename);
    temp.Time_s = temp.Time_s + second;
    full_data = [full_data; temp];
end
```

### Loading Data in R

```r
library(tidyverse)

# Load one second of data
df <- read_csv('Group_01/Group_01_second_05.csv')

# Load entire test
full_data <- map_df(0:29, function(second) {
  filename <- sprintf('Group_01/Group_01_second_%02d.csv', second)
  df <- read_csv(filename)
  df$Time_s <- df$Time_s + second
  return(df)
})
```

---

## Research Applications

This dataset is suitable for:

1. **Acoustic Wave Propagation Analysis**
   - Study wave speed and propagation delays
   - Analyze acoustic impedance effects
   - Investigate stratified flow effects

2. **Attenuation Mechanism Studies**
   - Quantify spatial attenuation rates
   - Compare attenuation across different leak locations
   - Model frequency-dependent attenuation

3. **Leak Detection Algorithm Development**
   - Train machine learning models
   - Test signal processing algorithms
   - Develop leak localization techniques

4. **Sensor Placement Optimization**
   - Analyze effect of sensor configuration
   - Optimize sensor density and spacing
   - Study detection sensitivity vs. distance

5. **Frequency Analysis**
   - Characterize leak acoustic signatures
   - Study harmonic content
   - Analyze broadband vs. tonal components

6. **Comparative Studies**
   - Compare different leak locations (A, D, E)
   - Analyze effect of leak intensity
   - Study sampling rate requirements

---

## Data Quality and Limitations

### Strengths
- High temporal resolution (10-17 kHz sampling)
- Multiple sensor configurations
- Controlled experimental conditions
- Repeatable and reproducible
- Well-documented parameters

### Limitations
- Synthetic data (not from physical experiments)
- Simplified physics model
- Single-phase flow assumption
- Idealized boundary conditions
- No environmental variations

### Validation Considerations
When using this data for algorithm development:
- Validate on real experimental data when available
- Consider additional noise sources in real systems
- Account for sensor calibration uncertainties
- Include environmental factors (temperature, pressure variations)

---

## Citation

If you use this dataset in your research, please cite:

```
Synthetic High-Frequency Acoustic Pressure Dataset for Leak Detection in Stratified Flows
Generated: February 2026
Version: 1.0
DOI: [To be assigned]
```

---

## Technical Support

For questions about this dataset:
- Review the `DATASET_DOCUMENTATION.md` file for technical details
- Check the `dataset_metadata.json` file for parameter specifications
- Examine the visualization files for data characteristics

---

## Version History

**Version 1.0 (February 2026)**
- Initial release
- 4 test groups (120 CSV files)
- 17 visualization figures
- Complete documentation

---

## License

This dataset is provided for research and educational purposes. Please provide appropriate attribution when using this data in publications or derivative works.

---

## Acknowledgments

This synthetic dataset was generated using physics-based models of acoustic wave propagation in fluid-filled pipelines, incorporating:
- Turbulent jet acoustic theory
- Wave propagation and attenuation models
- Pipeline reflection characteristics
- Realistic noise profiles

---

**End of README**
