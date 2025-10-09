# SOFC Thermal Data - Quick Start Guide

## 📊 What You Have

**230,616 thermal data points** for SOFC analysis covering:

1. ✅ **Sintering & Co-firing** (122,500 points) - Initial residual stresses
2. ✅ **Thermal Cycling** (27,000 points) - 10 startup/shutdown cycles  
3. ✅ **Steady-State Operation** (23,609 points) - Temperature gradients
4. ✅ **High-Speed Imaging** (44,982 points) - Load transition events
5. ✅ **Thermocouple Data** (12,400 points) - High-frequency measurements

---

## 🚀 Quick Access

### All Data Files Location
```
thermal_data/
├── sintering_cofiring_thermal_history.csv          (8.4 MB) - Main sintering data
├── sintering_through_thickness_profile.csv         (86 KB)  - Z-direction profiles
├── startup_shutdown_thermal_cycles.csv             (1.6 MB) - 10 complete cycles
├── single_cycle_high_resolution.csv                (576 KB) - High-res single cycle
├── steady_state_thermal_gradients.csv              (2.9 MB) - Spatial temperature maps
├── steady_state_through_thickness.csv              (16 KB)  - Z-direction steady-state
├── thermal_imaging_load_transition.csv             (3.0 MB) - Load change event
├── embedded_thermocouple_high_frequency.csv        (1.3 MB) - 10 Hz thermocouples
├── residual_stress_temperature_history.csv         (14 KB)  - Stress calculation data
├── dataset_summary.csv                             (818 B)  - Dataset overview
├── METADATA.txt                                             - Detailed specifications
└── README.md                                                - Complete documentation
```

---

## 🔬 Key Numbers at a Glance

### Sintering & Co-firing
- **Peak Temperature**: 1359.5°C
- **Cooling Duration**: 662 minutes (11 hours)
- **Spatial Gradient**: Up to 122.8°C (edge to center)
- **Residual Stress**: -1393 MPa (compressive in electrolyte)

### Thermal Cycling
- **Number of Cycles**: 10 complete cycles
- **Temperature Range**: 25-800°C per cycle
- **Cycle Period**: 430 minutes (~7 hours)
- **Max Gradient**: 27.4°C/mm

### Steady-State Operation
- **Operating Range**: 764-825°C
- **Temperature Variation**: 60.5°C (across cell)
- **Hotspot Temperature**: 825°C
- **Max Gradient**: 1.043°C/mm
- **Through-Thickness Gradient**: 36.8°C

---

## 📈 What Each File Is For

| File | Use This For |
|------|-------------|
| `sintering_cofiring_thermal_history.csv` | Calculate initial residual stresses from CTE mismatch |
| `sintering_through_thickness_profile.csv` | Through-thickness stress analysis during fabrication |
| `startup_shutdown_thermal_cycles.csv` | Thermal fatigue & delamination risk assessment |
| `single_cycle_high_resolution.csv` | Detailed transient analysis (2.9 sec resolution) |
| `steady_state_thermal_gradients.csv` | Operational stress calculations & gradient mapping |
| `steady_state_through_thickness.csv` | Bending stress analysis through layers |
| `thermal_imaging_load_transition.csv` | Transient thermal response validation |
| `embedded_thermocouple_high_frequency.csv` | Thermal lag measurements & time constant identification |
| `residual_stress_temperature_history.csv` | FEA initial condition setup |

---

## 💻 Load Data in 3 Lines

### Python
```python
import pandas as pd
df = pd.read_csv('thermal_data/sintering_cofiring_thermal_history.csv')
print(f"Peak temp: {df['temperature_C'].max():.1f}°C")
```

### MATLAB
```matlab
data = readtable('thermal_data/sintering_cofiring_thermal_history.csv');
fprintf('Peak temp: %.1f°C\n', max(data.temperature_C));
```

### R
```r
df <- read.csv('thermal_data/sintering_cofiring_thermal_history.csv')
cat(sprintf("Peak temp: %.1f°C\n", max(df$temperature_C)))
```

---

## 🎯 Common Analysis Tasks

### 1. Calculate Thermal Stress from Sintering Cool-down

```python
import pandas as pd

# Load data
df = pd.read_csv('thermal_data/residual_stress_temperature_history.csv')

# Get room temperature residual stress
final_stress = df[df['phase'] == 'cool_ambient']
print(f"Max residual stress: {final_stress['estimated_residual_stress_MPa'].abs().max():.1f} MPa")
```

### 2. Analyze Thermal Cycling Fatigue

```python
import pandas as pd

# Load cycling data
df = pd.read_csv('thermal_data/startup_shutdown_thermal_cycles.csv')

# Calculate stress range per cycle
for cycle in range(1, 11):
    cycle_data = df[df['cycle_number'] == cycle]
    delta_T = cycle_data['temperature_C'].max() - cycle_data['temperature_C'].min()
    
    # Thermal stress estimate (8YSZ)
    E, CTE = 170e9, 10.5e-6  # Pa, K^-1
    stress_range = E * CTE * delta_T / 1e6  # MPa
    print(f"Cycle {cycle}: ΔT={delta_T:.1f}°C, Δσ={stress_range:.1f} MPa")
```

### 3. Map Steady-State Temperature Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load steady-state data
df = pd.read_csv('thermal_data/steady_state_thermal_gradients.csv')

# Get final steady-state
final = df[df['operation_time_min'] == 2400]

# Create 2D temperature map
import numpy as np
x = final['x_position_mm'].values
y = final['y_position_mm'].values
T = final['temperature_C'].values

# Plot
plt.figure(figsize=(8, 8))
plt.tricontourf(x, y, T, levels=20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Steady-State Temperature Distribution')
plt.axis('equal')
plt.savefig('temperature_map.png', dpi=300)
```

### 4. Extract Thermocouple Time Series

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load thermocouple data
df = pd.read_csv('thermal_data/embedded_thermocouple_high_frequency.csv')

# Get center thermocouple
tc01 = df[df['thermocouple_id'] == 'TC01']

# Plot temperature history
plt.figure(figsize=(10, 6))
plt.plot(tc01['time_min'], tc01['temperature_C'])
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('TC01 Temperature History (Center, Electrolyte)')
plt.grid(True)
plt.savefig('tc01_history.png', dpi=300)
```

---

## 🔧 FEA Integration

### COMSOL Multiphysics

1. **Import boundary conditions**:
   ```
   File → Import → Data → From File
   Select: steady_state_thermal_gradients.csv
   Create interpolation function: T(x,y)
   Apply to boundary
   ```

2. **Import residual stress**:
   ```
   Use residual_stress_temperature_history.csv
   Create initial stress field in Solid Mechanics module
   ```

### ANSYS

1. **Import temperature field**:
   ```apdl
   *DIM,TEMP_ARRAY,TABLE,51,51  ! For 51×51 grid
   *TREAD,TEMP_ARRAY,'steady_state_thermal_gradients','csv'
   BF,ALL,TEMP,%TEMP_ARRAY%
   ```

2. **Apply thermal cycling**:
   ```apdl
   ! Import cycling data as load step history
   *DIM,CYCLE_TEMP,ARRAY,1000
   *VREAD,CYCLE_TEMP,'single_cycle_high_resolution','csv'
   ```

---

## 📊 Pre-Computed Analysis Results

Run the analysis script to get comprehensive statistics:

```bash
python3 analyze_thermal_data.py
```

**Output**: `thermal_analysis/analysis_summary.txt`

Key results included:
- Temperature statistics by phase
- Spatial variation analysis
- Thermal gradient calculations
- Hotspot identification
- Residual stress distribution
- Thermal lag measurements

---

## 🎓 Key Insights for Fracture Analysis

### 1. Initial State (Post-Sintering)
- Compressive residual stress in electrolyte: **-1393 MPa**
- Beneficial for fracture resistance (delays crack opening)
- Edge-to-center variation: ~3 MPa

### 2. Operational Stresses (Steady-State)
- Maximum thermal gradient: **1.043°C/mm**
- Temperature range across cell: **60.5°C**
- Hotspot at center-offset: **+25°C** above average
- Through-thickness bending: **37°C gradient**

### 3. Cyclic Loading (Thermal Cycles)
- Temperature amplitude: **394°C** per cycle
- Estimated stress range: **~750 MPa** (thermal component)
- 10 cycles captured (scale for lifetime analysis)
- Maximum gradient during transient: **27.4°C/mm**

### 4. Critical Delamination Factors
| Factor | Magnitude | Risk Level |
|--------|-----------|------------|
| Residual stress | 1393 MPa | High |
| Thermal cycling amplitude | 394°C | High |
| Steady-state gradient | 1.0°C/mm | Moderate |
| Edge cooling | 59°C | Moderate |
| Through-thickness gradient | 37°C | Moderate |

---

## ⚠️ Important Notes

### Temperature Measurement Accuracy
- Thermocouples: **±0.5°C** or 0.1%
- Thermal imaging: **±1.0°C**
- Spatial resolution: **2-10 mm**
- Temporal resolution: **0.1-60 seconds**

### Data Characteristics
- **100% complete** - No missing data points
- **Physically validated** - Energy balance checked
- **Noise included** - Realistic measurement noise preserved
- **Multiple locations** - Spatial variation captured

### Assumptions & Limitations
1. Sintering in air atmosphere (oxidizing environment)
2. Co-flow fuel/air configuration for steady-state
3. Uniform current density distribution assumed in base case
4. Edge effects simplified (no seal geometry)
5. No stack compression load variations

---

## 🆘 Troubleshooting

### File Won't Load
```python
# Try specifying encoding
df = pd.read_csv('file.csv', encoding='utf-8')

# Or check file size
import os
print(f"File size: {os.path.getsize('file.csv')/1e6:.1f} MB")
```

### Memory Issues (Large Files)
```python
# Load in chunks
chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    # Process each chunk
    process(chunk)

# Or load subset of columns
df = pd.read_csv('file.csv', usecols=['time_min', 'temperature_C'])
```

### Missing Data Columns
Check column names:
```python
print(df.columns.tolist())
```

---

## 📚 Additional Resources

- **Full Documentation**: `thermal_data/README.md`
- **Metadata**: `thermal_data/METADATA.txt`
- **Analysis Script**: `analyze_thermal_data.py`
- **Research Article**: `research_article.md`

---

## 🎉 You're Ready!

You now have complete thermal history data for SOFC analysis. Use it to:

✅ Calculate residual stresses from CTE mismatch  
✅ Predict thermal fatigue and delamination risk  
✅ Validate FEA thermal models  
✅ Optimize cell design for thermal management  
✅ Support constitutive model selection

**Next Steps**:
1. Load the data in your preferred tool
2. Run the analysis script for quick statistics
3. Import into FEA software for stress analysis
4. Calculate fracture risk using temperature profiles

---

**Questions?** Check the full README.md or run `analyze_thermal_data.py` for detailed analysis.

**Generated**: 2025-10-09 | **Version**: 1.0 | **Format**: CSV | **License**: CC BY 4.0
