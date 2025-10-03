# 🚀 Quick Start Guide - SOFC Experimental Data

## One-Command Setup

```bash
# Generate all experimental data
python3 generate_sofc_experimental_data.py

# Run example analysis
python3 example_analysis.py
```

That's it! You now have a complete SOFC experimental dataset.

---

## 📁 What Was Generated?

### Data Files (CSV)
- **DIC Data:** 6 CSV files with strain measurements
- **XRD Data:** 4 CSV files with stress and lattice measurements  
- **Post-Mortem:** 6 CSV files with crack, composition, and mechanical data
- **Total:** ~16 CSV files + metadata

### Visualizations (PNG)
- **DIC:** 13 strain map images
- **XRD:** 4 stress profile plots
- **Post-Mortem:** 6 analysis plots
- **Total:** ~23 high-resolution images

### Summary Documents
- `DATA_SUMMARY.txt` - Complete data inventory
- `ANALYSIS_SUMMARY_REPORT.txt` - Analysis results

---

## 📊 Key Files to Start With

### For Strain Analysis
```python
import pandas as pd

# Load thermal cycling strain data
df = pd.read_csv('sofc_experimental_data/dic_data/thermal_cycling_strain_data.csv')
print(df.head())
```

**Columns:** cycle, time_hours, temperature_C, region, strain_xx, strain_yy, strain_xy, von_mises_strain

### For Stress Analysis
```python
# Load residual stress profiles
df = pd.read_csv('sofc_experimental_data/xrd_data/residual_stress_profiles.csv')
print(df.head())
```

**Columns:** condition, depth_um, phase, sigma_11_MPa, sigma_22_MPa, sigma_33_MPa, von_mises_stress_MPa

### For Crack Analysis
```python
# Load microcrack threshold data
df = pd.read_csv('sofc_experimental_data/xrd_data/microcrack_threshold_data.csv')
print(df[df['cracked'] == True])
```

**Columns:** specimen_id, applied_strain, cracked, crack_density_per_mm2, crack_length_um

### For Composition Analysis
```python
# Load EDS line scan
df = pd.read_csv('sofc_experimental_data/postmortem_data/eds_analysis/eds_line_scan_cross_section.csv')
print(df[['distance_um', 'Ni_wt%', 'Zr_wt%', 'O_wt%']].head())
```

**Columns:** distance_um, Ni_wt%, Zr_wt%, Y_wt%, O_wt%, La_wt%, Sr_wt%, Mn_wt%

### For Mechanical Properties
```python
# Load nano-indentation map
df = pd.read_csv('sofc_experimental_data/postmortem_data/nanoindentation/nanoindentation_map.csv')
print(df[['depth_um', 'youngs_modulus_GPa', 'hardness_GPa']].head())
```

**Columns:** x_mm, y_mm, depth_um, youngs_modulus_GPa, hardness_GPa, creep_compliance

---

## 🎯 Common Analysis Tasks

### 1. Find Maximum Strain Location
```python
import pandas as pd

df = pd.read_csv('sofc_experimental_data/dic_data/sintering_strain_data.csv')
max_row = df.loc[df['von_mises_strain'].idxmax()]
print(f"Max strain: {max_row['von_mises_strain']*100:.2f}% at {max_row['region']}")
```

### 2. Plot Stress Profile
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sofc_experimental_data/xrd_data/residual_stress_profiles.csv')
condition = df[df['condition'] == 'as_sintered']

plt.figure(figsize=(10, 6))
plt.plot(condition['depth_um'], condition['von_mises_stress_MPa'], linewidth=2)
plt.xlabel('Depth (μm)')
plt.ylabel('Von Mises Stress (MPa)')
plt.title('Residual Stress Profile')
plt.grid(True)
plt.show()
```

### 3. Identify Cracked Specimens
```python
import pandas as pd

df = pd.read_csv('sofc_experimental_data/xrd_data/microcrack_threshold_data.csv')
cracked = df[df['cracked'] == True]
print(f"Cracked specimens: {len(cracked)}/{len(df)}")
print(f"Critical strain range: {cracked['applied_strain'].min()*100:.2f}% - {cracked['applied_strain'].max()*100:.2f}%")
```

### 4. Compare Mechanical Properties by Layer
```python
import pandas as pd

df = pd.read_csv('sofc_experimental_data/postmortem_data/nanoindentation/nanoindentation_map.csv')

# Define layers by depth
anode = df[df['depth_um'] < 300]
electrolyte = df[(df['depth_um'] >= 300) & (df['depth_um'] < 500)]
cathode = df[df['depth_um'] >= 500]

print(f"Anode E: {anode['youngs_modulus_GPa'].mean():.1f} GPa")
print(f"Electrolyte E: {electrolyte['youngs_modulus_GPa'].mean():.1f} GPa")
print(f"Cathode E: {cathode['youngs_modulus_GPa'].mean():.1f} GPa")
```

---

## 🔍 Data Structure Overview

```
sofc_experimental_data/
│
├── dic_data/                           # Digital Image Correlation
│   ├── sintering_strain_data.csv       # Strain during sintering (1200-1500°C)
│   ├── thermal_cycling_strain_data.csv # 10 thermal cycles (ΔT=400°C)
│   ├── startup_shutdown_cycles.csv     # 5 startup/shutdown cycles
│   ├── speckle_patterns/               # Speckle images + metadata
│   ├── lagrangian_tensors/             # Strain tensor fields
│   └── strain_hotspots/                # High strain locations (>1%)
│
├── xrd_data/                           # X-ray Diffraction
│   ├── residual_stress_profiles.csv    # Stress across cross-section
│   ├── lattice_strain_vs_temperature.csv # 25-1500°C lattice strain
│   ├── sin2psi_stress_analysis.csv     # Peak shift data
│   └── microcrack_threshold_data.csv   # Crack initiation (εcr>2%)
│
└── postmortem_data/                    # Post-Mortem Analysis
    ├── sem_analysis/
    │   └── crack_density_analysis.csv  # Crack measurements (cracks/mm²)
    ├── eds_analysis/
    │   ├── eds_line_scan_cross_section.csv # Elemental profiles
    │   └── eds_point_analysis.csv      # Point measurements
    └── nanoindentation/
        ├── nanoindentation_map.csv     # E, H, creep maps
        └── load_displacement_curves/   # Load-disp curves
```

---

## 📈 Key Findings Summary

| Parameter | Value | Location |
|-----------|-------|----------|
| Maximum Strain | ~2.2% | Anode-electrolyte interface |
| Maximum Stress | ~200 MPa | Interface regions (after cycling) |
| Critical Crack Strain | 2.0% | Threshold for microcrack initiation |
| YSZ Modulus | 184.7 GPa | Electrolyte layer |
| Ni-YSZ Modulus | 109.8 GPa | Anode layer |
| Interface Degradation | 15-20% | Property reduction at interfaces |

---

## 🛠️ Customization

### Change Temperature Range
Edit line ~68 in `generate_sofc_experimental_data.py`:
```python
temperatures = np.linspace(1200, 1600, 100)  # Extend to 1600°C
```

### Increase Thermal Cycles
Edit line ~103:
```python
n_cycles = 20  # Increase from 10 to 20 cycles
```

### Modify Material Properties
Edit lines ~520-525:
```python
E_YSZ = 184.7  # GPa
E_NiYSZ = 109.8  # GPa
E_LSM = 120.0  # GPa
```

### Add More Specimens
Edit line ~640:
```python
n_specimens = 50  # Increase from 30 to 50
```

---

## ✅ Verification Checklist

After generation, verify:
- [ ] `sofc_experimental_data/` directory exists
- [ ] 16+ CSV files created
- [ ] 20+ PNG visualization files
- [ ] `DATA_SUMMARY.txt` present
- [ ] No error messages during generation
- [ ] CSV files open correctly in Excel/Pandas
- [ ] PNG images display properly

---

## 💡 Tips

1. **Large Files:** All CSV files are reasonably sized (<5 MB each)
2. **Memory:** Generator uses ~500 MB RAM during execution
3. **Time:** Full generation takes ~30-60 seconds
4. **Reproducibility:** Fixed random seed (42) for consistent results
5. **Modification:** Change random seed for different data realizations

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install numpy pandas matplotlib scipy
```

### Error: "FileNotFoundError"
```bash
# Make sure you're in the correct directory
cd /workspace
python3 generate_sofc_experimental_data.py
```

### Empty or Missing Files
```bash
# Re-run the generator
rm -rf sofc_experimental_data/
python3 generate_sofc_experimental_data.py
```

### Analysis Script Errors
```bash
# Verify data was generated first
ls sofc_experimental_data/
python3 example_analysis.py
```

---

## 📚 Next Steps

1. **Explore the data:**
   - Open CSV files in Excel or Pandas
   - View PNG visualizations
   - Read `DATA_SUMMARY.txt`

2. **Run analysis:**
   - Execute `example_analysis.py`
   - View generated analysis plots
   - Read `ANALYSIS_SUMMARY_REPORT.txt`

3. **Integrate with your research:**
   - Train machine learning models
   - Validate computational simulations
   - Develop analysis algorithms
   - Create custom visualizations

4. **Customize:**
   - Modify data generation parameters
   - Add new measurement types
   - Extend analysis scripts

---

## 📞 Support

For questions:
1. Check `README.md` for detailed documentation
2. Review `DATA_SUMMARY.txt` for data structure
3. Examine the generator script for implementation details
4. Review example analysis for usage patterns

---

**Ready to use!** Your SOFC experimental dataset is complete. 🎉

**Dataset Size:** ~3-5 MB total  
**Files:** 40+ data and visualization files  
**Format:** CSV (data), PNG (images), JSON (metadata)  
**Applications:** ML training, simulation validation, method development
