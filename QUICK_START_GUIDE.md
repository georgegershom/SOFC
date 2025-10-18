# Phase 3 Microstructural Analysis - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Check What You Have

You should see this directory structure:

```
phase3_microstructural_analysis/
├── README.md                    ← Full documentation (591 lines)
├── DATASET_SUMMARY.md           ← This summary
├── QUICK_START_GUIDE.md         ← You are here!
├── requirements.txt             ← Python packages needed
├── sem_data/
│   └── sem_itz_analysis.csv    ← 58 SEM observations
├── xrd_data/
│   └── xrd_phase_composition.csv  ← 60 XRD measurements
├── tga_dta_data/
│   ├── tga_mass_loss_analysis.csv  ← 60 TGA measurements
│   └── dta_thermal_events.csv      ← 130 thermal events
├── microct_data/
│   ├── microct_porosity_analysis.csv      ← 63 porosity scans
│   └── microct_crack_network_analysis.csv ← 63 crack analyses
├── scripts/
│   ├── analyze_microstructural_data.py    ← Main analysis script
│   ├── visualize_microstructural_data.py  ← Generate figures
│   ├── generate_synthetic_microct_images.py
│   └── run_complete_analysis.py           ← Master script
└── figures/                     ← Output folder (will be created)
```

### Step 2: Verify Data Files

Quick check:

```bash
# Count data rows
wc -l *_data/*.csv
# Should show: 58, 60, 60, 130, 63, 63 (plus header rows)

# Preview data
head -n 3 sem_data/sem_itz_analysis.csv
head -n 3 xrd_data/xrd_phase_composition.csv
```

### Step 3: Install Dependencies (Optional, for analysis)

If you want to run the Python analysis scripts:

```bash
# Option 1: Install minimal requirements
pip install pandas numpy matplotlib seaborn scipy

# Option 2: Install from requirements file
pip install -r requirements.txt

# Option 3: Skip if you just want the raw data
# (The CSV files are ready to use in Excel, R, MATLAB, etc.)
```

### Step 4: Explore the Data

#### Option A: Use Excel/Spreadsheet
1. Open any CSV file in Excel or Google Sheets
2. Each file has clear column headers
3. Filter by temperature, rubber content, etc.

#### Option B: Use Python (if installed)
```python
import pandas as pd

# Load SEM data
sem = pd.read_csv('sem_data/sem_itz_analysis.csv')
print(f"SEM data: {len(sem)} observations")
print(sem.columns.tolist())

# Filter for specific condition
rubber_10_400C = sem[
    (sem['rubber_content_pct'] == 10) & 
    (sem['temperature_C'] == 400)
]
print(f"\nFound {len(rubber_10_400C)} observations for 10% rubber at 400°C")
print(rubber_10_400C[['itz_thickness_um', 'microcrack_density_per_mm2']])
```

#### Option C: Use R
```r
# Load SEM data
sem <- read.csv('sem_data/sem_itz_analysis.csv')
print(paste("SEM data:", nrow(sem), "observations"))

# Filter
rubber_10_400C <- sem[
    sem$rubber_content_pct == 10 & 
    sem$temperature_C == 400, 
]
summary(rubber_10_400C$itz_thickness_um)
```

### Step 5: Run Analysis (Python only)

```bash
cd scripts/

# Run complete analysis pipeline
python run_complete_analysis.py

# Or run individual scripts:
python analyze_microstructural_data.py      # Statistical analysis
python visualize_microstructural_data.py    # Generate figures
```

This will:
- ✅ Check dependencies
- ✅ Load all datasets
- ✅ Perform statistical analysis
- ✅ Generate 6 publication-quality figures
- ✅ Save figures to `../figures/`

---

## 📊 What's In Each Dataset?

### SEM Data (Interfacial Transition Zone)
**File**: `sem_data/sem_itz_analysis.csv`

**Key Columns**:
- `itz_thickness_um`: Interface thickness (micrometers)
- `microcrack_density_per_mm2`: Cracks per square millimeter
- `crack_width_um`: Average crack width
- `rubber_degradation_score`: 0-10 scale (0=none, 10=complete)
- `paste_morphology_score`: 0-10 scale (0=destroyed, 10=perfect)

**What it tells you**: How the rubber-paste and aggregate-paste interfaces degrade with temperature

### XRD Data (Phase Composition)
**File**: `xrd_data/xrd_phase_composition.csv`

**Key Columns**:
- `portlandite_wt_pct`: Ca(OH)₂ content (weight %)
- `free_lime_CaO_wt_pct`: Free lime from CH decomposition
- `CSH_amorphous_wt_pct`: C-S-H gel content
- `crystallinity_index`: Overall crystallinity (0-1)

**What it tells you**: Which phases decompose at each temperature and why strength is lost

### TGA Data (Mass Loss)
**File**: `tga_dta_data/tga_mass_loss_analysis.csv`

**Key Columns**:
- `free_water_loss_50_150C_pct`: Free water (%)
- `bound_water_loss_150_400C_pct`: C-S-H bound water (%)
- `CH_dehydrox_loss_400_500C_pct`: Portlandite decomposition (%)
- `rubber_combustion_loss_300_500C_pct`: Rubber burnout (%)
- `total_mass_loss_pct`: Total mass loss (%)

**What it tells you**: Exactly how much mass is lost and from which mechanisms

### DTA Data (Thermal Events)
**File**: `tga_dta_data/dta_thermal_events.csv`

**Key Columns**:
- `event_type`: endothermic or exothermic
- `onset_temp_C`, `peak_temp_C`, `endset_temp_C`: Event temperatures
- `enthalpy_J_g`: Energy of transformation
- `phase_identification`: What phase is changing

**What it tells you**: Temperature ranges for each decomposition reaction

### Micro-CT Porosity Data (3D Pore Structure)
**File**: `microct_data/microct_porosity_analysis.csv`

**Key Columns**:
- `total_porosity_pct`: Total void volume (%)
- `macro_porosity_50_1000um_pct`: Large pores affecting strength (%)
- `pore_connectivity_index`: How connected pores are (0-1)
- `permeability_m2`: Gas permeability (m²)
- `tortuosity_factor`: Path complexity (1 = straight)

**What it tells you**: How the 3D pore network evolves and affects transport properties

### Micro-CT Crack Data (3D Crack Network)
**File**: `microct_data/microct_crack_network_analysis.csv`

**Key Columns**:
- `crack_density_mm_mm3`: Total crack length per volume
- `avg_crack_width_um`: Average crack opening (micrometers)
- `crack_network_connectivity`: How interconnected (0-1)
- `damage_parameter`: Overall damage level (0=none, 1=failed)

**What it tells you**: How crack networks develop and when material has failed

---

## 🎯 Common Analysis Tasks

### Task 1: Compare Control vs Rubberized at 400°C

```python
import pandas as pd
import matplotlib.pyplot as plt

sem = pd.read_csv('sem_data/sem_itz_analysis.csv')

# Filter data
data_400C = sem[sem['temperature_C'] == 400]

# Group by rubber content
grouped = data_400C.groupby('rubber_content_pct')['itz_thickness_um'].mean()

# Plot
plt.figure(figsize=(8, 5))
plt.bar(grouped.index, grouped.values)
plt.xlabel('Rubber Content (%)')
plt.ylabel('ITZ Thickness (μm)')
plt.title('ITZ Thickness at 400°C')
plt.grid(axis='y', alpha=0.3)
plt.savefig('itz_comparison_400C.png', dpi=300)
print("✓ Figure saved!")
```

### Task 2: Track Portlandite Consumption

```python
xrd = pd.read_csv('xrd_data/xrd_phase_composition.csv')

# For 0% rubber
control = xrd[xrd['rubber_content_pct'] == 0]
grouped = control.groupby('temperature_C')['portlandite_wt_pct'].mean()

print("Portlandite Content (0% rubber):")
for temp, ch_content in grouped.items():
    print(f"  {temp}°C: {ch_content:.2f} wt%")

# Calculate consumption
initial = grouped[20]
for temp in [200, 400, 600, 800]:
    current = grouped[temp]
    consumed = ((initial - current) / initial) * 100
    print(f"  At {temp}°C: {consumed:.1f}% consumed")
```

### Task 3: Correlate Porosity with Temperature

```python
microct = pd.read_csv('microct_data/microct_porosity_analysis.csv')

# For 10% rubber
rubber10 = microct[microct['rubber_content_pct'] == 10]

# Get porosity at each temperature
result = rubber10.groupby('temperature_C')['total_porosity_pct'].mean()

print("Porosity Evolution (10% rubber):")
for temp, porosity in result.items():
    print(f"  {temp}°C: {porosity:.1f}%")
```

### Task 4: Identify Critical Temperature

```python
# When does damage parameter exceed 0.5?
cracks = pd.read_csv('microct_data/microct_crack_network_analysis.csv')

# For each rubber content
for rubber in [0, 5, 10, 15, 20]:
    data = cracks[cracks['rubber_content_pct'] == rubber]
    
    # Find first temperature where damage > 0.5
    critical_temps = data[data['damage_parameter'] > 0.5]
    
    if len(critical_temps) > 0:
        critical_T = critical_temps['temperature_C'].min()
        print(f"{rubber}% rubber: Critical temperature = {critical_T}°C")
```

---

## 📈 Expected Results

### Temperature Effects (Control Concrete, 0% Rubber)

| Temperature | ITZ Thickness | Portlandite | Porosity | Damage |
|-------------|---------------|-------------|----------|--------|
| 20°C | 25 μm | 18 wt% | 8% | 0.0 |
| 200°C | 28 μm | 17 wt% | 11% | 0.08 |
| 400°C | 32 μm | 12 wt% | 15% | 0.23 |
| 600°C | 40 μm | 4 wt% | 24% | 0.59 |
| 800°C | 55 μm | 0 wt% | 37% | 0.78 |

### Rubber Content Effects (at 400°C)

| Rubber | ITZ Thickness | Porosity | Damage | Notes |
|--------|---------------|----------|--------|-------|
| 0% | 32 μm | 15% | 0.23 | Moderate damage |
| 5% | 55 μm | 30% | 0.68 | Severe damage |
| 10% | 67 μm | 40% | 0.87 | Near failure |
| 15% | 78 μm | 48% | 0.97 | Failed |
| 20% | 90 μm | 58% | 0.99 | Total failure |

---

## 🆘 Troubleshooting

### Problem: "No such file or directory"
**Solution**: Make sure you're in the `phase3_microstructural_analysis/` directory
```bash
cd phase3_microstructural_analysis/
ls  # Should show sem_data, xrd_data, etc.
```

### Problem: "Module not found: pandas"
**Solution**: Either install pandas or use the CSV files directly in Excel
```bash
pip install pandas numpy matplotlib
```

### Problem: Figures not generating
**Solution**: Check that matplotlib is installed and figures/ folder exists
```bash
pip install matplotlib seaborn
mkdir -p figures/
```

### Problem: Data looks wrong
**Solution**: Check you're reading the correct columns and filtering properly
```python
# Always check what you loaded
print(df.columns.tolist())
print(df.head())
print(df['temperature_C'].unique())  # See available temperatures
```

---

## ✅ Validation Checklist

Before using this data, verify:

- [ ] All 6 CSV files present and readable
- [ ] Data has correct number of rows (see file summary)
- [ ] Column headers match documentation
- [ ] Temperature values are: 20, 200, 400, 600, 800°C
- [ ] Rubber values are: 0, 5, 10, 15, 20%
- [ ] No obviously wrong values (negative percentages, etc.)
- [ ] Trends make physical sense (porosity increases with temperature)

---

## 📞 Need Help?

1. **Start with**: `README.md` (complete documentation)
2. **Check**: CSV column headers and units
3. **Review**: Example scripts in this guide
4. **Examine**: Python script comments for detailed explanations

---

## 🎓 Learn More

### Recommended Reading

1. **SEM of Concrete**: 
   - Scrivener et al., "Backscattered electron imaging of cementitious microstructures"
   
2. **XRD Quantification**: 
   - Rietveld method, TOPAS/HighScore software
   
3. **TGA Analysis**: 
   - Ramachandran & Beaudoin, "Handbook of Thermal Analysis of Construction Materials"
   
4. **Micro-CT**: 
   - Cnudde & Boone, "High-resolution X-ray computed tomography in geosciences"

### Online Resources

- ImageJ/Fiji: https://imagej.net/
- TOPAS Academic: http://www.topas-academic.net/
- Avizo: https://www.thermofisher.com/avizo
- Python pandas: https://pandas.pydata.org/

---

## 🏁 You're Ready!

You now have:
- ✅ Complete microstructural dataset
- ✅ Understanding of what each file contains
- ✅ Example code to get started
- ✅ Expected results to validate against

**Go analyze your data!** 🔬📊

---

*Quick Start Guide - Phase 3 Microstructural Analysis*  
*Last updated: 2025-10-18*
