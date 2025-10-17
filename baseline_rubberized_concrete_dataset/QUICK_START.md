# Quick Start Guide
## Baseline Rubberized Concrete Dataset

Get started with the dataset in 5 minutes!

---

## 📦 What's Included?

This dataset contains **complete baseline characterization** for rubberized concrete:
- **4 mix designs:** 0%, 5%, 10%, 15% rubber replacement
- **18 data files:** Mechanical, physical, thermal, microstructural properties
- **>2000 data points** from 100+ specimens tested
- **Complete documentation:** README, data dictionary, analysis scripts

---

## 🚀 Quick Start (3 Steps)

### Step 1: Understand the Dataset Structure

```
baseline_rubberized_concrete_dataset/
│
├── Data Files (CSV format):
│   ├── 1_mixture_proportions.csv          ← Mix designs
│   ├── 8_compressive_strength.csv         ← Strength data
│   ├── 15_property_correlations.csv       ← Quick summary
│   └── ... (15 more data files)
│
├── Documentation:
│   ├── README.md                           ← Start here!
│   ├── DATA_DICTIONARY.md                  ← Variable definitions
│   └── QUICK_START.md                      ← You are here
│
└── Analysis Tools:
    ├── analysis_script.py                  ← Python analysis
    └── requirements.txt                    ← Dependencies
```

### Step 2: Quick Data Preview

**Option A: Use spreadsheet software (Excel, LibreOffice)**
1. Open `15_property_correlations.csv` for overview
2. Browse other CSV files as needed

**Option B: Use Python**
```python
import pandas as pd

# Load summary data
correlations = pd.read_csv('15_property_correlations.csv')
print(correlations)

# Load detailed compressive strength data
strength = pd.read_csv('8_compressive_strength.csv')
strength_28d = strength[strength['Age_Days'] == 28]
print(strength_28d.groupby('Mix_ID')['Compressive_Strength_MPa'].mean())
```

**Option C: Use command line**
```bash
# View summary
column -t -s, 15_property_correlations.csv | less -S

# View compressive strength averages
awk -F',' '$6==28 {sum[$1]+=$9; count[$1]++} END {for (mix in sum) print mix, sum[mix]/count[mix]}' 8_compressive_strength.csv
```

### Step 3: Run Analysis Script (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python analysis_script.py
```

**Output:**
- `mechanical_properties.png` - Property trends vs. rubber content
- `stress_strain_curves.png` - Complete stress-strain behavior
- `rubber_tga.png` - Thermal decomposition analysis
- `pore_distribution.png` - Pore structure characterization
- `analysis_summary.txt` - Text summary report

---

## 📊 Key Findings at a Glance

### Mechanical Properties @ 28 Days

| Mix | Rubber (%) | f'c (MPa) | f_t (MPa) | E (GPa) | Ductility |
|-----|------------|-----------|-----------|---------|-----------|
| RC-00 | 0 | 52.3 | 4.18 | 32.2 | 1.00× |
| RC-05 | 5 | 46.8 | 3.75 | 28.2 | 1.25× |
| RC-10 | 10 | 39.8 | 3.08 | 22.8 | 1.70× |
| RC-15 | 15 | 32.5 | 2.48 | 17.2 | 2.40× |

**Key Insights:**
- ✅ Strength decreases systematically with rubber content
- ✅ Ductility increases dramatically (2.4× at 15% rubber)
- ✅ Density reduces by 8% at 15% rubber (lighter concrete)
- ✅ Thermal insulation improves (conductivity drops 38%)
- ⚠️ Permeability increases (trade-off for durability)

---

## 🎯 Common Use Cases

### 1. For Thermo-Mechanical Modeling
**Files you need:**
- `1_mixture_proportions.csv` - Input material compositions
- `10_modulus_of_elasticity.csv` - Elastic properties (+ Poisson's ratio)
- `11_density_porosity.csv` - Density values
- `16_thermal_properties_ambient.csv` - Baseline thermal properties
- `18_stress_strain_curves.csv` - Constitutive behavior

**Key parameters:**
```
RC-00: E=32.2 GPa, ν=0.19, ρ=2385 kg/m³, k=1.82 W/(m·K), c_p=880 J/(kg·K)
RC-15: E=17.2 GPa, ν=0.31, ρ=2245 kg/m³, k=1.12 W/(m·K), c_p=1015 J/(kg·K)
```

### 2. For Fire Resistance Prediction
**Files you need:**
- `5_rubber_thermal_analysis.csv` - Rubber decomposition (TGA)
- `11_density_porosity.csv` - Initial porosity
- `12_pore_size_distribution_MIP.csv` - Pore structure
- `17_permeability_durability.csv` - Permeability

**Critical finding:** Rubber decomposes at 300-400°C, creating porosity for vapor escape (potential spalling reduction).

### 3. For Mix Design Optimization
**Files you need:**
- `1_mixture_proportions.csv` - Mix designs
- `7_fresh_state_properties.csv` - Workability
- `15_property_correlations.csv` - Performance summary

**Design guide:**
- Need f'c > 40 MPa? → Use ≤5% rubber
- Need high ductility? → Use 10-15% rubber
- Need lightweight? → Use 15% rubber (8% density reduction)

### 4. For Microstructural Understanding
**Files you need:**
- `14_microstructural_analysis.csv` - SEM/XRD data
- `12_pore_size_distribution_MIP.csv` - Pore structure

**Key finding:** Weak rubber-cement interface (ITZ thickness 95-165 μm, porosity 35-58%) causes strength reduction.

---

## 📖 Data File Descriptions (Brief)

| File | What's Inside | When to Use |
|------|---------------|-------------|
| `1_mixture_proportions.csv` | Complete mix designs | Model input, mix design |
| `2_aggregate_grading.csv` | Particle size distributions | Quality control, modeling |
| `3_rubber_characterization.csv` | Rubber physical properties | Understanding rubber behavior |
| `4_rubber_chemical_composition.csv` | Rubber composition | Chemical modeling |
| `5_rubber_thermal_analysis.csv` | TGA/DSC (25-800°C) | Fire modeling, decomposition |
| `6_rubber_ftir_peaks.csv` | FTIR spectroscopy | Chemical identification |
| `7_fresh_state_properties.csv` | Workability, slump, air | Mix design, placement |
| `8_compressive_strength.csv` | f'c @ 7 & 28 days | Structural design |
| `9_tensile_splitting_strength.csv` | f_t @ 28 days | Cracking analysis |
| `10_modulus_of_elasticity.csv` | E, ν, ultimate strain | Structural modeling |
| `11_density_porosity.csv` | Density, porosity | Modeling input |
| `12_pore_size_distribution_MIP.csv` | Pore structure (MIP) | Durability, fire modeling |
| `13_ultrasonic_pulse_velocity.csv` | UPV, dynamic modulus | NDT, quality assessment |
| `14_microstructural_analysis.csv` | SEM, XRD | Understanding mechanisms |
| `15_property_correlations.csv` | **Summary table** | **Start here!** |
| `16_thermal_properties_ambient.csv` | k, c_p, α @ 23°C | Thermal modeling baseline |
| `17_permeability_durability.csv` | Permeability, durability | Service life prediction |
| `18_stress_strain_curves.csv` | Complete σ-ε curves | Constitutive modeling |

---

## 💡 Pro Tips

### Tip 1: Always Start with the Summary
Open `15_property_correlations.csv` first to get a high-level overview.

### Tip 2: Check Units in DATA_DICTIONARY.md
Units vary by property (MPa, GPa, kg/m³, etc.). Always verify in `DATA_DICTIONARY.md`.

### Tip 3: Use Mix_ID for Filtering
All files use consistent `Mix_ID`: `RC-00`, `RC-05`, `RC-10`, `RC-15`.

### Tip 4: Statistical Data is Included
Most files include individual specimen data + mean, std dev, COV.

### Tip 5: Normalize to Control Mix
Many comparisons are easier if you normalize properties relative to RC-00 (control mix).

---

## 🔍 Example Analysis (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load correlation data
data = pd.read_csv('15_property_correlations.csv')

# Plot strength vs. rubber content
plt.figure(figsize=(10, 6))
plt.plot(data['Rubber_Content_Percent'], 
         data['Compressive_Strength_28d_MPa'], 
         'o-', linewidth=2, markersize=10)
plt.xlabel('Rubber Content (%)', fontsize=12)
plt.ylabel('Compressive Strength (MPa)', fontsize=12)
plt.title('Effect of Rubber on Compressive Strength', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('strength_vs_rubber.png', dpi=300)
plt.show()

# Calculate strength reduction
control_strength = data[data['Rubber_Content_Percent']==0]['Compressive_Strength_28d_MPa'].values[0]
data['Strength_Reduction_%'] = (1 - data['Compressive_Strength_28d_MPa']/control_strength) * 100
print(data[['Mix_ID', 'Rubber_Content_Percent', 'Compressive_Strength_28d_MPa', 'Strength_Reduction_%']])
```

---

## ❓ Frequently Asked Questions

### Q1: What testing standards were used?
**A:** ASTM, BS EN, and ISO standards throughout. See `DATA_DICTIONARY.md` for specific standards per property.

### Q2: How many specimens were tested?
**A:** 5 specimens per test per mix (20 total) for mechanical properties. >100 specimens total.

### Q3: What curing method was used?
**A:** 28 days in lime-saturated water at 23°C, 100% RH for all specimens.

### Q4: Why these rubber contents (5%, 10%, 15%)?
**A:** These levels represent practical ranges balancing workability, strength, and ductility based on literature.

### Q5: Is high-temperature data included?
**A:** No, this is **Pillar 1** (baseline characterization @ 23°C). High-temperature testing is a future phase.

### Q6: Can I use this for commercial projects?
**A:** This is research data. Engineering judgment and additional testing required for commercial applications.

### Q7: How do I cite this dataset?
**A:** See `CITATION.cff` file or the citation section in `README.md`.

### Q8: The analysis script doesn't work!
**A:** Check that you've installed dependencies: `pip install -r requirements.txt`

---

## 📚 Next Steps

1. **Read the full README.md** for comprehensive documentation
2. **Explore DATA_DICTIONARY.md** for variable definitions
3. **Run analysis_script.py** to generate visualizations
4. **Review CHANGELOG.md** for version history

---

## 🆘 Getting Help

**Documentation:**
- `README.md` - Main documentation (comprehensive)
- `DATA_DICTIONARY.md` - All variable definitions
- `CHANGELOG.md` - Version history

**Analysis:**
- `analysis_script.py` - Example Python analysis
- Comments in script explain each function

**Issues:**
- Check documentation first
- Review data dictionary for units/definitions
- Verify you're using the correct file for your purpose

---

## ✅ Checklist for First-Time Users

- [ ] Read this QUICK_START.md (you're almost done!)
- [ ] Open `15_property_correlations.csv` to see summary
- [ ] Browse `README.md` sections relevant to your work
- [ ] Install Python dependencies (if using analysis script)
- [ ] Run `analysis_script.py` to generate plots
- [ ] Consult `DATA_DICTIONARY.md` when you encounter unknown variables
- [ ] Check `CHANGELOG.md` to confirm you have the latest version

---

**Ready to dive deeper? Open README.md for the complete documentation!**

**Questions?** Review the FAQ section above or consult the detailed documentation.

---

*Last Updated: 2025-10-17*  
*Dataset Version: 1.0.0*
