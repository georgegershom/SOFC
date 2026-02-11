# Dataset Generation Summary

## ✅ Complete: Calibrated Parameters Dataset for Phase-Field Fracture Modeling

**Generated on**: February 11, 2026  
**Topic**: Phase-Field Fracture Modeling of Delamination in Electrolyte-Electrode Interfaces: The Role of Nanoscale Mixed Ionic-Electronic Conducting (MIEC) Interlayers

---

## 📦 Deliverables

### 1. CSV Data Files (8 files)

All CSV files are located in `csv_files/` directory:

✓ **01_main_calibrated_parameters.csv** (2.0 KB)
  - 15 core parameters with bounds and rationale
  - BK exponent, phase-field lengths, penalty parameters, etc.

✓ **02_interface_fracture_properties.csv** (1.3 KB)
  - 11 interface conditions (YSZ/GDC and GDC/LSCF)
  - Fracture energies, critical strengths, characteristic lengths

✓ **03_LSCF_nonstoichiometry_data.csv** (2.4 KB)
  - 35 data points across temperature-pO₂ space
  - Non-stoichiometry (Δδ) and anisotropic chemical strains

✓ **04_GDC_chemical_expansion_22delta_4T.csv** (2.2 KB)
  - 22 δ levels × 4 temperatures = 88 calibration points
  - Chemical expansion data for UMAT implementation

✓ **05_verification_QA_parameters.csv** (2.3 KB)
  - 20 verification and quality assurance parameters
  - Mesh convergence, tolerance, energy balance criteria

✓ **06_material_properties.csv** (2.0 KB)
  - 24 material properties for LSCF, YSZ, GDC
  - Mechanical, thermal, electrical, and microstructural data

✓ **07_operating_conditions.csv** (1.1 KB)
  - 12 operating scenarios
  - Load profiles, thermal/redox cycling, degradation timeline

✓ **08_cohesive_zone_model_parameters.csv** (1.8 KB)
  - 19 cohesive zone parameter sets
  - Mode-I, Mode-II, mixed-mode for different conditions

**Total CSV Data**: ~15 KB (text format, easily readable)

---

### 2. Visualization Figures (8 figures)

All figures located in `figures/` directory at 300 DPI, high-quality PNG:

✓ **Figure_01_Main_Parameters.png** (396 KB)
  - Bulk fracture energies, phase-field parameters, adhesion range

✓ **Figure_02_Interface_Properties.png** (538 KB)
  - YSZ/GDC and GDC/LSCF interface behavior and degradation

✓ **Figure_03_LSCF_Nonstoichiometry.png** (814 KB)
  - Temperature-pO₂-Δδ relationships and anisotropic strains

✓ **Figure_04_GDC_Chemical_Expansion.png** (822 KB)
  - 22δ × 4T dataset visualization and defect chemistry

✓ **Figure_05_Cohesive_Zone_Model.png** (576 KB)
  - Traction-separation laws and mixed-mode criteria

✓ **Figure_06_Verification_QA.png** (593 KB)
  - Mesh convergence, degradation function, tolerance selection

✓ **Figure_07_Material_Properties.png** (346 KB)
  - Elastic moduli, expansion coefficients, conductivities

✓ **Figure_08_Operating_Conditions.png** (559 KB)
  - Polarization curves, cycling profiles, pO₂ gradients

**Total Figures Size**: ~4.6 MB (publication-ready quality)

---

### 3. Documentation

✓ **README.md** (14 KB)
  - Comprehensive documentation with:
    - Dataset structure and file descriptions
    - Parameter summaries and key values
    - Implementation guidelines (UMAT/UEL code snippets)
    - Usage examples (Python, MATLAB, Fortran)
    - Data sources and calibration methods
    - Assumptions and limitations
    - File manifest

✓ **DATASET_SUMMARY.md** (This file)
  - Quick overview of deliverables
  - Statistics and key features

---

### 4. Automation Script

✓ **generate_figures.py** (29 KB)
  - Fully automated Python script to regenerate all 8 figures
  - Requires: pandas, numpy, matplotlib, seaborn
  - Can be run with: `python3 generate_figures.py`
  - Publication-quality plots with 300 DPI resolution

---

### 5. Compressed Archive

✓ **calibrated_parameters_dataset.zip** (7.2 KB)
  - Contains all 8 CSV files
  - Easy download and distribution
  - ~54-77% compression ratio

---

## 📊 Dataset Statistics

### Data Coverage

| Category | Count | Details |
|----------|-------|---------|
| **Main Parameters** | 15 | Core phase-field model parameters |
| **Interface Conditions** | 11 | YSZ/GDC and GDC/LSCF variations |
| **LSCF Data Points** | 35 | Temperature-pO₂ combinations |
| **GDC Calibration Points** | 88 | 22 δ levels × 4 temperatures |
| **Material Properties** | 24 | Across 3 materials (LSCF, YSZ, GDC) |
| **Operating Scenarios** | 12 | Load, thermal, redox conditions |
| **Cohesive Zone Sets** | 19 | Mode-I, II, mixed for different states |
| **QA Parameters** | 20 | Verification and convergence criteria |

**Total Unique Data Entries**: ~224 calibrated values

### Parameter Ranges

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| Fracture Energy (Gc) | 0.2 | 10.0 | J/m² |
| Interface Strength (σmax) | 80 | 280 | MPa |
| Phase-field Length (l₀) | 5 | 20 | nm |
| Non-stoichiometry (LSCF) | 0.009 | 0.047 | Δδ |
| Non-stoichiometry (GDC) | 0.0 | 0.0178 | δ |
| Temperature | 600 | 900 | °C |
| pO₂ | 10⁻²⁰ | 1.0 | atm |
| Current Density | 0.0 | 1.0 | A/cm² |

---

## 🎯 Key Features

### 1. **Multi-scale Coverage**
   - Nanoscale: Phase-field lengths (5-20 nm)
   - Microscale: Crack regularization (0.5 µm)
   - Interface: Cohesive zone parameters

### 2. **Multi-physics Integration**
   - Mechanical: Fracture energies, elastic properties
   - Chemical: Non-stoichiometry, expansion coefficients
   - Thermal: Temperature-dependent behavior (600-900°C)
   - Electrochemical: pO₂ gradients, current density effects

### 3. **Degradation Mechanisms**
   - Sr-segregation at GDC/LSCF (time-dependent: 0-1000h)
   - Interdiffusion at YSZ/GDC (enhancement effect)
   - SrZrO₃ formation (interface weakening)

### 4. **Implementation Ready**
   - Direct ABAQUS UMAT/UEL integration
   - Fortran code snippets in README
   - Python/MATLAB loading examples
   - Interpolation guidance for lookup tables

### 5. **Quality Assurance**
   - Mesh objectivity criteria
   - Convergence tolerance guidelines
   - Energy balance checks
   - Validation parameters

---

## 📁 Directory Structure

```
calibrated_parameters_dataset/
│
├── README.md                              (Main documentation)
├── DATASET_SUMMARY.md                     (This file - quick overview)
├── generate_figures.py                    (Python visualization script)
├── calibrated_parameters_dataset.zip      (Compressed CSV archive)
│
├── csv_files/                             (All data files)
│   ├── 01_main_calibrated_parameters.csv
│   ├── 02_interface_fracture_properties.csv
│   ├── 03_LSCF_nonstoichiometry_data.csv
│   ├── 04_GDC_chemical_expansion_22delta_4T.csv
│   ├── 05_verification_QA_parameters.csv
│   ├── 06_material_properties.csv
│   ├── 07_operating_conditions.csv
│   └── 08_cohesive_zone_model_parameters.csv
│
└── figures/                               (Publication-quality plots)
    ├── Figure_01_Main_Parameters.png
    ├── Figure_02_Interface_Properties.png
    ├── Figure_03_LSCF_Nonstoichiometry.png
    ├── Figure_04_GDC_Chemical_Expansion.png
    ├── Figure_05_Cohesive_Zone_Model.png
    ├── Figure_06_Verification_QA.png
    ├── Figure_07_Material_Properties.png
    └── Figure_08_Operating_Conditions.png
```

---

## 🚀 Quick Start

### Download CSV Data
```bash
# All CSV files are available in csv_files/ directory
# Or use the compressed archive:
unzip calibrated_parameters_dataset.zip
```

### Load Data (Python Example)
```python
import pandas as pd

# Load any CSV file
df = pd.read_csv('csv_files/01_main_calibrated_parameters.csv')
print(df.head())
```

### Regenerate Figures
```bash
# Install dependencies (if needed)
pip install pandas numpy matplotlib seaborn

# Generate all figures
python3 generate_figures.py
```

### View Figures
All figures are pre-generated in `figures/` directory as high-resolution PNG files.

---

## ✨ Highlights

### Most Comprehensive Dataset
- **88 calibration points** for GDC chemical expansion
- **35 data points** for LSCF non-stoichiometry
- **11 interface conditions** covering fresh to degraded states

### Publication-Ready Visualizations
- **8 multi-panel figures** with 4 subplots each
- **300 DPI resolution** for journal submission
- **Consistent styling** with scientific colormaps

### Direct Implementation Support
- **Fortran code snippets** for ABAQUS UMAT/UEL
- **Interpolation routines** for lookup tables
- **Verification workflow** included

### Time-Dependent Degradation
- **0 to 1000h operation** data for GDC/LSCF interface
- **Sr-segregation effects** quantified
- **Interdiffusion enhancement** at YSZ/GDC

---

## 📖 References to Key Data

### For Baseline Simulations:
- Main parameters: `01_main_calibrated_parameters.csv`
- YSZ/GDC interface: `02_interface_fracture_properties.csv` (Row 1: Baseline)
- GDC/LSCF interface: `02_interface_fracture_properties.csv` (Row 3: Baseline)

### For Degradation Studies:
- Time evolution: `02_interface_fracture_properties.csv` (Rows 8-11)
- Operating conditions: `07_operating_conditions.csv`

### For Chemo-Mechanical Coupling:
- LSCF: `03_LSCF_nonstoichiometry_data.csv` (35 points)
- GDC: `04_GDC_chemical_expansion_22delta_4T.csv` (88 points)

### For Verification:
- All QA parameters: `05_verification_QA_parameters.csv`

---

## 💾 File Sizes Summary

| Category | Number of Files | Total Size |
|----------|----------------|------------|
| CSV Files | 8 | ~15 KB |
| Figures (PNG) | 8 | ~4.6 MB |
| Documentation | 2 | ~20 KB |
| Scripts | 1 | ~29 KB |
| Archive | 1 | ~7.2 KB |
| **TOTAL** | **20 files** | **~4.7 MB** |

---

## ✅ Checklist

- [x] 8 comprehensive CSV data files
- [x] 8 publication-quality figures (300 DPI)
- [x] Complete README with implementation guide
- [x] Python visualization script
- [x] Compressed archive for easy download
- [x] Dataset summary (this document)
- [x] Code examples (Python, MATLAB, Fortran)
- [x] Parameter ranges and validation criteria
- [x] Multi-scale, multi-physics coverage
- [x] Time-dependent degradation data

---

**Dataset Status**: ✅ **COMPLETE AND READY FOR USE**

**Recommended Citation Format**:
```
Phase-Field Fracture Modeling Dataset for SOFC Interfaces (2026)
Calibrated Parameters for Electrolyte-Electrode Delamination with GDC Interlayers
```

---

**END OF SUMMARY**
