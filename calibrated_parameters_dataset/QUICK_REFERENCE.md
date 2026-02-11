# Quick Reference Guide

## 🎯 Where to Find What You Need

### Need Parameter Values?

| Your Goal | Go To File | Key Columns |
|-----------|-----------|-------------|
| Phase-field model setup | `01_main_calibrated_parameters.csv` | `Calibrated_Value`, `Lower_Bound`, `Upper_Bound` |
| Interface fracture energy | `02_interface_fracture_properties.csv` | `Gc_int`, `Sigma_max`, `Characteristic_Length` |
| LSCF chemical strain | `03_LSCF_nonstoichiometry_data.csv` | `Delta_delta`, `Chemical_Strain_xx/yy/zz` |
| GDC expansion lookup | `04_GDC_chemical_expansion_22delta_4T.csv` | `delta`, `T_600C/700C/800C/900C` |
| Mesh & convergence | `05_verification_QA_parameters.csv` | `Recommended_Value`, `Lower_Limit`, `Upper_Limit` |
| Material properties | `06_material_properties.csv` | `Value`, `Unit`, `Notes` |
| Operating conditions | `07_operating_conditions.csv` | `Temperature_C`, `pO2_cathode_atm`, `Current_Density_A_cm2` |
| Cohesive zone parameters | `08_cohesive_zone_model_parameters.csv` | `Gc`, `T_max`, `delta_0`, `delta_f` |

---

## 🔍 Common Use Cases

### Case 1: Setting up a baseline simulation (fresh cell, 800°C)

**Parameters needed:**
1. Phase-field length: `l₀ = 10 nm` → File `01`, row for `Phase-field Length Lower/Upper`
2. BK exponent: `η = 2.1` → File `01`, row 1
3. YSZ/GDC interface: `Gc = 2.15 J/m²` → File `02`, row 1 (Baseline)
4. GDC/LSCF interface: `Gc = 1.0 J/m²` → File `02`, row 3 (Baseline)

### Case 2: Modeling degradation after 500h operation

**Parameters needed:**
1. GDC/LSCF after 500h: `Gc = 0.6 J/m²` → File `02`, row 10
2. Reduced strength: `σmax = 135 MPa` → File `02`, row 10
3. Operating condition: File `07`, row 3 (Load 0.50 A/cm²)

### Case 3: Including chemical expansion for LSCF at 800°C, pO₂ = 0.21 atm

**Parameters needed:**
1. Look up in File `03`: Filter `Temperature_C = 800` AND `pO2_atm = 0.21`
2. Find: `Δδ = 0.028`, `Chemical_Strain_zz = 0.004046`
3. Anisotropy: `β₃₃/β₁₁ = 1.7` → File `01`, row 2

### Case 4: Setting up cohesive elements for YSZ/GDC interface

**Parameters needed:**
1. Cohesive zone parameters → File `08`, filter `Interface = YSZ/GDC` AND `Mode = Mode-I`
2. For baseline:
   - `Gc = 2.15 J/m²`
   - `T_max = 222.5 MPa`
   - `delta_0 = 0.019 µm` (onset)
   - `delta_f = 0.097 µm` (failure)
   - `K_penalty = 1000 GPa/m`

---

## 📊 Which Figure Shows What?

| Figure | Content | Use For |
|--------|---------|---------|
| **Figure 1** | Main parameters overview | Understanding parameter ranges |
| **Figure 2** | Interface properties | Selecting Gc for different conditions |
| **Figure 3** | LSCF non-stoichiometry | Interpolating Δδ from T and pO₂ |
| **Figure 4** | GDC expansion dataset | Visualizing 22δ×4T lookup table |
| **Figure 5** | Cohesive zone models | Choosing traction-separation law |
| **Figure 6** | Verification criteria | Setting up mesh and convergence |
| **Figure 7** | Material properties | Comparing elastic/thermal properties |
| **Figure 8** | Operating conditions | Understanding electrochemical environment |

---

## 🚀 Implementation Cheat Sheet

### For ABAQUS UMAT

```fortran
! 1. Read main parameters from File 01
parameter (eta_BK = 2.1)
parameter (l0_phasefield = 10.0E-9)  ! meters
parameter (k_residual = 1.0E-6)

! 2. Load LSCF expansion data from File 03
! Create lookup table for Chemical_Strain vs Delta_delta

! 3. Chemical strain calculation
epsilon_chem(1:2) = beta_11 * delta_current  ! In-plane
epsilon_chem(3) = beta_33 * delta_current     ! Out-of-plane
! where beta_33/beta_11 = 1.7 from File 01
```

### For ABAQUS UEL (Cohesive Interface)

```fortran
! 1. Read cohesive parameters from File 08
! For YSZ/GDC baseline, Mode-I:
G_c = 2.15      ! J/m²
T_max = 222.5   ! MPa
delta_0 = 0.019 ! µm
delta_f = 0.097 ! µm

! 2. Bilinear traction-separation law
if (delta_n < delta_0) then
    T_n = K_penalty * delta_n  ! K_penalty = 1000 GPa/m
else if (delta_n < delta_f) then
    T_n = T_max * (delta_f - delta_n) / (delta_f - delta_0)
else
    T_n = 0.0  ! Fully separated
end if
```

### For Python Analysis

```python
import pandas as pd

# Load interface properties
df_interface = pd.read_csv('csv_files/02_interface_fracture_properties.csv')

# Get baseline YSZ/GDC properties
ysz_gdc_baseline = df_interface[
    (df_interface['Interface'] == 'YSZ/GDC') & 
    (df_interface['Condition'] == 'Baseline')
]

Gc = ysz_gdc_baseline['Gc_int'].values[0]  # 2.15 J/m²
sigma_max = ysz_gdc_baseline['Sigma_max'].values[0]  # 222.5 MPa

# Load LSCF non-stoichiometry for interpolation
df_lscf = pd.read_csv('csv_files/03_LSCF_nonstoichiometry_data.csv')

# Interpolate for specific condition
import numpy as np
temp_query = 800  # °C
pO2_query = 0.21  # atm

data_800C = df_lscf[df_lscf['Temperature_C'] == temp_query]
delta = np.interp(np.log10(pO2_query), 
                  data_800C['log_pO2'], 
                  data_800C['Delta_delta'])
```

---

## ⚡ Quick Values at a Glance

### Most Important Parameters

| Parameter | Symbol | Value | File | Row/Filter |
|-----------|--------|-------|------|------------|
| **BK Exponent** | η | 2.1 | 01 | Row 1 |
| **Phase-field length** | l₀ | 10 nm | 01 | Rows 9-10 |
| **Crack regularization** | l | 0.5 µm | 01 | Row 11 |
| **YSZ/GDC Gc (fresh)** | Gc,int | 2.15 J/m² | 02 | Row 1 |
| **GDC/LSCF Gc (fresh)** | Gc,int | 1.0 J/m² | 02 | Row 3 |
| **GDC/LSCF Gc (1000h)** | Gc,int | 0.4 J/m² | 02 | Row 11 |
| **LSCF anisotropy** | β₃₃/β₁₁ | 1.7 | 01 | Row 2 |
| **Mesh ratio** | h/l₀ | 0.25 | 05 | Row 1 |
| **Convergence tolerance** | εtol | 10⁻⁷ | 05 | Row 7 |

### Material Properties at 800°C

| Material | E (GPa) | ν | TEC (ppm/K) | Kc (MPa·m⁰·⁵) |
|----------|---------|---|-------------|---------------|
| **LSCF** | 40 | 0.30 | 13.5 | 0.9 |
| **YSZ** | 200 | 0.31 | 10.5 | 2.8 |
| **GDC** | 160 | 0.32 | 12.0 | 1.2 |

*Source: File 06*

### Typical Operating Conditions

| Condition | T (°C) | j (A/cm²) | V (V) | pO₂ cathode |
|-----------|--------|-----------|-------|-------------|
| **OCV** | 800 | 0.00 | 1.10 | 0.21 atm |
| **Normal operation** | 800 | 0.50 | 0.75 | 0.21 atm |
| **High current** | 800 | 1.00 | 0.55 | 0.21 atm |

*Source: File 07*

---

## 📐 Dimensional Consistency Check

### Length Scales
- Phase-field interface: **5-20 nm**
- GDC grain size: **50 nm**
- Mesh size: **1.25-10 nm** (h/l₀ = 0.125-0.5)
- Cohesive zone: **0.018-0.097 µm** (18-97 nm)
- Crack regularization: **0.5 µm** (500 nm)

### Energy Scales
- Bulk fracture: **1.5-10 J/m²**
- Interface fresh: **0.5-2.5 J/m²**
- Interface degraded: **0.2-0.8 J/m²**

### Stress Scales
- Interface strength: **80-280 MPa**
- Operating stress: ~**50-150 MPa** (estimated from thermal/chemical mismatch)

---

## 🛠️ Troubleshooting

### Problem: Non-convergence in UMAT

**Check:**
1. File 05, row 7: Use εtol = 10⁻⁷ (not too strict)
2. File 05, row 8: Increase max iterations to 50
3. File 05, row 12: Reduce load step Δλ to 0.001

### Problem: Artificial crack branching

**Check:**
1. File 05, row 1: Ensure h/l₀ ∈ [0.125, 0.5]
2. File 01, row 9-10: Verify l₀ = 5-20 nm is appropriate for your GDC grain size
3. File 05, row 4: Check degradation function residual k_res = 10⁻⁶

### Problem: Interface delamination too early/late

**Check:**
1. File 02: Are you using the correct interface condition?
   - Fresh cell → Baseline
   - After operation → Use time-dependent row (100h/500h/1000h)
2. File 08: Verify cohesive zone parameters match your interface type
3. File 02: Check if interdiffusion/Sr-segregation should be included

---

## 📥 Download Instructions

### For CSV Data Only
```bash
# Download the zip file
unzip calibrated_parameters_dataset.zip

# All 8 CSV files will be extracted to csv_files/
```

### For Complete Dataset
```bash
# Clone or download the entire calibrated_parameters_dataset/ folder
# Contains: CSV files, figures, documentation, Python script
```

---

## 📞 Need More Information?

- **Detailed explanations**: See `README.md`
- **Dataset overview**: See `DATASET_SUMMARY.md`
- **Regenerate figures**: Run `python3 generate_figures.py`

---

**Quick Reference Version**: 1.0  
**Last Updated**: February 11, 2026  
**Total Parameters Available**: 224+ calibrated values

---

