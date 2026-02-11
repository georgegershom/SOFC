# Probabilistic Failure Maps: Uncertainty Quantification of Interfacial Toughness in SOCs

## Dataset Documentation

This repository contains fabricated datasets for the study "Probabilistic Failure Maps: Uncertainty Quantification of Interfacial Toughness in Solid Oxide Cells (SOCs)". These datasets address critical gaps in experimental data needed for stochastic phase-field simulations of interfacial delamination in co-sintered fuel cell stacks.

---

## 📁 Datasets Overview

### 1. **04_uncertainty_material_properties.csv**
**Description:** Statistical summary of material properties with uncertainty quantification for YSZ, GDC, and LSCF materials at both room temperature (25°C) and operating temperature (800°C).

**Key Features:**
- Material properties: Fracture Energy (Gc), Elastic Modulus, Poisson's Ratio, Thermal Expansion Coefficient
- Statistical metrics: Mean, Standard Deviation, Coefficient of Variation
- Interface-specific data for YSZ|GDC and GDC|LSCF interfaces
- Sample sizes and data sources included

**Columns:**
- `Material`: Material type (YSZ, GDC, LSCF)
- `Interface`: Interface designation
- `Property`: Property name
- `Temperature_C`: Temperature in Celsius
- `Mean_Value`: Mean value of the property
- `Std_Deviation`: Standard deviation
- `Coefficient_of_Variation`: CV = σ/μ
- `Distribution_Type`: Statistical distribution (Normal, Lognormal)
- `Unit`: Physical units
- `Sample_Size`: Number of experimental measurements
- `Data_Source`: Measurement technique

**Key Statistics:**
- YSZ|GDC Fracture Energy @ 25°C: μ = 2.15 J/m², σ = 0.42 J/m²
- YSZ|GDC Fracture Energy @ 800°C: μ = 1.89 J/m², σ = 0.37 J/m²
- GDC|LSCF Fracture Energy @ 25°C: μ = 1.02 J/m², σ = 0.21 J/m²
- GDC|LSCF Fracture Energy @ 800°C: μ = 0.88 J/m², σ = 0.18 J/m²

---

### 2. **16_micro_cantilever_fracture_data.csv**
**Description:** Individual micro-cantilever fracture test results for interfacial toughness measurements.

**Key Features:**
- 45 tests for YSZ|GDC interface at room temperature
- 32 tests for YSZ|GDC interface at 800°C
- 52 tests for GDC|LSCF interface at room temperature
- 28 tests for GDC|LSCF interface at 800°C
- Total: 157 individual fracture tests

**Columns:**
- `Test_ID`: Unique test identifier
- `Interface`: Interface type (YSZ|GDC or GDC|LSCF)
- `Material`: Material being tested
- `Temperature_C`: Test temperature
- `Beam_Width_um`: Cantilever beam width (μm)
- `Beam_Length_um`: Cantilever beam length (μm)
- `Beam_Thickness_um`: Cantilever beam thickness (μm)
- `Critical_Load_mN`: Critical load at fracture (mN)
- `Fracture_Energy_Gc_J_m2`: Calculated fracture energy (J/m²)
- `Elastic_Modulus_GPa`: Material elastic modulus (GPa)
- `Mode_Mixity_Angle_deg`: Mode mixity angle (degrees)
- `Failure_Mode`: Type of failure (Interfacial)
- `Notch_Depth_um`: Pre-notch depth (μm)
- `Testing_Date`: Date of test
- `Lab_ID`: Laboratory identifier
- `Operator`: Operator identifier

**Data Quality:**
- Multiple operators and laboratories for reproducibility
- Systematic variation in geometric parameters
- Temperature-dependent mechanical properties

---

### 3. **stochastic_inputs_RT_uncorrelated.csv**
**Description:** Stochastic input parameters for Abaqus UEL simulations at room temperature (25°C) assuming uncorrelated interfaces (ρ = 0.0).

**Features:**
- 500 random samples
- Independent sampling of YSZ|GDC and GDC|LSCF fracture energies
- Current assumption for baseline simulations

**Columns:**
- `Sample_ID`: Unique sample identifier (S0001-S0500)
- `Gc_YSZ_GDC`: Fracture energy for YSZ|GDC interface (J/m²)
- `Gc_GDC_LSCF`: Fracture energy for GDC|LSCF interface (J/m²)
- `Temperature_C`: Temperature (25°C)
- `Correlation_Coeff`: Correlation coefficient (0.0)

**Use Case:** Baseline probabilistic simulations assuming statistical independence between interface properties.

---

### 4. **stochastic_inputs_HT_uncorrelated.csv**
**Description:** Stochastic input parameters for high-temperature (800°C) simulations with uncorrelated interfaces.

**Features:**
- 500 random samples at operating temperature
- Proportional variance scaling from RT data
- Temperature-dependent mean values

**Application:** Operating temperature simulations for reliability assessment.

---

### 5. **stochastic_inputs_RT_correlated_rho050.csv**
**Description:** Sensitivity analysis dataset with moderate correlation (ρ = 0.5) between interfaces.

**Features:**
- 500 correlated samples
- Simulates effect of process defects affecting both interfaces
- Correlation coefficient ρ ≈ 0.5 (actual: 0.546)

**Scientific Rationale:** In co-sintered stacks, processing defects (e.g., porosity, grain growth anomalies) often affect multiple layers simultaneously. A "bad" sintering run would degrade both interfaces, implying positive correlation.

**Use Case:** Sensitivity analysis to determine if acquiring experimental correlation data is worth the effort.

---

## 🔬 Scientific Context

### Missing Data Gaps Addressed

#### **MISSING_DATASET_01: Interfacial Property Correlation Matrix**
**Problem:** No experimental data exists quantifying the statistical dependence between fracture toughness of YSZ|GDC and GDC|LSCF interfaces.

**Current Assumption:** ρ = 0.0 (uncorrelated)  
**Reality Check:** Process defects likely cause ρ > 0  
**Conservatism:** Current assumption may underestimate system failure probability

**Resolution:** Datasets with ρ = 0.0 (baseline) and ρ = 0.5 (sensitivity) provided.

#### **MISSING_DATASET_02: High-Temperature Fracture Statistics**
**Problem:** Most micro-cantilever data is collected at RT. High-T behavior may differ due to grain boundary softening or creep.

**Current Assumption:** Proportional Variance Scaling  
CV(800°C) = CV(25°C)

**Implementation:**
- Mean values scaled by elastic modulus ratio: Gc(800°C) = Gc(25°C) × E(800°C)/E(25°C)
- Standard deviation scaled proportionally: σ(800°C) = μ(800°C) × CV(25°C)

---

## 📊 Visualization Files

### **correlation_comparison_scatter.png**
Three-panel scatter plot showing:
1. **Series A:** Independent sampling (ρ = 0.0) - Current assumption
2. **Series B:** Moderate correlation (ρ = 0.5) - Hypothetical reality
3. **Sensitivity Case:** Strong correlation (ρ = 0.8)

**Purpose:** Visualize impact of missing correlation data on input parameter space.

### **fracture_energy_distributions.png**
Four-panel histogram showing:
1. YSZ|GDC fracture energy distribution @ 25°C
2. YSZ|GDC fracture energy distribution @ 800°C
3. GDC|LSCF fracture energy distribution @ 25°C
4. GDC|LSCF fracture energy distribution @ 800°C

**Purpose:** Display statistical distributions used for Monte Carlo sampling.

---

## 🐍 Python Scripts

### **generate_correlated_samples.py**
Comprehensive script for generating stochastic inputs with controllable correlation.

**Key Functions:**
- `generate_correlated_fracture_energies()`: Multivariate normal sampling with specified correlation
- `generate_stochastic_inputs()`: Complete parameter set generation for simulations
- `plot_correlation_comparison()`: Create comparison scatter plots
- `plot_distributions()`: Generate distribution histograms

**Usage:**
```python
python3 generate_correlated_samples.py
```

**Outputs:**
- 3 CSV files with stochastic inputs
- 2 PNG visualization files
- Console statistics summary

---

## 📋 Governing Equations

### Stochastic Sampling

**Independent Case (ρ = 0.0):**
```
Gc₁ ~ N(μ₁, σ₁²)  [YSZ|GDC]
Gc₂ ~ N(μ₂, σ₂²)  [GDC|LSCF]
```

**Correlated Case (ρ > 0):**
```
[Gc₁]     [μ₁]   [σ₁²      ρσ₁σ₂]
[Gc₂] ~ N([μ₂], [ρσ₁σ₂    σ₂²  ])
```

### High-Temperature Scaling
```
μ(800°C) = μ(25°C) × E(800°C) / E(25°C)
σ(800°C) = μ(800°C) × CV(25°C)
CV = σ / μ (constant across temperatures)
```

---

## 🔧 Integration with Abaqus UEL

### Workflow

1. **Preprocessing:**
   ```python
   df = pd.read_csv('stochastic_inputs_RT_uncorrelated.csv')
   for idx, row in df.iterrows():
       # Generate Abaqus input file with PROPS = [Gc_YSZ_GDC, Gc_GDC_LSCF]
   ```

2. **UEL Parameter Passing:**
   ```fortran
   SUBROUTINE UEL(...)
       REAL*8 PROPS(*)
       Gc_YSZ_GDC = PROPS(1)
       Gc_GDC_LSCF = PROPS(2)
       ! Phase-field formulation uses these values
   END SUBROUTINE
   ```

3. **Post-Processing:**
   - Collect failure probabilities for each sample
   - Generate probabilistic failure maps
   - Compute reliability metrics

---

## ✅ Verification & QA

### Data Integrity Checks

1. **Statistical Consistency:**
   - Verify generated samples match target distributions (mean ± 5%, std ± 10%)
   - Check positive definiteness of covariance matrices

2. **Correlation Validation:**
   ```python
   actual_rho = df[['Gc_YSZ_GDC', 'Gc_GDC_LSCF']].corr().iloc[0,1]
   assert abs(actual_rho - target_rho) < 0.05
   ```

3. **Physical Constraints:**
   - All fracture energies > 0
   - Elastic moduli decrease with temperature
   - CV values within realistic ranges (0.05-0.30)

### Sample Results (Verification)

**Room Temperature Uncorrelated Dataset:**
```
Gc_YSZ_GDC:  μ = 2.153 J/m², σ = 0.423 J/m²  ✓
Gc_GDC_LSCF: μ = 1.032 J/m², σ = 0.212 J/m²  ✓
Correlation: ρ = 0.002 (target: 0.000)        ✓
```

**Room Temperature Correlated Dataset (ρ = 0.5):**
```
Correlation: ρ = 0.546 (target: 0.500)        ✓
```

---

## 📌 Usage Recommendations

### For Baseline Simulations:
- Use `stochastic_inputs_RT_uncorrelated.csv` for room temperature
- Use `stochastic_inputs_HT_uncorrelated.csv` for operating temperature

### For Sensitivity Analysis:
- Compare results from uncorrelated vs. `stochastic_inputs_RT_correlated_rho050.csv`
- If correlation significantly affects failure probability, prioritize experimental measurement of ρ

### For Custom Studies:
- Modify `generate_correlated_samples.py` to:
  - Change number of samples
  - Adjust correlation coefficients
  - Include additional material parameters
  - Generate different temperature conditions

---

## 📖 Citation & References

If you use these datasets, please cite:

**Dataset:**
```
Probabilistic Failure Maps Dataset (2024)
Uncertainty Quantification of Interfacial Toughness in Solid Oxide Cells
Generated for stochastic phase-field simulations
```

**Related Material Systems:**
- YSZ: Yttria-Stabilized Zirconia (8YSZ)
- GDC: Gadolinium-Doped Ceria (Ce₀.₉Gd₀.₁O₂)
- LSCF: Lanthanum Strontium Cobalt Ferrite (La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃)

---

## 🔍 Key Assumptions Summary

| Parameter | Assumed Value | Justification |
|-----------|---------------|---------------|
| Correlation ρ | 0.0 (baseline) | Conservative; lack of experimental data |
| CV Scaling | Constant with T | Standard practice for ceramics |
| Gc Temperature Dependence | E(T) ratio | Thermodynamic consistency |
| Distribution Type | Normal | Central limit theorem; sufficient for Gc > 3σ |

---

## ⚠️ Important Notes

1. **Fabricated Data:** These datasets are synthetically generated for demonstration purposes. Real experimental measurements may differ.

2. **Non-Conservative Assumption:** ρ = 0.0 may underestimate system failure probability if actual correlation is positive.

3. **Temperature Effects:** High-temperature data assumes elastic scaling. Actual behavior may include creep, phase transformations, or oxidation effects not captured here.

4. **Weibull Statistics:** For brittle materials, Weibull distributions are often more appropriate than Normal distributions. Consider this for future refinements.

---

## 📦 File Manifest

### CSV Files (in SOC_datasets.zip):
- `04_uncertainty_material_properties.csv` (2.7 KB)
- `16_micro_cantilever_fracture_data.csv` (24.3 KB)
- `stochastic_inputs_RT_uncorrelated.csv` (18.5 KB)
- `stochastic_inputs_HT_uncorrelated.csv` (18.5 KB)
- `stochastic_inputs_RT_correlated_rho050.csv` (18.5 KB)

### Visualization Files:
- `correlation_comparison_scatter.png` (high-resolution, 300 DPI)
- `fracture_energy_distributions.png` (high-resolution, 300 DPI)

### Scripts:
- `generate_correlated_samples.py` (standalone Python script)

### Documentation:
- `README_DATASETS.md` (this file)

---

## 📧 Support & Questions

For questions about dataset usage, interpretation, or extensions, please refer to:
- The Python script for implementation details
- Scientific literature on SOC mechanical reliability
- Stochastic finite element analysis methodologies

---

**Last Updated:** February 11, 2026  
**Version:** 1.0  
**Status:** Complete - Ready for simulation deployment
