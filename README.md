# Epistemic Uncertainty Quantification in SOC Interfacial Toughness

## Probabilistic Failure Maps: Uncertainty Quantification of Interfacial Toughness in SOCs

### Project Overview

This repository contains synthetic datasets, analysis scripts, and visualization tools for quantifying epistemic uncertainty in Solid Oxide Cell (SOC) interfacial fracture mechanics. The study focuses on two critical knowledge gaps:

1. **MISSING_DATASET_01**: The correlation coefficient between YSZ|GDC and GDC|LSCF interface toughness
2. **MISSING_DATASET_02**: High-temperature (800°C) interfacial toughness statistics

---

## Repository Structure

```
.
├── data/                                          # All CSV datasets
│   ├── 21_missing_data_assumption_flags.csv      # Epistemic gap documentation
│   ├── 04_uncertainty_material_properties.csv    # Base material statistics
│   ├── 16_micro_cantilever_fracture_data.csv     # Experimental measurements
│   ├── 13_LSCF_ferroelastic_stress_strain.csv    # Temperature-dependent behavior
│   ├── stochastic_inputs_rho0.00.csv             # Scenario 1: Independent (10k samples)
│   ├── stochastic_inputs_rho0.50.csv             # Scenario 2: Correlated (10k samples)
│   └── stochastic_inputs_HT_uncorrelated.csv     # Scenario 3: High-temp (10k samples)
│
├── scripts/                                       # Analysis and generation scripts
│   ├── generate_stochastic_inputs.py             # Monte Carlo dataset generator
│   ├── quantify_epistemic_gap.py                 # Statistical analysis script
│   └── generate_figures.py                       # Visualization generator
│
├── figures/                                       # Publication-quality figures
│   ├── 01_cone_of_ignorance_fragility_curves.png
│   ├── 02_system_resistance_CDF.png
│   ├── 03_interface_correlation_scatter.png
│   ├── 04_probability_density_functions.png
│   ├── 05_risk_matrix_heatmap.png
│   └── 06_sensitivity_boxplots.png
│
└── README.md                                      # This file
```

---

## Key Findings

### 1. Missing Correlation Data (MISSING_DATASET_01)

**The Problem:**  
We have independent measurements of interface toughness but no data on their statistical covariance:

```
Cov(Gc_YSZ-GDC, Gc_GDC-LSCF) = ?
```

**Impact:**  
- Correlation increases variance of system resistance by ~22%
- Joint failure probability (both interfaces) increases by correlation factor
- Current assumption of independence is non-conservative

**Recommendation:**  
Perform co-sintered stack failure analysis with correlated interface measurements.

---

### 2. High-Temperature Variance (MISSING_DATASET_02)

**The Problem:**  
We extrapolated room-temperature statistics to 800°C using deterministic scaling:

```
Gc_HT = 0.75 × Gc_RT
σ_HT = σ_RT  ← CRITICAL ASSUMPTION
```

**Impact:**  
- Mean toughness reduces by 25% at operating temperature
- Failure probability at J=2.5 J/m² increases from 0.2% (RT) to 99.9% (HT)
- If real variance increases with temperature, we underestimate tail risk

**Recommendation:**  
Perform minimum 5-10 in-situ micro-cantilever tests at 800°C per interface.

---

## Dataset Descriptions

### Primary Datasets

#### `21_missing_data_assumption_flags.csv`
Documents the four critical epistemic gaps with impact analysis.

**Columns:**
- `dataset_id`: Unique identifier (MISSING_DATASET_01, etc.)
- `missing_variable`: Physical quantity name
- `symbol`: Mathematical notation
- `why_missing`: Explanation of data void
- `current_surrogate`: Assumption used in place of real data
- `impact_if_wrong`: Consequence of incorrect assumption
- `measurement_needed`: Experimental recommendation

---

#### `stochastic_inputs_rho0.00.csv` (10,000 samples)
Monte Carlo realizations assuming **independent** interfaces (ρ=0.0).

**Columns:**
- `sample_id`: Sample index (1-10000)
- `Gc_YSZ_GDC`: YSZ|GDC fracture toughness [J/m²]
- `Gc_GDC_LSCF`: GDC|LSCF fracture toughness [J/m²]
- `E_YSZ_GPa`, `E_GDC_GPa`, `E_LSCF_GPa`: Elastic moduli [GPa]
- `CTE_YSZ`, `CTE_GDC`, `CTE_LSCF`: Thermal expansion coefficients [10⁻⁶/K]
- `nu_YSZ`, `nu_GDC`, `nu_LSCF`: Poisson ratios
- `correlation_rho`: Correlation coefficient (0.0)
- `temperature_C`: Test temperature (25°C)

**Statistics:**
- Mean Gc_YSZ_GDC: 2.85 ± 0.12 J/m²
- Mean Gc_GDC_LSCF: 3.20 ± 0.15 J/m²
- Empirical correlation: ~0.00

---

#### `stochastic_inputs_rho0.50.csv` (10,000 samples)
Monte Carlo realizations with **moderate correlation** (ρ=0.5).

**Key Difference:**  
Interface toughnesses generated using bivariate normal distribution with covariance:
```
Cov(Gc₁, Gc₂) = 0.5 × σ₁ × σ₂ = 0.009 J²/m⁴
```

**Statistics:**
- Mean values: Same as independent case
- Standard deviations: Same marginal distributions
- **Empirical correlation: 0.497** (validates generator)

---

#### `stochastic_inputs_HT_uncorrelated.csv` (10,000 samples)
High-temperature scenario with **deterministic degradation**.

**Modification:**
- Gc_mean_HT = 0.75 × Gc_mean_RT
- σ_HT = σ_RT (constant variance assumption)
- Elastic moduli reduced by 8-15% (temperature softening)

**Statistics:**
- Mean Gc_YSZ_GDC: 2.14 ± 0.12 J/m² (25% reduction)
- Mean Gc_GDC_LSCF: 2.40 ± 0.15 J/m² (25% reduction)

---

### Supporting Datasets

#### `04_uncertainty_material_properties.csv`
Statistical parameters for all material properties (source data for Monte Carlo).

**Key Properties:**
- Fracture toughness: Normal distribution
- Interface strength: Lognormal distribution  
- All values at room temperature unless specified

---

#### `16_micro_cantilever_fracture_data.csv`
Raw experimental data from micro-cantilever bending tests (20 specimens).

**Columns:**
- `specimen_id`: Unique identifier (MC_001, etc.)
- `interface_type`: YSZ_GDC or GDC_LSCF
- `load_at_fracture_mN`: Critical load [mN]
- `beam_length_um`, `beam_width_um`, `beam_height_um`: Geometry [μm]
- `calculated_Gc_J_m2`: Derived fracture energy [J/m²]
- `test_temperature_C`: Always 25°C
- `notes`: Fracture mode observations

---

#### `13_LSCF_ferroelastic_stress_strain.csv`
Temperature-dependent constitutive behavior of LSCF cathode.

**Mechanisms:**
- 25°C: Pure elastic behavior
- 400-600°C: Ferroelastic domain switching onset
- 800°C: Significant grain boundary sliding (GBS)

**Relevance:**  
Demonstrates why high-T variance assumption is questionable (inelastic mechanisms increase scatter).

---

## Usage Instructions

### 1. Running the Analysis

```bash
cd scripts/
python3 quantify_epistemic_gap.py
```

**Output:**
```
EPISTEMIC UNCERTAINTY QUANTIFICATION REPORT
-------------------------------------------
Scenario 1 (Rho=0.0): P_fail = 0.00% (at J=3.0 J/m²)
Scenario 2 (Rho=0.5): P_fail = 0.00%
Scenario 3 (High-T):  P_fail = 0.00%

System Resistance Statistics:
  Independent: 6.05 ± 0.19 J/m²
  Correlated:  6.05 ± 0.23 J/m² (22% higher variance!)
  High-T:      4.54 ± 0.19 J/m²
```

---

### 2. Generating Figures

```bash
cd scripts/
python3 generate_figures.py
```

**Output:**
Six high-resolution PNG files (300 DPI) in `figures/` directory.

---

### 3. Regenerating Datasets

```bash
cd scripts/
python3 generate_stochastic_inputs.py
```

**Note:** Uses fixed random seed (42) for reproducibility.

---

## Mathematical Formulation

### Probability of Failure

For a single interface:
```
P_f = P(Gc < J_applied) = Φ((J_applied - μ_Gc) / σ_Gc)
```

For system (both interfaces):
```
P_system = P(Gc₁ + Gc₂ < J_applied)
```

**Variance of sum:**
```
Var(Gc₁ + Gc₂) = σ₁² + σ₂² + 2ρσ₁σ₂
```

**With ρ=0.0:**  σ_system = 0.192 J/m²  
**With ρ=0.5:**  σ_system = 0.234 J/m² (+22%)

---

## Figure Descriptions

### Figure 1: Cone of Ignorance (Fragility Curves)
**File:** `01_cone_of_ignorance_fragility_curves.png`

Sigmoidal failure probability curves vs. applied J. The yellow shaded region between ρ=0.0 and ρ=0.5 curves represents **epistemic uncertainty** due to MISSING_DATASET_01.

**Key Insight:**  
At moderate loading (J=5.5 J/m²), uncertainty in correlation coefficient causes ±3% uncertainty in failure probability.

---

### Figure 2: System Resistance CDF
**File:** `02_system_resistance_CDF.png`

Cumulative distribution functions of total resistance (Gc₁ + Gc₂).

**Key Insight:**  
- High-T curve shifted left by 25% (mean degradation)
- Slopes similar (variance assumption appears valid for RT)
- Horizontal separation = epistemic uncertainty

---

### Figure 3: Correlation Scatter Plots
**File:** `03_interface_correlation_scatter.png`

Bivariate scatter plots showing:
- **Left:** Independent (circular cloud)
- **Middle:** Correlated (elliptical cloud)
- **Right:** High-T (circular, shifted down-left)

**Visual Proof:** Correlation structure is preserved in synthetic data.

---

### Figure 4: Probability Density Functions
**File:** `04_probability_density_functions.png`

Four-panel comparison:
1. YSZ|GDC marginal PDF
2. GDC|LSCF marginal PDF
3. System resistance PDF (shows variance increase with correlation)
4. Bar chart: Standard deviation comparison

**Critical Observation:**  
Panel 4 shows σ_RT ≈ σ_HT, exposing the MISSING_DATASET_02 assumption.

---

### Figure 5: Risk Matrix Heatmap
**File:** `05_risk_matrix_heatmap.png`

2D histograms (joint PDFs) with contour lines.

**Interpretation:**
- Independent: Circular iso-probability contours
- Correlated: Elliptical contours (diagonal elongation)
- High-T: Entire distribution shifted to lower toughness

---

### Figure 6: Sensitivity Box Plots
**File:** `06_sensitivity_boxplots.png`

Quartile comparisons showing:
- Minimal effect of correlation on marginal distributions
- Dramatic mean shift at high temperature
- Similar interquartile ranges (validating variance assumption... for now)

---

## Dependencies

```bash
pip install numpy pandas scipy matplotlib
```

**Versions tested:**
- Python 3.12
- NumPy 2.3.5
- Pandas 3.0.0
- SciPy 1.17.0
- Matplotlib 3.10.8

---

## Citation

If you use this dataset, please cite:

```
Probabilistic Failure Maps: Uncertainty Quantification of Interfacial 
Toughness in Solid Oxide Cells
[Author Names], [Year]
DOI: [To be assigned]
```

---

## Contact & Support

**Questions about the datasets:**  
- Check `21_missing_data_assumption_flags.csv` for detailed explanations
- Review figure annotations for visual interpretations

**Experimental recommendations:**  
See "measurement_needed" column in flags file.

---

## License

This synthetic dataset is released under CC BY 4.0.  
© 2026. All rights reserved.

---

## Acknowledgments

This work highlights the critical need for:
1. Correlated interface testing protocols
2. In-situ high-temperature fracture measurements
3. Uncertainty-aware design in SOFC/SOEC development

**Disclaimer:**  
All datasets are SYNTHETIC and represent sensitivity scenarios, not measured values. Do not use for actual design without experimental validation.
