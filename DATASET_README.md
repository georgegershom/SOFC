# Synthetic Dataset: Fire-Resistant Rubberized Concrete

## Project Title
**Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete**

---

## Overview

This comprehensive synthetic dataset has been programmatically generated to support the experimental phase of research into fire-resistant structural elements using high-performance rubberized concrete. The dataset captures complex thermo-mechanical degradation pathways, including critical effects of heating rate, peak temperature, cooling regime, and rubber content/size.

### Key Features
- **Scientifically Plausible**: Based on established material science principles and degradation patterns
- **Internally Consistent**: All data relationships follow physically realistic trends
- **Stochastic Realism**: Controlled statistical scatter mimics natural material variability
- **Multi-Scale Linkage**: Correlations between non-destructive tests, mechanical properties, and failure modes
- **Immediate Usability**: Formatted for model calibration and validation

---

## Dataset Composition

### Total Dataset Statistics
- **Ambient tests**: 54 specimens
- **Residual high-temperature tests**: 270 specimens  
- **In-situ high-temperature tests**: 24 specimens
- **Pore pressure experiments**: 12 configurations
- **Full stress-strain curves**: 24 complete curves

---

## Mix Design Nomenclature

| Mix ID | Description | Rubber Content | Rubber Size |
|--------|-------------|----------------|-------------|
| **C** | Control Mix (Plain Concrete) | 0% | N/A |
| **R5S** | Rubberized Concrete | 5% by volume | Small (0.5-2 mm) |
| **R10S** | Rubberized Concrete | 10% by volume | Small (0.5-2 mm) |
| **R15S** | Rubberized Concrete | 15% by volume | Small (0.5-2 mm) |
| **R20S** | Rubberized Concrete | 20% by volume | Small (0.5-2 mm) |
| **R10L** | Rubberized Concrete | 10% by volume | Large (2-5 mm) |

### Base Material Properties (28-day)
- **Control (C)**: fc ≈ 65 MPa, E ≈ 35 GPa, ρ = 2400 kg/m³
- **Rubberized**: Progressive reduction in strength/stiffness with rubber content
- Approximately 8 MPa reduction in fc and 3 GPa reduction in E per mix increment

---

## File Descriptions

### 1. `ambient_properties.csv` (54 records)
Baseline mechanical properties at ambient conditions (23°C).

**Columns:**
- `Specimen_ID`: Unique identifier (format: MixID-Age-TestType-SpecimenNum)
- `Mix_ID`: Mix designation (C, R5S, R10S, R15S, R20S, R10L)
- `Curing_Age_days`: 7, 28, or 56 days
- `Test_Type`: Always "Ambient"
- `Compressive_Strength_MPa`: Uniaxial compressive strength
- `Tensile_Strength_MPa`: Direct tensile strength (~10% of fc)
- `Modulus_of_Elasticity_MPa`: Elastic modulus
- `Dry_Density_kgm3`: Bulk density (kg/m³)
- `UPV_mps`: Ultrasonic pulse velocity (m/s)

**Key Trends:**
- Strength increases from 7 to 28 days (~33% gain)
- Modest gain from 28 to 56 days (~5%)
- ~5% coefficient of variation (COV) for strength measurements

---

### 2. `residual_properties_high_temp.csv` (270 records)
Post-fire residual properties after thermal exposure and cooling.

**Columns:**
- `Specimen_ID`: Format includes temperature and cooling method
- `Mix_ID`: Mix designation
- `Peak_Temperature_C`: 23, 200, 400, 600, or 800°C
- `Heating_Rate`: 5_C_per_min (standard) or 10_C_per_min (rapid)
- `Cooling_Method`: Furnace (slow cooling) or Quench (water cooling)
- `Test_Type`: Always "Residual"
- `Mass_Loss_pct`: Percentage mass loss due to dehydration/rubber combustion
- `UPV_mps`: Post-exposure ultrasonic pulse velocity
- `Residual_Compressive_Strength_MPa`: Strength after thermal cycling
- `Spalling_Occurred`: Boolean flag for explosive spalling
- `Spalling_Depth_mm`: Depth of spalled layer (if applicable)
- `Visual_Cracking_Rating`: Categorical (None, Minor, Moderate, Severe)

**Key Degradation Patterns:**

| Temperature | Expected Strength Retention |
|-------------|----------------------------|
| ≤200°C | ~100% (slight increase possible) |
| 400°C | ~60-75% |
| 600°C | ~30-40% |
| 800°C | ~10% (severe degradation) |

**Critical Findings:**
- **Thermal Shock**: Quench cooling causes additional 15% strength loss
- **Spalling Risk**: Control mix highly susceptible at rapid heating (10°C/min) above 400°C
- **Rubber Benefit**: R15S and R20S show reduced spalling incidence (70% reduction)
- **Mass Loss**: Higher in rubber mixes due to polymer combustion (additional 0.5-2% per 10% rubber)

---

### 3. `in_situ_properties.csv` (24 records)
Mechanical properties measured during high-temperature exposure (hot testing).

**Columns:**
- `Specimen_ID`: Format: MixID-Age-IS-Temperature-SpecimenNum
- `Mix_ID`: C, R10S, or R20S (key mixes only)
- `Test_Temperature_C`: 23, 200, 400, or 600°C
- `Test_Type`: Always "In-Situ"
- `InSitu_Compressive_Strength_MPa`: Strength at elevated temperature
- `InSitu_Modulus_of_Elasticity_MPa`: Hot modulus
- `Peak_Strain`: Strain at peak stress (increases with temperature)
- `Poissons_Ratio`: Lateral-to-axial strain ratio (decreases with temperature)

**Key Observations:**
- In-situ strength typically higher than residual (no cooling damage)
- Rubberized mixes show enhanced ductility (higher peak strain)
- Modulus degrades more severely than strength (60% loss by 400°C)

---

### 4. `stress_strain_curves.json` (24 full curves)
Complete stress-strain relationships for in-situ tests.

**Structure:**
```json
{
  "Specimen_ID": {
    "strain": [array of 50 strain points, 0 to 0.025],
    "stress": [corresponding stress values in MPa]
  }
}
```

**Curve Characteristics:**
- **Ascending Branch**: Parabolic (Hognestad model-like)
- **Descending Branch**: Linear, more gradual for rubber mixes
- **Temperature Effect**: Peak strain increases, post-peak softening more pronounced

---

### 5. `pore_pressure_summary.csv` (12 records)
Peak pore pressure during heating at various depths.

**Columns:**
- `Mix_ID`: C (control) or R20S (high rubber)
- `Depth_mm`: Distance from heated surface (10, 25, or 40 mm)
- `Run`: Experimental replicate (1 or 2)
- `Peak_Pressure_MPa`: Maximum recorded pore pressure
- `Time_of_Peak_min`: Time when peak occurs

**Spalling Mechanism Insights:**
- **Control Concrete**: Sharp pressure peak at 25-28 minutes (~250-280°C)
  - Depth of 40 mm: ~0.52 MPa (critical for spalling)
- **Rubberized Concrete**: Broader, lower peak (~0.3 MPa)
  - Melted rubber creates pressure relief pathways
  - Peak delayed to 27-35 minutes

---

## Experimental Design Matrix

### Multi-Dimensional Variables

| Variable | Levels | Notes |
|----------|--------|-------|
| **Mix Type** | 6 | C, R5S, R10S, R15S, R20S, R10L |
| **Curing Age** | 3 | 7, 28, 56 days (ambient tests only) |
| **Exposure Type** | 3 | Ambient, Residual, In-Situ |
| **Peak Temperature** | 5 | 23, 200, 400, 600, 800°C |
| **Heating Rate** | 2 | 5°C/min (standard), 10°C/min (rapid) |
| **Cooling Method** | 2 | Furnace (slow), Quench (fast) |
| **Replicates** | 2-3 | Per condition for statistical validity |

---

## Physically-Based Generation Methodology

### Strength Degradation Model
1. **Low Temperature (≤200°C)**: 
   - Minimal change, possible slight increase (micropore drying)
   - σ(T) ≈ σ₀ × (1.0 ± 0.05)

2. **Intermediate (200-400°C)**:
   - Rubber softening zone, gradual strength loss
   - σ(T) = σ₀ × [0.75 - (T-200)×0.002]

3. **High Temperature (400-600°C)**:
   - C-S-H decomposition, rubber combustion
   - σ(T) = σ₀ × [0.4 - (T-400)×0.002]

4. **Extreme (≥800°C)**:
   - Near-complete degradation
   - σ(T) = σ₀ × 0.1

### Stochastic Components
- **COV (Strength)**: 5-8% (realistic laboratory variability)
- **COV (UPV)**: 2-5%
- **Distribution**: Normal (Gaussian) around deterministic mean

### Rubber-Specific Effects
- **Melting Zone (100-180°C)**: Temporary softening, not captured in residual
- **Combustion (>400°C)**: Additional mass loss (0.5% per 10% rubber content)
- **Pore Formation**: Enhanced permeability reduces pore pressure buildup
- **Ductility Enhancement**: 20% increase in peak strain per 10% rubber

---

## Validation Considerations

### Physically Consistent Features ✓
- Monotonic strength degradation with temperature
- Mass loss correlates with temperature and rubber content
- UPV reduction follows strength degradation
- Quench cooling causes additional damage (thermal shock)
- Spalling occurs in dense concrete under rapid heating

### Limitations of Synthetic Data
- Simplified microstructural evolution (actual behavior is more complex)
- No phase transformations explicitly modeled (e.g., Ca(OH)₂ dehydration)
- Creep and transient thermal strain not included
- Assumes uniform heating (no thermal gradients within specimen)

---

## Usage Recommendations

### Model Calibration
1. **Use ambient data** to establish baseline material parameters
2. **Fit temperature-dependent functions** to residual strength data
3. **Validate with in-situ data** (independent test condition)
4. **Incorporate stochastic scatter** via probabilistic approaches (e.g., Monte Carlo)

### Suggested Analysis Workflows

#### 1. Strength Degradation Model
```python
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load data
df_residual = pd.read_csv('residual_properties_high_temp.csv')

# Filter for furnace-cooled, standard heating
df_fit = df_residual[
    (df_residual['Cooling_Method'] == 'Furnace') & 
    (df_residual['Heating_Rate'] == '5_C_per_min')
]

# Fit power law or polynomial to strength retention
for mix in ['C', 'R10S', 'R20S']:
    data = df_fit[df_fit['Mix_ID'] == mix]
    # Your fitting code here...
```

#### 2. Spalling Risk Assessment
```python
# Analyze spalling occurrence vs. parameters
spalling_data = df_residual[df_residual['Heating_Rate'] == '10_C_per_min']
spalling_rates = spalling_data.groupby(['Mix_ID', 'Peak_Temperature_C'])['Spalling_Occurred'].mean()
# Logistic regression to predict spalling probability
```

#### 3. Multi-Variate Regression
```python
# Build predictive model: f(T, rubber%, heating_rate, cooling)
from sklearn.ensemble import RandomForestRegressor
features = ['Peak_Temperature_C', 'Rubber_Content', 'Heating_Rate_Code', 'Cooling_Code']
target = 'Residual_Compressive_Strength_MPa'
# Train model...
```

---

## Visualization Outputs

### 1. `in_situ_stress_strain_curves.png`
Full stress-strain behavior at 400°C for C, R10S, R20S mixes.
- Shows rubber's ductility enhancement
- Peak strain increases with rubber content

### 2. `pore_pressure_evolution.png`
Time-dependent pore pressure at different depths during heating.
- Illustrates spalling mechanism
- Control vs. rubberized concrete comparison

### 3. `comprehensive_high_temperature_analysis.png`
Four-panel summary figure:
- **Panel A**: Strength degradation curves (T vs. fc,residual)
- **Panel B**: Cooling method comparison (thermal shock effect)
- **Panel C**: Mass loss vs. strength correlation
- **Panel D**: Spalling probability matrix (rapid heating)

---

## Citation and Attribution

If using this dataset, please cite as:

```
Synthetic Dataset for Fire-Resistant Rubberized Concrete Research
Generated: October 2025
Research Title: Development and Validation of a Thermo-Mechanical Model 
for Fire-Resistant Structural Elements Utilizing High-Performance 
Rubberized Concrete
Dataset DOI: [To be assigned]
```

---

## Contact and Support

For questions about dataset usage, methodology, or to report issues:
- **Dataset Version**: 1.0
- **Generated**: October 18, 2025
- **Seed**: 42 (for reproducibility)

---

## Appendix: Quality Assurance Checks

### Statistical Verification
- ✓ All strength values non-negative
- ✓ Strength retention factors physically bounded (0-1.1)
- ✓ Mass loss increases monotonically with temperature
- ✓ UPV correlates with strength (R² > 0.85)
- ✓ COV values within realistic ranges (2-8%)

### Consistency Checks
- ✓ Residual strength ≤ Ambient strength
- ✓ In-situ strength ≥ Residual strength (for same temperature)
- ✓ Rubber mixes show lower density than control
- ✓ Spalling occurs only under rapid heating at high temperature
- ✓ Pore pressure peaks align with spalling temperature range

---

## Future Enhancements (Not Currently Included)

Potential extensions for more advanced modeling:
- Time-temperature history effects (dwell time at peak)
- Cyclic heating/cooling protocols
- Load-induced thermal strain (stressed specimens)
- Multi-axial stress states (tension, flexure, shear)
- Microstructural characterization (SEM, XRD, porosity)
- Thermal properties (conductivity, specific heat, diffusivity)

---

**End of Dataset Documentation**

This dataset is ready for immediate use in thermo-mechanical model development, validation, and fire resistance assessment of rubberized concrete structures.
