# Context Dataset for Residual Stress Prediction in Multi-layer Ceramic Structures

**Generated:** 2025-10-15  
**Purpose:** Predictive modeling of residual stress in co-sintered ceramic structures (e.g., Solid Oxide Fuel Cells)

## 📊 Dataset Overview

This comprehensive dataset enables **predictive** (not just descriptive) modeling of residual stress in multi-layer ceramic structures by capturing the causal relationships between process/material parameters and resulting stress states.

### Dataset Sizes Available

| Size | Samples | File Size (CSV) | File Size (Excel) | Use Case |
|------|---------|-----------------|-------------------|----------|
| **Small** | 100 | 149 KB | 101 KB | Quick testing, prototyping |
| **Medium** | 1,000 | 1.5 MB | 870 KB | Algorithm development |
| **Large** | 5,000 | 7.2 MB | 4.3 MB | Full model training |
| **XLarge** | 10,000 | 15 MB | 8.5 MB | Production models, deep learning |

### Features: 65 Total Parameters

- **Geometric Parameters:** 7 features
- **Anode Material Properties:** 11 features (Ni-YSZ)
- **Electrolyte Material Properties:** 11 features (YSZ/GDC)
- **Cathode Material Properties:** 11 features (LSM/LSCF)
- **Process Parameters:** 10 features (sintering profiles)
- **Derived Parameters:** 11 features (CTE mismatches, ratios)
- **Quality Flags:** 4 features (risk indicators)

## 🎯 Key Innovation: Causal Modeling

Unlike descriptive datasets that only capture correlations, this dataset includes:

1. **Input Parameters** (what you control):
   - Material properties (CTE, Young's modulus, creep)
   - Geometric design (layer thicknesses)
   - Process conditions (temperature profiles, atmosphere)

2. **Derived Features** (what causes stress):
   - CTE mismatches between layers
   - Stiffness ratios
   - Thickness ratios
   - Temperature history

This enables your ML model to answer **"what-if" questions**: *"What happens to residual stress if I increase the cooling rate by 2°C/min?"*

## 📁 Files Generated

For each dataset size, you get:

```
context_dataset_{size}.csv              # Main dataset (comma-separated)
context_dataset_{size}.xlsx             # Excel with multiple sheets
  ├── Full_Dataset                      # Complete data
  ├── Summary_Statistics                # Statistical overview
  └── Parameter_Ranges                  # DOE boundaries
context_dataset_{size}_metadata.json    # Detailed metadata
context_dataset_{size}_data_dictionary.txt  # Human-readable docs
```

## 🔬 Parameter Categories

### 1. Geometric Parameters

| Parameter | Range | Units | Description |
|-----------|-------|-------|-------------|
| `plate_length_mm` | 50 - 150 | mm | Plate length |
| `plate_width_mm` | 50 - 150 | mm | Plate width |
| `anode_thickness_um` | 300 - 1000 | μm | Anode final thickness |
| `electrolyte_thickness_um` | 5 - 50 | μm | Electrolyte final thickness |
| `cathode_thickness_um` | 20 - 100 | μm | Cathode final thickness |
| `green_density_fraction` | 0.45 - 0.65 | - | Initial green density |
| `green_shrinkage_factor` | 1.15 - 1.35 | - | Green-to-sintered multiplier |

### 2. Material Properties (per layer)

**Temperature-Dependent Properties:**
- Young's Modulus at 25°C and 1000°C
- CTE (Coefficient of Thermal Expansion) at 25°C and 1000°C
- Poisson's Ratio

**Sintering Behavior:**
- Sintering onset temperature (°C)
- Maximum shrinkage rate (/min)
- Total shrinkage fraction

**High-Temperature Deformation:**
- Creep activation energy (kJ/mol)
- Creep stress exponent (Norton law)

#### Typical Ranges (Anode - Ni-YSZ):
| Property | Range |
|----------|-------|
| Young's Modulus @ 25°C | 40 - 80 GPa |
| Young's Modulus @ 1000°C | 20 - 50 GPa |
| CTE @ 25°C | 10.5 - 13.5 ppm/K |
| CTE @ 1000°C | 12.0 - 15.0 ppm/K |
| Sintering Onset | 1100 - 1250 °C |

#### Typical Ranges (Electrolyte - YSZ):
| Property | Range |
|----------|-------|
| Young's Modulus @ 25°C | 180 - 220 GPa |
| Young's Modulus @ 1000°C | 120 - 160 GPa |
| CTE @ 25°C | 9.5 - 11.5 ppm/K |
| CTE @ 1000°C | 10.5 - 12.5 ppm/K |
| Sintering Onset | 1200 - 1350 °C |

#### Typical Ranges (Cathode - LSM/LSCF):
| Property | Range |
|----------|-------|
| Young's Modulus @ 25°C | 50 - 100 GPa |
| Young's Modulus @ 1000°C | 30 - 70 GPa |
| CTE @ 25°C | 11.0 - 14.0 ppm/K |
| CTE @ 1000°C | 12.5 - 16.0 ppm/K |
| Sintering Onset | 1000 - 1200 °C |

### 3. Process Parameters

| Parameter | Range | Units | Description |
|-----------|-------|-------|-------------|
| `heating_ramp_rate_C_per_min` | 1 - 10 | °C/min | Heating rate |
| `sintering_peak_temp_C` | 1300 - 1500 | °C | Maximum temperature |
| `sintering_hold_time_min` | 60 - 300 | min | Hold at peak |
| `cooling_ramp_rate_C_per_min` | 1 - 8 | °C/min | Cooling rate |
| `intermediate_hold_temp_C` | 800 - 1100 | °C | Optional intermediate hold |
| `intermediate_hold_time_min` | 0 - 120 | min | Intermediate hold duration |
| `oxygen_partial_pressure_atm` | 0.001 - 0.21 | atm | Oxygen concentration |
| `humidity_percent` | 0 - 5 | % | Moisture content |
| `atmosphere_type` | Categorical | - | Air, Argon, N₂, H₂/N₂, Vacuum |
| `temperature_profile_json` | JSON | - | Complete T(t) profile |

### 4. Derived Parameters (Critical for Stress)

These features capture the **physics** of residual stress generation:

| Parameter | Physical Meaning |
|-----------|------------------|
| `CTE_mismatch_anode_electrolyte_25C` | Differential expansion during cooling |
| `CTE_mismatch_cathode_electrolyte_25C` | Interface stress driver |
| `avg_CTE_mismatch_magnitude` | Overall mismatch severity |
| `stiffness_ratio_electrolyte_anode` | Load sharing ratio |
| `stiffness_ratio_electrolyte_cathode` | Constraint level |
| `thickness_ratio_anode_electrolyte` | Bending moment driver |
| `thickness_ratio_cathode_electrolyte` | Asymmetry measure |
| `total_process_time_min` | Time for stress relaxation |
| `cooling_heating_rate_ratio` | Thermal history asymmetry |

### 5. Quality Flags

Binary indicators for potentially problematic conditions:

| Flag | Condition | Risk |
|------|-----------|------|
| `flag_extreme_CTE_mismatch` | Avg mismatch > 3.5 ppm/K | High stress, cracking |
| `flag_thin_electrolyte` | Thickness < 10 μm | Mechanical fragility |
| `flag_fast_cooling` | Cooling rate > 6 °C/min | Quenching stresses |
| `flag_asymmetric_structure` | Large thickness imbalance | Warpage, curvature |

## 📈 Dataset Statistics (XLarge - 10,000 samples)

```
Total Features: 65
Memory Usage: ~10 MB

Key Distributions:
  CTE Mismatch Range: 0.02 - 4.14 ppm/K
  Sintering Temperature: 1300 - 1500 °C
  Total Thickness: 332.9 - 1134.1 μm

Atmosphere Distribution:
  Air:           48.5%
  Reducing H₂/N₂: 19.8%
  Argon:         15.6%
  Nitrogen:      10.6%
  Vacuum:         5.5%

Quality Flags:
  Extreme CTE mismatch:    167 samples (1.7%)
  Thin electrolyte:      1,111 samples (11.1%)
  Fast cooling:          2,857 samples (28.6%)
  Asymmetric structure:  7,270 samples (72.7%)
```

## 🚀 Usage Examples

### Loading the Dataset

```python
import pandas as pd
import json

# Load CSV
df = pd.read_csv('context_dataset_xlarge.csv')

# Load metadata
with open('context_dataset_xlarge_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

### Parsing Temperature Profiles

```python
# Extract and parse temperature profiles
sample = df.iloc[0]
profile = json.loads(sample['temperature_profile_json'])

import matplotlib.pyplot as plt
plt.plot(profile['time_points_min'], profile['temperature_C'])
plt.xlabel('Time (min)')
plt.ylabel('Temperature (°C)')
plt.title('Sintering Temperature Profile')
plt.grid(True)
plt.show()
```

### Feature Engineering for ML

```python
# Separate feature types
geometric_features = [col for col in df.columns if any(x in col for x in ['length', 'width', 'thickness', 'density'])]
material_features = [col for col in df.columns if any(x in col for x in ['youngs', 'CTE', 'poisson', 'shrinkage', 'creep'])]
process_features = [col for col in df.columns if any(x in col for x in ['ramp', 'temp', 'time', 'pressure', 'humidity'])]
derived_features = [col for col in df.columns if any(x in col for x in ['mismatch', 'ratio', 'total_'])]

# Create feature matrix
X = df[geometric_features + material_features + process_features + derived_features]

# One-hot encode atmosphere
X = pd.get_dummies(X, columns=['atmosphere_type'], prefix='atm')

print(f"Feature matrix shape: {X.shape}")
```

### Predictive Modeling Setup

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have a target variable (e.g., from FEA simulations)
# y = df['max_residual_stress_MPa']  # You would add this from FEA results

# Select input features
input_features = geometric_features + material_features + process_features + derived_features

X = df[input_features]
X = pd.get_dummies(X, columns=['atmosphere_type'])

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Ready for model training!")
```

## 🔗 Integration with FEA Results

This "Context Dataset" is designed to be **paired** with FEA simulation results:

```python
# Workflow:
# 1. Use this dataset as DOE inputs
# 2. Run FEA simulations for each sample_id
# 3. Extract stress/strain/warpage outputs
# 4. Merge with context data

# Example merge:
fea_results = pd.read_csv('fea_stress_results.csv')  # Your FEA outputs
full_dataset = df.merge(fea_results, on='sample_id')

# Now you have: X (context) -> y (stress)
X = full_dataset[input_features]
y = full_dataset['max_residual_stress_MPa']
```

## 🎓 Physical Insights

### Primary Residual Stress Drivers

1. **CTE Mismatch** (Most Critical)
   - Differential thermal contraction during cooling
   - Electrolyte-anode interface typically most stressed
   - ΔT × ΔCTE × E = stress

2. **Stiffness Ratios**
   - Stiffer layer (electrolyte) carries more load
   - Determines stress distribution

3. **Thickness Ratios**
   - Controls bending vs. membrane stress
   - Asymmetry causes warpage

4. **Cooling Rate**
   - Fast cooling → less stress relaxation
   - Creep at high T can reduce stress

5. **Sintering Sequence**
   - Layer-specific onset temperatures
   - Differential shrinkage generates stress

## 📊 Design of Experiments (DOE)

**Sampling Method:** Latin Hypercube Sampling (LHS)
- Space-filling design for efficient coverage
- Better than random sampling for metamodeling
- Ensures all parameter ranges are well-represented

**Advantages:**
- Uniform coverage of parameter space
- Efficient for high-dimensional problems
- Suitable for surrogate modeling

## ⚠️ Important Notes

1. **This is a synthetic dataset** based on realistic material property ranges from literature
2. **No actual FEA stress results** are included - you must generate these
3. **Temperature-dependent properties** are provided at 2 points (25°C and 1000°C) - interpolate as needed
4. **Material combinations** are randomized - some may be unrealistic (e.g., use quality flags)

## 🔧 Recommended Next Steps

1. **Run FEA Simulations**
   - Use ABAQUS, ANSYS, or COMSOL
   - Implement sintering models (shrinkage, creep)
   - Extract stress/strain/warpage for each `sample_id`

2. **Train ML Models**
   - Regression: Predict max stress, warpage
   - Classification: Predict failure/success
   - Multi-output: Predict stress field at multiple points

3. **Perform Sensitivity Analysis**
   - Which parameters most influence stress?
   - Interaction effects between CTE and thickness?

4. **Optimization**
   - Use trained model for design optimization
   - Minimize stress while meeting performance requirements

## 📚 References

Material property ranges based on:
- Ni-YSZ anodes: Typical SOFC cermet properties
- YSZ electrolytes: 8-YSZ (8 mol% Y₂O₃-stabilized ZrO₂)
- LSM/LSCF cathodes: Standard perovskite cathodes
- Sintering: Co-firing temperature profiles for SOFCs

## 📧 Dataset Information

- **Generated by:** Context Dataset Generator v1.0
- **Date:** 2025-10-15
- **Sampling:** Latin Hypercube (scipy.stats.qmc)
- **Seed:** 42 (reproducible)

## 📄 License

This dataset is provided for research and educational purposes. Please cite appropriately if used in publications.

---

**Ready to predict, not just describe! 🚀**
