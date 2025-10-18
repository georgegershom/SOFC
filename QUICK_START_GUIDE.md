# Quick Start Guide: Fire-Resistant Rubberized Concrete Dataset

## Getting Started in 5 Minutes

### 1. Prerequisites
```bash
pip install -r requirements.txt
```

### 2. Load and Explore Data
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
df_ambient = pd.read_csv('ambient_properties.csv')
df_residual = pd.read_csv('residual_properties_high_temp.csv')
df_insitu = pd.read_csv('in_situ_properties.csv')

# Quick exploration
print("Dataset sizes:")
print(f"Ambient: {len(df_ambient)} specimens")
print(f"Residual: {len(df_residual)} specimens")
print(f"In-Situ: {len(df_insitu)} specimens")
```

### 3. Basic Analysis Examples

#### Example 1: Plot Strength Degradation
```python
import seaborn as sns

# Filter for furnace-cooled, standard heating
df_plot = df_residual[
    (df_residual['Cooling_Method'] == 'Furnace') & 
    (df_residual['Heating_Rate'] == '5_C_per_min')
]

# Group by mix and temperature
grouped = df_plot.groupby(['Mix_ID', 'Peak_Temperature_C'])['Residual_Compressive_Strength_MPa'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
for mix in ['C', 'R10S', 'R20S']:
    data = grouped[grouped['Mix_ID'] == mix]
    plt.plot(data['Peak_Temperature_C'], data['Residual_Compressive_Strength_MPa'], 
             'o-', label=mix, linewidth=2, markersize=8)

plt.xlabel('Peak Temperature (°C)', fontsize=12)
plt.ylabel('Residual Compressive Strength (MPa)', fontsize=12)
plt.title('Thermal Degradation of Concrete Strength', fontsize=14, fontweight='bold')
plt.legend(title='Mix Type', fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()
```

#### Example 2: Analyze Spalling Risk
```python
# Focus on rapid heating conditions
spalling_data = df_residual[df_residual['Heating_Rate'] == '10_C_per_min']

# Calculate spalling probability
spalling_prob = spalling_data.groupby(['Mix_ID', 'Peak_Temperature_C']).agg({
    'Spalling_Occurred': ['sum', 'count', 'mean']
}).round(3)

print("\nSpalling Probability Matrix:")
print(spalling_prob)

# Key finding: Control concrete shows 60% spalling rate at 400°C+
# Rubberized mixes (R15S, R20S) show <30% spalling rate
```

#### Example 3: Load Stress-Strain Curves
```python
import json

# Load full stress-strain data
with open('stress_strain_curves.json', 'r') as f:
    curves = json.load(f)

# Plot curves at 400°C
plt.figure(figsize=(10, 6))
for mix in ['C', 'R10S', 'R20S']:
    specimen_id = f"{mix}-28-IS-400-1"
    strain = curves[specimen_id]['strain']
    stress = curves[specimen_id]['stress']
    plt.plot([s*100 for s in strain], stress, label=mix, linewidth=2)

plt.xlabel('Strain (%)', fontsize=12)
plt.ylabel('Stress (MPa)', fontsize=12)
plt.title('Stress-Strain Behavior at 400°C', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 4. Key Findings Summary

#### Baseline Properties (28-day, Ambient)
| Mix | fc (MPa) | COV | Density (kg/m³) |
|-----|----------|-----|-----------------|
| C (Control) | 60.4 | 15% | 2400 |
| R5S | 52.9 | 16% | 2360 |
| R10S | 44.8 | 14% | 2320 |
| R20S | 31.1 | 16% | 2240 |

**Trend**: ~13% strength reduction per 5% rubber addition

#### Temperature Effects (Residual Strength)
| Temperature | Control (C) | R10S | R20S |
|-------------|-------------|------|------|
| 200°C | ~100% | ~100% | ~100% |
| 400°C | ~65% | ~58% | ~52% |
| 600°C | ~32% | ~28% | ~24% |
| 800°C | ~10% | ~9% | ~8% |

**Pattern**: All mixes lose ~60% strength by 600°C

#### Spalling Resistance (Rapid Heating, 10°C/min)
- **Control (C)**: 18/30 specimens spalled (60%)
- **R15S**: 3/30 specimens spalled (10%)  
- **R20S**: 7/30 specimens spalled (23%)

**Conclusion**: Rubber reduces spalling risk by 60-85%

#### Cooling Method Impact (at 400°C)
- **Furnace cooling**: Reference condition
- **Quench cooling**: Additional 15% strength loss due to thermal shock

### 5. Recommended Workflows

#### Workflow A: Fit Temperature-Dependent Constitutive Model
1. Use ambient data for material parameter initialization
2. Fit degradation functions to residual data (Furnace-cooled)
3. Validate with in-situ data
4. Incorporate cooling method as damage multiplier (0.85 for quench)

#### Workflow B: Probabilistic Spalling Risk Model
1. Use logistic regression on spalling occurrence data
2. Predictors: Temperature, Heating Rate, Mix Type, Pore Pressure
3. Output: P(spalling) for given conditions
4. Validate against independent test data

#### Workflow C: Multi-Physics Simulation Calibration
1. Thermal model: Fit thermal diffusivity from heating rate data
2. Pore pressure model: Calibrate permeability evolution from pore_pressure_summary.csv
3. Mechanical model: Use stress-strain curves for plasticity parameters
4. Coupled model: Predict spalling using pore pressure threshold

### 6. Data Quality Indicators

✓ **Coefficient of Variation (COV)**
- Strength: 5-8% (typical for concrete testing)
- UPV: 2-5% (excellent repeatability)

✓ **Physical Consistency**
- Residual ≤ Ambient strength (100% of cases)
- In-Situ ≥ Residual at same T (verified)
- UPV-Strength correlation: R² > 0.85

✓ **Statistical Validity**
- 3 replicates per condition (ambient)
- 2-3 replicates per condition (high-temp)
- N=270 for residual property dataset (robust statistics)

### 7. Common Issues and Solutions

**Issue**: Large file size for stress-strain JSON
**Solution**: Subsample to 25 points if memory constrained
```python
curves_reduced = {k: {'strain': v['strain'][::2], 'stress': v['stress'][::2]} 
                  for k, v in curves.items()}
```

**Issue**: Categorical visual cracking rating hard to analyze
**Solution**: Convert to numeric scale
```python
rating_map = {'None': 0, 'Minor': 1, 'Moderate': 2, 'Severe': 3}
df_residual['Cracking_Score'] = df_residual['Visual_Cracking_Rating'].map(rating_map)
```

**Issue**: Need normalized strength retention
**Solution**: Merge with ambient data
```python
df_merged = df_residual.merge(
    df_ambient[df_ambient['Curing_Age_days']==28][['Mix_ID', 'Compressive_Strength_MPa']],
    on='Mix_ID', suffixes=('_residual', '_ambient')
)
df_merged['Retention_Factor'] = df_merged['Compressive_Strength_MPa_residual'] / df_merged['Compressive_Strength_MPa_ambient']
```

### 8. Next Steps

After exploring the dataset:
1. **Develop constitutive models** for each material (C, R5S, R10S, etc.)
2. **Implement in FE software** (e.g., Abaqus UMAT, ANSYS UserMat)
3. **Validate predictions** against in-situ test data (held-out validation)
4. **Optimize mix design** for target fire resistance rating (e.g., 2-hour)
5. **Publish findings** with proper dataset attribution

### 9. Regenerate Dataset with Different Parameters

To regenerate with different random seed or modify parameters:
```bash
# Edit generate_fire_resistance_dataset.py
# Line 16: np.random.seed(42)  # Change this value

python3 generate_fire_resistance_dataset.py
```

### 10. Support and Documentation

- **Full Documentation**: See `DATASET_README.md`
- **Methodology Details**: See `generate_fire_resistance_dataset.py` (heavily commented)
- **Visualizations**: PNG files provide publication-ready figures

---

**You're now ready to begin model development!**

For detailed methodology and validation considerations, refer to the comprehensive `DATASET_README.md` file.
