# Quick Start Guide: In-The-Wild SOFC Plate Dataset

## 🎯 Overview

This dataset contains **300 SOFC (Solid Oxide Fuel Cell) plate measurements** collected from a simulated production environment with realistic operational conditions, including:

- ✅ **Production variations** (temperature drift, material batches, furnace aging)
- ✅ **Measurement noise** (sensor drift, outliers, missing data)
- ✅ **Failure modes** (edge cracking, delamination, thermal shock)
- ✅ **Multiple stress patterns** (biaxial, gradient, edge-dominated, localized, mixed)
- ✅ **Complete metadata** (batch info, process parameters, quality labels)

**Dataset size**: 21 MB  
**Time span**: 2023-01-01 to 2023-12-10 (50 production batches)  
**Quality distribution**: 99 good, 87 marginal, 106 reject, 8 failed

---

## 📁 Dataset Structure

```
in_the_wild_dataset/
├── dataset_metadata.csv          # Master metadata file (300 plates)
├── dataset_summary.png            # Visual summary of dataset statistics
├── README.md                      # Comprehensive documentation
├── measurements/                  # Warp measurement data
│   ├── plate_00001_warp.csv      # CSV: x, y, z coordinates
│   ├── plate_00002_warp.csv
│   └── ... (300 files)
├── stress_fields/                 # Ground truth stress fields
│   ├── plate_00001_stress.npz    # NPZ: stress tensors + warp data
│   ├── plate_00002_stress.npz
│   └── ... (300 files)
└── visualizations/                # Sample plate visualizations
    ├── plate_00001.png           # Warp + stress field plots
    └── ... (18 files)
```

---

## 🚀 Quick Start Examples

### 1. Load and Explore Metadata

```python
import pandas as pd
import numpy as np

# Load metadata
metadata = pd.read_csv('in_the_wild_dataset/dataset_metadata.csv')

# Basic statistics
print(f"Total plates: {len(metadata)}")
print(f"Quality distribution:\n{metadata['quality_class'].value_counts()}")
print(f"Failure rate: {metadata['failed'].sum() / len(metadata) * 100:.1f}%")

# View first few records
print(metadata.head())
```

### 2. Load a Single Plate

```python
# Select a plate
plate = metadata.iloc[42]  # Example: plate 43

# Load warp measurements (surface profilometry data)
warp_df = pd.read_csv(f"in_the_wild_dataset/{plate['measurement_file']}")
print(warp_df.head())
# Columns: x_mm, y_mm, z_mm (may contain NaN for missing data)

# Load ground truth stress fields
stress_data = np.load(f"in_the_wild_dataset/{plate['stress_file']}")
stress_xx = stress_data['stress_xx']  # Stress in X direction (MPa)
stress_yy = stress_data['stress_yy']  # Stress in Y direction (MPa)
warp_true = stress_data['warp_true']  # True warp without noise (mm)
warp_measured = stress_data['warp_measured']  # Measured warp with noise (mm)
X = stress_data['X']  # X coordinate grid
Y = stress_data['Y']  # Y coordinate grid

print(f"Stress range: {stress_xx.min():.1f} to {stress_xx.max():.1f} MPa")
print(f"Warp range: {np.nanmin(warp_measured):.3f} to {np.nanmax(warp_measured):.3f} mm")
```

### 3. Filter by Quality and Conditions

```python
# Get only good quality plates
good_plates = metadata[metadata['quality_class'] == 'good']
print(f"Good plates: {len(good_plates)}")

# Get failed plates for failure analysis
failed_plates = metadata[metadata['failed'] == True]
print(f"\nFailed plates: {len(failed_plates)}")
print(failed_plates[['plate_id', 'failure_type', 'failure_location', 'max_stress_MPa']])

# Filter by process parameters
high_temp_plates = metadata[metadata['furnace_temp_C'] > 1400]
print(f"\nHigh temperature plates: {len(high_temp_plates)}")

# Filter by batch
batch_1 = metadata[metadata['batch_id'] == 'BATCH-0001']
print(f"\nBatch 1 plates: {len(batch_1)}")
```

### 4. Analyze Parameter Drift Over Time

```python
import matplotlib.pyplot as plt

# Convert date to datetime
metadata['date'] = pd.to_datetime(metadata['date'])

# Plot furnace temperature drift
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Temperature drift
axes[0, 0].scatter(metadata['furnace_age_days'], metadata['furnace_temp_C'], alpha=0.5)
axes[0, 0].set_xlabel('Furnace Age (days)')
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].set_title('Furnace Temperature Drift')

# Warp evolution
axes[0, 1].scatter(metadata['date'], metadata['max_warp_mm'], 
                   c=metadata['quality_class'].map({'good': 0, 'marginal': 1, 'reject': 2, 'failed': 3}),
                   cmap='RdYlGn_r', alpha=0.6)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Max Warp (mm)')
axes[0, 1].set_title('Warp Evolution Over Time')

# Stress vs Warp
axes[1, 0].scatter(metadata['max_stress_MPa'], metadata['max_warp_mm'], alpha=0.5)
axes[1, 0].set_xlabel('Max Stress (MPa)')
axes[1, 0].set_ylabel('Max Warp (mm)')
axes[1, 0].set_title('Stress-Warp Relationship')

# Quality by shift
shift_quality = pd.crosstab(metadata['shift'], metadata['quality_class'])
shift_quality.plot(kind='bar', stacked=True, ax=axes[1, 1])
axes[1, 1].set_xlabel('Shift')
axes[1, 1].set_title('Quality by Production Shift')

plt.tight_layout()
plt.show()
```

### 5. Visualize a Plate

```python
import matplotlib.pyplot as plt

# Load plate data
plate_id = 0
plate = metadata.iloc[plate_id]
stress_data = np.load(f"in_the_wild_dataset/{plate['stress_file']}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Measured warp
im1 = axes[0].contourf(stress_data['X'], stress_data['Y'], 
                       stress_data['warp_measured'], levels=20, cmap='RdYlBu_r')
axes[0].set_title(f"Measured Warp - {plate['plate_id']}")
axes[0].set_xlabel('X (mm)')
axes[0].set_ylabel('Y (mm)')
plt.colorbar(im1, ax=axes[0], label='Warp (mm)')

# Stress XX
im2 = axes[1].contourf(stress_data['X'], stress_data['Y'], 
                       stress_data['stress_xx'], levels=20, cmap='plasma')
axes[1].set_title('Residual Stress σ_xx')
axes[1].set_xlabel('X (mm)')
axes[1].set_ylabel('Y (mm)')
plt.colorbar(im2, ax=axes[1], label='Stress (MPa)')

# Stress YY
im3 = axes[2].contourf(stress_data['X'], stress_data['Y'], 
                       stress_data['stress_yy'], levels=20, cmap='plasma')
axes[2].set_title('Residual Stress σ_yy')
axes[2].set_xlabel('X (mm)')
axes[2].set_ylabel('Y (mm)')
plt.colorbar(im3, ax=axes[2], label='Stress (MPa)')

plt.tight_layout()
plt.show()

# Print plate info
print(f"\nPlate: {plate['plate_id']}")
print(f"Quality: {plate['quality_class']}")
print(f"Stress pattern: {plate['stress_pattern']}")
print(f"Max warp: {plate['max_warp_mm']:.3f} mm")
print(f"Max stress: {plate['max_stress_MPa']:.1f} MPa")
print(f"Furnace temp: {plate['furnace_temp_C']:.1f} °C")
print(f"Failed: {plate['failed']}")
```

---

## 🎓 Use Cases for ML Research

### 1. **Inverse Problem: Stress Reconstruction from Warp**
Train a model to predict `stress_xx` and `stress_yy` from `warp_measured`:

```python
# Example: Train/test split
from sklearn.model_selection import train_test_split

train_meta, test_meta = train_test_split(metadata, test_size=0.2, random_state=42)

# Load training data
for idx in train_meta.index[:10]:  # Example: first 10 plates
    plate = train_meta.loc[idx]
    warp_df = pd.read_csv(f"in_the_wild_dataset/{plate['measurement_file']}")
    stress_data = np.load(f"in_the_wild_dataset/{plate['stress_file']}")
    
    # Your ML model here
    # X = warp_df[['x_mm', 'y_mm', 'z_mm']].values
    # y = stress_data['stress_xx'], stress_data['stress_yy']
```

### 2. **Quality Classification**
Predict quality class from measurements and process parameters:

```python
features = ['max_warp_mm', 'mean_warp_mm', 'warp_std_mm', 
            'furnace_temp_C', 'cooling_rate_C_per_min', 
            'furnace_age_days']
X = metadata[features]
y = metadata['quality_class']

# Train classifier (e.g., Random Forest, Neural Network)
```

### 3. **Failure Prediction**
Predict failure probability from process parameters:

```python
metadata['failure_risk'] = (metadata['max_stress_MPa'] > 100).astype(int)
# Use process parameters to predict failure risk
```

### 4. **Domain Adaptation / Transfer Learning**
Train on early production data, test on later data with drift:

```python
early_data = metadata[metadata['furnace_age_days'] < 150]
late_data = metadata[metadata['furnace_age_days'] >= 150]
# Test model robustness to parameter drift
```

### 5. **Robustness to Noise**
Train on clean data, test on noisy measurements:

```python
# Compare warp_true vs warp_measured predictions
# Test model performance with missing data (NaN values)
```

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total plates | 300 |
| Production batches | 50 |
| Material batches | 10 |
| Operators | 5 |
| Date range | 2023-01-01 to 2023-12-10 |
| Furnace age range | 0-341 days |
| Mean warp | 1.013 ± 0.316 mm |
| Max warp (overall) | 5.286 mm |
| Mean max stress | 89.1 ± 17.9 MPa |
| Max stress (overall) | 154.5 MPa |
| Failure rate | 2.7% |
| Quality: Good | 33.0% |
| Quality: Marginal | 29.0% |
| Quality: Reject | 35.3% |
| Quality: Failed | 2.7% |

### Stress Patterns
- Biaxial: 150 plates (50%)
- Gradient: 80 plates (26.7%)
- Edge-dominated: 40 plates (13.3%)
- Localized: 20 plates (6.7%)
- Mixed: 10 plates (3.3%)

### Failure Modes
- No failure: 292 plates (97.3%)
- Edge cracking: 2 plates (0.7%)
- Delamination: 6 plates (2.0%)
- Thermal shock: 0 plates (0%)

---

## 🔬 Physical Validity Checks

When validating ML model predictions, ensure:

1. **Stress magnitude**: Typical range 20-150 MPa for SOFC ceramics
2. **Warp magnitude**: Typical range 0-5 mm for 150mm plates
3. **Edge effects**: Higher stress near edges due to boundary conditions
4. **Smoothness**: Stress fields should be continuous (except at cracks)
5. **Energy consistency**: Total strain energy should be physically plausible

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 2.0
- pandas >= 2.0
- scipy >= 1.10
- matplotlib >= 3.7
- seaborn >= 0.12

---

## 📚 Additional Resources

- **Comprehensive documentation**: See `in_the_wild_dataset/README.md`
- **Visual summary**: Open `in_the_wild_dataset/dataset_summary.png`
- **Sample visualizations**: Check `in_the_wild_dataset/visualizations/`

---

## 🎯 Key Features

✅ **Production-realistic**: Captures actual manufacturing variations  
✅ **Parameter drift**: Furnace aging, material batch changes  
✅ **Measurement noise**: Sensor artifacts, outliers, missing data  
✅ **Failure modes**: Edge cracks, delamination, thermal shock  
✅ **Complete metadata**: Full traceability of process conditions  
✅ **Ground truth**: True stress fields available for validation  
✅ **Multiple patterns**: 5 different stress distribution types  
✅ **Quality labels**: Pre-classified for supervised learning  

---

## 🤝 Citation

If you use this dataset in your research, please cite:

```
In-The-Wild SOFC Plate Dataset for ML-Augmented Inverse Modeling
Generated: 2025-10-16
Purpose: Residual Stress Quantification from Warped SOFC Plates
Source: Physics-based simulation with realistic production variations
```

---

**Ready to test your ML model's robustness? Start exploring!** 🚀
