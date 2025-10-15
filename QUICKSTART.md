# SOFC Dataset Quick Start Guide

## Installation & Setup

### 1. Install Dependencies

```bash
pip install numpy scipy matplotlib scikit-learn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
pip install scikit-learn  # Optional, for ML examples
```

### 2. Generate Dataset

Generate a dataset with 1000 samples (recommended for ML):

```bash
python3 sofc_dataset_generator.py --n-samples 1000 --output-dir sofc_dataset
```

For a quick test with 100 samples:

```bash
python3 sofc_dataset_generator.py --n-samples 100 --output-dir sofc_dataset
```

**Expected output:**
- Dataset size: ~440 MB for 1000 samples, ~44 MB for 100 samples
- Generation time: ~30 seconds for 100 samples, ~5 minutes for 1000 samples

### 3. Visualize Data

Create visualization summary:

```bash
python3 visualize_dataset.py --dataset-dir sofc_dataset --summary --n-samples 5
```

This creates visualizations in the `visualizations/` directory:
- Parameter distribution plots
- Warp field visualizations (top, bottom, mean)
- Stress field contour plots (σ_xx, σ_yy, σ_xy, von Mises)
- Through-thickness stress profiles

### 4. Load Data for ML

```python
from data_loader import SOFCDataLoader

# Initialize loader
loader = SOFCDataLoader(dataset_dir="sofc_dataset")

# Load all data
X, y = loader.load_all_samples(
    feature_type='warp',          # Input: warp field
    target_type='von_mises',      # Output: von Mises stress
    normalize_stress=True         # Normalize to [0, 1]
)

# Get train/test split
X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.2)

print(f"Training samples: {X_train.shape[0]}")
print(f"Feature dimensions: {X_train.shape[1]}")
print(f"Target dimensions: {y_train.shape[1]}")
```

## Quick Examples

### Example 1: Basic Data Exploration

```python
from data_loader import SOFCDataLoader
import matplotlib.pyplot as plt

# Load dataset
loader = SOFCDataLoader("sofc_dataset")

# Load a single sample
warp_data, stress_data, params = loader.load_sample(0)

# Visualize warp field
plt.figure(figsize=(10, 8))
plt.contourf(warp_data['X']*1000, warp_data['Y']*1000, 
             warp_data['warp_mean'], levels=20, cmap='viridis')
plt.colorbar(label='Warp (mm)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('SOFC Plate Warp Field')
plt.axis('equal')
plt.show()

# Print manufacturing parameters
print("Manufacturing Parameters:")
for key, value in params.items():
    print(f"  {key}: {value:.2f}")
```

### Example 2: Simple ML Model

```python
from data_loader import SOFCDataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Load data
loader = SOFCDataLoader("sofc_dataset")
X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.2)

# Dimensionality reduction
pca_X = PCA(n_components=50)
pca_y = PCA(n_components=20)

X_train_pca = pca_X.fit_transform(X_train)
X_test_pca = pca_X.transform(X_test)

y_train_pca = pca_y.fit_transform(y_train)
y_test_pca = pca_y.transform(y_test)

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train_pca, y_train_pca)

# Evaluate
y_pred_pca = model.predict(X_test_pca)
y_pred = pca_y.inverse_transform(y_pred_pca)

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
```

### Example 3: Parameter-to-Warp Prediction

```python
from data_loader import SOFCDataLoader
from sklearn.ensemble import RandomForestRegressor

# Load data with parameters as features
loader = SOFCDataLoader("sofc_dataset")

X, y = loader.load_all_samples(
    feature_type='params',  # Use manufacturing parameters
    target_type='von_mises'
)

# Note: y is still the full 3D stress field
# For this example, let's predict max warp from parameters
X_params = []
y_max_warp = []

for i in range(loader.n_samples):
    params = loader.get_parameter_features(i, normalize=True)
    warp = loader.get_warp_features(i, flatten=False)
    
    X_params.append(params)
    y_max_warp.append(np.abs(warp).max())

X_params = np.array(X_params)
y_max_warp = np.array(y_max_warp)

# Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_params, y_max_warp, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.4f}")

# Feature importance
import pandas as pd
param_names = [
    'peak_temp', 'heat_rate', 'cool_rate', 'dwell_time',
    'anode_thick', 'elec_thick', 'cath_thick',
    'ni_content', 'ysz_dopant', 'porosity',
    'green_dens', 'binder', 'length', 'width'
]
importance_df = pd.DataFrame({
    'parameter': param_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)
```

## Dataset Structure

```
sofc_dataset/
├── metadata/
│   ├── dataset_metadata.json       # Complete metadata
│   └── summary_statistics.json     # Statistical summary
├── parameters/
│   └── sample_XXXXX_params.json   # Manufacturing parameters (JSON)
├── warp_fields/
│   └── sample_XXXXX_warp.npz      # Warp field (NumPy compressed)
└── stress_fields/
    └── sample_XXXXX_stress.npz    # Stress field (NumPy compressed)
```

## Data Dimensions

- **Warp Field**: 2D array (50×50 by default)
  - Represents surface height map
  - Units: millimeters (mm)
  - Range: typically 0.1 - 200 mm

- **Stress Field**: 3D array (50×50×10 by default)
  - Represents full 3D stress tensor
  - Units: Pascals (Pa)
  - Components: σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz
  - Range: 1-1500 MPa (von Mises)

- **Parameters**: 14 scalar values
  - Normalized to [0, 1] for ML
  - Covers thermal, geometric, and material parameters

## Common Tasks

### Change Spatial Resolution

Generate dataset with higher resolution:

```bash
python3 sofc_dataset_generator.py --n-samples 500 --nx 100 --ny 100 --nz 20
```

**Warning**: Higher resolution = larger files and slower generation

### Extract Specific Stress Component

```python
loader = SOFCDataLoader("sofc_dataset")

# Get σ_xx component instead of von Mises
X, y = loader.load_all_samples(
    feature_type='warp',
    target_type='sigma_xx'  # Options: sigma_xx, sigma_yy, sigma_zz, etc.
)
```

### Combine Warp and Parameters as Features

```python
X, y = loader.load_all_samples(
    feature_type='both',  # Combines warp field + parameters
    target_type='von_mises'
)

print(f"Feature vector size: {X.shape[1]}")  
# = 2500 (warp) + 14 (params) = 2514
```

### Compute Dataset Statistics

```python
loader = SOFCDataLoader("sofc_dataset")
stats = loader.compute_statistics()

print(f"Max warp range: {stats['warp']['max']['min']:.2f} - "
      f"{stats['warp']['max']['max']:.2f} mm")
print(f"Max stress range: {stats['stress']['max']['min']:.2f} - "
      f"{stats['stress']['max']['max']:.2f} MPa")
```

## Troubleshooting

### Issue: Memory Error

**Solution**: Reduce spatial resolution or load data in batches

```python
# Load in batches
batch_size = 20
for i in range(0, loader.n_samples, batch_size):
    # Process batch
    batch_indices = range(i, min(i + batch_size, loader.n_samples))
    # ... your processing code
```

### Issue: Slow Data Loading

**Solution**: Use lower resolution or precompute features

```python
# Save preprocessed features
import pickle

X, y = loader.load_all_samples()
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)

# Load later
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    X, y = data['X'], data['y']
```

### Issue: Dataset Generation Too Slow

**Solution**: Generate smaller dataset first for prototyping

```bash
# Quick test dataset
python3 sofc_dataset_generator.py --n-samples 50 --nx 30 --ny 30 --nz 5

# Full dataset later
python3 sofc_dataset_generator.py --n-samples 1000
```

## Performance Benchmarks

**Dataset Generation:**
- 100 samples: ~30 seconds
- 1000 samples: ~5 minutes
- 5000 samples: ~25 minutes

**Data Loading (1000 samples):**
- Load all: ~10 seconds
- Load with PCA: ~15 seconds

**Memory Usage:**
- 100 samples: ~100 MB RAM
- 1000 samples: ~1 GB RAM
- 5000 samples: ~5 GB RAM

## Next Steps

1. **Explore the data**: Run visualizations to understand the dataset
2. **Train baseline model**: Use the example ML workflow
3. **Try advanced models**: CNNs, autoencoders, physics-informed NNs
4. **Read full documentation**: See `README_SOFC_Dataset.md` for details

## Getting Help

- Check `README_SOFC_Dataset.md` for full documentation
- Run `python3 data_loader.py` to see demo usage
- Run `python3 visualize_dataset.py --help` for visualization options

## Citation

```bibtex
@dataset{sofc_warp_stress_2025,
  title={Synthetic SOFC Warpage and Residual Stress Dataset},
  year={2025}
}
```
