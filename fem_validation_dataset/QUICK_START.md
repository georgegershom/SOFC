# Quick Start Guide

## Get Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Example Analysis

```bash
python example_analysis.py
```

This will generate 5 visualization files:
- `surface_stress_visualization.png` - Experimental stress field
- `fem_vs_experimental.png` - Model validation comparison
- `surrogate_model_performance.png` - GP surrogate model results
- `crack_analysis.png` - Crack initiation patterns
- `residual_analysis.png` - Stress by location type

### 3. Load Your First Dataset

```python
import pandas as pd

# Experimental ground truth
xrd = pd.read_csv('experimental_data/residual_stress/xrd_surface_residual_stress.csv')
print(xrd.head())

# FEM simulation output
fem = pd.read_csv('simulation_output/full_field/fem_full_field_solution.csv')
print(fem.head())

# Strategic collocation points
colloc = pd.read_csv('simulation_output/collocation_points/collocation_point_data.csv')
print(colloc.head())
```

## Common Use Cases

### Validate Your FEM Model

```python
# Load experimental measurements
exp_stress = xrd['sigma_xx_MPa'].values

# Load your FEM predictions at same locations
fem_stress = fem['stress_xx_MPa'].values

# Calculate error
error = np.abs(fem_stress - exp_stress)
relative_error = error / exp_stress

print(f"Mean absolute error: {error.mean():.2f} MPa")
print(f"Mean relative error: {100*relative_error.mean():.1f}%")
```

### Train a Surrogate Model

```python
from sklearn.gaussian_process import GaussianProcessRegressor

# Features: coordinates
X = colloc[['x_coord_mm', 'y_coord_mm', 'z_coord_mm']].values

# Target: stress
y = colloc['von_mises_MPa'].values

# Train
gp = GaussianProcessRegressor()
gp.fit(X, y)

# Predict at new locations
X_new = [[2.5, 2.5, 0.1]]  # mm
y_pred = gp.predict(X_new)
print(f"Predicted stress: {y_pred[0]:.1f} MPa")
```

### Analyze Crack Locations

```python
cracks = pd.read_csv('experimental_data/crack_analysis/crack_initiation_data.csv')

# Find stress threshold for cracking
import matplotlib.pyplot as plt

plt.scatter(cracks['x_position_mm'], cracks['y_position_mm'], 
            s=cracks['crack_length_um']*5, 
            c=cracks['stress_intensity_factor_MPa_sqrt_m'],
            cmap='hot')
plt.colorbar(label='K_I (MPa√m)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('Crack Locations and Intensity')
plt.show()
```

## Key Files

| File | What It Contains |
|------|------------------|
| `xrd_surface_residual_stress.csv` | Surface stress measurements (ground truth) |
| `fem_full_field_solution.csv` | FEM predictions (T, U, σ, ε) |
| `collocation_point_data.csv` | Strategic subset for surrogate models |
| `crack_initiation_data.csv` | Where and when cracks formed |
| `microstructure_characterization.csv` | Grain size, porosity, etc. |

## Data Structure

```
x_position_mm, y_position_mm, z_position_mm  → Location
sigma_xx_MPa, sigma_yy_MPa, sigma_zz_MPa    → Stress tensor
von_mises_MPa                                 → Equivalent stress
temperature_K                                 → Temperature field
displacement_x/y/z_um                        → Displacement field
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore [example_analysis.py](example_analysis.py) for complete workflows
3. Check [metadata/dataset_summary.json](metadata/dataset_summary.json) for statistics

## Need Help?

- **Full Documentation**: See [README.md](README.md)
- **Data Format Questions**: See comments in CSV headers
- **Analysis Examples**: Run `python example_analysis.py`
- **Units & Conventions**: See README.md "Units" section

## Citation

If you use this dataset in your research:

```
@dataset{fem_validation_sofc_2025,
  title={Multi-Scale FEM Validation Dataset for SOFC Electrolytes},
  year={2025},
  version={1.0.0}
}
```

---

**Pro Tip**: Start with the collocation points file - it has all the key information in a manageable size (30 points) before working with the full-field data (100K+ points in real applications).