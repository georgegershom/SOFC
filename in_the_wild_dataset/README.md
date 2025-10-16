
# In-The-Wild SOFC Plate Dataset Documentation

## Dataset Overview

This dataset contains 300 SOFC (Solid Oxide Fuel Cell) plate measurements 
collected from simulated production operations spanning from 2023-01-01 00:00:00 
to 2023-12-10 00:00:00.

**Purpose**: Demonstrate ML model robustness on real-world operational data with natural 
variations, measurement noise, and parameter drift.

## Dataset Characteristics

### Physical Properties
- Plate dimensions: 150 mm × 150 mm
- Nominal thickness: 0.5 mm
- Material: Ceramic (typical SOFC materials: YSZ, LSM, NiO-YSZ)
- Young's Modulus: 200 GPa
- Poisson's Ratio: 0.3
- CTE: 10.5 × 10⁻⁶ K⁻¹

### Measurement Grid
- Resolution: 25 × 25 points
- Measurement type: Surface profilometry (warp/deflection)
- Coordinate system: Cartesian (X, Y, Z) where Z is out-of-plane deflection

## Production Process Simulation

### Nominal Process Parameters
- Sintering temperature: 1400°C
- Cooling rate: 2.0 °C/min
- Oxygen partial pressure: 0.21 atm
- Ambient humidity: 30-60%

### Sources of Variation

1. **Furnace Aging Effect**
   - Temperature drift rate: -0.001 °C/day
   - Maximum age in dataset: 343 days
   - Total temperature drift: ~0.3 °C

2. **Material Batch Variations**
   - Number of material batches: 10
   - Powder size variations: 0.8-1.2 μm
   - Purity range: 99.5-99.9%
   - Green density range: 0.55-0.65

3. **Process Variations**
   - Temperature variations: ±5°C (random)
   - Cooling rate variations: ±0.3 °C/min
   - Furnace position effects: 5 zones (front, back, left, right, center)
   - Shift effects: 3 shifts (morning, afternoon, night)

4. **Measurement Noise**
   - Gaussian noise: σ = 0.02 mm (baseline)
   - Sensor drift: 15% of measurements affected
   - Outliers: 20% of measurements contain outliers (0.5% of points)
   - Missing data: 10% of measurements have missing points (2% of grid)

## Stress Patterns

The dataset includes 5 types of residual stress patterns:

1. **Biaxial** (150 plates): 
   Symmetric stress from uniform cooling
   
2. **Gradient** (80 plates): 
   Thermal gradients during cooling
   
3. **Edge-Dominated** (40 plates): 
   High stress near edges (common failure precursor)
   
4. **Localized** (20 plates): 
   Localized stress concentrations
   
5. **Mixed** (10 plates): 
   Complex multi-mode patterns

## Failure Modes

The dataset includes realistic failure modes observed in SOFC production:

1. **Edge Cracking** (2 cases):
   - Occurs when edge stress exceeds 100 MPa
   - Creates local stress relief and warp discontinuities
   - Most common failure mode

2. **Delamination** (6 cases):
   - Occurs in multilayer structures under high stress
   - Creates localized bubble-like deformations
   - Serious structural failure

3. **Thermal Shock** (0 cases):
   - Results from rapid cooling
   - Distributed microcracks throughout structure
   - Irregular surface features

4. **No Failure** (292 plates):
   - Normal production within acceptable limits

## Quality Classification

Plates are classified into 4 quality categories:

- **Good** (99 plates): 
  Max warp < 3.0 mm, max stress < 90 MPa, no failures
  
- **Marginal** (87 plates): 
  Max warp 3.0-4.0 mm or max stress 90-110 MPa
  
- **Reject** (106 plates): 
  Max warp > 4.0 mm or max stress > 110 MPa, but not failed
  
- **Failed** (8 plates): 
  Physical failure (cracks, delamination, etc.)

## File Structure

```
in_the_wild_dataset/
├── dataset_metadata.csv          # Complete metadata for all plates
├── dataset_summary.png            # Summary visualizations
├── README.md                      # This file
├── measurements/                  # Warp measurement data (CSV)
│   ├── plate_00001_warp.csv
│   ├── plate_00002_warp.csv
│   └── ...
├── stress_fields/                 # Ground truth stress fields (NPZ)
│   ├── plate_00001_stress.npz
│   ├── plate_00002_stress.npz
│   └── ...
└── visualizations/                # Sample visualizations
    ├── plate_00001.png
    ├── plate_00031.png
    └── ...
```

## Data Files

### 1. Measurement Files (CSV)
Each `plate_XXXXX_warp.csv` contains:
- `x_mm`: X coordinate (mm)
- `y_mm`: Y coordinate (mm)
- `z_mm`: Z deflection/warp (mm) [may contain NaN for missing data]

### 2. Stress Field Files (NPZ)
Each `plate_XXXXX_stress.npz` contains:
- `stress_xx`: Residual stress in X direction (MPa)
- `stress_yy`: Residual stress in Y direction (MPa)
- `warp_true`: True warp field without noise (mm)
- `warp_measured`: Measured warp with noise (mm)
- `X`: X coordinate meshgrid (mm)
- `Y`: Y coordinate meshgrid (mm)

### 3. Metadata File (CSV)
`dataset_metadata.csv` contains comprehensive information:
- Plate identification (plate_id, batch_id, date)
- Material tracking (material_batch)
- Process parameters (furnace_temp_C, cooling_rate_C_per_min, etc.)
- Production context (furnace_position, furnace_age_days, shift, operator)
- Stress characteristics (stress_pattern, max_stress_MPa)
- Warp measurements (max_warp_mm, mean_warp_mm, warp_std_mm)
- Quality assessment (quality_class, failed, failure_type, failure_location)
- File references (measurement_file, stress_file)

## Usage Examples

### Loading a Single Plate

```python
import pandas as pd
import numpy as np

# Load metadata
metadata = pd.read_csv('in_the_wild_dataset/dataset_metadata.csv')

# Select a plate
plate_info = metadata.iloc[0]

# Load warp measurements
warp_data = pd.read_csv(f"in_the_wild_dataset/{plate_info['measurement_file']}")

# Load ground truth stress
stress_data = np.load(f"in_the_wild_dataset/{plate_info['stress_file']}")
stress_xx = stress_data['stress_xx']
stress_yy = stress_data['stress_yy']
warp_true = stress_data['warp_true']
```

### Filtering by Quality

```python
# Get only good quality plates
good_plates = metadata[metadata['quality_class'] == 'good']

# Get failed plates  
failed_plates = metadata[metadata['failed'] == True]

# Get plates from specific batch
batch_plates = metadata[metadata['batch_id'] == 'BATCH-0001']
```

### Analyzing Parameter Drift

```python
import matplotlib.pyplot as plt

# Plot furnace temperature drift
plt.figure(figsize=(10, 6))
plt.scatter(metadata['furnace_age_days'], metadata['furnace_temp_C'])
plt.xlabel('Furnace Age (days)')
plt.ylabel('Temperature (°C)')
plt.title('Furnace Temperature Drift Over Time')
plt.show()

# Analyze correlation with quality
print(metadata.groupby('quality_class')['furnace_temp_C'].describe())
```

## Statistical Summary

### Warp Statistics
- Mean warp (across all plates): 1.013 ± 0.316 mm
- Maximum warp observed: 5.286 mm
- Minimum warp observed: 1.348 mm

### Stress Statistics
- Mean maximum stress: 89.1 ± 17.9 MPa
- Maximum stress observed: 154.5 MPa
- Minimum stress observed: 75.7 MPa

### Production Statistics
- Total production batches: 50
- Material batches used: 10
- Operators involved: 5
- Production span: 343 days

## Use Cases for ML Model Testing

### 1. Robustness to Noise
Test model performance on noisy measurements with outliers and missing data.

### 2. Domain Adaptation
Train on early production data, test on later data with parameter drift.

### 3. Failure Prediction
Predict failure modes from warp measurements and process parameters.

### 4. Quality Classification
Classify plates into quality categories based on measurements.

### 5. Stress Reconstruction
Inverse problem: reconstruct stress fields from warp measurements.

### 6. Process Optimization
Identify optimal process parameters for minimal warp/stress.

### 7. Anomaly Detection
Detect unusual patterns that deviate from normal production.

## Physical Validity Checks

When validating ML predictions, ensure:

1. **Stress Magnitude**: Typical range 20-150 MPa for SOFC ceramics
2. **Warp Magnitude**: Typical range 0-5 mm for 150mm plates
3. **Edge Effects**: Higher stress near edges (boundary conditions)
4. **Symmetry**: Patterns should respect plate symmetry unless there's asymmetric loading
5. **Smoothness**: Stress fields should be continuous except at crack locations
6. **Energy Consistency**: Total strain energy should be physically reasonable

## Known Limitations

1. This is simulated data based on physics models, not actual experimental measurements
2. Some simplifications in plate theory (Kirchhoff-Love assumptions)
3. Failure modes are idealized representations
4. Material properties assumed homogeneous except for batch variations
5. No microstructure effects or grain boundary effects included

## Citation

If you use this dataset, please cite:
```
In-The-Wild SOFC Plate Dataset for ML-Augmented Inverse Modeling
Generated: 2025-10-16
Purpose: Residual Stress Quantification from Warped SOFC Plates
```

## Contact & Support

This dataset was generated for research purposes in ML-augmented inverse modeling
for residual stress quantification in SOFC manufacturing.

---
Generated: 2025-10-16 00:37:37
Dataset Version: 1.0
