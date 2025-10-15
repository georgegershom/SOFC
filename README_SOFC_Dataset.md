# SOFC Warpage and Residual Stress Dataset

## Overview

This repository contains a **synthetic dataset generator** for ML-augmented inverse modeling of residual stress quantification from warped Solid Oxide Fuel Cell (SOFC) plates.

The dataset provides the critical "Ground Truth" Core Dataset with paired examples of:
- **Warp Fields** (easy-to-measure): 3D surface deformations of SOFC plates
- **Residual Stress Fields** (hard-to-measure): Full 3D stress tensor distributions throughout the volume

## Dataset Description

### Source and Methodology

This dataset is generated using a **Virtual Design of Experiments (DOE)** approach that simulates what would be obtained from expensive high-fidelity Finite Element Analysis (FEA) coupled thermo-mechanical simulations.

#### Manufacturing Parameter Space (14 Parameters)

**Sintering Parameters:**
- Peak sintering temperature: 1200-1600°C
- Heating rate: 1-10°C/min
- Cooling rate: 1-10°C/min
- Dwell time: 1-8 hours

**Layer Thickness:**
- Anode thickness: 300-800 μm
- Electrolyte thickness: 5-30 μm
- Cathode thickness: 20-80 μm

**Material Composition:**
- Anode Ni content: 40-60 wt%
- Electrolyte YSZ dopant: 6-10 mol% Y₂O₃
- Cathode porosity: 20-45%

**Green Body Properties:**
- Green density: 50-65% theoretical
- Binder content: 1-5 wt%

**Geometry:**
- Plate length: 50-150 mm
- Plate width: 50-150 mm

### Data Format

The dataset is organized as follows:

```
sofc_dataset/
├── metadata/
│   ├── dataset_metadata.json       # Complete dataset metadata
│   └── summary_statistics.json     # Statistical summary
├── parameters/
│   ├── sample_00000_params.json    # Manufacturing parameters
│   ├── sample_00001_params.json
│   └── ...
├── warp_fields/
│   ├── sample_00000_warp.npz       # Warp field data
│   ├── sample_00001_warp.npz
│   └── ...
└── stress_fields/
    ├── sample_00000_stress.npz     # Stress field data
    ├── sample_00001_stress.npz
    └── ...
```

#### Warp Field Data (`*_warp.npz`)

Each file contains:
- `X`, `Y`: 2D meshgrid coordinates (m)
- `warp_top`: Top surface deformation (mm)
- `warp_bottom`: Bottom surface deformation (mm)
- `warp_mean`: Mean warp field (mm)

**Format**: 2D arrays with shape (nx, ny) where nx, ny are spatial resolutions (default: 50×50)

#### Stress Field Data (`*_stress.npz`)

Each file contains:
- `sigma_xx`, `sigma_yy`, `sigma_zz`: Normal stress components (Pa)
- `sigma_xy`, `sigma_yz`, `sigma_xz`: Shear stress components (Pa)
- `z_coords`: Through-thickness coordinates (m)

**Format**: 3D arrays with shape (nx, ny, nz) where nz is through-thickness resolution (default: 10)

#### Parameter Data (`*_params.json`)

JSON file containing all 14 manufacturing parameters for each sample.

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0 (for visualization)

## Usage

### 1. Generate Dataset

Generate a synthetic dataset with 1000 samples:

```bash
python sofc_dataset_generator.py --n-samples 1000 --output-dir sofc_dataset
```

**Options:**
- `--n-samples`: Number of samples (default: 1000)
- `--output-dir`: Output directory (default: sofc_dataset)
- `--nx`, `--ny`: Spatial resolution for warp field (default: 50×50)
- `--nz`: Through-thickness resolution (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)

**Example - High Resolution Dataset:**
```bash
python sofc_dataset_generator.py --n-samples 500 --nx 100 --ny 100 --nz 20
```

### 2. Visualize Data

Visualize a specific sample:

```bash
python visualize_dataset.py --dataset-dir sofc_dataset --sample-idx 42
```

Create a comprehensive visualization summary:

```bash
python visualize_dataset.py --dataset-dir sofc_dataset --summary --n-samples 10 --output-dir visualizations
```

**Options:**
- `--dataset-dir`: Dataset directory (default: sofc_dataset)
- `--sample-idx`: Specific sample to visualize
- `--summary`: Create comprehensive visualization summary
- `--n-samples`: Number of samples for summary (default: 5)
- `--output-dir`: Output directory for visualizations

### 3. Load Data in Python

```python
import numpy as np
import json

# Load a sample
sample_id = "sample_00042"

# Load parameters
with open(f"sofc_dataset/parameters/{sample_id}_params.json", 'r') as f:
    params = json.load(f)

# Load warp field
warp_data = np.load(f"sofc_dataset/warp_fields/{sample_id}_warp.npz")
X = warp_data['X']
Y = warp_data['Y']
warp_top = warp_data['warp_top']

# Load stress field
stress_data = np.load(f"sofc_dataset/stress_fields/{sample_id}_stress.npz")
sigma_xx = stress_data['sigma_xx']  # Pa
sigma_yy = stress_data['sigma_yy']
z_coords = stress_data['z_coords']

# Compute von Mises stress
sigma_vm = np.sqrt(0.5 * (
    (sigma_xx - sigma_yy)**2 +
    (sigma_yy - stress_data['sigma_zz'])**2 +
    (stress_data['sigma_zz'] - sigma_xx)**2 +
    6 * (stress_data['sigma_xy']**2 + 
         stress_data['sigma_yz']**2 + 
         stress_data['sigma_xz']**2)
))
```

## Physics-Based Model

The synthetic data generator uses physics-informed models based on:

1. **Thermal Stress from CTE Mismatch**: 
   - Different coefficients of thermal expansion (CTE) between layers
   - Temperature-dependent material properties
   - Accounts for cooling rate effects

2. **Plate Bending Theory**:
   - Multi-layer composite beam theory
   - Neutral axis calculations
   - Curvature-stress relationships

3. **Sintering Effects**:
   - Green density influence on warpage
   - Binder content effects
   - Non-uniform sintering patterns

4. **Boundary Conditions**:
   - Edge constraint effects during sintering
   - Free edge stress concentrations

## Dataset Statistics

For a typical 1000-sample dataset:

**Warp Magnitudes:**
- Range: 0.01 - 2.5 mm (realistic for SOFC plates)
- Distribution: Depends on manufacturing parameter combinations

**Stress Magnitudes:**
- Range: 10 - 400 MPa (von Mises stress)
- Peak stresses typically in electrolyte layer
- Through-thickness variation: 3-5× between minimum and maximum

**Parameter Coverage:**
- Latin Hypercube Sampling ensures uniform coverage
- No bias toward specific parameter combinations
- Captures full parameter space interactions

## ML Applications

This dataset is designed for:

1. **Inverse Modeling**: Train ML models to predict stress fields from measured warp fields
2. **Dimensionality Reduction**: Learn compact representations of high-dimensional stress fields
3. **Physics-Informed Neural Networks**: Constrain ML models with known physics
4. **Manufacturing Optimization**: Predict stress outcomes from process parameters

### Example ML Workflow

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
n_samples = 1000
X_warp = []  # Input: warp fields
y_stress = []  # Target: stress fields

for i in range(n_samples):
    warp_data = np.load(f"sofc_dataset/warp_fields/sample_{i:05d}_warp.npz")
    stress_data = np.load(f"sofc_dataset/stress_fields/sample_{i:05d}_stress.npz")
    
    # Flatten spatial dimensions
    X_warp.append(warp_data['warp_mean'].flatten())
    y_stress.append(stress_data['sigma_xx'].flatten())

X_warp = np.array(X_warp)
y_stress = np.array(y_stress)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_warp, y_stress, test_size=0.2, random_state=42
)

# Train your ML model
# model.fit(X_train, y_train)
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sofc_warp_stress_2025,
  title={Synthetic SOFC Warpage and Residual Stress Dataset for ML-Augmented Inverse Modeling},
  author={Generated Dataset},
  year={2025},
  description={Physics-informed synthetic dataset for residual stress quantification from warped SOFC plates}
}
```

## Key Features

✅ **Physics-Informed**: Based on real thermal-mechanical physics of SOFC manufacturing  
✅ **Comprehensive**: 14 manufacturing parameters with realistic ranges  
✅ **Scalable**: Generate datasets of any size (100s to 10,000s of samples)  
✅ **Well-Documented**: Complete metadata and parameter tracking  
✅ **Visualization Tools**: Built-in tools for data exploration  
✅ **ML-Ready**: Paired input-output format ideal for supervised learning  
✅ **Reproducible**: Seeded random generation for reproducibility  

## Limitations and Considerations

⚠️ **Synthetic Data**: While physics-informed, this is synthetic data that approximates FEA results, not actual experimental measurements or full FEA simulations.

⚠️ **Simplifications**: 
- Isotropic material properties (real SOFCs may have anisotropy)
- Simplified creep and plasticity models
- No microstructural effects (grain boundaries, pores)
- No chemical composition gradients

⚠️ **Validation Required**: ML models trained on this data should be validated against real experimental data when available.

## Future Enhancements

Potential extensions to this dataset:
1. Add measurement noise to simulate real scanning profilometry
2. Include time-dependent effects (creep relaxation)
3. Add defect scenarios (cracks, delamination)
4. Multi-fidelity data (coarse + fine resolution)
5. Uncertainty quantification

## Contact and Support

For questions, issues, or contributions, please refer to the research article or open an issue in the repository.

## License

This dataset generator is provided for research and educational purposes.
