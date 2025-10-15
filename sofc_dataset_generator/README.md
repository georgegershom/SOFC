# SOFC Dataset Generator

## ML-Augmented Inverse Modeling for Residual Stress Quantification from Warped SOFC Plates

This repository contains a comprehensive synthetic dataset generator for creating paired warp and residual stress data from Solid Oxide Fuel Cell (SOFC) plates. The dataset is designed for training machine learning models to perform inverse modeling - predicting residual stress fields from easily measurable warp measurements.

## Overview

The "Ground Truth" Core Dataset provides direct, paired examples of warp and stress through high-fidelity finite element analysis (FEA) simulations. This synthetic dataset offers a perfect, noise-free, one-to-one mapping between warp and stress, serving as the primary training data for ML models.

### Key Features

- **Virtual Design of Experiments (DOE)**: Generate 500-1000+ realistic SOFC manufacturing scenarios
- **High-Fidelity FEA**: Coupled thermo-mechanical simulations of the complete manufacturing process
- **Multi-Format Output**: 
  - Warp fields as 2.5D height maps (like digital elevation models)
  - Residual stress fields as 3D voxelized tensors
  - Point clouds and structured mesh data
- **ML-Ready Exports**: HDF5, NumPy, and TensorFlow-compatible formats
- **Comprehensive Metadata**: Full traceability of simulation parameters and results

## Dataset Structure

### Input Features (Easy to Measure)
- **Warp Field**: 3D coordinates of deformed top and bottom surfaces
- **2.5D Height Maps**: Regular grid representation of surface deformation
- **Point Clouds**: Unstructured surface coordinate data

### Target Labels (Hard to Measure)
- **Residual Stress Field**: Full 3D stress tensor (σ_xx, σ_yy, σ_xy, σ_zz, etc.) at every point
- **Stress Invariants**: von Mises stress, principal stresses, hydrostatic stress
- **3D Voxelized Fields**: Regular grid representation for CNN/3D-CNN models

## Installation

### Prerequisites

```bash
# Core dependencies
pip install numpy scipy pandas matplotlib scikit-learn h5py
pip install pyvista meshio tqdm joblib

# Optional (for advanced FEA)
pip install fenics dolfin

# For DOE generation (if available)
pip install pyDOE2
```

### Quick Start

```bash
git clone <repository-url>
cd sofc_dataset_generator
pip install -r requirements.txt
```

## Usage

### Basic Dataset Generation

```python
from src.main_generator import create_dataset_generator

# Create generator with default configuration
generator = create_dataset_generator()

# Generate dataset
dataset_id = generator.generate_dataset()
print(f"Dataset created with ID: {dataset_id}")
```

### Custom Configuration

```python
# Load custom configuration
generator = create_dataset_generator("config/custom_config.yaml")

# Or modify configuration programmatically
generator.config['dataset']['n_samples'] = 500
generator.config['fea']['mesh_resolution'] = 1e-3  # 1mm elements

dataset_id = generator.generate_dataset()
```

### Command Line Interface

```bash
# Generate dataset with default settings
python -m src.main_generator

# Custom configuration
python -m src.main_generator --config config/my_config.yaml --n-samples 1000

# Resume interrupted generation
python -m src.main_generator --resume

# Validate configuration only
python -m src.main_generator --validate --config my_config.yaml
```

## Configuration

### Key Parameters

```yaml
dataset:
  name: "SOFC_Ground_Truth_Dataset"
  n_samples: 1000                    # Number of DOE points
  output_directory: "./dataset"

doe:
  sampling_method: "latin_hypercube"  # LHS, Sobol, random
  seed: 42

fea:
  mesh_resolution: 2.0e-3            # Element size (meters)
  n_time_steps: 50                   # Thermal history resolution
  solver_type: "simplified"          # or "fenics"

warp_extraction:
  grid_resolution: [64, 64]          # 2.5D height map resolution

stress_extraction:
  voxel_resolution: [32, 32, 16]     # 3D stress field resolution
```

### Manufacturing Parameter Space

The DOE covers realistic SOFC manufacturing variations:

- **Geometry**: Plate dimensions, layer thicknesses
- **Materials**: Porosity, composition, grain size
- **Thermal Profile**: Peak temperature, heating/cooling rates, dwell time
- **Manufacturing Conditions**: Atmosphere, support conditions, green density
- **Defects**: Controlled introduction of micro-cracks, delamination

## Output Data Format

### HDF5 Structure

```
dataset/
├── metadata.json                 # Dataset metadata
├── samples/
│   ├── sample_000001/
│   │   ├── doe_parameters.json   # Manufacturing parameters
│   │   ├── warp_data.h5         # Warp field data
│   │   ├── stress_data.h5       # Stress field data
│   │   └── simulation_metadata.json
│   └── ...
└── exports/
    ├── train_data.h5            # ML-ready training data
    ├── test_data.h5             # ML-ready test data
    └── ml_metadata.json
```

### ML-Ready Data Arrays

```python
import h5py

# Load training data
with h5py.File('exports/train_data.h5', 'r') as f:
    # Input: Warp height maps [N, 64, 64]
    warp_inputs = f['inputs/warp_height_maps'][:]
    
    # Target: Stress voxels [N, 32, 32, 16, 6]
    stress_targets = f['targets/stress_voxels'][:]
    
    # Parameters: DOE parameters [N, n_params]
    parameters = f['parameters/geometry.length'][:]
```

## Physics and Modeling

### FEA Simulation Process

1. **Mesh Generation**: Multi-layer hexahedral/tetrahedral meshes
2. **Material Models**: Temperature-dependent properties for Ni-YSZ, YSZ, LSM-YSZ
3. **Thermal Analysis**: Prescribed temperature profile during sintering
4. **Mechanical Analysis**: Thermal expansion, shrinkage, creep, and residual stress
5. **Coupling**: Sequential or monolithic thermo-mechanical coupling

### Material Models

- **Anode (Ni-YSZ)**: Porosity and Ni content dependent properties
- **Electrolyte (YSZ)**: Grain size and density dependent properties  
- **Cathode (LSM-YSZ)**: Porosity and LSM content dependent properties
- **Temperature Dependence**: Elastic modulus, thermal expansion, creep

### Stress Field Extraction

- **Element Stress**: Computed at integration points
- **Nodal Extrapolation**: Volume-weighted averaging
- **3D Voxelization**: Interpolation to regular grid
- **Invariant Calculation**: von Mises, principal stresses, etc.

## Examples

### Loading and Visualizing Data

```python
from src.utils.dataset_manager import create_dataset_manager

# Load dataset
manager = create_dataset_manager("./dataset")
manager.load_dataset()

# Get a sample
sample = manager.get_sample("sample_000001")

# Visualize warp field
from src.extraction.warp_extractor import WarpFieldExtractor
extractor = WarpFieldExtractor()
fig = extractor.visualize_warp_field(sample.warp_data)
fig.show()

# Visualize stress field  
from src.extraction.stress_extractor import StressFieldExtractor
extractor = StressFieldExtractor()
fig = extractor.visualize_stress_field(sample.stress_data)
fig.show()
```

### Training a Simple ML Model

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load ML-ready data
with h5py.File('exports/train_data.h5', 'r') as f:
    X_train = f['inputs/warp_height_maps'][:].reshape(len(f['inputs/warp_height_maps']), -1)
    y_train = f['targets/stress_voxels'][:].reshape(len(f['targets/stress_voxels']), -1)

with h5py.File('exports/test_data.h5', 'r') as f:
    X_test = f['inputs/warp_height_maps'][:].reshape(len(f['inputs/warp_height_maps']), -1)
    y_test = f['targets/stress_voxels'][:].reshape(len(f['targets/stress_voxels']), -1)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

## Performance and Scaling

### Computational Requirements

- **Memory**: ~8GB RAM for default settings
- **Storage**: ~100MB per sample (full resolution)
- **Time**: ~30 seconds per sample (simplified solver)
- **Parallelization**: Multi-process support for batch generation

### Scaling Guidelines

| Dataset Size | Recommended Hardware | Generation Time |
|-------------|---------------------|-----------------|
| 100 samples | 4 cores, 8GB RAM   | ~1 hour         |
| 1000 samples| 8 cores, 16GB RAM  | ~8 hours        |
| 5000 samples| 16 cores, 32GB RAM | ~40 hours       |

## Quality Control

### Automatic Validation

- **Parameter Coverage**: Ensures DOE spans full parameter space
- **Physics Checks**: Energy conservation, equilibrium validation
- **Data Quality**: NaN detection, outlier identification
- **Correlation Analysis**: Expected warp-stress relationships

### Manual Inspection

```python
# Generate dataset statistics
stats = manager.calculate_dataset_statistics()
print(f"Warp range: {stats['warp_statistics']['max_warp']}")
print(f"Stress range: {stats['stress_statistics']['max_von_mises']}")

# Visualize parameter distributions
import matplotlib.pyplot as plt
doe_matrix = generator.generate_doe_matrix()
doe_matrix.hist(figsize=(15, 10))
plt.show()
```

## Advanced Features

### Multi-Fidelity Modeling

Generate datasets with different mesh resolutions for multi-fidelity ML:

```yaml
advanced:
  multi_fidelity: true
  fidelity_levels: [1, 2, 4]  # Mesh refinement factors
```

### Adaptive Sampling

Iteratively improve dataset coverage:

```yaml
advanced:
  adaptive_sampling: true
  adaptive_criterion: "variance_reduction"
  adaptive_batch_size: 50
```

### Uncertainty Quantification

Include uncertainty estimates:

```yaml
advanced:
  uncertainty_quantification: true
  uq_method: "polynomial_chaos"
  uq_order: 2
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce mesh resolution or voxel resolution
2. **Slow Generation**: Enable parallel processing or use simplified solver
3. **Import Errors**: Install missing dependencies (see requirements.txt)
4. **FEA Convergence**: Adjust time step size or solver tolerances

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
python test_system.py
```

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code quality
flake8 src/
black src/
```

### Adding New Features

1. **Material Models**: Extend `src/materials/material_models.py`
2. **Extraction Methods**: Add to `src/extraction/`
3. **DOE Strategies**: Implement in `src/doe/`
4. **Export Formats**: Extend `src/utils/dataset_manager.py`

## Citation

If you use this dataset generator in your research, please cite:

```bibtex
@software{sofc_dataset_generator,
  title={SOFC Dataset Generator: ML-Augmented Inverse Modeling for Residual Stress Quantification},
  author={[Author Names]},
  year={2024},
  url={[Repository URL]}
}
```

## License

[License information]

## Contact

For questions, issues, or contributions:
- GitHub Issues: [Repository Issues URL]
- Email: [Contact Email]
- Documentation: [Documentation URL]