# SOFC Synthetic Dataset Generator

A comprehensive framework for generating synthetic datasets for ML-augmented inverse modeling of residual stress quantification from warped SOFC plates.

## Overview

This project generates the "Ground Truth" Core Dataset for SOFC residual stress analysis, providing direct, paired examples of warp and stress fields through virtual Design of Experiments (DOE) and high-fidelity finite element analysis (FEA) simulations.

## Key Features

- **Virtual DOE**: Generate 500-1000+ realistic SOFC manufacturing scenarios
- **FEA Simulation**: Coupled thermo-mechanical analysis with temperature-dependent material properties
- **Paired Data**: Extract warp fields (input) and residual stress fields (target) from each simulation
- **Multiple Formats**: Export data in HDF5, VTK, CSV, and NumPy formats
- **ML-Ready**: Structured datasets optimized for machine learning training

## Dataset Structure

### Input Features (Warp Field)
- **Point Cloud**: 3D coordinates of deformed SOFC plate surfaces
- **Height Maps**: 2.5D height maps (digital elevation model style)
- **Displacement Vectors**: 3D displacement field at surface nodes

### Target Labels (Stress Field)
- **3D Stress Tensor**: Full stress tensor (σ_xx, σ_yy, σ_xy, σ_zz, etc.) at integration points
- **Surface Stress Maps**: 2D stress distributions on critical surfaces
- **Stress Invariants**: Von Mises stress, principal stresses, hydrostatic stress

### Manufacturing Parameters
- **Geometric**: Cell dimensions, layer thicknesses
- **Material**: Property variations (Young's modulus, CTE, etc.)
- **Process**: Sintering temperature, cooling rate, assembly pressure
- **Environmental**: Operating temperature, thermal gradients

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sofc_dataset_generator

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Generate Small Dataset (Testing)
```bash
python examples/generate_small_dataset.py
```

### Generate Full Dataset
```bash
python generate_dataset.py --samples 500 --output_dir ./dataset --strategy lhs
```

### Load and Analyze Dataset
```bash
python examples/load_and_analyze_dataset.py
```

## Usage Examples

### Basic Dataset Generation
```python
from generate_dataset import SOFCDatasetGenerator

# Create generator
generator = SOFCDatasetGenerator()

# Generate dataset
dataset = generator.generate_dataset(
    n_samples=100,
    output_dir='./my_dataset',
    sampling_strategy='lhs',
    random_seed=42
)
```

### Load and Use Dataset
```python
from data_export.hdf5_exporter import HDF5Exporter

# Load dataset
exporter = HDF5Exporter('dataset/sofc_dataset.h5', mode='r')
dataset = exporter.load_dataset()

# Get ML training data
features = dataset.get_feature_matrix('height_map')
targets = dataset.get_target_matrix('surface_map')
parameters = dataset.get_parameter_matrix()
```

### Custom Configuration
```python
config = {
    'mesh_resolution': {
        'elements_per_mm': 15.0,
        'electrolyte_elements_z': 10
    },
    'warp_analysis': {
        'height_map_resolution': (100, 100),
        'surfaces': ['top', 'electrolyte_top']
    }
}

generator = SOFCDatasetGenerator(config)
```

## Project Structure

```
sofc_dataset_generator/
├── src/
│   ├── materials/          # Material property definitions
│   │   ├── sofc_materials.py
│   │   └── creep_models.py
│   ├── doe/               # Design of Experiments
│   │   ├── doe_generator.py
│   │   ├── sofc_parameters.py
│   │   └── sampling_strategies.py
│   ├── fea/               # FEA simulation framework
│   │   ├── mesh_generator.py
│   │   ├── fea_solver.py
│   │   ├── warp_analysis.py
│   │   └── stress_analysis.py
│   └── data_export/       # Data export utilities
│       ├── ml_dataset.py
│       └── hdf5_exporter.py
├── config/                # Configuration files
├── examples/              # Example scripts
├── tests/                 # Unit tests
├── generate_dataset.py    # Main generation script
└── requirements.txt       # Dependencies
```

## Configuration

The system can be configured through JSON files or Python dictionaries:

```json
{
  "mesh_resolution": {
    "elements_per_mm": 10.0,
    "electrolyte_elements_z": 8
  },
  "warp_analysis": {
    "height_map_resolution": [50, 50],
    "surfaces": ["top", "electrolyte_top"]
  },
  "simulation": {
    "load_cases": ["sintering_cooling", "operation"],
    "temperature_field_type": "linear"
  }
}
```

## Material Properties

The system includes realistic material properties for SOFC components:

- **8YSZ Electrolyte**: Temperature-dependent elastic properties, creep behavior
- **Ni-YSZ Anode**: Cermet properties with porosity effects
- **LSM-YSZ Cathode**: Composite material properties
- **Crofer 22 APU Interconnect**: Ferritic stainless steel properties

## Sampling Strategies

Multiple DOE sampling strategies are available:

- **Latin Hypercube Sampling (LHS)**: Recommended for most applications
- **Sobol Sequences**: Low-discrepancy quasi-random sampling
- **Random Sampling**: Uniform random sampling
- **Stratified Sampling**: Ensures representation across parameter ranges

## Output Formats

### HDF5 (Recommended)
- Efficient storage and compression
- Hierarchical structure for easy access
- Metadata preservation
- ML-ready aggregated data

### VTK
- Visualization in ParaView, VisIt
- 3D mesh and field data
- Compatible with scientific visualization tools

### CSV/NumPy
- Simple tabular format
- Easy integration with pandas/sklearn
- Individual sample files

## Performance

Typical performance on modern hardware:
- **Small dataset (10 samples)**: ~2-5 minutes
- **Medium dataset (100 samples)**: ~20-60 minutes  
- **Large dataset (500+ samples)**: ~2-8 hours

Performance scales with:
- Mesh resolution
- Number of surfaces analyzed
- Complexity of material models

## Validation

The generated datasets are validated against:
- Literature data for material properties
- Analytical solutions for simple cases
- Mesh convergence studies
- Physical plausibility checks

## Machine Learning Integration

The generated datasets are optimized for ML training:

```python
# Split dataset
train_data, val_data, test_data = dataset.split_dataset(
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

# Get training data
X_train = train_data.get_feature_matrix('height_map')
y_train = train_data.get_target_matrix('surface_map')

# Train ML model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this dataset generator in your research, please cite:

```bibtex
@software{sofc_dataset_generator,
  title={SOFC Synthetic Dataset Generator for ML-Augmented Inverse Modeling},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-repo/sofc-dataset-generator}
}
```

## Acknowledgments

- Based on research from "A Comparative Analysis of Constitutive Models for Predicting the Electrolyte's Fracture Risk in Planar SOFCs"
- Material properties sourced from literature and experimental data
- FEA framework inspired by commercial software capabilities

## Support

For questions and support:
- Create an issue on GitHub
- Check the examples directory
- Review the configuration options