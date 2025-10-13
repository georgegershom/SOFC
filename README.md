# SOFC Digital Twin Dataset Generator

A comprehensive system for generating multi-fidelity datasets for Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring.

## Overview

This repository provides a complete framework for generating three types of datasets essential for SOFC digital twin development:

1. **High-Fidelity Physics-Based Simulation Data** - Multi-physics simulations (electrochemical, thermal, structural)
2. **Experimental Validation Data** - Simulated lab test rig measurements with realistic noise and uncertainty
3. **Real-Time Monitoring Data** - High-frequency operational data streams for adaptive monitoring

## Features

### Multi-Physics Simulation
- **Electrochemical Modeling**: Charge conservation, species transport, Nernst potential calculation
- **Thermal Analysis**: Energy conservation, heat transfer, thermal stress calculation
- **Structural Analysis**: Linear elasticity, stress-strain relationships, failure prediction
- **Coupled Physics**: Iterative coupling between all physics domains

### Dataset Types

#### Dataset 1: High-Fidelity Simulation Data
- Parameter sweeps using Latin Hypercube Sampling
- 3D spatial field data (temperature, stress, strain, current density)
- Degradation state modeling
- Failure prediction metrics
- HDF5 storage with compression

#### Dataset 2: Experimental Data
- Simulated lab test rig measurements
- Electrochemical Impedance Spectroscopy (EIS)
- Thermal imaging data
- Strain gauge measurements
- Acoustic emission simulation
- Realistic noise and measurement uncertainty

#### Dataset 3: Real-Time Monitoring Data
- High-frequency operational data streams
- Low-frequency high-value measurements
- Data assimilation ready format
- Adaptive-scale monitoring triggers

### Degradation Modeling
- **Crack Propagation**: Paris' law, stress intensity factors, failure prediction
- **Material Aging**: Creep damage, oxidation, phase changes, property degradation
- **Multi-Scale Analysis**: From microstructural to component level

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt)

### Quick Install
```bash
# Clone the repository
git clone https://github.com/sofc-research/dataset-generator.git
cd dataset-generator

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development Install
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with GPU acceleration (optional)
pip install -e ".[gpu]"
```

## Quick Start

### Generate Complete Dataset
```python
from sofc_dataset_generator import SOFCDatasetGenerator

# Initialize dataset generator
generator = SOFCDatasetGenerator()

# Generate all three dataset types
datasets = generator.generate_complete_dataset(
    high_fidelity_samples=1000,
    experimental_duration=100.0,  # hours
    monitoring_duration=24.0,     # hours
    output_dir="sofc_datasets"
)
```

### Generate Individual Datasets

#### High-Fidelity Simulation Data
```python
# Generate high-fidelity data
hf_data = generator.generate_high_fidelity_data(
    n_samples=1000,
    operating_conditions_range={
        'current_density': (0.1, 1.0),  # A/cm²
        'fuel_utilization': (0.6, 0.9),  # %
        'inlet_fuel_temp': (973.15, 1173.15),  # K
    },
    material_properties_range={
        'anode_porosity': (0.2, 0.4),
        'electrolyte_thickness': (5e-6, 20e-6),  # m
    },
    output_file="high_fidelity_data.h5"
)
```

#### Experimental Validation Data
```python
# Generate experimental data
exp_data = generator.generate_experimental_data(
    test_duration_hours=100.0,
    operating_profile={
        'current_density': 0.5,  # A/cm²
        'inlet_fuel_temp': 1073.15,  # K
    },
    output_file="experimental_data.h5"
)
```

#### Real-Time Monitoring Data
```python
# Generate monitoring data
mon_data = generator.generate_monitoring_data(
    duration_hours=24.0,
    high_freq_sampling=1.0,    # Hz
    low_freq_sampling=1/3600.0, # Hz (hourly)
    output_file="monitoring_data.h5"
)
```

## Usage Examples

### Run Complete Example
```bash
# Run the complete dataset generation example
python sofc_dataset_generator/examples/generate_complete_dataset.py
```

This will generate:
- High-fidelity simulation data (100 samples)
- Experimental validation data (24 hours)
- Real-time monitoring data (6 hours)
- Comprehensive visualizations

### Custom Configuration
```python
# Custom configuration
config = {
    'electrochemical': {
        'fuel_composition': {'H2': 0.7, 'H2O': 0.3},
        'operating_pressure': 1.0e5,  # Pa
    },
    'thermal': {
        'ambient_temperature': 298.15,  # K
        'convection_coefficient': 10.0,  # W/m²K
    },
    'structural': {
        'material_properties': {
            'anode': {'E': 200e9, 'nu': 0.3, 'alpha': 12e-6},
            'electrolyte': {'E': 200e9, 'nu': 0.3, 'alpha': 10e-6},
        }
    }
}

generator = SOFCDatasetGenerator(config)
```

## Dataset Structure

### High-Fidelity Data
```
high_fidelity_data/
├── parameters/           # Input parameters for each simulation
├── mesh/                 # 3D mesh coordinates
├── potential_field/      # Electrochemical potential field
├── current_density_field/ # Current density vector field
├── temperature_field/    # Temperature field
├── stress_tensor/        # Stress tensor field
├── strain_tensor/        # Strain tensor field
├── cell_voltage/         # Global cell voltage
├── efficiency/           # Cell efficiency
└── failure_predictions/  # Failure analysis results
```

### Experimental Data
```
experimental_data/
├── time_points/          # Time series
├── global_operational/   # Global measurements (V, I, T, P)
├── eis_data/            # Electrochemical impedance spectroscopy
├── thermal_imaging/     # Thermal imaging data
├── strain_gauges/       # Strain gauge measurements
└── acoustic_emission/   # Acoustic emission events
```

### Monitoring Data
```
monitoring_data/
├── high_frequency_data/  # High-frequency measurements
├── low_frequency_data/   # Low-frequency measurements
├── data_assimilation/    # Data assimilation windows
└── adaptive_triggers/    # Monitoring triggers
```

## Visualization

The package includes comprehensive visualization tools:

- Operating conditions distribution
- Global performance metrics
- 3D field visualizations
- Time series plots
- EIS data (Nyquist and Bode plots)
- Adaptive monitoring triggers

## Advanced Features

### Degradation Modeling
```python
from sofc_dataset_generator.degradation_models import CrackPropagationModel, MaterialAgingModel

# Crack propagation
crack_model = CrackPropagationModel(config)
crack_results = crack_model.simulate_crack_growth(stress_history, time_points)

# Material aging
aging_model = MaterialAgingModel(config)
aging_results = aging_model.simulate_aging(material_type, stress_history, temperature_history, time_points)
```

### Data Storage and Retrieval
```python
from sofc_dataset_generator.data_formats.hdf5_utils import HDF5DatasetManager

# Save dataset
manager = HDF5DatasetManager("dataset.h5")
manager.save_dataset(dataset, "high_fidelity")

# Load dataset
loaded_dataset = manager.load_dataset("high_fidelity")
```

## Configuration

The system is highly configurable through YAML files or Python dictionaries:

```yaml
# config.yaml
electrochemical:
  fuel_composition:
    H2: 0.7
    H2O: 0.3
  operating_pressure: 1.0e5

thermal:
  ambient_temperature: 298.15
  convection_coefficient: 10.0

structural:
  material_properties:
    anode:
      youngs_modulus: 200e9
      poisson_ratio: 0.3
      thermal_expansion: 12e-6
```

## Performance Optimization

- **Parallel Processing**: Multi-threaded simulation execution
- **Memory Management**: Efficient array storage and retrieval
- **Compression**: HDF5 compression for large datasets
- **Chunking**: Optimized data access patterns
- **GPU Acceleration**: Optional CUDA support for large-scale simulations

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and install in development mode
git clone https://github.com/sofc-research/dataset-generator.git
cd dataset-generator
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 sofc_dataset_generator/
black sofc_dataset_generator/
```

## Citation

If you use this dataset generator in your research, please cite:

```bibtex
@software{sofc_dataset_generator,
  title={SOFC Digital Twin Dataset Generator},
  author={SOFC Research Team},
  year={2024},
  url={https://github.com/sofc-research/dataset-generator}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Multi-physics simulation framework
- HDF5 data storage utilities
- Visualization and analysis tools
- Degradation modeling algorithms

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the research team
- Check the documentation

## Roadmap

- [ ] Integration with COMSOL/ANSYS
- [ ] Machine learning model training utilities
- [ ] Cloud deployment support
- [ ] Advanced visualization dashboard
- [ ] Real-time data streaming
- [ ] Uncertainty quantification tools

---

**Note**: This is a research tool for generating synthetic datasets. For production use, ensure proper validation against experimental data.