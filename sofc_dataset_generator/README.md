# SOFC Digital Twin Dataset Generator

This repository contains a comprehensive dataset generation system for Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring.

## Dataset Overview

### Dataset 1: High-Fidelity Physics-Based Simulation Data
- Multi-physics simulations (electrochemical, thermal, structural)
- Parameter sweeps across operating conditions and material properties
- 3D spatial field data (temperature, stress, strain, current density)
- Degradation state modeling

### Dataset 2: Experimental Data Simulation
- Simulated lab test rig measurements
- Electrochemical Impedance Spectroscopy (EIS) data
- Thermal imaging and strain gauge data
- Acoustic emission simulation

### Dataset 3: Real-Time Monitoring Data
- High-frequency operational data streams
- Adaptive-scale monitoring capabilities
- Data assimilation ready format

## Project Structure

```
sofc_dataset_generator/
├── physics_simulators/     # Multi-physics simulation modules
├── data_generators/        # Dataset generation scripts
├── experimental_sim/       # Experimental data simulation
├── degradation_models/     # Crack propagation and aging models
├── data_formats/          # HDF5/NPZ storage utilities
├── visualization/         # Data visualization tools
├── validation/            # Dataset validation and quality tools
└── examples/              # Usage examples and tutorials
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- H5py for HDF5 storage
- Scikit-learn for ML utilities
- Optional: COMSOL/ANSYS integration modules

## Quick Start

```python
from sofc_dataset_generator import SOFCDatasetGenerator

# Initialize dataset generator
generator = SOFCDatasetGenerator()

# Generate high-fidelity simulation data
sim_data = generator.generate_high_fidelity_data(
    n_samples=1000,
    operating_conditions_range=operating_ranges,
    material_properties_range=material_ranges
)

# Generate experimental validation data
exp_data = generator.generate_experimental_data(
    test_duration_hours=100,
    sampling_frequency=1.0
)

# Generate real-time monitoring data
monitoring_data = generator.generate_monitoring_data(
    duration_hours=24,
    high_freq_sampling=1.0,
    low_freq_sampling=3600.0
)
```