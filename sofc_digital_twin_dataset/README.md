# Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring

## Dataset Overview

This dataset supports the development of an adaptive-scale physics-informed digital twin for Solid Oxide Fuel Cell (SOFC) thermo-structural integrity monitoring. The dataset follows the "Data-Model Fusion" trinity approach, providing comprehensive data across multiple scales and domains.

## Dataset Structure

```
sofc_digital_twin_dataset/
├── README.md
├── requirements.txt
├── config/
│   └── dataset_config.yaml
├── data/
│   ├── materials_geometry/
│   │   ├── microstructural/
│   │   └── macro_scale/
│   ├── operational_electrochemical/
│   │   ├── controlled_inputs/
│   │   └── electrochemical_response/
│   ├── thermo_structural/
│   │   ├── temperature_fields/
│   │   └── stress_strain/
│   ├── degradation_failure/
│   │   ├── accelerated_aging/
│   │   └── post_mortem/
│   └── synthetic_sensors/
│       ├── thermocouples/
│       ├── ir_camera/
│       └── strain_gauges/
├── generators/
│   ├── __init__.py
│   ├── base_generator.py
│   ├── materials_generator.py
│   ├── operational_generator.py
│   ├── thermo_structural_generator.py
│   ├── degradation_generator.py
│   └── sensor_generator.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── physics_utils.py
│   └── visualization.py
└── examples/
    ├── dataset_exploration.py
    ├── model_validation.py
    └── digital_twin_demo.py
```

## Core Philosophy: Data-Model Fusion Trinity

1. **High-Fidelity Physics-Based Model**: Trained and validated with high-resolution data
2. **Reduced-Order/Surrogate Model**: Trained on data from the high-fidelity model for real-time execution
3. **Real-World Sensor Data**: For calibration, assimilation, and continuous updating of the digital twin

## Dataset Components

### 1. Materials & Geometry Data
- **Microstructural Data (µ-scale)**: 3D tomography images, porosity, tortuosity
- **Macro-scale Geometry**: CAD models, assembly dimensions

### 2. Operational & Electrochemical Performance Data
- **Controlled Input Parameters**: Fuel composition, flow rates, temperatures, current density
- **Electrochemical Response**: Cell voltage, EIS spectra

### 3. Thermo-Structural Field Data
- **Temperature Fields**: Thermocouple data, IR camera measurements
- **Stress & Strain**: Strain gauge data, DIC measurements

### 4. Degradation & Failure Mode Data
- **Accelerated Aging**: Long-term performance data under various conditions
- **Post-Mortem Analysis**: Microscopy and spectroscopy data

## Usage

```python
from generators import SOFCDatasetGenerator

# Initialize the dataset generator
generator = SOFCDatasetGenerator()

# Generate complete dataset
generator.generate_complete_dataset()

# Access specific data components
temp_data = generator.get_temperature_field_data()
stress_data = generator.get_stress_strain_data()
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- OpenCV (for image processing)
- h5py (for data storage)
- scikit-learn (for machine learning utilities)

## Citation

If you use this dataset in your research, please cite:

```
Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring
[Your Name], [Institution], [Year]
```