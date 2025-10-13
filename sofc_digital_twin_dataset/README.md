# SOFC Digital Twin Multi-Fidelity Dataset

## Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring

This comprehensive dataset supports research in developing physics-informed machine learning models for Solid Oxide Fuel Cell (SOFC) digital twins, focusing on thermo-structural integrity monitoring and adaptive-scale modeling.

## 📊 Dataset Overview

### Core Philosophy: Multi-Fidelity & Multi-Physics

This dataset uniquely combines:
- **High-Fidelity Simulation Data**: Physics-based multi-field simulations
- **Synthetic Experimental Data**: Mimics real SOFC test rig measurements
- **Real-time Monitoring Streams**: Continuous operational data for adaptive models
- **Multi-Physics Coupling**: Electrochemical, thermal, and structural responses

### Key Features

- **100 Multi-physics simulations** with full 3D field data
- **1000 hours of operational data** at 0.1 Hz sampling
- **50 EIS measurements** for degradation tracking
- **100 thermal images** for validation
- **8 strain gauge sensors** continuous monitoring
- **54 acoustic emission events** for damage detection
- **3600 seconds of real-time stream** at 1 Hz

## 🚀 Quick Start

### Installation

```bash
# Clone or download the dataset
cd sofc_digital_twin_dataset

# Install required packages (optional for advanced features)
pip install numpy pandas matplotlib

# For full visualization capabilities
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
import pandas as pd
import json

# Load simulation metadata
with open('data/simulation/simulation_metadata.json', 'r') as f:
    simulations = json.load(f)

# Load a simulation field
temperature_field = np.load('data/simulation/temperature_0000.npy')
stress_field = np.load('data/simulation/stress_0000.npy')

# Load experimental data
op_data = pd.read_csv('data/experimental/operational_data.csv')
print(f"Operational data shape: {op_data.shape}")

# Load real-time monitoring stream
stream = pd.read_csv('data/monitoring/realtime_stream.csv')
```

## 📁 Dataset Structure

```
sofc_digital_twin_dataset/
│
├── data/
│   ├── simulation/           # Physics-based simulation data
│   │   ├── simulation_metadata.json
│   │   ├── temperature_XXXX.npy    # 3D temperature fields
│   │   ├── stress_XXXX.npy         # 3D stress fields
│   │   └── current_density_XXXX.npy # 3D current density fields
│   │
│   ├── experimental/         # Synthetic experimental measurements
│   │   ├── operational_data.csv    # Time-series operational data
│   │   ├── eis_data.json          # Impedance spectroscopy
│   │   ├── thermal_images.npy     # Thermal camera images
│   │   ├── strain_gauge_data.csv  # Strain measurements
│   │   └── acoustic_emission_events.csv # AE events
│   │
│   ├── monitoring/          # Real-time monitoring data
│   │   ├── realtime_stream.csv    # Continuous monitoring stream
│   │   └── adaptive_triggers.csv  # Adaptive model triggers
│   │
│   ├── metadata.json        # Dataset metadata and units
│   └── data_splits.json     # Train/validation/test splits
│
├── src/
│   ├── sofc_dataset_generator.py  # Full dataset generator
│   ├── data_loader.py             # PyTorch data loaders
│   └── visualization_tools.py     # Analysis and visualization
│
└── README.md
```

## 🔬 Data Description

### 1. Simulation Data

**Parameters Varied:**
- Current Density: 1000-10000 A/m²
- Fuel Utilization: 30-85%
- Air Utilization: 20-50%
- Fuel Temperature: 600-800°C
- Air Temperature: 500-700°C
- Crack Length: 0-5 mm
- Porosity Change: 0.9-1.2 (relative)

**Output Fields (20×20×5 grid):**
- 3D Temperature distribution
- 3D Von Mises stress field
- 3D Current density distribution
- Scalar outputs: Voltage, Max stress, Creep damage

### 2. Experimental Data

**Operational Data (10,000 points over 1000 hours):**
- Voltage, Current, Power
- Inlet/Outlet temperatures
- Fuel and air flow rates
- Efficiency calculations

**EIS Data (50 measurements):**
- Frequency: 0.01 Hz to 100 kHz
- Complex impedance (Z_real, Z_imag)
- Ohmic and charge transfer resistances

**Thermal Images (100 images, 32×32 pixels):**
- Surface temperature distributions
- Hot spot detection
- Degradation patterns

**Strain Gauge Data (8 sensors):**
- Continuous strain measurements
- Thermal and mechanical strain components
- Creep strain accumulation

**Acoustic Emission Events:**
- Event timestamps and characteristics
- Amplitude, duration, energy
- Event types: crack initiation/propagation, delamination

### 3. Monitoring Data

**Real-time Stream (3600 seconds at 1 Hz):**
- Voltage, Current, Power
- Temperature measurements
- Flow rates
- Timestamp for synchronization

## 🤖 Machine Learning Applications

### Physics-Informed Neural Networks (PINNs)

Use the simulation data to train PINNs that encode the governing PDEs:
- Charge conservation
- Energy conservation
- Linear momentum conservation

### Digital Twin Development

1. **Offline Training**: Use simulation data for surrogate model training
2. **Online Adaptation**: Use experimental data for model calibration
3. **Real-time Monitoring**: Use stream data for state estimation
4. **Integrity Prognosis**: Predict remaining useful life

### Example: Loading Data for ML

```python
from src.data_loader import SOFCSimulationDataset, ExperimentalDataLoader

# Load simulation dataset
train_dataset = SOFCSimulationDataset('data', split='train', normalize=True)
train_loader = train_dataset.get_dataloader(batch_size=32)

# Load experimental data
exp_loader = ExperimentalDataLoader('data')
degradation_indicators = exp_loader.get_degradation_indicators()

# Get a batch for training
for batch in train_loader:
    inputs = batch['inputs']  # Operating conditions
    fields = batch['fields']  # 3D field data
    outputs = batch['outputs']  # Scalar outputs
    break
```

## 📈 Visualization Tools

### Interactive Visualizations

```python
from src.visualization_tools import SOFCDataVisualizer

# Initialize visualizer
viz = SOFCDataVisualizer('data')

# Generate statistical summary
viz.generate_statistical_summary()

# Visualize simulation fields
viz.visualize_simulation_fields(sim_id=0)

# Create 3D field visualization
viz.create_3d_field_visualization(sim_id=0, field='temperature')

# Analyze experimental data
viz.analyze_experimental_data()

# Visualize EIS evolution
viz.visualize_eis_data()

# Display thermal images
viz.visualize_thermal_images()

# Analyze monitoring streams
viz.analyze_monitoring_streams()
```

## 🎯 Research Applications

### 1. Multi-Fidelity Modeling
- Train low-fidelity surrogate models on simulation data
- Calibrate with high-fidelity experimental measurements
- Develop transfer learning approaches

### 2. Degradation Prediction
- Track voltage degradation over time
- Monitor resistance evolution via EIS
- Detect damage accumulation through AE events

### 3. Adaptive Monitoring
- Real-time state estimation
- Dynamic model updating
- Anomaly detection and triggering

### 4. Failure Prognosis
- Stress concentration prediction
- Crack propagation modeling
- Remaining useful life estimation

## 📊 Dataset Statistics

| Category | Count | Description |
|----------|-------|-------------|
| Simulations | 100 | Multi-physics simulations with parameter sweeps |
| Grid Points | 2000 | Per simulation (20×20×5) |
| Time Series | 10,000 | Operational data points |
| EIS Spectra | 50 | Impedance measurements |
| Thermal Images | 100 | 32×32 pixel thermal maps |
| Strain Sensors | 8 | Continuous monitoring |
| AE Events | 54 | Damage indicators |
| Stream Data | 3600 | Real-time monitoring points |

## 🔧 Advanced Features

### Custom Data Generation

Modify parameters in `generate_dataset_simple.py`:
```python
# Adjust simulation count
n_simulations = 200  # Increase for more data

# Modify parameter ranges
param_ranges = {
    'current_density': (500, 15000),  # Wider range
    # ... adjust other parameters
}

# Change grid resolution
grid_size = (40, 40, 10)  # Higher resolution
```

### Data Augmentation

The dataset supports various augmentation strategies:
- Noise injection for robustness
- Interpolation for higher time resolution
- Field rotation/flipping for spatial invariance

## 📚 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sofc_digital_twin_2024,
  title={SOFC Digital Twin Multi-Fidelity Dataset},
  subtitle={Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring},
  year={2024},
  version={1.0},
  description={Multi-physics, multi-fidelity dataset for SOFC digital twin research}
}
```

## 🤝 Contributing

To extend or improve the dataset:

1. Modify simulation parameters in `sofc_dataset_generator.py`
2. Add new measurement types in experimental data
3. Implement additional degradation models
4. Create new visualization functions

## 📄 License

This dataset is provided for research purposes. Please ensure proper attribution when using in publications or presentations.

## 🆘 Support

For questions or issues:
1. Check the example scripts in `src/`
2. Review the visualization outputs
3. Examine the metadata files for parameter descriptions

## 🚀 Next Steps

1. **Explore the Data**: Use visualization tools to understand patterns
2. **Train Models**: Implement PINNs or surrogate models
3. **Validate**: Compare predictions with experimental data
4. **Deploy**: Test adaptive monitoring strategies
5. **Extend**: Add your own physics models or data sources

---

*This dataset represents a comprehensive foundation for SOFC digital twin research, combining multi-physics simulation with realistic experimental scenarios for advancing predictive maintenance and integrity monitoring technologies.*