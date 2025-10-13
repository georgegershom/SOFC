# SOFC Adaptive-Scale Physics-Informed Digital Twin Dataset

## Overview

This repository contains a comprehensive dataset generation framework for **Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring**. The framework implements a multi-fidelity, multi-physics approach that blends high-fidelity simulation data with experimental measurements to create a robust foundation for physics-informed machine learning models.

## Core Philosophy: Multi-Fidelity & Multi-Physics

The dataset architecture follows a three-tier approach:

1. **Dataset 1**: High-fidelity physics-based simulation data (abundant, computationally expensive)
2. **Dataset 2**: Experimental validation data (scarce, expensive, but ground truth)
3. **Dataset 3**: Real-time monitoring data (adaptive sampling for digital twin operation)

## Repository Structure

```
sofc_digital_twin/
├── config/
│   └── simulation_config.yaml          # Configuration parameters
├── datasets/                           # Generated datasets
│   ├── dataset1_physics_simulation/    # High-fidelity simulation data
│   ├── dataset2_experimental/          # Experimental validation data
│   └── dataset3_realtime/             # Real-time monitoring data
├── src/
│   ├── data_generation/               # Data generation scripts
│   │   ├── dataset1_generator.py      # Physics simulation generator
│   │   ├── dataset2_generator.py      # Experimental data generator
│   │   └── dataset3_generator.py      # Real-time data generator
│   ├── physics_models/                # Multi-physics models
│   │   └── sofc_physics.py           # Coupled SOFC physics model
│   └── utils/                         # Processing and visualization
│       ├── data_processor.py          # Data loading and analysis
│       └── visualizer.py             # Advanced visualization tools
├── examples/                          # Usage examples
├── docs/                             # Documentation
└── tests/                            # Unit tests
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sofc_digital_twin
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Generate Datasets

#### Dataset 1: High-Fidelity Physics Simulation

```bash
cd sofc_digital_twin
python src/data_generation/dataset1_generator.py
```

This generates comprehensive multi-physics simulation data including:
- **Electrochemical fields**: Current density, potential, species concentrations
- **Thermal fields**: Temperature distribution, heat generation, heat flux
- **Structural fields**: Displacement, stress/strain tensors, von Mises stress
- **Failure metrics**: Stress intensity factors, strain energy density

#### Dataset 2: Experimental Validation Data

```bash
python src/data_generation/dataset2_generator.py
```

This creates synthetic experimental data mimicking:
- **Global operational data**: I-V curves, temperatures, flow rates
- **EIS measurements**: Impedance spectroscopy for degradation diagnosis
- **Thermal imaging**: 2D temperature maps
- **Strain measurements**: Point strain measurements at critical locations
- **Acoustic emission**: Crack detection and propagation events
- **Post-mortem analysis**: SEM, X-ray tomography data

#### Dataset 3: Real-Time Monitoring Data

```bash
python src/data_generation/dataset3_generator.py
```

This simulates adaptive real-time monitoring including:
- **High-frequency operational data**: 1 Hz sampling of I, V, T, F
- **Adaptive sampling**: Increased frequency during critical events
- **Event-driven measurements**: EIS and thermal imaging triggered by anomalies
- **Acoustic emission monitoring**: Real-time crack detection

### Data Processing and Analysis

```python
from src.utils.data_processor import SOFCDataProcessor
from src.utils.visualizer import SOFCVisualizer

# Initialize processor
processor = SOFCDataProcessor()

# Load and analyze Dataset 1
simulations = processor.load_dataset1_batch(sim_ids=[0, 1, 2, 3, 4])
stats = processor.analyze_dataset1_statistics(simulations)

# Extract features for ML training
features, targets = processor.extract_features_dataset1(simulations)

# Create visualizations
processor.create_dataset1_visualizations(simulations)

# Advanced visualization
visualizer = SOFCVisualizer()
visualizer.create_comprehensive_report(
    simulation_data=simulations[0],
    experimental_data=processor.load_dataset2_experimental(),
    realtime_data=processor.load_dataset3_realtime()
)
```

## Dataset Details

### Dataset 1: High-Fidelity Physics Simulation

**Parameters Varied** (Latin Hypercube Sampling):

- **Operating Conditions**:
  - Current density: 0.1-1.5 A/cm²
  - Fuel utilization: 30-90%
  - Air utilization: 10-30%
  - Inlet temperatures: 600-900°C
  - Fuel composition: H₂, H₂O, CO, CH₄ mixtures

- **Material Properties**:
  - Electrode porosity: 0.3-0.6
  - Electrode tortuosity: 2.0-5.0
  - Conductivities: Variable ionic/electronic
  - Layer thicknesses: 10-200 μm

- **Degradation States**:
  - Initial crack lengths: 0-100 μm
  - Porosity degradation: 0-30% loss

**Output Fields** (3D spatial distributions):
- Temperature T(x,y,z)
- Current density i(x,y,z)
- Species concentrations (H₂, H₂O, O₂)
- Displacement vector U(x,y,z)
- Stress tensor σ(x,y,z)
- Strain tensor ε(x,y,z)
- Von Mises stress
- Strain energy density

### Dataset 2: Experimental Validation

**Global Operational Data**:
- Time-series: I(t), V(t), P(t)
- Temperatures: Inlet/outlet fuel and air
- Flow rates: Fuel and air
- Duration: 720 hours (30 days)

**Electrochemical Impedance Spectroscopy**:
- Frequency range: 0.01 Hz - 100 kHz
- Measurements: Every 24 hours
- Parameters: R_ohmic, R_activation, R_concentration

**In-Situ Measurements**:
- **Thermal imaging**: 2D temperature maps (64×48 pixels)
- **Strain gauges**: Point measurements at 4 locations
- **Acoustic emission**: Event detection with amplitude, duration, location

**Post-Mortem Analysis**:
- **SEM analysis**: Crack locations, delamination areas, porosity changes
- **X-ray tomography**: 3D microstructure (512×512×256 voxels)
- **Elemental analysis**: Chromium migration, nickel depletion

### Dataset 3: Real-Time Monitoring

**High-Frequency Stream** (1 Hz):
- Current density, voltage, power
- Inlet temperatures and flow rates
- Fuel/air utilization
- Degradation state indicator

**Adaptive Sampling**:
- Increased frequency during voltage drops
- Triggered EIS during instability
- Thermal imaging during temperature excursions

**Event-Driven Data**:
- Acoustic emission events with classification
- Severity assessment (low/medium/high)
- Spatial localization

## Physics Models

### Electrochemical Model
- Butler-Volmer kinetics for electrode reactions
- Charge conservation: ∇·(σ∇φ) = 0
- Species transport with consumption

### Thermal Model
- Heat conduction with generation: ∇·(k∇T) + q = 0
- Electrochemical heat generation: q = i·η
- Convective boundary conditions

### Structural Model
- Linear elasticity with thermal expansion
- Stress-strain relationships: σ = C:ε
- Thermal strain: ε_th = α(T - T_ref)

## Data Format

### HDF5 Structure
```
simulation_XXXXXX.h5
├── fields/
│   ├── temperature          [nx, ny, nz]
│   ├── current_density      [nx, ny, nz]
│   ├── stress_xx           [nx, ny, nz]
│   └── ...
├── parameters/
│   ├── operating_conditions/
│   └── material_properties/
├── derived_quantities/
│   ├── cell_voltage
│   ├── max_temperature
│   └── max_von_mises_stress
└── metadata/
    ├── simulation_id
    ├── timestamp
    └── geometry/
```

### CSV Format (Time-Series Data)
```csv
timestamp,time_hours,current_density,voltage,power,temp_fuel_inlet,...
2024-01-01 00:00:00,0.0,0.8,0.75,0.6,800,...
2024-01-01 00:00:01,0.000278,0.81,0.749,0.607,801,...
```

## Machine Learning Integration

### Feature Engineering
```python
# Extract features for ML training
features, targets = processor.extract_features_dataset1(simulations)

# Features (17D): Operating conditions + material properties
# Targets (11D): Derived quantities + field statistics

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

### Physics-Informed Neural Networks (PINNs)
The dataset is designed for training PINNs with physics constraints:

```python
# Physics loss terms
def physics_loss(model_output, inputs):
    # Charge conservation
    charge_residual = compute_charge_conservation(model_output, inputs)
    
    # Energy conservation  
    energy_residual = compute_energy_conservation(model_output, inputs)
    
    # Momentum conservation
    momentum_residual = compute_momentum_conservation(model_output, inputs)
    
    return charge_residual + energy_residual + momentum_residual
```

## Digital Twin Workflow

### 1. Offline Training
```python
# Train surrogate model on Dataset 1
model = train_physics_informed_model(features, targets, physics_constraints)
```

### 2. Online Adaptation
```python
# Real-time state estimation
current_state = model.predict(current_operating_conditions)

# Data assimilation with experimental measurements
updated_state = kalman_filter_update(current_state, measurements)

# Adapt model parameters
model.update_parameters(updated_state)
```

### 3. Integrity Monitoring
```python
# Predict internal stress fields
stress_field = model.predict_stress_field(current_conditions)

# Failure assessment
rul = failure_criterion(stress_field, material_properties)
```

## Validation and Benchmarking

### Model Validation
- **Dataset 1**: Cross-validation on simulation data
- **Dataset 2**: Validation against experimental measurements
- **Dataset 3**: Real-time prediction accuracy

### Performance Metrics
- **Accuracy**: RMSE, MAE for field predictions
- **Physics consistency**: Residual of governing equations
- **Computational efficiency**: Inference time vs. accuracy trade-off

## Applications

### Research Applications
- Physics-informed machine learning development
- Multi-fidelity modeling techniques
- Digital twin architectures
- Degradation mechanism studies

### Industrial Applications
- Real-time SOFC monitoring
- Predictive maintenance
- Optimal operation strategies
- Lifetime prediction

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sofc_digital_twin_2024,
  title={Adaptive-Scale Physics-Informed Digital Twin Dataset for SOFC Thermo-Structural Integrity Monitoring},
  author={[Your Name]},
  year={2024},
  publisher={[Institution]},
  version={1.0}
}
```

## License

This dataset is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Support

For questions and support:
- Create an issue on GitHub
- Email: [your-email@institution.edu]
- Documentation: [link-to-docs]

## Acknowledgments

This work was supported by [funding sources]. We thank [collaborators] for their contributions to the experimental validation data.