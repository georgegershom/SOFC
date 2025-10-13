# SOFC Digital Twin Dataset Specification

## Overview

This document provides detailed specifications for the three datasets generated for the Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring project.

## Dataset Architecture

The dataset follows a multi-fidelity approach with three complementary datasets:

1. **Dataset 1**: High-fidelity physics-based simulation data
2. **Dataset 2**: Experimental validation and ground truth data  
3. **Dataset 3**: Real-time adaptive monitoring data

## Dataset 1: High-Fidelity Physics Simulation Data

### Purpose
Provides comprehensive multi-physics simulation data for training physics-informed machine learning models. This dataset forms the foundation of the digital twin's physics knowledge.

### Data Generation Method
- **Solver**: Finite element/finite difference methods
- **Physics Models**: Coupled electrochemical-thermal-structural
- **Parameter Sampling**: Latin Hypercube Sampling for efficient parameter space coverage
- **Mesh Resolution**: 50×30×20 elements (configurable)

### Input Parameters

#### Operating Conditions
| Parameter | Range | Units | Samples | Description |
|-----------|--------|-------|---------|-------------|
| Current Density | 0.1 - 1.5 | A/cm² | 20 | Primary electrical load |
| Fuel Utilization | 0.3 - 0.9 | - | 15 | Fraction of fuel consumed |
| Air Utilization | 0.1 - 0.3 | - | 10 | Fraction of air consumed |
| Inlet Fuel Temperature | 700 - 900 | °C | 15 | Fuel inlet temperature |
| Inlet Air Temperature | 600 - 800 | °C | 12 | Air inlet temperature |

#### Fuel Composition
| Component | Range | Units | Samples | Description |
|-----------|--------|-------|---------|-------------|
| H₂ Percentage | 0.4 - 0.8 | - | 8 | Hydrogen fraction |
| H₂O Percentage | 0.1 - 0.4 | - | 6 | Water vapor fraction |
| CO Percentage | 0.05 - 0.2 | - | 5 | Carbon monoxide fraction |
| CH₄ Percentage | 0.0 - 0.1 | - | 3 | Methane fraction |

#### Material Properties
| Parameter | Range | Units | Samples | Description |
|-----------|--------|-------|---------|-------------|
| Electrode Porosity | 0.3 - 0.6 | - | 10 | Void fraction in electrodes |
| Electrode Tortuosity | 2.0 - 5.0 | - | 8 | Pore connectivity factor |
| Anode Conductivity | 1000 - 5000 | S/m | 10 | Ionic conductivity |
| Cathode Conductivity | 100 - 1000 | S/m | 10 | Electronic conductivity |
| Electrolyte Thickness | 10 - 50 | μm | 8 | Electrolyte layer thickness |
| Electrode Thickness | 50 - 200 | μm | 10 | Electrode layer thickness |

#### Degradation Parameters
| Parameter | Range | Units | Samples | Description |
|-----------|--------|-------|---------|-------------|
| Initial Crack Length | 0 - 100 | μm | 15 | Pre-existing crack size |
| Porosity Degradation | 0 - 0.3 | - | 10 | Fractional porosity loss |

### Output Fields

#### Electrochemical Fields (3D)
- **Potential**: φ(x,y,z) [V]
- **Current Density**: i(x,y,z) [A/m²]
- **Overpotential**: η(x,y,z) [V]
- **Species Concentrations**: 
  - H₂(x,y,z) [mol/m³]
  - H₂O(x,y,z) [mol/m³]
  - O₂(x,y,z) [mol/m³]

#### Thermal Fields (3D)
- **Temperature**: T(x,y,z) [K]
- **Heat Generation**: q(x,y,z) [W/m³]
- **Heat Flux**: q⃗(x,y,z) [W/m²]

#### Structural Fields (3D)
- **Displacement Vector**: U⃗(x,y,z) [m]
  - Ux(x,y,z), Uy(x,y,z), Uz(x,y,z)
- **Stress Tensor**: σ(x,y,z) [Pa]
  - σxx, σyy, σzz, σxy, σyz, σzx
- **Strain Tensor**: ε(x,y,z) [-]
  - εxx, εyy, εzz, εxy, εyz, εzx
- **Von Mises Stress**: σvm(x,y,z) [Pa]
- **Strain Energy Density**: W(x,y,z) [J/m³]

#### Derived Quantities
- **Cell Voltage**: V [V]
- **Maximum Temperature**: Tmax [K]
- **Maximum von Mises Stress**: σvm,max [Pa]
- **Field Statistics**: Mean, std, min, max for key fields

### File Format

#### HDF5 Structure
```
simulation_XXXXXX.h5
├── fields/                          # 3D field data
│   ├── temperature          [nx,ny,nz] float64
│   ├── current_density      [nx,ny,nz] float64
│   ├── potential           [nx,ny,nz] float64
│   ├── displacement_x      [nx,ny,nz] float64
│   ├── displacement_y      [nx,ny,nz] float64
│   ├── displacement_z      [nx,ny,nz] float64
│   ├── stress_xx           [nx,ny,nz] float64
│   ├── stress_yy           [nx,ny,nz] float64
│   ├── stress_zz           [nx,ny,nz] float64
│   ├── stress_xy           [nx,ny,nz] float64
│   ├── stress_yz           [nx,ny,nz] float64
│   ├── stress_zx           [nx,ny,nz] float64
│   ├── strain_xx           [nx,ny,nz] float64
│   ├── strain_yy           [nx,ny,nz] float64
│   ├── strain_zz           [nx,ny,nz] float64
│   ├── strain_xy           [nx,ny,nz] float64
│   ├── strain_yz           [nx,ny,nz] float64
│   ├── strain_zx           [nx,ny,nz] float64
│   ├── von_mises_stress    [nx,ny,nz] float64
│   ├── strain_energy_density [nx,ny,nz] float64
│   ├── h2_concentration    [nx,ny,nz] float64
│   ├── h2o_concentration   [nx,ny,nz] float64
│   └── o2_concentration    [nx,ny,nz] float64
├── parameters/                      # Input parameters
│   ├── operating_conditions/
│   │   ├── current_density         attr: float64
│   │   ├── fuel_utilization        attr: float64
│   │   ├── air_utilization         attr: float64
│   │   ├── inlet_fuel_temperature  attr: float64
│   │   ├── inlet_air_temperature   attr: float64
│   │   └── fuel_composition/
│   │       ├── h2_percentage       attr: float64
│   │       ├── h2o_percentage      attr: float64
│   │       ├── co_percentage       attr: float64
│   │       └── ch4_percentage      attr: float64
│   └── material_properties/
│       ├── electrode_porosity      attr: float64
│       ├── electrode_tortuosity    attr: float64
│       ├── anode_conductivity      attr: float64
│       ├── cathode_conductivity    attr: float64
│       ├── electrolyte_thickness   attr: float64
│       ├── electrode_thickness     attr: float64
│       ├── initial_crack_length    attr: float64
│       └── porosity_degradation    attr: float64
├── derived_quantities/              # Computed results
│   ├── cell_voltage                attr: float64
│   ├── max_temperature             attr: float64
│   └── max_von_mises_stress        attr: float64
└── metadata/                        # Simulation metadata
    ├── simulation_id               attr: int32
    ├── timestamp                   attr: float64
    └── geometry/
        ├── nx                      attr: int32
        ├── ny                      attr: int32
        ├── nz                      attr: int32
        ├── length                  attr: float64
        ├── width                   attr: float64
        └── height                  attr: float64
```

## Dataset 2: Experimental Validation Data

### Purpose
Provides experimental measurements for digital twin validation and model adaptation. Represents "ground truth" data that would be available from actual SOFC test rigs.

### Data Components

#### Global Operational Data
**File**: `operational_data.csv`
**Frequency**: 1 Hz
**Duration**: 720 hours (30 days)

| Column | Units | Description |
|--------|-------|-------------|
| timestamp | ISO 8601 | Measurement timestamp |
| time_hours | hours | Elapsed time |
| current_density | A/cm² | Applied current density |
| voltage | V | Cell voltage |
| power | W/cm² | Power output |
| temp_fuel_inlet | °C | Fuel inlet temperature |
| temp_air_inlet | °C | Air inlet temperature |
| temp_fuel_outlet | °C | Fuel outlet temperature |
| temp_air_outlet | °C | Air outlet temperature |
| flow_fuel | sccm | Fuel flow rate |
| flow_air | sccm | Air flow rate |

#### Electrochemical Impedance Spectroscopy (EIS)
**File**: `experimental_data.h5:/eis_data/`
**Frequency**: Every 24 hours
**Frequency Range**: 0.01 Hz - 100 kHz (50 points)

```
measurement_XXX/
├── test_time_hours         dataset: float64
├── frequencies            dataset: [50] float64
├── Z_real                 dataset: [50] float64
├── Z_imag                 dataset: [50] float64
├── R_ohmic                dataset: float64
├── R_activation           dataset: float64
└── R_concentration        dataset: float64
```

#### Thermal Imaging Data
**File**: `experimental_data.h5:/thermal_imaging/`
**Frequency**: During thermal transients
**Resolution**: 64×48 pixels

```
image_XXX/
├── test_time_hours        dataset: float64
├── temperature_map        dataset: [48,64] float64
├── max_temperature        dataset: float64
├── min_temperature        dataset: float64
├── mean_temperature       dataset: float64
└── image_dimensions       dataset: [2] int32
```

#### Strain Gauge Measurements
**File**: `strain_gauge_data.csv`
**Frequency**: 1/60 Hz (every minute)
**Locations**: 4 critical locations

| Column | Units | Description |
|--------|-------|-------------|
| timestamp | ISO 8601 | Measurement timestamp |
| time_hours | hours | Elapsed time |
| strain_interconnect_center | με | Strain at interconnect center |
| strain_seal_edge | με | Strain at seal edge |
| strain_current_collector | με | Strain at current collector |
| strain_frame_corner | με | Strain at frame corner |

#### Acoustic Emission Events
**File**: `experimental_data.h5:/acoustic_emission/`
**Frequency**: Event-driven

```
event_XXXX/
├── test_time_hours        attr: float64
├── amplitude_db           attr: float64
├── duration_us            attr: float64
├── energy                 attr: float64
├── frequency_peak_khz     attr: float64
├── x_location             attr: float64
├── y_location             attr: float64
└── event_type             attr: string
```

#### Post-Mortem Analysis
**File**: `experimental_data.h5:/postmortem_analysis/`

##### SEM Analysis
```
sem_analysis/
├── crack_locations/
│   ├── item_0/
│   │   ├── x                      attr: float64
│   │   ├── y                      attr: float64
│   │   ├── length_um              attr: float64
│   │   └── width_um               attr: float64
│   └── ...
├── delamination_areas/
├── porosity_measurements/
└── elemental_analysis/
```

##### X-ray Tomography
```
xray_tomography/
├── voxel_size_um              attr: float64
├── image_dimensions           attr: [3] int32
├── crack_volume_fraction      attr: float64
├── pore_size_distribution/
└── interconnectivity/
```

## Dataset 3: Real-Time Monitoring Data

### Purpose
Simulates real-time data streams for digital twin operation, including adaptive sampling and event-driven measurements.

### Data Components

#### High-Frequency Operational Stream
**File**: `realtime_operational_data.csv`
**Frequency**: 1 Hz (adaptive)
**Duration**: Configurable (default 168 hours)

| Column | Units | Description |
|--------|-------|-------------|
| timestamp | ISO 8601 | Measurement timestamp |
| simulation_time_hours | hours | Simulation time |
| current_density | A/cm² | Applied current density |
| voltage | V | Cell voltage |
| power | W/cm² | Power output |
| temp_fuel_inlet | °C | Fuel inlet temperature |
| temp_air_inlet | °C | Air inlet temperature |
| flow_fuel | sccm | Fuel flow rate |
| flow_air | sccm | Air flow rate |
| fuel_utilization | - | Fuel utilization |
| air_utilization | - | Air utilization |
| degradation_state | - | Degradation indicator (0-1) |

#### Adaptive EIS Measurements
**File**: `realtime_monitoring.h5:/eis_measurements/`
**Frequency**: Adaptive (baseline 24h, triggered by anomalies)

Structure similar to Dataset 2 EIS, with additional:
```
measurement_XXX/
├── degradation_state          dataset: float64
└── trigger_reason             attr: string
```

#### Adaptive Thermal Imaging
**File**: `realtime_monitoring.h5:/thermal_measurements/`
**Frequency**: Adaptive (baseline 6h, triggered by temperature excursions)

Structure similar to Dataset 2 thermal, with additional:
```
image_XXX/
├── hotspot_detected           dataset: bool
├── degradation_state          dataset: float64
└── trigger_reason             attr: string
```

#### Acoustic Emission Events
**File**: `acoustic_emission_events.csv`
**Frequency**: Event-driven

| Column | Units | Description |
|--------|-------|-------------|
| timestamp | ISO 8601 | Event timestamp |
| simulation_time_hours | hours | Simulation time |
| amplitude_db | dB | Event amplitude |
| duration_us | μs | Event duration |
| energy | - | Event energy |
| frequency_peak_khz | kHz | Peak frequency |
| event_type | - | Event classification |
| severity | - | Severity level |
| x_location | - | Normalized x location |
| y_location | - | Normalized y location |
| degradation_state | - | Current degradation state |

### Adaptive Sampling Logic

#### Trigger Conditions
1. **Voltage Degradation**: Rate > 1×10⁻⁴ V/hour
2. **Voltage Instability**: Standard deviation > 0.02 V
3. **Temperature Excursion**: Standard deviation > 15°C
4. **High Degradation**: Degradation state > 0.7

#### Adaptive Responses
- **Increased Sampling**: Up to 3× baseline frequency
- **Triggered EIS**: Immediate impedance measurement
- **Triggered Thermal**: Immediate thermal imaging
- **Alert Level**: Normal/Caution/Warning/Critical

## Data Quality and Validation

### Simulation Data Quality
- **Physics Consistency**: All simulations satisfy governing PDEs within tolerance
- **Mass Conservation**: Species mass balance maintained
- **Energy Conservation**: Energy balance maintained
- **Mesh Independence**: Results converged with respect to mesh size

### Experimental Data Realism
- **Noise Models**: Realistic measurement noise added
- **Sensor Limitations**: Bandwidth and resolution limits modeled
- **Degradation Trends**: Based on literature degradation rates
- **Event Statistics**: Acoustic emission frequency matches experimental observations

### Data Integrity Checks
- **Range Validation**: All values within physically reasonable bounds
- **Temporal Consistency**: Smooth temporal evolution
- **Correlation Checks**: Expected correlations between variables maintained
- **Missing Data**: No missing values in critical fields

## Usage Guidelines

### Loading Data
```python
from src.utils.data_processor import SOFCDataProcessor

processor = SOFCDataProcessor()

# Load Dataset 1
simulations = processor.load_dataset1_batch(sim_ids=[0, 1, 2])
features, targets = processor.extract_features_dataset1(simulations)

# Load Dataset 2
operational_data = processor.load_dataset2_operational()
experimental_data = processor.load_dataset2_experimental()

# Load Dataset 3
realtime_data = processor.load_dataset3_realtime()
```

### Machine Learning Applications
- **Feature Engineering**: Extract relevant features from multi-physics fields
- **Target Selection**: Choose appropriate targets for specific applications
- **Physics Constraints**: Incorporate governing equations as loss terms
- **Multi-Fidelity Training**: Combine all three datasets for robust models

### Digital Twin Workflow
1. **Offline Training**: Train surrogate models on Dataset 1
2. **Validation**: Validate against Dataset 2 experimental data
3. **Online Operation**: Use Dataset 3 for real-time state estimation
4. **Model Adaptation**: Update model parameters based on measurements

## File Size Estimates

### Dataset 1
- **Per Simulation**: ~50 MB (uncompressed), ~15 MB (compressed)
- **1000 Simulations**: ~15 GB (compressed)

### Dataset 2
- **Operational Data**: ~500 MB
- **EIS Data**: ~10 MB
- **Thermal Images**: ~100 MB
- **Other Data**: ~50 MB
- **Total**: ~660 MB

### Dataset 3
- **Operational Stream**: ~100 MB (1 week)
- **Adaptive Measurements**: ~50 MB
- **Events**: ~10 MB
- **Total**: ~160 MB (1 week)

## Computational Requirements

### Generation
- **Dataset 1**: High (physics simulations)
- **Dataset 2**: Low (synthetic data)
- **Dataset 3**: Medium (real-time simulation)

### Processing
- **Memory**: 8-16 GB RAM recommended
- **Storage**: 20+ GB for full datasets
- **CPU**: Multi-core recommended for parallel processing

## Future Extensions

### Additional Physics
- **Electrochemical Aging**: Detailed degradation mechanisms
- **Microstructural Evolution**: Grain growth, phase changes
- **Gas Transport**: Detailed porous media transport

### Enhanced Measurements
- **Distributed Sensors**: More measurement locations
- **Advanced Diagnostics**: Additional characterization techniques
- **Multi-Scale Data**: Coupling macro and micro-scale measurements

### Uncertainty Quantification
- **Parameter Uncertainty**: Probabilistic parameter distributions
- **Model Uncertainty**: Multiple physics model variants
- **Measurement Uncertainty**: Detailed uncertainty propagation