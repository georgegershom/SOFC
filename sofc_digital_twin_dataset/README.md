# SOFC Digital Twin Dataset
## Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring

### Overview

This comprehensive dataset supports the development of adaptive-scale physics-informed digital twins for Solid Oxide Fuel Cell (SOFC) thermo-structural integrity monitoring. The dataset follows the "Data-Model Fusion" Trinity approach, providing interconnected data across multiple scales and physics domains.

### Dataset Structure

The dataset is organized into five main categories:

#### 1. Materials & Geometry Data (`1_materials_geometry/`)
- **Microstructural Data**: 3D tomography-like data simulating FIB-SEM or X-ray nano-CT
  - Anode (Ni-YSZ) microstructure with percolating networks
  - Dense electrolyte (8YSZ) structure
  - Cathode (LSM-YSZ) composite microstructure
  - Effective property calculations (conductivity, porosity, TPB density)

- **Macro-scale Geometry**: CAD-like geometry and FEM mesh data
  - Cell dimensions and layer thicknesses
  - Flow field patterns and channel geometry
  - Structured hexahedral mesh for multi-physics simulations

#### 2. Operational & Electrochemical Performance Data (`2_operational_electrochemical/`)
- **Controlled Input Parameters**: Time-series operational conditions
  - Fuel composition (H₂, CO, CH₄, H₂O, CO₂) variations
  - Flow rates and utilization factors
  - Temperature and pressure conditions
  - Load profiles and current density variations

- **Electrochemical Response Data**: Performance measurements
  - Cell voltage evolution with degradation
  - Overpotential breakdown (activation, ohmic, concentration)
  - EIS spectra at various operating points and aging states
  - Power density calculations

#### 3. Thermo-Structural Field Data (`3_thermo_structural/`)
- **Temperature Fields**: Spatially and temporally resolved thermal data
  - 2D temperature distributions from IR camera simulation
  - Thermocouple point measurements at critical locations
  - Heat generation patterns and cooling effects

- **Stress & Strain Fields**: Mechanical integrity monitoring data
  - Von Mises stress distributions from FEM simulation
  - Strain gauge measurements at key locations
  - DIC (Digital Image Correlation) full-field strain data
  - Fracture risk assessment and safety factor evolution

#### 4. Degradation & Failure Mode Data (`4_degradation_failure/`)
- **Accelerated Aging Tests**: Multiple degradation mechanisms
  - Thermal cycling (delamination and cracking)
  - Redox cycling (anode degradation)
  - Steady-state aging (poisoning and coarsening)
  - High current stress testing

- **Post-Mortem Analysis**: Failure characterization
  - SEM/EDS microscopy data for different failure scenarios
  - Failure analysis reports with root cause identification
  - Degradation fingerprints for pattern recognition

#### 5. Synthesis Workflows (`5_synthesis_workflows/`)
- **High-Fidelity Model Training Data**: Physics-based model datasets
- **ROM Training Data**: Reduced-order model generation
- **Data Assimilation Workflows**: Real-time state estimation
- **Calibration Procedures**: Sensor and model calibration protocols

### Data Formats

- **HDF5 (.h5)**: Multi-dimensional arrays, field data, and large datasets
- **CSV (.csv)**: Time-series data and tabular datasets
- **JSON (.json)**: Configuration files, metadata, and structured parameters
- **PNG (.png)**: Visualizations and analysis plots

### Key Features

1. **Multi-Scale Integration**: Data spans from nano-scale microstructure to system-scale performance
2. **Multi-Physics Coupling**: Thermal, electrochemical, and mechanical phenomena
3. **Realistic Degradation**: Multiple failure modes with characteristic fingerprints
4. **Uncertainty Quantification**: Measurement noise and model uncertainty included
5. **Real-Time Compatibility**: Data assimilation and calibration workflows
6. **Comprehensive Documentation**: Detailed metadata and usage examples

### Usage Examples

See `documentation/usage_examples.py` for detailed code examples including:
- Loading and visualizing microstructural data
- Analyzing electrochemical performance trends
- Processing temperature and stress field data
- Implementing data assimilation workflows
- Training reduced-order models

### Data Quality and Validation

All synthetic data is generated using physically realistic models and validated against:
- Literature values for material properties
- Experimental trends from SOFC research
- Physics-based constraints and conservation laws
- Statistical consistency across datasets

### Citation

If you use this dataset in your research, please cite:

```
SOFC Digital Twin Dataset: Adaptive-Scale Physics-Informed Digital Twin for 
SOFC Thermo-Structural Integrity Monitoring. Generated 2025.
```

### Contact and Support

For questions, issues, or contributions, please refer to the documentation or contact the dataset maintainers.

### License

This dataset is provided for research and educational purposes. Please refer to the license file for detailed terms and conditions.
