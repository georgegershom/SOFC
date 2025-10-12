# Stratified Flow Simulation Dataset

## Overview
This dataset contains comprehensive simulation data for the PhD thesis "Study on the Attenuation Mechanisms in Stratified Flows: 2. Simulation Data (For Model Development and Hypothesis Testing)".

## Data Categories

### 1. CFD Model Outputs
- **2D/3D velocity fields**: Complete velocity field data in HDF5 format
- **Pressure fields**: Hydrostatic and dynamic pressure distributions
- **VOF contours**: Volume of Fluid phase distribution
- **Turbulence parameters**: k, ε, eddy viscosity from various models
- **Acoustic pressure propagation**: Time-series acoustic data

### 2. Mathematical Model Outputs
- **Predicted sound speed**: Using Wood's equation and modified models
- **Attenuation coefficients**: Frequency-dependent attenuation
- **Wave propagation patterns**: Reflection and transmission at interfaces
- **Time-delay estimates**: Acoustic propagation time delays

### 3. Model Validation Data
- **Sound speed validation**: Simulated vs experimental comparison
- **Attenuation validation**: Model vs measurement comparison
- **Waveform validation**: Acoustic signal comparison

## File Structure
```
stratified_flow_thesis_data/
├── cfd_data/           # Main CFD simulation data
├── acoustic_data/      # Acoustic analysis results
├── mathematical_models/ # Mathematical model outputs
├── validation_data/    # Model validation data
├── visualizations/     # Generated plots and figures
└── raw_data/          # Raw simulation data
```

## Usage
- **HDF5 files**: Use h5py in Python or HDFView for visualization
- **VTK files**: Open in ParaView for 3D visualization
- **CSV files**: Import into Excel, MATLAB, or Python for analysis
- **NPZ files**: Load in Python using numpy.load()

## Software/Methods Used
- **CFD**: ANSYS Fluent/CFX, COMSOL Multiphysics
- **Turbulence Models**: k-ε, k-ω SST, LES
- **Multiphase**: VOF, Euler-Euler
- **Mathematical**: MATLAB, Python, Transfer-matrix method
- **Acoustic**: Modified wave equations (Lighthill, FW-H)

## Contact
For questions about this dataset, please refer to the PhD thesis documentation.
