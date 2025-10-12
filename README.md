# Stratified Flow Simulation Dataset Generator

## Overview

This repository contains a comprehensive dataset generator for PhD thesis research on "Study on the Attenuation Mechanisms in Stratified Flows: 2. Simulation Data (For Model Development and Hypothesis Testing)". The generator creates realistic simulation data using both CFD and mathematical models for stratified air-water flows.

## Features

### CFD Model Outputs
- **2D/3D velocity fields** with stratified flow conditions
- **Pressure fields** including hydrostatic and dynamic components
- **Volume of Fluid (VOF) contours** for phase distribution
- **Turbulence parameters** (k, ε, eddy viscosity) from multiple models
- **Acoustic pressure propagation** over time

### Mathematical Model Outputs
- **Predicted sound speed** using Wood's equation and modified models
- **Attenuation coefficients** with frequency dependence
- **Wave propagation patterns** including reflection and transmission
- **Time-delay estimates** for acoustic propagation

### Model Validation Data
- **Synthetic experimental data** with realistic uncertainties
- **Direct comparison files** for simulated vs experimental results
- **Statistical validation metrics** (R², RMSE, etc.)

## Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Generate Complete Dataset

```bash
python run_simulation.py
```

This will generate the complete dataset in the `stratified_flow_thesis_data/` directory.

### Generate Specific Components

#### CFD Simulation Data Only
```python
from stratified_flow_simulation import StratifiedFlowSimulator

simulator = StratifiedFlowSimulator('config.json')
simulator.run_complete_simulation('output_directory')
```

#### Acoustic Analysis Data
```python
from acoustic_models import generate_acoustic_analysis_data

acoustic_data = generate_acoustic_analysis_data()
```

#### Turbulence Model Data
```python
from turbulence_models import generate_turbulence_data

turbulence_data = generate_turbulence_data()
```

#### Experimental Validation Data
```bash
python experimental_validation.py
```

### Analyze Generated Data

```bash
python data_analysis.py
```

## Configuration

Edit `config.json` to customize simulation parameters:

```json
{
  "domain": {
    "x_length": 10.0,    // Domain length in x (m)
    "y_length": 2.0,     // Domain length in y (m)
    "z_length": 1.0,     // Domain length in z (m)
    "nx": 200,           // Grid points in x
    "ny": 40,            // Grid points in y
    "nz": 20             // Grid points in z
  },
  "fluids": {
    "density_1": 1000.0,      // Water density (kg/m³)
    "density_2": 1.2,         // Air density (kg/m³)
    "viscosity_1": 0.001,     // Water viscosity (Pa·s)
    "viscosity_2": 0.000018,  // Air viscosity (Pa·s)
    "sound_speed_1": 1500.0,  // Water sound speed (m/s)
    "sound_speed_2": 343.0    // Air sound speed (m/s)
  },
  "flow": {
    "velocity_1": 0.1,        // Water velocity (m/s)
    "velocity_2": 5.0,        // Air velocity (m/s)
    "interface_height": 0.5,  // Interface position (m)
    "turbulence_intensity": 0.05  // Turbulence intensity
  },
  "acoustic": {
    "frequency_range": [100, 10000],  // Frequency range (Hz)
    "source_position": [0.5, 1.0, 0.5],  // Source position (m)
    "receiver_positions": [[5.0, 1.0, 0.5], [8.0, 1.0, 0.5]],  // Receiver positions
    "time_duration": 1.0,     // Simulation time (s)
    "sampling_rate": 44100    // Sampling rate (Hz)
  }
}
```

## Output Data Formats

### HDF5 Files
- **Main CFD data**: `cfd_data.h5`
- **Efficient storage** for large 3D fields
- **Hierarchical structure** for easy navigation

### VTK Files
- **3D visualization**: `stratified_flow_fields.vtk`
- **Compatible with ParaView** and other visualization tools
- **Field data** for velocity, pressure, VOF, etc.

### CSV Files
- **Analysis data**: Various CSV files for easy import
- **Validation data**: Experimental comparison data
- **Statistical results**: Analysis metrics and summaries

### NPZ Files
- **NumPy arrays**: Compressed arrays for Python analysis
- **Acoustic data**: Frequency sweep results
- **Turbulence data**: Multiple model comparisons

## Data Structure

```
stratified_flow_thesis_data/
├── cfd_data/                    # Main CFD simulation data
│   ├── cfd_data.h5             # HDF5 format CFD data
│   ├── stratified_flow_fields.vtk  # VTK visualization
│   ├── *.csv                   # CSV analysis files
│   └── metadata.json           # Simulation metadata
├── acoustic_data/              # Acoustic analysis results
│   ├── acoustic_analysis.npz   # Frequency sweep data
│   └── *.png                   # Acoustic plots
├── mathematical_models/        # Mathematical model outputs
│   ├── mathematical_models.npz # Model predictions
│   └── transfer_matrix.npz     # Transfer matrix results
├── validation_data/            # Model validation data
│   ├── experimental_*.json     # Experimental data
│   └── experimental_*.csv      # CSV validation data
├── visualizations/             # Generated plots and figures
│   ├── *.png                   # Analysis plots
│   └── *.pdf                   # Publication-quality figures
└── raw_data/                   # Raw simulation data
    ├── turbulence_models.npz   # Turbulence model data
    └── *.npz                   # Other raw data
```

## Software/Methods Used

### CFD Software
- **ANSYS Fluent/CFX** (simulated)
- **COMSOL Multiphysics** (simulated)
- **k-ε, k-ω SST, LES** turbulence models
- **VOF** and **Euler-Euler** multiphase models

### Mathematical Software
- **MATLAB** (simulated)
- **Python** with NumPy, SciPy
- **Transfer-matrix method**
- **Modified wave equations** (Lighthill, FW-H)

### Analysis Tools
- **ParaView** for 3D visualization
- **MATLAB/Python** for data analysis
- **Statistical validation** tools

## Usage Examples

### Load and Analyze CFD Data

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load CFD data
with h5py.File('stratified_flow_thesis_data/cfd_data/cfd_data.h5', 'r') as f:
    velocity_x = f['fields/velocity_x'][:]
    pressure = f['fields/pressure'][:]
    vof = f['fields/vof'][:]

# Plot velocity field
plt.figure(figsize=(10, 6))
plt.contourf(velocity_x[:, :, 10], levels=20)
plt.colorbar(label='Velocity (m/s)')
plt.title('Velocity Field')
plt.show()
```

### Analyze Acoustic Data

```python
import numpy as np

# Load acoustic data
acoustic_data = np.load('stratified_flow_thesis_data/acoustic_data/acoustic_analysis.npz')

frequencies = acoustic_data['frequencies']
attenuation = acoustic_data['attenuation_coefficients']

# Plot attenuation vs frequency
plt.loglog(frequencies, attenuation[:, 50])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation Coefficient (Np/m)')
plt.title('Attenuation vs Frequency')
plt.show()
```

### Validate Models

```python
import pandas as pd
from data_analysis import StratifiedFlowAnalyzer

# Load analyzer
analyzer = StratifiedFlowAnalyzer('stratified_flow_thesis_data')

# Run validation
validation_results = analyzer.validate_models()

# Print results
for freq, results in validation_results['sound_speed'].items():
    print(f"Frequency {freq} Hz: R² = {results['r2_score']:.3f}")
```

## Customization

### Adding New Turbulence Models

```python
class CustomTurbulenceModel:
    def __init__(self, grid_spacing, fluid_properties):
        self.dx, self.dy, self.dz = grid_spacing
        # Initialize model parameters
    
    def calculate_turbulence(self, velocity_field, density_field):
        # Implement custom turbulence model
        return k, epsilon, mu_t
```

### Adding New Acoustic Models

```python
class CustomAcousticModel:
    def __init__(self, fluid_properties):
        self.rho1 = fluid_properties['density_1']
        self.c1 = fluid_properties['sound_speed_1']
        # Initialize model parameters
    
    def calculate_sound_speed(self, volume_fraction):
        # Implement custom sound speed model
        return effective_sound_speed
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce grid resolution in `config.json`
2. **Missing dependencies**: Install all packages from `requirements.txt`
3. **File not found**: Ensure you're running from the correct directory

### Performance Tips

1. **Use HDF5** for large datasets
2. **Enable parallel processing** for large simulations
3. **Use appropriate turbulence models** for your application

## Citation

If you use this dataset in your research, please cite:

```
@phdthesis{stratified_flow_2024,
  title={Study on the Attenuation Mechanisms in Stratified Flows: 2. Simulation Data},
  author={[Your Name]},
  year={2024},
  school={[Your University]},
  type={PhD Thesis}
}
```

## License

This dataset generator is provided for academic research purposes. Please ensure proper attribution and follow your institution's guidelines for data usage.

## Contact

For questions or issues with this dataset generator, please refer to the PhD thesis documentation or contact the author.

## Version History

- **v1.0** (2024): Initial release with complete dataset generator
- Includes CFD simulation, acoustic analysis, turbulence modeling, and validation data
- Supports multiple output formats (HDF5, VTK, CSV, NPZ)
- Comprehensive analysis and visualization tools