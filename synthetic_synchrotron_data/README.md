# Synthetic Synchrotron X-ray Experimental Data for SOFC Creep Studies

This repository contains comprehensive synthetic synchrotron X-ray data designed for validating computational models of creep deformation and failure in Solid Oxide Fuel Cell (SOFC) materials.

## Overview

This dataset provides realistic synthetic synchrotron X-ray tomography and diffraction data that mimics real experimental conditions for studying creep deformation in SOFC metallic interconnects and anode support materials. The data includes:

- **4D Tomography Data**: 3D microstructural evolution over time
- **XRD Data**: Phase identification and residual stress/strain mapping
- **Comprehensive Metadata**: Complete experimental parameters and material specifications
- **Visualization Tools**: Scripts for analyzing and displaying the data

## Dataset Structure

```
synthetic_synchrotron_data/
├── README.md                           # This file
├── tomography/
│   ├── generate_tomography.py          # Script to generate initial 3D tomography data
│   ├── generate_time_evolution.py      # Script to simulate creep evolution over time
│   ├── initial/                        # Initial state data (pre-test)
│   │   ├── attenuation_map.npy         # 3D X-ray attenuation map
│   │   ├── porosity_map.npy            # Porosity distribution
│   │   ├── defect_map.npy              # Initial defects and cracks
│   │   ├── grain_map.npy               # Grain structure mapping
│   │   └── sinogram.npy                # Synthetic projection data
│   └── timeseries/                     # Time evolution data (16 time steps)
│       ├── attenuation_step_000.npy    # Attenuation at t=0
│       ├── attenuation_step_001.npy    # Attenuation at t=3.125h
│       ├── porosity_step_000.npy       # Porosity evolution
│       ├── defects_step_000.npy        # Defect evolution
│       └── evolution_summary.json      # Summary of simulation parameters
├── diffraction/
│   ├── generate_xrd.py                 # Script to generate XRD patterns and mapping
│   ├── crofer_22_apu_reference.json    # Reference XRD pattern for interconnect
│   ├── oxide_scale_reference.json      # Reference XRD pattern for oxide scale
│   ├── spatial_xrd_mapping.json        # Spatially resolved XRD data
│   └── xrd_timeseries/                 # Time evolution of XRD patterns
│       └── xrd_step_000.json           # XRD data at each time step
├── metadata/
│   └── experiment_metadata.json        # Complete experimental metadata
└── visualization/
    ├── visualize_tomography.py         # Interactive tomography visualization
    └── visualize_xrd.py                # Interactive XRD visualization
```

## Data Generation

### Prerequisites

The data generation scripts require:
- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- scikit-image (for 3D visualization)

Install dependencies:
```bash
pip install numpy scipy matplotlib scikit-image
```

### Generating Tomography Data

1. **Generate Initial State**:
```bash
cd tomography/
python generate_tomography.py
```
This creates the initial 3D microstructure with:
- Realistic grain structure using Voronoi tessellation
- Initial porosity and defects
- X-ray attenuation mapping
- Synthetic projection data

2. **Generate Time Evolution**:
```bash
python generate_time_evolution.py
```
This simulates creep deformation over 50 hours with:
- Pore nucleation and growth (cavitation)
- Crack initiation and propagation
- Grain boundary sliding
- Time-dependent attenuation changes

### Generating XRD Data

```bash
cd ../diffraction/
python generate_xrd.py
```
This creates:
- Reference XRD patterns for material phases
- Spatially resolved stress/strain mapping
- Time evolution of diffraction patterns

## Data Specifications

### Tomography Data

**Sample Specifications:**
- **Material**: Crofer 22 APU (Fe-22Cr metallic interconnect)
- **Sample Size**: 150 × 150 × 150 voxels
- **Voxel Size**: 0.5 μm
- **Physical Dimensions**: 75 × 75 × 75 μm³

**Test Conditions:**
- **Temperature**: 750°C
- **Applied Stress**: 60 MPa (uniaxial tension)
- **Test Duration**: 50 hours
- **Time Steps**: 16 (every 3.125 hours)

**Key Features:**
- Grain size: ~45 μm average
- Initial porosity: ~1.8%
- Realistic defect distribution
- Physically-based creep evolution

### XRD Data

**Measurement Parameters:**
- **X-ray Energy**: 10 keV (wavelength = 1.24 Å)
- **Detector**: 1024 × 1024 pixels
- **Spatial Grid**: 30 × 30 points (100 μm steps)
- **Measurement Time**: 30 seconds per point

**Material Phases:**
- **Crofer 22 APU**: BCC Fe-Cr alloy (a = 3.58 Å)
- **Oxide Scale**: Cr₂O₃ + (Mn,Cr)₃O₄ spinel (a = 4.18 Å)

## Usage Examples

### Basic Data Analysis

```python
import numpy as np

# Load tomography data
attenuation = np.load('tomography/initial/attenuation_map.npy')
porosity = np.load('tomography/initial/porosity_map.npy')

# Analyze microstructure
total_porosity = np.mean(porosity)  # 0.018 (1.8%)
grain_volume = np.bincount(grain_map.ravel())
grain_sizes = grain_volume[grain_volume > 0]  # Grain size distribution
```

### Visualization

```bash
# Launch interactive tomography visualizer
cd visualization/
python visualize_tomography.py ../tomography/

# Launch interactive XRD visualizer
python visualize_xrd.py ../diffraction/
```

### Custom Analysis

```python
import json
import numpy as np

# Load metadata
with open('metadata/experiment_metadata.json', 'r') as f:
    metadata = json.load(f)

# Access experimental parameters
temperature = metadata['operational_parameters']['test_conditions']['temperature_control']['target_temperature_celsius']
stress = metadata['operational_parameters']['test_conditions']['mechanical_loading']['applied_stress_mpa']['initial']

# Load time evolution data
evolution_summary = json.load(open('tomography/timeseries/evolution_summary.json'))
final_porosity = evolution_summary['final_porosity_fraction']
```

## Key Features and Validation

### Physical Realism

The synthetic data incorporates physically realistic features:

1. **Microstructural Evolution**:
   - Temperature and stress-dependent creep rates
   - Realistic pore nucleation and growth kinetics
   - Grain boundary sliding and cavitation
   - Crack propagation along stress concentrations

2. **XRD Characteristics**:
   - Accurate Bragg peak positions using lattice parameters
   - Realistic peak broadening with strain
   - Proper intensity relationships for different phases
   - Spatial variation in stress/strain fields

3. **Experimental Artifacts**:
   - Realistic noise levels in tomography data
   - Beam hardening effects (simulated)
   - Detector response characteristics
   - Time-dependent measurement uncertainties

### Validation Applications

This dataset is designed for validating:

1. **Crystal Plasticity Models**:
   - Compare predicted vs. measured strain evolution
   - Validate grain-level deformation mechanisms
   - Assess crack initiation criteria

2. **Phase Field Models**:
   - Validate porosity evolution predictions
   - Compare simulated vs. synthetic microstructure
   - Assess defect interaction models

3. **Machine Learning Models**:
   - Train on realistic synthetic data before applying to experimental data
   - Test segmentation algorithms on known ground truth
   - Validate feature extraction methods

4. **Image Analysis Methods**:
   - Develop and test pore segmentation algorithms
   - Validate crack detection and quantification
   - Assess grain boundary identification methods

## File Format Specifications

### Tomography Files

- **`.npy` files**: NumPy binary format containing 3D arrays
- **Dimensions**: (x, y, z) where x,y,z are voxel indices
- **Data types**:
  - `attenuation_map`: float32 (X-ray attenuation coefficients)
  - `porosity_map`: bool (binary porosity mask)
  - `defect_map`: bool (binary defect mask)
  - `grain_map`: int32 (grain ID labels)

### XRD Files

- **`.json` files**: JavaScript Object Notation
- **Structure**:
  ```json
  {
    "two_theta": [20.0, 20.1, ..., 80.0],
    "intensity": [10.5, 12.3, ..., 8.7],
    "phase": "Crofer 22 APU",
    "wavelength": 1.24
  }
  ```

### Metadata Files

- **`.json` files**: Hierarchical experimental metadata
- **Sections**:
  - `experiment_info`: Basic experiment details
  - `sample_specifications`: Material and geometry details
  - `operational_parameters`: Test conditions and acquisition settings
  - `data_processing`: Analysis methods and algorithms
  - `validation_and_uncertainty`: Uncertainty quantification

## Performance Considerations

### Computational Requirements

- **Tomography Generation**: ~2-5 minutes per dataset (depends on sample size)
- **Time Evolution**: ~1-2 minutes for 16 time steps
- **XRD Generation**: ~30 seconds for spatial mapping
- **Memory Usage**: ~500 MB for full dataset (150³ voxels × 16 time steps)

### Recommended Hardware

- **Minimum**: 8 GB RAM, 4-core CPU
- **Recommended**: 16 GB RAM, 8-core CPU, GPU for 3D visualization
- **Storage**: 1 GB free space for complete dataset

## Citation and Attribution

When using this synthetic dataset, please cite:

```
Synthetic Synchrotron X-ray Data for SOFC Creep Studies
Generated by: [Your Name/Organization]
Date: October 2024
Repository: [Repository URL]
```

## Contributing and Issues

This is a synthetic dataset generated for research purposes. For questions about the data generation methodology or suggestions for improvements, please contact the maintainers.

## License

This synthetic dataset is provided for research and educational purposes. Please respect the intellectual property of the underlying experimental techniques and methodologies that inspired this work.