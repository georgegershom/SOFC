# Synthetic Synchrotron X-ray Data for SOFC Creep Studies

## Overview

This repository contains a comprehensive synthetic data generator for simulating **4D (3D + Time) synchrotron X-ray experimental data** used in studying creep deformation and failure in **Solid Oxide Fuel Cell (SOFC)** materials. The data mimics real-world, in-operando synchrotron X-ray tomography and diffraction experiments conducted at facilities like ESRF, APS, or Diamond Light Source.

### Purpose

This synthetic dataset is designed for:
- **Model validation**: Testing and validating computational creep models
- **Algorithm development**: Developing image analysis and machine learning algorithms
- **Educational purposes**: Teaching synchrotron X-ray techniques and materials science
- **Proof-of-concept studies**: Planning experiments before costly beamtime

## Dataset Description

### 1. 4D Synchrotron X-ray Tomography Data

**File**: `synchrotron_data/tomography/tomography_4D.h5`

High-resolution 3D microstructure evolution captured at regular time intervals during high-temperature creep testing.

**Contents**:
- **Dimensions**: 512 × 512 × 512 voxels per time step
- **Time steps**: 11 (initial state + 10 measurements)
- **Voxel size**: 0.65 μm
- **Field of view**: ~333 μm in each dimension
- **Data format**: HDF5 with gzip compression

**Key Features Captured**:
- ✅ Creep cavity nucleation at grain boundaries
- ✅ Cavity growth and coalescence
- ✅ Crack propagation along grain boundaries
- ✅ Grain boundary sliding
- ✅ Microstructural degradation over time

**HDF5 Structure**:
```
tomography_4D.h5
├── tomography [11 × 512 × 512 × 512] float32
│   └── 4D array: [time, z, y, x]
├── time_hours [11] float32
│   └── Time stamp for each scan
└── Attributes:
    ├── temperature_C: 700.0
    ├── applied_stress_MPa: 50.0
    ├── voxel_size_um: 0.65
    ├── beam_energy_keV: 25.0
    └── ...
```

### 2. Grain Structure

**File**: `synchrotron_data/tomography/grain_map.h5`

3D map of grain IDs showing the polycrystalline microstructure.

**Contents**:
- **Grain IDs**: Integer array mapping each voxel to a grain
- **Number of grains**: 50-200 grains
- **Average grain size**: 25 μm

### 3. Tomography Metrics

**File**: `synchrotron_data/tomography/tomography_metrics.json`

Quantitative metrics extracted from tomography data at each time step.

**Metrics Tracked**:
- **Porosity [%]**: Total void volume fraction
- **Cavity count**: Number of discrete cavities/pores
- **Crack volume [mm³]**: Volume of propagating cracks
- **Grain boundary integrity**: Average density at grain boundaries

### 4. X-ray Diffraction (XRD) Data

#### 4a. Diffraction Patterns

**File**: `synchrotron_data/diffraction/xrd_patterns.json`

Phase identification patterns showing material phases present.

**Contents**:
- 2θ range: 20° - 80°
- Identified phases:
  - **Ferrite (α-Fe)**: 98% volume fraction (BCC structure)
  - **Chromia (Cr₂O₃)**: 2% volume fraction (surface oxide)

#### 4b. Strain/Stress Maps

**File**: `synchrotron_data/diffraction/strain_stress_maps.h5`

Spatial distribution of elastic strain and residual stress.

**Contents**:
- **Dimensions**: 256 × 256 pixels (2D maps)
- **Time steps**: 11
- **Datasets**:
  - `elastic_strain`: Strain tensor component
  - `residual_stress_MPa`: Von Mises equivalent stress
- **Evolution**: Strain accumulation and stress redistribution over time

#### 4c. Phase Maps

**File**: `synchrotron_data/diffraction/phase_map.h5`

3D spatial distribution of material phases.

**Contents**:
- Phase 1: Ferrite (bulk material)
- Phase 2: Chromia (surface oxide layer and grain boundary precipitates)

### 5. Metadata

#### 5a. Experimental Parameters

**File**: `synchrotron_data/metadata/experimental_parameters.json`

Complete experimental conditions and imaging parameters.

**Key Parameters**:
- **Temperature**: 700°C (typical SOFC operating temperature)
- **Applied stress**: 50 MPa (uniaxial tension)
- **Test duration**: 100 hours
- **Scan interval**: 10 hours
- **Beam energy**: 25 keV
- **Facility**: ESRF ID19 (simulated)

#### 5b. Material Specifications

**File**: `synchrotron_data/metadata/material_specifications.json`

Complete material characterization and properties.

**Material**: Ferritic Stainless Steel (Crofer 22 APU)

**Composition (wt%)**:
- Fe: 75.0%
- Cr: 22.0%
- Mn: 0.5%
- Ti, La, Si, C: < 0.1%

**Mechanical Properties**:
- Elastic modulus: 220 GPa
- Poisson's ratio: 0.29
- Yield strength: 280 MPa

**Microstructure**:
- Grain size: 25 μm (average)
- Grain morphology: Equiaxed
- Texture: Random

#### 5c. Sample Geometry

**File**: `synchrotron_data/metadata/sample_geometry.json`

Physical dimensions and preparation details.

**Geometry**: Cylindrical tensile specimen
- Diameter: 3.0 mm
- Height: 5.0 mm
- Gauge length: 3.0 mm

## Installation

### Requirements

```bash
# Python 3.8 or higher
pip install -r requirements.txt
```

**Core dependencies**:
- `numpy` >= 1.21.0
- `scipy` >= 1.7.0
- `h5py` >= 3.6.0
- `matplotlib` >= 3.5.0

### Optional Dependencies

For advanced visualization:
```bash
pip install pyvista  # 3D interactive visualization
pip install scikit-image  # Advanced image processing
pip install pandas  # Data analysis
```

## Usage

### 1. Generate Synthetic Data

```bash
python generate_synchrotron_data.py
```

**Options**:
```bash
python generate_synchrotron_data.py \
    --output-dir my_experiment \
    --temperature 750 \
    --stress 60 \
    --duration 200 \
    --seed 12345
```

**Parameters**:
- `--output-dir`: Output directory (default: `synchrotron_data`)
- `--temperature`: Test temperature in °C (default: 700)
- `--stress`: Applied stress in MPa (default: 50)
- `--duration`: Test duration in hours (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

**Output**:
```
synchrotron_data/
├── tomography/
│   ├── tomography_4D.h5          (~1.1 GB)
│   ├── grain_map.h5               (~512 MB)
│   └── tomography_metrics.json
├── diffraction/
│   ├── xrd_patterns.json
│   ├── strain_stress_maps.h5      (~3 MB)
│   └── phase_map.h5               (~512 MB)
├── metadata/
│   ├── experimental_parameters.json
│   ├── material_specifications.json
│   └── sample_geometry.json
└── dataset_summary.json
```

### 2. Visualize Data

```bash
python visualize_data.py
```

**Options**:
```bash
python visualize_data.py \
    --data-dir synchrotron_data \
    --output-dir visualizations
```

**Generated Visualizations**:
1. `tomography_initial.png` - Initial microstructure slices
2. `tomography_final.png` - Final microstructure after creep
3. `creep_evolution.png` - Time-series damage metrics
4. `xrd_patterns.png` - Phase identification patterns
5. `strain_maps_initial.png` - Initial strain distribution
6. `strain_maps_final.png` - Final strain distribution
7. `3d_voids.png` - 3D rendering of voids and cracks
8. `dashboard.png` - Comprehensive analysis dashboard

### 3. Analyze Data

```bash
python analyze_metrics.py
```

**Options**:
```bash
python analyze_metrics.py \
    --data-dir synchrotron_data \
    --output analysis_report.json
```

**Analysis Performed**:
- ✅ **Primary creep fitting**: Power-law model (ε = ε₀ + A·t^n)
- ✅ **Secondary creep fitting**: Linear steady-state creep rate
- ✅ **Cavity nucleation kinetics**: Nucleation rate vs. time
- ✅ **Crack propagation analysis**: Growth kinetics and exponential fit
- ✅ **Strain distribution statistics**: Mean, std, percentiles at each time

**Output**: JSON report with all quantitative results and model parameters.

## Data Access Examples

### Python

#### Read tomography data:
```python
import h5py
import numpy as np

# Open tomography file
with h5py.File('synchrotron_data/tomography/tomography_4D.h5', 'r') as f:
    # Load all time steps
    tomography = f['tomography'][:]  # Shape: (11, 512, 512, 512)
    time = f['time_hours'][:]         # Shape: (11,)
    
    # Get metadata
    temperature = f.attrs['temperature_C']
    stress = f.attrs['applied_stress_MPa']
    voxel_size = f.attrs['voxel_size_um']
    
    # Access specific time step
    initial_state = f['tomography'][0]    # t = 0
    final_state = f['tomography'][-1]     # t = 100 hours
    
    # Get 2D slice
    mid_slice = f['tomography'][5, 256, :, :]  # XY slice at mid-height
```

#### Read metrics:
```python
import json

with open('synchrotron_data/tomography/tomography_metrics.json', 'r') as f:
    metrics = json.load(f)

# Access time-series data
porosity = metrics['porosity_percent']
cavity_count = metrics['cavity_count']
time = metrics['time_hours']
```

#### Read XRD data:
```python
import h5py

# Strain/stress maps
with h5py.File('synchrotron_data/diffraction/strain_stress_maps.h5', 'r') as f:
    strain = f['elastic_strain'][:]       # Shape: (11, 256, 256)
    stress = f['residual_stress_MPa'][:]  # Shape: (11, 256, 256)
    time = f['time_hours'][:]
    
    # Plot strain evolution
    import matplotlib.pyplot as plt
    plt.imshow(strain[-1], cmap='RdYlBu_r')
    plt.colorbar(label='Elastic Strain')
    plt.title('Final Strain Distribution')
    plt.show()
```

### MATLAB

```matlab
% Read HDF5 tomography data
filename = 'synchrotron_data/tomography/tomography_4D.h5';
tomography = h5read(filename, '/tomography');  % 512×512×512×11
time = h5read(filename, '/time_hours');

% Get metadata
info = h5info(filename);
temp = h5readatt(filename, '/', 'temperature_C');
stress = h5readatt(filename, '/', 'applied_stress_MPa');

% Visualize slice
slice_data = squeeze(tomography(256,:,:,11));
imagesc(slice_data);
colormap gray;
colorbar;
title('Final State - Mid-slice');
```

## Data Validation

The synthetic data includes realistic features:

### Physical Accuracy
✅ **Creep deformation**: Power-law primary creep + steady-state secondary creep  
✅ **Cavity nucleation**: Preferentially at grain boundaries  
✅ **Crack propagation**: Along high-angle grain boundaries  
✅ **Strain localization**: Near defects and stress concentrators  
✅ **Phase distribution**: Ferrite bulk + chromia surface oxide  

### Imaging Realism
✅ **Voxel resolution**: 0.65 μm (typical for synchrotron μCT)  
✅ **Poisson noise**: Photon counting statistics  
✅ **Beam hardening**: Energy-dependent attenuation  
✅ **Grain structure**: Voronoi tessellation for realistic polycrystal  

### Metadata Completeness
✅ **Experimental parameters**: Temperature, stress, time, beam energy  
✅ **Material properties**: Composition, mechanical properties, microstructure  
✅ **Sample geometry**: Dimensions, preparation method  

## Limitations

This is **synthetic data** and has limitations:

1. **Simplified physics**: Real creep mechanisms are more complex
2. **Idealized grain structure**: Actual grains have more complex morphologies
3. **No instrumental artifacts**: Real data has ring artifacts, beam fluctuations, etc.
4. **Deterministic noise model**: Real noise is more complex
5. **Limited phase diversity**: Only 2 phases modeled

⚠️ **Do not use for publication-quality materials science conclusions**  
✅ **Suitable for algorithm development, model testing, and education**

## Scientific Background

### Creep Deformation in SOFCs

Solid Oxide Fuel Cells operate at high temperatures (600-1000°C), causing time-dependent plastic deformation (creep) in metallic interconnects. This leads to:

1. **Loss of electrical contact** between cell components
2. **Gas leakage** through cracked seals
3. **Mechanical failure** and cell degradation

### Creep Mechanisms

Three main mechanisms contribute to creep in SOFC interconnects:

1. **Diffusional creep** (Nabarro-Herring, Coble)
   - Vacancy diffusion through grains or grain boundaries
   - Dominant at low stresses

2. **Dislocation creep** (power-law creep)
   - Dislocation glide and climb
   - Dominant at higher stresses

3. **Grain boundary sliding**
   - Relative motion of adjacent grains
   - Accommodated by diffusion or dislocation motion

### Cavity Nucleation and Growth

- **Nucleation**: Cavities nucleate at grain boundaries, especially triple junctions
- **Growth**: Driven by stress concentration and vacancy diffusion
- **Coalescence**: Adjacent cavities link to form microcracks
- **Failure**: Microcracks propagate and coalesce, leading to macroscopic failure

## Citation

If you use this synthetic data in your research, please cite:

```bibtex
@software{synthetic_synchrotron_sofc_2025,
  title = {Synthetic Synchrotron X-ray Data Generator for SOFC Creep Studies},
  author = {Synthetic Data Generator},
  year = {2025},
  version = {1.0},
  url = {https://github.com/your-repo/synchrotron-sofc-data}
}
```

## References

### Experimental Techniques
1. Burnett, T. L., et al. (2016). "Large volume serial section tomography by automated on-stage scanning" *Ultramicroscopy* 161, 119-129.
2. Di Michiel, M., et al. (2005). "Fast microtomography using high energy synchrotron radiation" *Rev. Sci. Instrum.* 76, 043702.

### SOFC Materials and Creep
3. Yang, Z., et al. (2007). "Selection and Evaluation of Heat-Resistant Alloys for SOFC Interconnect Applications" *J. Electrochem. Soc.* 154, B107-B115.
4. Frost, H. J., & Ashby, M. F. (1982). *Deformation-Mechanism Maps: The Plasticity and Creep of Metals and Ceramics*. Pergamon Press.

### Computational Methods
5. Roters, F., et al. (2019). "DAMASK – The Düsseldorf Advanced Material Simulation Kit" *Comp. Mater. Sci.* 158, 420-478.

## License

This synthetic data generator is provided under the MIT License.

```
MIT License

Copyright (c) 2025 Synthetic Data Generator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## Contact and Support

For questions, bug reports, or feature requests:
- 📧 Email: your.email@example.com
- 🐛 Issues: GitHub Issues page
- 📖 Documentation: See this README

## Acknowledgments

This synthetic data generator was created to support:
- SOFC materials research and development
- Advanced characterization technique education
- Computational model validation studies

Special thanks to the synchrotron user community for inspiration and guidance on realistic data characteristics.

---

**Version**: 1.0  
**Last Updated**: 2025-10-04  
**Status**: Active Development
