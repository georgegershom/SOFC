# Quick Start Guide - Synthetic Synchrotron X-ray Data

## 🎯 What Has Been Generated

You now have a complete synthetic dataset simulating **in-operando synchrotron X-ray experiments** for SOFC creep studies, including:

✅ **4D Tomography Data** (3D + Time) - 6 time steps  
✅ **X-ray Diffraction Patterns** - Phase identification  
✅ **Strain/Stress Maps** - Residual stress evolution  
✅ **Grain Structure** - Polycrystalline microstructure  
✅ **Complete Metadata** - Experimental parameters, material specs, sample geometry  
✅ **Visualization Tools** - Ready-to-use plotting scripts  
✅ **Analysis Tools** - Quantitative creep analysis  

## 📊 Generated Data Summary

### Dataset Statistics
- **Total size**: ~22 MB (demo version)
- **Time steps**: 6 (0, 10, 20, 30, 40, 50 hours)
- **Volume dimensions**: 128 × 128 × 128 voxels
- **Voxel size**: 0.65 μm
- **Test conditions**: 700°C, 50 MPa uniaxial tension

### File Structure
```
synchrotron_data/
├── tomography/
│   ├── tomography_4D.h5          (19 MB)  - 4D microstructure evolution
│   ├── grain_map.h5               (240 KB) - Grain structure
│   └── tomography_metrics.json    (704 B)  - Extracted metrics
├── diffraction/
│   ├── xrd_patterns.json          (137 KB) - Phase identification
│   ├── strain_stress_maps.h5      (2.7 MB) - Strain/stress evolution
│   └── phase_map.h5               (104 KB) - Phase distribution
└── metadata/
    ├── experimental_parameters.json        - Test conditions
    ├── material_specifications.json        - Material properties
    └── sample_geometry.json                - Sample dimensions

visualizations/                     (8 PNG files)
├── dashboard.png                   - Comprehensive overview
├── creep_evolution.png            - Time-series metrics
├── tomography_initial.png         - Initial microstructure
├── tomography_final.png           - Final microstructure
├── xrd_patterns.png               - Diffraction patterns
├── strain_maps_initial.png        - Initial stress
├── strain_maps_final.png          - Final stress
└── 3d_voids.png                   - 3D void visualization
```

## 🚀 Quick Start in 3 Steps

### Step 1: Explore the Visualizations (Already Done!)
```bash
ls visualizations/
# View dashboard.png for a comprehensive overview
```

### Step 2: Run Example Usage
```bash
python3 example_usage.py
# This demonstrates 6 different ways to access and use the data
```

### Step 3: Try Custom Analysis
```python
import h5py
import numpy as np

# Load tomography data
with h5py.File('synchrotron_data/tomography/tomography_4D.h5', 'r') as f:
    initial = f['tomography'][0]     # t = 0 hours
    final = f['tomography'][-1]      # t = 50 hours
    
    # Calculate damage
    damage = initial - final
    print(f"Mean damage: {np.mean(damage):.4f}")
```

## 📚 Available Scripts

### 1. Data Generation
```bash
# Generate full-size dataset (512³ voxels, 11 time steps, ~1.6 GB)
python3 generate_synchrotron_data.py

# Generate demo dataset (128³ voxels, 6 time steps, ~22 MB)
python3 quick_generate.py

# Custom parameters
python3 generate_synchrotron_data.py \
    --temperature 750 \
    --stress 60 \
    --duration 200 \
    --seed 12345
```

### 2. Visualization
```bash
# Generate all visualizations
python3 visualize_data.py

# Custom output directory
python3 visualize_data.py --output-dir my_plots
```

### 3. Quantitative Analysis
```bash
# Run comprehensive creep analysis
python3 analyze_metrics.py

# Output:
#   - Primary creep model fitting
#   - Secondary creep rate
#   - Cavity nucleation kinetics
#   - Crack propagation analysis
#   - Strain distribution statistics
```

### 4. Example Usage
```bash
# Interactive examples showing how to use the data
python3 example_usage.py
```

## 🔍 What's in the Data?

### Tomography Features
- **Creep cavitation**: Voids nucleating at grain boundaries
- **Crack propagation**: Cracks growing along grain boundaries
- **Microstructural evolution**: Time-dependent damage accumulation
- **Grain structure**: Realistic polycrystalline morphology

### XRD Features
- **Phase identification**: Ferrite (98%) + Chromia (2%)
- **Strain mapping**: Elastic strain distribution
- **Stress mapping**: Residual stress evolution
- **Realistic peaks**: Based on actual crystal structures

### Metadata
- **Experimental parameters**: Temperature, stress, time, beam energy
- **Material properties**: Composition, mechanical properties, thermal properties
- **Sample geometry**: Dimensions, mass, preparation method

## 📖 Key Metrics Tracked

| Metric | Description | Unit |
|--------|-------------|------|
| **Porosity** | Total void volume fraction | % |
| **Cavity Count** | Number of discrete voids | count |
| **Crack Volume** | Volume of propagating cracks | mm³ |
| **GB Integrity** | Grain boundary degradation | normalized |
| **Elastic Strain** | Lattice strain | ε (dimensionless) |
| **Residual Stress** | Internal stress field | MPa |

## 🔬 Scientific Realism

### Physics Included ✅
- Power-law primary creep (ε ∝ t^n)
- Steady-state secondary creep (ε̇ = constant)
- Grain boundary cavity nucleation
- Stress-driven crack propagation
- Strain localization near defects

### Imaging Realism ✅
- Poisson noise (photon counting)
- Realistic voxel resolution (0.65 μm)
- Grain structure (Voronoi tessellation)
- Multi-phase material (ferrite + chromia)

## 💡 Use Cases

### ✅ Recommended For:
1. **Model validation**: Testing computational creep models
2. **Algorithm development**: Image analysis, ML training
3. **Educational purposes**: Teaching synchrotron techniques
4. **Method development**: Planning analysis workflows
5. **Proof-of-concept**: Testing ideas before experiments

### ⚠️ Not Suitable For:
1. Publication of materials science conclusions
2. Replacing real experimental data
3. Detailed mechanism studies (simplified physics)

## 🎓 Learning Resources

### Understanding the Data Format
- **HDF5 files**: Use `h5py` in Python or `h5read` in MATLAB
- **JSON files**: Plain text, readable in any text editor
- **Voxel data**: 3D arrays [z, y, x] convention

### Key Python Libraries
```python
import h5py          # Read HDF5 files
import numpy as np   # Numerical operations
import matplotlib    # Visualization
import scipy         # Analysis tools
```

### Example Workflows

**1. Load and visualize a slice:**
```python
import h5py
import matplotlib.pyplot as plt

with h5py.File('synchrotron_data/tomography/tomography_4D.h5', 'r') as f:
    data = f['tomography'][0, 64, :, :]  # Middle slice, t=0
    
plt.imshow(data, cmap='gray')
plt.colorbar()
plt.title('Initial Microstructure')
plt.show()
```

**2. Track porosity evolution:**
```python
import json
import matplotlib.pyplot as plt

with open('synchrotron_data/tomography/tomography_metrics.json') as f:
    metrics = json.load(f)

plt.plot(metrics['time_hours'], metrics['porosity_percent'])
plt.xlabel('Time (hours)')
plt.ylabel('Porosity (%)')
plt.title('Creep Damage Evolution')
plt.show()
```

**3. Analyze strain distribution:**
```python
import h5py
import numpy as np

with h5py.File('synchrotron_data/diffraction/strain_stress_maps.h5', 'r') as f:
    strain = f['elastic_strain'][-1]  # Final time step
    
print(f"Mean strain: {np.mean(strain):.6f}")
print(f"Max strain: {np.max(strain):.6f}")
print(f"Strain std: {np.std(strain):.6f}")
```

## 🆘 Troubleshooting

### Issue: "Module not found"
```bash
pip3 install numpy scipy h5py matplotlib
```

### Issue: "File not found"
Make sure you're in the `/workspace` directory:
```bash
cd /workspace
python3 example_usage.py
```

### Issue: "Memory error" with full dataset
Use the demo version first:
```bash
python3 quick_generate.py  # Generates 128³ instead of 512³
```

### Issue: Can't open HDF5 files
```bash
# Check if h5py is installed
python3 -c "import h5py; print(h5py.version.info)"
```

## 📞 Next Steps

1. ✅ **Explore visualizations**: Open files in `visualizations/`
2. ✅ **Run examples**: `python3 example_usage.py`
3. ✅ **Read documentation**: See `README.md` for full details
4. ✅ **Customize generation**: Modify parameters and regenerate
5. ✅ **Develop your analysis**: Use the data for your own research

## 📚 Documentation Files

- **README.md**: Complete documentation (15 KB)
- **QUICK_START.md**: This file
- **requirements.txt**: Python dependencies
- **dataset_summary.json**: Dataset statistics
- **analysis_report.json**: Quantitative analysis results

## 🎉 You're All Set!

The synthetic synchrotron X-ray dataset is ready to use. All scripts are working, visualizations are generated, and example code is provided.

**Start exploring:**
```bash
# View the dashboard
xdg-open visualizations/dashboard.png  # Linux
open visualizations/dashboard.png      # macOS

# Or run the examples
python3 example_usage.py
```

**Questions or issues?**
- Check README.md for detailed documentation
- Examine example_usage.py for code examples
- Review analyze_metrics.py for analysis methods

---
**Version**: 1.0  
**Generated**: 2025-10-04  
**Dataset Size**: 22 MB (demo) / 1.6 GB (full)
