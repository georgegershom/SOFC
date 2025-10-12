# Stratified Flow Simulation Dataset - Complete Index

**Version**: 1.0  
**Date**: 2025-10-12  
**Purpose**: PhD Thesis - Attenuation Mechanisms in Stratified Flows  

---

## Documentation Files

| File | Description | Start Here? |
|------|-------------|-------------|
| **README.md** | Complete dataset documentation with technical details | ✓ Yes |
| **QUICKSTART.md** | 5-minute tutorial to get started | ✓✓✓ Best for beginners |
| **DATASET_SUMMARY.md** | Quick reference guide with key information | ✓✓ For quick lookup |
| **INDEX.md** | This file - navigation guide | Start point |

---

## Python Scripts

| File | Description | Usage |
|------|-------------|-------|
| **generate_simulation_data.py** | Data generation script | Already run - regenerate data if needed |
| **visualize_data.py** | Creates all visualization figures | `python3 visualize_data.py` |
| **data_loader.py** | Helper functions for data loading | `from data_loader import StratifiedFlowData` |
| **requirements.txt** | Python package dependencies | `pip install -r requirements.txt` |

---

## Data Directories

### 1. cfd_outputs/ (CFD Simulation Results)

**Size**: ~220 MB  
**Files**: 10

| File | Size | Shape | Description |
|------|------|-------|-------------|
| velocity_u.npy | ~2 MB | (100,100,50) | X-velocity component |
| velocity_v.npy | ~2 MB | (100,100,50) | Y-velocity component |
| velocity_w.npy | ~2 MB | (100,100,50) | Z-velocity component |
| pressure.npy | ~2 MB | (100,100,50) | Pressure field |
| vof.npy | ~2 MB | (100,100,50) | Volume of Fluid |
| turbulence_k.npy | ~2 MB | (100,100,50) | Turbulent kinetic energy |
| turbulence_epsilon.npy | ~2 MB | (100,100,50) | Dissipation rate |
| eddy_viscosity.npy | ~2 MB | (100,100,50) | Eddy viscosity |
| **acoustic_pressure.npy** | ~**200 MB** | **(1000,100,50)** | **Acoustic propagation** |
| coordinates.json | <1 MB | - | Grid coordinates & metadata |

### 2. mathematical_model_outputs/ (Analytical & Semi-Analytical Models)

**Size**: ~0.5 MB  
**Files**: 9

| File | Size | Shape | Description |
|------|------|-------|-------------|
| sound_speed_wood.npy | <1 MB | (20,) | Wood's equation results |
| sound_speed_dispersive.npy | <1 MB | (50,20) | Dispersive model |
| attenuation_coefficients.npy | <1 MB | (50,20) | Attenuation (Np/m) |
| attenuation_dB.npy | <1 MB | (50,20) | Attenuation (dB/m) |
| reflection_vs_angle.npy | <1 MB | (90,) | Reflection coefficient |
| standing_waves.npy | <1 MB | (10,100) | Standing wave patterns |
| reflection_transmission.json | <1 MB | - | Interface coefficients |
| time_delays.json | <1 MB | - | Time-delay estimates |
| parameters.json | <1 MB | - | Model parameters |

### 3. validation_data/ (Model Validation & Comparison)

**Size**: ~0.5 MB  
**Files**: 6

| File | Size | Shape | Description |
|------|------|-------|-------------|
| waveform_simulated.npy | <1 MB | (1000,) | Simulated waveform |
| waveform_experimental.npy | <1 MB | (1000,) | Experimental waveform |
| waveform_time.npy | <1 MB | (1000,) | Time array |
| attenuation_comparison.json | <1 MB | - | Attenuation validation |
| sound_speed_comparison.json | <1 MB | - | Sound speed validation |
| validation_statistics.json | <1 MB | - | Statistical metrics |

### 4. figures/ (Visualizations)

**Size**: ~5 MB  
**Files**: 8 PNG images

| File | Description |
|------|-------------|
| velocity_fields.png | 3D velocity components (u, v, w, magnitude) |
| vof_pressure.png | Phase distribution and pressure field |
| turbulence_parameters.png | k, ε, μ_t fields |
| acoustic_propagation.png | Time snapshots of acoustic waves |
| sound_speed_predictions.png | Wood's equation and dispersive model |
| attenuation_coefficients.png | Attenuation vs. frequency and void fraction |
| wave_propagation.png | Reflection, transmission, standing waves |
| validation_comparison.png | Simulated vs. experimental comparisons |

---

## Quick Navigation Guide

### "I want to..."

**...understand what data is available**
→ Read: **README.md** (Section: Data Categories)

**...start analyzing data immediately**
→ Follow: **QUICKSTART.md** (5-minute tutorial)

**...load data in Python**
→ Use: **data_loader.py** 
```python
from data_loader import StratifiedFlowData
data = StratifiedFlowData()
velocity = data.load_velocity_field()
```

**...create visualizations**
→ Run: `python3 visualize_data.py`

**...find specific parameters**
→ Check: **DATASET_SUMMARY.md** (Section: Key Physical Parameters)

**...validate my models**
→ Use: **validation_data/** directory
→ Read: **DATASET_SUMMARY.md** (Section: Data Quality Metrics)

**...understand the file formats**
→ Read: **README.md** (Section: File Formats)

**...export to CSV or other formats**
→ See: **QUICKSTART.md** (Task 3: Export Data to CSV)

**...reproduce the dataset**
→ Run: `python3 generate_simulation_data.py`

---

## Research Workflow Suggestions

### Workflow A: Acoustic Attenuation Study

1. Load attenuation data:
   ```python
   from data_loader import StratifiedFlowData
   data = StratifiedFlowData()
   atten = data.load_attenuation(unit='dB')
   ```

2. Load acoustic propagation:
   ```python
   acoustic = data.load_acoustic_pressure(mmap=True)
   ```

3. Analyze frequency dependence
4. Compare with theoretical models
5. Use validation data for verification

**Key files**: `attenuation_dB.npy`, `acoustic_pressure.npy`, `attenuation_comparison.json`

### Workflow B: Sound Speed Characterization

1. Load sound speed models:
   ```python
   sound_speed = data.load_sound_speed(model='all')
   ```

2. Compare Wood's equation vs. dispersive model
3. Validate against experimental data:
   ```python
   validation = data.load_validation_data()
   speed_comp = validation['sound_speed']
   ```

4. Analyze void fraction effects

**Key files**: `sound_speed_*.npy`, `sound_speed_comparison.json`

### Workflow C: Interface Dynamics

1. Load VOF and velocity:
   ```python
   vof = data.load_vof()
   velocity = data.load_velocity_field()
   ```

2. Identify interface location
3. Analyze reflection/transmission:
   ```python
   wave_prop = data.load_wave_propagation()
   ```

4. Study interfacial waves

**Key files**: `vof.npy`, `velocity_*.npy`, `reflection_*.npy`

### Workflow D: Flow-Acoustic Coupling

1. Load all CFD fields:
   ```python
   velocity = data.load_velocity_field()
   turbulence = data.load_turbulence()
   acoustic = data.load_acoustic_pressure()
   ```

2. Analyze spatial correlations
3. Investigate turbulence effects
4. Study convective influence

**Key files**: All `cfd_outputs/` files

---

## File Dependencies

```
data_loader.py
  ├─ Requires: cfd_outputs/coordinates.json
  ├─ Requires: mathematical_model_outputs/parameters.json
  └─ Accesses: All .npy and .json files

visualize_data.py
  ├─ Requires: All cfd_outputs/*.npy
  ├─ Requires: All mathematical_model_outputs/*.npy
  ├─ Requires: All validation_data/*.npy
  └─ Creates: figures/*.png

generate_simulation_data.py
  ├─ Requires: numpy, json
  └─ Creates: All data files
```

---

## Data Access Patterns

### Pattern 1: Direct NumPy Loading
```python
import numpy as np
data = np.load('cfd_outputs/velocity_u.npy')
```
**Use when**: Simple, one-time data access

### Pattern 2: Using Data Loader Class
```python
from data_loader import StratifiedFlowData
data = StratifiedFlowData()
velocity = data.load_velocity_field()
```
**Use when**: Multiple accesses, need metadata, convenience functions

### Pattern 3: Memory-Mapped Loading
```python
import numpy as np
data = np.load('cfd_outputs/acoustic_pressure.npy', mmap_mode='r')
```
**Use when**: File is too large for RAM (acoustic_pressure.npy)

### Pattern 4: JSON Loading
```python
import json
with open('validation_data/validation_statistics.json', 'r') as f:
    stats = json.load(f)
```
**Use when**: Loading metadata, parameters, validation data

---

## Common Issues & Solutions

| Issue | Solution | Reference |
|-------|----------|-----------|
| File not found | Check working directory | QUICKSTART.md - Troubleshooting |
| Memory error | Use memory mapping | DATASET_SUMMARY.md - Issue 1 |
| Import error | Install packages | `pip install -r requirements.txt` |
| Need different units | Use conversion factors | DATASET_SUMMARY.md - Issue 3 |
| Plotting issues | Check matplotlib backend | QUICKSTART.md - Troubleshooting |

---

## Data Versioning

**Current Version**: 1.0 (2025-10-12)

**Version History**:
- v1.0: Initial release with full CFD, mathematical models, and validation data

**Future Versions** (planned):
- v1.1: Additional void fractions, extended frequency range
- v2.0: 3D acoustic propagation, LES turbulence model

---

## Citation

```bibtex
@dataset{stratified_flow_sim_2025,
  author = {[Your Name]},
  title = {Stratified Flow Simulation Dataset: Attenuation Mechanisms Study},
  year = {2025},
  version = {1.0},
  publisher = {[Your Institution]},
  note = {PhD Thesis Research Data}
}
```

---

## Support & Contact

**Questions?**
1. Check documentation: README.md, QUICKSTART.md, DATASET_SUMMARY.md
2. Run examples: `python3 data_loader.py`
3. View visualizations: `python3 visualize_data.py`
4. Contact: [Your Email]

---

## Quick Stats

```
Total Files: 33
Total Size: ~500 MB
Data Points: >500 million
Simulations: CFD + Mathematical Models
Validation: 3 comparison datasets
Figures: 8 publication-quality plots
Scripts: 3 Python scripts
Documentation: 5 markdown files
```

---

## License

[Specify license: CC BY 4.0, MIT, GPL, etc.]

This dataset is provided for academic and research purposes.

---

*Last Updated: 2025-10-12*  
*Dataset Version: 1.0*  
*Documentation Version: 1.0*
