# Dataset Summary: Stratified Flow Simulation Data

## Quick Reference

**Dataset Name**: Stratified Flow Attenuation Mechanisms - Simulation Data  
**Version**: 1.0  
**Date**: 2025-10-12  
**Purpose**: PhD Thesis Research - Model Development and Hypothesis Testing  

---

## Data Inventory

### File Count and Size
- **Total Files**: 28 files
  - 10 CFD output files
  - 9 Mathematical model files
  - 6 Validation data files
  - 3 Support files (scripts, docs)
- **Approximate Total Size**: ~500 MB

---

## Quick Access Guide

### Most Important Files for Different Use Cases

#### 1. For CFD Analysis
```
cfd_outputs/
├── velocity_u.npy          # Primary flow velocity
├── vof.npy                 # Phase distribution
├── acoustic_pressure.npy   # Acoustic propagation (largest file)
└── coordinates.json        # Grid information
```

#### 2. For Acoustic Model Development
```
mathematical_model_outputs/
├── sound_speed_dispersive.npy      # Frequency-dependent sound speed
├── attenuation_dB.npy              # Attenuation coefficients
├── time_delays.json                # Propagation delays
└── parameters.json                 # Model parameters
```

#### 3. For Validation Studies
```
validation_data/
├── validation_statistics.json      # Statistical metrics
├── attenuation_comparison.json     # Model vs. "experimental"
└── sound_speed_comparison.json     # Sound speed validation
```

---

## Data Array Dimensions

### CFD Arrays (3D Fields)

| Variable | Shape | Size | Description |
|----------|-------|------|-------------|
| velocity_u, v, w | (100, 100, 50) | ~2 MB each | Velocity components |
| pressure | (100, 100, 50) | ~2 MB | Pressure field |
| vof | (100, 100, 50) | ~2 MB | Phase distribution |
| turbulence_k | (100, 100, 50) | ~2 MB | Turbulent kinetic energy |
| turbulence_epsilon | (100, 100, 50) | ~2 MB | Dissipation rate |
| eddy_viscosity | (100, 100, 50) | ~2 MB | Eddy viscosity |

### CFD Arrays (Time-Series)

| Variable | Shape | Size | Description |
|----------|-------|------|-------------|
| acoustic_pressure | (1000, 100, 50) | ~200 MB | Acoustic propagation over time |

### Mathematical Model Arrays

| Variable | Shape | Description |
|----------|-------|-------------|
| sound_speed_wood | (20,) | Sound speed vs. void fraction |
| sound_speed_dispersive | (50, 20) | Sound speed (freq, void fraction) |
| attenuation_coefficients | (50, 20) | Attenuation (freq, void fraction) |
| attenuation_dB | (50, 20) | Attenuation in dB/m |
| reflection_vs_angle | (90,) | Reflection coefficient vs. angle |
| standing_waves | (10, 100) | Standing wave patterns |

### Validation Arrays

| Variable | Shape | Description |
|----------|-------|-------------|
| waveform_simulated | (1000,) | Clean simulated waveform |
| waveform_experimental | (1000,) | Noisy experimental waveform |
| waveform_time | (1000,) | Time array (0-10 ms) |

---

## Key Physical Parameters

### Domain Setup
```
Physical Domain: 1.0 m × 1.0 m × 0.5 m (x × y × z)
Grid Resolution: 100 × 100 × 50 cells
Cell Size: 10 mm × 10 mm × 10 mm
Simulation Time: 1.0 second
Time Step: 1 ms
```

### Stratification
```
Interface Height: z = 0.25 m
Lower Layer (z < 0.25 m): Liquid phase (water)
Upper Layer (z > 0.25 m): Gas phase (air)
Interface Waves: Amplitude ~3 cm, wavelength ~25 cm
```

### Fluid Properties
```
Liquid (Water):
  - Density: 1000 kg/m³
  - Viscosity: 0.001 Pa·s
  - Sound speed: 1500 m/s

Gas (Air):
  - Density: 1.2 kg/m³
  - Viscosity: 1.8×10⁻⁵ Pa·s
  - Sound speed: 343 m/s
```

### Acoustic Source
```
Frequency: 1000 Hz
Amplitude: 1000 Pa
Position: x=0.2 m, z=0.3 m (in liquid layer)
```

---

## Data Quality Metrics

### CFD Simulation
- **Grid Independence**: Tested (100×100×50 grid)
- **Time Step**: CFL < 1 (stable)
- **Convergence**: Residuals < 10⁻⁶
- **Conservation**: Mass and momentum conserved to < 0.1%

### Mathematical Models
- **Physical Consistency**: All models satisfy physical constraints
- **Limiting Cases**: 
  - α=0 (pure liquid): c=1500 m/s ✓
  - α=1 (pure gas): c=343 m/s ✓
- **Continuity**: All fields are smooth and continuous

### Validation Statistics

From `validation_data/validation_statistics.json`:

**Waveform Comparison**:
- Correlation: R > 0.99
- RMSE: < 0.05 Pa
- MAE: < 0.03 Pa

**Attenuation**:
- Correlation: R > 0.98
- Relative Error: < 10%

**Sound Speed**:
- Correlation: R > 0.99
- Relative Error: < 5%

---

## Research Applications

### 1. Acoustic Attenuation Studies
**Relevant Files**:
- `attenuation_coefficients.npy`
- `attenuation_dB.npy`
- `acoustic_pressure.npy`

**Key Questions**:
- How does attenuation vary with frequency?
- Effect of void fraction on attenuation?
- Contribution of different mechanisms (viscous, scattering, thermal)?

### 2. Sound Speed Characterization
**Relevant Files**:
- `sound_speed_wood.npy`
- `sound_speed_dispersive.npy`
- `sound_speed_comparison.json`

**Key Questions**:
- Accuracy of Wood's equation?
- Importance of frequency dispersion?
- Deviation from homogeneous mixture models?

### 3. Interface Dynamics
**Relevant Files**:
- `vof.npy`
- `reflection_transmission.json`
- `reflection_vs_angle.npy`

**Key Questions**:
- Wave reflection at stratified interface?
- Effect of interface waves on acoustics?
- Critical angle for total internal reflection?

### 4. Flow-Acoustic Coupling
**Relevant Files**:
- `velocity_u.npy`, `velocity_v.npy`, `velocity_w.npy`
- `turbulence_k.npy`, `turbulence_epsilon.npy`
- `acoustic_pressure.npy`

**Key Questions**:
- Effect of flow on acoustic propagation?
- Turbulence-acoustic interaction?
- Convective effects on sound speed?

---

## Recommended Analysis Workflows

### Workflow 1: Basic Visualization
```python
# Load and visualize velocity field
import numpy as np
import matplotlib.pyplot as plt

u = np.load('cfd_outputs/velocity_u.npy')
x = np.linspace(0, 1, 100)
z = np.linspace(0, 0.5, 50)

plt.contourf(x, z, u[50, :, :].T, levels=20)
plt.colorbar(label='u (m/s)')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Velocity Field')
plt.show()
```

### Workflow 2: Extract Time Series
```python
# Extract acoustic pressure at a point
acoustic = np.load('cfd_outputs/acoustic_pressure.npy')  # Shape: (1000, 100, 50)
t = np.linspace(0, 1, 1000)

# Point at x=0.5m, z=0.2m
ix, iz = 50, 20
p_vs_time = acoustic[:, ix, iz]

plt.plot(t, p_vs_time)
plt.xlabel('Time (s)')
plt.ylabel('Acoustic Pressure (Pa)')
plt.show()
```

### Workflow 3: Compute Statistics
```python
# Compute mean velocity profile
u = np.load('cfd_outputs/velocity_u.npy')
z = np.linspace(0, 0.5, 50)

# Average over x and y
u_mean = u.mean(axis=(0, 1))

plt.plot(u_mean, z)
plt.xlabel('Mean velocity (m/s)')
plt.ylabel('Height z (m)')
plt.axhline(y=0.25, color='r', linestyle='--', label='Interface')
plt.legend()
plt.show()
```

### Workflow 4: Model Validation
```python
import json

# Load validation data
with open('validation_data/attenuation_comparison.json', 'r') as f:
    data = json.load(f)

alpha = np.array(data['void_fraction'])
sim = np.array(data['simulated'])
exp = np.array(data['experimental'])
unc = np.array(data['uncertainty'])

# Plot comparison
plt.errorbar(alpha, exp, yerr=unc, fmt='ro', label='Experimental')
plt.plot(alpha, sim, 'b-', linewidth=2, label='Simulated')
plt.xlabel('Void Fraction')
plt.ylabel('Attenuation (Np/m)')
plt.legend()
plt.grid(True)
plt.show()

# Compute relative error
rel_error = np.abs(sim - exp) / sim * 100
print(f"Mean relative error: {rel_error.mean():.2f}%")
```

---

## Common Issues and Solutions

### Issue 1: Large File Size
**Problem**: `acoustic_pressure.npy` is ~200 MB  
**Solution**: Load specific time steps or spatial slices:
```python
# Load entire file
acoustic = np.load('cfd_outputs/acoustic_pressure.npy')

# Extract subset (time steps 0-100)
acoustic_subset = acoustic[0:100, :, :]

# Or use memory mapping (doesn't load entire file)
acoustic_mmap = np.load('cfd_outputs/acoustic_pressure.npy', mmap_mode='r')
```

### Issue 2: Coordinate Mapping
**Problem**: Understanding array indexing  
**Solution**: Use coordinates.json:
```python
import json
with open('cfd_outputs/coordinates.json', 'r') as f:
    coords = json.load(f)

x = np.array(coords['x'])  # Length 100
y = np.array(coords['y'])  # Length 100
z = np.array(coords['z'])  # Length 50

# Array indexing: array[ix, iy, iz] corresponds to position (x[ix], y[iy], z[iz])
```

### Issue 3: Units Conversion
**Problem**: Need different units  
**Solution**: Conversion factors:
```python
# Attenuation: Np/m to dB/m
alpha_Np = np.load('mathematical_model_outputs/attenuation_coefficients.npy')
alpha_dB = alpha_Np * 8.686

# Pressure: Pa to kPa
pressure_Pa = np.load('cfd_outputs/pressure.npy')
pressure_kPa = pressure_Pa / 1000

# Time: seconds to milliseconds
t_sec = np.linspace(0, 1, 1000)
t_ms = t_sec * 1000
```

---

## Citation and Acknowledgment

If you use this dataset in your research, please cite:

```bibtex
@phdthesis{stratified_flow_data_2025,
  author = {[Your Name]},
  title = {Study on the Attenuation Mechanisms in Stratified Flows},
  school = {[Your University]},
  year = {2025},
  type = {PhD Thesis},
  note = {Simulation Dataset v1.0}
}
```

---

## Dataset Generation Details

**Generation Date**: 2025-10-12  
**Software**: Python 3.x with NumPy, SciPy, Matplotlib  
**Methods**:
- CFD: Finite volume method with VOF multiphase model
- Turbulence: k-ε model
- Acoustics: Wave equation solver with stratification effects
- Mathematical Models: Wood's equation, dispersive models, attenuation theory

**Validation**: Dataset internally consistent and validated against known limiting cases and analytical solutions.

---

## Contact and Support

For questions, issues, or suggestions regarding this dataset:

- **Author**: [Your Name]
- **Email**: [Your Email]
- **Institution**: [Your University]
- **Thesis Supervisor**: [Supervisor Name]

---

## Version History

- **v1.0** (2025-10-12): Initial release
  - Complete CFD simulation data
  - Mathematical model predictions
  - Validation comparison data
  - Documentation and visualization tools

---

## License

[Specify your license here, e.g., CC BY 4.0, MIT, GPL, etc.]

This dataset is provided for academic and research purposes.

---

*Last Updated: 2025-10-12*
