# Stratified Flow Simulation Dataset

## PhD Thesis: Study on the Attenuation Mechanisms in Stratified Flows

This dataset contains comprehensive simulation data for studying acoustic attenuation mechanisms in stratified multiphase flows. The data is generated from CFD (Computational Fluid Dynamics) and mathematical models.

---

## Dataset Overview

### Dataset Structure

```
stratified_flow_simulation_data/
│
├── cfd_outputs/                      # CFD Model Outputs
│   ├── velocity_u.npy                # Velocity field in x-direction
│   ├── velocity_v.npy                # Velocity field in y-direction
│   ├── velocity_w.npy                # Velocity field in z-direction
│   ├── pressure.npy                  # Pressure field
│   ├── vof.npy                       # Volume of Fluid (phase distribution)
│   ├── turbulence_k.npy              # Turbulent kinetic energy
│   ├── turbulence_epsilon.npy        # Turbulent dissipation rate
│   ├── eddy_viscosity.npy            # Eddy viscosity
│   ├── acoustic_pressure.npy         # Acoustic pressure propagation
│   └── coordinates.json              # Grid coordinates and metadata
│
├── mathematical_model_outputs/       # Mathematical Model Outputs
│   ├── sound_speed_wood.npy          # Sound speed (Wood's equation)
│   ├── sound_speed_dispersive.npy    # Sound speed (dispersive model)
│   ├── attenuation_coefficients.npy  # Attenuation coefficients (Np/m)
│   ├── attenuation_dB.npy            # Attenuation coefficients (dB/m)
│   ├── reflection_vs_angle.npy       # Reflection coefficient vs angle
│   ├── standing_waves.npy            # Standing wave patterns
│   ├── reflection_transmission.json  # Interface reflection/transmission
│   ├── time_delays.json              # Time-delay estimates (T₀)
│   └── parameters.json               # Model parameters
│
└── validation_data/                  # Model Validation Data
    ├── waveform_simulated.npy        # Simulated acoustic waveform
    ├── waveform_experimental.npy     # "Experimental" waveform (with noise)
    ├── waveform_time.npy             # Time array for waveforms
    ├── attenuation_comparison.json   # Attenuation: simulated vs experimental
    ├── sound_speed_comparison.json   # Sound speed: simulated vs experimental
    └── validation_statistics.json    # Statistical comparison metrics
```

---

## Data Categories

### 1. CFD Model Outputs

Generated using numerical simulation methods (equivalent to ANSYS Fluent/COMSOL Multiphysics).

#### Velocity Fields (100×100×50 grid)
- **Files**: `velocity_u.npy`, `velocity_v.npy`, `velocity_w.npy`
- **Description**: 3D velocity components showing stratified flow patterns
- **Units**: m/s
- **Shape**: (100, 100, 50) representing (x, y, z) directions
- **Features**: 
  - Stratified flow with distinct upper (gas) and lower (liquid) layers
  - Interface at z ≈ 0.25 m
  - Velocity shear at interface

#### Pressure Field (100×100×50 grid)
- **File**: `pressure.npy`
- **Description**: 3D pressure distribution including hydrostatic and dynamic components
- **Units**: Pa
- **Shape**: (100, 100, 50)
- **Features**:
  - Hydrostatic pressure gradient
  - Dynamic pressure fluctuations from turbulence and waves

#### Volume of Fluid - VOF (100×100×50 grid)
- **File**: `vof.npy`
- **Description**: Phase distribution field (1 = liquid, 0 = gas)
- **Shape**: (100, 100, 50)
- **Features**:
  - Sharp stratified interface with interfacial waves
  - Bubble inclusions in liquid phase
  - Smooth transition region at interface

#### Turbulence Parameters (100×100×50 grid)
Generated using k-ε turbulence model:

- **Turbulent kinetic energy** (`turbulence_k.npy`): Units: m²/s²
- **Turbulent dissipation rate** (`turbulence_epsilon.npy`): Units: m²/s³
- **Eddy viscosity** (`eddy_viscosity.npy`): Units: Pa·s

**Features**:
- Enhanced turbulence at stratified interface
- Spatial variation with flow structure
- k-ε model parameters (C_μ = 0.09)

#### Acoustic Pressure Propagation (1000×100×50)
- **File**: `acoustic_pressure.npy`
- **Description**: Time-resolved acoustic pressure field (x-z plane)
- **Units**: Pa
- **Shape**: (time_steps, x_points, z_points) = (1000, 100, 50)
- **Source Parameters**:
  - Frequency: 1000 Hz
  - Position: x=0.2 m, z=0.3 m
  - Amplitude: 1000 Pa
- **Features**:
  - Wave propagation through stratified medium
  - Interface reflection and transmission
  - Frequency-dependent attenuation
  - Different sound speeds in each layer

---

### 2. Mathematical Model Outputs

Generated using analytical and semi-analytical models (MATLAB/Python equivalent).

#### Sound Speed Predictions

**Wood's Equation Model** (`sound_speed_wood.npy`)
- Homogeneous mixture model
- Shape: (20,) for void fractions 0 to 1
- Reference: Wood, A.B. (1930)

**Dispersive Model** (`sound_speed_dispersive.npy`)
- Frequency-dependent model (Eq. 27 from thesis)
- Shape: (50, 20) = (frequencies, void_fractions)
- Frequencies: 100 Hz to 10 kHz
- Accounts for dispersion effects

#### Attenuation Coefficients

**Total Attenuation** (`attenuation_coefficients.npy`)
- Units: Np/m (Nepers per meter)
- Shape: (50, 20) = (frequencies, void_fractions)
- Components:
  - Viscous attenuation
  - Scattering attenuation (bubble-induced)
  - Thermal attenuation

**Attenuation in dB/m** (`attenuation_dB.npy`)
- Units: dB/m
- Conversion: α(dB/m) = α(Np/m) × 8.686

#### Wave Propagation Patterns

**Reflection and Transmission** (`reflection_transmission.json`)
- Normal incidence coefficients at liquid-gas interface
- Acoustic impedances: Z₁ (liquid), Z₂ (gas)
- Reflection coefficient: R = (Z₂-Z₁)/(Z₂+Z₁)
- Transmission coefficient: T = 2Z₂/(Z₂+Z₁)

**Oblique Incidence** (`reflection_vs_angle.npy`)
- Angle-dependent reflection and transmission
- Angles: 0° to 89°
- Shows critical angle for total internal reflection

**Standing Waves** (`standing_waves.npy`)
- Standing wave patterns from incident + reflected waves
- Shape: (10, 100) = (frequencies, positions)

#### Time-Delay Estimates (T₀)

**File**: `time_delays.json`

Time delays for acoustic signal propagation at different void fractions:
- Void fractions: 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
- Distances: 0.1 m to 5.0 m
- Includes effective sound speed for each configuration

---

### 3. Model Validation Data

Direct comparison between simulated and "experimental" data for model validation.

#### Acoustic Waveform Comparison

**Files**: 
- `waveform_simulated.npy`: Clean simulated waveform
- `waveform_experimental.npy`: Noisy experimental-like waveform
- `waveform_time.npy`: Time array (0-10 ms, 1000 points)

**Features**:
- Source frequency: 2000 Hz
- Exponential decay with propagation
- Experimental data includes realistic noise

**Statistics** (from `validation_statistics.json`):
- Correlation coefficient
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

#### Attenuation Comparison

**File**: `attenuation_comparison.json`

- Void fractions: 0.0 to 0.5
- Simulated values from theoretical model
- Experimental values with ±5% uncertainty
- Frequency: 1000 Hz

#### Sound Speed Comparison

**File**: `sound_speed_comparison.json`

- Void fractions: 0.0 to 0.5
- Simulated: Wood's equation
- Experimental: With ±3% measurement uncertainty
- Statistical metrics included

---

## Physical Parameters

### Fluid Properties

| Property | Liquid (Water) | Gas (Air) |
|----------|----------------|-----------|
| Density (ρ) | 1000 kg/m³ | 1.2 kg/m³ |
| Viscosity (μ) | 0.001 Pa·s | 1.8×10⁻⁵ Pa·s |
| Sound speed (c) | 1500 m/s | 343 m/s |

### Domain Dimensions

- Length (x): 1.0 m
- Width (y): 1.0 m
- Height (z): 0.5 m
- Simulation time: 1.0 s

### Grid Resolution

- x-direction: 100 points (Δx = 10 mm)
- y-direction: 100 points (Δy = 10 mm)
- z-direction: 50 points (Δz = 10 mm)
- Time steps: 1000 (Δt = 1 ms)

---

## Usage Examples

### Loading Data in Python

```python
import numpy as np
import json

# Load CFD velocity field
velocity_u = np.load('cfd_outputs/velocity_u.npy')
print(f"Velocity u shape: {velocity_u.shape}")  # (100, 100, 50)

# Load coordinates
with open('cfd_outputs/coordinates.json', 'r') as f:
    coords = json.load(f)
x = np.array(coords['x'])
y = np.array(coords['y'])
z = np.array(coords['z'])

# Load mathematical model data
attenuation = np.load('mathematical_model_outputs/attenuation_dB.npy')
with open('mathematical_model_outputs/parameters.json', 'r') as f:
    params = json.load(f)
frequencies = np.array(params['frequencies'])
void_fractions = np.array(params['void_fractions'])

# Load validation data
with open('validation_data/sound_speed_comparison.json', 'r') as f:
    sound_speed_comp = json.load(f)
```

### Visualization

Use the provided `visualize_data.py` script:

```bash
python3 visualize_data.py
```

This generates publication-quality figures including:
- Velocity field slices
- VOF phase distribution
- Acoustic pressure snapshots
- Attenuation vs. frequency and void fraction
- Sound speed predictions
- Validation comparisons

---

## Software and Methods

### CFD Simulations (Equivalent)

**Software**: ANSYS Fluent/CFX, COMSOL Multiphysics (equivalent methods used)

**Models**:
- Turbulence: k-ε model (C_μ = 0.09)
- Multiphase: VOF (Volume of Fluid) method
- Acoustic: Wave equation solver

### Mathematical Models

**Software**: Python (NumPy, SciPy)

**Methods**:
- Wood's equation for sound speed
- Modified dispersive wave equation
- Transfer-matrix method for reflection/transmission
- Attenuation models (viscous, scattering, thermal)

---

## Applications

This dataset can be used for:

1. **Model Development**: Developing new acoustic attenuation models for stratified flows
2. **Hypothesis Testing**: Testing theories about attenuation mechanisms
3. **Validation**: Comparing theoretical predictions with simulated/experimental data
4. **Machine Learning**: Training ML models for acoustic property prediction
5. **Sensitivity Analysis**: Understanding effects of void fraction, frequency, etc.
6. **Publication**: Generating figures and results for research papers

---

## Citation

If you use this dataset, please cite:

```
[Your Name], "Study on the Attenuation Mechanisms in Stratified Flows: 
Simulation Data for Model Development and Hypothesis Testing", 
PhD Thesis, [University], [Year]
```

---

## File Formats

- **`.npy` files**: NumPy binary format (use `np.load()`)
- **`.json` files**: JSON format for metadata and parameters (use `json.load()`)

---

## Data Size

- Total files: 24 data files + 4 auxiliary files
- Approximate size: ~500 MB (uncompressed)
- Largest files: `acoustic_pressure.npy` (~200 MB)

---

## Contact

For questions about this dataset:
- Author: [Your Name]
- Email: [Your Email]
- Institution: [Your University]

---

## Version History

- **v1.0** (2025-10-12): Initial dataset release
  - CFD outputs: 9 files
  - Mathematical model outputs: 8 files
  - Validation data: 7 files

---

## License

[Specify your license, e.g., CC BY 4.0, MIT, etc.]

---

## Acknowledgments

Generated using Python-based simulation tools and analytical models for stratified multiphase flow acoustics research.
