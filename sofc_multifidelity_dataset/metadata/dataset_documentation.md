
# Multi-Fidelity SOFC Degradation Dataset
## PhD Thesis: Multi-Fidelity Digital Twin for SOFCs

Generated: 20251015_200820

## Dataset Overview

This dataset contains multi-fidelity data for training and validating Digital Twin models
for Solid Oxide Fuel Cell (SOFC) thermo-mechanical degradation prediction.

### Directory Structure

```
sofc_multifidelity_dataset/
├── phase1_LF/                          # Low-Fidelity (10,000 samples)
│   ├── phase1_LF_complete.csv          # All LF data
│   ├── phase1_LF_complete.h5           # HDF5 format
│   └── phase1_LF_statistics.csv        # Statistical summary
│
├── phase2_MF/                          # Mid-Fidelity (5,000 samples)
│   ├── phase2_MF_global.csv            # Global parameters
│   ├── phase2_MF_complete.h5           # With spatial fields
│   └── phase2_MF_statistics.csv        # Statistical summary
│
├── phase3_HF/                          # High-Fidelity (250 samples)
│   ├── phase3_HF_complete.csv          # All HF data
│   ├── phase3_HF_complete.h5           # With 3D fields
│   └── phase3_HF_statistics.csv        # Statistical summary
│
├── phase4_experimental/                # Experimental (15 cells)
│   ├── experimental_summary.csv        # Cell summary
│   ├── IV_curves_timeseries.csv        # I-V characterization
│   ├── EIS_measurements.csv            # Impedance spectroscopy
│   ├── microstructural_characterization.csv  # SEM/FIB data
│   └── experimental_complete.h5        # Complete experimental data
│
└── metadata/
    └── dataset_documentation.md        # This file
```

## Phase 1: Low-Fidelity Dataset

**Purpose:** Fast surrogate model training, uncertainty quantification, parameter screening

**Fidelity Level:** Low (1D/lumped parameter models)

**Sample Size:** 10,000

**Input Variables:**
- Operating conditions: Temperature, current density, fuel utilization, pressure
- Cycling parameters: Number of cycles, thermal cycling rate
- Geometry: Cell thickness, active area

**Output Variables:**
- **Thermo-electrical:** Voltage, power density, average temperature, temperature gradient
- **Mechanical:** Volume-averaged thermal stress, CTE mismatch stress, von Mises stress
- **Degradation:** Ni particle size, crack probability, delamination risk, time-to-failure

**Use Cases:**
- Training fast surrogate models
- Global sensitivity analysis
- Uncertainty propagation
- Initial parameter estimation

## Phase 2: Mid-Fidelity Dataset

**Purpose:** Multi-physics coupling, spatial pattern learning

**Fidelity Level:** Medium (2D/3D coarse grid CFD-FEM)

**Sample Size:** 5,000 global + 100 with spatial fields

**Input Variables:**
- All Phase 1 inputs plus:
- Detailed geometry: Layer thicknesses, rib/channel dimensions
- Material properties: Porosity, TPB density
- Flow conditions: H₂ and air flow rates

**Output Variables:**
- **Spatial fields (2D):** Temperature, current density, stress, H₂ and O₂ concentrations
- **Global metrics:** Max/min/std of all spatial quantities
- **Degradation:** Ni coarsening rate, crack density, delamination area

**Spatial Resolution:** 50×30 grid (coarse)

**Use Cases:**
- Training CNN/U-Net for spatial prediction
- Physics-informed neural networks
- Reduced-order modeling
- Multi-fidelity fusion

## Phase 3: High-Fidelity Dataset

**Purpose:** Ground truth for critical cases, damage prediction

**Fidelity Level:** High (3D fine grid FEM with microstructure)

**Sample Size:** 250 global + 20 with spatial fields

**Input Variables:**
- All Phase 2 inputs plus:
- Microstructure: Ni/YSZ grain sizes, interface roughness, tortuosity
- Temperature-dependent material properties
- Thermal cycling amplitude

**Output Variables:**
- **Detailed overpotentials:** Activation, ohmic, concentration (spatial)
- **3D stress/strain fields:** Principal stresses, von Mises, elastic & creep strain
- **Explicit damage:**
  - Crack initiation indicator, crack length, crack propagation rate
  - Strain energy release rate, delamination indicator
  - Ni coarsening (LSW + electrochemical)
  - TPB loss
- **Life prediction:** Cycles to failure, remaining life, performance degradation

**Spatial Resolution:** 100×60 grid for 2D slices (storage efficient)

**Use Cases:**
- Validating lower-fidelity models
- Training damage prediction models
- Failure criterion development
- Digital Twin calibration

## Phase 4: Experimental Validation Dataset

**Purpose:** Real-world validation, model calibration gold standard

**Fidelity Level:** Experimental measurements

**Sample Size:** 15 cells with multi-scale characterization

**Test Conditions:**
- Low-stress baseline (873 K, 0.5 A/cm²)
- Nominal operation (973 K, 1.0 A/cm²)
- High-stress (1073 K, 1.2 A/cm²)
- High fuel utilization (973 K, Uf=0.85)
- Thermal cycling

**Measurements:**

### Electrochemical Performance
- **I-V Curves:** Time-series at 0, 100, 500, 1000, 2000 hours
- **EIS:** Impedance spectra (0.01 Hz - 100 kHz) tracking degradation

### Microstructural Characterization (SEM-FIB)
- Ni particle size evolution (BOL, mid-life, EOL)
- TPB density changes
- Porosity evolution
- Crack observation (binary + length)
- Delamination detection (binary + area)

### Spatial Measurements
- **Thermography:** IR temperature maps (20×20 grid) for 5 cells
- Hot spot detection and evolution

**Use Cases:**
- Final model validation
- Uncertainty quantification bounds
- Failure mode identification
- Publication-quality results

## Variable Definitions

### Input Variables

| Variable | Symbol | Unit | Range | Description |
|----------|--------|------|-------|-------------|
| Fuel Utilization | Uf | - | 0.5-0.9 | Fraction of H₂ consumed |
| Operating Temperature | T | K | 873-1073 | Stack temperature |
| Current Density | i | A/cm² | 0.2-1.5 | Electrical load |
| Pressure | P | atm | 1.0-3.0 | Operating pressure |
| Thermal Cycles | N | - | 0-5000 | Number of start-stop cycles |
| Cell Thickness | δ | mm | 0.5-2.0 | Total cell thickness |
| Porosity | ε | - | 0.25-0.45 | Electrode porosity |
| TPB Density | λ | μm/μm³ | 1-8 | Three-phase boundary |

### Output Variables - Thermo-Electrical

| Variable | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Voltage | V | V | Cell voltage |
| Power Density | P | W/cm² | Electrical power output |
| Temperature Field | T(x,y,z) | K | Spatial temperature |
| Current Density Field | i(x,y,z) | A/cm² | Local current production |
| Overpotentials | η | V | Activation, ohmic, concentration |
| Species Concentrations | C | mol fraction | H₂, H₂O, O₂ distributions |

### Output Variables - Mechanical

| Variable | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Stress Field | σ(x,y,z) | MPa | Stress tensor components |
| Von Mises Stress | σᵥₘ | MPa | Equivalent stress |
| Strain Field | ε(x,y,z) | % | Elastic + creep strain |
| CTE Mismatch Stress | σ_CTE | MPa | Thermal expansion mismatch |

### Output Variables - Degradation

| Variable | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Ni Particle Size | r_Ni | nm | Anode Ni coarsening |
| TPB Density | λ(t) | μm/μm³ | Degrading TPB |
| Crack Indicator | Γ_crack | - | 0-2, >0.8 = initiated |
| Crack Length | a | μm | Physical crack size |
| Delamination Indicator | Γ_delam | - | Based on G/G_IC |
| Time to Failure | t_f | hours | Until 10% voltage drop |

## Physics Models Used

### Electrochemistry
- **Nernst Equation:** Reversible voltage
- **Butler-Volmer:** Activation overpotential
- **Ohm's Law:** Ionic resistance (temperature dependent)
- **Fick's Law:** Mass transport limitations

### Thermal
- **Heat Generation:** Joule heating + activation losses
- **Fourier's Law:** Conduction with spatial variations
- **Convection:** Channel flow effects

### Mechanics
- **Thermal Stress:** σ_th = α·ΔT·E/(1-2ν)
- **CTE Mismatch:** Multi-layer expansion mismatch
- **Creep:** Power-law creep (Arrhenius)
- **Fatigue:** Coffin-Manson relationship

### Degradation
- **Ni Coarsening:** LSW theory + electrochemical acceleration
- **Crack Initiation:** Griffith criterion
- **Crack Propagation:** Paris law (fatigue)
- **Delamination:** Strain energy release rate (G > G_IC)
- **TPB Loss:** Geometric scaling with particle size

## Data Formats

### CSV Files
- Standard comma-separated format
- Headers with variable names (units in description)
- One row per sample
- Missing data: NaN

### HDF5 Files
Hierarchical structure:

```
/
├── inputs/               # Input parameters
│   ├── fuel_utilization
│   ├── operating_temperature_K
│   └── ...
│
├── outputs/
│   ├── thermo_electrical/
│   │   ├── voltage_V
│   │   └── ...
│   ├── mechanical/
│   │   ├── stress_MPa
│   │   └── ...
│   └── degradation/
│       ├── Ni_particle_size_nm
│       └── ...
│
└── spatial_fields/       # For MF and HF only
    ├── sample_0/
    │   ├── temperature_K
    │   ├── stress_MPa
    │   └── damage_indicator
    └── ...
```

## Usage Examples

### Python - Loading Data

```python
import pandas as pd
import h5py
import numpy as np

# Load low-fidelity CSV
df_lf = pd.read_csv('phase1_LF/phase1_LF_complete.csv')

# Load high-fidelity HDF5
with h5py.File('phase3_HF/phase3_HF_complete.h5', 'r') as f:
    # Scalar data
    stress = f['scalar_outputs/max_stress_MPa'][:]
    
    # Spatial data
    T_field = f['spatial_fields_2D_slices/sample_0/temperature_K'][:]
    
    # Metadata
    print(f.attrs['description'])

# Load experimental I-V curves
df_iv = pd.read_csv('phase4_experimental/IV_curves_timeseries.csv')
cell1 = df_iv[df_iv['cell_id'] == 'SOFC_EXP_001']
```

## Multi-Fidelity Modeling Workflow

### Recommended Training Strategy

1. **Stage 1: Surrogate Model (LF)**
   - Train fast neural network on Phase 1 data
   - Input: Operating conditions → Output: Global degradation metrics
   - Model: Simple MLP (100-500k parameters)

2. **Stage 2: Spatial Model (MF)**
   - Train CNN/U-Net on Phase 2 spatial fields
   - Input: Operating + geometry → Output: 2D temperature, stress, damage maps
   - Model: U-Net or ConvLSTM

3. **Stage 3: High-Fidelity Correction (HF)**
   - Train correction model: Output_HF = Output_MF + Δ
   - Use Phase 3 data for residual learning
   - Model: Smaller network learning the bias

4. **Stage 4: Multi-Fidelity Fusion**
   - Combine all fidelities with uncertainty quantification
   - Gaussian Process, Bayesian Neural Network, or Ensemble
   - Calibrate with Phase 4 experimental data

5. **Validation**
   - Hold out 20% of Phase 4 for final testing
   - Compare predictions vs. real degradation
   - Publish uncertainty bounds

## Citation

If you use this dataset, please cite:

```
@dataset{sofc_multifidelity_2025,
  title={Multi-Fidelity Digital Twin Dataset for SOFC Thermo-Mechanical Degradation},
  author={[Your Name]},
  year={2025},
  institution={[Your University]},
  description={Synthetic and experimental multi-scale degradation data for SOFCs}
}
```

## Data Quality Notes

### Synthetic Data (Phases 1-3)
- Generated using physics-informed models
- Assumptions:
  - Ideal gas behavior
  - Uniform initial microstructure
  - No manufacturing defects (unless specified)
  - Simplified channel geometry

### Experimental Data (Phase 4)
- Realistic measurement noise included
- Limited sample size (n=15) typical for PhD
- Some cells have incomplete data (normal for long-term tests)
- Thermography only available for 5 cells

### Known Limitations
- No chemical degradation (sulfur poisoning, carbon deposition)
- No redox cycling effects
- Simplified seal mechanics
- 2D approximations in some MF cases

## Contact

For questions about this dataset:
- Email: [your.email@university.edu]
- GitHub: [repository link]

## License

This dataset is provided for academic research purposes only.

---

**Version:** 1.0
**Last Updated:** 20251015_200820
