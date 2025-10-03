# SOFC Experimental Measurement Dataset Generator

## 🧪 Overview

This repository contains a comprehensive synthetic experimental dataset generator for Solid Oxide Fuel Cell (SOFC) characterization research. The generated data simulates realistic measurements from:

1. **Digital Image Correlation (DIC)** analysis
2. **Synchrotron X-ray Diffraction (XRD)** measurements
3. **Post-Mortem Analysis** (SEM, EDS, nano-indentation)

The data is fabricated based on realistic physical models and typical experimental parameters for SOFC research.

---

## 📊 Generated Datasets

### 1. Digital Image Correlation (DIC) Data

**Directory:** `sofc_experimental_data/dic_data/`

#### a. Sintering Strain Data
- **File:** `sintering_strain_data.csv`
- **Temperature range:** 1200–1500°C
- **Duration:** 180 minutes
- **Contains:** Strain evolution (εxx, εyy, εxy) across different SOFC regions
- **Regions:** Anode, Electrolyte, Cathode, Interface zones

#### b. Thermal Cycling Data
- **File:** `thermal_cycling_strain_data.csv`
- **Temperature cycling:** ΔT = 400°C (800°C ± 400°C)
- **Number of cycles:** 10
- **Contains:** Real-time strain maps with damage accumulation
- **Visualization:** `thermal_cycling_analysis.png`

#### c. Startup/Shutdown Cycles
- **File:** `startup_shutdown_cycles.csv`
- **Number of cycles:** 5
- **Phases:** Startup (RT → 800°C), Operation (800°C hold), Shutdown (800°C → RT)
- **Contains:** Thermal strain evolution and mechanical stress accumulation

#### d. Speckle Pattern Images
- **Directory:** `speckle_patterns/`
- **Files:** 5 PNG images with metadata JSON files
- **Timestamps:** t0_25C, t1_400C, t2_800C, t3_1200C, t4_1500C
- **Format:** 512×512 pixels, 2.5 μm/pixel resolution
- **Equipment specs:** Allied Vision Manta G-504B camera, Computar 50mm f/2.8 lens

#### e. Lagrangian Strain Tensors (Vic-3D Format)
- **Directory:** `lagrangian_tensors/`
- **Files:** 5 CSV files at different temperatures (25, 400, 800, 1200, 1500°C)
- **Grid:** 100×80 spatial points (10mm × 8mm)
- **Components:** Exx, Eyy, Exy, Ezz, Von Mises strain
- **Visualizations:** Contour maps for each temperature

#### f. Strain Hotspots
- **Directory:** `strain_hotspots/`
- **File:** `strain_hotspot_catalog.csv`
- **Threshold:** Von Mises strain > 1.0%
- **Contains:** Localized high-strain regions (>1.0% strain at interfaces)
- **Classification:** Critical (>1.5%) and High (>1.0%) severity levels

---

### 2. Synchrotron X-ray Diffraction (XRD) Data

**Directory:** `sofc_experimental_data/xrd_data/`

#### a. Residual Stress Profiles
- **File:** `residual_stress_profiles.csv`
- **Depth range:** 0–800 μm across SOFC cross-section
- **Conditions:** As-sintered, After thermal cycling, After 100h operation
- **Stress components:** σ11, σ22, σ33, hydrostatic, Von Mises
- **Phases:** Ni-YSZ (anode), YSZ (electrolyte), LSM (cathode)

#### b. Lattice Strain Measurements
- **File:** `lattice_strain_vs_temperature.csv`
- **Temperature range:** 25–1500°C (50 points)
- **Phases:** YSZ (a₀=5.14 Å), Ni (a₀=3.52 Å), LSM (a₀=5.50 Å)
- **Contains:** Thermal expansion, elastic strain, lattice parameters
- **Visualization:** `lattice_strain_analysis.png`

#### c. Sin²ψ Method Peak Shift Data
- **File:** `sin2psi_stress_analysis.csv`
- **Method:** Standard sin²ψ stress measurement technique
- **Tilt angles:** ψ = -45° to +45° (15 angles)
- **Locations:** 5 positions (anode, interface_ae, electrolyte, interface_ec, cathode)
- **Radiation:** Cu Kα (λ = 1.5406 Å)
- **Contains:** d-spacing, 2θ angles, calculated stresses

#### d. Microcrack Initiation Thresholds
- **File:** `microcrack_threshold_data.csv`
- **Specimens:** 30 samples with varying strain levels (0.5%–3.5%)
- **Critical strain:** εcr > 0.02 (2%)
- **Contains:**
  - Crack density (cracks/mm²)
  - Average crack length and opening (μm)
  - XRD peak broadening (FWHM)
  - Crystallite size (nm)
- **Visualization:** `microcrack_threshold_analysis.png`

---

### 3. Post-Mortem Analysis Data

**Directory:** `sofc_experimental_data/postmortem_data/`

#### a. SEM Crack Density Quantification
- **Directory:** `sem_analysis/`
- **File:** `crack_density_analysis.csv`
- **Specimens:** 6 conditions (pristine, 10h, 50h, 100h, 200h operation, 10 thermal cycles)
- **ROIs:** 10 regions per specimen × 5 regions (anode, electrolyte, cathode, interfaces)
- **Measurements:**
  - Crack density (cracks/mm²)
  - Average crack length (μm)
  - Average crack width (μm)
  - Maximum crack length (μm)
- **Imaging parameters:** 5000× magnification, 15 kV accelerating voltage

#### b. EDS Line Scans for Elemental Composition
- **Directory:** `eds_analysis/`
- **File:** `eds_line_scan_cross_section.csv`
- **Scan length:** 0–800 μm across SOFC stack
- **Elements tracked:** Ni, Zr, Y, O, La, Sr, Mn (wt%)
- **Resolution:** 400 measurement points (2 μm spacing)
- **Contains:**
  - Anode composition (Ni-YSZ): ~40% Ni, 35% Zr, 5% Y, 20% O
  - Electrolyte (YSZ): ~53% Zr, 7% Y, 40% O
  - Cathode (LSM): ~15% La, 3% Sr, 7% Mn, 40% O
- **Point analysis:** `eds_point_analysis.csv` (5 replicates per location)
- **Visualization:** `eds_line_scan.png`

#### c. Nano-indentation Data
- **Directory:** `nanoindentation/`

**Mapping Data:**
- **File:** `nanoindentation_map.csv`
- **Spatial grid:** 50×40 indents (10mm × 0.8mm coverage)
- **Properties measured:**
  - **Young's Modulus:**
    - YSZ (electrolyte): 184.7 GPa
    - Ni-YSZ (anode): 109.8 GPa
    - LSM (cathode): 120.0 GPa
  - **Hardness:**
    - YSZ: 12.5 GPa
    - Ni-YSZ: 5.8 GPa
    - LSM: 6.5 GPa
  - **Creep compliance** values
- **Test parameters:**
  - Max load: 10 mN
  - Loading rate: 1 mN/s
  - Hold time: 10 s
- **Visualization:** `nanoindentation_property_maps.png` (3 contour maps)

**Load-Displacement Curves:**
- **Directory:** `load_displacement_curves/`
- **Files:** 4 CSV files + PNG plots for anode, electrolyte, cathode, interface
- **Contains:** Time-resolved load and displacement data
- **Phases:** Loading → Hold (creep) → Unloading
- **Analysis:** Oliver-Pharr method for property extraction

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download this repository
cd /workspace

# Install required dependencies
pip install numpy pandas matplotlib scipy

# Generate the synthetic dataset
python3 generate_sofc_experimental_data.py
```

### Generated Output

All data will be created in the `sofc_experimental_data/` directory with the following structure:

```
sofc_experimental_data/
├── DATA_SUMMARY.txt
├── dic_data/
│   ├── sintering_strain_data.csv
│   ├── thermal_cycling_strain_data.csv
│   ├── startup_shutdown_cycles.csv
│   ├── speckle_patterns/
│   ├── lagrangian_tensors/
│   └── strain_hotspots/
├── xrd_data/
│   ├── residual_stress_profiles.csv
│   ├── lattice_strain_vs_temperature.csv
│   ├── sin2psi_stress_analysis.csv
│   └── microcrack_threshold_data.csv
└── postmortem_data/
    ├── sem_analysis/
    ├── eds_analysis/
    └── nanoindentation/
```

---

## 📈 Data Usage Examples

### Example 1: Load and Plot DIC Sintering Data

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load sintering strain data
df = pd.read_csv('sofc_experimental_data/dic_data/sintering_strain_data.csv')

# Plot strain evolution for interface region
interface_data = df[df['region'] == 'interface_ae']
plt.figure(figsize=(10, 6))
plt.plot(interface_data['temperature_C'], interface_data['von_mises_strain'] * 100)
plt.xlabel('Temperature (°C)')
plt.ylabel('Von Mises Strain (%)')
plt.title('Strain Evolution at Anode-Electrolyte Interface')
plt.grid(True)
plt.show()
```

### Example 2: Analyze XRD Stress Profiles

```python
import pandas as pd

# Load residual stress data
df = pd.read_csv('sofc_experimental_data/xrd_data/residual_stress_profiles.csv')

# Compare stress profiles across conditions
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
for condition in df['condition'].unique():
    condition_data = df[df['condition'] == condition]
    ax.plot(condition_data['depth_um'], 
            condition_data['von_mises_stress_MPa'],
            label=condition.replace('_', ' '), linewidth=2)

ax.set_xlabel('Depth (μm)')
ax.set_ylabel('Von Mises Stress (MPa)')
ax.set_title('Residual Stress Profiles')
ax.legend()
ax.grid(True)
plt.show()
```

### Example 3: EDS Composition Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load EDS line scan
df = pd.read_csv('sofc_experimental_data/postmortem_data/eds_analysis/eds_line_scan_cross_section.csv')

# Plot elemental distribution
fig, ax = plt.subplots(figsize=(12, 6))
elements = ['Ni_wt%', 'Zr_wt%', 'O_wt%', 'La_wt%']
for element in elements:
    ax.plot(df['distance_um'], df[element], label=element.replace('_wt%', ''), linewidth=2)

ax.set_xlabel('Distance (μm)')
ax.set_ylabel('Weight %')
ax.set_title('EDS Line Scan Across SOFC')
ax.legend()
ax.grid(True)
plt.show()
```

### Example 4: Nano-indentation Property Maps

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load nano-indentation map
df = pd.read_csv('sofc_experimental_data/postmortem_data/nanoindentation/nanoindentation_map.csv')

# Create Young's modulus map
x_unique = sorted(df['x_mm'].unique())
y_unique = sorted(df['y_mm'].unique())
E_grid = df.pivot(index='y_mm', columns='x_mm', values='youngs_modulus_GPa').values

plt.figure(figsize=(10, 6))
plt.contourf(x_unique, y_unique, E_grid, levels=20, cmap='viridis')
plt.colorbar(label="Young's Modulus (GPa)")
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title("Young's Modulus Map")
plt.show()
```

---

## 🔬 Experimental Setup Details

### DIC System
- **Camera:** Allied Vision Manta G-504B
- **Lens:** Computar 50mm f/2.8
- **Resolution:** 2.5 μm/pixel
- **Acquisition rate:** 1 Hz
- **Software:** Vic-3D (Correlated Solutions)
- **Speckle pattern:** High-temperature resistant ceramic paint

### Synchrotron XRD
- **Beamline:** Typical synchrotron source (e.g., APS, ESRF)
- **Wavelength:** Cu Kα (λ = 1.5406 Å) equivalent
- **Spot size:** ~50 μm
- **Detector:** 2D area detector
- **Stress method:** Sin²ψ technique

### SEM/EDS System
- **Microscope:** Field-emission SEM
- **Accelerating voltage:** 15 kV
- **Magnification:** 5000× for crack analysis
- **EDS detector:** Energy-dispersive X-ray spectroscopy
- **Resolution:** ~2 μm lateral resolution

### Nano-indentation
- **System:** Berkovich indenter (three-sided pyramid)
- **Load range:** 0.1–10 mN
- **Displacement resolution:** < 0.1 nm
- **Analysis method:** Oliver-Pharr technique
- **Spacing:** 200 μm between indents

---

## 📝 Data File Formats

### CSV Files
All CSV files contain headers with self-explanatory column names:
- **Strain data:** Contains εxx, εyy, εxy, Von Mises strain
- **Stress data:** Contains σ11, σ22, σ33, hydrostatic, Von Mises
- **Composition data:** Contains weight percentages (wt%)
- **Mechanical properties:** Contains E (GPa), H (GPa), compliance

### JSON Metadata Files
Speckle pattern metadata includes:
- Timestamp
- Image dimensions
- Pixel size (μm)
- Camera model
- Lens specifications
- Exposure settings
- Acquisition rate

### PNG Images
All visualization plots are saved at 150 DPI resolution with:
- Clear axis labels
- Legends
- Colorbars (for contour plots)
- Grid lines for readability

---

## ⚙️ Customization

To modify the data generation parameters, edit `generate_sofc_experimental_data.py`:

```python
# Example: Change temperature range for sintering
temperatures = np.linspace(1200, 1500, 100)  # Modify range here

# Example: Change number of thermal cycles
n_cycles = 10  # Modify here

# Example: Change material properties
E_YSZ = 184.7  # GPa - Young's modulus of YSZ
E_NiYSZ = 109.8  # GPa - Young's modulus of Ni-YSZ
```

---

## 📚 Physical Models Used

### Strain Calculation
- **Thermal strain:** ε_thermal = α × ΔT (thermal expansion coefficient)
- **Von Mises strain:** ε_vm = √(εxx² + εyy² - εxx·εyy + 3·εxy²)
- **Interface effects:** 1.5–2.0× strain amplification

### Stress Calculation
- **Hooke's Law:** σ = E × ε (for elastic regime)
- **Thermal expansion mismatch:** Primary source of residual stress
- **Sin²ψ method:** d_ψ = d₀[1 + (σ(1+ν)/E)·sin²ψ]

### Damage Models
- **Crack initiation:** Critical strain εcr > 0.02 (2%)
- **Crack density:** Increases exponentially with operation time
- **Interface degradation:** Preferential cracking at material interfaces

---

## 🎯 Applications

This synthetic dataset can be used for:

1. **Machine Learning Model Training**
   - Predicting SOFC failure modes
   - Strain-to-stress correlations
   - Lifetime prediction models

2. **Method Development**
   - Testing data analysis pipelines
   - Validating strain measurement algorithms
   - Calibrating stress calculation methods

3. **Educational Purposes**
   - Teaching experimental characterization techniques
   - Understanding SOFC failure mechanisms
   - Learning materials science data analysis

4. **Benchmarking**
   - Comparing experimental setups
   - Validating computational models
   - Testing analysis software

---

## 📊 Key Findings from Generated Data

- **Strain hotspots:** Concentrated at anode-electrolyte and electrolyte-cathode interfaces
- **Residual stresses:** Highest tensile stresses (~200 MPa) at interfaces after thermal cycling
- **Crack initiation:** Occurs at strains exceeding 2% (εcr = 0.02)
- **Thermal cycling damage:** Progressive strain accumulation with each cycle (~5% increase)
- **Material properties:** Interface regions show 15-20% reduction in mechanical properties

---

## 📖 References

### Experimental Techniques
1. Digital Image Correlation (DIC) - Correlated Solutions, Vic-3D
2. Synchrotron X-ray Diffraction - Standard sin²ψ method
3. Nano-indentation - Oliver-Pharr analysis method

### SOFC Materials
- **YSZ (Yttria-Stabilized Zirconia):** E = 184.7 GPa, α = 10.5×10⁻⁶ K⁻¹
- **Ni-YSZ Cermet:** E = 109.8 GPa, α = 12.8×10⁻⁶ K⁻¹
- **LSM (Lanthanum Strontium Manganite):** E = 120.0 GPa, α = 11.2×10⁻⁶ K⁻¹

---

## 🤝 Contributing

To add new measurement types or improve physical models:
1. Edit the `SOFCDataGenerator` class
2. Add new generator methods following the existing pattern
3. Include appropriate visualizations
4. Update this README with documentation

---

## 📄 License

This synthetic dataset generator is provided for research and educational purposes.

---

## ✨ Features

- ✅ Physically realistic synthetic data
- ✅ Comprehensive measurement types
- ✅ Multiple operating conditions
- ✅ Automatic visualization generation
- ✅ Well-documented CSV formats
- ✅ Metadata tracking
- ✅ Ready for ML/AI training
- ✅ Scalable and customizable

---

## 🔍 Quality Assurance

All generated data includes:
- Realistic noise levels (Gaussian distribution)
- Physical constraints (e.g., no negative crack densities)
- Material property consistency
- Spatial correlations
- Temporal evolution patterns
- Interface effects and gradients

---

## 💡 Tips

1. **Large datasets:** For bigger grids, increase spatial resolution in the generator
2. **Custom conditions:** Modify temperature profiles and cycle counts as needed
3. **Validation:** Compare generated data with real experimental measurements
4. **Integration:** Use with FEA/simulation data for validation

---

## 📞 Support

For questions or issues with the data generator:
- Review the generated `DATA_SUMMARY.txt` file
- Check column headers in CSV files
- Examine the plotting functions for data structure
- Modify random seed for different data realizations

---

**Generated:** 2025  
**Version:** 1.0  
**Data Type:** Synthetic/Fabricated for Research Purposes
