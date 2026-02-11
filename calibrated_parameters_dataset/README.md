# Calibrated Parameters Dataset for Phase-Field Fracture Modeling

## Overview

This comprehensive dataset provides calibrated parameters for phase-field fracture modeling of delamination in Solid Oxide Fuel Cell (SOFC) electrolyte-electrode interfaces, with a focus on the role of nanoscale Mixed Ionic-Electronic Conducting (MIEC) interlayers (Gadolinium-Doped Ceria, GDC).

**Research Topic**: Phase-Field Fracture Modeling of Delamination in Electrolyte-Electrode Interfaces: The Role of Nanoscale Mixed Ionic-Electronic Conducting (MIEC) Interlayers

**Date Generated**: February 2026

---

## Dataset Structure

### CSV Files

The dataset consists of 8 comprehensive CSV files organized in the `csv_files/` directory:

#### 1. **01_main_calibrated_parameters.csv**
Core parameters for the phase-field fracture model including:
- BK Exponent (η = 2.1)
- Cathode anisotropy (β₃₃/β₁₁ = 1.7)
- Bulk fracture energies (Gc) for LSCF, YSZ, and GDC
- Interface adhesion range (Γᵢ: 0.2-3.2 J/m²)
- Penalty parameters (βₚₑₙ: 100-10,000 GPa/m)
- Phase-field length scales (l₀: 5-20 nm)
- Regularization parameters

#### 2. **02_interface_fracture_properties.csv**
Detailed interface fracture properties for:
- **YSZ/GDC Interface**: Interlayer initiation properties
  - Baseline conditions: Gc = 2.15 J/m²
  - With interdiffusion: Gc = 2.85 J/m² (enhanced by (Zr,Ce)O₂ solid solution)
  - Interlayer thickness effects (100nm, 500nm, 1µm)
  
- **GDC/LSCF Interface**: Cathode delamination properties
  - Fresh cell: Gc = 1.2 J/m²
  - Degradation timeline (0-1000h operation)
  - Sr-segregation effects: Gc reduced to 0.4 J/m² (SrO/SrZrO₃ formation)

#### 3. **03_LSCF_nonstoichiometry_data.csv**
LSCF cathode non-stoichiometry (Δδ) dataset from TGA experiments:
- Temperature range: 600-900°C
- pO₂ range: 1.0 to 10⁻²⁰ atm
- Δδ values: 0.009 to 0.047
- Anisotropic chemical strain components (εₓₓ, εᵧᵧ, εᵧᵧ)
- Chemical expansion coefficients

#### 4. **04_GDC_chemical_expansion_22delta_4T.csv**
Comprehensive GDC interlayer chemical expansion dataset:
- **22 non-stoichiometry levels** (δ: 0.0 to 0.0178)
- **4 temperature points** (600°C, 700°C, 800°C, 900°C)
- Linear strain values (µm/m)
- Chemical expansion coefficients (αchem)
- Corresponding pO₂ values at each condition
- Total: 88 data points for UMAT implementation

#### 5. **05_verification_QA_parameters.csv**
Quality assurance and verification parameters:
- Mesh objectivity criteria (h/l₀: 0.125-0.5)
- Degradation function residual stiffness (kres = 10⁻⁶)
- Newton-Raphson convergence tolerances (10⁻⁸ to 10⁻⁶)
- Energy balance checks
- Load step and time increment recommendations
- Interface element resolution guidelines

#### 6. **06_material_properties.csv**
Comprehensive material properties for LSCF, YSZ, and GDC:
- Elastic moduli (Young's modulus, Poisson's ratio)
- Thermal expansion coefficients
- Chemical expansion parameters
- Fracture toughness values
- Ionic and electronic conductivities
- Grain sizes and morphology (GDC: 50nm nanostructure)

#### 7. **07_operating_conditions.csv**
SOFC operating conditions for chemo-mechanical analysis:
- Open Circuit Voltage (OCV) conditions
- Load profiles (0.25-1.0 A/cm²)
- Temperature cycling (600-900°C)
- Redox cycling conditions
- Long-term degradation scenarios (up to 1000h)
- Thermal shock events

#### 8. **08_cohesive_zone_model_parameters.csv**
Cohesive zone model (CZM) parameters for UEL implementation:
- Mode-I, Mode-II, and Mixed-mode fracture energies
- Traction-separation law parameters (δ₀, δf, Tmax)
- Penalty stiffness values
- Model types (Bilinear, Exponential, Park-Paulino-Roesler, Xu-Needleman)
- Interface-specific and condition-dependent parameters

---

## Visualization Figures

The `figures/` directory contains 8 high-resolution PNG figures (300 DPI):

### Figure 1: Main Parameters Overview
- (a) Bulk fracture energies for LSCF, YSZ, GDC
- (b) Phase-field length scales and BK exponent
- (c) Interface adhesion energy range
- (d) Penalty parameter selection

### Figure 2: Interface Fracture Properties
- (a) YSZ/GDC interface: interdiffusion and thickness effects
- (b) GDC/LSCF degradation with Sr-segregation
- (c) Critical strength comparison
- (d) Process zone size vs fracture energy

### Figure 3: LSCF Non-stoichiometry
- (a) Δδ vs temperature at various pO₂
- (b) Δδ vs pO₂ at different temperatures
- (c) Anisotropic chemical strain at 800°C
- (d) Non-stoichiometry contour map

### Figure 4: GDC Chemical Expansion
- (a) Chemical expansion vs δ at 4 temperatures
- (b) Chemical expansion coefficient evolution
- (c) Defect chemistry: pO₂ vs δ (Brouwer diagram)
- (d) Expansion heatmap (22δ × 4T)

### Figure 5: Cohesive Zone Model
- (a) Mode-I fracture energy comparison
- (b) Traction-separation laws (Bilinear vs Exponential)
- (c) Mixed-mode fracture criterion
- (d) Interface stiffness vs fracture energy

### Figure 6: Verification and QA
- (a) Mesh objectivity study
- (b) Degradation function with residual stiffness
- (c) Convergence tolerance selection
- (d) Load step vs crack propagation

### Figure 7: Material Properties
- (a) Elastic modulus comparison
- (b) Thermal vs chemical expansion
- (c) Fracture toughness
- (d) Ionic vs electronic conductivity

### Figure 8: Operating Conditions
- (a) Polarization curve (I-V relationship)
- (b) Thermal cycling profile
- (c) Oxygen partial pressure profile
- (d) Long-term degradation evolution

---

## Key Parameters Summary

### Phase-Field Model Parameters

| Parameter | Symbol | Value | Unit | Notes |
|-----------|--------|-------|------|-------|
| BK Exponent | η | 2.1 | - | Standard for ceramic brittle fracture |
| Phase-field length (min) | l₀ | 5 | nm | Resolves GDC nanostructure |
| Phase-field length (max) | l₀ | 20 | nm | Upper bound for interface width |
| Bulk crack regularization | l | 0.5 | µm | Controls crack band width |
| Residual stiffness | kres | 10⁻⁶ | - | Prevents singularity in g(φ) |

### Interface Properties

#### YSZ/GDC Interface (Interlayer)
- **Baseline**: Gc = 1.8-2.5 J/m², σmax = 185-260 MPa
- **With Interdiffusion**: Gc = 2.5-3.2 J/m² (enhanced strength)
- **Mechanism**: (Zr,Ce)O₂ solid solution formation

#### GDC/LSCF Interface (Cathode)
- **Fresh Cell**: Gc = 0.5-1.5 J/m², σmax = 150-220 MPa
- **With Sr-Segregation**: Gc = 0.2-0.8 J/m² (drastically reduced)
- **Mechanism**: SrO/SrZrO₃ formation weakens interface
- **Characteristic Length**: 0.18-0.35 µm

### Chemo-Mechanical Coupling

#### LSCF Cathode
- Non-stoichiometry range: Δδ = 0.009 to 0.047
- Temperature range: 600-900°C
- pO₂ range: 1.0 to 10⁻¹⁰ atm
- Anisotropic expansion: β₃₃/β₁₁ = 1.7

#### GDC Interlayer
- Maximum δ = 0.0178 at 900°C, 10⁻²⁰ atm pO₂
- Chemical expansion coefficient: αchem = 0.085-0.23 (temperature dependent)
- Dataset: 22 δ levels × 4 temperatures = 88 calibration points

---

## Implementation Guidelines

### For ABAQUS UMAT (Bulk Material)

1. **Phase-field evolution**:
   ```fortran
   ! Degradation function
   g_phi = (1.0 - phi)**2 + k_res
   
   ! Phase-field driving force
   H_phi = max(psi_elastic_positive, H_history)
   ```

2. **Chemical strain** (LSCF):
   ```fortran
   ! Use 03_LSCF_nonstoichiometry_data.csv
   epsilon_chem(1) = beta_11 * delta_delta
   epsilon_chem(2) = beta_11 * delta_delta
   epsilon_chem(3) = beta_33 * delta_delta  ! Anisotropic
   ```

3. **Chemical strain** (GDC):
   ```fortran
   ! Use 04_GDC_chemical_expansion_22delta_4T.csv
   ! Interpolate based on current T and delta
   epsilon_chem = alpha_chem(T, delta) * delta
   ```

### For ABAQUS UEL (Cohesive Interface)

1. **Cohesive zone parameters** (from CSV file 08):
   ```fortran
   ! YSZ/GDC baseline
   G_c_int = 2.15  ! J/m²
   T_max = 222.5   ! MPa
   delta_0 = 0.019 ! µm (onset)
   delta_f = 0.097 ! µm (failure)
   K_penalty = 1000.0  ! GPa/m
   ```

2. **Bilinear traction-separation law**:
   ```fortran
   if (delta < delta_0) then
       T = K_penalty * delta
   else if (delta < delta_f) then
       T = T_max * (delta_f - delta) / (delta_f - delta_0)
   else
       T = 0.0  ! Complete failure
   end if
   ```

3. **Mixed-mode criterion** (Park-Paulino-Roesler):
   ```fortran
   beta_mode = atan(G_II / G_I)
   G_c_mixed = G_Ic * G_IIc / &
               (G_Ic * sin(beta_mode)**2 + G_IIc * cos(beta_mode)**2)
   ```

### Verification Workflow

1. **Mesh Convergence**:
   - Ensure h/l₀ ∈ [0.125, 0.5]
   - For l₀ = 10 nm: h = 1.25-5.0 nm
   - Run at least 3 mesh densities

2. **Energy Balance**:
   - Monitor: ΔΨ/Ψ₀ < 0.01
   - Check degradation function: g(φ=1) ≈ kres

3. **Convergence**:
   - Newton-Raphson: εtol = 10⁻⁷ (recommended)
   - Maximum iterations: 50
   - Load step size: Δλ = 0.001

---

## Data Sources and Calibration Methods

### Experimental Data
- **TGA (Thermogravimetric Analysis)**: Non-stoichiometry measurements for LSCF and GDC
- **XRD (X-ray Diffraction)**: Chemical expansion coefficients
- **Micro-cantilever Testing**: Bulk fracture energies
- **Nanoindentation**: Interface fracture properties
- **SEM/TEM**: Microstructural characterization (grain sizes, interface morphology)
- **EIS (Electrochemical Impedance Spectroscopy)**: Conductivity measurements

### Computational Methods
- **DFT (Density Functional Theory)**: Interface adhesion energies
- **Diffusion Models**: Interdiffusion and Sr-segregation effects
- **Defect Chemistry Models**: pO₂-δ-T relationships

### Calibration Approach
1. **Direct measurements**: Where available (TGA, XRD, mechanical testing)
2. **Literature synthesis**: Standard values for well-characterized materials
3. **DFT calculations**: Interface-specific properties
4. **Inverse modeling**: Fitting to experimental delamination data

---

## Citation

If you use this dataset in your research, please cite:

```
Phase-Field Fracture Modeling Dataset for SOFC Interfaces (2026)
Calibrated Parameters for Electrolyte-Electrode Delamination with GDC Interlayers
```

---

## File Formats

- **CSV Files**: UTF-8 encoded, comma-delimited
- **Figures**: PNG format, 300 DPI, RGB color space
- **Python Script**: Compatible with Python 3.8+, requires: pandas, numpy, matplotlib, seaborn

---

## Usage Examples

### Load Data in Python

```python
import pandas as pd

# Load main parameters
params = pd.read_csv('csv_files/01_main_calibrated_parameters.csv')

# Load LSCF non-stoichiometry data
lscf_data = pd.read_csv('csv_files/03_LSCF_nonstoichiometry_data.csv')

# Filter for specific temperature
data_800C = lscf_data[lscf_data['Temperature_C'] == 800]

# Interpolate chemical strain
import numpy as np
delta_query = 0.025
chem_strain = np.interp(delta_query, 
                         data_800C['Delta_delta'], 
                         data_800C['Chemical_Strain_zz'])
```

### Load Data in MATLAB

```matlab
% Load main parameters
params = readtable('csv_files/01_main_calibrated_parameters.csv');

% Load interface properties
interface = readtable('csv_files/02_interface_fracture_properties.csv');

% Extract YSZ/GDC baseline properties
ysz_gdc = interface(strcmp(interface.Interface, 'YSZ/GDC') & ...
                    strcmp(interface.Condition, 'Baseline'), :);
Gc_int = ysz_gdc.Gc_int;
```

### Load Data in Fortran (UMAT/UEL)

```fortran
! Pre-process CSV to include data as parameters
! Example: 03_LSCF_nonstoichiometry_data.csv

parameter (N_DELTA_POINTS = 35)
real*8 :: LSCF_DELTA(N_DELTA_POINTS) = (/ &
    0.009, 0.012, 0.018, ..., 0.047 /)
    
real*8 :: LSCF_STRAIN_ZZ(N_DELTA_POINTS) = (/ &
    0.001301, 0.001734, 0.002601, ..., 0.006791 /)

! Interpolation routine
call INTERP1D(LSCF_DELTA, LSCF_STRAIN_ZZ, &
              N_DELTA_POINTS, delta_current, strain_zz)
```

---

## Additional Notes

### Assumptions and Limitations

1. **Temperature Effects**: 
   - Most parameters calibrated at 800°C
   - Temperature-dependent data provided where available (LSCF, GDC)

2. **Microstructure**: 
   - GDC nanostructure: 50 nm grain size assumed
   - LSCF: Bulk polycrystalline behavior
   - YSZ: Dense electrolyte (>95% theoretical density)

3. **Chemical Environment**:
   - Cathode side: pO₂ = 0.21 atm (air)
   - Reducing side effects captured through pO₂ variation
   - Sr-segregation effects based on 1000h operation data

4. **Interface Degradation**:
   - Time-dependent data available for GDC/LSCF (0-1000h)
   - Sr-segregation is primary degradation mechanism
   - SrZrO₃ formation assumed at extended operation

### Recommended Parameter Selection

- **For fresh cell modeling**: Use baseline interface properties
- **For degradation studies**: Use time-dependent data from CSV file 02
- **For thermal cycling**: Include temperature-dependent expansion from CSV files 03, 04
- **For high-current operation**: Use current density effects from CSV file 07

---

## Contact and Support

For questions, corrections, or additional data requests, please refer to the main research publication.

**Dataset Version**: 1.0  
**Last Updated**: February 2026  
**License**: Academic use encouraged with proper citation

---

## File Manifest

```
calibrated_parameters_dataset/
├── README.md                          (This file)
├── generate_figures.py                (Python script for visualization)
├── calibrated_parameters_dataset.zip  (Compressed CSV files)
├── csv_files/
│   ├── 01_main_calibrated_parameters.csv
│   ├── 02_interface_fracture_properties.csv
│   ├── 03_LSCF_nonstoichiometry_data.csv
│   ├── 04_GDC_chemical_expansion_22delta_4T.csv
│   ├── 05_verification_QA_parameters.csv
│   ├── 06_material_properties.csv
│   ├── 07_operating_conditions.csv
│   └── 08_cohesive_zone_model_parameters.csv
└── figures/
    ├── Figure_01_Main_Parameters.png
    ├── Figure_02_Interface_Properties.png
    ├── Figure_03_LSCF_Nonstoichiometry.png
    ├── Figure_04_GDC_Chemical_Expansion.png
    ├── Figure_05_Cohesive_Zone_Model.png
    ├── Figure_06_Verification_QA.png
    ├── Figure_07_Material_Properties.png
    └── Figure_08_Operating_Conditions.png
```

---

**END OF README**
