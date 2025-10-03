# SOFC Experimental Measurement Dataset

This dataset contains fabricated but realistic experimental measurements for Solid Oxide Fuel Cell (SOFC) thermomechanical analysis research.

## Dataset Structure

### 1. Digital Image Correlation (DIC) Data (`dic_data/`)
- **Strain Maps**: Real-time strain field measurements during thermal testing
- **Speckle Patterns**: Image sequences with timestamps for correlation analysis  
- **Lagrangian Strain Tensors**: Vic-3D format strain tensor outputs

**Test Conditions:**
- Sintering: 1200-1500°C, 24 hours
- Thermal Cycling: 800-1200°C, ΔT=400°C, 10 cycles
- Startup/Shutdown: 25-800°C, 5 cycles

**Key Findings:**
- Maximum strain: 1.23%
- Strain hotspots >1.0% identified at material interfaces
- 1000 measurement points per condition

### 2. Synchrotron X-ray Diffraction (XRD) Data (`xrd_data/`)
- **Residual Stress Profiles**: Stress distribution across SOFC cross-section
- **Lattice Strain**: Temperature-dependent lattice parameter changes
- **sin²ψ Analysis**: Multi-angle stress measurements
- **Microcrack Thresholds**: Critical strain values for crack initiation

**Materials Analyzed:**
- YSZ (Yttria-Stabilized Zirconia)
- Ni (Nickel)
- LSM (Lanthanum Strontium Manganite)

**Stress Range:** -200 to +200 MPa
**Critical Strain Threshold:** 0.018-0.025

### 3. Post-Mortem Analysis Data (`postmortem_data/`)

#### SEM Analysis (`sem_images/`)
- Crack density quantification (cracks/mm²)
- Crack morphology measurements
- High-resolution microstructural imaging

**Crack Density Range:** 0.8-4.5 cracks/mm²
**Highest Density Location:** Anode-electrolyte interface

#### EDS Analysis (`eds_scans/`)
- Elemental composition line scans
- Point analysis at critical locations
- Interface chemistry characterization

**Elements Analyzed:** Ni, Zr, Y, La, Sr, Mn, O

#### Nano-indentation (`nanoindentation/`)
- Young's modulus mapping
- Hardness measurements  
- Creep compliance testing

**Material Properties:**
- YSZ: E = 184.7 GPa, H = 12.5 GPa
- Ni: E = 109.8 GPa, H = 2.1 GPa
- LSM: E = 95.2 GPa, H = 8.9 GPa

## File Formats

- **CSV**: Tabular measurement data
- **JSON**: Metadata and structured information
- **PNG**: Visualization plots and maps
- **NPY**: Raw numerical arrays (speckle patterns)

## Usage Notes

This is fabricated data generated for research and educational purposes. The values and trends are based on literature data and realistic material behavior, but are not from actual experiments.

## Data Quality

- Measurement uncertainties included
- Realistic noise and scatter applied
- Temperature-dependent material behavior modeled
- Interface effects and stress concentrations included

Generated on: 2025-10-03 05:47:56
Total Files: 35
Dataset Size: ~50 MB
