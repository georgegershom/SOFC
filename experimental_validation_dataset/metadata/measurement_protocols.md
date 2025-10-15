# SOFC Experimental Validation - Measurement Protocols

## Overview
This document describes the detailed measurement protocols used for the experimental validation dataset. All measurements follow industry standards and best practices for SOFC characterization.

## Warp Measurement Protocol

### Laser Scanning Confocal Microscopy (Primary Method)

**Instrument**: Keyence VK-X1000
**Resolution**: 1.0 μm lateral, 10 nm vertical
**Measurement Area**: 50 mm × 50 mm
**Grid Density**: 50 × 50 points (2500 total points)

#### Procedure:
1. **Sample Preparation**
   - Clean sample surface with isopropanol
   - Mount on vibration-isolated stage
   - Allow thermal equilibration (30 minutes)

2. **Calibration**
   - Verify instrument calibration using NIST traceable standards
   - Check laser power and detector sensitivity
   - Validate measurement range

3. **Measurement**
   - Set scan parameters: 50×50 grid, 1 μm step size
   - Acquire 3D surface profile
   - Verify data quality (check for outliers, missing points)
   - Repeat measurement 3 times for uncertainty analysis

4. **Data Processing**
   - Remove tilt and curvature from raw data
   - Calculate surface statistics (RMS, peak-to-valley, etc.)
   - Export point cloud data in standard format

#### Uncertainty Analysis:
- **Lateral uncertainty**: ±0.5 μm (instrument specification)
- **Vertical uncertainty**: ±0.02 μm (repeatability)
- **Measurement repeatability**: ±0.01 μm (3-sigma)

### White Light Interferometry (Secondary Method)

**Instrument**: Zygo NewView 8300
**Resolution**: 0.5 μm lateral, 0.1 nm vertical
**Measurement Area**: 1 mm × 1 mm (multiple fields)

#### Procedure:
1. **Sample Preparation**
   - Clean sample surface
   - Mount on anti-vibration table
   - Thermal equilibration (15 minutes)

2. **Measurement**
   - Select measurement fields (avoiding defects)
   - Acquire interferograms
   - Process using Zygo software
   - Stitch multiple fields if needed

3. **Data Processing**
   - Remove tilt and piston terms
   - Calculate surface parameters
   - Export height maps

## Residual Stress Measurement Protocols

### 1. Curvature-Based Method (Stoney's Formula)

**Principle**: Measure curvature of bilayer samples to calculate average stress

#### Procedure:
1. **Sample Preparation**
   - Prepare bilayer samples (anode/electrolyte, electrolyte/cathode)
   - Ensure uniform thickness
   - Clean surfaces

2. **Curvature Measurement**
   - Use laser profilometer or contact profilometer
   - Measure radius of curvature
   - Calculate stress using Stoney's formula:
     σ = E·h²/(6·R·(1-ν))
   - Where E = Young's modulus, h = thickness, R = radius, ν = Poisson's ratio

3. **Uncertainty Analysis**
   - Thickness measurement uncertainty: ±2%
   - Curvature measurement uncertainty: ±5%
   - Combined uncertainty: ±15%

### 2. X-Ray Diffraction (XRD)

**Instrument**: Bruker D8 Discover
**X-ray source**: Cu Kα (λ = 1.5418 Å)
**Spot size**: 1 mm diameter
**Measurement grid**: 5×5 points (25 total)

#### Procedure:
1. **Sample Preparation**
   - Clean sample surface
   - Mount on XRD stage
   - Align sample normal to beam

2. **Measurement**
   - Select measurement points (5×5 grid)
   - Acquire diffraction patterns
   - Measure peak positions
   - Calculate strain from peak shifts

3. **Stress Calculation**
   - Use sin²ψ method
   - Calculate stress tensor components
   - Apply corrections for sample geometry

4. **Uncertainty Analysis**
   - Peak position uncertainty: ±0.01°
   - Stress uncertainty: ±8%

### 3. Layer Removal Method (Destructive)

**Principle**: Measure warp change after sequential layer removal

#### Procedure:
1. **Initial Measurement**
   - Measure initial warp profile
   - Record baseline curvature

2. **Layer Removal**
   - Remove cathode layer (mechanical polishing)
   - Measure warp change
   - Remove electrolyte layer
   - Measure warp change
   - Remove anode layer
   - Measure final warp

3. **Stress Calculation**
   - Use inverse analysis (FEA-based)
   - Back-calculate stress profile
   - Account for removal effects

4. **Uncertainty Analysis**
   - Warp measurement uncertainty: ±0.01 mm
   - Layer removal uncertainty: ±5%
   - Combined uncertainty: ±20%

### 4. Raman Spectroscopy

**Instrument**: Renishaw inVia Raman Microscope
**Laser**: 532 nm, 10 mW
**Spot size**: 1 μm
**Measurement grid**: 4×4 points (16 total)

#### Procedure:
1. **Sample Preparation**
   - Clean sample surface
   - Mount on microscope stage
   - Focus laser spot

2. **Measurement**
   - Select measurement points
   - Acquire Raman spectra
   - Measure peak positions
   - Calculate peak shifts

3. **Stress Calculation**
   - Correlate peak shifts with stress
   - Use calibration curves
   - Calculate stress values

4. **Uncertainty Analysis**
   - Peak position uncertainty: ±0.1 cm⁻¹
   - Stress uncertainty: ±12%

## Quality Assurance

### Calibration Standards
- **Lateral calibration**: NIST traceable grid standards
- **Vertical calibration**: NIST traceable step height standards
- **Stress calibration**: Certified stress standards

### Measurement Conditions
- **Temperature**: 22.0 ± 0.5°C
- **Humidity**: 40 ± 5% RH
- **Vibration**: Isolated from building vibrations
- **Electromagnetic**: Shielded from EMI

### Data Validation
- **Repeatability**: 3 measurements per sample
- **Reproducibility**: Multiple operators
- **Cross-validation**: Compare different techniques
- **Outlier detection**: Statistical analysis

### Documentation
- **Raw data**: Preserved in original format
- **Processed data**: Standardized format
- **Metadata**: Complete measurement conditions
- **Uncertainty**: Full uncertainty analysis

## References

1. Stoney, G.G. (1909). "The tension of metallic films deposited by electrolysis"
2. NIST Special Publication 960-12: "Guide for the Use of the International System of Units"
3. ASTM E837-13: "Standard Test Method for Determining Residual Stresses by the Hole-Drilling Strain-Gage Method"
4. ISO 4287: "Geometrical Product Specifications (GPS) - Surface texture: Profile method"
