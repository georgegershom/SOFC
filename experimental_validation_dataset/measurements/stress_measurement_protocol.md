# SOFC Residual Stress Measurement Protocol

## Overview
This protocol describes multiple complementary techniques for measuring residual stresses in SOFC plates, addressing the challenge that full 3D stress tensors cannot be measured non-destructively.

## Measurement Techniques Overview

### 1. Curvature-Based Inverse Method (Stoney's Formula)
**Principle**: Measure curvature of bilayer samples to calculate average stress
**Advantages**: Non-destructive, well-established theory
**Limitations**: Through-thickness average only, requires bilayer samples

### 2. Layer Removal + Warp Measurement
**Principle**: Progressive layer removal with warp measurement after each step
**Advantages**: Through-thickness stress profile
**Limitations**: Destructive, requires careful material removal

### 3. X-Ray Diffraction (XRD) Stress Analysis
**Principle**: Measure lattice strain to calculate local stress
**Advantages**: Direct stress measurement, high spatial resolution
**Limitations**: Surface/near-surface only, requires crystalline materials

### 4. Raman Spectroscopy Stress Analysis
**Principle**: Stress-induced shifts in Raman peaks
**Advantages**: High spatial resolution, works on amorphous materials
**Limitations**: Requires calibration, limited penetration depth

## Sample Preparation

### Bilayer Samples for Curvature Method
1. **Fabrication Protocol**
   - Electrolyte-only samples (150 μm thick)
   - Electrolyte + anode bilayers
   - Electrolyte + cathode bilayers
   - Dimensions: 50 × 50 mm

2. **Reference Samples**
   - Measure curvature before layer deposition
   - Document initial stress state
   - Use same sintering conditions as multilayer samples

### Full Multilayer Samples
1. **Standard SOFC Configuration**
   - Anode/Electrolyte/Cathode trilayers
   - Same fabrication as main dataset
   - Additional samples for destructive testing

2. **Sample Sectioning**
   - Cut samples for cross-sectional analysis
   - Prepare surfaces for XRD and Raman
   - Maintain sample identification throughout

## Technique 1: Curvature-Based Stress Analysis

### Equipment Required
- **Profilometer**: Dektak XT (contact) or Zygo NewView (non-contact)
- **Sample Holder**: Kinematic mount with temperature control
- **Environmental Chamber**: For temperature-dependent measurements

### Theoretical Background
Stoney's formula for thin films on substrates:
```
σ_film = (E_s × t_s²) / (6 × t_f × R × (1 - ν_s))
```
Where:
- σ_film = average stress in film
- E_s, ν_s = elastic modulus and Poisson's ratio of substrate
- t_s, t_f = thickness of substrate and film
- R = radius of curvature

### Measurement Procedure

#### Pre-Deposition Curvature
1. **Sample Mounting**
   - Mount electrolyte substrate in kinematic holder
   - Ensure stress-free mounting (no clamping forces)
   - Allow thermal equilibration (30 minutes)

2. **Curvature Measurement**
   - Measure curvature along both principal axes
   - Take 5 measurements per axis
   - Calculate average and standard deviation

#### Post-Deposition Curvature
1. **Layer Deposition and Sintering**
   - Apply electrode layer using same process as main samples
   - Sinter according to fabrication protocol
   - Cool to measurement temperature

2. **Curvature Measurement**
   - Repeat measurement procedure
   - Use identical measurement parameters
   - Calculate change in curvature (Δκ)

#### Stress Calculation
1. **Data Processing**
   - Calculate radius of curvature: R = 1/κ
   - Apply Stoney's formula with temperature-dependent properties
   - Account for multilayer effects using modified equations

2. **Uncertainty Analysis**
   - Propagate measurement uncertainties
   - Consider systematic errors (temperature, thickness variations)
   - Report 95% confidence intervals

### Expected Results
- **Anode Stress**: -50 to -150 MPa (compressive)
- **Cathode Stress**: +20 to +80 MPa (tensile)
- **Measurement Uncertainty**: ±15 MPa

## Technique 2: Layer Removal Analysis

### Equipment Required
- **Ion Beam Milling**: FEI Helios NanoLab 650
- **Mechanical Polishing**: Struers LaboPol-5
- **Thickness Measurement**: Mitutoyo micrometer (±1 μm)
- **Warp Measurement**: Same equipment as warp protocol

### Procedure Overview
1. Initial warp measurement of full trilayer
2. Progressive removal of cathode layer
3. Warp measurement after each removal step
4. Inverse analysis to calculate original stress profile

### Detailed Procedure

#### Initial State Documentation
1. **Complete Warp Measurement**
   - Use LSCM for high-resolution mapping
   - Document initial warp state
   - Establish reference coordinate system

2. **Sample Characterization**
   - Measure layer thicknesses at multiple points
   - Document material properties
   - Photograph sample for reference

#### Layer Removal Sequence

##### Step 1: Cathode Removal
1. **Mechanical Polishing**
   - Use 15 μm diamond paste initially
   - Progress to 1 μm for final surface
   - Remove 10 μm increments with measurements

2. **Thickness Monitoring**
   - Measure remaining cathode thickness
   - Use optical microscopy of cross-sections
   - Stop when electrolyte surface is exposed

3. **Warp Measurement**
   - Clean surface thoroughly
   - Measure warp using same protocol as initial
   - Document changes in warp pattern

##### Step 2: Partial Electrolyte Removal
1. **Controlled Removal**
   - Remove 25%, 50%, 75% of electrolyte thickness
   - Use ion beam milling for precise control
   - Monitor removal rate continuously

2. **Surface Quality Control**
   - Minimize surface damage during removal
   - Check for artifacts using SEM
   - Document any processing-induced effects

##### Step 3: Complete Electrolyte Removal
1. **Final Removal**
   - Remove remaining electrolyte
   - Expose anode surface
   - Measure final warp state

#### Inverse Analysis
1. **Data Compilation**
   - Compile warp measurements at each removal step
   - Calculate incremental warp changes
   - Account for material removal effects

2. **Stress Reconstruction**
   - Use finite element inverse analysis
   - Fit stress distribution to match warp evolution
   - Validate against curvature measurements

### Expected Results
- **Through-thickness stress profiles**
- **Interface stress concentrations**
- **Validation of FEA stress predictions**

## Technique 3: X-Ray Diffraction Stress Analysis

### Equipment Required
- **XRD System**: Rigaku SmartLab with stress analysis package
- **X-ray Source**: Cu Kα (λ = 1.5406 Å)
- **Detector**: 2D detector for texture analysis
- **Sample Stage**: Eulerian cradle with precise positioning

### Theoretical Background
The sin²ψ method for stress analysis:
```
d_φψ = d₀[1 + (1+ν)/E × σ_φ × sin²ψ - ν/E × (σ₁₁ + σ₂₂)]
```
Where:
- d_φψ = lattice spacing at angle ψ
- σ_φ = stress in φ direction
- ψ = angle between surface normal and diffraction vector

### Measurement Procedure

#### Sample Preparation
1. **Surface Preparation**
   - Polish to 1 μm finish
   - Remove surface damage layer (5-10 μm)
   - Clean with ethanol and dry

2. **Measurement Points**
   - Map 5×5 grid across sample surface
   - Focus on high-stress regions identified by FEA
   - Include edge and center locations

#### XRD Measurements
1. **Peak Selection**
   - 8YSZ: (400) reflection at 2θ ≈ 73.2°
   - NiO: (200) reflection at 2θ ≈ 43.3°
   - LSM: (110) reflection at 2θ ≈ 32.8°

2. **Data Collection**
   - ψ angles: 0°, 15°, 30°, 45° (minimum)
   - φ angles: 0°, 90° (for biaxial stress)
   - Step size: 0.02° in 2θ
   - Count time: 10 seconds per step

#### Stress Calculation
1. **Peak Analysis**
   - Fit peaks with pseudo-Voigt function
   - Calculate d-spacing from peak position
   - Assess peak quality and texture effects

2. **Stress Analysis**
   - Plot d vs. sin²ψ
   - Calculate stress from slope
   - Apply corrections for texture and grain size

### Expected Results
- **Surface Stress Maps**: 2D stress distribution
- **Principal Stresses**: σ₁, σ₂ at each measurement point
- **Stress Gradients**: Near interfaces and edges
- **Measurement Uncertainty**: ±20 MPa

## Technique 4: Raman Spectroscopy Stress Analysis

### Equipment Required
- **Raman Spectrometer**: Horiba LabRAM HR Evolution
- **Laser**: 532 nm, 50 mW maximum power
- **Microscope**: 100× objective (spatial resolution ~1 μm)
- **Stage**: Motorized XYZ stage with 0.1 μm precision

### Calibration Requirements
1. **Stress-Free Reference**
   - Measure unstressed powder samples
   - Record reference peak positions
   - Establish temperature dependence

2. **Stress Calibration**
   - Use samples with known applied stress
   - Establish stress-shift relationship
   - Validate with XRD measurements

### Measurement Procedure

#### Sample Preparation
1. **Surface Requirements**
   - Optically smooth surface (Ra < 0.1 μm)
   - No surface coatings or contamination
   - Minimize laser heating effects

#### Spectral Acquisition
1. **Measurement Parameters**
   - Laser power: 10% to avoid heating
   - Acquisition time: 30 seconds per spectrum
   - Spectral range: 100-1000 cm⁻¹
   - Spatial resolution: 1 μm

2. **Mapping Strategy**
   - High-resolution maps in critical regions
   - Lower resolution overview maps
   - Focus on interfaces and stress concentrations

#### Data Analysis
1. **Peak Identification**
   - Identify stress-sensitive peaks
   - Account for temperature effects
   - Correct for instrumental drift

2. **Stress Calculation**
   - Apply calibrated stress-shift relationships
   - Calculate local stress values
   - Generate stress maps

### Expected Results
- **High-resolution stress maps** (1 μm spatial resolution)
- **Interface stress analysis**
- **Stress concentrations at defects**

## Quality Assurance and Validation

### Cross-Technique Validation
1. **Overlapping Measurements**
   - Compare XRD and Raman results in same regions
   - Validate curvature method with layer removal
   - Check consistency across techniques

2. **Reference Samples**
   - Use samples with known stress states
   - Validate against analytical solutions
   - Regular calibration checks

### Uncertainty Analysis
1. **Measurement Repeatability**
   - Multiple measurements on same sample
   - Different operators and conditions
   - Statistical analysis of variations

2. **Systematic Errors**
   - Temperature effects
   - Surface preparation artifacts
   - Equipment calibration drift

## Data Integration and Analysis

### Stress Tensor Reconstruction
1. **Data Fusion**
   - Combine results from all techniques
   - Weight data by measurement uncertainty
   - Use FEA as prior information

2. **3D Stress Field**
   - Interpolate between measurement points
   - Satisfy equilibrium constraints
   - Validate against physical expectations

### Comparison with FEA
1. **Direct Comparison**
   - Extract FEA results at measurement locations
   - Calculate correlation coefficients
   - Identify systematic differences

2. **Model Validation**
   - Use experimental data to validate FEA models
   - Identify missing physics or boundary conditions
   - Improve material property databases

This comprehensive stress measurement protocol provides multiple independent validation paths for FEA-based stress predictions, enabling robust validation of ML-augmented inverse modeling approaches.