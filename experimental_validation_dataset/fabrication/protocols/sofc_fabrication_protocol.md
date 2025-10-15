# SOFC Plate Fabrication Protocol

## Overview
This protocol describes the fabrication of SOFC plates with controlled parameter variations for experimental validation of stress-warp relationships.

## Materials

### Base Materials
- **8YSZ Powder**: Tosoh TZ-8Y, 99.9% purity, d50 = 0.5 μm
- **NiO Powder**: Sigma-Aldrich, 99.99% purity, d50 = 0.8 μm
- **LSM Powder**: Praxair, La0.8Sr0.2MnO3, d50 = 1.2 μm
- **Organic Binders**: PVB (polyvinyl butyral), PEG (polyethylene glycol)
- **Solvents**: Ethanol, toluene, isopropanol

### Additives
- **Plasticizers**: Dibutyl phthalate (DBP), benzyl butyl phthalate (BBP)
- **Dispersants**: Phosphate ester, polyethylene glycol
- **Sintering Aids**: None (pure ceramic system)

## Equipment Required

### Tape Casting
- **Doctor Blade System**: Custom-built with adjustable gap (10-500 μm)
- **Substrate**: Mylar film, 125 μm thickness
- **Drying Oven**: Forced convection, temperature control ±2°C

### Sintering
- **High-Temperature Furnace**: Nabertherm HT 08/17, max temp 1800°C
- **Atmosphere Control**: Air, N2, forming gas (5% H2/95% N2)
- **Temperature Monitoring**: Type-S thermocouples, ±1°C accuracy

## Fabrication Parameters

### Design of Experiments (DoE)
The fabrication follows a 2-level factorial design with center points:

| Parameter | Low Level | Center | High Level | Units |
|-----------|-----------|---------|------------|-------|
| Electrolyte Thickness | 100 | 150 | 200 | μm |
| Anode Thickness | 200 | 300 | 400 | μm |
| Cathode Thickness | 30 | 50 | 70 | μm |
| Sintering Temperature | 1300 | 1350 | 1400 | °C |
| Sintering Time | 2 | 4 | 6 | hours |
| Cooling Rate | 1 | 2 | 3 | °C/min |

### Sample Matrix
Total samples: 45 (36 factorial + 9 center points)

## Detailed Fabrication Steps

### Step 1: Slurry Preparation

#### Electrolyte Slurry (8YSZ)
1. **Powder Preparation**
   - Calcine 8YSZ powder at 600°C for 2 hours
   - Cool to room temperature in desiccator
   - Sieve through 325 mesh

2. **Slurry Mixing**
   - Combine 8YSZ powder (60 wt%)
   - Add binder solution: PVB (8 wt%), PEG (4 wt%) in ethanol
   - Add plasticizer: DBP (3 wt%)
   - Add dispersant: 0.5 wt%
   - Ball mill for 24 hours with YSZ media

3. **Rheology Adjustment**
   - Target viscosity: 2000-3000 cP at 10 s⁻¹
   - Adjust with ethanol if needed
   - Degas under vacuum for 30 minutes

#### Anode Slurry (NiO-YSZ)
1. **Powder Mixing**
   - NiO powder: 65 vol%
   - 8YSZ powder: 35 vol%
   - Dry mix for 2 hours

2. **Slurry Preparation**
   - Follow same procedure as electrolyte
   - Adjust powder loading to 58 wt%
   - Target viscosity: 1500-2500 cP

#### Cathode Slurry (LSM-YSZ)
1. **Powder Mixing**
   - LSM powder: 70 vol%
   - 8YSZ powder: 30 vol%
   - Dry mix for 2 hours

2. **Slurry Preparation**
   - Follow same procedure as electrolyte
   - Adjust powder loading to 55 wt%
   - Target viscosity: 1200-2000 cP

### Step 2: Tape Casting

#### Equipment Setup
1. Clean doctor blade and substrate thoroughly
2. Set blade gap according to target thickness:
   - Electrolyte: gap = 1.5 × target thickness
   - Anode: gap = 1.8 × target thickness
   - Cathode: gap = 2.0 × target thickness

#### Casting Procedure
1. **Electrolyte Layer**
   - Cast on Mylar substrate
   - Casting speed: 2 cm/s
   - Dry at 60°C for 2 hours

2. **Anode Layer**
   - Cast directly on dried electrolyte
   - Ensure good adhesion
   - Dry at 60°C for 2 hours

3. **Cathode Layer**
   - Cast on opposite side of electrolyte
   - Dry at 60°C for 2 hours

### Step 3: Green Body Processing

#### Cutting and Handling
1. Cut samples to 50 mm × 50 mm squares
2. Remove from substrate carefully
3. Store in controlled humidity (45-55% RH)

#### Binder Burnout
1. Heat at 1°C/min to 450°C
2. Hold for 2 hours in air
3. Cool at 2°C/min to room temperature

### Step 4: Sintering

#### Furnace Loading
1. Place samples on alumina setters
2. Use spacers to prevent sticking
3. Load maximum 9 samples per run

#### Sintering Profile
For each sample according to DoE matrix:

1. **Heating Phase**
   - Ramp at 3°C/min to sintering temperature
   - Atmosphere: Air

2. **Sintering Phase**
   - Hold at target temperature for specified time
   - Atmosphere: Air

3. **Cooling Phase**
   - Cool at specified rate to 1000°C
   - Natural cooling below 1000°C
   - Atmosphere: Air throughout

### Step 5: Quality Control

#### Dimensional Measurements
1. Measure thickness at 9 points using micrometer
2. Record length and width dimensions
3. Calculate shrinkage from green dimensions

#### Visual Inspection
1. Check for cracks, delamination, warpage
2. Document any defects with photography
3. Classify samples as acceptable/rejected

#### Sample Identification
1. Laser etch unique ID on each sample
2. Record fabrication parameters in database
3. Store samples in controlled environment

## Process Control and Documentation

### Critical Control Points
1. **Slurry Viscosity**: Check every batch
2. **Casting Thickness**: Measure wet thickness
3. **Drying Conditions**: Monitor temperature and humidity
4. **Sintering Temperature**: Calibrate thermocouples monthly
5. **Cooling Rate**: Monitor and record actual rates

### Documentation Requirements
1. **Batch Records**: Complete for each fabrication run
2. **Material Certificates**: Maintain for all raw materials
3. **Equipment Calibration**: Monthly for critical equipment
4. **Sample Tracking**: Unique ID from fabrication to testing

### Quality Metrics
- **Yield Target**: >90% acceptable samples
- **Thickness Uniformity**: ±5% across sample
- **Dimensional Tolerance**: ±2% from target
- **Surface Quality**: No visible cracks or delamination

## Safety Considerations

### Chemical Hazards
- Use fume hood for solvent-based operations
- Proper PPE: gloves, safety glasses, lab coat
- Maintain SDS for all chemicals

### High-Temperature Operations
- Furnace safety training required
- Heat-resistant gloves for hot samples
- Proper ventilation for sintering operations

### Sample Handling
- Fragile ceramic samples - handle with care
- Use appropriate storage containers
- Avoid contamination during processing

## Troubleshooting Guide

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Cracking during drying | Too fast drying | Reduce temperature, increase humidity |
| Delamination | Poor adhesion | Clean substrate, adjust binder content |
| Uneven thickness | Blade wear | Replace blade, check alignment |
| Warpage after sintering | Thermal gradients | Improve furnace uniformity |
| Low density | Insufficient sintering | Increase temperature or time |

## Expected Results

### Target Properties
- **Density**: >95% theoretical
- **Porosity**: <5% open porosity
- **Thickness Uniformity**: ±3% across sample
- **Surface Roughness**: Ra < 2 μm

### Typical Defect Rates
- **Cracking**: <5% of samples
- **Delamination**: <2% of samples
- **Dimensional Out-of-Spec**: <8% of samples

This protocol ensures reproducible fabrication of SOFC samples with controlled parameter variations suitable for experimental validation studies.