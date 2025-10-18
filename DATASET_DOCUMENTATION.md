# Fire-Resistant Concrete Microstructural Analysis Dataset

## Overview

This comprehensive dataset was generated for the research titled: **"Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete."**

The dataset provides multi-scale, quantitative evidence explaining thermo-mechanical degradation mechanisms observed in Phase 2, revealing phase transformations, microstructural evolution, and damage propagation at different length scales.

## Dataset Structure

### Core Identifiers
- **Sample_ID**: Links to Phase 2 specimens (e.g., C-28-R-800-Furnace)
- **Analysis_Type**: SEM, XRD, TGA, MicroCT
- **Measurement_Scale**: Micro (µm), Nano (nm), Bulk (mm)
- **Quantitative_Metric**: Specific measured parameter

### Sample Matrix
- **Mix IDs**: C-28-R-0, C-28-R-5, C-28-R-10, C-28-R-15, C-28-R-20
- **Rubber Content**: 0%, 5%, 10%, 15%, 20% by volume
- **Temperatures**: 25°C, 200°C, 400°C, 600°C, 800°C
- **Replicates**: 3 per condition
- **Total Records**: 4,548

## Analytical Techniques

### 1. Scanning Electron Microscopy (SEM)
**Scale**: Micro (µm)
**Fields of View**: 5 per sample
**Key Metrics**:
- Pore Size Distribution (mean, std, min, max, count)
- Crack Density (cracks/mm²)
- Interface Quality Index (dimensionless)
- Rubber Melt Fraction (fraction)
- Gas Evolution Pore Density (pores/mm²)

**Temperature-Dependent Features**:
- Pore size increases with temperature
- Crack density increases above 200°C
- Interface quality degrades with temperature
- Rubber-specific melt formation above 150°C

### 2. X-Ray Diffraction (XRD)
**Scale**: Bulk (mm)
**Key Metrics**:
- Cement phases: C3S, C2S, C3A, C4AF, CH, CSH, CASH, AFt, AFm
- Rubber phases: Natural_Rubber, SBR, Carbon_Black, Vulcanization_Products
- Peak characteristics: Intensity, Position (2θ), FWHM

**Phase Transformations**:
- 200°C: CSH formation begins
- 400°C: CH decomposition, CASH formation
- 600°C: AFm formation
- 800°C: Advanced phase decomposition

### 3. Thermogravimetric Analysis (TGA/DTA)
**Scale**: Bulk (mm)
**Key Metrics**:
- Total Mass Loss (wt%)
- Rubber Mass Loss (wt%)
- Cement Mass Loss (wt%)
- DTA Peak Analysis (μV/mg)

**Thermal Decomposition Sequence**:
- 100°C: Free water loss
- 200°C: Bound water loss
- 400°C: CH decomposition
- 600°C: CSH decomposition
- 150-500°C: Rubber decomposition

### 4. Micro-Computed Tomography (Micro-CT)
**Scale**: Micro (µm)
**3D Regions**: 3 per sample
**Key Metrics**:
- Total Porosity (vol%)
- Pore Connectivity (dimensionless)
- Tortuosity (dimensionless)
- Rubber Void Volume Fraction (vol%)
- Rubber Void Sphericity (dimensionless)
- Crack Volume Fraction (vol%)
- Crack Orientation Preference (dimensionless)

**3D Spatial Features**:
- Voxel size: 1.0 μm
- Analyzed volume: 1000 μm³
- Statistical analysis across multiple regions

## Rubber-Specific Degradation Signatures

### Melt Phase Formation
- Onset temperature: 150°C
- Complete melting: 450°C
- Quantified via SEM and Micro-CT

### Gas Evolution
- Pore formation from rubber decomposition
- Temperature-dependent evolution
- Measured via SEM pore density

### Interface Degradation
- Rubber-cement interface quality
- Temperature-dependent degradation
- Quantified via SEM interface quality index

### Void Morphology Changes
- Sphericity changes with temperature
- Volume fraction evolution
- 3D spatial distribution

## Statistical Robustness

### Replication Strategy
- 3 replicates per condition
- Multiple fields of view (SEM: 5, Micro-CT: 3)
- Statistical measures: mean, std, min, max, count

### Cross-Validation
- Internal consistency checks
- Temperature-dependent validation flags
- Measurement quality scores

## Data Quality Metrics

### Measurement Quality
- Quality scores: 0.8-1.0
- Cross-validation scores included
- Temperature and rubber content consistency flags

### Uncertainty Quantification
- Standard deviations for all measurements
- Confidence intervals based on replicates
- Measurement noise modeling

## File Formats

### Primary Dataset
- **CSV**: `fire_resistant_concrete_dataset.csv` (4,548 records)
- **JSON**: `fire_resistant_concrete_dataset.json` (structured format)

### Metadata
- **JSON**: `dataset_metadata.json` (generation parameters)
- **TXT**: `dataset_summary.txt` (summary statistics)

### Analysis Results
- **CSV**: `summary_statistics.csv` (comprehensive statistics)
- **JSON**: Analysis results for temperature evolution, rubber effects, phase transformations
- **PNG**: Visualization plots for all analytical techniques

## Usage Guidelines

### For Mechanistic Model Development
1. Use temperature evolution data to identify critical thresholds
2. Correlate phase transformations with microstructural changes
3. Apply rubber-specific degradation models
4. Validate against 3D spatial data

### For Statistical Analysis
1. Use replicate data for uncertainty quantification
2. Apply cross-validation metrics for model validation
3. Consider measurement quality scores in analysis

### For Multi-Scale Modeling
1. SEM data for micro-scale features
2. XRD data for phase composition
3. TGA data for thermal behavior
4. Micro-CT data for 3D spatial modeling

## Key Findings

### Temperature-Dependent Evolution
- Critical thresholds at 200°C, 400°C, 600°C, 800°C
- Phase transformations follow expected cement chemistry
- Rubber degradation follows polymer decomposition kinetics

### Rubber Content Effects
- Linear relationship with void formation
- Non-linear effects on interface quality
- Temperature-dependent degradation rates

### Multi-Technique Correlation
- Strong correlation between TGA mass loss and porosity
- XRD phase changes correlate with SEM microstructural changes
- 3D spatial data validates 2D SEM observations

## Technical Specifications

### Data Generation
- Python-based synthetic data generation
- Realistic parameter distributions
- Physically consistent relationships
- Statistical noise modeling

### Validation
- Cross-technique consistency checks
- Temperature-dependent validation
- Rubber content correlation validation
- Statistical significance testing

## Contact and Citation

This dataset was generated for research purposes. For questions or collaboration, please refer to the research team.

**Research Title**: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

**Phase**: Phase 3 - Microstructural and Chemical Analysis

**Generated**: 2025-01-27

---

*This documentation provides comprehensive guidance for using the fire-resistant concrete microstructural analysis dataset. The dataset is designed to support mechanistic model development rather than just phenomenological fitting, enabling deep understanding of thermo-mechanical degradation mechanisms.*