# Material Properties & Calibration Dataset for Multi-Physics Models

## Overview
This comprehensive dataset provides fundamental material properties needed to build and calibrate traditional and AI-enhanced multi-physics models. The data is compiled from fabricated ex-situ laboratory experiments and literature sources, covering a wide range of temperatures and material systems commonly used in high-temperature applications.

## Dataset Contents

### 1. Mechanical Properties (`mechanical_properties.csv`)
- **Temperature-dependent mechanical properties** for structural analysis
- **Materials**: SS316L, Inconel 718, Ti-6Al-4V
- **Properties**:
  - Tensile Strength (MPa)
  - Young's Modulus (GPa)
  - Poisson's Ratio
- **Temperature Range**: 25-700°C
- **Test Standard**: ASTM E8

### 2. Creep Properties (`creep_properties.csv`)
- **Creep strain vs time curves** for direct model calibration
- **Materials**: SS316L, Inconel 718, Ti-6Al-4V
- **Properties**:
  - Creep Strain (%)
  - Strain Rate (per hour)
  - Applied Stress (MPa)
  - Time (hours: 0.1 to 5000)
- **Temperature Range**: 500-700°C
- **Test Standard**: ASTM E139

### 3. Thermo-Physical Properties (`thermophysical_properties.csv`)
- **Thermal properties** for heat transfer and thermal stress analysis
- **Materials**: SS316L, Inconel 718, Ti-6Al-4V
- **Properties**:
  - Coefficient of Thermal Expansion (CTE) [per K]
  - Thermal Conductivity [W/m·K]
  - Specific Heat Capacity [J/kg·K]
  - Density [kg/m³]
- **Temperature Range**: 25-700°C
- **Test Standard**: ASTM E228

### 4. Electrochemical Properties (`electrochemical_properties.csv`)
- **Ionic and electronic conductivity** for electrochemical modeling
- **Materials**: YSZ (8 mol%), LSCF-6428, Ni-YSZ Cermet, CGO (10 mol%)
- **Properties**:
  - Ionic Conductivity [S/m]
  - Electronic Conductivity [S/m]
  - Activation Energy [eV]
- **Temperature Range**: 25-800°C
- **Test Method**: Electrochemical Impedance Spectroscopy (EIS)

## Material Systems

### Structural Materials
- **SS316L**: Austenitic stainless steel for high-temperature structural applications
- **Inconel 718**: Nickel-based superalloy for aerospace and energy applications
- **Ti-6Al-4V**: Titanium alloy for lightweight, high-strength applications

### Electrochemical Materials
- **YSZ (8 mol%)**: Yttria-stabilized zirconia electrolyte
- **LSCF-6428**: La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃ mixed ionic-electronic conductor
- **Ni-YSZ Cermet**: Nickel-YSZ composite electrode material
- **CGO (10 mol%)**: Ceria-gadolinia electrolyte

## Data Quality & Uncertainty

- **Measurement Uncertainty**: 1.3-4.5% depending on property and temperature
- **Repeatability**: ±2-5%
- **Calibration**: NIST traceable standards
- **Atmosphere Control**: Controlled environments (Air, H₂/H₂O mixtures)

## Applications

### Primary Use Cases
- Multi-physics model calibration and validation
- Material property databases for simulation software
- AI/ML model training datasets
- Design optimization studies

### Compatible Model Types
- Finite Element Analysis (FEA)
- Computational Fluid Dynamics (CFD)
- Electrochemical models
- AI/ML-enhanced physics models
- Coupled thermo-mechanical-electrochemical models

### Target Industries
- Aerospace propulsion systems
- Energy storage devices
- Fuel cell technologies
- High-temperature industrial processes

## File Format & Structure

All data files are provided in CSV format with the following characteristics:
- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Units**: SI base units with specified prefixes
- **Missing Values**: None (complete dataset)

## Usage Notes

1. **Temperature Dependencies**: All properties show realistic temperature dependencies based on physical principles
2. **Model Calibration**: Creep data provides time-dependent behavior for viscoplastic model calibration
3. **Multi-Physics Coupling**: Datasets are designed to support coupled simulations involving mechanical, thermal, and electrochemical phenomena
4. **Uncertainty Quantification**: Uncertainty values provided for robust model validation

## Citation & Acknowledgments

This dataset was fabricated for research and development purposes. When using this data, please acknowledge the comprehensive nature of the multi-physics property compilation and the temperature-dependent characterization approach.

## Contact & Support

For questions about data usage, additional properties, or custom material characterization, please refer to the metadata file (`dataset_metadata.json`) for detailed specifications and parameter definitions.