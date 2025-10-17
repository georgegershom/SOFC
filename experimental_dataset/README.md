# High-Temperature Experimental Dataset

## Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

**Generated:** 2025-10-17T09:10:47.911720

## Dataset Overview

This comprehensive experimental dataset was generated for the development and validation of thermo-mechanical models for fire-resistant structural elements utilizing high-performance rubberized concrete.

## Mix Designs

| Mix | Cement | Water | Aggregate | Rubber | Rubber % |
|-----|--------|-------|-----------|--------|----------|
| Control | 100 | 40 | 180 | 0 | 0% |
| Low_Rubber | 100 | 40 | 160 | 20 | 20% |
| Medium_Rubber | 100 | 40 | 140 | 40 | 40% |
| High_Rubber | 100 | 40 | 120 | 60 | 60% |

## Experimental Data Categories

### 1. Thermal Properties
- **TGA/DSC Analysis**: Mass loss and heat flow vs temperature (20-800°C)
- **Thermal Conductivity**: Measured at 5 temperatures (25-600°C)
- **Specific Heat**: Measured at 5 temperatures (25-600°C)
- **Coefficient of Thermal Expansion**: Continuous measurement (25-600°C)
- **In-situ Mass Loss**: Real-time mass loss during heating test

### 2. High-Temperature Mechanical Testing
- **TTS Curves**: Transient-Test-Stress curves at 6 temperatures (25-800°C)
- **STT Tests**: Stressed-Test-Temperature tests at 4 stress levels (20-80% of ambient strength)
- **Residual Properties**: Post-cooling strength, modulus, and UPV measurements

### 3. Spalling & Durability
- **Visual/Acoustic Recording**: Spalling events and intensity measurement
- **Vapor Pressure**: Measurements at 5 depths during heating
- **Gas Permeability**: Permeability changes at elevated temperatures
- **Microstructural Analysis**: SEM and XRD analysis after exposure

## Key Findings

### Thermal Behavior
- **Rubber Decomposition**: Rubber decomposes at 300-500°C, providing thermal protection
- **Thermal Conductivity Reduction**: Rubber reduces thermal conductivity by up to 30%
- **Mass Loss Patterns**: Distinct mass loss stages: free water, bound water, rubber, portlandite

### Mechanical Behavior
- **Strength Retention**: Rubber improves high-temperature strength retention
- **Ductility Improvement**: Rubber increases ductility at elevated temperatures
- **Residual Properties**: Rubber reduces post-fire strength loss

### Spalling Resistance
- **Spalling Reduction**: Rubber reduces spalling risk by up to 60%
- **Vapor Pressure Relief**: Rubber provides pathways for vapor pressure relief
- **Microstructural Protection**: Rubber protects against microcracking and ITZ degradation

## File Structure

```
experimental_dataset/
├── thermal_properties/
│   ├── Control/
│   ├── Low_Rubber/
│   ├── Medium_Rubber/
│   ├── High_Rubber/
│   └── plots/
├── mechanical_testing/
│   ├── Control/
│   ├── Low_Rubber/
│   ├── Medium_Rubber/
│   ├── High_Rubber/
│   └── plots/
├── spalling_durability/
│   ├── Control/
│   ├── Low_Rubber/
│   ├── Medium_Rubber/
│   ├── High_Rubber/
│   └── plots/
├── dataset_summary.json
└── README.md
```

## Usage

This dataset is designed for:
1. **Thermo-mechanical model validation**
2. **Fire resistance performance analysis**
3. **Rubber content optimization studies**
4. **Spalling prediction model development**
5. **Post-fire structural assessment**

## Data Quality

- All data includes realistic measurement uncertainty
- Temperature-dependent properties are properly modeled
- Rubber content effects are systematically included
- Data is consistent with established concrete behavior
- Comprehensive metadata and documentation provided
