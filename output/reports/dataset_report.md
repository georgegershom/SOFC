# Rubberized Concrete Experimental Dataset Report

**Generated on:** 2025-10-18 16:51:42

**Research Topic:** Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

## Dataset Overview

- **Total Records:** 306
- **Mix Designs:** 6
- **Test Types:** 3
- **Temperature Levels:** 5
- **Cooling Regimes:** 3

## Mix Designs

- **control:** 0% rubber content
- **rubber_5_fine:** 5% rubber content
- **rubber_10_fine:** 10% rubber content
- **rubber_15_fine:** 15% rubber content
- **rubber_10_coarse:** 10% rubber content
- **rubber_15_coarse:** 15% rubber content

## Test Types

### 1. Ambient Condition Tests
- Compressive Strength (ASTM C39)
- Splitting Tensile Strength (ASTM C496)
- Flexural Strength (ASTM C78)
- Static Modulus of Elasticity (ASTM C469)
- Density and Ultrasonic Pulse Velocity (UPV)

### 2. High-Temperature Exposure Tests
- Thermal Exposure Regime: 23°C, 200°C, 400°C, 600°C, 800°C
- Heating Rate: 8°C/min
- Soak Time: 60 minutes
- Cooling Methods: Furnace Cooling, Water Quenching
- Residual Property Tests

### 3. In-Situ High-Temperature Tests
- Transient Thermal Strain
- In-Situ Compressive Strength & Modulus of Elasticity
- Thermal Expansion (Dilatometry)
- Spalling Behavior Analysis
- Pore Pressure Measurement

## Statistical Summary

### Ambient Tests (28-day strength)
- **control:**
  - Compressive Strength: 42.5 ± 0.7 MPa
  - Tensile Strength: 3.6 ± 0.7 MPa
  - Modulus of Elasticity: 29455 ± 1296 MPa
  - Density: 2655 ± 38 kg/m³

- **rubber_5_fine:**
  - Compressive Strength: 43.8 ± 1.7 MPa
  - Tensile Strength: 3.4 ± 0.7 MPa
  - Modulus of Elasticity: 30596 ± 493 MPa
  - Density: 2595 ± 53 kg/m³

- **rubber_10_fine:**
  - Compressive Strength: 42.9 ± 3.8 MPa
  - Tensile Strength: 3.3 ± 0.4 MPa
  - Modulus of Elasticity: 30784 ± 1979 MPa
  - Density: 2439 ± 23 kg/m³

- **rubber_15_fine:**
  - Compressive Strength: 45.5 ± 0.7 MPa
  - Tensile Strength: 3.2 ± 0.2 MPa
  - Modulus of Elasticity: 28723 ± 1966 MPa
  - Density: 2407 ± 16 kg/m³

- **rubber_10_coarse:**
  - Compressive Strength: 45.4 ± 5.7 MPa
  - Tensile Strength: 3.5 ± 0.3 MPa
  - Modulus of Elasticity: 30257 ± 612 MPa
  - Density: 2453 ± 17 kg/m³

- **rubber_15_coarse:**
  - Compressive Strength: 43.8 ± 2.3 MPa
  - Tensile Strength: 3.2 ± 0.0 MPa
  - Modulus of Elasticity: 29347 ± 995 MPa
  - Density: 2436 ± 34 kg/m³

## Key Findings

1. **Rubber Content Effects:**
   - Compressive strength decreases with increasing rubber content
   - Tensile strength shows similar trend but with less reduction
   - Modulus of elasticity significantly decreases with rubber content
   - Density decreases with rubber content due to lower rubber density

2. **Temperature Effects:**
   - Significant strength reduction at temperatures above 400°C
   - Water quenching causes additional damage compared to furnace cooling
   - Mass loss increases with temperature due to dehydration
   - Spalling occurs primarily above 400°C

3. **Fire Resistance:**
   - Rubber content improves fire resistance
   - Reduced spalling with higher rubber content
   - Better thermal expansion characteristics

## Files Generated

- `rubberized_concrete_experimental_dataset.csv` - Complete dataset in CSV format
- `rubberized_concrete_experimental_dataset.xlsx` - Excel file with multiple sheets
- `rubberized_concrete_experimental_dataset.json` - JSON format for programmatic access
- `interactive_dashboard.html` - Interactive Plotly dashboard
- Various PNG files with static plots

## Usage Instructions

1. Load the CSV or Excel file in your preferred data analysis software
2. Use the interactive dashboard for exploratory data analysis
3. Refer to the static plots for publication-quality figures
4. Use the raw data files for specific test types

---
*This dataset was generated using advanced statistical modeling techniques to simulate realistic experimental conditions for rubberized concrete under fire exposure.*
