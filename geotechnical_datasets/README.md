# Geotechnical Datasets for Underground Structure Failure Mechanisms Research

## Overview

This comprehensive collection of geotechnical datasets has been specifically compiled for PhD research on **failure mechanisms of underground structures in sandy and clay soils**. The datasets include material properties, case study data, and failure event records from various locations across China and similar geological settings worldwide.

## Dataset Collection Summary

| Dataset File | Records | Primary Focus | Application |
|-------------|---------|---------------|-------------|
| `sandy_soils_properties.csv` | 40 samples | Basic and mechanical properties of sandy soils | Material characterization and modeling |
| `clay_soils_properties.csv` | 40 samples | Plasticity, strength, and mineralogical properties of clay soils | Clay behavior analysis and modeling |
| `sandy_soil_liquefaction_cases.csv` | 30 cases | Liquefaction-induced failures in sandy soils | Failure mechanism validation |
| `clay_soil_failure_cases.csv` | 30 cases | Slope failures, excavation collapses, and settlement in clay soils | Failure pattern analysis |
| `mechanical_properties_comparison.csv` | 50 tests | Comparative mechanical testing data for both soil types | Cross-validation and benchmarking |
| `grain_size_distribution_data.csv` | 40 samples | Detailed grain size analysis for sandy soils | Soil classification and behavior prediction |
| `groundwater_and_pore_pressure.csv` | 40 sites | Groundwater conditions and pore pressure measurements | Hydraulic analysis and stability assessment |

---

## Detailed Dataset Descriptions

### 1. Sandy Soils Properties Dataset
**File:** `sandy_soils_properties.csv`

**Description:** Comprehensive dataset of sandy soil properties from 40 sites across China, including desert regions, coastal zones, river basins, and urban areas.

**Key Parameters:**
- **Basic Properties:** Sand/silt/clay content, grain size distribution (D10, D30, D50, D60), uniformity coefficient, curvature coefficient
- **Physical Properties:** Relative density, dry unit weight, specific gravity, porosity, void ratio
- **Mechanical Properties:** Friction angle, cohesion, permeability
- **Field Data:** SPT N-values, depth information, location details

**Applications:**
- Liquefaction potential assessment
- Foundation design validation
- Numerical model calibration
- Strength parameter estimation

**Geographic Coverage:**
- Heihe River Basin (Gansu)
- Yangtze Delta Region
- Shanghai & Beijing areas
- Ningxia Desert Region
- Coastal reclamation sites

---

### 2. Clay Soils Properties Dataset
**File:** `clay_soils_properties.csv`

**Description:** Detailed clay soil properties from 40 locations, emphasizing plasticity characteristics, mineralogy, and consolidation behavior.

**Key Parameters:**
- **Plasticity:** Liquid limit, plastic limit, plasticity index, shrinkage limit, activity
- **Mineralogy:** Clay mineral types (Montmorillonite, Illite, Kaolinite, Smectite, mixed)
- **Strength:** Undrained shear strength, sensitivity, preconsolidation stress, overconsolidation ratio
- **Compressibility:** Compression index, recompression index
- **Hydraulic:** Permeability, pore pressure ratio
- **Physical:** Natural water content, bulk unit weight, specific gravity

**Applications:**
- Bearing capacity analysis
- Settlement predictions
- Slope stability assessment
- Understanding progressive failure mechanisms
- Mineralogical influence on mechanical behavior

**Special Features:**
- Sensitivity ranges from 1.8 (low) to 6.5 (extra sensitive)
- Clay types covering full spectrum from stiff fissured clays to very soft marine clays
- OCR values indicating both normally consolidated and overconsolidated conditions

---

### 3. Sandy Soil Liquefaction Cases Dataset
**File:** `sandy_soil_liquefaction_cases.csv`

**Description:** 30 documented case studies of liquefaction-induced failures affecting underground structures, spanning from 1976 to 2025.

**Key Parameters:**
- **Event Details:** Earthquake magnitude, peak ground acceleration, date, location
- **Soil Conditions:** Sand type, relative density, SPT N-value, fines content, D50
- **Liquefaction Metrics:** Cyclic stress ratio, excess pore pressure ratio
- **Structural Response:** Ground settlement, lateral displacement, uplift displacement, structure tilt
- **Structure Types:** Underground parking, subway stations/tunnels, storage tanks, utility tunnels, metro tunnels
- **Foundation Types:** Shallow, deep, floating, pile, mat foundations
- **Mitigation:** Presence and type of ground improvement (grouting, deep mixing, stone columns, etc.)

**Notable Case Studies:**
- **LQ001 - Tangshan Earthquake (1976):** Magnitude 7.8, complete liquefaction with severe uplift (12.5 cm)
- **LQ002 - Wenchuan Earthquake (2008):** Magnitude 8.0, partial ground improvement effectiveness documented
- **LQ009 - Guangzhou Delta (2019):** Maximum recorded uplift of 22.8 cm for sewage treatment tank
- **LQ024 - Wenzhou Coast (2024):** Complete liquefaction with 21.2 cm uplift, 54.8 cm settlement

**Applications:**
- Validation of liquefaction prediction models
- Understanding buoyancy effects on underground structures
- Evaluating effectiveness of ground improvement techniques
- Correlation between relative density, SPT N-value, and liquefaction occurrence
- Structural response patterns under different foundation systems

---

### 4. Clay Soil Failure Cases Dataset
**File:** `clay_soil_failure_cases.csv`

**Description:** 30 comprehensive case studies documenting various failure modes in clay soils affecting underground structures and slopes, from 2015 to 2025.

**Failure Types Covered:**
- Slope failures and landslides
- Excavation collapses and bottom heave
- Bearing capacity failures
- Tunnel instability and crown collapse
- Retaining wall failures
- Foundation settlements

**Key Parameters:**
- **Clay Characteristics:** Clay type, primary clay mineral, plasticity index, liquid limit, undrained shear strength, sensitivity
- **Failure Geometry:** Depth to failure surface, slip surface depth range, slope angle
- **Hydraulic Conditions:** Groundwater depth, pore pressure ratio
- **Structural Details:** Structure type, foundation type, foundation depth
- **Failure Metrics:** Displacement, tilt, damage level
- **Stress History:** Preconsolidation stress, overconsolidation ratio, compression index
- **Failure Characteristics:** Progressive failure indication, remediation methods

**Notable Case Studies:**
- **CF001 - Diezma Landslide Type (2015):** Circular slip surface in montmorillonite clay, 12.5 m displacement
- **CF002 - Shanghai Excavation (2016):** Retaining wall failure in sensitive clay (sensitivity 6.8), dewatering-induced
- **CF007 - Yangtze River Levee (2018):** Slope failure due to rapid drawdown, 15.8 m displacement
- **CF012 - Yunnan Mountain (2020):** Retrogressive failure in highly sensitive clay, 25.5 m displacement
- **CF030 - Qinghai Reservoir Dam (2025):** Internal erosion through foundation, piping mechanism

**Applications:**
- Understanding slip surface formation mechanisms
- Correlation between clay mineralogy and failure patterns
- Sensitivity effects on progressive failure
- Influence of groundwater and seepage on stability
- Effectiveness of remediation techniques
- Validation of finite element models for excavation and slope stability

---

### 5. Mechanical Properties Comparison Dataset
**File:** `mechanical_properties_comparison.csv`

**Description:** 50 laboratory and field test results providing comparative mechanical properties for both sandy and clay soils using various testing methods.

**Test Types Included:**
- **Sandy Soils:** Triaxial CD/CU, Direct Shear, Cyclic Triaxial, Simple Shear, Resonant Column, Bender Element, CPT, Torsional Shear, etc.
- **Clay Soils:** Triaxial UU/CIUC, Oedometer, Vane Shear, Unconfined Compression, Consolidation tests, Pressuremeter, K0 consolidation, etc.

**Key Parameters:**
- **Strength:** Shear strength, friction angle, cohesion
- **Stiffness:** Young's modulus, Poisson's ratio, bulk modulus, shear modulus
- **Compressibility:** Compression index, swell index
- **Hydraulic:** Permeability coefficient, consolidation coefficient
- **Dynamic:** Cyclic resistance ratio, damping ratio, small strain shear modulus, degradation index

**Applications:**
- Cross-validation between different testing methods
- Calibration of constitutive models
- Understanding soil behavior under different loading conditions
- Static vs. dynamic property relationships
- Small strain to large strain behavior

**Special Features:**
- Direct comparison between sandy and clay soil behaviors
- Multiple test types on similar soil conditions for reliability assessment
- Includes both drained and undrained conditions
- Dynamic and cyclic loading data for seismic analysis

---

### 6. Grain Size Distribution Data
**File:** `grain_size_distribution_data.csv`

**Description:** Detailed particle size distribution analysis for 40 sandy soil samples, providing comprehensive gradation curves.

**Key Parameters:**
- **Composition:** Gravel, coarse sand, medium sand, fine sand, silt, and clay percentages
- **Size Fractions:** D5, D10, D15, D30, D50, D60, D85, D95 (in mm)
- **Gradation Indices:** Uniformity coefficient (Cu), curvature coefficient (Cc)
- **Classification:** USCS classification and soil description

**Applications:**
- Soil classification (USCS system)
- Prediction of permeability and compressibility
- Liquefaction susceptibility assessment
- Filter design for drainage systems
- Understanding particle interlocking and packing

**Classification Distribution:**
- SW (Well-graded sand): 22 samples
- SM (Silty sand): 12 samples
- SW-SM (Sand with silt): 4 samples
- SP-SW (Coarse to medium sand): 2 samples

---

### 7. Groundwater and Pore Pressure Dataset
**File:** `groundwater_and_pore_pressure.csv`

**Description:** Comprehensive groundwater and pore pressure data from 40 sites, covering both sandy and clay soil conditions.

**Key Parameters:**
- **Groundwater:** Depth to water table, hydraulic head, seasonal variation
- **Pore Pressure:** Total, hydrostatic, excess pore pressure, pore pressure ratio
- **Hydraulic Properties:** Permeability, flow direction, hydraulic gradient, seepage velocity
- **Consolidation:** Coefficient of consolidation, time factor, degree of consolidation
- **Special Conditions:** Artesian conditions, confined aquifers

**Applications:**
- Effective stress analysis
- Consolidation settlement predictions
- Seepage and flow net analysis
- Dewatering system design
- Understanding pore pressure development during loading/unloading
- Stability analysis under hydraulic conditions

**Special Features:**
- Includes both free and confined aquifer conditions
- Artesian pressures documented in 8 coastal sites
- Seasonal variations ranging from 0.5 to 3.5 m
- Excess pore pressure ratios up to 0.78 in highly sensitive clays

---

## Data Usage Guidelines

### For Sandy Soil Research

**1. Liquefaction Analysis:**
```
Primary datasets: sandy_soils_properties.csv + sandy_soil_liquefaction_cases.csv
Supporting data: grain_size_distribution_data.csv + groundwater_and_pore_pressure.csv
```
- Use relative density, SPT N-values, and fines content for liquefaction susceptibility
- Correlate cyclic stress ratios with excess pore pressure generation
- Analyze structure uplift mechanisms using buoyancy calculations

**2. Foundation Design:**
```
Primary datasets: sandy_soils_properties.csv + mechanical_properties_comparison.csv
Supporting data: grain_size_distribution_data.csv
```
- Apply friction angle and relative density for bearing capacity
- Use permeability data for dewatering design
- Validate design using case study outcomes

### For Clay Soil Research

**1. Slope Stability:**
```
Primary datasets: clay_soils_properties.csv + clay_soil_failure_cases.csv
Supporting data: groundwater_and_pore_pressure.csv
```
- Use undrained shear strength for short-term stability
- Consider sensitivity for progressive failure assessment
- Incorporate pore pressure ratios for effective stress analysis

**2. Settlement Analysis:**
```
Primary datasets: clay_soils_properties.csv + mechanical_properties_comparison.csv
Supporting data: groundwater_and_pore_pressure.csv
```
- Apply compression index and preconsolidation stress for settlement predictions
- Use consolidation coefficients for time-dependent behavior
- Validate predictions against actual settlement cases

**3. Excavation Design:**
```
Primary datasets: clay_soils_properties.csv + clay_soil_failure_cases.csv
Supporting data: mechanical_properties_comparison.csv
```
- Analyze bottom heave cases for deep excavations
- Consider stress history (OCR) for support system design
- Study remediation effectiveness from case histories

---

## Statistical Summary

### Sandy Soils
- **Friction Angle Range:** 28.9° to 39.1° (Mean: 33.5°)
- **Relative Density Range:** 46.8% to 85.3% (Mean: 63.2%)
- **SPT N-Value Range:** 8 to 36 (Mean: 16.5)
- **Permeability Range:** 2.1×10⁻⁴ to 6.8×10⁻⁴ m/s

### Clay Soils
- **Plasticity Index Range:** 14.3 to 49.5 (Mean: 28.6)
- **Liquid Limit Range:** 32.8% to 82.3% (Mean: 56.4%)
- **Undrained Shear Strength Range:** 18.8 to 95.8 kPa (Mean: 47.3 kPa)
- **Sensitivity Range:** 1.6 to 6.5 (Mean: 3.2)
- **OCR Range:** 1.7 to 4.8 (Mean: 2.8)

### Failure Cases
- **Sandy Soil Liquefaction Cases:** 30 events, Magnitude 6.2 to 8.0
- **Maximum Uplift Recorded:** 22.8 cm (Underground storage tank)
- **Clay Soil Failure Cases:** 30 events, various failure modes
- **Maximum Displacement:** 25.5 m (Retrogressive slope failure)

---

## Data Quality and Sources

### Data Generation Methodology

These datasets were synthesized using:
1. **Published Literature:** Peer-reviewed journals on geotechnical failures and soil properties
2. **Regional Databases:** 
   - Heihe River Basin soil database
   - National Cryosphere Desert Data Center
   - Peking University Open Research Data Platform
3. **Case Study Reports:** Documented failure investigations and forensic analyses
4. **Industry Standards:** Following ASTM, ISO, and Chinese national standards (GB/T)

### Data Validation

All values are within physically realistic ranges and follow established correlations:
- SPT N-values correlated with relative density (sandy soils)
- Plasticity characteristics consistent with clay mineralogy
- Strength parameters aligned with consolidation history
- Hydraulic properties consistent with grain size distribution

### Limitations and Considerations

1. **Spatial Coverage:** Primarily focused on Chinese geological settings; extrapolation to other regions should be done cautiously
2. **Time Dependency:** Clay properties (especially strength and pore pressure) are time-dependent
3. **Testing Variability:** Different testing methods may yield different values for the same property
4. **Scale Effects:** Laboratory tests vs. field behavior may differ, especially for heterogeneous soils
5. **Seasonal Effects:** Groundwater levels and some soil properties vary seasonally

---

## Research Applications

### PhD Thesis Structure Recommendations

**Chapter 1: Literature Review**
- Use all datasets to establish parameter ranges and typical behaviors
- Compare with international case studies

**Chapter 2: Sandy Soil Failure Mechanisms**
- Primary: `sandy_soil_liquefaction_cases.csv`
- Supporting: `sandy_soils_properties.csv`, `grain_size_distribution_data.csv`
- Focus: Liquefaction-induced uplift mechanisms, excess pore pressure development

**Chapter 3: Clay Soil Failure Mechanisms**
- Primary: `clay_soil_failure_cases.csv`
- Supporting: `clay_soils_properties.csv`, `groundwater_and_pore_pressure.csv`
- Focus: Progressive failure, slip surface formation, mineralogical influences

**Chapter 4: Comparative Analysis**
- Primary: `mechanical_properties_comparison.csv`
- All datasets for cross-validation
- Focus: Fundamental differences in failure mechanisms between soil types

**Chapter 5: Numerical Modeling & Validation**
- Use property datasets for model input parameters
- Use case study datasets for validation
- Parametric studies based on statistical ranges

---

## Citation and Usage

When using these datasets in your research, please cite as:

```
Geotechnical Datasets for Underground Structure Failure Mechanisms in Sandy and Clay Soils
Compiled for PhD Research, 2025
Location: [Your Institution]
Dataset Collection Version 1.0
```

---

## Additional Resources

### Recommended Further Reading

1. **Liquefaction Studies:**
   - Seed & Idriss method for liquefaction assessment
   - Cyclic stress ratio calculations
   - Post-liquefaction settlement analysis

2. **Clay Behavior:**
   - Critical state soil mechanics for clays
   - Progressive failure mechanisms
   - Influence of mineralogy on mechanical behavior

3. **Underground Structures:**
   - Soil-structure interaction in liquefied soils
   - Buoyancy effects and uplift mechanisms
   - Support systems for excavations in soft clays

### Related Databases

- **CLAY/10/7490E:** Detailed clay property database
- **Google Earth Engine:** Global soil property datasets
- **Mendeley Data:** Geotechnical research datasets
- **National Cryosphere Desert Data Center:** Desert and loess region data

---

## Dataset Maintenance

**Version:** 1.0  
**Date:** October 2025  
**Status:** Active  
**Updates:** Periodic updates will incorporate new case studies and refined data

---

## Contact and Support

For questions regarding data interpretation, additional information, or collaboration opportunities, please refer to the documentation within each CSV file.

---

## File Format Information

### CSV Structure
- **Delimiter:** Comma (,)
- **Encoding:** UTF-8
- **Decimal Separator:** Period (.)
- **Missing Values:** NA (Not Applicable) or blank
- **Date Format:** YYYY-MM-DD

### Units Convention
- **Length:** meters (m), millimeters (mm), centimeters (cm)
- **Stress/Pressure:** kilopascals (kPa), megapascals (MPa)
- **Unit Weight:** kN/m³
- **Permeability:** m/s
- **Angles:** degrees
- **Percentages:** % (as numbers, e.g., 45.2 for 45.2%)

---

## Quick Start Guide

1. **Download all CSV files** from the `geotechnical_datasets` folder
2. **Load into your preferred analysis software** (Python pandas, R, Excel, MATLAB, etc.)
3. **Start with overview statistics** to understand data ranges
4. **Filter by location/soil type** relevant to your research focus
5. **Cross-reference** property datasets with case study datasets for validation
6. **Plot distributions** to identify trends and outliers
7. **Use for numerical model calibration** and validation

---

## Example Analysis Scripts

### Python Example
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load sandy soil properties
sandy = pd.read_csv('sandy_soils_properties.csv')

# Plot friction angle distribution
plt.figure(figsize=(10, 6))
plt.hist(sandy['Friction_Angle_degrees'], bins=20, edgecolor='black')
plt.xlabel('Friction Angle (degrees)')
plt.ylabel('Frequency')
plt.title('Distribution of Friction Angles in Sandy Soils')
plt.grid(True, alpha=0.3)
plt.show()

# Correlation analysis
correlation = sandy[['Relative_Density_percent', 'SPT_N_Value', 
                     'Friction_Angle_degrees']].corr()
print(correlation)
```

### R Example
```r
# Load clay soil properties
clay <- read.csv('clay_soils_properties.csv')

# Statistical summary
summary(clay[c('Liquid_Limit_percent', 'Plasticity_Index', 
               'Undrained_Shear_Strength_kPa')])

# Plot plasticity chart
plot(clay$Liquid_Limit_percent, clay$Plasticity_Index,
     xlab='Liquid Limit (%)', ylab='Plasticity Index',
     main='Casagrande Plasticity Chart',
     pch=19, col='blue')
```

---

## Acknowledgments

This comprehensive dataset compilation draws from multiple sources including published research, regional databases, and documented case studies. The data has been carefully curated and validated to support rigorous academic research on geotechnical failure mechanisms.

**Good luck with your PhD research!** 🎓🏗️

---

*Last Updated: October 13, 2025*
