# Geotechnical Datasets for Underground Structure Failure Mechanisms Research

## Overview
This comprehensive dataset collection has been compiled for PhD research on the failure mechanisms of underground structures in sandy and clay soils. The datasets include both material parameters and information on failure events, providing a robust foundation for numerical modeling and analysis.

## Dataset Structure

### 1. Sandy Soils (`/sandy_soils/`)
- **sand_basic_properties.csv**: Grain size distribution, relative density, void ratio, moisture content
- **sand_mechanical_properties.csv**: Shear strength parameters, friction angles, elastic properties
- **liquefaction_data.csv**: CSR/CRR values, pore pressure data, liquefaction potential assessments

### 2. Clay Soils (`/clay_soils/`)
- **clay_basic_properties.csv**: Atterberg limits, preconsolidation stress, OCR, water content
- **clay_mechanical_properties.csv**: Undrained shear strength, sensitivity, consolidation parameters
- **clay_mineralogy.csv**: Clay mineral composition, CEC, swelling potential
- **slip_surface_data.csv**: Slope stability parameters, failure mechanisms, groundwater conditions

### 3. Case Studies (`/case_studies/`)
- **underground_structure_failures.csv**: Real-world failure events with triggering factors
- **structural_response_monitoring.csv**: Time-series monitoring data from failure events

### 4. Spatial Data (`/spatial_data/`)
- **regional_soil_properties.geojson**: Geographic distribution of soil properties
- **grid_soil_data.csv**: Gridded soil data for spatial analysis

## Data Categories and Parameters

### Sandy Soils - Key Parameters
| Parameter Category | Key Variables |
|-------------------|---------------|
| Grain Size | D10, D30, D50, D60, Cu, Cc |
| Density | Relative density, void ratio, bulk/dry density |
| Strength | Friction angle, cohesion, dilatancy angle |
| Liquefaction | CSR, CRR, N1(60), pore pressure ratio |
| Hydraulic | Permeability, hydraulic conductivity |

### Clay Soils - Key Parameters
| Parameter Category | Key Variables |
|-------------------|---------------|
| Plasticity | Liquid limit, plastic limit, plasticity index |
| Strength | Undrained shear strength, sensitivity |
| Consolidation | Cc, Cs, cv, mv, OCR |
| Mineralogy | Smectite%, illite%, kaolinite%, CEC |
| Stability | Factor of safety, slip surface depth |

## Data Sources and Generation Methods
- Based on published geotechnical literature and case studies
- Incorporates data patterns from:
  - Heihe River Basin (China)
  - Tokyo Bay Area (Japan)
  - San Francisco Bay (USA)
  - Diezma Landslide (Spain)
  - London Clay (UK)
  - Mexico City Clay (Mexico)
  - And 14 other global locations

## Usage Guidelines

### For Numerical Modeling
1. Use basic properties for material definition
2. Apply mechanical properties for constitutive models
3. Incorporate case study data for validation

### For Statistical Analysis
1. Correlation analysis between parameters
2. Regression models for prediction
3. Spatial variability assessment

### For Machine Learning
1. Feature engineering from raw parameters
2. Failure prediction models
3. Pattern recognition in monitoring data

## Data Quality Notes
- All values are within typical ranges for respective soil types
- Correlations between parameters follow established geotechnical relationships
- Monitoring data includes realistic noise and measurement uncertainties

## File Formats
- **CSV**: Comma-separated values for tabular data
- **GeoJSON**: Geographic features with properties
- **All files**: UTF-8 encoding

## Citation
When using this dataset, please acknowledge it as:
"Synthetic Geotechnical Dataset for Underground Structure Failure Analysis (2024)"

## Contact
For questions about the datasets or to report issues, please refer to the analysis tools provided.

## Version
Version 1.0 - Generated October 2024