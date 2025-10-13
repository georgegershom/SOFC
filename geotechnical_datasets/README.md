# Geotechnical Datasets for PhD Research

## Failure Mechanisms of Underground Structures in Sandy and Clay Soils

This repository contains comprehensive synthetic geotechnical datasets designed for PhD research on failure mechanisms of underground structures. The datasets cover both sandy and clay soil properties, case studies of actual failures, and geospatial data for regional analysis.

## 📁 Dataset Structure

```
geotechnical_datasets/
├── sandy_soils/           # Sandy soil properties and liquefaction data
├── clay_soils/            # Clay soil properties and mineralogy data
├── case_studies/          # Failure case studies and monitoring data
├── geospatial_data/       # Regional soil maps and hazard assessments
├── scripts/               # Data generation and analysis tools
└── documentation/         # Analysis results and visualizations
```

## 🏗️ Dataset Categories

### 1. Sandy Soil Properties

**Files:**
- `sandy_soil_basic_properties.csv` - Basic soil properties (500 samples)
- `sandy_soil_mechanical_properties.csv` - Strength and deformation parameters
- `sandy_soil_liquefaction_data.csv` - Liquefaction susceptibility data
- `sandy_soil_complete_dataset.csv` - Combined comprehensive dataset

**Key Parameters:**
- Grain size distribution (d10, d30, d60, uniformity coefficient)
- Relative density and void ratio
- Friction angle and cohesion
- Liquefaction resistance (CRR, CSR, Factor of Safety)
- Pore pressure parameters

### 2. Clay Soil Properties

**Files:**
- `clay_soil_basic_properties.csv` - Basic soil properties (400 samples)
- `clay_soil_mineralogy.csv` - Clay mineral composition
- `clay_soil_mechanical_properties.csv` - Strength and consolidation parameters
- `clay_soil_failure_susceptibility.csv` - Failure potential assessment
- `clay_soil_complete_dataset.csv` - Combined comprehensive dataset

**Key Parameters:**
- Atterberg limits (liquid limit, plastic limit, plasticity index)
- Clay mineralogy (smectite, illite, kaolinite, chlorite percentages)
- Undrained shear strength and sensitivity
- Consolidation parameters (compression index, OCR)
- Swelling and shrinkage properties

### 3. Case Studies

**Files:**
- `sandy_soil_failure_cases.csv` - 50 sandy soil failure cases
- `clay_soil_failure_cases.csv` - 40 clay soil failure cases
- `monitoring_data_time_series.csv` - Time-series monitoring data
- `case_study_summary.json` - Statistical summary

**Failure Types Covered:**

*Sandy Soils:*
- Liquefaction-induced settlement and lateral spreading
- Liquefaction-induced uplift
- Cyclic mobility and flow liquefaction
- Foundation bearing capacity failure

*Clay Soils:*
- Progressive slope failure and circular slip
- Retrogressive landslides
- Foundation heave and tunnel instability
- Quick clay landslides

### 4. Geospatial Data

**Files:**
- `regional_soil_properties_grid.csv` - Regional soil property grid (93,080 points)
- `borehole_data_detailed.csv` - Detailed borehole logs (298 boreholes)
- `liquefaction_hazard_map.csv` - Liquefaction hazard assessment
- `landslide_hazard_map.csv` - Landslide hazard assessment
- `geospatial_metadata.json` - Spatial data metadata

**Coverage Areas:**
- San Francisco Bay Area, CA
- Los Angeles Basin, CA
- Puget Sound, WA

## 🔬 Research Applications

### For Sandy Soils
1. **Liquefaction Analysis**: Use CRR, CSR, and factor of safety data to study liquefaction potential
2. **Strength Characterization**: Analyze friction angle relationships with density and grain size
3. **Permeability Studies**: Investigate flow characteristics and drainage behavior
4. **Case Study Validation**: Compare theoretical predictions with observed failure data

### For Clay Soils
1. **Plasticity Analysis**: Use Atterberg limits and plasticity charts for soil classification
2. **Mineralogical Effects**: Study influence of clay minerals on engineering behavior
3. **Consolidation Studies**: Analyze compression behavior and settlement predictions
4. **Stability Analysis**: Investigate slope stability and progressive failure mechanisms

### For Geospatial Analysis
1. **Regional Hazard Mapping**: Assess liquefaction and landslide susceptibility
2. **Spatial Correlation Studies**: Analyze spatial variability of soil properties
3. **Site Characterization**: Use borehole data for detailed subsurface modeling
4. **Risk Assessment**: Combine hazard maps with infrastructure data

## 📊 Key Statistics

| Dataset Category | Samples/Records | Key Features |
|------------------|----------------|--------------|
| Sandy Soils | 500 samples | 41 parameters including liquefaction data |
| Clay Soils | 400 samples | 45 parameters including mineralogy |
| Sandy Failures | 50 cases | Earthquake-induced failures |
| Clay Failures | 40 cases | Various triggering mechanisms |
| Geospatial Grid | 93,080 points | Regional soil property mapping |
| Boreholes | 298 locations | Detailed stratigraphic data |
| Monitoring Data | 3,686 records | Time-series failure evolution |

## 🛠️ Analysis Tools

### Data Generation Scripts
- `generate_sandy_soil_data.py` - Generate sandy soil datasets
- `generate_clay_soil_data.py` - Generate clay soil datasets
- `generate_case_studies.py` - Generate failure case studies
- `generate_geospatial_data.py` - Generate regional datasets

### Analysis Tools
- `data_analysis_tools.py` - Comprehensive analysis suite
  - Statistical analysis and correlations
  - Visualization and plotting
  - Predictive modeling with machine learning
  - Failure mechanism analysis

## 📈 Analysis Results

### Sandy Soil Correlations
- Relative Density vs Friction Angle: r = 0.809 (strong positive)
- Sand Content vs Cyclic Resistance: r = -0.799 (strong negative)
- Fines content significantly affects liquefaction resistance

### Clay Soil Correlations
- Liquid Limit vs Compression Index: r = 0.851 (very strong positive)
- Plasticity Index vs Undrained Strength: r = 0.180 (weak positive)
- OCR is the strongest predictor of undrained strength (69% importance)

### Failure Analysis
- **Sandy Soil Failures**: Total damage $53.9M across 50 cases
- **Clay Soil Failures**: Total damage $46.1M, 83 casualties across 40 cases
- Most common sandy failure: Foundation bearing capacity failure
- Most common clay failure: Retrogressive landslide

## 🎯 Research Focus Areas

### 1. Liquefaction Mechanisms
- Pore pressure generation and dissipation
- Cyclic loading effects on sandy soils
- Post-liquefaction behavior and reconsolidation

### 2. Progressive Failure in Clays
- Strain localization and shear band development
- Rate effects and time-dependent behavior
- Influence of mineralogy on failure mechanisms

### 3. Underground Structure Response
- Soil-structure interaction during failure
- Foundation performance in liquefiable soils
- Tunnel stability in clay formations

### 4. Mitigation Strategies
- Ground improvement techniques
- Early warning systems based on monitoring data
- Risk-based design approaches

## 📚 Data Usage Guidelines

### Quality Assurance
- All datasets are synthetic but based on realistic geotechnical relationships
- Statistical distributions match published literature values
- Correlations reflect established geotechnical principles

### Recommended Analyses
1. **Correlation Studies**: Use complete datasets for property relationships
2. **Classification**: Apply standard geotechnical classification systems
3. **Predictive Modeling**: Use case studies for validation of theoretical models
4. **Spatial Analysis**: Leverage geospatial data for regional assessments

### Citation
When using these datasets, please cite as:
```
Geotechnical Datasets for PhD Research: Failure Mechanisms of Underground Structures 
in Sandy and Clay Soils. Generated Dataset Collection. 2024.
```

## 🔧 Requirements

### Software Dependencies
```python
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
scipy >= 1.7.0
```

### Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## 📞 Support

For questions about the datasets or analysis methods:
- Review the analysis scripts in `/scripts/`
- Check the generated plots in `/documentation/`
- Examine the metadata files for detailed parameter descriptions

## 🔄 Updates and Versions

- **Version 1.0** (2024): Initial comprehensive dataset release
- Includes all major soil types and failure mechanisms
- Complete geospatial coverage for three major regions
- Validated analysis tools and visualization suite

---

*This dataset collection supports PhD research on geotechnical failure mechanisms and provides a comprehensive foundation for advanced studies in soil mechanics and underground structure design.*