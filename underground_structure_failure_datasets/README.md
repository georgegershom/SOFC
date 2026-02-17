# Underground Structure Failure Mechanism Dataset
## Sandy and Clay Soils Analysis

This comprehensive dataset collection provides synthetic and realistic data for analyzing failure mechanisms of underground structures in sandy and clay soils. The dataset is designed for research, machine learning, and geotechnical engineering applications.

---

## Dataset Overview

This collection contains **11 CSV datasets** and **10 visualization figures** covering various aspects of underground structure failure analysis.

### Dataset Categories

1. **Physical Model Data**
2. **Field Monitoring Data**
3. **Numerical Simulation Data**
4. **Geohazard Susceptibility Data**
5. **Environmental/Meteorological Data**
6. **Soil Property Data**
7. **Structural Monitoring Data**
8. **Experimental Physical Modeling Data**
9. **Synthetic FEM Parametric Data**
10. **Summary Statistics**

---

## Detailed Dataset Descriptions

### 1. Pipeline Leakage Sinkholes (`01_pipeline_leakage_sinkholes.csv`)
**Purpose**: Physical model data for manmade sinkholes caused by pipeline leakage  
**Samples**: 500  
**Soil Type**: Sandy clay with various strata

**Key Features**:
- `Time_hours`: Time elapsed since leakage start (0-240 hours)
- `Flow_Rate_L_per_min`: Water flow rate (5-50 L/min)
- `Settlement_mm`: Ground settlement measurement
- `Sinkhole_Diameter_m`: Diameter of formed sinkhole (0.5-5.0 m)
- `Moisture_Content_percent`: Soil moisture content (15-35%)
- `Pipeline_Depth_m`: Depth of buried pipeline (1.5-6.0 m)
- `Cavity_Volume_m3`: Volume of subsurface cavity (0.1-15.0 m³)
- `Failure_Occurred`: Binary indicator (0=No, 1=Yes)

**Applications**: Sinkhole prediction, pipeline risk assessment, urban geohazard analysis

---

### 2. Instrumented Embankment (`02_instrumented_embankment.csv`)
**Purpose**: Field monitoring data from instrumented embankment on clay formation  
**Samples**: 400  
**Soil Type**: Clay (Mudstone Formation, London Clay, Keuper Marl)

**Key Features**:
- `Depth_m`: Measurement depth (0-30 m)
- `P_Wave_Velocity_m_per_s`: Primary wave velocity (300-1800 m/s)
- `S_Wave_Velocity_m_per_s`: Shear wave velocity (150-800 m/s)
- `Extensometer_Displacement_mm`: Vertical displacement (-50 to 150 mm)
- `Pressure_Cell_kPa`: Earth pressure measurement (0-500 kPa)
- `Loading_Stage`: Construction loading stage (1-5)
- `Undrained_Shear_Strength_kPa`: Clay shear strength (20-150 kPa)
- `Plasticity_Index`: Clay plasticity (15-55)
- `Failure_Mode`: None, Heave, Settlement, Lateral

**Applications**: Embankment stability analysis, ground improvement validation, geophysical characterization

---

### 3. Deep Excavation in London Clay (`03_deep_excavation_london_clay.csv`)
**Purpose**: Numerical simulation data for deep excavation analysis  
**Samples**: 600  
**Soil Type**: London Clay

**Key Features**:
- `Excavation_Depth_m`: Depth of excavation (5-30 m)
- `Excavation_Width_m`: Width of excavation (10-50 m)
- `Wall_Thickness_m`: Retaining wall thickness (0.3-1.5 m)
- `Prop_Spacing_m`: Support prop spacing (2-8 m)
- `Horizontal_Displacement_mm`: Wall lateral movement (0-150 mm)
- `Max_Bending_Moment_kNm`: Maximum moment in wall (50-2000 kNm)
- `Basal_Heave_mm`: Heave at excavation base (-10 to 80 mm)
- `Safety_Factor`: Overall safety factor (0.8-3.0)
- `Failure_Risk`: Low, Medium, High, Critical

**Applications**: Deep excavation design, retaining wall optimization, urban construction planning

---

### 4. Geohazard Susceptibility (`04_geohazard_susceptibility.csv`)
**Purpose**: Geohazard and corroded asset failure data (BGS-inspired)  
**Samples**: 800  
**Soil Type**: Swelling clay, compressible ground, running sand, mixed

**Key Features**:
- `Latitude`, `Longitude`: Geographic coordinates (UK region)
- `Ground_Movement_Potential`: Very Low to Very High
- `Shrink_Swell_Index`: Volumetric change potential (0-5)
- `Compressibility_Index_Cc`: Consolidation parameter (0.05-0.5)
- `Permeability_m_per_s`: Hydraulic conductivity (10⁻⁹ to 10⁻⁴)
- `Asset_Age_years`: Infrastructure age (10-100 years)
- `Corrosion_Rate_mm_per_year`: Material degradation rate (0.01-0.5)
- `Asset_Type`: Pipeline, Foundation, Tunnel, Basement
- `Failure_Probability`: Risk score (0-1)

**Applications**: Infrastructure risk mapping, asset management, ground movement prediction

---

### 5. Meteorological Data (`05_meteorological_embankment.csv`)
**Purpose**: Environmental data for clay embankment failure prediction  
**Samples**: 1095 (3 years daily data)  
**Soil Type**: Clay

**Key Features**:
- `Date`: Daily timestamp (2020-2023)
- `Precipitation_mm`: Daily rainfall
- `Temperature_C`: Daily temperature (seasonal variation)
- `Evapotranspiration_mm`: Daily ET (0-5 mm)
- `Soil_Moisture_percent`: Soil moisture content (10-40%)
- `Pore_Water_Pressure_kPa`: Pore pressure measurement (-50 to 150 kPa)
- `Matric_Suction_kPa`: Soil suction (0-300 kPa)
- `Embankment_Settlement_mm`: Cumulative settlement
- `Alert_Level`: Green, Yellow, Orange, Red

**Applications**: Climate impact assessment, early warning systems, long-term monitoring

---

### 6a. Sandy Soil Properties (`06a_sandy_soil_properties.csv`)
**Purpose**: Constitutive properties of sandy soils  
**Samples**: 300

**Key Features**:
- `Relative_Density_percent`: Compaction state (30-95%)
- `D10_mm`, `D50_mm`, `D90_mm`: Grain size distribution
- `Uniformity_Coefficient_Cu`: Gradation parameter (1.5-15)
- `Internal_Friction_Angle_degrees`: Shear strength parameter (28-42°)
- `Dilation_Angle_degrees`: Volumetric strain parameter (0-15°)
- `Permeability_m_per_s`: Hydraulic conductivity (10⁻⁵ to 10⁻³)
- `Youngs_Modulus_MPa`: Elastic stiffness (10-80 MPa)
- `Classification`: SP, SW, SM, SC (USCS)

**Applications**: Material characterization, FEM input parameters, soil classification

---

### 6b. Clay Soil Properties (`06b_clay_soil_properties.csv`)
**Purpose**: Constitutive properties of clay soils  
**Samples**: 300

**Key Features**:
- `Liquid_Limit_percent`: Atterberg limit (30-90%)
- `Plastic_Limit_percent`: Atterberg limit (15-40%)
- `Plasticity_Index`: Liquid limit - Plastic limit (10-60)
- `Undrained_Shear_Strength_kPa`: Cohesion (15-200 kPa)
- `Effective_Friction_Angle_degrees`: Drained shear strength (18-32°)
- `Compression_Index_Cc`: Consolidation parameter (0.1-0.6)
- `Overconsolidation_Ratio`: Stress history (1.0-8.0)
- `Permeability_m_per_s`: Hydraulic conductivity (10⁻¹⁰ to 10⁻⁷)
- `Classification`: CL, CH, CI, CV (USCS)

**Applications**: Clay behavior modeling, consolidation analysis, foundation design

---

### 7. Tunnel Monitoring (`07_tunnel_monitoring.csv`)
**Purpose**: Structural monitoring data for tunnel infrastructure (Kaggle-inspired)  
**Samples**: 1000  
**Soil Type**: Clay, Sand, Mixed, Rock

**Key Features**:
- `Tunnel_Depth_m`: Depth below surface (5-50 m)
- `Tunnel_Diameter_m`: Internal diameter (3-12 m)
- `Cover_Depth_Ratio`: Cover to diameter ratio (1.0-8.0)
- `Vertical_Settlement_mm`: Crown settlement (-5 to 80 mm)
- `Horizontal_Displacement_mm`: Lateral movement (-20 to 50 mm)
- `Earth_Pressure_kPa`: Overburden pressure (50-800 kPa)
- `Construction_Method`: TBM, NATM, Cut-and-Cover, Pipe-Jacking
- `Service_Years`: Operational age (0-100 years)
- `Risk_Level`: Low, Medium, High, Critical
- `Failure_Mode`: None, Excessive Settlement, Lining Crack, Water Ingress, Collapse

**Applications**: Tunnel health monitoring, risk assessment, maintenance planning

---

### 8. Centrifuge Physical Modeling (`08_centrifuge_physical_modeling.csv`)
**Purpose**: Experimental data from centrifuge testing  
**Samples**: 250  
**Soil Type**: Dry sand, saturated sand, clay, silty clay

**Key Features**:
- `Centrifuge_g_Level`: Acceleration level (20-100 g)
- `Model_Scale`: Scaling factor (1:20 to 1:100)
- `Structure_Type`: Tunnel, Foundation, Retaining Wall, Pile
- `Embedment_Depth_mm`: Model embedment (50-300 mm)
- `Applied_Load_kN`: Loading magnitude (0-50 kN)
- `Settlement_mm`: Vertical displacement (0-30 mm)
- `Bearing_Capacity_kPa`: Ultimate capacity (50-500 kPa)
- `PIV_Max_Shear_Strain_percent`: Strain field measurement (0-15%)
- `Failure_Mechanism`: General Shear, Local Shear, Punching, Bearing, Piping

**Applications**: Physical model validation, mechanism visualization, benchmark testing

---

### 9. Synthetic FEM Parametric Study (`09_synthetic_fem_parametric.csv`)
**Purpose**: Comprehensive parametric study using finite element method  
**Samples**: 1500  
**Soil Type**: Sand, Clay, Mixed

**Key Features**:
- `Soil_Type`: Sand, Clay, Mixed
- `Structure_Type`: Tunnel, Basement, Foundation, Retaining Wall
- `Embedment_Depth_m`: Structure depth (2-30 m)
- `HD_Ratio`: Height/depth ratio (0.5-5.0)
- `Groundwater_Condition`: Dry, Saturated, Partially Saturated
- `Cohesion_kPa`: Soil cohesion (0-100 kPa)
- `Friction_Angle_degrees`: Soil friction (0-40°)
- `Max_Vertical_Displacement_mm`: Maximum settlement (-10 to 150 mm)
- `Max_Shear_Strain_percent`: Strain concentration (0-20%)
- `Min_Safety_Factor`: Minimum factor of safety (0.5-4.0)
- `Failure_Mode`: None, Heave, Piping, Structural Buckling, Bearing Failure, Excessive Settlement
- `Failure_Occurred`: Binary outcome (0=No, 1=Yes)

**Applications**: Machine learning training, parametric optimization, failure prediction

---

### 10. Summary Parameters (`10_summary_parameters.csv`)
**Purpose**: Reference table of all key variables and their sources  
**Samples**: 16 key parameters

**Categories**:
- Constitutive: E, ν, c, φ, γ
- Geometric: Wall thickness, tunnel diameter, embedment depth, excavation width
- Observational: Settlement, displacement, pore pressure, earth pressure
- Risk/Mode: Safety factor, failure probability, failure mode

**Applications**: Parameter reference, data dictionary, research planning

---

## Visualization Figures

All figures are high-resolution PNG files (300 DPI) located in the `figures/` directory:

1. **01_pipeline_leakage_analysis.png**: Time-settlement relationships, flow rate effects
2. **02_embankment_monitoring.png**: Wave velocity profiles, loading stage analysis
3. **03_deep_excavation_analysis.png**: Parametric effects on wall displacement
4. **04_geohazard_susceptibility.png**: Spatial distribution, asset risk mapping
5. **05_meteorological_analysis.png**: Climate impact on embankment performance
6. **06_soil_properties_comparison.png**: Sand vs Clay characteristics
7. **07_tunnel_monitoring.png**: Construction method effects, risk assessment
8. **08_centrifuge_modeling.png**: Physical modeling results, g-level effects
9. **09_fem_parametric_study.png**: Comprehensive parametric analysis
10. **10_comprehensive_summary_dashboard.png**: Integrated multi-dataset summary

---

## File Structure

```
underground_structure_failure_datasets/
├── README.md (this file)
├── csv_data/
│   ├── 01_pipeline_leakage_sinkholes.csv
│   ├── 02_instrumented_embankment.csv
│   ├── 03_deep_excavation_london_clay.csv
│   ├── 04_geohazard_susceptibility.csv
│   ├── 05_meteorological_embankment.csv
│   ├── 06a_sandy_soil_properties.csv
│   ├── 06b_clay_soil_properties.csv
│   ├── 07_tunnel_monitoring.csv
│   ├── 08_centrifuge_physical_modeling.csv
│   ├── 09_synthetic_fem_parametric.csv
│   └── 10_summary_parameters.csv
├── figures/
│   ├── 01_pipeline_leakage_analysis.png
│   ├── 02_embankment_monitoring.png
│   ├── 03_deep_excavation_analysis.png
│   ├── 04_geohazard_susceptibility.png
│   ├── 05_meteorological_analysis.png
│   ├── 06_soil_properties_comparison.png
│   ├── 07_tunnel_monitoring.png
│   ├── 08_centrifuge_modeling.png
│   ├── 09_fem_parametric_study.png
│   └── 10_comprehensive_summary_dashboard.png
└── underground_structure_failure_datasets.zip (CSV files)
```

---

## Usage Examples

### Python (Pandas)
```python
import pandas as pd

# Load any dataset
fem_data = pd.read_csv('csv_data/09_synthetic_fem_parametric.csv')

# Filter for clay soil failures
clay_failures = fem_data[(fem_data['Soil_Type'] == 'Clay') & 
                         (fem_data['Failure_Occurred'] == 1)]

# Analyze safety factors
mean_sf = fem_data.groupby('Soil_Type')['Min_Safety_Factor'].mean()
print(mean_sf)
```

### R
```r
# Load dataset
tunnel_data <- read.csv('csv_data/07_tunnel_monitoring.csv')

# Statistical analysis
summary(tunnel_data$Vertical_Settlement_mm)

# Regression model
model <- lm(Vertical_Settlement_mm ~ Tunnel_Depth_m + Soil_Type, 
            data=tunnel_data)
```

### Machine Learning
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load FEM data
df = pd.read_csv('csv_data/09_synthetic_fem_parametric.csv')

# Prepare features and target
features = ['Embedment_Depth_m', 'HD_Ratio', 'Cohesion_kPa', 
            'Friction_Angle_degrees', 'Youngs_Modulus_MPa']
X = df[features]
y = df['Failure_Occurred']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

---

## Data Generation Methodology

All datasets were synthetically generated using realistic parameter ranges derived from:
- Published geotechnical literature
- International building codes and standards
- Real-world case studies (NIH, BGS, Zenodo, University research)
- Typical FEM simulation outputs

**Random Seed**: 42 (for reproducibility)

**Distributions Used**:
- Uniform: Most geometric and material properties
- Exponential: Time-dependent settlement
- Gamma: Precipitation patterns
- Normal: Temperature variations
- Categorical: Soil types, failure modes, risk levels

---

## Research Applications

### 1. Machine Learning & AI
- Failure prediction models
- Classification of failure modes
- Risk assessment algorithms
- Feature importance analysis
- Neural network training

### 2. Geotechnical Engineering
- Parametric design studies
- Sensitivity analysis
- Probabilistic risk assessment
- Ground improvement optimization
- Monitoring system development

### 3. Data Science
- Exploratory data analysis
- Statistical modeling
- Time series analysis
- Spatial analysis
- Correlation studies

### 4. Education & Training
- Student projects
- Teaching datasets
- Case study development
- Software training
- Visualization exercises

---

## Citation

If you use this dataset in your research, please cite:

```
Underground Structure Failure Mechanism Dataset (2026)
Failure Mechanisms of Underground Structures in Sandy and Clay Soils
Synthetic Dataset Collection for Geotechnical Engineering Research
https://github.com/[your-repo]
```

---

## Technical Specifications

- **Total Datasets**: 11 CSV files
- **Total Samples**: 6,240+ data points
- **Total Visualizations**: 10 high-resolution figures
- **File Format**: CSV (UTF-8)
- **Missing Values**: None
- **Data Quality**: Cleaned and validated
- **Outliers**: Realistic ranges maintained

---

## Key Parameter Ranges Summary

| Parameter | Sandy Soil | Clay Soil | Unit |
|-----------|-----------|-----------|------|
| Young's Modulus (E) | 10-80 | 5-50 | MPa |
| Poisson's Ratio (ν) | 0.15-0.35 | 0.25-0.45 | - |
| Cohesion (c) | 0-5 | 15-200 | kPa |
| Friction Angle (φ) | 28-42 | 18-32 | degrees |
| Unit Weight (γ) | 16-20 | 17-22 | kN/m³ |
| Permeability (k) | 10⁻⁵-10⁻³ | 10⁻¹⁰-10⁻⁷ | m/s |
| Plasticity Index (PI) | N/A | 10-60 | % |
| Relative Density (Dr) | 30-95 | N/A | % |

---

## Software Requirements

### For CSV Analysis:
- Python 3.7+ with pandas, numpy
- R 4.0+ with tidyverse
- Excel, LibreOffice, or any spreadsheet software

### For Visualization Reproduction:
- Python 3.7+ with matplotlib, seaborn
- R 4.0+ with ggplot2

### For Machine Learning:
- scikit-learn, TensorFlow, PyTorch
- Caret (R), tidymodels (R)

---

## License

This dataset is provided for educational and research purposes. Please attribute appropriately when using in publications.

---

## Contact & Support

For questions, suggestions, or issues with the dataset:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

---

## Version History

- **v1.0** (February 2026): Initial release
  - 11 CSV datasets
  - 10 visualization figures
  - Comprehensive documentation

---

## Acknowledgments

This synthetic dataset was inspired by real-world data sources including:
- National Institutes of Health (NIH) / Mendeley Data
- British Geological Survey (BGS)
- University of Bath
- Imperial College London / Zenodo
- Newcastle University
- International geotechnical research community

---

**Last Updated**: February 17, 2026
