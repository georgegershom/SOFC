# Geotechnical Datasets for Underground Structure Failure Analysis

This repository contains comprehensive geotechnical datasets generated and collected for PhD research on the failure mechanisms of underground structures in sandy and clay soils.

## Dataset Overview

### Synthetic Datasets

#### 1. Sandy Soil Properties (`sandy_soils/sandy_soil_properties.csv`)
- **Samples**: 1,000
- **Variables**: 25
- **Description**: Comprehensive sandy soil properties including grain size distribution, strength parameters, and liquefaction potential

**Key Variables**:
- `sand_content_pct`: Sand content percentage (80-100%)
- `D10_mm`, `D30_mm`, `D50_mm`, `D60_mm`: Grain size distribution parameters
- `Cu`, `Cc`: Uniformity and curvature coefficients
- `relative_density_pct`: Relative density (0-100%)
- `friction_angle_deg`: Friction angle in degrees
- `cohesion_kPa`: Cohesion in kPa
- `SPT_N60`: Standard Penetration Test N-value
- `CRR_7_5`: Cyclic Resistance Ratio for M7.5 earthquake
- `permeability_m_s`: Soil permeability

#### 2. Clay Soil Properties (`clay_soils/clay_soil_properties.csv`)
- **Samples**: 1,000
- **Variables**: 20
- **Description**: Comprehensive clay soil properties including Atterberg limits, mineralogy, and strength parameters

**Key Variables**:
- `liquid_limit_pct`: Liquid limit percentage
- `plastic_limit_pct`: Plastic limit percentage
- `plasticity_index`: Plasticity index
- `preconsolidation_stress_kPa`: Preconsolidation stress
- `OCR`: Overconsolidation ratio
- `undrained_shear_strength_kPa`: Undrained shear strength
- `sensitivity`: Soil sensitivity
- `smectite_pct`, `illite_pct`, `kaolinite_pct`, `chlorite_pct`: Clay mineral percentages

#### 3. Failure Case Studies (`case_studies/failure_case_studies.csv`)
- **Samples**: 100
- **Variables**: 15
- **Description**: Synthetic failure case studies including various failure mechanisms

**Key Variables**:
- `failure_type`: Type of failure (Liquefaction, Landslide, Settlement, etc.)
- `soil_type`: Soil type at failure location
- `failure_depth_m`: Depth of failure
- `earthquake_magnitude`: Earthquake magnitude if applicable
- `max_displacement_mm`: Maximum displacement
- `economic_loss_usd`: Economic impact

#### 4. Underground Structures (`case_studies/underground_structures.csv`)
- **Samples**: 200
- **Variables**: 12
- **Description**: Underground structure data for performance analysis

**Key Variables**:
- `structure_type`: Type of underground structure
- `depth_m`: Structure depth
- `surrounding_soil_type`: Surrounding soil conditions
- `max_crack_width_mm`: Maximum crack width
- `leakage_rate_L_per_day`: Leakage rate

### Real Datasets

#### 5. Earthquake Data (`real_data/usgs_earthquakes.csv`)
- **Source**: USGS Earthquake Database
- **Description**: Real earthquake data for liquefaction analysis
- **Variables**: 12

#### 6. Regional Soil Data (`real_data/regional_soil_data.csv`)
- **Source**: Synthesized from published regional studies
- **Description**: Regional soil data from various locations
- **Variables**: 12

#### 7. Liquefaction Cases (`real_data/liquefaction_cases.csv`)
- **Source**: Published case studies from major earthquakes
- **Description**: Real liquefaction case studies
- **Variables**: 18

#### 8. Landslide Cases (`real_data/landslide_cases.csv`)
- **Source**: International landslide database
- **Description**: Real landslide case studies
- **Variables**: 20

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Generate Datasets

```bash
python generate_datasets.py
```

### Download Real Data

```bash
python download_real_data.py
```

### Analyze Datasets

```bash
python analyze_datasets.py
```

## Data Quality and Validation

### Synthetic Data
- Generated using realistic statistical distributions based on published literature
- Correlations between variables follow established geotechnical relationships
- Ranges and distributions match typical field and laboratory measurements

### Real Data
- Sourced from reputable public databases and published case studies
- Quality indicators included where available
- Metadata provided for all datasets

## Applications

These datasets are suitable for:

1. **Machine Learning**: Training models for failure prediction
2. **Statistical Analysis**: Understanding relationships between soil properties
3. **Numerical Modeling**: Parameter estimation and validation
4. **Risk Assessment**: Developing failure probability models
5. **Research**: PhD thesis and academic publications

## File Structure

```
geotechnical_datasets/
├── sandy_soils/
│   └── sandy_soil_properties.csv
├── clay_soils/
│   └── clay_soil_properties.csv
├── case_studies/
│   ├── failure_case_studies.csv
│   └── underground_structures.csv
├── real_data/
│   ├── usgs_earthquakes.csv
│   ├── regional_soil_data.csv
│   ├── liquefaction_cases.csv
│   └── landslide_cases.csv
├── analysis/
│   ├── sandy_soils_statistics.csv
│   ├── clay_soils_statistics.csv
│   └── summary_report.md
├── visualizations/
│   ├── sandy_soils_analysis.png
│   ├── clay_soils_analysis.png
│   ├── failure_cases_analysis.png
│   ├── earthquake_analysis.png
│   └── interactive_dashboard.html
├── generate_datasets.py
├── download_real_data.py
├── analyze_datasets.py
├── requirements.txt
├── dataset_metadata.json
└── README.md
```

## Key Features

- **Comprehensive Coverage**: Both sandy and clay soil properties
- **Real-world Data**: Actual case studies and earthquake records
- **Statistical Rigor**: Realistic distributions and correlations
- **Multiple Formats**: CSV for analysis, HTML for interactive visualization
- **Documentation**: Detailed metadata and usage instructions
- **Reproducibility**: Fixed random seeds for consistent results

## Research Applications

### For Sandy Soils:
- Liquefaction potential assessment
- Strength parameter estimation
- Pore pressure analysis
- Seismic response modeling

### For Clay Soils:
- Consolidation analysis
- Mineralogical effects on behavior
- Sensitivity studies
- Preconsolidation stress evaluation

### For Failure Analysis:
- Case study validation
- Failure mechanism identification
- Economic impact assessment
- Risk factor analysis

## Citation

If you use these datasets in your research, please cite:

```
Geotechnical Datasets for Underground Structure Failure Analysis
PhD Research Dataset Collection
[Your Institution], [Year]
```

## Contact

For questions about the datasets or research applications, please contact [your email].

## License

This dataset is provided for academic research purposes. Please ensure proper attribution when using the data.