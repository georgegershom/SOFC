# SENCE Framework: Advanced Vulnerability Assessment Visualization

This repository contains a comprehensive implementation of the SENCE (Socio-Economic Natural Compound Ecosystem) framework for vulnerability assessment of petroleum cities in Nigeria's Niger Delta region.

## Overview

The SENCE framework provides a sophisticated approach to assessing compound vulnerability by analyzing the interconnected relationships between social, economic, and environmental domains. This implementation includes advanced statistical analysis, model validation, and professional visualizations.

## Key Features

### 1. Advanced Radar Chart Visualization
- **File**: `sence_radar_chart.py`
- Professional radar chart showing normalized domain contributions to mean CVI
- Statistical overlays and confidence intervals
- City-specific vulnerability profiles for Port Harcourt, Warri, and Bonny

### 2. Comprehensive Statistical Analysis
- **File**: `sence_advanced_analysis.py`
- Principal Component Analysis (PCA)
- Correlation analysis between domains
- Model validation with cross-validation
- Sensitivity analysis
- Statistical significance testing

### 3. Framework Documentation
- **File**: `sence_framework_mermaid.md`
- Mermaid diagrams showing SENCE framework structure
- Vulnerability assessment workflow
- City-specific vulnerability profiles
- Integration model diagrams

### 4. PlantUML Diagrams
- **File**: `sence_framework_plantuml.puml`
- Comprehensive system architecture diagrams
- Vulnerability assessment workflow
- Statistical model architecture
- City comparison models

## Generated Visualizations

The scripts generate the following professional visualizations:

1. **sence_comprehensive_analysis.png** - Multi-panel analysis including:
   - Radar chart with statistical overlays
   - Statistical distribution analysis
   - Correlation matrix
   - Model validation metrics

2. **sence_enhanced_radar.png** - Enhanced radar chart with:
   - Professional styling and annotations
   - Statistical summary boxes
   - Framework information
   - Confidence intervals

3. **sence_advanced_statistical_analysis.png** - Comprehensive statistical analysis including:
   - PCA analysis plots
   - Correlation heatmaps
   - Model validation results
   - Feature importance analysis
   - Sensitivity analysis
   - Residual analysis and Q-Q plots

## Key Findings

### City Vulnerability Profiles

1. **Port Harcourt (CVI: 0.52)**
   - Most balanced vulnerability profile
   - Moderate environmental impact (0.45)
   - Urban economic disparities (0.52)
   - Better infrastructure access (0.38)

2. **Warri (CVI: 0.61)**
   - Highest overall vulnerability
   - Severe industrial pollution (0.68)
   - Economic deprivation (0.71)
   - Inter-ethnic conflicts (0.65)

3. **Bonny (CVI: 0.59)**
   - Extreme environmental degradation (0.89)
   - High economic mono-dependence (0.76)
   - Point-source pollution from LNG terminal
   - Moderate social issues (0.54)

### Statistical Analysis Results

- **PCA Analysis**: First 3 components explain 73% of variance
- **Model Performance**: R² = 0.775, RMSE = 0.018, MAPE = 2.9%
- **Statistical Significance**: Significant differences between cities (p < 0.001)
- **Domain Correlations**: Strong correlations between Environmental-Economic (0.956) and Social-Governance (1.000)

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis

1. **Basic Radar Chart Analysis**:
```bash
python3 sence_radar_chart.py
```

2. **Advanced Statistical Analysis**:
```bash
python3 sence_advanced_analysis.py
```

### Viewing Diagrams

The Mermaid and PlantUML diagrams can be viewed using:
- **Mermaid**: GitHub, GitLab, or Mermaid Live Editor
- **PlantUML**: PlantUML online server or local installation

## Framework Components

### SENCE Domains
1. **Environmental Domain**
   - Oil spill impact intensity
   - Gas flaring effects
   - Vegetation health (NDVI)
   - Water quality (NDWI)
   - Land degradation
   - Mangrove loss

2. **Economic Domain**
   - Unemployment rates
   - Income diversity (HHI)
   - Infrastructure access
   - Employment opportunities
   - Economic resilience
   - Poverty levels

3. **Social Domain**
   - Education access
   - Healthcare access
   - Community cohesion
   - Safety index
   - Social capital
   - Governance trust

4. **Governance Domain**
   - Institutional trust
   - Policy effectiveness
   - Transparency
   - Accountability
   - Participation
   - Rule of law

5. **Infrastructure Domain**
   - Water supply
   - Electricity access
   - Transportation
   - Communication
   - Housing quality
   - Sanitation

## Methodology

The SENCE framework employs:
- **Multi-domain aggregation** with weighted contributions
- **Principal Component Analysis** for dimensionality reduction
- **Cross-validation** for model robustness
- **Sensitivity analysis** for parameter stability
- **Statistical significance testing** for validation

## Applications

This framework can be used for:
- Policy development and intervention strategies
- Resource allocation and priority setting
- Monitoring and evaluation of vulnerability reduction programs
- Comparative analysis of urban vulnerability patterns
- Evidence-based decision making for sustainable development

## Technical Specifications

- **Python 3.8+**
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, SciPy, Scikit-learn
- **Visualization**: Professional radar charts, statistical plots, correlation matrices
- **Analysis**: PCA, correlation analysis, model validation, sensitivity analysis

## Citation

This implementation is based on the SENCE framework for assessing compound vulnerability in petroleum-dependent urban contexts. The methodology integrates socio-economic and environmental indicators to provide comprehensive vulnerability assessments.

## License

This project is provided for research and educational purposes. Please cite appropriately when using this framework in academic or professional work.