# Quick Start Guide - Geotechnical Datasets

## 📋 What's Included

This comprehensive dataset collection includes everything needed for PhD research on underground structure failure mechanisms:

### 1. **Sandy Soils Data** (`/sandy_soils/`)
- **Basic Properties**: Grain size distribution, density, moisture content (20 samples)
- **Mechanical Properties**: Friction angles, elastic moduli, permeability (20 samples)  
- **Liquefaction Data**: CSR/CRR ratios, safety factors, pore pressure (20 samples)

### 2. **Clay Soils Data** (`/clay_soils/`)
- **Basic Properties**: Atterberg limits, OCR, water content (20 samples)
- **Mechanical Properties**: Shear strength, consolidation parameters (20 samples)
- **Mineralogy**: Clay minerals composition, CEC, swelling potential (20 samples)
- **Slip Surface**: Slope stability, failure mechanisms (20 samples)

### 3. **Case Studies** (`/case_studies/`)
- **Failure Events**: 20 real-world underground structure failures with causes
- **Monitoring Data**: Time-series structural response measurements

### 4. **Spatial Data** (`/spatial_data/`)
- **Regional Properties**: GeoJSON with 11 geographic features
- **Grid Data**: 20 grid points with spatial soil properties

### 5. **Analysis Tools** (`/analysis_tools/`)
- **Data Loader**: Load and preprocess all datasets
- **Visualization**: Generate plots and charts
- **Statistical Analysis**: ML models, PCA, clustering

## 🚀 Getting Started

### Basic Usage (No Dependencies)
```bash
# View dataset summary
python3 dataset_summary.py
```

### Full Analysis (With Dependencies)
```bash
# Install required packages
pip install -r requirements.txt

# Run comprehensive analysis
python3 run_analysis.py
```

## 📊 Key Parameters Included

### For Sandy Soils
- Grain size (D10, D30, D50, D60)
- Friction angle (peak, residual)
- Liquefaction potential (CSR, CRR, N1(60))
- Relative density & void ratio

### For Clay Soils
- Liquid limit & plasticity index
- Undrained shear strength
- Sensitivity & OCR
- Clay mineralogy (smectite, illite, kaolinite)

## 🎯 Research Applications

1. **Numerical Modeling**: Use mechanical properties for FEM/DEM simulations
2. **Risk Assessment**: Analyze failure cases and triggering factors
3. **Machine Learning**: Predict failure modes and safety factors
4. **Spatial Analysis**: Regional hazard mapping
5. **Time-Series Analysis**: Monitor structural response evolution

## 📈 Sample Analyses Available

The analysis tools provide:
- Grain size distribution curves
- Plasticity charts (Casagrande)
- Liquefaction assessment plots
- Failure mode distributions
- Correlation matrices
- PCA and clustering results
- Regression models for prediction

## 🌍 Global Coverage

Data represents conditions from:
- Asia: Heihe Basin, Tokyo Bay, Shanghai, Bangkok
- North America: San Francisco, Boston, Mexico City
- Europe: London, Spain (Diezma), Venice, Amsterdam
- And 10+ other locations worldwide

## 📝 Data Quality

- Realistic parameter ranges based on literature
- Proper correlations between related parameters
- Includes uncertainty and measurement noise
- Suitable for statistical analysis and ML

## 🔗 Next Steps

1. Explore the `README.md` for detailed documentation
2. Check `documentation/data_dictionary.md` for parameter definitions
3. Use Python scripts in `analysis_tools/` for analysis
4. Visualize relationships between parameters
5. Build predictive models for your specific research needs

## 💡 Tips

- Start with `data_loader.py` to understand data structure
- Use `visualization.py` for quick insights
- Apply `statistical_analysis.py` for hypothesis testing
- Combine datasets for comprehensive analysis
- Export results for further processing in other tools

---
Generated for PhD Research on Underground Structure Failure Mechanisms
Version 1.0 | October 2024