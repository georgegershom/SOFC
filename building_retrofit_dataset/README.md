# 🏢 Integrated Building Retrofit Dataset for AI/IoT Optimization

## 📊 Overview

This comprehensive dataset is designed for PhD research on **AI- and IoT-driven optimization of building retrofits**. It integrates real-time IoT sensor data, detailed building attributes, historical energy performance, and lifecycle assessment (LCA) information to enable advanced machine learning and optimization research.

### 🎯 Research Applications

- **Energy Consumption Forecasting**: Predict building energy usage patterns
- **Retrofit Optimization**: Identify optimal retrofit strategies for maximum energy savings
- **Carbon Footprint Reduction**: Analyze lifecycle environmental impacts
- **Smart Building Control**: Develop AI-driven building management systems
- **Portfolio-Level Analysis**: Optimize retrofit strategies across multiple buildings

## 📁 Dataset Structure

```
building_retrofit_dataset/
├── data/
│   ├── raw/                           # Raw generated data
│   │   ├── iot_sensors/               # Time-series IoT data
│   │   │   ├── energy_data.csv        # Energy consumption measurements
│   │   │   ├── ieq_data.csv           # Indoor environmental quality
│   │   │   ├── weather_data.csv       # Outdoor weather conditions
│   │   │   └── occupancy_data.csv     # Building occupancy patterns
│   │   ├── building_attributes/       # Static building properties
│   │   │   └── buildings.csv          # Comprehensive building characteristics
│   │   ├── energy_performance/        # Historical energy data
│   │   │   ├── historical_consumption.csv
│   │   │   ├── energy_ratings.csv
│   │   │   ├── retrofit_savings.csv
│   │   │   └── benchmarking.csv
│   │   ├── lca/                       # Lifecycle assessment data
│   │   │   ├── building_materials_lca.csv
│   │   │   ├── retrofit_materials_lca.csv
│   │   │   ├── lifecycle_impacts.csv
│   │   │   └── carbon_offset_potential.csv
│   │   └── integrated_building_data.csv  # Merged dataset for ML
│   └── processed/                     # Processed datasets for research
├── src/                               # Source code
│   ├── generators/                    # Data generation modules
│   ├── pipeline/                      # Data integration utilities
│   └── generate_dataset.py            # Main generation script
├── notebooks/                         # Analysis notebooks
│   └── explore_dataset.ipynb         # Exploratory data analysis
└── docs/                             # Documentation
```

## 🔧 Installation & Setup

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Required Packages

- `pandas>=2.1.3`
- `numpy>=1.24.3`
- `scikit-learn>=1.3.2`
- `matplotlib>=3.8.2`
- `seaborn>=0.13.0`
- `plotly>=5.18.0`
- `pyarrow>=14.0.1` (for Parquet support)

### Generate the Dataset

```bash
cd src
python generate_dataset.py
```

This will generate:
- 100 buildings with comprehensive attributes
- 350,000+ IoT sensor records
- 5 years of historical energy data
- Lifecycle assessment for 50 buildings
- Retrofit analysis for 30% of buildings

## 📊 Data Categories

### 1. IoT Sensor Data 🌡️

Real-time measurements at 15-60 minute intervals:

| Category | Parameters | Update Frequency |
|----------|------------|------------------|
| **Energy** | Total consumption, HVAC, lighting, equipment breakdown | 15-60 min |
| **IEQ** | Temperature, humidity, CO₂, TVOC, PM2.5 | 15-60 min |
| **Weather** | Outdoor temp, humidity, solar radiation, wind, precipitation | 15-60 min |
| **Occupancy** | Occupant count, occupancy rate | 15-60 min |

### 2. Building Attributes 🏗️

Comprehensive static building properties:

| Category | Features | Description |
|----------|----------|-------------|
| **Geometry** | Floor area, height, volume, shape factor | Physical dimensions |
| **Construction** | Year, materials, structural system | Building fabric |
| **Thermal** | U-values, R-values, thermal mass | Envelope performance |
| **Systems** | HVAC type, controls, renewable energy | Building services |
| **Quality** | Condition scores, maintenance needs | Current state |

### 3. Energy Performance ⚡

Historical and benchmarking data:

| Dataset | Content | Temporal Coverage |
|---------|---------|-------------------|
| **Historical Consumption** | Monthly energy use by fuel type | 5 years |
| **Energy Ratings** | EU A-G ratings, recommendations | Bi-annual assessments |
| **Retrofit Savings** | Measured savings post-retrofit | 2 years post-retrofit |
| **Benchmarking** | Peer comparisons, percentiles | Current snapshot |

### 4. Lifecycle Assessment (LCA) 🔄

Environmental impact data following EN 15804 standards:

| Phase | Data Included | Metrics |
|-------|---------------|---------|
| **A1-A3: Product** | Material production impacts | kg CO₂e, embodied energy |
| **A4-A5: Construction** | Transport, installation | kg CO₂e, waste |
| **B1-B7: Use** | Operational impacts | Annual emissions |
| **C1-C4: End of Life** | Demolition, disposal | kg CO₂e, recycling potential |
| **D: Beyond System** | Recycling benefits | Carbon credits |

## 🤖 Machine Learning Features

The integrated dataset includes 77+ features optimized for ML applications:

### Feature Categories

- **Continuous Variables**: Floor area, energy consumption, U-values, carbon emissions
- **Categorical Variables**: Building type, energy rating, construction period, materials
- **Time-Series Features**: Hourly/daily energy patterns, seasonal variations
- **Derived Metrics**: EUI, carbon intensity, retrofit potential scores

### Pre-Processing

The data integration pipeline (`src/pipeline/data_integration.py`) provides:

- Automatic feature engineering
- Missing value imputation
- Categorical encoding
- Time-series aggregation
- Cross-dataset joins

## 📈 Sample Analysis

### Quick Start with Jupyter

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load integrated dataset
buildings = pd.read_csv('data/raw/integrated_building_data.csv')

# Analyze energy performance
buildings.groupby('building_type')['avg_eui'].mean().plot(kind='bar')
plt.title('Average Energy Use Intensity by Building Type')
plt.ylabel('EUI (kWh/m²/year)')
plt.show()
```

### Explore the Dataset

Open the provided Jupyter notebook for comprehensive analysis:

```bash
jupyter notebook notebooks/explore_dataset.ipynb
```

## 🎓 Research Use Cases

### 1. Energy Forecasting

```python
from src.pipeline.data_integration import BuildingDataIntegrator

integrator = BuildingDataIntegrator()
time_series = integrator.create_time_series_dataset()
# Use for LSTM/ARIMA models
```

### 2. Retrofit Optimization

```python
retrofit_data = integrator.create_retrofit_recommendation_dataset()
# Apply multi-objective optimization algorithms
```

### 3. Carbon Impact Analysis

```python
lca_data = integrator.load_lca_data()
# Analyze embodied vs operational carbon trade-offs
```

## 📚 Data Dictionary

### Key Fields

| Field | Type | Description | Unit |
|-------|------|-------------|------|
| `building_id` | String | Unique building identifier | - |
| `gross_floor_area_m2` | Float | Total floor area | m² |
| `construction_year` | Integer | Year of construction | Year |
| `energy_rating` | String | EU energy rating | A-G |
| `total_consumption_kw` | Float | Total energy consumption | kW |
| `u_value_wall` | Float | Wall thermal transmittance | W/m²K |
| `retrofit_potential` | String | Retrofit opportunity level | high/medium/low |
| `net_lifecycle_carbon_kg_co2` | Float | Total lifecycle emissions | kg CO₂ |

Full data dictionary available in `data/processed/data_dictionary.json`

## 🔬 Validation & Quality

### Data Generation Methods

- **Stochastic Models**: Realistic variability using statistical distributions
- **Physical Constraints**: Energy balance, thermal physics principles
- **Temporal Patterns**: Daily, weekly, and seasonal variations
- **Building Standards**: Based on EU building codes and ASHRAE standards

### Quality Assurance

- Validated ranges for all physical parameters
- Consistent relationships between correlated variables
- No missing values in core fields
- Temporal continuity in time-series data

## 🚀 Advanced Features

### Custom Data Generation

Modify generation parameters in `src/generate_dataset.py`:

```python
generate_complete_dataset(
    n_buildings=500,        # Increase building count
    output_dir='custom/',   # Custom output location
    seed=123               # Different random seed
)
```

### Data Integration Pipeline

Use the provided pipeline for research-specific datasets:

```python
from src.pipeline.data_integration import BuildingDataIntegrator

integrator = BuildingDataIntegrator(data_dir='data/raw/')
ml_dataset = integrator.integrate_for_ml(include_iot=True)
integrator.export_for_research(output_path='data/research/')
```

## 📖 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{building_retrofit_dataset_2024,
  title={Integrated Building Retrofit Dataset for AI/IoT Optimization},
  author={[Your Name]},
  year={2024},
  note={Synthetic dataset for PhD research on building retrofit optimization}
}
```

## 🤝 Contributing

This dataset is designed for academic research. Contributions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your enhancements

## ⚖️ License

This dataset is provided for academic research purposes. Please check with your institution's guidelines for data usage and sharing.

## 📧 Contact

For questions or collaborations related to this dataset:

- **Research Area**: AI-driven Building Retrofit Optimization
- **Application Domain**: Smart Buildings, Energy Efficiency, Carbon Reduction

## 🎯 Future Enhancements

Planned additions to the dataset:

- [ ] Grid electricity carbon intensity time-series
- [ ] Detailed HVAC system specifications
- [ ] Social and behavioral factors
- [ ] Cost data for retrofit measures
- [ ] Building geometry (3D models)
- [ ] District-level energy networks
- [ ] Renewable energy generation profiles
- [ ] Demand response capabilities

---

**Note**: This is a synthetic dataset generated for research purposes. While based on realistic distributions and relationships, it should not be used for real-world building assessment without validation against actual building data.