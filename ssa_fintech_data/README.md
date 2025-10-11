# Sub-Saharan Africa FinTech Early Warning Model Dataset

## 🎯 Project Overview

This repository contains a comprehensive dataset designed for research on **"FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies"**. The dataset combines real macroeconomic data from international sources with synthetic FinTech-specific indicators to create a robust foundation for early warning model development.

## 📊 Dataset Highlights

- **Coverage**: 20 Sub-Saharan African countries
- **Time Period**: 2010-2023 (14 years)
- **Total Observations**: 280
- **Total Indicators**: 93
- **Data Sources**: World Bank, IMF, AfDB + Synthetic FinTech indicators

## 🌍 Countries Included

| Country | Code | FinTech Status |
|---------|------|----------------|
| Kenya | KE | 🏆 Leading mobile money market |
| Nigeria | NG | 🏆 Largest economy in SSA |
| South Africa | ZA | 🏆 Most developed financial sector |
| Ghana | GH | 🚀 Growing FinTech hub |
| Uganda | UG | 🚀 Mobile money pioneer |
| Tanzania | TZ | 🚀 Large mobile money market |
| Rwanda | RW | 🚀 Digital transformation leader |
| Senegal | SN | 🚀 West African FinTech hub |
| Côte d'Ivoire | CI | 📈 Emerging market |
| Zambia | ZM | 📈 Copper-dependent economy |
| Botswana | BW | 📈 Stable middle-income |
| Malawi | MW | 📈 Agriculture-based |
| Mozambique | MZ | 📈 Post-conflict recovery |
| Ethiopia | ET | 📈 Largest population |
| Zimbabwe | ZW | ⚠️ Economic challenges |
| Cameroon | CM | 📈 Central African hub |
| Burkina Faso | BF | 📈 Sahel region |
| Mali | ML | ⚠️ Security challenges |
| Benin | BJ | 📈 Small open economy |
| Togo | TG | 📈 Transit economy |

## 📁 Repository Structure

```
ssa_fintech_data/
├── README.md                           # This file
├── scripts/                            # Data collection and processing scripts
│   ├── ssa_macro_data_collector.py    # Main World Bank data collector
│   ├── simple_data_collector.py       # Simplified data collector
│   ├── data_enhancer.py               # Adds calculated indicators
│   ├── synthetic_data_generator.py    # Generates FinTech indicators
│   └── create_summary_report.py       # Creates final reports
├── processed_data/                     # Final datasets and reports
│   ├── ssa_comprehensive_dataset.csv  # 🎯 MAIN DATASET (CSV)
│   ├── ssa_comprehensive_dataset.xlsx # 🎯 MAIN DATASET (Excel)
│   ├── ssa_macro_data_simple.csv      # Basic World Bank data only
│   ├── ssa_macro_data_enhanced.csv    # Enhanced with calculations
│   ├── comprehensive_summary_report.json # Detailed statistics
│   ├── enhanced_data_report.json      # Data quality report
│   ├── crisis_scenarios.json          # Stress test scenarios
│   └── dataset_overview_charts.png    # Visualization summary
├── raw_data/                           # Raw data files (if any)
└── documentation/                      # Detailed documentation
    └── dataset_documentation.md       # Complete data dictionary
```

## 🚀 Quick Start

### 1. Main Dataset Files
- **Primary**: `processed_data/ssa_comprehensive_dataset.csv` - Complete dataset with all indicators
- **Excel**: `processed_data/ssa_comprehensive_dataset.xlsx` - Same data in Excel format
- **Basic**: `processed_data/ssa_macro_data_simple.csv` - World Bank indicators only

### 2. Key Variables for FinTech Early Warning Models

#### 🔴 **Risk Indicators** (Dependent Variables)
- `gdp_growth_risk` - GDP growth risk categories
- `inflation_risk` - Inflation risk categories  
- `economic_instability` - Composite economic instability index
- `external_vulnerability` - External shock vulnerability

#### 📈 **Economic Indicators** (Independent Variables)
- `gdp_growth` + `gdp_growth_volatility` - Economic growth and stability
- `inflation` + `inflation_volatility` - Price stability
- `unemployment` - Labor market conditions
- `exchange_rate_volatility` - Currency stability
- `debt_gdp` - Fiscal sustainability

#### 💻 **Digital Infrastructure**
- `internet_users` - Internet penetration
- `mobile_subs` - Mobile phone penetration
- `digital_infrastructure` - Composite digital readiness
- `digital_divide` - Gap between mobile and internet access

#### 🏦 **FinTech-Specific Indicators** (Synthetic)
- `fintech_adoption_rate` - FinTech service usage
- `mobile_money_penetration` - Mobile money accounts
- `digital_payment_volume_gdp` - Digital payment volume
- `cybersecurity_incidents_per_100k` - Cyber risk
- `fintech_regulatory_score` - Regulatory environment

## 📊 Data Quality Overview

| Category | Completeness | Examples |
|----------|-------------|----------|
| **Excellent (>95%)** | ✅ | GDP growth, unemployment, mobile subs, internet users |
| **Good (90-95%)** | ✅ | Inflation, FDI, financial depth, money supply |
| **Fair (70-90%)** | ⚠️ | Trade openness, interest rates |
| **Synthetic (100%)** | 🔮 | All FinTech-specific indicators |

## 🔧 Usage Examples

### Loading the Data
```python
import pandas as pd

# Load main dataset
df = pd.read_csv('processed_data/ssa_comprehensive_dataset.csv')

# Filter for specific countries
kenya_data = df[df['country_name'] == 'Kenya']

# Get latest year data
latest_year = df[df['year'] == df['year'].max()]
```

### Early Warning Model Example
```python
# Prepare features for early warning model
features = [
    'gdp_growth_volatility', 'inflation_volatility', 
    'exchange_rate_volatility', 'unemployment',
    'digital_infrastructure', 'fintech_adoption_rate',
    'cybersecurity_incidents_per_100k'
]

# Create binary crisis indicator
df['crisis'] = (df['gdp_growth'] < 0) | (df['inflation'] > 15)

# Model ready data
X = df[features].fillna(df[features].mean())
y = df['crisis']
```

## 📈 Key Insights from the Dataset

### 🏆 **Top Performers (2023)**
- **GDP Growth**: Rwanda (8.2%), Senegal (4.1%), Kenya (5.6%)
- **FinTech Adoption**: Botswana (38.3%), South Africa (35.2%), Kenya (32.4%)
- **Digital Infrastructure**: South Africa (777.2), Botswana (62.7), Ghana (37.4)

### ⚠️ **Risk Indicators**
- **Highest Economic Instability**: Ghana, Rwanda, Malawi
- **Cybersecurity Concerns**: Varies by digital adoption level
- **Financial Exclusion**: Still significant in rural economies

## 🔮 Synthetic Data Methodology

FinTech-specific indicators were generated using:
- **Statistical Distributions**: Beta, log-normal, normal distributions
- **Economic Correlations**: Based on real economic conditions
- **Regional Benchmarks**: Validated against known patterns
- **Realistic Constraints**: Bounded by feasible ranges

## 📚 Research Applications

### 1. **Early Warning Models**
- Binary classification (crisis/no crisis)
- Multi-class risk categorization
- Time series forecasting
- Panel data analysis

### 2. **FinTech Risk Analysis**
- Digital adoption patterns
- Cybersecurity risk assessment
- Regulatory impact analysis
- Financial inclusion gaps

### 3. **Policy Research**
- Digital divide analysis
- Infrastructure investment priorities
- Regulatory framework effectiveness
- Cross-border payment efficiency

## 🔄 Reproducibility

### Re-running Data Collection
```bash
# Install requirements
pip install pandas numpy requests wbdata matplotlib seaborn openpyxl scipy

# Run data collection pipeline
python scripts/simple_data_collector.py
python scripts/data_enhancer.py
python scripts/synthetic_data_generator.py
python scripts/create_summary_report.py
```

### Updating the Dataset
- Modify country list in scripts
- Adjust time periods
- Add new indicators
- Update synthetic data models

## 📋 Citation

When using this dataset, please cite:

```bibtex
@dataset{ssa_fintech_dataset_2025,
  title={Sub-Saharan Africa FinTech Early Warning Model Dataset},
  author={Research Team},
  year={2025},
  note={Comprehensive macroeconomic and FinTech indicators for Sub-Saharan African countries, 2010-2023},
  url={https://github.com/your-repo/ssa-fintech-dataset}
}
```

## ⚠️ Important Notes

### **Real Data Limitations**
- Some indicators have reporting lags
- Methodological changes over time
- Limited coverage for certain countries/periods

### **Synthetic Data Limitations**
- Based on statistical models, not actual observations
- May not capture all real-world relationships
- Should be validated against actual FinTech data when available

## 🤝 Contributing

We welcome contributions to improve the dataset:
- Additional data sources
- Better synthetic data models
- Enhanced documentation
- Bug fixes and improvements

## 📞 Contact

For questions about the dataset or research collaboration:
- **Research Topic**: FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies
- **Dataset Version**: 1.0
- **Last Updated**: October 11, 2025

---

## 🎉 Dataset Summary

✅ **280 observations** across **20 countries** and **14 years**  
✅ **93 indicators** covering macroeconomic, digital, and FinTech variables  
✅ **Real World Bank data** + **Synthetic FinTech indicators**  
✅ **Risk categorizations** and **regional comparisons**  
✅ **Crisis scenarios** for stress testing  
✅ **Comprehensive documentation** and **visualizations**  

**Ready for immediate use in FinTech early warning model research! 🚀**