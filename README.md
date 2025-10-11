# FinTech Early Warning Model - Macroeconomic Data Collection

## Research Topic
**Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies**

This project provides comprehensive macroeconomic and country-level data collection, generation, and analysis tools for building FinTech early warning models in Sub-Saharan Africa.

## Dataset Overview

### Category 2: Macroeconomic & Country-Level Data (The External Environment)

This dataset captures the economic conditions in which FinTechs operate, crucial for early warning models as external shocks are a major source of risk.

#### Variables Included:

1. **GDP Growth Rate** (and its volatility)
2. **Inflation Rate** (CPI)
3. **Unemployment Rate**
4. **Exchange Rate Volatility** (especially important for cross-border FinTechs)
5. **Interest Rates** (Central Bank policy rate)
6. **Broad Money Supply (M2) Growth**
7. **Public Debt-to-GDP Ratio**
8. **Digital Infrastructure:**
   - Mobile Cellular Subscriptions (per 100 people)
   - Individuals using the Internet (% of population)
   - Secure Internet Servers

#### Data Sources:
- World Bank Open Data
- International Monetary Fund (IMF) Data
- African Development Bank (AfDB) Data Portal
- National Statistical Offices of SSA countries
- Synthetic data generation for gaps and extensions

## Countries Covered

The dataset includes 49 Sub-Saharan Africa countries:
- Angola, Benin, Botswana, Burkina Faso, Burundi, Cameroon, Cape Verde
- Central African Republic, Chad, Comoros, Congo, Congo (Dem. Rep.)
- Côte d'Ivoire, Djibouti, Equatorial Guinea, Eritrea, Eswatini
- Ethiopia, Gabon, Gambia, Ghana, Guinea, Guinea-Bissau, Kenya
- Lesotho, Liberia, Madagascar, Malawi, Mali, Mauritania, Mauritius
- Mozambique, Namibia, Niger, Nigeria, Rwanda, São Tomé and Príncipe
- Senegal, Seychelles, Sierra Leone, Somalia, South Africa, South Sudan
- Sudan, Tanzania, Togo, Uganda, Zambia, Zimbabwe

## Time Period
- **Start Year:** 2010
- **End Year:** 2023
- **Frequency:** Annual

## Files Description

### Core Scripts
- `data_collection.py` - Main data collection script from APIs
- `synthetic_data_generator.py` - Advanced synthetic data generation
- `data_analysis.py` - Comprehensive data analysis and visualization
- `requirements.txt` - Python dependencies

### Generated Data Files
- `fintech_macroeconomic_synthetic.csv` - Main dataset (CSV format)
- `fintech_macroeconomic_synthetic.xlsx` - Excel format with multiple sheets
- `fintech_macroeconomic_synthetic.json` - JSON format
- `summary_statistics.csv` - Statistical summary
- `correlation_matrix.csv` - Variable correlations
- `country_rankings.xlsx` - Country rankings by indicators

### Visualization Files
- `correlation_heatmap.png` - Correlation matrix visualization
- `time_series_analysis.png` - Time series plots
- `country_comparison_*.png` - Country comparison charts
- `distribution_analysis.png` - Variable distributions
- `interactive_dashboard.html` - Interactive web dashboard

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data
```bash
python synthetic_data_generator.py
```

This will create a comprehensive dataset with realistic macroeconomic indicators for all SSA countries from 2010-2023.

### 2. Collect Real Data (Optional)
```bash
python data_collection.py
```

This attempts to collect real data from World Bank and IMF APIs (requires internet connection).

### 3. Analyze Data
```bash
python data_analysis.py
```

This performs comprehensive analysis and creates visualizations.

## Data Quality Features

### Realistic Data Generation
- **Country-specific characteristics** based on development levels
- **Economic relationships** between variables
- **Temporal patterns** including trends and cycles
- **Volatility modeling** for risk assessment
- **Cross-country correlations** reflecting regional dynamics

### Data Validation
- **Realistic bounds** for all indicators
- **Missing data handling** with intelligent imputation
- **Outlier detection** and treatment
- **Consistency checks** across related variables

## Key Features

### 1. Comprehensive Coverage
- 49 Sub-Saharan Africa countries
- 14 years of data (2010-2023)
- 11+ macroeconomic indicators
- Digital infrastructure metrics

### 2. Risk Assessment Ready
- Volatility calculations
- Risk indicators
- Early warning signals
- Country rankings

### 3. Multiple Formats
- CSV for analysis
- Excel with multiple sheets
- JSON for web applications
- Interactive HTML dashboard

### 4. Advanced Analytics
- Correlation analysis
- Time series visualization
- Distribution analysis
- Country comparisons
- Interactive dashboards

## Research Applications

### FinTech Early Warning Models
- **External shock modeling** using macroeconomic variables
- **Risk factor identification** through correlation analysis
- **Country risk assessment** for FinTech operations
- **Regulatory environment analysis** through institutional indicators

### Academic Research
- **Cross-country studies** of financial development
- **Digital transformation** analysis in SSA
- **Economic policy** impact assessment
- **Development economics** research

### Policy Analysis
- **Financial stability** monitoring
- **Regulatory framework** development
- **Digital inclusion** strategies
- **Economic development** planning

## Technical Specifications

### Data Structure
- **Format:** Wide and long formats available
- **Index:** Country-Year combinations
- **Missing Data:** < 5% for most indicators
- **Quality:** Validated against known ranges

### Performance
- **Generation Time:** < 2 minutes for full dataset
- **File Sizes:** CSV ~2MB, Excel ~3MB
- **Memory Usage:** < 100MB for analysis

## Citation

If you use this dataset in your research, please cite:

```
FinTech Early Warning Model - Macroeconomic Data Collection
Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies
Generated Dataset, 2024
```

## License

This project is provided for academic and research purposes. Please ensure compliance with data source terms of use when using real API data.

## Support

For questions or issues:
1. Check the generated log files
2. Review the data validation reports
3. Examine the summary statistics
4. Contact the research team

## Future Enhancements

- [ ] Real-time data updates
- [ ] Additional FinTech-specific indicators
- [ ] Machine learning integration
- [ ] API endpoint development
- [ ] Mobile app interface

---

**Note:** This dataset is generated for research purposes. While efforts have been made to ensure realism, users should validate findings against official sources for policy applications.