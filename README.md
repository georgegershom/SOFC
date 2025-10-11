# Sub-Saharan Africa Macroeconomic Data for FinTech Risk Modeling

## 📊 Dataset Overview

This comprehensive dataset has been compiled for research on **"FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies"**. It provides macroeconomic and country-level data for 48 Sub-Saharan African countries from 2010 to 2024.

### Quick Statistics
- **Countries Covered**: 48 Sub-Saharan African nations
- **Time Period**: 2010 - 2024 (15 years)
- **Total Records**: 720
- **Variables**: 23 (including derived risk indicators)
- **Data Sources**: World Bank API + Synthetic generation for missing values

## 📁 Files Generated

1. **`ssa_macroeconomic_data_full.csv`** - Complete dataset with all indicators
2. **`ssa_macroeconomic_data.xlsx`** - Excel workbook with multiple analytical sheets:
   - Full_Dataset: Complete data
   - Country_Summary: Statistical summaries by country
   - Recent_Data_2020_2024: Focus on recent years
   - High_Risk_Countries: Countries with elevated FinTech risk
3. **`ssa_macroeconomic_analysis.png`** - Comprehensive visualization dashboard
4. **`ssa_correlation_matrix.png`** - Correlation heatmap of key indicators
5. **`ssa_macroeconomic_summary_report.txt`** - Executive summary report

## 📋 Variables Included

### Category 2: Macroeconomic & Country-Level Data

#### Economic Indicators
- **GDP Growth Rate (%)** - Annual GDP growth with volatility measures
- **Inflation Rate (CPI) (%)** - Consumer Price Index inflation
- **Unemployment Rate (%)** - National unemployment levels
- **Exchange Rate Volatility** - Calculated from official exchange rates
- **Real Interest Rate (%)** - Central bank policy rates adjusted for inflation
- **Broad Money Supply (M2) (% of GDP)** - Money supply growth indicator
- **Central Government Debt (% of GDP)** - Public debt levels

#### Digital Infrastructure
- **Mobile Cellular Subscriptions** (per 100 people)
- **Individuals using the Internet** (% of population)
- **Secure Internet Servers** (per 1 million people)

#### Derived Risk Indicators
- **Digital Infrastructure Index** - Composite measure of digital readiness
- **Economic Stability Index** - Measure of economic stability
- **Financial Development Index** - Overall financial system development
- **FinTech Risk Score** - Comprehensive risk assessment (0-100 scale)
- **Risk Category** - Categorical risk classification (Low/Medium/High/Very High)

## 🌍 Countries Included

The dataset covers all 48 Sub-Saharan African countries:

**East Africa**: Ethiopia, Kenya, Tanzania, Uganda, Rwanda, Burundi, South Sudan, Sudan, Eritrea, Somalia, Comoros, Madagascar, Mauritius, Seychelles

**West Africa**: Nigeria, Ghana, Senegal, Mali, Burkina Faso, Niger, Benin, Togo, Côte d'Ivoire, Guinea, Guinea-Bissau, Liberia, Sierra Leone, Cape Verde, Gambia, Mauritania

**Central Africa**: Cameroon, Chad, Central African Republic, Congo (Rep.), Congo (DRC), Equatorial Guinea, Gabon, São Tomé and Príncipe

**Southern Africa**: South Africa, Angola, Botswana, Namibia, Zimbabwe, Zambia, Mozambique, Malawi, Lesotho, Eswatini

## 🔑 Key Findings

### Risk Distribution (2010-2024)
- **Low Risk**: 82.4% of observations
- **Medium Risk**: 14.9% of observations
- **High Risk**: 1.5% of observations
- **Very High Risk**: 1.2% of observations

### Top 5 Highest Risk Countries (2024)
1. Zimbabwe (Risk Score: 100.00)
2. Sierra Leone (Risk Score: 91.79)
3. South Sudan (Risk Score: 55.00)
4. Congo, DRC (Risk Score: 47.42)
5. Malawi (Risk Score: 44.20)

### Digital Infrastructure Leaders (2024)
1. Botswana (Index: 113.29)
2. Mauritius (Index: 97.61)
3. São Tomé and Príncipe (Index: 90.46)
4. Nigeria (Index: 88.55)
5. Mauritania (Index: 88.16)

## 🔧 How to Use This Dataset

### Python Example
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('ssa_macroeconomic_data_full.csv')

# Filter for high-risk countries in 2024
high_risk_2024 = df[(df['Year'] == 2024) & (df['Risk_Category'].isin(['High', 'Very High']))]

# Analyze specific country trends
kenya_data = df[df['Country_Name'] == 'Kenya'].sort_values('Year')

# Calculate average risk by country
avg_risk = df.groupby('Country_Name')['FinTech_Risk_Score'].mean().sort_values(ascending=False)
```

### Excel Analysis
Open `ssa_macroeconomic_data.xlsx` to access pre-formatted sheets with:
- Pivot tables for country comparisons
- Time series data for trend analysis
- Pre-filtered high-risk country data

## 📊 Data Collection Methodology

1. **Primary Data Source**: World Bank Open Data API
   - Real-time data fetching for available indicators
   - Coverage varies by country and indicator

2. **Synthetic Data Generation**: For missing values, realistic synthetic data was generated using:
   - Historical economic patterns
   - Regional trends and correlations
   - Economic theory-based constraints
   - Random shocks to simulate economic crises

3. **Risk Score Calculation**: Weighted composite of:
   - Exchange rate volatility (20%)
   - GDP growth volatility (15%)
   - Inflation rate (15%)
   - Unemployment rate (10%)
   - Government debt levels (10%)
   - Digital infrastructure gaps (15%)
   - Economic instability (15%)

## ⚠️ Data Limitations

- Some data points are synthetically generated where official data was unavailable
- Exchange rate volatility calculations require at least 3 years of data
- Digital infrastructure metrics may lag actual adoption rates
- Risk scores are calculated using a standardized model that may not capture all country-specific factors

## 📚 Recommended Use Cases

1. **FinTech Risk Modeling**: Build early warning systems for FinTech operations
2. **Market Entry Analysis**: Assess market readiness for FinTech expansion
3. **Regulatory Planning**: Understand economic contexts for policy development
4. **Academic Research**: Study relationships between economic indicators and FinTech development
5. **Investment Decision Making**: Evaluate country-level risks for FinTech investments

## 🛠️ Running the Scripts

### Prerequisites
```bash
pip install pandas numpy requests openpyxl xlsxwriter tqdm matplotlib seaborn
```

### Generate Fresh Data
```bash
python3 ssa_macroeconomic_data_collector.py
```

### Create Visualizations
```bash
python3 visualize_macroeconomic_data.py
```

## 📈 Visualization Highlights

The analysis includes:
- Risk distribution pie charts
- GDP growth trends with confidence intervals
- Digital infrastructure evolution over time
- Correlation matrices between key indicators
- Exchange rate volatility distributions
- Digital infrastructure vs. risk scatter plots

## 🤝 Contributing

For updates or corrections to this dataset:
1. Report issues with specific country data
2. Suggest additional indicators relevant to FinTech risk
3. Provide access to alternative data sources

## 📄 License

This dataset is compiled from publicly available sources and synthetic generation for research purposes.

## 📞 Contact

For questions about this dataset or the FinTech Early Warning Model research, please refer to the research documentation.

---

*Dataset generated on: 2025-10-11*
*Version: 1.0*