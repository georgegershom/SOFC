# FinTech Early Warning Model - Data Dictionary

## Dataset: Macroeconomic & Country-Level Data for Sub-Saharan Africa

### Overview
- **Dataset Name:** fintech_macroeconomic_synthetic
- **Time Period:** 2010-2023 (14 years)
- **Countries:** 49 Sub-Saharan Africa countries
- **Observations:** 686 (49 countries × 14 years)
- **Variables:** 14 total (3 metadata + 11 indicators)

---

## Variable Descriptions

### Metadata Variables

| Variable | Type | Description | Values |
|----------|------|-------------|---------|
| `Country` | String | 3-letter ISO country code | AGO, BEN, BWA, ..., ZWE |
| `Country_Name` | String | Full country name | Angola, Benin, Botswana, ... |
| `Year` | Integer | Calendar year | 2010, 2011, ..., 2023 |

### Economic Indicators

| Variable | Type | Description | Unit | Range | Source |
|----------|------|-------------|------|-------|---------|
| `GDP_Growth_Rate` | Float | Annual GDP growth rate | Percentage | -10 to 15 | Synthetic |
| `GDP_Growth_Volatility` | Float | 3-year rolling standard deviation of GDP growth | Percentage | 0 to 10 | Calculated |
| `Inflation_Rate_CPI` | Float | Consumer Price Index inflation rate | Percentage | -2 to 50 | Synthetic |
| `Unemployment_Rate` | Float | Unemployment as % of labor force | Percentage | 1 to 30 | Synthetic |
| `Exchange_Rate_Volatility` | Float | Exchange rate volatility measure | Percentage | 1 to 50 | Synthetic |
| `Central_Bank_Policy_Rate` | Float | Central bank policy interest rate | Percentage | 0 to 30 | Synthetic |
| `Broad_Money_Supply_M2_Growth` | Float | Annual growth in M2 money supply | Percentage | -5 to 50 | Synthetic |
| `Public_Debt_to_GDP_Ratio` | Float | Government debt as % of GDP | Percentage | 10 to 150 | Synthetic |

### Digital Infrastructure Indicators

| Variable | Type | Description | Unit | Range | Source |
|----------|------|-------------|------|-------|---------|
| `Mobile_Cellular_Subscriptions_per_100` | Float | Mobile subscriptions per 100 people | Per 100 people | 0 to 100 | Synthetic |
| `Internet_Users_Percent` | Float | Internet users as % of population | Percentage | 0 to 100 | Synthetic |
| `Secure_Internet_Servers` | Float | Number of secure internet servers | Count | 0 to 15 | Synthetic |

---

## Data Quality Metrics

### Completeness
- **Overall Missing Data:** 0.7% (only GDP_Growth_Volatility has missing values)
- **Country Coverage:** 100% (all 49 SSA countries)
- **Time Coverage:** 100% (2010-2023)

### Validity
- **Range Checks:** All variables within realistic bounds
- **Consistency:** Cross-variable relationships maintained
- **Temporal Consistency:** Smooth transitions over time

### Reliability
- **Synthetic Generation:** Based on real economic relationships
- **Country Characteristics:** Reflects actual development levels
- **Volatility Modeling:** Realistic risk patterns

---

## Key Relationships

### High Correlations (|r| > 0.7)
1. **Mobile Subscriptions ↔ Internet Users** (r = 0.705)
   - Digital infrastructure development correlation
   
2. **Internet Users ↔ Secure Servers** (r = 0.728)
   - Digital security infrastructure correlation

### Economic Relationships
- **GDP Growth ↔ Inflation:** Negative correlation (typical Phillips curve)
- **Unemployment ↔ GDP Growth:** Negative correlation (Okun's law)
- **Interest Rates ↔ Inflation:** Positive correlation (Taylor rule)
- **Debt-to-GDP ↔ Interest Rates:** Positive correlation (risk premium)

---

## Risk Indicators

### High Risk Thresholds
- **High Inflation:** > 10% annually
- **High Unemployment:** > 15% of labor force
- **High Debt:** > 80% of GDP
- **High Volatility:** GDP growth volatility > 5%

### Country Risk Profiles (2023)
- **High Inflation Risk:** 11 countries (Angola, CAR, Chad, Ethiopia, Gabon, Lesotho, Malawi, Mozambique, Namibia, Seychelles, Sudan)
- **High Unemployment Risk:** 2 countries (Burundi, Ethiopia)
- **High Debt Risk:** 5 countries (Angola, Ethiopia, Guinea, Madagascar, Seychelles)

---

## Usage Guidelines

### For FinTech Early Warning Models
1. **External Shock Modeling:** Use GDP growth, inflation, and exchange rate volatility
2. **Regulatory Environment:** Use institutional quality proxies (debt levels, policy rates)
3. **Digital Readiness:** Use mobile and internet penetration indicators
4. **Economic Stability:** Use unemployment and inflation rates

### For Academic Research
1. **Cross-country Analysis:** Full panel data structure
2. **Time Series Analysis:** 14-year time series per country
3. **Development Studies:** Country characteristics vary by development level
4. **Policy Analysis:** Multiple policy-relevant indicators

### For Policy Analysis
1. **Financial Stability:** Monitor high-risk countries
2. **Digital Inclusion:** Track digital infrastructure progress
3. **Economic Development:** Assess growth and stability patterns
4. **Regulatory Framework:** Use institutional indicators

---

## Data Sources and Methodology

### Synthetic Data Generation
- **Base Values:** Country-specific characteristics (development level, institutional quality)
- **Volatility:** Political stability and resource dependency factors
- **Trends:** Institutional quality and development level impacts
- **Cycles:** Economic cycle patterns (7-year cycles)
- **Shocks:** Random economic shocks with realistic distributions

### Validation
- **Range Validation:** All values within realistic bounds
- **Relationship Validation:** Economic relationships maintained
- **Temporal Validation:** Smooth time series transitions
- **Cross-country Validation:** Regional patterns preserved

---

## File Formats

### Available Formats
1. **CSV:** `fintech_macroeconomic_synthetic.csv` (main dataset)
2. **Excel:** `fintech_macroeconomic_synthetic.xlsx` (multiple sheets)
3. **JSON:** `fintech_macroeconomic_synthetic.json` (web applications)

### Additional Files
- **Summary Statistics:** `summary_statistics.csv`
- **Correlation Matrix:** `correlation_matrix.csv`
- **Country Rankings:** `country_rankings.xlsx`
- **Visualizations:** Multiple PNG files
- **Interactive Dashboard:** `interactive_dashboard.html`

---

## Citation and Usage

### Citation
```
FinTech Early Warning Model - Macroeconomic Data Collection
Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies
Generated Dataset, 2024
```

### License
Academic and research use. Validate findings against official sources for policy applications.

### Support
- Check generated log files for data generation details
- Review summary statistics for data quality assessment
- Examine correlation matrix for variable relationships
- Use interactive dashboard for exploratory analysis

---

*This data dictionary provides comprehensive information about the FinTech Early Warning Model dataset. For technical details about data generation methodology, refer to the source code files.*