# Data Dictionary - SSA Macroeconomic Dataset

## Variable Definitions and Descriptions

| Variable Name | Type | Unit | Description | Source |
|--------------|------|------|-------------|--------|
| **Country_Code** | String | ISO3 | Three-letter country code (e.g., KEN for Kenya) | ISO Standard |
| **Country_Name** | String | Text | Full country name | World Bank |
| **Year** | Integer | Year | Calendar year (2010-2024) | - |
| **GDP Growth Rate (%)** | Float | Percentage | Annual percentage growth rate of GDP at market prices | World Bank/Synthetic |
| **GDP_Growth_Volatility** | Float | Percentage | Standard deviation of GDP growth over 3-year rolling window | Calculated |
| **Inflation Rate (CPI) (%)** | Float | Percentage | Annual consumer price index inflation rate | World Bank/Synthetic |
| **Unemployment Rate (%)** | Float | Percentage | Unemployment as % of total labor force | World Bank/Synthetic |
| **Official Exchange Rate (LCU per US$)** | Float | Local Currency Units | Official exchange rate (annual average) | World Bank/Synthetic |
| **Exchange_Rate_Volatility** | Float | Percentage | Volatility of exchange rate changes (3-year window) | Calculated |
| **Real Interest Rate (%)** | Float | Percentage | Lending interest rate adjusted for inflation | World Bank/Synthetic |
| **Broad Money (% of GDP)** | Float | Percentage | M2 money supply as percentage of GDP | World Bank/Synthetic |
| **M2_Growth_Rate** | Float | Percentage | Year-over-year growth rate of broad money | Calculated |
| **Central Government Debt (% of GDP)** | Float | Percentage | Gross government debt as % of GDP | World Bank/Synthetic |
| **Mobile Cellular Subscriptions (per 100 people)** | Float | Per 100 | Mobile phone subscriptions per 100 population | World Bank/Synthetic |
| **Individuals using the Internet (% of population)** | Float | Percentage | Internet users as % of total population | World Bank/Synthetic |
| **Secure Internet Servers (per 1 million people)** | Float | Per million | Number of secure internet servers per million people | World Bank/Synthetic |
| **Digital_Infrastructure_Index** | Float | Index (0-100+) | Composite index of digital readiness | Calculated |
| **Economic_Stability_Index** | Float | Index (0-100) | Composite measure of economic stability | Calculated |
| **Financial_Development_Index** | Float | Index (0-100) | Overall financial system development score | Calculated |
| **FinTech_Risk_Score** | Float | Score (0-100) | Comprehensive FinTech operational risk score | Calculated |
| **Risk_Category** | String | Category | Risk classification (Low/Medium/High/Very High) | Calculated |
| **Data_Source** | String | Text | Indicates data origin (World Bank API + Synthetic) | Metadata |
| **Data_Collection_Date** | String | Date | Date when data was collected/generated | Metadata |

## Calculated Indices Formulas

### Digital Infrastructure Index
```
Digital_Infrastructure_Index = 
    (Mobile_Subscriptions × 0.4) + 
    (Internet_Users × 0.4) + 
    (Secure_Servers × 0.2)
```

### Economic Stability Index
```
Economic_Stability_Index = 100 - [
    (GDP_Volatility × 2) + 
    |Inflation_Rate| + 
    (Exchange_Volatility × 0.5)
]
```

### Financial Development Index
```
Financial_Development_Index = 
    (Broad_Money × 0.5) + 
    ((100 - Gov_Debt) × 0.3) + 
    (Digital_Infrastructure × 0.2)
```

### FinTech Risk Score
```
FinTech_Risk_Score = 
    (Exchange_Volatility × 0.20) +
    (GDP_Volatility × 0.15) +
    |Inflation_Rate| × 0.15) +
    (Unemployment × 0.10) +
    (Gov_Debt × 0.10) +
    ((100 - Digital_Infrastructure) × 0.15) +
    ((100 - Economic_Stability) × 0.15)
```

## Risk Categories Thresholds

| Category | Risk Score Range | Description |
|----------|-----------------|-------------|
| **Low** | 0-25 | Stable environment, low operational risk |
| **Medium** | 26-50 | Moderate risk, requires monitoring |
| **High** | 51-75 | Elevated risk, mitigation strategies needed |
| **Very High** | 76-100 | Critical risk level, intensive monitoring required |

## Data Quality Indicators

### Synthetic Data Generation Parameters

Each synthetic indicator follows specific distributions:

| Indicator | Mean | Std Dev | Min | Max | Annual Trend |
|-----------|------|---------|-----|-----|--------------|
| GDP Growth | 4.5% | 3.0% | -5% | 15% | +0.1% |
| Inflation | 6.0% | 4.0% | -2% | 25% | -0.05% |
| Unemployment | 8.0% | 3.0% | 2% | 25% | +0.05% |
| Exchange Rate | 100 | 50 | 0.5 | 5000 | +5 |
| Interest Rate | 5.0% | 3.0% | -10% | 20% | -0.1% |
| Broad Money | 35% | 15% | 10% | 100% | +0.5% |
| Gov Debt | 45% | 20% | 10% | 150% | +1.0% |
| Mobile Subs | 80 | 30 | 10 | 150 | +3 |
| Internet Users | 30% | 20% | 1% | 85% | +2.5% |
| Secure Servers | 50 | 100 | 0.1 | 1000 | +10 |

### Economic Shocks Simulation
- 10% probability of economic shock per year
- Shock magnitude: ±2 standard deviations from mean
- Ensures realistic economic crisis representation

## Missing Data Treatment

1. **Primary Approach**: Fetch from World Bank API
2. **Secondary Approach**: Generate synthetic values based on:
   - Historical patterns
   - Regional correlations
   - Economic theory constraints
   - Random variation within realistic bounds

## Usage Notes

- **Exchange_Rate_Volatility** and **GDP_Growth_Volatility** require minimum 3 years of data
- **M2_Growth_Rate** is undefined for the first year of each country
- Values may exceed 100 for indices where this is economically meaningful (e.g., mobile subscriptions)
- Synthetic data maintains temporal consistency within countries

## Data Validation Rules

- All percentages are bounded appropriately
- Exchange rates are positive
- Risk scores are normalized to 0-100 scale
- Categorical variables follow defined classifications
- Time series maintain logical progression

## Update Frequency

This is a static dataset generated for research purposes. For production use:
- World Bank data updates: Quarterly
- Volatility calculations: Monthly recommended
- Risk scores: Real-time or daily for operational systems

---
*Last Updated: 2025-10-11*