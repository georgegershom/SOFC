# Data Dictionary: SSA Macroeconomic Dataset
## For FinTech Early Warning Model Research

**Dataset Name:** Sub-Saharan Africa Macroeconomic & Country-Level Data  
**Version:** 1.0  
**Date Created:** 2025-10-11  
**Coverage:** 20 Sub-Saharan African Countries, 2010-2024  

---

## Variable Descriptions

### Identifier Variables

| Variable | Type | Description |
|----------|------|-------------|
| Country_Code | String | ISO 3166-1 alpha-3 country code |
| Country_Name | String | Full country name |
| Year | Integer | Year of observation (2010-2024) |

---

### Macroeconomic Variables

#### 1. GDP_Growth
- **Description:** Annual GDP growth rate
- **Unit:** Percentage (%)
- **Source:** Generated based on World Bank patterns
- **Range:** Typically -10% to +15% (varies by country)
- **Notes:** Includes COVID-19 shock in 2020

#### 2. GDP_Growth_Volatility
- **Description:** Rolling 3-year standard deviation of GDP growth
- **Unit:** Percentage points
- **Calculation:** 3-year rolling window standard deviation
- **Use:** Measures macroeconomic stability/instability

#### 3. Inflation
- **Description:** Annual inflation rate (Consumer Price Index)
- **Unit:** Percentage (%)
- **Source:** Generated based on historical patterns
- **Range:** -2% to 200% (Zimbabwe extreme case)
- **Notes:** Country-specific inflation regimes

#### 4. Inflation_Volatility
- **Description:** Rolling 3-year standard deviation of inflation
- **Unit:** Percentage points
- **Calculation:** 3-year rolling window standard deviation
- **Use:** Measures price stability

#### 5. Unemployment
- **Description:** Unemployment rate (% of total labor force)
- **Unit:** Percentage (%)
- **Range:** 2% to 35%
- **Notes:** Spike during COVID-19 period

#### 6. Exchange_Rate
- **Description:** Official exchange rate (Local Currency Units per US$)
- **Unit:** LCU per USD
- **Source:** Generated with country-specific depreciation trends
- **Notes:** Higher values indicate currency depreciation

#### 7. Exchange_Rate_Volatility
- **Description:** Rolling 3-year standard deviation of exchange rate
- **Unit:** LCU per USD
- **Calculation:** 3-year rolling window standard deviation
- **Use:** Critical for cross-border FinTech risk assessment

#### 8. Interest_Rate
- **Description:** Central Bank policy interest rate (real)
- **Unit:** Percentage (%)
- **Range:** Typically 3% to 25%
- **Notes:** Generally set above inflation rate

#### 9. M2_Growth
- **Description:** Broad Money Supply (M2) growth rate
- **Unit:** Percentage (%)
- **Calculation:** Annual growth in M2
- **Use:** Indicates monetary policy stance and liquidity

#### 10. Debt_to_GDP
- **Description:** Public debt as percentage of GDP
- **Unit:** Percentage (%)
- **Range:** 20% to 100%
- **Notes:** Generally increasing trend over period

---

### Digital Infrastructure Variables

#### 11. Mobile_Subscriptions_per_100
- **Description:** Mobile cellular subscriptions per 100 people
- **Unit:** Subscriptions per 100 inhabitants
- **Range:** 30 to 150
- **Trend:** Strong upward trend 2010-2024
- **Notes:** Can exceed 100 due to multiple SIM ownership

#### 12. Internet_Users_Percent
- **Description:** Percentage of population using the Internet
- **Unit:** Percentage (%)
- **Range:** 5% to 80%
- **Trend:** Rapid increase, especially post-2015
- **Use:** Proxy for FinTech adoption potential

#### 13. Secure_Servers_per_Million
- **Description:** Secure Internet servers per 1 million people
- **Unit:** Servers per million population
- **Range:** 0.5 to 50
- **Trend:** Increasing over time
- **Use:** Indicator of digital security infrastructure

---

## Countries Included

| Code | Country Name | Region |
|------|--------------|--------|
| NGA | Nigeria | West Africa |
| ZAF | South Africa | Southern Africa |
| KEN | Kenya | East Africa |
| ETH | Ethiopia | East Africa |
| GHA | Ghana | West Africa |
| TZA | Tanzania | East Africa |
| UGA | Uganda | East Africa |
| CIV | Côte d'Ivoire | West Africa |
| SEN | Senegal | West Africa |
| RWA | Rwanda | East Africa |
| ZMB | Zambia | Southern Africa |
| MOZ | Mozambique | Southern Africa |
| BWA | Botswana | Southern Africa |
| NAM | Namibia | Southern Africa |
| ZWE | Zimbabwe | Southern Africa |
| CMR | Cameroon | Central Africa |
| AGO | Angola | Central Africa |
| BEN | Benin | West Africa |
| BFA | Burkina Faso | West Africa |
| MLI | Mali | West Africa |

**Total:** 20 countries representing major SSA economies across all sub-regions

---

## Time Period

- **Start Year:** 2010
- **End Year:** 2024
- **Frequency:** Annual
- **Total Observations:** 300 (20 countries × 15 years)

---

## Data Quality Notes

### Data Generation Methodology

This dataset combines:

1. **Real-world patterns:** Parameters based on historical World Bank, IMF, and AfDB data
2. **Country-specific characteristics:** Different growth trajectories, inflation regimes, and development levels
3. **Economic shocks:** COVID-19 pandemic impact (2020)
4. **Realistic volatility:** Stochastic variation within empirically-observed ranges
5. **Trend components:** Digital infrastructure growth, debt accumulation

### Special Considerations

- **COVID-19 Impact (2020):** GDP shock of approximately -5%, unemployment spike
- **Zimbabwe:** Extreme inflation scenario reflecting hyperinflation history
- **Digital Infrastructure:** Monotonic increasing trend reflecting technology adoption
- **Exchange Rates:** Country-specific depreciation patterns

### Use Cases

This dataset is specifically designed for:

1. FinTech early warning models
2. Macroeconomic risk assessment in SSA
3. Digital economy analysis
4. Cross-country comparative studies
5. Time series forecasting models
6. Financial stability research

---

## Data Sources & References

### Primary Sources (for parameter calibration):

1. **World Bank Open Data**
   - URL: https://data.worldbank.org/
   - Indicators: GDP growth, inflation, unemployment, digital infrastructure

2. **International Monetary Fund (IMF)**
   - URL: https://www.imf.org/en/Data
   - Financial soundness indicators, monetary data

3. **African Development Bank (AfDB)**
   - URL: https://dataportal.opendataforafrica.org/
   - Regional economic statistics

### Data Processing

- **Random Seed:** 42 (for reproducibility)
- **Generation Date:** 2025-10-11
- **Software:** Python 3.x with pandas, numpy

---

## Citation

When using this dataset, please cite:

```
Sub-Saharan Africa Macroeconomic Dataset (2010-2024)
Generated for: Research on FinTech Early Warning Model in Nexus of Fintech Risk 
              in Sub-Sahara Africa Economies
Date: October 2025
Coverage: 20 SSA countries, 15 years, 13 macroeconomic and infrastructure variables
```

---

## Contact & Updates

For questions or data updates, refer to the research project documentation.

**Last Updated:** 2025-10-11
