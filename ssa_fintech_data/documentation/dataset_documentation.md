# Sub-Saharan Africa FinTech Early Warning Model Dataset

## Overview

This comprehensive dataset has been compiled for research on "FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies". The dataset combines real macroeconomic data from international sources with synthetic FinTech-specific indicators to create a robust foundation for early warning model development.

**Dataset Version**: 1.0  
**Last Updated**: October 11, 2025  
**Coverage**: 20 Sub-Saharan African countries, 2010-2023  
**Total Observations**: 280 (20 countries × 14 years)  
**Total Indicators**: 93

## Countries Covered

The dataset includes the following 20 Sub-Saharan African countries, selected based on their FinTech market development and data availability:

1. **Kenya (KE)** - Leading mobile money market
2. **Nigeria (NG)** - Largest economy in SSA
3. **South Africa (ZA)** - Most developed financial sector
4. **Ghana (GH)** - Growing FinTech hub
5. **Uganda (UG)** - Mobile money pioneer
6. **Tanzania (TZ)** - Large mobile money market
7. **Rwanda (RW)** - Digital transformation leader
8. **Senegal (SN)** - West African FinTech hub
9. **Côte d'Ivoire (CI)** - Emerging market
10. **Zambia (ZM)** - Copper-dependent economy
11. **Botswana (BW)** - Stable middle-income country
12. **Malawi (MW)** - Low-income, agriculture-based
13. **Mozambique (MZ)** - Post-conflict recovery
14. **Ethiopia (ET)** - Largest population in region
15. **Zimbabwe (ZW)** - Economic challenges
16. **Cameroon (CM)** - Central African hub
17. **Burkina Faso (BF)** - Sahel region
18. **Mali (ML)** - Security challenges
19. **Benin (BJ)** - Small open economy
20. **Togo (TG)** - Transit economy

## Data Sources

### Primary Sources (Real Data)
- **World Bank Open Data**: Macroeconomic and development indicators
- **International Monetary Fund (IMF)**: Financial and monetary data
- **African Development Bank (AfDB)**: Regional development statistics

### Synthetic Data Sources
- **FinTech-specific indicators**: Generated using statistical models based on real economic conditions
- **Missing indicator imputation**: Synthetic data for indicators with low availability

## Dataset Structure

### File Formats Available
1. **ssa_comprehensive_dataset.csv** - Main dataset in CSV format
2. **ssa_comprehensive_dataset.xlsx** - Excel format with formatting
3. **ssa_macro_data_simple.csv** - Basic World Bank indicators only
4. **ssa_macro_data_enhanced.csv** - Real data with calculated indicators
5. **enhanced_data_report.json** - Detailed statistical summary

## Variable Categories

### Category 1: Core Macroeconomic Indicators

#### Economic Growth & Stability
- `gdp_growth` - GDP growth rate (annual %)
- `gdp_growth_volatility` - 3-year rolling standard deviation of GDP growth
- `inflation` - Consumer price inflation (annual %)
- `inflation_volatility` - 3-year rolling standard deviation of inflation
- `unemployment` - Unemployment rate (% of total labor force)

#### Monetary & Financial
- `interest_rate` - Real interest rate (%)
- `central_bank_policy_rate` - Central bank policy rate (%) [Synthetic]
- `money_supply` - Broad money supply (% of GDP)
- `financial_depth` - Domestic credit to private sector (% of GDP)
- `financial_depth_growth` - Year-over-year growth in financial depth

#### External Sector
- `exchange_rate` - Official exchange rate (LCU per USD)
- `exchange_rate_volatility` - 3-year rolling standard deviation of exchange rate changes
- `current_account_balance_gdp` - Current account balance (% of GDP) [Synthetic]
- `fx_reserves_months_imports` - Foreign exchange reserves (months of imports) [Synthetic]
- `trade_openness` - Trade (% of GDP)
- `fdi_inflows` - Foreign direct investment inflows (% of GDP)

#### Fiscal & Governance
- `debt_gdp` - Government debt (% of GDP)
- `government_debt_gdp` - Alternative debt measure [Synthetic]
- `political_stability_index` - Political stability index (-2.5 to 2.5) [Synthetic]
- `credit_rating_score` - Country credit rating (0-100) [Synthetic]
- `ease_doing_business_rank` - World Bank Doing Business rank [Synthetic]

### Category 2: Digital Infrastructure

#### Connectivity & Access
- `mobile_subs` - Mobile cellular subscriptions (per 100 people)
- `mobile_subs_growth` - Year-over-year growth in mobile subscriptions
- `internet_users` - Internet users (% of population)
- `internet_users_growth` - Year-over-year growth in internet users
- `secure_servers` - Secure internet servers (per 1 million people)

#### Digital Readiness
- `digital_infrastructure` - Composite digital infrastructure index
- `digital_literacy_rate` - Digital literacy rate (% of population) [Synthetic]
- `digital_divide` - Gap between mobile and internet penetration

### Category 3: FinTech-Specific Indicators [All Synthetic]

#### FinTech Adoption & Usage
- `fintech_adoption_rate` - FinTech service usage (% of population)
- `mobile_money_penetration` - Mobile money account ownership (% of adults)
- `digital_payment_volume_gdp` - Digital payment volume (% of GDP)
- `financial_inclusion_proxy` - Financial inclusion composite indicator

#### FinTech Infrastructure & Investment
- `fintech_investment_usd_millions` - FinTech investment (USD millions)
- `fintech_regulatory_score` - FinTech regulatory environment score (0-100)
- `banking_sector_concentration_hhi` - Banking sector concentration (HHI index)
- `cross_border_payment_costs_pct` - Cross-border payment costs (% of transaction)

#### Risk Indicators
- `cybersecurity_incidents_per_100k` - Cybersecurity incidents per 100k transactions
- `financial_exclusion_rate` - Financial exclusion rate (% of adults)
- `external_vulnerability` - External vulnerability composite index
- `economic_instability` - Economic instability composite index

### Category 4: Risk Classifications

#### Categorical Risk Indicators
- `gdp_growth_risk` - GDP growth risk category (High/Medium/Low/Very Low Risk)
- `inflation_risk` - Inflation risk category (Deflation/Low/Medium/High Risk)
- `unemployment_risk` - Unemployment risk category (Low/Medium/High/Very High Risk)
- `digital_infrastructure_level` - Digital infrastructure level (Low/Medium-Low/Medium-High/High)

### Category 5: Regional Comparisons

For each numeric indicator, the following regional comparison metrics are available:
- `[indicator]_regional_avg` - Regional average for the same year
- `[indicator]_regional_percentile` - Percentile rank within region (0-100)

## Data Quality Assessment

### Completeness Rates (Real Data Indicators)
- **Excellent (>95%)**: GDP growth, unemployment, secure servers, mobile subscriptions, internet users
- **Good (90-95%)**: Inflation, FDI inflows, financial depth, money supply, exchange rate
- **Fair (70-90%)**: Trade openness, interest rate
- **Poor (<70%)**: Government debt

### Synthetic Data Quality
- **FinTech indicators**: 100% complete, generated using econometric relationships
- **Missing economic indicators**: Generated to fill gaps in real data
- **Validation**: Synthetic data validated against regional benchmarks and economic theory

## Usage Guidelines

### For Early Warning Models
1. **Dependent Variables**: Use risk classifications or create binary crisis indicators
2. **Leading Indicators**: Focus on volatility measures and growth rates
3. **Structural Breaks**: Consider 2020-2021 COVID-19 period
4. **Cross-validation**: Use country-based or time-based splits

### For FinTech Risk Analysis
1. **FinTech Adoption**: Use synthetic FinTech indicators as proxies
2. **Digital Divide**: Analyze gaps between mobile and internet penetration
3. **Regulatory Environment**: Consider regulatory scores in model specification
4. **Cybersecurity**: Include cyber risk indicators in risk models

### Statistical Considerations
1. **Panel Data**: Use country and time fixed effects
2. **Autocorrelation**: Consider lagged variables for dynamic models
3. **Heteroskedasticity**: Robust standard errors recommended
4. **Missing Data**: Multiple imputation for real indicators with gaps

## Crisis Scenarios

The dataset includes predefined crisis scenarios for stress testing:

1. **Global Financial Crisis**: GDP shock, inflation increase, exchange rate depreciation
2. **Commodity Price Shock**: GDP decline, deflation risk, currency pressure
3. **Cyber Attack Scenario**: FinTech adoption decline, cybersecurity incidents spike
4. **Regulatory Crackdown**: FinTech restrictions, adoption decline

## Limitations

### Real Data Limitations
- **Timeliness**: Some indicators have reporting lags
- **Consistency**: Methodological changes over time
- **Coverage**: Some countries have limited data for certain indicators

### Synthetic Data Limitations
- **Model Assumptions**: Based on statistical relationships, not actual observations
- **Validation**: Limited ability to validate against actual FinTech data
- **Correlation Structure**: May not capture all real-world relationships

## Recommended Citation

When using this dataset, please cite:

```
SSA FinTech Early Warning Model Dataset (2025). Comprehensive macroeconomic and 
FinTech indicators for Sub-Saharan African countries, 2010-2023. 
Research on FinTech Early Warning Model in Nexus of Fintech Risk in 
Sub-Sahara Africa Economies.
```

## Contact Information

For questions about the dataset or methodology, please contact the research team.

## Version History

- **v1.0** (October 2025): Initial release with 20 countries, 93 indicators
- Real World Bank data collection and processing
- Synthetic FinTech indicator generation
- Risk categorization and regional comparisons

## Appendix: Variable Definitions

### Real Data Sources and Codes
- GDP Growth: World Bank NY.GDP.MKTP.KD.ZG
- Inflation: World Bank FP.CPI.TOTL.ZG
- Unemployment: World Bank SL.UEM.TOTL.ZS
- Mobile Subscriptions: World Bank IT.CEL.SETS.P2
- Internet Users: World Bank IT.NET.USER.ZS
- [Full mapping available in country_mapping.json]

### Synthetic Data Generation Methods
- **Beta Distribution**: For bounded indicators (0-100%)
- **Log-normal Distribution**: For positive skewed variables
- **Normal Distribution**: For symmetric variables
- **Correlation Preservation**: Synthetic data correlated with real economic conditions

---

*This documentation is part of the SSA FinTech Early Warning Model research project.*