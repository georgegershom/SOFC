# FinTech SSA Dataset - Data Dictionary

## Research Context
**Research Topic**: Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

This dataset contains quarterly time-series data for {num_companies} FinTech companies operating across 10 Sub-Sahara African countries from 2020-2023.

## Dataset Structure
- **Total Records**: {total_records}
- **Time Period**: Quarterly data for 16 quarters (Q1 2020 - Q4 2023)
- **Countries Covered**: Nigeria, Kenya, South Africa, Ghana, Uganda, Tanzania, Rwanda, Senegal, Zambia, Ethiopia

## Variable Definitions

### Identifier Variables

| Variable | Type | Description |
|----------|------|-------------|
| company_id | String | Unique identifier for each FinTech company (FT001-FT150) |
| company_name | String | Name of the FinTech company |
| quarter | Integer | Sequential quarter number (1-16) |
| year | Integer | Year of observation |
| quarter_of_year | Integer | Quarter within year (1-4) |
| date | Date | Date of the quarter (YYYY-MM-DD) |
| country | String | Country where FinTech operates |
| fintech_type | String | Type of FinTech service (Mobile Money, Payment Gateway, Digital Lending, etc.) |
| company_age_years | Float | Age of company in years at the time of observation |

### Dependent Variables (What we're predicting)

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| **fintech_failure** | Binary | 0-1 | Primary DV: 1 if company has failed/become insolvent/closed, 0 otherwise |
| **fintech_distress** | Binary | 0-1 | 1 if company shows signs of distress (multiple negative indicators), 0 otherwise |
| **regulatory_sanction** | Binary | 0-1 | 1 if company received regulatory sanction/fine/suspension in this quarter, 0 otherwise |

### Independent Variables - Financial Performance

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| revenue_usd | Float | USD | Quarterly revenue in US Dollars |
| revenue_growth_pct | Float | Percentage | Year-over-year revenue growth rate |
| net_income_usd | Float | USD | Quarterly net income (profit/loss) |
| profit_margin_pct | Float | Percentage | Net profit margin (net_income / revenue * 100) |
| burn_rate_usd | Float | USD | Cash burn rate for loss-making companies |
| funding_amount_usd | Float | USD | Amount of funding received in this quarter (0 if none) |
| funding_stage | String | Category | Current funding stage (Seed, Series A, B, C, Bootstrapped) |
| total_funding_to_date_usd | Float | USD | Cumulative funding received to date |

### Independent Variables - Operational Metrics

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| active_users | Integer | Count | Number of active users/customers in the quarter |
| user_growth_pct | Float | Percentage | User growth rate compared to baseline |
| transaction_volume_usd | Float | USD | Total value of transactions processed in the quarter |
| transaction_count | Integer | Count | Total number of transactions in the quarter |
| avg_transaction_value_usd | Float | USD | Average transaction size (volume/count) |
| num_agents | Integer | Count | Number of agents (for mobile money operators; 0 for other types) |
| customer_acquisition_cost_usd | Float | USD | Cost to acquire each new customer (CAC) |
| customer_churn_rate_pct | Float | Percentage | Customer churn rate (% of customers lost) |

### Contextual Variables (Country-level indicators)

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| country_market_size_index | Float | 0-1 | Relative market size indicator for the country |
| country_regulatory_strength_index | Float | 0-1 | Regulatory framework strength (higher = stronger) |
| country_economic_stability_index | Float | 0-1 | Economic stability indicator (higher = more stable) |
| quarters_since_last_funding | Integer | 0-4 | Number of quarters since last funding round |

## Data Notes

### Dependent Variable Details

1. **fintech_failure**: 
   - This is the primary outcome variable for building early warning models
   - A value of 1 indicates the company has permanently ceased operations
   - Once a company fails (1), it remains failed in subsequent quarters
   - Approximately {failure_rate}% of companies in the dataset experience failure

2. **fintech_distress**:
   - Leading indicator that may precede failure
   - Coded as 1 when company exhibits 2+ distress signals:
     * Large losses (net income < -50% of revenue)
     * Significant revenue decline (> 20% drop)
     * High customer churn (> 25%)
     * High burn rate (burning more than earning)

3. **regulatory_sanction**:
   - Indicates regulatory intervention
   - Can be a warning sign of operational or compliance issues
   - More common in companies with low health scores

### Data Quality & Limitations

1. **Synthetic Data**: This is a fabricated dataset designed for research and model development. While based on realistic parameters and relationships, it does not represent actual company data.

2. **Data Relationships**: The data includes realistic correlations:
   - Failed companies show deteriorating metrics in quarters leading up to failure
   - Healthier companies (better regulatory environment, larger markets) tend to perform better
   - Seasonal patterns are included in transaction volumes

3. **Missing Data**: This dataset has complete records (no missing values) to facilitate initial model development. Real-world data would have substantial missing values, especially for private company financial data.

## Suggested Usage

### For Early Warning Models:
- Use `fintech_failure` as the primary target variable
- Consider `fintech_distress` as an intermediate outcome
- Build models to predict failure 1-4 quarters in advance
- Use lagged variables (previous quarter metrics) as predictors

### Feature Engineering Suggestions:
- Create trend variables (3-quarter moving averages)
- Calculate quarter-over-quarter changes for key metrics
- Create interaction terms (e.g., burn_rate * funding_stage)
- Develop composite health scores from multiple indicators

### Model Development:
- Split data temporally (train on early periods, test on later)
- Consider company-level fixed effects
- Account for country-level clustering
- Handle class imbalance (failures are relatively rare)

## Data Sources Simulated

This synthetic dataset emulates data that would be collected from:
1. **Company Reports**: Financial statements, earnings reports
2. **Regulatory Filings**: Central Bank payment system statistics
3. **VC Databases**: Crunchbase, Disrupt Africa, Partech Africa
4. **Industry Reports**: GSMA Mobile Money reports, World Bank
5. **App Stores**: Download and usage statistics

## Citation

If using this dataset, please cite:
```
FinTech Early Warning Model Dataset for Sub-Sahara Africa
Generated: {generation_date}
Research Topic: FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies
```

## Contact & Support

For questions about variable definitions or data generation methodology, refer to the generation script: `generate_fintech_ssa_dataset.py`
