# FinTech Distress Dataset - Data Dictionary

## Overview
This document provides detailed descriptions of all variables in the FinTech Early Warning Model Dataset for Sub-Saharan Africa.

## 1. companies.csv

| Variable | Type | Description | Values/Format |
|----------|------|-------------|---------------|
| `company_id` | String | Unique identifier for each company | FT0001, FT0002, ... |
| `company_name` | String | Name of the FinTech company | Based on real SSA FinTech companies |
| `country` | String | Country where company operates | Nigeria, Kenya, South Africa, Ghana, Uganda, Tanzania, Rwanda, Senegal, Ethiopia, Zambia |
| `fintech_type` | String | Type of FinTech service | Mobile Money, Digital Banking, Payment Gateway, Lending Platform, Investment Platform, Insurance Tech, Crypto Exchange, Remittance |
| `age_years` | Float | Company age in years | 1.0 - 15.0 |
| `company_size` | String | Company size category | Startup, Small, Medium, Large |
| `employees` | Integer | Number of employees | 5 - 1000 |
| `regulatory_status` | String | Regulatory compliance status | Licensed, Pending, Unlicensed, Suspended |
| `funding_stage` | String | Current funding stage | Bootstrap, Seed, Series A, Series B, Series C, IPO, Acquired |
| `gdp_per_capita` | Float | Country GDP per capita in USD | 822 - 6994 |
| `mobile_penetration` | Float | Mobile phone penetration rate | 0.45 - 0.95 |
| `fintech_maturity` | String | Country's FinTech ecosystem maturity | low, medium, high |

## 2. financial_metrics.csv

| Variable | Type | Description | Values/Format |
|----------|------|-------------|---------------|
| `company_id` | String | Company identifier | FT0001, FT0002, ... |
| `quarter` | String | Quarter identifier | Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8 |
| `quarter_date` | String | Date of the quarter | YYYY-MM-DD format |
| `revenue_usd` | Float | Revenue in USD | 10,000 - 2,000,000 |
| `revenue_growth_rate` | Float | Quarter-over-quarter revenue growth | -0.2 to 0.5 |
| `net_income_usd` | Float | Net income in USD | Can be negative |
| `profit_margin` | Float | Profit margin ratio | -0.5 to 0.6 |
| `operating_expenses_usd` | Float | Operating expenses in USD | 5,000 - 1,500,000 |
| `burn_rate_usd` | Float | Monthly burn rate in USD | 0 - 500,000 (for startups) |

## 3. operational_metrics.csv

| Variable | Type | Description | Values/Format |
|----------|------|-------------|---------------|
| `company_id` | String | Company identifier | FT0001, FT0002, ... |
| `quarter` | String | Quarter identifier | Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8 |
| `quarter_date` | String | Date of the quarter | YYYY-MM-DD format |
| `active_users` | Integer | Number of active users | 100 - 50,000 |
| `user_growth_rate` | Float | User growth rate | -0.1 to 0.3 |
| `churn_rate` | Float | Customer churn rate | 0.02 - 0.08 |
| `total_transactions` | Integer | Total transaction count | 200 - 1,000,000 |
| `transaction_volume_usd` | Float | Total transaction volume in USD | 10,000 - 100,000,000 |
| `avg_transaction_value_usd` | Float | Average transaction value in USD | 10 - 500 |
| `customer_acquisition_cost_usd` | Float | CAC in USD | 5 - 50 |
| `agents_count` | Integer | Number of agents (for mobile money) | 0 - 500 |

## 4. funding_data.csv

| Variable | Type | Description | Values/Format |
|----------|------|-------------|---------------|
| `company_id` | String | Company identifier | FT0001, FT0002, ... |
| `round_number` | Integer | Sequential round number | 1 - 8 |
| `round_date` | String | Date of funding round | YYYY-MM-DD format |
| `round_type` | String | Type of funding round | Seed, Series A, Series B, Series C, Series D, Bridge, Series D+ |
| `amount_raised_usd` | Float | Amount raised in USD | 10,000 - 20,000,000 |
| `valuation_usd` | Float | Company valuation in USD | 50,000 - 100,000,000 |
| `num_investors` | Integer | Number of investors | 1 - 5 |
| `investors` | String | List of investors | Comma-separated list |
| `months_since_last_round` | Integer | Months since previous round | 0 - 24 |

## 5. regulatory_data.csv

| Variable | Type | Description | Values/Format |
|----------|------|-------------|---------------|
| `company_id` | String | Company identifier | FT0001, FT0002, ... |
| `event_date` | String | Date of regulatory event | YYYY-MM-DD format |
| `event_type` | String | Type of regulatory action | Warning, Fine, Suspension, License Revocation, Compliance Review |
| `severity` | String | Severity level | Low, Medium, High |
| `fine_amount_usd` | Float | Fine amount in USD | 0 - 100,000 |
| `resolution_status` | String | Current resolution status | Open, Resolved, Appealed |
| `description` | String | Event description | Text description |

## 6. distress_indicators.csv

| Variable | Type | Description | Values/Format |
|----------|------|-------------|---------------|
| `company_id` | String | Company identifier | FT0001, FT0002, ... |
| `distress_score` | Float | Continuous distress score | 0.0 - 1.0 (higher = more distressed) |
| `is_distressed` | Integer | Binary distress indicator | 0 (not distressed), 1 (distressed) |
| `revenue_decline_rate` | Float | Rate of revenue decline | -0.5 to 0.5 |
| `user_decline_rate` | Float | Rate of user decline | -0.3 to 0.3 |
| `has_revenue_decline` | Integer | Binary revenue decline indicator | 0, 1 |
| `has_user_decline` | Integer | Binary user decline indicator | 0, 1 |
| `has_regulatory_issues` | Integer | Binary regulatory issues indicator | 0, 1 |
| `regulatory_issues_count` | Integer | Count of regulatory issues | 0 - 5 |
| `months_since_founded` | Integer | Months since company founding | 12 - 180 |

## Key Relationships

### Primary Keys
- `company_id` is the primary key that links all datasets

### Foreign Keys
- All other datasets reference `companies.company_id`

### Time Series
- Financial and operational metrics are provided for 8 quarters (Q1-Q8)
- Q8 represents the most recent quarter
- Q1 represents 7 quarters ago

### Distress Classification
- `is_distressed = 1` if `distress_score > 0.4`
- Distress score is calculated based on multiple risk factors
- 42% of companies in the dataset are classified as distressed

## Data Quality Notes

1. **Missing Values**: Some companies may have fewer than 8 quarters of data if they are newer
2. **Currency**: All monetary values are in USD
3. **Temporal Consistency**: Data represents a 2-year period ending at the time of generation
4. **Realistic Ranges**: All values are within realistic ranges for SSA FinTech companies
5. **Synthetic Data**: This is synthetic data generated for research purposes

## Usage Guidelines

1. **Research Applications**: Suitable for early warning model development, risk assessment, and policy analysis
2. **Model Development**: Use `is_distressed` as the dependent variable for classification models
3. **Feature Engineering**: Combine variables from different datasets for comprehensive analysis
4. **Temporal Analysis**: Use quarter-level data for time series analysis
5. **Country Analysis**: Group by country for regional insights

## Citation

When using this dataset, please cite:
```
FinTech Early Warning Model Dataset for Sub-Saharan Africa
Generated for research on FinTech risk assessment in SSA economies
[Your Institution], [Year]
```