# FinTech Early Warning Model Dataset for Sub-Saharan Africa

## Overview

This dataset was generated for research on FinTech early warning models in Sub-Saharan Africa economies. It contains comprehensive data on 200 FinTech companies across 10 SSA countries, including financial performance, operational metrics, funding data, regulatory information, and distress indicators.

## Dataset Structure

The dataset consists of 6 main CSV files:

### 1. companies.csv
Core company information including:
- `company_id`: Unique identifier
- `company_name`: Company name (based on real SSA FinTech companies)
- `country`: Operating country
- `fintech_type`: Type of FinTech service
- `age_years`: Company age in years
- `company_size`: Size category (Startup, Small, Medium, Large)
- `employees`: Number of employees
- `regulatory_status`: Regulatory compliance status
- `funding_stage`: Current funding stage
- `gdp_per_capita`: Country GDP per capita
- `mobile_penetration`: Mobile phone penetration rate
- `fintech_maturity`: Country's FinTech ecosystem maturity

### 2. financial_metrics.csv
Quarterly financial performance data:
- `company_id`: Company identifier
- `quarter`: Quarter identifier (Q1-Q8, representing last 2 years)
- `quarter_date`: Date of the quarter
- `revenue_usd`: Revenue in USD
- `revenue_growth_rate`: Quarter-over-quarter revenue growth
- `net_income_usd`: Net income in USD
- `profit_margin`: Profit margin ratio
- `operating_expenses_usd`: Operating expenses in USD
- `burn_rate_usd`: Monthly burn rate (for startups)

### 3. operational_metrics.csv
Operational performance indicators:
- `company_id`: Company identifier
- `quarter`: Quarter identifier
- `quarter_date`: Date of the quarter
- `active_users`: Number of active users
- `user_growth_rate`: User growth rate
- `churn_rate`: Customer churn rate
- `total_transactions`: Total transaction count
- `transaction_volume_usd`: Total transaction volume in USD
- `avg_transaction_value_usd`: Average transaction value
- `customer_acquisition_cost_usd`: CAC in USD
- `agents_count`: Number of agents (for mobile money)

### 4. funding_data.csv
Funding rounds and investment information:
- `company_id`: Company identifier
- `round_number`: Sequential round number
- `round_date`: Date of funding round
- `round_type`: Type of funding round
- `amount_raised_usd`: Amount raised in USD
- `valuation_usd`: Company valuation in USD
- `num_investors`: Number of investors
- `investors`: List of investors
- `months_since_last_round`: Months since previous round

### 5. regulatory_data.csv
Regulatory sanctions and compliance events:
- `company_id`: Company identifier
- `event_date`: Date of regulatory event
- `event_type`: Type of regulatory action
- `severity`: Severity level (Low, Medium, High)
- `fine_amount_usd`: Fine amount in USD
- `resolution_status`: Current resolution status
- `description`: Event description

### 6. distress_indicators.csv
Distress prediction variables:
- `company_id`: Company identifier
- `distress_score`: Continuous distress score (0-1)
- `is_distressed`: Binary distress indicator (0/1)
- `revenue_decline_rate`: Rate of revenue decline
- `user_decline_rate`: Rate of user decline
- `has_revenue_decline`: Binary revenue decline indicator
- `has_user_decline`: Binary user decline indicator
- `has_regulatory_issues`: Binary regulatory issues indicator
- `regulatory_issues_count`: Count of regulatory issues
- `months_since_founded`: Months since company founding

## Key Statistics

- **Total Companies**: 200
- **Countries Covered**: 10 (Nigeria, Kenya, South Africa, Ghana, Uganda, Tanzania, Rwanda, Senegal, Ethiopia, Zambia)
- **FinTech Types**: 8 (Mobile Money, Digital Banking, Payment Gateway, Lending Platform, Investment Platform, Insurance Tech, Crypto Exchange, Remittance)
- **Distress Rate**: 42% (84 out of 200 companies)
- **Time Period**: 8 quarters (2 years) of historical data

## FinTech Types Distribution

1. **Mobile Money** (25%): MTN Mobile Money, Airtel Money, M-Pesa
2. **Digital Banking** (20%): Kuda, Carbon, Tala
3. **Payment Gateway** (15%): Flutterwave, Paystack, Yoco
4. **Lending Platform** (12%): Branch, JUMO
5. **Investment Platform** (10%): PiggyVest, Cowrywise, EasyEquities
6. **Insurance Tech** (8%): Various insurtech companies
7. **Crypto Exchange** (5%): Luno, various crypto platforms
8. **Remittance** (5%): Mukuru, various remittance services

## Country Distribution

1. **Nigeria** (25%): Largest FinTech ecosystem in SSA
2. **Kenya** (20%): M-Pesa dominated market
3. **South Africa** (15%): Most mature financial sector
4. **Ghana** (10%): Growing mobile money adoption
5. **Uganda** (8%): Emerging FinTech market
6. **Tanzania** (7%): M-Pesa and mobile money growth
7. **Rwanda** (5%): Government-supported digital transformation
8. **Senegal** (4%): Francophone West Africa leader
9. **Ethiopia** (3%): Large population, low penetration
10. **Zambia** (3%): Emerging market with growth potential

## Data Generation Methodology

The dataset was generated using a sophisticated synthetic data generation approach that:

1. **Real Company Names**: Used actual FinTech company names from each SSA country
2. **Realistic Financial Models**: Applied country-specific economic indicators (GDP, mobile penetration)
3. **Industry-Specific Characteristics**: Different risk profiles and growth patterns for each FinTech type
4. **Regulatory Realism**: Based on actual regulatory frameworks in SSA countries
5. **Temporal Consistency**: Generated 8 quarters of consistent time-series data
6. **Distress Modeling**: Created realistic distress indicators based on multiple risk factors

## Usage for Research

This dataset is designed for:

1. **Early Warning Models**: Predicting FinTech company distress
2. **Risk Assessment**: Understanding risk factors in SSA FinTech
3. **Policy Analysis**: Regulatory impact on FinTech stability
4. **Investment Research**: Funding patterns and success factors
5. **Market Analysis**: FinTech ecosystem development in SSA

## Data Quality Notes

- All financial figures are in USD
- Revenue and transaction volumes are realistic for SSA markets
- Regulatory events are based on actual SSA regulatory frameworks
- Distress indicators are calculated using multiple risk factors
- Data includes both successful and distressed companies for balanced analysis

## Citation

If you use this dataset in your research, please cite:

```
FinTech Early Warning Model Dataset for Sub-Saharan Africa
Generated for research on FinTech risk assessment in SSA economies
[Your Institution], [Year]
```

## Contact

For questions about this dataset or research collaboration, please contact [Your Contact Information].