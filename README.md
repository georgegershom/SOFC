# FinTech Early Warning Dataset for Sub-Saharan Africa

## Research Context

**Title**: Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

**Focus**: Category 1: FinTech-Specific Data (The Micro Foundation)

This dataset provides comprehensive, fabricated data for developing early warning models to predict FinTech distress in Sub-Saharan African economies. The dataset addresses the critical challenge of limited publicly available FinTech performance data in the SSA region while maintaining research-grade quality and realism.

## Dataset Overview

- **Total Observations**: 3,332 quarterly records
- **Unique Companies**: 150 FinTech companies
- **Geographic Coverage**: 18 Sub-Saharan African countries
- **FinTech Types**: 12 different categories
- **Time Period**: Q1 2019 to Q4 2024 (6 years)
- **Distressed Companies**: 29 companies (19.3%) with distress events
- **Total Distress Events**: 181 observations

## Key Features

### Dependent Variables (What we predict)
- **Primary**: `is_distressed` - Binary indicator of company distress
- **Risk Scores**: `closure_risk`, `acquisition_risk` - Continuous risk measures (0-1)
- **Distress Types**: `distress_type` - Categories: closure, acquisition, regulatory_action, severe_downturn
- **Regulatory Actions**: `regulatory_action` - Binary indicator of regulatory sanctions

### Independent Variables (Predictors)

#### Financial Performance Metrics
- `quarterly_revenue` - Total revenue per quarter (USD)
- `net_income` - Net profit/loss per quarter (USD)
- `burn_rate` - Cash burn rate for startups (USD)
- `revenue_growth_rate` - Quarter-over-quarter growth (%)
- `profit_margin` - Net profit margin (%)

#### Operational Metrics
- `active_users` - Number of active users
- `transaction_count` - Total transactions processed
- `transaction_volume` - Total transaction value (USD)
- `customer_acquisition_cost` - Cost to acquire new customers (USD)
- `churn_rate` - Customer churn rate (%)
- `number_of_agents` - Number of agents/partners

#### Funding Information
- `funding_round` - Whether funding was raised this quarter
- `funding_stage` - Stage of funding (Pre-Seed to Series D)
- `funding_amount` - Amount raised (USD)

## Geographic Distribution

The dataset covers 18 Sub-Saharan African countries with realistic market distributions:

**Top Countries by Observations**:
1. Ivory Coast (328 observations)
2. Zambia (300 observations)
3. Senegal (264 observations)
4. Mozambique (252 observations)
5. Ghana (216 observations)

**Countries with Highest Distress Rates**:
1. Mali (19.6% distress rate)
2. Tanzania (15.0% distress rate)
3. Malawi (12.5% distress rate)

## FinTech Type Coverage

**12 FinTech Categories Represented**:
- Digital Banking (496 observations)
- Investment Platform (360 observations)
- Crypto Exchange (316 observations)
- Lending Platform (308 observations)
- Remittance Service (284 observations)
- Mobile Money (244 observations)
- Payment Gateway (228 observations)
- And 5 others...

**Distress Risk by Type**:
- Microfinance Tech: 10.8% distress rate
- Crowdfunding Platform: 7.6% distress rate
- Lending Platform: 7.4% distress rate

## Data Quality Metrics

- **Missing Values**: 5.4% (primarily in optional fields like funding_stage)
- **Duplicate Records**: 0
- **Temporal Consistency**: ✓ All dates and ages are logically consistent
- **Value Ranges**: ✓ All metrics within realistic bounds
- **Correlations**: Strong logical correlations (e.g., revenue vs costs: 0.98)

## Key Statistics

**Financial Performance**:
- Average Quarterly Revenue: $12.5M (median: $2.5M)
- Profitable Companies: 77.3%
- Average Users per Company: 782K
- Average Transaction Volume: $631M per quarter

**Distress Patterns**:
- Distress events increase with company age (most in 10+ year companies)
- Q4 shows highest distress frequency (seasonal pattern)
- Strong correlation between distress and operational metrics

## Files Included

1. **`fintech_distress_dataset.csv`** - Main dataset (3,332 rows × 35 columns)
2. **`dataset_metadata.json`** - Comprehensive metadata and variable descriptions
3. **`generate_fintech_dataset.py`** - Data generation script with full methodology
4. **`validate_dataset.py`** - Data quality validation and analysis script
5. **`requirements.txt`** - Python dependencies
6. **`README.md`** - This documentation file

## Usage Instructions

### Loading the Dataset

```python
import pandas as pd
import json

# Load main dataset
df = pd.read_csv('fintech_distress_dataset.csv')

# Load metadata
with open('dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Dataset shape: {df.shape}")
print(f"Companies: {df['company_id'].nunique()}")
print(f"Date range: {df['quarter'].min()} to {df['quarter'].max()}")
```

### Basic Analysis

```python
# Distress analysis
distressed_companies = df[df['is_distressed'] == True]
print(f"Distressed observations: {len(distressed_companies)}")

# Financial performance by distress status
performance = df.groupby('is_distressed').agg({
    'quarterly_revenue': 'mean',
    'profit_margin': 'mean',
    'churn_rate': 'mean'
})
print(performance)
```

### Time Series Analysis

```python
# Convert quarter to datetime
df['quarter'] = pd.to_datetime(df['quarter'])

# Company-level time series
company_ts = df[df['company_id'] == 'FT_001'].set_index('quarter')
company_ts[['quarterly_revenue', 'active_users', 'closure_risk']].plot()
```

## Research Applications

### Early Warning Model Development

1. **Binary Classification**: Predict `is_distressed` using financial and operational metrics
2. **Risk Scoring**: Develop continuous risk scores using `closure_risk` and `acquisition_risk`
3. **Multi-class Classification**: Predict specific `distress_type`
4. **Time Series Forecasting**: Predict future distress using historical patterns

### Recommended Modeling Approaches

1. **Logistic Regression**: Baseline interpretable model
2. **Random Forest**: Handle non-linear relationships and feature interactions
3. **Gradient Boosting**: XGBoost/LightGBM for high performance
4. **LSTM/GRU**: Capture temporal dependencies
5. **Survival Analysis**: Time-to-distress modeling

### Feature Engineering Suggestions

```python
# Lag features for early warning
df['revenue_lag1'] = df.groupby('company_id')['quarterly_revenue'].shift(1)
df['revenue_change'] = df['quarterly_revenue'] - df['revenue_lag1']

# Moving averages
df['revenue_ma3'] = df.groupby('company_id')['quarterly_revenue'].rolling(3).mean()

# Ratios and derived metrics
df['revenue_per_user'] = df['quarterly_revenue'] / df['active_users']
df['cost_efficiency'] = df['quarterly_costs'] / df['transaction_volume']
```

### Cross-Validation Strategy

Use temporal splits to avoid data leakage:

```python
from sklearn.model_selection import TimeSeriesSplit

# Sort by company and date
df_sorted = df.sort_values(['company_id', 'quarter'])

# Use TimeSeriesSplit for validation
tscv = TimeSeriesSplit(n_splits=5)
```

## Data Generation Methodology

The dataset was generated using a sophisticated simulation approach that ensures:

1. **Realistic Company Profiles**: Based on actual SSA FinTech landscape
2. **Temporal Consistency**: Logical progression of metrics over time
3. **Distress Patterns**: Realistic early warning signals 2-4 quarters before distress
4. **Market Dynamics**: Country-specific and sector-specific variations
5. **Correlation Structure**: Maintains expected relationships between variables

### Key Assumptions

- **Growth Phases**: Companies follow startup → growth → expansion → mature phases
- **Seasonal Effects**: Q4 typically shows stronger performance
- **Distress Probability**: 15% base rate, adjusted for company characteristics
- **Market Tiers**: Tier 1 (major markets), Tier 2 (emerging), Tier 3 (nascent)

## Limitations and Considerations

1. **Fabricated Data**: While realistic, this is simulated data for research purposes
2. **SSA Context**: Patterns may not generalize to other regions
3. **Time Period**: Limited to 2019-2024 timeframe
4. **Company Selection**: Focus on formal FinTech companies (excludes informal sector)
5. **Regulatory Environment**: Simplified regulatory action modeling

## Citation

If you use this dataset in your research, please cite:

```
FinTech Early Warning Dataset for Sub-Saharan Africa (2024)
Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies
Generated for academic research purposes
```

## Contact and Support

For questions about the dataset methodology, additional variables, or research collaboration:

- Review the `generate_fintech_dataset.py` script for detailed methodology
- Run `validate_dataset.py` for comprehensive data quality analysis
- Check `dataset_metadata.json` for complete variable descriptions

## License

This dataset is provided for academic and research purposes. Please ensure appropriate attribution when using in publications or presentations.

---

**Last Updated**: October 2024  
**Dataset Version**: 1.0  
**Research Focus**: FinTech Early Warning Systems for Sub-Saharan Africa