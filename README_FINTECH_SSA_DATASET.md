# FinTech Early Warning Model Dataset for Sub-Saharan Africa

## Dataset Overview

This synthetic dataset has been generated for research on **FinTech Early Warning Models in the Nexus of FinTech Risk in Sub-Saharan Africa Economies**. The dataset simulates realistic patterns of FinTech company performance, distress indicators, and failure patterns across 10 SSA countries.

### Key Features
- **500 FinTech companies** across Sub-Saharan Africa
- **20 quarterly observations** per company (5 years: 2019-2023)
- **10,000 total observations**
- **38 variables** covering financial, operational, and risk metrics
- **Realistic distress patterns** with ~23% of companies experiencing distress

## Files Generated

1. **`fintech_ssa_distress_dataset.csv`** - Main dataset (10,000 rows × 38 columns)
2. **`fintech_ssa_data_dictionary.csv`** - Complete variable descriptions
3. **`fintech_ssa_dataset_generator.py`** - Python script for dataset generation (reproducible)

## Dataset Structure

### Geographic Coverage
The dataset covers 10 major SSA economies with varying FinTech penetration:
- **Nigeria** (25% weight) - Largest FinTech market
- **Kenya** (20%) - M-Pesa pioneer, strong regulatory framework
- **South Africa** (15%) - Most developed financial infrastructure
- **Ghana** (10%) - Growing digital finance hub
- **Uganda** (8%) - Mobile money leader
- **Tanzania** (7%) - Emerging market
- **Rwanda** (5%) - Innovation-friendly regulation
- **Senegal** (4%) - West African francophone hub
- **Ivory Coast** (3%) - WAEMU financial center
- **Ethiopia** (3%) - Large untapped market

### FinTech Categories
- **Mobile Money** (35%) - Dominant in SSA
- **Digital Banking** (20%)
- **Payment Processing** (15%)
- **Lending** (12%)
- **Insurance** (8%)
- **Investment** (5%)
- **Cryptocurrency** (5%)

## Key Variables

### Dependent Variables (Distress Indicators)
- **`distress_flag`** - Binary indicator of company distress (1 = distressed)
- **`failure_imminent`** - Early warning (1 = failure within 2 quarters)
- **`early_warning_signal`** - Composite early warning indicator
- **`regulatory_sanction`** - Regulatory action taken against company

### Financial Performance Metrics
- **Revenue metrics**: `revenue`, `revenue_growth_qoq`, `revenue_volatility`
- **Profitability**: `net_income`, `profitability_ratio`
- **Funding**: `funding_amount`, `cumulative_funding`, `burn_rate`
- **Operating costs**: `operating_costs`

### Operational Metrics
- **User metrics**: `active_users`, `user_growth_qoq`, `churn_rate`
- **Transaction metrics**: `transaction_volume`, `transaction_count`, `avg_transaction_value`
- **Network**: `num_agents` (for mobile money/digital banking)
- **Efficiency**: `customer_acquisition_cost`

### Risk Indicators
- **`risk_score`** - Composite risk metric (0-1 scale)
- **`regulatory_fine`** - Amount of regulatory fines
- **`revenue_volatility`** - Revenue growth volatility
- **Moving averages** for trend analysis

## Usage Instructions

### Loading the Dataset
```python
import pandas as pd

# Load main dataset
df = pd.read_csv('fintech_ssa_distress_dataset.csv')

# Load data dictionary
data_dict = pd.read_csv('fintech_ssa_data_dictionary.csv')

# Convert date columns
df['quarter_date'] = pd.to_datetime(df['quarter_date'])
df['founding_date'] = pd.to_datetime(df['founding_date'])
```

### Example Analysis
```python
# Filter distressed companies
distressed = df[df['distress_flag'] == 1]

# Analyze by country
country_risk = df.groupby('country')['distress_flag'].mean()

# Early warning analysis
early_warnings = df[df['early_warning_signal'] == 1]

# Time series for a specific company
company_ts = df[df['company_id'] == 'FT_NGA_0001'].sort_values('quarter')
```

## Data Generation Methodology

### Realistic Patterns Simulated
1. **Growth trajectories** vary by:
   - Company stage (Seed to Mature)
   - Market size (small/medium/large)
   - Regulatory environment strength
   - FinTech category

2. **Distress patterns** include:
   - Gradual decline over 4-6 quarters before failure
   - Increased churn rates
   - Revenue volatility
   - Funding difficulties

3. **Market dynamics**:
   - Higher growth in large markets (Nigeria, Kenya, South Africa)
   - Category-specific transaction patterns
   - Regulatory events (2% probability per quarter)

### Key Assumptions
- Base failure rates: 35% (Seed), 25% (Series A), 15% (Series B), 8% (Series C+), 5% (Mature)
- Quarterly growth rates: 6-10% for healthy companies
- Churn rates: 2-8% baseline, up to 25% when distressed
- Customer acquisition costs vary by category

## Research Applications

This dataset is suitable for:
1. **Early Warning Model Development**
   - Binary classification models for distress prediction
   - Time series forecasting of failure risk
   - Survival analysis

2. **Risk Factor Analysis**
   - Identifying key predictors of FinTech failure
   - Country-specific risk assessment
   - Category-specific vulnerabilities

3. **Policy Research**
   - Regulatory environment impact analysis
   - Market development studies
   - Financial inclusion research

## Statistical Summary

### Distress Statistics
- Companies experiencing distress: 115 (23.0%)
- Average time to failure: 12 quarters
- Early warning accuracy potential: ~84% of distressed companies show warning signals

### Latest Quarter Metrics (Q20)
- Average Revenue: $108.5M
- Average Active Users: 4.95M
- Average Churn Rate: 7.23%
- Companies with Early Warning Signals: 16.8%

## Limitations and Disclaimers

1. **Synthetic Data**: This is fabricated data for research purposes only
2. **Simplified Patterns**: Real-world dynamics are more complex
3. **Coverage Gaps**: Does not include all SSA countries or FinTech categories
4. **Time Period**: Limited to 5-year window (2019-2023)
5. **Regulatory Simplification**: Regulatory environments are more nuanced in reality

## Reproducibility

The dataset can be regenerated using the provided Python script:
```bash
python3 fintech_ssa_dataset_generator.py
```

The random seed is fixed (42) for reproducibility. Modify the generator parameters to create variations:
- `n_companies`: Number of companies to generate
- `n_quarters`: Number of quarterly observations
- Country weights and categories can be adjusted in the class initialization

## Citation

If using this dataset for research, please cite:
```
FinTech Early Warning Model Dataset for Sub-Saharan Africa
Generated for: Research on FinTech Early Warning Model in Nexus of 
               FinTech Risk in Sub-Sahara Africa Economies
Date: October 2025
Version: 1.0
```

## Contact and Support

This dataset was generated as a research tool for studying FinTech risk in Sub-Saharan Africa. 
For questions about the methodology or to report issues, please refer to the generation script documentation.

---
*Note: This is synthetic data created for academic research and should not be used for actual business decisions or regulatory purposes.*