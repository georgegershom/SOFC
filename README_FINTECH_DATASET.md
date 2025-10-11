# FinTech Early Warning Model Dataset for Sub-Sahara Africa

## 📊 Overview

This comprehensive synthetic dataset has been generated to support research on **FinTech Early Warning Models in Sub-Sahara Africa Economies**. The dataset contains quarterly time-series data for 150 FinTech companies operating across 10 SSA countries from Q1 2020 to Q4 2023.

**Research Topic**: Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

## 🎯 Dataset Purpose

This dataset enables the development and testing of early warning systems that can predict FinTech company distress or failure using a combination of:
- Financial performance indicators
- Operational metrics
- Regulatory environment factors
- Country-level economic indicators

## 📁 Files Generated

| File | Description | Use Case |
|------|-------------|----------|
| `fintech_ssa_dataset.csv` | Main dataset with all variables | Primary data for analysis and modeling |
| `fintech_ssa_dataset.xlsx` | Excel format with multiple sheets | Easy exploration, includes summary sheets |
| `fintech_ssa_dataset.json` | JSON format | API integration, web applications |
| `fintech_company_profiles.csv` | Company-level metadata | Company characteristics analysis |
| `dataset_summary.json` | Statistical summary | Quick overview of dataset characteristics |
| `data_dictionary.md` | Complete variable definitions | Reference guide for all variables |
| `generate_fintech_ssa_dataset.py` | Data generation script | Reproduce or modify the dataset |

## 📈 Dataset Statistics

- **Total Records**: 2,400 (150 companies × 16 quarters)
- **Time Period**: Q1 2020 - Q4 2023 (16 quarters)
- **Countries**: 10 Sub-Sahara African nations
- **FinTech Types**: 8 different categories
- **Failed Companies**: ~8% of companies experience failure
- **Companies in Distress**: Multiple quarters with distress signals

### Countries Covered
- 🇳🇬 Nigeria (largest market)
- 🇰🇪 Kenya (strongest M-Pesa presence)
- 🇿🇦 South Africa (most developed regulatory framework)
- 🇬🇭 Ghana
- 🇺🇬 Uganda
- 🇹🇿 Tanzania
- 🇷🇼 Rwanda
- 🇸🇳 Senegal
- 🇿🇲 Zambia
- 🇪🇹 Ethiopia

### FinTech Categories
1. **Mobile Money** (e.g., M-Pesa, MTN Mobile Money, Airtel Money)
2. **Payment Gateway** (e.g., Paystack, Flutterwave)
3. **Digital Lending**
4. **Remittance Services**
5. **Insurance Tech (InsurTech)**
6. **Investment Platforms**
7. **Digital Banking**
8. **Crypto Exchanges**

## 🔑 Key Variables

### Dependent Variables (Target for Prediction)

1. **`fintech_failure`** (Binary: 0/1)
   - Primary target variable
   - 1 = Company has failed/become insolvent/closed
   - 0 = Company is operational
   - Use this to build early warning models

2. **`fintech_distress`** (Binary: 0/1)
   - Leading indicator of potential failure
   - 1 = Company shows multiple distress signals
   - Useful for intermediate predictions

3. **`regulatory_sanction`** (Binary: 0/1)
   - 1 = Company received regulatory action (fine, suspension, warning)
   - Can be both a predictor and outcome

### Independent Variables

#### Financial Performance Metrics
- `revenue_usd` - Quarterly revenue
- `revenue_growth_pct` - Year-over-year growth
- `net_income_usd` - Profitability
- `profit_margin_pct` - Efficiency indicator
- `burn_rate_usd` - Cash burn (critical for startups)
- `funding_amount_usd` - Capital raised
- `funding_stage` - Investment maturity

#### Operational Metrics
- `active_users` - Customer base size
- `user_growth_pct` - User acquisition rate
- `transaction_volume_usd` - Total value processed
- `transaction_count` - Transaction frequency
- `avg_transaction_value_usd` - Transaction size
- `num_agents` - Agent network (for mobile money)
- `customer_acquisition_cost_usd` - CAC
- `customer_churn_rate_pct` - Retention indicator

#### Contextual Variables
- `country_market_size_index` - Market opportunity
- `country_regulatory_strength_index` - Regulatory environment
- `country_economic_stability_index` - Economic conditions

## 🚀 Quick Start

### Loading the Data

**Python (Pandas)**
```python
import pandas as pd

# Load main dataset
df = pd.read_csv('fintech_ssa_dataset.csv')

# Load company profiles
companies = pd.read_csv('fintech_company_profiles.csv')

# View summary
print(df.info())
print(df.head())
```

**R**
```r
# Load main dataset
df <- read.csv('fintech_ssa_dataset.csv')

# Load company profiles
companies <- read.csv('fintech_company_profiles.csv')

# View structure
str(df)
head(df)
```

**Excel**
Simply open `fintech_ssa_dataset.xlsx` with multiple sheets:
- Sheet 1: Full Dataset
- Sheet 2: Company Summary
- Sheet 3: Failed Companies

### Basic Analysis Examples

**1. Failure Rate by Country**
```python
failure_by_country = df.groupby('country')['fintech_failure'].mean() * 100
print(failure_by_country.sort_values(ascending=False))
```

**2. Average Metrics for Failed vs. Successful Companies**
```python
comparison = df.groupby('fintech_failure').agg({
    'revenue_usd': 'mean',
    'active_users': 'mean',
    'customer_churn_rate_pct': 'mean',
    'profit_margin_pct': 'mean'
})
print(comparison)
```

**3. Identify Companies Showing Distress**
```python
distressed = df[df['fintech_distress'] == 1]
print(f"Companies in distress: {distressed['company_id'].nunique()}")
```

## 🎓 Suggested Research Applications

### 1. Early Warning Model Development
Build predictive models to forecast FinTech failure 1-4 quarters in advance:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Select features
features = ['revenue_growth_pct', 'profit_margin_pct', 'burn_rate_usd', 
            'active_users', 'customer_churn_rate_pct', 'user_growth_pct']

# Create lagged features (use previous quarter to predict current)
df['target'] = df.groupby('company_id')['fintech_failure'].shift(-1)

# Build model
X = df[features].dropna()
y = df.loc[X.index, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 2. Survival Analysis
Analyze time-to-failure using survival analysis:
```python
from lifelines import CoxPHFitter

# Prepare survival data
survival_df = df.groupby('company_id').agg({
    'quarter': 'max',  # Duration
    'fintech_failure': 'max',  # Event occurred
    'revenue_growth_pct': 'mean',
    'profit_margin_pct': 'mean'
}).reset_index()

cph = CoxPHFitter()
cph.fit(survival_df, duration_col='quarter', event_col='fintech_failure')
cph.print_summary()
```

### 3. Cluster Analysis
Identify groups of companies with similar risk profiles:
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
cluster_features = ['revenue_usd', 'active_users', 'transaction_volume_usd', 
                    'profit_margin_pct', 'customer_churn_rate_pct']

# Normalize and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[cluster_features].dropna())

kmeans = KMeans(n_clusters=4, random_state=42)
df['risk_cluster'] = kmeans.fit_predict(X_scaled)
```

### 4. Time Series Forecasting
Predict future performance metrics:
```python
from statsmodels.tsa.arima.model import ARIMA

# Select one company
company_data = df[df['company_id'] == 'FT001'].set_index('date')

# Forecast revenue
model = ARIMA(company_data['revenue_usd'], order=(1,1,1))
results = model.fit()
forecast = results.forecast(steps=4)
```

### 5. Country-Level Analysis
Compare FinTech ecosystems across SSA countries:
```python
country_analysis = df.groupby('country').agg({
    'fintech_failure': 'mean',
    'revenue_usd': 'mean',
    'active_users': 'sum',
    'transaction_volume_usd': 'sum',
    'regulatory_sanction': 'sum'
})
```

## 📊 Feature Engineering Suggestions

To enhance model performance, consider creating:

1. **Trend Variables**
   ```python
   # 3-quarter moving averages
   df['revenue_ma3'] = df.groupby('company_id')['revenue_usd'].transform(
       lambda x: x.rolling(3, min_periods=1).mean()
   )
   ```

2. **Change Variables**
   ```python
   # Quarter-over-quarter changes
   df['revenue_qoq_change'] = df.groupby('company_id')['revenue_usd'].pct_change()
   ```

3. **Ratio Variables**
   ```python
   # Burn rate to revenue ratio
   df['burn_revenue_ratio'] = df['burn_rate_usd'] / df['revenue_usd']
   ```

4. **Lag Variables**
   ```python
   # Previous quarter metrics
   df['revenue_lag1'] = df.groupby('company_id')['revenue_usd'].shift(1)
   df['revenue_lag2'] = df.groupby('company_id')['revenue_usd'].shift(2)
   ```

5. **Composite Scores**
   ```python
   # Financial health score
   df['financial_health'] = (
       df['profit_margin_pct'] * 0.3 +
       df['revenue_growth_pct'] * 0.3 +
       (100 - df['customer_churn_rate_pct']) * 0.2 +
       df['user_growth_pct'] * 0.2
   )
   ```

## ⚠️ Important Considerations

### Data Characteristics
1. **Synthetic Data**: This is fabricated data designed for research. While realistic, it does not represent actual companies.
2. **Realistic Relationships**: The data includes authentic correlations (e.g., failed companies show deteriorating metrics before failure).
3. **No Missing Values**: Real-world data would have substantial missing values, especially for private companies.
4. **Temporal Structure**: Data is quarterly time series - use appropriate modeling techniques.

### Modeling Considerations
1. **Class Imbalance**: Failures are relatively rare (~8%) - use appropriate techniques:
   - SMOTE for oversampling
   - Class weights in models
   - Precision-recall metrics instead of accuracy

2. **Temporal Split**: Always split data by time, not randomly:
   ```python
   train = df[df['quarter'] <= 12]  # First 3 years
   test = df[df['quarter'] > 12]    # Last year
   ```

3. **Panel Data Structure**: Consider company-level fixed effects:
   ```python
   from linearmodels import PanelOLS
   df_panel = df.set_index(['company_id', 'quarter'])
   model = PanelOLS.from_formula('revenue_usd ~ active_users + EntityEffects', df_panel)
   ```

4. **Country Clustering**: Standard errors should account for country-level clustering.

## 🔄 Regenerating or Modifying the Dataset

To create a new version or adjust parameters:

```bash
# Edit the parameters in the script
nano generate_fintech_ssa_dataset.py

# Regenerate
python3 generate_fintech_ssa_dataset.py
```

Key parameters you can modify:
- `NUM_COMPANIES`: Number of FinTech firms (default: 150)
- `NUM_QUARTERS`: Time periods (default: 16)
- `START_DATE`: Beginning of observation period
- Add/remove countries or FinTech types

## 📚 Data Sources Emulated

This synthetic dataset emulates real-world data sources:

1. **Company Reports**: Safaricom (M-Pesa), MTN Mobile Money, Flutterwave, Paystack annual reports
2. **Regulatory Bodies**: 
   - Central Bank of Nigeria Payment System Statistics
   - Central Bank of Kenya National Payment System Reports
   - South African Reserve Bank Payment System Reports
3. **VC Databases**: Crunchbase, Disrupt Africa, Partech Africa Reports
4. **Industry Reports**: 
   - GSMA State of the Industry Report on Mobile Money
   - World Bank Financial Inclusion Reports
   - IMF FinTech Notes
5. **App Store Data**: Google Play/Apple App Store download statistics

## 📖 Citation

If you use this dataset in your research, please cite:

```
FinTech Early Warning Model Dataset for Sub-Sahara Africa (2024)
Research Topic: Research on FinTech Early Warning Model in Nexus of Fintech Risk 
in Sub-Sahara Africa Economies
Dataset Type: Synthetic/Fabricated for Research Purposes
Generated: October 2025
```

## 🤝 Support & Questions

For questions about:
- **Variable Definitions**: See `data_dictionary.md`
- **Data Generation**: Review `generate_fintech_ssa_dataset.py`
- **Summary Statistics**: Check `dataset_summary.json`

## 📝 License

This synthetic dataset is provided for academic and research purposes. Feel free to use, modify, and share with appropriate attribution.

---

**Happy Researching! 🚀📊**

*Building better early warning systems for sustainable FinTech growth in Africa*
