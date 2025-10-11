# FinTech Nexus Dataset - Category 4
## Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

---

## 📊 Quick Start

Your complete **Category 4: Nexus-Specific & Alternative Data** dataset is ready!

### Generated Files

| File | Description |
|------|-------------|
| **`fintech_nexus_data_category4.csv`** | Main dataset (405 records × 22 variables) |
| **`DATA_DICTIONARY_CATEGORY4.md`** | Complete variable definitions & methodology |
| **`DATASET_SUMMARY.md`** | Statistical summaries & usage guide |
| **`generate_nexus_data.py`** | Python script to regenerate/modify data |

---

## 📈 Dataset Overview

- **Time Period:** Q1 2018 - Q3 2024 (27 quarters)
- **Countries:** 15 Sub-Saharan African economies
- **Total Records:** 405 quarterly observations
- **Variables:** 22 comprehensive measures

### Countries Covered
Nigeria • Kenya • South Africa • Ghana • Uganda • Tanzania • Rwanda • Senegal • Ivory Coast • Zambia • Ethiopia • Botswana • Mozambique • Zimbabwe • Cameroon

---

## 📋 Variable Categories

### 1️⃣ Cyber Risk Exposure (5 variables)
- Total cybersecurity incidents
- Phishing attacks
- Malware incidents  
- Data breaches
- Mobile money fraud search trends (Google Trends)

### 2️⃣ Consumer Sentiment & Trust (4 variables)
- Social media sentiment score (-1 to 1)
- Consumer trust index (0-100)
- Social media mention volume
- Complaint rates per 10,000 transactions

### 3️⃣ Competitive Dynamics (5 variables)
- Herfindahl-Hirschman Index (market concentration)
- New FinTech licenses issued annually
- Total active licenses
- Top 3 firms market share
- New entrants per quarter

### 4️⃣ Additional Nexus Metrics (4 variables)
- Financial inclusion rate (%)
- Mobile money transaction volume (millions)
- Regulatory risk score
- Technology adoption index

### 5️⃣ Identifiers (4 variables)
- Country, Year, Quarter, Date

---

## 🚀 Quick Loading Guide

### Python (pandas)
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('fintech_nexus_data_category4.csv')

# View basic info
print(df.head())
print(df.info())
print(df.describe())

# Filter by country
kenya_data = df[df['country'] == 'Kenya']

# Filter by time period
recent_data = df[df['year'] >= 2022]

# Analyze cyber risk trends
import matplotlib.pyplot as plt
df.groupby('year')['cyber_incidents_total'].mean().plot()
plt.title('Average Cyber Incidents Over Time')
plt.show()
```

### R
```r
# Load the dataset
df <- read.csv('fintech_nexus_data_category4.csv')

# View structure
str(df)
summary(df)

# Filter by country
kenya_data <- df[df$country == 'Kenya', ]

# Plot trends
library(ggplot2)
ggplot(df, aes(x=year, y=cyber_incidents_total, color=country)) +
  geom_line() +
  theme_minimal() +
  labs(title='Cyber Incidents by Country Over Time')
```

### Excel
1. Open Excel
2. Go to **Data → From Text/CSV**
3. Select `fintech_nexus_data_category4.csv`
4. Use PivotTables and charts for analysis

---

## 📊 Sample Analysis Ideas

### 1. Early Warning System Development
```python
# Create binary crisis indicator (high cyber risk)
df['crisis'] = (df['cyber_incidents_total'] > df['cyber_incidents_total'].quantile(0.75)).astype(int)

# Feature selection for prediction
features = ['sentiment_score', 'hhi_index', 'mobile_fraud_search_trend', 
            'complaint_rate_per_10k', 'regulatory_risk_score']
X = df[features]
y = df['crisis']

# Train model (example with Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

### 2. Time Series Forecasting
```python
import statsmodels.api as sm

# Select one country
kenya = df[df['country']=='Kenya'].sort_values('date')

# Fit ARIMA model for cyber incidents
model = sm.tsa.ARIMA(kenya['cyber_incidents_total'], order=(1,1,1))
results = model.fit()

# Forecast next 4 quarters
forecast = results.forecast(steps=4)
print(forecast)
```

### 3. Panel Data Regression
```python
from linearmodels import PanelOLS

# Set multi-index for panel data
df_panel = df.set_index(['country', 'date'])

# Run fixed effects regression
model = PanelOLS.from_formula(
    'sentiment_score ~ cyber_incidents_total + hhi_index + EntityEffects',
    data=df_panel
)
results = model.fit()
print(results)
```

### 4. Clustering Countries by Risk Profile
```python
from sklearn.cluster import KMeans

# Aggregate by country
country_profiles = df.groupby('country').mean()

# Select risk variables
risk_vars = ['cyber_incidents_total', 'sentiment_score', 'hhi_index', 
             'complaint_rate_per_10k', 'regulatory_risk_score']

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
country_profiles['cluster'] = kmeans.fit_predict(country_profiles[risk_vars])

print(country_profiles[['cluster'] + risk_vars])
```

---

## ⚠️ Important Notes

### This is Synthetic Data
- ✅ **Use for:** Academic research, methodology development, model testing
- ❌ **Don't use for:** Real-world policy decisions, investment strategies, regulatory compliance

### Data Quality
- Realistic time trends and cross-country variation
- Logical correlations between variables
- Random shock events simulating crises
- Based on industry knowledge and academic literature

### Reproducibility
To regenerate with different parameters:
```bash
python3 generate_nexus_data.py
```

Edit the script to:
- Change baseline values
- Adjust growth rates
- Modify shock probabilities
- Add/remove countries
- Extend time period

---

## 📚 Documentation

For detailed information, see:
- **`DATA_DICTIONARY_CATEGORY4.md`** - Complete variable definitions, sources, and interpretation
- **`DATASET_SUMMARY.md`** - Statistical summaries, sample analyses, and use cases

---

## 🎯 Thesis Integration

### Recommended Approach

1. **Exploratory Data Analysis**
   - Visualize trends across countries
   - Identify patterns and anomalies
   - Calculate correlations

2. **Feature Engineering**
   - Create lag variables (t-1, t-2, t-3)
   - Rolling averages (3-quarter, 4-quarter)
   - Derive composite risk indices

3. **Model Development**
   - Binary classification (crisis/no crisis)
   - Multi-class risk levels (low/medium/high)
   - Time series forecasting
   - Panel data analysis

4. **Validation & Interpretation**
   - Cross-validation
   - Out-of-sample testing
   - Feature importance analysis
   - Policy recommendations

---

## 📈 Key Statistics

### Cyber Risk (Mean Values)
- Cyber Incidents: 51.4 per quarter
- Phishing Attacks: 19.7 per quarter
- Data Breaches: 7.7 per quarter
- Mobile Fraud Search Trend: 83.1/100

### Market Structure
- Average HHI: 0.19 (moderate competition)
- Active Licenses: 24.4 per country
- New Entrants: 4.5 per quarter

### Consumer Metrics
- Sentiment Score: 0.08 (slightly positive)
- Trust Index: 54.1/100
- Complaint Rate: 45.3 per 10k transactions

---

## 🔄 Updates & Modifications

Need to adjust the dataset?

1. Open `generate_nexus_data.py`
2. Modify parameters:
   - Line 11: Change random seed
   - Lines 14-18: Add/remove countries
   - Lines 41-56: Adjust baseline values
   - Lines 58-62: Modify growth trends
3. Regenerate: `python3 generate_nexus_data.py`

---

## 💡 Tips for Success

### For Your Thesis

1. **Always disclose** this is synthetic data in your methodology section
2. **Validate** your early warning model framework with this data
3. **Compare** results with real-world case studies where available
4. **Discuss limitations** of synthetic data in your analysis
5. **Emphasize** the methodological contribution rather than empirical findings

### Data Analysis Best Practices

1. Start with **exploratory visualizations**
2. Check for **multicollinearity** among predictors
3. Consider **lag structures** (financial crises don't happen instantly)
4. Use **cross-validation** to prevent overfitting
5. Test **multiple model specifications**

---

## 📧 Questions?

Review the comprehensive documentation:
- Variable definitions → `DATA_DICTIONARY_CATEGORY4.md`
- Statistical summaries → `DATASET_SUMMARY.md`  
- Generation code → `generate_nexus_data.py`

---

## ✅ Dataset Validation Checklist

- ✓ 405 records covering 15 countries
- ✓ 27 quarters (2018 Q1 - 2024 Q3)
- ✓ 22 variables across 4 categories
- ✓ No missing values
- ✓ Realistic distributions and trends
- ✓ Logical correlations between variables
- ✓ Country-specific characteristics preserved
- ✓ Time trends and seasonality incorporated

---

## 🎓 Citation

```
FinTech Nexus Risk Dataset - Category 4 (2025)
Sub-Saharan Africa: Nexus-Specific & Alternative Data
Synthetic dataset for thesis: "Research on FinTech Early Warning Model 
in Nexus of Fintech Risk in Sub-Sahara Africa Economies"
Generated: October 2025
```

---

**Good luck with your thesis research! 🚀**

*This dataset provides a solid foundation for developing and testing your FinTech early warning model methodology.*
