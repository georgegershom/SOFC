# FinTech Nexus Dataset - Category 4 Summary Report

## Dataset Information
**Generated:** October 11, 2025  
**Thesis Topic:** Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies  
**Category:** Nexus-Specific & Alternative Data  

---

## Files Generated

| File | Description | Size |
|------|-------------|------|
| `fintech_nexus_data_category4.csv` | Main dataset with all variables | 405 rows × 22 columns |
| `DATA_DICTIONARY_CATEGORY4.md` | Complete data dictionary and methodology | Comprehensive documentation |
| `generate_nexus_data.py` | Python script to regenerate data | Reproducible code |

---

## Dataset Overview

### Coverage
- **Countries:** 15 Sub-Saharan African economies
- **Time Period:** Q1 2018 - Q3 2024 (27 quarters)
- **Total Observations:** 405 quarterly records
- **Variables:** 22 comprehensive measures

### Countries Included
1. Nigeria
2. Kenya
3. South Africa
4. Ghana
5. Uganda
6. Tanzania
7. Rwanda
8. Senegal
9. Ivory Coast
10. Zambia
11. Ethiopia
12. Botswana
13. Mozambique
14. Zimbabwe
15. Cameroon

---

## Variable Categories

### 1. Cyber Risk Exposure (5 variables)
- `cyber_incidents_total` - Total cybersecurity incidents
- `phishing_attacks` - Phishing attack count
- `malware_incidents` - Malware-related incidents
- `data_breaches` - Confirmed data breaches
- `mobile_fraud_search_trend` - Google Trends for "mobile money fraud"

### 2. Consumer Sentiment & Trust (4 variables)
- `sentiment_score` - Social media sentiment analysis (-1 to 1)
- `trust_index` - Consumer trust index (0-100)
- `social_media_mentions` - Volume of FinTech brand mentions
- `complaint_rate_per_10k` - Complaints per 10,000 transactions

### 3. Competitive Dynamics (5 variables)
- `hhi_index` - Herfindahl-Hirschman Index (market concentration)
- `new_licenses_issued_annual` - New FinTech licenses per year
- `total_active_licenses` - Total active licenses
- `top3_market_share_pct` - Top 3 firms' market share
- `new_entrants_quarter` - New market entrants per quarter

### 4. Additional Nexus Metrics (4 variables)
- `financial_inclusion_rate` - % population with financial access
- `transaction_volume_millions` - Mobile money volume (millions)
- `regulatory_risk_score` - Regulatory uncertainty score
- `tech_adoption_index` - Technology readiness index

### 5. Identifier Variables (4 variables)
- `country` - Country name
- `year` - Calendar year
- `quarter` - Quarter (1-4)
- `date` - Formatted date (YYYY-Qq)

---

## Key Statistical Summaries

### Cyber Risk Variables
| Variable | Mean | Std Dev | Min | Max |
|----------|------|---------|-----|-----|
| Cyber Incidents Total | 62.3 | 39.8 | 5 | 250+ |
| Phishing Attacks | 24.1 | 15.7 | 2 | 100+ |
| Malware Incidents | 18.4 | 12.1 | 1 | 80+ |
| Data Breaches | 9.3 | 6.2 | 0 | 40+ |
| Mobile Fraud Search Trend | 64.2 | 16.8 | 20 | 100 |

**Trend:** All cyber risk metrics show upward trends from 2018-2024, reflecting increased digitalization and threat sophistication.

### Consumer Sentiment Variables
| Variable | Mean | Std Dev | Min | Max |
|----------|------|---------|-----|-----|
| Sentiment Score | 0.28 | 0.18 | -0.3 | 0.8 |
| Trust Index | 64.1 | 9.1 | 35 | 90 |
| Social Media Mentions | 2,104 | 1,256 | 500 | 6,000 |
| Complaint Rate (per 10k) | 38.5 | 11.2 | 5 | 85 |

**Trend:** Sentiment remains moderately positive but shows volatility linked to cyber incident spikes. Trust levels correlate inversely with complaint rates.

### Competitive Dynamics Variables
| Variable | Mean | Std Dev | Min | Max |
|----------|------|---------|-----|-----|
| HHI Index | 0.28 | 0.08 | 0.10 | 0.60 |
| Total Active Licenses | 42.7 | 24.3 | 5 | 140 |
| New Entrants (Quarter) | 4.8 | 2.6 | 0 | 14 |
| Top 3 Market Share % | 26.5 | 8.9 | 10 | 65 |

**Trend:** Market concentration (HHI) decreasing over time, indicating growing competition. License issuance and new entrants both increasing.

### Additional Nexus Metrics
| Variable | Mean | Std Dev | Min | Max |
|----------|------|---------|-----|-----|
| Financial Inclusion Rate % | 58.3 | 12.8 | 30 | 95 |
| Transaction Volume (M) | 102.7 | 56.4 | 40 | 500+ |
| Regulatory Risk Score | 38.4 | 11.7 | 17 | 61 |
| Tech Adoption Index | 60.8 | 10.3 | 38 | 84 |

**Trend:** Strong growth in financial inclusion and transaction volumes. Technology adoption steadily improving across all countries.

---

## Data Quality Features

### ✓ Realistic Patterns
- Time trends reflecting FinTech sector growth
- Seasonal variations in transaction volumes
- Random shock events (10% probability) simulating crises
- Cross-country heterogeneity based on development levels

### ✓ Logical Correlations
- Cyber incidents → negative sentiment
- Higher competition → lower HHI
- Tech adoption → financial inclusion
- Complaint rates ↔ trust levels

### ✓ Country-Specific Characteristics
- **High Development:** South Africa, Nigeria, Kenya (1.5× multiplier)
- **Moderate Development:** Ghana, Rwanda, Tanzania (1.0× baseline)
- **Emerging Markets:** Ethiopia, Zimbabwe (baseline with higher HHI)

---

## Sample Data: Kenya (First 8 Quarters)

| Date | Cyber Incidents | Sentiment | HHI | Trust Index | Transaction Vol (M) |
|------|----------------|-----------|-----|-------------|-------------------|
| 2018-Q1 | 48 | 0.087 | 0.184 | 54 | 88 |
| 2018-Q2 | 39 | 0.024 | 0.166 | 51 | 108 |
| 2018-Q3 | 62 | -0.214 | 0.151 | 39 | 103 |
| 2018-Q4 | 31 | 0.044 | 0.119 | 52 | 114 |
| 2019-Q1 | 51 | 0.188 | 0.145 | 59 | 121 |
| 2019-Q2 | 48 | 0.271 | 0.131 | 63 | 140 |
| 2019-Q3 | 57 | 0.249 | 0.115 | 62 | 92 |
| 2019-Q4 | 59 | 0.203 | 0.110 | 60 | 87 |

**Observations:**
- Kenya shows high FinTech activity (major hub)
- Decreasing HHI indicates growing competition
- Q3 2018 shows negative sentiment spike (possible fraud wave)
- Transaction volumes trending upward

---

## Suggested Use Cases

### 1. Early Warning System Development
- **Binary Classification:** Predict crisis vs. normal periods
- **Risk Scoring:** Develop composite risk indices
- **Lead Indicators:** Identify predictive variables

### 2. Time Series Forecasting
- **ARIMA/SARIMA:** Model cyber incident trends
- **VAR Models:** Analyze multivariate relationships
- **Prophet:** Forecast transaction volumes with seasonality

### 3. Panel Data Analysis
- **Fixed Effects:** Control for country-specific factors
- **Random Effects:** Cross-country comparisons
- **Difference-in-Differences:** Policy impact evaluation

### 4. Machine Learning Models
- **Random Forest:** Feature importance for risk prediction
- **XGBoost:** Gradient boosting for early warning
- **LSTM Networks:** Deep learning for time series
- **Clustering:** Identify country risk profiles

### 5. Network & Contagion Analysis
- **Correlation Networks:** Identify interconnected risks
- **Spillover Effects:** Cross-country transmission
- **VAR/VECM:** Dynamic relationships

---

## Important Disclaimers

### ⚠️ Synthetic Data
This dataset is **synthetically generated** for academic purposes. While based on realistic parameters and industry knowledge, it does not represent actual recorded data.

### ✓ Appropriate Uses
- Thesis methodology development
- Model testing and validation
- Proof-of-concept analyses
- Academic research and learning

### ✗ Inappropriate Uses
- Real-world policy decisions
- Investment or risk management
- Publication without synthetic data disclaimer
- Regulatory compliance reporting

---

## Reproducibility

To regenerate the dataset:
```bash
python3 generate_nexus_data.py
```

To modify parameters:
1. Edit `generate_nexus_data.py`
2. Adjust baseline values, growth factors, or shock probabilities
3. Change random seed for different realizations
4. Regenerate dataset

---

## Major FinTech Brands by Country

### East Africa
- **Kenya:** M-Pesa, M-Shwari, Tala, Branch, Cellulant
- **Tanzania:** M-Pesa, Tigo Pesa, Airtel Money, HaloPesa
- **Uganda:** MTN Mobile Money, Airtel Money, Chipper Cash, Xente
- **Rwanda:** MTN Mobile Money, Airtel Money, Chipper Cash, IREMBO
- **Ethiopia:** M-Birr, HelloCash, Amole, Kacha Digital

### West Africa
- **Nigeria:** Flutterwave, Paystack, OPay, PalmPay, Kuda
- **Ghana:** MTN MoMo, Zeepay, ExpressPay, Slydepay, hubtel
- **Senegal:** Orange Money, Wave, Free Money, E-Money
- **Ivory Coast:** Orange Money, MTN Mobile Money, Moov Money, Wave
- **Cameroon:** Orange Money, MTN Mobile Money, Express Union Mobile

### Southern Africa
- **South Africa:** Yoco, TymeBank, Discovery Bank, Luno, Zapper
- **Zambia:** MTN Mobile Money, Airtel Money, Zoona, Kazang
- **Botswana:** Orange Money, MyZaka, Smega, BluePay
- **Zimbabwe:** EcoCash, OneMoney, Telecash, ZimSwitch
- **Mozambique:** M-Pesa, Mkesh, e-Mola, PagaLu

---

## Next Steps

### For Your Thesis

1. **Data Exploration**
   - Load CSV into Python/R
   - Perform exploratory data analysis
   - Visualize trends and patterns

2. **Feature Engineering**
   - Create lag variables
   - Calculate rolling averages
   - Derive risk indices

3. **Model Development**
   - Split train/test sets
   - Select relevant features
   - Train early warning models
   - Validate predictions

4. **Results & Analysis**
   - Interpret model outputs
   - Identify key risk drivers
   - Develop policy recommendations

### Additional Data Sources

To complement this dataset, consider collecting:
- **World Bank:** GDP, inflation, poverty rates
- **IMF:** Financial stability indicators
- **GSMA:** Mobile penetration, connectivity
- **Central Banks:** Payment system statistics
- **Regulatory Agencies:** Actual license data

---

## Support & Questions

For technical questions or to report issues:
- Review `DATA_DICTIONARY_CATEGORY4.md` for detailed documentation
- Check `generate_nexus_data.py` for implementation details
- Modify parameters to suit your research needs

---

**Citation:**
```
FinTech Nexus Risk Dataset - Category 4 (2025)
Sub-Saharan Africa FinTech Early Warning System Data
Synthetic dataset covering 15 countries, 2018-2024
Generated for academic research purposes
```

---

*Generated: October 11, 2025*  
*Version: 1.0*  
*Status: Ready for Analysis*
