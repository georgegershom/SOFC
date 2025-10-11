# FinTech Nexus Data - Category 4: Data Dictionary

## Dataset Overview
**File:** `fintech_nexus_data_category4.csv`  
**Research Topic:** FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies  
**Category:** Nexus-Specific & Alternative Data  
**Time Period:** 2018 Q1 - 2024 Q3  
**Countries Covered:** 15 Sub-Saharan African economies  
**Total Records:** 405 quarterly observations  
**Generated:** October 2025

---

## Geographic Coverage
The dataset covers the following Sub-Saharan African countries with significant FinTech presence:

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

## Variable Definitions

### Identifier Variables

| Variable | Type | Description |
|----------|------|-------------|
| `country` | String | Name of the Sub-Saharan African country |
| `year` | Integer | Calendar year |
| `quarter` | Integer | Quarter of the year (1-4) |
| `date` | String | Date in format YYYY-Qq (e.g., 2024-Q3) |

---

### Category 1: Cyber Risk Exposure

These variables capture the cybersecurity threats and digital fraud risks facing the FinTech sector.

| Variable | Type | Range | Description | Sources |
|----------|------|-------|-------------|---------|
| `cyber_incidents_total` | Integer | 5-200+ | Total number of cybersecurity incidents reported in the financial sector per quarter | National CERTs, industry reports, financial regulators |
| `phishing_attacks` | Integer | 2-100+ | Number of phishing attacks targeting financial institutions and FinTech platforms | Cybersecurity incident reports |
| `malware_incidents` | Integer | 1-80+ | Number of malware-related security incidents affecting financial services | Security operations centers, threat intelligence |
| `data_breaches` | Integer | 0-40+ | Number of confirmed data breaches in the financial sector | Data protection authorities, media reports |
| `mobile_fraud_search_trend` | Integer | 0-100 | Google Trends search volume for "mobile money fraud" (normalized 0-100 scale) | Google Trends API |

**Key Insights:**
- Higher values indicate greater cyber risk exposure
- Trending upward from 2018-2024 across all countries
- Significant spikes occur during major fraud waves or cyber attack campaigns
- More economically developed countries (SA, Nigeria, Kenya) show higher absolute numbers due to larger digital ecosystems

---

### Category 2: Consumer Sentiment & Trust

These variables measure public perception and confidence in FinTech services.

| Variable | Type | Range | Description | Sources |
|----------|------|-------|-------------|---------|
| `sentiment_score` | Float | -1.0 to 1.0 | Social media sentiment analysis score for major FinTech brands (-1=very negative, 0=neutral, 1=very positive) | Twitter/X API, sentiment analysis algorithms |
| `trust_index` | Integer | 0-100 | Consumer trust index derived from sentiment scores (0=no trust, 100=complete trust) | Survey data, social media analysis |
| `social_media_mentions` | Integer | 500-5000+ | Volume of social media mentions about major FinTech brands per quarter | Social media monitoring tools |
| `complaint_rate_per_10k` | Integer | 1-100+ | Consumer complaints per 10,000 transactions | Consumer protection agencies, FinTech platforms |

**Major FinTech Brands by Country:**
- **Nigeria:** Flutterwave, Paystack, OPay, PalmPay, Kuda
- **Kenya:** M-Pesa, M-Shwari, Tala, Branch, Cellulant
- **South Africa:** Yoco, TymeBank, Discovery Bank, Luno, Zapper
- **Ghana:** MTN MoMo, Zeepay, ExpressPay, Slydepay, hubtel
- *(See full list in generation script)*

**Key Insights:**
- Sentiment scores generally range from -0.2 to 0.6 (slightly negative to moderately positive)
- Trust index transformation: `(sentiment_score + 1) * 50`
- Negative spikes correlate with cyber incidents and fraud waves
- Complaint rates inversely related to sentiment scores

---

### Category 3: Competitive Dynamics

These variables capture market structure and competitive intensity in the FinTech sector.

| Variable | Type | Range | Description | Sources |
|----------|------|-------|-------------|---------|
| `hhi_index` | Float | 0.10-0.70 | Herfindahl-Hirschman Index measuring market concentration (0=perfect competition, 1=monopoly) | Central bank data, market research reports |
| `new_licenses_issued_annual` | Integer | 0-50+ | Number of new FinTech licenses issued in the year (recorded in Q4 only) | Financial regulators, central banks |
| `total_active_licenses` | Integer | 5-150+ | Total number of active FinTech operating licenses | Regulatory databases |
| `top3_market_share_pct` | Float | 10-70% | Combined market share of top 3 FinTech firms | Industry reports, market analysis |
| `new_entrants_quarter` | Integer | 0-15+ | Number of new FinTech companies entering the market per quarter | Business registrations, regulatory filings |

**HHI Interpretation:**
- **< 0.15:** Highly competitive market
- **0.15-0.25:** Unconcentrated market
- **0.25-0.50:** Moderate concentration
- **> 0.50:** High concentration

**Key Insights:**
- HHI generally decreasing over time (market becoming more competitive)
- Significant variation across countries (Ethiopia, Zimbabwe more concentrated)
- Licensing regimes maturing, with more structured issuance processes
- New entrants increasing as regulatory frameworks stabilize

---

### Category 4: Additional Nexus Metrics

These supplementary variables provide context for the nexus-specific data.

| Variable | Type | Range | Description | Sources |
|----------|------|-------|-------------|---------|
| `financial_inclusion_rate` | Integer | 30-95% | Percentage of adult population with access to formal financial services | World Bank Findex, central banks |
| `transaction_volume_millions` | Integer | 40-500+ | Total mobile money transaction volume in millions per quarter | Payment system operators, central banks |
| `regulatory_risk_score` | Integer | 17-61 | Assessment of regulatory uncertainty and compliance risk (0=low risk, 100=high risk) | Expert assessments, regulatory change index |
| `tech_adoption_index` | Integer | 38-84 | Technology adoption and digital readiness index (0-100 scale) | ITU, GSMA, national statistics |

**Key Insights:**
- Financial inclusion steadily increasing across all countries
- Transaction volumes show strong growth and seasonal patterns
- Regulatory risk scores inversely correlated with sentiment
- Tech adoption index shows convergence across countries

---

## Data Generation Methodology

### Synthetic Data Approach
This dataset is **synthetically generated** using realistic parameters based on:
1. **Industry Reports:** World Bank, IMF, GSMA, various FinTech industry reports
2. **Academic Literature:** Research papers on African FinTech markets
3. **Public Data Sources:** Google Trends patterns, regulatory announcements
4. **Expert Knowledge:** Understanding of regional FinTech dynamics

### Statistical Properties
- **Time Trends:** Incorporated gradual growth patterns reflecting FinTech sector expansion
- **Seasonal Effects:** Quarterly variations in transaction volumes and consumer behavior
- **Shock Events:** Random spikes representing cyber attacks, fraud waves, regulatory changes (10% probability per quarter)
- **Cross-Country Variation:** Different baseline parameters for each country based on economic development
- **Correlations:** Logical relationships between variables (e.g., cyber incidents → negative sentiment)

### Baseline Factors
1. **Economic Development Factor:**
   - South Africa, Nigeria, Kenya: 1.5x multiplier
   - Other countries: 1.0x baseline
   
2. **Time Factor:**
   - Compound growth: 5% per quarter
   
3. **Random Variation:**
   - Most variables: ±20-30% random noise
   - Maintains realistic uncertainty while preserving trends

---

## Data Quality & Limitations

### Strengths
✓ Comprehensive coverage of nexus-specific variables  
✓ Realistic time trends and cross-country variation  
✓ Logical correlations between related variables  
✓ Quarterly granularity for time-series analysis  
✓ 6+ years of historical data  

### Limitations
⚠ **Synthetic Data:** Not based on actual recorded incidents or transactions  
⚠ **Simplified Relationships:** Real-world dynamics are more complex  
⚠ **Data Availability:** Some variables (e.g., exact HHI) are difficult to obtain in reality  
⚠ **Country Coverage:** Limited to 15 countries, may not represent all SSA  
⚠ **Lagging Indicators:** Some real-world data may lag by 1-2 quarters  

### Recommended Usage
This dataset is suitable for:
- **Academic Research:** Thesis development, methodology testing, model validation
- **Preliminary Analysis:** Exploratory data analysis, variable selection
- **Model Development:** Training early warning models, testing algorithms
- **Proof of Concept:** Demonstrating analytical approaches

**NOT recommended for:**
- Policy decisions based solely on this data
- Publication without disclaimer about synthetic nature
- Real-world investment or risk management decisions

---

## Statistical Summary

### Cyber Risk Variables
```
cyber_incidents_total:     mean=62.3,  std=39.8,  range=[5, 250]
phishing_attacks:          mean=24.1,  std=15.7,  range=[2, 100]
malware_incidents:         mean=18.4,  std=12.1,  range=[1, 80]
data_breaches:             mean=9.3,   std=6.2,   range=[0, 40]
mobile_fraud_search_trend: mean=64.2,  std=16.8,  range=[20, 100]
```

### Sentiment Variables
```
sentiment_score:           mean=0.28,  std=0.18,  range=[-0.3, 0.8]
trust_index:               mean=64.1,  std=9.1,   range=[35, 90]
social_media_mentions:     mean=2104,  std=1256,  range=[500, 6000]
complaint_rate_per_10k:    mean=38.5,  std=11.2,  range=[5, 85]
```

### Competitive Variables
```
hhi_index:                 mean=0.28,  std=0.08,  range=[0.10, 0.60]
total_active_licenses:     mean=42.7,  std=24.3,  range=[5, 140]
new_entrants_quarter:      mean=4.8,   std=2.6,   range=[0, 14]
top3_market_share_pct:     mean=26.5,  std=8.9,   range=[10, 65]
```

---

## Suggested Analysis Approaches

### 1. Time Series Analysis
- Trend analysis for cyber risk indicators
- Seasonal decomposition of transaction volumes
- Autoregressive models for sentiment forecasting

### 2. Panel Data Analysis
- Fixed effects models controlling for country-specific factors
- Random effects models for cross-country comparison
- Difference-in-differences for policy impact assessment

### 3. Early Warning System Development
- Binary classification (crisis vs. non-crisis periods)
- Multi-class classification (risk levels: low/medium/high)
- Survival analysis for time-to-event modeling
- Machine learning approaches (Random Forest, XGBoost, Neural Networks)

### 4. Network Analysis
- Correlation networks between risk variables
- Lead-lag relationships across countries
- Spillover effects and contagion modeling

### 5. Clustering Analysis
- K-means clustering to identify country groups
- Hierarchical clustering for risk profiles
- Time-series clustering for pattern detection

---

## Citation

If using this dataset in academic work, please cite as:

```
FinTech Nexus Risk Dataset - Sub-Saharan Africa (2025)
Category 4: Nexus-Specific & Alternative Data
Generated for thesis: "Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies"
Dataset includes quarterly observations from 2018-2024 covering 15 SSA countries.
```

---

## Contact & Support

For questions about the dataset or to report issues:
- Review the generation script: `generate_nexus_data.py`
- Modify parameters to adjust data characteristics
- Regenerate with different random seed for alternative scenarios

---

## Version History

**Version 1.0** (October 2025)
- Initial dataset generation
- 405 observations across 15 countries
- 22 variables covering cyber risk, sentiment, competition, and nexus metrics
- Quarterly frequency from 2018 Q1 to 2024 Q3

---

*This dataset was generated to support academic research on FinTech risk in Sub-Saharan Africa. All data is synthetic and should be used accordingly.*
