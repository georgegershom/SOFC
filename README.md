# FinTech Risk Nexus Dataset - Sub-Saharan Africa

## Overview
This repository contains a comprehensive synthetic dataset for researching FinTech early warning models in the nexus of FinTech risk across Sub-Saharan African economies.

## Thesis Topic
**Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies**

## Dataset Category
**Category 4: Nexus-Specific & Alternative Data** - Capturing interconnectedness and modern risks unique to FinTech ecosystems.

## Quick Start

### Dataset Files
- `fintech_risk_nexus_dataset.csv` - Main dataset in CSV format (recommended for analysis)
- `fintech_risk_nexus_dataset.xlsx` - Excel workbook with multiple analytical sheets
- `fintech_risk_nexus_dataset.json` - JSON format for programmatic access
- `dataset_metadata.json` - Complete metadata and variable descriptions

### Documentation
- `DATASET_DOCUMENTATION.md` - Comprehensive documentation of all variables and methodology
- `README.md` - This file

### Analysis Tools
- `generate_fintech_risk_dataset.py` - Dataset generation script
- `sample_analysis.py` - Example analysis demonstrating dataset usage

## Key Features

### 1. Cyber Risk Exposure
- **Cybersecurity incidents** reported in financial sector
- **Google search trends** for "mobile money fraud"
- **Data breach severity** index
- **Phishing attempts** frequency

### 2. Consumer Sentiment & Trust
- **Social media sentiment** analysis scores
- **Consumer trust** index
- **Net Promoter Score** (NPS)
- **Customer complaint** rates

### 3. Competitive Dynamics
- **Herfindahl-Hirschman Index** (HHI) for market concentration
- **New FinTech licenses** issued quarterly
- **Market entry** rates
- **Innovation** index

### 4. Additional Risk Indicators
- System interconnectedness
- Regulatory compliance scores
- Operational risk indices
- Liquidity risk indicators

## Dataset Specifications

- **Countries:** 15 Sub-Saharan African economies
- **Time Period:** Q1 2018 - Q3 2024
- **Frequency:** Quarterly observations
- **Total Records:** 405
- **Variables:** 31 comprehensive indicators

## Countries Included

1. Kenya (M-Pesa leader)
2. Nigeria (Largest economy)
3. South Africa (Most developed financial sector)
4. Ghana, Uganda, Tanzania, Rwanda
5. Senegal, Ivory Coast, Ethiopia
6. Zambia, Zimbabwe, Mozambique
7. Cameroon, Angola

## Usage Instructions

### Loading the Data (Python)
```python
import pandas as pd

# Load CSV
df = pd.read_csv('fintech_risk_nexus_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

# Basic exploration
print(df.info())
print(df.describe())
```

### Running Sample Analysis
```bash
python3 sample_analysis.py
```

### Key Metrics to Monitor
1. **Composite Cyber Risk** - Overall cybersecurity threat level
2. **Market Health Score** - Combined market vitality indicator
3. **Consumer Trust Index** - Public confidence in FinTech
4. **HHI** - Market concentration (competition level)

## Analysis Recommendations

### For Early Warning Systems
- Monitor threshold breaches in key risk indicators
- Track year-over-year changes in critical metrics
- Use composite scores for holistic risk assessment
- Implement multi-metric warning signals

### Statistical Approaches
- Panel data analysis with fixed/random effects
- Time series forecasting with ARIMA/LSTM
- Clustering for country segmentation
- Machine learning for risk classification

## Key Insights from Initial Analysis

### Risk Distribution
- Most countries fall in medium risk category
- Cyber risks higher in rapidly growing markets
- Trust levels generally improving over time

### Market Evolution
- Market concentration decreasing (HHI down 13.5% from 2018-2024)
- Kenya leads in license issuance (175 total)
- South Africa tops innovation index (61.3/100)

### Warning Signals (Q3 2024)
- Zimbabwe and Cameroon show multiple warning indicators
- Focus areas: Mobile fraud searches and complaint rates
- Cyber incidents elevated in Kenya and South Africa

## Data Quality Notes

- **Synthetic Data:** Generated for research purposes
- **Realistic Patterns:** Based on regional characteristics
- **Temporal Consistency:** Includes trends and seasonality
- **Cross-Country Variation:** Reflects actual market differences

## Citation

```bibtex
@dataset{fintech_risk_nexus_2025,
  title={FinTech Risk Nexus Dataset - Sub-Saharan Africa},
  category={Nexus-Specific & Alternative Data},
  year={2025},
  description={Synthetic dataset for FinTech early warning model research},
  countries={15 Sub-Saharan African economies},
  period={2018-2024}
}
```

## Files Summary

| File | Size | Description |
|------|------|-------------|
| fintech_risk_nexus_dataset.csv | ~100KB | Main dataset |
| fintech_risk_nexus_dataset.xlsx | ~150KB | Excel with analysis sheets |
| fintech_risk_nexus_dataset.json | ~300KB | JSON format |
| dataset_metadata.json | ~5KB | Variable descriptions |
| DATASET_DOCUMENTATION.md | ~15KB | Full documentation |

## Next Steps

1. **Exploratory Data Analysis:** Use `sample_analysis.py` as starting point
2. **Model Development:** Build early warning models using key indicators
3. **Validation:** Compare patterns with actual market data where available
4. **Refinement:** Adjust thresholds based on empirical findings

---

**Note:** This is synthetic data generated for academic research. It should not be used for actual investment decisions or operational risk assessment.