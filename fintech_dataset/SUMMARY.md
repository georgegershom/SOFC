# FinTech Early Warning Model Dataset - Summary

## 🎯 Project Overview

This dataset was created for research on **"FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Saharan Africa Economies"**. It provides comprehensive data on 200 FinTech companies across 10 SSA countries, enabling the development of early warning systems for FinTech distress prediction.

## 📊 Dataset Statistics

- **Total Companies**: 200
- **Countries Covered**: 10 (Nigeria, Kenya, South Africa, Ghana, Uganda, Tanzania, Rwanda, Senegal, Ethiopia, Zambia)
- **FinTech Types**: 8 (Mobile Money, Digital Banking, Payment Gateway, Lending Platform, Investment Platform, Insurance Tech, Crypto Exchange, Remittance)
- **Time Period**: 8 quarters (2 years) of historical data
- **Distress Rate**: 42% (84 out of 200 companies)
- **Total Records**: 4,000+ data points across all datasets

## 🗂️ Dataset Components

### 1. **companies.csv** (200 records)
Core company information including demographics, regulatory status, and market characteristics.

### 2. **financial_metrics.csv** (1,600 records)
Quarterly financial performance data including revenue, profit margins, and growth rates.

### 3. **operational_metrics.csv** (1,600 records)
User metrics, transaction volumes, churn rates, and operational KPIs.

### 4. **funding_data.csv** (588 records)
Funding rounds, investment amounts, valuations, and investor information.

### 5. **regulatory_data.csv** (134 records)
Regulatory sanctions, compliance events, and enforcement actions.

### 6. **distress_indicators.csv** (200 records)
Distress scores, risk indicators, and binary distress classification.

## 🔍 Key Findings

### Risk Factors (by importance)
1. **User Decline Rate** (13.7% importance)
2. **Revenue Decline Rate** (11.2% importance)
3. **Active Users** (7.9% importance)
4. **User Growth Rate** (6.0% importance)
5. **Transaction Volume** (5.6% importance)

### Distress Patterns
- **Crypto Exchanges**: 79.2% distress rate (highest risk)
- **Lending Platforms**: 50.0% distress rate
- **Mobile Money**: 23.8% distress rate (lowest risk)
- **Digital Banking**: 27.3% distress rate

### Financial Performance
- **Distressed companies**: 23.9% average revenue decline
- **Healthy companies**: 11.6% average revenue decline
- **Distressed companies**: 15.1% average user decline
- **Healthy companies**: -13.1% average user growth

## 🤖 Model Performance

The Random Forest early warning model achieved:
- **Accuracy**: 85%
- **ROC AUC Score**: 0.914
- **Precision**: 87% (healthy), 82% (distressed)
- **Recall**: 87% (healthy), 82% (distressed)

## 📈 Research Applications

### 1. Early Warning Models
- Binary classification for distress prediction
- Continuous distress scoring
- Risk factor identification and ranking

### 2. Risk Assessment
- Country-specific risk analysis
- FinTech type risk profiling
- Regulatory impact assessment

### 3. Policy Analysis
- Regulatory effectiveness evaluation
- Market stability monitoring
- Intervention strategy development

### 4. Investment Research
- Due diligence support
- Portfolio risk management
- Market opportunity identification

## 🛠️ Usage Instructions

### Quick Start
```python
import pandas as pd

# Load the main datasets
companies = pd.read_csv('fintech_dataset/companies.csv')
distress = pd.read_csv('fintech_dataset/distress_indicators.csv')

# Merge for analysis
data = companies.merge(distress, on='company_id')
print(f"Dataset shape: {data.shape}")
print(f"Distress rate: {data['is_distressed'].mean():.1%}")
```

### Advanced Analysis
```python
# Run the sample analysis script
python3 sample_analysis.py
```

## 📋 File Structure

```
fintech_dataset/
├── companies.csv              # Company demographics
├── financial_metrics.csv      # Financial performance
├── operational_metrics.csv    # Operational KPIs
├── funding_data.csv          # Investment data
├── regulatory_data.csv       # Regulatory events
├── distress_indicators.csv   # Risk indicators
├── README.md                 # Detailed documentation
├── data_dictionary.md        # Variable descriptions
├── metadata.json            # Dataset metadata
└── SUMMARY.md               # This summary
```

## 🎯 Research Recommendations

### 1. **Primary Focus Areas**
- Revenue decline monitoring (strongest predictor)
- User growth and churn analysis
- Regulatory compliance tracking

### 2. **Model Development**
- Ensemble methods for improved accuracy
- Time series analysis for trend detection
- Country-specific model calibration

### 3. **Feature Engineering**
- Rolling averages for trend analysis
- Interaction terms between risk factors
- External economic indicators integration

### 4. **Validation Studies**
- Cross-country model validation
- Temporal validation with new data
- Regulatory change impact assessment

## 🔬 Data Quality & Limitations

### Strengths
- ✅ Realistic company names and characteristics
- ✅ Country-specific economic indicators
- ✅ Comprehensive risk factor coverage
- ✅ Temporal consistency across quarters
- ✅ Balanced distressed/healthy sample

### Limitations
- ⚠️ Synthetic data (not real company data)
- ⚠️ Limited to 2-year time period
- ⚠️ Simplified regulatory framework
- ⚠️ No external market shock modeling

## 📚 Citation

When using this dataset in your research, please cite:

```
FinTech Early Warning Model Dataset for Sub-Saharan Africa
Generated for research on FinTech risk assessment in SSA economies
[Your Institution], [Year]
```

## 🤝 Collaboration

This dataset is designed for academic and research purposes. For collaboration opportunities or questions about the data generation methodology, please contact [Your Contact Information].

---

**Generated on**: January 2025  
**Dataset Version**: 1.0  
**Total Size**: ~2MB  
**Compatibility**: Python 3.7+, R, Stata, SPSS