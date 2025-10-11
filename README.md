# FinTech Risk Nexus Dataset - Sub-Saharan Africa

## Research Context
**Thesis Topic:** Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

**Dataset Category:** Category 4: Nexus-Specific & Alternative Data

This dataset captures the interconnectedness and modern risks in FinTech ecosystems across Sub-Saharan African economies, providing unique insights for early warning model development.

## Dataset Overview

### Countries Covered (27)
Nigeria, Kenya, South Africa, Ghana, Uganda, Tanzania, Ethiopia, Angola, Mozambique, Zambia, Zimbabwe, Rwanda, Senegal, Burkina Faso, Mali, Niger, Chad, Cameroon, Côte d'Ivoire, Madagascar, Malawi, Botswana, Namibia, Mauritius, Seychelles, Eswatini, Lesotho

### Data Components

#### 1. Cyber Risk Exposure (`fintech_risk_nexus_cyber_risk.csv`)
- **Records:** 540 (27 countries × 5 years × 4 quarters)
- **Key Variables:**
  - `cyber_incidents`: Number of cybersecurity incidents reported in financial sector
  - `mobile_money_fraud_search_trends`: Google search trends for "mobile money fraud" (0-100 scale)
  - `mobile_money_fraud_cases`: Actual fraud cases reported
  - `data_breach_severity`: Severity of data breaches (1-10 scale)
  - `digital_payment_volume_millions_usd`: Digital payment transaction volume
  - `cyber_risk_score`: Composite cyber risk assessment

#### 2. Consumer Sentiment & Trust (`fintech_risk_nexus_consumer_sentiment.csv`)
- **Records:** 8,100 (27 countries × 5 years × 12 months × 5 brands per country)
- **Key Variables:**
  - `sentiment_score`: Social media sentiment analysis (-1 to 1)
  - `trust_index`: Consumer trust in FinTech brands (0-100)
  - `social_media_mentions`: Volume of social media mentions
  - `customer_satisfaction`: Customer satisfaction scores (1-10)
  - `app_store_rating`: Mobile app store ratings (1-5)
  - `customer_complaints`: Monthly complaint volume

#### 3. Competitive Dynamics (`fintech_risk_nexus_competitive_dynamics.csv`)
- **Records:** 135 (27 countries × 5 years)
- **Key Variables:**
  - `new_fintech_licenses`: Number of new FinTech licenses issued annually
  - `hhi_index`: Herfindahl-Hirschman Index for market concentration
  - `market_concentration`: Categorical concentration level (High/Moderate/Low)
  - `top5_market_share`: Market share of top 5 companies
  - `fintech_investment_millions_usd`: Annual FinTech investment
  - `partnerships_mergers`: Number of partnerships/mergers
  - `regulatory_changes`: Binary indicator of regulatory changes
  - `digital_adoption_rate`: Digital adoption rate

#### 4. Macro-Economic Indicators (`fintech_risk_nexus_macro_economic.csv`)
- **Records:** 135 (27 countries × 5 years)
- **Key Variables:**
  - `gdp_per_capita_usd`: GDP per capita in USD
  - `inflation_rate_pct`: Annual inflation rate
  - `unemployment_rate_pct`: Unemployment rate
  - `mobile_penetration_pct`: Mobile phone penetration rate
  - `internet_penetration_pct`: Internet penetration rate
  - `financial_inclusion_pct`: Financial inclusion rate

#### 5. Summary Statistics (`fintech_risk_nexus_summary_statistics.csv`)
- **Records:** 27 (one per country)
- **Key Variables:**
  - Aggregated metrics across all data components
  - Risk level classification (High/Medium/Low)
  - Country-level performance indicators

## FinTech Brands Covered (29)
M-Pesa, MTN Mobile Money, Airtel Money, Orange Money, Tigo Pesa, Vodacom M-Pesa, Ecobank Mobile, GTBank, Access Bank, First Bank, UBA, Zenith Bank, Fidelity Bank, Sterling Bank, Kuda Bank, Opay, PalmPay, Carbon, FairMoney, Branch, Tala, Jumo, MFS Africa, Flutterwave, Paystack, Interswitch, Cellulant, Paga, Korapay

## Data Characteristics

### Temporal Coverage
- **Period:** 5 years (2020-2024)
- **Frequency:** 
  - Cyber Risk: Quarterly
  - Consumer Sentiment: Monthly
  - Competitive Dynamics: Annual
  - Macro-Economic: Annual

### Data Quality Features
- **Realistic Relationships:** Variables are correlated to reflect real-world dynamics
- **Country-Specific Patterns:** Each country has unique risk profiles and development levels
- **Temporal Trends:** Data includes realistic time-series patterns and growth trajectories
- **Cross-Sector Integration:** Variables are interconnected to reflect nexus relationships

## Usage for Research

### Early Warning Model Development
1. **Risk Identification:** Use cyber risk and sentiment data to identify early warning signals
2. **Market Dynamics:** Leverage competitive dynamics to understand market stability
3. **Macro Context:** Incorporate economic indicators for comprehensive risk assessment
4. **Nexus Analysis:** Explore interconnections between different risk categories

### Key Research Questions
- How do cyber risks correlate with consumer trust in FinTech services?
- What is the relationship between market concentration and financial stability?
- How do regulatory changes impact competitive dynamics and risk levels?
- What role do macro-economic factors play in FinTech risk propagation?

## File Structure
```
/workspace/
├── fintech_risk_nexus_dataset.py          # Data generation script
├── fintech_risk_nexus_cyber_risk.csv      # Cyber risk data
├── fintech_risk_nexus_consumer_sentiment.csv  # Sentiment & trust data
├── fintech_risk_nexus_competitive_dynamics.csv  # Market dynamics data
├── fintech_risk_nexus_macro_economic.csv  # Economic indicators
├── fintech_risk_nexus_summary_statistics.csv  # Country summaries
├── dataset_metadata.json                  # Dataset metadata
└── README.md                              # This documentation
```

## Technical Notes
- **Data Generation:** Synthetic data created using realistic statistical distributions
- **Reproducibility:** Random seeds set for consistent results
- **Scalability:** Script can generate data for different time periods
- **Extensibility:** Framework allows addition of new variables and countries

## Citation
If you use this dataset in your research, please cite:
```
FinTech Risk Nexus Dataset - Sub-Saharan Africa (2024)
Category 4: Nexus-Specific & Alternative Data
Generated for: Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies
```

## Contact
For questions about the dataset or research collaboration, please refer to the thesis documentation.