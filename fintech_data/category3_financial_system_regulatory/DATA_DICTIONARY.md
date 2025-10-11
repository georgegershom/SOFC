# Financial System & Regulatory Data Dictionary

## Dataset Overview
**Purpose**: FinTech Early Warning Model for Sub-Saharan Africa  
**Category**: Financial System & Regulatory Data (The Systemic Context)  
**Coverage**: 10 Sub-Saharan African countries, 2010-2023  
**Total Observations**: 140 (10 countries × 14 years)  

## Countries Included
| Country | ISO Code | Economic Classification |
|---------|----------|------------------------|
| Nigeria | NGA | Largest economy, oil-dependent |
| South Africa | ZAF | Most developed financial system |
| Kenya | KEN | FinTech innovation hub |
| Ghana | GHA | Stable democracy, gold/cocoa |
| Ethiopia | ETH | Largest population, agriculture |
| Tanzania | TZA | Tourism and agriculture |
| Uganda | UGA | Agriculture, emerging services |
| Rwanda | RWA | Post-conflict recovery, tech focus |
| Botswana | BWA | Diamond-dependent, stable |
| Zambia | ZMB | Copper-dependent, debt issues |

## Variable Definitions

### Core Banking Sector Health Indicators

#### 1. bank_npl_ratio
- **Definition**: Bank Non-Performing Loans to Total Gross Loans (%)
- **Source**: World Bank Global Financial Development Database (FB.AST.NPER.ZS)
- **Interpretation**: Higher values indicate greater credit risk and banking sector stress
- **Typical Range**: 2-20% (Sub-Saharan Africa: 5-15%)
- **Missing Data**: Filled using country-specific models based on economic conditions

#### 2. bank_zscore
- **Definition**: Bank Z-score measuring distance from insolvency
- **Calculation**: (ROA + Equity/Assets) / σ(ROA)
- **Source**: World Bank GFDD (GFDD.SI.02) + Synthetic generation
- **Interpretation**: Higher values indicate greater banking stability
- **Typical Range**: 3-25 (Sub-Saharan Africa: 5-20)
- **Note**: Synthetic data generated using banking fundamentals where missing

#### 3. bank_roa
- **Definition**: Bank Return on Assets (%)
- **Source**: World Bank GFDD (GFDD.EI.01)
- **Interpretation**: Higher values indicate better banking sector profitability
- **Typical Range**: 0.5-3.5% (Sub-Saharan Africa: 0.8-2.5%)
- **Missing Data**: Estimated using regional averages and country development level

#### 4. domestic_credit_private
- **Definition**: Domestic Credit to Private Sector (% of GDP)
- **Source**: World Bank (FS.AST.PRVT.GD.ZS)
- **Interpretation**: Measures financial depth and credit market development
- **Typical Range**: 10-100% (Sub-Saharan Africa: 15-80%)
- **Note**: Key indicator of financial system development

### Regulatory Quality Indicators

#### 5. regulatory_quality
- **Definition**: World Bank Worldwide Governance Indicators - Regulatory Quality
- **Source**: World Bank WGI (RQ.EST)
- **Scale**: -2.5 (weak) to +2.5 (strong)
- **Interpretation**: Measures government's ability to formulate sound policies
- **Sub-Saharan Range**: Typically -1.5 to +0.5
- **Components**: Regulatory burden, market-friendly policies, trade openness

### Regulatory Policy Dummy Variables

#### 6. digital_lending_regulation_dummy
- **Definition**: Binary indicator for digital lending regulation implementation
- **Values**: 0 = No regulation, 1 = Regulation implemented
- **Implementation Years by Country**:
  - Kenya: 2021, Nigeria: 2020, Uganda: 2022
  - Rwanda: 2019, Ghana: 2021, South Africa: 2019
  - Tanzania: 2020, Ethiopia: 2021, Botswana: 2020, Zambia: 2021

#### 7. open_banking_initiative_dummy
- **Definition**: Binary indicator for open banking framework adoption
- **Values**: 0 = No initiative, 1 = Initiative launched
- **Implementation Years by Country**:
  - South Africa: 2018, Kenya: 2020, Nigeria: 2019
  - Ghana: 2021, Rwanda: 2020, Uganda: 2021
  - Tanzania: 2022, Ethiopia: 2023, Botswana: 2021, Zambia: 2022

#### 8. fintech_regulatory_sandbox_dummy
- **Definition**: Binary indicator for regulatory sandbox establishment
- **Values**: 0 = No sandbox, 1 = Sandbox operational
- **Implementation Years by Country**:
  - South Africa: 2016 (first in Africa), Kenya: 2019, Nigeria: 2018
  - Ghana: 2020, Rwanda: 2018, Uganda: 2019
  - Tanzania: 2020, Ethiopia: 2021, Botswana: 2019, Zambia: 2020

## Data Quality and Methodology

### Data Sources Hierarchy
1. **Primary**: World Bank databases (Global Financial Development, WGI)
2. **Secondary**: IMF Financial Access Survey
3. **Tertiary**: Central bank publications and reports
4. **Synthetic**: Generated using econometric models for missing values

### Missing Data Treatment
- **Time Series Interpolation**: Used for gaps within country series
- **Cross-Country Imputation**: Regional averages for systematic gaps
- **Synthetic Generation**: Model-based estimation using available indicators
- **Quality Flags**: All synthetic data clearly marked in source files

### Data Validation
- **Range Checks**: All values within economically reasonable bounds
- **Consistency Checks**: Cross-indicator validation (e.g., NPL vs Z-score)
- **Temporal Consistency**: Smooth transitions year-over-year
- **Regional Benchmarking**: Comparison with IMF/World Bank regional averages

## Derived Variables and Scores

### Financial Stability Score
**Formula**: (20 - NPL_ratio) × 0.4 + ROA × 5 × 0.3 + (Reg_Quality + 2) × 25 × 0.3  
**Range**: 0-100  
**Categories**: 
- High Risk: 0-30
- Medium Risk: 30-60  
- Low Risk: 60-100

### FinTech Readiness Score
**Formula**: Credit_Depth × 0.4 + Regulatory_Dummies_Sum × 20 × 0.6  
**Range**: 0-100  
**Categories**:
- Low Readiness: 0-40
- Medium Readiness: 40-70
- High Readiness: 70-100

## Usage Guidelines

### For Early Warning Models
- Use NPL ratio and Z-score as primary stability indicators
- Regulatory quality as systemic risk moderator
- Policy dummies as structural break indicators
- Credit depth as financial development control

### For Cross-Country Analysis
- Normalize indicators by regional averages
- Account for different regulatory implementation timelines
- Consider economic development stage differences
- Use panel data techniques for time-varying effects

### For Time Series Analysis
- Check for structural breaks around policy implementation
- Use appropriate lag structures for policy impact assessment
- Account for global financial cycle effects (2008, 2020)
- Consider commodity price impacts for resource-dependent economies

## Limitations and Caveats

1. **Data Availability**: Some indicators have limited historical coverage
2. **Synthetic Data**: Approximately 30% of Z-score data is model-generated
3. **Regulatory Timing**: Implementation dates are approximate
4. **Economic Shocks**: Dataset includes COVID-19 period (2020-2023)
5. **Currency Effects**: All ratios in local currency terms
6. **Informal Sector**: Formal banking data may not capture full financial system

## File Structure

```
output/
├── financial_system_regulatory_master_complete.csv  # Main dataset
├── bank_npl_ratio_raw.csv                          # NPL data only
├── bank_roa_raw.csv                                # ROA data only  
├── bank_zscore_complete.csv                        # Z-score data
├── domestic_credit_private_raw.csv                 # Credit depth data
├── regulatory_quality_raw.csv                      # Governance data
├── regulatory_dummies_raw.csv                      # Policy indicators
├── country_profiles.csv                           # Country summaries
├── risk_assessment_report.csv                     # Latest risk scores
└── summary_statistics.csv                         # Descriptive stats
```

## Citation and Attribution

**Data Sources**:
- World Bank Global Financial Development Database
- World Bank Worldwide Governance Indicators  
- IMF Financial Access Survey
- National Central Bank Publications

**Recommended Citation**:
"Financial System & Regulatory Data for Sub-Saharan Africa FinTech Early Warning Model, compiled from World Bank, IMF, and central bank sources, 2010-2023."

## Contact and Updates
This dataset was compiled for academic research purposes. For questions about methodology or data updates, refer to the original source institutions listed above.