# Financial System & Regulatory Data - Dataset Documentation

## Overview
This dataset contains Category 3: Financial System & Regulatory Data for FinTech Early Warning Model research in Sub-Saharan African economies. The data measures the health of traditional financial systems and regulatory landscapes that FinTechs both compete with and integrate into.

## Dataset Information
- **File Name**: `financial_system_regulatory_data.csv`
- **Format**: CSV (Comma-Separated Values)
- **Period**: 2020 Q1 - 2024 Q2 (18 quarters)
- **Countries**: 8 Sub-Saharan African countries
- **Total Observations**: 144 (8 countries × 18 quarters)
- **Variables**: 25 indicators

## Countries Included
1. **Nigeria** - Largest economy in Africa, major FinTech hub
2. **Kenya** - Pioneer in mobile money (M-Pesa), advanced FinTech ecosystem
3. **South Africa** - Most developed financial sector in SSA
4. **Ghana** - Growing FinTech sector, stable banking system
5. **Uganda** - Emerging FinTech market, mobile money adoption
6. **Tanzania** - Developing FinTech ecosystem, mobile financial services
7. **Rwanda** - Strong regulatory framework, digital transformation focus
8. **Ethiopia** - Recently opened financial sector, emerging FinTech market

## Variable Descriptions

### Banking Sector Health Indicators

#### 1. bank_npl_ratio
- **Description**: Bank Non-Performing Loans to Total Loans (%)
- **Range**: 2.8% - 15.0%
- **Interpretation**: Higher values indicate deteriorating loan quality
- **Source**: Central Bank data, IMF Financial Access Survey

#### 2. bank_z_score
- **Description**: Bank Z-score (measure of bank stability)
- **Range**: 0.4 - 15.6
- **Interpretation**: Higher values indicate greater bank stability
- **Calculation**: (ROA + Capital/Assets) / σ(ROA)

#### 3. bank_roa
- **Description**: Return on Assets of the banking sector (%)
- **Range**: -1.6% to 3.5%
- **Interpretation**: Higher values indicate better profitability
- **Source**: Central Bank financial statements

#### 4. domestic_credit_private_sector_gdp
- **Description**: Domestic Credit to Private Sector (% of GDP)
- **Range**: 7.3% - 145.8%
- **Interpretation**: Higher values indicate more developed financial intermediation
- **Source**: World Bank Global Financial Development Database

### Regulatory Quality Indicators

#### 5. regulatory_quality_wgi
- **Description**: World Bank Worldwide Governance Indicators - Regulatory Quality
- **Range**: -0.45 to 1.23
- **Interpretation**: Higher values indicate better regulatory quality
- **Scale**: -2.5 (weak) to +2.5 (strong)

#### 6. financial_regulation_index
- **Description**: Composite Financial Regulation Index (0-1 scale)
- **Range**: 0.48 - 1.22
- **Interpretation**: Higher values indicate more comprehensive financial regulation
- **Components**: Banking regulation, FinTech regulation, consumer protection

### Regulatory Dummy Variables

#### 7. digital_lending_guidelines
- **Description**: Dummy variable for digital lending regulations (1 = introduced, 0 = not)
- **Values**: 0, 1
- **Implementation**: Varies by country (2020-2021)

#### 8. open_banking_regulation
- **Description**: Dummy variable for open banking regulations (1 = introduced, 0 = not)
- **Values**: 0, 1
- **Implementation**: Varies by country (2021-2022)

#### 9. fintech_sandbox
- **Description**: Dummy variable for FinTech regulatory sandbox (1 = active, 0 = not)
- **Values**: 0, 1
- **Implementation**: Varies by country (2020-2023)

#### 10. cybersecurity_regulation
- **Description**: Dummy variable for cybersecurity regulations (1 = introduced, 0 = not)
- **Values**: 0, 1
- **Implementation**: Varies by country (2023-2024)

#### 11. data_protection_law
- **Description**: Dummy variable for data protection laws (1 = introduced, 0 = not)
- **Values**: 0, 1
- **Implementation**: Varies by country (2020-2021)

#### 12. digital_identity_framework
- **Description**: Dummy variable for digital identity frameworks (1 = introduced, 0 = not)
- **Values**: 0, 1
- **Implementation**: Varies by country (2020-2021)

#### 13. central_bank_digital_currency
- **Description**: Dummy variable for CBDC development/launch (1 = active, 0 = not)
- **Values**: 0, 1
- **Implementation**: Limited to South Africa (2024)

### Additional Financial System Indicators

#### 14. regulatory_sandbox_participants
- **Description**: Number of participants in regulatory sandbox
- **Range**: 0 - 100
- **Interpretation**: Higher values indicate more active FinTech innovation

#### 15. financial_inclusion_index
- **Description**: Composite Financial Inclusion Index (0-1 scale)
- **Range**: 0.52 - 1.06
- **Interpretation**: Higher values indicate better financial inclusion
- **Components**: Account ownership, digital payments, credit access

#### 16. credit_bureau_coverage
- **Description**: Credit Bureau Coverage (% of adult population)
- **Range**: 38.2% - 100%
- **Interpretation**: Higher values indicate better credit information systems

#### 17. deposit_insurance_coverage
- **Description**: Deposit Insurance Coverage (% of deposits)
- **Range**: 60% - 100%
- **Interpretation**: Higher values indicate stronger depositor protection

#### 18. capital_adequacy_ratio
- **Description**: Capital Adequacy Ratio (%)
- **Range**: 10.8% - 20.5%
- **Interpretation**: Higher values indicate stronger capital buffers

#### 19. loan_loss_provision_ratio
- **Description**: Loan Loss Provision Ratio (%)
- **Range**: 1.2% - 5.9%
- **Interpretation**: Higher values indicate higher risk provisioning

#### 20. interest_rate_spread
- **Description**: Interest Rate Spread (lending - deposit rates, %)
- **Range**: 4.5% - 32.1%
- **Interpretation**: Higher values indicate less efficient financial intermediation

#### 21. liquidity_ratio
- **Description**: Liquidity Ratio (%)
- **Range**: 13.5% - 28.4%
- **Interpretation**: Higher values indicate better liquidity management

#### 22. financial_stability_index
- **Description**: Composite Financial Stability Index (0-1 scale)
- **Range**: 0.48 - 0.92
- **Interpretation**: Higher values indicate greater financial system stability

## Data Characteristics

### Temporal Patterns
- **COVID-19 Impact**: Deterioration in banking sector health indicators (2020-2021)
- **Recovery Phase**: Gradual improvement in regulatory frameworks (2021-2024)
- **FinTech Evolution**: Progressive introduction of digital financial regulations

### Cross-Country Variations
- **South Africa**: Most advanced regulatory framework and financial system
- **Kenya**: Strong FinTech ecosystem with comprehensive mobile money regulations
- **Nigeria**: Large market with evolving regulatory landscape
- **Rwanda**: Proactive regulatory approach with strong digital transformation focus
- **Ethiopia**: Recently opened financial sector with emerging regulatory framework

## Data Sources (Simulated)
- Global Financial Development Database (World Bank)
- Financial Access Survey (IMF)
- Bank for International Settlements (BIS) Statistics
- Central Bank Websites and Financial Statements
- Worldwide Governance Indicators (World Bank)
- Global Partnership for Financial Inclusion (GPFI) Reports

## Usage Notes
- Data represents realistic patterns based on Sub-Saharan African financial systems
- Values are simulated but reflect actual ranges and trends observed in the region
- Dummy variables reflect the progressive nature of FinTech regulation adoption
- Some indicators show seasonal and cyclical patterns typical of financial systems

## Research Applications
This dataset is suitable for:
- FinTech early warning model development
- Financial stability analysis
- Regulatory impact assessment
- Cross-country comparative studies
- Time series analysis of financial system health
- Machine learning model training for risk prediction

## Data Quality
- **Completeness**: 100% (no missing values)
- **Consistency**: Values follow logical patterns and relationships
- **Accuracy**: Simulated data based on real-world ranges and trends
- **Timeliness**: Quarterly data from 2020-2024

## Contact Information
For questions about this dataset or research methodology, please refer to the FinTech Early Warning Model research project documentation.