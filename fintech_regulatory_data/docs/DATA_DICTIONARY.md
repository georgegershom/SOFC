# Financial System & Regulatory Data Dictionary
## Sub-Saharan Africa FinTech Risk Early Warning Model

### Dataset Overview
This dataset contains comprehensive financial system and regulatory data for 48 Sub-Saharan African countries, covering the period from 2010 to 2024 (quarterly data).

### Purpose
This data supports research on FinTech early warning models, measuring the health of traditional financial systems and the regulatory landscape that FinTechs operate within.

---

## Variable Definitions

### 1. Identification Variables

| Variable | Type | Description |
|----------|------|-------------|
| Country_Code | String | ISO 3-letter country code |
| Country_Name | String | Full country name |
| Year | Integer | Year (2010-2024) |
| Quarter | String | Quarter (Q1-Q4) |
| Date | String | Combined Year-Quarter identifier |
| Region | String | Geographic region (Sub-Saharan Africa) |

### 2. Banking Sector Health Indicators

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| Bank_NPL_to_Total_Loans_% | Float | 0-30 | Non-performing loans as percentage of total gross loans |
| Bank_Z_Score | Float | 2-50 | Distance to insolvency (higher = more stable) |
| Bank_ROA_% | Float | -2 to 5 | Return on Assets of the banking sector |
| Domestic_Credit_to_Private_Sector_%_GDP | Float | 0-200 | Private sector credit as % of GDP |
| Capital_Adequacy_Ratio_% | Float | 8-30 | Bank capital to risk-weighted assets |

### 3. Regulatory Quality Indicators

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| WGI_Regulatory_Quality | Float | -2.5 to 2.5 | World Bank Governance Indicator for regulatory quality |
| Financial_Regulation_Index | Float | 0-100 | Composite index of financial regulation strength |
| Digital_Lending_Regulation | Binary | 0/1 | 1 if digital lending guidelines exist |
| Mobile_Money_Regulation | Binary | 0/1 | 1 if mobile money regulations exist |
| Data_Protection_Law | Binary | 0/1 | 1 if data protection laws exist |
| Regulatory_Sandbox | Binary | 0/1 | 1 if regulatory sandbox exists |
| Total_FinTech_Regulations | Integer | 0-4 | Count of FinTech-specific regulations |

### 4. Financial Inclusion Metrics

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| Account_Ownership_%_Adults | Float | 0-100 | Percentage of adults with financial accounts |
| Mobile_Money_Account_%_Adults | Float | 0-100 | Percentage of adults with mobile money accounts |
| Bank_Branches_per_100k_Adults | Float | 0.1-50 | Commercial bank branches per 100,000 adults |
| ATMs_per_100k_Adults | Float | 0.1-100 | ATMs per 100,000 adults |

### 5. Market Structure Indicators

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| Banking_Market_HHI | Float | 1000-10000 | Herfindahl-Hirschman Index (market concentration) |
| Interest_Rate_Spread_% | Float | 2-25 | Difference between lending and deposit rates |
| Liquid_Assets_to_Deposits_% | Float | 10-60 | Bank liquid assets to deposits ratio |

### 6. Systemic Risk Indicators

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| Forex_Reserves_Months_Imports | Float | 0.1-12 | Foreign exchange reserves in months of imports |
| Financial_Stress_Index | Float | 0-100 | Composite measure of financial system stress |
| Systemic_Risk_Score | Float | 0-100 | Overall systemic risk assessment |

### 7. Composite Indices

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| Banking_Health_Index | Float | 0-100 | Composite measure of banking sector health |
| Regulatory_Strength_Index | Float | 0-100 | Composite measure of regulatory quality |
| Financial_Development_Index | Float | 0-100 | Composite measure of financial development |

### 8. Metadata

| Variable | Type | Description |
|----------|------|-------------|
| Data_Source | String | Source of data (Synthetic/Estimated) |
| Last_Updated | Date | Date of last update |

---

## Countries Included

The dataset covers all 48 Sub-Saharan African countries:

**East Africa:** Burundi, Comoros, Djibouti, Eritrea, Ethiopia, Kenya, Madagascar, Malawi, Mauritius, Mozambique, Rwanda, Seychelles, Somalia, South Sudan, Sudan, Tanzania, Uganda, Zambia, Zimbabwe

**West Africa:** Benin, Burkina Faso, Cabo Verde, Ivory Coast, Gambia, Ghana, Guinea, Guinea-Bissau, Liberia, Mali, Mauritania, Niger, Nigeria, Senegal, Sierra Leone, Togo

**Central Africa:** Angola, Cameroon, Central African Republic, Chad, Democratic Republic of Congo, Republic of Congo, Equatorial Guinea, Gabon, Sao Tome and Principe

**Southern Africa:** Botswana, Eswatini, Lesotho, Namibia, South Africa

---

## Data Quality Notes

1. **Synthetic Data Generation**: This dataset uses advanced synthetic data generation techniques to create realistic patterns based on known economic relationships and regional characteristics.

2. **COVID-19 Impact**: The data includes realistic COVID-19 pandemic effects from Q1 2020 to Q4 2021, showing increased NPLs, reduced bank stability, and heightened financial stress.

3. **Regulatory Evolution**: The dataset captures the gradual introduction of FinTech regulations across different countries, reflecting the actual regulatory landscape development in SSA.

4. **Country Groupings**: Countries are grouped into three development levels for more realistic data generation:
   - Advanced: South Africa, Mauritius, Seychelles, Botswana, Namibia
   - Emerging: Kenya, Ghana, Nigeria, Senegal, Rwanda, Uganda, Tanzania
   - Developing: All other countries

---

## Usage Guidelines

1. **Time Series Analysis**: Data is suitable for panel data analysis, time series forecasting, and early warning system development.

2. **Cross-Country Comparisons**: Consistent methodology allows for valid cross-country comparisons.

3. **Model Training**: Can be used for training machine learning models for FinTech risk prediction.

4. **Research Applications**: Suitable for academic research on financial inclusion, regulatory impact, and systemic risk in SSA.

---

## Citation

If using this dataset, please cite:
```
Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies
Financial System & Regulatory Data (Category 3)
Generated: 2024
```

---

## Contact & Updates

For questions or updates about this dataset, please refer to the accompanying research documentation.