# Lagos Food Processing FDI Dataset Documentation

## Research Topic
**The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy**

## Dataset Overview
This synthetic dataset has been generated to support research on FDI impact on food processing firms in Lagos, Nigeria. The dataset simulates realistic patterns based on Nigerian economic conditions and food processing industry characteristics.

## Generated Date
October 13, 2025

## Dataset Files

### 1. Primary Data Files

#### `lagos_food_firms_primary_data.csv`
- **Description**: Firm-level survey data from 300 food processing firms in Lagos
- **Records**: 300 firms (200 SMEs, 100 large firms)
- **Variables**: 43 variables covering firm characteristics, FDI status, performance metrics, and policy perceptions
- **Structure**: Cross-sectional data as of Q3 2024

#### `lagos_firms_panel_data.csv`
- **Description**: Panel data for all firms over 5 years (2020-2024)
- **Records**: 1,500 firm-year observations
- **Variables**: Same as primary data with year-specific variations
- **Use**: For panel data analysis and trend identification

### 2. Secondary Data Files

#### `lagos_macro_secondary_data.csv`
- **Description**: Quarterly macro-economic and sectoral data
- **Time Period**: Q1 2019 to Q3 2024 (23 quarters)
- **Variables**: 21 macro-economic indicators including FDI flows, GDP growth, infrastructure indices
- **Sources Simulated**: CBN, NBS, World Bank, UNCTAD patterns

### 3. Integrated Dataset

#### `lagos_fdi_integrated_dataset.csv`
- **Description**: Combined firm-level and macro-economic data
- **Records**: 300 firms with latest macro indicators
- **Variables**: 58 variables (firm-level + macro context)
- **Use**: Main analysis file for regression and SEM models

### 4. Supporting Files

#### `data_dictionary.json`
- **Description**: Comprehensive variable definitions and measurement scales
- **Format**: JSON structure with categories and descriptions

## Variable Categories

### 1. Firm Identifiers & Demographics
- FirmID: Unique identifier (FIRM_001 to FIRM_300)
- Size: SME (<250 employees) or Large (≥250 employees)
- Subsector: 10 food processing subsectors
- Firm Age: Years since establishment (1-45 years)
- Ownership Type: 5 categories including foreign and joint ventures

### 2. FDI Variables
- **FDI_Presence**: Binary indicator (0/1)
- **FDI_Percentage**: Foreign ownership percentage (0-100%)
- **FDI_Origin**: Source country for FDI
- **FDI_Type**: Greenfield, M&A, or Joint Venture
- **FDI_Years**: Duration since FDI entry

### 3. FDI Constructs (Likert Scale 1-5)
- **Knowledge_Absorption**: Adapted from Zahra & George (2002)
- **Task_Performance**: Based on Koopmans et al. (2013)
- **Innovation_Score**: Following OECD Oslo Manual
- **Firm_Resources_Score**: Composite resource capability

### 4. Performance Metrics
- **Financial**: ROI, ROA (%)
- **Market**: Market share, Export intensity (%)
- **Operational**: Efficiency score (0-100)
- **Export**: Export value, destination count

### 5. Government Policy Perception (Likert Scale 1-7)
- Tax incentives effectiveness
- Regulatory stability
- Infrastructure support
- Corruption experience (reverse coded)
- Policy effectiveness index (composite)

### 6. Resource Variables
- **Human Capital**: Skilled labor ratio, training hours
- **Technology**: IoT adoption, R&D spending, automation level
- **Financial**: Liquidity ratio, revenue, assets

### 7. Control Variables
- Employees, Assets (continuous)
- ISO certification, Quality certifications (categorical)
- Local content percentage
- Supply chain metrics

### 8. Macro-Economic Context
- FDI inflows (Lagos, sectoral)
- Economic indicators (GDP growth, inflation, exchange rate)
- Infrastructure indices (power, logistics, port efficiency)
- Institutional quality (corruption index, regulatory quality)
- Policy indicators (tax incentives, budget allocations)

## Measurement Scales

### Likert Scales
- **1-5 Scale**: FDI constructs (knowledge, performance, innovation, resources)
- **1-7 Scale**: Government policy perceptions
- **Binary (0/1)**: FDI presence, certifications, technology adoption

### Continuous Variables
- **Percentages**: ROI, ROA, export intensity, market share, local content
- **Monetary**: Revenue, assets, export values (Million NGN)
- **Ratios**: Liquidity, inventory turnover, employment ratios
- **Counts**: Employees, export destinations, firm age

## Data Quality Notes

### Realistic Patterns Simulated
1. **FDI Distribution**: Higher FDI presence in large firms (70%) vs SMEs (30%)
2. **Performance Premium**: FDI firms show 30% better performance on average
3. **Sectoral Distribution**: Reflects actual Lagos food processing composition
4. **COVID Impact**: 2020 data shows 15% performance reduction

### Correlations Built In
- FDI presence positively correlated with:
  - Performance metrics (r ≈ 0.3-0.4)
  - Technology adoption (r ≈ 0.4-0.5)
  - Export intensity (r ≈ 0.5-0.6)
  - Innovation scores (r ≈ 0.3-0.4)

### Nigerian Context Features
- Exchange rate depreciation trend (2019-2024)
- Inflation rates (11-18% range)
- Power supply constraints (12-20 hours/day)
- Corruption index (24-28, typical for Nigeria)
- Lagos captures 35-45% of national FDI

## Statistical Properties

### Sample Distribution
- **Total Firms**: 300
- **SMEs**: 200 (66.7%)
- **Large Firms**: 100 (33.3%)
- **FDI Presence**: 215 firms (71.7%)

### Key Statistics (Mean ± SD)
- **ROI**: 18.30% ± 7.11%
- **ROA**: 12.77% ± 5.28%
- **Export Intensity**: 21.13% ± 15.84%
- **Market Share**: 4.85% ± 4.21%
- **Operational Efficiency**: 70.2 ± 12.3

### FDI Origin Distribution
- China (19%), Germany (15%), UK (14%)
- South Africa (14%), USA (13%)
- India (13%), Netherlands (12%)

### Subsector Coverage
- All 10 major food processing subsectors represented
- Balanced distribution (30-43 firms per subsector)

## Usage Guidelines

### For Regression Analysis
Use `lagos_fdi_integrated_dataset.csv`:
```
DV: Performance metrics (ROI, ROA, Export_Intensity)
IV: FDI variables
Moderator: Policy_Effectiveness_Index
Controls: Size, Age, Subsector, Macro variables
```

### For Panel Analysis
Use `lagos_firms_panel_data.csv`:
```
Entity: FirmID
Time: Year
Fixed effects: Firm, Year
Random effects: Subsector
```

### For SEM/Path Analysis
Key constructs:
```
Latent Variables:
- FDI Impact (Knowledge, Task, Innovation, Resources)
- Firm Performance (ROI, ROA, Market_Share)
- Policy Environment (Tax, Regulatory, Infrastructure, Corruption)
```

## Ethical Considerations

1. **Synthetic Data**: All data is fabricated for research purposes
2. **Anonymization**: Firm IDs are generic (no real company names)
3. **Realistic Patterns**: Based on actual Nigerian economic indicators
4. **Research Use Only**: Not for commercial or policy decisions

## Data Limitations

1. **Synthetic Nature**: Not actual firm data
2. **Simplifications**: Some complex relationships simplified
3. **Time Period**: Limited to 2019-2024
4. **Geographic Scope**: Lagos only (not nationwide)
5. **Self-reported Bias**: Simulated but not accounting for actual reporting biases

## Recommended Analysis Tools

### Software
- **STATA**: For panel data and SEM
- **R**: Packages - lavaan (SEM), plm (panel), ggplot2 (visualization)
- **SPSS**: For basic regression and PROCESS macro
- **Python**: pandas, statsmodels, scikit-learn

### Analysis Methods
1. **Descriptive Statistics**: Mean, SD, correlations
2. **Regression Models**: OLS, Fixed/Random Effects
3. **Moderation Analysis**: Hayes PROCESS or interaction terms
4. **SEM**: Measurement and structural models
5. **Robustness Checks**: Bootstrap, alternative specifications

## Contact & Citation

**Generated by**: AI Assistant
**Date**: October 13, 2025
**Purpose**: Academic Research Support

### Suggested Citation
"Synthetic Dataset: The Influence of FDI on Food Processing Firms in Lagos, Nigeria (2024). 
Generated for academic research purposes."

## Version History

- **v1.0** (Oct 13, 2025): Initial generation
  - 300 firms, 5-year panel
  - 43 firm variables, 21 macro variables
  - Complete documentation

---

*Note: This is synthetic data created for research purposes. All patterns and relationships are simulated based on typical Nigerian economic conditions and should not be used for actual policy or business decisions.*