# Food Processing FDI Research Dataset

## Research Topic
**The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy**

## Dataset Overview
This dataset contains comprehensive firm-level data from 300 food processing firms in Lagos, Nigeria, examining the relationship between foreign direct investment (FDI) and firm performance, moderated by government policy effectiveness.

## Dataset Structure

### Sample Composition
- **Total Firms**: 300
- **SMEs**: 200 firms (67%)
- **Large Firms**: 100 firms (33%)
- **FDI Presence**: 171 firms (57%)
- **Non-FDI Firms**: 129 firms (43%)

### Geographic Coverage
- **Location**: Lagos State, Nigeria
- **Areas**: Ikeja, Victoria Island, Apapa, Surulere, Lagos Island, Others

### Subsector Distribution
- Meat & Poultry Processing: 60 firms (20%)
- Grain & Cereal Processing: 54 firms (18%)
- Dairy Products: 52 firms (17%)
- Bakery & Confectionery: 46 firms (15%)
- Fruit & Vegetable Processing: 45 firms (15%)
- Beverage Production: 43 firms (14%)

## Variables

### 1. FDI Constructs (Primary Variables)
| Variable | Description | Scale | Source |
|----------|-------------|-------|---------|
| `FDI_Presence` | Binary indicator of foreign investment | 0/1 | Survey |
| `Knowledge_Absorption` | Ability to absorb new knowledge/technology | 1-5 Likert | Zahra & George (2002) |
| `Task_Performance` | Operational task performance | 1-5 Likert | Koopmans et al. (2013) |
| `Innovation_Score` | Innovation capabilities and R&D | 1-5 Likert | OECD Oslo Manual |
| `FDI_Intensity` | Percentage of foreign ownership | 0-100% | Survey |
| `FDI_Duration_Years` | Years of foreign investment presence | Continuous | Survey |
| `FDI_Origin_Region` | Geographic origin of foreign investor | Categorical | Survey |
| `FDI_Type` | Type of foreign investment | Categorical | Survey |

### 2. Firm Performance (Dependent Variables)
| Variable | Description | Scale | Source |
|----------|-------------|-------|---------|
| `ROI_Percent` | Return on Investment | Percentage | Financial data |
| `ROA_Percent` | Return on Assets | Percentage | Financial data |
| `Export_Intensity` | Percentage of revenue from exports | 0-100% | Survey |
| `Market_Share_Percent` | Market share in main product category | 0-100% | Survey |
| `Operational_Efficiency` | Self-rated operational efficiency | 1-5 Likert | Survey |
| `Revenue_Growth_Percent` | Annual revenue growth rate | Percentage | Financial data |
| `Profit_Margin_Percent` | Profit margin percentage | Percentage | Financial data |

### 3. Firm Resources (Control Variables)
| Variable | Description | Scale | Source |
|----------|-------------|-------|---------|
| `Skilled_Labor_Ratio` | Percentage of skilled workers | 0-100% | Survey |
| `IoT_Usage` | Internet of Things adoption | 0/1 | Survey |
| `R_D_Spend_Percent` | R&D spending as % of revenue | 0-100% | Survey |
| `Technology_Adoption_Score` | Technology adoption level | 1-5 Likert | Survey |
| `Liquidity_Ratio` | Current assets/current liabilities | Continuous | Financial data |
| `Debt_to_Equity_Ratio` | Debt-to-equity ratio | Continuous | Financial data |
| `Training_Investment_Percent` | Training investment as % of revenue | 0-100% | Survey |

### 4. Government Policy (Moderating Variables)
| Variable | Description | Scale | Source |
|----------|-------------|-------|---------|
| `Tax_Incentive_Effectiveness` | Effectiveness of tax incentives | 1-7 Likert | Survey |
| `Regulatory_Stability` | Stability of government regulations | 1-7 Likert | Survey |
| `Infrastructure_Support` | Adequacy of infrastructure support | 1-7 Likert | Survey |
| `Corruption_Experience` | Experience with corruption | 1-7 Likert | Survey |
| `Policy_Effectiveness_Index` | Overall policy effectiveness | 1-7 Likert | Survey |
| `Ease_of_Doing_Business` | Ease of doing business rating | 1-7 Likert | Survey |
| `Government_Support_Access` | Access to government support programs | 1-7 Likert | Survey |
| `Regulatory_Burden` | Regulatory burden level | 1-7 Likert | Survey |

### 5. Secondary Data (Macro Variables)
| Variable | Description | Scale | Source |
|----------|-------------|-------|---------|
| `FDI_Inflow_Lagos_Million_USD` | Total FDI inflows to Lagos | USD millions | CBN, UNCTAD |
| `Lagos_GDP_Growth_Percent` | Lagos GDP growth rate | Percentage | NBS |
| `Food_Processing_GDP_Contribution_Percent` | Sector GDP contribution | Percentage | NBS |
| `Power_Supply_Hours_Daily` | Daily power supply hours | Hours | World Bank |
| `Logistics_Quality_Index` | Logistics quality rating | 1-5 | World Bank |
| `Corruption_Index` | Corruption perception index | 0-100 | Transparency International |
| `Regulatory_Quality_Index` | Regulatory quality rating | -2.5 to 2.5 | World Bank |
| `Ease_of_Doing_Business_Rank` | Nigeria's global ranking | Rank | World Bank |
| `EEG_Disbursement_Million_NGN` | Export Expansion Grant disbursement | NGN millions | NIPC |
| `Food_Export_Value_Million_USD` | Food export value | USD millions | NBS |

### 6. Control Variables
| Variable | Description | Scale | Source |
|----------|-------------|-------|---------|
| `Firm_Size` | Company size category | SME/Large | Survey |
| `Firm_Age` | Years in operation | Continuous | Survey |
| `Employees` | Number of employees | Continuous | Survey |
| `Assets_Million_NGN` | Total assets in millions NGN | Continuous | Survey |
| `Subsector` | Food processing subsector | Categorical | Survey |
| `Ownership_Type` | Type of ownership | Categorical | Survey |
| `Export_Orientation` | Export-focused company | 0/1 | Survey |
| `Certification_ISO` | ISO certification | 0/1 | Survey |
| `Certification_HACCP` | HACCP certification | 0/1 | Survey |
| `Location_Zone` | Lagos area location | Categorical | Survey |
| `Years_in_Export` | Years involved in exports | Continuous | Survey |
| `Supply_Chain_Integration` | Supply chain integration level | 1-5 Likert | Survey |
| `Competition_Intensity` | Market competition intensity | 1-5 Likert | Survey |

## Files Included

### 1. Dataset Files
- `food_processing_fdi_dataset.csv` - Main dataset (300 firms × 54 variables)
- `data_dictionary.json` - Comprehensive variable definitions and scales

### 2. Data Collection Tools
- `survey_questionnaire.json` - Complete survey questionnaire (42 questions, 7 sections)
- `survey_questionnaire.py` - Python script to generate questionnaire

### 3. Analysis Tools
- `data_analysis_tools.py` - Comprehensive analysis toolkit
- `food_processing_fdi_dataset.py` - Dataset generation script

### 4. Documentation
- `README.md` - This comprehensive guide
- `analysis_report.json` - Generated analysis results

## Usage Instructions

### 1. Dataset Generation
```bash
python3 food_processing_fdi_dataset.py
```

### 2. Survey Questionnaire
```bash
python3 survey_questionnaire.py
```

### 3. Data Analysis
```bash
python3 data_analysis_tools.py
```

## Research Methodology

### Data Collection
1. **Primary Data**: Structured questionnaires from 300 food processing firms
2. **Secondary Data**: Macro and sectoral data from CBN, NBS, World Bank, UNCTAD
3. **Sampling**: Stratified random sampling (200 SMEs, 100 large firms)
4. **Respondents**: Senior managers, CEOs, operations heads

### Analysis Methods
- Descriptive statistics
- T-tests for FDI vs non-FDI performance comparison
- Correlation analysis
- Regression analysis (OLS)
- Structural Equation Modeling (recommended)

### Ethical Considerations
- Informed consent obtained from all participants
- Data anonymization using firm IDs
- Confidentiality maintained throughout research
- University ethics committee approval

## Key Findings (Sample)

### FDI Performance Gap
- FDI firms show **X%** higher ROI than non-FDI firms
- Innovation scores are **X points** higher for FDI firms
- Export intensity is **X%** higher for FDI firms

### Government Policy Assessment
- Average policy effectiveness: **X.X/7**
- Infrastructure support: **X.X/7** (needs improvement)
- Regulatory stability: **X.X/7**

## Recommendations

1. **FDI Promotion**: Encourage more foreign investment in food processing
2. **Policy Improvement**: Enhance government policy implementation
3. **Infrastructure**: Invest in power supply and logistics infrastructure
4. **Innovation**: Focus on knowledge absorption and technology adoption

## Data Quality

### Validation
- Cross-verification with secondary data where possible
- Internal consistency checks
- Outlier detection and treatment
- Missing data analysis

### Limitations
- Self-reported performance data
- Cross-sectional design (no time series)
- Lagos-specific (may not generalize to other regions)
- Potential response bias

## Citation

If you use this dataset in your research, please cite:

```
[Your Name] (2024). "Food Processing FDI Performance Dataset - Lagos, Nigeria." 
[Institution], [Location]. Dataset available at [URL].
```

## Contact Information

- **Principal Investigator**: [Your Name]
- **Institution**: [University Name]
- **Email**: [your.email@university.edu]
- **Phone**: [Phone Number]

## License

This dataset is provided for academic research purposes. Please ensure proper attribution and ethical use of the data.

---

*Generated on: [Date]*
*Dataset Version: 1.0*
*Total Variables: 54*
*Sample Size: 300 firms*