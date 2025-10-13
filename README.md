# Food Processing Firms FDI Research Dataset

## 🎯 Research Topic
**The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy**

## 📁 Dataset Overview
This repository contains a comprehensive, fabricated dataset designed for academic research on the relationship between Foreign Direct Investment (FDI) and firm performance in Nigeria's food processing sector, with a focus on government policy moderation effects.

### 🔢 Dataset Statistics
- **Total Firms**: 300 food processing companies
- **Geographic Scope**: Lagos State, Nigeria  
- **Time Period**: 2022-2024 (cross-sectional with time-varying macro data)
- **Sample Structure**: 200 SMEs + 100 Large firms (stratified sampling)
- **FDI Presence**: 181 firms (60.3%) have FDI
- **Variables**: 113 total variables across firm, FDI, performance, policy, and macro dimensions

## 📊 Files Included

### 🗂️ Main Datasets
1. **`integrated_food_processing_dataset.csv`** - Complete integrated dataset (300 firms × 113 variables)
2. **`primary_survey_data.csv`** - Primary firm-level survey data (300 firms × 44 variables)  
3. **`secondary_macro_data.csv`** - Secondary macro/sectoral data (6 years × 46 variables)

### 📋 Documentation
4. **`data_dictionary.md`** - Comprehensive variable definitions and measurement scales
5. **`methodology_documentation.md`** - Detailed research methodology and data collection procedures
6. **`README.md`** - This overview file

### 🔧 Code Files
7. **`generate_primary_data.py`** - Script to generate primary survey data
8. **`generate_secondary_data.py`** - Script to generate secondary macro/sectoral data
9. **`integrate_datasets.py`** - Script to merge primary and secondary data
10. **`sample_analysis.py`** - Complete sample analysis demonstrating key research approaches

## 🎯 Research Framework

### Research Questions
1. **Direct Effects**: How does FDI presence/intensity affect food processing firm performance?
2. **Mechanisms**: Through what channels (knowledge absorption, innovation, task performance) does FDI influence performance?
3. **Moderation**: How do Nigerian government policies moderate the FDI-performance relationship?
4. **Heterogeneity**: Do effects vary across firm sizes, subsectors, and time periods?

### Theoretical Foundation
- **Resource-Based View (RBV)** - Firm resources and capabilities
- **Dynamic Capabilities Theory** - Knowledge absorption and innovation
- **Institutional Theory** - Government policy environment effects
- **FDI Theory** - Internalization and eclectic paradigm

## 📈 Key Variables

### 🎯 Dependent Variables (Performance)
- `overall_performance_index` - Composite performance measure (0-10 scale)
- `roi_percent` - Return on Investment (%)
- `roa_percent` - Return on Assets (%)
- `export_intensity_percent` - Export revenue as % of total revenue
- `operational_efficiency_score` - Operational efficiency (1-10 scale)

### 🏭 Independent Variables (FDI)
- `fdi_presence` - Binary FDI indicator (0/1)
- `fdi_intensity` - FDI amount relative to firm assets
- `fdi_type` - Type of FDI (Greenfield, M&A, Joint Venture)
- `fdi_origin_country` - Origin country of FDI
- `knowledge_absorption_score` - Composite score (1-5 scale)
- `innovation_score` - Innovation capability (1-5 scale)

### 🏛️ Moderating Variables (Government Policy)
- `policy_effectiveness_index` - Composite policy effectiveness (1-7 scale)
- `gp_tax_incentives_effectiveness` - Tax incentive effectiveness perception
- `gp_regulatory_stability` - Regulatory stability perception
- `gp_infrastructure_support` - Infrastructure support quality
- `macro_ease_of_doing_business_score` - World Bank EODB score

### 🎛️ Control Variables
- **Firm**: Size, age, subsector, ownership type, resources
- **Macro**: GDP growth, inflation, exchange rate, interest rate
- **Institutional**: Corruption index, regulatory quality, government effectiveness

## 🔍 Sample Analysis Results

### Key Findings from Sample Analysis
- **FDI Prevalence**: 60.3% of firms have FDI presence
- **Performance Gap**: FDI firms outperform non-FDI firms by 0.06 points (modest effect)
- **Policy Moderation**: FDI effects stronger in high policy effectiveness environments (0.076 vs -0.002 correlation)
- **Mechanisms**: Innovation shows strongest FDI differential (+0.09 points for FDI firms)
- **Sectoral Variation**: Sugar & Confectionery has highest FDI rate (79%), Fruits & Vegetables highest performance

### Statistical Summary
- **R-squared**: Basic FDI model explains 12.3% of performance variance
- **Mechanism Model**: Adding FDI constructs improves R-squared to 15.9%
- **Policy Moderation**: Interaction effects provide additional explanatory power

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Quick Start
```python
# Load the integrated dataset
import pandas as pd
df = pd.read_csv('integrated_food_processing_dataset.csv')

# Basic analysis
print(f"Dataset shape: {df.shape}")
print(f"FDI firms: {df['fdi_presence'].sum()} ({df['fdi_presence'].mean()*100:.1f}%)")

# Performance comparison
fdi_performance = df[df['fdi_presence']==1]['overall_performance_index'].mean()
non_fdi_performance = df[df['fdi_presence']==0]['overall_performance_index'].mean()
print(f"Performance gap: {fdi_performance - non_fdi_performance:.3f}")
```

### Run Complete Analysis
```bash
python3 sample_analysis.py
```

## 📊 Data Structure

### Primary Data (Firm-Level)
- **FDI Constructs**: Knowledge absorption, task performance, innovation (Likert 1-5)
- **Performance Metrics**: ROI, ROA, export intensity, market share, efficiency
- **Firm Resources**: Human capital, technology adoption, financial resources
- **Policy Perception**: Tax incentives, regulatory stability, infrastructure (Likert 1-7)

### Secondary Data (Macro-Level)
- **FDI Flows**: Total FDI, sectoral FDI, origin countries (CBN, UNCTAD)
- **Economic Indicators**: GDP growth, inflation, exchange rates (NBS, CBN)
- **Governance**: Ease of business, corruption, regulatory quality (World Bank)
- **Infrastructure**: Power, logistics, internet, financial inclusion

## 🎓 Research Applications

### Suitable for:
- **Academic Research**: PhD dissertations, journal articles, conference papers
- **Policy Analysis**: Government policy evaluation and recommendations
- **Teaching**: Graduate courses in international business, development economics
- **Methodology Demonstration**: Mixed-methods research, survey design, data integration

### Analysis Approaches:
- **Descriptive Analysis**: Cross-tabulations, correlation analysis
- **Inferential Statistics**: OLS regression, moderated regression
- **Advanced Methods**: SEM, propensity score matching, multilevel modeling
- **Robustness Checks**: Alternative specifications, sensitivity analysis

## ⚠️ Important Notes

### Data Fabrication
- This is a **fabricated dataset** created for research demonstration purposes
- Values are realistic and based on actual Nigerian economic conditions and literature
- **Not suitable for real policy decisions** - use actual data for policy analysis
- Designed to demonstrate research methodology and analytical approaches

### Ethical Considerations
- Real data collection would require IRB approval and informed consent
- Firm anonymization protocols are demonstrated in the dataset structure
- Data protection and security measures are outlined in methodology documentation

### Limitations
- Cross-sectional design limits causal inference
- Self-reported performance data may have bias
- Geographic scope limited to Lagos State
- Temporal scope reflects 2022-2024 economic conditions

## 📚 References and Sources

### Data Sources Simulated
- **Central Bank of Nigeria (CBN)**: FDI flows, exchange rates, financial data
- **National Bureau of Statistics (NBS)**: GDP, inflation, sectoral data
- **World Bank**: Governance indicators, ease of business rankings
- **UNCTAD**: Global FDI statistics and classifications
- **Nigerian Investment Promotion Commission (NIPC)**: Policy incentives

### Theoretical References
- Zahra, S. A., & George, G. (2002). Absorptive capacity: A review, reconceptualization, and extension
- Koopmans, L., et al. (2013). Development of an individual work performance questionnaire
- OECD Oslo Manual (2018). Guidelines for collecting, reporting and using data on innovation

## 🤝 Contributing

This dataset is designed for educational and research purposes. Suggestions for improvements to the methodology, additional variables, or analytical approaches are welcome.

## 📄 License

This fabricated dataset is provided for academic and educational use. Please cite appropriately if used in research or teaching materials.

## 📧 Contact

For questions about the dataset structure, methodology, or potential applications, please refer to the comprehensive documentation provided in `data_dictionary.md` and `methodology_documentation.md`.

---

**Generated**: October 2024  
**Version**: 1.0  
**Status**: Research Demonstration Dataset