# FDI Influence on Food Processing Firms - Synthetic Dataset

## 📚 PhD Research Project
**Institution:** Jiangsu University  
**Research Topic:** The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy  
**Generated:** October 13, 2025  

---

## 📊 Dataset Overview

This synthetic dataset was created for PhD research examining the relationship between Foreign Direct Investment (FDI) and firm performance in Nigeria's food processing industry. The data simulates survey responses from 300 food processing firms in Lagos, Nigeria, with realistic relationships based on theoretical frameworks and empirical patterns from existing literature.

### Key Statistics:
- **Total Sample Size:** 300 firms
- **SMEs:** 200 firms (66.7%)
- **Large Firms:** 100 firms (33.3%)
- **Firms with FDI:** 159 (53.0%)
- **Response Rate Simulation:** 70% (reflecting real-world survey conditions)
- **Missing Values:** ~2% (realistic non-response patterns)

---

## 📁 Files Generated

1. **`fdi_survey_data.csv`** - Main dataset in CSV format (79 KB)
2. **`fdi_survey_data.xlsx`** - Excel workbook with multiple sheets (101 KB)
   - Survey_Data: Main dataset
   - Variable_Labels: Complete data dictionary
   - Summary_Statistics: Descriptive statistics
   - Correlations: Correlation matrix
3. **`fdi_survey_data.dta`** - Stata format dataset (153 KB)
4. **`import_fdi_data.sps`** - SPSS syntax file for importing and labeling data
5. **`fdi_data_summary_report.txt`** - Detailed statistical summary report
6. **`fdi_data_visualization.png`** - Comprehensive visualization of key relationships
7. **`sem_validation_plots.png`** - SEM model validation plots

---

## 🏗️ Data Structure

### A. Firm Identification Variables
- `firm_code`: Unique identifier (SME001-SME200, LRG001-LRG100)
- `firm_size`: Categorical (SME/Large)
- `survey_date`: Date of survey completion
- `industrial_zone`: Lagos industrial zone location
- `response_time_minutes`: Time to complete survey

### B. Firm Characteristics
- `years_operation`: Years in business (1-50)
- `employees_category`: 1=1-50, 2=51-250, 3=251-500, 4=500+
- `revenue_category`: 1=<50M₦, 2=50M-500M₦, 3=500M-5B₦, 4=>5B₦
- `ownership_type`: 1=Local, 2=Foreign-owned, 3=Joint venture

### C. FDI Engagement Variables
- `has_fdi`: Binary (0=No, 1=Yes)
- `fdi_equity`: Equity investment (0/1)
- `fdi_joint_venture`: Joint venture (0/1)
- `fdi_tech_transfer`: Technology transfer (0/1)
- `fdi_mgmt_contract`: Management contract (0/1)
- `years_fdi_partnership`: Duration of FDI partnership

### D. Knowledge Absorption (Zahra & George Scale, 1-5)
- `ka_technical_manuals`: Acquisition of technical manuals
- `ka_staff_training`: Staff training from foreign partners
- `ka_tech_adaptation`: Technology adaptation to local needs
- `ka_knowledge_commercialization`: Knowledge commercialization
- `ka_average`: Composite score

### E. Task Performance (Koopmans et al. Scale, 1-5)
- `tp_production_efficiency`: Production efficiency rating
- `tp_quality_control`: Quality control rating
- `tp_order_fulfillment`: Order fulfillment performance
- `tp_employee_productivity`: Employee productivity rating
- `tp_average`: Composite score

### F. Innovation Metrics (OECD Oslo Manual)
- `rd_spending_percent`: R&D as % of revenue
- `new_products_3years`: New products launched (3-year period)
- `process_inn_iot`: IoT systems adoption (0/1)
- `process_inn_automation`: Automation adoption (0/1)
- `process_inn_quality`: Quality management systems (0/1)
- `process_inn_other`: Other innovations (0/1)
- `innovation_score`: Composite innovation score

### G. Firm Resources
- `hr_skilled_workforce_percent`: % of skilled workers
- `hr_training_hours`: Annual training hours per employee
- `tech_modern_equipment`: Modern equipment usage (0/1)
- `tech_machinery_age`: Age of machinery (years)
- `fin_access_credit`: Credit access (1=Easy, 2=Moderate, 3=Difficult)
- `fin_reinvestment_rate`: Reinvestment rate (%)
- `firm_resources_score`: Composite resources score

### H. Government Policy Perception (1-7 scale)
- `gp_tax_incentives`: Tax incentives effectiveness
- `gp_regulatory_stability`: Regulatory stability rating
- `gp_infrastructure_support`: Infrastructure support rating
- `gp_ease_permits`: Ease of obtaining permits
- `gp_average`: Composite policy score

### I. Performance Metrics
- `perf_roi`: Return on Investment (%)
- `perf_roa`: Return on Assets (%)
- `perf_export_intensity`: Export intensity (%)
- `perf_capacity_utilization`: Capacity utilization (%)
- `perf_market_share`: Market share in Lagos (%)
- `performance_composite`: Overall performance score (0-100)

---

## 📈 Key Findings from Synthetic Data

### Correlations with Performance:
- **FDI → Performance:** r = 0.771 (strong positive)
- **Knowledge Absorption → Performance:** r = 0.647
- **Task Performance → Performance:** r = 0.590
- **Innovation → Performance:** r = 0.568
- **Firm Resources → Performance:** r = 0.420
- **Government Policy → Performance:** r = 0.381

### Performance Differences:
- **ROI:** FDI firms (22.0%) vs Non-FDI firms (16.2%)
- **ROA:** FDI firms (15.5%) vs Non-FDI firms (11.2%)
- **Export Intensity:** FDI firms (28.6%) vs Non-FDI firms (0.0%)
- **Capacity Utilization:** FDI firms (75.2%) vs Non-FDI firms (58.9%)

---

## 🔬 SEM Model Structure

The dataset is structured for Structural Equation Modeling (SEM) analysis with:

### Measurement Model:
- **Knowledge Absorption** (4 reflective indicators)
- **Task Performance** (4 reflective indicators)
- **Innovation** (6 indicators)
- **Government Policies** (4 formative indicators)
- **Firm Resources** (6 indicators)
- **Performance** (5 indicators)

### Structural Model Paths:
1. **Direct Effects:** FDI → Performance
2. **Mediation Paths:** FDI → KA/TP/INN → Performance
3. **Moderation Effects:** Government Policy × FDI interactions

---

## 💻 Usage Instructions

### Loading in Python:
```python
import pandas as pd
df = pd.read_csv('fdi_survey_data.csv')
print(df.info())
print(df.describe())
```

### Loading in R:
```r
library(tidyverse)
data <- read_csv('fdi_survey_data.csv')
summary(data)
```

### Loading in SPSS:
1. Open SPSS
2. Run the syntax file: `import_fdi_data.sps`
3. Or directly open the Excel file

### Loading in Stata:
```stata
use "fdi_survey_data.dta", clear
describe
summarize
```

---

## 🎯 SEM Analysis with lavaan (R)

```r
library(lavaan)

model <- '
  # Measurement model
  KA =~ ka_technical_manuals + ka_staff_training + ka_tech_adaptation + ka_knowledge_commercialization
  TP =~ tp_production_efficiency + tp_quality_control + tp_order_fulfillment + tp_employee_productivity
  INN =~ rd_spending_percent + new_products_3years + process_inn_iot + process_inn_automation
  
  # Structural model
  performance_composite ~ has_fdi + KA + TP + INN + firm_resources_score
  KA ~ has_fdi
  TP ~ has_fdi
  INN ~ has_fdi
  
  # Moderation (interaction terms need to be created separately)
  performance_composite ~ has_fdi:gp_average
'

fit <- sem(model, data = df)
summary(fit, fit.measures = TRUE, standardized = TRUE)
```

---

## ⚠️ Important Notes

1. **Synthetic Data:** This is artificially generated data for research purposes. While relationships are realistic, actual empirical data should be collected for final analysis.

2. **Ethical Considerations:** No real firm data is included. All firm codes and data points are fictional.

3. **Statistical Validity:** The data includes realistic:
   - Missing value patterns (~2%)
   - Variable distributions based on empirical literature
   - Correlation structures reflecting theoretical expectations
   - Heterogeneity between SMEs and large firms

4. **Limitations:**
   - Simplified linear relationships (real-world may be non-linear)
   - Perfect stratification (real sampling may have imbalances)
   - No temporal dynamics (cross-sectional only)

---

## 📊 Visualization Highlights

The generated visualizations include:
1. FDI distribution by firm size
2. Performance comparisons (box plots)
3. Knowledge absorption components
4. Innovation metrics comparison
5. ROI distribution histograms
6. Government policy ratings
7. Correlation heatmap
8. Human resources comparison
9. Market share relationships
10. Task performance radar charts
11. Export intensity distributions
12. Performance driver coefficients

---

## 🔄 Reproducibility

To regenerate the dataset with different parameters:

```python
python3 generate_fdi_survey_data.py
```

Modify the generator parameters in the script:
- Change sample sizes: `FDIDataGenerator(n_sme=200, n_large=100)`
- Adjust missing rate: `add_missing_values(missing_rate=0.03)`
- Modify random seed for different samples: `np.random.seed(42)`

---

## 📚 References

### Theoretical Frameworks:
1. **Knowledge Absorption:** Zahra, S. A., & George, G. (2002). Absorptive capacity: A review, reconceptualization, and extension.
2. **Task Performance:** Koopmans, L., et al. (2011). Conceptual frameworks of individual work performance.
3. **Innovation Metrics:** OECD Oslo Manual (2018). Guidelines for collecting and interpreting innovation data.
4. **SEM Methodology:** Hair, J. F., et al. (2019). Multivariate data analysis (8th ed.).

---

## 📧 Contact

For questions about the dataset or research methodology:
- Research Institution: Jiangsu University
- Dataset Generator Version: 1.0
- Generated: October 13, 2025

---

## 📄 License

This synthetic dataset is provided for academic research purposes only. Users should:
- Acknowledge the source when using the data
- Not represent synthetic data as real empirical findings
- Collect actual data for publication-quality research

---

## 🚀 Next Steps

1. **Validate Measurement Model:** Confirm factor structure with CFA
2. **Test Hypotheses:** Run full SEM model with mediation and moderation
3. **Robustness Checks:** Multi-group analysis, alternative specifications
4. **Collect Real Data:** Use this as a template for actual survey deployment

---

*End of Documentation*