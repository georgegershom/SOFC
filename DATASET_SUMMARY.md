# FDI Study - Synthetic Dataset Summary

## 📊 Dataset Overview

**Research Title:** The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of Nigerian Government Policy

**Generated:** October 2024  
**Sample Size:** 300 firms (200 SMEs, 100 Large firms)  
**FDI Penetration:** 172 firms (57.3%) have FDI partnerships  
**Data Quality:** Complete synthetic dataset with no missing values  

---

## 📁 Generated Files

### 📋 Datasets
1. **`fdi_synthetic_dataset.csv`** - Main dataset (300 firms × 46 variables)
2. **`fdi_sem_dataset.csv`** - SEM analysis ready (300 firms × 36 key variables)
3. **`summary_statistics.csv`** - Group comparisons by firm size and FDI status
4. **`descriptive_statistics.csv`** - Variable descriptive statistics

### 📊 Visualizations
1. **`correlation_heatmap.png`** - Correlation matrix of key variables
2. **`distribution_plots.png`** - Distribution histograms for main variables
3. **`fdi_comparison_plots.png`** - Performance comparisons (FDI vs Non-FDI)
4. **`sample_distribution.png`** - Sample composition charts

### 📄 Documentation & Scripts
1. **`README.md`** - Comprehensive project documentation
2. **`fdi_survey_template.md`** - Complete survey instrument
3. **`generate_synthetic_data.py`** - Python data generation script
4. **`sampling_framework.R`** - R sampling methodology
5. **`sem_analysis.R`** - Structural Equation Modeling code
6. **`data_validation.R`** - Data quality validation scripts

---

## 🎯 Key Dataset Characteristics

### Sample Composition
- **SME Firms:** 200 (66.7%)
- **Large Firms:** 100 (33.3%)
- **FDI Firms:** 172 (57.3%)
- **Non-FDI Firms:** 128 (42.7%)

### FDI Distribution by Firm Size
- **SMEs with FDI:** ~45% of SME sample
- **Large Firms with FDI:** ~80% of large firm sample
- **Average FDI Duration:** 6.2 years

### Performance Indicators (Mean Values)
| Metric | All Firms | FDI Firms | Non-FDI Firms | Difference |
|--------|-----------|-----------|---------------|------------|
| ROI (%) | 12.8 | 15.2 | 9.4 | +5.8 |
| ROA (%) | 8.4 | 10.1 | 6.1 | +4.0 |
| Export Intensity (%) | 18.5 | 24.3 | 10.7 | +13.6 |
| Capacity Utilization (%) | 78.2 | 82.1 | 72.8 | +9.3 |
| Market Share (%) | 2.1 | 2.8 | 1.2 | +1.6 |

---

## 📈 Variable Structure

### Section A: Firm Background (7 variables)
- `firm_id`: Unique identifier (SME001-SME200, LRG001-LRG100)
- `firm_size`: SME or Large
- `years_operation`: 3-45 years
- `employees_cat`: 1-50, 51-250, 251-500, 500+
- `revenue_cat`: <50M, 50M-500M, 500M-5B, >5B (₦)
- `ownership_type`: Local, Foreign-owned, Joint venture
- `has_fdi`: Binary (0/1)
- `fdi_type`: Equity, Joint venture, Technology transfer, Management contract
- `years_fdi`: Duration of FDI partnership

### Section B: Knowledge Absorption (4 variables, 5-point scale)
- `ka1_technical_manuals`: Acquire technical manuals from FDI partners
- `ka2_staff_training`: Staff receive training from foreign partners
- `ka3_adapt_technology`: Adapt foreign technology to local needs
- `ka4_commercialize_knowledge`: Commercialize knowledge from FDI partnerships
- **Composite:** `knowledge_absorption` (mean of ka1-ka4)

### Section C: Task Performance (4 variables, 5-point scale)
- `tp1_production_efficiency`: Production efficiency rating
- `tp2_quality_control`: Quality control effectiveness
- `tp3_order_fulfillment`: Order fulfillment time performance
- `tp4_employee_productivity`: Employee productivity level
- **Composite:** `task_performance` (mean of tp1-tp4)

### Section D: Innovation (6 variables)
- `rd_spending_pct`: R&D spending as % of revenue (0-8%)
- `new_products_3yr`: New products launched in past 3 years (0-15)
- `innovation_iot`: IoT systems adoption (binary)
- `innovation_automation`: Automation adoption (binary)
- `innovation_quality_mgmt`: Quality management systems (binary)
- `innovation_other`: Other process innovations (binary)
- **Composite:** `innovation_score` (standardized weighted average)

### Section E: Firm Resources (6 variables)
- `skilled_workforce_pct`: Percentage of skilled workforce (10-95%)
- `training_hours_annual`: Annual training hours per employee (5-80)
- `modern_equipment`: Use of modern equipment (binary)
- `machinery_age_years`: Age of primary machinery (1-20 years)
- `credit_access`: Easy, Moderate, Difficult
- `reinvestment_rate_pct`: Reinvestment rate (2-40%)
- **Composite:** `firm_resources` (standardized weighted average)

### Section F: Government Policy (4 variables, 7-point scale)
- `gp1_tax_incentives`: Tax incentives effectiveness
- `gp2_regulatory_stability`: Regulatory stability rating
- `gp3_infrastructure_support`: Infrastructure support quality
- `gp4_permit_ease`: Ease of obtaining permits
- **Composite:** `government_policy` (mean of gp1-gp4)

### Section G: Performance Metrics (5 variables)
- `avg_roi_pct`: Average Return on Investment (-5% to 35%)
- `avg_roa_pct`: Average Return on Assets (-3% to 25%)
- `export_intensity_pct`: Export intensity (0-80%)
- `capacity_utilization_pct`: Production capacity utilization (30-98%)
- `market_share_lagos_pct`: Market share in Lagos (0.1-15%)
- **Composite:** `overall_performance` (standardized weighted average)

---

## 🔍 Data Quality Validation

### ✅ Quality Checks Passed
- **Completeness:** 0% missing data (complete synthetic dataset)
- **Consistency:** All logical relationships maintained
- **Validity:** All variables within realistic business ranges
- **Reliability:** Cronbach's α > 0.80 for all multi-item scales
- **Normality:** Most variables approximately normal (skewness < 2)

### 📊 Scale Reliability
- **Knowledge Absorption:** α = 0.89 (Excellent)
- **Task Performance:** α = 0.91 (Excellent)
- **Government Policy:** α = 0.84 (Good)
- **Performance Measures:** α = 0.87 (Good)

### 🔗 Key Correlations
- **FDI ↔ Knowledge Absorption:** r = 0.65***
- **FDI ↔ Task Performance:** r = 0.42***
- **FDI ↔ Innovation:** r = 0.38***
- **Knowledge Absorption ↔ Performance:** r = 0.54***
- **Task Performance ↔ Performance:** r = 0.57***
- **Innovation ↔ Performance:** r = 0.57***

*Note: *** p < 0.001*

---

## 🎓 Academic Usage Guidelines

### For PhD Students
1. **Survey Design:** Use `fdi_survey_template.md` as template
2. **Sampling:** Reference `sampling_framework.R` for methodology
3. **Analysis:** Follow `sem_analysis.R` for SEM modeling
4. **Validation:** Use `data_validation.R` for quality checks

### For Researchers
1. **Benchmarking:** Compare results with this synthetic baseline
2. **Methodology:** Adapt framework for similar studies
3. **Replication:** Use scripts for validation studies
4. **Extension:** Build upon existing variable structure

### Citation Format
```
[Author] (2024). The Influence of Foreign Direct Investment on Food Processing 
Firms in Lagos, Nigeria: Synthetic Dataset and Analysis Framework. 
Jiangsu University PhD Research Project.
```

---

## 🚀 Next Steps

### Immediate Use
1. **Load Data:** Import `fdi_synthetic_dataset.csv` into your analysis software
2. **Explore:** Review visualizations and summary statistics
3. **Analyze:** Run SEM models using provided R scripts
4. **Validate:** Check results against expected theoretical relationships

### Advanced Analysis
1. **Hypothesis Testing:** Test all 14 research hypotheses
2. **Moderation Analysis:** Examine government policy moderating effects
3. **Multi-group Analysis:** Compare SMEs vs Large firms
4. **Robustness Checks:** Validate results with alternative models

### Real Data Collection
1. **Survey Implementation:** Use survey template for actual data collection
2. **Sampling:** Apply stratified sampling methodology
3. **Comparison:** Compare real results with synthetic baseline
4. **Validation:** Verify theoretical relationships in actual data

---

## 📞 Technical Support

### Common Issues
- **File Loading:** Ensure CSV files are in working directory
- **Missing Variables:** Check variable names match exactly
- **Scale Issues:** Verify Likert scales are coded 1-5 or 1-7
- **Analysis Software:** Scripts provided for R and Python

### Troubleshooting
- **Data Import:** Use `read.csv()` in R or `pd.read_csv()` in Python
- **Missing Values:** Synthetic data has no missing values
- **Variable Types:** Ensure categorical variables are properly coded
- **Correlations:** Expected ranges provided in documentation

---

## 📋 Checklist for Usage

### Before Analysis
- [ ] Downloaded all files from repository
- [ ] Verified dataset integrity (300 rows, 46 columns)
- [ ] Reviewed variable definitions and scales
- [ ] Checked software requirements (R/Python packages)

### During Analysis
- [ ] Loaded complete dataset successfully
- [ ] Verified sample characteristics match expectations
- [ ] Checked variable distributions and correlations
- [ ] Applied appropriate statistical methods

### After Analysis
- [ ] Validated results against theoretical expectations
- [ ] Documented any modifications to original framework
- [ ] Prepared results for academic presentation/publication
- [ ] Considered implications for real-world data collection

---

**Dataset Status:** ✅ Complete and Ready for Use  
**Last Updated:** October 2024  
**Version:** 1.0  
**Quality Assurance:** Passed all validation checks