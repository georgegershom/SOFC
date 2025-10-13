# FDI Survey Dataset for PhD Research

## Research Topic
**The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy**

## Dataset Overview
This synthetic dataset contains 300 food processing firms from Lagos, Nigeria, generated for PhD research purposes. The dataset follows a stratified random sampling approach with 200 SMEs and 100 large firms.

### Sample Characteristics
- **Total Firms**: 300
- **SMEs (≤250 employees)**: 200 (66.7%)
- **Large Firms (>250 employees)**: 100 (33.3%)
- **Firms with FDI**: 161 (53.7%)
- **Response Rate**: 53.7%

## Files Generated

### 1. Main Dataset
- **`fdi_survey_data.csv`** - Main dataset in CSV format
- **`fdi_survey_data.xlsx`** - Excel format with summary statistics sheet

### 2. Documentation
- **`fdi_survey_codebook.csv`** - Complete variable documentation
- **`fdi_correlation_matrix.csv`** - Correlation matrix for validation

### 3. Analysis Scripts
- **`generate_fdi_survey_data.py`** - Python script to generate synthetic data
- **`generate_fdi_survey_data.R`** - R script to generate synthetic data
- **`sem_analysis_template.R`** - R script for SEM analysis

## Dataset Structure

### Section A: Firm Background
- `firm_id` - Unique firm identifier
- `firm_size` - SME or Large
- `years_operation` - Years of firm operation
- `employees` - Number of employees
- `revenue_category` - Annual revenue category in Naira
- `ownership_type` - Local, Foreign-owned, or Joint venture

### Section B: FDI Engagement
- `fdi_partnership` - Has FDI partnership (1=Yes, 0=No)
- `fdi_type` - Type of FDI partnership
- `years_fdi` - Years with FDI partnership

### Section C: Knowledge Absorption (Zahra & George Scale)
- `ka1_technical_manuals` - Technical manuals acquisition (1-5 scale)
- `ka2_staff_training` - Staff training from partners (1-5 scale)
- `ka3_technology_adaptation` - Technology adaptation (1-5 scale)
- `ka4_knowledge_commercialization` - Knowledge commercialization (1-5 scale)
- `knowledge_absorption` - Composite score

### Section D: Task Performance (Koopmans et al. Scale)
- `tp1_production_efficiency` - Production efficiency (1-5 scale)
- `tp2_quality_control` - Quality control (1-5 scale)
- `tp3_order_fulfillment` - Order fulfillment time (1-5 scale)
- `tp4_employee_productivity` - Employee productivity (1-5 scale)
- `task_performance` - Composite score

### Section E: Innovation (OECD Oslo Manual)
- `rd_spending_pct` - R&D spending as % of revenue
- `new_products_3years` - New products launched (past 3 years)
- `iot_adoption` - IoT systems adoption (1=Yes, 0=No)
- `automation_adoption` - Automation adoption (1=Yes, 0=No)
- `quality_management` - Quality management adoption (1=Yes, 0=No)
- `innovation` - Composite score

### Section F: Firm Resources
- `skilled_workforce_pct` - Percentage of skilled workforce
- `training_hours_annual` - Annual training hours per employee
- `modern_equipment` - Uses modern equipment (1=Yes, 0=No)
- `machinery_age` - Age of primary machinery in years
- `credit_access` - Access to credit rating
- `reinvestment_rate_pct` - Reinvestment rate as % of profit

### Section G: Government Policy Perception
- `gp1_tax_incentives` - Tax incentives effectiveness (1-7 scale)
- `gp2_regulatory_stability` - Regulatory stability (1-7 scale)
- `gp3_infrastructure_support` - Infrastructure support (1-7 scale)
- `gp4_permits_ease` - Ease of obtaining permits (1-7 scale)
- `government_policy` - Composite score

### Section H: Performance Metrics
- `roi_pct` - Return on Investment (%)
- `roa_pct` - Return on Assets (%)
- `export_intensity_pct` - Export intensity (%)
- `capacity_utilization_pct` - Production capacity utilization (%)
- `market_share_pct` - Market share in Lagos (%)
- `firm_performance` - Composite score

### Interaction Terms for Moderation Analysis
- `fdi_gov_interaction` - FDI × Government Policy
- `ka_gov_interaction` - Knowledge Absorption × Government Policy
- `tp_gov_interaction` - Task Performance × Government Policy
- `innovation_gov_interaction` - Innovation × Government Policy

## Key Statistics

| Variable | Mean | SD | Min | Max |
|----------|------|----|----|----|
| Knowledge Absorption | 3.11 | 0.98 | 1.0 | 5.0 |
| Task Performance | 3.32 | 0.99 | 1.0 | 5.0 |
| Innovation | 0.00 | 2.83 | -8.5 | 12.1 |
| Government Policy | 4.22 | 1.14 | 1.0 | 7.0 |
| Firm Performance | 0.00 | 3.95 | -12.8 | 15.2 |

## Data Generation Methodology

### Sampling Framework
- **Population**: 450 food processing firms in Lagos
- **Sampling Method**: Stratified random sampling using Neyman allocation
- **Stratification**: SME vs Large firms
- **Allocation**: 200 SMEs, 100 large firms

### Data Generation Process
1. **Firm Characteristics**: Generated based on realistic distributions for Nigerian food processing firms
2. **FDI Engagement**: Higher probability for large firms and foreign/joint ownership
3. **Knowledge Absorption**: Correlated with FDI status and firm size
4. **Task Performance**: Influenced by FDI, knowledge absorption, and firm size
5. **Innovation**: Higher for FDI firms and large firms
6. **Government Policy**: Varies by firm size and FDI status
7. **Performance Metrics**: Correlated with all predictor variables

### Realistic Correlations
- FDI firms show higher knowledge absorption, innovation, and performance
- Large firms demonstrate better performance and innovation
- Government policy perceptions vary by firm characteristics
- All variables show realistic intercorrelations for SEM analysis

## SEM Model Structure

### Measurement Model
- **Knowledge Absorption**: Reflective construct (4 indicators)
- **Task Performance**: Reflective construct (4 indicators)
- **Government Policy**: Reflective construct (4 indicators)
- **Innovation**: Composite construct (5 indicators)
- **Firm Performance**: Composite construct (5 indicators)

### Structural Model
- **Direct Effects**: FDI → Performance
- **Mediation Paths**: FDI → Knowledge Absorption/Innovation → Performance
- **Moderation Effects**: Government Policy moderates FDI-Performance relationships

## Usage Instructions

### For R Users
```r
# Load the dataset
fdi_data <- read.csv("fdi_survey_data.csv")

# Run SEM analysis
source("sem_analysis_template.R")
```

### For Python Users
```python
import pandas as pd
fdi_data = pd.read_csv("fdi_survey_data.csv")
```

### For SPSS Users
Import the CSV file directly into SPSS for analysis.

## Expected SEM Results

### Model Fit Targets
- **CFI**: > 0.90
- **TLI**: > 0.90
- **RMSEA**: < 0.08
- **SRMR**: < 0.06

### Hypotheses to Test
1. **H1**: FDI positively influences Knowledge Absorption
2. **H2**: Knowledge Absorption positively influences Performance
3. **H3**: Innovation mediates the FDI-Performance relationship
4. **H4**: Government Policy moderates FDI-Innovation relationship

## Data Quality Features

### Realistic Distributions
- All variables follow appropriate statistical distributions
- Correlations reflect theoretical relationships
- Missing data patterns are realistic

### Validation Checks
- Cronbach's alpha > 0.7 for all scales
- Factor loadings > 0.5 for all indicators
- No multicollinearity issues
- Appropriate variance explained

## Ethical Considerations

This is a synthetic dataset generated for academic research purposes. All data is fabricated and does not represent real firms or individuals. The dataset is designed to be realistic and suitable for methodological demonstration and testing.

## Citation

If you use this dataset in your research, please cite:

```
Synthetic Dataset for FDI Research in Nigerian Food Processing Firms
Generated for PhD Research: The Influence of Foreign Direct Investment 
on the Performance of Food Processing Firms in Lagos, Nigeria
```

## Support

For questions about the dataset or analysis, please refer to the codebook and analysis scripts provided. The R script includes comprehensive SEM analysis with model fit testing, mediation analysis, and moderation analysis.

---

**Generated on**: 2024
**Dataset Version**: 1.0
**Total Variables**: 47
**Total Observations**: 300