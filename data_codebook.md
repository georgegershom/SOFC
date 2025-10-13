# DATA CODEBOOK
## The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria

**Dataset:** `fdi_lagos_survey_data.csv`  
**Sample Size:** 210 firms (70% response rate)  
**Data Collection Period:** July - September 2024  
**Research Institution:** Jiangsu University, PhD Research

---

## VARIABLE DICTIONARY

### IDENTIFIERS & METADATA

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `firm_code` | String | Unique firm identifier | FIRM001 - FIRM210 |
| `survey_date` | Date | Date of survey completion | DD/MM/YYYY format |
| `firm_size` | Categorical | Firm size classification | SME, Large |

---

### SECTION A: FIRM BACKGROUND

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `years_operation` | Integer | Years firm has been in operation | 5-40 years |
| `num_employees` | Categorical | Number of employees | 1-50, 51-250, 251-500, 500+ |
| `annual_revenue` | Categorical | Annual revenue in Naira | <50M, 50M-500M, 500M-5B, >5B |
| `ownership_type` | Categorical | Ownership structure | Local, Foreign-owned, Joint venture |
| `has_fdi` | Binary | Has foreign direct investment | Yes, No |
| `fdi_type` | Categorical | Type of FDI arrangement | Equity, Joint venture, Technology transfer, Management contract (blank if no FDI) |
| `years_fdi` | Integer | Years with FDI partnership | 0-15 years (0 if no FDI) |

---

### SECTION B: KNOWLEDGE ABSORPTION (Zahra & George Absorptive Capacity Scale)

**Scale:** 1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree

| Variable | Description | Construct |
|----------|-------------|-----------|
| `ka1_technical_manuals` | We regularly acquire technical manuals from FDI partners | Acquisition |
| `ka2_staff_training` | Our staff receive training from foreign partners | Assimilation |
| `ka3_adapt_technology` | We adapt foreign technology to local production needs | Transformation |
| `ka4_commercialize_knowledge` | We commercialize knowledge from FDI partnerships | Exploitation |

**Composite Score:** KA_SCORE = (ka1 + ka2 + ka3 + ka4) / 4

---

### SECTION C: TASK PERFORMANCE (Koopmans et al. Individual Work Performance Scale)

**Scale:** 1 = Very Poor, 2 = Poor, 3 = Fair, 4 = Good, 5 = Excellent

| Variable | Description |
|----------|-------------|
| `tp1_production_efficiency` | Production efficiency rating |
| `tp2_quality_control` | Quality control effectiveness |
| `tp3_order_fulfillment` | Order fulfillment time performance |
| `tp4_employee_productivity` | Employee productivity level |

**Composite Score:** TP_SCORE = (tp1 + tp2 + tp3 + tp4) / 4

---

### SECTION D: INNOVATION (OECD Oslo Manual Framework)

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `rd_spending_pct` | Float | R&D spending as % of revenue | 0-8% |
| `new_products_3yrs` | Integer | New products launched in past 3 years | 0-15 products |
| `innov_iot` | Binary | IoT systems adopted | 0 = No, 1 = Yes |
| `innov_automation` | Binary | Automation systems adopted | 0 = No, 1 = Yes |
| `innov_quality_mgmt` | Binary | Quality management systems adopted | 0 = No, 1 = Yes |
| `innov_other` | Binary | Other process innovations | 0 = No, 1 = Yes |

**Innovation Index:** INN_INDEX = (innov_iot + innov_automation + innov_quality_mgmt + innov_other) / 4

---

### SECTION E: FIRM RESOURCES

#### Human Resources

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `skilled_workforce_pct` | Float | Percentage of skilled workforce | 20-90% |
| `training_hours_per_emp` | Integer | Annual training hours per employee | 10-200 hours |

#### Technological Resources

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `modern_equipment` | Binary | Use of modern equipment | Yes, No |
| `machinery_age_years` | Integer | Age of primary machinery | 1-25 years |

#### Financial Resources

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `access_credit` | Categorical | Ease of accessing credit | Easy, Moderate, Difficult |
| `reinvestment_rate_pct` | Float | Percentage of profits reinvested | 5-40% |

---

### SECTION F: GOVERNMENT POLICY PERCEPTION

**Scale:** 1 = Very Ineffective, 2 = Ineffective, 3 = Somewhat Ineffective, 4 = Neutral, 5 = Somewhat Effective, 6 = Effective, 7 = Very Effective

| Variable | Description |
|----------|-------------|
| `gp1_tax_incentives` | Effectiveness of tax incentives |
| `gp2_regulatory_stability` | Regulatory stability perception |
| `gp3_infrastructure` | Infrastructure support quality |
| `gp4_permits_ease` | Ease of obtaining permits |

**Government Policy Index:** GP_INDEX = (gp1 + gp2 + gp3 + gp4) / 4

---

### SECTION G: PERFORMANCE METRICS

#### Financial Performance

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `avg_roi_pct` | Float | Average Return on Investment (past 3 years) | -5% to 45% |
| `avg_roa_pct` | Float | Average Return on Assets (past 3 years) | -3% to 35% |
| `export_intensity_pct` | Float | Exports as % of total sales | 0-60% |

#### Operational Performance

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `capacity_utilization_pct` | Float | Production capacity utilization | 40-95% |
| `market_share_pct` | Float | Market share in Lagos food processing sector | 0.5-25% |

---

## COMPOSITE SCORES FOR SEM ANALYSIS

### Latent Variables (Reflective Constructs)

1. **KNOWLEDGE_ABSORPTION (KA)**
   - Formula: `(ka1 + ka2 + ka3 + ka4) / 4`
   - Range: 1-5
   - Reliability: Cronbach's α expected > 0.70

2. **TASK_PERFORMANCE (TP)**
   - Formula: `(tp1 + tp2 + tp3 + tp4) / 4`
   - Range: 1-5
   - Reliability: Cronbach's α expected > 0.70

3. **INNOVATION (INN)**
   - Indicators: `rd_spending_pct`, `new_products_3yrs`, innovation index
   - Multi-indicator construct

4. **GOVERNMENT_POLICY (GP)** - Formative Construct
   - Indicators: `gp1`, `gp2`, `gp3`, `gp4`
   - Range: 1-7

5. **FIRM_PERFORMANCE (PERFORM)**
   - Indicators: `avg_roi_pct`, `avg_roa_pct`, `capacity_utilization_pct`, `export_intensity_pct`, `market_share_pct`
   - Multi-dimensional construct

---

## DATA CHARACTERISTICS

### Missing Data
- **Design:** No missing data in synthetic dataset (100% completion rate)
- **Real-world expectation:** 5-10% missing data typical

### Distribution Characteristics
- **Likert scales:** Approximately normal distribution with slight positive skew for FDI firms
- **Performance metrics:** Right-skewed distributions (typical for financial data)
- **Innovation metrics:** Poisson distribution for count variables

### Key Correlations (Expected)
- FDI presence ↔ Knowledge absorption: r ≈ 0.45-0.65
- Knowledge absorption ↔ Performance: r ≈ 0.40-0.60
- Innovation ↔ Performance: r ≈ 0.35-0.55
- Government policy ↔ FDI effectiveness: Moderating effect

---

## SEM MODEL VARIABLES

### Exogenous Variables
- `has_fdi` (FDI presence, coded: 0 = No, 1 = Yes)
- `firm_size` (coded: 0 = SME, 1 = Large)
- Firm resources: `skilled_workforce_pct`, `modern_equipment`, `access_credit`

### Mediating Variables
- Knowledge Absorption (KA)
- Task Performance (TP)
- Innovation (INN)

### Moderating Variable
- Government Policy (GP)

### Dependent Variable
- Firm Performance (PERFORM)

---

## USAGE NOTES

### For Descriptive Analysis
```r
# Load data
data <- read.csv("fdi_lagos_survey_data.csv")

# Create composite scores
data$KA_score <- (data$ka1_technical_manuals + data$ka2_staff_training + 
                  data$ka3_adapt_technology + data$ka4_commercialize_knowledge) / 4

data$TP_score <- (data$tp1_production_efficiency + data$tp2_quality_control + 
                  data$tp3_order_fulfillment + data$tp4_employee_productivity) / 4
```

### For SEM Analysis (lavaan)
```r
# Convert categorical to dummy
data$fdi_dummy <- ifelse(data$has_fdi == "Yes", 1, 0)
data$size_dummy <- ifelse(data$firm_size == "Large", 1, 0)

# Define measurement model
measurement_model <- '
  KA =~ ka1_technical_manuals + ka2_staff_training + 
        ka3_adapt_technology + ka4_commercialize_knowledge
  TP =~ tp1_production_efficiency + tp2_quality_control + 
        tp3_order_fulfillment + tp4_employee_productivity
'
```

---

## DATA QUALITY CHECKS

### Recommended Validation Steps
1. **Normality tests:** Shapiro-Wilk for continuous variables
2. **Outlier detection:** Boxplots for performance metrics
3. **Reliability analysis:** Cronbach's alpha for multi-item scales
4. **Correlation matrix:** Check for multicollinearity (VIF < 5)
5. **Common method bias:** Harman's single factor test

### Expected Reliability Coefficients
- Knowledge Absorption: α > 0.80
- Task Performance: α > 0.75
- Government Policy: α > 0.70

---

## CITATION

If using this dataset, cite as:

> Synthetic Survey Data: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria. Jiangsu University PhD Research, 2024. N=210 firms.

---

**Last Updated:** October 13, 2025  
**Version:** 1.0  
**Contact:** PhD Candidate, Jiangsu University
