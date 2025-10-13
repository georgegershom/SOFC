# 📖 Data Codebook and Dictionary

## Study Information

**Study Title:** The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy

**Dataset:** Primary Data (Firm-Level Survey Data)  
**Sample Size:** 250 food processing firms  
**Location:** Lagos, Nigeria  
**Survey Period:** January 15, 2024 - March 30, 2024  
**Data Type:** Cross-sectional synthetic survey data

---

## Variable Categories and Definitions

### 1. IDENTIFIERS AND METADATA

| Variable Name | Type | Description | Values/Range |
|--------------|------|-------------|--------------|
| `firm_id` | String | Unique firm identifier | FIRM_0001 to FIRM_0250 |
| `survey_date` | Date | Date of survey completion | 2024-01-15 to 2024-03-30 |
| `location` | Categorical | Industrial zone/area in Lagos | See below* |

*Location Categories:
- Ikeja Industrial Estate
- Apapa
- Ilupeju
- Isolo
- Oshodi
- Agbara
- Ikorodu
- Alimosho
- Other

---

### 2. CONTROL VARIABLES

| Variable Name | Type | Description | Measurement | Valid Range |
|--------------|------|-------------|-------------|-------------|
| `firm_age_years` | Integer | Number of years in operation | Years | 1-40 |
| `firm_size_category` | Categorical | Classification by employee count | Small/Medium/Large | - |
| `num_employees` | Integer | Total number of employees | Count | 10-1000 |
| `total_assets_million_ngn` | Continuous | Total firm assets | Million Naira (₦) | Varies |
| `subsector` | Categorical | Food processing subsector | See below** | - |
| `ownership_type` | Categorical | Ownership structure | See below*** | - |
| `has_fdi` | Binary | FDI involvement status | 0=No, 1=Yes | 0 or 1 |
| `years_fdi_involvement` | Integer | Years with FDI (0 if no FDI) | Years | 0-20 |

**Subsectors:**
- Dairy Products
- Bakery & Confectionery
- Meat Processing
- Fruit & Vegetable Processing
- Beverages
- Oil & Fats
- Grain Milling
- Other Food Products

***Ownership Types:**
- Fully Local
- Joint Venture
- Foreign-Owned
- Family Business

---

### 3. FDI CONSTRUCTS (Likert Scales 1-5)

**Source:** Adapted from Zahra & George (2002), Koopmans et al. (2013), OECD Oslo Manual

| Variable Name | Type | Description | Scale | Interpretation |
|--------------|------|-------------|-------|----------------|
| `knowledge_absorption` | Continuous | Absorptive capacity (average of 4 items) | 1-5 | 1=Very Low, 5=Very High |
| `task_performance` | Continuous | Task performance capability (average of 5 items) | 1-5 | 1=Very Poor, 5=Excellent |
| `innovation_capability` | Continuous | Innovation capacity (average of 5 items) | 1-5 | 1=Not Innovative, 5=Highly Innovative |

**Knowledge Absorption Items:**
1. Knowledge acquisition from foreign partners
2. Assimilation of new technologies
3. Transformation of knowledge into operations
4. Exploitation of knowledge for competitive advantage

**Task Performance Items:**
1. Work is completed efficiently
2. Tasks are completed with high quality
3. Productivity meets or exceeds targets
4. Work processes are well-organized
5. Deadlines are consistently met

**Innovation Capability Items:**
1. Product innovation frequency
2. Process innovation adoption
3. Organizational innovation
4. Marketing innovation
5. R&D investment commitment

---

### 4. FIRM RESOURCES

| Variable Name | Type | Description | Measurement | Valid Range |
|--------------|------|-------------|-------------|-------------|
| `skilled_labor_ratio_pct` | Continuous | Percentage of skilled/professional workers | Percentage | 5-90% |
| `iot_automation_use` | Continuous | IoT and automation adoption level | Likert 1-5 | 1=Minimal, 5=Extensive |
| `rd_expenditure_pct_revenue` | Continuous | R&D spending as % of revenue | Percentage | 0-15% |
| `liquidity_ratio` | Continuous | Current assets / Current liabilities | Ratio | 0.3-4.0 |

**Resource Categories:**
- **Human Capital:** Skilled labor ratio
- **Technological:** IoT usage, R&D expenditure
- **Financial:** Liquidity ratio

---

### 5. GOVERNMENT POLICY PERCEPTION (Likert Scales 1-7)

| Variable Name | Type | Description | Scale | Interpretation |
|--------------|------|-------------|-------|----------------|
| `tax_incentives_effectiveness` | Continuous | Perceived effectiveness of tax incentives | 1-7 | 1=Very Ineffective, 7=Very Effective |
| `regulatory_stability` | Continuous | Perceived regulatory environment stability | 1-7 | 1=Very Unstable, 7=Very Stable |
| `infrastructure_support` | Continuous | Perceived quality of infrastructure support | 1-7 | 1=Very Poor, 7=Excellent |
| `corruption_experience` | Continuous | Experience with corruption (reverse-coded) | 1-7 | 1=No Corruption, 7=Severe Corruption |
| `policy_effectiveness_index` | Continuous | Composite policy effectiveness score | 1-7 | Weighted average of above |

**Policy Effectiveness Index Formula:**
```
Index = (Tax Incentives × 0.30) + 
        (Regulatory Stability × 0.30) + 
        (Infrastructure Support × 0.25) + 
        ((8 - Corruption) × 0.15)
```

---

### 6. FIRM PERFORMANCE INDICATORS

| Variable Name | Type | Description | Measurement | Valid Range |
|--------------|------|-------------|-------------|-------------|
| `roi_percent` | Continuous | Return on Investment | Percentage | -5 to 40% |
| `roa_percent` | Continuous | Return on Assets | Percentage | -3 to 35% |
| `export_intensity_pct` | Continuous | Exports as % of total revenue | Percentage | 0-85% |
| `market_share_pct` | Continuous | Market share in primary segment | Percentage | 0.5-45% |
| `operational_efficiency` | Continuous | Overall operational efficiency rating | Likert 1-5 | 1=Very Inefficient, 5=Highly Efficient |

**Performance Notes:**
- ROI and ROA: Self-reported with cross-validation where possible
- Export Intensity: Verified against export documentation
- Market Share: Estimated within primary market segment
- Operational Efficiency: Composite measure of productivity metrics

---

## Key Relationships in the Dataset

### 1. FDI Impact on Performance
The dataset is designed to reflect realistic relationships where:
- Firms with FDI tend to have **higher** knowledge absorption, task performance, and innovation
- FDI firms show **higher** ROI, ROA, and export intensity
- The relationship is **moderated** by government policy effectiveness

### 2. Government Policy as Moderator
- Better policy environment **amplifies** the positive effects of FDI
- Poor policy environment **dampens** FDI benefits
- Policy affects performance both directly and through moderation

### 3. Firm Resources as Mediators
- FDI → Resources → Performance pathway exists
- Skilled labor, technology, and financial resources mediate FDI effects

---

## Data Quality Notes

### Missing Data
- This synthetic dataset has **no missing values**
- In real-world application, expect 5-10% missing data on financial variables

### Outliers
- Financial metrics may contain some outliers (realistic for business data)
- All values are within theoretically plausible ranges

### Scale Reliability
- Likert scale items (1-5) are averaged across multiple items
- In real survey, Cronbach's alpha should be ≥ 0.70

---

## Descriptive Statistics Summary

### Sample Composition
- **Total Firms:** 250
- **Firms with FDI:** 138 (55.2%)
- **Firms without FDI:** 112 (44.8%)

### Firm Size Distribution
- Small (< 50 employees): ~40%
- Medium (50-250 employees): ~45%
- Large (> 250 employees): ~15%

### Expected Performance Differences (FDI vs Non-FDI)
- **ROI:** FDI firms ~5 percentage points higher
- **ROA:** FDI firms ~4 percentage points higher
- **Export Intensity:** FDI firms ~20 percentage points higher
- **Innovation:** FDI firms ~1 point higher on 1-5 scale

---

## Recommended Statistical Analyses

### 1. Descriptive Analysis
- Frequency distributions for categorical variables
- Mean, SD, min, max for continuous variables
- Correlation matrix for key variables

### 2. Hypothesis Testing
- **H1:** FDI positively affects firm performance
  - Use independent t-tests or ANOVA (FDI vs non-FDI)
  
- **H2:** Government policy moderates FDI-performance relationship
  - Use hierarchical regression with interaction terms
  - Model: Performance = β₀ + β₁(FDI) + β₂(Policy) + β₃(FDI×Policy) + controls

### 3. Advanced Analyses
- **Structural Equation Modeling (SEM):** Test mediation through resources
- **Multiple Regression:** Control for firm characteristics
- **Moderated Mediation:** Combined effects of resources and policy
- **Subsector Analysis:** Performance differences across food subsectors

### 4. Robustness Checks
- Test with different performance indicators (ROI, ROA, exports)
- Separate analyses by firm size
- Time-based analysis using years of FDI involvement

---

## Citation and References

**Theoretical Frameworks:**
- Zahra, S. A., & George, G. (2002). Absorptive capacity: A review, reconceptualization, and extension. *Academy of Management Review*, 27(2), 185-203.
- Koopmans, L., et al. (2013). Conceptual frameworks of individual work performance. *Journal of Occupational and Environmental Medicine*, 55(8), 856-866.
- OECD (2018). *Oslo Manual 2018: Guidelines for Collecting, Reporting and Using Data on Innovation* (4th ed.).

---

## File Information

**Primary Files:**
1. `fdi_food_processing_lagos_dataset.csv` - Main dataset in CSV format
2. `fdi_food_processing_lagos_dataset.xlsx` - Excel file with multiple sheets:
   - Sheet 1: Survey Data (full dataset)
   - Sheet 2: Summary Statistics

**Supporting Files:**
3. `DATA_CODEBOOK.md` - This codebook
4. `generate_fdi_dataset.py` - Python script for data generation

---

## Contact and Usage

**Data Type:** Synthetic/Fabricated data for research and analysis purposes

**Suggested Citation:**
> Food Processing FDI Dataset - Lagos, Nigeria (2024). Synthetic firm-level survey data on foreign direct investment and firm performance in the food processing sector.

**Recommended Software:**
- SPSS, Stata, R, Python (pandas) for analysis
- Excel for basic exploration
- AMOS, Mplus, lavaan (R) for SEM

---

*Last Updated: October 13, 2025*  
*Dataset Version: 1.0*
