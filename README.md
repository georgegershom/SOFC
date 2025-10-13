# FDI Performance Study - Synthetic Dataset Package

## 📊 Research Topic
**The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy**

PhD Research - Jiangsu University

---

## 📦 Package Contents

This package contains a complete synthetic dataset and analysis tools for the FDI performance study:

### 1. **Dataset**
- `fdi_lagos_survey_data.csv` - Main dataset with 210 firm responses
  - 140 SMEs (66.7%)
  - 70 Large firms (33.3%)
  - 111 firms with FDI (52.9%)
  - 99 firms without FDI (47.1%)

### 2. **Documentation**
- `data_codebook.md` - Complete variable dictionary and measurement scales
- `descriptive_statistics_report.txt` - Comprehensive statistical summary
- `summary_statistics.csv` - Quick reference statistics
- `README.md` - This file

### 3. **Analysis Scripts**
- `generate_fdi_survey_data.py` - Python script to regenerate the dataset
- `descriptive_statistics_report.py` - Generate descriptive statistics
- `sem_analysis_code.R` - Complete R code for SEM analysis using lavaan

### 4. **Requirements**
- `requirements.txt` - Python package dependencies

---

## 🚀 Quick Start

### Option 1: Use the Pre-generated Dataset

Simply open `fdi_lagos_survey_data.csv` in your preferred analysis software:

**Excel/SPSS:**
```
File → Open → fdi_lagos_survey_data.csv
```

**R:**
```r
data <- read.csv("fdi_lagos_survey_data.csv")
```

**Python:**
```python
import pandas as pd
data = pd.read_csv("fdi_lagos_survey_data.csv")
```

### Option 2: Regenerate the Dataset

If you want to modify the data generation parameters:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the generator
python3 generate_fdi_survey_data.py
```

---

## 📈 Data Structure

### Survey Sections

| Section | Variables | Description |
|---------|-----------|-------------|
| **A: Firm Background** | 8 vars | Years in operation, size, revenue, ownership, FDI status |
| **B: Knowledge Absorption** | 4 vars | Zahra & George Scale (5-point Likert) |
| **C: Task Performance** | 4 vars | Koopmans et al. Scale (5-point Likert) |
| **D: Innovation** | 6 vars | R&D spending, new products, process innovations |
| **E: Firm Resources** | 6 vars | Human, technological, and financial resources |
| **F: Government Policy** | 4 vars | Policy perception (7-point Likert) |
| **G: Performance Metrics** | 5 vars | ROI, ROA, capacity utilization, exports, market share |

**Total Variables:** 42

---

## 🔬 Running SEM Analysis

### Prerequisites

Install R packages:
```r
install.packages(c("lavaan", "semPlot", "psych", "ggplot2", 
                   "corrplot", "semTools"))
```

### Run the Analysis

```r
# Open R or RStudio
source("sem_analysis_code.R")
```

### What the Analysis Does

1. **Descriptive Statistics**
   - Sample characteristics
   - Group comparisons (FDI vs Non-FDI)
   - Distribution analysis

2. **Reliability Testing**
   - Cronbach's alpha for all scales
   - Internal consistency checks

3. **Confirmatory Factor Analysis (CFA)**
   - Validates measurement model
   - Factor loadings and fit indices
   - Generates visualization: `cfa_model.png`

4. **Structural Equation Modeling (SEM)**
   - Tests hypothesized relationships
   - Direct and indirect effects
   - Mediation analysis
   - Generates visualization: `sem_model_main.png`

5. **Moderation Analysis**
   - Tests government policy as moderator
   - Interaction effects

6. **Multi-group Analysis**
   - SMEs vs Large firms comparison
   - Invariance testing

7. **Outputs Generated**
   - `correlation_matrix.png` - Visual correlation matrix
   - `cfa_model.png` - CFA path diagram
   - `sem_model_main.png` - Full SEM model diagram
   - `sem_results_summary.txt` - Complete results report

---

## 📊 Key Statistics

### Sample Characteristics

| Metric | Value |
|--------|-------|
| Total Firms | 210 |
| Response Rate | 70% (of 300 target) |
| Survey Period | July - September 2024 |
| Mean Years in Operation | ~15 years |
| Mean ROI | 12.33% ± 7.57% |
| Mean ROA | 8.48% ± 5.79% |

### Performance Comparison: FDI vs Non-FDI

| Metric | FDI Firms | Non-FDI Firms | Difference |
|--------|-----------|---------------|------------|
| ROI | ~14.5% | ~9.9% | +4.6% |
| Knowledge Absorption | ~3.8/5 | ~2.9/5 | +0.9 |
| R&D Spending | ~1.5% | ~0.5% | +1.0% |
| New Products (3yrs) | ~5.0 | ~2.0 | +3.0 |

---

## 🎯 Hypotheses Tested

### Direct Effects
- **H1:** FDI → Knowledge Absorption (+)
- **H2:** FDI → Task Performance (+)
- **H3:** FDI → Innovation (+)
- **H4:** Knowledge Absorption → Firm Performance (+)
- **H5:** Task Performance → Firm Performance (+)
- **H6:** Innovation → Firm Performance (+)

### Mediation Effects
- **H7:** Knowledge Absorption mediates FDI → Performance
- **H8:** Innovation mediates FDI → Performance

### Moderation Effects
- **H9:** Government Policy moderates FDI → Performance
- **H10:** Government Policy moderates KA → Performance
- **H11:** Government Policy moderates Innovation → Performance

---

## 📐 SEM Model Specification

### Measurement Model (CFA)

```
Knowledge Absorption (KA)
  ├─ ka1: Technical manuals acquisition
  ├─ ka2: Staff training
  ├─ ka3: Technology adaptation
  └─ ka4: Knowledge commercialization

Task Performance (TP)
  ├─ tp1: Production efficiency
  ├─ tp2: Quality control
  ├─ tp3: Order fulfillment
  └─ tp4: Employee productivity

Innovation (INN)
  ├─ rd_spending_pct
  ├─ new_products_3yrs
  └─ innovation_index

Government Policy (GP) - Formative
  ├─ gp1: Tax incentives
  ├─ gp2: Regulatory stability
  ├─ gp3: Infrastructure
  └─ gp4: Permits ease

Firm Performance (PERFORM)
  ├─ avg_roi_pct
  ├─ avg_roa_pct
  ├─ capacity_utilization_pct
  └─ export_intensity_pct
```

### Structural Model

```
FDI → Knowledge Absorption → Performance
FDI → Task Performance → Performance
FDI → Innovation → Performance

Moderator: Government Policy
```

---

## 📋 Variable Naming Convention

| Prefix | Construct | Example |
|--------|-----------|---------|
| `ka` | Knowledge Absorption | `ka1_technical_manuals` |
| `tp` | Task Performance | `tp1_production_efficiency` |
| `innov` | Innovation | `innov_automation` |
| `gp` | Government Policy | `gp1_tax_incentives` |
| `avg` | Performance Metrics | `avg_roi_pct` |

---

## 🔍 Data Quality

### Reliability (Expected Cronbach's α)
- Knowledge Absorption: α > 0.80
- Task Performance: α > 0.75
- Government Policy: α > 0.70

### Model Fit Targets
- CFI: > 0.90
- RMSEA: < 0.08
- SRMR: < 0.06
- χ²/df: < 3.0

### Validity
- Construct validity: Confirmed through CFA
- Discriminant validity: AVE > squared correlations
- Convergent validity: Factor loadings > 0.60

---

## 💡 Usage Examples

### Example 1: Basic Descriptive Analysis (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('fdi_lagos_survey_data.csv')

# Create composite scores
df['KA_score'] = df[['ka1_technical_manuals', 'ka2_staff_training', 
                      'ka3_adapt_technology', 'ka4_commercialize_knowledge']].mean(axis=1)

# Compare FDI vs Non-FDI
fdi_comparison = df.groupby('has_fdi').agg({
    'avg_roi_pct': 'mean',
    'avg_roa_pct': 'mean',
    'KA_score': 'mean',
    'rd_spending_pct': 'mean'
})

print(fdi_comparison)
```

### Example 2: Correlation Analysis (R)

```r
library(corrplot)

data <- read.csv("fdi_lagos_survey_data.csv")

# Select numeric variables
numeric_vars <- c("avg_roi_pct", "avg_roa_pct", "rd_spending_pct", 
                  "capacity_utilization_pct", "new_products_3yrs")

# Correlation matrix
cor_matrix <- cor(data[, numeric_vars], use = "complete.obs")

# Visualize
corrplot(cor_matrix, method = "color", type = "upper", 
         addCoef.col = "black", tl.col = "black")
```

### Example 3: T-tests (R)

```r
# Compare performance between FDI and Non-FDI firms
t.test(avg_roi_pct ~ has_fdi, data = data)
t.test(avg_roa_pct ~ has_fdi, data = data)
t.test(rd_spending_pct ~ has_fdi, data = data)
```

---

## 📚 References

### Measurement Scales Used

1. **Knowledge Absorption:** Zahra, S. A., & George, G. (2002). Absorptive capacity: A review, reconceptualization, and extension. *Academy of Management Review, 27*(2), 185-203.

2. **Task Performance:** Koopmans, L., et al. (2011). Conceptual frameworks of individual work performance. *Journal of Occupational and Environmental Medicine, 53*(8), 856-866.

3. **Innovation:** OECD (2018). *Oslo Manual 2018: Guidelines for Collecting, Reporting and Using Data on Innovation* (4th ed.).

### Analytical Framework

- **SEM Methodology:** Hair, J. F., et al. (2017). *A Primer on Partial Least Squares Structural Equation Modeling (PLS-SEM)* (2nd ed.).
- **Moderation Analysis:** Hayes, A. F. (2017). *Introduction to Mediation, Moderation, and Conditional Process Analysis* (2nd ed.).

---

## ⚠️ Important Notes

### Data Characteristics

1. **Synthetic Data:** This dataset is computer-generated for research purposes. While it simulates realistic patterns based on FDI theory, it should be validated with actual field data.

2. **Realistic Distributions:** Data was generated with:
   - Theory-driven correlations (FDI → Performance)
   - Appropriate statistical distributions
   - Realistic value ranges based on Lagos context

3. **No Missing Data:** The synthetic dataset has 100% completion rate. Real surveys typically have 5-10% missing data.

### Recommended Adjustments for Real Data

When collecting actual survey data:
- Expect 60-75% response rate
- Plan for missing data imputation
- Include data validation questions
- Add reverse-coded items for bias detection
- Pilot test with 20-30 firms first

---

## 🛠️ Troubleshooting

### Python Issues

**Problem:** `ModuleNotFoundError`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem:** Dataset not loading
```python
# Solution: Check file path
import os
print(os.getcwd())  # Verify you're in correct directory
```

### R Issues

**Problem:** Package installation fails
```r
# Solution: Install from CRAN with dependencies
install.packages("lavaan", dependencies = TRUE)
```

**Problem:** SEM model won't converge
```r
# Solution: Check for multicollinearity
library(car)
vif_values <- vif(lm(avg_roi_pct ~ ., data = data))
print(vif_values)  # VIF > 10 indicates multicollinearity
```

---

## 📞 Support

For questions about:
- **Dataset structure:** See `data_codebook.md`
- **Statistical methods:** See `sem_analysis_code.R` comments
- **Variable definitions:** See section headers in CSV file

---

## 📄 License

This synthetic dataset is provided for academic research purposes. Feel free to use, modify, and distribute with proper attribution.

---

## ✅ Checklist for Your Analysis

- [ ] Read `data_codebook.md` to understand all variables
- [ ] Review `descriptive_statistics_report.txt` for sample characteristics
- [ ] Check reliability coefficients (Cronbach's α) in R output
- [ ] Validate measurement model (CFA) before structural model
- [ ] Test for common method bias (Harman's test)
- [ ] Check model fit indices meet thresholds
- [ ] Test hypotheses systematically
- [ ] Run robustness checks (multi-group analysis)
- [ ] Interpret results in context of Nigerian food industry
- [ ] Compare findings with existing FDI literature

---

**Generated:** October 13, 2025  
**Version:** 1.0  
**Research Institution:** Jiangsu University, PhD Program

Good luck with your research! 🎓
