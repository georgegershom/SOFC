# 📊 FDI Food Processing Dataset - Lagos, Nigeria

## 🎯 Overview

This is a comprehensive **synthetic dataset** of 250 food processing firms in Lagos, Nigeria, designed for research on **Foreign Direct Investment (FDI)** and its impact on firm performance, moderated by government policy.

### Research Topic
**"The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy"**

---

## 📁 Dataset Files

| File | Format | Description | Size |
|------|--------|-------------|------|
| **fdi_food_processing_lagos_dataset.csv** | CSV | Main dataset (250 firms × 28 variables) | ~50 KB |
| **fdi_food_processing_lagos_dataset.xlsx** | Excel | Multi-sheet workbook with data + summary stats | ~60 KB |
| **DATA_CODEBOOK.md** | Markdown | Complete variable descriptions and methodology | Documentation |
| **generate_fdi_dataset.py** | Python | Script to regenerate the dataset | Code |
| **analysis_example.py** | Python | Example statistical analyses | Code |
| **README_DATASET.md** | Markdown | This file | Documentation |

---

## 📋 Quick Start

### Option 1: Use the Pre-Generated Data
```bash
# Open in Excel
open fdi_food_processing_lagos_dataset.xlsx

# Or load in Python
import pandas as pd
df = pd.read_csv('fdi_food_processing_lagos_dataset.csv')
print(df.head())
```

### Option 2: Regenerate the Dataset
```bash
# Install dependencies
pip install numpy pandas openpyxl scipy

# Generate fresh dataset
python3 generate_fdi_dataset.py

# Run example analyses
python3 analysis_example.py
```

### Option 3: Import to SPSS/Stata
```bash
# For SPSS: Import CSV file directly
# For Stata: 
import delimited fdi_food_processing_lagos_dataset.csv, clear
```

---

## 📊 Dataset Characteristics

### Sample Composition
- **Total Firms:** 250
- **Firms with FDI:** 138 (55.2%)
- **Survey Period:** Jan-Mar 2024
- **Location:** Lagos, Nigeria
- **Sectors:** 8 food processing subsectors

### Key Features
✅ **28 variables** covering all research constructs  
✅ **Realistic correlations** between FDI, resources, policy, and performance  
✅ **No missing data** (synthetic dataset)  
✅ **Multiple performance indicators** (ROI, ROA, exports, market share)  
✅ **Policy moderation effects** built into the data  
✅ **Control variables** (firm size, age, subsector, ownership)  

---

## 🔬 Variable Categories

### 1️⃣ FDI Constructs (Likert 1-5)
- Knowledge Absorption
- Task Performance  
- Innovation Capability

### 2️⃣ Firm Performance Indicators
- ROI (Return on Investment %)
- ROA (Return on Assets %)
- Export Intensity (%)
- Market Share (%)
- Operational Efficiency (1-5 scale)

### 3️⃣ Government Policy Perception (Likert 1-7)
- Tax Incentives Effectiveness
- Regulatory Stability
- Infrastructure Support
- Corruption Experience
- Policy Effectiveness Index (composite)

### 4️⃣ Firm Resources
- Skilled Labor Ratio (%)
- IoT/Automation Use (1-5 scale)
- R&D Expenditure (% of revenue)
- Liquidity Ratio

### 5️⃣ Control Variables
- Firm Size (employees, assets)
- Firm Age
- Subsector
- Ownership Type
- Location in Lagos

📖 **Full details in:** `DATA_CODEBOOK.md`

---

## 📈 Expected Research Findings

Based on the dataset design, you should find:

### ✅ Hypothesis 1: FDI → Performance
- **FDI firms outperform non-FDI firms** on all metrics
- Expected differences:
  - ROI: +5 percentage points
  - Export Intensity: +20 percentage points
  - Innovation: +1 point (on 1-5 scale)

### ✅ Hypothesis 2: Policy Moderation
- **Government policy moderates FDI effects**
- Better policy environment → stronger FDI benefits
- Interaction term (FDI × Policy) should be significant

### ✅ Hypothesis 3: Resource Mediation
- **Firm resources mediate FDI-performance link**
- Path: FDI → Resources → Performance

---

## 🎓 Recommended Analyses

### Basic Analyses
```python
# 1. Descriptive Statistics
df.describe()
df.groupby('has_fdi').mean()

# 2. T-tests (FDI vs Non-FDI)
from scipy.stats import ttest_ind
ttest_ind(df[df.has_fdi==1].roi_percent, 
          df[df.has_fdi==0].roi_percent)

# 3. Correlation Matrix
df[['knowledge_absorption', 'innovation_capability', 
    'policy_effectiveness_index', 'roi_percent']].corr()
```

### Advanced Analyses
1. **Hierarchical Regression** - Test moderation
   ```
   Model 1: Performance = β₀ + β₁(FDI) + β₂(Controls)
   Model 2: Performance = β₀ + β₁(FDI) + β₂(Policy) + β₃(FDI×Policy) + β₄(Controls)
   ```

2. **Structural Equation Modeling (SEM)** - Test mediation
   - Use AMOS, Mplus, or lavaan (R)
   - Test: FDI → Resources → Performance

3. **Robustness Checks**
   - Different performance indicators
   - Subsample analyses (by size, subsector)
   - Sensitivity to outliers

---

## 📊 Sample Statistics

### Performance Comparison: FDI vs Non-FDI

| Metric | FDI Firms | Non-FDI Firms | Difference |
|--------|-----------|---------------|------------|
| ROI (%) | 21.42 | 16.18 | +5.24 ✅ |
| ROA (%) | 15.82 | 11.33 | +4.49 ✅ |
| Export (%) | 37.59 | 18.19 | +19.40 ✅ |
| Innovation (1-5) | 3.48 | 2.41 | +1.07 ✅ |

### Policy Perception (Mean ± SD)
- Tax Incentives: 4.04 ± 1.49
- Regulatory Stability: 3.73 ± 1.46  
- Infrastructure Support: 3.55 ± 1.52
- Corruption: 3.96 ± 1.63 (reverse coded)

---

## 🛠️ Software Compatibility

| Software | Status | Notes |
|----------|--------|-------|
| **Excel** | ✅ Full | Open .xlsx directly |
| **SPSS** | ✅ Full | Import CSV with variable labels |
| **Stata** | ✅ Full | Use `import delimited` |
| **R** | ✅ Full | `read.csv()` or `read_excel()` |
| **Python** | ✅ Full | `pandas.read_csv()` |
| **SAS** | ✅ Full | PROC IMPORT |
| **AMOS/Mplus** | ✅ Full | For SEM analyses |

---

## 📚 Theoretical Frameworks Used

### FDI Constructs
- **Zahra & George (2002)** - Absorptive Capacity Model
- **Koopmans et al. (2013)** - Task Performance Framework
- **OECD Oslo Manual (2018)** - Innovation Measurement

### Performance Metrics
- Financial: ROI, ROA (accounting standards)
- Market: Export intensity, market share
- Operational: Efficiency ratings

### Policy Framework
- Tax incentives
- Regulatory environment
- Infrastructure quality
- Corruption perception

---

## ⚠️ Important Notes

### This is Synthetic Data
- **Purpose:** Research, teaching, methodology testing
- **Not real firm data** - fabricated for study purposes
- **Realistic patterns** built in based on literature
- **No confidentiality issues** - fully synthetic

### Quality Assurance
✅ No missing values  
✅ All values within valid ranges  
✅ Realistic correlations  
✅ Appropriate distributions  
✅ Consistent relationships  

### Limitations
- Cross-sectional (not longitudinal)
- Self-reported measures (like real surveys)
- Simplified relationships (for demonstration)
- No extreme outliers (cleaned data)

---

## 🎯 Use Cases

### 1. Academic Research
- Thesis/dissertation work
- Research methodology practice
- Testing statistical techniques

### 2. Teaching
- Research methods courses
- Statistics classes
- International business modules

### 3. Practice
- SEM/regression practice
- SPSS/Stata/R tutorials
- Data visualization exercises

---

## 📞 Dataset Information

**Topic:** FDI Impact on Food Processing Firms in Lagos, Nigeria  
**Data Type:** Primary survey data (synthetic)  
**Sample:** 250 firms  
**Variables:** 28  
**Time Period:** Cross-sectional (2024)  
**Status:** Complete, ready for analysis

---

## 🚀 Next Steps

1. **📖 Read the Codebook** (`DATA_CODEBOOK.md`)
2. **📊 Explore the Data** (Excel or statistical software)
3. **🔬 Run Example Analysis** (`python3 analysis_example.py`)
4. **📈 Conduct Your Analysis** (regression, SEM, etc.)
5. **✍️ Write Your Results** (thesis, paper, report)

---

## 📖 Citation

If using this dataset for academic purposes, suggested citation:

> **Food Processing FDI Dataset - Lagos, Nigeria (2024).** Synthetic firm-level survey data on foreign direct investment and firm performance in the food processing sector. Variables: FDI constructs, performance indicators, government policy perception, and firm resources (N=250).

---

## 🤝 Support

For questions about:
- **Variables:** See `DATA_CODEBOOK.md`
- **Methodology:** See `generate_fdi_dataset.py`
- **Analysis:** See `analysis_example.py`

---

## ✨ Features Summary

| Feature | Status |
|---------|--------|
| Complete dataset (250 firms) | ✅ |
| All required variables | ✅ |
| FDI constructs (Likert 1-5) | ✅ |
| Performance indicators | ✅ |
| Policy perception (Likert 1-7) | ✅ |
| Firm resources | ✅ |
| Control variables | ✅ |
| Realistic correlations | ✅ |
| Moderation effects | ✅ |
| Excel + CSV formats | ✅ |
| Complete documentation | ✅ |
| Example analyses | ✅ |
| Ready for statistical analysis | ✅ |

---

**Generated:** October 13, 2025  
**Version:** 1.0  
**Status:** ✅ Complete and ready to use

🎉 **Happy analyzing!**
