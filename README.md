# Food Processing Firms FDI Dataset - Lagos, Nigeria

## 📊 Dataset Overview

This is a synthetic dataset created for studying **"The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of Nigerian Government Policy"**

### 🎯 Research Focus
- **Primary Relationship**: FDI → Firm Performance
- **Moderator**: Nigerian Government Policy
- **Industry**: Food Processing
- **Location**: Lagos, Nigeria
- **Sample Size**: 500 firms

## 📁 Files Included

| File | Description |
|------|-------------|
| `lagos_food_processing_fdi_dataset.csv` | Main dataset (CSV format) |
| `lagos_food_processing_fdi_dataset.xlsx` | Excel version with multiple sheets |
| `data_dictionary.json` | Complete variable definitions |
| `CODEBOOK.txt` | Detailed codebook and methodology |
| `variable_list.csv` | Quick reference for all variables |
| `generate_fdi_dataset.py` | Dataset generation script |
| `data_dictionary.py` | Documentation generator |
| `analyze_dataset.py` | Analysis and visualization script |

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Generate Dataset
```bash
python generate_fdi_dataset.py
```

### Create Documentation
```bash
python data_dictionary.py
```

### Run Analysis
```bash
python analyze_dataset.py
```

## 📈 Variable Categories

### 1. Control Variables (9 variables)
- Firm identification and demographics
- Firm size, age, ownership structure
- Subsector classification
- Export intensity

### 2. FDI Constructs (15 variables)
- **Knowledge Absorption** (Zahra & George, 2002)
  - Acquisition, Assimilation, Transformation, Exploitation
- **Task Performance** (Koopmans et al., 2013)
  - Quality, Efficiency, Innovation
- **Innovation Capacity** (OECD Oslo Manual)
  - Product, Process, Marketing, Organizational
- **FDI Ownership Percentage**

### 3. Firm Performance (6 variables)
- Financial: ROI, ROA, Revenue Growth
- Operational: Efficiency, Productivity Index
- Market: Market Share

### 4. Firm Resources (10 variables)
- **Human Capital**: Skilled labor, Training, Management quality
- **Technological**: IoT adoption, R&D spending, ERP systems
- **Financial**: Liquidity, Credit access, Investment capacity

### 5. Government Policy (10 variables)
- Tax incentives effectiveness (1-7 scale)
- Regulatory stability
- Infrastructure support
- Investment protection
- Bureaucratic burden (reverse coded)
- Corruption experience (reverse coded)
- Policy effectiveness index (composite)

### 6. Interaction Variables (5 variables)
- FDI × Policy interaction
- Performance composites
- Three-way interactions with firm size

## 📊 Key Features

### Measurement Scales
- **Likert 1-5**: FDI constructs, firm resources
- **Likert 1-7**: Government policy variables
- **Binary (0/1)**: Technology adoption indicators
- **Continuous**: Financial metrics, percentages

### Built-in Relationships
1. **Positive correlations**:
   - FDI exposure → Knowledge absorption → Innovation
   - Policy effectiveness → Firm performance
   - Firm size → Resource availability

2. **Moderation effect**:
   - Government policy moderates FDI-performance relationship
   - Stronger FDI benefits under supportive policy environment

3. **Realistic patterns**:
   - Foreign firms show higher export intensity
   - Larger firms have better credit access
   - Subsector-specific performance variations

## 📈 Sample Statistics

- **Average ROI**: ~15-20%
- **FDI Participation**: ~65% of firms
- **Average Policy Effectiveness**: 4.2/7.0
- **Innovation Score**: 3.1/5.0
- **Export Intensity**: 20-25% average

## 🔍 Analysis Recommendations

1. **Regression Analysis**
   ```
   Performance = β₀ + β₁(FDI) + β₂(Policy) + β₃(FDI×Policy) + Controls
   ```

2. **Required Tests**
   - Multicollinearity (VIF < 10)
   - Heteroscedasticity
   - Normality of residuals

3. **Advanced Methods**
   - Hierarchical regression for moderation
   - Structural Equation Modeling (SEM)
   - Multilevel modeling for subsector effects

## 📚 Theoretical Framework

### Theories Applied
1. **Resource-Based View (RBV)**: Firm resources as competitive advantage
2. **Institutional Theory**: Government policy as institutional factor
3. **Knowledge-Based View**: FDI as knowledge transfer mechanism
4. **Absorptive Capacity**: Ability to acquire and utilize external knowledge

## ⚠️ Data Notes

- **Missing Data**: ~2% MCAR (Missing Completely at Random)
- **Synthetic Data**: Generated for research demonstration
- **Realistic Patterns**: Based on empirical literature
- **Outliers**: Winsorized at 1st and 99th percentiles

## 📝 Citation

When using this dataset, please cite:
```
Synthetic Dataset for FDI Study of Food Processing Firms in Lagos, Nigeria (2024)
Research Topic: The Influence of Foreign Direct Investment on the Performance 
of Food Processing Firms in Lagos, Nigeria: The Moderating Role of Nigerian 
Government Policy
```

## 🤝 Contact & Support

For questions about the dataset structure or variables, refer to:
- `CODEBOOK.txt` for detailed documentation
- `data_dictionary.json` for variable definitions
- `variable_list.csv` for quick reference

## 📊 Quick Start Analysis

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('lagos_food_processing_fdi_dataset.csv')

# View basic info
print(df.info())
print(df.describe())

# Test moderation hypothesis
from scipy import stats

# Calculate correlations
fdi_performance = stats.pearsonr(df['fdi_composite'], df['performance_composite'])
print(f"FDI-Performance Correlation: {fdi_performance[0]:.3f}")

# Split by policy support
high_policy = df[df['policy_effectiveness_index'] >= df['policy_effectiveness_index'].median()]
low_policy = df[df['policy_effectiveness_index'] < df['policy_effectiveness_index'].median()]

# Compare effects
print("High Policy Support:", stats.pearsonr(high_policy['fdi_composite'], 
                                            high_policy['performance_composite'])[0])
print("Low Policy Support:", stats.pearsonr(low_policy['fdi_composite'], 
                                           low_policy['performance_composite'])[0])
```

## 🎯 Research Applications

This dataset is suitable for:
- Testing moderation/mediation hypotheses
- Policy impact assessment
- FDI effectiveness studies
- Firm performance analysis
- Resource-based view testing
- Cross-sectional regression analysis

---

**Generated**: 2024
**Version**: 1.0
**License**: For research and educational purposes