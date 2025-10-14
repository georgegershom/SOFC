# Quick Start Guide - Welding Inverse Design Dataset

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Explore the Data
```python
import pandas as pd

# Load the master dataset
df = pd.read_csv('master_dataset.csv')

print(f"Total samples: {len(df)}")
print(f"Features: {len(df.columns)}")
print(f"\nFirst few samples:")
print(df.head())
```

### 3. Run Analysis
```bash
python3 data_analysis.py
```

This will generate:
- Comprehensive statistical analysis
- Correlation analysis
- Material performance comparisons
- Defect impact analysis
- Optimal parameter identification

### 4. Train ML Models
```bash
python3 inverse_design_example.py
```

This demonstrates:
- Forward design models (parameters → performance)
- Multi-fidelity learning
- Model evaluation metrics

## 📊 Quick Data Overview

### Files
- `master_dataset.csv` - Full dataset (10,500 samples)
- `tier1_experimental_data.csv` - Experimental only (500 samples)
- `tier2_computational_data.csv` - Simulation only (10,000 samples)
- `dataset_metadata.json` - Complete metadata

### Key Columns

**Inputs (X):**
- Laser_Power_W
- Welding_Speed_mm_s
- Material_Combination
- ... (13 total)

**Outputs (Y):**
- Tensile_Shear_Strength_N
- Contact_Resistance_uOhm
- Cycles_to_Failure
- Overall_Quality_Score
- ... (19 total)

## 🎯 Common Tasks

### Find Best Performing Welds
```python
import pandas as pd

df = pd.read_csv('master_dataset.csv')

# Top 10 by quality score
top_10 = df.nlargest(10, 'Overall_Quality_Score')
print(top_10[['Weld_ID', 'Material_Combination', 'Laser_Power_W', 
              'Welding_Speed_mm_s', 'Cycles_to_Failure', 
              'Overall_Quality_Score']])
```

### Filter by Material
```python
# Get all Cu-Al welds
cu_al = df[df['Material_Combination'] == 'Cu-Al']
print(f"Average quality: {cu_al['Overall_Quality_Score'].mean():.2f}")
print(f"Average cycles: {cu_al['Cycles_to_Failure'].mean():.0f}")
```

### Find Optimal Parameters
```python
# Get top 10% performers
threshold = df['Overall_Quality_Score'].quantile(0.9)
top_performers = df[df['Overall_Quality_Score'] >= threshold]

# Average parameters
print("Optimal parameter ranges:")
print(f"Laser Power: {top_performers['Laser_Power_W'].mean():.0f} W")
print(f"Welding Speed: {top_performers['Welding_Speed_mm_s'].mean():.1f} mm/s")
print(f"Heat Input: {top_performers['Heat_Input_J_mm'].mean():.2f} J/mm")
```

### Train a Simple Model
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Prepare data
input_cols = ['Laser_Power_W', 'Welding_Speed_mm_s', 'Heat_Input_J_mm']
output_col = 'Overall_Quality_Score'

# Encode categorical if needed
df_encoded = df.copy()
le = LabelEncoder()
df_encoded['Material_encoded'] = le.fit_transform(df['Material_Combination'])
input_cols.append('Material_encoded')

X = df_encoded[input_cols]
y = df_encoded[output_col]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.3f}")
```

## 📈 Key Statistics

- **Total Samples:** 10,500
- **Materials:** Cu-Al, Al-Al, Al-Steel, Cu-Cu
- **Average Quality Score:** 86.9/100
- **Average Cycles to Failure:** 1,242
- **Average Tensile Strength:** 3,858 N
- **Defect Rate:** 0.21%

## 🎓 Next Steps

1. **Read the README.md** for detailed documentation
2. **Check DATASET_SUMMARY.md** for comprehensive overview
3. **Explore data_analysis.py** for analysis tools
4. **Study inverse_design_example.py** for ML examples

## 🔗 Recommended Workflow

### For Inverse Design Research:

1. **Exploratory Analysis** → Run `data_analysis.py`
2. **Forward Model** → Train on full dataset
3. **Validation** → Test on experimental data only
4. **Optimization** → Use with Bayesian optimization
5. **Verification** → Compare predictions with held-out experimental data

### For Multi-Fidelity Learning:

1. **Pre-train** → Use computational data (10,000 samples)
2. **Fine-tune** → Use experimental data (500 samples)
3. **Validate** → Cross-validation on experimental set
4. **Deploy** → Use for inverse design predictions

## 🆘 Troubleshooting

### Import Errors
```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

### File Not Found
Make sure you're in the `welding_datasets` directory:
```bash
cd welding_datasets
```

### Memory Issues
If loading all 10,500 samples is too large:
```python
# Load in chunks
df = pd.read_csv('master_dataset.csv', nrows=1000)  # First 1000 rows
```

Or use just experimental data:
```python
df = pd.read_csv('tier1_experimental_data.csv')  # Only 500 rows
```

## 📧 Support

For detailed information, see:
- `README.md` - Full documentation
- `DATASET_SUMMARY.md` - Project overview
- `dataset_metadata.json` - Technical specifications

---

**Happy Researching!** 🚀
