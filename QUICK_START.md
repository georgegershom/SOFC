# Quick Start Guide

## 📦 Download the Dataset

**Main ZIP file:** `SOC_Epistemic_Uncertainty_Dataset.zip` (2.9 MB)

Contains:
- 7 CSV files (30,000+ data points)
- Full documentation
- Dataset summary

---

## 🚀 Quick Analysis (60 seconds)

```bash
# 1. Extract the ZIP (if needed)
unzip SOC_Epistemic_Uncertainty_Dataset.zip

# 2. Run the analysis
cd scripts/
python3 quantify_epistemic_gap.py

# 3. View the figures
cd ../figures/
# Open PNG files in any image viewer
```

---

## 📊 Key Files to Check First

### For Dataset Overview:
1. `README.md` - Complete documentation
2. `data/DATASET_SUMMARY.txt` - Quick statistics

### For Understanding Missing Data:
1. `data/21_missing_data_assumption_flags.csv` - **START HERE**
   - Explains what's missing and why it matters

### For Visualization:
1. `figures/01_cone_of_ignorance_fragility_curves.png` - **Main result**
   - Shows uncertainty bounds due to missing correlation data

### For Monte Carlo Inputs:
1. `data/stochastic_inputs_rho0.00.csv` - Independent scenario
2. `data/stochastic_inputs_rho0.50.csv` - Correlated scenario  
3. `data/stochastic_inputs_HT_uncorrelated.csv` - High-temp scenario

---

## 🎯 Three Use Cases

### Use Case 1: "I need the datasets for my simulation"
```python
import pandas as pd

# Load scenario
df = pd.read_csv('data/stochastic_inputs_rho0.50.csv')

# Extract properties for sample i
sample_id = 100
Gc_anode = df.loc[sample_id, 'Gc_YSZ_GDC']
Gc_cathode = df.loc[sample_id, 'Gc_GDC_LSCF']
```

### Use Case 2: "I want to see the uncertainty impact"
```bash
cd scripts/
python3 quantify_epistemic_gap.py

# Look for section:
# "EPISTEMIC UNCERTAINTY QUANTIFICATION REPORT"
```

### Use Case 3: "I need publication-quality figures"
All figures are 300 DPI PNG in `figures/` directory:
- Figure 1: Fragility curves (the "cone")
- Figure 2: CDF comparison
- Figure 3: Correlation scatter plots
- Figure 4: PDF histograms
- Figure 5: Risk heatmaps
- Figure 6: Box plot sensitivity

---

## 📈 Understanding the Results

### What is "Epistemic Uncertainty"?
Uncertainty due to **lack of knowledge** (not randomness).

**In this study:**
- We don't know the correlation coefficient ρ between interfaces
- We assume it's between 0.0 and 0.5
- This creates the "Cone of Ignorance" (shaded region in Figure 1)

### What is the Main Finding?
At room temperature:
- System resistance: 6.05 J/m²
- Variance increases by **22%** if interfaces are correlated (ρ=0.5)

At high temperature (800°C):
- System resistance drops to 4.54 J/m² (25% weaker)
- Failure probability dramatically increases

---

## ⚠️ Important Warnings

1. **This is SYNTHETIC data** - Created for uncertainty analysis
2. **Do NOT use for actual design** without experimental validation
3. **Two critical missing datasets:**
   - Real correlation coefficient (currently guessed)
   - High-temperature variance (currently assumed constant)

---

## 🔬 Recommended Next Steps

### For Experimentalists:
1. Perform 5-10 co-sintered stack tests to measure real ρ
2. Conduct in-situ 800°C micro-cantilever tests
3. Measure high-temperature variance (not just mean)

### For Modelers:
1. Use all three scenarios to bound predictions
2. Report results as **intervals** not single values:
   ```
   P_fail ∈ [P_indep, P_corr]  (epistemic bounds)
   ```
3. Propagate through FEA/Abaqus models

### For Decision Makers:
1. Review Figure 1 ("Cone of Ignorance")
2. Check `data/21_missing_data_assumption_flags.csv`
3. Prioritize MISSING_DATASET_02 (high-T variance)

---

## 💬 Common Questions

**Q: Which scenario should I use?**  
A: Run all three and report the range. Use ρ=0.5 for conservative design.

**Q: Are the 10,000 samples independent?**  
A: Yes, each row is an independent Monte Carlo realization.

**Q: Can I change the material properties?**  
A: Yes, edit `scripts/generate_stochastic_inputs.py` and re-run.

**Q: What's the random seed?**  
A: Seed = 42 (for reproducibility). Change it for different realizations.

**Q: Why is the high-T failure probability so high?**  
A: Because Gc drops by 25% while the applied load (J=2.5 J/m²) stays constant.

---

## 📚 Full Documentation

See `README.md` for:
- Complete mathematical formulation
- Detailed column descriptions
- Citation information
- Contact details

---

## 🐛 Troubleshooting

**Error: "No module named 'pandas'"**  
Solution:
```bash
pip install pandas scipy matplotlib numpy
```

**Figures look blurry**  
Solution: All figures are 300 DPI. View at 100% zoom or import into LaTeX.

**Need different correlation values?**  
Solution: Edit line 138 in `generate_stochastic_inputs.py`:
```python
rho = 0.7  # Change from 0.5 to any value in [-1, 1]
```

---

## ⏱️ Expected Runtime

- `generate_stochastic_inputs.py`: ~2 seconds
- `quantify_epistemic_gap.py`: <1 second
- `generate_figures.py`: ~8 seconds

**Total:** Less than 15 seconds for complete regeneration.

---

## ✅ Verification Checklist

After extraction, you should have:
- [ ] 7 CSV files in `data/`
- [ ] 6 PNG figures in `figures/`
- [ ] 3 Python scripts in `scripts/`
- [ ] 1 ZIP file (2.9 MB)
- [ ] 3 markdown files (README, QUICK_START, DATASET_SUMMARY)

---

**Happy analyzing! 🚀**

For issues or questions, review the detailed README.md first.
