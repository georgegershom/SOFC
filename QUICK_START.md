# ⚡ QUICK START - SENCE Framework Figure 9

## 🎯 Get Started in 60 Seconds

### Step 1: Install Dependencies (15 seconds)
```bash
pip install numpy matplotlib pandas scipy seaborn
```

### Step 2: Generate Visualizations (30 seconds)
```bash
python3 sence_radar_visualization.py
```

### Step 3: View Results (15 seconds)
```bash
# Open these files:
outputs/figure9_sence_radar_chart.png    # Main radar chart
outputs/figure9_sence_radar_chart.pdf    # Vector version
outputs/sence_statistical_report.txt     # Statistical analysis
```

---

## 📊 What You'll Get

✅ **13 Output Files** (6.5 MB total)
- Main radar chart (PNG + PDF)
- 3D temporal evolution
- 6 demonstration examples
- Statistical reports
- Data exports (CSV, JSON, LaTeX)

---

## 🔍 Preview

### Main Visualization Components:
1. **8-Axis Radar Chart** - Comparing 3 cities across SENCE domains
2. **Statistical Comparison** - Mean CVI with confidence intervals
3. **Domain Breakdown** - Stacked bar charts by category
4. **Typology Analysis** - Vulnerability signature scatter plot

### Key Findings:
- **Warri**: Highest CVI (0.61) - "Compound Vortex" 
- **Bonny**: Environmental extreme (0.91) - "Environmental Hotspot"
- **Port Harcourt**: Balanced profile (0.52) - "Urban Disparity"

---

## 🎨 Run Demonstrations

```bash
python3 demo_interactive.py
```

**7 Demo Scenarios:**
1. Basic usage
2. Adding custom cities
3. Domain-specific analysis
4. Temporal evolution
5. Correlation analysis
6. Policy targeting
7. Data export

---

## 📚 Learn More

| Level | Document | Time |
|-------|----------|------|
| **Quick** | [README.md](README.md) | 5 min |
| **Detailed** | [USAGE_GUIDE.md](USAGE_GUIDE.md) | 30 min |
| **Complete** | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | 1 hour |

---

## 🔧 Customize

### Add Your Own City
```python
from sence_radar_visualization import SENCERadarChart

radar = SENCERadarChart()
radar.city_data['Your City'] = {
    'values': [0.6, 0.7, 0.65, 0.55, 0.62, 0.68, 0.58, 0.64],  # 8 domains
    'mean_cvi': 0.63,
    'color': '#FF6B6B',
    'linestyle': '-',
    'marker': 'D',
    'alpha': 0.25
}

fig = radar.create_advanced_radar_chart()
fig.savefig('my_radar.png', dpi=300)
```

---

## 🆘 Troubleshooting

**Issue**: `ModuleNotFoundError`
```bash
pip install -r requirements.txt
```

**Issue**: Font warnings
```python
# Ignore or install fonts
import warnings
warnings.filterwarnings('ignore')
```

**Issue**: Python not found
```bash
# Use python3
python3 sence_radar_visualization.py
```

---

## 📧 Support

- **Documentation**: See [INDEX.md](INDEX.md) for complete navigation
- **Issues**: Check [USAGE_GUIDE.md](USAGE_GUIDE.md) → Troubleshooting
- **Contact**: research@sence-framework.org

---

## ✅ Verification

After running, verify:
```bash
ls outputs/  # Should show 13 files
```

Expected output:
```
demo_basic.png
demo_correlation_matrix.png
demo_custom_city.png
demo_environmental_analysis.png
demo_policy_targeting.png
demo_temporal_comparison.png
figure9_sence_3d_temporal.png
figure9_sence_radar_chart.pdf
figure9_sence_radar_chart.png
sence_statistical_report.txt
sence_table.tex
sence_vulnerability_data.csv
sence_vulnerability_data.json
```

---

## 🚀 Next Steps

1. ✅ View `outputs/figure9_sence_radar_chart.png`
2. ✅ Read `outputs/sence_statistical_report.txt`
3. ✅ Explore [USAGE_GUIDE.md](USAGE_GUIDE.md) for customization
4. ✅ Check [INDEX.md](INDEX.md) for complete navigation

---

**Ready to dive deeper?** → [README.md](README.md)

**Version**: 1.0.0 | **License**: MIT | **Status**: ✅ Production Ready
