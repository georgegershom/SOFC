# 📑 Project Index: Petroleum Cities Systems Dynamics Model

## 🚀 Quick Navigation

### For First-Time Users
1. **Start Here**: Read `USAGE_GUIDE.md` for quick start instructions
2. **Run Model**: Execute `python3 petroleum_cities_systems_model.py`
3. **View Results**: Open `systems_map.png` and `simulation_results.png`

### For Researchers
1. **Technical Details**: Read `README.md` for comprehensive documentation
2. **Validation**: See `PROJECT_SUMMARY.md` for validation metrics
3. **Customize**: Modify parameters in `petroleum_cities_systems_model.py`

---

## 📁 File Directory

### Core Implementation
| File | Size | Purpose | Priority |
|------|------|---------|----------|
| `petroleum_cities_systems_model.py` | 36 KB | Main model implementation | ⭐⭐⭐⭐⭐ |
| `requirements.txt` | 76 B | Python dependencies | ⭐⭐⭐⭐⭐ |

### Output Visualizations
| File | Size | Description | Priority |
|------|------|-------------|----------|
| `systems_map.png` | 986 KB | Conceptual feedback loop diagram | ⭐⭐⭐⭐⭐ |
| `simulation_results.png` | 4.5 MB | 9-panel dynamic analysis | ⭐⭐⭐⭐⭐ |

### Documentation
| File | Size | Content | Audience |
|------|------|---------|----------|
| `README.md` | 13 KB | Complete technical documentation | Researchers |
| `USAGE_GUIDE.md` | 8.3 KB | Quick start guide | First-time users |
| `PROJECT_SUMMARY.md` | 11 KB | Executive overview | Decision-makers |
| `INDEX.md` | This file | Navigation guide | Everyone |

### Interactive Tools
| File | Description |
|------|-------------|
| `interactive_analysis.ipynb` | Jupyter notebook template for exploration |

---

## 🎯 Use Case → File Mapping

### "I want to understand the model quickly"
→ Start with: `USAGE_GUIDE.md`  
→ Then run: `python3 petroleum_cities_systems_model.py`  
→ View: `systems_map.png`

### "I need to cite this in a paper"
→ Read: `PROJECT_SUMMARY.md` (§ Citation)  
→ Reference: `README.md` (§ Scientific Context)  
→ Include: Both PNG files in manuscript

### "I want to customize the model"
→ Read: `README.md` (§ Customization)  
→ Edit: `petroleum_cities_systems_model.py`  
→ Reference: `USAGE_GUIDE.md` (§ Customization Examples)

### "I need to validate the approach"
→ Read: `PROJECT_SUMMARY.md` (§ Validation Metrics)  
→ Check: Console output from running model  
→ Review: `simulation_results.png` (panels 4, 8, 9)

### "I want to test policy scenarios"
→ Read: `USAGE_GUIDE.md` (§ Test Policy Interventions)  
→ Modify: Parameters in model class  
→ Compare: Multiple simulation runs

---

## 📊 Model Components Breakdown

### Input: Initial Conditions (12 variables)
Set in `petroleum_cities_systems_model.py` → `initial_state` dictionary

**R1 Loop (Environmental-Livelihood):**
- `oil_spills`: 0.65 (high pollution)
- `ecosystem_health`: 0.35 (degraded)
- `livelihood_loss`: 0.60 (significant)
- `artisanal_refining`: 0.45 (moderate illegal activity)

**R2 Loop (Governance):**
- `compound_vulnerability`: 0.70 (high)
- `institutional_capacity`: 0.30 (weak)
- `public_trust`: 0.25 (low)
- `informal_governance`: 0.55 (strong informal systems)

**R3 Loop (Economic):**
- `oil_sector_dominance`: 0.72 (HHI from study)
- `economic_diversity`: 0.28 (low)
- `adaptive_capacity`: 0.20 (limited)
- `economic_shocks`: 0.50 (moderate)

### Processing: Systems Dynamics
Implemented in `system_dynamics()` method:
- 12 coupled differential equations
- 6 cross-loop coupling parameters
- Boundary dampening for [0,1] range
- LSODA ODE solver (SciPy)

### Output: Visualizations & Metrics

**Graphical Outputs:**
1. `systems_map.png`: Shows structure of feedback loops
2. `simulation_results.png`: Shows temporal evolution

**Console Metrics:**
- Initial/final states for all variables
- Loop intensity changes (R1, R2, R3)
- Composite Vulnerability Index (CVI)
- System Resilience Index
- Correlations and validation

---

## 🔧 Dependencies Explained

| Library | Purpose | Used For |
|---------|---------|----------|
| **NumPy** | Numerical arrays | State variables, time series |
| **SciPy** | Scientific computing | ODE integration (odeint) |
| **Matplotlib** | Plotting | All visualizations |
| **NetworkX** | Graph analysis | Systems map creation |
| **Seaborn** | Statistical graphics | Color palettes, styling |

Install all: `pip install -r requirements.txt`

---

## 📈 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Read Documentation     │
                 │  (USAGE_GUIDE.md)       │
                 └─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Install Dependencies   │
                 │  (requirements.txt)     │
                 └─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Run Model              │
                 │  (petroleum_cities...py)│
                 └─────────────────────────┘
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                             │
        ▼                                             ▼
┌───────────────┐                           ┌────────────────┐
│ Console Output│                           │  PNG Files     │
│ • Initial     │                           │ • systems_map  │
│ • Final       │                           │ • simulation   │
│ • Analysis    │                           │   _results     │
└───────────────┘                           └────────────────┘
        │                                             │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Interpret Results      │
                 │  (README.md §           │
                 │   Interpreting Results) │
                 └─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Customize/Extend       │
                 │  (Modify parameters,    │
                 │   test scenarios)       │
                 └─────────────────────────┘
```

---

## 🎓 Learning Path

### Beginner Level
1. ✅ Read `USAGE_GUIDE.md` introduction
2. ✅ Run model with default settings
3. ✅ View and understand `systems_map.png`
4. ✅ Read console output

### Intermediate Level
5. ✅ Read `README.md` model architecture section
6. ✅ Modify initial conditions
7. ✅ Run sensitivity analysis
8. ✅ Interpret `simulation_results.png` all panels

### Advanced Level
9. ✅ Read `PROJECT_SUMMARY.md` mathematical details
10. ✅ Modify differential equations
11. ✅ Add new variables or loops
12. ✅ Integrate with your own data

---

## 🔍 Key Concepts Explained

### What is a "Reinforcing Loop"?
A positive feedback cycle where changes amplify the original cause.
- **Example (R1)**: More pollution → Less fish → More illegal refining → More pollution
- **Visualization**: See circular arrows in `systems_map.png`
- **Proof**: Loop intensity increasing over time in `simulation_results.png`

### What is "Cross-Loop Coupling"?
Interactions between different feedback loops that create system-level effects.
- **Example**: Environmental damage (R1) weakens governance (R2)
- **Visualization**: Dashed arrows in `systems_map.png`
- **Quantification**: 6 coupling parameters in model

### What is "Composite Vulnerability Index (CVI)"?
Weighted average of key stressors: 0.3×Oil + 0.25×Livelihood + 0.25×Compound + 0.2×Shocks
- **Range**: 0 (low vulnerability) to 1 (high vulnerability)
- **Interpretation**: Higher values = worse conditions
- **Visualization**: Panel 4 in `simulation_results.png`

---

## 📞 Quick Reference

### Running the Model
```bash
python3 petroleum_cities_systems_model.py
```

### Viewing Outputs
```bash
# Console output: appears immediately
# PNG files: open with any image viewer
xdg-open systems_map.png           # Linux
open systems_map.png                # macOS
start systems_map.png               # Windows
```

### Modifying Parameters
```python
# In petroleum_cities_systems_model.py
model = PetroleumCitySystemsModel()
model.params['oil_dependency'] = 0.85  # Change from 0.72
model.initial_state['ecosystem_health'] = 0.50  # Change from 0.35
```

### Getting Help
1. **Technical issues**: Check `README.md` § Common Issues
2. **Conceptual questions**: See `USAGE_GUIDE.md` § Key Concepts
3. **Implementation details**: Read inline comments in `.py` file

---

## ✨ Best Practices

### For Reproducibility
1. ✅ Document all parameter changes
2. ✅ Use version control for modifications
3. ✅ Record random seeds if adding stochasticity
4. ✅ Save outputs with descriptive filenames

### For Publications
1. ✅ Cite the Nigeria case study source
2. ✅ Include both PNG files in manuscript
3. ✅ Report all parameter values used
4. ✅ Describe validation approach

### For Teaching
1. ✅ Start with `systems_map.png` conceptual overview
2. ✅ Run live demo with students
3. ✅ Show parameter sensitivity
4. ✅ Compare baseline vs. intervention scenarios

---

## 🎉 Success Checklist

After running the model, you should have:

- [ ] Console output with 7 sections of analysis
- [ ] `systems_map.png` showing 3 loops and interconnections
- [ ] `simulation_results.png` with 9 analysis panels
- [ ] Understanding of reinforcing feedback mechanisms
- [ ] Ability to modify parameters for your use case
- [ ] Validation that model demonstrates positive feedback

If all checked: **You're ready to use the model!** 🚀

---

## 📚 Additional Resources

### Theoretical Background
- SENCE Framework: Systemic, Environmental, Networked, Contextual, Emergent
- Systems Dynamics: Differential equation-based modeling
- Feedback Loops: Positive (reinforcing) vs. negative (balancing)

### Related Fields
- Urban sustainability science
- Socio-ecological systems
- Complexity theory
- Vulnerability assessment
- Resource curse literature

### Software Skills
- Python programming
- Scientific computing (NumPy/SciPy)
- Data visualization (Matplotlib)
- Network analysis (NetworkX)

---

## 🌟 Project Highlights

| Metric | Value |
|--------|-------|
| **State Variables** | 12 |
| **Feedback Loops** | 3 (R1, R2, R3) |
| **Cross-Couplings** | 6 parameters |
| **Simulation Horizon** | 50 years |
| **Time Resolution** | 0.1 years |
| **Calibration Points** | r=0.78, HHI=0.72 |
| **Visualization Panels** | 9 |
| **Documentation Pages** | 30+ (combined) |
| **Code Lines** | ~1,200 (commented) |
| **Runtime** | <15 seconds |
| **Output Resolution** | 300 DPI |

---

**Version**: 1.0  
**Last Updated**: October 3, 2025  
**Status**: ✅ Complete and Validated

---

**Happy Modeling! 🎓🔬📊**
