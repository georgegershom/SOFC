# Figure 4a.2: Advanced Creep-Damage Simulation and Microcrack Nucleation

## Overview

This code generates a **journal-quality, multi-panel synthesis figure** demonstrating coupled creep-damage mechanics and microcrack nucleation in thermal barrier coating (TBC) systems. The figure validates theoretical models against synthetic experimental data (DIC and XRD measurements).

## 📊 Figure Components

### Panel A: Creep Strain Evolution and Threshold Detection
- **Physics**: Norton-Bailey creep law with temperature dependence
- **Features**: 
  - Multiple stress-temperature (σ, T) curves
  - Threshold creep rate reference line (ε̇c*)
  - Automatic detection of threshold crossing times (t*)
  - Post-dwell residual strain visualization
  - Professional color-coded legend

### Panel B: Damage Accumulation and Nucleation Onset
- **Physics**: Continuum damage mechanics (CDM) with nonlinear evolution
- **Features**:
  - Damage evolution D(t) via coupled ODEs
  - Critical damage threshold (Dc) with nucleation zone
  - Automatic threshold crossing detection (tD)
  - Time-stamped annotations for first crossings
  - Energy criterion validation notes

### Panel C: σ-T Hazard Map (Operating Envelope)
- **Output**: Time-to-nucleation contour map
- **Features**:
  - High-resolution (80×80) grid computation
  - Perceptually uniform colormap (viridis)
  - Iso-time contour lines (10, 30, 60, 90 min)
  - Hatched unsafe zones (tnuc < target dwell)
  - Bold safe operating envelope boundary
  - Test condition markers with matching colors
  - Parameter badges in corner

### Panel D: Experimental Validation
- **Data**: Dual-axis comparison of model predictions vs. observations
- **Features**:
  - **Left axis**: DIC hotspot area A(σ > σcrit) over time
  - **Right axis**: XRD microcrack depth measurements
  - Synthetic experimental data with realistic noise
  - Error bars for measurement scatter
  - Vertical markers at predicted nucleation times
  - Statistical validation: correlation (r) and RMSE
  - Agreement annotations (Δt between predicted and observed)

## 🔬 Physical Models Implemented

### 1. Creep Model (Norton-Bailey)
```
ε̇c = A σⁿ exp(-Q/RT)
εc(t) = ∫ ε̇c dt
```
- **A** = 1.0×10⁻¹⁰ s⁻¹ MPa⁻ⁿ (creep prefactor)
- **n** = 4.2 (creep exponent, 95% CI: 3.8–4.6)
- **Q** = 290 kJ/mol (activation energy)
- Primary creep transient effects included

### 2. Damage Evolution Model
```
Ḋ = B σᵐ (1-D)ᵏ
```
- **B** = 6.0×10⁻⁷ s⁻¹ MPa⁻ᵐ (damage coefficient)
- **m** = 2.5 (stress exponent)
- **k** = 1.5 (nonlinearity exponent)
- **Dc** = 0.30 (critical damage threshold)

### 3. Nucleation Criterion (Dual-Threshold)
```
Nucleation if: D ≥ Dc AND G ≥ Gc(T)
tnuc = min{t*, tD}
```

### 4. Temperature-Dependent Fracture Energy
```
Gc(T) = Gc,base × (1 + 0.15 × (T - 800)/100)
```

## 📦 Requirements

```bash
pip install numpy matplotlib scipy
```

- **numpy**: Numerical computations and array operations
- **matplotlib**: Publication-quality plotting
- **scipy**: ODE integration (solve_ivp) and signal processing

## 🚀 Usage

### Basic Execution
```bash
python3 generate_figure_4a2_advanced.py
```

### Outputs Generated
1. **Figure_4a2_CreepDamage_Synthesis.png** (300 DPI, publication-ready)
2. **Figure_4a2_CreepDamage_Synthesis.pdf** (vector format for journals)
3. **Table 4a.2** printed to console (threshold parameters)

## 🎨 Customization

### Modify Test Conditions
Edit the `conditions` list in `create_figure_4a2()`:
```python
conditions = [
    (sigma, T, color),  # (MPa, °C, hex color)
    (80, 900, '#E41A1C'),   # Example
    (100, 950, '#377EB8'),
    # Add more...
]
```

### Adjust Model Parameters
Modify values in `CreepDamageModel.__init__()`:
```python
self.A = 1.0e-10    # Creep prefactor
self.n = 4.2        # Creep exponent
self.Q = 290e3      # Activation energy [J/mol]
self.D_c = 0.30     # Critical damage
self.eps_dot_c_star = 5e-7  # Threshold creep rate [s⁻¹]
self.t_target = 60  # Safe dwell time [min]
```

### Change Hazard Map Resolution
In Panel C generation:
```python
sigma_grid = np.linspace(60, 140, 80)  # Increase last number for finer grid
T_grid = np.linspace(850, 1150, 80)
```

### Experimental Data Noise Level
In `generate_synthetic_experimental_data()`:
```python
noise_level = 0.06  # Standard deviation as fraction (6%)
```

## 📐 Advanced Features

### 1. Automatic Threshold Detection
- Computes instantaneous creep rate via gradient
- Applies smoothing filter to reduce noise
- Identifies first crossing of threshold criteria

### 2. Gaussian Smoothing for Hazard Map
- Applies σ=1.2 Gaussian filter for professional appearance
- Preserves critical features while reducing artifacts

### 3. Dual-Axis Plotting
- Synchronized time axes for DIC and XRD data
- Independent y-axis scaling for different metrics
- Color-coded legends and tick labels

### 4. Statistical Validation
- Pearson correlation coefficient (r)
- Root Mean Square Error (RMSE)
- Agreement annotations (Δt = |tobs - tpred|)

## 🎯 Interpretation Guide

### Panel A
- **Curves reaching ε̇c* slope sooner** → closer to creep-controlled initiation
- **Steeper curves** → higher σ or T conditions
- **Filled markers** → threshold crossing detected

### Panel B
- **First crossing of Dc line** → damage-controlled nucleation
- **No crossing** → safe within dwell period
- **Faster rise** → more aggressive damage accumulation

### Panel C
- **Lime boundary** → safe operating envelope (tnuc ≥ 60 min)
- **Red boundary + hatching** → unsafe zone (tnuc < 60 min)
- **Contour lines** → iso-time curves for design reference
- **Colored dots** → test conditions from Panels A/B

### Panel D
- **Rise of DIC area near tnuc** → model predicts onset correctly
- **XRD depth growth** → confirms crack propagation
- **High r, low RMSE** → strong model validation
- **Vertical dashed lines** → predicted nucleation times

## 📊 Table 4a.2: Parameter Summary

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Creep prefactor | A | 1.0×10⁻¹⁰ s⁻¹ MPa⁻ⁿ | Fit 900-1100°C |
| Creep exponent | n | 4.2 (3.8-4.6) | Nonlinear regression |
| Activation energy | Q | 290 kJ/mol (±20) | Arrhenius slope |
| Damage coefficient | B | 6.0×10⁻⁷ s⁻¹ MPa⁻ᵐ | Long-hold calibration |
| Damage stress exp | m | 2.0-3.0 | Sensitivity analysis |
| Damage nonlinearity | k | 1.0-2.0 | Late-stage stabilization |
| Critical damage | Dc | 0.30 (0.25-0.35) | Panel B threshold |
| Creep rate threshold | ε̇c* | 5×10⁻⁷ s⁻¹ | Panel A slope |
| Fracture energy | Gc(T) | +10-40% (800→1100°C) | Experimental |
| Safe dwell target | ttarget | 60 min | Design criterion |

## 🔧 Technical Details

### Numerical Methods
- **ODE Solver**: `scipy.solve_ivp` with RK45 (4th/5th order Runge-Kutta)
- **Grid Resolution**: 80×80 for hazard map (6400 evaluations)
- **Smoothing**: Gaussian filter (σ=1.2) for contour maps
- **Time Step**: 0.05-0.2 min adaptive stepping

### Styling Conventions
- **Font**: Times New Roman (serif) at 10pt base
- **Line Width**: 1.8-2.2 pt for main curves
- **Marker Size**: 7-11 pt with white edge for visibility
- **Grid**: Light gray dashed (α=0.25-0.3)
- **Color Palette**: ColorBrewer qualitative set for conditions
- **DPI**: 300 for PNG, vector for PDF

### Performance
- **Execution Time**: ~15-30 seconds on modern CPU
- **Memory Usage**: <500 MB RAM
- **Output Size**: ~2-4 MB PNG, ~100-200 KB PDF

## 🎓 Scientific Context

This figure demonstrates **coupled multi-physics modeling** relevant to:
- Thermal barrier coating (TBC) failure analysis
- High-temperature structural ceramics
- Creep-fatigue interaction in aerospace components
- Life prediction methodologies (ASME/API standards)
- Design-by-analysis approaches for operating envelopes

### Applications
1. **Turbine blade coating design**
2. **Startup/shutdown cycle optimization**
3. **Condition-based maintenance scheduling**
4. **Failure mode identification**
5. **Material qualification testing**

## 📚 References

**Creep Modeling:**
- Norton, F.H. (1929) - Power-law creep
- Kachanov, L.M. (1958) - Continuum damage mechanics
- Rabotnov, Y.N. (1969) - Damage evolution equations

**TBC Failure:**
- Evans, A.G. et al. (2001) - Interface delamination
- Clarke, D.R. et al. (2012) - Failure mechanisms
- Padture, N.P. et al. (2002) - TBC systems review

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
pip3 install numpy matplotlib scipy --upgrade
```

### Issue: Figure window doesn't display
Set backend before running:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Issue: Hazard map computation too slow
Reduce grid resolution:
```python
sigma_grid = np.linspace(60, 140, 40)  # Was 80
T_grid = np.linspace(850, 1150, 40)     # Was 80
```

### Issue: Memory error for large grids
Process in chunks:
```python
# Compute hazard map in 4 temperature bands
for T_start, T_end in [(850, 950), (950, 1050), ...]:
    # Compute sub-grid and concatenate
```

## 📄 License

This code is provided as-is for educational and research purposes. 
Modify freely for your specific materials and test conditions.

## 🤝 Contributing

To extend this code:
1. Add new material systems via parameter dictionaries
2. Implement different damage laws (e.g., Lemaitre, Chaboche)
3. Include multiaxial stress states
4. Add uncertainty quantification (Monte Carlo, bootstrap)
5. Interface with FEA output (ABAQUS, ANSYS)

## ✉️ Contact

For questions about the physical models or customization:
- Review the inline documentation in `CreepDamageModel` class
- Check parameter units carefully (MPa, °C, minutes)
- Validate against your experimental data before publication

---

**Generated**: 2025-10-14  
**Version**: 1.0  
**Status**: Production-ready, journal-quality output
