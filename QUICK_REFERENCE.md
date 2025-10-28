# 🔋 SOFC Materials - Quick Reference Card

## 📊 Material Properties at a Glance

### ANODE: Ni-YSZ (Nickel-Yttria Stabilized Zirconia)
```
TEC:         12.5 × 10⁻⁶ K⁻¹
E (Young):   45 GPa
Density:     5,800 kg/m³
Porosity:    35%
σ_e:         15,000 S/m
σ_i:         0.8 S/m
i₀:          4,500 A/m² (H₂ oxidation)
Temp:        1073 K
```
**Function:** Fuel oxidation, current collection  
**Key Feature:** Composite material, percolated Ni for conductivity

---

### ELECTROLYTE: 8YSZ (8 mol% Y₂O₃-ZrO₂)
```
TEC:         10.5 × 10⁻⁶ K⁻¹
E (Young):   200 GPa
Density:     5,900 kg/m³
Porosity:    2% (gas-tight)
σ_e:         0.001 S/m (negligible)
σ_i:         3.2 S/m @ 1073K
Eₐ:          80 kJ/mol
Temp range:  800-1273 K
```
**Function:** Oxygen ion conduction, gas separation  
**Key Feature:** Pure ionic conductor, temperature-dependent (Arrhenius)

---

### ELECTROLYTE: CGO (Gadolinium-Doped Ceria)
```
TEC:         12.5 × 10⁻⁶ K⁻¹
E (Young):   180 GPa
Density:     7,200 kg/m³
Porosity:    3%
σ_i:         8.5 S/m @ 873K
Eₐ:          65 kJ/mol
Temp range:  700-1073 K
```
**Function:** Intermediate-temperature electrolyte  
**Key Feature:** Higher conductivity than YSZ at lower temperatures

---

### CATHODE: LSM (La₀.₈Sr₀.₂MnO₃)
```
TEC:         11.8 × 10⁻⁶ K⁻¹
E (Young):   55 GPa
Density:     5,200 kg/m³
Porosity:    30%
σ_e:         25,000 S/m
σ_i:         0.05 S/m
i₀:          2,800 A/m² (O₂ reduction)
Temp:        1073 K
```
**Function:** Oxygen reduction, current collection  
**Key Feature:** Electronic conductor, requires YSZ for TPB

---

### CATHODE: LSM-YSZ Composite
```
TEC:         11.2 × 10⁻⁶ K⁻¹
E (Young):   50 GPa
Porosity:    35%
σ_e:         18,000 S/m
σ_i:         1.2 S/m
i₀:          3,500 A/m²
Temp:        1073 K
```
**Function:** Enhanced cathode performance  
**Key Feature:** Extended triple-phase boundary length

---

### CATHODE: LSCF (La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃)
```
TEC:         14.5 × 10⁻⁶ K⁻¹
E (Young):   60 GPa
Density:     5,500 kg/m³
Porosity:    35%
σ_e:         35,000 S/m
σ_i:         2.5 S/m
i₀:          5,200 A/m² (highest!)
Temp:        873 K
```
**Function:** Intermediate-temperature cathode  
**Key Feature:** Mixed ionic-electronic conductor (MIEC)

---

### INTERCONNECT: Crofer 22 APU
```
TEC:         12.0 × 10⁻⁶ K⁻¹
E (Young):   170 GPa
Density:     7,600 kg/m³
Porosity:    0% (dense metal)
σ_e:         1.2 × 10⁶ S/m
Temp:        1073 K
```
**Function:** Current collection, gas separation  
**Key Feature:** Ferritic steel, TEC-matched to ceramics

---

## 🔬 Critical Design Parameters

### TEC Matching (Thermal Expansion)
```
Goal: Minimize TEC mismatch to prevent cracking

Good matches:
  Ni-YSZ (12.5) ≈ Crofer (12.0) ≈ LSM (11.8) ≈ 8YSZ (10.5)
  
Mismatch concern:
  LSCF (14.5) vs 8YSZ (10.5) → Use CGO buffer layer
```

### Porosity Requirements
```
Anode:       30-40%  (H₂ diffusion)
Cathode:     30-35%  (O₂ diffusion)
Electrolyte: <5%     (gas-tight, typically 2-3%)
Interconnect: 0%     (dense metal)
```

### Operating Temperatures
```
High-Temp SOFC:           800-1000°C (1073-1273 K)
Intermediate-Temp SOFC:   600-800°C  (873-1073 K)
Low-Temp SOFC:            400-600°C  (673-873 K)
```

---

## 📐 Essential Equations

### 1. Ionic Conductivity (Arrhenius)
```
σ(T) = σ₀ × exp(-Eₐ / (R × T))

Where:
  σ₀ = pre-exponential factor [S/m]
  Eₐ = activation energy [J/mol]
  R  = 8.314 J/(mol·K)
  T  = temperature [K]

Example (8YSZ at 1073K):
  σ = 3.2 S/m, Eₐ = 80 kJ/mol
```

### 2. Norton-Bailey Creep
```
ε̇ = B × σⁿ × exp(-Q / (R × T))

Where:
  B = pre-exponential [Pa⁻ⁿ s⁻¹]
  σ = stress [Pa]
  n = stress exponent
  Q = activation energy [J/mol]

Material Parameters:
  Ni-YSZ: B=2.8×10⁻¹³, n=2.1, Q=320 kJ/mol
  8YSZ:   B=1.5×10⁻¹⁵, n=1.0, Q=520 kJ/mol
  Crofer: B=8.5×10⁻¹², n=4.2, Q=280 kJ/mol
```

### 3. Butler-Volmer (Electrode Kinetics)
```
i = 2 × i₀ × sinh(F × η / (2 × R × T))

Where:
  i₀ = exchange current density [A/m²]
  η  = activation overpotential [V]
  F  = 96,485 C/mol

Exchange Currents @ 1073K:
  Ni-YSZ anode:       4,500 A/m²
  LSM cathode:        2,800 A/m²
  LSM-YSZ composite:  3,500 A/m²
  LSCF @ 873K:        5,200 A/m²
```

### 4. Ohmic Loss
```
R_ohm = t / (σ × A)

Where:
  t = thickness [m]
  σ = conductivity [S/m]
  A = area [m²]

Example: 10 μm YSZ @ 1073K, 100 cm²
  R = 10×10⁻⁶ / (3.2 × 100×10⁻⁴) = 0.031 Ω
  ASR = 0.31 Ω·cm²
```

---

## 🎯 Typical SOFC Performance Targets

### Cell Level
```
OCV (Open Circuit):     1.0-1.1 V @ 1073K
Operating Voltage:      0.7-0.8 V @ 0.5 A/cm²
Power Density:          0.3-1.0 W/cm²
ASR (Area Specific):    0.15-0.3 Ω·cm² total
```

### Component Resistances
```
Electrolyte:   0.05-0.15 Ω·cm² (YSZ, 10-20 μm)
Anode:         0.02-0.05 Ω·cm²
Cathode:       0.05-0.15 Ω·cm²
Interconnect:  <0.01 Ω·cm²
```

### Lifetime Targets
```
Degradation:  <1% per 1000 hours
Target life:  40,000-80,000 hours (5-10 years)
Thermal cycles: >50-100 cycles
```

---

## 🔧 Material Selection Guide

### For High-Temperature SOFC (800-1000°C)
```
Anode:        Ni-YSZ (standard)
Electrolyte:  8YSZ (10-20 μm)
Cathode:      LSM or LSM-YSZ
Interconnect: Crofer 22 APU or similar
```

### For Intermediate-Temperature SOFC (600-800°C)
```
Anode:        Ni-YSZ (modified) or Cu-Ceria
Electrolyte:  CGO (5-10 μm) or thin YSZ
Cathode:      LSCF or SSC
Interconnect: Crofer or other Fe-Cr alloys
```

### TEC Compatibility Matrix
```
✅ Good Match (ΔTec < 2):
  Ni-YSZ ↔ 8YSZ ↔ LSM ↔ Crofer

⚠️ Acceptable (ΔTec 2-3):
  CGO ↔ Ni-YSZ
  LSM-YSZ ↔ 8YSZ

❌ Poor Match (ΔTec > 3):
  LSCF ↔ 8YSZ (use CGO buffer)
```

---

## 💻 Quick Code Snippets

### Python: Get Material Properties
```python
from sofc_material_properties import SOFCMaterialDatabase

db = SOFCMaterialDatabase()
props = db.get_material_properties('anode', 'Ni-YSZ')
tec = props['thermo_physical'].thermal_expansion_coefficient
print(f"TEC: {tec*1e6:.1f} × 10⁻⁶ /K")
```

### Python: Calculate Conductivity at Temperature
```python
sigma_800C = db.get_ionic_conductivity('electrolyte', '8YSZ', 
                                       temperature=1073)
print(f"σ_ion = {sigma_800C:.2f} S/m")
```

### Python: Calculate Creep Rate
```python
creep = db.calculate_creep_rate('anode', 'Ni-YSZ',
                                stress=50e6,      # 50 MPa
                                temperature=1073) # 800°C
print(f"Creep rate: {creep:.3e} /s")
```

### MATLAB: Load CSV
```matlab
data = readtable('sofc_material_properties.csv');
ni_ysz = data(strcmp(data.Layer, 'Ni-YSZ'), :);
```

---

## 📊 Units Quick Reference

| Property | Symbol | Unit |
|----------|--------|------|
| Temperature | T | K (Kelvin) |
| TEC | α | K⁻¹ or °C⁻¹ |
| Young's Modulus | E | Pa or GPa |
| Stress | σ | Pa or MPa |
| Strain Rate | ε̇ | s⁻¹ |
| Conductivity | σ | S/m or S/cm |
| Current Density | i, j | A/m² or A/cm² |
| Voltage | V, E, η | V (Volt) |
| Resistance | R | Ω (Ohm) |
| ASR | - | Ω·cm² |
| Activation Energy | Eₐ, Q | J/mol or kJ/mol |
| Power Density | P | W/cm² |
| Porosity | ε, φ | - (fraction) or % |
| Density | ρ | kg/m³ or g/cm³ |

---

## 🚨 Common Pitfalls to Avoid

1. **TEC Mismatch**
   - Always check thermal expansion compatibility
   - Use buffer layers when necessary
   - Consider entire thermal cycle (RT → operating T)

2. **Electrolyte Thickness**
   - Too thick → high ohmic resistance
   - Too thin → mechanical fragility, pinholes
   - Optimal: 10-20 μm for YSZ

3. **Porosity**
   - Anode/cathode need 30-40% for gas transport
   - Electrolyte must be <5% for gas-tightness
   - Don't forget tortuosity effects

4. **Temperature Extrapolation**
   - Properties are temperature-dependent
   - Don't extrapolate beyond measured ranges
   - Use Arrhenius equations when available

5. **Creep at High Temperature**
   - Significant above 800°C
   - Consider long-term deformation
   - Metallic components most susceptible

---

## 📁 File Quick Reference

```
Main Data:
  sofc_material_properties.csv    → Flat table (Excel, MATLAB)
  sofc_material_properties.json   → Hierarchical (Python, JS)
  sofc_material_properties.py     → Python module with methods

Documentation:
  README.md                       → Full documentation
  EQUATIONS_AND_MODELS.md        → Math models & equations
  INDEX.md                       → Complete project index
  QUICK_REFERENCE.md             → This file

Visualization:
  visualize_properties.py         → Generate all plots
  *.png (6 files)                → Material property charts
```

---

## 🔗 Constants to Remember

```
R  = 8.314 J/(mol·K)      Gas constant
F  = 96,485 C/mol         Faraday constant
kB = 1.381×10⁻²³ J/K      Boltzmann constant
NA = 6.022×10²³ mol⁻¹     Avogadro number
```

---

## ✅ Quick Checklist for SOFC Design

- [ ] TEC matched within ±2 × 10⁻⁶ K⁻¹?
- [ ] Electrolyte <5% porosity (gas-tight)?
- [ ] Electrodes 30-40% porosity (gas diffusion)?
- [ ] Operating temperature appropriate for materials?
- [ ] Ionic conductivity >1 S/m at operating temp?
- [ ] Exchange current density >1000 A/m²?
- [ ] Total ASR <0.3 Ω·cm²?
- [ ] Mechanical strength adequate (E > 40 GPa)?
- [ ] Creep rate acceptable for lifetime?
- [ ] Thermal cycling compatibility verified?

---

**Last Updated:** October 3, 2025  
**For detailed information, see README.md and EQUATIONS_AND_MODELS.md**

---

*Print this card for quick reference during SOFC design and simulation work!* 🚀
