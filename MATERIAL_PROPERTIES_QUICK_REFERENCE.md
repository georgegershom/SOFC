# SOFC Material Properties - Quick Reference Tables

**Last Updated:** 2025-10-09 | **Database Version:** 1.0

---

## Table 1: Elastic Properties at Operating Temperature (800°C)

| Material | E (GPa) | ν | G (GPa) | K (GPa) | ρ (kg/m³) |
|----------|---------|-----|---------|---------|-----------|
| 8YSZ Electrolyte | 170 ± 4 | 0.226 ± 0.006 | 69.3 | 152 | 5900 |
| Ni-YSZ Anode (30% porous) | 29 ± 4 | 0.300 ± 0.02 | 11.2 | 24.2 | 6800 |
| LSM-YSZ Cathode (35% porous) | 42 ± 5 | 0.280 ± 0.02 | 16.4 | 38.2 | 5200 |
| Ni Metal | 162 ± 6 | 0.325 ± 0.007 | 61.1 | 243 | 8900 |
| NiO | 205 ± 10 | 0.27 ± 0.02 | 80.7 | 203 | 6670 |

**Notes:**
- E = Young's Modulus
- ν = Poisson's Ratio  
- G = Shear Modulus (calculated: G = E/[2(1+ν)])
- K = Bulk Modulus (calculated: K = E/[3(1-2ν)])
- ρ = Density

---

## Table 2: Fracture Properties at 25°C and 800°C

| Material | K_Ic @ 25°C (MPa√m) | K_Ic @ 800°C (MPa√m) | Reduction | G_Ic @ 800°C (J/m²) |
|----------|---------------------|----------------------|-----------|---------------------|
| **8YSZ Electrolyte** | 2.85 ± 0.18 | 2.28 ± 0.13 | 20% | 28.9 ± 2.5 |
| **Ni-YSZ Anode** | 5.8 ± 0.9 | 4.5 ± 0.7 | 22% | 65.2 ± 10.2 |
| **LSM-YSZ Cathode** | 2.2 ± 0.4 | 1.8 ± 0.3 | 18% | 22.1 ± 4.2 |
| NiO | 1.8 ± 0.3 | — | — | — |

**Flexural Strength (σ_f):**
- 8YSZ @ 25°C: 385 ± 42 MPa (Weibull m = 8.5)
- 8YSZ @ 800°C: 215 ± 22 MPa (Weibull m = 11.2) ⚠️ **Design Critical**
- Ni-YSZ @ 800°C: 55 ± 10 MPa

---

## Table 3: Interface Fracture Properties ⚠️ **MOST CRITICAL**

| Interface | Location | K_Ic @ 25°C | K_Ic @ 800°C | G_Ic @ 800°C | Criticality |
|-----------|----------|-------------|--------------|--------------|-------------|
| **YSZ/Ni-YSZ** | Electrolyte/Anode | 1.85 ± 0.35 | **1.38 ± 0.28** | **10.5 ± 3.2** | **HIGHEST** ⚠️ |
| YSZ/LSM-YSZ | Electrolyte/Cathode | 2.15 ± 0.42 | 1.72 ± 0.38 | 16.8 ± 4.8 | HIGH |

**Interfacial Strength (Tensile):**
- YSZ/Ni-YSZ @ 800°C: 48 ± 10 MPa ⚠️
- YSZ/LSM @ 800°C: 62 ± 14 MPa

**Key Insight:**  
Interface toughness is **35-40% lower** than bulk YSZ - this is the primary failure location!

---

## Table 4: Thermal Expansion Coefficients (CTE)

| Material | CTE @ 25-400°C (×10⁻⁶ K⁻¹) | CTE @ 400-800°C (×10⁻⁶ K⁻¹) | Mean CTE (×10⁻⁶ K⁻¹) | Δα vs YSZ |
|----------|---------------------------|-----------------------------|-----------------------|-----------|
| **8YSZ Electrolyte** | 10.2 ± 0.2 | 10.7 ± 0.2 | **10.5** | — |
| **Ni-YSZ Anode** | 12.7 ± 0.4 | 13.3 ± 0.4 | **13.0** | +2.5 ⚠️ |
| **LSM-YSZ Cathode** | 11.4 ± 0.3 | 11.9 ± 0.3 | **11.6** | +1.1 |
| Ni Metal | 13.8 ± 0.3 | 14.8 ± 0.4 | 14.0 | +3.5 |
| **NiO** | 14.5 ± 0.5 | 15.2 ± 0.5 | **15.0** | +4.5 ⚠️⚠️ |

**CTE Mismatch Impact:**
- Δα = 2.5 (Anode-Electrolyte) → Residual stress ≈ **100-150 MPa**
- Critical for thermal cycling failure!

---

## Table 5: Chemical Expansion Coefficients @ 800°C

| Material | Coefficient (per Δδ) | pO₂ Range (atm) | Expansion Type | Severity |
|----------|----------------------|-----------------|----------------|----------|
| 8YSZ | 0.02 ± 0.01 | 10⁻²⁰ to 1 | Minimal | LOW ✓ |
| Ni-YSZ | 0.15 ± 0.03 | 10⁻²⁰ to 10⁻¹⁵ | Moderate | MEDIUM ⚠️ |
| LSM-YSZ | 0.08 ± 0.02 | 10⁻⁵ to 1 | Moderate | MEDIUM ⚠️ |
| NiO | 0.22 ± 0.05 | 10⁻²⁰ to 10⁻⁵ | Significant | HIGH ⚠️ |

**Redox Transformation (Most Critical):**
- **Ni → NiO:**
  - Volumetric expansion: **165%**
  - Linear strain: **17.5%** ⚠️⚠️⚠️
  - **CATASTROPHIC** for anode structure!

---

## Table 6: Thermal Conductivity @ 800°C

| Material | k (W/m·K) | Contribution to Thermal Gradients |
|----------|-----------|-----------------------------------|
| 8YSZ Electrolyte | 2.30 ± 0.07 | Low conductivity → high gradients |
| Ni-YSZ Anode | 4.5 ± 0.5 | Moderate |
| LSM-YSZ Cathode | 2.8 ± 0.4 | Low |
| Ni Metal | 52.8 ± 2.2 | High (good for current collection) |

---

## Table 7: Creep Parameters (Norton-Bailey Law: ε̇ = B σⁿ exp(-Q/RT))

| Material | B (s⁻¹ MPa⁻ⁿ) | n | Q (kJ/mol) | Valid T Range | Dominant Mechanism |
|----------|---------------|-----|------------|---------------|-------------------|
| 8YSZ | 8.5 × 10⁻¹² | 1.8 | 385 | 700-1200°C | Diffusional creep |
| Ni-YSZ | 2.8 × 10⁻¹⁰ | 2.5 | 275 | 600-1000°C | Ni-dominated creep |

**Time-Dependent Effects @ 800°C:**
- 8YSZ: ~20% stress relaxation after 100 hours
- Ni-YSZ: >40% stress relaxation after 100 hours

---

## Critical Property Summary for FEA Input

### 8YSZ Electrolyte @ 800°C (Most Used)

```
Material: 8YSZ_Electrolyte_800C
---------------------------------------
E = 170 GPa
ν = 0.226
α = 10.8 × 10⁻⁶ K⁻¹
ρ = 5900 kg/m³
k = 2.30 W/m·K
Cp = 590 J/kg·K

Fracture:
K_Ic = 2.28 MPa√m
G_Ic = 28.9 J/m²
σ_f = 215 MPa (mean)

Creep:
B = 8.5 × 10⁻¹² s⁻¹ MPa⁻¹·⁸
n = 1.8
Q = 385 kJ/mol
```

### Ni-YSZ Anode @ 800°C

```
Material: NiYSZ_Anode_800C
---------------------------------------
E = 29 GPa (30% porosity)
ν = 0.30
α = 13.3 × 10⁻⁶ K⁻¹
ρ = 6800 kg/m³
k = 4.5 W/m·K
Cp = 520 J/kg·K

Fracture:
K_Ic = 4.5 MPa√m
σ_f = 55 MPa

Chemical expansion:
β_chem = 0.15 per Δδ
```

### YSZ/Ni-YSZ Interface @ 800°C ⚠️

```
Interface: YSZ_NiYSZ_800C
---------------------------------------
K_Ic = 1.38 MPa√m  ← CRITICAL!
G_Ic = 10.5 J/m²
σ_interface = 48 MPa

Interface adhesion: 1.2 J/m²
Mode II toughness: 2.1 MPa√m
```

---

## Stress Limits and Safety Factors

### Maximum Allowable Stresses (@ 800°C)

| Location | Material | σ_max (MPa) | Safety Factor | Source |
|----------|----------|-------------|---------------|--------|
| Electrolyte bulk | 8YSZ | 215 | 1.5 (design) | Flexural strength |
| Electrolyte edge | 8YSZ | 165 | 1.3 | With stress concentration |
| **Anode/Electrolyte interface** | **Interface** | **48** | **1.5** | **Most critical** ⚠️ |
| Cathode/Electrolyte interface | Interface | 62 | 1.5 | |

**Typical Operating Stresses (Literature):**
- Electrolyte: 100-150 MPa (Von Mises)
- Principal stress: 120-145 MPa
- Interface shear: 20-30 MPa

**Safety Assessment:**
- SF = σ_limit / σ_operating
- SF > 1.5 → Acceptable
- SF < 1.3 → High risk
- SF < 1.0 → Predicted failure

---

## Temperature-Dependent Properties (Curve Fits)

### Young's Modulus vs Temperature

**8YSZ:**
```
E(T) = 205 - 0.0408 × T  [GPa, T in °C]
R² = 0.996
Valid: 25-1200°C
```

**Ni-YSZ (30% porosity):**
```
E(T) = 65 × exp(-0.0018 × T)  [GPa, T in °C]
Valid: 25-1000°C
```

### Fracture Toughness vs Temperature

**8YSZ:**
```
K_Ic(T) = 2.85 - 0.00078 × T  [MPa√m, T in °C]
Valid: 25-1200°C
```

**YSZ/Ni-YSZ Interface:**
```
K_Ic(T) = 1.85 - 0.000608 × T  [MPa√m, T in °C]
Valid: 25-800°C
```

### CTE vs Temperature

**8YSZ:**
```
α(T) = 10.0 + 0.0011 × T  [×10⁻⁶ K⁻¹, T in °C]
Valid: 25-1200°C
```

**Ni-YSZ:**
```
α(T) = 12.3 + 0.0017 × T  [×10⁻⁶ K⁻¹, T in °C]
Valid: 25-1000°C
```

---

## Key Takeaways for Design

### ⚠️ Critical Failure Modes (In Order of Likelihood)

1. **Interface Delamination** (YSZ/Ni-YSZ)
   - Driven by: CTE mismatch, residual stresses
   - Limiting property: K_Ic,interface = 1.38 MPa√m
   - Mitigation: Graded interfaces, optimized sintering

2. **Electrolyte Edge Cracking**
   - Driven by: Stress concentrations, thermal cycling
   - Limiting property: K_Ic,YSZ = 2.28 MPa√m
   - Mitigation: Chamfering, edge sealing

3. **Anode Redox Damage**
   - Driven by: Ni→NiO expansion (17.5%)
   - Catastrophic if exposed to air during operation
   - Mitigation: Fuel supply redundancy, gradual reduction

4. **Thermal Cycling Fatigue**
   - Driven by: Cyclic stresses (Δσ ≈ 90 MPa)
   - Cumulative damage over 1000+ cycles
   - Mitigation: Slow ramp rates, creep relaxation

### ✓ Design Guidelines

**Layer Thicknesses:**
- Electrolyte: 150 μm (balance strength vs. ionic resistance)
- Anode: 300-500 μm
- Cathode: 30-50 μm

**Operating Limits:**
- Max temperature: 850°C (creep, strength reduction)
- Max thermal gradient: 50°C/cm
- Max ramp rate: 2°C/min (startup/shutdown)
- Max thermal cycles: Design for 500-1000 cycles

**Material Selection Criteria:**
- CTE matching: Δα < 2.0 × 10⁻⁶ K⁻¹
- Interface strength: σ_i > 50 MPa @ 800°C
- Fracture toughness: K_Ic > 2.0 MPa√m @ 800°C

---

## Units and Conversions

### Fracture Mechanics

```
K_Ic [MPa√m] ↔ G_Ic [J/m²]:
G_Ic = K_Ic² × (1-ν²) / E

where E in MPa, result in J/m² (= N/m = MPa·mm)
```

### Stress-Strain

```
ε = σ / E    (elastic)
ε_thermal = α × ΔT
ε_creep = ∫ B σⁿ exp(-Q/RT) dt
```

### Temperature

```
T[K] = T[°C] + 273.15
R = 8.314 J/(mol·K) = 0.008314 kJ/(mol·K)
```

---

## Data Quality Flags

| Property Type | Typical Uncertainty | Confidence | Number of Sources |
|---------------|---------------------|------------|-------------------|
| Elastic (E, ν) | ±5% | HIGH | 10-15 per material |
| CTE | ±3% | HIGH | 8-12 per material |
| Fracture (bulk) | ±15% | MEDIUM | 5-8 per material |
| **Fracture (interface)** | **±25%** | **MEDIUM-LOW** | **2-4 per interface** ⚠️ |
| Chemical expansion | ±30% | MEDIUM-LOW | 3-6 per material |
| Creep | ±40% | LOW-MEDIUM | 2-5 per material |

**Note:** Interface properties have highest uncertainty due to measurement difficulty!

---

## Recommended Usage

**For Preliminary Design:**
- Use mean values from Tables 1-4
- Apply safety factors: SF = 1.5-2.0

**For Detailed FEA:**
- Use temperature-dependent curve fits
- Include creep for time > 100 hours
- Model interfaces explicitly with cohesive zone elements

**For Probabilistic Analysis:**
- Use Weibull distributions for strength
- Include uncertainties from data quality flags
- Monte Carlo with ±20% variation on interface properties

---

## Quick Access - Most Critical Values

```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️  TOP 5 CRITICAL PARAMETERS FOR SOFC FRACTURE RISK  ⚠️  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Interface Toughness (YSZ/Ni-YSZ):  1.38 MPa√m         │
│  2. Electrolyte Strength (800°C):      215 MPa            │
│  3. CTE Mismatch (Anode-Electrolyte):  2.5 × 10⁻⁶ K⁻¹    │
│  4. Interface Strength (800°C):        48 MPa             │
│  5. Ni→NiO Expansion:                  17.5% strain       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0  
**For Full Details See:** `material_property_dataset.json`  
**For Visualizations See:** `*.png` files  
**For Raw Data See:** `*.csv` files
