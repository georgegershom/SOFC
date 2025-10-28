# SOFC Material Models and Equations

## Overview
This document provides the mathematical models and equations used for SOFC material behavior, including temperature dependencies, mechanical models, and electrochemical relationships.

---

## 1. Temperature-Dependent Ionic Conductivity

### Arrhenius Equation

The ionic conductivity of electrolyte materials (YSZ, CGO) follows Arrhenius behavior:

```
σ(T) = σ₀ × exp(-Eₐ / (R × T))
```

**Parameters:**
- `σ(T)` = ionic conductivity at temperature T [S/m]
- `σ₀` = pre-exponential factor / reference conductivity [S/m]
- `Eₐ` = activation energy for ionic conduction [J/mol]
- `R` = universal gas constant = 8.314 J/(mol·K)
- `T` = absolute temperature [K]

**Example for 8YSZ:**
- `σ₀` = 3.2 S/m (at T_ref = 1073 K)
- `Eₐ` = 80,000 J/mol = 80 kJ/mol
- Temperature range: 800-1273 K

```python
# Python implementation
import numpy as np

def ionic_conductivity_ysz(T):
    """Calculate YSZ ionic conductivity at temperature T (Kelvin)"""
    sigma_0 = 3.2  # S/m at 1073K
    Ea = 80e3      # J/mol
    R = 8.314      # J/(mol·K)
    T_ref = 1073   # K
    
    sigma = sigma_0 * np.exp(-Ea/R * (1/T - 1/T_ref))
    return sigma

# Example
T = 973  # K
print(f"σ_ion at {T}K = {ionic_conductivity_ysz(T):.3f} S/m")
```

---

## 2. Creep Models

### Norton-Bailey Power Law Creep

Describes time-dependent plastic deformation at high temperatures:

```
ε̇_creep = B × σⁿ × exp(-Q / (R × T))
```

**Parameters:**
- `ε̇_creep` = creep strain rate [s⁻¹]
- `B` = pre-exponential constant [Pa⁻ⁿ s⁻¹]
- `σ` = applied stress [Pa]
- `n` = stress exponent (material constant) [-]
- `Q` = activation energy for creep [J/mol]
- `R` = gas constant = 8.314 J/(mol·K)
- `T` = absolute temperature [K]

**Material-Specific Parameters:**

| Material | B [Pa⁻ⁿ s⁻¹] | n [-] | Q [kJ/mol] | Notes |
|----------|--------------|-------|------------|-------|
| Ni-YSZ | 2.8 × 10⁻¹³ | 2.1 | 320 | Anode composite |
| 8YSZ | 1.5 × 10⁻¹⁵ | 1.0 | 520 | Dense electrolyte |
| Crofer 22 APU | 8.5 × 10⁻¹² | 4.2 | 280 | Metallic interconnect |

**Physical Meaning:**
- `n = 1`: Newtonian viscous flow (diffusion-controlled)
- `n > 1`: Power-law creep (dislocation-controlled)
- Higher `Q`: More temperature-sensitive creep

**Example Implementation:**

```python
def norton_bailey_creep(stress, temperature, B, n, Q):
    """
    Calculate creep strain rate using Norton-Bailey law
    
    Args:
        stress: Applied stress [Pa]
        temperature: Temperature [K]
        B: Pre-exponential factor [Pa^-n s^-1]
        n: Stress exponent [-]
        Q: Activation energy [J/mol]
    
    Returns:
        Creep strain rate [s^-1]
    """
    R = 8.314  # J/(mol·K)
    strain_rate = B * (stress ** n) * np.exp(-Q / (R * temperature))
    return strain_rate

# Example for Ni-YSZ at 1073K, 50 MPa
epsilon_dot = norton_bailey_creep(
    stress=50e6,      # 50 MPa
    temperature=1073,  # K
    B=2.8e-13,
    n=2.1,
    Q=320e3
)
print(f"Creep rate: {epsilon_dot:.3e} s⁻¹")
```

---

## 3. Plasticity: Johnson-Cook Model

For Ni-YSZ under mechanical loading with strain rate and temperature effects:

```
σ_y = [A + B × εₚⁿ] × [1 + C × ln(ε̇*/ε̇₀)] × [1 - T*ᵐ]
```

**Parameters:**
- `σ_y` = yield stress [Pa]
- `εₚ` = equivalent plastic strain [-]
- `ε̇*` = normalized strain rate = ε̇/ε̇₀ [-]
- `T*` = homologous temperature = (T - T_room)/(T_melt - T_room) [-]

**Material Constants (Ni-YSZ):**
- `A` = 180 MPa (initial yield stress)
- `B` = 420 MPa (hardening modulus)
- `n` = 0.42 (hardening exponent)
- `C` = 0.015 (strain rate sensitivity)
- `m` = 1.1 (thermal softening exponent)
- `ε̇₀` = 1.0 s⁻¹ (reference strain rate)

**Physical Interpretation:**
- **First bracket** `[A + B × εₚⁿ]`: Strain hardening
- **Second bracket** `[1 + C × ln(ε̇*/ε̇₀)]`: Strain rate sensitivity
- **Third bracket** `[1 - T*ᵐ]`: Thermal softening

```python
def johnson_cook_stress(eps_plastic, strain_rate, temperature,
                        A=180e6, B=420e6, n=0.42, C=0.015, m=1.1,
                        eps_dot_0=1.0, T_room=298, T_melt=1700):
    """
    Calculate yield stress using Johnson-Cook model
    
    Args:
        eps_plastic: Equivalent plastic strain
        strain_rate: Strain rate [s^-1]
        temperature: Temperature [K]
        A, B, n, C, m: Johnson-Cook parameters
        
    Returns:
        Yield stress [Pa]
    """
    # Strain hardening term
    term1 = A + B * (eps_plastic ** n)
    
    # Strain rate term
    eps_star = strain_rate / eps_dot_0
    term2 = 1 + C * np.log(max(eps_star, 1e-10))
    
    # Thermal softening term
    T_star = (temperature - T_room) / (T_melt - T_room)
    T_star = np.clip(T_star, 0, 1)  # Keep in [0, 1]
    term3 = 1 - (T_star ** m)
    
    sigma_y = term1 * term2 * term3
    return sigma_y

# Example
stress = johnson_cook_stress(
    eps_plastic=0.01,      # 1% plastic strain
    strain_rate=1e-3,      # s^-1
    temperature=1073       # K
)
print(f"Yield stress: {stress/1e6:.1f} MPa")
```

---

## 4. Electrochemical Models

### 4.1 Butler-Volmer Equation

Describes electrode kinetics at triple-phase boundaries:

```
i = i₀ × [exp(α_a × F × η / (R × T)) - exp(-α_c × F × η / (R × T))]
```

**For symmetric charge transfer (α_a = α_c = 0.5):**

```
i = 2 × i₀ × sinh(F × η / (2 × R × T))
```

**Parameters:**
- `i` = current density [A/m²]
- `i₀` = exchange current density [A/m²]
- `α_a` = anodic charge transfer coefficient (typically 0.5) [-]
- `α_c` = cathodic charge transfer coefficient (typically 0.5) [-]
- `η` = activation overpotential [V]
- `F` = Faraday constant = 96,485 C/mol
- `R` = gas constant = 8.314 J/(mol·K)
- `T` = temperature [K]

**Exchange Current Densities (at 1073K):**
- Ni-YSZ anode (H₂ oxidation): i₀ = 4,500 A/m²
- LSM cathode (O₂ reduction): i₀ = 2,800 A/m²
- LSM-YSZ composite cathode: i₀ = 3,500 A/m²
- LSCF cathode (873K): i₀ = 5,200 A/m²

```python
def butler_volmer_current(eta, i0, T=1073, alpha=0.5):
    """
    Calculate current density from overpotential
    
    Args:
        eta: Activation overpotential [V]
        i0: Exchange current density [A/m2]
        T: Temperature [K]
        alpha: Charge transfer coefficient
    
    Returns:
        Current density [A/m2]
    """
    F = 96485  # C/mol
    R = 8.314  # J/(mol·K)
    
    i = 2 * i0 * np.sinh(F * eta / (2 * R * T))
    return i

# Example: Ni-YSZ anode at 100 mV overpotential
eta = 0.1  # V
i = butler_volmer_current(eta, i0=4500, T=1073)
print(f"Current density: {i:.0f} A/m²")
```

### 4.2 Activation Overpotential (Tafel Approximation)

For large overpotentials (|η| > 50-100 mV):

```
η_act = (R × T) / (α × F) × ln(i / i₀)
```

**Simplified for α = 0.5:**

```
η_act = (2 × R × T) / F × ln(i / i₀)
```

At T = 1073K:
```
η_act ≈ 0.186 V × ln(i / i₀)
```

---

## 5. Effective Properties for Porous Media

### 5.1 Effective Thermal Conductivity (Parallel Model)

For composite materials like Ni-YSZ:

```
k_eff = (1 - ε) × [φ_Ni × k_Ni + φ_YSZ × k_YSZ] + ε × k_gas
```

**Where:**
- `k_eff` = effective thermal conductivity [W/(m·K)]
- `ε` = porosity [-]
- `φ_i` = volume fraction of phase i (in solid skeleton) [-]
- `k_i` = thermal conductivity of phase i [W/(m·K)]
- `k_gas` = gas thermal conductivity ≈ 0.05-0.1 W/(m·K) [W/(m·K)]

### 5.2 Effective Electrical Conductivity (Percolation)

For electronic conductivity in Ni-YSZ (Ni content > percolation threshold):

```
σ_eff,e = σ_Ni × (φ_Ni - φ_c)^t
```

**For ionic conductivity:**

```
σ_eff,i = σ_YSZ × (φ_YSZ - φ_c)^t × (1 - ε)^τ
```

**Parameters:**
- `φ_c` = percolation threshold ≈ 0.15-0.20 [-]
- `t` = percolation exponent ≈ 1.5-2.0 [-]
- `τ` = tortuosity factor ≈ 1.5-3.0 [-]

### 5.3 Effective Young's Modulus (Porosity Correction)

```
E_eff = E₀ × (1 - ε)^m
```

**Where:**
- `E_eff` = effective Young's modulus [Pa]
- `E₀` = dense material modulus [Pa]
- `ε` = porosity [-]
- `m` = material constant (typically 2-3) [-]

**Example:**
```python
def effective_youngs_modulus(E0, porosity, m=2.5):
    """Calculate effective Young's modulus accounting for porosity"""
    E_eff = E0 * (1 - porosity) ** m
    return E_eff

# Ni-YSZ with 35% porosity
E_dense = 100e9  # Pa (hypothetical dense value)
E_porous = effective_youngs_modulus(E_dense, porosity=0.35)
print(f"Effective modulus: {E_porous/1e9:.1f} GPa")
```

---

## 6. Thermal Stress Calculation

### Thermal Strain

```
ε_thermal = α × ΔT
```

**Where:**
- `α` = coefficient of thermal expansion [K⁻¹]
- `ΔT` = temperature change [K]

### Thermal Stress (Constrained)

For bi-layer structures (e.g., electrolyte on anode):

```
σ_thermal = (E / (1 - ν)) × Δα × ΔT
```

**Where:**
- `E` = Young's modulus [Pa]
- `ν` = Poisson's ratio [-]
- `Δα` = TEC mismatch = α₁ - α₂ [K⁻¹]
- `ΔT` = temperature difference [K]

**Example: YSZ electrolyte on Ni-YSZ anode**
```python
def thermal_stress_mismatch(E, nu, alpha1, alpha2, delta_T):
    """Calculate thermal stress due to TEC mismatch"""
    sigma = (E / (1 - nu)) * (alpha1 - alpha2) * delta_T
    return sigma

# Cooling from 1073K to 300K
stress = thermal_stress_mismatch(
    E=200e9,               # YSZ modulus
    nu=0.31,               # YSZ Poisson's ratio
    alpha1=10.5e-6,        # YSZ TEC
    alpha2=12.5e-6,        # Ni-YSZ TEC
    delta_T=-773           # Cooling
)
print(f"Thermal stress: {stress/1e6:.1f} MPa")
```

---

## 7. Triple-Phase Boundary (TPB) Length

### TPB Density

Empirical relationship for composite electrodes:

```
L_TPB = k × φ_ion × φ_elec × (1 - ε) / d_particle
```

**Where:**
- `L_TPB` = TPB length per unit volume [m/m³ or m⁻²]
- `k` = geometric factor ≈ 2-4 [-]
- `φ_ion` = ionic conductor volume fraction [-]
- `φ_elec` = electronic conductor volume fraction [-]
- `ε` = porosity [-]
- `d_particle` = particle diameter [m]

**Optimal composition:** φ_ion ≈ φ_elec ≈ 0.5 in solid skeleton

---

## 8. Nernst Potential

Open circuit voltage for H₂-O₂ fuel cell:

```
E_Nernst = E⁰ + (R × T) / (4 × F) × ln[(p_H₂ × p_O₂^0.5) / p_H₂O]
```

**At standard conditions:**
- `E⁰` ≈ 1.0-1.1 V (at 1073K)
- `R` = 8.314 J/(mol·K)
- `T` = temperature [K]
- `F` = 96,485 C/mol
- `p_i` = partial pressure of species i [atm]

---

## 9. Ohmic Loss

### Electrolyte Resistance

```
R_ohm = t / (σ_ion × A)
```

**Voltage drop:**

```
V_ohm = I × R_ohm = I × t / (σ_ion × A)
```

**Where:**
- `t` = electrolyte thickness [m]
- `σ_ion` = ionic conductivity [S/m]
- `A` = electrode area [m²]
- `I` = total current [A]

**Example:** 10 μm YSZ electrolyte at 1073K
```python
def ohmic_resistance(thickness, conductivity, area):
    """Calculate ohmic resistance"""
    R = thickness / (conductivity * area)
    return R

R = ohmic_resistance(
    thickness=10e-6,       # 10 μm
    conductivity=3.2,      # S/m for YSZ at 1073K
    area=100e-4            # 100 cm²
)
print(f"ASR: {R * 100e-4:.4f} Ω·cm²")
```

---

## 10. Summary of Key Constants

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Gas constant | R | 8.314 | J/(mol·K) |
| Faraday constant | F | 96,485 | C/mol |
| Boltzmann constant | k_B | 1.381×10⁻²³ | J/K |
| Elementary charge | e | 1.602×10⁻¹⁹ | C |
| Avogadro number | N_A | 6.022×10²³ | mol⁻¹ |

---

## References

1. Singhal, S.C. & Kendall, K. (Eds.). *High Temperature Solid Oxide Fuel Cells: Fundamentals, Design and Applications*. Elsevier, 2003.

2. O'Hayre, R., Cha, S.-W., Colella, W., & Prinz, F.B. *Fuel Cell Fundamentals*. Wiley, 2016.

3. Bove, R. & Ubertini, S. (Eds.). *Modeling Solid Oxide Fuel Cells*. Springer, 2008.

4. Ivers-Tiffée, E. & Weber, A. "Evaluation of electrochemical impedance spectra by the distribution of relaxation times." *Journal of the Ceramic Society of Japan*, 2017.

---

**Note:** All equations and models are provided for educational and research purposes. Validate parameters experimentally for critical applications.
