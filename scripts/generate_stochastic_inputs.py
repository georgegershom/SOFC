"""
Generate stochastic input files for epistemic uncertainty analysis
Three scenarios:
1. rho=0.0: Independent interfaces
2. rho=0.5: Moderately correlated interfaces
3. High-T: High temperature degradation
"""

import numpy as np
import pandas as pd
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples for Monte Carlo
N_SAMPLES = 10000

# Room temperature material properties (from 04_uncertainty_material_properties.csv)
# YSZ|GDC interface
Gc_YSZ_GDC_mean = 2.85  # J/m²
Gc_YSZ_GDC_std = 0.12   # J/m²

# GDC|LSCF interface
Gc_GDC_LSCF_mean = 3.20  # J/m²
Gc_GDC_LSCF_std = 0.15   # J/m²

# Elastic moduli
E_YSZ_mean, E_YSZ_std = 210, 8.4  # GPa
E_GDC_mean, E_GDC_std = 180, 7.2  # GPa
E_LSCF_mean, E_LSCF_std = 120, 6.0  # GPa

# Coefficient of Thermal Expansion
CTE_YSZ_mean, CTE_YSZ_std = 10.5, 0.3  # 10^-6/K
CTE_GDC_mean, CTE_GDC_std = 12.4, 0.4  # 10^-6/K
CTE_LSCF_mean, CTE_LSCF_std = 13.2, 0.5  # 10^-6/K

# Poisson ratios
nu_YSZ_mean, nu_YSZ_std = 0.31, 0.01
nu_GDC_mean, nu_GDC_std = 0.29, 0.01
nu_LSCF_mean, nu_LSCF_std = 0.28, 0.015

def generate_correlated_normals(mean1, std1, mean2, std2, rho, n_samples):
    """
    Generate correlated normal random variables using Cholesky decomposition
    """
    # Covariance matrix
    cov = [[std1**2, rho*std1*std2],
           [rho*std1*std2, std2**2]]
    
    # Generate samples
    samples = np.random.multivariate_normal([mean1, mean2], cov, n_samples)
    
    return samples[:, 0], samples[:, 1]

# ============================================================================
# SCENARIO 1: rho = 0.0 (Independent interfaces)
# ============================================================================
print("Generating Scenario 1: Independent interfaces (rho=0.0)...")

Gc_YSZ_GDC_indep = np.random.normal(Gc_YSZ_GDC_mean, Gc_YSZ_GDC_std, N_SAMPLES)
Gc_GDC_LSCF_indep = np.random.normal(Gc_GDC_LSCF_mean, Gc_GDC_LSCF_std, N_SAMPLES)

# Other material properties (assumed independent)
E_YSZ = np.random.normal(E_YSZ_mean, E_YSZ_std, N_SAMPLES)
E_GDC = np.random.normal(E_GDC_mean, E_GDC_std, N_SAMPLES)
E_LSCF = np.random.normal(E_LSCF_mean, E_LSCF_std, N_SAMPLES)

CTE_YSZ = np.random.normal(CTE_YSZ_mean, CTE_YSZ_std, N_SAMPLES)
CTE_GDC = np.random.normal(CTE_GDC_mean, CTE_GDC_std, N_SAMPLES)
CTE_LSCF = np.random.normal(CTE_LSCF_mean, CTE_LSCF_std, N_SAMPLES)

nu_YSZ = np.random.normal(nu_YSZ_mean, nu_YSZ_std, N_SAMPLES)
nu_GDC = np.random.normal(nu_GDC_mean, nu_GDC_std, N_SAMPLES)
nu_LSCF = np.random.normal(nu_LSCF_mean, nu_LSCF_std, N_SAMPLES)

df_indep = pd.DataFrame({
    'sample_id': range(1, N_SAMPLES + 1),
    'Gc_YSZ_GDC': Gc_YSZ_GDC_indep,
    'Gc_GDC_LSCF': Gc_GDC_LSCF_indep,
    'E_YSZ_GPa': E_YSZ,
    'E_GDC_GPa': E_GDC,
    'E_LSCF_GPa': E_LSCF,
    'CTE_YSZ': CTE_YSZ,
    'CTE_GDC': CTE_GDC,
    'CTE_LSCF': CTE_LSCF,
    'nu_YSZ': nu_YSZ,
    'nu_GDC': nu_GDC,
    'nu_LSCF': nu_LSCF,
    'correlation_rho': 0.0,
    'temperature_C': 25
})

df_indep.to_csv('../data/stochastic_inputs_rho0.00.csv', index=False)
print(f"  Saved: stochastic_inputs_rho0.00.csv ({N_SAMPLES} samples)")

# ============================================================================
# SCENARIO 2: rho = 0.5 (Correlated interfaces)
# ============================================================================
print("Generating Scenario 2: Correlated interfaces (rho=0.5)...")

rho = 0.5
Gc_YSZ_GDC_corr, Gc_GDC_LSCF_corr = generate_correlated_normals(
    Gc_YSZ_GDC_mean, Gc_YSZ_GDC_std,
    Gc_GDC_LSCF_mean, Gc_GDC_LSCF_std,
    rho, N_SAMPLES
)

# Reuse the same material properties for fair comparison
df_corr = pd.DataFrame({
    'sample_id': range(1, N_SAMPLES + 1),
    'Gc_YSZ_GDC': Gc_YSZ_GDC_corr,
    'Gc_GDC_LSCF': Gc_GDC_LSCF_corr,
    'E_YSZ_GPa': E_YSZ,
    'E_GDC_GPa': E_GDC,
    'E_LSCF_GPa': E_LSCF,
    'CTE_YSZ': CTE_YSZ,
    'CTE_GDC': CTE_GDC,
    'CTE_LSCF': CTE_LSCF,
    'nu_YSZ': nu_YSZ,
    'nu_GDC': nu_GDC,
    'nu_LSCF': nu_LSCF,
    'correlation_rho': 0.5,
    'temperature_C': 25
})

df_corr.to_csv('../data/stochastic_inputs_rho0.50.csv', index=False)
print(f"  Saved: stochastic_inputs_rho0.50.csv ({N_SAMPLES} samples)")

# ============================================================================
# SCENARIO 3: High Temperature (800°C) - Uncorrelated with degradation
# ============================================================================
print("Generating Scenario 3: High-Temperature (800°C)...")

# High-temperature degradation factor (deterministic scaling)
T_degradation_factor = 0.75

# CRITICAL ASSUMPTION: We keep the same std dev (this is the problem!)
Gc_YSZ_GDC_HT_mean = Gc_YSZ_GDC_mean * T_degradation_factor
Gc_GDC_LSCF_HT_mean = Gc_GDC_LSCF_mean * T_degradation_factor

# Generate samples (uncorrelated)
Gc_YSZ_GDC_HT = np.random.normal(Gc_YSZ_GDC_HT_mean, Gc_YSZ_GDC_std, N_SAMPLES)
Gc_GDC_LSCF_HT = np.random.normal(Gc_GDC_LSCF_HT_mean, Gc_GDC_LSCF_std, N_SAMPLES)

# Temperature-dependent elastic moduli (slight softening)
E_YSZ_HT = np.random.normal(E_YSZ_mean * 0.92, E_YSZ_std, N_SAMPLES)
E_GDC_HT = np.random.normal(E_GDC_mean * 0.88, E_GDC_std, N_SAMPLES)
E_LSCF_HT = np.random.normal(E_LSCF_mean * 0.85, E_LSCF_std, N_SAMPLES)

df_ht = pd.DataFrame({
    'sample_id': range(1, N_SAMPLES + 1),
    'Gc_YSZ_GDC': Gc_YSZ_GDC_HT,
    'Gc_GDC_LSCF': Gc_GDC_LSCF_HT,
    'E_YSZ_GPa': E_YSZ_HT,
    'E_GDC_GPa': E_GDC_HT,
    'E_LSCF_GPa': E_LSCF_HT,
    'CTE_YSZ': CTE_YSZ,
    'CTE_GDC': CTE_GDC,
    'CTE_LSCF': CTE_LSCF,
    'nu_YSZ': nu_YSZ,
    'nu_GDC': nu_GDC,
    'nu_LSCF': nu_LSCF,
    'correlation_rho': 0.0,
    'temperature_C': 800
})

df_ht.to_csv('../data/stochastic_inputs_HT_uncorrelated.csv', index=False)
print(f"  Saved: stochastic_inputs_HT_uncorrelated.csv ({N_SAMPLES} samples)")

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\nScenario 1 (rho=0.0):")
print(f"  Gc_YSZ_GDC: μ={Gc_YSZ_GDC_indep.mean():.3f}, σ={Gc_YSZ_GDC_indep.std():.3f}")
print(f"  Gc_GDC_LSCF: μ={Gc_GDC_LSCF_indep.mean():.3f}, σ={Gc_GDC_LSCF_indep.std():.3f}")
print(f"  Empirical correlation: {np.corrcoef(Gc_YSZ_GDC_indep, Gc_GDC_LSCF_indep)[0,1]:.4f}")

print("\nScenario 2 (rho=0.5):")
print(f"  Gc_YSZ_GDC: μ={Gc_YSZ_GDC_corr.mean():.3f}, σ={Gc_YSZ_GDC_corr.std():.3f}")
print(f"  Gc_GDC_LSCF: μ={Gc_GDC_LSCF_corr.mean():.3f}, σ={Gc_GDC_LSCF_corr.std():.3f}")
print(f"  Empirical correlation: {np.corrcoef(Gc_YSZ_GDC_corr, Gc_GDC_LSCF_corr)[0,1]:.4f}")

print("\nScenario 3 (High-T):")
print(f"  Gc_YSZ_GDC: μ={Gc_YSZ_GDC_HT.mean():.3f}, σ={Gc_YSZ_GDC_HT.std():.3f}")
print(f"  Gc_GDC_LSCF: μ={Gc_GDC_LSCF_HT.mean():.3f}, σ={Gc_GDC_LSCF_HT.std():.3f}")
print(f"  Mean reduction factor: {Gc_YSZ_GDC_HT.mean()/Gc_YSZ_GDC_indep.mean():.3f}")

print("\n" + "="*70)
print("Generation complete!")
