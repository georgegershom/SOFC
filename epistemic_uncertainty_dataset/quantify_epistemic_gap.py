#!/usr/bin/env python3
"""
quantify_epistemic_gap.py
=========================
Quantify the "Ignorance Gap" in Probabilistic Failure Maps for SOC Interfaces.

This script loads the surrogate stochastic input datasets and computes
the failure probabilities under different assumptions, exposing the
epistemic uncertainty due to missing data.

Usage:
    python3 quantify_epistemic_gap.py
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import os

CSV_DIR = "csv"

# Load the surrogate datasets
df_indep = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.00.csv'))
df_corr  = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.50.csv'))
df_ht    = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_HT_uncorrelated.csv'))

# Calculate "System" Resistance (Sum of Gc for both interfaces)
# This represents the total energy barrier to full delamination
R_indep = df_indep['Gc_YSZ_GDC'] + df_indep['Gc_GDC_LSCF']
R_corr  = df_corr['Gc_YSZ_GDC'] + df_corr['Gc_GDC_LSCF']
R_ht    = df_ht['Gc_YSZ_GDC'] + df_ht['Gc_GDC_LSCF']

# =============================================================================
# Impact of Missing Data
# =============================================================================
print("=" * 60)
print("EPISTEMIC UNCERTAINTY QUANTIFICATION")
print("Probabilistic Failure Maps for SOC Interfacial Toughness")
print("=" * 60)

# Calculate Failure Probability for a hypothetical Driving Force J_applied = 3.0 J/m^2
J_app = 3.0
Pf_indep = np.mean(R_indep < J_app)
Pf_corr  = np.mean(R_corr < J_app)
Pf_ht    = np.mean(R_ht < J_app)

print("\n--- IMPACT OF MISSING DATA ---")
print(f"Applied load:  J_app = {J_app:.1f} J/m²")
print(f"N_samples:     {len(R_indep)}")
print()
print(f"Scenario 1 (ρ=0.0, RT):  P_fail = {Pf_indep:.4f}  ({Pf_indep*100:.2f}%)")
print(f"Scenario 2 (ρ=0.5, RT):  P_fail = {Pf_corr:.4f}  ({Pf_corr*100:.2f}%)")
print(f"Scenario 3 (ρ=0.0, HT):  P_fail = {Pf_ht:.4f}  ({Pf_ht*100:.2f}%)")
print("-" * 60)
print(f"Epistemic Gap (ρ unknown):     ΔP_f = {abs(Pf_indep - Pf_corr)*100:.2f}%")
print(f"Epistemic Gap (T unknown):     ΔP_f = {abs(Pf_indep - Pf_ht)*100:.2f}%")
print(f"Total Epistemic Range:         [{min(Pf_indep, Pf_corr, Pf_ht)*100:.2f}%, "
      f"{max(Pf_indep, Pf_corr, Pf_ht)*100:.2f}%]")

# =============================================================================
# Bounded Interval for P_f(system)
# =============================================================================
print("\n--- BOUNDED INTERVAL FOR P_f(system) ---")
print("Because ρ(Gc_anode, Gc_cathode) is unknown and f(Gc|T=800°C) is unknown,")
print("the true failure probability lies within:")
print()
Pf_lower = min(Pf_indep, Pf_corr, Pf_ht)
Pf_upper = max(Pf_indep, Pf_corr, Pf_ht)
print(f"   P_f(system) ∈ [{Pf_lower:.4f}, {Pf_upper:.4f}]")
print(f"                = [{Pf_lower*100:.2f}%, {Pf_upper*100:.2f}%]")
print()
print("This interval represents the EPISTEMIC uncertainty.")
print("It can only be reduced by acquiring the missing data.")

# =============================================================================
# Statistics summary
# =============================================================================
print("\n--- DISTRIBUTION STATISTICS ---")
for name, R in [("Scenario 1 (ρ=0.0, RT)", R_indep),
                ("Scenario 2 (ρ=0.5, RT)", R_corr),
                ("Scenario 3 (ρ=0.0, HT)", R_ht)]:
    print(f"\n{name}:")
    print(f"  Mean(R_system)   = {R.mean():.3f} J/m²")
    print(f"  StdDev(R_system) = {R.std():.3f} J/m²")
    print(f"  CoV              = {R.std()/R.mean()*100:.1f}%")
    print(f"  5th percentile   = {np.percentile(R, 5):.3f} J/m²")
    print(f"  Median           = {np.percentile(R, 50):.3f} J/m²")
    print(f"  95th percentile  = {np.percentile(R, 95):.3f} J/m²")

# =============================================================================
# Sweep over J_applied
# =============================================================================
print("\n--- FAILURE PROBABILITY vs J_APPLIED ---")
print(f"{'J_app (J/m²)':<15} {'P_f(ρ=0)':<12} {'P_f(ρ=0.5)':<12} {'P_f(HT)':<12} {'Max Gap (%)':<12}")
print("-" * 63)
for J in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    pf1 = np.mean(R_indep < J)
    pf2 = np.mean(R_corr < J)
    pf3 = np.mean(R_ht < J)
    gap = (max(pf1, pf2, pf3) - min(pf1, pf2, pf3)) * 100
    print(f"{J:<15.1f} {pf1:<12.4f} {pf2:<12.4f} {pf3:<12.4f} {gap:<12.2f}")

print("\n" + "=" * 60)
print("Analysis complete.")
print("=" * 60)
