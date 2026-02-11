# filename: quantify_epistemic_gap.py
import pandas as pd
import numpy as np
import scipy.stats as stats

# Load the surrogate datasets
df_indep = pd.read_csv('../data/stochastic_inputs_rho0.00.csv')
df_corr  = pd.read_csv('../data/stochastic_inputs_rho0.50.csv')
df_ht    = pd.read_csv('../data/stochastic_inputs_HT_uncorrelated.csv')

# Calculate "System" Resistance (Sum of Gc for both interfaces)
# This represents the total energy barrier to full delamination
R_indep = df_indep['Gc_YSZ_GDC'] + df_indep['Gc_GDC_LSCF']
R_corr  = df_corr['Gc_YSZ_GDC'] + df_corr['Gc_GDC_LSCF']
R_ht    = df_ht['Gc_YSZ_GDC'] + df_ht['Gc_GDC_LSCF']

# Calculate Failure Probability for a hypothetical Driving Force J_applied = 3.0 J/m^2
J_app = 3.0
Pf_indep = np.mean(R_indep < J_app)
Pf_corr  = np.mean(R_corr < J_app)
Pf_ht    = np.mean(R_ht < J_app)

print("=" * 70)
print("          EPISTEMIC UNCERTAINTY QUANTIFICATION REPORT")
print("=" * 70)
print("\n--- IMPACT OF MISSING DATA ---")
print(f"\nApplied Energy Release Rate: J_applied = {J_app:.2f} J/m²")
print(f"\nScenario 1 (Rho=0.0): P_fail = {Pf_indep:.4f} ({Pf_indep*100:.2f}%)")
print(f"Scenario 2 (Rho=0.5): P_fail = {Pf_corr:.4f} ({Pf_corr*100:.2f}%)")
print(f"Scenario 3 (High-T):  P_fail = {Pf_ht:.4f} ({Pf_ht*100:.2f}%)")
print("-" * 70)
print(f"\nEpistemic Uncertainty Gap (Rho effect): {abs(Pf_indep - Pf_corr)*100:.2f}%")
print(f"Temperature Effect Gap: {abs(Pf_indep - Pf_ht)*100:.2f}%")
print(f"Maximum uncertainty range: {(max(Pf_indep, Pf_corr, Pf_ht) - min(Pf_indep, Pf_corr, Pf_ht))*100:.2f}%")

# Statistical summary
print("\n" + "=" * 70)
print("          SYSTEM RESISTANCE STATISTICS")
print("=" * 70)

print("\nScenario 1 (Independent, ρ=0.0):")
print(f"  Mean System Resistance: {R_indep.mean():.3f} J/m²")
print(f"  Std Dev: {R_indep.std():.3f} J/m²")
print(f"  5th Percentile: {np.percentile(R_indep, 5):.3f} J/m²")
print(f"  95th Percentile: {np.percentile(R_indep, 95):.3f} J/m²")

print("\nScenario 2 (Correlated, ρ=0.5):")
print(f"  Mean System Resistance: {R_corr.mean():.3f} J/m²")
print(f"  Std Dev: {R_corr.std():.3f} J/m²")
print(f"  5th Percentile: {np.percentile(R_corr, 5):.3f} J/m²")
print(f"  95th Percentile: {np.percentile(R_corr, 95):.3f} J/m²")

print("\nScenario 3 (High-Temperature, 800°C):")
print(f"  Mean System Resistance: {R_ht.mean():.3f} J/m²")
print(f"  Std Dev: {R_ht.std():.3f} J/m²")
print(f"  5th Percentile: {np.percentile(R_ht, 5):.3f} J/m²")
print(f"  95th Percentile: {np.percentile(R_ht, 95):.3f} J/m²")

# Interface-level failure probabilities
print("\n" + "=" * 70)
print("          INDIVIDUAL INTERFACE FAILURE PROBABILITIES")
print("=" * 70)

# Using a lower threshold for individual interfaces
J_interface = 2.5  # J/m²

print(f"\nCritical Energy Release Rate: {J_interface:.2f} J/m²")

print("\nYSZ|GDC Interface:")
print(f"  P_fail (Independent): {np.mean(df_indep['Gc_YSZ_GDC'] < J_interface)*100:.2f}%")
print(f"  P_fail (Correlated):  {np.mean(df_corr['Gc_YSZ_GDC'] < J_interface)*100:.2f}%")
print(f"  P_fail (High-T):      {np.mean(df_ht['Gc_YSZ_GDC'] < J_interface)*100:.2f}%")

print("\nGDC|LSCF Interface:")
print(f"  P_fail (Independent): {np.mean(df_indep['Gc_GDC_LSCF'] < J_interface)*100:.2f}%")
print(f"  P_fail (Correlated):  {np.mean(df_corr['Gc_GDC_LSCF'] < J_interface)*100:.2f}%")
print(f"  P_fail (High-T):      {np.mean(df_ht['Gc_GDC_LSCF'] < J_interface)*100:.2f}%")

# Joint failure probability (both interfaces fail)
print("\n" + "=" * 70)
print("          JOINT FAILURE PROBABILITY (CATASTROPHIC)")
print("=" * 70)

Pf_joint_indep = np.mean((df_indep['Gc_YSZ_GDC'] < J_interface) & (df_indep['Gc_GDC_LSCF'] < J_interface))
Pf_joint_corr = np.mean((df_corr['Gc_YSZ_GDC'] < J_interface) & (df_corr['Gc_GDC_LSCF'] < J_interface))
Pf_joint_ht = np.mean((df_ht['Gc_YSZ_GDC'] < J_interface) & (df_ht['Gc_GDC_LSCF'] < J_interface))

print(f"\nProbability of BOTH interfaces failing:")
print(f"  Independent (ρ=0.0): {Pf_joint_indep*100:.3f}%")
print(f"  Correlated (ρ=0.5):  {Pf_joint_corr*100:.3f}%")
print(f"  High-Temperature:    {Pf_joint_ht*100:.3f}%")
print(f"\nCorrelation Effect: {(Pf_joint_corr/Pf_joint_indep - 1)*100:.1f}% increase in joint failure risk")

print("\n" + "=" * 70)
print("          KEY FINDINGS")
print("=" * 70)

print("\n1. MISSING_DATASET_01 (Correlation) Impact:")
print(f"   The correlation coefficient uncertainty contributes ±{abs(Pf_indep - Pf_corr)*100:.2f}%")
print(f"   to the failure probability estimate.")

print("\n2. MISSING_DATASET_02 (High-T) Impact:")
print(f"   Temperature degradation increases failure risk by {(Pf_ht/Pf_indep - 1)*100:.1f}%")
print(f"   (assuming constant variance - likely non-conservative)")

print("\n3. Joint Failure Risk:")
print(f"   Correlation increases catastrophic failure probability by")
print(f"   {(Pf_joint_corr/Pf_joint_indep - 1)*100:.1f}%, demonstrating the criticality")
print(f"   of measuring interfacial covariance.")

print("\n" + "=" * 70)
print("Analysis complete. See ../figures/ for visualizations.")
print("=" * 70)
