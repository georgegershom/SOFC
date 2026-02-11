"""
Generate visualization figures for epistemic uncertainty analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
from scipy import stats

# Set publication-quality plot parameters
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18

# Load datasets
print("Loading datasets...")
df_indep = pd.read_csv('../data/stochastic_inputs_rho0.00.csv')
df_corr  = pd.read_csv('../data/stochastic_inputs_rho0.50.csv')
df_ht    = pd.read_csv('../data/stochastic_inputs_HT_uncorrelated.csv')

# Calculate system resistance
R_indep = df_indep['Gc_YSZ_GDC'] + df_indep['Gc_GDC_LSCF']
R_corr  = df_corr['Gc_YSZ_GDC'] + df_corr['Gc_GDC_LSCF']
R_ht    = df_ht['Gc_YSZ_GDC'] + df_ht['Gc_GDC_LSCF']

# ============================================================================
# FIGURE 1: THE "CONE OF IGNORANCE" - Fragility Curves
# ============================================================================
print("Generating Figure 1: Cone of Ignorance (Fragility Curves)...")

fig, ax = plt.subplots(figsize=(12, 8))

# Range of applied energy release rates
J_range = np.linspace(3.5, 7.0, 200)

# Calculate failure probabilities for each scenario
Pf_indep_curve = [np.mean(R_indep < J) for J in J_range]
Pf_corr_curve = [np.mean(R_corr < J) for J in J_range]
Pf_ht_curve = [np.mean(R_ht < J) for J in J_range]

# Plot the curves
ax.plot(J_range, Pf_indep_curve, 'b-', linewidth=2.5, label='Independent (ρ=0.0, RT)')
ax.plot(J_range, Pf_corr_curve, 'g--', linewidth=2.5, label='Correlated (ρ=0.5, RT)')
ax.plot(J_range, Pf_ht_curve, 'r-.', linewidth=3.0, label='High-Temperature (800°C)', alpha=0.8)

# Fill the "cone of ignorance" between curves
ax.fill_between(J_range, Pf_indep_curve, Pf_corr_curve, 
                alpha=0.2, color='yellow', 
                label='Epistemic Uncertainty\n(MISSING_DATASET_01)')

# Annotations
ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.text(6.8, 0.015, '1% Failure Threshold', fontsize=10, color='gray')

ax.axhline(y=0.10, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.text(6.8, 0.11, '10% Failure Threshold', fontsize=10, color='gray')

# Labels and formatting
ax.set_xlabel('Applied Energy Release Rate, $J_{applied}$ [J/m²]', fontsize=14, fontweight='bold')
ax.set_ylabel('Probability of System Failure, $P_f$', fontsize=14, fontweight='bold')
ax.set_title('Probabilistic Failure Maps: The "Cone of Ignorance"\nUncertainty Due to Missing Correlation & Temperature Data', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(3.5, 7.0)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('../figures/01_cone_of_ignorance_fragility_curves.png', dpi=300, bbox_inches='tight')
print("  Saved: 01_cone_of_ignorance_fragility_curves.png")
plt.close()

# ============================================================================
# FIGURE 2: Cumulative Distribution Functions (CDFs)
# ============================================================================
print("Generating Figure 2: Cumulative Distribution Functions...")

fig, ax = plt.subplots(figsize=(12, 8))

# Sort for CDF plotting
R_indep_sorted = np.sort(R_indep)
R_corr_sorted = np.sort(R_corr)
R_ht_sorted = np.sort(R_ht)

# Calculate empirical CDF
cdf_vals = np.arange(1, len(R_indep) + 1) / len(R_indep)

# Plot CDFs
ax.plot(R_indep_sorted, cdf_vals, 'b-', linewidth=2.5, label='Independent (ρ=0.0, RT)')
ax.plot(R_corr_sorted, cdf_vals, 'g--', linewidth=2.5, label='Correlated (ρ=0.5, RT)')
ax.plot(R_ht_sorted, cdf_vals, 'r-.', linewidth=3.0, label='High-Temperature (800°C)', alpha=0.8)

# Mark percentiles
for percentile, color, ls in [(5, 'orange', ':'), (50, 'purple', '-.'), (95, 'brown', '--')]:
    val_indep = np.percentile(R_indep, percentile)
    val_corr = np.percentile(R_corr, percentile)
    val_ht = np.percentile(R_ht, percentile)
    
    ax.axhline(y=percentile/100, color=color, linestyle=ls, linewidth=1.2, alpha=0.5)
    ax.text(7.2, percentile/100 - 0.02, f'{percentile}th', fontsize=9, color=color)

# Labels
ax.set_xlabel('System Resistance, $R = G_{c,YSZ|GDC} + G_{c,GDC|LSCF}$ [J/m²]', 
              fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Probability, $F_R(r)$', fontsize=14, fontweight='bold')
ax.set_title('Cumulative Distribution Functions of System Resistance\nQuantifying Aleatory vs. Epistemic Uncertainty', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(3.5, 7.5)

plt.tight_layout()
plt.savefig('../figures/02_system_resistance_CDF.png', dpi=300, bbox_inches='tight')
print("  Saved: 02_system_resistance_CDF.png")
plt.close()

# ============================================================================
# FIGURE 3: Interface Correlation Scatter Plot
# ============================================================================
print("Generating Figure 3: Interface Correlation Scatter Plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scenario 1: Independent
axes[0].scatter(df_indep['Gc_YSZ_GDC'], df_indep['Gc_GDC_LSCF'], 
                alpha=0.3, s=10, color='blue')
rho_indep = np.corrcoef(df_indep['Gc_YSZ_GDC'], df_indep['Gc_GDC_LSCF'])[0, 1]
axes[0].set_title(f'Independent\nρ = {rho_indep:.3f}', fontweight='bold')
axes[0].set_xlabel('$G_{c,YSZ|GDC}$ [J/m²]')
axes[0].set_ylabel('$G_{c,GDC|LSCF}$ [J/m²]')
axes[0].grid(True, alpha=0.3)

# Scenario 2: Correlated
axes[1].scatter(df_corr['Gc_YSZ_GDC'], df_corr['Gc_GDC_LSCF'], 
                alpha=0.3, s=10, color='green')
rho_corr = np.corrcoef(df_corr['Gc_YSZ_GDC'], df_corr['Gc_GDC_LSCF'])[0, 1]
axes[1].set_title(f'Correlated\nρ = {rho_corr:.3f}', fontweight='bold')
axes[1].set_xlabel('$G_{c,YSZ|GDC}$ [J/m²]')
axes[1].set_ylabel('$G_{c,GDC|LSCF}$ [J/m²]')
axes[1].grid(True, alpha=0.3)

# Scenario 3: High-T
axes[2].scatter(df_ht['Gc_YSZ_GDC'], df_ht['Gc_GDC_LSCF'], 
                alpha=0.3, s=10, color='red')
rho_ht = np.corrcoef(df_ht['Gc_YSZ_GDC'], df_ht['Gc_GDC_LSCF'])[0, 1]
axes[2].set_title(f'High-Temperature\nρ = {rho_ht:.3f}', fontweight='bold')
axes[2].set_xlabel('$G_{c,YSZ|GDC}$ [J/m²]')
axes[2].set_ylabel('$G_{c,GDC|LSCF}$ [J/m²]')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Interfacial Toughness Correlation Structures\nVisualization of MISSING_DATASET_01', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('../figures/03_interface_correlation_scatter.png', dpi=300, bbox_inches='tight')
print("  Saved: 03_interface_correlation_scatter.png")
plt.close()

# ============================================================================
# FIGURE 4: Probability Density Functions
# ============================================================================
print("Generating Figure 4: Probability Density Functions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# YSZ|GDC Interface
axes[0, 0].hist(df_indep['Gc_YSZ_GDC'], bins=50, alpha=0.5, density=True, 
                color='blue', label='RT Independent')
axes[0, 0].hist(df_corr['Gc_YSZ_GDC'], bins=50, alpha=0.5, density=True, 
                color='green', label='RT Correlated')
axes[0, 0].hist(df_ht['Gc_YSZ_GDC'], bins=50, alpha=0.5, density=True, 
                color='red', label='High-T')
axes[0, 0].set_xlabel('$G_{c,YSZ|GDC}$ [J/m²]')
axes[0, 0].set_ylabel('Probability Density')
axes[0, 0].set_title('YSZ|GDC Interface Toughness', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# GDC|LSCF Interface
axes[0, 1].hist(df_indep['Gc_GDC_LSCF'], bins=50, alpha=0.5, density=True, 
                color='blue', label='RT Independent')
axes[0, 1].hist(df_corr['Gc_GDC_LSCF'], bins=50, alpha=0.5, density=True, 
                color='green', label='RT Correlated')
axes[0, 1].hist(df_ht['Gc_GDC_LSCF'], bins=50, alpha=0.5, density=True, 
                color='red', label='High-T')
axes[0, 1].set_xlabel('$G_{c,GDC|LSCF}$ [J/m²]')
axes[0, 1].set_ylabel('Probability Density')
axes[0, 1].set_title('GDC|LSCF Interface Toughness', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# System Resistance
axes[1, 0].hist(R_indep, bins=50, alpha=0.5, density=True, 
                color='blue', label='RT Independent')
axes[1, 0].hist(R_corr, bins=50, alpha=0.5, density=True, 
                color='green', label='RT Correlated')
axes[1, 0].hist(R_ht, bins=50, alpha=0.5, density=True, 
                color='red', label='High-T')
axes[1, 0].set_xlabel('System Resistance [J/m²]')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].set_title('Total System Resistance', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Comparison of variance (showing the problem with MISSING_DATASET_02)
scenarios = ['Independent\n(ρ=0.0, RT)', 'Correlated\n(ρ=0.5, RT)', 'High-Temp\n(800°C)']
std_devs = [R_indep.std(), R_corr.std(), R_ht.std()]
colors_bar = ['blue', 'green', 'red']

bars = axes[1, 1].bar(scenarios, std_devs, color=colors_bar, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Standard Deviation [J/m²]', fontweight='bold')
axes[1, 1].set_title('Variability Comparison\n(MISSING_DATASET_02 Issue)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, std in zip(bars, std_devs):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{std:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

fig.suptitle('Probability Density Functions and Variability Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('../figures/04_probability_density_functions.png', dpi=300, bbox_inches='tight')
print("  Saved: 04_probability_density_functions.png")
plt.close()

# ============================================================================
# FIGURE 5: Risk Matrix Heatmap
# ============================================================================
print("Generating Figure 5: Risk Matrix Heatmap...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Create 2D histograms
J_bins = np.linspace(1.5, 4.5, 30)

for idx, (df, title, color) in enumerate([
    (df_indep, 'Independent (ρ=0.0)', 'Blues'),
    (df_corr, 'Correlated (ρ=0.5)', 'Greens'),
    (df_ht, 'High-T (800°C)', 'Reds')
]):
    h, xedges, yedges = np.histogram2d(df['Gc_YSZ_GDC'], df['Gc_GDC_LSCF'], bins=30)
    
    im = axes[idx].imshow(h.T, origin='lower', aspect='auto', cmap=color,
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    axes[idx].set_xlabel('$G_{c,YSZ|GDC}$ [J/m²]')
    axes[idx].set_ylabel('$G_{c,GDC|LSCF}$ [J/m²]')
    axes[idx].set_title(title, fontweight='bold')
    
    # Add contour lines
    axes[idx].contour(h.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      colors='white', alpha=0.3, linewidths=1)
    
    plt.colorbar(im, ax=axes[idx], label='Sample Count')

fig.suptitle('Joint Probability Density Maps\nInterfacial Toughness Phase Space', 
             fontsize=16, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('../figures/05_risk_matrix_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved: 05_risk_matrix_heatmap.png")
plt.close()

# ============================================================================
# FIGURE 6: Sensitivity Analysis - Box Plots
# ============================================================================
print("Generating Figure 6: Sensitivity Analysis Box Plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Individual interface comparisons
data_ysz = [df_indep['Gc_YSZ_GDC'], df_corr['Gc_YSZ_GDC'], df_ht['Gc_YSZ_GDC']]
data_lscf = [df_indep['Gc_GDC_LSCF'], df_corr['Gc_GDC_LSCF'], df_ht['Gc_GDC_LSCF']]

bp1 = axes[0].boxplot(data_ysz, labels=['ρ=0.0\n(RT)', 'ρ=0.5\n(RT)', 'High-T\n(800°C)'],
                      patch_artist=True, showmeans=True)
for patch, color in zip(bp1['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
    patch.set_facecolor(color)

axes[0].set_ylabel('$G_{c,YSZ|GDC}$ [J/m²]', fontweight='bold')
axes[0].set_title('YSZ|GDC Interface Sensitivity', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

bp2 = axes[1].boxplot(data_lscf, labels=['ρ=0.0\n(RT)', 'ρ=0.5\n(RT)', 'High-T\n(800°C)'],
                      patch_artist=True, showmeans=True)
for patch, color in zip(bp2['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
    patch.set_facecolor(color)

axes[1].set_ylabel('$G_{c,GDC|LSCF}$ [J/m²]', fontweight='bold')
axes[1].set_title('GDC|LSCF Interface Sensitivity', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Statistical Distribution Comparison Across Scenarios', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('../figures/06_sensitivity_boxplots.png', dpi=300, bbox_inches='tight')
print("  Saved: 06_sensitivity_boxplots.png")
plt.close()

print("\n" + "="*70)
print("All figures generated successfully!")
print("Location: /workspace/figures/")
print("="*70)
