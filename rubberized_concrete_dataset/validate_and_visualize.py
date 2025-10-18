#!/usr/bin/env python3
"""
Data Validation and Visualization Script for Rubberized Concrete Dataset

This script validates the dataset integrity and generates summary visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load all datasets
print("Loading datasets...")
ambient = pd.read_csv("rubberized_concrete_dataset/01_ambient_condition_tests.csv")
residual = pd.read_csv("rubberized_concrete_dataset/02_residual_properties_post_heat.csv")
insitu = pd.read_csv("rubberized_concrete_dataset/03_insitu_hot_strength.csv")
thermal_exp = pd.read_csv("rubberized_concrete_dataset/04_thermal_expansion_dilatometry.csv")
transient = pd.read_csv("rubberized_concrete_dataset/05_transient_thermal_strain_loaded.csv")
spalling = pd.read_csv("rubberized_concrete_dataset/06_spalling_and_pore_pressure.csv")
stress_strain = pd.read_csv("rubberized_concrete_dataset/07_stress_strain_curves.csv")

print(f"✓ Ambient tests: {len(ambient)} specimens")
print(f"✓ Residual properties: {len(residual)} specimens")
print(f"✓ In-situ hot tests: {len(insitu)} specimens")
print(f"✓ Thermal expansion: {len(thermal_exp)} measurements")
print(f"✓ Transient strain: {len(transient)} measurements")
print(f"✓ Spalling data: {len(spalling)} specimens")
print(f"✓ Stress-strain curves: {len(stress_strain)} points")

# Data validation checks
print("\n=== DATA VALIDATION ===")

# Check for missing values
print("\nMissing values check:")
for name, df in [("ambient", ambient), ("residual", residual), ("insitu", insitu)]:
    missing = df.isnull().sum().sum()
    print(f"  {name}: {missing} missing values")

# Check data ranges
print("\nData range validation:")
print(f"  Compressive strength: {ambient['compressive_strength_MPa'].min():.1f} - {ambient['compressive_strength_MPa'].max():.1f} MPa")
print(f"  Residual strength retention: {residual['strength_retention_percent'].min():.1f} - {residual['strength_retention_percent'].max():.1f} %")
print(f"  Temperature range: {residual['target_temperature_C'].min()} - {residual['target_temperature_C'].max()} °C")

# Statistical checks
print("\nStatistical consistency:")
for mix in ambient['mix_design'].unique():
    mix_data = ambient[(ambient['mix_design'] == mix) & (ambient['curing_age_days'] == 28)]
    if len(mix_data) > 0:
        mean_strength = mix_data['compressive_strength_MPa'].mean()
        cov = (mix_data['compressive_strength_MPa'].std() / mean_strength) * 100
        print(f"  {mix} @ 28d: {mean_strength:.1f} MPa (CoV: {cov:.1f}%)")

# Generate key plots
print("\n=== GENERATING VISUALIZATIONS ===")

# Create output directory for plots
os.makedirs("rubberized_concrete_dataset/plots", exist_ok=True)

# Plot 1: Ambient strength vs rubber content
plt.figure(figsize=(10, 6))
ambient_28 = ambient[ambient['curing_age_days'] == 28]
for mix in ambient_28['mix_design'].unique():
    mix_data = ambient_28[ambient_28['mix_design'] == mix]
    plt.scatter(mix_data['rubber_content_pct'], 
               mix_data['compressive_strength_MPa'],
               label=mix, s=100, alpha=0.7)
plt.xlabel('Rubber Content (%)', fontsize=12)
plt.ylabel('Compressive Strength (MPa)', fontsize=12)
plt.title('Effect of Rubber Content on Ambient Strength (28 days)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rubberized_concrete_dataset/plots/01_rubber_content_vs_strength.png', dpi=300)
print("✓ Generated: rubber content vs strength")

# Plot 2: Residual strength retention
plt.figure(figsize=(12, 6))
for cooling in ['furnace', 'water_quench']:
    cooling_data = residual[residual['cooling_method'] == cooling]
    temp_means = cooling_data.groupby('target_temperature_C')['strength_retention_percent'].mean()
    temp_std = cooling_data.groupby('target_temperature_C')['strength_retention_percent'].std()
    plt.errorbar(temp_means.index, temp_means.values, yerr=temp_std.values,
                marker='o', markersize=8, linewidth=2, capsize=5,
                label=f'{cooling.replace("_", " ").title()}')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Strength Retention (%)', fontsize=12)
plt.title('Residual Strength Retention vs Temperature', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rubberized_concrete_dataset/plots/02_strength_retention.png', dpi=300)
print("✓ Generated: strength retention plot")

# Plot 3: Thermal expansion curves
plt.figure(figsize=(12, 6))
for mix in ['RC-0', 'RC-20', 'RC-20-SF']:
    mix_data = thermal_exp[(thermal_exp['mix_design'] == mix) & 
                          (thermal_exp['specimen_number'] == 1)]
    plt.plot(mix_data['temperature_C'], mix_data['thermal_strain_microstrain'],
            linewidth=2, label=mix, alpha=0.8)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Thermal Strain (με)', fontsize=12)
plt.title('Thermal Expansion Behavior', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rubberized_concrete_dataset/plots/03_thermal_expansion.png', dpi=300)
print("✓ Generated: thermal expansion curves")

# Plot 4: Stress-strain curves
plt.figure(figsize=(12, 8))
for temp in [23, 400, 600]:
    temp_data = stress_strain[(stress_strain['mix_design'] == 'RC-20') & 
                             (stress_strain['temperature_C'] == temp)]
    plt.plot(temp_data['strain_percent'] * 100, temp_data['stress_MPa'],
            linewidth=2, label=f'{temp}°C', alpha=0.8)
plt.xlabel('Strain (%)', fontsize=12)
plt.ylabel('Stress (MPa)', fontsize=12)
plt.title('Stress-Strain Curves for RC-20 at Different Temperatures', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rubberized_concrete_dataset/plots/04_stress_strain_curves.png', dpi=300)
print("✓ Generated: stress-strain curves")

# Plot 5: Spalling behavior
plt.figure(figsize=(12, 6))
spalling_summary = spalling.groupby(['target_temperature_C', 'cooling_method'])['spalling_area_percent'].mean().reset_index()
for cooling in ['furnace', 'water_quench']:
    cooling_data = spalling_summary[spalling_summary['cooling_method'] == cooling]
    plt.bar(cooling_data['target_temperature_C'] + (10 if cooling == 'water_quench' else -10),
           cooling_data['spalling_area_percent'],
           width=15, label=cooling.replace('_', ' ').title(), alpha=0.8)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Average Spalling Area (%)', fontsize=12)
plt.title('Spalling Behavior vs Temperature and Cooling Method', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('rubberized_concrete_dataset/plots/05_spalling_behavior.png', dpi=300)
print("✓ Generated: spalling behavior plot")

print("\n=== VALIDATION COMPLETE ===")
print("All plots saved to: rubberized_concrete_dataset/plots/")
