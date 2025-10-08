#!/usr/bin/env python3
"""
Data Validation Script for Sintering Microstructure Dataset
===========================================================

This script performs comprehensive validation of the generated dataset
to ensure physical realism and data quality.

Author: AI Assistant
Date: October 8, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def validate_dataset():
    """Perform comprehensive dataset validation."""
    
    print("Loading dataset for validation...")
    df = pd.read_csv('/workspace/datasets/sintering_microstructure_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print("\nPerforming validation checks...\n")
    
    validation_results = {}
    
    # 1. Check for missing values
    missing_values = df.isnull().sum().sum()
    validation_results['missing_values'] = missing_values
    print(f"✓ Missing values: {missing_values} (should be 0)")
    
    # 2. Check physical constraints
    print("\n--- Physical Constraint Checks ---")
    
    # Density constraints
    density_valid = ((df['final_relative_density_percent'] >= df['initial_relative_density_percent']) & 
                    (df['final_relative_density_percent'] <= 100)).all()
    validation_results['density_physical'] = density_valid
    print(f"✓ Final density ≥ Initial density and ≤ 100%: {density_valid}")
    
    # Porosity = 100 - density
    porosity_check = np.allclose(df['porosity_percent'], 
                                100 - df['final_relative_density_percent'], 
                                rtol=1e-10)
    validation_results['porosity_consistency'] = porosity_check
    print(f"✓ Porosity = 100 - Density: {porosity_check}")
    
    # Grain size should be >= particle size (generally)
    grain_size_valid = (df['grain_size_mean_um'] >= df['particle_size_um'] * 0.8).sum() / len(df)
    validation_results['grain_size_growth'] = grain_size_valid
    print(f"✓ Grain size ≥ 80% of particle size: {grain_size_valid:.1%} of samples")
    
    # Temperature ranges
    temp_valid = ((df['peak_temperature_C'] >= 1200) & 
                 (df['peak_temperature_C'] <= 1600)).all()
    validation_results['temperature_range'] = temp_valid
    print(f"✓ Temperature in range 1200-1600°C: {temp_valid}")
    
    # 3. Check correlations make physical sense
    print("\n--- Correlation Validation ---")
    
    # Temperature should positively correlate with density
    temp_density_corr = df['peak_temperature_C'].corr(df['final_relative_density_percent'])
    validation_results['temp_density_corr'] = temp_density_corr
    print(f"✓ Temperature-Density correlation: {temp_density_corr:.3f} (should be > 0.3)")
    
    # Temperature should positively correlate with grain size
    temp_grain_corr = df['peak_temperature_C'].corr(df['grain_size_mean_um'])
    validation_results['temp_grain_corr'] = temp_grain_corr
    print(f"✓ Temperature-Grain Size correlation: {temp_grain_corr:.3f} (should be > 0.3)")
    
    # Pressure should positively correlate with density
    pressure_density_corr = df['applied_pressure_MPa'].corr(df['final_relative_density_percent'])
    validation_results['pressure_density_corr'] = pressure_density_corr
    print(f"✓ Pressure-Density correlation: {pressure_density_corr:.3f} (should be > 0.2)")
    
    # 4. Check distributions
    print("\n--- Distribution Checks ---")
    
    # Check for reasonable standard deviations
    grain_size_cv = df['grain_size_std_um'].mean() / df['grain_size_mean_um'].mean()
    validation_results['grain_size_cv'] = grain_size_cv
    print(f"✓ Grain size coefficient of variation: {grain_size_cv:.3f} (should be 0.2-0.6)")
    
    # Check atmosphere distribution
    atm_dist = df['atmosphere'].value_counts(normalize=True)
    validation_results['atmosphere_dist'] = atm_dist
    print(f"✓ Atmosphere distribution:")
    for atm, frac in atm_dist.items():
        print(f"    {atm}: {frac:.1%}")
    
    # 5. Check experimental realism
    print("\n--- Experimental Realism ---")
    
    # SEM resolution should decrease with magnification
    sem_resolution_check = df.groupby('sem_magnification')['sem_resolution_nm'].mean()
    resolution_trend = sem_resolution_check.diff().dropna()
    resolution_decreasing = (resolution_trend < 0).all()
    validation_results['sem_resolution_trend'] = resolution_decreasing
    print(f"✓ SEM resolution decreases with magnification: {resolution_decreasing}")
    
    # Date range check
    date_range = pd.to_datetime(df['experiment_date']).max() - pd.to_datetime(df['experiment_date']).min()
    validation_results['date_range_days'] = date_range.days
    print(f"✓ Experiment date range: {date_range.days} days")
    
    # 6. Statistical checks
    print("\n--- Statistical Validation ---")
    
    # Check for outliers (values beyond 3 standard deviations)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    
    for col in numerical_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = (z_scores > 3).sum()
        outlier_counts[col] = outliers
    
    total_outliers = sum(outlier_counts.values())
    validation_results['total_outliers'] = total_outliers
    print(f"✓ Total statistical outliers (>3σ): {total_outliers}")
    
    # Columns with most outliers
    top_outlier_cols = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    print("   Top outlier columns:")
    for col, count in top_outlier_cols:
        print(f"    {col}: {count}")
    
    # 7. Generate validation summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    # Count passed checks
    passed_checks = 0
    total_checks = 0
    
    critical_checks = [
        ('No missing values', validation_results['missing_values'] == 0),
        ('Physical density constraints', validation_results['density_physical']),
        ('Porosity consistency', validation_results['porosity_consistency']),
        ('Temperature range valid', validation_results['temperature_range']),
        ('Temp-density correlation', validation_results['temp_density_corr'] > 0.3),
        ('Temp-grain correlation', validation_results['temp_grain_corr'] > 0.3),
        ('Pressure-density correlation', validation_results['pressure_density_corr'] > 0.2)
    ]
    
    for check_name, passed in critical_checks:
        status = "PASS" if passed else "FAIL"
        print(f"{check_name}: {status}")
        if passed:
            passed_checks += 1
        total_checks += 1
    
    print(f"\nOverall validation: {passed_checks}/{total_checks} critical checks passed")
    
    if passed_checks == total_checks:
        print("🎉 Dataset validation SUCCESSFUL! Data is ready for use.")
    else:
        print("⚠️  Some validation checks failed. Review data generation logic.")
    
    return validation_results

def create_validation_plots():
    """Create validation plots."""
    
    df = pd.read_csv('/workspace/datasets/sintering_microstructure_dataset.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Density vs Temperature (should show positive trend)
    axes[0,0].scatter(df['peak_temperature_C'], df['final_relative_density_percent'], 
                     alpha=0.6, s=30)
    axes[0,0].set_xlabel('Peak Temperature (°C)')
    axes[0,0].set_ylabel('Final Density (%)')
    axes[0,0].set_title('Validation: Temperature vs Density')
    
    # Add trend line
    z = np.polyfit(df['peak_temperature_C'], df['final_relative_density_percent'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['peak_temperature_C'], p(df['peak_temperature_C']), 
                  "r--", alpha=0.8, label=f'Trend (r={df["peak_temperature_C"].corr(df["final_relative_density_percent"]):.3f})')
    axes[0,0].legend()
    
    # 2. Grain size vs particle size (grain should be larger)
    axes[0,1].scatter(df['particle_size_um'], df['grain_size_mean_um'], 
                     alpha=0.6, s=30, c='orange')
    axes[0,1].plot([0, 6], [0, 6], 'k--', alpha=0.5, label='1:1 line')
    axes[0,1].set_xlabel('Particle Size (μm)')
    axes[0,1].set_ylabel('Grain Size (μm)')
    axes[0,1].set_title('Validation: Particle vs Grain Size')
    axes[0,1].legend()
    
    # 3. Porosity vs Density (should be perfectly anti-correlated)
    axes[1,0].scatter(df['final_relative_density_percent'], df['porosity_percent'], 
                     alpha=0.6, s=30, c='green')
    axes[1,0].plot([70, 100], [30, 0], 'r--', alpha=0.8, label='Perfect anti-correlation')
    axes[1,0].set_xlabel('Final Density (%)')
    axes[1,0].set_ylabel('Porosity (%)')
    axes[1,0].set_title('Validation: Density vs Porosity')
    axes[1,0].legend()
    
    # 4. Distribution of key variables
    axes[1,1].hist(df['final_relative_density_percent'], bins=30, alpha=0.7, 
                  label='Final Density', density=True)
    axes[1,1].hist(df['grain_size_mean_um'], bins=30, alpha=0.7, 
                  label='Grain Size', density=True)
    axes[1,1].set_xlabel('Value')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Validation: Key Variable Distributions')
    axes[1,1].legend()
    
    plt.suptitle('Dataset Validation Plots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save validation plots
    plt.savefig('/workspace/analysis_plots/validation_plots.png', dpi=300, bbox_inches='tight')
    print("Validation plots saved: /workspace/analysis_plots/validation_plots.png")
    
    return fig

if __name__ == "__main__":
    # Run validation
    validation_results = validate_dataset()
    
    # Create validation plots
    validation_fig = create_validation_plots()
    
    print(f"\nValidation complete. Results saved to validation plots.")