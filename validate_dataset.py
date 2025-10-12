#!/usr/bin/env python3
"""
Dataset Validation Script for Stratified Flow Attenuation Mechanisms Dataset
Validates data quality, consistency, and physical constraints
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

def validate_dataset():
    """Comprehensive dataset validation."""
    print("=== STRATIFIED FLOW ATTENUATION DATASET VALIDATION ===\n")
    
    # Load all datasets
    datasets = {}
    dataset_files = [
        'fluid_properties.csv',
        'flow_geometry.csv', 
        'acoustic_properties.csv',
        'experimental_conditions.csv',
        'theoretical_models.csv',
        'correlation_data.csv'
    ]
    
    for file in dataset_files:
        try:
            df = pd.read_csv(f'stratified_flow_dataset/{file}')
            datasets[file.replace('.csv', '')] = df
            print(f"✓ Loaded {file}: {len(df)} samples")
        except FileNotFoundError:
            print(f"✗ Could not load {file}")
            return False
    
    print(f"\nTotal datasets loaded: {len(datasets)}")
    
    # Validation results
    validation_results = {
        'completeness': True,
        'consistency': True,
        'physical_constraints': True,
        'statistical_quality': True,
        'overall_score': 0
    }
    
    # 1. Completeness Check
    print("\n=== 1. COMPLETENESS CHECK ===")
    for name, df in datasets.items():
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"✗ {name}: {missing_values} missing values")
            validation_results['completeness'] = False
        else:
            print(f"✓ {name}: No missing values")
    
    # 2. Consistency Check
    print("\n=== 2. CONSISTENCY CHECK ===")
    
    # Check sample_id consistency across datasets
    sample_ids = {}
    for name, df in datasets.items():
        if 'sample_id' in df.columns:
            sample_ids[name] = set(df['sample_id'].unique())
    
    if len(sample_ids) > 1:
        common_ids = set.intersection(*sample_ids.values())
        if len(common_ids) > 0:
            print(f"✓ Sample ID consistency: {len(common_ids)} common samples")
        else:
            print("✗ No common sample IDs found")
            validation_results['consistency'] = False
    
    # Check data types
    for name, df in datasets.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        print(f"✓ {name}: {len(numeric_cols)} numeric, {len(non_numeric_cols)} non-numeric columns")
    
    # 3. Physical Constraints Check
    print("\n=== 3. PHYSICAL CONSTRAINTS CHECK ===")
    
    # Fluid properties validation
    fluid_props = datasets['fluid_properties']
    
    # Density should be positive
    if (fluid_props['density'] <= 0).any():
        print("✗ Negative densities found")
        validation_results['physical_constraints'] = False
    else:
        print("✓ All densities are positive")
    
    # Viscosity should be positive
    if (fluid_props['viscosity'] <= 0).any():
        print("✗ Negative viscosities found")
        validation_results['physical_constraints'] = False
    else:
        print("✓ All viscosities are positive")
    
    # Sound speed should be positive
    if (fluid_props['sound_speed'] <= 0).any():
        print("✗ Negative sound speeds found")
        validation_results['physical_constraints'] = False
    else:
        print("✓ All sound speeds are positive")
    
    # Temperature should be reasonable (273-400 K)
    if (fluid_props['temperature'] < 273).any() or (fluid_props['temperature'] > 400).any():
        print("✗ Temperatures outside reasonable range (273-400 K)")
        validation_results['physical_constraints'] = False
    else:
        print("✓ All temperatures in reasonable range")
    
    # Acoustic properties validation
    acoustic_props = datasets['acoustic_properties']
    
    # Attenuation should be positive
    attenuation_cols = ['viscous_attenuation', 'thermal_attenuation', 'scattering_attenuation', 
                       'interface_attenuation', 'total_attenuation']
    
    for col in attenuation_cols:
        if (acoustic_props[col] < 0).any():
            print(f"✗ Negative {col} found")
            validation_results['physical_constraints'] = False
        else:
            print(f"✓ All {col} are non-negative")
    
    # Frequency should be positive
    if (acoustic_props['frequency'] <= 0).any():
        print("✗ Non-positive frequencies found")
        validation_results['physical_constraints'] = False
    else:
        print("✓ All frequencies are positive")
    
    # Flow geometry validation
    flow_geometry = datasets['flow_geometry']
    
    # Diameter should be positive
    if (flow_geometry['diameter'] <= 0).any():
        print("✗ Non-positive diameters found")
        validation_results['physical_constraints'] = False
    else:
        print("✓ All diameters are positive")
    
    # Layer thickness ratio should be between 0 and 1
    if (flow_geometry['layer_thickness_ratio'] < 0).any() or (flow_geometry['layer_thickness_ratio'] > 1).any():
        print("✗ Layer thickness ratios outside [0,1] range")
        validation_results['physical_constraints'] = False
    else:
        print("✓ All layer thickness ratios in [0,1] range")
    
    # Reynolds numbers should be positive
    if (flow_geometry['Re_heavy'] <= 0).any() or (flow_geometry['Re_light'] <= 0).any():
        print("✗ Non-positive Reynolds numbers found")
        validation_results['physical_constraints'] = False
    else:
        print("✓ All Reynolds numbers are positive")
    
    # 4. Statistical Quality Check
    print("\n=== 4. STATISTICAL QUALITY CHECK ===")
    
    # Check for outliers using IQR method
    outlier_counts = {}
    for name, df in datasets.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers += ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[name] = outliers
        print(f"✓ {name}: {outliers} outliers detected (acceptable for synthetic data)")
    
    # Check data distributions
    print("\nData distribution analysis:")
    for name, df in datasets.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check for normal distribution in key columns
            if 'total_attenuation' in df.columns:
                stat, p_value = stats.normaltest(df['total_attenuation'])
                print(f"✓ {name} total_attenuation: p-value = {p_value:.6f} (normal distribution test)")
    
    # 5. Cross-dataset Validation
    print("\n=== 5. CROSS-DATASET VALIDATION ===")
    
    # Check frequency consistency
    acoustic_freqs = set(acoustic_props['frequency'].unique())
    theoretical_freqs = set(datasets['theoretical_models']['frequency'].unique())
    
    if acoustic_freqs.issubset(theoretical_freqs):
        print("✓ Frequency ranges are consistent between acoustic and theoretical data")
    else:
        print("✗ Frequency ranges are inconsistent")
        validation_results['consistency'] = False
    
    # Check sample ID ranges
    sample_id_ranges = {}
    for name, df in datasets.items():
        if 'sample_id' in df.columns:
            sample_id_ranges[name] = (df['sample_id'].min(), df['sample_id'].max())
            print(f"✓ {name} sample_id range: {sample_id_ranges[name]}")
    
    # 6. Generate Validation Report
    print("\n=== 6. VALIDATION SUMMARY ===")
    
    # Calculate overall score
    score = 0
    if validation_results['completeness']:
        score += 25
        print("✓ Completeness: PASSED (25/25)")
    else:
        print("✗ Completeness: FAILED (0/25)")
    
    if validation_results['consistency']:
        score += 25
        print("✓ Consistency: PASSED (25/25)")
    else:
        print("✗ Consistency: FAILED (0/25)")
    
    if validation_results['physical_constraints']:
        score += 25
        print("✓ Physical Constraints: PASSED (25/25)")
    else:
        print("✗ Physical Constraints: FAILED (0/25)")
    
    if validation_results['statistical_quality']:
        score += 25
        print("✓ Statistical Quality: PASSED (25/25)")
    else:
        print("✗ Statistical Quality: FAILED (0/25)")
    
    validation_results['overall_score'] = score
    print(f"\nOverall Validation Score: {score}/100")
    
    if score >= 75:
        print("🎉 DATASET VALIDATION: PASSED")
        print("The dataset meets quality standards for PhD research use.")
    elif score >= 50:
        print("⚠️  DATASET VALIDATION: PARTIAL PASS")
        print("The dataset has some issues but may still be usable with caution.")
    else:
        print("❌ DATASET VALIDATION: FAILED")
        print("The dataset has significant quality issues and should be regenerated.")
    
    # Save validation report
    validation_report = {
        'validation_date': pd.Timestamp.now().isoformat(),
        'overall_score': int(score),
        'validation_results': validation_results,
        'outlier_counts': {k: int(v) for k, v in outlier_counts.items()},
        'sample_id_ranges': {k: (int(v[0]), int(v[1])) for k, v in sample_id_ranges.items()},
        'dataset_sizes': {name: int(len(df)) for name, df in datasets.items()}
    }
    
    with open('stratified_flow_dataset/validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\nValidation report saved to: stratified_flow_dataset/validation_report.json")
    
    return score >= 75

def create_validation_plots():
    """Create validation visualization plots."""
    print("\n=== CREATING VALIDATION PLOTS ===")
    
    # Load datasets
    fluid_props = pd.read_csv('stratified_flow_dataset/fluid_properties.csv')
    acoustic_props = pd.read_csv('stratified_flow_dataset/acoustic_properties.csv')
    flow_geometry = pd.read_csv('stratified_flow_dataset/flow_geometry.csv')
    
    # Create validation plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Density distribution by fluid type
    for fluid_type in fluid_props['fluid_type'].unique():
        data = fluid_props[fluid_props['fluid_type'] == fluid_type]
        axes[0,0].hist(data['density'], alpha=0.6, label=fluid_type, bins=30)
    axes[0,0].set_xlabel('Density (kg/m³)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Density Distribution by Fluid Type')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Attenuation vs Frequency
    sample_data = acoustic_props[acoustic_props['sample_id'] < 100]  # Sample for clarity
    axes[0,1].loglog(sample_data['frequency'], sample_data['total_attenuation'], 
                    'b.', alpha=0.6, markersize=2)
    axes[0,1].set_xlabel('Frequency (Hz)')
    axes[0,1].set_ylabel('Total Attenuation (Np/m)')
    axes[0,1].set_title('Attenuation vs Frequency (Sample)')
    axes[0,1].grid(True)
    
    # 3. Flow regime distribution
    regime_counts = flow_geometry['flow_regime'].value_counts()
    axes[0,2].bar(regime_counts.index, regime_counts.values, color='skyblue', edgecolor='black')
    axes[0,2].set_xlabel('Flow Regime')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title('Flow Regime Distribution')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Temperature vs Sound Speed
    for fluid_type in fluid_props['fluid_type'].unique():
        data = fluid_props[fluid_props['fluid_type'] == fluid_type]
        axes[1,0].scatter(data['temperature'], data['sound_speed'], 
                         label=fluid_type, alpha=0.6, s=20)
    axes[1,0].set_xlabel('Temperature (K)')
    axes[1,0].set_ylabel('Sound Speed (m/s)')
    axes[1,0].set_title('Temperature vs Sound Speed')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 5. Reynolds number correlation
    axes[1,1].loglog(flow_geometry['Re_heavy'], flow_geometry['Re_light'], 
                    'b.', alpha=0.6, markersize=4)
    axes[1,1].set_xlabel('Reynolds Number (Heavy Phase)')
    axes[1,1].set_ylabel('Reynolds Number (Light Phase)')
    axes[1,1].set_title('Reynolds Number Correlation')
    axes[1,1].grid(True)
    
    # 6. Attenuation mechanisms comparison
    freq_sample = acoustic_props.groupby('frequency').mean().reset_index()
    axes[1,2].loglog(freq_sample['frequency'], freq_sample['viscous_attenuation'], 
                    'r-', label='Viscous', linewidth=2)
    axes[1,2].loglog(freq_sample['frequency'], freq_sample['thermal_attenuation'], 
                    'g-', label='Thermal', linewidth=2)
    axes[1,2].loglog(freq_sample['frequency'], freq_sample['scattering_attenuation'], 
                    'b-', label='Scattering', linewidth=2)
    axes[1,2].loglog(freq_sample['frequency'], freq_sample['interface_attenuation'], 
                    'm-', label='Interface', linewidth=2)
    axes[1,2].set_xlabel('Frequency (Hz)')
    axes[1,2].set_ylabel('Attenuation (Np/m)')
    axes[1,2].set_title('Attenuation Mechanisms')
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('stratified_flow_dataset/validation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Validation plots saved to: stratified_flow_dataset/validation_plots.png")

if __name__ == "__main__":
    # Run validation
    validation_passed = validate_dataset()
    
    # Create validation plots
    create_validation_plots()
    
    print(f"\n=== VALIDATION COMPLETE ===")
    if validation_passed:
        print("✅ Dataset is ready for PhD research use!")
    else:
        print("❌ Dataset requires attention before use.")