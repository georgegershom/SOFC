"""
Sample Analysis Script for Context Dataset
Demonstrates how to load, explore, and prepare the dataset for ML modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_dataset(size='medium'):
    """Load dataset and metadata"""
    print(f"Loading {size} dataset...")
    
    df = pd.read_csv(f'context_dataset_{size}.csv')
    
    with open(f'context_dataset_{size}_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
    return df, metadata


def explore_basic_statistics(df):
    """Print basic dataset statistics"""
    print("\n" + "="*80)
    print("BASIC DATASET STATISTICS")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values!")
    else:
        print(missing[missing > 0])
    
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    print("\nAtmosphere Distribution:")
    print(df['atmosphere_type'].value_counts())
    
    print("\nQuality Flags:")
    flag_cols = [col for col in df.columns if col.startswith('flag_')]
    for flag in flag_cols:
        count = df[flag].sum()
        pct = count / len(df) * 100
        print(f"  {flag}: {count} samples ({pct:.1f}%)")


def analyze_cte_mismatch(df):
    """Analyze CTE mismatch distributions"""
    print("\n" + "="*80)
    print("CTE MISMATCH ANALYSIS")
    print("="*80)
    
    cte_cols = [col for col in df.columns if 'CTE_mismatch' in col]
    
    print("\nCTE Mismatch Statistics (ppm/K):")
    print(df[cte_cols].describe())
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of average CTE mismatch
    axes[0, 0].hist(df['avg_CTE_mismatch_magnitude'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Average CTE Mismatch Magnitude (ppm/K)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of CTE Mismatch')
    axes[0, 0].axvline(3.5, color='red', linestyle='--', label='Extreme threshold')
    axes[0, 0].legend()
    
    # Anode-Electrolyte vs Cathode-Electrolyte
    axes[0, 1].scatter(
        df['CTE_mismatch_anode_electrolyte_25C'],
        df['CTE_mismatch_cathode_electrolyte_25C'],
        alpha=0.3, s=10
    )
    axes[0, 1].set_xlabel('Anode-Electrolyte CTE Mismatch (ppm/K)')
    axes[0, 1].set_ylabel('Cathode-Electrolyte CTE Mismatch (ppm/K)')
    axes[0, 1].set_title('CTE Mismatch Relationships')
    axes[0, 1].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 1].axvline(0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # CTE vs Temperature for each layer
    layers = ['anode', 'electrolyte', 'cathode']
    colors = ['red', 'blue', 'green']
    
    for layer, color in zip(layers, colors):
        cte_25 = df[f'{layer}_CTE_25C_ppm_K'].values
        cte_1000 = df[f'{layer}_CTE_1000C_ppm_K'].values
        
        axes[1, 0].scatter([25]*len(cte_25), cte_25, alpha=0.2, s=5, 
                          color=color, label=f'{layer} @ 25°C')
        axes[1, 0].scatter([1000]*len(cte_1000), cte_1000, alpha=0.2, s=5, 
                          color=color, label=f'{layer} @ 1000°C')
    
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('CTE (ppm/K)')
    axes[1, 0].set_title('Temperature-Dependent CTE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot by atmosphere type
    df_melted = pd.DataFrame({
        'Atmosphere': df['atmosphere_type'],
        'CTE_Mismatch': df['avg_CTE_mismatch_magnitude']
    })
    
    df_melted.boxplot(by='Atmosphere', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Atmosphere Type')
    axes[1, 1].set_ylabel('Average CTE Mismatch (ppm/K)')
    axes[1, 1].set_title('CTE Mismatch by Atmosphere')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('analysis_cte_mismatch.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: analysis_cte_mismatch.png")
    plt.close()


def analyze_material_properties(df):
    """Analyze material property distributions"""
    print("\n" + "="*80)
    print("MATERIAL PROPERTY ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    layers = ['anode', 'electrolyte', 'cathode']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # Young's modulus at 25°C
    for layer, color in zip(layers, colors):
        axes[0, 0].hist(df[f'{layer}_youngs_modulus_25C_GPa'], 
                       bins=30, alpha=0.5, label=layer.capitalize(), 
                       color=color, edgecolor='black')
    axes[0, 0].set_xlabel("Young's Modulus @ 25°C (GPa)")
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title("Young's Modulus Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Young's modulus temperature dependence
    for layer, color in zip(layers, colors):
        E_25 = df[f'{layer}_youngs_modulus_25C_GPa'].values
        E_1000 = df[f'{layer}_youngs_modulus_1000C_GPa'].values
        
        axes[0, 1].scatter(E_25, E_1000, alpha=0.3, s=10, 
                          color=color, label=layer.capitalize())
    
    axes[0, 1].set_xlabel("Young's Modulus @ 25°C (GPa)")
    axes[0, 1].set_ylabel("Young's Modulus @ 1000°C (GPa)")
    axes[0, 1].set_title('Temperature Dependence of Stiffness')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Stiffness ratios
    axes[0, 2].hist(df['stiffness_ratio_electrolyte_anode'], 
                   bins=40, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 2].set_xlabel('Electrolyte/Anode Stiffness Ratio')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Stiffness Ratio Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Sintering onset temperatures
    for layer, color in zip(layers, colors):
        axes[1, 0].hist(df[f'{layer}_sintering_onset_C'], 
                       bins=30, alpha=0.5, label=layer.capitalize(), 
                       color=color, edgecolor='black')
    axes[1, 0].set_xlabel('Sintering Onset Temperature (°C)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Sintering Onset Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Creep parameters
    for layer, color in zip(layers, colors):
        axes[1, 1].scatter(
            df[f'{layer}_creep_activation_energy_kJ_mol'],
            df[f'{layer}_creep_stress_exponent'],
            alpha=0.3, s=10, color=color, label=layer.capitalize()
        )
    axes[1, 1].set_xlabel('Creep Activation Energy (kJ/mol)')
    axes[1, 1].set_ylabel('Creep Stress Exponent')
    axes[1, 1].set_title('Creep Parameter Space')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Total shrinkage
    for layer, color in zip(layers, colors):
        axes[1, 2].hist(df[f'{layer}_total_shrinkage_fraction'] * 100, 
                       bins=30, alpha=0.5, label=layer.capitalize(), 
                       color=color, edgecolor='black')
    axes[1, 2].set_xlabel('Total Shrinkage (%)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Sintering Shrinkage Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_material_properties.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: analysis_material_properties.png")
    plt.close()


def analyze_process_parameters(df):
    """Analyze sintering process parameters"""
    print("\n" + "="*80)
    print("PROCESS PARAMETER ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Peak temperature distribution
    axes[0, 0].hist(df['sintering_peak_temp_C'], bins=40, 
                   edgecolor='black', alpha=0.7, color='orangered')
    axes[0, 0].set_xlabel('Peak Sintering Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Peak Temperature Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Heating vs cooling rates
    axes[0, 1].scatter(df['heating_ramp_rate_C_per_min'],
                      df['cooling_ramp_rate_C_per_min'],
                      alpha=0.3, s=10)
    axes[0, 1].plot([0, 10], [0, 10], 'r--', label='Equal rates')
    axes[0, 1].set_xlabel('Heating Rate (°C/min)')
    axes[0, 1].set_ylabel('Cooling Rate (°C/min)')
    axes[0, 1].set_title('Thermal Ramp Rate Relationships')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hold time distribution
    axes[0, 2].hist(df['sintering_hold_time_min'], bins=40, 
                   edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 2].set_xlabel('Peak Hold Time (min)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Hold Time Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Total process time
    axes[1, 0].hist(df['total_process_time_min'] / 60, bins=40, 
                   edgecolor='black', alpha=0.7, color='darkgreen')
    axes[1, 0].set_xlabel('Total Process Time (hours)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Total Sintering Cycle Duration')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Peak temp vs process time
    axes[1, 1].scatter(df['sintering_peak_temp_C'],
                      df['total_process_time_min'] / 60,
                      c=df['cooling_ramp_rate_C_per_min'],
                      alpha=0.5, s=20, cmap='coolwarm')
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Cooling Rate (°C/min)')
    axes[1, 1].set_xlabel('Peak Temperature (°C)')
    axes[1, 1].set_ylabel('Total Process Time (hours)')
    axes[1, 1].set_title('Process Complexity Map')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Oxygen partial pressure
    axes[1, 2].hist(df['oxygen_partial_pressure_atm'], bins=40, 
                   edgecolor='black', alpha=0.7, color='coral')
    axes[1, 2].set_xlabel('Oxygen Partial Pressure (atm)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Atmosphere Composition')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_process_parameters.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: analysis_process_parameters.png")
    plt.close()


def analyze_geometric_parameters(df):
    """Analyze geometric design space"""
    print("\n" + "="*80)
    print("GEOMETRIC PARAMETER ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Layer thicknesses
    layers = ['anode', 'electrolyte', 'cathode']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    thickness_data = [df[f'{layer}_thickness_um'].values for layer in layers]
    
    bp = axes[0, 0].boxplot(thickness_data, labels=[l.capitalize() for l in layers],
                            patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[0, 0].set_ylabel('Thickness (μm)')
    axes[0, 0].set_title('Layer Thickness Distributions')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Thickness ratios
    axes[0, 1].scatter(df['thickness_ratio_anode_electrolyte'],
                      df['thickness_ratio_cathode_electrolyte'],
                      alpha=0.3, s=10)
    axes[0, 1].set_xlabel('Anode/Electrolyte Thickness Ratio')
    axes[0, 1].set_ylabel('Cathode/Electrolyte Thickness Ratio')
    axes[0, 1].set_title('Thickness Ratio Design Space')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total thickness distribution
    axes[1, 0].hist(df['total_sintered_thickness_um'], bins=50, 
                   edgecolor='black', alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Total Sintered Thickness (μm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Total Structure Thickness')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plate dimensions
    axes[1, 1].scatter(df['plate_length_mm'], df['plate_width_mm'],
                      c=df['total_sintered_thickness_um'],
                      alpha=0.5, s=20, cmap='viridis')
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Total Thickness (μm)')
    axes[1, 1].set_xlabel('Plate Length (mm)')
    axes[1, 1].set_ylabel('Plate Width (mm)')
    axes[1, 1].set_title('Plate Geometry Design Space')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('analysis_geometric_parameters.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: analysis_geometric_parameters.png")
    plt.close()


def analyze_temperature_profiles(df, n_samples=10):
    """Visualize sample temperature profiles"""
    print("\n" + "="*80)
    print("TEMPERATURE PROFILE ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot sample profiles
    for i in range(min(n_samples, len(df))):
        profile = json.loads(df.iloc[i]['temperature_profile_json'])
        axes[0].plot(profile['time_points_min'], profile['temperature_C'], 
                    alpha=0.6, linewidth=1.5)
    
    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title(f'Sample Temperature Profiles (n={n_samples})')
    axes[0].grid(True, alpha=0.3)
    
    # Cooling rate vs peak temperature
    axes[1].scatter(df['sintering_peak_temp_C'],
                   df['cooling_ramp_rate_C_per_min'],
                   c=df['heating_ramp_rate_C_per_min'],
                   alpha=0.5, s=30, cmap='plasma')
    cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cbar.set_label('Heating Rate (°C/min)')
    axes[1].set_xlabel('Peak Temperature (°C)')
    axes[1].set_ylabel('Cooling Rate (°C/min)')
    axes[1].set_title('Thermal Cycle Parameter Space')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_temperature_profiles.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: analysis_temperature_profiles.png")
    plt.close()


def create_correlation_heatmap(df):
    """Create correlation heatmap for key parameters"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Select key numeric features
    key_features = [
        'avg_CTE_mismatch_magnitude',
        'stiffness_ratio_electrolyte_anode',
        'thickness_ratio_anode_electrolyte',
        'sintering_peak_temp_C',
        'cooling_ramp_rate_C_per_min',
        'total_process_time_min',
        'total_sintered_thickness_um',
        'electrolyte_thickness_um',
        'green_density_fraction'
    ]
    
    corr_matrix = df[key_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix - Key Parameters', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('analysis_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: analysis_correlation_heatmap.png")
    plt.close()


def prepare_for_ml(df):
    """Prepare dataset for machine learning"""
    print("\n" + "="*80)
    print("PREPARING DATASET FOR ML")
    print("="*80)
    
    # Remove non-feature columns
    exclude_cols = ['sample_id', 'generation_timestamp', 'temperature_profile_json']
    
    # Separate features by type
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove excluded columns
    numeric_features = [f for f in numeric_features if f not in exclude_cols]
    categorical_features = [f for f in categorical_features if f not in exclude_cols]
    
    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # One-hot encode categorical
    X = df[numeric_features + categorical_features].copy()
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Total features after encoding: {X.shape[1]}")
    
    # Save feature names
    feature_names = X.columns.tolist()
    with open('ml_feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(name + '\n')
    
    print("\n✓ Saved: ml_feature_names.txt")
    
    return X, feature_names


def generate_summary_report(df, metadata):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    with open('dataset_analysis_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONTEXT DATASET - ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generation Date: {metadata['generation_date']}\n")
        f.write(f"Number of Samples: {len(df)}\n")
        f.write(f"Number of Features: {len(df.columns)}\n")
        f.write(f"Sampling Method: {metadata['sampling_method']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("KEY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("CTE Mismatch:\n")
        f.write(f"  Range: [{df['avg_CTE_mismatch_magnitude'].min():.3f}, "
                f"{df['avg_CTE_mismatch_magnitude'].max():.3f}] ppm/K\n")
        f.write(f"  Mean: {df['avg_CTE_mismatch_magnitude'].mean():.3f} ppm/K\n")
        f.write(f"  Std: {df['avg_CTE_mismatch_magnitude'].std():.3f} ppm/K\n\n")
        
        f.write("Sintering Temperature:\n")
        f.write(f"  Range: [{df['sintering_peak_temp_C'].min():.1f}, "
                f"{df['sintering_peak_temp_C'].max():.1f}] °C\n")
        f.write(f"  Mean: {df['sintering_peak_temp_C'].mean():.1f} °C\n\n")
        
        f.write("Total Thickness:\n")
        f.write(f"  Range: [{df['total_sintered_thickness_um'].min():.1f}, "
                f"{df['total_sintered_thickness_um'].max():.1f}] μm\n")
        f.write(f"  Mean: {df['total_sintered_thickness_um'].mean():.1f} μm\n\n")
        
        f.write("Atmosphere Distribution:\n")
        for atm, count in df['atmosphere_type'].value_counts().items():
            f.write(f"  {atm}: {count} ({count/len(df)*100:.1f}%)\n")
        
        f.write("\nQuality Flags:\n")
        flag_cols = [col for col in df.columns if col.startswith('flag_')]
        for flag in flag_cols:
            count = df[flag].sum()
            pct = count / len(df) * 100
            f.write(f"  {flag}: {count} samples ({pct:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("GENERATED VISUALIZATIONS\n")
        f.write("="*80 + "\n\n")
        f.write("1. analysis_cte_mismatch.png\n")
        f.write("2. analysis_material_properties.png\n")
        f.write("3. analysis_process_parameters.png\n")
        f.write("4. analysis_geometric_parameters.png\n")
        f.write("5. analysis_temperature_profiles.png\n")
        f.write("6. analysis_correlation_heatmap.png\n")
    
    print("\n✓ Saved: dataset_analysis_report.txt")


def main():
    """Main analysis workflow"""
    print("="*80)
    print("CONTEXT DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Load dataset (change size as needed)
    df, metadata = load_dataset(size='medium')
    
    # Run analyses
    explore_basic_statistics(df)
    analyze_cte_mismatch(df)
    analyze_material_properties(df)
    analyze_process_parameters(df)
    analyze_geometric_parameters(df)
    analyze_temperature_profiles(df, n_samples=20)
    create_correlation_heatmap(df)
    
    # Prepare for ML
    X, feature_names = prepare_for_ml(df)
    
    # Generate report
    generate_summary_report(df, metadata)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - dataset_analysis_report.txt")
    print("  - ml_feature_names.txt")
    print("  - analysis_cte_mismatch.png")
    print("  - analysis_material_properties.png")
    print("  - analysis_process_parameters.png")
    print("  - analysis_geometric_parameters.png")
    print("  - analysis_temperature_profiles.png")
    print("  - analysis_correlation_heatmap.png")
    print("\n✓ Ready for machine learning model training!")


if __name__ == "__main__":
    main()
