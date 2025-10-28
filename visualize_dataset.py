"""
Dataset Visualization and Analysis
Generates plots and analysis for the ML training dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_datasets():
    """Load training and validation datasets"""
    data_dir = Path('/workspace/ml_training_data')
    
    train_df = pd.read_csv(data_dir / 'training_dataset.csv')
    val_df = pd.read_csv(data_dir / 'validation_dataset.csv')
    
    return train_df, val_df


def plot_input_distributions(df, output_path):
    """Plot distributions of input parameters"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Input Parameter Distributions', fontsize=16, fontweight='bold')
    
    params = [
        ('sintering_temperature_C', 'Sintering Temperature (°C)'),
        ('cooling_rate_C_per_min', 'Cooling Rate (°C/min)'),
        ('porosity_percent', 'Porosity (%)'),
        ('youngs_modulus_Pa', "Young's Modulus (Pa)"),
        ('density_g_cm3', 'Density (g/cm³)'),
        ('thermal_conductivity_W_mK', 'Thermal Conductivity (W/m·K)'),
        ('grain_size_um', 'Grain Size (μm)'),
        ('TEC_mismatch_K-1', 'TEC Mismatch (K⁻¹)')
    ]
    
    for idx, (col, label) in enumerate(params):
        ax = axes[idx // 4, idx % 4]
        ax.hist(df[col], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.text(0.95, 0.95, f'μ={mean_val:.2e}\nσ={std_val:.2e}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved input distributions to {output_path}")
    plt.close()


def plot_output_distributions(df, output_path):
    """Plot distributions of output labels"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Output Label Distributions', fontsize=16, fontweight='bold')
    
    outputs = [
        ('thermal_stress_Pa', 'Thermal Stress (Pa)', 'viridis'),
        ('thermal_strain', 'Thermal Strain', 'plasma'),
        ('stress_hotspot_intensity', 'Stress Hotspot Intensity', 'coolwarm'),
        ('crack_initiation_risk', 'Crack Initiation Risk', 'Reds'),
        ('delamination_probability', 'Delamination Probability', 'Blues')
    ]
    
    for idx, (col, label, cmap) in enumerate(outputs):
        ax = axes[idx // 3, idx % 3]
        n, bins, patches = ax.hist(df[col], bins=50, alpha=0.7, edgecolor='black')
        
        # Color gradient
        for i, patch in enumerate(patches):
            plt.setp(patch, 'facecolor', plt.cm.get_cmap(cmap)(i/len(patches)))
        
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.text(0.95, 0.95, f'μ={mean_val:.2e}\nσ={std_val:.2e}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved output distributions to {output_path}")
    plt.close()


def plot_correlation_matrix(df, output_path):
    """Plot correlation matrix of key features"""
    # Select subset of features for clarity
    features = [
        'sintering_temperature_C', 'cooling_rate_C_per_min', 'porosity_percent',
        'thermal_stress_Pa', 'thermal_strain', 'stress_hotspot_intensity',
        'crack_initiation_risk', 'delamination_probability'
    ]
    
    corr_matrix = df[features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved correlation matrix to {output_path}")
    plt.close()


def plot_stress_vs_temperature(df, output_path):
    """Plot stress vs temperature relationship"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Thermal Stress vs Process Parameters', fontsize=16, fontweight='bold')
    
    # Stress vs Temperature
    scatter = axes[0].scatter(df['sintering_temperature_C'], 
                             df['thermal_stress_Pa'], 
                             c=df['cooling_rate_C_per_min'],
                             cmap='viridis', alpha=0.5, s=10)
    axes[0].set_xlabel('Sintering Temperature (°C)')
    axes[0].set_ylabel('Thermal Stress (Pa)')
    axes[0].set_title('Colored by Cooling Rate')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=axes[0])
    cbar1.set_label('Cooling Rate (°C/min)')
    
    # Stress vs Porosity
    scatter2 = axes[1].scatter(df['porosity_percent'], 
                              df['thermal_stress_Pa'],
                              c=df['sintering_temperature_C'],
                              cmap='plasma', alpha=0.5, s=10)
    axes[1].set_xlabel('Porosity (%)')
    axes[1].set_ylabel('Thermal Stress (Pa)')
    axes[1].set_title('Colored by Temperature')
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved stress analysis to {output_path}")
    plt.close()


def plot_risk_analysis(df, output_path):
    """Plot risk analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
    
    # Crack risk vs stress
    scatter1 = axes[0, 0].scatter(df['thermal_stress_Pa'], 
                                  df['crack_initiation_risk'],
                                  c=df['porosity_percent'],
                                  cmap='YlOrRd', alpha=0.5, s=10)
    axes[0, 0].set_xlabel('Thermal Stress (Pa)')
    axes[0, 0].set_ylabel('Crack Initiation Risk')
    axes[0, 0].set_title('Crack Risk vs Stress (colored by Porosity)')
    axes[0, 0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Porosity (%)')
    
    # Delamination vs cooling rate
    scatter2 = axes[0, 1].scatter(df['cooling_rate_C_per_min'], 
                                  df['delamination_probability'],
                                  c=df['TEC_mismatch_K-1'],
                                  cmap='Blues', alpha=0.5, s=10)
    axes[0, 1].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 1].set_ylabel('Delamination Probability')
    axes[0, 1].set_title('Delamination vs Cooling Rate (colored by TEC mismatch)')
    axes[0, 1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('TEC Mismatch (K⁻¹)')
    
    # Stress hotspot vs porosity
    axes[1, 0].hexbin(df['porosity_percent'], df['stress_hotspot_intensity'],
                      gridsize=30, cmap='hot', mincnt=1)
    axes[1, 0].set_xlabel('Porosity (%)')
    axes[1, 0].set_ylabel('Stress Hotspot Intensity')
    axes[1, 0].set_title('Hotspot Intensity vs Porosity (hexbin)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined risk heatmap
    axes[1, 1].scatter(df['crack_initiation_risk'], 
                      df['delamination_probability'],
                      c=df['stress_hotspot_intensity'],
                      cmap='RdYlBu_r', alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Crack Initiation Risk')
    axes[1, 1].set_ylabel('Delamination Probability')
    axes[1, 1].set_title('Combined Risk Analysis')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved risk analysis to {output_path}")
    plt.close()


def plot_validation_comparison(train_df, val_df, output_path):
    """Compare training and validation datasets"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training vs Validation Dataset Comparison', fontsize=16, fontweight='bold')
    
    comparisons = [
        ('thermal_stress_Pa', 'Thermal Stress (Pa)'),
        ('thermal_strain', 'Thermal Strain'),
        ('stress_hotspot_intensity', 'Stress Hotspot Intensity'),
        ('crack_initiation_risk', 'Crack Initiation Risk'),
        ('delamination_probability', 'Delamination Probability')
    ]
    
    for idx, (col, label) in enumerate(comparisons):
        ax = axes[idx // 3, idx % 3]
        
        ax.hist(train_df[col], bins=50, alpha=0.5, label='Training', 
                color='blue', density=True)
        ax.hist(val_df[col], bins=50, alpha=0.5, label='Validation', 
                color='red', density=True)
        
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved validation comparison to {output_path}")
    plt.close()


def plot_measurement_noise(val_df, output_path):
    """Plot measurement noise in validation data"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Experimental Measurement Noise (DIC/XRD)', fontsize=16, fontweight='bold')
    
    # DIC strain measurement
    strain_error = (val_df['thermal_strain_measured'] - val_df['thermal_strain']) / val_df['thermal_strain'] * 100
    axes[0].hist(strain_error, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[0].set_xlabel('DIC Strain Measurement Error (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Digital Image Correlation (DIC) - Strain')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.95, 0.95, f'Mean Error: {strain_error.mean():.2f}%\nStd: {strain_error.std():.2f}%',
                transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # XRD stress measurement
    stress_error = (val_df['thermal_stress_Pa_measured'] - val_df['thermal_stress_Pa']) / val_df['thermal_stress_Pa'] * 100
    axes[1].hist(stress_error, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('XRD Stress Measurement Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('X-Ray Diffraction (XRD) - Stress')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.95, 0.95, f'Mean Error: {stress_error.mean():.2f}%\nStd: {stress_error.std():.2f}%',
                transform=axes[1].transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved measurement noise analysis to {output_path}")
    plt.close()


def generate_summary_report(train_df, val_df, output_path):
    """Generate a comprehensive summary report"""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ML TRAINING DATASET SUMMARY REPORT\n")
        f.write("ANN and PINN Models - Sintering Analysis\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Training samples:   {len(train_df):,}\n")
        f.write(f"Validation samples: {len(val_df):,}\n")
        f.write(f"Total samples:      {len(train_df) + len(val_df):,}\n")
        f.write(f"Features per sample: {len(train_df.columns)}\n\n")
        
        f.write("INPUT PARAMETER RANGES\n")
        f.write("-"*80 + "\n")
        f.write(f"Sintering Temperature: {train_df['sintering_temperature_C'].min():.1f} - {train_df['sintering_temperature_C'].max():.1f} °C\n")
        f.write(f"Cooling Rate:          {train_df['cooling_rate_C_per_min'].min():.2f} - {train_df['cooling_rate_C_per_min'].max():.2f} °C/min\n")
        f.write(f"Porosity:              {train_df['porosity_percent'].min():.2f} - {train_df['porosity_percent'].max():.2f} %\n")
        f.write(f"TEC Mismatch:          {train_df['TEC_mismatch_K-1'].min():.2e} - {train_df['TEC_mismatch_K-1'].max():.2e} K⁻¹\n\n")
        
        f.write("OUTPUT LABEL STATISTICS (Training)\n")
        f.write("-"*80 + "\n")
        outputs = ['thermal_stress_Pa', 'thermal_strain', 'stress_hotspot_intensity',
                   'crack_initiation_risk', 'delamination_probability']
        for col in outputs:
            f.write(f"\n{col}:\n")
            f.write(f"  Mean:   {train_df[col].mean():.4e}\n")
            f.write(f"  Std:    {train_df[col].std():.4e}\n")
            f.write(f"  Min:    {train_df[col].min():.4e}\n")
            f.write(f"  Max:    {train_df[col].max():.4e}\n")
            f.write(f"  25%:    {train_df[col].quantile(0.25):.4e}\n")
            f.write(f"  Median: {train_df[col].median():.4e}\n")
            f.write(f"  75%:    {train_df[col].quantile(0.75):.4e}\n")
        
        f.write("\n\nVALIDATION DATA CHARACTERISTICS\n")
        f.write("-"*80 + "\n")
        strain_error = ((val_df['thermal_strain_measured'] - val_df['thermal_strain']) / val_df['thermal_strain'] * 100).abs()
        stress_error = ((val_df['thermal_stress_Pa_measured'] - val_df['thermal_stress_Pa']) / val_df['thermal_stress_Pa'] * 100).abs()
        
        f.write(f"DIC (Strain) Measurement Error:\n")
        f.write(f"  Mean:   {strain_error.mean():.2f}%\n")
        f.write(f"  Std:    {strain_error.std():.2f}%\n")
        f.write(f"  Max:    {strain_error.max():.2f}%\n")
        
        f.write(f"\nXRD (Stress) Measurement Error:\n")
        f.write(f"  Mean:   {stress_error.mean():.2f}%\n")
        f.write(f"  Std:    {stress_error.std():.2f}%\n")
        f.write(f"  Max:    {stress_error.max():.2f}%\n")
        
        f.write(f"\nMeasurement Confidence:\n")
        f.write(f"  DIC Mean Confidence: {val_df['dic_measurement_confidence'].mean():.2f}\n")
        f.write(f"  XRD Mean Confidence: {val_df['xrd_measurement_confidence'].mean():.2f}\n")
        
        f.write("\n\nRECOMMENDED USAGE\n")
        f.write("-"*80 + "\n")
        f.write("1. Use training_dataset.csv/parquet for model training\n")
        f.write("2. Use validation_dataset.csv/parquet for model validation\n")
        f.write("3. Consider measurement uncertainty in validation metrics\n")
        f.write("4. Input features are physics-based and realistic\n")
        f.write("5. Output labels include probability/risk scores (0-1 scale)\n")
        f.write("6. Both CSV and Parquet formats available\n")
        f.write("   - CSV: Easy to inspect, universal compatibility\n")
        f.write("   - Parquet: Faster loading, smaller file size\n\n")
        
        f.write("="*80 + "\n")
        f.write("Report generated successfully\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Saved summary report to {output_path}")


def main():
    """Main visualization function"""
    print("="*70)
    print("Dataset Visualization and Analysis")
    print("="*70)
    
    # Create output directory
    viz_dir = Path('/workspace/ml_training_data/visualizations')
    viz_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_df, val_df = load_datasets()
    print(f"✓ Loaded {len(train_df)} training samples")
    print(f"✓ Loaded {len(val_df)} validation samples")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_input_distributions(train_df, viz_dir / 'input_distributions.png')
    plot_output_distributions(train_df, viz_dir / 'output_distributions.png')
    plot_correlation_matrix(train_df, viz_dir / 'correlation_matrix.png')
    plot_stress_vs_temperature(train_df, viz_dir / 'stress_analysis.png')
    plot_risk_analysis(train_df, viz_dir / 'risk_analysis.png')
    plot_validation_comparison(train_df, val_df, viz_dir / 'train_val_comparison.png')
    plot_measurement_noise(val_df, viz_dir / 'measurement_noise.png')
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(train_df, val_df, viz_dir / 'summary_report.txt')
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print(f"All visualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    print("  - input_distributions.png")
    print("  - output_distributions.png")
    print("  - correlation_matrix.png")
    print("  - stress_analysis.png")
    print("  - risk_analysis.png")
    print("  - train_val_comparison.png")
    print("  - measurement_noise.png")
    print("  - summary_report.txt")
    print("="*70)


if __name__ == "__main__":
    main()
