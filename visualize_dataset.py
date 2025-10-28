#!/usr/bin/env python3
"""
Dataset Visualization Script for Sintering Process Analysis
Creates comprehensive visualizations of the generated dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import h5py
import json

def load_dataset():
    """Load the generated dataset"""
    # Load training data
    df = pd.read_csv('/workspace/sintering_training_data.csv')
    
    # Load validation data
    val_df = pd.read_csv('/workspace/sintering_validation_data.csv')
    
    # Load metadata
    with open('/workspace/dataset_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return df, val_df, metadata

def create_parameter_distributions(df, output_dir='/workspace'):
    """Create visualizations of parameter distributions"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sintering Process Parameter Distributions', fontsize=16, fontweight='bold')
    
    # Temperature distribution
    axes[0, 0].hist(df['temperature'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Sintering Temperature Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cooling rate distribution
    axes[0, 1].hist(df['cooling_rate'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Cooling Rate Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Porosity distribution
    axes[0, 2].hist(df['porosity'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].set_xlabel('Porosity')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Porosity Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Material distribution
    material_counts = df['material'].value_counts()
    axes[1, 0].pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Material Distribution')
    
    # Stress distribution
    axes[1, 1].hist(df['stress'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Stress (Pa)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Stress Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Strain distribution
    axes[1, 2].hist(df['strain'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].set_xlabel('Strain')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Strain Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_output_analysis(df, output_dir='/workspace'):
    """Create visualizations of output variables"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Output Variable Analysis', fontsize=16, fontweight='bold')
    
    # Stress hotspot density
    axes[0, 0].hist(df['stress_hotspot_density'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Stress Hotspot Density')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Stress Hotspot Density Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Crack risk
    axes[0, 1].hist(df['crack_risk'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Crack Risk')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Crack Risk Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Delamination probability
    axes[1, 0].hist(df['delamination_probability'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('Delamination Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Delamination Probability Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Output correlation heatmap
    output_cols = ['stress_hotspot_density', 'crack_risk', 'delamination_probability']
    corr_matrix = df[output_cols].corr()
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 1].set_xticks(range(len(output_cols)))
    axes[1, 1].set_yticks(range(len(output_cols)))
    axes[1, 1].set_xticklabels(output_cols, rotation=45)
    axes[1, 1].set_yticklabels(output_cols)
    axes[1, 1].set_title('Output Variable Correlations')
    
    # Add correlation values to heatmap
    for i in range(len(output_cols)):
        for j in range(len(output_cols)):
            text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/output_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_correlations(df, output_dir='/workspace'):
    """Create correlation analysis between parameters and outputs"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Parameter-Output Correlations', fontsize=16, fontweight='bold')
    
    # Temperature vs outputs
    axes[0, 0].scatter(df['temperature'], df['stress_hotspot_density'], alpha=0.5, s=1)
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Stress Hotspot Density')
    axes[0, 0].set_title('Temperature vs Stress Hotspots')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cooling rate vs outputs
    axes[0, 1].scatter(df['cooling_rate'], df['delamination_probability'], alpha=0.5, s=1)
    axes[0, 1].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 1].set_ylabel('Delamination Probability')
    axes[0, 1].set_title('Cooling Rate vs Delamination')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Porosity vs outputs
    axes[1, 0].scatter(df['porosity'], df['crack_risk'], alpha=0.5, s=1)
    axes[1, 0].set_xlabel('Porosity')
    axes[1, 0].set_ylabel('Crack Risk')
    axes[1, 0].set_title('Porosity vs Crack Risk')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Stress vs strain
    axes[1, 1].scatter(df['stress'], df['strain'], alpha=0.5, s=1)
    axes[1, 1].set_xlabel('Stress (Pa)')
    axes[1, 1].set_ylabel('Strain')
    axes[1, 1].set_title('Stress vs Strain Relationship')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_material_analysis(df, output_dir='/workspace'):
    """Create material-specific analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Material-Specific Analysis', fontsize=16, fontweight='bold')
    
    # Stress by material
    df.boxplot(column='stress', by='material', ax=axes[0, 0])
    axes[0, 0].set_title('Stress Distribution by Material')
    axes[0, 0].set_xlabel('Material')
    axes[0, 0].set_ylabel('Stress (Pa)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Strain by material
    df.boxplot(column='strain', by='material', ax=axes[0, 1])
    axes[0, 1].set_title('Strain Distribution by Material')
    axes[0, 1].set_xlabel('Material')
    axes[0, 1].set_ylabel('Strain')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Stress hotspots by material
    df.boxplot(column='stress_hotspot_density', by='material', ax=axes[1, 0])
    axes[1, 0].set_title('Stress Hotspot Density by Material')
    axes[1, 0].set_xlabel('Material')
    axes[1, 0].set_ylabel('Stress Hotspot Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Delamination probability by material
    df.boxplot(column='delamination_probability', by='material', ax=axes[1, 1])
    axes[1, 1].set_title('Delamination Probability by Material')
    axes[1, 1].set_xlabel('Material')
    axes[1, 1].set_ylabel('Delamination Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/material_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_validation_analysis(val_df, output_dir='/workspace'):
    """Create validation dataset analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Validation Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Validation temperature distribution
    axes[0, 0].hist(val_df['temperature'], bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Validation Temperature Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation cooling rate distribution
    axes[0, 1].hist(val_df['cooling_rate'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Validation Cooling Rate Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Measurement type distribution
    measurement_counts = val_df['measurement_type'].value_counts()
    axes[1, 0].pie(measurement_counts.values, labels=measurement_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Measurement Type Distribution')
    
    # Measurement uncertainty
    axes[1, 1].hist(val_df['measurement_uncertainty'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_xlabel('Measurement Uncertainty')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Measurement Uncertainty Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, val_df, metadata, output_dir='/workspace'):
    """Generate a comprehensive summary report"""
    report = f"""
# Sintering Process Machine Learning Dataset Summary

## Dataset Overview
- **Total Training Samples**: {len(df):,}
- **Total Validation Samples**: {len(val_df):,}
- **Features**: {len(df.columns)}
- **Materials**: {len(metadata['materials'])}

## Parameter Ranges
- **Temperature**: {metadata['parameter_ranges']['temperature_range'][0]} - {metadata['parameter_ranges']['temperature_range'][1]} °C
- **Cooling Rate**: {metadata['parameter_ranges']['cooling_rate_range'][0]} - {metadata['parameter_ranges']['cooling_rate_range'][1]} °C/min
- **Porosity**: {metadata['parameter_ranges']['porosity_range'][0]} - {metadata['parameter_ranges']['porosity_range'][1]}
- **TEC Mismatch**: {metadata['parameter_ranges']['tec_mismatch']:.2e} K⁻¹

## Material Properties
"""
    
    for material, props in metadata['materials'].items():
        report += f"""
### {material.title()}
- Young's Modulus (E): {props['E']:.0e} Pa
- Poisson's Ratio (ν): {props['nu']:.2f}
- Thermal Expansion (α): {props['alpha']:.2e} K⁻¹
- Density: {props['density']:.0f} kg/m³
"""
    
    report += f"""
## Output Statistics
- **Stress Hotspot Density**: {df['stress_hotspot_density'].mean():.3f} ± {df['stress_hotspot_density'].std():.3f}
- **Crack Risk**: {df['crack_risk'].mean():.3f} ± {df['crack_risk'].std():.3f}
- **Delamination Probability**: {df['delamination_probability'].mean():.3f} ± {df['delamination_probability'].std():.3f}

## Material Distribution
"""
    
    for material, count in df['material'].value_counts().items():
        percentage = (count / len(df)) * 100
        report += f"- {material.title()}: {count:,} samples ({percentage:.1f}%)\n"
    
    report += f"""
## Dataset Features
### Input Features
- Temperature, cooling rate, porosity
- Material properties (E, ν, α, density)
- TEC mismatch
- Derived features (stress/strain ratio, thermal gradient)

### Output Labels
- Stress hotspot density
- Crack initiation risk
- Delamination probability

### Validation Data
- DIC/XRD measurement simulations
- Controlled parameter ranges
- Measurement uncertainty modeling

## File Structure
- `sintering_training_data.csv`: Main training dataset
- `sintering_validation_data.csv`: Validation dataset
- `sintering_dataset.h5`: HDF5 format with spatial data
- `dataset_metadata.json`: Dataset metadata and parameters

## Usage for ML Models
This dataset is optimized for:
- **ANN (Artificial Neural Networks)**: Tabular data with engineered features
- **PINN (Physics-Informed Neural Networks)**: Physics-based constraints and spatial data
- **Regression tasks**: Predicting stress hotspots, crack risk, delamination
- **Classification tasks**: Material failure prediction

## Next Steps
1. Load dataset using provided HDF5 or CSV files
2. Preprocess features using StandardScaler
3. Split into train/validation/test sets
4. Train ANN/PINN models
5. Validate against experimental DIC/XRD data
"""
    
    with open(f'{output_dir}/dataset_summary_report.md', 'w') as f:
        f.write(report)
    
    print("Summary report saved to dataset_summary_report.md")

def main():
    """Main visualization function"""
    print("📊 Creating dataset visualizations...")
    
    # Load dataset
    df, val_df, metadata = load_dataset()
    
    # Create visualizations
    print("Creating parameter distributions...")
    create_parameter_distributions(df)
    
    print("Creating output analysis...")
    create_output_analysis(df)
    
    print("Creating parameter correlations...")
    create_parameter_correlations(df)
    
    print("Creating material analysis...")
    create_material_analysis(df)
    
    print("Creating validation analysis...")
    create_validation_analysis(val_df)
    
    print("Generating summary report...")
    generate_summary_report(df, val_df, metadata)
    
    print("✅ All visualizations created!")
    print("Files created:")
    print("  - parameter_distributions.png")
    print("  - output_analysis.png")
    print("  - parameter_correlations.png")
    print("  - material_analysis.png")
    print("  - validation_analysis.png")
    print("  - dataset_summary_report.md")

if __name__ == "__main__":
    main()