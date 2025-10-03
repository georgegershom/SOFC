#!/usr/bin/env python3
"""
Visualization script for Optimization and Validation Dataset
Generates plots for FEM vs experimental data, crack depth analysis, 
sintering parameters, and geometric design variations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_fem_vs_experimental():
    """Plot FEM predicted vs experimental stress/strain profiles"""
    data = pd.read_csv('fem_experimental_comparison/stress_strain_profiles.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Stress comparison
    for sample in data['sample_id'].unique()[:3]:
        sample_data = data[data['sample_id'] == sample]
        axes[0, 0].plot(sample_data['position_mm'], sample_data['fem_stress_mpa'], 
                       'o-', label=f'{sample} FEM', alpha=0.7)
        axes[0, 0].plot(sample_data['position_mm'], sample_data['experimental_stress_mpa'], 
                       's--', label=f'{sample} Exp', alpha=0.7)
    
    axes[0, 0].set_xlabel('Position (mm)')
    axes[0, 0].set_ylabel('Stress (MPa)')
    axes[0, 0].set_title('FEM vs Experimental Stress Profiles')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Strain comparison
    for sample in data['sample_id'].unique()[:3]:
        sample_data = data[data['sample_id'] == sample]
        axes[0, 1].plot(sample_data['position_mm'], sample_data['fem_strain'], 
                       'o-', label=f'{sample} FEM', alpha=0.7)
        axes[0, 1].plot(sample_data['position_mm'], sample_data['experimental_strain'], 
                       's--', label=f'{sample} Exp', alpha=0.7)
    
    axes[0, 1].set_xlabel('Position (mm)')
    axes[0, 1].set_ylabel('Strain')
    axes[0, 1].set_title('FEM vs Experimental Strain Profiles')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation plot - Stress
    axes[1, 0].scatter(data['fem_stress_mpa'], data['experimental_stress_mpa'], 
                      alpha=0.6, s=30)
    min_val = min(data['fem_stress_mpa'].min(), data['experimental_stress_mpa'].min())
    max_val = max(data['fem_stress_mpa'].max(), data['experimental_stress_mpa'].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')
    axes[1, 0].set_xlabel('FEM Stress (MPa)')
    axes[1, 0].set_ylabel('Experimental Stress (MPa)')
    axes[1, 0].set_title('Stress Correlation: FEM vs Experimental')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation plot - Strain
    axes[1, 1].scatter(data['fem_strain'], data['experimental_strain'], 
                      alpha=0.6, s=30)
    min_val = min(data['fem_strain'].min(), data['experimental_strain'].min())
    max_val = max(data['fem_strain'].max(), data['experimental_strain'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')
    axes[1, 1].set_xlabel('FEM Strain')
    axes[1, 1].set_ylabel('Experimental Strain')
    axes[1, 1].set_title('Strain Correlation: FEM vs Experimental')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fem_experimental_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print correlation coefficients
    stress_corr = np.corrcoef(data['fem_stress_mpa'], data['experimental_stress_mpa'])[0, 1]
    strain_corr = np.corrcoef(data['fem_strain'], data['experimental_strain'])[0, 1]
    print(f"Stress correlation coefficient: {stress_corr:.4f}")
    print(f"Strain correlation coefficient: {strain_corr:.4f}")

def plot_crack_depth_analysis():
    """Plot crack depth estimates from synchrotron XRD vs model predictions"""
    data = pd.read_csv('crack_depth_analysis/crack_depth_estimates.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Depth comparison by sample
    samples = data['sample_id'].unique()[:4]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(samples)))
    
    for i, sample in enumerate(samples):
        sample_data = data[data['sample_id'] == sample]
        axes[0, 0].plot(sample_data['location_x_mm'], sample_data['synchrotron_xrd_depth_mm'], 
                       'o-', label=f'{sample} XRD', color=colors[i], alpha=0.7)
        axes[0, 0].plot(sample_data['location_x_mm'], sample_data['model_prediction_mm'], 
                       's--', label=f'{sample} Model', color=colors[i], alpha=0.5)
    
    axes[0, 0].set_xlabel('Location X (mm)')
    axes[0, 0].set_ylabel('Crack Depth (mm)')
    axes[0, 0].set_title('Crack Depth: Synchrotron XRD vs Model Predictions')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSO optimization comparison
    axes[0, 1].scatter(data['synchrotron_xrd_depth_mm'], data['pso_optimized_mm'], 
                      c=data['uncertainty_mm'], cmap='coolwarm', alpha=0.6, s=50)
    min_val = data[['synchrotron_xrd_depth_mm', 'pso_optimized_mm']].min().min()
    max_val = data[['synchrotron_xrd_depth_mm', 'pso_optimized_mm']].max().max()
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Uncertainty (mm)')
    axes[0, 1].set_xlabel('Synchrotron XRD Depth (mm)')
    axes[0, 1].set_ylabel('PSO Optimized Depth (mm)')
    axes[0, 1].set_title('PSO Optimization Results')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    errors_model = data['model_prediction_mm'] - data['synchrotron_xrd_depth_mm']
    errors_pso = data['pso_optimized_mm'] - data['synchrotron_xrd_depth_mm']
    
    axes[1, 0].hist(errors_model, bins=15, alpha=0.5, label='Model Error', edgecolor='black')
    axes[1, 0].hist(errors_pso, bins=15, alpha=0.5, label='PSO Error', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Prediction Error (mm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Strain gradient vs crack depth
    axes[1, 1].scatter(data['strain_gradient'], data['synchrotron_xrd_depth_mm'], 
                      c=data['diffraction_intensity'], cmap='plasma', alpha=0.6, s=50)
    cbar2 = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar2.set_label('Diffraction Intensity')
    axes[1, 1].set_xlabel('Strain Gradient')
    axes[1, 1].set_ylabel('Crack Depth (mm)')
    axes[1, 1].set_title('Strain Gradient vs Crack Depth')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('crack_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\nCrack Depth Analysis Statistics:")
    print(f"Model MAE: {np.abs(errors_model).mean():.4f} mm")
    print(f"PSO MAE: {np.abs(errors_pso).mean():.4f} mm")
    print(f"Model RMSE: {np.sqrt((errors_model**2).mean()):.4f} mm")
    print(f"PSO RMSE: {np.sqrt((errors_pso**2).mean()):.4f} mm")

def plot_sintering_optimization():
    """Plot optimal sintering parameters and temperature profiles"""
    data = pd.read_csv('sintering_optimization/optimal_parameters.csv')
    
    with open('sintering_optimization/temperature_profiles.json', 'r') as f:
        temp_profiles = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Cooling rate vs density
    scatter = axes[0, 0].scatter(data['cooling_rate_c_per_min'], data['density_percent'], 
                                c=data['optimization_score'], cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter, ax=axes[0, 0], label='Optimization Score')
    axes[0, 0].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 0].set_ylabel('Density (%)')
    axes[0, 0].set_title('Cooling Rate vs Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cooling rate vs residual stress
    axes[0, 1].scatter(data['cooling_rate_c_per_min'], data['residual_stress_mpa'], 
                      c=data['grain_size_um'], cmap='coolwarm', s=50, alpha=0.6)
    plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Grain Size (μm)')
    axes[0, 1].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 1].set_ylabel('Residual Stress (MPa)')
    axes[0, 1].set_title('Cooling Rate vs Residual Stress')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature profiles
    for profile_name, profile_data in list(temp_profiles['temperature_profiles'].items())[:3]:
        times = [p['time_min'] for p in profile_data['profile_data']]
        temps = [p['temperature_c'] for p in profile_data['profile_data']]
        axes[0, 2].plot(times, temps, '-', label=f"Rate: {profile_data['cooling_rate']}°C/min", 
                       linewidth=2, alpha=0.8)
    
    axes[0, 2].set_xlabel('Time (min)')
    axes[0, 2].set_ylabel('Temperature (°C)')
    axes[0, 2].set_title('Temperature Profiles')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Hardness vs fracture toughness
    axes[1, 0].scatter(data['hardness_hv'], data['fracture_toughness_mpa_m05'], 
                      c=data['cooling_rate_c_per_min'], cmap='plasma', s=50, alpha=0.6)
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Cooling Rate (°C/min)')
    axes[1, 0].set_xlabel('Hardness (HV)')
    axes[1, 0].set_ylabel('Fracture Toughness (MPa·m^0.5)')
    axes[1, 0].set_title('Hardness vs Fracture Toughness Trade-off')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PSO convergence history
    convergence = temp_profiles['pso_optimization_results']['convergence_history']
    iterations = [int(k.split('_')[1]) for k in convergence.keys()]
    values = list(convergence.values())
    axes[1, 1].plot(iterations, values, 'o-', linewidth=2, markersize=8, alpha=0.8)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Objective Function Value')
    axes[1, 1].set_title('PSO Convergence History')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Parameter correlation heatmap
    param_cols = ['cooling_rate_c_per_min', 'max_temperature_c', 'holding_time_min', 
                  'density_percent', 'hardness_hv', 'residual_stress_mpa']
    corr_matrix = data[param_cols].corr()
    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_xticks(range(len(param_cols)))
    axes[1, 2].set_yticks(range(len(param_cols)))
    axes[1, 2].set_xticklabels([col.replace('_', '\n') for col in param_cols], rotation=45, ha='right')
    axes[1, 2].set_yticklabels(param_cols)
    axes[1, 2].set_title('Parameter Correlation Matrix')
    plt.colorbar(im, ax=axes[1, 2], label='Correlation')
    
    # Add correlation values
    for i in range(len(param_cols)):
        for j in range(len(param_cols)):
            text = axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('sintering_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print optimal parameters
    optimal = temp_profiles['pso_optimization_results']['optimal_parameters']
    print("\nPSO-Optimized Sintering Parameters:")
    for param, value in optimal.items():
        print(f"  {param}: {value}")

def plot_geometric_designs():
    """Plot geometric design variations and optimization results"""
    data = pd.read_csv('geometric_designs/channel_designs.csv')
    
    with open('geometric_designs/design_optimization_results.json', 'r') as f:
        design_results = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Channel type comparison
    bow_data = data[data['channel_type'] == 'bow_shaped']
    rect_data = data[data['channel_type'] == 'rectangular']
    
    axes[0, 0].boxplot([bow_data['optimization_score'], rect_data['optimization_score']], 
                       labels=['Bow-shaped', 'Rectangular'])
    axes[0, 0].set_ylabel('Optimization Score')
    axes[0, 0].set_title('Design Performance Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pressure drop vs heat transfer
    for channel_type in data['channel_type'].unique():
        type_data = data[data['channel_type'] == channel_type]
        axes[0, 1].scatter(type_data['pressure_drop_kpa'], type_data['heat_transfer_coefficient'],
                          label=channel_type.replace('_', ' ').title(), alpha=0.6, s=50)
    
    axes[0, 1].set_xlabel('Pressure Drop (kPa)')
    axes[0, 1].set_ylabel('Heat Transfer Coefficient')
    axes[0, 1].set_title('Pressure Drop vs Heat Transfer Trade-off')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Aspect ratio effects
    axes[0, 2].scatter(data['aspect_ratio'], data['structural_integrity_score'], 
                      c=data['flow_rate_ml_min'], cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2], label='Flow Rate (ml/min)')
    axes[0, 2].set_xlabel('Aspect Ratio')
    axes[0, 2].set_ylabel('Structural Integrity Score')
    axes[0, 2].set_title('Aspect Ratio Impact on Structural Integrity')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Pareto front
    pareto = design_results['pso_optimization']['pareto_front']
    heat_transfer = [p['heat_transfer'] for p in pareto]
    pressure_drop = [p['pressure_drop'] for p in pareto]
    structural = [p['structural_score'] for p in pareto]
    
    scatter = axes[1, 0].scatter(pressure_drop, heat_transfer, c=structural, 
                                cmap='RdYlGn', s=100, alpha=0.8, edgecolors='black', linewidth=2)
    plt.colorbar(scatter, ax=axes[1, 0], label='Structural Score')
    axes[1, 0].set_xlabel('Pressure Drop (kPa)')
    axes[1, 0].set_ylabel('Heat Transfer Coefficient')
    axes[1, 0].set_title('Pareto Optimal Front')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sensitivity analysis
    sensitivity = design_results['sensitivity_analysis']['most_influential_parameters']
    params = [s['parameter'] for s in sensitivity]
    indices = [s['sensitivity_index'] for s in sensitivity]
    colors_sens = ['green' if s['effect_on_performance'] == 'positive' else 'orange' 
                  for s in sensitivity]
    
    axes[1, 1].barh(params, indices, color=colors_sens, alpha=0.7)
    axes[1, 1].set_xlabel('Sensitivity Index')
    axes[1, 1].set_title('Parameter Sensitivity Analysis')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # Manufacturing complexity vs performance
    complexity_groups = data.groupby('manufacturing_complexity').agg({
        'optimization_score': 'mean',
        'structural_integrity_score': 'mean',
        'thermal_resistance_k_w': 'mean'
    }).reset_index()
    
    x = np.arange(len(complexity_groups))
    width = 0.25
    
    axes[1, 2].bar(x - width, complexity_groups['optimization_score'], width, 
                  label='Optimization Score', alpha=0.7)
    axes[1, 2].bar(x, complexity_groups['structural_integrity_score'], width, 
                  label='Structural Score', alpha=0.7)
    axes[1, 2].bar(x + width, 1/complexity_groups['thermal_resistance_k_w']/1000, width, 
                  label='Thermal Performance', alpha=0.7)
    
    axes[1, 2].set_xlabel('Manufacturing Complexity')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(complexity_groups['manufacturing_complexity'])
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Manufacturing Complexity vs Performance')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('geometric_designs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print optimal design
    optimal = design_results['pso_optimization']['optimal_design']
    print("\nPSO-Optimized Channel Design:")
    print(f"  Type: {optimal['type']}")
    print(f"  Width: {optimal['width_mm']} mm")
    print(f"  Height: {optimal['height_mm']} mm")
    print(f"  Overall Score: {optimal['predicted_performance']['overall_optimization_score']}")

def main():
    """Main function to run all visualizations"""
    print("=" * 60)
    print("Optimization and Validation Dataset Visualization")
    print("=" * 60)
    
    # Change to dataset directory
    import os
    if os.path.exists('optimization_validation_dataset'):
        os.chdir('optimization_validation_dataset')
    
    print("\n1. FEM vs Experimental Analysis")
    print("-" * 40)
    plot_fem_vs_experimental()
    
    print("\n2. Crack Depth Analysis")
    print("-" * 40)
    plot_crack_depth_analysis()
    
    print("\n3. Sintering Optimization")
    print("-" * 40)
    plot_sintering_optimization()
    
    print("\n4. Geometric Design Variations")
    print("-" * 40)
    plot_geometric_designs()
    
    print("\n" + "=" * 60)
    print("Visualization complete! Check the generated PNG files.")
    print("=" * 60)

if __name__ == "__main__":
    main()