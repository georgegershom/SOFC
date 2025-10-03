"""
Visualization script for optimization and validation datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)

def load_datasets(data_dir='/workspace/optimization_datasets'):
    """Load all generated datasets"""
    datasets = {
        'stress_strain': pd.read_csv(f'{data_dir}/stress_strain_profiles.csv'),
        'crack_depth': pd.read_csv(f'{data_dir}/crack_depth_estimates.csv'),
        'sintering': pd.read_csv(f'{data_dir}/sintering_parameters.csv'),
        'geometric': pd.read_csv(f'{data_dir}/geometric_designs.csv'),
        'pso_history': pd.read_csv(f'{data_dir}/pso_optimization_history.csv')
    }
    
    with open(f'{data_dir}/dataset_summary.json', 'r') as f:
        summary = json.load(f)
    
    return datasets, summary

def plot_stress_strain_comparison(df):
    """Plot FEM vs Experimental stress/strain profiles"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Select a few samples for visualization
    sample_ids = df['sample_id'].unique()[:3]
    
    # Plot 1: Stress profiles for sample 1
    sample_data = df[df['sample_id'] == sample_ids[0]]
    axes[0, 0].plot(sample_data['position_mm'], sample_data['fem_stress_MPa'], 
                    label='FEM', linewidth=2)
    axes[0, 0].plot(sample_data['position_mm'], sample_data['experimental_stress_MPa'], 
                    label='Experimental (XRD)', linewidth=2, linestyle='--')
    axes[0, 0].set_xlabel('Position (mm)')
    axes[0, 0].set_ylabel('Stress (MPa)')
    axes[0, 0].set_title(f'Stress Profile Comparison - Sample {sample_ids[0]}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Strain profiles for sample 1
    axes[0, 1].plot(sample_data['position_mm'], sample_data['fem_strain_microstrain'], 
                    label='FEM', linewidth=2)
    axes[0, 1].plot(sample_data['position_mm'], sample_data['experimental_strain_microstrain'], 
                    label='Experimental (XRD)', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Position (mm)')
    axes[0, 1].set_ylabel('Strain (μɛ)')
    axes[0, 1].set_title(f'Strain Profile Comparison - Sample {sample_ids[0]}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Stress residual distribution
    axes[1, 0].hist(df['stress_residual_MPa'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(df['stress_residual_MPa'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean = {df["stress_residual_MPa"].mean():.2f} MPa')
    axes[1, 0].set_xlabel('Stress Residual (MPa)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Stress Residuals (FEM vs Experimental)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation plot
    axes[1, 1].scatter(df['fem_stress_MPa'], df['experimental_stress_MPa'], 
                      alpha=0.3, s=5)
    min_val = min(df['fem_stress_MPa'].min(), df['experimental_stress_MPa'].min())
    max_val = max(df['fem_stress_MPa'].max(), df['experimental_stress_MPa'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
    axes[1, 1].set_xlabel('FEM Stress (MPa)')
    axes[1, 1].set_ylabel('Experimental Stress (MPa)')
    axes[1, 1].set_title('FEM vs Experimental Stress Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/optimization_datasets/stress_strain_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: stress_strain_analysis.png")
    plt.close()

def plot_crack_depth_analysis(df):
    """Plot crack depth estimates: XRD vs PSO predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: XRD vs True crack depth
    axes[0, 0].scatter(df['true_crack_depth_mm'], df['xrd_crack_depth_mm'], 
                      alpha=0.6, label='XRD Measurements')
    axes[0, 0].plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect Agreement')
    axes[0, 0].set_xlabel('True Crack Depth (mm)')
    axes[0, 0].set_ylabel('XRD Measured Depth (mm)')
    axes[0, 0].set_title('Synchrotron XRD Measurement Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: PSO vs True crack depth
    axes[0, 1].scatter(df['true_crack_depth_mm'], df['pso_predicted_depth_mm'], 
                      alpha=0.6, color='orange', label='PSO Predictions')
    axes[0, 1].plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect Agreement')
    axes[0, 1].set_xlabel('True Crack Depth (mm)')
    axes[0, 1].set_ylabel('PSO Predicted Depth (mm)')
    axes[0, 1].set_title('PSO Inverse Model Prediction Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error comparison
    error_data = pd.DataFrame({
        'XRD Error': df['xrd_measurement_error_mm'],
        'PSO Error': df['pso_prediction_error_mm']
    })
    error_data.boxplot(ax=axes[1, 0])
    axes[1, 0].set_ylabel('Absolute Error (mm)')
    axes[1, 0].set_title('Measurement/Prediction Error Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: PSO convergence vs accuracy
    scatter = axes[1, 1].scatter(df['pso_iterations'], df['pso_prediction_error_mm'], 
                                c=df['pso_convergence_rate'], cmap='viridis', alpha=0.6)
    axes[1, 1].set_xlabel('PSO Iterations')
    axes[1, 1].set_ylabel('PSO Prediction Error (mm)')
    axes[1, 1].set_title('PSO Performance: Iterations vs Accuracy')
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Convergence Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/optimization_datasets/crack_depth_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: crack_depth_analysis.png")
    plt.close()

def plot_sintering_optimization(df):
    """Plot sintering parameter optimization results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Cooling rate vs quality score
    axes[0, 0].scatter(df['cooling_rate_C_per_min'], df['quality_score'], alpha=0.6)
    axes[0, 0].axvspan(1, 2, alpha=0.2, color='green', label='Optimal Range')
    axes[0, 0].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 0].set_ylabel('Quality Score')
    axes[0, 0].set_title('Cooling Rate vs Quality Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cooling rate vs density
    axes[0, 1].scatter(df['cooling_rate_C_per_min'], df['density_percent_theoretical'], alpha=0.6)
    axes[0, 1].axvspan(1, 2, alpha=0.2, color='green', label='Optimal Range')
    axes[0, 1].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 1].set_ylabel('Density (% Theoretical)')
    axes[0, 1].set_title('Cooling Rate vs Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cooling rate vs crack density
    axes[0, 2].scatter(df['cooling_rate_C_per_min'], df['crack_density_per_cm2'], alpha=0.6, color='red')
    axes[0, 2].axvspan(1, 2, alpha=0.2, color='green', label='Optimal Range')
    axes[0, 2].set_xlabel('Cooling Rate (°C/min)')
    axes[0, 2].set_ylabel('Crack Density (cracks/cm²)')
    axes[0, 2].set_title('Cooling Rate vs Crack Density')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Hold temperature vs quality
    axes[1, 0].scatter(df['hold_temperature_C'], df['quality_score'], alpha=0.6)
    axes[1, 0].set_xlabel('Hold Temperature (°C)')
    axes[1, 0].set_ylabel('Quality Score')
    axes[1, 0].set_title('Sintering Temperature vs Quality')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Comparison of optimal vs non-optimal
    optimal = df[df['optimal_range_cooling'] == True]
    non_optimal = df[df['optimal_range_cooling'] == False]
    
    comparison_data = pd.DataFrame({
        'Optimal Range\n(1-2°C/min)': [optimal['quality_score'].mean()],
        'Outside Range': [non_optimal['quality_score'].mean()]
    })
    comparison_data.T.plot(kind='bar', ax=axes[1, 1], legend=False)
    axes[1, 1].set_ylabel('Average Quality Score')
    axes[1, 1].set_title('Optimal vs Non-Optimal Cooling Rate')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Material comparison
    material_quality = df.groupby('material')['quality_score'].mean().sort_values(ascending=False)
    material_quality.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_ylabel('Average Quality Score')
    axes[1, 2].set_title('Quality Score by Material')
    axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/optimization_datasets/sintering_optimization.png', dpi=300, bbox_inches='tight')
    print("Saved: sintering_optimization.png")
    plt.close()

def plot_geometric_designs(df):
    """Plot geometric design variations analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Design type comparison - stress
    design_stress = df.groupby('design_type')['max_von_mises_stress_MPa'].mean().sort_values()
    design_stress.plot(kind='barh', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_xlabel('Average Max von Mises Stress (MPa)')
    axes[0, 0].set_title('Stress Levels by Design Type')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Design type comparison - thermal performance
    design_thermal = df.groupby('design_type')['thermal_performance_score'].mean().sort_values(ascending=False)
    design_thermal.plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_xlabel('Average Thermal Performance Score')
    axes[0, 1].set_title('Thermal Performance by Design Type')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Efficiency score comparison
    design_efficiency = df.groupby('design_type')['efficiency_score'].mean().sort_values(ascending=False)
    design_efficiency.plot(kind='bar', ax=axes[1, 0], color='green')
    axes[1, 0].set_ylabel('Average Efficiency Score')
    axes[1, 0].set_title('Overall Efficiency by Design Type')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Stress concentration vs thermal performance
    for design_type in df['design_type'].unique():
        design_data = df[df['design_type'] == design_type]
        axes[1, 1].scatter(design_data['stress_concentration_factor'], 
                          design_data['thermal_performance_score'],
                          label=design_type, alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Stress Concentration Factor')
    axes[1, 1].set_ylabel('Thermal Performance Score')
    axes[1, 1].set_title('Design Trade-off: Stress vs Thermal Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/optimization_datasets/geometric_designs.png', dpi=300, bbox_inches='tight')
    print("Saved: geometric_designs.png")
    plt.close()

def plot_pso_convergence(df):
    """Plot PSO optimization convergence"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Convergence curves for selected runs
    runs = df['run_id'].unique()[:5]
    for run_id in runs:
        run_data = df[df['run_id'] == run_id].sort_values('iteration')
        axes[0, 0].plot(run_data['iteration'], run_data['best_fitness'], 
                       label=f'Run {run_id}', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Best Fitness')
    axes[0, 0].set_title('PSO Convergence Curves')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Swarm diversity over iterations
    for run_id in runs:
        run_data = df[df['run_id'] == run_id].sort_values('iteration')
        axes[0, 1].plot(run_data['iteration'], run_data['swarm_diversity'], 
                       label=f'Run {run_id}', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Swarm Diversity')
    axes[0, 1].set_title('Swarm Diversity Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final fitness vs particle count
    final_fitness = df.groupby(['run_id', 'n_particles'])['best_fitness'].min().reset_index()
    particle_counts = final_fitness.groupby('n_particles')['best_fitness'].mean()
    particle_counts.plot(kind='bar', ax=axes[1, 0], color='purple')
    axes[1, 0].set_xlabel('Number of Particles')
    axes[1, 0].set_ylabel('Average Final Best Fitness')
    axes[1, 0].set_title('PSO Performance vs Swarm Size')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Computation time distribution
    axes[1, 1].hist(df['computation_time_sec'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(df['computation_time_sec'].mean(), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Mean = {df["computation_time_sec"].mean():.2f} s')
    axes[1, 1].set_xlabel('Computation Time (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('PSO Computation Time Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/optimization_datasets/pso_convergence.png', dpi=300, bbox_inches='tight')
    print("Saved: pso_convergence.png")
    plt.close()

def generate_all_visualizations():
    """Generate all visualization plots"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    datasets, summary = load_datasets()
    
    print("1. Stress/Strain Analysis...")
    plot_stress_strain_comparison(datasets['stress_strain'])
    
    print("2. Crack Depth Analysis...")
    plot_crack_depth_analysis(datasets['crack_depth'])
    
    print("3. Sintering Optimization...")
    plot_sintering_optimization(datasets['sintering'])
    
    print("4. Geometric Designs...")
    plot_geometric_designs(datasets['geometric'])
    
    print("5. PSO Convergence...")
    plot_pso_convergence(datasets['pso_history'])
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nAll plots saved to /workspace/optimization_datasets/\n")

if __name__ == "__main__":
    generate_all_visualizations()
