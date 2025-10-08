#!/usr/bin/env python3
"""
Example Analysis Script for Multi-Scale FEM Validation Dataset
==============================================================

This script demonstrates how to:
1. Load and visualize experimental stress data
2. Compare FEM predictions with experimental measurements
3. Train a surrogate model using collocation points
4. Analyze crack initiation correlations
5. Perform residual analysis

Requirements:
    pip install pandas numpy matplotlib scipy scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set up paths
BASE_DIR = Path(__file__).parent
EXPERIMENTAL_DIR = BASE_DIR / "experimental_data"
SIMULATION_DIR = BASE_DIR / "simulation_output"
MULTISCALE_DIR = BASE_DIR / "multi_scale_data"


def load_experimental_stress_data():
    """Load all experimental residual stress measurements."""
    print("Loading experimental stress data...")
    
    xrd_data = pd.read_csv(EXPERIMENTAL_DIR / "residual_stress" / "xrd_surface_residual_stress.csv")
    raman_data = pd.read_csv(EXPERIMENTAL_DIR / "residual_stress" / "raman_spectroscopy_stress.csv")
    synchrotron_data = pd.read_csv(EXPERIMENTAL_DIR / "residual_stress" / "synchrotron_xrd_subsurface.csv")
    
    print(f"  XRD measurements: {len(xrd_data)}")
    print(f"  Raman measurements: {len(raman_data)}")
    print(f"  Synchrotron measurements: {len(synchrotron_data)}")
    
    return xrd_data, raman_data, synchrotron_data


def visualize_surface_stress_field(xrd_data):
    """Create 2D contour plot of surface residual stress."""
    print("\nVisualizing surface stress field...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    x = xrd_data['x_position_mm'].values
    y = xrd_data['y_position_mm'].values
    sigma_xx = xrd_data['sigma_xx_MPa'].values
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    Zi = griddata((x, y), sigma_xx, (Xi, Yi), method='cubic')
    
    # Plot 1: Contour plot
    contour = axes[0].contourf(Xi, Yi, Zi, levels=15, cmap='coolwarm')
    axes[0].scatter(x, y, c='black', s=20, marker='x', label='Measurement points')
    axes[0].set_xlabel('X Position (mm)')
    axes[0].set_ylabel('Y Position (mm)')
    axes[0].set_title('Surface Residual Stress σ_xx (MPa)')
    axes[0].legend()
    plt.colorbar(contour, ax=axes[0])
    
    # Plot 2: Histogram of stress values
    axes[1].hist(sigma_xx, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Stress σ_xx (MPa)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Surface Residual Stress')
    axes[1].axvline(sigma_xx.mean(), color='red', linestyle='--', 
                    label=f'Mean: {sigma_xx.mean():.1f} MPa')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'surface_stress_visualization.png', dpi=300)
    print(f"  Saved: surface_stress_visualization.png")
    
    return fig


def compare_fem_vs_experimental():
    """Compare FEM predictions with experimental measurements."""
    print("\nComparing FEM vs. Experimental data...")
    
    # Load data
    fem_data = pd.read_csv(SIMULATION_DIR / "full_field" / "fem_full_field_solution.csv")
    xrd_data = pd.read_csv(EXPERIMENTAL_DIR / "residual_stress" / "xrd_surface_residual_stress.csv")
    
    # For demonstration, find nearest FEM nodes to experimental points
    # In practice, use proper spatial interpolation
    
    # Simple validation at surface (z≈0)
    fem_surface = fem_data[fem_data['z_coord_mm'] == 0.0]
    
    print(f"  FEM surface nodes: {len(fem_surface)}")
    print(f"  Experimental surface points: {len(xrd_data)}")
    
    # Calculate statistics
    fem_stress_mean = fem_surface['stress_xx_MPa'].mean()
    exp_stress_mean = xrd_data['sigma_xx_MPa'].mean()
    
    print(f"\n  Mean FEM stress σ_xx: {fem_stress_mean:.2f} MPa")
    print(f"  Mean Experimental stress σ_xx: {exp_stress_mean:.2f} MPa")
    print(f"  Difference: {abs(fem_stress_mean - exp_stress_mean):.2f} MPa")
    print(f"  Relative error: {100*abs(fem_stress_mean - exp_stress_mean)/abs(exp_stress_mean):.1f}%")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fem_surface['stress_xx_MPa'], fem_surface['von_mises_MPa'], 
               label='FEM', alpha=0.6, s=50)
    ax.scatter(xrd_data['sigma_xx_MPa'], 
               xrd_data['sigma_xx_MPa']*1.1,  # Approximate von Mises
               label='Experimental (XRD)', alpha=0.6, s=50, marker='s')
    ax.set_xlabel('Stress σ_xx (MPa)')
    ax.set_ylabel('Von Mises Stress (MPa)')
    ax.set_title('FEM vs. Experimental Stress Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'fem_vs_experimental.png', dpi=300)
    print(f"  Saved: fem_vs_experimental.png")
    
    return fig


def train_surrogate_model():
    """Train a Gaussian Process surrogate model using collocation points."""
    print("\nTraining surrogate model with collocation points...")
    
    # Load collocation points (sparse "measurements")
    colloc_data = pd.read_csv(SIMULATION_DIR / "collocation_points" / "collocation_point_data.csv")
    
    # Load full-field data (ground truth for validation)
    full_field = pd.read_csv(SIMULATION_DIR / "full_field" / "fem_full_field_solution.csv")
    
    # Prepare features (spatial coordinates) and target (von Mises stress)
    X_colloc = colloc_data[['x_coord_mm', 'y_coord_mm', 'z_coord_mm']].values
    y_colloc = colloc_data['von_mises_MPa'].values
    
    X_full = full_field[['x_coord_mm', 'y_coord_mm', 'z_coord_mm']].values
    y_full = full_field['von_mises_MPa'].values
    
    print(f"  Training points (collocation): {len(X_colloc)}")
    print(f"  Test points (full field): {len(X_full)}")
    
    # Train Gaussian Process
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True)
    
    print("  Training GP model...")
    gp.fit(X_colloc, y_colloc)
    
    # Predict on full field
    y_pred, y_std = gp.predict(X_full, return_std=True)
    
    # Calculate errors
    mse = mean_squared_error(y_full, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_full, y_pred)
    mean_error = np.mean(np.abs(y_full - y_pred))
    
    print(f"\n  Surrogate Model Performance:")
    print(f"    RMSE: {rmse:.2f} MPa")
    print(f"    MAE: {mean_error:.2f} MPa")
    print(f"    R²: {r2:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Predicted vs. Actual
    axes[0].scatter(y_full, y_pred, alpha=0.6, s=30)
    axes[0].plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('True Von Mises Stress (MPa)')
    axes[0].set_ylabel('Predicted Von Mises Stress (MPa)')
    axes[0].set_title(f'Surrogate Model Performance (R²={r2:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_full - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Von Mises Stress (MPa)')
    axes[1].set_ylabel('Residual (MPa)')
    axes[1].set_title('Residual Analysis')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'surrogate_model_performance.png', dpi=300)
    print(f"  Saved: surrogate_model_performance.png")
    
    return gp, fig


def analyze_crack_initiation():
    """Analyze crack initiation locations and correlate with stress."""
    print("\nAnalyzing crack initiation data...")
    
    # Load crack data
    crack_data = pd.read_csv(EXPERIMENTAL_DIR / "crack_analysis" / "crack_initiation_data.csv")
    
    print(f"  Total cracks observed: {len(crack_data)}")
    print(f"  Grain boundary cracks: {crack_data['grain_boundary_crack'].sum()}")
    print(f"  Pore-associated cracks: {crack_data['pore_associated'].sum()}")
    
    # Statistics
    print(f"\n  Crack Statistics:")
    print(f"    Mean crack length: {crack_data['crack_length_um'].mean():.2f} ± {crack_data['crack_length_um'].std():.2f} μm")
    print(f"    Mean critical temperature: {crack_data['critical_temperature_K'].mean():.1f} K")
    print(f"    Mean stress intensity factor: {crack_data['stress_intensity_factor_MPa_sqrt_m'].mean():.2f} MPa√m")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Crack locations
    ax = axes[0, 0]
    gb_cracks = crack_data[crack_data['grain_boundary_crack'] == True]
    pore_cracks = crack_data[crack_data['pore_associated'] == True]
    
    ax.scatter(crack_data['x_position_mm'], crack_data['y_position_mm'], 
               s=crack_data['crack_length_um']*3, alpha=0.5, label='All cracks')
    ax.scatter(gb_cracks['x_position_mm'], gb_cracks['y_position_mm'], 
               marker='s', s=100, c='red', alpha=0.7, label='GB cracks')
    ax.scatter(pore_cracks['x_position_mm'], pore_cracks['y_position_mm'], 
               marker='^', s=100, c='blue', alpha=0.7, label='Pore cracks')
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_title('Crack Locations (size = crack length)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Critical temperature distribution
    axes[0, 1].hist(crack_data['critical_temperature_K'], bins=10, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Critical Temperature (K)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Critical Temperature for Cracking')
    
    # Plot 3: Stress intensity vs crack length
    axes[1, 0].scatter(crack_data['crack_length_um'], 
                       crack_data['stress_intensity_factor_MPa_sqrt_m'], s=50)
    axes[1, 0].set_xlabel('Crack Length (μm)')
    axes[1, 0].set_ylabel('Stress Intensity Factor (MPa√m)')
    axes[1, 0].set_title('Crack Length vs. Stress Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Crack type comparison
    crack_types = ['GB only', 'Pore only', 'Both']
    counts = [
        (crack_data['grain_boundary_crack'] & ~crack_data['pore_associated']).sum(),
        (~crack_data['grain_boundary_crack'] & crack_data['pore_associated']).sum(),
        (crack_data['grain_boundary_crack'] & crack_data['pore_associated']).sum()
    ]
    axes[1, 1].bar(crack_types, counts, edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('Number of Cracks')
    axes[1, 1].set_title('Crack Initiation Mechanisms')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'crack_analysis.png', dpi=300)
    print(f"  Saved: crack_analysis.png")
    
    return fig


def residual_analysis():
    """Perform residual analysis to identify model inaccuracies."""
    print("\nPerforming residual analysis...")
    
    # Load collocation points (selected locations)
    colloc_data = pd.read_csv(SIMULATION_DIR / "collocation_points" / "collocation_point_data.csv")
    
    # Group by location type
    location_types = colloc_data['location_type'].unique()
    
    print(f"\n  Residual Statistics by Location Type:")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Analyze stress by location type
    stress_by_location = []
    labels = []
    
    for loc_type in location_types:
        subset = colloc_data[colloc_data['location_type'] == loc_type]
        stress_by_location.append(subset['von_mises_MPa'].values)
        labels.append(loc_type)
        
        print(f"    {loc_type}: mean={subset['von_mises_MPa'].mean():.1f} MPa, "
              f"std={subset['von_mises_MPa'].std():.1f} MPa, n={len(subset)}")
    
    # Box plot
    axes[0].boxplot(stress_by_location, labels=labels)
    axes[0].set_ylabel('Von Mises Stress (MPa)')
    axes[0].set_title('Stress Distribution by Location Type')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Distance to defects vs. stress
    axes[1].scatter(colloc_data['distance_to_pore_um'], 
                   colloc_data['von_mises_MPa'], 
                   c=colloc_data['distance_to_grain_boundary_um'],
                   cmap='viridis', s=80, alpha=0.7)
    axes[1].set_xlabel('Distance to Pore (μm)')
    axes[1].set_ylabel('Von Mises Stress (MPa)')
    axes[1].set_title('Stress vs. Distance to Defects')
    axes[1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cbar.set_label('Distance to GB (μm)')
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'residual_analysis.png', dpi=300)
    print(f"  Saved: residual_analysis.png")
    
    return fig


def main():
    """Run all analysis examples."""
    print("="*70)
    print("Multi-Scale FEM Validation Dataset - Example Analysis")
    print("="*70)
    
    # 1. Load and visualize experimental data
    xrd_data, raman_data, synchrotron_data = load_experimental_stress_data()
    visualize_surface_stress_field(xrd_data)
    
    # 2. Compare FEM vs. experimental
    compare_fem_vs_experimental()
    
    # 3. Train surrogate model
    train_surrogate_model()
    
    # 4. Analyze cracks
    analyze_crack_initiation()
    
    # 5. Residual analysis
    residual_analysis()
    
    print("\n" + "="*70)
    print("Analysis complete! Check the generated PNG files.")
    print("="*70)


if __name__ == "__main__":
    main()