#!/usr/bin/env python3
"""
Example script demonstrating how to load and use the validation dataset
for FEM model validation and residual analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_experimental_data():
    """Load experimental residual stress measurements"""
    with open('validation_dataset/experimental_residual_stress.json', 'r') as f:
        exp_data = json.load(f)
    
    print("=== Experimental Data Overview ===")
    print(f"XRD measurements: {len(exp_data['xrd_surface']['measurement_points'])}")
    print(f"Raman measurements: {len(exp_data['raman_spectroscopy']['measurement_points'])}")
    print(f"Synchrotron measurements: {len(exp_data['synchrotron_bulk']['measurement_points'])}")
    
    # Extract XRD stress data for analysis
    xrd_points = np.array(exp_data['xrd_surface']['measurement_points'])
    xrd_stress_xx = np.array(exp_data['xrd_surface']['stress_xx'])
    xrd_stress_yy = np.array(exp_data['xrd_surface']['stress_yy'])
    
    print(f"XRD Stress Range (σ_xx): {np.min(xrd_stress_xx)*1e-6:.1f} to {np.max(xrd_stress_xx)*1e-6:.1f} MPa")
    print(f"XRD Stress Range (σ_yy): {np.min(xrd_stress_yy)*1e-6:.1f} to {np.max(xrd_stress_yy)*1e-6:.1f} MPa")
    
    return exp_data

def load_crack_data():
    """Load crack initiation and propagation data"""
    with open('validation_dataset/crack_initiation_propagation.json', 'r') as f:
        crack_data = json.load(f)
    
    print("\n=== Crack Data Overview ===")
    print(f"Total cracks observed: {len(crack_data['crack_locations'])}")
    
    # Analyze crack types
    crack_types = [cond['crack_type'] for cond in crack_data['critical_conditions']]
    for crack_type in set(crack_types):
        count = crack_types.count(crack_type)
        print(f"  {crack_type}: {count} cracks")
    
    # Critical stress analysis
    critical_stresses = [cond['stress'] for cond in crack_data['critical_conditions']]
    print(f"Critical stress range: {np.min(critical_stresses)*1e-6:.1f} to {np.max(critical_stresses)*1e-6:.1f} MPa")
    
    return crack_data

def load_simulation_data():
    """Load FEM simulation results"""
    # Load full-field data (compressed numpy arrays)
    sim_data = np.load('validation_dataset/full_field_simulation.npz')
    
    print("\n=== Simulation Data Overview ===")
    print(f"Mesh points: {len(sim_data['coordinates'])}")
    print(f"Temperature range: {np.min(sim_data['temperature']):.1f} to {np.max(sim_data['temperature']):.1f} K")
    
    # Stress analysis
    stress_tensors = sim_data['stress']
    stress_xx = stress_tensors[:, 0, 0]  # Extract σ_xx component
    print(f"Stress σ_xx range: {np.min(stress_xx)*1e-6:.1f} to {np.max(stress_xx)*1e-6:.1f} MPa")
    
    # Load collocation points
    with open('validation_dataset/collocation_points.json', 'r') as f:
        collocation_data = json.load(f)
    
    print(f"Collocation points: {len(collocation_data['points'])}")
    
    # Analyze point types
    point_types = [data['point_type'] for data in collocation_data['data']]
    for point_type in set(point_types):
        count = point_types.count(point_type)
        print(f"  {point_type}: {count} points")
    
    return sim_data, collocation_data

def load_material_data():
    """Load multi-scale material characterization data"""
    with open('validation_dataset/multiscale_material_data.json', 'r') as f:
        material_data = json.load(f)
    
    print("\n=== Material Data Overview ===")
    
    # Macro-scale properties
    bulk_props = material_data['macro_scale']['bulk_properties']
    print(f"Young's modulus: {bulk_props['young_modulus']*1e-9:.1f} GPa")
    print(f"Poisson's ratio: {bulk_props['poisson_ratio']:.3f}")
    print(f"CTE: {bulk_props['cte']*1e6:.1f} ppm/K")
    
    # Meso-scale microstructure
    grain_data = material_data['meso_scale']['microstructure']['grain_size_distribution']
    grain_sizes = [g['equivalent_diameter'] for g in grain_data]
    print(f"Grain size: {np.mean(grain_sizes)*1e6:.2f} ± {np.std(grain_sizes)*1e6:.2f} μm")
    
    pore_data = material_data['meso_scale']['microstructure']['pore_size_distribution']
    pore_sizes = [p['equivalent_diameter'] for p in pore_data]
    print(f"Pore size: {np.mean(pore_sizes)*1e6:.2f} ± {np.std(pore_sizes)*1e6:.2f} μm")
    
    return material_data

def validate_fem_model_example():
    """Example of how to use this data for FEM model validation"""
    print("\n=== FEM Model Validation Example ===")
    
    # Load experimental and simulation data
    exp_data = load_experimental_data()
    sim_data, collocation_data = load_simulation_data()
    
    # Extract experimental XRD stress measurements
    exp_points = np.array(exp_data['xrd_surface']['measurement_points'])
    exp_stress = np.array(exp_data['xrd_surface']['stress_xx'])
    
    # Extract simulation stress at similar locations
    sim_coords = sim_data['coordinates']
    sim_stress = sim_data['stress'][:, 0, 0]  # σ_xx component
    
    # Find simulation points closest to experimental measurements
    validation_results = []
    
    for i, exp_point in enumerate(exp_points):
        # Find closest simulation point (simple nearest neighbor)
        distances = np.sqrt(np.sum((sim_coords - exp_point)**2, axis=1))
        closest_idx = np.argmin(distances)
        
        sim_stress_val = sim_stress[closest_idx]
        exp_stress_val = exp_stress[i]
        
        # Calculate residual (difference between simulation and experiment)
        residual = sim_stress_val - exp_stress_val
        
        validation_results.append({
            'exp_point': exp_point,
            'exp_stress': exp_stress_val,
            'sim_stress': sim_stress_val,
            'residual': residual,
            'relative_error': abs(residual) / abs(exp_stress_val) if exp_stress_val != 0 else np.inf
        })
    
    # Analyze validation results
    residuals = [r['residual'] for r in validation_results]
    rel_errors = [r['relative_error'] for r in validation_results if r['relative_error'] != np.inf]
    
    print(f"Mean residual: {np.mean(residuals)*1e-6:.2f} MPa")
    print(f"RMS residual: {np.sqrt(np.mean(np.array(residuals)**2))*1e-6:.2f} MPa")
    print(f"Mean relative error: {np.mean(rel_errors)*100:.1f}%")
    
    # Identify regions with high residuals (model deficiencies)
    high_residual_threshold = 2 * np.std(residuals)
    high_residual_points = [r for r in validation_results if abs(r['residual']) > high_residual_threshold]
    
    print(f"Points with high residuals (>{high_residual_threshold*1e-6:.1f} MPa): {len(high_residual_points)}")
    
    return validation_results

def residual_analysis_example():
    """Example of residual analysis for model improvement"""
    print("\n=== Residual Analysis Example ===")
    
    # Load collocation point data
    with open('validation_dataset/collocation_points.json', 'r') as f:
        collocation_data = json.load(f)
    
    # Analyze stress concentrations at different point types
    point_analysis = {}
    
    for data in collocation_data['data']:
        point_type = data['point_type']
        stress_tensor = np.array(data['stress'])
        von_mises = np.sqrt(0.5 * ((stress_tensor[0,0] - stress_tensor[1,1])**2 + 
                                   (stress_tensor[1,1] - stress_tensor[2,2])**2 + 
                                   (stress_tensor[2,2] - stress_tensor[0,0])**2 + 
                                   6 * (stress_tensor[0,1]**2 + stress_tensor[1,2]**2 + stress_tensor[2,0]**2)))
        
        if point_type not in point_analysis:
            point_analysis[point_type] = []
        point_analysis[point_type].append(von_mises)
    
    print("Von Mises stress by location type:")
    for point_type, stresses in point_analysis.items():
        mean_stress = np.mean(stresses)
        max_stress = np.max(stresses)
        print(f"  {point_type}: {mean_stress*1e-6:.1f} ± {np.std(stresses)*1e-6:.1f} MPa (max: {max_stress*1e-6:.1f} MPa)")
    
    # Identify critical regions for model refinement
    critical_stress_threshold = 100e6  # Pa
    critical_points = []
    
    for i, data in enumerate(collocation_data['data']):
        stress_tensor = np.array(data['stress'])
        von_mises = np.sqrt(0.5 * ((stress_tensor[0,0] - stress_tensor[1,1])**2 + 
                                   (stress_tensor[1,1] - stress_tensor[2,2])**2 + 
                                   (stress_tensor[2,2] - stress_tensor[0,0])**2 + 
                                   6 * (stress_tensor[0,1]**2 + stress_tensor[1,2]**2 + stress_tensor[2,0]**2)))
        
        if von_mises > critical_stress_threshold:
            critical_points.append({
                'index': i,
                'location': collocation_data['points'][i],
                'point_type': data['point_type'],
                'von_mises_stress': von_mises
            })
    
    print(f"\nCritical stress regions (>{critical_stress_threshold*1e-6:.0f} MPa): {len(critical_points)}")
    
    # Recommend mesh refinement regions
    if critical_points:
        print("Recommended mesh refinement locations:")
        for point in critical_points[:5]:  # Show top 5
            loc = point['location']
            print(f"  {point['point_type']} at ({loc[0]*1e6:.1f}, {loc[1]*1e6:.1f}, {loc[2]*1e6:.1f}) μm: {point['von_mises_stress']*1e-6:.1f} MPa")

def create_summary_plot():
    """Create a summary visualization of the dataset"""
    print("\n=== Creating Summary Plot ===")
    
    # Load data
    exp_data = load_experimental_data()
    crack_data = load_crack_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # XRD stress map
    xrd_points = np.array(exp_data['xrd_surface']['measurement_points'])
    xrd_stress = np.array(exp_data['xrd_surface']['stress_xx'])
    
    scatter1 = axes[0, 0].scatter(xrd_points[:, 0]*1e6, xrd_points[:, 1]*1e6, 
                                 c=xrd_stress*1e-6, cmap='RdBu_r', s=100)
    axes[0, 0].set_xlabel('X (μm)')
    axes[0, 0].set_ylabel('Y (μm)')
    axes[0, 0].set_title('XRD Surface Stress σ_xx (MPa)')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # Crack locations
    crack_points = np.array(crack_data['crack_locations'])
    crack_lengths = np.array(crack_data['crack_lengths'])
    
    scatter2 = axes[0, 1].scatter(crack_points[:, 0]*1e6, crack_points[:, 1]*1e6, 
                                 c=crack_lengths*1e6, cmap='Reds', s=100)
    axes[0, 1].set_xlabel('X (μm)')
    axes[0, 1].set_ylabel('Y (μm)')
    axes[0, 1].set_title('Crack Locations & Lengths (μm)')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # Critical stress histogram
    critical_stresses = [cond['stress'] for cond in crack_data['critical_conditions']]
    axes[1, 0].hist(np.array(critical_stresses)*1e-6, bins=10, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Critical Stress (MPa)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Critical Stress Distribution')
    
    # Stress comparison (exp vs sim example)
    # This is a simplified example - in practice you'd do proper spatial interpolation
    sim_stress_example = xrd_stress + np.random.normal(0, 10e6, len(xrd_stress))  # Simulated data with noise
    
    axes[1, 1].scatter(xrd_stress*1e-6, sim_stress_example*1e-6, alpha=0.6)
    axes[1, 1].plot([-150, 50], [-150, 50], 'r--', label='Perfect agreement')
    axes[1, 1].set_xlabel('Experimental Stress (MPa)')
    axes[1, 1].set_ylabel('Simulation Stress (MPa)')
    axes[1, 1].set_title('Model Validation')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('dataset_usage_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Summary plot saved as 'dataset_usage_summary.png'")

def main():
    """Main function demonstrating dataset usage"""
    print("=== SOFC Electrolyte Validation Dataset Usage Example ===")
    
    # Check if dataset exists
    if not Path('validation_dataset').exists():
        print("Error: validation_dataset directory not found!")
        print("Please run validation_analysis_dataset.py first to generate the dataset.")
        return
    
    # Load and analyze all data components
    exp_data = load_experimental_data()
    crack_data = load_crack_data()
    sim_data, collocation_data = load_simulation_data()
    material_data = load_material_data()
    
    # Demonstrate validation workflow
    validation_results = validate_fem_model_example()
    
    # Demonstrate residual analysis
    residual_analysis_example()
    
    # Create summary visualization
    create_summary_plot()
    
    print("\n=== Usage Example Complete ===")
    print("This example demonstrates how to:")
    print("1. Load experimental measurements for validation")
    print("2. Compare simulation results with experiments")
    print("3. Perform residual analysis to identify model deficiencies")
    print("4. Use collocation points for surrogate model training")
    print("5. Identify regions requiring mesh refinement")
    print("\nThe dataset is now ready for your FEM validation and analysis!")

if __name__ == "__main__":
    main()