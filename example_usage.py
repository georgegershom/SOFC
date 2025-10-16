"""
Example Usage: Loading and Using the Experimental Validation Dataset
=====================================================================

This script demonstrates how to load and use the experimental dataset
for ML model validation.
"""

import numpy as np
import pandas as pd
import h5py
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_fabrication_parameters():
    """Load fabrication parameters for all samples."""
    print("="*80)
    print("LOADING FABRICATION PARAMETERS")
    print("="*80)
    
    fab_params = pd.read_csv('experimental_validation_dataset/fabrication_parameters.csv')
    
    print(f"\nLoaded {len(fab_params)} samples")
    print(f"\nFirst 5 samples:")
    print(fab_params[['sample_id', 'anode_thickness_um', 'electrolyte_thickness_um', 
                      'cathode_thickness_um', 'electrolyte_sinter_temp_C']].head())
    
    return fab_params


def load_warp_data(sample_id):
    """Load 3D warp measurement for a specific sample."""
    print(f"\n{'='*80}")
    print(f"LOADING WARP DATA: {sample_id}")
    print(f"{'='*80}")
    
    with h5py.File('experimental_validation_dataset/warp_measurements_3d.h5', 'r') as f:
        if sample_id not in f:
            print(f"Sample {sample_id} not found!")
            return None
        
        grp = f[sample_id]
        
        warp_data = {
            'X': grp['X'][:],
            'Y': grp['Y'][:],
            'Z': grp['Z'][:],
            'max_warp_um': grp.attrs['max_warp_um'],
            'rms_warp_um': grp.attrs['rms_warp_um'],
            'technique': grp.attrs['technique'],
            'resolution_um': grp.attrs['resolution_um']
        }
    
    print(f"\n  Technique: {warp_data['technique']}")
    print(f"  Resolution: {warp_data['resolution_um']:.2f} μm")
    print(f"  Grid size: {warp_data['X'].shape}")
    print(f"  Max warp: {warp_data['max_warp_um']:.2f} μm")
    print(f"  RMS warp: {warp_data['rms_warp_um']:.2f} μm")
    
    return warp_data


def load_stress_measurements(sample_id):
    """Load all stress measurements for a specific sample."""
    print(f"\n{'='*80}")
    print(f"LOADING STRESS MEASUREMENTS: {sample_id}")
    print(f"{'='*80}")
    
    stress_data = {}
    
    # 1. Curvature-based stress
    curv_stress = pd.read_csv('experimental_validation_dataset/curvature_stress_measurements.csv')
    sample_curv = curv_stress[curv_stress['sample_id'] == sample_id]
    
    if not sample_curv.empty:
        stress_data['curvature'] = {
            'anode_GPa': sample_curv['anode_stress_GPa'].values[0],
            'electrolyte_GPa': sample_curv['electrolyte_stress_GPa'].values[0],
            'cathode_GPa': sample_curv['cathode_stress_GPa'].values[0],
            'uncertainty_pct': sample_curv['uncertainty_percent'].values[0]
        }
        print(f"\n1. Curvature-based stress (Stoney's formula):")
        print(f"   Anode: {stress_data['curvature']['anode_GPa']:.3f} GPa")
        print(f"   Electrolyte: {stress_data['curvature']['electrolyte_GPa']:.3f} GPa")
        print(f"   Cathode: {stress_data['curvature']['cathode_GPa']:.3f} GPa")
        print(f"   Uncertainty: ±{stress_data['curvature']['uncertainty_pct']:.1f}%")
    
    # 2. XRD measurements
    xrd_stress = pd.read_csv('experimental_validation_dataset/xrd_stress_measurements.csv')
    sample_xrd = xrd_stress[xrd_stress['sample_id'] == sample_id]
    
    if not sample_xrd.empty:
        stress_data['xrd'] = sample_xrd[['x_position_mm', 'y_position_mm', 
                                         'stress_xx_GPa', 'stress_yy_GPa', 
                                         'uncertainty_GPa']].to_dict('records')
        print(f"\n2. XRD measurements:")
        print(f"   Points measured: {len(sample_xrd)}")
        print(f"   Stress range: {sample_xrd['stress_xx_GPa'].min():.3f} - {sample_xrd['stress_xx_GPa'].max():.3f} GPa")
    
    # 3. Raman measurements
    raman_stress = pd.read_csv('experimental_validation_dataset/raman_stress_measurements.csv')
    sample_raman = raman_stress[raman_stress['sample_id'] == sample_id]
    
    if not sample_raman.empty:
        stress_data['raman'] = sample_raman[['x_position_mm', 'y_position_mm', 
                                             'stress_estimate_GPa', 
                                             'uncertainty_GPa']].to_dict('records')
        print(f"\n3. Raman spectroscopy:")
        print(f"   Points measured: {len(sample_raman)}")
        print(f"   Stress range: {sample_raman['stress_estimate_GPa'].min():.3f} - {sample_raman['stress_estimate_GPa'].max():.3f} GPa")
    
    # 4. Layer removal profile
    layer_file = Path('experimental_validation_dataset/layer_removal_stress_profiles.h5')
    if layer_file.exists():
        with h5py.File(layer_file, 'r') as f:
            if sample_id in f:
                grp = f[sample_id]
                stress_data['layer_removal'] = {
                    'z_positions_um': grp['z_positions_um'][:],
                    'stress_profile_GPa': grp['stress_profile_GPa'][:],
                    'uncertainty_GPa': grp.attrs['uncertainty_GPa']
                }
                print(f"\n4. Layer removal profile:")
                print(f"   Depth points: {len(stress_data['layer_removal']['z_positions_um'])}")
                print(f"   Stress range: {stress_data['layer_removal']['stress_profile_GPa'].min():.3f} - "
                      f"{stress_data['layer_removal']['stress_profile_GPa'].max():.3f} GPa")
    
    return stress_data


def visualize_sample(sample_id):
    """Create a comprehensive visualization for a single sample."""
    print(f"\n{'='*80}")
    print(f"VISUALIZING SAMPLE: {sample_id}")
    print(f"{'='*80}")
    
    # Load data
    warp_data = load_warp_data(sample_id)
    stress_data = load_stress_measurements(sample_id)
    
    if warp_data is None:
        print(f"Cannot visualize {sample_id} - data not found")
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 3D Warp surface
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    X_mm = warp_data['X'] * 1000
    Y_mm = warp_data['Y'] * 1000
    surf = ax1.plot_surface(X_mm, Y_mm, warp_data['Z'], cmap='RdYlBu_r', alpha=0.9)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Warp (μm)')
    ax1.set_title(f'{sample_id}\n3D Warp Measurement', fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2. Warp contour
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contourf(X_mm, Y_mm, warp_data['Z'], levels=20, cmap='RdYlBu_r')
    ax2.contour(X_mm, Y_mm, warp_data['Z'], levels=10, colors='black', 
                linewidths=0.5, alpha=0.3)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Warp Contour Map', fontweight='bold')
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2, label='Warp (μm)')
    
    # 3. Curvature-based stress (bar chart)
    if 'curvature' in stress_data:
        ax3 = fig.add_subplot(2, 3, 3)
        layers = ['Anode', 'Electrolyte', 'Cathode']
        stresses = [
            stress_data['curvature']['anode_GPa'],
            stress_data['curvature']['electrolyte_GPa'],
            stress_data['curvature']['cathode_GPa']
        ]
        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        bars = ax3.bar(layers, stresses, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_ylabel('Stress (GPa)')
        ax3.set_title('Curvature-Based Stress', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. XRD stress map
    if 'xrd' in stress_data and stress_data['xrd']:
        ax4 = fig.add_subplot(2, 3, 4)
        xrd_df = pd.DataFrame(stress_data['xrd'])
        scatter = ax4.scatter(xrd_df['x_position_mm'], xrd_df['y_position_mm'],
                             c=xrd_df['stress_xx_GPa'], cmap='viridis', 
                             s=100, edgecolors='black')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')
        ax4.set_title('XRD Stress Measurements', fontweight='bold')
        ax4.set_aspect('equal')
        fig.colorbar(scatter, ax=ax4, label='Stress σxx (GPa)')
    
    # 5. Raman stress map
    if 'raman' in stress_data and stress_data['raman']:
        ax5 = fig.add_subplot(2, 3, 5)
        raman_df = pd.DataFrame(stress_data['raman'])
        ax5.hexbin(raman_df['x_position_mm'], raman_df['y_position_mm'],
                  C=raman_df['stress_estimate_GPa'], gridsize=10, cmap='plasma')
        ax5.set_xlabel('X (mm)')
        ax5.set_ylabel('Y (mm)')
        ax5.set_title('Raman Stress Estimates', fontweight='bold')
        ax5.set_aspect('equal')
    
    # 6. Layer removal profile
    if 'layer_removal' in stress_data:
        ax6 = fig.add_subplot(2, 3, 6)
        lr = stress_data['layer_removal']
        ax6.plot(lr['stress_profile_GPa'], lr['z_positions_um'], 'b-', linewidth=2)
        ax6.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax6.fill_betweenx(lr['z_positions_um'], 
                         lr['stress_profile_GPa'] - lr['uncertainty_GPa'],
                         lr['stress_profile_GPa'] + lr['uncertainty_GPa'],
                         alpha=0.3, color='blue')
        ax6.set_xlabel('Stress (GPa)')
        ax6.set_ylabel('Depth (μm)')
        ax6.set_title('Through-Thickness Stress Profile', fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('experimental_validation_dataset/sample_visualizations')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{sample_id}_complete_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()


def ml_validation_workflow_example():
    """
    Example workflow for ML model validation using experimental data.
    """
    print("\n" + "="*80)
    print("ML MODEL VALIDATION WORKFLOW EXAMPLE")
    print("="*80)
    
    print("""
    Step 1: Train ML Model on FEA Data
    -----------------------------------
    # Train on Dataset 1 (High-Fidelity FEA) and Dataset 2 (Synthetic Diverse)
    model = YourMLModel()
    model.train(fea_warp_data, fea_stress_data)
    
    
    Step 2: Load Experimental Warp Data
    ------------------------------------
    """)
    
    # Load a sample
    sample_id = 'SOFC-EXP-001'
    warp_data = load_warp_data(sample_id)
    
    print("""
    Step 3: Predict Stress Using ML Model
    --------------------------------------
    # Use your trained model to predict stress from experimental warp
    stress_predicted = model.predict(warp_data['Z'])
    
    
    Step 4: Load Experimental Stress Measurements
    ----------------------------------------------
    """)
    
    stress_data = load_stress_measurements(sample_id)
    
    print("""
    Step 5: Calculate Validation Metrics
    -------------------------------------
    # Compare predicted stress with experimental measurements
    
    # A. Layer-averaged comparison (curvature method)
    rmse_curvature = calculate_rmse(
        stress_predicted.layer_average(),
        stress_data['curvature']
    )
    
    # B. Point-wise comparison (XRD/Raman)
    rmse_pointwise = calculate_rmse(
        stress_predicted.at_points(xrd_positions),
        stress_data['xrd']['stress_xx_GPa']
    )
    
    # C. Through-thickness profile comparison
    rmse_profile = calculate_rmse(
        stress_predicted.depth_profile(),
        stress_data['layer_removal']['stress_profile_GPa']
    )
    
    
    Step 6: Analyze Discrepancies
    ------------------------------
    # Discrepancies indicate:
    # - Physics missing from FEA (e.g., grain boundaries, microcracking)
    # - ML model limitations (generalization issues)
    # - Measurement uncertainties (must be accounted for)
    
    
    Step 7: Iterate and Improve
    ----------------------------
    # Based on validation results:
    # 1. Refine FEA model (add missing physics)
    # 2. Retrain ML model with improved FEA data
    # 3. Re-validate on experimental data
    # 4. Repeat until satisfactory accuracy achieved
    """)
    
    print("="*80)


def main():
    """
    Main demonstration.
    """
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "EXPERIMENTAL DATASET USAGE EXAMPLE" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    
    # 1. Load fabrication parameters
    fab_params = load_fabrication_parameters()
    
    # 2. Demonstrate loading data for a specific sample
    sample_id = 'SOFC-EXP-001'
    warp_data = load_warp_data(sample_id)
    stress_data = load_stress_measurements(sample_id)
    
    # 3. Visualize sample
    print("\nCreating comprehensive visualization...")
    visualize_sample(sample_id)
    
    # 4. Show ML validation workflow
    ml_validation_workflow_example()
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)
    print("\n✓ Dataset successfully loaded and demonstrated!")
    print("✓ Use this as a template for your ML validation pipeline")


if __name__ == '__main__':
    main()
