#!/usr/bin/env python3
"""
Example Usage of Synthetic Synchrotron X-ray Data
=================================================

This script demonstrates how to load and analyze the generated
synthetic synchrotron data for SOFC creep studies.
"""

import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path


def example_1_load_tomography():
    """
    Example 1: Load and inspect 4D tomography data
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Loading 4D Tomography Data")
    print("="*70)
    
    # Open the HDF5 file
    with h5py.File('synchrotron_data/tomography/tomography_4D.h5', 'r') as f:
        # Load data
        tomography = f['tomography'][:]
        time = f['time_hours'][:]
        
        # Load metadata
        temp = f.attrs['temperature_C']
        stress = f.attrs['applied_stress_MPa']
        voxel_size = f.attrs['voxel_size_um']
        
        print(f"\nData shape: {tomography.shape}")
        print(f"  Time steps: {len(time)}")
        print(f"  Volume dimensions: {tomography.shape[1:]} voxels")
        print(f"  Voxel size: {voxel_size} μm")
        print(f"  Physical size: {np.array(tomography.shape[1:]) * voxel_size / 1000} mm")
        
        print(f"\nTest conditions:")
        print(f"  Temperature: {temp}°C")
        print(f"  Applied stress: {stress} MPa")
        print(f"  Total duration: {time[-1]} hours")
        
        # Calculate porosity at each time step
        print(f"\nPorosity evolution:")
        for i, t in enumerate(time):
            volume = tomography[i]
            porosity = np.sum(volume < 0.3) / volume.size * 100
            print(f"  t = {t:5.1f} h: Porosity = {porosity:5.2f}%")
        
        # Compare initial and final states
        initial = tomography[0]
        final = tomography[-1]
        damage = initial - final
        
        print(f"\nMicrostructural damage:")
        print(f"  Mean damage: {np.mean(damage):.4f}")
        print(f"  Max damage: {np.max(damage):.4f}")
        print(f"  Damaged voxels: {np.sum(damage > 0.1)} ({np.sum(damage > 0.1)/damage.size*100:.2f}%)")


def example_2_analyze_grain_structure():
    """
    Example 2: Analyze grain structure
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Analyzing Grain Structure")
    print("="*70)
    
    with h5py.File('synchrotron_data/tomography/grain_map.h5', 'r') as f:
        grain_map = f['grain_ids'][:]
        num_grains = f.attrs['num_grains']
        avg_grain_size = f.attrs['average_grain_size_um']
        
        print(f"\nGrain statistics:")
        print(f"  Total number of grains: {num_grains}")
        print(f"  Average grain size: {avg_grain_size} μm")
        
        # Calculate grain size distribution
        grain_ids, counts = np.unique(grain_map, return_counts=True)
        volumes_voxels = counts
        
        print(f"\nGrain size distribution:")
        print(f"  Smallest grain: {np.min(volumes_voxels)} voxels")
        print(f"  Largest grain: {np.max(volumes_voxels)} voxels")
        print(f"  Mean grain volume: {np.mean(volumes_voxels):.0f} voxels")
        print(f"  Median grain volume: {np.median(volumes_voxels):.0f} voxels")


def example_3_xrd_analysis():
    """
    Example 3: Analyze X-ray diffraction data
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: X-ray Diffraction Analysis")
    print("="*70)
    
    # Load diffraction patterns
    with open('synchrotron_data/diffraction/xrd_patterns.json', 'r') as f:
        xrd_data = json.load(f)
    
    print("\nPhase identification:")
    phases = xrd_data['phases_detected']
    for phase_name, phase_data in phases.items():
        print(f"\n  Phase: {phase_name.replace('_', ' ')}")
        print(f"    Volume fraction: {phase_data['fraction']*100:.1f}%")
        print(f"    Crystal system: {phase_data['crystal_system']}")
        print(f"    Lattice parameter: {phase_data['lattice_parameter_angstrom']} Å")
        print(f"    Main peaks: {phase_data['peaks_deg']} (2θ degrees)")
    
    # Analyze strain/stress evolution
    print("\n" + "-"*70)
    print("Strain and stress evolution:")
    print("-"*70)
    
    with h5py.File('synchrotron_data/diffraction/strain_stress_maps.h5', 'r') as f:
        strain = f['elastic_strain'][:]
        stress = f['residual_stress_MPa'][:]
        time = f['time_hours'][:]
        
        print(f"\nStrain evolution (mean values):")
        for i, t in enumerate(time):
            mean_strain = np.mean(strain[i])
            mean_stress = np.mean(stress[i])
            print(f"  t = {t:5.1f} h: ε = {mean_strain:.6f}, σ = {mean_stress:.1f} MPa")


def example_4_track_damage_metrics():
    """
    Example 4: Track damage metrics over time
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Tracking Damage Metrics")
    print("="*70)
    
    # Load metrics
    with open('synchrotron_data/tomography/tomography_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    time = metrics['time_hours']
    porosity = metrics['porosity_percent']
    cavity_count = metrics['cavity_count']
    crack_volume = metrics['crack_volume_mm3']
    
    print("\nCreep damage progression:")
    print("-"*70)
    print(f"{'Time (h)':>10} {'Porosity (%)':>15} {'Cavities':>12} {'Crack Vol (mm³)':>18}")
    print("-"*70)
    
    for i in range(len(time)):
        print(f"{time[i]:>10.1f} {porosity[i]:>15.2f} {cavity_count[i]:>12d} {crack_volume[i]:>18.6f}")
    
    # Calculate rates
    print("\n" + "-"*70)
    print("Damage rates:")
    print("-"*70)
    
    dt = time[-1] - time[0]
    porosity_rate = (porosity[-1] - porosity[0]) / dt
    cavity_rate = (cavity_count[-1] - cavity_count[0]) / dt
    crack_rate = (crack_volume[-1] - crack_volume[0]) / dt
    
    print(f"  Porosity increase rate: {porosity_rate:.4f} %/hour")
    print(f"  Cavity nucleation rate: {cavity_rate:.2f} cavities/hour")
    print(f"  Crack growth rate: {crack_rate:.6e} mm³/hour")


def example_5_compare_initial_final():
    """
    Example 5: Visualize initial vs final microstructure
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Comparing Initial and Final Microstructure")
    print("="*70)
    
    with h5py.File('synchrotron_data/tomography/tomography_4D.h5', 'r') as f:
        initial = f['tomography'][0]
        final = f['tomography'][-1]
        time_final = f['time_hours'][-1]
    
    # Get middle slice
    mid_z = initial.shape[0] // 2
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initial state
    im1 = axes[0].imshow(initial[mid_z], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Initial State (t=0)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Density', fraction=0.046)
    
    # Final state
    im2 = axes[1].imshow(final[mid_z], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Final State (t={time_final:.0f}h)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Density', fraction=0.046)
    
    # Damage map
    damage = initial[mid_z] - final[mid_z]
    im3 = axes[2].imshow(damage, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Damage (Density Loss)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], label='Δ Density', fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('comparison_initial_final.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison figure to: comparison_initial_final.png")
    
    # Statistics
    print("\nQuantitative comparison:")
    print(f"  Initial mean density: {np.mean(initial):.4f}")
    print(f"  Final mean density: {np.mean(final):.4f}")
    print(f"  Mean density loss: {np.mean(initial) - np.mean(final):.4f}")
    print(f"  Maximum local damage: {np.max(damage):.4f}")
    print(f"  Severely damaged area: {np.sum(damage > 0.3) / damage.size * 100:.2f}%")


def example_6_load_metadata():
    """
    Example 6: Access experimental metadata
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Accessing Experimental Metadata")
    print("="*70)
    
    # Load experimental parameters
    with open('synchrotron_data/metadata/experimental_parameters.json', 'r') as f:
        exp_params = json.load(f)
    
    print("\nExperimental Setup:")
    print(f"  Facility: {exp_params['facility']}")
    print(f"  Beamline: {exp_params['beamline']}")
    print(f"  Date: {exp_params['date']}")
    
    print("\nTest Conditions:")
    tc = exp_params['test_conditions']
    print(f"  Temperature: {tc['temperature_C']}°C ± {tc['temperature_stability_C']}°C")
    print(f"  Stress: {tc['applied_stress_MPa']} MPa ({tc['stress_type']})")
    print(f"  Atmosphere: {tc['atmosphere']}")
    print(f"  Duration: {tc['test_duration_hours']} hours")
    
    print("\nImaging Parameters:")
    ip = exp_params['imaging_parameters']
    print(f"  Beam energy: {ip['beam_energy_keV']} keV")
    print(f"  Voxel size: {ip['voxel_size_um']} μm")
    print(f"  Field of view: {ip['field_of_view_mm']} mm")
    print(f"  Detector: {ip['detector']}")
    
    # Load material specifications
    with open('synchrotron_data/metadata/material_specifications.json', 'r') as f:
        material = json.load(f)
    
    print("\n" + "-"*70)
    print("Material Specifications:")
    print("-"*70)
    print(f"  Material: {material['material_name']}")
    print(f"  Application: {material['application']}")
    
    print("\n  Composition (wt%):")
    for element, percentage in material['composition_wt_percent'].items():
        print(f"    {element}: {percentage}%")
    
    print("\n  Mechanical Properties:")
    mp = material['mechanical_properties']
    print(f"    Elastic modulus: {mp['elastic_modulus_GPa']} GPa")
    print(f"    Yield strength: {mp['yield_strength_MPa']} MPa")
    print(f"    Tensile strength: {mp['tensile_strength_MPa']} MPa")
    
    # Load sample geometry
    with open('synchrotron_data/metadata/sample_geometry.json', 'r') as f:
        sample = json.load(f)
    
    print("\n" + "-"*70)
    print("Sample Geometry:")
    print("-"*70)
    print(f"  Sample ID: {sample['sample_id']}")
    print(f"  Geometry: {sample['geometry']}")
    print(f"  Diameter: {sample['dimensions_mm']['diameter']} mm")
    print(f"  Height: {sample['dimensions_mm']['height']} mm")
    print(f"  Mass: {sample['mass_mg']} mg")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SYNTHETIC SYNCHROTRON DATA EXAMPLES" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    
    # Run all examples
    example_1_load_tomography()
    example_2_analyze_grain_structure()
    example_3_xrd_analysis()
    example_4_track_damage_metrics()
    example_5_compare_initial_final()
    example_6_load_metadata()
    
    print("\n" + "="*70)
    print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - comparison_initial_final.png")
    print("\nFor more examples, see:")
    print("  - visualize_data.py: Comprehensive visualization tools")
    print("  - analyze_metrics.py: Quantitative analysis and model fitting")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
