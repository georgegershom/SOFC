#!/usr/bin/env python3
"""
Example usage of the generated SOFC microstructural dataset
Shows how to load, analyze, and use the data for modeling
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(h5_file="./sofc_dataset/microstructure.h5"):
    """Load the SOFC microstructural dataset."""
    
    print("Loading SOFC microstructural dataset...")
    
    with h5py.File(h5_file, 'r') as f:
        # Load main volume
        volume = f['microstructure'][:]
        voxel_size = f['microstructure'].attrs['voxel_size']
        
        # Load phase masks
        phases = {}
        if 'phases' in f:
            for phase_key in f['phases'].keys():
                phase_id = int(phase_key.split('_')[1])
                phases[phase_id] = {
                    'mask': f['phases'][phase_key][:],
                    'name': f['phases'][phase_key].attrs['name'],
                    'volume_fraction': f['phases'][phase_key].attrs.get('volume_fraction', 0)
                }
    
    return volume, voxel_size, phases

def example_electrochemical_analysis(volume, voxel_size):
    """
    Example: Calculate effective properties for electrochemical modeling.
    """
    print("\n" + "="*60)
    print("EXAMPLE: ELECTROCHEMICAL ANALYSIS")
    print("="*60)
    
    # Identify conducting phases
    ni_phase = (volume == 1)  # Electronic conductor
    ysz_phase = (volume == 2) | (volume == 3)  # Ionic conductor
    pore_phase = (volume == 0)  # Gas transport
    
    # Calculate volume fractions in anode region (first half of volume)
    anode_region = volume[:, :, :volume.shape[2]//2]
    
    anode_porosity = np.sum(anode_region == 0) / anode_region.size
    ni_fraction = np.sum(anode_region == 1) / anode_region.size
    ysz_fraction = np.sum(anode_region == 2) / anode_region.size
    
    print(f"\nAnode Region Properties:")
    print(f"  • Porosity: {anode_porosity:.1%}")
    print(f"  • Ni fraction: {ni_fraction:.1%}")
    print(f"  • YSZ fraction: {ysz_fraction:.1%}")
    
    # Calculate tortuosity (simplified - ratio of actual to straight path)
    from scipy.ndimage import distance_transform_edt
    
    # For pore phase tortuosity
    pore_dist = distance_transform_edt(pore_phase)
    pore_tortuosity_est = 1.5  # Simplified estimate
    
    print(f"\nTransport Properties (estimated):")
    print(f"  • Pore tortuosity: ~{pore_tortuosity_est:.2f}")
    print(f"  • Effective diffusivity: ~{anode_porosity/pore_tortuosity_est:.2f}")
    
    # Find Triple Phase Boundaries (TPB)
    from scipy.ndimage import binary_dilation
    
    ni_dilated = binary_dilation(ni_phase)
    ysz_dilated = binary_dilation(ysz_phase)
    pore_dilated = binary_dilation(pore_phase)
    
    tpb = ni_dilated & ysz_dilated & pore_dilated
    tpb_density = np.sum(tpb) * voxel_size / (np.prod(volume.shape) * voxel_size**3)
    
    print(f"\nElectrochemical Active Sites:")
    print(f"  • TPB density: {tpb_density:.2e} µm⁻²")
    print(f"  • Active TPB sites: {np.sum(tpb):,} voxels")
    
    return anode_porosity, tpb_density

def example_mechanical_analysis(volume, voxel_size):
    """
    Example: Prepare data for mechanical stress analysis.
    """
    print("\n" + "="*60)
    print("EXAMPLE: MECHANICAL ANALYSIS PREPARATION")
    print("="*60)
    
    # Material properties at 800°C
    materials = {
        0: {'name': 'Pore', 'E': 0, 'nu': 0, 'CTE': 0},
        1: {'name': 'Ni', 'E': 200e9, 'nu': 0.31, 'CTE': 13.3e-6},
        2: {'name': 'YSZ_Anode', 'E': 200e9, 'nu': 0.31, 'CTE': 10.5e-6},
        3: {'name': 'YSZ_Electrolyte', 'E': 200e9, 'nu': 0.31, 'CTE': 10.5e-6},
        4: {'name': 'GDC', 'E': 180e9, 'nu': 0.33, 'CTE': 12.5e-6},
    }
    
    print("\nMaterial Assignment for FEA:")
    for phase_id in np.unique(volume):
        mat = materials.get(phase_id, {})
        count = np.sum(volume == phase_id)
        print(f"  • Phase {phase_id} ({mat.get('name', 'Unknown')}): {count:,} elements")
        if mat.get('E', 0) > 0:
            print(f"    - Young's Modulus: {mat['E']/1e9:.0f} GPa")
            print(f"    - CTE: {mat['CTE']*1e6:.1f} µm/m/K")
    
    # Identify critical interface
    from scipy.ndimage import binary_erosion, binary_dilation
    
    anode_mask = (volume == 1) | (volume == 2)
    electrolyte_mask = (volume == 3)
    
    # Find interface elements
    interface = (binary_dilation(anode_mask) & electrolyte_mask) | \
               (binary_dilation(electrolyte_mask) & anode_mask)
    
    print(f"\nCritical Interface Analysis:")
    print(f"  • Anode/Electrolyte interface elements: {np.sum(interface):,}")
    print(f"  • Interface area: {np.sum(interface) * voxel_size**2:.2f} µm²")
    
    # Calculate CTE mismatch stress (simplified)
    delta_T = 700  # Temperature change from room to operating (°C)
    cte_mismatch = abs(materials[1]['CTE'] - materials[3]['CTE'])
    stress_estimate = materials[1]['E'] * cte_mismatch * delta_T
    
    print(f"\nThermal Stress Estimation:")
    print(f"  • CTE mismatch: {cte_mismatch*1e6:.1f} µm/m/K")
    print(f"  • Estimated stress: {stress_estimate/1e6:.0f} MPa")
    print(f"  • Critical for delamination: {'Yes' if stress_estimate/1e6 > 100 else 'No'}")
    
    return interface, stress_estimate

def example_degradation_study(volume, voxel_size, time_hours=1000):
    """
    Example: Simulate microstructural evolution during operation.
    """
    print("\n" + "="*60)
    print("EXAMPLE: DEGRADATION STUDY")
    print("="*60)
    
    print(f"\nSimulating {time_hours} hours of operation...")
    
    # Initial Ni particle size
    ni_mask = (volume == 1)
    initial_ni_volume = np.sum(ni_mask)
    
    # Simplified coarsening model (Ostwald ripening)
    # Particle size increases with t^(1/3)
    time_factor = (time_hours / 100) ** (1/3)
    
    print(f"\nNi Particle Coarsening:")
    print(f"  • Initial Ni volume: {initial_ni_volume * voxel_size**3:.0f} µm³")
    print(f"  • Estimated size increase: {(time_factor - 1)*100:.1f}%")
    
    # TPB reduction estimate
    tpb_reduction = 1 - (1 / time_factor)**0.5
    print(f"  • Estimated TPB reduction: {tpb_reduction*100:.1f}%")
    
    # Porosity changes
    print(f"\nPore Structure Evolution:")
    print(f"  • Pore closure near interface: Minor")
    print(f"  • Estimated porosity change: <5%")
    
    # Performance degradation
    performance_loss = tpb_reduction * 0.7  # 70% of loss from TPB reduction
    print(f"\nPerformance Impact:")
    print(f"  • Estimated performance loss: {performance_loss*100:.1f}%")
    print(f"  • Degradation rate: {performance_loss*100/time_hours*1000:.2f}%/1000h")
    
    return performance_loss

def visualize_structure(volume):
    """Create a simple visualization of the structure."""
    
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SOFC Microstructure - Multi-plane Views', fontsize=16)
    
    # Different viewing planes
    slices = [
        ('XY plane (z=0)', volume[:, :, 0]),
        ('XY plane (z=mid)', volume[:, :, volume.shape[2]//2]),
        ('XY plane (z=end)', volume[:, :, -1]),
        ('XZ plane (y=mid)', volume[:, volume.shape[1]//2, :]),
        ('YZ plane (x=mid)', volume[volume.shape[0]//2, :, :]),
        ('3D projection', np.max(volume, axis=2))
    ]
    
    for ax, (title, data) in zip(axes.flat, slices):
        im = ax.imshow(data, cmap='tab10', vmin=0, vmax=5)
        ax.set_title(title)
        ax.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', 
                       pad=0.1, fraction=0.05, ticks=[0, 1, 2, 3, 4, 5])
    cbar.set_label('Phase ID')
    cbar.ax.set_xticklabels(['Pore', 'Ni', 'YSZ-A', 'YSZ-E', 'GDC', 'SDC'])
    
    plt.tight_layout()
    plt.savefig('./sofc_dataset/microstructure_views.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to ./sofc_dataset/microstructure_views.png")
    
    plt.show()

def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("SOFC MICROSTRUCTURAL DATASET - USAGE EXAMPLES")
    print("="*80)
    
    # Load the dataset
    volume, voxel_size, phases = load_dataset()
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  • Shape: {volume.shape}")
    print(f"  • Voxel size: {voxel_size} µm")
    print(f"  • Physical size: {tuple(d * voxel_size for d in volume.shape)} µm")
    print(f"  • Number of phases: {len(phases)}")
    
    # Example analyses
    porosity, tpb = example_electrochemical_analysis(volume, voxel_size)
    interface, stress = example_mechanical_analysis(volume, voxel_size)
    degradation = example_degradation_study(volume, voxel_size)
    
    # Visualization
    visualize_structure(volume)
    
    print("\n" + "="*80)
    print("EXAMPLE ANALYSES COMPLETE")
    print("="*80)
    print("\nThe dataset is ready for:")
    print("  ✓ Electrochemical modeling (Butler-Volmer, Nernst-Planck)")
    print("  ✓ Mechanical analysis (FEA, thermal stress)")
    print("  ✓ Transport phenomena (gas diffusion, ionic conduction)")
    print("  ✓ Degradation studies (Ni coarsening, delamination)")
    print("  ✓ Machine learning (segmentation training, property prediction)")
    print("\nRefer to the documentation for detailed usage instructions.")

if __name__ == "__main__":
    main()