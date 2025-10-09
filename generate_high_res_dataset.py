#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import tifffile
import os
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("HIGH-RESOLUTION SOFC MICROSTRUCTURE DATASET GENERATION")
    print("="*60)
    
    # Parameters - higher resolution
    resolution = (256, 256, 128)  # Higher resolution
    voxel_size = 0.1  # 100 nm
    porosity = 0.3
    ni_ysz_ratio = 0.6
    ysz_thickness = 10.0  # 10 μm
    
    print(f"Resolution: {resolution}")
    print(f"Voxel size: {voxel_size} μm")
    print(f"Target porosity: {porosity}")
    print(f"Ni/YSZ ratio: {ni_ysz_ratio}")
    print(f"YSZ thickness: {ysz_thickness} μm")
    
    # Phase labels
    PORE = 0
    NI = 1
    YSZ_ANODE = 2
    YSZ_ELECTROLYTE = 3
    INTERLAYER = 4
    
    # Initialize microstructure
    print("\nGenerating microstructure...")
    microstructure = np.zeros(resolution, dtype=np.uint8)
    
    # 1. Generate pore network
    print("  Creating pore network...")
    pore_mask = np.zeros(resolution, dtype=bool)
    
    # Create pores using random spheres
    n_pores = int(porosity * np.prod(resolution) / 2000)  # Adjusted for higher resolution
    print(f"    Creating {n_pores} pores...")
    
    for i in range(n_pores):
        # Random center
        center = np.random.uniform(0, min(resolution), 3)
        radius = np.random.uniform(3, 8)  # Larger pores for higher resolution
        
        # Create sphere
        x, y, z = np.ogrid[:resolution[0], :resolution[1], :resolution[2]]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        sphere = dist <= radius
        pore_mask |= sphere
        
        if i % 50 == 0:
            print(f"      Created {i+1}/{n_pores} pores")
    
    # Apply morphological operations
    pore_mask = morphology.binary_opening(pore_mask, morphology.ball(2))
    pore_mask = morphology.binary_closing(pore_mask, morphology.ball(1))
    
    microstructure[pore_mask] = PORE
    print(f"    Pore phase: {np.sum(pore_mask)} voxels")
    
    # 2. Generate Ni phase
    print("  Creating Ni phase...")
    solid_mask = ~pore_mask
    ni_mask = np.zeros_like(solid_mask)
    
    # Create Ni particles
    n_ni_particles = int(ni_ysz_ratio * np.sum(solid_mask) / 1000)  # Adjusted for higher resolution
    print(f"    Creating {n_ni_particles} Ni particles...")
    
    for i in range(n_ni_particles):
        # Find random solid voxel
        solid_indices = np.where(solid_mask)
        if len(solid_indices[0]) > 0:
            idx = np.random.randint(0, len(solid_indices[0]))
            center = [solid_indices[j][idx] for j in range(3)]
            
            # Create Ni particle
            radius = np.random.uniform(2, 5)  # Larger particles for higher resolution
            x, y, z = np.ogrid[:resolution[0], :resolution[1], :resolution[2]]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            sphere = (dist <= radius) & solid_mask
            ni_mask |= sphere
            
            if i % 100 == 0:
                print(f"      Created {i+1}/{n_ni_particles} Ni particles")
    
    microstructure[ni_mask] = NI
    print(f"    Ni phase: {np.sum(ni_mask)} voxels")
    
    # 3. Generate YSZ anode (remaining solid)
    print("  Creating YSZ anode...")
    ysz_anode_mask = solid_mask & ~ni_mask
    microstructure[ysz_anode_mask] = YSZ_ANODE
    print(f"    YSZ anode: {np.sum(ysz_anode_mask)} voxels")
    
    # 4. Generate YSZ electrolyte
    print("  Creating YSZ electrolyte...")
    electrolyte_thickness_voxels = int(ysz_thickness / voxel_size)
    start_z = resolution[2] - electrolyte_thickness_voxels
    
    if start_z >= 0:
        electrolyte_mask = np.zeros(resolution, dtype=bool)
        electrolyte_mask[:, :, start_z:] = True
        
        # Add some roughness
        for z in range(start_z, resolution[2]):
            height_variation = np.random.normal(0, 0.5, (resolution[0], resolution[1]))
            height_variation = np.round(height_variation).astype(int)
            
            for x in range(resolution[0]):
                for y in range(resolution[1]):
                    new_z = z + height_variation[x, y]
                    if 0 <= new_z < resolution[2]:
                        electrolyte_mask[x, y, new_z] = True
        
        microstructure[electrolyte_mask] = YSZ_ELECTROLYTE
        print(f"    YSZ electrolyte: {np.sum(electrolyte_mask)} voxels")
    
    # 5. Generate interlayer
    print("  Creating interlayer...")
    anode_mask = (microstructure == NI) | (microstructure == YSZ_ANODE)
    electrolyte_mask = (microstructure == YSZ_ELECTROLYTE)
    
    anode_dilated = morphology.binary_dilation(anode_mask, morphology.ball(2))
    electrolyte_dilated = morphology.binary_dilation(electrolyte_mask, morphology.ball(2))
    interlayer_mask = anode_dilated & electrolyte_dilated & ~anode_mask & ~electrolyte_mask
    
    microstructure[interlayer_mask] = INTERLAYER
    print(f"    Interlayer: {np.sum(interlayer_mask)} voxels")
    
    # Calculate phase properties
    print("\nCalculating phase properties...")
    total_voxels = np.prod(resolution)
    
    phase_names = {0: 'Pore', 1: 'Ni', 2: 'YSZ_Anode', 3: 'YSZ_Electrolyte', 4: 'Interlayer'}
    properties = {}
    
    for phase_id, name in phase_names.items():
        count = np.sum(microstructure == phase_id)
        volume_fraction = count / total_voxels
        volume_um3 = count * (voxel_size ** 3)
        
        properties[name] = {
            'count': count,
            'volume_fraction': volume_fraction,
            'volume_um3': volume_um3
        }
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save HDF5
    print("\nSaving data...")
    h5_filename = 'output/sofc_microstructure_high_res.h5'
    print(f"  Saving HDF5: {h5_filename}")
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('microstructure', data=microstructure, compression='gzip')
        f.attrs['resolution'] = microstructure.shape
        f.attrs['voxel_size_um'] = voxel_size
        f.attrs['porosity'] = porosity
        f.attrs['ni_ysz_ratio'] = ni_ysz_ratio
        f.attrs['ysz_thickness_um'] = ysz_thickness
        
        # Save phase properties
        props_group = f.create_group('phase_properties')
        for phase, props in properties.items():
            phase_group = props_group.create_group(phase)
            for key, value in props.items():
                phase_group.attrs[key] = value
    
    # Save TIFF stack
    print("  Saving TIFF stack...")
    for z in range(microstructure.shape[2]):
        slice_data = microstructure[:, :, z]
        tiff_filename = f'output/sofc_microstructure_high_res_z{z:03d}.tif'
        tifffile.imwrite(tiff_filename, slice_data.astype(np.uint8))
    
    # Create visualizations
    print("  Creating visualizations...")
    
    # 2D slices
    z_indices = [resolution[2]//4, resolution[2]//2, 3*resolution[2]//4, resolution[2]-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = {0: [1, 1, 1], 1: [1, 0.84, 0], 2: [0.68, 0.85, 0.9], 3: [0, 0, 0.55], 4: [1, 0, 0]}
    
    for i, z in enumerate(z_indices):
        if i >= 4:
            break
            
        slice_data = microstructure[:, :, z]
        
        # Create colored image
        colored_slice = np.zeros((*slice_data.shape, 3))
        for phase_id, color in colors.items():
            mask = slice_data == phase_id
            colored_slice[mask] = color
        
        axes[i].imshow(colored_slice)
        axes[i].set_title(f'Z = {z * voxel_size:.1f} μm')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/sofc_microstructure_high_res_slices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Phase distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    phases = list(properties.keys())
    fractions = [props['volume_fraction'] for props in properties.values()]
    color_list = [colors[i] for i in range(len(phases))]
    
    # Bar chart
    bars = ax1.bar(phases, fractions, color=color_list, edgecolor='black')
    ax1.set_title('Phase Volume Fractions')
    ax1.set_ylabel('Volume Fraction')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, fraction in zip(bars, fractions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{fraction:.3f}', ha='center', va='bottom')
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(fractions, labels=phases, colors=color_list, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Phase Distribution')
    
    plt.tight_layout()
    plt.savefig('output/sofc_microstructure_high_res_phase_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*50)
    print("HIGH-RESOLUTION GENERATION COMPLETE")
    print("="*50)
    print(f"Resolution: {microstructure.shape}")
    print(f"Voxel size: {voxel_size} μm")
    print(f"Total volume: {np.prod(microstructure.shape) * (voxel_size**3):.2f} μm³")
    print("\nPhase Distribution:")
    for phase, props in properties.items():
        print(f"  {phase}: {props['volume_fraction']:.3f} ({props['volume_um3']:.2f} μm³)")
    
    print(f"\nFiles saved to 'output/' directory:")
    print("  - sofc_microstructure_high_res.h5 (HDF5 format)")
    print("  - sofc_microstructure_high_res_z*.tif (TIFF stack)")
    print("  - sofc_microstructure_high_res_slices.png (2D slices)")
    print("  - sofc_microstructure_high_res_phase_distribution.png (Phase distribution)")
    
    # Create analysis report
    print("\nCreating analysis report...")
    with open('output/analysis_report_high_res.txt', 'w') as f:
        f.write("SOFC Microstructure Analysis Report (High Resolution)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Resolution: {microstructure.shape}\n")
        f.write(f"Voxel size: {voxel_size} μm\n")
        f.write(f"Total volume: {np.prod(microstructure.shape) * (voxel_size**3):.2f} μm³\n\n")
        f.write("Phase Distribution:\n")
        for phase, props in properties.items():
            f.write(f"  {phase}: {props['volume_fraction']:.3f} ({props['volume_um3']:.2f} μm³)\n")
    
    print("Analysis report saved to output/analysis_report_high_res.txt")

if __name__ == "__main__":
    main()