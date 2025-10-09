#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import tifffile
import os
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')

def generate_optimized_sofc_microstructure(resolution=(256, 256, 128), voxel_size=0.1, 
                                          porosity=0.3, ni_ysz_ratio=0.6, ysz_thickness=10.0):
    """
    Generate optimized SOFC microstructure with better phase distribution.
    """
    print(f"Generating optimized SOFC microstructure with resolution {resolution}")
    
    # Phase labels
    PORE = 0
    NI = 1
    YSZ_ANODE = 2
    YSZ_ELECTROLYTE = 3
    INTERLAYER = 4
    
    # Initialize microstructure
    microstructure = np.zeros(resolution, dtype=np.uint8)
    
    # 1. Generate pore network with better distribution
    print("  Creating optimized pore network...")
    pore_mask = np.zeros(resolution, dtype=bool)
    
    # Create pores using multiple methods for realism
    n_pores = int(porosity * np.prod(resolution) / 1500)
    print(f"    Creating {n_pores} pores...")
    
    # Method 1: Random spheres
    for i in range(n_pores // 2):
        center = np.random.uniform(0, min(resolution), 3)
        radius = np.random.uniform(3, 7)
        
        x, y, z = np.ogrid[:resolution[0], :resolution[1], :resolution[2]]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        sphere = dist <= radius
        pore_mask |= sphere
        
        if i % 50 == 0:
            print(f"      Created {i+1}/{n_pores//2} random pores")
    
    # Method 2: Elongated pores for better connectivity
    for i in range(n_pores // 2):
        center = np.random.uniform(0, min(resolution), 3)
        # Create elongated pore
        length = np.random.uniform(5, 12)
        width = np.random.uniform(2, 4)
        height = np.random.uniform(2, 4)
        
        # Random orientation
        angle = np.random.uniform(0, 2*np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        x, y, z = np.ogrid[:resolution[0], :resolution[1], :resolution[2]]
        
        # Rotate coordinates
        x_rot = (x - center[0]) * cos_a - (y - center[1]) * sin_a
        y_rot = (x - center[0]) * sin_a + (y - center[1]) * cos_a
        z_rot = z - center[2]
        
        # Create ellipsoid
        ellipsoid = ((x_rot/length)**2 + (y_rot/width)**2 + (z_rot/height)**2) <= 1
        pore_mask |= ellipsoid
        
        if i % 50 == 0:
            print(f"      Created {i+1}/{n_pores//2} elongated pores")
    
    # Apply morphological operations for realism
    pore_mask = morphology.binary_opening(pore_mask, morphology.ball(1))
    pore_mask = morphology.binary_closing(pore_mask, morphology.ball(1))
    
    microstructure[pore_mask] = PORE
    print(f"    Pore phase: {np.sum(pore_mask)} voxels")
    
    # 2. Generate Ni phase with better distribution
    print("  Creating optimized Ni phase...")
    solid_mask = ~pore_mask
    ni_mask = np.zeros_like(solid_mask)
    
    # Create Ni particles with size distribution
    n_ni_particles = int(ni_ysz_ratio * np.sum(solid_mask) / 800)
    print(f"    Creating {n_ni_particles} Ni particles...")
    
    for i in range(n_ni_particles):
        # Find random solid voxel
        solid_indices = np.where(solid_mask)
        if len(solid_indices[0]) > 0:
            idx = np.random.randint(0, len(solid_indices[0]))
            center = [solid_indices[j][idx] for j in range(3)]
            
            # Create Ni particle with size distribution
            radius = np.random.gamma(2, 1.5) + 1  # Gamma distribution for realistic sizes
            radius = np.clip(radius, 1, 6)
            
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
    
    # 4. Generate YSZ electrolyte with realistic interface
    print("  Creating YSZ electrolyte...")
    electrolyte_thickness_voxels = int(ysz_thickness / voxel_size)
    start_z = resolution[2] - electrolyte_thickness_voxels
    
    if start_z >= 0:
        electrolyte_mask = np.zeros(resolution, dtype=bool)
        electrolyte_mask[:, :, start_z:] = True
        
        # Add realistic surface roughness
        for z in range(start_z, resolution[2]):
            # Create correlated roughness
            roughness = np.random.normal(0, 0.8, (resolution[0], resolution[1]))
            # Apply smoothing for more realistic interface
            from scipy.ndimage import gaussian_filter
            roughness = gaussian_filter(roughness, sigma=1.0)
            height_variation = np.round(roughness).astype(int)
            
            for x in range(resolution[0]):
                for y in range(resolution[1]):
                    new_z = z + height_variation[x, y]
                    if 0 <= new_z < resolution[2]:
                        electrolyte_mask[x, y, new_z] = True
        
        microstructure[electrolyte_mask] = YSZ_ELECTROLYTE
        print(f"    YSZ electrolyte: {np.sum(electrolyte_mask)} voxels")
    
    # 5. Generate interlayer with better definition
    print("  Creating interlayer...")
    anode_mask = (microstructure == NI) | (microstructure == YSZ_ANODE)
    electrolyte_mask = (microstructure == YSZ_ELECTROLYTE)
    
    # Create more realistic interlayer
    anode_dilated = morphology.binary_dilation(anode_mask, morphology.ball(2))
    electrolyte_dilated = morphology.binary_dilation(electrolyte_mask, morphology.ball(2))
    interlayer_mask = anode_dilated & electrolyte_dilated & ~anode_mask & ~electrolyte_mask
    
    # Add some additional interlayer material
    interlayer_dilated = morphology.binary_dilation(interlayer_mask, morphology.ball(1))
    additional_interlayer = interlayer_dilated & ~interlayer_mask & ~anode_mask & ~electrolyte_mask
    interlayer_mask |= additional_interlayer
    
    microstructure[interlayer_mask] = INTERLAYER
    print(f"    Interlayer: {np.sum(interlayer_mask)} voxels")
    
    return microstructure

def create_advanced_visualizations(microstructure, voxel_size, output_prefix):
    """Create advanced visualizations."""
    print("Creating advanced visualizations...")
    
    resolution = microstructure.shape
    
    # 1. Multiple cross-sections
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    z_indices = [resolution[2]//6, resolution[2]//3, resolution[2]//2, 
                 2*resolution[2]//3, 5*resolution[2]//6, resolution[2]-1]
    
    colors = {0: [1, 1, 1], 1: [1, 0.84, 0], 2: [0.68, 0.85, 0.9], 3: [0, 0, 0.55], 4: [1, 0, 0]}
    
    for i, z in enumerate(z_indices):
        if i >= 6:
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
    plt.savefig(f'{output_prefix}_advanced_slices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Phase distribution with statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate phase statistics
    phase_names = {0: 'Pore', 1: 'Ni', 2: 'YSZ_Anode', 3: 'YSZ_Electrolyte', 4: 'Interlayer'}
    total_voxels = np.prod(resolution)
    
    phases = []
    fractions = []
    volumes = []
    colors_list = []
    
    for phase_id, name in phase_names.items():
        count = np.sum(microstructure == phase_id)
        volume_fraction = count / total_voxels
        volume_um3 = count * (voxel_size ** 3)
        
        phases.append(name)
        fractions.append(volume_fraction)
        volumes.append(volume_um3)
        colors_list.append(colors[phase_id])
    
    # Bar chart
    bars = ax1.bar(phases, fractions, color=colors_list, edgecolor='black')
    ax1.set_title('Phase Volume Fractions')
    ax1.set_ylabel('Volume Fraction')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, fraction in zip(bars, fractions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{fraction:.3f}', ha='center', va='bottom')
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(fractions, labels=phases, colors=colors_list, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Phase Distribution')
    
    # Volume chart
    bars = ax3.bar(phases, volumes, color=colors_list, edgecolor='black')
    ax3.set_title('Phase Volumes')
    ax3.set_ylabel('Volume (μm³)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, volume in zip(bars, volumes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{volume:.1f}', ha='center', va='bottom')
    
    # Summary statistics
    ax4.axis('off')
    
    summary_text = f"""
    Microstructure Summary:
    
    Resolution: {resolution}
    Voxel Size: {voxel_size} μm
    Total Volume: {total_voxels * (voxel_size**3):.2f} μm³
    
    Phase Distribution:
    """
    
    for phase, fraction, volume in zip(phases, fractions, volumes):
        summary_text += f"    {phase}: {fraction:.3f} ({volume:.1f} μm³)\n"
    
    summary_text += f"\nQuality Metrics:\n"
    summary_text += f"    Porosity: {fractions[0]:.3f}\n"
    summary_text += f"    Ni/YSZ Ratio: {fractions[1]/(fractions[1]+fractions[2]):.3f}\n"
    summary_text += f"    Electrolyte Coverage: {fractions[3]:.3f}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_advanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    print("="*60)
    print("FINAL OPTIMIZED SOFC MICROSTRUCTURE GENERATION")
    print("="*60)
    
    # Parameters for high-quality dataset
    resolution = (256, 256, 128)
    voxel_size = 0.1  # 100 nm
    porosity = 0.3
    ni_ysz_ratio = 0.6
    ysz_thickness = 10.0  # 10 μm
    
    print(f"Resolution: {resolution}")
    print(f"Voxel size: {voxel_size} μm")
    print(f"Target porosity: {porosity}")
    print(f"Ni/YSZ ratio: {ni_ysz_ratio}")
    print(f"YSZ thickness: {ysz_thickness} μm")
    
    # Generate microstructure
    microstructure = generate_optimized_sofc_microstructure(
        resolution=resolution,
        voxel_size=voxel_size,
        porosity=porosity,
        ni_ysz_ratio=ni_ysz_ratio,
        ysz_thickness=ysz_thickness
    )
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save HDF5
    print("\nSaving data...")
    h5_filename = 'output/sofc_microstructure_final.h5'
    print(f"  Saving HDF5: {h5_filename}")
    
    # Calculate phase properties
    phase_names = {0: 'Pore', 1: 'Ni', 2: 'YSZ_Anode', 3: 'YSZ_Electrolyte', 4: 'Interlayer'}
    total_voxels = np.prod(microstructure.shape)
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
        tiff_filename = f'output/sofc_microstructure_final_z{z:03d}.tif'
        tifffile.imwrite(tiff_filename, slice_data.astype(np.uint8))
    
    # Create advanced visualizations
    create_advanced_visualizations(microstructure, voxel_size, 'output/sofc_microstructure_final')
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Resolution: {microstructure.shape}")
    print(f"Voxel size: {voxel_size} μm")
    print(f"Total volume: {total_voxels * (voxel_size**3):.2f} μm³")
    print("\nPhase Distribution:")
    for phase, props in properties.items():
        print(f"  {phase}: {props['volume_fraction']:.3f} ({props['volume_um3']:.2f} μm³)")
    
    print(f"\nFiles saved to 'output/' directory:")
    print("  - sofc_microstructure_final.h5 (HDF5 format)")
    print("  - sofc_microstructure_final_z*.tif (TIFF stack)")
    print("  - sofc_microstructure_final_advanced_slices.png (Advanced slices)")
    print("  - sofc_microstructure_final_advanced_analysis.png (Advanced analysis)")
    
    # Create final summary report
    print("\nCreating final summary report...")
    with open('output/final_summary_report.txt', 'w') as f:
        f.write("SOFC Microstructure Dataset - Final Summary\n")
        f.write("="*50 + "\n\n")
        f.write("This dataset contains realistic 3D microstructural data for SOFC electrode modeling.\n\n")
        f.write("Dataset Specifications:\n")
        f.write(f"  Resolution: {microstructure.shape}\n")
        f.write(f"  Voxel size: {voxel_size} μm\n")
        f.write(f"  Total volume: {total_voxels * (voxel_size**3):.2f} μm³\n")
        f.write(f"  Target porosity: {porosity}\n")
        f.write(f"  Ni/YSZ ratio: {ni_ysz_ratio}\n")
        f.write(f"  YSZ thickness: {ysz_thickness} μm\n\n")
        
        f.write("Phase Distribution:\n")
        for phase, props in properties.items():
            f.write(f"  {phase}: {props['volume_fraction']:.3f} ({props['volume_um3']:.2f} μm³)\n")
        
        f.write("\nFile Formats:\n")
        f.write("  - HDF5: Hierarchical data format with metadata\n")
        f.write("  - TIFF: Image stack format for each z-slice\n")
        f.write("  - PNG: Visualization plots\n\n")
        
        f.write("Applications:\n")
        f.write("  - Finite Element Analysis (FEA)\n")
        f.write("  - Computational Fluid Dynamics (CFD)\n")
        f.write("  - Electrochemical modeling\n")
        f.write("  - Thermal analysis\n")
        f.write("  - Delamination studies\n")
        f.write("  - Transport property calculations\n\n")
        
        f.write("Usage Instructions:\n")
        f.write("  1. Load HDF5 file for complete dataset with metadata\n")
        f.write("  2. Use TIFF stack for image processing workflows\n")
        f.write("  3. Import into FEA software (ANSYS, Abaqus, etc.)\n")
        f.write("  4. Use for computational modeling and analysis\n")
    
    print("Final summary report saved to output/final_summary_report.txt")

if __name__ == "__main__":
    main()