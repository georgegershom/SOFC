"""
Simplified 3D Microstructural Dataset Generator for SOFC Electrode Modeling

This module generates realistic 3D microstructural data for SOFC electrodes,
including proper phase segmentation, interface geometry, and volume fractions.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from skimage import morphology, segmentation, measure, filters
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import tifffile
import pandas as pd
from tqdm import tqdm
import os
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class SOFCMicrostructureGenerator:
    """
    Generator for realistic 3D SOFC electrode microstructures.
    
    Creates voxelated data with proper phase segmentation including:
    - Pore phase
    - Ni-YSZ anode material
    - YSZ electrolyte
    - Interlayers
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int, int] = (256, 256, 128),
                 voxel_size: float = 0.1,  # micrometers
                 porosity: float = 0.3,
                 ni_ysz_ratio: float = 0.6,
                 ysz_thickness: float = 10.0):  # micrometers
        """
        Initialize the microstructure generator.
        
        Parameters:
        -----------
        resolution : tuple
            (width, height, depth) in voxels
        voxel_size : float
            Size of each voxel in micrometers
        porosity : float
            Target porosity (0-1)
        ni_ysz_ratio : float
            Ratio of Ni to YSZ in anode (0-1)
        ysz_thickness : float
            Thickness of electrolyte layer in micrometers
        """
        self.resolution = resolution
        self.voxel_size = voxel_size
        self.porosity = porosity
        self.ni_ysz_ratio = ni_ysz_ratio
        self.ysz_thickness = ysz_thickness
        
        # Phase labels
        self.PORE = 0
        self.NI = 1
        self.YSZ_ANODE = 2
        self.YSZ_ELECTROLYTE = 3
        self.INTERLAYER = 4
        
        # Initialize empty microstructure
        self.microstructure = np.zeros(resolution, dtype=np.uint8)
        self.phase_properties = {}
        
    def generate_realistic_pore_network(self) -> np.ndarray:
        """
        Generate a realistic pore network using a combination of methods.
        """
        print("Generating realistic pore network...")
        
        # Create base pore structure using random spheres
        pore_mask = np.zeros(self.resolution, dtype=bool)
        
        # Generate pore centers with some clustering
        n_pores = int(self.porosity * np.prod(self.resolution) / 1000)
        pore_centers = []
        
        # Create clustered pore centers
        n_clusters = max(1, n_pores // 20)
        cluster_centers = np.random.uniform(0, min(self.resolution), (n_clusters, 3))
        
        for i in range(n_pores):
            if i < n_clusters:
                # Place some pores at cluster centers
                center = cluster_centers[i] + np.random.normal(0, 5, 3)
            else:
                # Place remaining pores near existing clusters
                cluster_idx = np.random.randint(0, n_clusters)
                center = cluster_centers[cluster_idx] + np.random.normal(0, 15, 3)
            
            center = np.clip(center, 0, np.array(self.resolution) - 1)
            pore_centers.append(center)
        
        # Create pores with varying sizes
        for center in pore_centers:
            # Random pore radius (2-8 voxels)
            radius = np.random.uniform(2, 8)
            
            # Create sphere
            x, y, z = np.ogrid[:self.resolution[0], :self.resolution[1], :self.resolution[2]]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            sphere = dist <= radius
            pore_mask |= sphere
        
        # Apply morphological operations for realism
        pore_mask = morphology.binary_opening(pore_mask, morphology.ball(2))
        pore_mask = morphology.binary_closing(pore_mask, morphology.ball(1))
        
        # Add some connectivity between pores
        pore_mask = morphology.binary_dilation(pore_mask, morphology.ball(1))
        
        return pore_mask
    
    def generate_ni_ysz_anode(self, pore_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Ni-YSZ anode structure with proper phase distribution.
        """
        print("Generating Ni-YSZ anode structure...")
        
        # Solid phase (non-pore)
        solid_mask = ~pore_mask
        
        # Create Ni and YSZ phases within solid regions
        ni_mask = np.zeros_like(solid_mask)
        ysz_anode_mask = np.zeros_like(solid_mask)
        
        # Generate Ni phase using random spheres
        ni_centers = []
        n_ni_particles = int(self.ni_ysz_ratio * np.sum(solid_mask) / 500)
        
        for _ in range(n_ni_particles):
            # Find random solid voxel
            solid_indices = np.where(solid_mask)
            if len(solid_indices[0]) > 0:
                idx = np.random.randint(0, len(solid_indices[0]))
                center = [solid_indices[i][idx] for i in range(3)]
                ni_centers.append(center)
        
        # Create Ni particles
        for center in ni_centers:
            radius = np.random.uniform(1.5, 4)
            x, y, z = np.ogrid[:self.resolution[0], :self.resolution[1], :self.resolution[2]]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            sphere = (dist <= radius) & solid_mask
            ni_mask |= sphere
        
        # YSZ anode is the remaining solid phase
        ysz_anode_mask = solid_mask & ~ni_mask
        
        # Apply some smoothing
        ni_mask = morphology.binary_opening(ni_mask, morphology.ball(1))
        ysz_anode_mask = morphology.binary_opening(ysz_anode_mask, morphology.ball(1))
        
        return ni_mask, ysz_anode_mask
    
    def generate_ysz_electrolyte(self) -> np.ndarray:
        """
        Generate YSZ electrolyte layer with realistic interface.
        """
        print("Generating YSZ electrolyte layer...")
        
        # Calculate electrolyte thickness in voxels
        electrolyte_thickness_voxels = int(self.ysz_thickness / self.voxel_size)
        
        # Create electrolyte layer at the top
        electrolyte_mask = np.zeros(self.resolution, dtype=bool)
        start_z = self.resolution[2] - electrolyte_thickness_voxels
        
        if start_z >= 0:
            electrolyte_mask[:, :, start_z:] = True
        
        # Add some surface roughness to the interface
        for z in range(start_z, self.resolution[2]):
            # Add random height variations
            height_variation = np.random.normal(0, 0.5, (self.resolution[0], self.resolution[1]))
            height_variation = np.round(height_variation).astype(int)
            
            for x in range(self.resolution[0]):
                for y in range(self.resolution[1]):
                    new_z = z + height_variation[x, y]
                    if 0 <= new_z < self.resolution[2]:
                        electrolyte_mask[x, y, new_z] = True
        
        return electrolyte_mask
    
    def generate_interlayer(self, anode_mask: np.ndarray, electrolyte_mask: np.ndarray) -> np.ndarray:
        """
        Generate interlayer between anode and electrolyte.
        """
        print("Generating interlayer...")
        
        # Find interface between anode and electrolyte
        anode_dilated = morphology.binary_dilation(anode_mask, morphology.ball(2))
        electrolyte_dilated = morphology.binary_dilation(electrolyte_mask, morphology.ball(2))
        
        # Interlayer is the intersection of dilated regions
        interlayer_mask = anode_dilated & electrolyte_dilated
        
        # Remove original anode and electrolyte from interlayer
        interlayer_mask = interlayer_mask & ~anode_mask & ~electrolyte_mask
        
        return interlayer_mask
    
    def generate_microstructure(self) -> np.ndarray:
        """
        Generate the complete 3D microstructure.
        """
        print("Generating complete 3D SOFC microstructure...")
        
        # Generate pore network
        pore_mask = self.generate_realistic_pore_network()
        
        # Generate Ni-YSZ anode
        ni_mask, ysz_anode_mask = self.generate_ni_ysz_anode(pore_mask)
        
        # Generate YSZ electrolyte
        electrolyte_mask = self.generate_ysz_electrolyte()
        
        # Generate interlayer
        interlayer_mask = self.generate_interlayer(
            ni_mask | ysz_anode_mask, electrolyte_mask
        )
        
        # Combine all phases
        microstructure = np.zeros(self.resolution, dtype=np.uint8)
        microstructure[pore_mask] = self.PORE
        microstructure[ni_mask] = self.NI
        microstructure[ysz_anode_mask] = self.YSZ_ANODE
        microstructure[electrolyte_mask] = self.YSZ_ELECTROLYTE
        microstructure[interlayer_mask] = self.INTERLAYER
        
        self.microstructure = microstructure
        
        # Calculate phase properties
        self._calculate_phase_properties()
        
        return microstructure
    
    def _calculate_phase_properties(self):
        """Calculate volume fractions and other properties."""
        print("Calculating phase properties...")
        
        total_voxels = np.prod(self.resolution)
        
        self.phase_properties = {
            'total_voxels': total_voxels,
            'voxel_size_um': self.voxel_size,
            'total_volume_um3': total_voxels * (self.voxel_size ** 3),
            'phases': {}
        }
        
        phase_names = {
            self.PORE: 'Pore',
            self.NI: 'Ni',
            self.YSZ_ANODE: 'YSZ_Anode',
            self.YSZ_ELECTROLYTE: 'YSZ_Electrolyte',
            self.INTERLAYER: 'Interlayer'
        }
        
        for phase_id, name in phase_names.items():
            count = np.sum(self.microstructure == phase_id)
            volume_fraction = count / total_voxels
            volume_um3 = count * (self.voxel_size ** 3)
            
            self.phase_properties['phases'][name] = {
                'count': count,
                'volume_fraction': volume_fraction,
                'volume_um3': volume_um3
            }
    
    def save_hdf5(self, filename: str):
        """Save microstructure to HDF5 format."""
        print(f"Saving microstructure to {filename}...")
        
        with h5py.File(filename, 'w') as f:
            # Save microstructure data
            f.create_dataset('microstructure', data=self.microstructure, compression='gzip')
            
            # Save metadata
            f.attrs['resolution'] = self.resolution
            f.attrs['voxel_size_um'] = self.voxel_size
            f.attrs['porosity'] = self.porosity
            f.attrs['ni_ysz_ratio'] = self.ni_ysz_ratio
            f.attrs['ysz_thickness_um'] = self.ysz_thickness
            
            # Save phase properties
            props_group = f.create_group('phase_properties')
            for phase, props in self.phase_properties['phases'].items():
                phase_group = props_group.create_group(phase)
                for key, value in props.items():
                    phase_group.attrs[key] = value
    
    def save_tiff_stack(self, filename_prefix: str):
        """Save microstructure as TIFF stack."""
        print(f"Saving TIFF stack with prefix {filename_prefix}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename_prefix) if os.path.dirname(filename_prefix) else '.', exist_ok=True)
        
        # Save each slice
        for z in range(self.resolution[2]):
            slice_data = self.microstructure[:, :, z]
            filename = f"{filename_prefix}_z{z:03d}.tif"
            tifffile.imwrite(filename, slice_data.astype(np.uint8))
    
    def visualize_2d_slices(self, n_slices: int = 4, save_path: str = None):
        """Create 2D slice visualizations."""
        print("Creating 2D slice visualizations...")
        
        # Select z-slices
        z_indices = np.linspace(0, self.resolution[2]-1, n_slices, dtype=int)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        phase_names = {0: 'Pore', 1: 'Ni', 2: 'YSZ_Anode', 3: 'YSZ_Electrolyte', 4: 'Interlayer'}
        colors = {0: 'white', 1: 'gold', 2: 'lightblue', 3: 'darkblue', 4: 'red'}
        
        for i, z in enumerate(z_indices):
            if i >= 4:
                break
                
            slice_data = self.microstructure[:, :, z]
            
            # Create colored image
            colored_slice = np.zeros((*slice_data.shape, 3))
            for phase_id, color in colors.items():
                mask = slice_data == phase_id
                if color == 'white':
                    colored_slice[mask] = [1, 1, 1]
                elif color == 'gold':
                    colored_slice[mask] = [1, 0.84, 0]
                elif color == 'lightblue':
                    colored_slice[mask] = [0.68, 0.85, 0.9]
                elif color == 'darkblue':
                    colored_slice[mask] = [0, 0, 0.55]
                elif color == 'red':
                    colored_slice[mask] = [1, 0, 0]
            
            axes[i].imshow(colored_slice)
            axes[i].set_title(f'Z = {z * self.voxel_size:.1f} μm')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D slices saved to {save_path}")
        
        plt.show()
    
    def create_phase_distribution_plot(self, save_path: str = None):
        """Create phase distribution visualization."""
        print("Creating phase distribution plot...")
        
        # Calculate phase fractions
        total_voxels = np.prod(self.resolution)
        phase_fractions = {}
        phase_names = {0: 'Pore', 1: 'Ni', 2: 'YSZ_Anode', 3: 'YSZ_Electrolyte', 4: 'Interlayer'}
        
        for phase_id, name in phase_names.items():
            count = np.sum(self.microstructure == phase_id)
            phase_fractions[name] = count / total_voxels
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        phases = list(phase_fractions.keys())
        fractions = list(phase_fractions.values())
        colors = ['white', 'gold', 'lightblue', 'darkblue', 'red']
        
        bars = ax1.bar(phases, fractions, color=colors, edgecolor='black')
        ax1.set_title('Phase Volume Fractions')
        ax1.set_ylabel('Volume Fraction')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, fraction in zip(bars, fractions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{fraction:.3f}', ha='center', va='bottom')
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(fractions, labels=phases, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('Phase Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Phase distribution plot saved to {save_path}")
        
        plt.show()


def main():
    """Main function to generate and save the dataset."""
    print("Starting 3D SOFC Microstructure Generation...")
    
    # Create generator with realistic parameters
    generator = SOFCMicrostructureGenerator(
        resolution=(256, 256, 128),
        voxel_size=0.1,  # 100 nm voxel size
        porosity=0.3,
        ni_ysz_ratio=0.6,
        ysz_thickness=10.0  # 10 micrometers
    )
    
    # Generate microstructure
    microstructure = generator.generate_microstructure()
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save in multiple formats
    generator.save_hdf5('output/sofc_microstructure.h5')
    generator.save_tiff_stack('output/sofc_microstructure')
    
    # Create visualizations
    generator.visualize_2d_slices(save_path='output/microstructure_slices.png')
    generator.create_phase_distribution_plot(save_path='output/phase_distribution.png')
    
    # Print summary
    print("\n" + "="*50)
    print("MICROSTRUCTURE GENERATION COMPLETE")
    print("="*50)
    print(f"Resolution: {generator.resolution}")
    print(f"Voxel size: {generator.voxel_size} μm")
    print(f"Total volume: {generator.phase_properties['total_volume_um3']:.2f} μm³")
    print("\nPhase Distribution:")
    for phase, props in generator.phase_properties['phases'].items():
        print(f"  {phase}: {props['volume_fraction']:.3f} ({props['volume_um3']:.2f} μm³)")
    
    print(f"\nFiles saved to 'output/' directory:")
    print("  - sofc_microstructure.h5 (HDF5 format)")
    print("  - sofc_microstructure_*.tif (TIFF stack)")
    print("  - microstructure_slices.png (2D slice visualization)")
    print("  - phase_distribution.png (Phase distribution plot)")


if __name__ == "__main__":
    main()