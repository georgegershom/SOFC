#!/usr/bin/env python3
"""
Synthetic Synchrotron X-ray Tomography Data Generator for SOFC Materials

This script generates realistic synthetic 3D tomography data for SOFC metallic interconnects
and anode support materials, including grain structures, porosity, and defects.
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import Voronoi
import json
import os
from datetime import datetime

class SyntheticTomographyGenerator:
    def __init__(self, sample_size=(200, 200, 200), voxel_size=0.5, material_type='interconnect'):
        """
        Initialize the tomography generator

        Parameters:
        -----------
        sample_size : tuple
            (x, y, z) dimensions in voxels
        voxel_size : float
            Physical size of each voxel in microns
        material_type : str
            'interconnect' for metallic interconnect or 'anode' for anode support
        """
        self.sample_size = sample_size
        self.voxel_size = voxel_size  # microns
        self.material_type = material_type

        # Material properties for realistic attenuation
        self.material_properties = {
            'interconnect': {
                'base_attenuation': 2.5,  # Higher for metallic materials
                'grain_attenuation_variation': 0.3,
                'porosity_attenuation': 0.1,
                'typical_grain_size': 50,  # microns
                'porosity_fraction': 0.02
            },
            'anode': {
                'base_attenuation': 1.2,  # Lower for ceramic materials
                'grain_attenuation_variation': 0.2,
                'porosity_attenuation': 0.05,
                'typical_grain_size': 30,  # microns
                'porosity_fraction': 0.15
            }
        }

        self.props = self.material_properties[material_type]

    def generate_grain_structure(self):
        """
        Generate realistic grain structure using Voronoi tessellation
        """
        print("Generating grain structure...")

        # Generate random grain centers
        n_grains = max(10, int(np.prod(self.sample_size) / (self.props['typical_grain_size']**3 / self.voxel_size**3)))
        grain_centers = np.random.rand(n_grains, 3) * self.sample_size

        # Create Voronoi diagram
        vor = Voronoi(grain_centers)

        # Assign each voxel to nearest grain
        grain_map = np.zeros(self.sample_size, dtype=int)

        for i in range(self.sample_size[0]):
            for j in range(self.sample_size[1]):
                for k in range(self.sample_size[2]):
                    # Find closest grain center
                    distances = np.sqrt((vor.points[:, 0] - i)**2 +
                                      (vor.points[:, 1] - j)**2 +
                                      (vor.points[:, 2] - k)**2)
                    grain_map[i, j, k] = np.argmin(distances)

        return grain_map

    def add_porosity_and_defects(self, grain_map):
        """
        Add realistic porosity and initial defects
        """
        print("Adding porosity and defects...")

        # Create porosity mask
        porosity_map = np.random.rand(*self.sample_size) < self.props['porosity_fraction']

        # Add some structured defects (cracks, inclusions)
        # Simulate some grain boundary defects
        defect_map = np.zeros_like(grain_map, dtype=bool)

        # Add some random defect locations
        n_defects = int(np.prod(self.sample_size) * 0.001)  # 0.1% defects
        defect_locations = np.random.randint(0, self.sample_size[0], n_defects), \
                          np.random.randint(0, self.sample_size[1], n_defects), \
                          np.random.randint(0, self.sample_size[2], n_defects)

        defect_map[defect_locations] = True

        # Expand defects slightly
        defect_map = ndimage.binary_dilation(defect_map, iterations=2)

        return porosity_map, defect_map

    def generate_attenuation_map(self, grain_map, porosity_map, defect_map):
        """
        Generate realistic X-ray attenuation map
        """
        print("Generating attenuation map...")

        # Base attenuation for each grain (with some variation)
        attenuation_map = np.zeros(self.sample_size)

        for grain_id in np.unique(grain_map):
            grain_mask = grain_map == grain_id
            # Add some variation within grains
            grain_base = self.props['base_attenuation'] + \
                        np.random.normal(0, self.props['grain_attenuation_variation'])

            attenuation_map[grain_mask] = grain_base + \
                                        np.random.normal(0, 0.1, size=np.sum(grain_mask))

        # Apply porosity effects (lower attenuation)
        attenuation_map[porosity_map] = self.props['porosity_attenuation']

        # Apply defect effects (very low attenuation or cracks)
        attenuation_map[defect_map] = 0.01  # Nearly transparent

        # Add some noise to simulate real measurement conditions
        noise_level = 0.05
        attenuation_map += np.random.normal(0, noise_level, self.sample_size)

        # Ensure non-negative values
        attenuation_map = np.maximum(attenuation_map, 0)

        return attenuation_map

    def generate_projection_data(self, attenuation_map, n_angles=360):
        """
        Generate synthetic projection data (sinogram) from attenuation map
        """
        print(f"Generating projection data ({n_angles} angles)...")

        # Simple parallel beam projection simulation
        sinogram = np.zeros((n_angles, self.sample_size[1], self.sample_size[0]))

        angles = np.linspace(0, 180, n_angles)  # degrees

        for i, angle in enumerate(angles):
            # Rotate the attenuation map
            rotated = ndimage.rotate(attenuation_map, angle, axes=(0, 2),
                                   reshape=False, prefilter=False)

            # Project along one axis (simplified)
            projection = np.sum(rotated, axis=2)
            sinogram[i] = projection

        return sinogram

    def reconstruct_tomography(self, sinogram):
        """
        Simple filtered back projection reconstruction (simplified)
        """
        print("Reconstructing 3D volume...")

        # This is a simplified reconstruction - in reality this would use FBP
        # For synthetic data, we'll use a simple back projection approach
        reconstructed = np.zeros(self.sample_size)

        angles = np.linspace(0, 180, sinogram.shape[0])

        for i, angle in enumerate(angles):
            # Back project each projection
            projection = sinogram[i]

            # Create back projection volume
            back_proj = np.zeros(self.sample_size)
            for z in range(self.sample_size[2]):
                back_proj[:, :, z] = projection

            # Rotate back
            back_proj_rotated = ndimage.rotate(back_proj, -angle, axes=(0, 2),
                                             reshape=False, prefilter=False)

            reconstructed += back_proj_rotated

        reconstructed /= len(angles)

        return reconstructed

    def generate_dataset(self, output_dir='output'):
        """
        Generate complete tomography dataset
        """
        print("Starting complete dataset generation...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate grain structure
        grain_map = self.generate_grain_structure()

        # Add porosity and defects
        porosity_map, defect_map = self.add_porosity_and_defects(grain_map)

        # Generate attenuation map
        attenuation_map = self.generate_attenuation_map(grain_map, porosity_map, defect_map)

        # Generate projection data
        sinogram = self.generate_projection_data(attenuation_map)

        # Reconstruct (for synthetic ground truth)
        reconstructed = self.reconstruct_tomography(sinogram)

        # Save data
        np.save(f'{output_dir}/grain_map.npy', grain_map)
        np.save(f'{output_dir}/porosity_map.npy', porosity_map)
        np.save(f'{output_dir}/defect_map.npy', defect_map)
        np.save(f'{output_dir}/attenuation_map.npy', attenuation_map)
        np.save(f'{output_dir}/sinogram.npy', sinogram)
        np.save(f'{output_dir}/reconstructed_volume.npy', reconstructed)

        print(f"Dataset saved to {output_dir}")

        return {
            'grain_map': grain_map,
            'porosity_map': porosity_map,
            'defect_map': defect_map,
            'attenuation_map': attenuation_map,
            'sinogram': sinogram,
            'reconstructed_volume': reconstructed
        }

def main():
    """Main function to generate tomography data"""
    print("Synthetic Synchrotron X-ray Tomography Generator")
    print("=" * 50)

    # Generate initial state (pre-test)
    print("\nGenerating initial state (pre-test)...")
    generator = SyntheticTomographyGenerator(
        sample_size=(80, 80, 80),
        voxel_size=0.5,
        material_type='interconnect'
    )

    initial_data = generator.generate_dataset('synthetic_synchrotron_data/tomography/initial')

    print("\nInitial state generation complete!")
    print(f"Sample size: {initial_data['attenuation_map'].shape}")
    print(f"Number of grains: {len(np.unique(initial_data['grain_map']))}")
    print(f"Porosity fraction: {np.mean(initial_data['porosity_map']):.3f}")

if __name__ == "__main__":
    main()