#!/usr/bin/env python3
"""
Simplified Synthetic Synchrotron X-ray Tomography Data Generator

This is a simplified version that generates essential tomography data
without complex Voronoi computations that may cause memory issues.
"""

import numpy as np
import json
import os

def generate_simple_grain_structure(sample_size=(80, 80, 80)):
    """
    Generate simple grain structure using random assignment
    """
    print("Generating grain structure...")

    # Create grain map with realistic grain sizes
    grain_map = np.zeros(sample_size, dtype=np.int32)

    # Generate grain centers
    n_grains = 50  # Fixed number for simplicity
    grain_centers = np.random.rand(n_grains, 3) * sample_size

    # Assign each voxel to nearest grain center
    for i in range(sample_size[0]):
        for j in range(sample_size[1]):
            for k in range(sample_size[2]):
                distances = np.sqrt((grain_centers[:, 0] - i)**2 +
                                  (grain_centers[:, 1] - j)**2 +
                                  (grain_centers[:, 2] - k)**2)
                grain_map[i, j, k] = np.argmin(distances)

    return grain_map, grain_centers

def generate_porosity_and_defects(sample_size=(80, 80, 80)):
    """
    Generate porosity and defect maps
    """
    print("Generating porosity and defects...")

    # Create porosity (cavitation sites)
    porosity_map = np.random.rand(*sample_size) < 0.02  # 2% porosity

    # Create defect map (initial cracks and inclusions)
    defect_map = np.random.rand(*sample_size) < 0.005  # 0.5% defects

    # Add some structured defects along grain boundaries
    # (This is simplified - in full version would use actual grain boundaries)

    return porosity_map, defect_map

def generate_attenuation_map(sample_size=(80, 80, 80), grain_map=None, porosity_map=None, defect_map=None):
    """
    Generate X-ray attenuation map
    """
    print("Generating attenuation map...")

    if grain_map is None or porosity_map is None or defect_map is None:
        raise ValueError("Missing required maps")

    # Base attenuation values
    attenuation_map = np.ones(sample_size) * 2.5  # Base for metallic material

    # Add grain-to-grain variation
    for grain_id in np.unique(grain_map):
        grain_mask = grain_map == grain_id
        variation = np.random.normal(0, 0.3)
        attenuation_map[grain_mask] *= (1 + variation)

    # Reduce attenuation in porous regions
    attenuation_map[porosity_map] = 0.1

    # Very low attenuation in defects
    attenuation_map[defect_map] = 0.01

    # Add noise
    noise = np.random.normal(0, 0.05, sample_size)
    attenuation_map += noise

    # Ensure non-negative
    attenuation_map = np.maximum(attenuation_map, 0)

    return attenuation_map

def generate_sinogram(attenuation_map, n_angles=180):
    """
    Generate simplified sinogram (projection data)
    """
    print(f"Generating sinogram ({n_angles} angles)...")

    sinogram = np.zeros((n_angles, attenuation_map.shape[1], attenuation_map.shape[0]))

    angles = np.linspace(0, 180, n_angles)

    for i, angle in enumerate(angles):
        # Simplified projection - just sum along one axis
        # (In reality this would be proper Radon transform)
        projection = np.sum(attenuation_map, axis=2)
        sinogram[i] = projection

    return sinogram

def save_dataset(data_dict, output_dir):
    """
    Save dataset to files
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, data in data_dict.items():
        filename = f'{output_dir}/{name}.npy'
        np.save(filename, data)
        print(f"Saved {name} to {filename}")

def main():
    """Main function to generate simplified tomography data"""
    print("Simplified Synthetic Synchrotron X-ray Tomography Generator")
    print("=" * 60)

    # Set sample size
    sample_size = (80, 80, 80)

    print(f"\nGenerating dataset with sample size: {sample_size}")
    print("This may take a few minutes...")

    # Generate grain structure
    grain_map, grain_centers = generate_simple_grain_structure(sample_size)

    # Generate porosity and defects
    porosity_map, defect_map = generate_porosity_and_defects(sample_size)

    # Generate attenuation map
    attenuation_map = generate_attenuation_map(sample_size, grain_map, porosity_map, defect_map)

    # Generate sinogram
    sinogram = generate_sinogram(attenuation_map, n_angles=180)

    # Create data dictionary
    dataset = {
        'grain_map': grain_map,
        'porosity_map': porosity_map,
        'defect_map': defect_map,
        'attenuation_map': attenuation_map,
        'sinogram': sinogram,
        'grain_centers': grain_centers
    }

    # Save dataset
    output_dir = 'synthetic_synchrotron_data/tomography/initial'
    save_dataset(dataset, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"Sample size: {attenuation_map.shape}")
    print(f"Number of grains: {len(np.unique(grain_map))}")
    print(f"Porosity fraction: {np.mean(porosity_map):.4f}")
    print(f"Defect fraction: {np.mean(defect_map):.4f}")
    print(f"Attenuation range: {np.min(attenuation_map):.3f} - {np.max(attenuation_map):.3f}")
    print(f"Sinogram shape: {sinogram.shape}")
    print(f"Files saved to: {output_dir}")

    # Save metadata
    metadata = {
        'sample_size': sample_size,
        'voxel_size_microns': 0.5,
        'material_type': 'interconnect',
        'n_grains': len(np.unique(grain_map)),
        'porosity_fraction': float(np.mean(porosity_map)),
        'defect_fraction': float(np.mean(defect_map)),
        'attenuation_range': [float(np.min(attenuation_map)), float(np.max(attenuation_map))],
        'sinogram_angles': 180,
        'generation_method': 'simplified_synthetic'
    }

    with open(f'{output_dir}/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {output_dir}/dataset_metadata.json")

if __name__ == "__main__":
    main()