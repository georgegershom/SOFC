#!/usr/bin/env python3
"""
Example: Load and Analyze SOFC Dataset

This example demonstrates how to load a generated dataset and perform
basic analysis and visualization.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_export.hdf5_exporter import HDF5Exporter
from data_export.ml_dataset import MLDataset


def load_dataset(hdf5_file: str) -> MLDataset:
    """Load dataset from HDF5 file"""
    print(f"Loading dataset from {hdf5_file}...")
    exporter = HDF5Exporter(hdf5_file, mode='r')
    dataset = exporter.load_dataset()
    print(f"Loaded dataset with {len(dataset.samples)} samples")
    return dataset


def analyze_dataset(dataset: MLDataset):
    """Analyze dataset statistics and properties"""
    print("\nDataset Analysis:")
    print("=" * 40)
    
    # Basic statistics
    print(f"Dataset name: {dataset.dataset_name}")
    print(f"Creation date: {dataset.dataset_date}")
    print(f"Number of samples: {len(dataset.samples)}")
    
    # Warp statistics
    warp_magnitudes = [sample.warp_magnitude for sample in dataset.samples]
    print(f"\nWarp Magnitude Statistics:")
    print(f"  Mean: {np.mean(warp_magnitudes):.3f} mm")
    print(f"  Std:  {np.std(warp_magnitudes):.3f} mm")
    print(f"  Min:  {np.min(warp_magnitudes):.3f} mm")
    print(f"  Max:  {np.max(warp_magnitudes):.3f} mm")
    
    # Stress statistics
    max_stresses = [sample.max_stress for sample in dataset.samples]
    print(f"\nMax Stress Statistics:")
    print(f"  Mean: {np.mean(max_stresses):.1f} MPa")
    print(f"  Std:  {np.std(max_stresses):.1f} MPa")
    print(f"  Min:  {np.min(max_stresses):.1f} MPa")
    print(f"  Max:  {np.max(max_stresses):.1f} MPa")
    
    # Parameter statistics
    print(f"\nParameter Statistics:")
    for param, stats in dataset.statistics['parameter_ranges'].items():
        print(f"  {param}: {stats['mean']:.3f} ± {stats['std']:.3f}")


def visualize_sample(dataset: MLDataset, sample_idx: int = 0):
    """Visualize a sample from the dataset"""
    if sample_idx >= len(dataset.samples):
        print(f"Sample index {sample_idx} out of range")
        return
    
    sample = dataset.samples[sample_idx]
    print(f"\nVisualizing sample {sample.sample_id}...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Warp height map
    if sample.warp_height_map is not None:
        im1 = axes[0, 0].contourf(sample.warp_x_coords, sample.warp_y_coords, 
                                 sample.warp_height_map, levels=20, cmap='viridis')
        axes[0, 0].set_title('Warp Height Map')
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=axes[0, 0], label='Height (mm)')
    
    # Displacement magnitude
    displacement_magnitude = np.linalg.norm(sample.warp_displacements, axis=1)
    scatter = axes[0, 1].scatter(sample.warp_points[:, 0], sample.warp_points[:, 1], 
                                c=displacement_magnitude, cmap='plasma', s=20)
    axes[0, 1].set_title('Displacement Magnitude')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Displacement (mm)')
    
    # Surface stress map
    if sample.surface_stress_map is not None:
        im3 = axes[1, 0].contourf(sample.surface_x_coords, sample.surface_y_coords, 
                                 sample.surface_stress_map, levels=20, cmap='plasma')
        axes[1, 0].set_title('Surface Stress Map')
        axes[1, 0].set_xlabel('X (mm)')
        axes[1, 0].set_ylabel('Y (mm)')
        plt.colorbar(im3, ax=axes[1, 0], label='Stress (MPa)')
    
    # Stress distribution histogram
    axes[1, 1].hist(sample.stress_von_mises, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Von Mises Stress Distribution')
    axes[1, 1].set_xlabel('Stress (MPa)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(sample.max_stress, color='red', linestyle='--', 
                      label=f'Max: {sample.max_stress:.1f} MPa')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print sample information
    print(f"Sample {sample.sample_id} Information:")
    print(f"  Warp magnitude: {sample.warp_magnitude:.3f} mm")
    print(f"  Max stress: {sample.max_stress:.1f} MPa")
    print(f"  Simulation time: {sample.simulation_time:.3f} s")
    print(f"  Mesh elements: {sample.mesh_info.get('n_elements', 'N/A')}")


def prepare_ml_data(dataset: MLDataset):
    """Prepare data for ML training"""
    print("\nPreparing ML Training Data:")
    print("=" * 30)
    
    # Get feature and target matrices
    features = dataset.get_feature_matrix('height_map')
    targets = dataset.get_target_matrix('surface_map')
    parameters = dataset.get_parameter_matrix()
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Target matrix shape: {targets.shape}")
    print(f"Parameter matrix shape: {parameters.shape}")
    
    # Check for missing values
    print(f"Features with NaN: {np.isnan(features).sum()}")
    print(f"Targets with NaN: {np.isnan(targets).sum()}")
    
    # Basic statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {np.nanmean(features):.6f}")
    print(f"  Std:  {np.nanstd(features):.6f}")
    print(f"  Min:  {np.nanmin(features):.6f}")
    print(f"  Max:  {np.nanmax(features):.6f}")
    
    print(f"\nTarget statistics:")
    print(f"  Mean: {np.nanmean(targets):.3f}")
    print(f"  Std:  {np.nanstd(targets):.3f}")
    print(f"  Min:  {np.nanmin(targets):.3f}")
    print(f"  Max:  {np.nanmax(targets):.3f}")
    
    return features, targets, parameters


def main():
    """Main function"""
    # Check if dataset exists
    dataset_file = './small_dataset/sofc_dataset.h5'
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found.")
        print("Please run generate_small_dataset.py first.")
        return
    
    # Load dataset
    dataset = load_dataset(dataset_file)
    
    # Analyze dataset
    analyze_dataset(dataset)
    
    # Visualize a sample
    visualize_sample(dataset, sample_idx=0)
    
    # Prepare ML data
    features, targets, parameters = prepare_ml_data(dataset)
    
    print("\nDataset analysis completed!")


if __name__ == "__main__":
    main()