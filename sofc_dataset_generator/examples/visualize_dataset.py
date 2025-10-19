#!/usr/bin/env python3
"""
Example: Visualize SOFC Dataset

This example demonstrates how to load and visualize the generated
SOFC synthetic dataset.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_export.visualization import DatasetVisualizer


def load_dataset(dataset_path: str):
    """Load dataset from HDF5 file"""
    dataset = {
        'warp_data': [],
        'stress_data': [],
        'parameters': [],
        'metadata': []
    }
    
    with h5py.File(dataset_path, 'r') as f:
        # Load warp data
        for sample_id in f['warp_data'].keys():
            sample_group = f['warp_data'][sample_id]
            sample_data = {
                'top_height_map': sample_group['top_height_map'][:],
                'top_height_x': sample_group['top_height_x'][:],
                'top_height_y': sample_group['top_height_y'][:],
                'bottom_height_map': sample_group['bottom_height_map'][:],
                'top_point_cloud': sample_group['top_point_cloud'][:],
                'bottom_point_cloud': sample_group['bottom_point_cloud'][:],
                'top_warp_metrics': dict(sample_group.attrs),
                'bottom_warp_metrics': {}
            }
            dataset['warp_data'].append(sample_data)
        
        # Load stress data
        for sample_id in f['stress_data'].keys():
            sample_group = f['stress_data'][sample_id]
            sample_data = {
                'electrolyte_stress_tensor': sample_group['electrolyte_stress_tensor'][:],
                'electrolyte_von_mises': sample_group['electrolyte_von_mises'][:],
                'electrolyte_principal_stresses': sample_group['electrolyte_principal_stresses'][:],
                'electrolyte_stress_maps': {
                    'von_mises_map': sample_group['von_mises_map'][:],
                    'principal_1_map': sample_group['principal_1_map'][:],
                    'x_grid': sample_group['x_grid'][:],
                    'y_grid': sample_group['y_grid'][:]
                },
                'stress_metrics': {
                    'electrolyte': {}
                }
            }
            dataset['stress_data'].append(sample_data)
        
        # Load parameters
        param_df = f['parameters/dataframe'][:]
        param_names = f['parameters/dataframe'].attrs['columns']
        dataset['parameters'] = [dict(zip(param_names, row)) for row in param_df]
    
    return dataset


def main():
    """Main visualization example"""
    print("SOFC Dataset Visualization Example")
    print("=" * 40)
    
    # Check if dataset exists
    dataset_path = './small_dataset/sofc_dataset.h5'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run generate_small_dataset.py first")
        return
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset['warp_data'])} samples")
    
    # Create visualizer
    visualizer = DatasetVisualizer(dataset)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Dataset overview
    visualizer.plot_dataset_overview(save_path='dataset_overview.png')
    
    # Sample visualizations
    for i in range(min(3, len(dataset['warp_data']))):
        print(f"Visualizing sample {i}...")
        
        # Warp visualization
        visualizer.plot_warp_field(i, save_path=f'warp_sample_{i}.png')
        
        # Stress visualization
        visualizer.plot_stress_field(i, save_path=f'stress_sample_{i}.png')
    
    # Parameter analysis
    visualizer.plot_parameter_distributions(save_path='parameter_distributions.png')
    
    # Correlation analysis
    visualizer.plot_parameter_correlations(save_path='parameter_correlations.png')
    
    print("\nVisualization completed!")
    print("Check the current directory for generated plots.")


if __name__ == "__main__":
    main()