"""
Example script for exploring the SOFC Digital Twin Dataset.

This script demonstrates how to load, explore, and analyze the generated dataset.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DataUtils, VisualizationUtils


def load_dataset(dataset_path: str = "data/integrated_dataset.h5"):
    """Load the integrated dataset."""
    print(f"Loading dataset from: {dataset_path}")
    
    dataset = {}
    with h5py.File(dataset_path, 'r') as f:
        # Load all groups
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                dataset[key] = {}
                load_group(f[key], dataset[key])
            else:
                dataset[key] = f[key][:]
    
    print("Dataset loaded successfully!")
    return dataset


def load_group(group, data_dict):
    """Recursively load HDF5 group."""
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            data_dict[key] = {}
            load_group(group[key], data_dict[key])
        else:
            data_dict[key] = group[key][:]


def explore_dataset_structure(dataset):
    """Explore the dataset structure."""
    print("\n" + "="*60)
    print("DATASET STRUCTURE EXPLORATION")
    print("="*60)
    
    def print_structure(data, indent=0):
        for key, value in data.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}/")
                print_structure(value, indent + 1)
            elif isinstance(value, np.ndarray):
                print("  " * indent + f"{key}: {value.shape} {value.dtype}")
            else:
                print("  " * indent + f"{key}: {type(value).__name__}")
    
    print_structure(dataset)


def analyze_time_series_data(dataset):
    """Analyze time series data."""
    print("\n" + "="*60)
    print("TIME SERIES DATA ANALYSIS")
    print("="*60)
    
    if 'time_series' in dataset:
        time_series = dataset['time_series']
        
        # Get time points
        if 'time_points' in time_series:
            time_points = time_series['time_points']
            print(f"Time range: {time_points[0]:.1f} - {time_points[-1]:.1f} seconds")
            print(f"Time step: {time_points[1] - time_points[0]:.1f} seconds")
            print(f"Total duration: {time_points[-1] - time_points[0]:.1f} seconds")
            print(f"Number of time points: {len(time_points)}")
        
        # Analyze each time series
        for key, value in time_series.items():
            if key != 'time_points' and isinstance(value, dict):
                print(f"\n{key}:")
                if 'time_points' in value:
                    print(f"  Time points: {len(value['time_points'])}")
                    print(f"  Time range: {value['time_points'][0]:.1f} - {value['time_points'][-1]:.1f} s")
                
                # Find numeric data
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray) and subvalue.dtype in [np.float64, np.float32, np.int64, np.int32]:
                        print(f"  {subkey}: {subvalue.shape} {subvalue.dtype}")
                        if len(subvalue) > 0:
                            print(f"    Range: {np.min(subvalue):.3f} - {np.max(subvalue):.3f}")
                            print(f"    Mean: {np.mean(subvalue):.3f} ± {np.std(subvalue):.3f}")


def analyze_spatial_data(dataset):
    """Analyze spatial data."""
    print("\n" + "="*60)
    print("SPATIAL DATA ANALYSIS")
    print("="*60)
    
    if 'spatial_data' in dataset:
        spatial_data = dataset['spatial_data']
        
        for key, value in spatial_data.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: {value.shape} {value.dtype}")
                if len(value.shape) >= 2:
                    print(f"  Spatial dimensions: {value.shape[1:]}")
                    print(f"  Time steps: {value.shape[0]}")
                    print(f"  Value range: {np.min(value):.3f} - {np.max(value):.3f}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        print(f"  {subkey}: {subvalue.shape} {subvalue.dtype}")


def analyze_sensor_network(dataset):
    """Analyze sensor network data."""
    print("\n" + "="*60)
    print("SENSOR NETWORK ANALYSIS")
    print("="*60)
    
    if 'sensor_network' in dataset:
        sensor_network = dataset['sensor_network']
        
        for sensor_type, sensors in sensor_network.items():
            if isinstance(sensors, dict):
                print(f"{sensor_type}: {len(sensors)} sensors")
                
                # Analyze first sensor as example
                if sensors:
                    first_sensor = list(sensors.values())[0]
                    if isinstance(first_sensor, dict):
                        print(f"  Example sensor data keys: {list(first_sensor.keys())}")
                        
                        # Check for location data
                        if 'location' in first_sensor:
                            print(f"  Location data available: {first_sensor['location']}")
                        
                        # Check for time series data
                        for key, value in first_sensor.items():
                            if isinstance(value, np.ndarray) and len(value) > 0:
                                print(f"    {key}: {value.shape} {value.dtype}")


def create_summary_plots(dataset, output_dir="plots"):
    """Create summary plots of the dataset."""
    print("\n" + "="*60)
    print("CREATING SUMMARY PLOTS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualization utilities
    viz_utils = VisualizationUtils()
    
    # Plot time series data
    if 'time_series' in dataset:
        time_series = dataset['time_series']
        
        # Find temperature data
        for key, value in time_series.items():
            if 'temperature' in key and isinstance(value, dict):
                if 'temperature' in value and 'time_points' in value:
                    plt.figure(figsize=(12, 6))
                    plt.plot(value['time_points'], value['temperature'])
                    plt.xlabel('Time (s)')
                    plt.ylabel('Temperature (°C)')
                    plt.title(f'Temperature Time Series: {key}')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(output_dir, f'temperature_{key}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Find voltage data
        for key, value in time_series.items():
            if 'voltage' in key and isinstance(value, dict):
                if 'voltage' in value and 'time_points' in value:
                    plt.figure(figsize=(12, 6))
                    plt.plot(value['time_points'], value['voltage'])
                    plt.xlabel('Time (s)')
                    plt.ylabel('Voltage (V)')
                    plt.title(f'Voltage Time Series: {key}')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(output_dir, f'voltage_{key}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
    
    # Plot spatial data
    if 'spatial_data' in dataset:
        spatial_data = dataset['spatial_data']
        
        # Plot temperature field
        if 'temperature_field_2d' in spatial_data:
            temp_field = spatial_data['temperature_field_2d']
            if len(temp_field.shape) == 3:
                # Plot first time step
                plt.figure(figsize=(10, 8))
                plt.imshow(temp_field[0], cmap='hot', origin='lower')
                plt.colorbar(label='Temperature (°C)')
                plt.title('Temperature Field (t=0)')
                plt.xlabel('X (voxels)')
                plt.ylabel('Y (voxels)')
                plt.savefig(os.path.join(output_dir, 'temperature_field_2d.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    print(f"Summary plots saved to: {output_dir}")


def create_data_quality_report(dataset, output_file="data_quality_report.txt"):
    """Create a data quality report."""
    print("\n" + "="*60)
    print("CREATING DATA QUALITY REPORT")
    print("="*60)
    
    data_utils = DataUtils()
    quality_metrics = data_utils.validate_data_quality(dataset)
    
    with open(output_file, 'w') as f:
        f.write("SOFC Digital Twin Dataset - Data Quality Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated at: {pd.Timestamp.now()}\n\n")
        
        f.write("Data Quality Metrics:\n")
        f.write("-" * 30 + "\n")
        
        for component, metrics in quality_metrics.items():
            f.write(f"\n{component}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
        
        # Summary statistics
        f.write("\n\nSummary Statistics:\n")
        f.write("-" * 30 + "\n")
        
        total_components = len(quality_metrics)
        components_with_issues = sum(1 for metrics in quality_metrics.values() 
                                   if metrics.get('missing_values', 0) > 0 or 
                                      metrics.get('infinite_values', 0) > 0)
        
        f.write(f"Total components: {total_components}\n")
        f.write(f"Components with issues: {components_with_issues}\n")
        f.write(f"Data quality: {(total_components - components_with_issues) / total_components * 100:.1f}%\n")
    
    print(f"Data quality report saved to: {output_file}")


def main():
    """Main function for dataset exploration."""
    print("SOFC Digital Twin Dataset Exploration")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset()
    
    # Explore dataset structure
    explore_dataset_structure(dataset)
    
    # Analyze different data types
    analyze_time_series_data(dataset)
    analyze_spatial_data(dataset)
    analyze_sensor_network(dataset)
    
    # Create visualizations
    create_summary_plots(dataset)
    
    # Create data quality report
    create_data_quality_report(dataset)
    
    print("\n" + "="*60)
    print("Dataset exploration complete!")
    print("="*60)


if __name__ == "__main__":
    main()