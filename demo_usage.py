#!/usr/bin/env python3
"""
Demonstration of SOFC Fracture Dataset Usage
===========================================

This script demonstrates how to load and use the generated fracture dataset
for various applications including visualization, analysis, and PINN training.

Author: AI Assistant
Date: 2025-10-09
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from pathlib import Path

def demo_data_loading():
    """Demonstrate how to load different types of data from the dataset."""
    print("=== Data Loading Demonstration ===")
    
    sample_dir = Path('fracture_dataset/sample_000')
    
    # 1. Load phase field evolution data
    print("Loading phase field evolution data...")
    with h5py.File(sample_dir / 'phase_field_data.h5', 'r') as f:
        phase_field = f['phase_field'][:]
        time_array = f['physical_time'][:]
        
        print(f"Phase field shape: {phase_field.shape}")
        print(f"Time array shape: {time_array.shape}")
        print(f"Time range: {time_array[0]/3600:.1f} - {time_array[-1]/3600:.1f} hours")
        print(f"Phase field range: {phase_field.min():.3f} - {phase_field.max():.3f}")
    
    # 2. Load SEM data
    print("\nLoading SEM data...")
    with h5py.File(sample_dir / 'sem_data.h5', 'r') as f:
        first_image = f['images/image_0/image_data'][:]
        print(f"SEM image shape: {first_image.shape}")
        print(f"SEM image range: {first_image.min():.3f} - {first_image.max():.3f}")
    
    # 3. Load performance data
    print("\nLoading performance data...")
    with open(sample_dir / 'performance_data.json', 'r') as f:
        performance = json.load(f)
    
    # Parse string arrays from JSON
    voltages = np.fromstring(performance['electrochemical_performance']['voltage_V'].strip('[]'), sep=' ')
    times = np.fromstring(performance['time_hours'].strip('[]'), sep=' ')
    
    print(f"Performance time points: {len(times)}")
    print(f"Voltage range: {voltages.min():.3f} - {voltages.max():.3f} V")
    print(f"Voltage degradation: {(voltages[-1] - voltages[0])*1000:.1f} mV over {times[-1]:.0f} hours")

def demo_visualization():
    """Demonstrate basic visualization of the fracture data."""
    print("\n=== Visualization Demonstration ===")
    
    sample_dir = Path('fracture_dataset/sample_000')
    
    # Load data
    with h5py.File(sample_dir / 'phase_field_data.h5', 'r') as f:
        phase_field = f['phase_field'][:]
        time_array = f['physical_time'][:]
    
    with open(sample_dir / 'performance_data.json', 'r') as f:
        performance = json.load(f)
    
    voltages = np.fromstring(performance['electrochemical_performance']['voltage_V'].strip('[]'), sep=' ')
    times = np.fromstring(performance['time_hours'].strip('[]'), sep=' ')
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SOFC Fracture Dataset - Sample 000', fontsize=16)
    
    # Phase field evolution at different times
    time_indices = [0, len(time_array)//4, len(time_array)//2, -1]
    z_mid = phase_field.shape[2] // 2  # Middle z-slice
    
    for i, t_idx in enumerate(time_indices[:3]):
        ax = axes[0, i]
        im = ax.imshow(phase_field[:, :, z_mid, t_idx], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f't = {time_array[t_idx]/3600:.1f} hours')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        plt.colorbar(im, ax=ax, label='Phase field')
    
    # Crack area evolution
    ax = axes[0, 2]
    crack_areas = []
    for t in range(phase_field.shape[3]):
        crack_area = np.sum(phase_field[:, :, :, t] > 0.5)
        crack_areas.append(crack_area)
    
    ax.plot(time_array / 3600, crack_areas, 'b-', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Crack Area (voxels)')
    ax.set_title('Crack Evolution')
    ax.grid(True, alpha=0.3)
    
    # Performance degradation
    ax = axes[1, 0]
    ax.plot(times, voltages, 'g-', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Voltage Degradation')
    ax.grid(True, alpha=0.3)
    
    # SEM image
    with h5py.File(sample_dir / 'sem_data.h5', 'r') as f:
        sem_image = f['images/image_0/image_data'][:]
    
    ax = axes[1, 1]
    ax.imshow(sem_image, cmap='gray')
    ax.set_title('SEM Post-mortem Image')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Correlation plot
    ax = axes[1, 2]
    delamination_area = np.fromstring(performance['delamination_area_m2'].strip('[]'), sep=' ')
    ax.scatter(delamination_area * 1e6, voltages, alpha=0.7)  # Convert to mm²
    ax.set_xlabel('Delamination Area (mm²)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Area vs. Performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_pinn_data_preparation():
    """Demonstrate how to prepare data for PINN training."""
    print("\n=== PINN Data Preparation Demonstration ===")
    
    # Load multiple samples for training
    dataset_dir = Path('fracture_dataset')
    sample_dirs = sorted(dataset_dir.glob('sample_*'))[:3]  # Use first 3 samples
    
    all_coords = []
    all_phi = []
    
    for sample_dir in sample_dirs:
        print(f"Processing {sample_dir.name}...")
        
        # Load phase field data
        with h5py.File(sample_dir / 'phase_field_data.h5', 'r') as f:
            phase_field = f['phase_field'][:]
            time_array = f['physical_time'][:]
        
        # Create normalized coordinate grid
        nx, ny, nz, nt = phase_field.shape
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        t = time_array / time_array[-1]  # Normalize time to [0,1]
        
        X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')
        
        # Flatten for PINN input
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel(), T.ravel()], axis=1)
        phi_values = phase_field.ravel()
        
        all_coords.append(coords)
        all_phi.append(phi_values)
    
    # Combine all samples
    coords_combined = np.concatenate(all_coords, axis=0)
    phi_combined = np.concatenate(all_phi, axis=0)
    
    print(f"Total training points: {len(coords_combined):,}")
    print(f"Input dimensions: {coords_combined.shape[1]} (x, y, z, t)")
    print(f"Output dimensions: 1 (phase field)")
    print(f"Coordinate ranges:")
    print(f"  X: {coords_combined[:, 0].min():.3f} - {coords_combined[:, 0].max():.3f}")
    print(f"  Y: {coords_combined[:, 1].min():.3f} - {coords_combined[:, 1].max():.3f}")
    print(f"  Z: {coords_combined[:, 2].min():.3f} - {coords_combined[:, 2].max():.3f}")
    print(f"  T: {coords_combined[:, 3].min():.3f} - {coords_combined[:, 3].max():.3f}")
    print(f"Phase field range: {phi_combined.min():.3f} - {phi_combined.max():.3f}")
    
    # Subsample for demonstration
    n_sample = min(10000, len(coords_combined))
    indices = np.random.choice(len(coords_combined), n_sample, replace=False)
    
    coords_sample = coords_combined[indices]
    phi_sample = phi_combined[indices]
    
    print(f"\nSubsampled to {n_sample:,} points for training efficiency")
    
    return coords_sample, phi_sample

def demo_statistical_analysis():
    """Demonstrate statistical analysis of the dataset."""
    print("\n=== Statistical Analysis Demonstration ===")
    
    dataset_dir = Path('fracture_dataset')
    
    # Collect statistics from all samples
    final_crack_areas = []
    final_voltages = []
    nucleation_counts = []
    
    for sample_dir in sorted(dataset_dir.glob('sample_*')):
        # Load phase field data
        with h5py.File(sample_dir / 'phase_field_data.h5', 'r') as f:
            phase_field = f['phase_field'][:]
        
        # Load performance data
        with open(sample_dir / 'performance_data.json', 'r') as f:
            performance = json.load(f)
        
        # Calculate final crack area
        final_crack_area = np.sum(phase_field[:, :, :, -1] > 0.5)
        final_crack_areas.append(final_crack_area)
        
        # Extract final voltage
        voltages = np.fromstring(performance['electrochemical_performance']['voltage_V'].strip('[]'), sep=' ')
        final_voltages.append(voltages[-1])
        
        # Count nucleation sites (simplified)
        with open(sample_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Count cracks from first SEM measurement
        crack_data = metadata['crack_measurements'][0]
        nucleation_counts.append(crack_data['num_cracks'])
    
    # Print statistics
    print(f"Dataset Statistics (n={len(final_crack_areas)} samples):")
    print(f"Final Crack Area:")
    print(f"  Mean: {np.mean(final_crack_areas):.1f} ± {np.std(final_crack_areas):.1f} voxels")
    print(f"  Range: {np.min(final_crack_areas):.0f} - {np.max(final_crack_areas):.0f} voxels")
    
    print(f"Final Voltage:")
    print(f"  Mean: {np.mean(final_voltages):.3f} ± {np.std(final_voltages):.3f} V")
    print(f"  Range: {np.min(final_voltages):.3f} - {np.max(final_voltages):.3f} V")
    
    print(f"Nucleation Sites:")
    print(f"  Mean: {np.mean(nucleation_counts):.1f} sites per sample")
    print(f"  Range: {np.min(nucleation_counts)} - {np.max(nucleation_counts)} sites")
    
    # Calculate correlation
    correlation = np.corrcoef(final_crack_areas, final_voltages)[0, 1]
    print(f"Crack Area - Voltage Correlation: r = {correlation:.3f}")
    
    # Create summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Crack area distribution
    axes[0].hist(final_crack_areas, bins=8, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Final Crack Area (voxels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Crack Area Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Voltage distribution
    axes[1].hist(final_voltages, bins=8, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('Final Voltage (V)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Voltage Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Correlation
    axes[2].scatter(final_crack_areas, final_voltages, alpha=0.7)
    axes[2].set_xlabel('Final Crack Area (voxels)')
    axes[2].set_ylabel('Final Voltage (V)')
    axes[2].set_title(f'Correlation (r = {correlation:.3f})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run all demonstrations."""
    print("SOFC Fracture Dataset Usage Demonstration")
    print("=" * 50)
    
    # Check if dataset exists
    if not Path('fracture_dataset').exists():
        print("Error: Dataset not found. Please run 'python3 fracture_dataset_generator.py' first.")
        return
    
    try:
        # Run demonstrations
        demo_data_loading()
        demo_visualization()
        coords, phi = demo_pinn_data_preparation()
        demo_statistical_analysis()
        
        print("\n" + "=" * 50)
        print("Demonstration complete!")
        print("\nNext steps:")
        print("1. Use 'python3 pinn_fracture_model.py' for PINN training")
        print("2. Use 'python3 dataset_analysis.py' for comprehensive analysis")
        print("3. Use 'python3 visualize_dataset.py' for interactive exploration")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Please ensure all required packages are installed:")
        print("pip3 install numpy matplotlib h5py scipy scikit-image")

if __name__ == '__main__':
    main()