#!/usr/bin/env python3
"""
Visualization script for the SOFC fracture dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from pathlib import Path

def visualize_sample(sample_id=0):
    """Visualize data from a specific sample."""
    sample_dir = Path(f'fracture_dataset/sample_{sample_id:03d}')
    
    # Load phase field data
    with h5py.File(sample_dir / 'phase_field_data.h5', 'r') as f:
        phase_field = f['phase_field'][:]
        time_array = f['physical_time'][:]
    
    # Load performance data
    with open(sample_dir / 'performance_data.json', 'r') as f:
        performance = json.load(f)
    
    # Load SEM data
    with h5py.File(sample_dir / 'sem_data.h5', 'r') as f:
        first_image = f['images/image_0/image_data'][:]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'SOFC Fracture Dataset - Sample {sample_id}', fontsize=16)
    
    # Phase field evolution
    times_to_show = [0, len(time_array)//4, len(time_array)//2, -1]
    for i, t_idx in enumerate(times_to_show[:3]):
        ax = axes[0, i]
        # Show middle z-slice
        z_mid = phase_field.shape[2] // 2
        im = ax.imshow(phase_field[:, :, z_mid, t_idx], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f't = {time_array[t_idx]/3600:.1f} hours')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        plt.colorbar(im, ax=ax, label='Phase field')
    
    # SEM image
    axes[0, 2].remove()
    ax_sem = fig.add_subplot(2, 3, 3)
    ax_sem.imshow(first_image, cmap='gray')
    ax_sem.set_title('SEM Post-mortem Image')
    ax_sem.set_xlabel('X (pixels)')
    ax_sem.set_ylabel('Y (pixels)')
    
    # Performance degradation
    time_hours = np.array(performance['time_hours'])
    
    ax = axes[1, 0]
    ax.plot(time_hours, performance['electrochemical_performance']['voltage_V'])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Voltage Degradation')
    ax.grid(True)
    
    ax = axes[1, 1]
    ax.plot(time_hours, performance['electrochemical_performance']['area_specific_resistance_ohm_cm2'])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('ASR (Ω·cm²)')
    ax.set_title('Resistance Increase')
    ax.grid(True)
    
    ax = axes[1, 2]
    ax.plot(time_hours, np.array(performance['delamination_area_m2']) * 1e6)  # Convert to mm²
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Delamination Area (mm²)')
    ax.set_title('Crack Area Growth')
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_dataset_statistics():
    """Plot statistics across all samples in the dataset."""
    dataset_dir = Path('fracture_dataset')
    
    # Collect statistics from all samples
    final_crack_areas = []
    final_voltages = []
    num_nucleation_sites = []
    
    for sample_dir in sorted(dataset_dir.glob('sample_*')):
        # Load performance data
        with open(sample_dir / 'performance_data.json', 'r') as f:
            performance = json.load(f)
        
        final_crack_areas.append(performance['delamination_area_m2'][-1] * 1e6)  # mm²
        final_voltages.append(performance['electrochemical_performance']['voltage_V'][-1])
        
        # Load nucleation site count
        with open(sample_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Count nucleation sites from crack measurements
        crack_data = metadata['crack_measurements'][0]  # First SEM image
        num_nucleation_sites.append(crack_data['num_cracks'])
    
    # Create statistics plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dataset Statistics (All Samples)', fontsize=16)
    
    # Final crack area distribution
    axes[0, 0].hist(final_crack_areas, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Final Crack Area (mm²)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Final Crack Areas')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Final voltage distribution
    axes[0, 1].hist(final_voltages, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Final Voltage (V)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Final Voltages')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation between crack area and voltage
    axes[1, 0].scatter(final_crack_areas, final_voltages, alpha=0.6)
    axes[1, 0].set_xlabel('Final Crack Area (mm²)')
    axes[1, 0].set_ylabel('Final Voltage (V)')
    axes[1, 0].set_title('Crack Area vs. Voltage Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    correlation = np.corrcoef(final_crack_areas, final_voltages)[0, 1]
    axes[1, 0].text(0.05, 0.95, f'R = {correlation:.3f}', transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Number of nucleation sites
    axes[1, 1].hist(num_nucleation_sites, bins=range(1, max(num_nucleation_sites)+2), 
                   alpha=0.7, edgecolor='black', color='green')
    axes[1, 1].set_xlabel('Number of Nucleation Sites')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Nucleation Sites')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nDataset Summary Statistics:")
    print(f"Number of samples: {len(final_crack_areas)}")
    print(f"Final crack area: {np.mean(final_crack_areas):.3f} ± {np.std(final_crack_areas):.3f} mm²")
    print(f"Final voltage: {np.mean(final_voltages):.3f} ± {np.std(final_voltages):.3f} V")
    print(f"Crack-voltage correlation: {correlation:.3f}")
    print(f"Average nucleation sites: {np.mean(num_nucleation_sites):.1f}")

if __name__ == '__main__':
    print("SOFC Fracture Dataset Visualization")
    print("1. Visualizing sample 0...")
    visualize_sample(0)
    
    print("\n2. Generating dataset statistics...")
    plot_dataset_statistics()
