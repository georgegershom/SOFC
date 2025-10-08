#!/usr/bin/env python3
"""
Data Visualization Script for FEM Validation Dataset
==================================================

This script creates visualizations of the generated validation dataset
to help understand the data structure and relationships.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

def load_dataset(filename='validation_dataset.json'):
    """Load the validation dataset"""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_residual_stress_distribution(dataset):
    """Plot residual stress distribution from XRD measurements"""
    xrd_data = dataset['macro_scale']['xrd_measurements']
    positions = np.array(xrd_data['positions'])
    stresses = np.array(xrd_data['stress_values'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Stress vs position
    ax1.plot(positions * 1000, stresses / 1e6, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Residual Stress (MPa)')
    ax1.set_title('Surface Residual Stress Distribution (XRD)')
    ax1.grid(True, alpha=0.3)
    
    # Stress histogram
    ax2.hist(stresses / 1e6, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Residual Stress (MPa)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Stress Distribution Histogram')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('residual_stress_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_microstructure(dataset):
    """Plot microstructure with grains and pores"""
    micro_data = dataset['meso_scale']['microstructure']
    grain_centers = np.array(micro_data['grain_centers'])
    grain_sizes = np.array(micro_data['grain_sizes'])
    pore_centers = np.array(micro_data['pore_centers'])
    pore_sizes = np.array(micro_data['pore_sizes'])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot grains
    for i, (center, size) in enumerate(zip(grain_centers, grain_sizes)):
        if i < 100:  # Limit for visualization
            circle = Circle(center, size/2, alpha=0.3, color='lightblue', edgecolor='blue')
            ax.add_patch(circle)
    
    # Plot pores
    for i, (center, size) in enumerate(zip(pore_centers, pore_sizes)):
        circle = Circle(center, size/2, alpha=0.8, color='red', edgecolor='darkred')
        ax.add_patch(circle)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Position (μm)')
    ax.set_title('Microstructure: Grains (blue) and Pores (red)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('microstructure_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_grain_size_distribution(dataset):
    """Plot grain size distribution"""
    grain_data = dataset['meso_scale']['grain_sizes']
    sizes = np.array(grain_data['distribution'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(sizes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.set_xlabel('Grain Size (μm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Grain Size Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Log-normal fit
    ax2.hist(sizes, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black', label='Data')
    
    # Fit log-normal distribution
    mu, sigma = np.mean(np.log(sizes)), np.std(np.log(sizes))
    x = np.linspace(sizes.min(), sizes.max(), 100)
    pdf = (1/(x * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x) - mu) / sigma)**2)
    ax2.plot(x, pdf, 'r-', linewidth=2, label=f'Log-normal fit (μ={mu:.2f}, σ={sigma:.2f})')
    
    ax2.set_xlabel('Grain Size (μm)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Grain Size Distribution with Log-normal Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grain_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_crack_analysis(dataset):
    """Plot crack locations and analysis"""
    crack_data = dataset['crack_data']['crack_locations']
    positions = np.array(crack_data['positions'])
    lengths = np.array(crack_data['lengths'])
    orientations = np.array(crack_data['orientations'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Crack locations
    ax1.scatter(positions[:, 0], positions[:, 1], c=lengths, cmap='Reds', s=100, alpha=0.7)
    ax1.set_xlabel('Position (μm)')
    ax1.set_ylabel('Position (μm)')
    ax1.set_title('Crack Locations (color = length)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Crack length distribution
    ax2.hist(lengths, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Crack Length (μm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Crack Length Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Crack orientation distribution
    ax3.hist(orientations, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Orientation (radians)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Crack Orientation Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Length vs orientation
    ax4.scatter(orientations, lengths, alpha=0.7, color='green')
    ax4.set_xlabel('Orientation (radians)')
    ax4.set_ylabel('Crack Length (μm)')
    ax4.set_title('Crack Length vs Orientation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('crack_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_fem_simulation_data(dataset):
    """Plot FEM simulation results"""
    fem_data = dataset['fem_simulation']['full_field']
    coordinates = fem_data['coordinates']
    temperature = np.array(fem_data['temperature'])
    stress_xx = np.array(fem_data['stress']['xx'])
    
    # Create 2D slices for visualization
    nx, ny, nz = fem_data['mesh_info']['nx'], fem_data['mesh_info']['ny'], fem_data['mesh_info']['nz']
    
    # Reshape data for 2D plotting
    T_2d = temperature[:, :, nz//2]  # Middle slice
    S_2d = stress_xx[:, :, nz//2]   # Middle slice
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Temperature field
    im1 = ax1.imshow(T_2d, cmap='hot', origin='lower', aspect='equal')
    ax1.set_title('Temperature Field (K)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax1)
    
    # Stress field
    im2 = ax2.imshow(S_2d / 1e6, cmap='RdBu_r', origin='lower', aspect='equal')
    ax2.set_title('Stress Field (MPa)')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('fem_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_collocation_points(dataset):
    """Plot collocation point data"""
    collocation_data = dataset['fem_simulation']['collocation_points']
    points = collocation_data['points']
    
    # Extract coordinates and data
    coords = np.array([point['coordinates'] for point in points])
    temperatures = np.array([point['temperature'] for point in points])
    stresses = np.array([point['stress'] for point in points])
    point_types = [point['point_type'] for point in points]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Collocation point locations
    colors = {'pore': 'red', 'grain_boundary': 'blue', 'surface': 'green', 'bulk': 'gray'}
    for i, (coord, ptype) in enumerate(zip(coords, point_types)):
        ax1.scatter(coord[0] * 1000, coord[1] * 1000, c=colors[ptype], s=50, alpha=0.7, label=ptype if i < 4 else "")
    
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Collocation Point Locations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature distribution
    ax2.hist(temperatures, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Temperature Distribution at Collocation Points')
    ax2.grid(True, alpha=0.3)
    
    # Stress distribution
    stress_magnitudes = np.sqrt(np.sum(np.array(stresses)**2, axis=1))
    ax3.hist(stress_magnitudes / 1e6, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Stress Magnitude (MPa)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Stress Magnitude Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Temperature vs stress
    ax4.scatter(temperatures, stress_magnitudes / 1e6, alpha=0.7, color='green')
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Stress Magnitude (MPa)')
    ax4.set_title('Temperature vs Stress at Collocation Points')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('collocation_points_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(dataset):
    """Create summary statistics table"""
    print("FEM Validation Dataset Summary Statistics")
    print("=" * 50)
    
    # Macro-scale statistics
    xrd_data = dataset['macro_scale']['xrd_measurements']
    stresses = np.array(xrd_data['stress_values'])
    print(f"\nMacro-Scale Residual Stress:")
    print(f"  Mean: {np.mean(stresses)/1e6:.1f} MPa")
    print(f"  Std:  {np.std(stresses)/1e6:.1f} MPa")
    print(f"  Min:  {np.min(stresses)/1e6:.1f} MPa")
    print(f"  Max:  {np.max(stresses)/1e6:.1f} MPa")
    
    # Meso-scale statistics
    grain_data = dataset['meso_scale']['grain_sizes']
    grain_sizes = np.array(grain_data['distribution'])
    print(f"\nMeso-Scale Grain Sizes:")
    print(f"  Mean: {np.mean(grain_sizes):.2f} μm")
    print(f"  Std:  {np.std(grain_sizes):.2f} μm")
    print(f"  Count: {len(grain_sizes)} grains")
    
    # Crack statistics
    crack_data = dataset['crack_data']['crack_locations']
    crack_lengths = np.array(crack_data['lengths'])
    print(f"\nCrack Analysis:")
    print(f"  Total cracks: {len(crack_lengths)}")
    print(f"  Mean length: {np.mean(crack_lengths):.2f} μm")
    print(f"  Max length:  {np.max(crack_lengths):.2f} μm")
    
    # FEM simulation statistics
    fem_data = dataset['fem_simulation']['full_field']
    mesh_info = fem_data['mesh_info']
    print(f"\nFEM Simulation:")
    print(f"  Total nodes: {mesh_info['total_nodes']:,}")
    print(f"  Total elements: {mesh_info['total_elements']:,}")
    print(f"  Mesh size: {mesh_info['nx']} × {mesh_info['ny']} × {mesh_info['nz']}")
    
    # Collocation points
    collocation_data = dataset['fem_simulation']['collocation_points']
    print(f"\nCollocation Points:")
    print(f"  Total points: {collocation_data['total_points']}")
    for ptype, count in collocation_data['point_types'].items():
        print(f"  {ptype}: {count} points")

def main():
    """Main visualization function"""
    print("Loading validation dataset...")
    dataset = load_dataset()
    
    print("Creating visualizations...")
    
    # Create all visualizations
    plot_residual_stress_distribution(dataset)
    plot_microstructure(dataset)
    plot_grain_size_distribution(dataset)
    plot_crack_analysis(dataset)
    plot_fem_simulation_data(dataset)
    plot_collocation_points(dataset)
    
    # Print summary statistics
    create_summary_statistics(dataset)
    
    print("\nVisualization complete! Generated files:")
    print("- residual_stress_analysis.png")
    print("- microstructure_visualization.png")
    print("- grain_size_distribution.png")
    print("- crack_analysis.png")
    print("- fem_simulation_results.png")
    print("- collocation_points_analysis.png")

if __name__ == "__main__":
    main()