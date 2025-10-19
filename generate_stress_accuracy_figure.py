#!/usr/bin/env python3
"""
Generate Figure: Spatial Accuracy of MF-DL Predictions: Thermo-mechanical Stress (σ_VM) after 5,000 Hours

This script creates a professional side-by-side comparison of MF-DL predictions vs HF simulation
for von Mises stress in the anode-electrolyte region of a SOFC, mimicking ABAQUS-style results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_custom_colormap():
    """Create a professional scientific colormap similar to ABAQUS"""
    colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#FFFFFF']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('stress_map', colors, N=n_bins)
    return cmap

def generate_ni_particles(nx, ny, n_particles=25, particle_size_range=(3, 8)):
    """Generate Ni nanoparticle cluster positions and sizes"""
    particles = []
    
    # Focus particles in the anode region (upper 70% of domain)
    anode_region_y_min = int(0.3 * ny)
    
    for _ in range(n_particles):
        # Random position in anode region
        x = np.random.randint(5, nx-5)
        y = np.random.randint(anode_region_y_min, ny-5)
        size = np.random.uniform(*particle_size_range)
        particles.append((x, y, size))
    
    return particles

def create_base_stress_field(nx, ny):
    """Create the base stress field with CTE mismatch and microstructural features"""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize stress field
    stress = np.zeros((ny, nx))
    
    # 1. CTE mismatch - high stress band at anode-electrolyte interface
    interface_y = 0.3  # Interface at 30% from bottom
    interface_width = 0.05
    
    # Create Gaussian profile for interface stress
    interface_stress = 85 * np.exp(-((Y - interface_y) / interface_width)**2)
    stress += interface_stress
    
    # 2. Add background stress gradient
    # Higher stress in anode, lower in electrolyte
    background_anode = 25 * (Y > interface_y) * (1 + 0.3 * Y)
    background_electrolyte = 15 * (Y <= interface_y) * (1 - 0.5 * Y)
    stress += background_anode + background_electrolyte
    
    # 3. Add edge effects
    edge_factor = 1.2
    edge_width = 0.1
    
    # Left and right edges
    left_edge = edge_factor * np.exp(-X / edge_width)
    right_edge = edge_factor * np.exp(-(1-X) / edge_width)
    stress += 10 * (left_edge + right_edge)
    
    # Top and bottom edges
    bottom_edge = edge_factor * np.exp(-Y / edge_width)
    top_edge = edge_factor * np.exp(-(1-Y) / edge_width)
    stress += 8 * (bottom_edge + top_edge)
    
    return stress

def add_ni_particle_stress_concentrations(stress, particles, nx, ny, strength_factor=1.0):
    """Add stress concentrations around Ni particles"""
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    
    for px, py, size in particles:
        # Create distance field from particle center
        distances = np.sqrt((x_coords - px)**2 + (y_coords - py)**2)
        
        # Stress concentration around particle
        # High stress at particle boundary, decaying with distance
        particle_stress = np.zeros_like(distances, dtype=float)
        
        # Inside particle - moderate stress
        inside_mask = distances <= size/2
        particle_stress[inside_mask] = 20 * strength_factor
        
        # Particle boundary - high stress concentration
        boundary_mask = (distances > size/2) & (distances <= size)
        particle_stress[boundary_mask] = 60 * strength_factor * np.exp(-2*(distances[boundary_mask] - size/2))
        
        # Outside particle - stress concentration decaying with distance
        outside_mask = distances > size
        decay_factor = np.exp(-0.3 * (distances[outside_mask] - size))
        particle_stress[outside_mask] = 35 * strength_factor * decay_factor
        
        stress += particle_stress
    
    return stress

def add_realistic_noise_and_smoothing(stress, noise_level=0.02, smooth_sigma=0.8):
    """Add realistic noise and smooth the field to mimic FEM results"""
    # Add small amount of noise
    noise = np.random.normal(0, noise_level * np.max(stress), stress.shape)
    stress_noisy = stress + noise
    
    # Smooth to mimic FEM interpolation
    stress_smooth = gaussian_filter(stress_noisy, sigma=smooth_sigma)
    
    # Ensure non-negative values
    stress_smooth = np.maximum(stress_smooth, 0)
    
    return stress_smooth

def create_mf_dl_prediction(hf_stress, correlation=0.98):
    """Create MF-DL prediction that correlates highly with HF but has subtle differences"""
    # Start with HF stress
    mf_stress = hf_stress.copy()
    
    # Add small systematic differences
    # 1. Slightly different smoothing
    mf_stress = gaussian_filter(mf_stress, sigma=0.9)
    
    # 2. Small scaling differences in high-stress regions
    high_stress_mask = hf_stress > 70
    mf_stress[high_stress_mask] *= 0.95
    
    # 3. Add small random variations to achieve target correlation
    current_corr = np.corrcoef(hf_stress.flatten(), mf_stress.flatten())[0, 1]
    
    if current_corr > correlation:
        # Add noise to reduce correlation
        noise_strength = 0.1
        while current_corr > correlation and noise_strength < 1.0:
            noise = np.random.normal(0, noise_strength * np.std(hf_stress), hf_stress.shape)
            mf_stress_test = mf_stress + noise
            mf_stress_test = np.maximum(mf_stress_test, 0)  # Keep non-negative
            test_corr = np.corrcoef(hf_stress.flatten(), mf_stress_test.flatten())[0, 1]
            if abs(test_corr - correlation) < abs(current_corr - correlation):
                mf_stress = mf_stress_test
                current_corr = test_corr
            noise_strength += 0.05
    
    return mf_stress

def create_layer_boundaries(nx, ny):
    """Create layer boundary information for visualization"""
    interface_y = int(0.3 * ny)  # Anode-electrolyte interface
    
    # Create masks for different regions
    anode_mask = np.zeros((ny, nx), dtype=bool)
    electrolyte_mask = np.zeros((ny, nx), dtype=bool)
    
    anode_mask[interface_y:, :] = True
    electrolyte_mask[:interface_y, :] = True
    
    return anode_mask, electrolyte_mask, interface_y

def plot_stress_comparison():
    """Create the main comparison figure"""
    # Parameters
    nx, ny = 200, 150  # Grid resolution
    
    # Generate Ni particles
    particles = generate_ni_particles(nx, ny, n_particles=28, particle_size_range=(4, 10))
    
    # Create HF simulation (ground truth)
    print("Generating HF simulation stress field...")
    hf_base = create_base_stress_field(nx, ny)
    hf_stress = add_ni_particle_stress_concentrations(hf_base, particles, nx, ny, strength_factor=1.0)
    hf_stress = add_realistic_noise_and_smoothing(hf_stress, noise_level=0.015, smooth_sigma=0.7)
    
    # Create MF-DL prediction
    print("Generating MF-DL prediction stress field...")
    mf_stress = create_mf_dl_prediction(hf_stress, correlation=0.98)
    
    # Calculate actual correlation
    actual_correlation = np.corrcoef(hf_stress.flatten(), mf_stress.flatten())[0, 1]
    print(f"Achieved spatial correlation: {actual_correlation:.3f}")
    
    # Create layer boundaries
    anode_mask, electrolyte_mask, interface_y = create_layer_boundaries(nx, ny)
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Spatial Accuracy of MF-DL Predictions: Thermo-mechanical Stress (σ_VM) after 5,000 Hours', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Set consistent color limits
    vmin, vmax = 0, 100
    
    # Coordinate arrays for plotting
    x = np.linspace(0, 10, nx)  # 10 mm domain
    y = np.linspace(0, 7.5, ny)  # 7.5 mm domain
    
    # Plot MF-DL prediction (left panel)
    im1 = ax1.contourf(x, y, mf_stress, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title('(a) MF-DL Prediction', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Distance (mm)', fontsize=12)
    ax1.set_ylabel('Distance (mm)', fontsize=12)
    
    # Add interface line
    interface_y_mm = interface_y * 7.5 / ny
    ax1.axhline(y=interface_y_mm, color='black', linewidth=2, linestyle='--', alpha=0.7)
    
    # Add Ni particles as circles
    for px, py, size in particles:
        px_mm = px * 10 / nx
        py_mm = py * 7.5 / ny
        size_mm = size * 10 / nx * 0.3  # Scale particle size
        circle = plt.Circle((px_mm, py_mm), size_mm, fill=False, 
                          edgecolor='black', linewidth=1, alpha=0.6)
        ax1.add_patch(circle)
    
    # Plot HF simulation (right panel)
    im2 = ax2.contourf(x, y, hf_stress, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title('(b) HF Simulation (Ground Truth)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Distance (mm)', fontsize=12)
    ax2.set_ylabel('Distance (mm)', fontsize=12)
    
    # Add interface line
    ax2.axhline(y=interface_y_mm, color='black', linewidth=2, linestyle='--', alpha=0.7)
    
    # Add Ni particles as circles
    for px, py, size in particles:
        px_mm = px * 10 / nx
        py_mm = py * 7.5 / ny
        size_mm = size * 10 / nx * 0.3
        circle = plt.Circle((px_mm, py_mm), size_mm, fill=False, 
                          edgecolor='black', linewidth=1, alpha=0.6)
        ax2.add_patch(circle)
    
    # Add layer labels
    ax1.text(1, 6, 'Anode\n(Ni-YSZ)', fontsize=11, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax1.text(1, 1, 'Electrolyte\n(8YSZ)', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.text(1, 6, 'Anode\n(Ni-YSZ)', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax2.text(1, 1, 'Electrolyte\n(8YSZ)', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('von Mises Stress, σ_VM (MPa)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add spatial correlation annotation
    fig.text(0.5, 0.08, f'Spatial Correlation = {actual_correlation:.2f}', 
             fontsize=14, fontweight='bold', ha='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Add callouts
    # CTE Mismatch callout
    ax1.annotate('CTE Mismatch', xy=(5, interface_y_mm), xytext=(7, 4),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Ni Cluster Stress Concentration callout
    if particles:
        px, py, _ = particles[15]  # Pick a representative particle
        px_mm = px * 10 / nx
        py_mm = py * 7.5 / ny
        ax2.annotate('Ni Cluster Stress\nConcentration', xy=(px_mm, py_mm), xytext=(7.5, 6),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15, right=0.9)
    
    # Save the figure
    plt.savefig('/workspace/stress_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('/workspace/stress_accuracy_comparison.pdf', bbox_inches='tight')
    
    print(f"\nFigure saved as:")
    print(f"- stress_accuracy_comparison.png (high resolution)")
    print(f"- stress_accuracy_comparison.pdf (vector format)")
    
    # Display statistics
    print(f"\nStress Field Statistics:")
    print(f"HF Simulation - Max: {np.max(hf_stress):.1f} MPa, Mean: {np.mean(hf_stress):.1f} MPa")
    print(f"MF-DL Prediction - Max: {np.max(mf_stress):.1f} MPa, Mean: {np.mean(mf_stress):.1f} MPa")
    print(f"Spatial Correlation: {actual_correlation:.3f}")
    
    # Calculate regions above 80 MPa threshold
    hf_high_stress = np.sum(hf_stress > 80) / hf_stress.size * 100
    mf_high_stress = np.sum(mf_stress > 80) / mf_stress.size * 100
    print(f"Regions >80 MPa - HF: {hf_high_stress:.1f}%, MF-DL: {mf_high_stress:.1f}%")
    
    plt.show()
    
    return fig, actual_correlation

def create_supplementary_analysis():
    """Create supplementary analysis plots"""
    print("\nCreating supplementary analysis...")
    
    # Generate the same data
    nx, ny = 200, 150
    particles = generate_ni_particles(nx, ny, n_particles=28, particle_size_range=(4, 10))
    
    hf_base = create_base_stress_field(nx, ny)
    hf_stress = add_ni_particle_stress_concentrations(hf_base, particles, nx, ny, strength_factor=1.0)
    hf_stress = add_realistic_noise_and_smoothing(hf_stress, noise_level=0.015, smooth_sigma=0.7)
    
    mf_stress = create_mf_dl_prediction(hf_stress, correlation=0.98)
    
    # Create analysis figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Difference plot
    diff = hf_stress - mf_stress
    im1 = ax1.imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10, origin='lower')
    ax1.set_title('Stress Difference (HF - MF-DL)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Stress Difference (MPa)')
    
    # 2. Scatter plot correlation
    ax2.scatter(hf_stress.flatten()[::100], mf_stress.flatten()[::100], alpha=0.6, s=1)
    ax2.plot([0, 100], [0, 100], 'r--', linewidth=2)
    ax2.set_xlabel('HF Simulation (MPa)')
    ax2.set_ylabel('MF-DL Prediction (MPa)')
    ax2.set_title('Point-by-Point Correlation')
    corr = np.corrcoef(hf_stress.flatten(), mf_stress.flatten())[0, 1]
    ax2.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax2.transAxes, fontweight='bold')
    
    # 3. Histogram comparison
    ax3.hist(hf_stress.flatten(), bins=50, alpha=0.7, label='HF Simulation', density=True)
    ax3.hist(mf_stress.flatten(), bins=50, alpha=0.7, label='MF-DL Prediction', density=True)
    ax3.set_xlabel('Stress (MPa)')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Stress Distribution Comparison')
    ax3.legend()
    
    # 4. Line profiles
    y_profile = ny // 2
    x_coords = np.linspace(0, 10, nx)
    ax4.plot(x_coords, hf_stress[y_profile, :], 'b-', linewidth=2, label='HF Simulation')
    ax4.plot(x_coords, mf_stress[y_profile, :], 'r--', linewidth=2, label='MF-DL Prediction')
    ax4.set_xlabel('Distance (mm)')
    ax4.set_ylabel('Stress (MPa)')
    ax4.set_title('Cross-sectional Stress Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/stress_analysis_supplementary.png', dpi=300, bbox_inches='tight')
    print("Supplementary analysis saved as: stress_analysis_supplementary.png")
    
    return fig

if __name__ == "__main__":
    print("Generating Spatial Accuracy Comparison Figure...")
    print("=" * 60)
    
    # Create main figure
    main_fig, correlation = plot_stress_comparison()
    
    # Create supplementary analysis
    supp_fig = create_supplementary_analysis()
    
    print("\n" + "=" * 60)
    print("Figure generation completed successfully!")
    print(f"Final spatial correlation achieved: {correlation:.3f}")
    print("\nFiles generated:")
    print("- stress_accuracy_comparison.png (main figure)")
    print("- stress_accuracy_comparison.pdf (main figure, vector)")
    print("- stress_analysis_supplementary.png (supplementary analysis)")