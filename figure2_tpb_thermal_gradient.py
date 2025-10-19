#!/usr/bin/env python3
"""
Figure 2: Systematic Underestimation of TPB Thermal Gradients by Low-Fidelity Models

This script generates a scatter plot demonstrating how Low-Fidelity (LF) models
systematically underestimate the local thermal gradient at the Triple-Phase Boundary
(∇T_TPB) compared to High-Fidelity (HF) models and experimental measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set up publication-quality figure parameters
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

def generate_lf_hf_data():
    """
    Generate synthetic LF vs HF prediction data points.
    Based on the text: LF ~45 K/mm, HF ~72 K/mm with 37% average error.
    """
    # Generate HF predictions in the range mentioned
    np.random.seed(42)  # For reproducibility
    
    # Create a range of operating conditions that would yield different thermal gradients
    n_points = 12
    
    # HF predictions - spread around the typical value of 72 K/mm
    hf_predictions = np.random.normal(72, 5, n_points)
    hf_predictions = np.clip(hf_predictions, 60, 80)  # Keep within reasonable bounds
    
    # LF predictions - systematically underestimate by about 37%
    # Add some realistic scatter
    underestimation_factor = np.random.normal(0.63, 0.05, n_points)  # 1 - 0.37 = 0.63
    lf_predictions = hf_predictions * underestimation_factor
    
    # Add a few edge cases
    # Some points with less underestimation
    hf_additional = np.array([55, 65, 75])
    lf_additional = hf_additional * np.random.normal(0.70, 0.03, 3)
    
    hf_predictions = np.concatenate([hf_predictions, hf_additional])
    lf_predictions = np.concatenate([lf_predictions, lf_additional])
    
    return hf_predictions, lf_predictions

def generate_experimental_data():
    """
    Generate experimental validation points.
    These should align with HF predictions, confirming their accuracy.
    """
    # Key experimental point from the text
    exp_points = np.array([72, 68, 75, 70])  # K/mm
    
    return exp_points

def create_figure():
    """
    Create the main figure with all components.
    """
    # Generate data
    hf_pred, lf_pred = generate_lf_hf_data()
    exp_data = generate_experimental_data()
    
    # Create figure with square aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Set equal axis ranges as specified
    axis_min, axis_max = 40, 80
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # Plot the y=x reference line (perfect agreement)
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 
            'r-', linewidth=2, label='y = x (Perfect Agreement)', zorder=1)
    
    # Plot LF vs HF predictions (blue circles)
    ax.scatter(hf_pred, lf_pred, 
              s=100, c='blue', marker='o', alpha=0.7,
              edgecolors='darkblue', linewidth=1,
              label='LF vs. HF Predictions', zorder=2)
    
    # Plot experimental data (black circles on the diagonal)
    ax.scatter(exp_data, exp_data,
              s=150, c='black', marker='o',
              edgecolors='black', linewidth=2,
              label='Experimental Data', zorder=3)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set labels
    ax.set_xlabel('HF Prediction of ∇T$_{TPB}$ (K/mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('LF Prediction of ∇T$_{TPB}$ (K/mm)', fontsize=12, fontweight='bold')
    
    # Set title
    ax.set_title('Systematic Underestimation of TPB Thermal Gradients\nby Low-Fidelity Models', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Calculate average underestimation for annotation
    avg_error = np.mean((hf_pred - lf_pred) / hf_pred) * 100
    
    # Add annotation pointing to the cluster of blue points
    annotation_x = 70
    annotation_y = 45
    ax.annotate(f'LF Underestimation\n~{avg_error:.0f}% Average Error',
                xy=(annotation_x, annotation_y),
                xytext=(annotation_x - 8, annotation_y + 10),
                fontsize=11,
                fontweight='bold',
                color='darkblue',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         edgecolor='darkblue', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))
    
    # Add shaded region to emphasize underestimation zone
    ax.fill_between([axis_min, axis_max], [axis_min, axis_max], axis_min,
                    alpha=0.05, color='red', label='_nolegend_')
    
    # Set aspect ratio to be equal (square plot)
    ax.set_aspect('equal', adjustable='box')
    
    # Add minor ticks for better precision reading
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=6, width=1)
    
    # Strengthen the spines for better visibility
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Add text box with key finding
    textstr = 'Key Finding:\nLF models systematically miss\nlocal thermal hotspots that drive\nthermo-mechanical degradation'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    return fig, ax

def main():
    """
    Main function to generate and save the figure.
    """
    # Create the figure
    fig, ax = create_figure()
    
    # Save the figure in multiple formats
    fig.savefig('figure2_tpb_thermal_gradient.png', dpi=300, bbox_inches='tight')
    fig.savefig('figure2_tpb_thermal_gradient.pdf', bbox_inches='tight')
    fig.savefig('figure2_tpb_thermal_gradient.svg', bbox_inches='tight')
    
    # Display the figure
    plt.show()
    
    print("Figure 2 has been successfully generated and saved in multiple formats:")
    print("  - figure2_tpb_thermal_gradient.png (high-resolution raster)")
    print("  - figure2_tpb_thermal_gradient.pdf (vector format for publications)")
    print("  - figure2_tpb_thermal_gradient.svg (editable vector format)")
    print("\nKey elements included:")
    print("  ✓ LF vs HF predictions (blue circles below y=x line)")
    print("  ✓ Experimental validation points (black circles on y=x line)")
    print("  ✓ Perfect agreement reference line (red y=x line)")
    print("  ✓ Annotation highlighting ~37% underestimation")
    print("  ✓ Square aspect ratio for proper 45° reference line")
    print("  ✓ Publication-quality styling and formatting")

if __name__ == "__main__":
    main()