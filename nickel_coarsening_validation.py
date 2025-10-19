#!/usr/bin/env python3
"""
High-Fidelity Model Validation: Predicting Nickel Coarsening
Bar chart comparing HF model predictions against SEM experimental measurements
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up the figure style
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Data for the bars based on specifications
methods = ['High-Fidelity (HF)\nModel', 'Scanning Electron\nMicroscopy (SEM)']
mean_values = [15.0, 14.9]  # Mean Ni coarsening values in nm
confidence_intervals = [1.5, 2.0]  # 95% CI in nm

# Create the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Define colors for the bars
colors = ['#E74C3C', '#2C3E50']  # Red for HF Model, Dark grey for SEM

# Create the bar chart
x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, mean_values, width=0.6, color=colors, 
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add error bars (95% confidence intervals)
error_bars = ax.errorbar(x_pos, mean_values, yerr=confidence_intervals,
                         fmt='none', ecolor='black', elinewidth=2,
                         capsize=8, capthick=2, label='95% CI')

# Add value labels on top of each bar
for i, (bar, value, ci) in enumerate(zip(bars, mean_values, confidence_intervals)):
    # Position the text slightly above the error bar
    y_position = value + ci + 0.5
    ax.text(bar.get_x() + bar.get_width()/2, y_position,
            f'{value:.1f} ± {ci:.1f} nm',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add the R² annotation
r_squared_text = r'$\mathbf{R^2 = 0.97}$'
ax.text(0.5, 0.85, r_squared_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                  edgecolor='black', linewidth=1.5),
        ha='center', va='center')

# Optional: Add a subtle connection line between bars to emphasize closeness
connection_y = max(mean_values) - 0.5
ax.plot([x_pos[0] + 0.3, x_pos[1] - 0.3], [connection_y, connection_y],
        'k--', alpha=0.3, linewidth=1)

# Customize the axes
ax.set_xlabel('Measurement Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Ni Coarsening, Δd$_{Ni}$ (nm)', fontsize=12, fontweight='bold')
ax.set_title('High-Fidelity Model Validation: Predicting Nickel Coarsening',
             fontsize=14, fontweight='bold', pad=15)

# Set x-axis properties
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)

# Set y-axis properties
ax.set_ylim(0, 20)
ax.set_yticks(np.arange(0, 21, 2))

# Add horizontal grid lines for better readability
ax.grid(axis='y', linestyle='-', alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)

# Create legend for the colors
hf_patch = mpatches.Patch(color=colors[0], label='HF Model')
sem_patch = mpatches.Patch(color=colors[1], label='SEM Experimental')
ax.legend(handles=[hf_patch, sem_patch], loc='upper left', 
          frameon=True, fancybox=True, shadow=False,
          edgecolor='black', framealpha=0.9)

# Add a subtle annotation about validation success
validation_text = "Excellent agreement between\nmodel and experiment"
ax.text(0.98, 0.15, validation_text,
        transform=ax.transAxes,
        fontsize=9,
        style='italic',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                  edgecolor='gray', alpha=0.8, linewidth=0.5))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_filename = 'nickel_coarsening_validation.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Figure saved as: {output_filename}")

# Also save as PDF for publication quality
pdf_filename = 'nickel_coarsening_validation.pdf'
plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
print(f"Figure also saved as: {pdf_filename}")

# Display the figure
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("VALIDATION SUMMARY")
print("="*50)
print(f"High-Fidelity Model: {mean_values[0]:.1f} ± {confidence_intervals[0]:.1f} nm")
print(f"SEM Experimental:    {mean_values[1]:.1f} ± {confidence_intervals[1]:.1f} nm")
print(f"R² Value:            0.97")
print(f"Model Accuracy:      Excellent (within experimental uncertainty)")
print("="*50)