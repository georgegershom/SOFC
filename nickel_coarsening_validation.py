import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Data for the validation chart
# High-Fidelity Model data
hf_mean = 15.1  # Mean Ni coarsening prediction (nm)
hf_error = 1.5  # 95% confidence interval (nm)

# SEM Experimental data
sem_mean = 14.9  # Mean experimental measurement (nm)
sem_error = 2.0  # 95% confidence interval (nm)

# Bar positions
x_positions = [0, 1]
bar_width = 0.6

# Create the bars
bars = ax.bar(x_positions, [hf_mean, sem_mean], width=bar_width, 
              color=['#d62728', '#2c2c2c'],  # Red for HF, Black for SEM
              alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars
ax.errorbar(x_positions, [hf_mean, sem_mean], 
            yerr=[hf_error, sem_error], 
            fmt='none', color='black', capsize=8, capthick=2, 
            elinewidth=2, markersize=8)

# Customize the plot
ax.set_xlabel('Measurement Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Ni Coarsening, Δd_Ni (nm)', fontsize=14, fontweight='bold')
ax.set_title('High-Fidelity Model Validation: Predicting Nickel Coarsening', 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(x_positions)
ax.set_xticklabels(['High-Fidelity (HF) Model', 'Scanning Electron Microscopy (SEM)'], 
                   fontsize=12, fontweight='bold')

# Set y-axis range and add grid
ax.set_ylim(0, 20)
ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Add data value labels on top of bars
ax.text(0, hf_mean + hf_error + 0.5, f'{hf_mean:.1f} nm', 
        ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.text(1, sem_mean + sem_error + 0.5, f'{sem_mean:.1f} nm', 
        ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add R² annotation with a fancy box
r2_text = 'R² = 0.97'
r2_box = FancyBboxPatch((0.25, 16.5), 0.5, 2.5, 
                        boxstyle="round,pad=0.1", 
                        facecolor='lightblue', 
                        edgecolor='navy', 
                        linewidth=2,
                        alpha=0.8)
ax.add_patch(r2_box)
ax.text(0.5, 17.75, r2_text, ha='center', va='center', 
        fontsize=14, fontweight='bold', color='navy')

# Add connection line between bars to emphasize closeness
ax.plot([0.3, 0.7], [max(hf_mean + hf_error, sem_mean + sem_error) + 1, 
                     max(hf_mean + hf_error, sem_mean + sem_error) + 1], 
        'k--', alpha=0.6, linewidth=1)

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor='#d62728', alpha=0.8, label='HF Model'),
                   plt.Rectangle((0,0),1,1, facecolor='#2c2c2c', alpha=0.8, label='SEM Experiment')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

# Add statistical significance annotation
ax.annotate('Validation R² = 0.97\nHighly Accurate Model', 
            xy=(0.5, 12), xytext=(0.5, 8),
            ha='center', va='center',
            fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

# Customize tick parameters
ax.tick_params(axis='both', which='major', labelsize=11)
ax.tick_params(axis='y', which='major', length=6, width=1)

# Add minor ticks for better readability
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
ax.tick_params(axis='y', which='minor', length=3, width=0.5)

# Tight layout and save
plt.tight_layout()
plt.savefig('/workspace/nickel_coarsening_validation.png', dpi=300, bbox_inches='tight')
plt.savefig('/workspace/nickel_coarsening_validation.pdf', bbox_inches='tight')
plt.show()

# Print summary statistics
print("High-Fidelity Model Validation Summary:")
print("="*50)
print(f"HF Model Prediction: {hf_mean:.1f} ± {hf_error:.1f} nm")
print(f"SEM Experimental:    {sem_mean:.1f} ± {sem_error:.1f} nm")
print(f"Validation R²:       {0.97:.2f}")
print(f"Difference:          {abs(hf_mean - sem_mean):.1f} nm ({abs(hf_mean - sem_mean)/sem_mean*100:.1f}%)")
print("="*50)