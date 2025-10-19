import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

# Set up the figure with better styling
plt.style.use('default')
fig, ax = plt.subplots(figsize=(12, 10))

# Data for the validation chart
# High-Fidelity Model data
hf_mean = 15.1  # Mean Ni coarsening prediction (nm)
hf_error = 1.5  # 95% confidence interval (nm)
hf_rmse = 2.0   # Root Mean Square Error

# SEM Experimental data
sem_mean = 14.9  # Mean experimental measurement (nm)
sem_error = 2.0  # 95% confidence interval (nm)

# Bar positions
x_positions = [0, 1]
bar_width = 0.6

# Create the bars with enhanced styling
bars = ax.bar(x_positions, [hf_mean, sem_mean], width=bar_width, 
              color=['#d62728', '#2c2c2c'],  # Red for HF, Black for SEM
              alpha=0.85, edgecolor='black', linewidth=2,
              capsize=0)

# Add error bars with enhanced styling
error_bars = ax.errorbar(x_positions, [hf_mean, sem_mean], 
                        yerr=[hf_error, sem_error], 
                        fmt='none', color='black', capsize=12, capthick=3, 
                        elinewidth=3, markersize=0)

# Customize the plot
ax.set_xlabel('Measurement Method', fontsize=16, fontweight='bold', labelpad=15)
ax.set_ylabel('Ni Coarsening, Δd_Ni (nm)', fontsize=16, fontweight='bold', labelpad=15)
ax.set_title('High-Fidelity Model Validation: Predicting Nickel Coarsening\nQuantitative Comparison with Experimental SEM Data', 
             fontsize=18, fontweight='bold', pad=25)

# Set x-axis labels with better formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(['High-Fidelity (HF) Model', 'Scanning Electron Microscopy (SEM)'], 
                   fontsize=14, fontweight='bold')

# Set y-axis range and add grid
ax.set_ylim(0, 22)
ax.grid(True, axis='y', alpha=0.4, linestyle='-', linewidth=0.8)
ax.set_axisbelow(True)

# Add data value labels on top of bars with enhanced formatting
ax.text(0, hf_mean + hf_error + 0.8, f'{hf_mean:.1f} nm', 
        ha='center', va='bottom', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='red'))
ax.text(1, sem_mean + sem_error + 0.8, f'{sem_mean:.1f} nm', 
        ha='center', va='bottom', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black'))

# Add error bar labels
ax.text(0, hf_mean - hf_error - 1.2, f'±{hf_error:.1f} nm', 
        ha='center', va='top', fontsize=11, style='italic', color='red')
ax.text(1, sem_mean - sem_error - 1.2, f'±{sem_error:.1f} nm', 
        ha='center', va='top', fontsize=11, style='italic', color='black')

# Add R² annotation with enhanced styling
r2_text = 'R² = 0.97\nValidation R²'
r2_box = FancyBboxPatch((0.15, 18), 0.7, 3.5, 
                        boxstyle="round,pad=0.15", 
                        facecolor='lightblue', 
                        edgecolor='navy', 
                        linewidth=3,
                        alpha=0.9)
ax.add_patch(r2_box)
ax.text(0.5, 19.75, r2_text, ha='center', va='center', 
        fontsize=16, fontweight='bold', color='navy')

# Add connection line between bars to emphasize closeness
connection_y = max(hf_mean + hf_error, sem_mean + sem_error) + 2.5
ax.plot([0.3, 0.7], [connection_y, connection_y], 
        'k--', alpha=0.7, linewidth=2)
ax.text(0.5, connection_y + 0.3, 'Excellent Agreement', 
        ha='center', va='bottom', fontsize=11, style='italic', fontweight='bold')

# Add statistical summary box
stats_text = f'Statistical Summary:\n• Difference: {abs(hf_mean - sem_mean):.1f} nm ({abs(hf_mean - sem_mean)/sem_mean*100:.1f}%)\n• HF RMSE: {hf_rmse:.1f} nm\n• Model Accuracy: Excellent'
stats_box = FancyBboxPatch((0.02, 0.02), 0.96, 0.15, 
                          boxstyle="round,pad=0.02", 
                          facecolor='lightgreen', 
                          edgecolor='darkgreen', 
                          linewidth=2,
                          alpha=0.8,
                          transform=ax.transAxes)
ax.add_patch(stats_box)
ax.text(0.5, 0.095, stats_text, ha='center', va='center', 
        fontsize=11, fontweight='bold', color='darkgreen',
        transform=ax.transAxes)

# Enhanced legend
legend_elements = [Rectangle((0,0),1,1, facecolor='#d62728', alpha=0.85, label='High-Fidelity Model'),
                   Rectangle((0,0),1,1, facecolor='#2c2c2c', alpha=0.85, label='SEM Experiment')]
legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                  framealpha=0.95, fancybox=True, shadow=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1)

# Customize tick parameters
ax.tick_params(axis='both', which='major', labelsize=12, length=8, width=1.5)
ax.tick_params(axis='y', which='major', length=8, width=1.5)

# Add minor ticks for better readability
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
ax.tick_params(axis='y', which='minor', length=4, width=0.8)

# Add horizontal reference lines
ax.axhline(y=15, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(y=5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Tight layout and save
plt.tight_layout()
plt.savefig('/workspace/enhanced_nickel_coarsening_validation.png', dpi=300, bbox_inches='tight')
plt.savefig('/workspace/enhanced_nickel_coarsening_validation.pdf', bbox_inches='tight')
plt.show()

# Print enhanced summary statistics
print("="*60)
print("HIGH-FIDELITY MODEL VALIDATION SUMMARY")
print("="*60)
print(f"HF Model Prediction:     {hf_mean:.1f} ± {hf_error:.1f} nm (95% CI)")
print(f"SEM Experimental:        {sem_mean:.1f} ± {sem_error:.1f} nm (95% CI)")
print(f"Validation R²:           {0.97:.2f}")
print(f"HF Model RMSE:           {hf_rmse:.1f} nm")
print(f"Absolute Difference:     {abs(hf_mean - sem_mean):.1f} nm")
print(f"Relative Difference:     {abs(hf_mean - sem_mean)/sem_mean*100:.1f}%")
print(f"Model Accuracy:          Excellent")
print("="*60)
print("CONCLUSION: The High-Fidelity model demonstrates exceptional")
print("accuracy in predicting Nickel coarsening, with predictions")
print("within experimental uncertainty bounds.")
print("="*60)