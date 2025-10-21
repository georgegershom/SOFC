#!/usr/bin/env python3
"""
Enhanced Thermal Boundary Fit Visualization
Creates a more realistic scientific figure showing T(t) with 95% CI + IR Frame & Residual Map
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

# Set up the figure with 16:9 aspect ratio (1920x1080)
fig = plt.figure(figsize=(19.2, 10.8), facecolor='white')
fig.suptitle('Thermal Boundary Fit: T(t) with 95% CI + IR Frame & Residual Map', 
             fontsize=48, fontweight='bold', y=0.95)

# Create grid layout: 70% left for main chart, 30% right for insets
gs = gridspec.GridSpec(2, 2, width_ratios=[7, 3], height_ratios=[1, 1], 
                       hspace=0.3, wspace=0.2, left=0.08, right=0.95, 
                       top=0.88, bottom=0.12)

# Main chart (left 70%)
ax_main = fig.add_subplot(gs[:, 0])

# Generate more realistic thermal data
time = np.linspace(0, 240, 1000)
np.random.seed(42)  # For reproducible results

# Realistic thermal model with multiple phases
# Initial heating phase (0-120 min)
T_initial = 550 + 100 * (1 - np.exp(-time/40))

# XRD snapshot phase (120-165 min) - slight cooling
T_xrd = T_initial + 20 * np.exp(-(time-120)/30) * (time >= 120) * (time < 165)

# Hold phase (165-195 min) - steady state with small fluctuations
T_hold = T_xrd + 30 * (1 - np.exp(-(time-165)/20)) * (time >= 165) * (time < 195)
T_hold += 5 * np.sin((time-165) * np.pi/15) * (time >= 165) * (time < 195)

# Loading phase (195-210 min) - rapid heating
T_load = T_hold + 80 * (1 - np.exp(-(time-195)/8)) * (time >= 195) * (time < 210)

# Post-shock phase (210-240 min) - cooling
T_post = T_load + 40 * np.exp(-(time-210)/25) * (time >= 210)

# Combine all phases
T_mdl = np.where(time < 120, T_initial, 
                np.where(time < 165, T_xrd,
                        np.where(time < 195, T_hold,
                                np.where(time < 210, T_load, T_post))))

# Add realistic noise and smoothing
T_mdl = T_mdl + 2 * np.sin(time/10) + 1 * np.cos(time/7)
T_mdl = np.convolve(T_mdl, np.ones(5)/5, mode='same')  # Smooth

# IR data points (2.5 Hz sampling)
ir_times = np.linspace(0, 240, 600)
T_mdl_ir = np.interp(ir_times, time, T_mdl)

# Add realistic measurement noise (higher during transients)
noise_level = np.where((ir_times >= 195) & (ir_times <= 210), 3.5, 2.0)  # Higher noise during loading
T_IR = T_mdl_ir + np.random.normal(0, noise_level, len(ir_times))

# 95% confidence interval (varies with time)
ci_width = np.where((time >= 195) & (time <= 210), 4.5, 3.0)
ci_upper = T_mdl + ci_width
ci_lower = T_mdl - ci_width

# Plot main chart
ax_main.plot(time, T_mdl, 'b-', linewidth=3, label='Model T_mdl(t)', color='#2563EB')
ax_main.scatter(ir_times, T_IR, s=8, alpha=0.7, color='#DC2626', label='IR T_IR(t)')
ax_main.fill_between(time, ci_lower, ci_upper, alpha=0.4, color='lightblue', label='95% CI')

# Event markers with better positioning
events = [120, 165, 180, 210, 238]
event_labels = ['XRD snapshot', 'Hold start', 'Under load', 'Peak load', 'Post-shock']
colors = ['purple', 'green', 'orange', 'red', 'brown']

# Get temperature values at event times for better label positioning
event_temps = np.interp(events, time, T_mdl)

for i, (t, label, color, temp) in enumerate(zip(events, event_labels, colors, event_temps)):
    ax_main.axvline(x=t, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    ax_main.text(t, temp + 15, label, rotation=90, ha='right', va='bottom', 
                fontsize=18, fontweight='bold', color=color)

# Shaded regions with better styling
ax_main.axvspan(165, 195, alpha=0.1, color='gray', label='Steady/hold')
ax_main.axvspan(200, 210, alpha=0.1, color='red', label='Peak window')

# Callout boxes with better positioning
# RMSE callout
rmse_box = patches.FancyBboxPatch((30, 750), 140, 50, 
                                 boxstyle="round,pad=5", 
                                 facecolor='white', edgecolor='black', 
                                 linewidth=1.5, alpha=0.9)
ax_main.add_patch(rmse_box)
ax_main.text(100, 775, 'RMSE = 2.8 K (hold)\n4.3 K (transient)', 
            ha='center', va='center', fontsize=22, fontweight='bold')

# BCs callout
bc_box = patches.FancyBboxPatch((30, 650), 200, 60, 
                               boxstyle="round,pad=5", 
                               facecolor='white', edgecolor='black', 
                               linewidth=1.5, alpha=0.9)
ax_main.add_patch(bc_box)
ax_main.text(130, 680, 'h = 22 ± 4 W·m⁻²·K⁻¹\nε = 0.92 ± 0.02 (95% CI)', 
            ha='center', va='center', fontsize=22, fontweight='bold')

# Formatting
ax_main.set_xlim(0, 240)
ax_main.set_ylim(550, 900)
ax_main.set_xlabel('Time [min]', fontsize=28, fontweight='bold')
ax_main.set_ylabel('Temperature [°C]', fontsize=28, fontweight='bold')
ax_main.tick_params(axis='both', which='major', labelsize=24)
ax_main.grid(True, alpha=0.3)
ax_main.legend(loc='upper right', fontsize=20, framealpha=0.9)

# Set major ticks every 30 minutes
ax_main.set_xticks(np.arange(0, 241, 30))

# Top inset - IR frame at t* = 180 min
ax_ir = fig.add_subplot(gs[0, 1])

# Generate more realistic IR temperature field
x = np.linspace(0, 20, 50)  # 20mm field
y = np.linspace(0, 15, 40)  # 15mm field
X, Y = np.meshgrid(x, y)

# Create realistic temperature field with spatial gradients
# Base temperature at t=180 min (around 700°C)
T_base = 700

# Add spatial variations typical of thermal experiments
T_IR_field = (T_base + 
              30 * np.exp(-((X-10)**2 + (Y-7.5)**2)/15) +  # Hot spot
              15 * np.sin(X/4) * np.cos(Y/3) +  # Spatial variations
              10 * np.exp(-((X-2)**2 + (Y-2)**2)/8) +  # Corner heating
              8 * np.exp(-((X-18)**2 + (Y-13)**2)/6))  # Another hot spot

# Add some noise to simulate measurement uncertainty
T_IR_field += np.random.normal(0, 1, T_IR_field.shape)

# Plot IR frame
im1 = ax_ir.imshow(T_IR_field, extent=[0, 20, 0, 15], cmap='viridis', aspect='equal')
ax_ir.set_title('IR @ t = 180 min; ε = 0.92', fontsize=22, fontweight='bold', pad=10)

# Add ROI outline
roi_rect = patches.Rectangle((2, 2), 16, 11, linewidth=2, edgecolor='white', 
                           facecolor='none', linestyle='-')
ax_ir.add_patch(roi_rect)

# Add scale bar
ax_ir.plot([2, 12], [1, 1], 'w-', linewidth=3)
ax_ir.text(7, 0.5, '10 mm', ha='center', va='top', color='white', fontsize=18, fontweight='bold')

# Add crosshair at center
ax_ir.axhline(y=7.5, color='white', linestyle='--', alpha=0.7, linewidth=1)
ax_ir.axvline(x=10, color='white', linestyle='--', alpha=0.7, linewidth=1)

# Colorbar for IR
cbar1 = plt.colorbar(im1, ax=ax_ir, shrink=0.8, aspect=20)
cbar1.set_label('Temperature [°C]', fontsize=18, fontweight='bold')
cbar1.ax.tick_params(labelsize=16)

# Bottom inset - Residual map at t* = 180 min
ax_res = fig.add_subplot(gs[1, 1])

# Generate model temperature field (slightly different from IR)
T_mdl_field = (T_base + 
               28 * np.exp(-((X-10)**2 + (Y-7.5)**2)/18) +  # Slightly different hot spot
               12 * np.sin(X/4.2) * np.cos(Y/3.1) +  # Slightly different spatial variations
               8 * np.exp(-((X-2)**2 + (Y-2)**2)/10) +  # Different corner heating
               6 * np.exp(-((X-18)**2 + (Y-13)**2)/8))  # Different hot spot

# Calculate residual
residual = T_mdl_field - T_IR_field

# Create diverging colormap (blue-white-red)
colors_div = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
n_bins = 100
cmap_div = LinearSegmentedColormap.from_list('div', colors_div, N=n_bins)

# Plot residual map
im2 = ax_res.imshow(residual, extent=[0, 20, 0, 15], cmap=cmap_div, 
                   aspect='equal', vmin=-6, vmax=6)

# Add zero contour
contour = ax_res.contour(X, Y, residual, levels=[0], colors='black', linewidths=1.5)

# Add ROI outline
roi_rect2 = patches.Rectangle((2, 2), 16, 11, linewidth=2, edgecolor='black', 
                            facecolor='none', linestyle='-')
ax_res.add_patch(roi_rect2)

# Calculate actual residual statistics
residual_flat = residual.flatten()
mean_res = np.mean(residual_flat)
std_res = np.std(residual_flat)
max_abs_res = np.max(np.abs(residual_flat))

# Add stats box with actual calculated values
stats_text = f'Mean = {mean_res:.1f} K\nStd = {std_res:.1f} K\nMax |ΔT| = {max_abs_res:.1f} K'
ax_res.text(0.02, 0.98, stats_text, transform=ax_res.transAxes, 
           fontsize=16, fontweight='bold', va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

# Colorbar for residual
cbar2 = plt.colorbar(im2, ax=ax_res, shrink=0.8, aspect=20)
cbar2.set_label('ΔT [K]', fontsize=18, fontweight='bold')
cbar2.set_ticks([-6, -3, 0, 3, 6])
cbar2.ax.tick_params(labelsize=16)

# Format residual plot
ax_res.set_title('Residual Map ΔT(x,y)', fontsize=22, fontweight='bold', pad=10)
ax_res.set_xlabel('x [mm]', fontsize=18, fontweight='bold')
ax_res.set_ylabel('y [mm]', fontsize=18, fontweight='bold')

# Set font to Inter/Helvetica
plt.rcParams['font.family'] = ['Inter', 'Helvetica', 'sans-serif']

# Adjust layout and save
plt.tight_layout()
plt.savefig('thermal_boundary_fit_enhanced.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('thermal_boundary_fit_enhanced.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Enhanced thermal boundary fit visualization created successfully!")
print("Files saved: thermal_boundary_fit_enhanced.png and thermal_boundary_fit_enhanced.pdf")
print(f"Residual statistics: Mean = {mean_res:.1f} K, Std = {std_res:.1f} K, Max |ΔT| = {max_abs_res:.1f} K")