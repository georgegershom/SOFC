#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
with frame synchronization and continuous capture windows.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set up the figure with exact specifications
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)  # 1920x1080 at 100 DPI
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove all default spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Set exact figure bounds
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Color scheme
colors = {
    'text': '#111827',
    'grid': '#E5E7EB',
    'dic': '#2563EB',
    'ir': '#DC2626',
    'xrd': '#7C3AED',
    'background': 'white'
}

# Layout parameters
title_y = 0.95
subtitle_y = 0.90
main_panel_top = 0.85
main_panel_bottom = 0.15
timeline_y = 0.80
raster_start_y = 0.75
raster_height = 0.50
legend_y = 0.08

# Time parameters
time_start = 0
time_end = 240  # minutes
time_range = time_end - time_start

def time_to_x(time_min):
    """Convert time in minutes to x coordinate (0-1)"""
    return 0.1 + (time_min / time_range) * 0.8

def add_text(x, y, text, fontsize, weight='normal', color=colors['text'], ha='left', va='center'):
    """Add text with consistent styling"""
    ax.text(x, y, text, fontsize=fontsize, weight=weight, color=color, 
            ha=ha, va=va, fontfamily='sans-serif')

# Title and subtitle
add_text(0.5, title_y, "Synchronized Time Raster — DIC, IR, XRD", 48, 'bold', ha='center')
add_text(0.5, subtitle_y, "(Aligned Frames & Continuous Capture)", 28, 'normal', ha='center')
add_text(0.5, subtitle_y - 0.03, "Unified clock; hardware trigger; Δt alignment", 20, 'normal', ha='center')

# Main timeline
timeline_x_start = time_to_x(0)
timeline_x_end = time_to_x(240)
ax.plot([timeline_x_start, timeline_x_end], [timeline_y, timeline_y], 
        color=colors['text'], linewidth=2)

# Major time ticks and labels
for i in range(0, 241, 30):
    x = time_to_x(i)
    ax.plot([x, x], [timeline_y - 0.01, timeline_y + 0.01], color=colors['text'], linewidth=2)
    add_text(x, timeline_y - 0.03, str(i), 16, ha='center')

# Minor grid lines
for i in range(0, 241, 5):
    x = time_to_x(i)
    ax.plot([x, x], [timeline_y - 0.005, timeline_y + 0.005], color=colors['grid'], linewidth=1)

# Event markers
events = [
    (120, "XRD snapshot"),
    (150, "Pre-hold"),
    (165, "Hold start"),
    (180, "Under load"),
    (210, "Peak load"),
    (238, "Post-shock")
]

for time, label in events:
    x = time_to_x(time)
    ax.plot([x, x], [timeline_y + 0.02, raster_start_y], color=colors['text'], 
            linestyle='--', linewidth=1.5, alpha=0.7)
    add_text(x, timeline_y + 0.04, label, 18, ha='center')

# Raster rows
raster_labels = [
    "DIC Cam A (1–5 Hz)",
    "DIC Cam B (1–5 Hz)", 
    "IR Camera (1–5 Hz)",
    "XRD (event snapshots)",
    "Thermocouples/logs (1–2 Hz)"
]

row_height = raster_height / len(raster_labels)
row_colors = [colors['dic'], colors['dic'], colors['ir'], colors['xrd'], colors['text']]

for i, (label, color) in enumerate(zip(raster_labels, row_colors)):
    y = raster_start_y - i * row_height - row_height/2
    
    # Row label
    add_text(0.05, y, label, 26, color=color)
    
    # Row line
    ax.plot([timeline_x_start, timeline_x_end], [y, y], color=colors['grid'], linewidth=1, alpha=0.5)
    
    # Frame ticks for DIC and IR rows (every 12 seconds = 0.2 min for 5 Hz)
    if i < 3:  # DIC A, DIC B, IR
        frame_interval = 0.2  # minutes
        for t in np.arange(0, 241, frame_interval):
            x = time_to_x(t)
            # Simulate occasional dropped frames
            if i == 1 and (171.8 <= t <= 172.2 or 203.8 <= t <= 204.2):
                continue
            ax.plot([x, x], [y - 0.008, y + 0.008], color=color, linewidth=1.5)
    
    # XRD snapshots
    elif i == 3:  # XRD row
        xrd_times = [120, 180, 210, 238]
        for t in xrd_times:
            x = time_to_x(t)
            circle = plt.Circle((x, y), 0.008, color=color, zorder=5)
            ax.add_patch(circle)

# Continuous capture bars for DIC and IR
continuous_windows = [(165, 195), (200, 210)]
for i in range(3):  # DIC A, DIC B, IR
    y = raster_start_y - i * row_height - row_height/2
    bar_height = 0.015
    color = row_colors[i]
    
    for start_time, end_time in continuous_windows:
        x_start = time_to_x(start_time)
        x_end = time_to_x(end_time)
        width = x_end - x_start
        
        rect = Rectangle((x_start, y - bar_height/2), width, bar_height, 
                        facecolor=color, alpha=0.6, edgecolor=color, linewidth=1)
        ax.add_patch(rect)

# Right-side callout boxes
callout_x = 0.75
callout_y_start = 0.65
callout_spacing = 0.08

callouts = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%",
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, text in enumerate(callouts):
    y = callout_y_start - i * callout_spacing
    
    # Create rounded rectangle background
    box = FancyBboxPatch((callout_x - 0.08, y - 0.025), 0.16, 0.05,
                        boxstyle="round,pad=0.01", 
                        facecolor='white', edgecolor=colors['text'], 
                        linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    
    add_text(callout_x, y, text, 20, ha='center', weight='normal')

# Legend
legend_x = 0.75
legend_y_pos = 0.20

# Legend background
legend_box = FancyBboxPatch((legend_x - 0.12, legend_y_pos - 0.08), 0.24, 0.15,
                           boxstyle="round,pad=0.01",
                           facecolor='white', edgecolor=colors['text'],
                           linewidth=1.5, alpha=0.9)
ax.add_patch(legend_box)

# Legend items
legend_items = [
    ("Thin ticks = aligned frames", 0.02),
    ("Thick bar = continuous video segment", 0.06),
    ("● = XRD snapshot", 0.10)
]

for text, y_offset in legend_items:
    add_text(legend_x - 0.10, legend_y_pos - y_offset, text, 18, ha='left')

# Add visual legend elements
# Thin tick example
legend_tick_x = legend_x - 0.11
legend_tick_y = legend_y_pos - 0.02
ax.plot([legend_tick_x, legend_tick_x], [legend_tick_y - 0.005, legend_tick_y + 0.005], 
        color=colors['text'], linewidth=1.5)

# Thick bar example
bar_x_start = legend_x - 0.11
bar_x_end = legend_x - 0.08
bar_y = legend_y_pos - 0.06
rect = Rectangle((bar_x_start, bar_y - 0.003), bar_x_end - bar_x_start, 0.006,
                facecolor=colors['dic'], alpha=0.6, edgecolor=colors['dic'])
ax.add_patch(rect)

# XRD dot example
xrd_dot_x = legend_x - 0.11
xrd_dot_y = legend_y_pos - 0.10
circle = plt.Circle((xrd_dot_x, xrd_dot_y), 0.003, color=colors['xrd'], zorder=5)
ax.add_patch(circle)

# Final adjustments
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Save the figure
plt.tight_layout()
plt.savefig('synchronized_time_raster.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Figure saved as 'synchronized_time_raster.png' and 'synchronized_time_raster.pdf'")
print("Dimensions: 1920×1080 pixels")
print("Features:")
print("- Time-aligned sensing streams with frame synchronization")
print("- Continuous capture windows highlighted")
print("- Event markers for key test phases")
print("- Professional styling with modality-specific colors")
print("- Accessibility-compliant design")