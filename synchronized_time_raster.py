#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
with hardware-triggered synchronization for DIC, IR, and XRD modalities.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from datetime import datetime, timedelta

# Set up the figure with exact specifications
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove all spines and ticks initially
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Define colors (professional palette)
colors = {
    'text': '#111827',
    'grid': '#E5E7EB',
    'dic': '#2563EB',
    'ir': '#DC2626',
    'xrd': '#7C3AED',
    'thermocouple': '#059669'
}

# Time range and settings
time_start = 0
time_end = 240  # minutes
major_tick_interval = 30
minor_tick_interval = 5

# Event markers (time in minutes)
events = {
    120: "XRD snapshot",
    150: "Pre-hold", 
    165: "Hold start",
    180: "Under load",
    210: "Peak load",
    238: "Post-shock"
}

# Continuous capture windows
continuous_windows = [
    (165, 195),  # steady/hold
    (200, 210)   # peak-load window
]

# XRD snapshot times
xrd_snapshots = [120, 180, 210, 238]

# Frame rates and dropped frame simulation
frame_rates = {
    'DIC Cam A': 3,  # Hz (1-5 Hz range)
    'DIC Cam B': 3,  # Hz (1-5 Hz range) 
    'IR Camera': 3,     # Hz (1-5 Hz range)
    'XRD': 0,    # event snapshots only
    'Thermocouples/logs': 1.5  # Hz (1-2 Hz range)
}

# Calculate positions
header_height = 0.15
main_panel_height = 0.7
legend_height = 0.15

# Main panel area (70% of height)
main_panel_bottom = 0.15
main_panel_top = 0.85

# Raster rows (5 rows total)
row_height = (main_panel_top - main_panel_bottom) / 5
row_labels = ['DIC Cam A (1–5 Hz)', 'DIC Cam B (1–5 Hz)', 'IR Camera (1–5 Hz)', 
              'XRD (event snapshots)', 'Thermocouples/logs (1–2 Hz)']

# Time axis position (bottom of main panel)
time_axis_y = main_panel_bottom - 0.05

# Add title and subheader
ax.text(0.5, 0.95, 'Synchronized Time Raster — DIC, IR, XRD (Aligned Frames & Continuous Capture)', 
        fontsize=48, fontweight='bold', ha='center', va='top', color=colors['text'],
        transform=ax.transAxes, fontfamily='sans-serif')

ax.text(0.5, 0.88, 'Unified clock; hardware trigger; Δt alignment', 
        fontsize=28, ha='center', va='top', color=colors['text'],
        transform=ax.transAxes, fontfamily='sans-serif')

# Create time axis
time_axis_left = 0.1
time_axis_right = 0.9
time_axis_width = time_axis_right - time_axis_left

# Draw time axis line
ax.plot([time_axis_left, time_axis_right], [time_axis_y, time_axis_y], 
        color=colors['text'], linewidth=2, transform=ax.transAxes)

# Add major time ticks and labels
for i in range(0, int(time_end) + 1, major_tick_interval):
    x_pos = time_axis_left + (i / time_end) * time_axis_width
    ax.plot([x_pos, x_pos], [time_axis_y - 0.01, time_axis_y + 0.01], 
            color=colors['text'], linewidth=2, transform=ax.transAxes)
    ax.text(x_pos, time_axis_y - 0.03, str(i), fontsize=16, ha='center', va='top',
            color=colors['text'], transform=ax.transAxes)

# Add minor grid lines
for i in range(0, int(time_end) + 1, minor_tick_interval):
    x_pos = time_axis_left + (i / time_end) * time_axis_width
    ax.plot([x_pos, x_pos], [main_panel_bottom, main_panel_top], 
            color=colors['grid'], linewidth=0.5, alpha=0.5, transform=ax.transAxes)

# Add event markers (vertical dashed lines)
for time, label in events.items():
    x_pos = time_axis_left + (time / time_end) * time_axis_width
    ax.plot([x_pos, x_pos], [main_panel_bottom, main_panel_top], 
            color=colors['text'], linestyle='--', linewidth=1.5, alpha=0.7, transform=ax.transAxes)
    ax.text(x_pos, main_panel_top + 0.01, label, fontsize=20, ha='center', va='bottom',
            color=colors['text'], transform=ax.transAxes, rotation=0)

# Create raster rows
for i, (label, color_key) in enumerate(zip(row_labels, ['dic', 'dic', 'ir', 'xrd', 'thermocouple'])):
    row_center = main_panel_bottom + (4 - i) * row_height + row_height/2
    
    # Add row label
    ax.text(0.02, row_center, label, fontsize=26, ha='left', va='center',
            color=colors[color_key], transform=ax.transAxes, fontweight='bold')
    
    # Add frame ticks for DIC, IR, and thermocouples
    if color_key in ['dic', 'ir', 'thermocouple']:
        # Extract the key for frame rates
        if 'DIC Cam A' in label:
            frame_rate = frame_rates['DIC Cam A']
        elif 'DIC Cam B' in label:
            frame_rate = frame_rates['DIC Cam B']
        elif 'IR Camera' in label:
            frame_rate = frame_rates['IR Camera']
        elif 'Thermocouples' in label:
            frame_rate = frame_rates['Thermocouples/logs']
        else:
            frame_rate = 0
            
        if frame_rate > 0:
            frame_interval = 60 / frame_rate  # seconds between frames
            frame_times = np.arange(0, time_end * 60, frame_interval) / 60  # convert to minutes
            
            # Simulate dropped frames for DIC B (around t=172 and t=204)
            if 'DIC Cam B' in label:
                frame_times = frame_times[~((frame_times > 170) & (frame_times < 174))]
                frame_times = frame_times[~((frame_times > 202) & (frame_times < 206))]
            
            for frame_time in frame_times:
                if 0 <= frame_time <= time_end:
                    x_pos = time_axis_left + (frame_time / time_end) * time_axis_width
                    ax.plot([x_pos, x_pos], [row_center - 0.008, row_center + 0.008], 
                           color=colors[color_key], linewidth=2, transform=ax.transAxes)
    
    # Add continuous capture bars for DIC and IR
    if color_key in ['dic', 'ir']:
        for start_time, end_time in continuous_windows:
            start_x = time_axis_left + (start_time / time_end) * time_axis_width
            end_x = time_axis_left + (end_time / time_end) * time_axis_width
            bar_height = 0.012
            
            rect = FancyBboxPatch((start_x, row_center - bar_height/2), 
                                 end_x - start_x, bar_height,
                                 boxstyle="round,pad=0.002",
                                 facecolor=colors[color_key], 
                                 edgecolor=colors[color_key],
                                 alpha=0.8, transform=ax.transAxes)
            ax.add_patch(rect)
    
    # Add XRD snapshots
    if color_key == 'xrd':
        for snapshot_time in xrd_snapshots:
            x_pos = time_axis_left + (snapshot_time / time_end) * time_axis_width
            circle = plt.Circle((x_pos, row_center), 0.008, 
                              color=colors[color_key], transform=ax.transAxes)
            ax.add_patch(circle)

# Add right-side metric callouts
callout_x = 0.92
callout_y_start = 0.75
callout_spacing = 0.08

metrics = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%", 
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, metric in enumerate(metrics):
    y_pos = callout_y_start - i * callout_spacing
    
    # Create rounded box
    box = FancyBboxPatch((callout_x - 0.08, y_pos - 0.03), 0.16, 0.06,
                        boxstyle="round,pad=0.01",
                        facecolor='white',
                        edgecolor=colors['text'],
                        linewidth=1.5,
                        transform=ax.transAxes)
    ax.add_patch(box)
    
    ax.text(callout_x, y_pos, metric, fontsize=24, ha='center', va='center',
            color=colors['text'], transform=ax.transAxes, fontweight='bold')

# Add legend (bottom-right)
legend_x = 0.75
legend_y = 0.12

legend_items = [
    ("Thin ticks", "aligned frames", colors['text']),
    ("Thick bar", "continuous video segment", colors['dic']),
    ("●", "XRD snapshot", colors['xrd'])
]

for i, (symbol, description, color) in enumerate(legend_items):
    y_pos = legend_y + i * 0.04
    
    if symbol == "Thin ticks":
        ax.plot([legend_x, legend_x + 0.02], [y_pos, y_pos], 
               color=color, linewidth=2, transform=ax.transAxes)
    elif symbol == "Thick bar":
        rect = FancyBboxPatch((legend_x, y_pos - 0.005), 0.02, 0.01,
                             boxstyle="round,pad=0.001",
                             facecolor=color, edgecolor=color,
                             transform=ax.transAxes)
        ax.add_patch(rect)
    elif symbol == "●":
        circle = plt.Circle((legend_x + 0.01, y_pos), 0.004, 
                          color=color, transform=ax.transAxes)
        ax.add_patch(circle)
    
    ax.text(legend_x + 0.03, y_pos, f"{symbol} = {description}", 
            fontsize=22, ha='left', va='center',
            color=colors['text'], transform=ax.transAxes)

# Set axis limits and remove all ticks
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Ensure clean output
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

# Save the figure
plt.savefig('synchronized_time_raster.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Synchronized Time Raster figure created successfully!")
print("Files saved: synchronized_time_raster.png, synchronized_time_raster.pdf")
print("\nFigure specifications:")
print(f"- Dimensions: 1920×1080 pixels")
print(f"- Time range: 0-240 minutes")
print(f"- Event markers: {len(events)} events")
print(f"- Raster rows: {len(row_labels)} sensing modalities")
print(f"- Continuous capture windows: {len(continuous_windows)}")
print(f"- XRD snapshots: {len(xrd_snapshots)}")