#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
with hardware-triggered synchronization for DIC, IR, and XRD data.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set up the figure with exact specifications
fig_width = 1920 / 100  # Convert pixels to inches (assuming 100 DPI)
fig_height = 1080 / 100
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Define colors (professional palette)
colors = {
    'text': '#111827',  # Near-black
    'grid': '#E5E7EB',  # Light gray
    'dic': '#2563EB',   # Blue
    'ir': '#DC2626',    # Red
    'xrd': '#7C3AED',   # Violet
    'thermocouple': '#059669'  # Green
}

# Time range and settings
time_start = 0
time_end = 240  # minutes
major_tick_interval = 30
minor_tick_interval = 5

# Create time axis
time_axis = np.linspace(time_start, time_end, int((time_end - time_start) / minor_tick_interval) + 1)
major_ticks = np.arange(time_start, time_end + major_tick_interval, major_tick_interval)

# Set up the plot area
ax.set_xlim(time_start, time_end)
ax.set_ylim(0, 10)  # 10 units for 5 rows + spacing

# Remove default axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add title and subheader
ax.text(0.5, 9.5, 'Synchronized Time Raster — DIC, IR, XRD (Aligned Frames & Continuous Capture)', 
        fontsize=48, fontweight='bold', ha='center', va='center', color=colors['text'],
        transform=ax.transAxes)

ax.text(0.5, 9.0, 'Unified clock; hardware trigger; Δt alignment', 
        fontsize=28, ha='center', va='center', color=colors['text'],
        transform=ax.transAxes)

# Draw horizontal time axis
ax.axhline(y=7.5, xmin=0.05, xmax=0.95, color=colors['text'], linewidth=2)

# Add major tick marks and labels
for i, tick in enumerate(major_ticks):
    if tick <= time_end:
        # Tick mark
        ax.plot([tick, tick], [7.3, 7.7], color=colors['text'], linewidth=2)
        # Label
        ax.text(tick, 7.0, f'{int(tick)}', fontsize=20, ha='center', va='top', color=colors['text'])

# Add minor grid lines
for tick in time_axis[::2]:  # Every other minor tick to avoid clutter
    ax.axvline(x=tick, ymin=0.1, ymax=0.9, color=colors['grid'], linewidth=0.5, alpha=0.7)

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
    ax.axvline(x=time, ymin=0.1, ymax=0.9, color=colors['text'], linewidth=1, linestyle='--', alpha=0.7)
    ax.text(time, 8.5, label, fontsize=20, ha='center', va='bottom', color=colors['text'], rotation=0)

# Define raster rows
row_configs = [
    {'name': 'DIC Cam A (1–5 Hz)', 'y': 6.5, 'color': colors['dic'], 'has_bars': True},
    {'name': 'DIC Cam B (1–5 Hz)', 'y': 5.5, 'color': colors['dic'], 'has_bars': True},
    {'name': 'IR Camera (1–5 Hz)', 'y': 4.5, 'color': colors['ir'], 'has_bars': True},
    {'name': 'XRD (event snapshots)', 'y': 3.5, 'color': colors['xrd'], 'has_bars': False},
    {'name': 'Thermocouples/logs (1–2 Hz)', 'y': 2.5, 'color': colors['thermocouple'], 'has_bars': False}
]

# Add row labels
for config in row_configs:
    ax.text(0.02, config['y'], config['name'], fontsize=26, ha='left', va='center', 
            color=colors['text'], transform=ax.transAxes)

# Generate frame data (simulating 1-5 Hz for DIC/IR, 1-2 Hz for thermocouples)
frame_rates = {'dic': 3, 'ir': 3, 'thermocouple': 1.5}  # Hz
frame_times = {}

for modality in ['dic', 'ir', 'thermocouple']:
    rate = frame_rates[modality]
    times = []
    current_time = 0
    while current_time <= time_end:
        times.append(current_time)
        current_time += 60 / rate  # Convert Hz to minutes
    frame_times[modality] = times

# Add frame ticks for each row
for i, config in enumerate(row_configs):
    if config['name'].startswith('DIC'):
        modality = 'dic'
    elif config['name'].startswith('IR'):
        modality = 'ir'
    elif config['name'].startswith('Thermocouples'):
        modality = 'thermocouple'
    else:
        continue
    
    times = frame_times[modality]
    
    # Add thin vertical ticks for frames
    for time in times:
        if time <= time_end:
            # Add some random dropped frames for DIC B around t=172 and t=204
            if 'DIC Cam B' in config['name'] and (170 <= time <= 175 or 202 <= time <= 206):
                if np.random.random() < 0.3:  # 30% chance to drop frame
                    continue
            
            ax.plot([time, time], [config['y'] - 0.15, config['y'] + 0.15], 
                   color=config['color'], linewidth=2)

# Add continuous capture bars
continuous_windows = [(165, 195), (200, 210)]

for config in row_configs:
    if config['has_bars']:
        for start_time, end_time in continuous_windows:
            # Convert time to data coordinates
            x_start = start_time
            x_end = end_time
            y_bottom = config['y'] - 0.25
            y_top = config['y'] + 0.25
            
            rect = Rectangle((x_start, y_bottom), x_end - x_start, y_top - y_bottom,
                           facecolor=config['color'], alpha=0.7, edgecolor=config['color'])
            ax.add_patch(rect)

# Add XRD snapshot markers
xrd_times = [120, 180, 210, 238]
xrd_config = next(config for config in row_configs if 'XRD' in config['name'])

for time in xrd_times:
    ax.scatter(time, xrd_config['y'], s=100, color=xrd_config['color'], 
              marker='o', zorder=5, edgecolor='white', linewidth=2)

# Add right-side callouts
callout_x = 0.75
callout_y_start = 0.6
callout_spacing = 0.08

callouts = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%",
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, callout_text in enumerate(callouts):
    y_pos = callout_y_start - i * callout_spacing
    
    # Create rounded rectangle background
    bbox = FancyBboxPatch((callout_x, y_pos), 0.2, 0.06, 
                         boxstyle="round,pad=0.01", 
                         facecolor='lightgray', alpha=0.3,
                         edgecolor=colors['text'], linewidth=1,
                         transform=ax.transAxes)
    ax.add_patch(bbox)
    
    ax.text(callout_x + 0.1, y_pos + 0.03, callout_text, 
            fontsize=24, ha='center', va='center', color=colors['text'],
            transform=ax.transAxes)

# Add legend
legend_x = 0.75
legend_y = 0.25

# Legend background
legend_bbox = FancyBboxPatch((legend_x, legend_y), 0.2, 0.15, 
                           boxstyle="round,pad=0.01", 
                           facecolor='white', alpha=0.9,
                           edgecolor=colors['text'], linewidth=1,
                           transform=ax.transAxes)
ax.add_patch(legend_bbox)

# Legend items
legend_items = [
    ("Thin ticks = aligned frames", 0.02, 0.12),
    ("Thick bar = continuous video", 0.02, 0.08),
    ("● = XRD snapshot", 0.02, 0.04)
]

for text, x_offset, y_offset in legend_items:
    ax.text(legend_x + x_offset, legend_y + y_offset, text, 
            fontsize=22, ha='left', va='center', color=colors['text'],
            transform=ax.transAxes)

# Add modality color indicators in legend
ax.plot([legend_x + 0.15, legend_x + 0.18], [legend_y + 0.12, legend_y + 0.12], 
        color=colors['dic'], linewidth=3, transform=ax.transAxes)
ax.plot([legend_x + 0.15, legend_x + 0.18], [legend_y + 0.08, legend_y + 0.08], 
        color=colors['ir'], linewidth=6, transform=ax.transAxes)
ax.scatter(legend_x + 0.165, legend_y + 0.04, s=60, color=colors['xrd'], 
          marker='o', transform=ax.transAxes)

# Final adjustments - fix layout to avoid warnings
plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05)

# Save the figure
plt.savefig('synchronized_time_raster.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Synchronized Time Raster figure created successfully!")
print("Files saved: synchronized_time_raster.png and synchronized_time_raster.pdf")