#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
with continuous capture windows and discrete events.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import numpy as np
from datetime import datetime, timedelta

# Set up the figure with exact specifications
fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='white')
fig.patch.set_facecolor('white')

# Canvas dimensions: 1920x1080
plt.xlim(0, 1920)
plt.ylim(0, 1080)
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Color scheme
colors = {
    'text': '#111827',  # near-black
    'grid': '#E5E7EB',  # light gray
    'dic': '#2563EB',   # blue
    'ir': '#DC2626',    # red
    'xrd': '#7C3AED',   # violet
    'thermocouple': '#059669'  # green
}

# Time range: 0-240 minutes
time_range = (0, 240)
time_width = 1200  # pixels for time axis
time_start_x = 200
time_end_x = time_start_x + time_width

# Header section (top 15% of height)
header_y = 900
title_y = 950
subheader_y = 900

# Add title
ax.text(960, title_y, 'Synchronized Time Raster — DIC, IR, XRD (Aligned Frames & Continuous Capture)', 
        fontsize=48, fontweight='bold', ha='center', va='center', color=colors['text'],
        fontfamily='sans-serif')

# Add subheader
ax.text(960, subheader_y, 'Unified clock; hardware trigger; Δt alignment', 
        fontsize=28, ha='center', va='center', color=colors['text'],
        fontfamily='sans-serif')

# Main panel (center 70% height)
panel_top = 800
panel_bottom = 200
panel_height = panel_top - panel_bottom

# Time axis
ax.plot([time_start_x, time_end_x], [panel_top, panel_top], 
        color=colors['text'], linewidth=2)

# Major time ticks (every 30 minutes)
major_ticks = np.arange(0, 241, 30)
for i, minute in enumerate(major_ticks):
    x_pos = time_start_x + (minute / 240) * time_width
    # Tick mark
    ax.plot([x_pos, x_pos], [panel_top, panel_top - 20], 
            color=colors['text'], linewidth=2)
    # Label
    ax.text(x_pos, panel_top - 35, str(minute), 
            fontsize=16, ha='center', va='top', color=colors['text'])

# Minor grid lines
minor_ticks = np.arange(0, 241, 10)
for minute in minor_ticks:
    if minute not in major_ticks:
        x_pos = time_start_x + (minute / 240) * time_width
        ax.plot([x_pos, x_pos], [panel_top, panel_bottom], 
                color=colors['grid'], linewidth=0.5, alpha=0.5)

# Event markers (vertical dashed lines)
events = [
    (120, "XRD snapshot"),
    (150, "Pre-hold"),
    (165, "Hold start"),
    (180, "Under load"),
    (210, "Peak load"),
    (238, "Post-shock")
]

for minute, label in events:
    x_pos = time_start_x + (minute / 240) * time_width
    ax.plot([x_pos, x_pos], [panel_top, panel_bottom], 
            color=colors['text'], linewidth=1, linestyle='--', alpha=0.7)
    ax.text(x_pos, panel_top + 15, label, 
            fontsize=20, ha='center', va='bottom', color=colors['text'],
            rotation=0)

# Raster rows
row_height = 80
row_spacing = 20
rows = [
    ("DIC Cam A (1–5 Hz)", colors['dic']),
    ("DIC Cam B (1–5 Hz)", colors['dic']),
    ("IR Camera (1–5 Hz)", colors['ir']),
    ("XRD (event snapshots)", colors['xrd']),
    ("Thermocouples/logs (1–2 Hz)", colors['thermocouple'])
]

row_centers = []
for i, (label, color) in enumerate(rows):
    y_center = panel_top - 50 - i * (row_height + row_spacing)
    row_centers.append(y_center)
    
    # Row label
    ax.text(50, y_center, label, 
            fontsize=26, ha='left', va='center', color=colors['text'],
            fontweight='bold')
    
    # Row line
    ax.plot([time_start_x, time_end_x], [y_center, y_center], 
            color=colors['text'], linewidth=1, alpha=0.3)

# Add frame ticks for DIC and IR rows (first 3 rows)
frame_rate = 3  # Hz average
frame_interval = 1.0 / frame_rate  # seconds between frames
frame_interval_min = frame_interval / 60  # minutes between frames

for row_idx in range(3):  # DIC A, DIC B, IR
    y_center = row_centers[row_idx]
    # Add frame ticks every frame interval
    frame_times = np.arange(0, 240, frame_interval_min)
    
    for minute in frame_times:
        if minute <= 240:
            x_pos = time_start_x + (minute / 240) * time_width
            # Thin vertical tick
            ax.plot([x_pos, x_pos], [y_center - 3, y_center + 3], 
                    color=colors['text'], linewidth=1)
    
    # Add some dropped frames for DIC B (around t≈172 and t≈204)
    if row_idx == 1:  # DIC B
        # Remove a few ticks around 172 and 204 minutes
        drop_windows = [(170, 175), (202, 207)]
        for start_min, end_min in drop_windows:
            for minute in np.arange(start_min, end_min, frame_interval_min):
                if minute <= 240:
                    x_pos = time_start_x + (minute / 240) * time_width
                    # Draw a small gap instead of tick
                    ax.plot([x_pos-1, x_pos+1], [y_center, y_center], 
                            color='white', linewidth=3)

# Add continuous capture bars for DIC and IR rows
continuous_windows = [(165, 195), (200, 210)]

for row_idx in range(3):  # DIC A, DIC B, IR
    y_center = row_centers[row_idx]
    color = colors['dic'] if row_idx < 2 else colors['ir']
    
    for start_min, end_min in continuous_windows:
        start_x = time_start_x + (start_min / 240) * time_width
        end_x = time_start_x + (end_min / 240) * time_width
        width = end_x - start_x
        height = 8
        
        rect = Rectangle((start_x, y_center - height/2), width, height,
                        facecolor=color, alpha=0.7, edgecolor=color, linewidth=1)
        ax.add_patch(rect)

# Add XRD snapshots (filled circles)
xrd_row_idx = 3
y_center = row_centers[xrd_row_idx]
xrd_times = [120, 180, 210, 238]

for minute in xrd_times:
    x_pos = time_start_x + (minute / 240) * time_width
    circle = Circle((x_pos, y_center), 4, 
                   facecolor=colors['xrd'], edgecolor=colors['xrd'], linewidth=1)
    ax.add_patch(circle)

# Add thermocouple ticks (simpler, lower frequency)
thermocouple_row_idx = 4
y_center = row_centers[thermocouple_row_idx]
thermocouple_interval = 1.0 / 1.5  # 1.5 Hz
thermocouple_interval_min = thermocouple_interval / 60

thermocouple_times = np.arange(0, 240, thermocouple_interval_min)
for minute in thermocouple_times:
    if minute <= 240:
        x_pos = time_start_x + (minute / 240) * time_width
        ax.plot([x_pos, x_pos], [y_center - 2, y_center + 2], 
                color=colors['thermocouple'], linewidth=1)

# Right-side metric callouts
callout_x = 1500
callout_y_start = 700
callout_spacing = 60

callouts = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%",
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, callout_text in enumerate(callouts):
    y_pos = callout_y_start - i * callout_spacing
    
    # Rounded rectangle background
    rect = Rectangle((callout_x - 10, y_pos - 20), 200, 40,
                    facecolor='white', edgecolor=colors['text'], 
                    linewidth=1, alpha=0.9)
    ax.add_patch(rect)
    
    ax.text(callout_x + 90, y_pos, callout_text, 
            fontsize=24, ha='center', va='center', color=colors['text'])

# Legend (bottom-right)
legend_x = 1600
legend_y = 150

legend_items = [
    ("Thin ticks = aligned frames", 0),
    ("Thick bar = continuous video segment", 1),
    ("● = XRD snapshot", 2)
]

for i, (text, item_type) in enumerate(legend_items):
    y_pos = legend_y - i * 30
    
    if item_type == 0:  # Thin tick
        ax.plot([legend_x, legend_x + 20], [y_pos, y_pos], 
                color=colors['text'], linewidth=2)
    elif item_type == 1:  # Thick bar
        rect = Rectangle((legend_x, y_pos - 4), 20, 8,
                        facecolor=colors['dic'], alpha=0.7, edgecolor=colors['dic'])
        ax.add_patch(rect)
    elif item_type == 2:  # XRD circle
        circle = Circle((legend_x + 10, y_pos), 3, 
                       facecolor=colors['xrd'], edgecolor=colors['xrd'])
        ax.add_patch(circle)
    
    ax.text(legend_x + 30, y_pos, text, 
            fontsize=22, ha='left', va='center', color=colors['text'])

# Add accessibility note (hidden, for reference)
accessibility_text = ("Horizontal timeline from 0 to 240 minutes with five stacked rows for DIC A, DIC B, IR, XRD, and thermocouples. "
                     "Thin ticks align across rows at the same times. Thick bars on DIC and IR indicate continuous recording "
                     "from 165–195 and 200–210 minutes. XRD shows discrete dots at 120, 180, 210, and 238 minutes. "
                     "Vertical dashed lines label key events. Side notes report ≤30 ms desynchronization and <1% dropped frames.")

# Save the figure
plt.tight_layout()
plt.savefig('synchronized_time_raster.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Figure saved as 'synchronized_time_raster.png' and 'synchronized_time_raster.pdf'")
print(f"Accessibility alt-text: {accessibility_text}")

# Display the figure
plt.show()