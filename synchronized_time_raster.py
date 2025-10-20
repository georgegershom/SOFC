#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from datetime import datetime, timedelta

# Set up the figure with exact specifications
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove all default spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Set exact dimensions
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)

# Colors
BLACK = '#111827'
GRAY = '#E5E7EB'
DIC_BLUE = '#2563EB'
IR_RED = '#DC2626'
XRD_VIOLET = '#7C3AED'

# Time axis parameters
TIME_START = 0
TIME_END = 240  # minutes
MAJOR_TICK_INTERVAL = 30
PIXELS_PER_MINUTE = 7.5  # 1800 pixels for 240 minutes

def time_to_x(time_min):
    """Convert time in minutes to x coordinate"""
    return 100 + time_min * PIXELS_PER_MINUTE

# Header section
ax.text(960, 1000, 'Synchronized Time Raster — DIC, IR, XRD (Aligned Frames & Continuous Capture)', 
        fontsize=48, fontweight='bold', ha='center', va='center', color=BLACK, fontfamily='sans-serif')
ax.text(960, 950, 'Unified clock; hardware trigger; Δt alignment', 
        fontsize=28, ha='center', va='center', color=BLACK, fontfamily='sans-serif')

# Main panel area (center 70% height)
panel_top = 800
panel_bottom = 200
panel_height = panel_top - panel_bottom

# Time axis
time_axis_y = panel_bottom - 50
ax.plot([time_to_x(0), time_to_x(240)], [time_axis_y, time_axis_y], 
        color=BLACK, linewidth=2)

# Major time ticks and labels
for i in range(0, 241, 30):
    x = time_to_x(i)
    ax.plot([x, x], [time_axis_y - 10, time_axis_y + 10], color=BLACK, linewidth=2)
    ax.text(x, time_axis_y - 30, str(i), ha='center', va='top', fontsize=16, color=BLACK)

# Minor grid lines
for i in range(0, 241, 5):
    x = time_to_x(i)
    ax.plot([x, x], [panel_bottom, panel_top], color=GRAY, linewidth=0.5, alpha=0.3)

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
    ax.plot([x, x], [panel_bottom, panel_top], color=BLACK, linewidth=1, 
            linestyle='--', alpha=0.7)
    ax.text(x, panel_top + 20, label, ha='center', va='bottom', fontsize=20, 
            color=BLACK, rotation=0)

# Raster rows
row_height = panel_height / 4
row_labels = [
    "DIC Cam A (1–5 Hz)",
    "DIC Cam B (1–5 Hz)", 
    "IR Camera (1–5 Hz)",
    "XRD (event snapshots)"
]
row_colors = [DIC_BLUE, DIC_BLUE, IR_RED, XRD_VIOLET]

for i, (label, color) in enumerate(zip(row_labels, row_colors)):
    row_y = panel_top - (i + 1) * row_height + row_height/2
    
    # Row label
    ax.text(20, row_y, label, ha='left', va='center', fontsize=26, 
            color=BLACK, fontweight='bold')
    
    # Row background
    ax.add_patch(Rectangle((100, row_y - row_height/2 + 5), 
                          time_to_x(240) - 100, row_height - 10, 
                          facecolor='white', edgecolor=GRAY, linewidth=0.5))
    
    # Frame ticks for DIC and IR rows (not XRD)
    if i < 3:  # DIC A, DIC B, IR
        # Generate frame times (1-5 Hz means 0.2-1 second intervals)
        # For visualization, use 0.5 second intervals (2 Hz)
        frame_interval = 0.5 / 60  # Convert to minutes
        frame_times = np.arange(0, 240, frame_interval)
        
        # Add some dropped frames for realism (around t=172 and t=204 for DIC B)
        if i == 1:  # DIC B
            frame_times = frame_times[~((frame_times > 171) & (frame_times < 173))]
            frame_times = frame_times[~((frame_times > 203) & (frame_times < 205))]
        
        for t in frame_times:
            if t <= 240:
                x = time_to_x(t)
                ax.plot([x, x], [row_y - 8, row_y + 8], color=color, linewidth=1.5)
    
    # Continuous capture bars for DIC and IR
    if i < 3:  # DIC A, DIC B, IR
        # Bar 1: 165-195 min
        bar1_x = time_to_x(165)
        bar1_width = time_to_x(195) - time_to_x(165)
        ax.add_patch(Rectangle((bar1_x, row_y - 6), bar1_width, 12, 
                              facecolor=color, alpha=0.7, edgecolor=color))
        
        # Bar 2: 200-210 min
        bar2_x = time_to_x(200)
        bar2_width = time_to_x(210) - time_to_x(200)
        ax.add_patch(Rectangle((bar2_x, row_y - 6), bar2_width, 12, 
                              facecolor=color, alpha=0.7, edgecolor=color))
    
    # XRD snapshots (only for XRD row)
    if i == 3:  # XRD row
        xrd_times = [120, 180, 210, 238]
        for t in xrd_times:
            x = time_to_x(t)
            circle = plt.Circle((x, row_y), 4, color=color, zorder=5)
            ax.add_patch(circle)

# Right-side callouts
callout_x = time_to_x(240) + 50
callout_y_start = panel_top - 50
callout_texts = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%",
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, text in enumerate(callout_texts):
    y = callout_y_start - i * 80
    # Create rounded rectangle background
    callout = FancyBboxPatch((callout_x, y - 25), 200, 50,
                            boxstyle="round,pad=5",
                            facecolor='white',
                            edgecolor=BLACK,
                            linewidth=1.5)
    ax.add_patch(callout)
    ax.text(callout_x + 100, y, text, ha='center', va='center', 
            fontsize=24, color=BLACK, fontweight='bold')

# Legend (bottom-right)
legend_x = 1600
legend_y = 150
legend_items = [
    ("Thin ticks = aligned frames", "line", BLACK),
    ("Thick bar = continuous video segment", "bar", DIC_BLUE),
    ("● = XRD snapshot", "circle", XRD_VIOLET)
]

for i, (text, item_type, color) in enumerate(legend_items):
    y = legend_y - i * 30
    
    if item_type == "line":
        ax.plot([legend_x, legend_x + 20], [y, y], color=color, linewidth=2)
    elif item_type == "bar":
        ax.add_patch(Rectangle((legend_x, y - 3), 20, 6, facecolor=color, alpha=0.7))
    elif item_type == "circle":
        circle = plt.Circle((legend_x + 10, y), 3, color=color)
        ax.add_patch(circle)
    
    ax.text(legend_x + 30, y, text, ha='left', va='center', fontsize=22, color=BLACK)

# Add subtle modality color hints to row labels
for i, (label, color) in enumerate(zip(row_labels, row_colors)):
    row_y = panel_top - (i + 1) * row_height + row_height/2
    # Add small color indicator
    ax.add_patch(Rectangle((5, row_y - 8), 8, 16, facecolor=color, alpha=0.3))

# Final styling
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)
ax.set_aspect('equal')

# Save the figure
plt.tight_layout()
plt.savefig('synchronized_time_raster.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Synchronized Time Raster figure created successfully!")
print("Files saved: synchronized_time_raster.png, synchronized_time_raster.pdf")