#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
with DIC, IR, and XRD data synchronized to a unified clock.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
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

# Set exact figure size
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

# Color scheme
colors = {
    'dic': '#2563EB',      # Blue
    'ir': '#DC2626',       # Red  
    'xrd': '#7C3AED',      # Violet
    'text': '#111827',     # Near black
    'grid': '#E5E7EB',     # Light gray
    'event': '#6B7280'     # Gray for event lines
}

# Time range and parameters
time_start = 0
time_end = 240  # minutes
major_ticks = np.arange(0, 241, 30)
minor_ticks = np.arange(0, 241, 5)

# Event times and labels
events = {
    120: "XRD snapshot",
    150: "Pre-hold", 
    165: "Hold start",
    180: "Under load",
    210: "Peak load",
    238: "Post-shock"
}

# XRD snapshot times
xrd_times = [120, 180, 210, 238]

# Continuous capture windows
continuous_windows = [
    (165, 195),  # Steady/hold
    (200, 210)   # Peak-load
]

# Raster row definitions
rows = [
    {"name": "DIC Cam A (1–5 Hz)", "color": colors['dic'], "y": 0.7},
    {"name": "DIC Cam B (1–5 Hz)", "color": colors['dic'], "y": 0.6},
    {"name": "IR Camera (1–5 Hz)", "color": colors['ir'], "y": 0.5},
    {"name": "XRD (snapshots)", "color": colors['xrd'], "y": 0.4}
]

# Main panel area (center 70% height)
panel_top = 0.85
panel_bottom = 0.15
panel_left = 0.15
panel_right = 0.75

# Convert time to x coordinates
def time_to_x(time):
    return panel_left + (time / time_end) * (panel_right - panel_left)

# Convert y position to figure coordinates
def row_to_y(row_idx):
    return panel_bottom + (row_idx + 1) * (panel_top - panel_bottom) / (len(rows) + 1)

# Add title and subheader
ax.text(0.5, 0.95, "Synchronized Time Raster — DIC, IR, XRD (Aligned Frames & Continuous Capture)", 
        fontsize=48, fontweight='bold', ha='center', va='top', color=colors['text'],
        transform=ax.transAxes, fontfamily='sans-serif')

ax.text(0.5, 0.90, "Unified clock; hardware trigger; Δt alignment", 
        fontsize=28, ha='center', va='top', color=colors['text'],
        transform=ax.transAxes, fontfamily='sans-serif')

# Draw main time axis
ax.plot([panel_left, panel_right], [panel_bottom, panel_bottom], 
        color=colors['text'], linewidth=2, transform=ax.transAxes)

# Add major time ticks and labels
for time in major_ticks:
    x_pos = time_to_x(time)
    ax.plot([x_pos, x_pos], [panel_bottom - 0.01, panel_bottom], 
            color=colors['text'], linewidth=1.5, transform=ax.transAxes)
    ax.text(x_pos, panel_bottom - 0.03, str(time), 
            fontsize=16, ha='center', va='top', color=colors['text'],
            transform=ax.transAxes, fontfamily='sans-serif')

# Add minor grid lines
for time in minor_ticks[1:-1]:  # Skip first and last to avoid overlap
    x_pos = time_to_x(time)
    ax.plot([x_pos, x_pos], [panel_bottom, panel_top], 
            color=colors['grid'], linewidth=0.5, alpha=0.5, transform=ax.transAxes)

# Add event markers (vertical dashed lines)
for time, label in events.items():
    x_pos = time_to_x(time)
    ax.plot([x_pos, x_pos], [panel_bottom, panel_top], 
            color=colors['event'], linewidth=1, linestyle='--', alpha=0.8, transform=ax.transAxes)
    ax.text(x_pos, panel_top + 0.01, label, 
            fontsize=20, ha='center', va='bottom', color=colors['text'],
            transform=ax.transAxes, fontfamily='sans-serif')

# Create raster rows
for i, row in enumerate(rows):
    y_pos = row_to_y(i)
    
    # Row label
    ax.text(panel_left - 0.02, y_pos, row['name'], 
            fontsize=26, ha='right', va='center', color=colors['text'],
            transform=ax.transAxes, fontfamily='sans-serif')
    
    # Draw row line
    ax.plot([panel_left, panel_right], [y_pos, y_pos], 
            color=colors['text'], linewidth=1, alpha=0.3, transform=ax.transAxes)
    
    # Add frame ticks (simulate 1-5 Hz sampling)
    if 'DIC' in row['name'] or 'IR' in row['name']:
        # Simulate variable frame rate with some dropped frames
        frame_times = []
        for t in np.arange(0, time_end, 0.2):  # ~5 Hz base rate
            # Simulate some dropped frames around t=172 and t=204 for DIC B
            if 'DIC Cam B' in row['name'] and (170 <= t <= 175 or 202 <= t <= 206):
                if np.random.random() < 0.3:  # 30% chance of dropped frame
                    continue
            frame_times.append(t)
        
        # Draw thin vertical ticks for each frame
        for t in frame_times:
            x_pos = time_to_x(t)
            ax.plot([x_pos, x_pos], [y_pos - 0.008, y_pos + 0.008], 
                    color=row['color'], linewidth=1, transform=ax.transAxes)
        
        # Add continuous capture bars
        for start_time, end_time in continuous_windows:
            start_x = time_to_x(start_time)
            end_x = time_to_x(end_time)
            width = end_x - start_x
            height = 0.016
            
            rect = Rectangle((start_x, y_pos - height/2), width, height,
                           facecolor=row['color'], alpha=0.7, transform=ax.transAxes)
            ax.add_patch(rect)
    
    elif 'XRD' in row['name']:
        # Add XRD snapshot markers (filled circles)
        for t in xrd_times:
            x_pos = time_to_x(t)
            circle = plt.Circle((x_pos, y_pos), 0.008, 
                              facecolor=row['color'], edgecolor=row['color'], 
                              transform=ax.transAxes)
            ax.add_patch(circle)

# Add right-side metric callouts
callout_x = panel_right + 0.02
callout_y_start = 0.7
callout_spacing = 0.08

callouts = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%", 
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, callout in enumerate(callouts):
    y_pos = callout_y_start - i * callout_spacing
    
    # Callout box
    box = Rectangle((callout_x - 0.01, y_pos - 0.025), 0.18, 0.05,
                   facecolor='white', edgecolor=colors['text'], 
                   linewidth=1, transform=ax.transAxes)
    ax.add_patch(box)
    
    # Callout text
    ax.text(callout_x + 0.08, y_pos, callout, 
            fontsize=24, ha='center', va='center', color=colors['text'],
            transform=ax.transAxes, fontfamily='sans-serif')

# Add legend (bottom-right)
legend_x = panel_right - 0.15
legend_y = panel_bottom - 0.08

legend_items = [
    ("Thin ticks", "aligned frames", colors['text']),
    ("Thick bar", "continuous video segment", colors['dic']),
    ("●", "XRD snapshot", colors['xrd'])
]

for i, (symbol, description, color) in enumerate(legend_items):
    y_pos = legend_y - i * 0.03
    
    if symbol == "●":
        circle = plt.Circle((legend_x, y_pos), 0.008, 
                          facecolor=color, edgecolor=color, 
                          transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(legend_x + 0.02, y_pos, f"= {description}", 
                fontsize=22, ha='left', va='center', color=colors['text'],
                transform=ax.transAxes, fontfamily='sans-serif')
    else:
        ax.text(legend_x, y_pos, symbol, 
                fontsize=22, ha='left', va='center', color=color,
                transform=ax.transAxes, fontfamily='sans-serif')
        ax.text(legend_x + 0.02, y_pos, f"= {description}", 
                fontsize=22, ha='left', va='center', color=colors['text'],
                transform=ax.transAxes, fontfamily='sans-serif')

# Set axis limits and save
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Save the figure
plt.savefig('synchronized_time_raster.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Synchronized Time Raster figure created successfully!")
print("Files saved: synchronized_time_raster.png and synchronized_time_raster.pdf")