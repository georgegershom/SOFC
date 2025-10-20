#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
with continuous capture windows and discrete events.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime, timedelta

# Set up the figure with exact specifications
fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='white')
fig.patch.set_facecolor('white')

# Set DPI for 1920x1080 output
plt.rcParams['figure.dpi'] = 100

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

# Event markers
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
    (165, 195),  # Steady/hold
    (200, 210)   # Peak-load window
]

# XRD snapshots
xrd_times = [120, 180, 210, 238]

# Raster rows configuration
rows = [
    {"name": "DIC Cam A (1–5 Hz)", "color": colors['dic'], "y": 0.75},
    {"name": "DIC Cam B (1–5 Hz)", "color": colors['dic'], "y": 0.65},
    {"name": "IR Camera (1–5 Hz)", "color": colors['ir'], "y": 0.55},
    {"name": "XRD (snapshots)", "color": colors['xrd'], "y": 0.45}
]

# Set up the main plotting area
ax.set_xlim(0, 240)
ax.set_ylim(0, 1)
ax.set_facecolor('white')

# Remove default axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add title and subheader
fig.text(0.5, 0.95, 'Synchronized Time Raster — DIC, IR, XRD (Aligned Frames & Continuous Capture)', 
         fontsize=48, fontweight='bold', ha='center', va='top', color=colors['text'],
         fontfamily='sans-serif')

fig.text(0.5, 0.90, 'Unified clock; hardware trigger; Δt alignment', 
         fontsize=28, ha='center', va='top', color=colors['text'],
         fontfamily='sans-serif')

# Create time axis
ax.axhline(y=0.35, xmin=0.05, xmax=0.95, color=colors['text'], linewidth=2)

# Add major time ticks and labels
for t in major_ticks:
    x_pos = 0.05 + (t / 240) * 0.9
    ax.plot([x_pos, x_pos], [0.33, 0.37], color=colors['text'], linewidth=2)
    ax.text(x_pos, 0.30, str(t), ha='center', va='top', fontsize=16, color=colors['text'])

# Add minor grid lines
for t in minor_ticks[1:-1]:  # Skip first and last to avoid overlap
    x_pos = 0.05 + (t / 240) * 0.9
    ax.axvline(x=x_pos, ymin=0.35, ymax=0.95, color=colors['grid'], linewidth=0.5, alpha=0.5)

# Add event markers (vertical dashed lines)
for t, label in events.items():
    x_pos = 0.05 + (t / 240) * 0.9
    ax.axvline(x=x_pos, ymin=0.35, ymax=0.95, color=colors['event'], 
               linewidth=1.5, linestyle='--', alpha=0.8)
    ax.text(x_pos, 0.97, label, ha='center', va='bottom', fontsize=20, 
            color=colors['text'], rotation=0)

# Create raster rows
for i, row in enumerate(rows):
    y_pos = row['y']
    
    # Row label
    ax.text(0.02, y_pos, row['name'], ha='left', va='center', fontsize=26, 
            color=colors['text'], fontweight='bold')
    
    # Add frame ticks (simulating 1-5 Hz)
    frame_rate = 3  # Average of 1-5 Hz
    frame_interval = 60 / frame_rate  # seconds per frame in minutes
    frame_times = np.arange(0, 240, frame_interval)
    
    for t in frame_times:
        if 0 <= t <= 240:
            x_pos = 0.05 + (t / 240) * 0.9
            # Add small gap to simulate dropped frames (around t=172 and t=204 for DIC B)
            if i == 1 and (170 <= t <= 175 or 202 <= t <= 206):
                continue
            ax.plot([x_pos, x_pos], [y_pos-0.02, y_pos+0.02], 
                   color=row['color'], linewidth=1.5)
    
    # Add continuous capture bars for DIC and IR rows
    if i < 3:  # DIC A, DIC B, IR rows
        for start, end in continuous_windows:
            x_start = 0.05 + (start / 240) * 0.9
            x_end = 0.05 + (end / 240) * 0.9
            width = x_end - x_start
            rect = Rectangle((x_start, y_pos-0.03), width, 0.06, 
                           facecolor=row['color'], alpha=0.3, edgecolor=row['color'], linewidth=2)
            ax.add_patch(rect)
    
    # Add XRD snapshots for XRD row
    if i == 3:  # XRD row
        for t in xrd_times:
            x_pos = 0.05 + (t / 240) * 0.9
            circle = plt.Circle((x_pos, y_pos), 0.015, color=row['color'], zorder=5)
            ax.add_patch(circle)

# Add right-side callout boxes
callout_x = 0.75
callout_y_start = 0.85
callout_spacing = 0.08

callouts = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%", 
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, callout_text in enumerate(callouts):
    y_pos = callout_y_start - i * callout_spacing
    
    # Create rounded rectangle background
    rect = patches.FancyBboxPatch((callout_x, y_pos-0.03), 0.22, 0.06,
                                 boxstyle="round,pad=0.01", 
                                 facecolor='white', edgecolor=colors['text'], 
                                 linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    
    # Add text
    ax.text(callout_x + 0.11, y_pos, callout_text, ha='center', va='center', 
            fontsize=24, color=colors['text'], fontweight='bold')

# Add legend in bottom-right
legend_x = 0.75
legend_y = 0.15

# Legend background
legend_rect = patches.FancyBboxPatch((legend_x, legend_y-0.08), 0.22, 0.16,
                                   boxstyle="round,pad=0.01",
                                   facecolor='white', edgecolor=colors['text'],
                                   linewidth=1.5, alpha=0.9)
ax.add_patch(legend_rect)

# Legend items
legend_items = [
    ("Thin ticks", "aligned frames", colors['text']),
    ("Thick bar", "continuous video segment", colors['dic']),
    ("●", "XRD snapshot", colors['xrd'])
]

for i, (symbol, description, color) in enumerate(legend_items):
    y_pos = legend_y + 0.12 - i * 0.04
    
    if symbol == "Thin ticks":
        ax.plot([legend_x + 0.02, legend_x + 0.02], [y_pos-0.01, y_pos+0.01], 
               color=color, linewidth=2)
    elif symbol == "Thick bar":
        rect = Rectangle((legend_x + 0.015, y_pos-0.01), 0.01, 0.02, 
                       facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
        ax.add_patch(rect)
    else:  # XRD dot
        circle = plt.Circle((legend_x + 0.02, y_pos), 0.005, color=color)
        ax.add_patch(circle)
    
    ax.text(legend_x + 0.04, y_pos, f"{symbol} = {description}", 
            ha='left', va='center', fontsize=22, color=colors['text'])

# Add time axis label
ax.text(0.5, 0.25, 'Time (minutes)', ha='center', va='center', 
        fontsize=20, color=colors['text'], fontweight='bold')

# Ensure proper layout
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1)

# Save the figure
plt.savefig('synchronized_time_raster.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Figure saved as 'synchronized_time_raster.png' and 'synchronized_time_raster.pdf'")
print("Dimensions: 1920×1080 pixels")
print("Format: Clean vector graphics with professional styling")