#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
for DIC, IR, XRD with continuous capture windows and event markers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

# Set up the figure with exact specifications
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)  # 1920x1080 at 100 DPI
fig.patch.set_facecolor('white')

# Create grid layout
gs = gridspec.GridSpec(4, 3, height_ratios=[0.15, 0.7, 0.1, 0.05], 
                      width_ratios=[0.8, 0.15, 0.05], hspace=0.3, wspace=0.1)

# Color scheme
DIC_COLOR = '#2563EB'  # Blue
IR_COLOR = '#DC2626'   # Red
XRD_COLOR = '#7C3AED'  # Violet
THERMO_COLOR = '#059669'  # Green
AXIS_COLOR = '#111827'  # Near black
GRID_COLOR = '#E5E7EB'  # Light gray

# Time parameters
time_start = 0
time_end = 240
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
continuous_windows = [(165, 195), (200, 210)]

# XRD snapshot times
xrd_snapshots = [120, 180, 210, 238]

# Raster row definitions
raster_rows = [
    ("DIC Cam A (1–5 Hz)", DIC_COLOR),
    ("DIC Cam B (1–5 Hz)", DIC_COLOR), 
    ("IR Camera (1–5 Hz)", IR_COLOR),
    ("XRD (snapshots)", XRD_COLOR),
    ("Thermocouples (1–2 Hz)", THERMO_COLOR)
]

# Create main plotting area
ax_main = fig.add_subplot(gs[1, 0])

# Set up the time axis
ax_main.set_xlim(time_start, time_end)
ax_main.set_ylim(-0.5, len(raster_rows) - 0.5)

# Draw major and minor grid
ax_main.set_xticks(major_ticks)
ax_main.set_xticks(minor_ticks, minor=True)
ax_main.grid(True, which='major', color=GRID_COLOR, linewidth=0.5, alpha=0.7)
ax_main.grid(True, which='minor', color=GRID_COLOR, linewidth=0.3, alpha=0.5)

# Set y-axis labels for raster rows
y_positions = np.arange(len(raster_rows))
ax_main.set_yticks(y_positions)
ax_main.set_yticklabels([row[0] for row in raster_rows], fontsize=26, color=AXIS_COLOR)

# Draw event markers
for time, label in events.items():
    ax_main.axvline(x=time, color=AXIS_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
    ax_main.text(time, len(raster_rows) - 0.3, label, ha='center', va='bottom', 
                fontsize=20, color=AXIS_COLOR, rotation=0)

# Generate frame data for each row
frame_rates = [3, 3, 3, 0, 1.5]  # Hz for each row
dropped_frames = {1: [172, 204]}  # DIC Cam B dropped frames

for i, (row_name, color) in enumerate(raster_rows):
    y_pos = y_positions[i]
    
    if i < 3:  # DIC and IR rows
        # Generate frame times
        frame_rate = frame_rates[i]
        frame_times = np.arange(time_start, time_end, 1/frame_rate)
        
        # Remove dropped frames for DIC Cam B
        if i == 1 and i in dropped_frames:
            for drop_time in dropped_frames[i]:
                frame_times = frame_times[np.abs(frame_times - drop_time) > 0.5]
        
        # Draw thin vertical ticks for frames
        for frame_time in frame_times:
            ax_main.plot([frame_time, frame_time], [y_pos - 0.15, y_pos + 0.15], 
                        color=color, linewidth=2, alpha=0.8)
        
        # Draw continuous capture bars
        for start_time, end_time in continuous_windows:
            rect = Rectangle((start_time, y_pos - 0.25), end_time - start_time, 0.5,
                           facecolor=color, alpha=0.6, edgecolor=color, linewidth=0)
            ax_main.add_patch(rect)
    
    elif i == 3:  # XRD row
        # Draw XRD snapshot markers
        for snapshot_time in xrd_snapshots:
            circle = plt.Circle((snapshot_time, y_pos), 0.08, 
                              facecolor=XRD_COLOR, edgecolor=XRD_COLOR, alpha=0.9)
            ax_main.add_patch(circle)
    
    else:  # Thermocouples row
        # Generate thermocouple data points
        thermo_rate = frame_rates[i]
        thermo_times = np.arange(time_start, time_end, 1/thermo_rate)
        
        for thermo_time in thermo_times:
            ax_main.plot([thermo_time, thermo_time], [y_pos - 0.1, y_pos + 0.1], 
                        color=color, linewidth=1.5, alpha=0.7)

# Format the main axis
ax_main.set_xlabel('Time (minutes)', fontsize=24, color=AXIS_COLOR, fontweight='bold')
ax_main.tick_params(axis='x', labelsize=20, colors=AXIS_COLOR)
ax_main.tick_params(axis='y', labelsize=0)  # Hide y-axis tick labels
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
ax_main.spines['left'].set_visible(False)
ax_main.spines['bottom'].set_color(AXIS_COLOR)
ax_main.spines['bottom'].set_linewidth(2)

# Add title and subheader
ax_title = fig.add_subplot(gs[0, :])
ax_title.text(0.5, 0.7, 'Synchronized Time Raster — DIC, IR, XRD (Aligned Frames & Continuous Capture)', 
              ha='center', va='center', fontsize=48, fontweight='bold', color=AXIS_COLOR)
ax_title.text(0.5, 0.2, 'Unified clock; hardware trigger; Δt alignment', 
              ha='center', va='center', fontsize=28, color=AXIS_COLOR, style='italic')
ax_title.set_xlim(0, 1)
ax_title.set_ylim(0, 1)
ax_title.axis('off')

# Add callout boxes
ax_callouts = fig.add_subplot(gs[1, 1])
ax_callouts.axis('off')

callouts = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%", 
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, callout in enumerate(callouts):
    y_pos = 0.8 - i * 0.25
    # Create rounded rectangle
    rect = patches.FancyBboxPatch((0.1, y_pos - 0.08), 0.8, 0.12,
                                 boxstyle="round,pad=0.02", 
                                 facecolor='#F3F4F6', edgecolor=AXIS_COLOR, 
                                 linewidth=1.5, alpha=0.9)
    ax_callouts.add_patch(rect)
    ax_callouts.text(0.5, y_pos, callout, ha='center', va='center', 
                    fontsize=24, color=AXIS_COLOR, fontweight='bold')

# Add legend
ax_legend = fig.add_subplot(gs[2, 2])
ax_legend.axis('off')

legend_items = [
    ("Thin ticks", "aligned frames", DIC_COLOR),
    ("Thick bar", "continuous video segment", DIC_COLOR), 
    ("●", "XRD snapshot", XRD_COLOR)
]

y_start = 0.8
for i, (symbol, description, color) in enumerate(legend_items):
    y_pos = y_start - i * 0.25
    if symbol == "Thin ticks":
        ax_legend.plot([0.1, 0.1], [y_pos - 0.05, y_pos + 0.05], 
                      color=color, linewidth=2)
    elif symbol == "Thick bar":
        rect = Rectangle((0.05, y_pos - 0.05), 0.1, 0.1, 
                        facecolor=color, alpha=0.6, edgecolor=color)
        ax_legend.add_patch(rect)
    else:  # XRD symbol
        circle = plt.Circle((0.1, y_pos), 0.03, facecolor=color, edgecolor=color)
        ax_legend.add_patch(circle)
    
    ax_legend.text(0.25, y_pos, f"{symbol} = {description}", 
                  fontsize=22, color=AXIS_COLOR, va='center')

ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)

# Add accessibility note
ax_access = fig.add_subplot(gs[3, :])
ax_access.text(0.5, 0.5, 
              "Accessibility: Horizontal timeline from 0 to 240 minutes with five stacked rows for DIC A, DIC B, IR, XRD, and thermocouples. "
              "Thin ticks align across rows at the same times. Thick bars on DIC and IR indicate continuous recording from 165–195 and 200–210 minutes. "
              "XRD shows discrete dots at 120, 180, 210, and 238 minutes. Vertical dashed lines label key events. "
              "Side notes report ≤30 ms desynchronization and <1% dropped frames.",
              ha='center', va='center', fontsize=14, color='#6B7280', style='italic', wrap=True)
ax_access.axis('off')

# Save the figure
plt.tight_layout()
plt.savefig('synchronized_time_raster.png', dpi=100, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Synchronized Time Raster figure created successfully!")
print("Files saved: synchronized_time_raster.png and synchronized_time_raster.pdf")