#!/usr/bin/env python3
"""
Synchronized Time Raster Figure Generator
Creates a professional visualization showing time-aligned sensing streams
with continuous capture windows and discrete events.
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

# Figure dimensions (16:9, 1920×1080)
fig.set_size_inches(19.2, 10.8)

# Color scheme
colors = {
    'dic': '#2563EB',      # Blue
    'ir': '#DC2626',       # Red  
    'xrd': '#7C3AED',      # Violet
    'text': '#111827',     # Near black
    'grid': '#E5E7EB',     # Light gray
    'event': '#6B7280'     # Gray for event lines
}

# Time range and settings
time_start = 0
time_end = 240
major_ticks = np.arange(0, 241, 30)
minor_ticks = np.arange(0, 241, 5)

# Event timestamps and labels
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
    (165, 195),  # Steady/hold window
    (200, 210)   # Peak-load window
]

# XRD snapshot times
xrd_snapshots = [120, 180, 210, 238]

# Frame rates and dropped frame simulation
frame_rates = {
    'dic_a': 3,  # Hz
    'dic_b': 3,  # Hz  
    'ir': 3,     # Hz
    'thermo': 1.5  # Hz
}

# Simulate dropped frames for DIC B around t=172 and t=204
def get_frame_times(rate, start_time, end_time, dropped_intervals=None):
    """Generate frame times with optional dropped frame simulation"""
    frame_interval = 1.0 / rate  # seconds per frame
    times = []
    current_time = start_time
    
    while current_time <= end_time:
        # Check if this frame should be dropped
        if dropped_intervals:
            should_drop = False
            for drop_start, drop_end in dropped_intervals:
                if drop_start <= current_time <= drop_end:
                    should_drop = True
                    break
            if not should_drop:
                times.append(current_time)
        else:
            times.append(current_time)
        current_time += frame_interval
    
    return times

# Set up the main plotting area
ax.set_xlim(time_start, time_end)
ax.set_ylim(-0.5, 4.5)

# Remove default axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Add title and subheader
fig.text(0.5, 0.95, 'Synchronized Time Raster — DIC, IR, XRD (Aligned Frames & Continuous Capture)', 
         fontsize=48, fontweight='bold', ha='center', va='top', color=colors['text'])
fig.text(0.5, 0.90, 'Unified clock; hardware trigger; Δt alignment', 
         fontsize=28, ha='center', va='top', color=colors['text'])

# Add time axis
ax.axhline(y=0, color=colors['text'], linewidth=2)
ax.set_xlabel('Time (minutes)', fontsize=24, color=colors['text'], labelpad=20)

# Add major and minor grid lines
for tick in major_ticks:
    ax.axvline(x=tick, color=colors['grid'], linewidth=0.5, alpha=0.7)
    ax.text(tick, -0.3, str(int(tick)), ha='center', va='top', fontsize=20, color=colors['text'])

for tick in minor_ticks[1:-1]:  # Skip first and last to avoid overlap
    if tick not in major_ticks:
        ax.axvline(x=tick, color=colors['grid'], linewidth=0.3, alpha=0.5)

# Add event markers (vertical dashed lines)
for time, label in events.items():
    ax.axvline(x=time, color=colors['event'], linewidth=2, linestyle='--', alpha=0.8)
    ax.text(time, 4.2, label, ha='center', va='bottom', fontsize=20, 
            color=colors['text'], rotation=0)

# Define row positions and labels
row_data = [
    {'name': 'DIC Cam A (1–5 Hz)', 'y': 3.5, 'color': colors['dic']},
    {'name': 'DIC Cam B (1–5 Hz)', 'y': 2.5, 'color': colors['dic']},
    {'name': 'IR Camera (1–5 Hz)', 'y': 1.5, 'color': colors['ir']},
    {'name': 'XRD (event snapshots)', 'y': 0.5, 'color': colors['xrd']}
]

# Add row labels
for row in row_data:
    ax.text(-15, row['y'], row['name'], ha='right', va='center', 
            fontsize=26, color=colors['text'], fontweight='bold')

# Generate and plot frame ticks for each row
dropped_intervals_dic_b = [(171, 173), (203, 205)]  # Simulate dropped frames

for i, row in enumerate(row_data):
    if row['name'].startswith('DIC') or row['name'].startswith('IR'):
        # Generate frame times
        if 'DIC B' in row['name']:
            frame_times = get_frame_times(frame_rates['dic_b'], time_start, time_end, dropped_intervals_dic_b)
        else:
            frame_times = get_frame_times(frame_rates['dic_a'], time_start, time_end)
        
        # Plot thin vertical ticks for frames
        for frame_time in frame_times:
            if time_start <= frame_time <= time_end:
                ax.plot([frame_time, frame_time], 
                       [row['y'] - 0.15, row['y'] + 0.15], 
                       color=row['color'], linewidth=2, alpha=0.8)
        
        # Add continuous capture bars
        for window_start, window_end in continuous_windows:
            rect = Rectangle((window_start, row['y'] - 0.25), 
                           window_end - window_start, 0.5,
                           facecolor=row['color'], alpha=0.6, edgecolor=row['color'])
            ax.add_patch(rect)
    
    elif row['name'].startswith('XRD'):
        # Add XRD snapshot dots
        for snapshot_time in xrd_snapshots:
            circle = plt.Circle((snapshot_time, row['y']), 0.08, 
                              facecolor=row['color'], edgecolor=row['color'], alpha=0.9)
            ax.add_patch(circle)

# Add metric callout boxes on the right side
callout_x = 250
callout_y_start = 3.5
callout_height = 0.4
callout_width = 25

callouts = [
    "Max desync ≤ 30 ms\nduring ramps",
    "Dropped frames < 1%",
    "Co-registration error\n≤ 0.3–0.5 px (DIC↔IR)"
]

for i, callout_text in enumerate(callouts):
    y_pos = callout_y_start - i * 0.8
    callout_box = FancyBboxPatch((callout_x, y_pos - callout_height/2), 
                                callout_width, callout_height,
                                boxstyle="round,pad=0.1", 
                                facecolor='white', edgecolor=colors['text'],
                                linewidth=1.5)
    ax.add_patch(callout_box)
    ax.text(callout_x + callout_width/2, y_pos, callout_text, 
            ha='center', va='center', fontsize=20, color=colors['text'])

# Add legend in bottom-right
legend_x = 180
legend_y = 0.2
legend_items = [
    ("Thin ticks", "aligned frames", colors['text']),
    ("Thick bar", "continuous video segment", colors['dic']),
    ("●", "XRD snapshot", colors['xrd'])
]

for i, (symbol, description, color) in enumerate(legend_items):
    y_pos = legend_y - i * 0.3
    if symbol == "●":
        circle = plt.Circle((legend_x, y_pos), 0.05, facecolor=color, edgecolor=color)
        ax.add_patch(circle)
        ax.text(legend_x + 0.2, y_pos, f"{symbol} = {description}", 
                ha='left', va='center', fontsize=22, color=colors['text'])
    else:
        ax.text(legend_x, y_pos, f"{symbol} = {description}", 
                ha='left', va='center', fontsize=22, color=colors['text'])

# Set axis properties
ax.set_xlim(time_start - 20, time_end + 50)  # Extra space for labels
ax.set_ylim(-0.5, 4.5)
ax.set_xticks(major_ticks)
ax.set_xticklabels([str(int(t)) for t in major_ticks], fontsize=18)
ax.tick_params(axis='x', which='major', length=0, pad=10)
ax.tick_params(axis='y', which='both', length=0, pad=0)

# Remove y-axis ticks and labels
ax.set_yticks([])

# Add subtle grid for alignment
ax.grid(True, axis='x', alpha=0.3, color=colors['grid'], linewidth=0.5)

# Adjust layout to prevent clipping
plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.12, right=0.85)

# Save the figure
plt.savefig('synchronized_time_raster.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('synchronized_time_raster.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Synchronized Time Raster figure generated successfully!")
print("Files saved: synchronized_time_raster.png, synchronized_time_raster.pdf")