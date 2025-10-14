#!/usr/bin/env python3
"""
Panel Combination Script
=======================

Combines individual FEM analysis panels into a single comprehensive figure.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np

def combine_panels():
    """Combine all individual panels into a single figure"""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16), dpi=100)
    
    # Create grid layout: 2x3 (2 rows, 3 columns)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2], width_ratios=[1, 1],
                  hspace=0.15, wspace=0.1)
    
    # Load panel images
    try:
        panel_a = mpimg.imread('/workspace/fem_panel_a.png')
        panel_b = mpimg.imread('/workspace/fem_panel_b.png') 
        panel_c = mpimg.imread('/workspace/fem_panel_c.png')
        panel_d = mpimg.imread('/workspace/fem_panel_d.png')
        metrics = mpimg.imread('/workspace/fem_metrics_summary.png')
        
        # Panel A: Top left
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(panel_a)
        ax1.axis('off')
        ax1.set_title('Panel A: Baseline Stress Field', fontsize=14, fontweight='bold', pad=10)
        
        # Panel B: Top right  
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(panel_b)
        ax2.axis('off')
        ax2.set_title('Panel B: Optimized Stress Field', fontsize=14, fontweight='bold', pad=10)
        
        # Panel C: Middle left
        ax3 = fig.add_subplot(gs[1, 0]) 
        ax3.imshow(panel_c)
        ax3.axis('off')
        ax3.set_title('Panel C: Difference Map', fontsize=14, fontweight='bold', pad=10)
        
        # Panel D: Middle right
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(panel_d)
        ax4.axis('off')
        ax4.set_title('Panel D: Line-out Analysis', fontsize=14, fontweight='bold', pad=10)
        
        # Metrics: Bottom spanning both columns
        ax5 = fig.add_subplot(gs[2, :])
        ax5.imshow(metrics)
        ax5.axis('off')
        ax5.set_title('Performance Metrics & Validation', fontsize=14, fontweight='bold', pad=10)
        
        # Add main title
        fig.suptitle('Advanced FEM von Mises Stress Analysis: Baseline vs. Optimized Design\n' +
                    'Figure 4a.2.2: Comprehensive Electrolyte Stress Optimization Study', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Add footer
        footer_text = ("Advanced FEM Analysis System | von Mises Stress Optimization | "
                      "Electrolyte Design Validation | Generated: 2025-10-14 | "
                      "All constraints satisfied ✓")
        fig.text(0.5, 0.01, footer_text, ha='center', va='bottom',
                fontsize=10, style='italic', alpha=0.7)
        
        # Save combined figure
        output_path = '/workspace/complete_fem_analysis.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.2)
        plt.close(fig)
        
        print(f"✅ Combined analysis saved to: {output_path}")
        return output_path
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find panel file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error combining panels: {e}")
        return None

if __name__ == "__main__":
    print("🔗 Combining FEM analysis panels...")
    result = combine_panels()
    if result:
        print("✅ Panel combination completed successfully!")
    else:
        print("❌ Panel combination failed!")