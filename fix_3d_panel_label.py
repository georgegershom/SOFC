"""
Fix for 3D Panel Label Issue in Chemical Expansion Tensor Visualization

The issue: Axes3D.text() requires (x, y, z, s) arguments, not (x, y, s)
Solution: Detect 3D axes and use text2D() or check axis type
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def add_panel_label(ax, label, cfg, x=-0.08, y=1.06):
    """
    Add professional panel label
    
    Fixed version that handles both 2D and 3D axes correctly.
    """
    # Check if this is a 3D axis
    if hasattr(ax, 'zaxis') or isinstance(ax, Axes3D):
        # For 3D axes, use text2D which works with 2D coordinates
        ax.text2D(x, y, f'({label})', 
                  transform=ax.transAxes,
                  fontsize=cfg.TITLE_SIZE + 2, 
                  fontweight='bold',
                  color=cfg.COLORS['text_dark'],
                  verticalalignment='top', 
                  horizontalalignment='left')
    else:
        # For 2D axes, use regular text
        ax.text(x, y, f'({label})', 
                transform=ax.transAxes,
                fontsize=cfg.TITLE_SIZE + 2, 
                fontweight='bold',
                color=cfg.COLORS['text_dark'],
                verticalalignment='top', 
                horizontalalignment='left')


# Alternative simpler fix: Just always use text2D for consistency
def add_panel_label_v2(ax, label, cfg, x=-0.08, y=1.06):
    """
    Add professional panel label (simplified version)
    
    Uses text2D which works for both 2D and 3D axes in matplotlib.
    """
    # text2D works for both 2D and 3D axes
    if hasattr(ax, 'text2D'):
        # 3D axis
        ax.text2D(x, y, f'({label})', 
                  transform=ax.transAxes,
                  fontsize=cfg.TITLE_SIZE + 2, 
                  fontweight='bold',
                  color=cfg.COLORS['text_dark'],
                  verticalalignment='top', 
                  horizontalalignment='left')
    else:
        # 2D axis
        ax.text(x, y, f'({label})', 
                transform=ax.transAxes,
                fontsize=cfg.TITLE_SIZE + 2, 
                fontweight='bold',
                color=cfg.COLORS['text_dark'],
                verticalalignment='top', 
                horizontalalignment='left')


print("""
=================================================================
FIX FOR 3D PANEL LABEL ISSUE
=================================================================

Replace your add_panel_label function with the fixed version above.

The issue was that Axes3D.text() requires (x, y, z, string) arguments,
while regular Axes.text() requires (x, y, string) arguments.

The solution uses text2D() for 3D axes, which accepts 2D coordinates.

=================================================================
""")
