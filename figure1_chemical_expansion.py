"""
Figure 1: Chemical Expansion Tensor - Ni to NiO Phase Transformation
Application: SOFC Ni-YSZ Redox Eigenstrain Analysis
Model: Temperature-Dependent Tensor

This script generates a comprehensive visualization of the chemical expansion
tensor for the Ni → NiO phase transformation used in SOFC thermomechanical modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from dataclasses import dataclass
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FigureConfig:
    """Configuration class for figure styling"""
    # Figure dimensions
    FIG_WIDTH: float = 14
    FIG_HEIGHT: float = 10
    DPI: int = 150
    
    # Font sizes
    TITLE_SIZE: int = 12
    LABEL_SIZE: int = 10
    TICK_SIZE: int = 9
    LEGEND_SIZE: int = 9
    
    # Color scheme
    COLORS: Dict = None
    
    def __post_init__(self):
        self.COLORS = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'text_dark': '#2c3e50',
            'grid': '#ecf0f1',
            'background': '#ffffff',
            'strain_xx': '#e74c3c',
            'strain_yy': '#3498db',
            'strain_zz': '#27ae60',
            'volumetric': '#9b59b6',
            'tensor_face': '#3498db',
            'tensor_edge': '#2c3e50',
        }


def generate_chemical_expansion_data(T_range: np.ndarray) -> Dict:
    """
    Generate chemical expansion tensor data for Ni → NiO transformation.
    
    The chemical eigenstrain tensor accounts for the volume change during
    oxidation of Ni to NiO in SOFC anode materials.
    
    Parameters:
        T_range: Temperature range in Celsius
        
    Returns:
        Dictionary containing strain components and derived quantities
    """
    # Base volumetric expansion ratio for Ni → NiO (approximately 1.67)
    # This is based on the Pilling-Bedworth ratio
    base_expansion = 0.67  # 67% volume increase
    
    # Temperature dependence parameters
    T_ref = 25.0  # Reference temperature (°C)
    T_normalized = (T_range - T_ref) / (800.0 - T_ref)
    
    # Chemical eigenstrain components (anisotropic expansion)
    # Based on crystallographic considerations of NiO formation
    epsilon_xx = 0.18 * (1 + 0.15 * T_normalized)  # Principal strain 1
    epsilon_yy = 0.22 * (1 + 0.12 * T_normalized)  # Principal strain 2
    epsilon_zz = 0.27 * (1 + 0.18 * T_normalized)  # Principal strain 3 (max)
    
    # Volumetric strain (trace of strain tensor)
    epsilon_vol = epsilon_xx + epsilon_yy + epsilon_zz
    
    # Deviatoric components
    epsilon_mean = epsilon_vol / 3
    epsilon_dev_xx = epsilon_xx - epsilon_mean
    epsilon_dev_yy = epsilon_yy - epsilon_mean
    epsilon_dev_zz = epsilon_zz - epsilon_mean
    
    # Strain rate (derivative with respect to temperature)
    d_epsilon_xx = np.gradient(epsilon_xx, T_range)
    d_epsilon_yy = np.gradient(epsilon_yy, T_range)
    d_epsilon_zz = np.gradient(epsilon_zz, T_range)
    
    return {
        'T': T_range,
        'epsilon_xx': epsilon_xx,
        'epsilon_yy': epsilon_yy,
        'epsilon_zz': epsilon_zz,
        'epsilon_vol': epsilon_vol,
        'epsilon_mean': epsilon_mean,
        'epsilon_dev_xx': epsilon_dev_xx,
        'epsilon_dev_yy': epsilon_dev_yy,
        'epsilon_dev_zz': epsilon_dev_zz,
        'd_epsilon_xx': d_epsilon_xx,
        'd_epsilon_yy': d_epsilon_yy,
        'd_epsilon_zz': d_epsilon_zz,
    }


def add_panel_label(ax, label: str, cfg: FigureConfig, x: float = -0.08, y: float = 1.06):
    """
    Add professional panel label (a), (b), (c), etc.
    
    This function handles both 2D and 3D axes by checking for the text2D method
    which is specific to 3D axes in matplotlib.
    
    Parameters:
        ax: Matplotlib axes object (2D or 3D)
        label: Panel label character (e.g., 'a', 'b', 'c')
        cfg: Figure configuration object
        x: X position in axes coordinates
        y: Y position in axes coordinates
    """
    # Check if this is a 3D axis - use text2D for 3D axes
    if hasattr(ax, 'text2D'):
        ax.text2D(x, y, f'({label})', transform=ax.transAxes,
                  fontsize=cfg.TITLE_SIZE + 2, fontweight='bold',
                  color=cfg.COLORS['text_dark'],
                  verticalalignment='top', horizontalalignment='left')
    else:
        ax.text(x, y, f'({label})', transform=ax.transAxes,
                fontsize=cfg.TITLE_SIZE + 2, fontweight='bold',
                color=cfg.COLORS['text_dark'],
                verticalalignment='top', horizontalalignment='left')


def create_strain_components_panel(ax, cfg: FigureConfig, data: Dict):
    """
    Panel (a): Chemical strain components vs temperature.
    
    Shows the three principal strain components (εxx, εyy, εzz) as functions
    of temperature during the Ni → NiO transformation.
    """
    T = data['T']
    
    # Plot strain components
    ax.plot(T, data['epsilon_xx'] * 100, '-', color=cfg.COLORS['strain_xx'], 
            linewidth=2, label=r'$\varepsilon_{xx}^{ch}$')
    ax.plot(T, data['epsilon_yy'] * 100, '--', color=cfg.COLORS['strain_yy'], 
            linewidth=2, label=r'$\varepsilon_{yy}^{ch}$')
    ax.plot(T, data['epsilon_zz'] * 100, '-.', color=cfg.COLORS['strain_zz'], 
            linewidth=2, label=r'$\varepsilon_{zz}^{ch}$')
    
    # Formatting
    ax.set_xlabel('Temperature (°C)', fontsize=cfg.LABEL_SIZE)
    ax.set_ylabel('Chemical Strain (%)', fontsize=cfg.LABEL_SIZE)
    ax.set_title('Strain Components', fontsize=cfg.TITLE_SIZE, fontweight='bold')
    ax.legend(loc='upper left', fontsize=cfg.LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=0.3, color=cfg.COLORS['grid'])
    ax.tick_params(labelsize=cfg.TICK_SIZE)
    ax.set_xlim([T.min(), T.max()])
    
    # Add operating temperature annotation
    ax.axvline(x=800, color='gray', linestyle=':', alpha=0.7)
    ax.annotate('Operating\nTemp.', xy=(800, data['epsilon_xx'][-1]*100), 
                xytext=(700, data['epsilon_xx'][-1]*100 + 3),
                fontsize=cfg.TICK_SIZE - 1, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))


def create_volumetric_panel(ax, cfg: FigureConfig, data: Dict):
    """
    Panel (b): Volumetric expansion and mean strain.
    
    Shows the total volumetric strain and its relationship to mean hydrostatic
    strain during the oxidation process.
    """
    T = data['T']
    
    # Fill between for volumetric expansion
    ax.fill_between(T, 0, data['epsilon_vol'] * 100, 
                    alpha=0.3, color=cfg.COLORS['volumetric'])
    ax.plot(T, data['epsilon_vol'] * 100, '-', color=cfg.COLORS['volumetric'], 
            linewidth=2.5, label=r'$\varepsilon_{vol}^{ch} = tr(\varepsilon^{ch})$')
    ax.plot(T, data['epsilon_mean'] * 100 * 3, '--', color=cfg.COLORS['primary'], 
            linewidth=1.5, label=r'$3\bar{\varepsilon}^{ch}$ (hydrostatic)')
    
    # Formatting
    ax.set_xlabel('Temperature (°C)', fontsize=cfg.LABEL_SIZE)
    ax.set_ylabel('Volumetric Strain (%)', fontsize=cfg.LABEL_SIZE)
    ax.set_title('Volumetric Expansion', fontsize=cfg.TITLE_SIZE, fontweight='bold')
    ax.legend(loc='upper left', fontsize=cfg.LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=0.3, color=cfg.COLORS['grid'])
    ax.tick_params(labelsize=cfg.TICK_SIZE)
    ax.set_xlim([T.min(), T.max()])
    ax.set_ylim([0, data['epsilon_vol'].max() * 100 * 1.15])
    
    # Pilling-Bedworth ratio annotation
    pb_ratio = 1.0 + data['epsilon_vol'][-1]
    ax.annotate(f'P-B Ratio ≈ {pb_ratio:.2f}', 
                xy=(T[-1], data['epsilon_vol'][-1] * 100),
                xytext=(T[-1] - 200, data['epsilon_vol'][-1] * 100 - 10),
                fontsize=cfg.TICK_SIZE, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=cfg.COLORS['volumetric'], alpha=0.9),
                arrowprops=dict(arrowstyle='->', color=cfg.COLORS['volumetric']))


def create_tensor_3d_panel(ax, cfg: FigureConfig, data: Dict):
    """
    Panel (c): 3D visualization of the strain tensor ellipsoid.
    
    Visualizes the chemical eigenstrain tensor as a 3D ellipsoid where the
    principal semi-axes correspond to the principal strain values.
    """
    # Get strain values at operating temperature (last index)
    eps_xx = data['epsilon_xx'][-1]
    eps_yy = data['epsilon_yy'][-1]
    eps_zz = data['epsilon_zz'][-1]
    
    # Scale factor for visualization
    scale = 3.0
    
    # Create ellipsoid
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    
    x = scale * eps_xx * np.outer(np.cos(u), np.sin(v))
    y = scale * eps_yy * np.outer(np.sin(u), np.sin(v))
    z = scale * eps_zz * np.outer(np.ones_like(u), np.cos(v))
    
    # Plot ellipsoid surface
    ax.plot_surface(x, y, z, alpha=0.6, color=cfg.COLORS['tensor_face'],
                    edgecolor=cfg.COLORS['tensor_edge'], linewidth=0.2)
    
    # Draw principal axes
    axis_scale = scale * max(eps_xx, eps_yy, eps_zz) * 1.3
    ax.quiver(0, 0, 0, axis_scale, 0, 0, color=cfg.COLORS['strain_xx'], 
              arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_scale, 0, color=cfg.COLORS['strain_yy'], 
              arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_scale, color=cfg.COLORS['strain_zz'], 
              arrow_length_ratio=0.1, linewidth=2)
    
    # Add axis labels using text (not text2D since these are positioned in 3D space)
    ax.text(axis_scale * 1.1, 0, 0, r'$\varepsilon_{xx}$', fontsize=cfg.LABEL_SIZE,
            color=cfg.COLORS['strain_xx'])
    ax.text(0, axis_scale * 1.1, 0, r'$\varepsilon_{yy}$', fontsize=cfg.LABEL_SIZE,
            color=cfg.COLORS['strain_yy'])
    ax.text(0, 0, axis_scale * 1.1, r'$\varepsilon_{zz}$', fontsize=cfg.LABEL_SIZE,
            color=cfg.COLORS['strain_zz'])
    
    # Formatting
    ax.set_xlabel('X', fontsize=cfg.LABEL_SIZE)
    ax.set_ylabel('Y', fontsize=cfg.LABEL_SIZE)
    ax.set_zlabel('Z', fontsize=cfg.LABEL_SIZE)
    ax.set_title('3D Strain Tensor\n(at 800°C)', fontsize=cfg.TITLE_SIZE, fontweight='bold')
    ax.tick_params(labelsize=cfg.TICK_SIZE - 1)
    
    # Set equal aspect ratio
    max_range = axis_scale
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Set view angle
    ax.view_init(elev=20, azim=45)


def create_strain_rate_panel(ax, cfg: FigureConfig, data: Dict):
    """
    Panel (d): Temperature derivative of strain components.
    
    Shows how the strain rate (dε/dT) varies with temperature, important for
    understanding the kinetics of the transformation.
    """
    T = data['T']
    
    # Convert to per-degree-C (multiply by 1e4 for better visualization)
    scale = 1e4
    
    ax.plot(T, data['d_epsilon_xx'] * scale, '-', color=cfg.COLORS['strain_xx'], 
            linewidth=2, label=r'$d\varepsilon_{xx}/dT$')
    ax.plot(T, data['d_epsilon_yy'] * scale, '--', color=cfg.COLORS['strain_yy'], 
            linewidth=2, label=r'$d\varepsilon_{yy}/dT$')
    ax.plot(T, data['d_epsilon_zz'] * scale, '-.', color=cfg.COLORS['strain_zz'], 
            linewidth=2, label=r'$d\varepsilon_{zz}/dT$')
    
    # Formatting
    ax.set_xlabel('Temperature (°C)', fontsize=cfg.LABEL_SIZE)
    ax.set_ylabel(r'Strain Rate ($\times 10^{-4}$ /°C)', fontsize=cfg.LABEL_SIZE)
    ax.set_title('Strain Rate vs Temperature', fontsize=cfg.TITLE_SIZE, fontweight='bold')
    ax.legend(loc='upper right', fontsize=cfg.LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=0.3, color=cfg.COLORS['grid'])
    ax.tick_params(labelsize=cfg.TICK_SIZE)
    ax.set_xlim([T.min(), T.max()])
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)


def create_figure1() -> plt.Figure:
    """
    Create the complete Figure 1 with all four panels.
    
    Returns:
        matplotlib Figure object
    """
    # Initialize configuration
    cfg = FigureConfig()
    
    # Generate data
    T_range = np.linspace(25, 800, 200)
    data = generate_chemical_expansion_data(T_range)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(cfg.FIG_WIDTH, cfg.FIG_HEIGHT), dpi=cfg.DPI)
    
    # Create grid spec for layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # Panel (a): Strain components
    print("Generating Panel (a): Strain Components...")
    ax_strain = fig.add_subplot(gs[0, 0])
    create_strain_components_panel(ax_strain, cfg, data)
    add_panel_label(ax_strain, 'a', cfg)
    
    # Panel (b): Volumetric expansion
    print("Generating Panel (b): Volumetric Expansion...")
    ax_vol = fig.add_subplot(gs[0, 1])
    create_volumetric_panel(ax_vol, cfg, data)
    add_panel_label(ax_vol, 'b', cfg)
    
    # Panel (c): 3D tensor visualization
    print("Generating Panel (c): 3D Tensor...")
    ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
    create_tensor_3d_panel(ax_3d, cfg, data)
    add_panel_label(ax_3d, 'c', cfg, x=-0.05, y=1.02)  # This now uses text2D for 3D axes
    
    # Panel (d): Strain rate
    print("Generating Panel (d): Strain Rate...")
    ax_rate = fig.add_subplot(gs[1, 1])
    create_strain_rate_panel(ax_rate, cfg, data)
    add_panel_label(ax_rate, 'd', cfg)
    
    # Add main title
    fig.suptitle('Chemical Expansion Tensor: Ni → NiO Phase Transformation\n'
                 'SOFC Ni-YSZ Redox Eigenstrain Analysis',
                 fontsize=cfg.TITLE_SIZE + 2, fontweight='bold', y=0.98)
    
    return fig


def main():
    """Main entry point for figure generation."""
    print("=" * 65)
    print("FIGURE 1: CHEMICAL EXPANSION TENSOR INPUT")
    print("=" * 65)
    print("Material:     Ni → NiO Phase Transformation")
    print("Application:  SOFC Ni-YSZ Redox Eigenstrain Analysis")
    print("Model:        Temperature-Dependent Tensor")
    print("Author:       georgegershom")
    print("Date:         2026-02-01")
    print("=" * 65 + "\n")
    
    fig = create_figure1()
    
    print("\n✨ Figure generation complete!\n")
    
    # Save figure
    fig.savefig('figure1_chemical_expansion_tensor.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig('figure1_chemical_expansion_tensor.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("Saved: figure1_chemical_expansion_tensor.png")
    print("Saved: figure1_chemical_expansion_tensor.pdf")
    
    plt.show()
    
    return fig


if __name__ == '__main__':
    main()
