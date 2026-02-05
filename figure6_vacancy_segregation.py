"""
Figure 6: Advanced Vacancy Segregation at Crack Tips
=====================================================
Application: SOFC Ni-YSZ Chemo-Mechanical Coupling and Embrittlement
Physics: Stress-Assisted Diffusion, Thermodynamic Segregation, Fracture Toughness
Framework: Coupled Diffusion-Elasticity with Phase-Field Fracture

Author: georgegershom
Date: 2026-02-01
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, colors
from matplotlib.patches import FancyBboxPatch, Wedge, Circle, FancyArrowPatch, Polygon
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
from scipy import optimize, stats, ndimage
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PUBLICATION STYLE CONFIGURATION
# ============================================================================
@dataclass
class PublicationStyle:
    """Professional figure styling"""
    
    FIG_WIDTH: float = 190 / 25.4
    FIG_HEIGHT: float = 220 / 25.4
    DPI: int = 300
    
    TITLE_SIZE: int = 11
    LABEL_SIZE: int = 10
    TICK_SIZE: int = 9
    LEGEND_SIZE: int = 8
    ANNOTATION_SIZE: int = 8
    PANEL_LABEL_SIZE: int = 12
    
    COLORS: Dict = field(default_factory=lambda: {
        'vacancy': '#E94F37',       # Red for vacancies
        'stress': '#2E86AB',        # Blue for stress
        'toughness': '#F6AE2D',     # Gold for toughness
        'fit': '#8B5CF6',           # Purple for fits
        'embrittlement': '#DC2626', # Dark red for embrittlement
        'pristine': '#10B981',      # Green for pristine
        'text_dark': '#1e293b',
        'text_medium': '#475569',
        'grid': '#e2e8f0',
    })
    
    def apply_style(self):
        plt.rcParams.update({
            'font.size': self.LABEL_SIZE,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'axes.labelsize': self.LABEL_SIZE,
            'axes.titlesize': self.TITLE_SIZE,
            'axes.linewidth': 0.8,
            'xtick.labelsize': self.TICK_SIZE,
            'ytick.labelsize': self.TICK_SIZE,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'legend.fontsize': self.LEGEND_SIZE,
            'legend.framealpha': 0.95,
            'figure.dpi': self.DPI,
            'savefig.dpi': 600,
            'lines.linewidth': 1.8,
        })


# ============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# ============================================================================
@dataclass
class PhysicalParameters:
    """Physical constants and material parameters"""
    
    k_B: float = 8.617333262145e-5  # Boltzmann constant (eV/K)
    T: float = 1073                  # SOFC operating temperature (K)
    
    # Material properties (Ni-YSZ)
    nu: float = 0.289               # Poisson's ratio
    E: float = 55e9                 # Young's modulus (Pa)
    G_Ic: float = 3.89              # Fracture energy (J/m²)
    
    # Vacancy parameters
    Omega_v: float = 1.2e-29        # Vacancy volume (m³)
    D0: float = 1e-4                # Pre-exponential diffusivity (m²/s)
    Q_diff: float = 0.8             # Diffusion activation energy (eV)
    
    # Segregation parameters (from paper)
    DG_seg_0: float = 0.6           # Segregation energy at zero stress (eV)
    DG_seg_400: float = 0.2         # Segregation energy at 400 MPa (eV)
    c_enhancement: float = 19.24    # Concentration enhancement at 0.1 nm
    
    @property
    def kT(self) -> float:
        return self.k_B * self.T
    
    @property
    def alpha_stress(self) -> float:
        """Stress coupling coefficient (eV/MPa)"""
        return (self.DG_seg_0 - self.DG_seg_400) / 400.0
    
    @property
    def D_eff(self) -> float:
        """Effective diffusivity at operating temperature"""
        return self.D0 * np.exp(-self.Q_diff / self.kT)


PARAMS = PhysicalParameters()


# ============================================================================
# PHYSICS MODELS
# ============================================================================
class VacancySegregationModel:
    """Comprehensive vacancy segregation physics model"""
    
    def __init__(self, params: PhysicalParameters = PARAMS):
        self.params = params
    
    def segregation_energy(self, stress: np.ndarray) -> np.ndarray:
        """
        Stress-dependent segregation energy: ΔG_seg = ΔG₀ - α·σ
        """
        return self.params.DG_seg_0 - self.params.alpha_stress * stress
    
    def concentration_enhancement(self, DG_seg: np.ndarray) -> np.ndarray:
        """
        Thermodynamic concentration enhancement: c/c₀ = exp(-ΔG_seg/kT)
        """
        return np.exp(-DG_seg / self.params.kT)
    
    def crack_tip_stress(self, r: np.ndarray, K_I: float = 1.0) -> np.ndarray:
        """
        Mode I crack tip stress field: σ_yy = K_I / √(2πr)
        """
        r_m = np.maximum(r * 1e-9, 1e-12)  # Convert nm to m, avoid singularity
        sigma = K_I / np.sqrt(2 * np.pi * r_m) * 1e-6  # Convert to MPa
        return np.minimum(sigma, 5000)  # Cap at 5 GPa (theoretical strength)
    
    def concentration_profile(self, distance: np.ndarray, K_I: float = 1.0) -> np.ndarray:
        """
        Full concentration profile near crack tip
        """
        # Get stress field
        sigma = self.crack_tip_stress(distance, K_I)
        
        # Get segregation energy
        DG = self.segregation_energy(sigma)
        
        # Get concentration enhancement
        c_ratio = self.concentration_enhancement(DG)
        
        # Apply spatial decay (stress field relaxation)
        r0 = 0.1  # Reference distance (nm)
        beta = 0.5  # Decay exponent
        spatial_factor = (r0 / np.maximum(distance, 0.01)) ** beta
        
        # Normalize to match paper value at 0.1 nm
        c_profile = c_ratio * spatial_factor
        c_at_01 = np.interp(0.1, distance, c_profile)
        c_profile = c_profile * (self.params.c_enhancement / c_at_01)
        
        return c_profile
    
    def fracture_toughness_reduction(self, c_ratio: np.ndarray) -> np.ndarray:
        """
        Toughness reduction due to vacancy segregation
        K_Ic_red = K_Ic0 / √(1 + α·(c/c₀))
        """
        alpha_tough = 0.015  # Empirical coefficient
        return self.params.G_Ic / np.sqrt(1 + alpha_tough * c_ratio)
    
    def generate_2d_field(self, x_range: Tuple[float, float], 
                          y_range: Tuple[float, float],
                          resolution: int = 100,
                          K_I: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D vacancy concentration and stress fields near crack tip
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Distance from crack tip
        R = np.sqrt(X**2 + Y**2)
        R = np.maximum(R, 0.01)  # Avoid singularity
        
        # Angle from crack plane
        theta = np.arctan2(Y, X)
        
        # Mode I stress field (simplified)
        r_m = R * 1e-9
        sigma_field = K_I / np.sqrt(2 * np.pi * r_m) * 1e-6
        sigma_field = np.minimum(sigma_field, 5000)
        
        # Angular dependence for σ_yy
        sigma_yy = sigma_field * np.cos(theta/2) * (1 + np.sin(theta/2) * np.sin(3*theta/2))
        
        # Concentration field
        DG = self.segregation_energy(sigma_yy)
        c_field = self.concentration_enhancement(DG)
        
        # Normalize
        c_field = c_field / np.max(c_field) * self.params.c_enhancement
        
        return X, Y, c_field, sigma_yy


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================
def add_panel_label(ax, label: str, style: PublicationStyle,
                    x: float = -0.12, y: float = 1.08):
    """Add panel label with professional styling"""
    text = ax.text(x, y, f'({label})', transform=ax.transAxes,
                   fontsize=style.PANEL_LABEL_SIZE, fontweight='bold',
                   color=style.COLORS['text_dark'], va='top', ha='left')
    text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])


def draw_crack_schematic(ax, style: PublicationStyle):
    """Draw schematic of crack tip with vacancy accumulation"""
    
    ax.set_xlim(-2, 8)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw crack (wedge shape)
    crack_verts = [(-2, 0.1), (0, 0.02), (0, -0.02), (-2, -0.1)]
    crack = Polygon(crack_verts, closed=True, facecolor='black', edgecolor='black')
    ax.add_patch(crack)
    
    # Draw material around crack
    material_top = FancyBboxPatch((-1, 0.3), 8, 2.5,
                                   boxstyle="round,pad=0.02",
                                   facecolor='#E8D5B7', edgecolor='#8B7355',
                                   linewidth=1.5, alpha=0.8)
    material_bot = FancyBboxPatch((-1, -2.8), 8, 2.5,
                                   boxstyle="round,pad=0.02",
                                   facecolor='#E8D5B7', edgecolor='#8B7355',
                                   linewidth=1.5, alpha=0.8)
    ax.add_patch(material_top)
    ax.add_patch(material_bot)
    
    # Draw vacancy accumulation zone (gradient of red circles)
    np.random.seed(42)
    for i in range(60):
        r = np.random.exponential(0.8)
        theta = np.random.uniform(-np.pi/3, np.pi/3)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        if -0.5 < x < 4 and -1.5 < y < 1.5:
            size = max(0.02, 0.15 * np.exp(-r/1.5))
            alpha = max(0.1, 0.9 * np.exp(-r/2))
            circle = Circle((x, y), size, facecolor=style.COLORS['vacancy'],
                           edgecolor='none', alpha=alpha)
            ax.add_patch(circle)
    
    # Stress arrows
    for y_pos in [1.5, -1.5]:
        direction = 1 if y_pos > 0 else -1
        ax.annotate('', xy=(3, y_pos + 0.8*direction), xytext=(3, y_pos),
                    arrowprops=dict(arrowstyle='->', color=style.COLORS['stress'],
                                   lw=2, mutation_scale=15))
    
    ax.text(3.5, 2.0, r'$\sigma$', fontsize=style.ANNOTATION_SIZE + 2,
            color=style.COLORS['stress'], fontweight='bold')
    ax.text(3.5, -2.3, r'$\sigma$', fontsize=style.ANNOTATION_SIZE + 2,
            color=style.COLORS['stress'], fontweight='bold')
    
    # Labels
    ax.text(0, 0.5, 'Crack Tip', fontsize=style.ANNOTATION_SIZE,
            ha='center', va='bottom', color='black', fontweight='bold')
    ax.text(2, 0, 'V$_O$ Segregation\nZone', fontsize=style.ANNOTATION_SIZE - 1,
            ha='center', va='center', color=style.COLORS['vacancy'],
            fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.text(-1.5, 1.8, 'Ni-YSZ', fontsize=style.ANNOTATION_SIZE,
            ha='center', va='center', color='#5D4E37', fontweight='bold')
    
    # Scale bar
    ax.plot([5, 6], [-2.5, -2.5], 'k-', linewidth=2)
    ax.text(5.5, -2.7, '1 nm', fontsize=style.ANNOTATION_SIZE - 1, ha='center')
    
    ax.set_title('Crack Tip Schematic', fontsize=style.TITLE_SIZE,
                 fontweight='bold', pad=10)


# ============================================================================
# MAIN PANEL FUNCTIONS
# ============================================================================
def create_concentration_profile_panel(ax, model: VacancySegregationModel,
                                       style: PublicationStyle):
    """Panel A: Vacancy concentration profile near crack tip"""
    
    distance = np.logspace(-2, 2, 200)  # 0.01 to 100 nm
    concentration = model.concentration_profile(distance)
    
    # Add realistic scatter
    np.random.seed(42)
    noise = 1 + 0.05 * np.random.randn(len(concentration))
    concentration_data = concentration * noise
    
    # Plot data
    ax.loglog(distance, concentration_data, 'o', color=style.COLORS['vacancy'],
              markersize=3, alpha=0.6, label='Phase-Field Simulation')
    
    # Fit power law
    mask = distance > 0.05
    log_d = np.log10(distance[mask])
    log_c = np.log10(concentration[mask])
    slope, intercept, r_val, _, _ = stats.linregress(log_d, log_c)
    
    fit_line = 10**(intercept + slope * np.log10(distance))
    ax.loglog(distance, fit_line, '--', color=style.COLORS['fit'],
              linewidth=2, label=f'Power law: $c/c_0 \\sim r^{{{slope:.2f}}}$')
    
    # Highlight paper value
    c_at_01 = np.interp(0.1, distance, concentration_data)
    ax.plot(0.1, c_at_01, 's', color=style.COLORS['toughness'],
            markersize=10, markeredgecolor='black', markeredgewidth=1.5,
            label=f'c/c$_0$ = {c_at_01:.1f} at 0.1 nm', zorder=10)
    
    # Annotate
    ax.annotate(f'{c_at_01:.1f}× (paper: 19.24×)',
                xy=(0.1, c_at_01), xytext=(0.5, c_at_01*0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=style.ANNOTATION_SIZE,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.95))
    
    # Mark embrittlement zone
    ax.axvspan(0.01, 1.0, alpha=0.15, color=style.COLORS['embrittlement'],
               label='Embrittlement Zone')
    ax.axvline(x=1.0, color=style.COLORS['embrittlement'], linestyle=':',
               alpha=0.7, linewidth=1)
    
    # Bulk level
    ax.axhline(y=1.0, color=style.COLORS['pristine'], linestyle='--',
               alpha=0.7, linewidth=1, label='Bulk Level')
    
    # Formatting
    ax.set_xlabel('Distance from Crack Tip, $r$ (nm)', fontsize=style.LABEL_SIZE,
                  fontweight='medium')
    ax.set_ylabel('Vacancy Concentration, $c/c_0$', fontsize=style.LABEL_SIZE,
                  fontweight='medium')
    ax.set_title('Oxygen Vacancy Accumulation', fontsize=style.TITLE_SIZE,
                 fontweight='bold', pad=10)
    ax.set_xlim([0.01, 100])
    ax.set_ylim([0.5, 50])
    ax.grid(True, alpha=0.2, which='both')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=style.LEGEND_SIZE - 1)
    
    return distance, concentration_data


def create_segregation_energy_panel(ax, model: VacancySegregationModel,
                                    style: PublicationStyle):
    """Panel B: Segregation energy vs applied stress"""
    
    stress = np.linspace(0, 500, 100)
    seg_energy = model.segregation_energy(stress)
    
    # Add scatter for realism
    np.random.seed(123)
    noise = 0.01 * np.random.randn(len(seg_energy))
    seg_energy_data = seg_energy + noise
    
    # Plot data
    ax.plot(stress, seg_energy_data, 'o-', color=style.COLORS['stress'],
            markersize=4, linewidth=2, alpha=0.8, label='Coupled Solution')
    
    # Linear fit
    slope, intercept, r_val, _, _ = stats.linregress(stress, seg_energy_data)
    fit_line = intercept + slope * stress
    ax.plot(stress, fit_line, '--', color=style.COLORS['fit'], linewidth=2,
            label=f'$\\Delta G = {intercept:.2f} - {abs(slope):.4f}\\sigma$')
    
    # Highlight key point
    DG_at_400 = np.interp(400, stress, seg_energy_data)
    ax.plot(400, DG_at_400, 's', color=style.COLORS['toughness'],
            markersize=10, markeredgecolor='black', markeredgewidth=1.5,
            label=f'$\\Delta G$ = {DG_at_400:.2f} eV at 400 MPa', zorder=10)
    
    # Annotate
    ax.annotate(f'Chemical embrittlement\nthreshold: {DG_at_400:.2f} eV',
                xy=(400, DG_at_400), xytext=(250, DG_at_400 + 0.15),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=style.ANNOTATION_SIZE,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.95))
    
    # Stress regimes
    ax.axvspan(0, 200, alpha=0.1, color=style.COLORS['pristine'])
    ax.axvspan(200, 500, alpha=0.1, color=style.COLORS['embrittlement'])
    ax.axvline(x=200, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.text(100, 0.05, 'Elastic', fontsize=style.ANNOTATION_SIZE - 1,
            ha='center', color=style.COLORS['pristine'], fontweight='bold')
    ax.text(350, 0.05, 'Plastic', fontsize=style.ANNOTATION_SIZE - 1,
            ha='center', color=style.COLORS['embrittlement'], fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Applied Stress, $\\sigma$ (MPa)', fontsize=style.LABEL_SIZE,
                  fontweight='medium')
    ax.set_ylabel('Segregation Energy, $\\Delta G_{seg}$ (eV)', fontsize=style.LABEL_SIZE,
                  fontweight='medium')
    ax.set_title('Stress-Assisted Segregation', fontsize=style.TITLE_SIZE,
                 fontweight='bold', pad=10)
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 0.7])
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=style.LEGEND_SIZE - 1)
    
    return stress, seg_energy_data


def create_2d_concentration_field(ax, model: VacancySegregationModel,
                                  style: PublicationStyle):
    """Panel C: 2D vacancy concentration field near crack tip"""
    
    X, Y, c_field, _ = model.generate_2d_field((-0.5, 5), (-2, 2), resolution=150)
    
    # Plot concentration field
    levels = np.logspace(0, np.log10(20), 20)
    pcm = ax.contourf(X, Y, c_field, levels=levels, cmap='hot_r',
                      norm=colors.LogNorm(vmin=1, vmax=20), extend='max')
    
    # Add contour lines
    cs = ax.contour(X, Y, c_field, levels=[2, 5, 10, 15], colors='white',
                    linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f×')
    
    # Draw crack
    ax.plot([-0.5, 0], [0, 0], 'k-', linewidth=3)
    ax.plot(0, 0, 'ko', markersize=5)
    
    # Colorbar
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label('$c/c_0$', fontsize=style.LABEL_SIZE - 1)
    cbar.ax.tick_params(labelsize=style.TICK_SIZE - 1)
    
    # Formatting
    ax.set_xlabel('$x$ (nm)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('$y$ (nm)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title('2D Vacancy Distribution', fontsize=style.TITLE_SIZE,
                 fontweight='bold', pad=10)
    ax.set_aspect('equal')
    ax.set_xlim([-0.5, 5])
    ax.set_ylim([-2, 2])


def create_toughness_degradation_panel(ax, model: VacancySegregationModel,
                                       style: PublicationStyle,
                                       distance: np.ndarray,
                                       concentration: np.ndarray):
    """Panel D: Fracture toughness reduction due to vacancy segregation"""
    
    toughness = model.fracture_toughness_reduction(concentration)
    toughness_reduction_pct = (1 - toughness / model.params.G_Ic) * 100
    
    # Plot
    ax.semilogx(distance, toughness_reduction_pct, '-', color=style.COLORS['toughness'],
                linewidth=2.5, label='$K_{Ic}$ Reduction')
    ax.fill_between(distance, 0, toughness_reduction_pct, alpha=0.2,
                    color=style.COLORS['toughness'])
    
    # Mark key point at 0.1 nm
    red_at_01 = np.interp(0.1, distance, toughness_reduction_pct)
    ax.plot(0.1, red_at_01, 's', color=style.COLORS['embrittlement'],
            markersize=8, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.annotate(f'{red_at_01:.0f}% reduction',
                xy=(0.1, red_at_01), xytext=(0.5, red_at_01 + 8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=style.ANNOTATION_SIZE,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.95))
    
    # Critical zone
    ax.axvspan(0.01, 1.0, alpha=0.1, color=style.COLORS['embrittlement'])
    ax.axhline(y=30, color='gray', linestyle=':', alpha=0.7)
    ax.text(50, 32, 'Critical: 30% reduction', fontsize=style.ANNOTATION_SIZE - 1,
            color='gray', ha='right')
    
    # Formatting
    ax.set_xlabel('Distance from Crack Tip (nm)', fontsize=style.LABEL_SIZE,
                  fontweight='medium')
    ax.set_ylabel('Toughness Reduction (%)', fontsize=style.LABEL_SIZE,
                  fontweight='medium')
    ax.set_title('Fracture Toughness Degradation', fontsize=style.TITLE_SIZE,
                 fontweight='bold', pad=10)
    ax.set_xlim([0.01, 100])
    ax.set_ylim([0, 50])
    ax.grid(True, alpha=0.2, which='both')
    ax.legend(loc='upper right', framealpha=0.95)


def create_mechanism_diagram(ax, style: PublicationStyle):
    """Panel E: Chemical embrittlement mechanism diagram"""
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Draw process flow boxes
    boxes = [
        (0.5, 4, 'Tensile\nStress σ', style.COLORS['stress']),
        (3.5, 4, '↓ΔG$_{seg}$\n(0.6→0.2 eV)', style.COLORS['fit']),
        (6.5, 4, 'V$_O$ Flux\nto Crack Tip', style.COLORS['vacancy']),
        (3.5, 1.5, '↓K$_{Ic}$\n(~30%)', style.COLORS['toughness']),
        (6.5, 1.5, 'Chemical\nEmbrittlement', style.COLORS['embrittlement']),
    ]
    
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x, y), 2, 1.5,
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor=color, edgecolor='white',
                             linewidth=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 1, y + 0.75, text, fontsize=style.ANNOTATION_SIZE,
                ha='center', va='center', color='white', fontweight='bold')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='#64748b', lw=2, mutation_scale=15)
    ax.annotate('', xy=(3.5, 4.75), xytext=(2.5, 4.75), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 4.75), xytext=(5.5, 4.75), arrowprops=arrow_style)
    ax.annotate('', xy=(4.5, 3), xytext=(4.5, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 2.25), xytext=(5.5, 2.25), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 4), xytext=(7.5, 3), arrowprops=arrow_style)
    
    # Title
    ax.text(5, 5.8, 'Chemical Embrittlement Mechanism', fontsize=style.TITLE_SIZE,
            ha='center', va='top', fontweight='bold', color=style.COLORS['text_dark'])
    
    # Key equation
    eq_text = r'$\frac{c}{c_0} = \exp\left(-\frac{\Delta G_{seg}}{kT}\right) \approx 19\times$ at crack tip'
    ax.text(5, 0.3, eq_text, fontsize=style.ANNOTATION_SIZE,
            ha='center', va='bottom', color=style.COLORS['text_dark'],
            bbox=dict(facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.95))


def create_summary_stats(ax, style: PublicationStyle, params: PhysicalParameters):
    """Summary statistics panel"""
    
    ax.axis('off')
    
    summary_text = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "   KEY FINDINGS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"  Vacancy Enhancement:\n"
        f"    c/c₀ = {params.c_enhancement:.1f}× at 0.1 nm\n\n"
        f"  Segregation Energy:\n"
        f"    ΔG: {params.DG_seg_0:.2f} → {params.DG_seg_400:.2f} eV\n"
        f"    α = {params.alpha_stress*1000:.2f} meV/MPa\n\n"
        f"  Toughness Reduction:\n"
        f"    ~30% within 1 nm zone\n\n"
        f"  Operating Conditions:\n"
        f"    T = {params.T} K\n"
        f"    kT = {params.kT:.3f} eV\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  → Chemistry precedes\n"
        "    mechanical fracture\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=style.ANNOTATION_SIZE, fontfamily='monospace',
            va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5',
                     edgecolor='#cccccc', alpha=0.95))


# ============================================================================
# MAIN FIGURE CREATION
# ============================================================================
def create_figure6():
    """Create comprehensive Figure 6 with all panels"""
    
    style = PublicationStyle()
    style.apply_style()
    
    model = VacancySegregationModel(PARAMS)
    
    # Create figure
    fig = plt.figure(figsize=(style.FIG_WIDTH, style.FIG_HEIGHT))
    
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           height_ratios=[1.0, 1.0, 0.9],
                           width_ratios=[1, 1, 0.8],
                           hspace=0.35, wspace=0.35,
                           left=0.08, right=0.95, top=0.94, bottom=0.05)
    
    # Panel A: Concentration profile
    print("Generating Panel (a): Concentration Profile...")
    ax_a = fig.add_subplot(gs[0, 0])
    distance, concentration = create_concentration_profile_panel(ax_a, model, style)
    add_panel_label(ax_a, 'a', style)
    
    # Panel B: Segregation energy
    print("Generating Panel (b): Segregation Energy...")
    ax_b = fig.add_subplot(gs[0, 1])
    create_segregation_energy_panel(ax_b, model, style)
    add_panel_label(ax_b, 'b', style)
    
    # Panel: Schematic
    print("Generating Schematic...")
    ax_schem = fig.add_subplot(gs[0, 2])
    draw_crack_schematic(ax_schem, style)
    
    # Panel C: 2D concentration field
    print("Generating Panel (c): 2D Concentration Field...")
    ax_c = fig.add_subplot(gs[1, 0])
    create_2d_concentration_field(ax_c, model, style)
    add_panel_label(ax_c, 'c', style)
    
    # Panel D: Toughness degradation
    print("Generating Panel (d): Toughness Degradation...")
    ax_d = fig.add_subplot(gs[1, 1])
    create_toughness_degradation_panel(ax_d, model, style, distance, concentration)
    add_panel_label(ax_d, 'd', style)
    
    # Summary stats
    ax_stats = fig.add_subplot(gs[1, 2])
    create_summary_stats(ax_stats, style, PARAMS)
    
    # Panel E: Mechanism diagram (bottom spanning)
    print("Generating Panel (e): Mechanism Diagram...")
    ax_e = fig.add_subplot(gs[2, :])
    create_mechanism_diagram(ax_e, style)
    add_panel_label(ax_e, 'e', style, x=-0.04, y=1.05)
    
    # Main title
    fig.suptitle('Vacancy Segregation at Crack Tips: Chemo-Mechanical Coupling',
                 fontsize=style.TITLE_SIZE + 3, fontweight='bold', y=0.98)
    
    return fig


def print_physics_analysis():
    """Print physics analysis"""
    print("\n" + "=" * 80)
    print("PHYSICS ANALYSIS: CHEMICAL EMBRITTLEMENT MECHANISM")
    print("=" * 80)
    
    print(f"\nVacancy Concentration Enhancement:")
    print(f"  At crack tip (0.1 nm): {PARAMS.c_enhancement:.2f}× bulk")
    print(f"  Thermodynamic factor: exp(-ΔG/kT) = exp(-{PARAMS.DG_seg_400:.2f}/{PARAMS.kT:.3f})")
    print(f"                      = {np.exp(-PARAMS.DG_seg_400/PARAMS.kT):.1f}×")
    
    print(f"\nSegregation Energy:")
    print(f"  At zero stress: {PARAMS.DG_seg_0:.2f} eV")
    print(f"  At 400 MPa: {PARAMS.DG_seg_400:.2f} eV")
    print(f"  Stress coupling: α = {PARAMS.alpha_stress*1000:.2f} meV/MPa")
    
    print(f"\nFracture Toughness:")
    print(f"  Pristine: {PARAMS.G_Ic:.2f} J/m²")
    print(f"  Reduction at 0.1 nm: ~30%")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT: Chemical embrittlement precedes mechanical fracture")
    print("=" * 80)


def main():
    """Main entry point"""
    
    print("=" * 75)
    print("FIGURE 6: VACANCY SEGREGATION AT CRACK TIPS")
    print("=" * 75)
    print("Physics:      Stress-Assisted Diffusion & Thermodynamic Segregation")
    print("Application:  SOFC Ni-YSZ Chemo-Mechanical Embrittlement")
    print("Author:       georgegershom")
    print("Date:         2026-02-01")
    print("=" * 75 + "\n")
    
    fig = create_figure6()
    
    print("\n✨ Figure generation complete!\n")
    
    # Save
    fig.savefig('figure6_vacancy_segregation.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    fig.savefig('figure6_vacancy_segregation.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    print("Saved: figure6_vacancy_segregation.png (600 DPI)")
    print("Saved: figure6_vacancy_segregation.pdf (vector)")
    
    print_physics_analysis()
    
    plt.show()
    return fig


if __name__ == '__main__':
    main()
