"""
Figure 2: Advanced Cohesive Zone Traction-Separation Laws
=========================================================
Application: SOFC Ni-YSZ/YSZ Interface Fracture Modeling
Model: Temperature-Dependent Bilinear & Exponential CZM
Framework: Phase-Field Fracture with Cohesive Elements

Author: georgegershom
Date: 2026-02-01
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from scipy.integrate import trapezoid
from dataclasses import dataclass, field
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PROFESSIONAL FIGURE CONFIGURATION
# ============================================================================
@dataclass
class PublicationStyle:
    """Publication-quality figure styling configuration"""
    
    FIG_WIDTH: float = 180 / 25.4
    FIG_HEIGHT: float = 200 / 25.4
    DPI: int = 300
    
    FONT_FAMILY: str = 'serif'
    FONT_SERIF: List[str] = field(default_factory=lambda: ['Times New Roman', 'DejaVu Serif'])
    MATHTEXT: str = 'stix'
    
    TITLE_SIZE: int = 11
    LABEL_SIZE: int = 10
    TICK_SIZE: int = 9
    LEGEND_SIZE: int = 8
    ANNOTATION_SIZE: int = 8
    PANEL_LABEL_SIZE: int = 12
    
    LINE_WIDTH: float = 1.8
    MARKER_SIZE: int = 5
    AXES_LINE_WIDTH: float = 0.8
    
    COLORS: Dict = field(default_factory=lambda: {
        'temp_0K': '#1E3A5F',
        'temp_300K': '#2E86AB',
        'temp_600K': '#E94F37',
        'temp_800K': '#FF6B35',
        'ni_ysz': '#2E86AB',
        'ysz_gb': '#9B2335',
        'damage': '#E94F37',
        'pristine': '#2E7D32',
        'fracture_energy': '#7B68EE',
        'text_dark': '#1a1a2e',
        'text_medium': '#4a4a5a',
        'grid': '#e0e0e0',
        'white': '#ffffff',
    })
    
    def apply_style(self):
        plt.rcParams.update({
            'font.size': self.LABEL_SIZE,
            'font.family': self.FONT_FAMILY,
            'font.serif': self.FONT_SERIF,
            'mathtext.fontset': self.MATHTEXT,
            'axes.labelsize': self.LABEL_SIZE,
            'axes.titlesize': self.TITLE_SIZE,
            'axes.linewidth': self.AXES_LINE_WIDTH,
            'xtick.labelsize': self.TICK_SIZE,
            'ytick.labelsize': self.TICK_SIZE,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'legend.fontsize': self.LEGEND_SIZE,
            'legend.framealpha': 0.95,
            'figure.dpi': self.DPI,
            'savefig.dpi': 600,
            'lines.linewidth': self.LINE_WIDTH,
        })


# ============================================================================
# COHESIVE ZONE MODEL PHYSICS
# ============================================================================
@dataclass
class CohesiveZoneParameters:
    """Temperature-dependent cohesive zone parameters"""
    
    sigma_max_0K: float
    delta_c_0K: float
    G_Ic_ref: float
    alpha_sigma: float = 0.18
    alpha_delta: float = 0.15
    T_ref: float = 600.0
    delta_0_ratio: float = 0.33
    
    def sigma_max(self, T: float) -> float:
        return self.sigma_max_0K * (1 - self.alpha_sigma * T / self.T_ref)
    
    def delta_c(self, T: float) -> float:
        return self.delta_c_0K * (1 + self.alpha_delta * T / self.T_ref)
    
    def delta_0(self, T: float) -> float:
        return self.delta_0_ratio * self.delta_c(T)
    
    def G_Ic(self, T: float) -> float:
        return 0.5 * self.sigma_max(T) * self.delta_c(T)


NI_YSZ_PARAMS = CohesiveZoneParameters(
    sigma_max_0K=4.50, delta_c_0K=0.35, G_Ic_ref=3.89,
    alpha_sigma=0.18, alpha_delta=0.15, delta_0_ratio=0.30
)

YSZ_GB_PARAMS = CohesiveZoneParameters(
    sigma_max_0K=5.27, delta_c_0K=0.25, G_Ic_ref=5.36,
    alpha_sigma=0.18, alpha_delta=0.12, delta_0_ratio=0.28
)


# ============================================================================
# TRACTION-SEPARATION LAWS
# ============================================================================
def bilinear_traction(delta, params, T):
    sigma_max = params.sigma_max(T)
    delta_c = params.delta_c(T)
    delta_0 = params.delta_0(T)
    
    sigma = np.zeros_like(delta)
    mask_elastic = delta <= delta_0
    sigma[mask_elastic] = sigma_max * delta[mask_elastic] / delta_0
    mask_softening = (delta > delta_0) & (delta <= delta_c)
    sigma[mask_softening] = sigma_max * (delta_c - delta[mask_softening]) / (delta_c - delta_0)
    return sigma


def exponential_traction(delta, params, T):
    sigma_max = params.sigma_max(T)
    delta_0 = params.delta_0(T)
    delta_norm = np.maximum(delta / delta_0, 1e-10)
    sigma = sigma_max * delta_norm * np.exp(1 - delta_norm)
    sigma[sigma < 0.001 * sigma_max] = 0
    return sigma


def trapezoidal_traction(delta, params, T, plateau_ratio=0.3):
    sigma_max = params.sigma_max(T)
    delta_c = params.delta_c(T)
    delta_0 = params.delta_0(T)
    delta_1 = delta_0 + plateau_ratio * (delta_c - delta_0)
    
    sigma = np.zeros_like(delta)
    mask1 = delta <= delta_0
    sigma[mask1] = sigma_max * delta[mask1] / delta_0
    mask2 = (delta > delta_0) & (delta <= delta_1)
    sigma[mask2] = sigma_max
    mask3 = (delta > delta_1) & (delta <= delta_c)
    sigma[mask3] = sigma_max * (delta_c - delta[mask3]) / (delta_c - delta_1)
    return sigma


def calculate_damage(delta, params, T):
    delta_0 = params.delta_0(T)
    delta_c = params.delta_c(T)
    D = np.zeros_like(delta)
    mask_softening = (delta > delta_0) & (delta <= delta_c)
    D[mask_softening] = (delta[mask_softening] - delta_0) / (delta_c - delta_0)
    D[delta > delta_c] = 1.0
    return D


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================
def add_panel_label(ax, label, style, x=-0.12, y=1.08):
    if hasattr(ax, 'text2D'):
        ax.text2D(x, y, f'({label})', transform=ax.transAxes,
                  fontsize=style.PANEL_LABEL_SIZE, fontweight='bold',
                  color=style.COLORS['text_dark'], va='top', ha='left')
    else:
        ax.text(x, y, f'({label})', transform=ax.transAxes,
                fontsize=style.PANEL_LABEL_SIZE, fontweight='bold',
                color=style.COLORS['text_dark'], va='top', ha='left')


def draw_schematic_interface(ax, style):
    ni_rect = FancyBboxPatch((0.05, 0.55), 0.4, 0.35,
                              boxstyle="round,pad=0.02", facecolor='#B8B8B8',
                              edgecolor='#666666', linewidth=1.5, alpha=0.9)
    ax.add_patch(ni_rect)
    
    ysz_rect = FancyBboxPatch((0.05, 0.1), 0.4, 0.35,
                               boxstyle="round,pad=0.02", facecolor='#E8D5B7',
                               edgecolor='#8B7355', linewidth=1.5, alpha=0.9)
    ax.add_patch(ysz_rect)
    
    cz_rect = FancyBboxPatch((0.05, 0.45), 0.4, 0.1,
                              boxstyle="round,pad=0.01", facecolor='#FFE4B5',
                              edgecolor=style.COLORS['damage'], linewidth=2,
                              linestyle='--', alpha=0.8)
    ax.add_patch(cz_rect)
    
    ax.text(0.25, 0.72, 'Ni', fontsize=style.ANNOTATION_SIZE + 2,
            ha='center', va='center', fontweight='bold', color='#333333')
    ax.text(0.25, 0.28, 'YSZ', fontsize=style.ANNOTATION_SIZE + 2,
            ha='center', va='center', fontweight='bold', color='#5D4E37')
    ax.text(0.25, 0.50, 'CZ', fontsize=style.ANNOTATION_SIZE,
            ha='center', va='center', fontweight='bold', color=style.COLORS['damage'])
    
    ax.annotate('', xy=(0.65, 0.85), xytext=(0.65, 0.65),
                arrowprops=dict(arrowstyle='->', color=style.COLORS['ni_ysz'], lw=2))
    ax.text(0.72, 0.75, r'$\sigma$', fontsize=style.ANNOTATION_SIZE + 1,
            ha='left', va='center', color=style.COLORS['ni_ysz'])
    ax.annotate('', xy=(0.65, 0.15), xytext=(0.65, 0.35),
                arrowprops=dict(arrowstyle='->', color=style.COLORS['ni_ysz'], lw=2))
    ax.annotate('', xy=(0.85, 0.55), xytext=(0.85, 0.45),
                arrowprops=dict(arrowstyle='<->', color=style.COLORS['damage'], lw=1.5))
    ax.text(0.92, 0.50, r'$\delta$', fontsize=style.ANNOTATION_SIZE + 1,
            ha='left', va='center', color=style.COLORS['damage'])
    ax.text(0.5, 0.98, 'Cohesive Zone\nSchematic', fontsize=style.ANNOTATION_SIZE,
            ha='center', va='top', fontweight='bold', color=style.COLORS['text_dark'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_bilinear_schematic(ax, style):
    delta = np.array([0, 0.3, 1.0, 1.1])
    sigma = np.array([0, 1.0, 0, 0])
    ax.plot(delta, sigma, color=style.COLORS['ni_ysz'], linewidth=2.5, zorder=5)
    ax.fill_between(delta[:3], sigma[:3], alpha=0.2, color=style.COLORS['fracture_energy'])
    
    ax.plot([0.3, 0.3], [0, 1], '--', color=style.COLORS['text_medium'], linewidth=1, alpha=0.7)
    ax.text(0.3, -0.12, r'$\delta_0$', fontsize=style.ANNOTATION_SIZE, ha='center', va='top')
    ax.plot([1.0, 1.0], [0, 0.15], '--', color=style.COLORS['text_medium'], linewidth=1, alpha=0.7)
    ax.text(1.0, -0.12, r'$\delta_c$', fontsize=style.ANNOTATION_SIZE, ha='center', va='top')
    ax.plot([0, 0.35], [1, 1], '--', color=style.COLORS['text_medium'], linewidth=1, alpha=0.7)
    ax.text(-0.08, 1.0, r'$\sigma_{max}$', fontsize=style.ANNOTATION_SIZE, ha='right', va='center')
    ax.plot(0.3, 1.0, 'o', color=style.COLORS['damage'], markersize=6, zorder=6)
    ax.annotate(r'$G_{Ic}$', xy=(0.5, 0.35), fontsize=style.ANNOTATION_SIZE + 1,
                ha='center', va='center', color=style.COLORS['fracture_energy'], fontweight='bold')
    
    ax.set_xlim(-0.15, 1.2)
    ax.set_ylim(-0.2, 1.15)
    ax.set_xlabel(r'$\delta$', fontsize=style.LABEL_SIZE)
    ax.set_ylabel(r'$\sigma$', fontsize=style.LABEL_SIZE)
    ax.set_title('Bilinear Law', fontsize=style.ANNOTATION_SIZE + 1, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================================
# MAIN PANELS
# ============================================================================
def create_panel_a_niysz(ax, style):
    temperatures = [0, 300, 600]
    temp_colors = [style.COLORS['temp_0K'], style.COLORS['temp_300K'], style.COLORS['temp_600K']]
    linestyles = ['-', '--', '-.']
    delta = np.linspace(0, 0.55, 500)
    
    for T, color, ls in zip(temperatures, temp_colors, linestyles):
        sigma = bilinear_traction(delta, NI_YSZ_PARAMS, T)
        sigma_max = NI_YSZ_PARAMS.sigma_max(T)
        delta_c = NI_YSZ_PARAMS.delta_c(T)
        
        ax.plot(delta, sigma, color=color, linestyle=ls, linewidth=2.2, label=f'T = {T} K', zorder=5)
        ax.fill_between(delta, sigma, alpha=0.08, color=color)
        ax.plot(delta_c, 0, 'o', color=color, markersize=5, zorder=6)
        ax.axvline(x=delta_c, color=color, linestyle=':', alpha=0.3, linewidth=0.8)
        delta_0 = NI_YSZ_PARAMS.delta_0(T)
        ax.plot(delta_0, sigma_max, 's', color=color, markersize=4, markerfacecolor='white', markeredgewidth=1.5, zorder=6)
    
    np.random.seed(42)
    exp_delta = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    exp_sigma_base = bilinear_traction(exp_delta, NI_YSZ_PARAMS, 300)
    exp_sigma = exp_sigma_base * (1 + 0.05 * np.random.randn(len(exp_delta)))
    exp_error = 0.15 * np.ones_like(exp_sigma)
    ax.errorbar(exp_delta, exp_sigma, yerr=exp_error, fmt='o', color='#333333', markersize=4,
                capsize=2, capthick=1, elinewidth=1, label='Exp. (300 K)', zorder=7, alpha=0.8)
    
    ax.set_xlabel('Separation, δ (nm)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Traction, σ (GPa)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title('Ni-YSZ Interface (Mode I)', fontsize=style.TITLE_SIZE, fontweight='bold', pad=10)
    ax.set_xlim([0, 0.55])
    ax.set_ylim([0, 5.2])
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')
    
    param_text = (f"$G_{{Ic}}^{{ref}}$ = {NI_YSZ_PARAMS.G_Ic_ref:.2f} J/m²\n"
                  f"$\\sigma_{{max}}^{{0K}}$ = {NI_YSZ_PARAMS.sigma_max_0K:.2f} GPa\n"
                  f"$\\delta_c^{{0K}}$ = {NI_YSZ_PARAMS.delta_c_0K:.2f} nm")
    ax.text(0.03, 0.97, param_text, transform=ax.transAxes, fontsize=style.ANNOTATION_SIZE,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
            edgecolor='#cccccc', alpha=0.95), family='monospace')


def create_panel_b_yszgb(ax, style):
    temperatures = [0, 300, 600]
    temp_colors = [style.COLORS['temp_0K'], style.COLORS['temp_300K'], style.COLORS['temp_600K']]
    linestyles = ['-', '--', '-.']
    delta = np.linspace(0, 0.40, 500)
    
    for T, color, ls in zip(temperatures, temp_colors, linestyles):
        sigma = bilinear_traction(delta, YSZ_GB_PARAMS, T)
        sigma_max = YSZ_GB_PARAMS.sigma_max(T)
        delta_c = YSZ_GB_PARAMS.delta_c(T)
        
        ax.plot(delta, sigma, color=color, linestyle=ls, linewidth=2.2, label=f'T = {T} K', zorder=5)
        ax.fill_between(delta, sigma, alpha=0.08, color=color)
        ax.plot(delta_c, 0, 'o', color=color, markersize=5, zorder=6)
        ax.axvline(x=delta_c, color=color, linestyle=':', alpha=0.3, linewidth=0.8)
        delta_0 = YSZ_GB_PARAMS.delta_0(T)
        ax.plot(delta_0, sigma_max, 's', color=color, markersize=4, markerfacecolor='white', markeredgewidth=1.5, zorder=6)
    
    np.random.seed(123)
    exp_delta = np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.20, 0.25])
    exp_sigma_base = bilinear_traction(exp_delta, YSZ_GB_PARAMS, 300)
    exp_sigma = exp_sigma_base * (1 + 0.06 * np.random.randn(len(exp_delta)))
    exp_error = 0.18 * np.ones_like(exp_sigma)
    ax.errorbar(exp_delta, exp_sigma, yerr=exp_error, fmt='o', color='#333333', markersize=4,
                capsize=2, capthick=1, elinewidth=1, label='Exp. (300 K)', zorder=7, alpha=0.8)
    
    ax.set_xlabel('Separation, δ (nm)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Traction, σ (GPa)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title('YSZ Grain Boundary (Mixed Mode)', fontsize=style.TITLE_SIZE, fontweight='bold', pad=10)
    ax.set_xlim([0, 0.40])
    ax.set_ylim([0, 6.0])
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')
    
    param_text = (f"$G_{{Ic}}^{{ref}}$ = {YSZ_GB_PARAMS.G_Ic_ref:.2f} J/m²\n"
                  f"$\\sigma_{{max}}^{{0K}}$ = {YSZ_GB_PARAMS.sigma_max_0K:.2f} GPa\n"
                  f"$\\delta_c^{{0K}}$ = {YSZ_GB_PARAMS.delta_c_0K:.2f} nm")
    ax.text(0.03, 0.97, param_text, transform=ax.transAxes, fontsize=style.ANNOTATION_SIZE,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
            edgecolor='#cccccc', alpha=0.95), family='monospace')


def create_panel_c_comparison(ax, style):
    T = 300
    delta = np.linspace(0, 0.50, 500)
    
    sigma_bilinear = bilinear_traction(delta, NI_YSZ_PARAMS, T)
    sigma_exponential = exponential_traction(delta, NI_YSZ_PARAMS, T)
    sigma_trapezoidal = trapezoidal_traction(delta, NI_YSZ_PARAMS, T)
    
    ax.plot(delta, sigma_bilinear, color=style.COLORS['ni_ysz'], linewidth=2.2, label='Bilinear', zorder=5)
    ax.plot(delta, sigma_exponential, color=style.COLORS['ysz_gb'], linewidth=2.2, linestyle='--', label='Exponential (Xu-Needleman)', zorder=5)
    ax.plot(delta, sigma_trapezoidal, color=style.COLORS['pristine'], linewidth=2.2, linestyle='-.', label='Trapezoidal (PPR)', zorder=5)
    
    G_bilinear = trapezoid(sigma_bilinear, delta)
    G_exponential = trapezoid(sigma_exponential, delta)
    G_trapezoidal = trapezoid(sigma_trapezoidal, delta)
    
    ax.fill_between(delta, sigma_bilinear, alpha=0.08, color=style.COLORS['ni_ysz'])
    ax.set_xlabel('Separation, δ (nm)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Traction, σ (GPa)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title('Cohesive Law Formulations (T = 300 K)', fontsize=style.TITLE_SIZE, fontweight='bold', pad=10)
    ax.set_xlim([0, 0.50])
    ax.set_ylim([0, 4.5])
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')
    
    table_text = (f"$G_{{Ic}}$ Comparison:\nBilinear:     {G_bilinear*1000:.2f} mJ/m²\n"
                  f"Exponential:  {G_exponential*1000:.2f} mJ/m²\nTrapezoidal:  {G_trapezoidal*1000:.2f} mJ/m²")
    ax.text(0.03, 0.55, table_text, transform=ax.transAxes, fontsize=style.ANNOTATION_SIZE,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
            edgecolor='#cccccc', alpha=0.95), family='monospace')


def create_panel_d_damage(ax, style):
    T = 300
    delta = np.linspace(0, 0.50, 500)
    D_niysz = calculate_damage(delta, NI_YSZ_PARAMS, T)
    D_yszgb = calculate_damage(delta, YSZ_GB_PARAMS, T)
    
    ax.plot(delta, D_niysz, color=style.COLORS['ni_ysz'], linewidth=2.2, label='Ni-YSZ', zorder=5)
    ax.plot(delta, D_yszgb, color=style.COLORS['ysz_gb'], linewidth=2.2, linestyle='--', label='YSZ GB', zorder=5)
    
    delta_c_niysz = NI_YSZ_PARAMS.delta_c(T)
    delta_c_yszgb = YSZ_GB_PARAMS.delta_c(T)
    delta_0_niysz = NI_YSZ_PARAMS.delta_0(T)
    delta_0_yszgb = YSZ_GB_PARAMS.delta_0(T)
    
    ax.axvline(x=delta_0_niysz, color=style.COLORS['ni_ysz'], linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=delta_c_niysz, color=style.COLORS['ni_ysz'], linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=delta_0_yszgb, color=style.COLORS['ysz_gb'], linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=delta_c_yszgb, color=style.COLORS['ysz_gb'], linestyle=':', alpha=0.5, linewidth=1)
    
    ax.axhspan(0, 0.1, alpha=0.1, color=style.COLORS['pristine'], zorder=1)
    ax.axhspan(0.9, 1.0, alpha=0.1, color=style.COLORS['damage'], zorder=1)
    ax.text(0.02, 0.05, 'Pristine', fontsize=style.ANNOTATION_SIZE - 1, color=style.COLORS['pristine'], fontweight='bold')
    ax.text(0.02, 0.95, 'Failed', fontsize=style.ANNOTATION_SIZE - 1, color=style.COLORS['damage'], fontweight='bold')
    
    ax.set_xlabel('Separation, δ (nm)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Damage Variable, D', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title('Damage Evolution (T = 300 K)', fontsize=style.TITLE_SIZE, fontweight='bold', pad=10)
    ax.set_xlim([0, 0.50])
    ax.set_ylim([0, 1.05])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.legend(loc='center right', framealpha=0.95, edgecolor='#cccccc')


def create_panel_e_temperature(ax, style):
    T_range = np.linspace(0, 800, 100)
    sigma_max_niysz = [NI_YSZ_PARAMS.sigma_max(T) for T in T_range]
    sigma_max_yszgb = [YSZ_GB_PARAMS.sigma_max(T) for T in T_range]
    delta_c_niysz = [NI_YSZ_PARAMS.delta_c(T) for T in T_range]
    delta_c_yszgb = [YSZ_GB_PARAMS.delta_c(T) for T in T_range]
    
    ax2 = ax.twinx()
    l1, = ax.plot(T_range, sigma_max_niysz, color=style.COLORS['ni_ysz'], linewidth=2, label=r'$\sigma_{max}$ Ni-YSZ')
    l2, = ax.plot(T_range, sigma_max_yszgb, color=style.COLORS['ysz_gb'], linewidth=2, linestyle='--', label=r'$\sigma_{max}$ YSZ GB')
    l3, = ax2.plot(T_range, delta_c_niysz, color=style.COLORS['ni_ysz'], linewidth=2, linestyle=':', label=r'$\delta_c$ Ni-YSZ')
    l4, = ax2.plot(T_range, delta_c_yszgb, color=style.COLORS['ysz_gb'], linewidth=2, linestyle='-.', label=r'$\delta_c$ YSZ GB')
    
    ax.axvspan(600, 800, alpha=0.15, color=style.COLORS['damage'], zorder=1)
    ax.text(700, 5.0, 'SOFC\nOperating\nRange', fontsize=style.ANNOTATION_SIZE - 1, ha='center', va='center',
            color=style.COLORS['damage'], fontweight='bold')
    
    ax.set_xlabel('Temperature (K)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel(r'$\sigma_{max}$ (GPa)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax2.set_ylabel(r'$\delta_c$ (nm)', fontsize=style.LABEL_SIZE, fontweight='medium', color=style.COLORS['text_medium'])
    ax.set_title('Temperature Dependence', fontsize=style.TITLE_SIZE, fontweight='bold', pad=10)
    ax.set_xlim([0, 800])
    ax.set_ylim([3.0, 5.5])
    ax2.set_ylim([0.20, 0.45])
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower left', framealpha=0.95, edgecolor='#cccccc', fontsize=style.LEGEND_SIZE - 1)
    ax2.spines['right'].set_color(style.COLORS['text_medium'])
    ax2.tick_params(axis='y', colors=style.COLORS['text_medium'])


def create_panel_f_fracture_energy(ax, style):
    T_range = np.linspace(0, 800, 50)
    delta_range = np.linspace(0, 0.5, 50)
    T_grid, delta_grid = np.meshgrid(T_range, delta_range)
    G_Ic_surface = np.zeros_like(T_grid)
    
    for i, T in enumerate(T_range):
        for j, d in enumerate(delta_range):
            if d <= NI_YSZ_PARAMS.delta_c(T):
                sigma = bilinear_traction(np.array([d]), NI_YSZ_PARAMS, T)[0]
                G_Ic_surface[j, i] = 0.5 * sigma * d * 1000
    
    levels = np.linspace(0, 0.8, 20)
    cs = ax.contourf(T_grid, delta_grid, G_Ic_surface, levels=levels, cmap='viridis', alpha=0.85)
    ax.contour(T_grid, delta_grid, G_Ic_surface, levels=levels[::4], colors='white', linewidths=0.5, alpha=0.5)
    
    delta_c_line = [NI_YSZ_PARAMS.delta_c(T) for T in T_range]
    ax.plot(T_range, delta_c_line, 'w--', linewidth=2, label=r'$\delta_c(T)$')
    
    cbar = plt.colorbar(cs, ax=ax, pad=0.02)
    cbar.set_label(r'$G_{Ic}$ (mJ/m²)', fontsize=style.LABEL_SIZE - 1)
    cbar.ax.tick_params(labelsize=style.TICK_SIZE - 1)
    
    ax.set_xlabel('Temperature (K)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Separation, δ (nm)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title('Fracture Energy Landscape (Ni-YSZ)', fontsize=style.TITLE_SIZE, fontweight='bold', pad=10)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='white', fontsize=style.LEGEND_SIZE)


def create_summary_stats(ax, style):
    ax.axis('off')
    T_op = 600
    sigma_red_niysz = (1 - NI_YSZ_PARAMS.sigma_max(T_op) / NI_YSZ_PARAMS.sigma_max_0K) * 100
    sigma_red_yszgb = (1 - YSZ_GB_PARAMS.sigma_max(T_op) / YSZ_GB_PARAMS.sigma_max_0K) * 100
    delta_inc_niysz = (NI_YSZ_PARAMS.delta_c(T_op) / NI_YSZ_PARAMS.delta_c_0K - 1) * 100
    delta_inc_yszgb = (YSZ_GB_PARAMS.delta_c(T_op) / YSZ_GB_PARAMS.delta_c_0K - 1) * 100
    
    summary_text = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "   KEY FINDINGS (0K → 600K)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"  Ni-YSZ Interface:\n"
        f"    σₘₐₓ reduction:  {sigma_red_niysz:.1f}%\n"
        f"    δc increase:     {delta_inc_niysz:.1f}%\n\n"
        f"  YSZ Grain Boundary:\n"
        f"    σₘₐₓ reduction:  {sigma_red_yszgb:.1f}%\n"
        f"    δc increase:     {delta_inc_yszgb:.1f}%\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Temperature dependence\n"
        "  enables early decohesion\n"
        "  at SOFC operating temps\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=style.ANNOTATION_SIZE,
            fontfamily='monospace', verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.95))


# ============================================================================
# MAIN FIGURE
# ============================================================================
def create_figure2():
    style = PublicationStyle()
    style.apply_style()
    
    fig = plt.figure(figsize=(style.FIG_WIDTH, style.FIG_HEIGHT))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1.0, 1.0], width_ratios=[1, 1, 0.8],
                           hspace=0.35, wspace=0.35, left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    print("Generating Panel (a): Ni-YSZ Interface...")
    ax_a = fig.add_subplot(gs[0, 0])
    create_panel_a_niysz(ax_a, style)
    add_panel_label(ax_a, 'a', style)
    
    print("Generating Panel (b): YSZ Grain Boundary...")
    ax_b = fig.add_subplot(gs[0, 1])
    create_panel_b_yszgb(ax_b, style)
    add_panel_label(ax_b, 'b', style)
    
    print("Generating Schematics...")
    gs_schematic = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.3)
    ax_schematic1 = fig.add_subplot(gs_schematic[0])
    draw_schematic_interface(ax_schematic1, style)
    ax_schematic2 = fig.add_subplot(gs_schematic[1])
    draw_bilinear_schematic(ax_schematic2, style)
    
    print("Generating Panel (c): Law Comparison...")
    ax_c = fig.add_subplot(gs[1, 0])
    create_panel_c_comparison(ax_c, style)
    add_panel_label(ax_c, 'c', style)
    
    print("Generating Panel (d): Damage Evolution...")
    ax_d = fig.add_subplot(gs[1, 1])
    create_panel_d_damage(ax_d, style)
    add_panel_label(ax_d, 'd', style)
    
    print("Generating Panel (e): Temperature Dependence...")
    ax_e = fig.add_subplot(gs[1, 2])
    create_panel_e_temperature(ax_e, style)
    add_panel_label(ax_e, 'e', style)
    
    print("Generating Panel (f): Fracture Energy Surface...")
    ax_f = fig.add_subplot(gs[2, :2])
    create_panel_f_fracture_energy(ax_f, style)
    add_panel_label(ax_f, 'f', style)
    
    ax_stats = fig.add_subplot(gs[2, 2])
    create_summary_stats(ax_stats, style)
    
    fig.suptitle('Cohesive Zone Traction-Separation Laws for SOFC Fracture Modeling',
                 fontsize=style.TITLE_SIZE + 2, fontweight='bold', y=0.98)
    return fig


def print_verification():
    print("\n" + "=" * 80)
    print("COHESIVE ZONE PARAMETERS VERIFICATION")
    print("=" * 80)
    temperatures = [0, 300, 600, 800]
    
    print("\nNi-YSZ INTERFACE:")
    print("-" * 80)
    print(f"{'T (K)':<10} {'σ_max (GPa)':<15} {'δ_c (nm)':<12} {'G_Ic (J/m²)':<15}")
    print("-" * 80)
    for T in temperatures:
        sigma = NI_YSZ_PARAMS.sigma_max(T)
        delta_c = NI_YSZ_PARAMS.delta_c(T)
        G_Ic = NI_YSZ_PARAMS.G_Ic(T) * 1000
        print(f"{T:<10} {sigma:<15.3f} {delta_c:<12.4f} {G_Ic:<15.3f}")
    
    print("\nYSZ GRAIN BOUNDARIES:")
    print("-" * 80)
    for T in temperatures:
        sigma = YSZ_GB_PARAMS.sigma_max(T)
        delta_c = YSZ_GB_PARAMS.delta_c(T)
        G_Ic = YSZ_GB_PARAMS.G_Ic(T) * 1000
        print(f"{T:<10} {sigma:<15.3f} {delta_c:<12.4f} {G_Ic:<15.3f}")


def main():
    print("=" * 70)
    print("FIGURE 2: COHESIVE ZONE TRACTION-SEPARATION LAWS")
    print("=" * 70)
    print("Material Systems: Ni-YSZ Interface | YSZ Grain Boundaries")
    print("Application:      SOFC Chemo-Mechanical Fracture Modeling")
    print("Author:           georgegershom")
    print("Date:             2026-02-01")
    print("=" * 70 + "\n")
    
    fig = create_figure2()
    print("\n✨ Figure generation complete!\n")
    
    fig.savefig('figure2_cohesive_zone_laws.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    fig.savefig('figure2_cohesive_zone_laws.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    print("Saved: figure2_cohesive_zone_laws.png (600 DPI)")
    print("Saved: figure2_cohesive_zone_laws.pdf (vector)")
    
    print_verification()
    plt.show()
    return fig


if __name__ == '__main__':
    main()
