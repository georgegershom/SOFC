#!/usr/bin/env python3
"""
Comprehensive Visualization Script for Microstructural Analysis
Phase 3: PhD-Level Figures for Publication

Generates publication-quality figures for:
- SEM ITZ analysis
- XRD phase evolution
- TGA/DTA thermal decomposition
- Micro-CT porosity and crack networks

Author: Generated for PhD Research
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Color palette
COLORS = {
    0: '#2E86AB',   # Blue - control
    5: '#06A77D',   # Green
    10: '#F5B700',  # Yellow/Gold
    15: '#D62828',  # Red
    20: '#7209B7'   # Purple
}

class MicrostructuralVisualizer:
    """Publication-quality visualization suite"""
    
    def __init__(self, data_dir='..', output_dir='../figures'):
        """Initialize visualizer"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.load_data()
        
    def load_data(self):
        """Load all datasets"""
        print("Loading data for visualization...")
        self.sem = pd.read_csv(f'{self.data_dir}/sem_data/sem_itz_analysis.csv')
        self.xrd = pd.read_csv(f'{self.data_dir}/xrd_data/xrd_phase_composition.csv')
        self.tga = pd.read_csv(f'{self.data_dir}/tga_dta_data/tga_mass_loss_analysis.csv')
        self.dta = pd.read_csv(f'{self.data_dir}/tga_dta_data/dta_thermal_events.csv')
        self.porosity = pd.read_csv(f'{self.data_dir}/microct_data/microct_porosity_analysis.csv')
        self.cracks = pd.read_csv(f'{self.data_dir}/microct_data/microct_crack_network_analysis.csv')
        print("✓ Data loaded successfully\n")
        
    def plot_sem_itz_evolution(self):
        """Figure 1: SEM ITZ thickness and microcracking evolution"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: ITZ thickness vs temperature for rubber-paste interface
        ax1 = fig.add_subplot(gs[0, 0])
        rubber_itz = self.sem[self.sem['itz_type'] == 'rubber_paste']
        for rubber in [5, 10, 15, 20]:
            data = rubber_itz[rubber_itz['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['itz_thickness_um'].mean()
            ax1.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('ITZ Thickness (μm)')
        ax1.set_title('(a) ITZ Thickness Evolution at Rubber-Paste Interface')
        ax1.legend(frameon=False)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Plot 2: Microcrack density
        ax2 = fig.add_subplot(gs[0, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.sem[self.sem['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['microcrack_density_per_mm2'].mean()
            ax2.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Microcrack Density (cracks/mm²)')
        ax2.set_title('(b) Microcrack Density Development')
        ax2.legend(frameon=False, ncol=2)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_yscale('log')
        
        # Plot 3: Rubber degradation score
        ax3 = fig.add_subplot(gs[1, 0])
        for rubber in [5, 10, 15, 20]:
            data = self.sem[self.sem['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['rubber_degradation_score'].mean()
            ax3.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Rubber Degradation Score (0-10)')
        ax3.set_title('(c) Rubber Particle Degradation Progression')
        ax3.legend(frameon=False)
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.set_ylim(-0.5, 10.5)
        
        # Plot 4: Paste morphology score
        ax4 = fig.add_subplot(gs[1, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.sem[self.sem['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['paste_morphology_score'].mean()
            ax4.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Paste Morphology Score (0-10)')
        ax4.set_title('(d) Paste Quality Degradation')
        ax4.legend(frameon=False, ncol=2)
        ax4.grid(alpha=0.3, linestyle='--')
        
        plt.suptitle('Figure 1: SEM Analysis - ITZ Degradation and Microstructural Changes', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/fig1_sem_itz_evolution.png', bbox_inches='tight')
        print("✓ Figure 1 saved: SEM ITZ Evolution")
        plt.close()
        
    def plot_xrd_phase_evolution(self):
        """Figure 2: XRD phase composition evolution"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Portlandite consumption
        ax1 = fig.add_subplot(gs[0, 0])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.xrd[self.xrd['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['portlandite_wt_pct'].mean()
            ax1.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Ca(OH)₂ Content (wt%)')
        ax1.set_title('(a) Portlandite Consumption')
        ax1.legend(frameon=False, ncol=2)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Free lime formation
        ax2 = fig.add_subplot(gs[0, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.xrd[self.xrd['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['free_lime_CaO_wt_pct'].mean()
            ax2.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Free CaO Content (wt%)')
        ax2.set_title('(b) Free Lime Formation')
        ax2.legend(frameon=False, ncol=2)
        ax2.grid(alpha=0.3, linestyle='--')
        
        # Plot 3: C-S-H amorphous content
        ax3 = fig.add_subplot(gs[1, 0])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.xrd[self.xrd['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['CSH_amorphous_wt_pct'].mean()
            ax3.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('C-S-H Content (wt%)')
        ax3.set_title('(c) C-S-H Gel Degradation')
        ax3.legend(frameon=False, ncol=2)
        ax3.grid(alpha=0.3, linestyle='--')
        
        # Plot 4: Crystallinity index
        ax4 = fig.add_subplot(gs[1, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.xrd[self.xrd['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['crystallinity_index'].mean()
            ax4.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Crystallinity Index')
        ax4.set_title('(d) Overall Crystallinity Evolution')
        ax4.legend(frameon=False, ncol=2)
        ax4.grid(alpha=0.3, linestyle='--')
        
        plt.suptitle('Figure 2: XRD Analysis - Phase Composition Evolution', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/fig2_xrd_phase_evolution.png', bbox_inches='tight')
        print("✓ Figure 2 saved: XRD Phase Evolution")
        plt.close()
        
    def plot_tga_mass_loss_mechanisms(self):
        """Figure 3: TGA/DTA mass loss mechanisms"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Stacked mass loss for 0% rubber
        ax1 = fig.add_subplot(gs[0, 0])
        rubber_0 = self.tga[self.tga['rubber_content_pct'] == 0]
        temps = sorted(rubber_0['temperature_C'].unique())
        
        free_water = [rubber_0[rubber_0['temperature_C'] == t]['free_water_loss_50_150C_pct'].mean() for t in temps]
        bound_water = [rubber_0[rubber_0['temperature_C'] == t]['bound_water_loss_150_400C_pct'].mean() for t in temps]
        ch_loss = [rubber_0[rubber_0['temperature_C'] == t]['CH_dehydrox_loss_400_500C_pct'].mean() for t in temps]
        caco3_loss = [rubber_0[rubber_0['temperature_C'] == t]['CaCO3_decomp_loss_600_800C_pct'].mean() for t in temps]
        
        ax1.bar(temps, free_water, label='Free Water (50-150°C)', color='#4ECDC4', edgecolor='black', linewidth=0.5)
        ax1.bar(temps, bound_water, bottom=free_water, label='Bound Water (150-400°C)', 
               color='#44AF69', edgecolor='black', linewidth=0.5)
        bottom1 = np.array(free_water) + np.array(bound_water)
        ax1.bar(temps, ch_loss, bottom=bottom1, label='CH Dehydroxylation (400-500°C)', 
               color='#F7B801', edgecolor='black', linewidth=0.5)
        bottom2 = bottom1 + np.array(ch_loss)
        ax1.bar(temps, caco3_loss, bottom=bottom2, label='CaCO₃ Decomposition (600-800°C)', 
               color='#F77F00', edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Exposure Temperature (°C)')
        ax1.set_ylabel('Mass Loss (%)')
        ax1.set_title('(a) Mass Loss Mechanisms - 0% Rubber')
        ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
        ax1.grid(alpha=0.3, linestyle='--', axis='y')
        
        # Plot 2: Stacked mass loss for 10% rubber
        ax2 = fig.add_subplot(gs[0, 1])
        rubber_10 = self.tga[self.tga['rubber_content_pct'] == 10]
        temps = sorted(rubber_10['temperature_C'].unique())
        
        free_water = [rubber_10[rubber_10['temperature_C'] == t]['free_water_loss_50_150C_pct'].mean() for t in temps]
        bound_water = [rubber_10[rubber_10['temperature_C'] == t]['bound_water_loss_150_400C_pct'].mean() for t in temps]
        rubber_comb = [rubber_10[rubber_10['temperature_C'] == t]['rubber_combustion_loss_300_500C_pct'].mean() for t in temps]
        ch_loss = [rubber_10[rubber_10['temperature_C'] == t]['CH_dehydrox_loss_400_500C_pct'].mean() for t in temps]
        caco3_loss = [rubber_10[rubber_10['temperature_C'] == t]['CaCO3_decomp_loss_600_800C_pct'].mean() for t in temps]
        
        ax2.bar(temps, free_water, label='Free Water (50-150°C)', color='#4ECDC4', edgecolor='black', linewidth=0.5)
        ax2.bar(temps, bound_water, bottom=free_water, label='Bound Water (150-400°C)', 
               color='#44AF69', edgecolor='black', linewidth=0.5)
        bottom1 = np.array(free_water) + np.array(bound_water)
        ax2.bar(temps, rubber_comb, bottom=bottom1, label='Rubber Combustion (300-500°C)', 
               color='#E63946', edgecolor='black', linewidth=0.5)
        bottom2 = bottom1 + np.array(rubber_comb)
        ax2.bar(temps, ch_loss, bottom=bottom2, label='CH Dehydroxylation (400-500°C)', 
               color='#F7B801', edgecolor='black', linewidth=0.5)
        bottom3 = bottom2 + np.array(ch_loss)
        ax2.bar(temps, caco3_loss, bottom=bottom3, label='CaCO₃ Decomposition (600-800°C)', 
               color='#F77F00', edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Exposure Temperature (°C)')
        ax2.set_ylabel('Mass Loss (%)')
        ax2.set_title('(b) Mass Loss Mechanisms - 10% Rubber')
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
        ax2.grid(alpha=0.3, linestyle='--', axis='y')
        
        # Plot 3: Bound water loss comparison
        ax3 = fig.add_subplot(gs[1, 0])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.tga[self.tga['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['bound_water_loss_150_400C_pct'].mean()
            ax3.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax3.set_xlabel('Exposure Temperature (°C)')
        ax3.set_ylabel('Bound Water Loss (%)')
        ax3.set_title('(c) C-S-H Degradation (Bound Water Loss)')
        ax3.legend(frameon=False, ncol=2)
        ax3.grid(alpha=0.3, linestyle='--')
        
        # Plot 4: Total mass loss
        ax4 = fig.add_subplot(gs[1, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.tga[self.tga['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['total_mass_loss_pct'].mean()
            ax4.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax4.set_xlabel('Exposure Temperature (°C)')
        ax4.set_ylabel('Total Mass Loss (%)')
        ax4.set_title('(d) Total Mass Loss Evolution')
        ax4.legend(frameon=False, ncol=2)
        ax4.grid(alpha=0.3, linestyle='--')
        
        plt.suptitle('Figure 3: TGA/DTA Analysis - Mass Loss Mechanisms', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/fig3_tga_mass_loss.png', bbox_inches='tight')
        print("✓ Figure 3 saved: TGA Mass Loss Mechanisms")
        plt.close()
        
    def plot_microct_porosity_evolution(self):
        """Figure 4: Micro-CT porosity evolution"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Total porosity
        ax1 = fig.add_subplot(gs[0, 0])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.porosity[self.porosity['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['total_porosity_pct'].mean()
            ax1.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Total Porosity (%)')
        ax1.set_title('(a) Total Porosity Evolution')
        ax1.legend(frameon=False, ncol=2)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Plot 2: Macro-porosity (structural impact)
        ax2 = fig.add_subplot(gs[0, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.porosity[self.porosity['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['macro_porosity_50_1000um_pct'].mean()
            ax2.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Macro-porosity 50-1000 μm (%)')
        ax2.set_title('(b) Macro-porosity Development')
        ax2.legend(frameon=False, ncol=2)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_yscale('log')
        
        # Plot 3: Pore connectivity
        ax3 = fig.add_subplot(gs[1, 0])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.porosity[self.porosity['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['pore_connectivity_index'].mean()
            ax3.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Pore Connectivity Index (0-1)')
        ax3.set_title('(c) Pore Network Connectivity')
        ax3.legend(frameon=False, ncol=2)
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Permeability
        ax4 = fig.add_subplot(gs[1, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = self.porosity[self.porosity['rubber_content_pct'] == rubber]
            grouped = data.groupby('temperature_C')['permeability_m2'].mean() * 1e18
            ax4.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                    linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Permeability (×10⁻¹⁸ m²)')
        ax4.set_title('(d) Permeability Increase')
        ax4.legend(frameon=False, ncol=2)
        ax4.grid(alpha=0.3, linestyle='--')
        ax4.set_yscale('log')
        
        plt.suptitle('Figure 4: Micro-CT Analysis - Porosity and Pore Network Evolution', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/fig4_microct_porosity.png', bbox_inches='tight')
        print("✓ Figure 4 saved: Micro-CT Porosity Evolution")
        plt.close()
        
    def plot_crack_network_development(self):
        """Figure 5: Crack network development"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        heated = self.cracks[self.cracks['temperature_C'] > 20]
        
        # Plot 1: Crack density
        ax1 = fig.add_subplot(gs[0, 0])
        for rubber in [0, 5, 10, 15, 20]:
            data = heated[heated['rubber_content_pct'] == rubber]
            if len(data) > 0:
                grouped = data.groupby('temperature_C')['crack_density_mm_mm3'].mean()
                ax1.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                        linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Crack Density (mm/mm³)')
        ax1.set_title('(a) Crack Density Development')
        ax1.legend(frameon=False, ncol=2)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_yscale('log')
        
        # Plot 2: Average crack width
        ax2 = fig.add_subplot(gs[0, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = heated[heated['rubber_content_pct'] == rubber]
            if len(data) > 0:
                grouped = data.groupby('temperature_C')['avg_crack_width_um'].mean()
                ax2.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                        linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Average Crack Width (μm)')
        ax2.set_title('(b) Crack Width Evolution')
        ax2.legend(frameon=False, ncol=2)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_yscale('log')
        
        # Plot 3: Crack network connectivity
        ax3 = fig.add_subplot(gs[1, 0])
        for rubber in [0, 5, 10, 15, 20]:
            data = heated[heated['rubber_content_pct'] == rubber]
            if len(data) > 0:
                grouped = data.groupby('temperature_C')['crack_network_connectivity'].mean()
                ax3.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                        linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Crack Network Connectivity (0-1)')
        ax3.set_title('(c) Crack Network Connectivity')
        ax3.legend(frameon=False, ncol=2)
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Damage parameter
        ax4 = fig.add_subplot(gs[1, 1])
        for rubber in [0, 5, 10, 15, 20]:
            data = heated[heated['rubber_content_pct'] == rubber]
            if len(data) > 0:
                grouped = data.groupby('temperature_C')['damage_parameter'].mean()
                ax4.plot(grouped.index, grouped.values, 'o-', color=COLORS[rubber], 
                        linewidth=2, markersize=8, label=f'{rubber}% Rubber')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Damage Parameter (0-1)')
        ax4.set_title('(d) Overall Damage Evolution')
        ax4.legend(frameon=False, ncol=2)
        ax4.grid(alpha=0.3, linestyle='--')
        ax4.set_ylim(0, 1.05)
        
        plt.suptitle('Figure 5: Micro-CT Analysis - Crack Network Development', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/fig5_crack_networks.png', bbox_inches='tight')
        print("✓ Figure 5 saved: Crack Network Development")
        plt.close()
        
    def plot_integrated_degradation_map(self):
        """Figure 6: Integrated degradation mechanism map"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Heatmap data preparation
        rubber_contents = [0, 5, 10, 15, 20]
        temperatures = [20, 200, 400, 600, 800]
        
        # 1. ITZ thickness heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        itz_matrix = np.zeros((len(temperatures), len(rubber_contents)))
        for i, temp in enumerate(temperatures):
            for j, rubber in enumerate(rubber_contents):
                if rubber > 0:
                    data = self.sem[(self.sem['temperature_C'] == temp) & 
                                   (self.sem['rubber_content_pct'] == rubber) &
                                   (self.sem['itz_type'] == 'rubber_paste')]
                    if len(data) > 0:
                        itz_matrix[i, j] = data['itz_thickness_um'].mean()
        im1 = ax1.imshow(itz_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(rubber_contents)))
        ax1.set_yticks(range(len(temperatures)))
        ax1.set_xticklabels(rubber_contents)
        ax1.set_yticklabels(temperatures)
        ax1.set_xlabel('Rubber Content (%)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('(a) ITZ Thickness (μm)')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Portlandite consumption heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ch_matrix = np.zeros((len(temperatures), len(rubber_contents)))
        for i, temp in enumerate(temperatures):
            for j, rubber in enumerate(rubber_contents):
                data = self.xrd[(self.xrd['temperature_C'] == temp) & 
                               (self.xrd['rubber_content_pct'] == rubber)]
                if len(data) > 0:
                    ch_matrix[i, j] = data['portlandite_wt_pct'].mean()
        im2 = ax2.imshow(ch_matrix, cmap='Blues_r', aspect='auto')
        ax2.set_xticks(range(len(rubber_contents)))
        ax2.set_yticks(range(len(temperatures)))
        ax2.set_xticklabels(rubber_contents)
        ax2.set_yticklabels(temperatures)
        ax2.set_xlabel('Rubber Content (%)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('(b) Portlandite Content (wt%)')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Total mass loss heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        mass_matrix = np.zeros((len(temperatures), len(rubber_contents)))
        for i, temp in enumerate(temperatures):
            for j, rubber in enumerate(rubber_contents):
                data = self.tga[(self.tga['temperature_C'] == temp) & 
                               (self.tga['rubber_content_pct'] == rubber)]
                if len(data) > 0:
                    mass_matrix[i, j] = data['total_mass_loss_pct'].mean()
        im3 = ax3.imshow(mass_matrix, cmap='Purples', aspect='auto')
        ax3.set_xticks(range(len(rubber_contents)))
        ax3.set_yticks(range(len(temperatures)))
        ax3.set_xticklabels(rubber_contents)
        ax3.set_yticklabels(temperatures)
        ax3.set_xlabel('Rubber Content (%)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('(c) Total Mass Loss (%)')
        plt.colorbar(im3, ax=ax3)
        
        # 4. Total porosity heatmap
        ax4 = fig.add_subplot(gs[1, 0])
        por_matrix = np.zeros((len(temperatures), len(rubber_contents)))
        for i, temp in enumerate(temperatures):
            for j, rubber in enumerate(rubber_contents):
                data = self.porosity[(self.porosity['temperature_C'] == temp) & 
                                    (self.porosity['rubber_content_pct'] == rubber)]
                if len(data) > 0:
                    por_matrix[i, j] = data['total_porosity_pct'].mean()
        im4 = ax4.imshow(por_matrix, cmap='Greens', aspect='auto')
        ax4.set_xticks(range(len(rubber_contents)))
        ax4.set_yticks(range(len(temperatures)))
        ax4.set_xticklabels(rubber_contents)
        ax4.set_yticklabels(temperatures)
        ax4.set_xlabel('Rubber Content (%)')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('(d) Total Porosity (%)')
        plt.colorbar(im4, ax=ax4)
        
        # 5. Pore connectivity heatmap
        ax5 = fig.add_subplot(gs[1, 1])
        conn_matrix = np.zeros((len(temperatures), len(rubber_contents)))
        for i, temp in enumerate(temperatures):
            for j, rubber in enumerate(rubber_contents):
                data = self.porosity[(self.porosity['temperature_C'] == temp) & 
                                    (self.porosity['rubber_content_pct'] == rubber)]
                if len(data) > 0:
                    conn_matrix[i, j] = data['pore_connectivity_index'].mean()
        im5 = ax5.imshow(conn_matrix, cmap='RdYlGn_r', aspect='auto')
        ax5.set_xticks(range(len(rubber_contents)))
        ax5.set_yticks(range(len(temperatures)))
        ax5.set_xticklabels(rubber_contents)
        ax5.set_yticklabels(temperatures)
        ax5.set_xlabel('Rubber Content (%)')
        ax5.set_ylabel('Temperature (°C)')
        ax5.set_title('(e) Pore Connectivity (0-1)')
        plt.colorbar(im5, ax=ax5)
        
        # 6. Damage parameter heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        damage_matrix = np.zeros((len(temperatures), len(rubber_contents)))
        for i, temp in enumerate(temperatures):
            for j, rubber in enumerate(rubber_contents):
                data = self.cracks[(self.cracks['temperature_C'] == temp) & 
                                  (self.cracks['rubber_content_pct'] == rubber)]
                if len(data) > 0:
                    damage_matrix[i, j] = data['damage_parameter'].mean()
        im6 = ax6.imshow(damage_matrix, cmap='Reds', aspect='auto')
        ax6.set_xticks(range(len(rubber_contents)))
        ax6.set_yticks(range(len(temperatures)))
        ax6.set_xticklabels(rubber_contents)
        ax6.set_yticklabels(temperatures)
        ax6.set_xlabel('Rubber Content (%)')
        ax6.set_ylabel('Temperature (°C)')
        ax6.set_title('(f) Damage Parameter (0-1)')
        plt.colorbar(im6, ax=ax6)
        
        plt.suptitle('Figure 6: Integrated Degradation Mechanism Map', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/fig6_integrated_degradation_map.png', bbox_inches='tight')
        print("✓ Figure 6 saved: Integrated Degradation Map")
        plt.close()
        
    def generate_all_figures(self):
        """Generate all publication-quality figures"""
        print("\n" + "=" * 80)
        print("GENERATING PUBLICATION-QUALITY FIGURES")
        print("=" * 80 + "\n")
        
        self.plot_sem_itz_evolution()
        self.plot_xrd_phase_evolution()
        self.plot_tga_mass_loss_mechanisms()
        self.plot_microct_porosity_evolution()
        self.plot_crack_network_development()
        self.plot_integrated_degradation_map()
        
        print("\n" + "=" * 80)
        print("ALL FIGURES GENERATED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n✓ All figures saved to: {self.output_dir}/")
        print("✓ Ready for publication and dissertation")


def main():
    """Main execution function"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  MICROSTRUCTURAL DATA VISUALIZATION SUITE                                  ║
║  Phase 3: Publication-Quality Figures                                      ║
║                                                                            ║
║  PhD-Level Analysis and Visualization                                      ║
║  Generated: 2025-10-18                                                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    visualizer = MicrostructuralVisualizer()
    visualizer.generate_all_figures()
    
    print("\n✓ Visualization complete!")
    print("\nNext steps:")
    print("  1. Review figures for publication")
    print("  2. Export data for thermo-mechanical model validation")
    print("  3. Prepare manuscript figures")


if __name__ == "__main__":
    main()
