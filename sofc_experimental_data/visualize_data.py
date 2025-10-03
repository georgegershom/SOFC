"""
Visualization tools for SOFC experimental datasets
Creates plots and visualizations for DIC, XRD, and post-mortem analysis data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DataVisualizer:
    def __init__(self, base_path="."):
        self.base_path = base_path
        self.fig_path = os.path.join(base_path, "figures")
        os.makedirs(self.fig_path, exist_ok=True)
    
    def visualize_dic_data(self, experiment_type='thermal_cycling'):
        """Create visualizations for DIC data"""
        print(f"\nVisualizing DIC data for {experiment_type}...")
        
        # Load summary data
        dic_data = pd.read_csv(f'dic_data/{experiment_type}/dic_summary.csv')
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Temperature profile
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dic_data['time_min'], dic_data['temperature_C'], 'r-', linewidth=2)
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Temperature (°C)', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'DIC Analysis - {experiment_type.replace("_", " ").title()}')
        
        # Panel 2: Strain evolution
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(dic_data['time_min'], dic_data['mean_exx']*100, label='εxx', linewidth=2)
        ax2.plot(dic_data['time_min'], dic_data['mean_eyy']*100, label='εyy', linewidth=2)
        ax2.plot(dic_data['time_min'], dic_data['mean_von_mises']*100, 
                label='von Mises', linewidth=2, linestyle='--')
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Strain (%)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Maximum strain and hotspots
        ax3 = fig.add_subplot(gs[2, 0:2])
        ax3.plot(dic_data['time_min'], dic_data['max_von_mises']*100, 
                'g-', label='Max von Mises Strain', linewidth=2)
        ax3.axhline(y=1.0, color='r', linestyle='--', label='Critical Strain (1%)')
        ax3.fill_between(dic_data['time_min'], 0, dic_data['max_von_mises']*100, 
                         where=(dic_data['max_von_mises'] > 0.01), alpha=0.3, color='r')
        ax3.set_xlabel('Time (min)')
        ax3.set_ylabel('Maximum Strain (%)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Hotspot count
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.plot(dic_data['time_min'], dic_data['hotspot_count'], 
                'o-', color='darkred', markersize=3)
        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('Hotspot Count\n(ε > 1%)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Digital Image Correlation Results - {experiment_type}', 
                    fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(self.fig_path, f'dic_{experiment_type}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # Load and visualize strain maps
        with open(f'dic_data/{experiment_type}/strain_maps.json', 'r') as f:
            strain_maps = json.load(f)
        
        if strain_maps:
            # Plot first and last strain maps
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            for idx, map_idx in enumerate([0, -1]):
                strain_map = strain_maps[map_idx]
                row = idx
                
                # Von Mises strain
                im1 = axes[row, 0].imshow(strain_map['strain_tensor']['von_mises_strain'],
                                          cmap='jet', aspect='auto')
                axes[row, 0].set_title(f'Von Mises Strain - {strain_map["time_min"]} min')
                plt.colorbar(im1, ax=axes[row, 0], label='Strain')
                
                # εxx strain
                im2 = axes[row, 1].imshow(strain_map['strain_tensor']['exx'],
                                          cmap='RdBu_r', aspect='auto')
                axes[row, 1].set_title(f'εxx - {strain_map["time_min"]} min')
                plt.colorbar(im2, ax=axes[row, 1], label='Strain')
                
                # εyy strain
                im3 = axes[row, 2].imshow(strain_map['strain_tensor']['eyy'],
                                          cmap='RdBu_r', aspect='auto')
                axes[row, 2].set_title(f'εyy - {strain_map["time_min"]} min')
                plt.colorbar(im3, ax=axes[row, 2], label='Strain')
            
            plt.suptitle('Strain Field Evolution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_path, f'strain_maps_{experiment_type}.png'),
                       dpi=150, bbox_inches='tight')
            plt.show()
    
    def visualize_xrd_data(self, experiment_type='thermal_cycling'):
        """Create visualizations for XRD data"""
        print(f"\nVisualizing XRD data for {experiment_type}...")
        
        # Load residual stress profile
        stress_data = pd.read_csv(f'xrd_data/{experiment_type}/residual_stress_profile.csv')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Stress profiles
        ax1 = axes[0, 0]
        ax1.plot(stress_data['position_um'], stress_data['sigma_xx_MPa'], 
                label='σxx', linewidth=2)
        ax1.plot(stress_data['position_um'], stress_data['sigma_yy_MPa'], 
                label='σyy', linewidth=2)
        ax1.plot(stress_data['position_um'], stress_data['sigma_zz_MPa'], 
                label='σzz', linewidth=2)
        ax1.set_xlabel('Position (μm)')
        ax1.set_ylabel('Residual Stress (MPa)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Residual Stress Distribution')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Von Mises stress
        ax2 = axes[0, 1]
        ax2.plot(stress_data['position_um'], stress_data['von_mises_MPa'], 
                'r-', linewidth=2)
        ax2.set_xlabel('Position (μm)')
        ax2.set_ylabel('von Mises Stress (MPa)')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('von Mises Stress Profile')
        
        # Load lattice strain data
        lattice_data = pd.read_csv(f'xrd_data/{experiment_type}/lattice_strain_data.csv')
        
        # Lattice strain vs temperature
        ax3 = axes[1, 0]
        for material in lattice_data['material'].unique():
            mat_data = lattice_data[lattice_data['material'] == material]
            ax3.plot(mat_data['temperature_C'], mat_data['lattice_strain']*100,
                    'o-', label=material, markersize=4, alpha=0.7)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Lattice Strain (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Thermal Lattice Strain')
        
        # Load sin²ψ results
        sin2psi_stress = pd.read_csv(f'xrd_data/{experiment_type}/sin2psi_stress_results.csv')
        
        # Stress from sin²ψ method
        ax4 = axes[1, 1]
        for material in sin2psi_stress['material'].unique():
            mat_data = sin2psi_stress[sin2psi_stress['material'] == material]
            ax4.plot(mat_data['position_um'], mat_data['stress_MPa'],
                    'o-', label=material, markersize=6)
        ax4.set_xlabel('Position (μm)')
        ax4.set_ylabel('Calculated Stress (MPa)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title('sin²ψ Method Stress Results')
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.suptitle(f'Synchrotron XRD Analysis - {experiment_type}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_path, f'xrd_{experiment_type}.png'),
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_postmortem_data(self, experiment_type='thermal_cycling'):
        """Create visualizations for post-mortem analysis data"""
        print(f"\nVisualizing post-mortem data for {experiment_type}...")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Load SEM crack analysis
        crack_data = pd.read_csv(f'post_mortem/{experiment_type}/sem_crack_analysis.csv')
        
        # Crack density by region
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.bar(crack_data['region'], crack_data['crack_density_per_mm2'], 
               color='darkred', alpha=0.7)
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Crack Density (cracks/mm²)')
        ax1.set_title('Crack Density Distribution')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(crack_data['crack_density_per_mm2']):
            ax1.text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom')
        
        # Crack characteristics
        ax2 = fig.add_subplot(gs[0, 2])
        crack_types = ['transgranular_fraction', 'intergranular_fraction', 'interface_fraction']
        mean_fractions = crack_data[crack_types].mean()
        ax2.pie(mean_fractions, labels=['Trans.', 'Inter.', 'Interface'],
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Crack Type Distribution')
        
        # Load EDS data
        eds_data = pd.read_csv(f'post_mortem/{experiment_type}/eds_line_scan.csv')
        
        # EDS line scan
        ax3 = fig.add_subplot(gs[1, :])
        elements = ['Ni_at_percent', 'Zr_at_percent', 'Ce_at_percent', 'La_at_percent']
        colors = ['blue', 'green', 'red', 'purple']
        for elem, color in zip(elements, colors):
            if elem in eds_data.columns:
                ax3.plot(eds_data['position_um'], eds_data[elem], 
                        label=elem.split('_')[0], linewidth=2, color=color)
        ax3.set_xlabel('Position (μm)')
        ax3.set_ylabel('Atomic %')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('EDS Elemental Line Scan')
        
        # Load nanoindentation data
        nano_data = pd.read_csv(f'post_mortem/{experiment_type}/nanoindentation_grid.csv')
        
        # Young's modulus by material
        ax4 = fig.add_subplot(gs[2, 0])
        materials = nano_data['material'].unique()
        modulus_data = [nano_data[nano_data['material'] == m]['youngs_modulus_GPa'] 
                       for m in materials]
        bp = ax4.boxplot(modulus_data, labels=materials, patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette('husl', len(materials))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax4.set_ylabel('Young\'s Modulus (GPa)')
        ax4.set_title('Mechanical Properties')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Hardness vs Modulus
        ax5 = fig.add_subplot(gs[2, 1])
        for material in materials:
            mat_data = nano_data[nano_data['material'] == material]
            ax5.scatter(mat_data['youngs_modulus_GPa'], mat_data['hardness_GPa'],
                       label=material, alpha=0.6, s=30)
        ax5.set_xlabel('Young\'s Modulus (GPa)')
        ax5.set_ylabel('Hardness (GPa)')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Hardness vs Modulus')
        
        # Load porosity data
        porosity_data = pd.read_csv(f'post_mortem/{experiment_type}/porosity_analysis.csv')
        
        # Porosity comparison
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.bar(porosity_data['material'], porosity_data['porosity_percent'],
               color='steelblue', alpha=0.7)
        ax6.set_ylabel('Porosity (%)')
        ax6.set_title('Porosity Analysis')
        ax6.tick_params(axis='x', rotation=45)
        for i, v in enumerate(porosity_data['porosity_percent']):
            ax6.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.suptitle(f'Post-Mortem Analysis - {experiment_type}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_path, f'postmortem_{experiment_type}.png'),
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self):
        """Create a summary report of all datasets"""
        print("\n" + "="*60)
        print("SOFC EXPERIMENTAL DATA SUMMARY REPORT")
        print("="*60)
        
        experiments = ['sintering', 'thermal_cycling', 'startup_shutdown']
        
        for exp in experiments:
            print(f"\n{exp.upper().replace('_', ' ')} EXPERIMENT")
            print("-"*40)
            
            # DIC Summary
            if os.path.exists(f'dic_data/{exp}/dic_summary.csv'):
                dic_data = pd.read_csv(f'dic_data/{exp}/dic_summary.csv')
                print(f"DIC Data:")
                print(f"  - Time points: {len(dic_data)}")
                print(f"  - Max strain: {dic_data['max_von_mises'].max():.4f}")
                print(f"  - Max hotspots: {dic_data['hotspot_count'].max()}")
            
            # XRD Summary
            if os.path.exists(f'xrd_data/{exp}/residual_stress_profile.csv'):
                stress_data = pd.read_csv(f'xrd_data/{exp}/residual_stress_profile.csv')
                print(f"XRD Data:")
                print(f"  - Max compressive stress: {stress_data['sigma_xx_MPa'].min():.1f} MPa")
                print(f"  - Max tensile stress: {stress_data['sigma_xx_MPa'].max():.1f} MPa")
            
            # Post-mortem Summary
            if os.path.exists(f'post_mortem/{exp}/sem_crack_analysis.csv'):
                crack_data = pd.read_csv(f'post_mortem/{exp}/sem_crack_analysis.csv')
                print(f"Post-Mortem Analysis:")
                print(f"  - Max crack density: {crack_data['crack_density_per_mm2'].max():.1f} cracks/mm²")
                print(f"  - Total cracks analyzed: {crack_data['total_cracks'].sum()}")
        
        print("\n" + "="*60)
        print("Data generation and visualization complete!")
        print("="*60)

if __name__ == "__main__":
    visualizer = DataVisualizer()
    
    # Run all visualizations for one experiment type
    experiment = 'thermal_cycling'
    
    visualizer.visualize_dic_data(experiment)
    visualizer.visualize_xrd_data(experiment)
    visualizer.visualize_postmortem_data(experiment)
    visualizer.create_summary_report()