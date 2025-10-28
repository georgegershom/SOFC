#!/usr/bin/env python3
"""
SOFC Experimental Data Visualizer
Creates comprehensive visualizations for the generated experimental datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json
from scipy.interpolate import griddata
import os

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

class SOFCDataVisualizer:
    def __init__(self, data_dir='/workspace/sofc_experimental_data'):
        self.data_dir = data_dir
        self.figsize = (12, 8)
        
    def load_data(self):
        """Load all experimental data"""
        print("Loading experimental data...")
        
        with open(f'{self.data_dir}/dic_data.json', 'r') as f:
            self.dic_data = json.load(f)
        
        with open(f'{self.data_dir}/xrd_data.json', 'r') as f:
            self.xrd_data = json.load(f)
        
        with open(f'{self.data_dir}/post_mortem_data.json', 'r') as f:
            self.post_mortem_data = json.load(f)
        
        print("Data loaded successfully!")
    
    def plot_dic_strain_maps(self):
        """Plot DIC strain maps for different conditions"""
        print("Creating DIC strain map visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Digital Image Correlation (DIC) Strain Maps', fontsize=16, fontweight='bold')
        
        # Sintering strain maps at different temperatures
        sintering_data = self.dic_data['sintering']
        selected_temps = [1200, 1300, 1400, 1500]
        
        for i, temp in enumerate(selected_temps):
            if i < 4:
                ax = axes[0, i] if i < 2 else axes[1, i-2]
                
                # Find closest temperature data
                temp_data = min(sintering_data, key=lambda x: abs(x['temperature'] - temp))
                strain_map = np.array(temp_data['strain_map'], dtype=float)
                
                im = ax.imshow(strain_map, cmap='jet', aspect='auto', origin='lower')
                ax.set_title(f'Sintering at {temp}°C\nMax Strain: {temp_data["max_strain"]:.4f}')
                ax.set_xlabel('X Position (pixels)')
                ax.set_ylabel('Y Position (pixels)')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Strain', rotation=270, labelpad=20)
        
        # Thermal cycling strain evolution
        ax = axes[1, 2]
        thermal_data = self.dic_data['thermal_cycling']
        temps = [data['temperature'] for data in thermal_data[:20]]  # First 20 points
        max_strains = [data['max_strain'] for data in thermal_data[:20]]
        
        ax.plot(temps, max_strains, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Maximum Strain')
        ax.set_title('Thermal Cycling Strain Evolution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/dic_strain_maps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_xrd_stress_profiles(self):
        """Plot XRD residual stress profiles"""
        print("Creating XRD stress profile visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Synchrotron X-ray Diffraction Analysis', fontsize=16, fontweight='bold')
        
        # Residual stress profiles
        ax = axes[0, 0]
        residual_data = self.xrd_data['residual_stresses']
        positions = [data['position'] for data in residual_data]
        stresses = [data['stress'] for data in residual_data]
        layers = [data['layer'] for data in residual_data]
        
        # Color by layer
        colors = {'anode': 'red', 'electrolyte': 'blue', 'cathode': 'green'}
        for layer in ['anode', 'electrolyte', 'cathode']:
            layer_pos = [pos for pos, l in zip(positions, layers) if l == layer]
            layer_stress = [stress for stress, l in zip(stresses, layers) if l == layer]
            ax.scatter(layer_pos, layer_stress, c=colors[layer], label=layer.capitalize(), alpha=0.7, s=50)
        
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Residual Stress (MPa)')
        ax.set_title('Residual Stress Profiles Across SOFC Cross-Section')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Lattice strain vs temperature
        ax = axes[0, 1]
        lattice_data = self.xrd_data['lattice_strains']
        materials = list(set([data['material'] for data in lattice_data]))
        
        for material in materials:
            mat_data = [data for data in lattice_data if data['material'] == material]
            temps = [data['temperature'] for data in mat_data]
            strains = [data['lattice_strain'] for data in mat_data]
            ax.plot(temps, strains, 'o-', label=material, linewidth=2, markersize=4)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Lattice Strain')
        ax.set_title('Lattice Strain vs Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Peak shift data (sin²ψ method)
        ax = axes[1, 0]
        peak_data = self.xrd_data['peak_shifts']
        psi_angles = [data['psi_angle'] for data in peak_data]
        peak_shifts = [data['peak_shift'] for data in peak_data]
        
        ax.scatter(psi_angles, peak_shifts, c='purple', alpha=0.7, s=50)
        ax.set_xlabel('ψ Angle (degrees)')
        ax.set_ylabel('Peak Shift')
        ax.set_title('Peak Shift Data (sin²ψ Method)')
        ax.grid(True, alpha=0.3)
        
        # Microcrack initiation
        ax = axes[1, 1]
        microcrack_data = self.xrd_data['microcrack_data']
        strains = [data['strain'] for data in microcrack_data]
        crack_initiated = [data['crack_initiated'] for data in microcrack_data]
        
        # Separate data by crack initiation
        no_crack_strains = [s for s, c in zip(strains, crack_initiated) if not c]
        crack_strains = [s for s, c in zip(strains, crack_initiated) if c]
        
        ax.scatter(no_crack_strains, [0]*len(no_crack_strains), c='green', alpha=0.6, s=30, label='No Crack')
        ax.scatter(crack_strains, [1]*len(crack_strains), c='red', alpha=0.6, s=30, label='Crack Initiated')
        
        # Add critical strain threshold line
        ax.axvline(x=0.02, color='black', linestyle='--', linewidth=2, label='Critical Strain (0.02)')
        
        ax.set_xlabel('Strain')
        ax.set_ylabel('Crack Initiation')
        ax.set_title('Microcrack Initiation Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/xrd_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_post_mortem_analysis(self):
        """Plot post-mortem analysis results"""
        print("Creating post-mortem analysis visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Post-Mortem Analysis Results', fontsize=16, fontweight='bold')
        
        # Crack density from SEM images
        ax = axes[0, 0]
        sem_data = self.post_mortem_data['sem_images']
        crack_densities = [data['crack_density'] for data in sem_data]
        magnifications = [data['magnification'] for data in sem_data]
        
        scatter = ax.scatter(magnifications, crack_densities, c=crack_densities, cmap='Reds', s=100, alpha=0.7)
        ax.set_xlabel('Magnification')
        ax.set_ylabel('Crack Density (cracks/mm²)')
        ax.set_title('Crack Density vs Magnification')
        ax.set_xscale('log')
        plt.colorbar(scatter, ax=ax, label='Crack Density')
        ax.grid(True, alpha=0.3)
        
        # EDS elemental composition
        ax = axes[0, 1]
        eds_data = self.post_mortem_data['eds_scans'][0]['scan_data']  # First scan
        positions = [data['position'] for data in eds_data]
        ni_content = [data['Ni'] for data in eds_data]
        zr_content = [data['Zr'] for data in eds_data]
        y_content = [data['Y'] for data in eds_data]
        
        ax.plot(positions, ni_content, 'r-', linewidth=2, label='Ni')
        ax.plot(positions, zr_content, 'b-', linewidth=2, label='Zr')
        ax.plot(positions, y_content, 'g-', linewidth=2, label='Y')
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Elemental Content (at%)')
        ax.set_title('EDS Line Scan - Elemental Composition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Nano-indentation Young's modulus
        ax = axes[0, 2]
        nano_data = self.post_mortem_data['nano_indentation']
        phases = [data['phase'] for data in nano_data]
        youngs_modulus = [data['youngs_modulus'] for data in nano_data]
        
        # Box plot by phase
        phase_data = {}
        for phase, modulus in zip(phases, youngs_modulus):
            if phase not in phase_data:
                phase_data[phase] = []
            phase_data[phase].append(modulus)
        
        box_data = [phase_data[phase] for phase in phase_data.keys()]
        box_labels = list(phase_data.keys())
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Young\'s Modulus (GPa)')
        ax.set_title('Nano-indentation Young\'s Modulus by Phase')
        ax.grid(True, alpha=0.3)
        
        # Hardness vs Young's modulus
        ax = axes[1, 0]
        hardness = [data['hardness'] for data in nano_data]
        
        scatter = ax.scatter(youngs_modulus, hardness, c=[colors[box_labels.index(phase)] for phase in phases], 
                           s=100, alpha=0.7)
        ax.set_xlabel('Young\'s Modulus (GPa)')
        ax.set_ylabel('Hardness (GPa)')
        ax.set_title('Hardness vs Young\'s Modulus')
        ax.grid(True, alpha=0.3)
        
        # Creep compliance
        ax = axes[1, 1]
        creep_compliance = [data['creep_compliance'] for data in nano_data]
        
        ax.hist(creep_compliance, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('Creep Compliance (1/GPa)')
        ax.set_ylabel('Frequency')
        ax.set_title('Creep Compliance Distribution')
        ax.grid(True, alpha=0.3)
        
        # Phase distribution
        ax = axes[1, 2]
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        wedges, texts, autotexts = ax.pie(phase_counts.values(), labels=phase_counts.keys(), 
                                        autopct='%1.1f%%', startangle=90)
        ax.set_title('Phase Distribution in Nano-indentation')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/post_mortem_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        print("Creating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('SOFC Experimental Measurement Dataset Summary', fontsize=20, fontweight='bold', y=0.95)
        
        # DIC summary - strain evolution
        ax1 = fig.add_subplot(gs[0, :2])
        sintering_data = self.dic_data['sintering']
        temps = [data['temperature'] for data in sintering_data]
        max_strains = [data['max_strain'] for data in sintering_data]
        mean_strains = [data['mean_strain'] for data in sintering_data]
        
        ax1.plot(temps, max_strains, 'r-', linewidth=2, label='Maximum Strain')
        ax1.plot(temps, mean_strains, 'b-', linewidth=2, label='Mean Strain')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Strain')
        ax1.set_title('DIC Strain Evolution During Sintering')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # XRD stress profile
        ax2 = fig.add_subplot(gs[0, 2:])
        residual_data = self.xrd_data['residual_stresses']
        positions = [data['position'] for data in residual_data]
        stresses = [data['stress'] for data in residual_data]
        
        ax2.plot(positions, stresses, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Position (mm)')
        ax2.set_ylabel('Residual Stress (MPa)')
        ax2.set_title('XRD Residual Stress Profile')
        ax2.grid(True, alpha=0.3)
        
        # Thermal cycling
        ax3 = fig.add_subplot(gs[1, :2])
        thermal_data = self.dic_data['thermal_cycling']
        cycle_temps = [data['temperature'] for data in thermal_data[:50]]
        cycle_strains = [data['max_strain'] for data in thermal_data[:50]]
        
        ax3.scatter(cycle_temps, cycle_strains, c='orange', alpha=0.6, s=30)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Maximum Strain')
        ax3.set_title('Thermal Cycling Strain Response')
        ax3.grid(True, alpha=0.3)
        
        # Lattice strain
        ax4 = fig.add_subplot(gs[1, 2:])
        lattice_data = self.xrd_data['lattice_strains']
        lattice_temps = [data['temperature'] for data in lattice_data if data['material'] == 'YSZ']
        lattice_strains = [data['lattice_strain'] for data in lattice_data if data['material'] == 'YSZ']
        
        ax4.plot(lattice_temps, lattice_strains, 'purple', linewidth=2, marker='s', markersize=4)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Lattice Strain')
        ax4.set_title('YSZ Lattice Strain vs Temperature')
        ax4.grid(True, alpha=0.3)
        
        # Crack density
        ax5 = fig.add_subplot(gs[2, :2])
        sem_data = self.post_mortem_data['sem_images']
        crack_densities = [data['crack_density'] for data in sem_data]
        
        ax5.hist(crack_densities, bins=10, alpha=0.7, color='red', edgecolor='black')
        ax5.set_xlabel('Crack Density (cracks/mm²)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Crack Density Distribution')
        ax5.grid(True, alpha=0.3)
        
        # Young's modulus by phase
        ax6 = fig.add_subplot(gs[2, 2:])
        nano_data = self.post_mortem_data['nano_indentation']
        phases = [data['phase'] for data in nano_data]
        youngs_modulus = [data['youngs_modulus'] for data in nano_data]
        
        phase_means = {}
        for phase in set(phases):
            phase_moduli = [mod for mod, p in zip(youngs_modulus, phases) if p == phase]
            phase_means[phase] = np.mean(phase_moduli)
        
        bars = ax6.bar(phase_means.keys(), phase_means.values(), 
                      color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        ax6.set_ylabel('Young\'s Modulus (GPa)')
        ax6.set_title('Average Young\'s Modulus by Phase')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, phase_means.values()):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Statistics summary
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Calculate key statistics
        max_strain = max(max_strains)
        mean_stress = np.mean(stresses)
        total_cracks = sum(crack_densities)
        avg_youngs = np.mean(youngs_modulus)
        
        stats_text = f"""
        📊 EXPERIMENTAL DATA SUMMARY
        
        🔬 Digital Image Correlation (DIC):
        • Maximum strain during sintering: {max_strain:.4f}
        • Temperature range: 1200-1500°C
        • Strain hotspots detected: {sum(len(data['hotspot_locations']) for data in sintering_data)}
        
        🔬 Synchrotron X-ray Diffraction (XRD):
        • Average residual stress: {mean_stress:.1f} MPa
        • Stress measurement points: {len(residual_data)}
        • Lattice strain measurements: {len(lattice_data)}
        
        🔬 Post-Mortem Analysis:
        • Total crack density: {total_cracks:.2f} cracks/mm²
        • Average Young's modulus: {avg_youngs:.1f} GPa
        • Nano-indentation points: {len(nano_data)}
        • SEM images analyzed: {len(sem_data)}
        """
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.savefig(f'{self.data_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("🎨 Generating comprehensive visualizations...")
        
        self.load_data()
        
        print("\n1. Creating DIC strain map visualizations...")
        self.plot_dic_strain_maps()
        
        print("\n2. Creating XRD analysis visualizations...")
        self.plot_xrd_stress_profiles()
        
        print("\n3. Creating post-mortem analysis visualizations...")
        self.plot_post_mortem_analysis()
        
        print("\n4. Creating summary dashboard...")
        self.create_summary_dashboard()
        
        print("\n✅ All visualizations generated successfully!")
        print(f"📁 Plots saved to: {self.data_dir}/")

def main():
    """Main function to generate all visualizations"""
    visualizer = SOFCDataVisualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()