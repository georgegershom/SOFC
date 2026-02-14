#!/usr/bin/env python3
"""
Visualization script for material properties

Generates publication-quality figures:
1. Young's modulus vs. Temperature
2. CTE comparison bar chart
3. Fracture toughness hierarchy
4. Chemical expansion strain fields
5. Interface properties comparison

Usage:
    python visualize_properties.py --input data/ --output figures/

Author: Generated for SOFC Fracture Dataset
Date: February 13, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path


# Set publication-quality defaults
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['figure.dpi'] = 300


class MaterialVisualizer:
    """Generate visualization figures for material database"""
    
    def __init__(self, data_dir: str):
        """
        Initialize visualizer with data directory
        
        Parameters:
        -----------
        data_dir : str or Path
            Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.master_df = pd.read_csv(self.data_dir / 'master_material_database.csv')
        self.thermoelastic_df = pd.read_csv(self.data_dir / 'thermoelastic_properties.csv')
        self.fracture_df = pd.read_csv(self.data_dir / 'fracture_cohesive_properties.csv')
        self.chemical_df = pd.read_csv(self.data_dir / 'chemical_expansion_data.csv')
        
        # Color scheme
        self.colors = {
            'YSZ': '#2E86AB',  # Blue
            '8YSZ': '#2E86AB',
            'GDC': '#A23B72',  # Purple
            'GDC10': '#A23B72',
            'LSCF': '#F18F01', # Orange
        }
    
    def plot_elastic_modulus_temperature(self, output_path: Path) -> None:
        """Plot Young's modulus vs. Temperature for all materials"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        materials = ['8YSZ', 'GDC10', 'LSCF']
        labels = ['8YSZ (Electrolyte)', 'GDC10 (Interlayer)', 'LSCF (Cathode)']
        
        for material, label in zip(materials, labels):
            data = self.thermoelastic_df[
                (self.thermoelastic_df['Material'] == material) &
                (self.thermoelastic_df['Property'] == 'Youngs_Modulus')
            ].sort_values('Temperature_C')
            
            if len(data) > 0:
                temps = data['Temperature_C'].values
                E_values = data['Value'].values
                
                ax.plot(temps, E_values, 'o-', linewidth=2.5, markersize=8,
                       color=self.colors[material], label=label)
        
        ax.set_xlabel('Temperature (°C)', fontsize=13, fontweight='bold')
        ax.set_ylabel("Young's Modulus (GPa)", fontsize=13, fontweight='bold')
        ax.set_title('Temperature Dependence of Elastic Modulus', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 850)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {output_path.name}")
    
    def plot_cte_comparison(self, output_path: Path) -> None:
        """Bar chart comparing CTEs"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        materials = ['8YSZ', 'GDC10', 'LSCF']
        labels = ['8YSZ', 'GDC10', 'LSCF']
        cte_values = []
        
        for material in materials:
            cte = self.thermoelastic_df[
                (self.thermoelastic_df['Material'] == material) &
                (self.thermoelastic_df['Property'] == 'CTE')
            ]['Value'].values
            
            if len(cte) > 0:
                cte_values.append(cte[0])
            else:
                cte_values.append(0)
        
        x_pos = np.arange(len(labels))
        colors_list = [self.colors[m] for m in materials]
        
        bars = ax.bar(x_pos, cte_values, color=colors_list, alpha=0.8,
                     edgecolor='black', linewidth=1.5, width=0.6)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, cte_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
        
        ax.set_ylabel('CTE (ppm/K)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Material', fontsize=13, fontweight='bold')
        ax.set_title('Coefficient of Thermal Expansion (25-800°C)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, max(cte_values) * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add annotation for mismatch
        delta_alpha = cte_values[2] - cte_values[0]  # LSCF - YSZ
        ax.annotate(f'Δα = {delta_alpha:.1f} ppm/K',
                   xy=(0.5, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=11, ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {output_path.name}")
    
    def plot_fracture_toughness_hierarchy(self, output_path: Path) -> None:
        """Plot fracture toughness hierarchy (bulk and interfaces)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel A: Bulk fracture toughness
        bulk_data = self.fracture_df[
            self.fracture_df['Interface/Material'].isin(['8YSZ', 'GDC10', 'LSCF'])
        ].copy()
        
        materials = []
        Gc_bulk = []
        for mat in ['8YSZ', 'GDC10', 'LSCF']:
            data = bulk_data[bulk_data['Interface/Material'] == mat]
            Gc_val = data[data['Property'] == 'Bulk_Fracture_Toughness']['Value'].values
            if len(Gc_val) > 0:
                materials.append(mat)
                Gc_bulk.append(Gc_val[0])
        
        x_pos = np.arange(len(materials))
        colors_list = [self.colors[m] for m in materials]
        
        bars1 = ax1.bar(x_pos, Gc_bulk, color=colors_list, alpha=0.8,
                       edgecolor='black', linewidth=1.5, width=0.6)
        
        for bar, val in zip(bars1, Gc_bulk):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        
        ax1.set_ylabel('Fracture Toughness $G_c$ (J/m²)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Material', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Bulk Fracture Toughness', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(materials, fontsize=11)
        ax1.set_ylim(0, max(Gc_bulk) * 1.25)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Panel B: Interface toughness (Mode I and II)
        interfaces = ['YSZ/GDC', 'GDC/LSCF']
        Gc_I_values = []
        Gc_II_values = []
        
        for iface in interfaces:
            iface_symbol = iface.replace('/', '_')
            Gc_I = self.fracture_df[
                (self.fracture_df['Interface/Material'] == iface) &
                (self.fracture_df['Mode'] == 'I')
            ]['Value'].values
            Gc_II = self.fracture_df[
                (self.fracture_df['Interface/Material'] == iface) &
                (self.fracture_df['Mode'] == 'II')
            ]['Value'].values
            
            if len(Gc_I) > 0:
                Gc_I_values.append(Gc_I[0])
            else:
                Gc_I_values.append(0)
            
            if len(Gc_II) > 0:
                Gc_II_values.append(Gc_II[0])
            else:
                Gc_II_values.append(0)
        
        x_pos = np.arange(len(interfaces))
        width = 0.35
        
        bars_I = ax2.bar(x_pos - width/2, Gc_I_values, width, label='Mode I',
                        color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars_II = ax2.bar(x_pos + width/2, Gc_II_values, width, label='Mode II',
                         color='#E24A4A', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars_I, Gc_I_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)
        
        for bar, val in zip(bars_II, Gc_II_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)
        
        ax2.set_ylabel('Interface Toughness $G_c$ (J/m²)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Interface', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Interface Fracture Toughness', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(interfaces, fontsize=11)
        ax2.legend(loc='upper right', fontsize=10, frameon=True)
        ax2.set_ylim(0, max(max(Gc_I_values), max(Gc_II_values)) * 1.25)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {output_path.name}")
    
    def plot_chemical_expansion_schematic(self, output_path: Path) -> None:
        """Plot chemical expansion strain fields"""
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
        
        # Panel A: GDC isotropic expansion
        ax1 = fig.add_subplot(gs[0, 0])
        
        beta_iso_GDC = 0.025
        delta_delta_GDC = 0.05
        eps_chem_GDC = beta_iso_GDC * delta_delta_GDC * 100
        
        pO2_values = np.logspace(-3, -0.7, 50)  # 0.001 to 0.2 atm
        delta_values = 0.05 + 0.025 * np.log10(0.21 / pO2_values)  # Simplified model
        eps_values = beta_iso_GDC * delta_values * 100
        
        ax1.plot(pO2_values, eps_values, linewidth=2.5, color=self.colors['GDC'])
        ax1.set_xlabel('$pO_2$ (atm)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Chemical Strain (%)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) GDC Isotropic Chemical Expansion', fontsize=12, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(eps_chem_GDC, color='red', linestyle='--', linewidth=1.5,
                   label=f'Cathode: {eps_chem_GDC:.3f}%')
        ax1.legend(fontsize=9)
        
        # Panel B: LSCF anisotropic expansion
        ax2 = fig.add_subplot(gs[0, 1])
        
        beta_11_LSCF = 0.08
        beta_33_LSCF = 0.12
        delta_delta_LSCF = 0.15
        eps_11 = beta_11_LSCF * delta_delta_LSCF * 100
        eps_33 = beta_33_LSCF * delta_delta_LSCF * 100
        
        directions = ['a-axis\n(in-plane)', 'c-axis\n(out-of-plane)']
        strains = [eps_11, eps_33]
        colors_dir = ['#4A90E2', '#E24A4A']
        
        bars = ax2.bar(directions, strains, color=colors_dir, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, strains):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{val:.3f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        
        ax2.set_ylabel('Chemical Strain (%)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) LSCF Anisotropic Chemical Expansion', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, max(strains) * 1.3)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Panel C: Tri-layer schematic with expansion
        ax3 = fig.add_subplot(gs[1, :])
        
        # Draw tri-layer structure
        layer_y = [0, 1, 1.5, 4.5]  # Scaled for visualization
        layer_names = ['8YSZ\n(10 μm)', 'GDC\n(5 μm)', 'LSCF\n(30 μm)']
        layer_colors = [self.colors['8YSZ'], self.colors['GDC'], self.colors['LSCF']]
        
        for i in range(len(layer_names)):
            y_bottom = layer_y[i]
            y_top = layer_y[i+1]
            
            rect = mpatches.Rectangle((0, y_bottom), 10, y_top - y_bottom,
                                     facecolor=layer_colors[i], alpha=0.6,
                                     edgecolor='black', linewidth=2)
            ax3.add_patch(rect)
            
            # Add layer label
            y_center = (y_bottom + y_top) / 2
            ax3.text(5, y_center, layer_names[i], ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Add expansion arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='red')
        
        # Thermal expansion arrows (left side)
        ax3.annotate('', xy=(11, 0.5), xytext=(12, 0.5), arrowprops=arrow_props)
        ax3.text(13, 0.5, 'α = 10.5 ppm/K', va='center', fontsize=9)
        
        ax3.annotate('', xy=(11, 1.25), xytext=(12.3, 1.25), arrowprops=arrow_props)
        ax3.text(13, 1.25, 'α = 12.5 ppm/K', va='center', fontsize=9)
        
        ax3.annotate('', xy=(11, 3), xytext=(13.5, 3), arrowprops=arrow_props)
        ax3.text(14.2, 3, 'α = 15.8 ppm/K', va='center', fontsize=9)
        
        # Chemical expansion arrows (right side - only for GDC and LSCF)
        chem_arrow_props = dict(arrowstyle='->', lw=2, color='blue')
        
        ax3.annotate('', xy=(-1, 1.25), xytext=(-1.5, 1.25), arrowprops=chem_arrow_props)
        ax3.text(-3, 1.25, f'ε_chem = {eps_chem_GDC:.3f}%', va='center', fontsize=9, color='blue')
        
        ax3.annotate('', xy=(-1, 3), xytext=(-2, 3), arrowprops=chem_arrow_props)
        ax3.text(-3, 3, f'ε_chem = {eps_33:.3f}%', va='center', fontsize=9, color='blue')
        
        ax3.set_xlim(-4, 15)
        ax3.set_ylim(-0.5, 5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title('(c) Tri-Layer Structure with Thermal and Chemical Expansion',
                     fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {output_path.name}")
    
    def plot_validation_data(self, output_path: Path) -> None:
        """Plot experimental validation data"""
        try:
            validation_df = pd.read_csv(self.data_dir / 'experimental_validation_data.csv')
        except:
            print("⚠️  Validation data file not found, skipping...")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel A: Curvature evolution
        curvature_data = validation_df[validation_df['Measurement'] == 'Global_Curvature']
        
        if len(curvature_data) > 0:
            temps = curvature_data['Temperature_C'].values
            kappas = curvature_data['Value'].values * 1000  # Convert to 1/m
            
            # Sort by temperature
            sorted_idx = np.argsort(temps)
            temps = temps[sorted_idx]
            kappas = kappas[sorted_idx]
            
            ax1.plot(temps, kappas, 'o-', linewidth=2.5, markersize=10,
                    color='#2E86AB', markerfacecolor='white', markeredgewidth=2)
            
            ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Curvature κ (m⁻¹)', fontsize=12, fontweight='bold')
            ax1.set_title('(a) Global Curvature Evolution', fontsize=13, fontweight='bold', pad=10)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.invert_xaxis()  # Cooldown direction
            
            # Add annotation
            ax1.annotate('Cooling', xy=(0.5, 0.95), xycoords='axes fraction',
                        fontsize=11, ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Panel B: Delamination data
        delam_data = validation_df[validation_df['Measurement'] == 'Delamination_Length']
        
        if len(delam_data) > 0:
            sample_ids = delam_data['Sample_ID'].values
            L_delam = delam_data['Value'].astype(float).values
            
            x_pos = np.arange(len(sample_ids))
            bars = ax2.bar(x_pos, L_delam, color='#F18F01', alpha=0.8,
                          edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, L_delam):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{float(val):.2f} mm', ha='center', va='bottom',
                        fontweight='bold', fontsize=10)
            
            ax2.set_ylabel('Delamination Length (mm)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Sample ID', fontsize=12, fontweight='bold')
            ax2.set_title('(b) Delamination at GDC/LSCF Interface (RT)',
                         fontsize=13, fontweight='bold', pad=10)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(sample_ids, fontsize=10)
            ax2.set_ylim(0, max(L_delam) * 1.3)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {output_path.name}")
    
    def generate_all_figures(self, output_dir: Path) -> None:
        """
        Generate all visualization figures
        
        Parameters:
        -----------
        output_dir : Path
            Output directory for figures
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating Visualization Figures...")
        print("=" * 60)
        
        self.plot_elastic_modulus_temperature(output_dir / 'fig1_elastic_modulus_vs_temperature.png')
        self.plot_cte_comparison(output_dir / 'fig2_cte_comparison.png')
        self.plot_fracture_toughness_hierarchy(output_dir / 'fig3_fracture_toughness.png')
        self.plot_chemical_expansion_schematic(output_dir / 'fig4_chemical_expansion.png')
        self.plot_validation_data(output_dir / 'fig5_validation_data.png')
        
        print("=" * 60)
        print(f"✓ All figures saved to: {output_dir}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate visualization figures from material database'
    )
    parser.add_argument('--input', '-i',
                       default='data/',
                       help='Input data directory path')
    parser.add_argument('--output', '-o',
                       default='figures/',
                       help='Output figures directory path')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1
    
    # Generate figures
    print(f"Reading data from: {input_dir}")
    visualizer = MaterialVisualizer(input_dir)
    visualizer.generate_all_figures(Path(args.output))
    
    print("\n✓ Success! Figures ready for publication.")
    
    return 0


if __name__ == '__main__':
    exit(main())
