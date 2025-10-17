#!/usr/bin/env python3
"""
Baseline Rubberized Concrete Dataset - Analysis Script

This script loads, analyzes, and visualizes the comprehensive baseline dataset
for rubberized concrete characterization.

Author: Rubberized Concrete Fire Resistance Research Project
Date: 2025-10-17
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class RubberizedConcreteAnalyzer:
    """Class for analyzing rubberized concrete baseline dataset"""
    
    def __init__(self, data_dir='./'):
        """Initialize analyzer with data directory path"""
        self.data_dir = Path(data_dir)
        self.data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all CSV files into memory"""
        csv_files = {
            'mixture_proportions': '1_mixture_proportions.csv',
            'aggregate_grading': '2_aggregate_grading.csv',
            'rubber_characterization': '3_rubber_characterization.csv',
            'rubber_chemical': '4_rubber_chemical_composition.csv',
            'rubber_tga': '5_rubber_thermal_analysis.csv',
            'rubber_ftir': '6_rubber_ftir_peaks.csv',
            'fresh_properties': '7_fresh_state_properties.csv',
            'compressive_strength': '8_compressive_strength.csv',
            'tensile_strength': '9_tensile_splitting_strength.csv',
            'elastic_modulus': '10_modulus_of_elasticity.csv',
            'density_porosity': '11_density_porosity.csv',
            'pore_distribution': '12_pore_size_distribution_MIP.csv',
            'upv': '13_ultrasonic_pulse_velocity.csv',
            'microstructure': '14_microstructural_analysis.csv',
            'correlations': '15_property_correlations.csv',
            'thermal_properties': '16_thermal_properties_ambient.csv',
            'permeability': '17_permeability_durability.csv',
            'stress_strain': '18_stress_strain_curves.csv'
        }
        
        print("Loading dataset files...")
        for key, filename in csv_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self.data[key] = pd.read_csv(filepath)
                print(f"  ✓ Loaded: {filename}")
            else:
                print(f"  ✗ Missing: {filename}")
        print(f"\nTotal files loaded: {len(self.data)}/{len(csv_files)}\n")
    
    def summary_statistics(self):
        """Print summary statistics for key properties"""
        print("="*80)
        print("SUMMARY STATISTICS - 28-DAY MECHANICAL PROPERTIES")
        print("="*80)
        
        # Compressive strength
        comp_data = self.data['compressive_strength']
        comp_28d = comp_data[comp_data['Age_Days'] == 28]
        comp_summary = comp_28d.groupby('Mix_ID')['Compressive_Strength_MPa'].agg(['mean', 'std', 'min', 'max'])
        print("\n1. COMPRESSIVE STRENGTH (MPa) @ 28 days:")
        print(comp_summary.to_string())
        
        # Tensile strength
        tensile_data = self.data['tensile_strength']
        tensile_summary = tensile_data.groupby('Mix_ID')['Tensile_Splitting_Strength_MPa'].agg(['mean', 'std', 'min', 'max'])
        print("\n2. TENSILE SPLITTING STRENGTH (MPa) @ 28 days:")
        print(tensile_summary.to_string())
        
        # Elastic modulus
        modulus_data = self.data['elastic_modulus']
        modulus_summary = modulus_data.groupby('Mix_ID')['Static_Modulus_GPa'].agg(['mean', 'std', 'min', 'max'])
        print("\n3. ELASTIC MODULUS (GPa) @ 28 days:")
        print(modulus_summary.to_string())
        
        # Correlations
        corr_data = self.data['correlations']
        print("\n4. PROPERTY CORRELATIONS:")
        print(corr_data.to_string(index=False))
        
        print("\n" + "="*80 + "\n")
    
    def plot_mechanical_properties(self, save_path='mechanical_properties.png'):
        """Create comprehensive mechanical properties plot"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Mechanical Properties vs. Rubber Content', fontsize=16, fontweight='bold')
        
        rubber_content = [0, 5, 10, 15]
        corr_data = self.data['correlations']
        
        # 1. Compressive Strength
        ax = axes[0, 0]
        comp_strength = corr_data['Compressive_Strength_28d_MPa'].values
        ax.plot(rubber_content, comp_strength, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Rubber Content (%)', fontsize=11)
        ax.set_ylabel('Compressive Strength (MPa)', fontsize=11)
        ax.set_title('(a) Compressive Strength', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, (x, y) in enumerate(zip(rubber_content, comp_strength)):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        
        # 2. Tensile Strength
        ax = axes[0, 1]
        tensile_strength = corr_data['Tensile_Strength_28d_MPa'].values
        ax.plot(rubber_content, tensile_strength, 's-', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('Rubber Content (%)', fontsize=11)
        ax.set_ylabel('Tensile Splitting Strength (MPa)', fontsize=11)
        ax.set_title('(b) Tensile Splitting Strength', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, (x, y) in enumerate(zip(rubber_content, tensile_strength)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        
        # 3. Elastic Modulus
        ax = axes[0, 2]
        modulus = corr_data['Modulus_Elasticity_GPa'].values
        ax.plot(rubber_content, modulus, '^-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Rubber Content (%)', fontsize=11)
        ax.set_ylabel('Elastic Modulus (GPa)', fontsize=11)
        ax.set_title('(c) Static Elastic Modulus', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, (x, y) in enumerate(zip(rubber_content, modulus)):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        
        # 4. Density
        ax = axes[1, 0]
        density = corr_data['Density_kg_m3'].values
        ax.plot(rubber_content, density, 'D-', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Rubber Content (%)', fontsize=11)
        ax.set_ylabel('Density (kg/m³)', fontsize=11)
        ax.set_title('(d) Oven-Dry Density', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, (x, y) in enumerate(zip(rubber_content, density)):
            ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        
        # 5. Porosity
        ax = axes[1, 1]
        porosity = corr_data['Porosity_Percent'].values
        ax.plot(rubber_content, porosity, 'v-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Rubber Content (%)', fontsize=11)
        ax.set_ylabel('Total Porosity (%)', fontsize=11)
        ax.set_title('(e) Total Porosity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, (x, y) in enumerate(zip(rubber_content, porosity)):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        
        # 6. Ductility Index
        ax = axes[1, 2]
        ductility = corr_data['Ductility_Index'].values
        ax.plot(rubber_content, ductility, 'p-', linewidth=2, markersize=8, color='brown')
        ax.set_xlabel('Rubber Content (%)', fontsize=11)
        ax.set_ylabel('Ductility Index (Relative)', fontsize=11)
        ax.set_title('(f) Ductility Index', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        for i, (x, y) in enumerate(zip(rubber_content, ductility)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_stress_strain_curves(self, save_path='stress_strain_curves.png'):
        """Plot complete stress-strain behavior"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        stress_strain = self.data['stress_strain']
        
        colors = {'RC-00': 'blue', 'RC-05': 'orange', 'RC-10': 'green', 'RC-15': 'red'}
        labels = {'RC-00': 'RC-00 (0% rubber)', 'RC-05': 'RC-05 (5% rubber)',
                  'RC-10': 'RC-10 (10% rubber)', 'RC-15': 'RC-15 (15% rubber)'}
        
        for mix_id in ['RC-00', 'RC-05', 'RC-10', 'RC-15']:
            mix_data = stress_strain[stress_strain['Mix_ID'] == mix_id]
            ax.plot(mix_data['Strain_Microstrain'], mix_data['Stress_MPa'], 
                    linewidth=2.5, color=colors[mix_id], label=labels[mix_id])
        
        ax.set_xlabel('Strain (με)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
        ax.set_title('Compressive Stress-Strain Curves @ 28 Days', fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 7000)
        ax.set_ylim(0, 60)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_rubber_tga(self, save_path='rubber_tga.png'):
        """Plot thermogravimetric analysis of rubber"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        tga_data = self.data['rubber_tga']
        
        # TGA curve
        ax1.plot(tga_data['Temperature_C'], tga_data['Weight_Loss_Percent'], 
                 'b-', linewidth=2.5, label='Mass Loss')
        ax1.set_xlabel('Temperature (°C)', fontsize=12)
        ax1.set_ylabel('Mass Loss (%)', fontsize=12, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('(a) Thermogravimetric Analysis (TGA) of Crumb Rubber', 
                      fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # DTG curve (derivative)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(tga_data['Temperature_C'], tga_data['DTG_Peak_Percent_per_C']*100, 
                      'r--', linewidth=2, label='DTG (dm/dT)')
        ax1_twin.set_ylabel('Mass Loss Rate (%/°C) × 100', fontsize=12, color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1_twin.legend(loc='upper right')
        
        # Annotate key temperatures
        key_temps = [250, 350, 500]
        for temp in key_temps:
            idx = (tga_data['Temperature_C'] - temp).abs().idxmin()
            mass_loss = tga_data.loc[idx, 'Weight_Loss_Percent']
            ax1.annotate(f'{temp}°C\n({mass_loss:.1f}%)', 
                        xy=(temp, mass_loss), xytext=(temp+30, mass_loss-5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                        fontsize=10, fontweight='bold')
        
        # Heat flow (DSC)
        ax2.plot(tga_data['Temperature_C'], tga_data['Heat_Flow_mW'], 
                 'g-', linewidth=2.5)
        ax2.set_xlabel('Temperature (°C)', fontsize=12)
        ax2.set_ylabel('Heat Flow (mW)', fontsize=12)
        ax2.set_title('(b) Differential Scanning Calorimetry (DSC)', 
                      fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_pore_distribution(self, save_path='pore_distribution.png'):
        """Plot pore size distribution from MIP"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        pore_data = self.data['pore_distribution']
        
        mixes = ['RC00', 'RC05', 'RC10', 'RC15']
        labels = ['RC-00 (0%)', 'RC-05 (5%)', 'RC-10 (10%)', 'RC-15 (15%)']
        colors = ['blue', 'orange', 'green', 'red']
        
        # Cumulative intrusion
        for mix, label, color in zip(mixes, labels, colors):
            ax1.semilogx(pore_data['Pore_Diameter_nm'], 
                        pore_data[f'{mix}_Cumulative_Intrusion_ml_g'],
                        linewidth=2.5, label=label, color=color)
        
        ax1.set_xlabel('Pore Diameter (nm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Intrusion (ml/g)', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Cumulative Pore Volume', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Add shaded regions for pore classifications
        ax1.axvspan(2, 10, alpha=0.1, color='yellow', label='Gel pores')
        ax1.axvspan(10, 50000, alpha=0.1, color='cyan', label='Capillary pores')
        ax1.axvspan(50000, 500000, alpha=0.1, color='pink', label='Macro pores')
        
        # Incremental intrusion (pore size distribution)
        for mix, label, color in zip(mixes, labels, colors):
            ax2.semilogx(pore_data['Pore_Diameter_nm'], 
                        pore_data[f'{mix}_Incremental_Intrusion_ml_g'],
                        linewidth=2.5, label=label, color=color)
        
        ax2.set_xlabel('Pore Diameter (nm)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Incremental Intrusion (ml/g)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Pore Size Distribution', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\nGenerating visualization plots...")
        self.plot_mechanical_properties()
        self.plot_stress_strain_curves()
        self.plot_rubber_tga()
        self.plot_pore_distribution()
        print("\n✓ All plots generated successfully!\n")
    
    def export_summary_report(self, filename='analysis_summary.txt'):
        """Export comprehensive text summary"""
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BASELINE RUBBERIZED CONCRETE DATASET - ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Mix designs
            f.write("1. MIXTURE DESIGNS\n")
            f.write("-" * 80 + "\n")
            mix_data = self.data['mixture_proportions']
            for col in ['Mix_ID', 'Cement_kg_m3', 'Fine_Aggregate_kg_m3', 
                       'Rubber_Aggregate_kg_m3', 'Water_kg_m3', 'Water_Cement_Ratio']:
                if col in mix_data.columns:
                    f.write(f"\n{col}:\n")
                    f.write(mix_data[col].to_string(index=False) + "\n")
            
            # Mechanical properties
            f.write("\n\n2. MECHANICAL PROPERTIES @ 28 DAYS\n")
            f.write("-" * 80 + "\n")
            corr_data = self.data['correlations']
            f.write(corr_data.to_string(index=False))
            
            # Rubber characterization
            f.write("\n\n3. RUBBER CHARACTERIZATION\n")
            f.write("-" * 80 + "\n")
            rubber_data = self.data['rubber_characterization']
            key_props = ['Specific_Gravity', 'Water_Absorption_24h', 'Hardness_Shore_A']
            for prop in key_props:
                row = rubber_data[rubber_data['Property'] == prop]
                if not row.empty:
                    f.write(f"{prop}: {row['Value'].values[0]} {row['Unit'].values[0]}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF SUMMARY REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Saved: {filename}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BASELINE RUBBERIZED CONCRETE DATASET - ANALYSIS TOOL")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = RubberizedConcreteAnalyzer('./')
    
    # Display summary statistics
    analyzer.summary_statistics()
    
    # Generate visualizations
    analyzer.generate_all_plots()
    
    # Export summary report
    analyzer.export_summary_report()
    
    print("="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    print("Generated files:")
    print("  • mechanical_properties.png")
    print("  • stress_strain_curves.png")
    print("  • rubber_tga.png")
    print("  • pore_distribution.png")
    print("  • analysis_summary.txt")
    print("\n")


if __name__ == "__main__":
    main()
