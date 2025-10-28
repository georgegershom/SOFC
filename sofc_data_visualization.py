#!/usr/bin/env python3
"""
SOFC Materials Properties Data Visualization and Analysis Tool

This script provides comprehensive visualization and analysis capabilities for SOFC materials data.
It loads the generated datasets and creates various plots to explore material properties.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SOFCDataAnalyzer:
    """Class for analyzing and visualizing SOFC materials data."""
    
    def __init__(self):
        """Initialize the analyzer and load all datasets."""
        self.data_dir = Path('/workspace')
        self.load_data()
        
    def load_data(self):
        """Load all JSON and CSV datasets."""
        print("Loading SOFC materials datasets...")
        
        # Load JSON files
        with open(self.data_dir / 'sofc_materials_thermal_mechanical.json', 'r') as f:
            self.thermal_mechanical = json.load(f)
            
        with open(self.data_dir / 'sofc_creep_parameters.json', 'r') as f:
            self.creep_params = json.load(f)
            
        with open(self.data_dir / 'sofc_plasticity_parameters.json', 'r') as f:
            self.plasticity_params = json.load(f)
            
        with open(self.data_dir / 'sofc_electrochemical_properties.json', 'r') as f:
            self.electrochemical = json.load(f)
            
        # Load CSV file
        self.temp_dependent = pd.read_csv(self.data_dir / 'sofc_temperature_dependent_properties.csv')
        
        print("✓ All datasets loaded successfully!")
        
    def plot_thermal_expansion_comparison(self):
        """Plot and compare thermal expansion coefficients of different materials."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract TEC values at different temperatures
        materials_data = {}
        for material, props in self.thermal_mechanical['materials'].items():
            if 'thermal_properties' in props:
                tec_data = props['thermal_properties']['thermal_expansion_coefficient']
                if 'temperature_dependent' in tec_data:
                    temps = list(tec_data['temperature_dependent'].keys())
                    values = list(tec_data['temperature_dependent'].values())
                    materials_data[material.replace('_', ' ')] = {
                        'temps': [float(t) for t in temps],
                        'values': [v * 1e6 for v in values]  # Convert to 10^-6/K
                    }
        
        # Plot temperature-dependent TEC
        for material, data in materials_data.items():
            ax1.plot(data['temps'], data['values'], marker='o', label=material, linewidth=2)
        
        ax1.set_xlabel('Temperature (K)', fontsize=12)
        ax1.set_ylabel('TEC (×10⁻⁶ K⁻¹)', fontsize=12)
        ax1.set_title('Temperature-Dependent Thermal Expansion Coefficients', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Bar chart comparison at 1073K
        tec_at_1073 = {}
        for material, data in materials_data.items():
            if 1073 in data['temps']:
                idx = data['temps'].index(1073)
                tec_at_1073[material] = data['values'][idx]
        
        materials = list(tec_at_1073.keys())
        values = list(tec_at_1073.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(materials)))
        
        bars = ax2.bar(range(len(materials)), values, color=colors)
        ax2.set_xticks(range(len(materials)))
        ax2.set_xticklabels(materials, rotation=45, ha='right')
        ax2.set_ylabel('TEC at 1073K (×10⁻⁶ K⁻¹)', fontsize=12)
        ax2.set_title('TEC Comparison at Operating Temperature (1073K)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'thermal_expansion_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Thermal expansion comparison plot saved!")
        
    def plot_conductivity_arrhenius(self):
        """Plot Arrhenius plots for ionic and electronic conductivities."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Temperature range
        temps_K = np.array([773, 873, 973, 1073, 1173, 1273])
        temps_inv = 1000 / temps_K  # 1000/T for better scale
        
        # Ionic conductivity data
        ionic_materials = {
            '8YSZ': [0.008, 0.018, 0.036, 0.056, 0.078, 0.100],
            'GDC': [0.015, 0.035, 0.065, 0.095, 0.125, 0.155],
            'LSCF': [0.04, 0.10, 0.16, 0.22, 0.28, 0.35]
        }
        
        for material, conductivities in ionic_materials.items():
            log_cond = np.log10(conductivities)
            ax1.plot(temps_inv, log_cond, marker='o', label=material, linewidth=2, markersize=8)
            
            # Fit line for activation energy
            z = np.polyfit(temps_inv, log_cond, 1)
            p = np.poly1d(z)
            ax1.plot(temps_inv, p(temps_inv), '--', alpha=0.5)
        
        ax1.set_xlabel('1000/T (K⁻¹)', fontsize=12)
        ax1.set_ylabel('log₁₀(σ) [S/cm]', fontsize=12)
        ax1.set_title('Ionic Conductivity - Arrhenius Plot', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        # Electronic conductivity data
        electronic_materials = {
            'Ni-YSZ': [3600, 3400, 3200, 3000, 2800, 2600],
            'LSM': [110, 120, 130, 135, 138, 140],
            'LSCF': [295, 308, 315, 320, 322, 324],
            'Crofer22APU': [13200, 12700, 12500, 12200, 11800, 11400]
        }
        
        for material, conductivities in electronic_materials.items():
            log_cond = np.log10(conductivities)
            ax2.plot(temps_inv, log_cond, marker='s', label=material, linewidth=2, markersize=8)
        
        ax2.set_xlabel('1000/T (K⁻¹)', fontsize=12)
        ax2.set_ylabel('log₁₀(σ) [S/cm]', fontsize=12)
        ax2.set_title('Electronic Conductivity - Temperature Dependence', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'conductivity_arrhenius.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Conductivity Arrhenius plots saved!")
        
    def plot_mechanical_properties(self):
        """Plot mechanical properties comparison."""
        fig = plt.figure(figsize=(15, 10))
        
        # Young's modulus vs temperature
        ax1 = plt.subplot(2, 2, 1)
        temp_data = self.temp_dependent
        materials = ['Ni-YSZ_Anode', '8YSZ_Electrolyte', 'LSM_Cathode', 'Crofer22APU']
        
        for material in materials:
            material_data = temp_data[temp_data['Material'] == material]
            youngs_data = material_data[material_data['Property'] == 'Youngs_Modulus']
            if not youngs_data.empty:
                temps = [float(col.split('_')[1].replace('K', '')) for col in youngs_data.columns if col.startswith('T_')]
                values = [youngs_data[f'T_{int(t)}K'].values[0] for t in temps]
                ax1.plot(temps, values, marker='o', label=material.replace('_', ' '), linewidth=2)
        
        ax1.set_xlabel('Temperature (K)', fontsize=12)
        ax1.set_ylabel("Young's Modulus (GPa)", fontsize=12)
        ax1.set_title("Young's Modulus vs Temperature", fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Yield stress vs temperature for metals
        ax2 = plt.subplot(2, 2, 2)
        metals = ['Ni-YSZ_Anode', 'Crofer22APU']
        
        for material in metals:
            material_data = temp_data[temp_data['Material'] == material]
            yield_data = material_data[material_data['Property'] == 'Yield_Stress']
            if not yield_data.empty:
                temps = [float(col.split('_')[1].replace('K', '')) for col in yield_data.columns if col.startswith('T_')]
                values = [yield_data[f'T_{int(t)}K'].values[0] for t in temps]
                ax2.plot(temps, values, marker='s', label=material.replace('_', ' '), linewidth=2)
        
        ax2.set_xlabel('Temperature (K)', fontsize=12)
        ax2.set_ylabel('Yield Stress (MPa)', fontsize=12)
        ax2.set_title('Yield Stress vs Temperature', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Creep parameters comparison
        ax3 = plt.subplot(2, 2, 3)
        creep_n_values = {}
        for material, props in self.creep_params['materials'].items():
            if 'Norton_Bailey_Parameters' in props:
                n_data = props['Norton_Bailey_Parameters']['primary_creep']['n']
                if isinstance(n_data, dict):
                    n_value = n_data['value']
                else:
                    n_value = n_data
                creep_n_values[material.replace('_', ' ')] = n_value
        
        materials = list(creep_n_values.keys())
        n_values = list(creep_n_values.values())
        colors = plt.cm.Spectral(np.linspace(0, 1, len(materials)))
        
        bars = ax3.bar(range(len(materials)), n_values, color=colors)
        ax3.set_xticks(range(len(materials)))
        ax3.set_xticklabels(materials, rotation=45, ha='right')
        ax3.set_ylabel('Stress Exponent n', fontsize=12)
        ax3.set_title('Norton-Bailey Creep Stress Exponent', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, n_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Activation energies
        ax4 = plt.subplot(2, 2, 4)
        activation_energies = {}
        for material, props in self.creep_params['materials'].items():
            if 'Norton_Bailey_Parameters' in props:
                Q_data = props['Norton_Bailey_Parameters']['primary_creep']['Q']
                if isinstance(Q_data, dict):
                    Q_value = Q_data['value']
                else:
                    Q_value = Q_data
                activation_energies[material.replace('_', ' ')] = Q_value
        
        materials = list(activation_energies.keys())
        Q_values = list(activation_energies.values())
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(materials)))
        
        bars = ax4.bar(range(len(materials)), Q_values, color=colors)
        ax4.set_xticks(range(len(materials)))
        ax4.set_xticklabels(materials, rotation=45, ha='right')
        ax4.set_ylabel('Activation Energy (kJ/mol)', fontsize=12)
        ax4.set_title('Creep Activation Energy', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, Q_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'mechanical_properties.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Mechanical properties comparison saved!")
        
    def plot_electrochemical_performance(self):
        """Plot electrochemical performance parameters."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Exchange current densities
        materials_i0 = {
            'Ni-YSZ (H₂)': 0.53,
            'LSM (O₂)': 0.15,
            'LSCF (O₂)': 0.48
        }
        
        materials = list(materials_i0.keys())
        i0_values = list(materials_i0.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax1.bar(materials, i0_values, color=colors)
        ax1.set_ylabel('Exchange Current Density (A/cm²)', fontsize=12)
        ax1.set_title('Exchange Current Densities at 1073K', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, i0_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Area specific resistance breakdown
        asr_components = {
            'Anode': 0.15,
            'Electrolyte': 0.10,
            'Cathode': 0.20,
            'Interconnect': 0.03,
            'Contact': 0.02
        }
        
        labels = list(asr_components.keys())
        sizes = list(asr_components.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        explode = (0.05, 0.05, 0.05, 0, 0)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax2.set_title('ASR Breakdown at 1073K', fontsize=14, fontweight='bold')
        
        # Charge transfer resistance vs temperature
        temps = np.array([973, 1023, 1073, 1123, 1173])
        lsm_ctr = np.array([0.65, 0.48, 0.35, 0.25, 0.18])
        lscf_ctr = np.array([0.35, 0.22, 0.10, 0.06, 0.04])
        
        ax3.plot(temps, lsm_ctr, marker='o', label='LSM', linewidth=2, markersize=8)
        ax3.plot(temps, lscf_ctr, marker='s', label='LSCF', linewidth=2, markersize=8)
        ax3.set_xlabel('Temperature (K)', fontsize=12)
        ax3.set_ylabel('Charge Transfer Resistance (Ω·cm²)', fontsize=12)
        ax3.set_title('Cathode Charge Transfer Resistance', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Degradation rates
        degradation_data = {
            'Initial\n(0-1000h)': 2.0,
            'Steady State\n(>1000h)': 0.5
        }
        
        periods = list(degradation_data.keys())
        rates = list(degradation_data.values())
        colors = ['#FF6B6B', '#95E77E']
        
        bars = ax4.bar(periods, rates, color=colors, width=0.6)
        ax4.set_ylabel('Voltage Degradation Rate (%/1000h)', fontsize=12)
        ax4.set_title('Typical SOFC Degradation Rates', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 2.5)
        
        for bar, val in zip(bars, rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'electrochemical_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Electrochemical performance plots saved!")
        
    def generate_summary_report(self):
        """Generate a summary report of key material properties."""
        print("\n" + "="*60)
        print("SOFC MATERIALS PROPERTIES SUMMARY REPORT")
        print("="*60)
        
        # Operating conditions
        print("\n📊 TYPICAL OPERATING CONDITIONS:")
        print("  • Temperature: 973-1273 K (700-1000°C)")
        print("  • Pressure: 1-10 atm")
        print("  • Fuel: H₂, CO, CH₄ (reformed)")
        print("  • Oxidant: Air or O₂")
        
        # Material summary
        print("\n🔬 KEY MATERIAL PROPERTIES AT 1073K:")
        
        materials_summary = {
            'Ni-YSZ Anode': {
                'TEC': '12.8 ×10⁻⁶ K⁻¹',
                'E': '75 GPa',
                'σ_ionic': '0.05 S/cm',
                'σ_electronic': '3400 S/cm',
                'Porosity': '30-40%'
            },
            '8YSZ Electrolyte': {
                'TEC': '10.6 ×10⁻⁶ K⁻¹',
                'E': '195 GPa',
                'σ_ionic': '0.056 S/cm',
                'σ_electronic': '~10⁻⁸ S/cm',
                'Density': '>95%'
            },
            'LSM Cathode': {
                'TEC': '12.0 ×10⁻⁶ K⁻¹',
                'E': '66 GPa',
                'σ_electronic': '130 S/cm',
                'i₀': '0.15 A/cm²',
                'Porosity': '30-40%'
            },
            'LSCF Cathode': {
                'TEC': '15.7 ×10⁻⁶ K⁻¹',
                'E': '62 GPa',
                'σ_electronic': '315 S/cm',
                'σ_ionic': '0.16 S/cm',
                'i₀': '0.48 A/cm²'
            }
        }
        
        for material, props in materials_summary.items():
            print(f"\n  {material}:")
            for prop, value in props.items():
                print(f"    • {prop}: {value}")
        
        # Performance targets
        print("\n🎯 PERFORMANCE TARGETS:")
        print("  • Power Density: >0.5 W/cm² at 0.7V")
        print("  • Total ASR: <0.5 Ω·cm²")
        print("  • Fuel Utilization: 70-90%")
        print("  • Lifetime: >40,000 hours (stationary)")
        print("  • Degradation Rate: <0.5%/1000h (steady state)")
        
        # Critical issues
        print("\n⚠️ CRITICAL DESIGN CONSIDERATIONS:")
        print("  • TEC Mismatch: Must be <2×10⁻⁶ K⁻¹ between layers")
        print("  • Ni Coarsening: Major degradation mechanism in anode")
        print("  • Cr Poisoning: From interconnect, affects cathode")
        print("  • Thermal Cycling: Causes mechanical failure")
        print("  • Redox Cycling: Ni oxidation/reduction in anode")
        
        print("\n" + "="*60)
        print("✅ Report generation complete!")
        print("="*60)
        
    def run_all_analyses(self):
        """Run all analysis and visualization functions."""
        print("\n🚀 Starting comprehensive SOFC data analysis...\n")
        
        self.plot_thermal_expansion_comparison()
        self.plot_conductivity_arrhenius()
        self.plot_mechanical_properties()
        self.plot_electrochemical_performance()
        self.generate_summary_report()
        
        print("\n✨ All analyses complete! Check the generated PNG files for visualizations.")
        print(f"📁 Output directory: {self.data_dir}")


def main():
    """Main function to run the SOFC data analyzer."""
    analyzer = SOFCDataAnalyzer()
    analyzer.run_all_analyses()
    
    # Print file summary
    print("\n📋 GENERATED FILES SUMMARY:")
    print("  1. sofc_materials_thermal_mechanical.json - Thermal & mechanical properties")
    print("  2. sofc_creep_parameters.json - Norton-Bailey creep parameters")
    print("  3. sofc_plasticity_parameters.json - Johnson-Cook plasticity parameters")
    print("  4. sofc_electrochemical_properties.json - Electrochemical properties")
    print("  5. sofc_temperature_dependent_properties.csv - Temperature-dependent data")
    print("  6. thermal_expansion_comparison.png - TEC comparison plots")
    print("  7. conductivity_arrhenius.png - Arrhenius conductivity plots")
    print("  8. mechanical_properties.png - Mechanical properties plots")
    print("  9. electrochemical_performance.png - Electrochemical performance plots")


if __name__ == "__main__":
    main()