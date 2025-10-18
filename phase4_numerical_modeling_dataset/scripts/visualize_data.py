"""
Data Visualization Script for Phase 4 Numerical Modeling Dataset
Creates comprehensive plots for rubberized concrete fire resistance data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_loader import RubberizedConcreteDataLoader

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DataVisualizer:
    """
    Visualization tools for rubberized concrete dataset
    """
    
    def __init__(self, data_loader: RubberizedConcreteDataLoader):
        """
        Initialize visualizer
        
        Args:
            data_loader: Instance of RubberizedConcreteDataLoader with loaded data
        """
        self.loader = data_loader
        self.output_path = Path("../visualizations")
        self.output_path.mkdir(exist_ok=True)
        
    def plot_thermal_properties(self):
        """Plot temperature-dependent thermal properties"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rubber_contents = [0, 10, 20, 30]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Thermal conductivity
        ax = axes[0, 0]
        cond_data = self.loader.thermal_data['conductivity']
        for rc, color in zip(rubber_contents, colors):
            data = cond_data[cond_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Temperature_C'], data['Thermal_Conductivity_W_mK'], 
                   'o-', label=f'{rc}% Rubber', color=color, linewidth=2)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Thermal Conductivity (W/mK)', fontsize=12)
        ax.set_title('Thermal Conductivity vs Temperature', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Specific heat
        ax = axes[0, 1]
        heat_data = self.loader.thermal_data['specific_heat']
        for rc, color in zip(rubber_contents, colors):
            data = heat_data[heat_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Temperature_C'], data['Specific_Heat_J_kgK'], 
                   'o-', label=f'{rc}% Rubber', color=color, linewidth=2)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Specific Heat (J/kgK)', fontsize=12)
        ax.set_title('Specific Heat Capacity vs Temperature', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Density
        ax = axes[1, 0]
        dens_data = self.loader.thermal_data['density']
        for rc, color in zip(rubber_contents, colors):
            data = dens_data[dens_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Temperature_C'], data['Density_kg_m3'], 
                   'o-', label=f'{rc}% Rubber', color=color, linewidth=2)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Density (kg/m³)', fontsize=12)
        ax.set_title('Density vs Temperature', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Thermal diffusivity (calculated)
        ax = axes[1, 1]
        for rc, color in zip(rubber_contents, colors):
            cond = cond_data[cond_data['Rubber_Content_Percent'] == rc]
            heat = heat_data[heat_data['Rubber_Content_Percent'] == rc]
            dens = dens_data[dens_data['Rubber_Content_Percent'] == rc]
            
            # Diffusivity = k / (rho * cp)
            diffusivity = (cond['Thermal_Conductivity_W_mK'].values * 1e6 / 
                          (dens['Density_kg_m3'].values * heat['Specific_Heat_J_kgK'].values))
            
            ax.plot(cond['Temperature_C'], diffusivity, 
                   'o-', label=f'{rc}% Rubber', color=color, linewidth=2)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Thermal Diffusivity (mm²/s)', fontsize=12)
        ax.set_title('Thermal Diffusivity vs Temperature', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'thermal_properties.png', dpi=300, bbox_inches='tight')
        print("  Saved: thermal_properties.png")
        
    def plot_mechanical_properties(self):
        """Plot temperature-dependent mechanical properties"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rubber_contents = [0, 10, 20, 30]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Compressive strength
        ax = axes[0, 0]
        comp_data = self.loader.mechanical_data['compressive_strength']
        for rc, color in zip(rubber_contents, colors):
            data = comp_data[comp_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Temperature_C'], data['Compressive_Strength_MPa'], 
                   'o-', label=f'{rc}% Rubber', color=color, linewidth=2)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Compressive Strength (MPa)', fontsize=12)
        ax.set_title('Compressive Strength vs Temperature', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Residual strength ratio
        ax = axes[0, 1]
        for rc, color in zip(rubber_contents, colors):
            data = comp_data[comp_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Temperature_C'], data['Residual_Strength_Ratio'], 
                   'o-', label=f'{rc}% Rubber', color=color, linewidth=2)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Residual Strength Ratio', fontsize=12)
        ax.set_title('Residual Compressive Strength Ratio', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Elastic modulus
        ax = axes[1, 0]
        mod_data = self.loader.mechanical_data['elastic_modulus']
        for rc, color in zip(rubber_contents, colors):
            data = mod_data[mod_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Temperature_C'], data['Elastic_Modulus_GPa'], 
                   'o-', label=f'{rc}% Rubber', color=color, linewidth=2)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Elastic Modulus (GPa)', fontsize=12)
        ax.set_title('Elastic Modulus vs Temperature', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tensile strength
        ax = axes[1, 1]
        tens_data = self.loader.mechanical_data['tensile_strength']
        for rc, color in zip(rubber_contents, colors):
            data = tens_data[tens_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Temperature_C'], data['Tensile_Strength_MPa'], 
                   'o-', label=f'{rc}% Rubber', color=color, linewidth=2)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Tensile Strength (MPa)', fontsize=12)
        ax.set_title('Tensile Strength vs Temperature', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'mechanical_properties.png', dpi=300, bbox_inches='tight')
        print("  Saved: mechanical_properties.png")
        
    def plot_validation_temperature_profiles(self):
        """Plot validation temperature profiles"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        rubber_contents = [0, 10, 20, 30]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # ISO834 - Core temperature
        ax = axes[0, 0]
        iso_data = self.loader.validation_data['iso834_temps']
        for rc, color in zip(rubber_contents, colors):
            data = iso_data[iso_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Time_min'], data['TC6_Center_C'], 
                   '-', label=f'{rc}% Rubber', color=color, linewidth=2.5)
        ax.set_xlabel('Time (min)', fontsize=12)
        ax.set_ylabel('Core Temperature (°C)', fontsize=12)
        ax.set_title('ISO834 Fire - Core Temperature Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ISO834 - Temperature gradient
        ax = axes[0, 1]
        for rc, color in zip(rubber_contents, colors):
            data = iso_data[(iso_data['Rubber_Content_Percent'] == rc) & 
                           (iso_data['Time_min'] == 60)]
            if len(data) > 0:
                temps = [data['TC1_Surface_C'].values[0],
                        data['TC2_25mm_C'].values[0],
                        data['TC3_50mm_C'].values[0],
                        data['TC4_75mm_C'].values[0],
                        data['TC5_100mm_C'].values[0],
                        data['TC6_Center_C'].values[0]]
                positions = [0, 25, 50, 75, 100, 125]
                ax.plot(positions, temps, 'o-', label=f'{rc}% Rubber', 
                       color=color, linewidth=2, markersize=8)
        ax.set_xlabel('Depth from Surface (mm)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Temperature Gradient at 60 min (ISO834)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ASTM E119 - Core temperature
        ax = axes[1, 0]
        astm_data = self.loader.validation_data['astm_e119_temps']
        for rc, color in zip(rubber_contents, colors):
            data = astm_data[astm_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Time_min'], data['TC6_Center_C'], 
                   '-', label=f'{rc}% Rubber', color=color, linewidth=2.5)
        ax.set_xlabel('Time (min)', fontsize=12)
        ax.set_ylabel('Core Temperature (°C)', fontsize=12)
        ax.set_title('ASTM E119 Fire - Core Temperature Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Hydrocarbon - Core temperature
        ax = axes[1, 1]
        hc_data = self.loader.validation_data['hydrocarbon_temps']
        for rc, color in zip(rubber_contents, colors):
            data = hc_data[hc_data['Rubber_Content_Percent'] == rc]
            ax.plot(data['Time_min'], data['TC6_Center_C'], 
                   '-', label=f'{rc}% Rubber', color=color, linewidth=2.5)
        ax.set_xlabel('Time (min)', fontsize=12)
        ax.set_ylabel('Core Temperature (°C)', fontsize=12)
        ax.set_title('Hydrocarbon Fire - Core Temperature Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'validation_temperature_profiles.png', dpi=300, bbox_inches='tight')
        print("  Saved: validation_temperature_profiles.png")
        
    def plot_spalling_analysis(self):
        """Plot spalling behavior analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rubber_contents = [0, 10, 20, 30]
        
        # Time to first spall
        ax = axes[0, 0]
        iso_spall = self.loader.validation_data['spalling_iso834']
        iso_grouped = iso_spall[iso_spall['Applied_Load_MPa'] == 10.0].groupby('Rubber_Content_Percent')['First_Spall_Time_min'].mean()
        ax.bar(iso_grouped.index, iso_grouped.values, color='steelblue', alpha=0.7, label='ISO834')
        ax.set_xlabel('Rubber Content (%)', fontsize=12)
        ax.set_ylabel('Time to First Spall (min)', fontsize=12)
        ax.set_title('Time to First Spalling Event', fontsize=14, fontweight='bold')
        ax.set_xticks(rubber_contents)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Total spalled mass
        ax = axes[0, 1]
        spall_mass = iso_spall[iso_spall['Applied_Load_MPa'] == 10.0].groupby('Rubber_Content_Percent')['Total_Spalled_Mass_g'].mean()
        ax.bar(spall_mass.index, spall_mass.values, color='coral', alpha=0.7)
        ax.set_xlabel('Rubber Content (%)', fontsize=12)
        ax.set_ylabel('Total Spalled Mass (g)', fontsize=12)
        ax.set_title('Total Spalled Mass (ISO834, 10 MPa)', fontsize=14, fontweight='bold')
        ax.set_xticks(rubber_contents)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Number of spalls over time
        ax = axes[1, 0]
        width = 6
        x = np.arange(len(rubber_contents))
        for i, rc in enumerate(rubber_contents):
            data = iso_spall[(iso_spall['Rubber_Content_Percent'] == rc) & 
                            (iso_spall['Applied_Load_MPa'] == 10.0)]
            spalls_30 = data['Number_Spalls_30min'].mean()
            spalls_60 = data['Number_Spalls_60min'].mean() - spalls_30
            spalls_90 = data['Number_Spalls_90min'].mean() - data['Number_Spalls_60min'].mean()
            
            ax.bar(rc, spalls_30, width, label='0-30 min' if i == 0 else '', color='#8dd3c7')
            ax.bar(rc, spalls_60, width, bottom=spalls_30, label='30-60 min' if i == 0 else '', color='#fb8072')
            ax.bar(rc, spalls_90, width, bottom=spalls_30+spalls_60, label='60-90 min' if i == 0 else '', color='#bebada')
        
        ax.set_xlabel('Rubber Content (%)', fontsize=12)
        ax.set_ylabel('Number of Spalling Events', fontsize=12)
        ax.set_title('Spalling Event Timeline', fontsize=14, fontweight='bold')
        ax.set_xticks(rubber_contents)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Failure time comparison
        ax = axes[1, 1]
        failure_times = iso_spall[iso_spall['Applied_Load_MPa'] == 10.0].groupby('Rubber_Content_Percent')['Failure_Time_min'].mean()
        bars = ax.bar(failure_times.index, failure_times.values, color='forestgreen', alpha=0.7)
        ax.set_xlabel('Rubber Content (%)', fontsize=12)
        ax.set_ylabel('Time to Failure (min)', fontsize=12)
        ax.set_title('Time to Failure (ISO834, 10 MPa)', fontsize=14, fontweight='bold')
        ax.set_xticks(rubber_contents)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'spalling_analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: spalling_analysis.png")
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\nGenerating visualizations...")
        
        self.plot_thermal_properties()
        self.plot_mechanical_properties()
        self.plot_validation_temperature_profiles()
        self.plot_spalling_analysis()
        
        print("\nAll visualizations complete!")


def main():
    """Main execution function"""
    
    # Load data
    loader = RubberizedConcreteDataLoader()
    loader.load_all_data()
    
    # Create visualizations
    visualizer = DataVisualizer(loader)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
