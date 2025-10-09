#!/usr/bin/env python3
"""
SOFC Electrochemical Data Visualization Script

This script creates comprehensive visualizations of the generated SOFC electrochemical dataset,
focusing on the relationship between electrochemical performance and mechanical stress.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SOFCDataVisualizer:
    """Visualization class for SOFC electrochemical data."""
    
    def __init__(self, data_dir='/workspace/sofc_data'):
        """Initialize with data directory."""
        self.data_dir = data_dir
        self.data = {}
        self.load_data()
    
    def load_data(self):
        """Load all datasets."""
        print("Loading SOFC electrochemical data...")
        
        datasets = ['iv_curves', 'eis_spectra', 'overpotential_analysis', 'operating_conditions']
        
        for dataset in datasets:
            file_path = os.path.join(self.data_dir, f'{dataset}.csv')
            if os.path.exists(file_path):
                self.data[dataset] = pd.read_csv(file_path)
                print(f"Loaded {dataset}: {len(self.data[dataset])} records")
            else:
                print(f"Warning: {file_path} not found")
    
    def plot_iv_characteristics(self, save_path=None):
        """Plot I-V characteristics at different temperatures."""
        print("Creating I-V characteristics plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # I-V curves
        temperatures = [700, 750, 800, 850]
        colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))
        
        for i, temp in enumerate(temperatures):
            temp_data = self.data['iv_curves'][self.data['iv_curves']['temperature_c'] == temp]
            ax1.plot(temp_data['current_density_acm2'], temp_data['voltage_v'], 
                    color=colors[i], linewidth=2, label=f'{temp}°C')
        
        ax1.set_xlabel('Current Density (A/cm²)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('SOFC I-V Characteristics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Power density curves
        for i, temp in enumerate(temperatures):
            temp_data = self.data['iv_curves'][self.data['iv_curves']['temperature_c'] == temp]
            ax2.plot(temp_data['current_density_acm2'], temp_data['power_density_wcm2'], 
                    color=colors[i], linewidth=2, label=f'{temp}°C')
        
        ax2.set_xlabel('Current Density (A/cm²)')
        ax2.set_ylabel('Power Density (W/cm²)')
        ax2.set_title('Power Density Characteristics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_overpotential_breakdown(self, save_path=None):
        """Plot overpotential breakdown analysis."""
        print("Creating overpotential breakdown plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample data for a specific temperature
        sample_data = self.data['iv_curves'][self.data['iv_curves']['temperature_c'] == 800].iloc[:50]
        
        # Overpotential components
        ax1.plot(sample_data['current_density_acm2'], sample_data['eta_ohmic_v'], 
                'b-', linewidth=2, label='Ohmic')
        ax1.plot(sample_data['current_density_acm2'], sample_data['eta_activation_v'], 
                'r-', linewidth=2, label='Activation')
        ax1.plot(sample_data['current_density_acm2'], sample_data['eta_concentration_v'], 
                'g-', linewidth=2, label='Concentration')
        
        ax1.set_xlabel('Current Density (A/cm²)')
        ax1.set_ylabel('Overpotential (V)')
        ax1.set_title('Overpotential Components at 800°C')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Temperature effect on overpotentials
        temp_data = self.data['iv_curves'][self.data['iv_curves']['current_density_acm2'] == 0.5]
        ax2.plot(temp_data['temperature_c'], temp_data['eta_ohmic_v'], 
                'bo-', linewidth=2, label='Ohmic')
        ax2.plot(temp_data['temperature_c'], temp_data['eta_activation_v'], 
                'ro-', linewidth=2, label='Activation')
        ax2.plot(temp_data['temperature_c'], temp_data['eta_concentration_v'], 
                'go-', linewidth=2, label='Concentration')
        
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Overpotential (V)')
        ax2.set_title('Temperature Effect on Overpotentials (0.5 A/cm²)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ASR vs Temperature
        ax3.plot(temp_data['temperature_c'], temp_data['asr_ohmic_cm2'], 
                'ko-', linewidth=2, markersize=6)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Area Specific Resistance (Ω·cm²)')
        ax3.set_title('Ohmic ASR vs Temperature')
        ax3.grid(True, alpha=0.3)
        
        # Power density vs Temperature
        ax4.plot(temp_data['temperature_c'], temp_data['power_density_wcm2'], 
                'mo-', linewidth=2, markersize=6)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Power Density (W/cm²)')
        ax4.set_title('Power Density vs Temperature (0.5 A/cm²)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_impedance_spectra(self, save_path=None):
        """Plot electrochemical impedance spectroscopy data."""
        print("Creating EIS spectra plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Nyquist plot
        spectrum_ids = self.data['eis_spectra']['spectrum_id'].unique()[:5]
        colors = plt.cm.tab10(np.linspace(0, 1, len(spectrum_ids)))
        
        for i, spec_id in enumerate(spectrum_ids):
            spec_data = self.data['eis_spectra'][self.data['eis_spectra']['spectrum_id'] == spec_id]
            ax1.plot(spec_data['z_real_ohm_cm2'], -spec_data['z_imag_ohm_cm2'], 
                    color=colors[i], linewidth=2, 
                    label=f'T={spec_data["temperature_c"].iloc[0]:.0f}°C')
        
        ax1.set_xlabel('Real Impedance (Ω·cm²)')
        ax1.set_ylabel('-Imaginary Impedance (Ω·cm²)')
        ax1.set_title('Nyquist Plot - EIS Spectra')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Bode plot
        sample_spectrum = self.data['eis_spectra'][self.data['eis_spectra']['spectrum_id'] == spectrum_ids[0]]
        ax2.semilogx(sample_spectrum['frequency_hz'], sample_spectrum['z_magnitude_ohm_cm2'], 
                    'b-', linewidth=2, label='Magnitude')
        ax2_twin = ax2.twinx()
        ax2_twin.semilogx(sample_spectrum['frequency_hz'], sample_spectrum['phase_angle_deg'], 
                         'r-', linewidth=2, label='Phase')
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Impedance Magnitude (Ω·cm²)', color='b')
        ax2_twin.set_ylabel('Phase Angle (degrees)', color='r')
        ax2.set_title('Bode Plot - Impedance vs Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_anode_oxidation_effects(self, save_path=None):
        """Plot anode oxidation effects and mechanical stress."""
        print("Creating anode oxidation effects plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Ni oxidation factor vs temperature
        ax1.scatter(self.data['overpotential_analysis']['temperature_c'], 
                   self.data['overpotential_analysis']['ni_oxidation_factor'], 
                   c=self.data['overpotential_analysis']['time_hours'], 
                   cmap='viridis', alpha=0.7, s=50)
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('Operating Time (hours)')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Ni Oxidation Factor')
        ax1.set_title('Ni Oxidation vs Temperature')
        ax1.grid(True, alpha=0.3)
        
        # Induced stress vs Ni oxidation
        ax2.scatter(self.data['overpotential_analysis']['ni_oxidation_factor'], 
                   self.data['overpotential_analysis']['induced_stress_mpa'], 
                   c=self.data['overpotential_analysis']['temperature_c'], 
                   cmap='plasma', alpha=0.7, s=50)
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('Temperature (°C)')
        ax2.set_xlabel('Ni Oxidation Factor')
        ax2.set_ylabel('Induced Stress (MPa)')
        ax2.set_title('Mechanical Stress vs Ni Oxidation')
        ax2.grid(True, alpha=0.3)
        
        # Volume change vs time
        ax3.scatter(self.data['overpotential_analysis']['time_hours'], 
                   self.data['overpotential_analysis']['ni_volume_change_percent'], 
                   c=self.data['overpotential_analysis']['temperature_c'], 
                   cmap='coolwarm', alpha=0.7, s=50)
        cbar3 = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar3.set_label('Temperature (°C)')
        ax3.set_xlabel('Operating Time (hours)')
        ax3.set_ylabel('Ni Volume Change (%)')
        ax3.set_title('Volume Change vs Operating Time')
        ax3.grid(True, alpha=0.3)
        
        # Anode overpotentials vs Ni oxidation
        ax4.scatter(self.data['overpotential_analysis']['ni_oxidation_factor'], 
                   self.data['overpotential_analysis']['eta_anode_activation_v'], 
                   c=self.data['overpotential_analysis']['eta_anode_concentration_v'], 
                   cmap='RdYlBu', alpha=0.7, s=50)
        cbar4 = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar4.set_label('Anode Concentration Overpotential (V)')
        ax4.set_xlabel('Ni Oxidation Factor')
        ax4.set_ylabel('Anode Activation Overpotential (V)')
        ax4.set_title('Anode Overpotentials vs Ni Oxidation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_operating_conditions_analysis(self, save_path=None):
        """Plot operating conditions and performance analysis."""
        print("Creating operating conditions analysis plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Efficiency vs Temperature
        ax1.scatter(self.data['operating_conditions']['temperature_c'], 
                   self.data['operating_conditions']['efficiency'], 
                   c=self.data['operating_conditions']['current_density_acm2'], 
                   cmap='viridis', alpha=0.7, s=50)
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('Current Density (A/cm²)')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Efficiency')
        ax1.set_title('Cell Efficiency vs Temperature')
        ax1.grid(True, alpha=0.3)
        
        # Power density vs Current density
        ax2.scatter(self.data['operating_conditions']['current_density_acm2'], 
                   self.data['operating_conditions']['power_density_wcm2'], 
                   c=self.data['operating_conditions']['temperature_c'], 
                   cmap='plasma', alpha=0.7, s=50)
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('Temperature (°C)')
        ax2.set_xlabel('Current Density (A/cm²)')
        ax2.set_ylabel('Power Density (W/cm²)')
        ax2.set_title('Power Density vs Current Density')
        ax2.grid(True, alpha=0.3)
        
        # Fuel utilization vs Efficiency
        ax3.scatter(self.data['operating_conditions']['fuel_utilization'], 
                   self.data['operating_conditions']['efficiency'], 
                   c=self.data['operating_conditions']['air_utilization'], 
                   cmap='coolwarm', alpha=0.7, s=50)
        cbar3 = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar3.set_label('Air Utilization')
        ax3.set_xlabel('Fuel Utilization')
        ax3.set_ylabel('Efficiency')
        ax3.set_title('Efficiency vs Gas Utilization')
        ax3.grid(True, alpha=0.3)
        
        # Oxygen chemical potential gradient
        ax4.scatter(self.data['operating_conditions']['oxygen_chemical_potential_gradient'], 
                   self.data['operating_conditions']['eta_activation_v'], 
                   c=self.data['operating_conditions']['temperature_c'], 
                   cmap='RdYlBu', alpha=0.7, s=50)
        cbar4 = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar4.set_label('Temperature (°C)')
        ax4.set_xlabel('Oxygen Chemical Potential Gradient')
        ax4.set_ylabel('Activation Overpotential (V)')
        ax4.set_title('Oxygen Gradient vs Activation Overpotential')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, save_path=None):
        """Create a comprehensive dashboard with all key visualizations."""
        print("Creating comprehensive dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # I-V characteristics
        ax1 = fig.add_subplot(gs[0, 0])
        temperatures = [700, 750, 800, 850]
        colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))
        
        for i, temp in enumerate(temperatures):
            temp_data = self.data['iv_curves'][self.data['iv_curves']['temperature_c'] == temp]
            ax1.plot(temp_data['current_density_acm2'], temp_data['voltage_v'], 
                    color=colors[i], linewidth=2, label=f'{temp}°C')
        
        ax1.set_xlabel('Current Density (A/cm²)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('I-V Characteristics')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Power density
        ax2 = fig.add_subplot(gs[0, 1])
        for i, temp in enumerate(temperatures):
            temp_data = self.data['iv_curves'][self.data['iv_curves']['temperature_c'] == temp]
            ax2.plot(temp_data['current_density_acm2'], temp_data['power_density_wcm2'], 
                    color=colors[i], linewidth=2, label=f'{temp}°C')
        
        ax2.set_xlabel('Current Density (A/cm²)')
        ax2.set_ylabel('Power Density (W/cm²)')
        ax2.set_title('Power Density')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # EIS Nyquist plot
        ax3 = fig.add_subplot(gs[0, 2])
        spectrum_ids = self.data['eis_spectra']['spectrum_id'].unique()[:3]
        colors_eis = plt.cm.tab10(np.linspace(0, 1, len(spectrum_ids)))
        
        for i, spec_id in enumerate(spectrum_ids):
            spec_data = self.data['eis_spectra'][self.data['eis_spectra']['spectrum_id'] == spec_id]
            ax3.plot(spec_data['z_real_ohm_cm2'], -spec_data['z_imag_ohm_cm2'], 
                    color=colors_eis[i], linewidth=2, 
                    label=f'T={spec_data["temperature_c"].iloc[0]:.0f}°C')
        
        ax3.set_xlabel('Real Impedance (Ω·cm²)')
        ax3.set_ylabel('-Imaginary Impedance (Ω·cm²)')
        ax3.set_title('EIS Nyquist Plot')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Ni oxidation vs temperature
        ax4 = fig.add_subplot(gs[1, 0])
        scatter = ax4.scatter(self.data['overpotential_analysis']['temperature_c'], 
                             self.data['overpotential_analysis']['ni_oxidation_factor'], 
                             c=self.data['overpotential_analysis']['time_hours'], 
                             cmap='viridis', alpha=0.7, s=30)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Ni Oxidation Factor')
        ax4.set_title('Ni Oxidation vs Temperature')
        ax4.grid(True, alpha=0.3)
        
        # Induced stress vs Ni oxidation
        ax5 = fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(self.data['overpotential_analysis']['ni_oxidation_factor'], 
                             self.data['overpotential_analysis']['induced_stress_mpa'], 
                             c=self.data['overpotential_analysis']['temperature_c'], 
                             cmap='plasma', alpha=0.7, s=30)
        ax5.set_xlabel('Ni Oxidation Factor')
        ax5.set_ylabel('Induced Stress (MPa)')
        ax5.set_title('Stress vs Ni Oxidation')
        ax5.grid(True, alpha=0.3)
        
        # Efficiency vs temperature
        ax6 = fig.add_subplot(gs[1, 2])
        scatter = ax6.scatter(self.data['operating_conditions']['temperature_c'], 
                             self.data['operating_conditions']['efficiency'], 
                             c=self.data['operating_conditions']['current_density_acm2'], 
                             cmap='viridis', alpha=0.7, s=30)
        ax6.set_xlabel('Temperature (°C)')
        ax6.set_ylabel('Efficiency')
        ax6.set_title('Efficiency vs Temperature')
        ax6.grid(True, alpha=0.3)
        
        # Overpotential breakdown
        ax7 = fig.add_subplot(gs[2, :])
        sample_data = self.data['iv_curves'][self.data['iv_curves']['temperature_c'] == 800].iloc[:50]
        
        ax7.plot(sample_data['current_density_acm2'], sample_data['eta_ohmic_v'], 
                'b-', linewidth=2, label='Ohmic')
        ax7.plot(sample_data['current_density_acm2'], sample_data['eta_activation_v'], 
                'r-', linewidth=2, label='Activation')
        ax7.plot(sample_data['current_density_acm2'], sample_data['eta_concentration_v'], 
                'g-', linewidth=2, label='Concentration')
        
        ax7.set_xlabel('Current Density (A/cm²)')
        ax7.set_ylabel('Overpotential (V)')
        ax7.set_title('Overpotential Breakdown at 800°C')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Volume change vs time
        ax8 = fig.add_subplot(gs[3, 0])
        scatter = ax8.scatter(self.data['overpotential_analysis']['time_hours'], 
                             self.data['overpotential_analysis']['ni_volume_change_percent'], 
                             c=self.data['overpotential_analysis']['temperature_c'], 
                             cmap='coolwarm', alpha=0.7, s=30)
        ax8.set_xlabel('Operating Time (hours)')
        ax8.set_ylabel('Ni Volume Change (%)')
        ax8.set_title('Volume Change vs Time')
        ax8.grid(True, alpha=0.3)
        
        # Power density vs current density
        ax9 = fig.add_subplot(gs[3, 1])
        scatter = ax9.scatter(self.data['operating_conditions']['current_density_acm2'], 
                             self.data['operating_conditions']['power_density_wcm2'], 
                             c=self.data['operating_conditions']['temperature_c'], 
                             cmap='plasma', alpha=0.7, s=30)
        ax9.set_xlabel('Current Density (A/cm²)')
        ax9.set_ylabel('Power Density (W/cm²)')
        ax9.set_title('Power vs Current Density')
        ax9.grid(True, alpha=0.3)
        
        # Oxygen chemical potential gradient
        ax10 = fig.add_subplot(gs[3, 2])
        scatter = ax10.scatter(self.data['operating_conditions']['oxygen_chemical_potential_gradient'], 
                              self.data['operating_conditions']['eta_activation_v'], 
                              c=self.data['operating_conditions']['temperature_c'], 
                              cmap='RdYlBu', alpha=0.7, s=30)
        ax10.set_xlabel('Oxygen Chemical Potential Gradient')
        ax10.set_ylabel('Activation Overpotential (V)')
        ax10.set_title('Oxygen Gradient vs Overpotential')
        ax10.grid(True, alpha=0.3)
        
        plt.suptitle('SOFC Electrochemical Loading Dataset - Comprehensive Analysis Dashboard', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_plots(self, output_dir='/workspace/sofc_data/plots'):
        """Generate all plots and save them."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating all plots in {output_dir}...")
        
        # Individual plots
        self.plot_iv_characteristics(os.path.join(output_dir, 'iv_characteristics.png'))
        self.plot_overpotential_breakdown(os.path.join(output_dir, 'overpotential_breakdown.png'))
        self.plot_impedance_spectra(os.path.join(output_dir, 'impedance_spectra.png'))
        self.plot_anode_oxidation_effects(os.path.join(output_dir, 'anode_oxidation_effects.png'))
        self.plot_operating_conditions_analysis(os.path.join(output_dir, 'operating_conditions.png'))
        
        # Comprehensive dashboard
        self.create_comprehensive_dashboard(os.path.join(output_dir, 'comprehensive_dashboard.png'))
        
        print("All plots generated successfully!")

def main():
    """Main function to generate all visualizations."""
    print("SOFC Electrochemical Data Visualization")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = SOFCDataVisualizer()
    
    # Generate all plots
    visualizer.generate_all_plots()
    
    print("\nVisualization complete!")
    print("Check /workspace/sofc_data/plots/ for all generated plots")

if __name__ == "__main__":
    main()