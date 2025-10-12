#!/usr/bin/env python3
"""
Visualization Tools for Stratified Flow Acoustic Dataset

This module provides comprehensive visualization functions for exploring
and presenting the experimental data in various formats.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class DataVisualizer:
    """
    Class for creating various visualizations of the stratified flow dataset
    """
    
    def __init__(self, data_path="../"):
        """Initialize with path to dataset"""
        self.data_path = data_path
        self.load_all_data()
        self.setup_style()
    
    def load_all_data(self):
        """Load all datasets"""
        try:
            self.flow_data = pd.read_csv(f"{self.data_path}/experimental_data/flow_regime_characterization.csv")
            self.acoustic_data = pd.read_csv(f"{self.data_path}/acoustic_signals/acoustic_measurements.csv")
            self.attenuation_data = pd.read_csv(f"{self.data_path}/attenuation_metrics/attenuation_coefficients.csv")
            self.freq_dependent = pd.read_csv(f"{self.data_path}/attenuation_metrics/frequency_dependent_attenuation.csv")
            self.fluid_data = pd.read_csv(f"{self.data_path}/fluid_properties/fluid_conditions.csv")
            self.turbulence_data = pd.read_csv(f"{self.data_path}/turbulence_data/turbulence_measurements.csv")
            self.velocity_data = pd.read_csv(f"{self.data_path}/turbulence_data/velocity_profiles.csv")
            print("All datasets loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
    
    def setup_style(self):
        """Setup matplotlib style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Custom color palette
        self.colors = {
            'smooth_stratified': '#2E86AB',
            'wavy_stratified': '#A23B72',
            'gas': '#F18F01',
            'liquid': '#C73E1D',
            'interface': '#7209B7'
        }
    
    def overview_dashboard(self):
        """Create comprehensive overview dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Flow regime map
        ax1 = plt.subplot(3, 4, 1)
        for pattern in self.flow_data['flow_pattern'].unique():
            data = self.flow_data[self.flow_data['flow_pattern'] == pattern]
            ax1.scatter(data['superficial_gas_velocity_usg_ms'],
                       data['superficial_liquid_velocity_usl_ms'],
                       c=self.colors[pattern], label=pattern.replace('_', ' ').title(),
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('USG (m/s)')
        ax1.set_ylabel('USL (m/s)')
        ax1.set_title('Flow Regime Map')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 2. Void fraction distribution
        ax2 = plt.subplot(3, 4, 2)
        for pattern in self.flow_data['flow_pattern'].unique():
            data = self.flow_data[self.flow_data['flow_pattern'] == pattern]
            ax2.hist(data['void_fraction_alpha'], bins=10, alpha=0.7, 
                    color=self.colors[pattern], label=pattern.replace('_', ' ').title())
        ax2.set_xlabel('Void Fraction')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Void Fraction Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Interface characteristics
        ax3 = plt.subplot(3, 4, 3)
        ax3.scatter(self.flow_data['interface_height_mm'], 
                   self.flow_data['wave_amplitude_mm'],
                   c=[self.colors[p] for p in self.flow_data['flow_pattern']],
                   s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Interface Height (mm)')
        ax3.set_ylabel('Wave Amplitude (mm)')
        ax3.set_title('Interface Characteristics')
        ax3.grid(True, alpha=0.3)
        
        # 4. Attenuation vs frequency
        ax4 = plt.subplot(3, 4, 4)
        exp_ids = ['EXP001', 'EXP003', 'EXP005', 'EXP009']
        for exp_id in exp_ids:
            data = self.attenuation_data[self.attenuation_data['experiment_id'] == exp_id]
            ax4.loglog(data['frequency_hz'], data['attenuation_coefficient_np_per_m'], 
                      'o-', label=exp_id, linewidth=2, markersize=6)
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Attenuation (Np/m)')
        ax4.set_title('Frequency-Dependent Attenuation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. SNR distribution
        ax5 = plt.subplot(3, 4, 5)
        ax5.hist(self.acoustic_data['signal_to_noise_ratio_db'], bins=20, 
                color='skyblue', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('SNR (dB)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Signal-to-Noise Ratio Distribution')
        ax5.grid(True, alpha=0.3)
        
        # 6. Temperature and pressure conditions
        ax6 = plt.subplot(3, 4, 6)
        ax6.scatter(self.fluid_data['temperature_c'], 
                   self.fluid_data['system_pressure_kpa'],
                   c='green', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax6.set_xlabel('Temperature (°C)')
        ax6.set_ylabel('Pressure (kPa)')
        ax6.set_title('Operating Conditions')
        ax6.grid(True, alpha=0.3)
        
        # 7. Turbulence intensity
        ax7 = plt.subplot(3, 4, 7)
        for phase in self.turbulence_data['phase'].unique():
            data = self.turbulence_data[self.turbulence_data['phase'] == phase]
            ax7.scatter(data['mean_velocity_ms'], 
                       data['turbulence_intensity_percent'],
                       c=self.colors.get(phase, 'gray'), label=phase.title(),
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax7.set_xlabel('Mean Velocity (m/s)')
        ax7.set_ylabel('Turbulence Intensity (%)')
        ax7.set_title('Turbulence Characteristics')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Reynolds stress
        ax8 = plt.subplot(3, 4, 8)
        interface_data = self.turbulence_data[self.turbulence_data['phase'] == 'interface']
        ax8.scatter(interface_data['mean_velocity_ms'], 
                   interface_data['reynolds_stress_pa'],
                   c=self.colors['interface'], s=80, alpha=0.7, 
                   edgecolors='black', linewidth=0.5)
        ax8.set_xlabel('Interface Velocity (m/s)')
        ax8.set_ylabel('Reynolds Stress (Pa)')
        ax8.set_title('Interface Reynolds Stress')
        ax8.grid(True, alpha=0.3)
        
        # 9. Velocity profiles example
        ax9 = plt.subplot(3, 4, 9)
        exp_data = self.velocity_data[self.velocity_data['experiment_id'] == 'EXP005']
        ax9.plot(exp_data['axial_velocity_ms'], exp_data['radial_position_mm'], 
                'o-', linewidth=2, markersize=6)
        ax9.set_xlabel('Axial Velocity (m/s)')
        ax9.set_ylabel('Radial Position (mm)')
        ax9.set_title('Velocity Profile (EXP005)')
        ax9.grid(True, alpha=0.3)
        
        # 10. Attenuation mechanisms
        ax10 = plt.subplot(3, 4, 10)
        exp_data = self.freq_dependent[self.freq_dependent['experiment_id'] == 'EXP005']
        ax10.loglog(exp_data['frequency_hz'], exp_data['scattering_loss_np_per_m'], 'o-', label='Scattering')
        ax10.loglog(exp_data['frequency_hz'], exp_data['viscous_loss_np_per_m'], 's-', label='Viscous')
        ax10.loglog(exp_data['frequency_hz'], exp_data['interface_loss_np_per_m'], '^-', label='Interface')
        ax10.set_xlabel('Frequency (Hz)')
        ax10.set_ylabel('Loss (Np/m)')
        ax10.set_title('Attenuation Mechanisms')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # 11. Phase velocity
        ax11 = plt.subplot(3, 4, 11)
        ax11.scatter(self.attenuation_data['frequency_hz'], 
                    self.attenuation_data['phase_velocity_ms'],
                    c='purple', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax11.set_xlabel('Frequency (Hz)')
        ax11.set_ylabel('Phase Velocity (m/s)')
        ax11.set_title('Phase Velocity vs Frequency')
        ax11.grid(True, alpha=0.3)
        
        # 12. Quality factor
        ax12 = plt.subplot(3, 4, 12)
        ax12.scatter(self.attenuation_data['frequency_hz'], 
                    self.attenuation_data['quality_factor_q'],
                    c='orange', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax12.set_xlabel('Frequency (Hz)')
        ax12.set_ylabel('Quality Factor')
        ax12.set_title('Quality Factor vs Frequency')
        ax12.grid(True, alpha=0.3)
        ax12.set_yscale('log')
        
        plt.suptitle('Stratified Flow Acoustic Dataset - Overview Dashboard', fontsize=20, y=0.98)
        plt.tight_layout()
        plt.show()
    
    def create_3d_flow_map(self):
        """Create 3D flow regime map"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for pattern in self.flow_data['flow_pattern'].unique():
            data = self.flow_data[self.flow_data['flow_pattern'] == pattern]
            ax.scatter(data['superficial_gas_velocity_usg_ms'],
                      data['superficial_liquid_velocity_usl_ms'],
                      data['void_fraction_alpha'],
                      c=self.colors[pattern], label=pattern.replace('_', ' ').title(),
                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Superficial Gas Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Superficial Liquid Velocity (m/s)', fontsize=12)
        ax.set_zlabel('Void Fraction', fontsize=12)
        ax.set_title('3D Flow Regime Map', fontsize=16)
        ax.legend()
        
        plt.show()
    
    def interactive_attenuation_plot(self):
        """Create interactive attenuation plot using Plotly"""
        # Merge data for interactive plotting
        merged_data = self.attenuation_data.merge(self.flow_data, on='experiment_id')
        
        fig = px.scatter(merged_data, 
                        x='frequency_hz', 
                        y='attenuation_coefficient_np_per_m',
                        color='flow_pattern',
                        size='void_fraction_alpha',
                        hover_data=['experiment_id', 'superficial_gas_velocity_usg_ms', 
                                  'superficial_liquid_velocity_usl_ms'],
                        log_x=True, log_y=True,
                        title='Interactive Attenuation vs Frequency')
        
        fig.update_layout(
            xaxis_title='Frequency (Hz)',
            yaxis_title='Attenuation Coefficient (Np/m)',
            width=1000, height=600
        )
        
        fig.show()
    
    def velocity_profile_animation(self, experiment_ids=['EXP001', 'EXP003', 'EXP005', 'EXP009']):
        """Create animated velocity profiles"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            ax.clear()
            exp_id = experiment_ids[frame % len(experiment_ids)]
            
            # Get data for this experiment
            exp_data = self.velocity_data[self.velocity_data['experiment_id'] == exp_id]
            void_fraction = self.flow_data[self.flow_data['experiment_id'] == exp_id]['void_fraction_alpha'].iloc[0]
            
            # Plot velocity profile
            ax.plot(exp_data['axial_velocity_ms'], exp_data['radial_position_mm'], 
                   'o-', linewidth=3, markersize=8, color='blue')
            
            # Mark interface
            interface_data = exp_data[exp_data['phase'] == 'interface']
            if not interface_data.empty:
                ax.axhline(y=interface_data['radial_position_mm'].iloc[0], 
                          color='red', linestyle='--', linewidth=2, label='Interface')
            
            ax.set_xlabel('Axial Velocity (m/s)', fontsize=14)
            ax.set_ylabel('Radial Position (mm)', fontsize=14)
            ax.set_title(f'Velocity Profile - {exp_id} (α = {void_fraction:.2f})', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 102)
        
        anim = FuncAnimation(fig, animate, frames=len(experiment_ids)*3, 
                           interval=1500, repeat=True)
        plt.show()
        
        return anim
    
    def correlation_heatmap(self):
        """Create comprehensive correlation heatmap"""
        # Merge all relevant data
        merged_data = (self.flow_data
                      .merge(self.fluid_data, on='experiment_id')
                      .merge(self.attenuation_data.groupby('experiment_id').mean().reset_index(), 
                            on='experiment_id'))
        
        # Select numerical columns for correlation
        numerical_cols = merged_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = merged_data[numerical_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'},
                   fmt='.2f', annot_kws={'size': 8})
        
        ax.set_title('Comprehensive Parameter Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def turbulence_characteristics_plot(self):
        """Plot comprehensive turbulence characteristics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Turbulent kinetic energy vs velocity
        ax1 = axes[0, 0]
        for phase in self.turbulence_data['phase'].unique():
            data = self.turbulence_data[self.turbulence_data['phase'] == phase]
            ax1.scatter(data['mean_velocity_ms'], data['turbulent_kinetic_energy_m2_s2'],
                       c=self.colors.get(phase, 'gray'), label=phase.title(),
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Mean Velocity (m/s)')
        ax1.set_ylabel('Turbulent Kinetic Energy (m²/s²)')
        ax1.set_title('TKE vs Mean Velocity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Dissipation rate vs TKE
        ax2 = axes[0, 1]
        ax2.scatter(self.turbulence_data['turbulent_kinetic_energy_m2_s2'],
                   self.turbulence_data['turbulent_dissipation_rate_m2_s3'],
                   c=[self.colors.get(p, 'gray') for p in self.turbulence_data['phase']],
                   s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Turbulent Kinetic Energy (m²/s²)')
        ax2.set_ylabel('Dissipation Rate (m²/s³)')
        ax2.set_title('Dissipation Rate vs TKE')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # 3. Reynolds stress distribution
        ax3 = axes[0, 2]
        for phase in self.turbulence_data['phase'].unique():
            data = self.turbulence_data[self.turbulence_data['phase'] == phase]
            ax3.hist(data['reynolds_stress_pa'], bins=10, alpha=0.7,
                    color=self.colors.get(phase, 'gray'), label=phase.title())
        ax3.set_xlabel('Reynolds Stress (Pa)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reynolds Stress Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Turbulence intensity vs Reynolds number
        ax4 = axes[1, 0]
        for phase in self.velocity_data['phase'].unique():
            data = self.velocity_data[self.velocity_data['phase'] == phase]
            ax4.scatter(data['reynolds_number_local'], data['velocity_fluctuation_rms_ms'],
                       c=self.colors.get(phase, 'gray'), label=phase.title(),
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Local Reynolds Number')
        ax4.set_ylabel('Velocity Fluctuation RMS (m/s)')
        ax4.set_title('Velocity Fluctuations vs Reynolds Number')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        # 5. Integral length scale
        ax5 = axes[1, 1]
        ax5.scatter(self.turbulence_data['mean_velocity_ms'],
                   self.turbulence_data['integral_length_scale_mm'],
                   c=[self.colors.get(p, 'gray') for p in self.turbulence_data['phase']],
                   s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax5.set_xlabel('Mean Velocity (m/s)')
        ax5.set_ylabel('Integral Length Scale (mm)')
        ax5.set_title('Integral Length Scale vs Velocity')
        ax5.grid(True, alpha=0.3)
        
        # 6. Shear stress comparison
        ax6 = axes[1, 2]
        wall_stress = self.turbulence_data[self.turbulence_data['wall_shear_stress_pa'] > 0]['wall_shear_stress_pa']
        interface_stress = self.turbulence_data[self.turbulence_data['interface_shear_stress_pa'] > 0]['interface_shear_stress_pa']
        
        ax6.hist(wall_stress, bins=10, alpha=0.7, color='blue', label='Wall Shear Stress')
        ax6.hist(interface_stress, bins=10, alpha=0.7, color='red', label='Interface Shear Stress')
        ax6.set_xlabel('Shear Stress (Pa)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Shear Stress Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        
        plt.suptitle('Turbulence Characteristics Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate visualization capabilities"""
    visualizer = DataVisualizer()
    
    print("=== Stratified Flow Dataset Visualization Tools ===\n")
    
    # 1. Overview dashboard
    print("1. Creating overview dashboard...")
    visualizer.overview_dashboard()
    
    # 2. 3D flow map
    print("2. Creating 3D flow regime map...")
    visualizer.create_3d_flow_map()
    
    # 3. Correlation heatmap
    print("3. Creating correlation heatmap...")
    visualizer.correlation_heatmap()
    
    # 4. Turbulence characteristics
    print("4. Plotting turbulence characteristics...")
    visualizer.turbulence_characteristics_plot()
    
    # 5. Interactive plot (if plotly is available)
    try:
        print("5. Creating interactive attenuation plot...")
        visualizer.interactive_attenuation_plot()
    except ImportError:
        print("5. Plotly not available, skipping interactive plot...")
    
    # 6. Velocity profile animation
    print("6. Creating velocity profile animation...")
    anim = visualizer.velocity_profile_animation()
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()