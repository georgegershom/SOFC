#!/usr/bin/env python3
"""
Dataset Analysis and Visualization Script

This script provides comprehensive analysis and visualization tools for the
stratified flow acoustic attenuation dataset.

Usage: python analyze_dataset.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import signal
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class DatasetAnalyzer:
    """Analyze and visualize the stratified flow dataset."""
    
    def __init__(self, data_dir='stratified_flow_dataset'):
        """Initialize analyzer with dataset directory."""
        self.data_dir = data_dir
        self.load_data()
        
    def load_data(self):
        """Load all dataset files."""
        print("Loading dataset files...")
        
        # Load CSV files
        self.flow_data = pd.read_csv(
            os.path.join(self.data_dir, 'flow_regime_characterization.csv')
        )
        self.attenuation_data = pd.read_csv(
            os.path.join(self.data_dir, 'acoustic_attenuation_data.csv')
        )
        self.turbulence_data = pd.read_csv(
            os.path.join(self.data_dir, 'turbulence_shear_data.csv')
        )
        self.velocity_profiles = pd.read_csv(
            os.path.join(self.data_dir, 'velocity_profiles.csv')
        )
        
        # Load JSON files
        with open(os.path.join(self.data_dir, 'acoustic_timeseries_data.json'), 'r') as f:
            self.timeseries_data = json.load(f)
        
        with open(os.path.join(self.data_dir, 'dataset_summary.json'), 'r') as f:
            self.summary = json.load(f)
            
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✓ Loaded {len(self.flow_data)} experiments")
        print(f"✓ Loaded {len(self.attenuation_data)} attenuation measurements")
        print(f"✓ Loaded {len(self.turbulence_data)} turbulence records")
        print(f"✓ Loaded {len(self.velocity_profiles)} velocity profile points")
        print(f"✓ Loaded {len(self.timeseries_data)} time-series records\n")
    
    def print_summary(self):
        """Print dataset summary statistics."""
        print("="*70)
        print("DATASET SUMMARY")
        print("="*70)
        
        print(f"\n📊 Total Experiments: {self.summary['dataset_info']['total_experiments']}")
        print(f"📅 Generation Date: {self.summary['dataset_info']['generation_date']}")
        print(f"🔧 Pipe Diameter: {self.summary['dataset_info']['pipe_diameter_m']} m")
        print(f"📏 Pipe Length: {self.summary['dataset_info']['pipe_length_m']} m")
        
        print("\n🌊 Flow Patterns:")
        for pattern, count in self.summary['flow_regime_statistics']['flow_patterns'].items():
            print(f"   - {pattern}: {count} experiments")
        
        print("\n📈 Operating Ranges:")
        print(f"   Void Fraction: {self.summary['flow_regime_statistics']['void_fraction']['min']:.3f} - "
              f"{self.summary['flow_regime_statistics']['void_fraction']['max']:.3f}")
        print(f"   Gas Velocity: {self.summary['flow_regime_statistics']['superficial_gas_velocity']['min']:.2f} - "
              f"{self.summary['flow_regime_statistics']['superficial_gas_velocity']['max']:.2f} m/s")
        print(f"   Liquid Velocity: {self.summary['flow_regime_statistics']['superficial_liquid_velocity']['min']:.3f} - "
              f"{self.summary['flow_regime_statistics']['superficial_liquid_velocity']['max']:.3f} m/s")
        
        print(f"\n🔊 Acoustic Measurements:")
        print(f"   Frequencies: {self.summary['acoustic_statistics']['frequencies_tested_Hz']}")
        print(f"   Attenuation: {self.summary['acoustic_statistics']['attenuation_coefficient']['min']:.4f} - "
              f"{self.summary['acoustic_statistics']['attenuation_coefficient']['max']:.4f} Np/m")
        print(f"   SNR: {self.summary['acoustic_statistics']['SNR']['min']:.1f} - "
              f"{self.summary['acoustic_statistics']['SNR']['max']:.1f} dB")
        
        print("="*70 + "\n")
    
    def plot_flow_map(self, save=True):
        """Plot Taitel-Dukler flow pattern map."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Separate by flow pattern
        smooth = self.flow_data[self.flow_data['flow_pattern'] == 'smooth_stratified']
        wavy = self.flow_data[self.flow_data['flow_pattern'] == 'wavy_stratified']
        
        ax.scatter(smooth['U_SL'], smooth['U_SG'], 
                  c='blue', marker='o', s=100, alpha=0.6, 
                  label='Smooth Stratified', edgecolors='k')
        ax.scatter(wavy['U_SL'], wavy['U_SG'], 
                  c='red', marker='s', s=100, alpha=0.6, 
                  label='Wavy Stratified', edgecolors='k')
        
        ax.set_xlabel('Superficial Liquid Velocity, $U_{SL}$ (m/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Superficial Gas Velocity, $U_{SG}$ (m/s)', fontsize=12, fontweight='bold')
        ax.set_title('Flow Pattern Map (Taitel-Dukler Type)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(self.flow_data['U_SL']) * 1.1)
        ax.set_ylim(0, max(self.flow_data['U_SG']) * 1.1)
        
        plt.tight_layout()
        if save:
            plt.savefig('flow_pattern_map.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: flow_pattern_map.png")
        plt.show()
    
    def plot_attenuation_vs_frequency(self, n_experiments=5, save=True):
        """Plot attenuation coefficient vs frequency for multiple experiments."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select experiments with different void fractions
        exp_ids = self.flow_data.nlargest(n_experiments, 'void_fraction')['experiment_id'].values
        
        for exp_id in exp_ids:
            exp_atten = self.attenuation_data[self.attenuation_data['experiment_id'] == exp_id]
            exp_flow = self.flow_data[self.flow_data['experiment_id'] == exp_id].iloc[0]
            
            alpha = exp_flow['void_fraction']
            U_SG = exp_flow['U_SG']
            pattern = exp_flow['flow_pattern']
            
            label = f"Exp {exp_id}: α={alpha:.2f}, $U_{{SG}}$={U_SG:.1f} m/s ({pattern})"
            ax.loglog(exp_atten['frequency'], exp_atten['attenuation_coefficient'], 
                     'o-', linewidth=2, markersize=8, label=label, alpha=0.8)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attenuation Coefficient (Np/m)', fontsize=12, fontweight='bold')
        ax.set_title('Acoustic Attenuation vs Frequency', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig('attenuation_vs_frequency.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: attenuation_vs_frequency.png")
        plt.show()
    
    def plot_attenuation_mechanisms(self, save=True):
        """Plot contribution of different attenuation mechanisms."""
        # Merge datasets
        merged = self.attenuation_data.merge(
            self.flow_data[['experiment_id', 'flow_pattern']], 
            on='experiment_id'
        )
        
        # Group by frequency
        freq_groups = merged.groupby('frequency').agg({
            'viscous_atten_contribution': 'mean',
            'scattering_atten_contribution': 'mean',
            'turbulence_atten_contribution': 'mean'
        })
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        frequencies = freq_groups.index
        width = 0.25
        x = np.arange(len(frequencies))
        
        ax.bar(x - width, freq_groups['viscous_atten_contribution'], width, 
               label='Viscous Absorption', color='steelblue', edgecolor='black')
        ax.bar(x, freq_groups['scattering_atten_contribution'], width, 
               label='Scattering', color='coral', edgecolor='black')
        ax.bar(x + width, freq_groups['turbulence_atten_contribution'], width, 
               label='Turbulence', color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Attenuation Contribution (Np/m)', fontsize=12, fontweight='bold')
        ax.set_title('Attenuation Mechanism Contributions', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(f)}' for f in frequencies])
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        if save:
            plt.savefig('attenuation_mechanisms.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: attenuation_mechanisms.png")
        plt.show()
    
    def plot_void_fraction_effect(self, save=True):
        """Plot effect of void fraction on attenuation at different frequencies."""
        # Select a few frequencies
        test_freqs = [1000, 2000, 5000]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for idx, freq in enumerate(test_freqs):
            ax = axes[idx]
            
            # Filter data for this frequency
            freq_data = self.attenuation_data[self.attenuation_data['frequency'] == freq]
            
            # Merge with flow data to get void fraction
            merged = freq_data.merge(
                self.flow_data[['experiment_id', 'void_fraction', 'flow_pattern']], 
                on='experiment_id'
            )
            
            # Separate by flow pattern
            smooth = merged[merged['flow_pattern'] == 'smooth_stratified']
            wavy = merged[merged['flow_pattern'] == 'wavy_stratified']
            
            ax.scatter(smooth['void_fraction'], smooth['attenuation_coefficient'], 
                      c='blue', marker='o', s=80, alpha=0.6, label='Smooth')
            ax.scatter(wavy['void_fraction'], wavy['attenuation_coefficient'], 
                      c='red', marker='s', s=80, alpha=0.6, label='Wavy')
            
            ax.set_xlabel('Void Fraction, α', fontsize=11, fontweight='bold')
            ax.set_ylabel('Attenuation (Np/m)', fontsize=11, fontweight='bold')
            ax.set_title(f'{freq} Hz', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Effect of Void Fraction on Acoustic Attenuation', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save:
            plt.savefig('void_fraction_effect.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: void_fraction_effect.png")
        plt.show()
    
    def plot_velocity_profiles(self, exp_id=1, save=True):
        """Plot velocity profiles for a specific experiment."""
        exp_profiles = self.velocity_profiles[self.velocity_profiles['experiment_id'] == exp_id]
        exp_flow = self.flow_data[self.flow_data['experiment_id'] == exp_id].iloc[0]
        
        gas_profile = exp_profiles[exp_profiles['phase'] == 'gas']
        liquid_profile = exp_profiles[exp_profiles['phase'] == 'liquid']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Dimensional velocities
        ax1.plot(gas_profile['velocity'], gas_profile['radial_position']*1000, 
                'r-o', linewidth=2, markersize=6, label='Gas Phase')
        ax1.plot(liquid_profile['velocity'], liquid_profile['radial_position']*1000, 
                'b-s', linewidth=2, markersize=6, label='Liquid Phase')
        ax1.set_xlabel('Velocity (m/s)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Radial Position (mm)', fontsize=11, fontweight='bold')
        ax1.set_title('Velocity Profiles', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Normalized velocities
        ax2.plot(gas_profile['normalized_velocity'], gas_profile['normalized_position'], 
                'r-o', linewidth=2, markersize=6, label='Gas Phase')
        ax2.plot(liquid_profile['normalized_velocity'], liquid_profile['normalized_position'], 
                'b-s', linewidth=2, markersize=6, label='Liquid Phase')
        ax2.set_xlabel('Normalized Velocity (u/U)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Normalized Position (y/D)', fontsize=11, fontweight='bold')
        ax2.set_title('Normalized Velocity Profiles', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Experiment {exp_id}: α={exp_flow["void_fraction"]:.2f}, '
                    f'$U_{{SG}}$={exp_flow["U_SG"]:.1f} m/s, $U_{{SL}}$={exp_flow["U_SL"]:.3f} m/s', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(f'velocity_profiles_exp{exp_id}.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: velocity_profiles_exp{exp_id}.png")
        plt.show()
    
    def plot_acoustic_timeseries(self, exp_id=1, save=True):
        """Plot acoustic time series data."""
        # Find the time series data for this experiment
        ts_data = None
        for ts in self.timeseries_data:
            if ts['experiment_id'] == exp_id:
                ts_data = ts
                break
        
        if ts_data is None:
            print(f"No time-series data for experiment {exp_id}")
            return
        
        time = np.array(ts_data['time'])
        source = np.array(ts_data['source_signal'])
        received = np.array(ts_data['received_signal'])
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Time series
        axes[0].plot(time[:500], source[:500], 'b-', linewidth=1.5, label='Source')
        axes[0].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        axes[0].set_title('Source Signal', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(time[:500], received[:500], 'r-', linewidth=1.5, label='Received')
        axes[1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        axes[1].set_title('Received Signal (with Attenuation & Noise)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot 2: FFT
        N = len(source)
        sr = ts_data['sampling_rate']
        freqs = np.fft.rfftfreq(N, 1/sr)
        
        source_fft = np.abs(np.fft.rfft(source))
        received_fft = np.abs(np.fft.rfft(received))
        
        axes[2].semilogy(freqs[:5000], source_fft[:5000], 'b-', linewidth=1.5, 
                        alpha=0.7, label='Source')
        axes[2].semilogy(freqs[:5000], received_fft[:5000], 'r-', linewidth=1.5, 
                        alpha=0.7, label='Received')
        axes[2].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Magnitude', fontsize=11, fontweight='bold')
        axes[2].set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_xlim(0, 5000)
        
        plt.suptitle(f'Acoustic Signal Analysis - Experiment {exp_id}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'acoustic_timeseries_exp{exp_id}.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: acoustic_timeseries_exp{exp_id}.png")
        plt.show()
    
    def plot_turbulence_analysis(self, save=True):
        """Plot turbulence parameters."""
        # Merge turbulence data with flow data
        merged = self.turbulence_data.merge(
            self.flow_data[['experiment_id', 'U_SG', 'U_SL', 'void_fraction', 'flow_pattern']], 
            on='experiment_id'
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Reynolds number
        axes[0, 0].scatter(merged['U_SG'], merged['Re_gas'], 
                          c=merged['void_fraction'], cmap='viridis', 
                          s=80, alpha=0.7, edgecolors='k')
        axes[0, 0].set_xlabel('Gas Velocity (m/s)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Reynolds Number (Gas)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Gas Phase Reynolds Number', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: TKE
        axes[0, 1].scatter(merged['U_SG'], merged['TKE_gas'], 
                          c=merged['void_fraction'], cmap='viridis', 
                          s=80, alpha=0.7, edgecolors='k')
        axes[0, 1].set_xlabel('Gas Velocity (m/s)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('TKE (m²/s²)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Turbulent Kinetic Energy (Gas)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Interfacial shear
        scatter = axes[1, 0].scatter(merged['U_SG'], merged['tau_interface'], 
                                     c=merged['void_fraction'], cmap='viridis', 
                                     s=80, alpha=0.7, edgecolors='k')
        axes[1, 0].set_xlabel('Gas Velocity (m/s)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Interfacial Shear Stress (Pa)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Interfacial Shear Stress', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Dissipation rate
        axes[1, 1].scatter(merged['TKE_gas'], merged['dissipation_rate_gas'], 
                          c=merged['void_fraction'], cmap='viridis', 
                          s=80, alpha=0.7, edgecolors='k')
        axes[1, 1].set_xlabel('TKE (m²/s²)', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Dissipation Rate (m²/s³)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Turbulent Dissipation Rate', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes, label='Void Fraction', 
                          orientation='horizontal', pad=0.05, aspect=40)
        cbar.set_label('Void Fraction', fontsize=11, fontweight='bold')
        
        plt.suptitle('Turbulence Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig('turbulence_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: turbulence_analysis.png")
        plt.show()
    
    def generate_all_plots(self):
        """Generate all analysis plots."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATION PLOTS")
        print("="*70 + "\n")
        
        self.plot_flow_map()
        self.plot_attenuation_vs_frequency()
        self.plot_attenuation_mechanisms()
        self.plot_void_fraction_effect()
        self.plot_velocity_profiles(exp_id=1)
        self.plot_acoustic_timeseries(exp_id=1)
        self.plot_turbulence_analysis()
        
        print("\n" + "="*70)
        print("ALL PLOTS GENERATED SUCCESSFULLY!")
        print("="*70 + "\n")


def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("STRATIFIED FLOW DATASET ANALYSIS")
    print("="*70 + "\n")
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate all plots
    analyzer.generate_all_plots()
    
    print("Analysis complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
