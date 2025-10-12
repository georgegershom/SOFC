#!/usr/bin/env python3
"""
Stratified Flow Acoustics Data Analysis Script
Comprehensive analysis tools for the generated dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import welch
import json
import warnings
warnings.filterwarnings('ignore')

class StratifiedFlowAnalyzer:
    """Comprehensive analyzer for stratified flow acoustics data."""
    
    def __init__(self, dataset_file='stratified_flow_acoustics_dataset.json'):
        """Initialize analyzer with dataset."""
        self.dataset_file = dataset_file
        self.load_dataset()
        
    def load_dataset(self):
        """Load the dataset from JSON file."""
        print(f"Loading dataset from {self.dataset_file}...")
        with open(self.dataset_file, 'r') as f:
            self.dataset = json.load(f)
        
        # Convert to DataFrames for easier analysis
        self.flow_df = pd.DataFrame(self.dataset['flow_regime_data'])
        self.fluid_df = pd.DataFrame(self.dataset['fluid_properties_data'])
        self.acoustic_data = self.dataset['acoustic_transmission_data']
        self.turbulence_data = self.dataset['turbulence_data']
        self.attenuation_data = self.dataset['attenuation_metrics_data']
        
        print(f"Dataset loaded successfully!")
        print(f"Number of experiments: {len(self.flow_df)}")
    
    def analyze_flow_regime_correlations(self):
        """Analyze correlations between flow regime parameters."""
        print("\nAnalyzing flow regime correlations...")
        
        # Select numerical columns for correlation analysis
        numerical_cols = ['void_fraction', 'superficial_gas_velocity', 'superficial_liquid_velocity',
                         'interface_height', 'wave_amplitude', 'temperature', 'pressure']
        
        corr_matrix = self.flow_df[numerical_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Flow Regime Parameters Correlation Matrix')
        plt.tight_layout()
        plt.savefig('flow_regime_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def analyze_attenuation_mechanisms(self):
        """Analyze attenuation mechanisms and their dependencies."""
        print("\nAnalyzing attenuation mechanisms...")
        
        # Extract attenuation data
        atten_df = pd.DataFrame(self.attenuation_data)
        
        # Create scatter plots for attenuation vs flow parameters
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Attenuation vs void fraction
        axes[0,0].scatter(atten_df['void_fraction'], atten_df['average_attenuation'], 
                         alpha=0.6, c=atten_df['superficial_gas_velocity'], cmap='viridis')
        axes[0,0].set_xlabel('Void Fraction')
        axes[0,0].set_ylabel('Average Attenuation Coefficient')
        axes[0,0].set_title('Attenuation vs Void Fraction')
        axes[0,0].grid(True, alpha=0.3)
        
        # Attenuation vs gas velocity
        axes[0,1].scatter(atten_df['superficial_gas_velocity'], atten_df['average_attenuation'],
                         alpha=0.6, c=atten_df['void_fraction'], cmap='plasma')
        axes[0,1].set_xlabel('Superficial Gas Velocity (m/s)')
        axes[0,1].set_ylabel('Average Attenuation Coefficient')
        axes[0,1].set_title('Attenuation vs Gas Velocity')
        axes[0,1].grid(True, alpha=0.3)
        
        # Attenuation vs liquid velocity
        axes[1,0].scatter(atten_df['superficial_liquid_velocity'], atten_df['average_attenuation'],
                         alpha=0.6, c=atten_df['void_fraction'], cmap='inferno')
        axes[1,0].set_xlabel('Superficial Liquid Velocity (m/s)')
        axes[1,0].set_ylabel('Average Attenuation Coefficient')
        axes[1,0].set_title('Attenuation vs Liquid Velocity')
        axes[1,0].grid(True, alpha=0.3)
        
        # 3D scatter plot data preparation
        axes[1,1].scatter(atten_df['void_fraction'], atten_df['superficial_gas_velocity'],
                         s=atten_df['average_attenuation']*1000, alpha=0.6,
                         c=atten_df['average_attenuation'], cmap='coolwarm')
        axes[1,1].set_xlabel('Void Fraction')
        axes[1,1].set_ylabel('Superficial Gas Velocity (m/s)')
        axes[1,1].set_title('Attenuation Bubble Chart\n(Size = Attenuation)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('attenuation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return atten_df
    
    def analyze_frequency_dependent_attenuation(self):
        """Analyze frequency-dependent attenuation characteristics."""
        print("\nAnalyzing frequency-dependent attenuation...")
        
        # Extract frequency data
        frequencies = np.array(self.attenuation_data[0]['frequencies'])
        n_freq = len(frequencies)
        n_exp = len(self.attenuation_data)
        
        # Create matrix of attenuation coefficients
        atten_matrix = np.zeros((n_exp, n_freq))
        for i, exp in enumerate(self.attenuation_data):
            atten_matrix[i, :] = exp['attenuation_coefficient']
        
        # Calculate statistics
        mean_attenuation = np.mean(atten_matrix, axis=0)
        std_attenuation = np.std(atten_matrix, axis=0)
        
        # Plot frequency-dependent attenuation
        plt.figure(figsize=(12, 8))
        
        # Mean attenuation with error bars
        plt.errorbar(frequencies, mean_attenuation, yerr=std_attenuation,
                    fmt='-o', capsize=3, capthick=1, alpha=0.8)
        
        # Individual experiments (sample)
        sample_indices = np.random.choice(n_exp, min(50, n_exp), replace=False)
        for idx in sample_indices:
            plt.plot(frequencies, atten_matrix[idx, :], alpha=0.1, color='gray')
        
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Attenuation Coefficient')
        plt.title('Frequency-Dependent Attenuation Characteristics')
        plt.grid(True, alpha=0.3)
        plt.legend(['Mean ± Std', 'Individual Experiments'])
        
        plt.tight_layout()
        plt.savefig('frequency_dependent_attenuation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return frequencies, mean_attenuation, std_attenuation
    
    def analyze_acoustic_signal_characteristics(self):
        """Analyze acoustic signal characteristics and SNR."""
        print("\nAnalyzing acoustic signal characteristics...")
        
        # Extract SNR data
        snr_values = [exp['snr_db'] for exp in self.acoustic_data]
        
        # Create SNR analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # SNR histogram
        axes[0,0].hist(snr_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_xlabel('SNR (dB)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('SNR Distribution')
        axes[0,0].grid(True, alpha=0.3)
        
        # SNR vs void fraction
        void_fractions = [self.flow_df.iloc[i]['void_fraction'] for i in range(len(snr_values))]
        axes[0,1].scatter(void_fractions, snr_values, alpha=0.6, color='orange')
        axes[0,1].set_xlabel('Void Fraction')
        axes[0,1].set_ylabel('SNR (dB)')
        axes[0,1].set_title('SNR vs Void Fraction')
        axes[0,1].grid(True, alpha=0.3)
        
        # SNR vs gas velocity
        gas_velocities = [self.flow_df.iloc[i]['superficial_gas_velocity'] for i in range(len(snr_values))]
        axes[1,0].scatter(gas_velocities, snr_values, alpha=0.6, color='green')
        axes[1,0].set_xlabel('Superficial Gas Velocity (m/s)')
        axes[1,0].set_ylabel('SNR (dB)')
        axes[1,0].set_title('SNR vs Gas Velocity')
        axes[1,0].grid(True, alpha=0.3)
        
        # SNR vs flow pattern
        flow_patterns = [self.flow_df.iloc[i]['flow_pattern'] for i in range(len(snr_values))]
        pattern_snr = pd.DataFrame({'pattern': flow_patterns, 'snr': snr_values})
        pattern_snr.boxplot(column='snr', by='pattern', ax=axes[1,1])
        axes[1,1].set_title('SNR vs Flow Pattern')
        axes[1,1].set_xlabel('Flow Pattern')
        axes[1,1].set_ylabel('SNR (dB)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('acoustic_signal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return snr_values
    
    def analyze_turbulence_characteristics(self):
        """Analyze turbulence and shear layer characteristics."""
        print("\nAnalyzing turbulence characteristics...")
        
        # Extract turbulence data
        turb_df = pd.DataFrame(self.turbulence_data)
        
        # Create turbulence analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # TKE vs void fraction
        axes[0,0].scatter(turb_df['turbulent_kinetic_energy_gas'], 
                         turb_df['turbulent_kinetic_energy_liquid'],
                         c=self.flow_df['void_fraction'], cmap='viridis', alpha=0.6)
        axes[0,0].set_xlabel('Gas TKE (m²/s²)')
        axes[0,0].set_ylabel('Liquid TKE (m²/s²)')
        axes[0,0].set_title('Turbulent Kinetic Energy Comparison')
        axes[0,0].grid(True, alpha=0.3)
        
        # Dissipation rate vs void fraction
        axes[0,1].scatter(turb_df['turbulent_dissipation_rate_gas'],
                         turb_df['turbulent_dissipation_rate_liquid'],
                         c=self.flow_df['void_fraction'], cmap='plasma', alpha=0.6)
        axes[0,1].set_xlabel('Gas Dissipation Rate (m²/s³)')
        axes[0,1].set_ylabel('Liquid Dissipation Rate (m²/s³)')
        axes[0,1].set_title('Turbulent Dissipation Rate Comparison')
        axes[0,1].grid(True, alpha=0.3)
        
        # Shear stress analysis
        axes[1,0].scatter(turb_df['shear_stress_interface'],
                         turb_df['wall_shear_stress_gas'],
                         c=self.flow_df['void_fraction'], cmap='coolwarm', alpha=0.6)
        axes[1,0].set_xlabel('Interface Shear Stress (Pa)')
        axes[1,0].set_ylabel('Wall Shear Stress - Gas (Pa)')
        axes[1,0].set_title('Shear Stress Analysis')
        axes[1,0].grid(True, alpha=0.3)
        
        # TKE ratio vs void fraction
        tke_ratio = turb_df['turbulent_kinetic_energy_gas'] / (turb_df['turbulent_kinetic_energy_liquid'] + 1e-10)
        axes[1,1].scatter(self.flow_df['void_fraction'], tke_ratio, alpha=0.6, color='purple')
        axes[1,1].set_xlabel('Void Fraction')
        axes[1,1].set_ylabel('TKE Ratio (Gas/Liquid)')
        axes[1,1].set_title('TKE Ratio vs Void Fraction')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('turbulence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return turb_df
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*70)
        
        # Run all analyses
        flow_correlations = self.analyze_flow_regime_correlations()
        attenuation_analysis = self.analyze_attenuation_mechanisms()
        freq_attenuation = self.analyze_frequency_dependent_attenuation()
        acoustic_analysis = self.analyze_acoustic_signal_characteristics()
        turbulence_analysis = self.analyze_turbulence_characteristics()
        
        # Generate summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        print(f"Dataset contains {len(self.flow_df)} experiments")
        print(f"Void fraction range: {self.flow_df['void_fraction'].min():.3f} - {self.flow_df['void_fraction'].max():.3f}")
        print(f"Gas velocity range: {self.flow_df['superficial_gas_velocity'].min():.3f} - {self.flow_df['superficial_gas_velocity'].max():.3f} m/s")
        print(f"Liquid velocity range: {self.flow_df['superficial_liquid_velocity'].min():.3f} - {self.flow_df['superficial_liquid_velocity'].max():.3f} m/s")
        print(f"Average SNR: {np.mean(acoustic_analysis):.1f} ± {np.std(acoustic_analysis):.1f} dB")
        print(f"Average attenuation coefficient: {np.mean(attenuation_analysis['average_attenuation']):.4f}")
        
        print("\nAnalysis complete! Check the generated plots for detailed visualizations.")
        
        return {
            'flow_correlations': flow_correlations,
            'attenuation_analysis': attenuation_analysis,
            'frequency_analysis': freq_attenuation,
            'acoustic_analysis': acoustic_analysis,
            'turbulence_analysis': turbulence_analysis
        }

def main():
    """Main function to run comprehensive analysis."""
    print("Stratified Flow Acoustics Data Analyzer")
    print("="*50)
    
    # Initialize analyzer
    analyzer = StratifiedFlowAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.generate_comprehensive_report()
    
    print("\nAnalysis completed successfully!")
    print("Generated files:")
    print("- flow_regime_correlations.png")
    print("- attenuation_analysis.png")
    print("- frequency_dependent_attenuation.png")
    print("- acoustic_signal_analysis.png")
    print("- turbulence_analysis.png")

if __name__ == "__main__":
    main()