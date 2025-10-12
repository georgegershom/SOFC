#!/usr/bin/env python3
"""
Attenuation Analysis Tools for Stratified Flow Acoustic Dataset

This module provides functions for analyzing acoustic attenuation in stratified flows,
including frequency-dependent analysis, mechanism decomposition, and correlation studies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.stats import pearsonr
import seaborn as sns

class AttenuationAnalyzer:
    """
    Class for analyzing acoustic attenuation in stratified flows
    """
    
    def __init__(self, data_path="../"):
        """Initialize with path to dataset"""
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        """Load all relevant datasets"""
        try:
            self.flow_data = pd.read_csv(f"{self.data_path}/experimental_data/flow_regime_characterization.csv")
            self.acoustic_data = pd.read_csv(f"{self.data_path}/acoustic_signals/acoustic_measurements.csv")
            self.attenuation_data = pd.read_csv(f"{self.data_path}/attenuation_metrics/attenuation_coefficients.csv")
            self.freq_dependent = pd.read_csv(f"{self.data_path}/attenuation_metrics/frequency_dependent_attenuation.csv")
            self.fluid_data = pd.read_csv(f"{self.data_path}/fluid_properties/fluid_conditions.csv")
            print("Data loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
    
    def plot_attenuation_vs_frequency(self, experiment_ids=None, save_fig=False):
        """
        Plot attenuation coefficient vs frequency for selected experiments
        """
        if experiment_ids is None:
            experiment_ids = ['EXP001', 'EXP003', 'EXP005', 'EXP009']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(experiment_ids)))
        
        for i, exp_id in enumerate(experiment_ids):
            data = self.attenuation_data[self.attenuation_data['experiment_id'] == exp_id]
            void_fraction = self.flow_data[self.flow_data['experiment_id'] == exp_id]['void_fraction_alpha'].iloc[0]
            
            ax.loglog(data['frequency_hz'], data['attenuation_coefficient_np_per_m'], 
                     'o-', color=colors[i], linewidth=2, markersize=8,
                     label=f'{exp_id} (α = {void_fraction:.2f})')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=14)
        ax.set_ylabel('Attenuation Coefficient (Np/m)', fontsize=14)
        ax.set_title('Frequency-Dependent Attenuation in Stratified Flows', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig('attenuation_vs_frequency.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_attenuation_mechanisms(self, experiment_id='EXP005'):
        """
        Analyze different attenuation mechanisms for a specific experiment
        """
        data = self.freq_dependent[self.freq_dependent['experiment_id'] == experiment_id]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Absolute contributions
        ax1.loglog(data['frequency_hz'], data['scattering_loss_np_per_m'], 'o-', label='Scattering', linewidth=2)
        ax1.loglog(data['frequency_hz'], data['viscous_loss_np_per_m'], 's-', label='Viscous', linewidth=2)
        ax1.loglog(data['frequency_hz'], data['thermal_loss_np_per_m'], '^-', label='Thermal', linewidth=2)
        ax1.loglog(data['frequency_hz'], data['interface_loss_np_per_m'], 'd-', label='Interface', linewidth=2)
        ax1.loglog(data['frequency_hz'], data['attenuation_coefficient_np_per_m'], 'k-', label='Total', linewidth=3)
        
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Attenuation Coefficient (Np/m)', fontsize=12)
        ax1.set_title(f'Attenuation Mechanisms - {experiment_id}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Relative contributions
        total_atten = data['attenuation_coefficient_np_per_m']
        ax2.semilogx(data['frequency_hz'], 100*data['scattering_loss_np_per_m']/total_atten, 'o-', label='Scattering')
        ax2.semilogx(data['frequency_hz'], 100*data['viscous_loss_np_per_m']/total_atten, 's-', label='Viscous')
        ax2.semilogx(data['frequency_hz'], 100*data['thermal_loss_np_per_m']/total_atten, '^-', label='Thermal')
        ax2.semilogx(data['frequency_hz'], 100*data['interface_loss_np_per_m']/total_atten, 'd-', label='Interface')
        
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Relative Contribution (%)', fontsize=12)
        ax2.set_title('Relative Contribution of Mechanisms', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self):
        """
        Analyze correlations between flow parameters and attenuation
        """
        # Merge datasets for correlation analysis
        merged_data = self.attenuation_data.merge(self.flow_data, on='experiment_id')
        
        # Select relevant columns for correlation
        corr_columns = ['void_fraction_alpha', 'superficial_gas_velocity_usg_ms', 
                       'superficial_liquid_velocity_usl_ms', 'interface_height_mm',
                       'wave_amplitude_mm', 'frequency_hz', 'attenuation_coefficient_np_per_m']
        
        corr_data = merged_data[corr_columns]
        correlation_matrix = corr_data.corr()
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Correlation Matrix: Flow Parameters vs Attenuation', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def fit_attenuation_model(self, experiment_id='EXP005'):
        """
        Fit empirical model to attenuation data
        """
        data = self.attenuation_data[self.attenuation_data['experiment_id'] == experiment_id]
        
        # Power law model: α = A * f^n
        def power_law(f, A, n):
            return A * (f ** n)
        
        # Fit the model
        popt, pcov = optimize.curve_fit(power_law, data['frequency_hz'], 
                                       data['attenuation_coefficient_np_per_m'])
        
        A_fit, n_fit = popt
        A_err, n_err = np.sqrt(np.diag(pcov))
        
        # Generate fitted curve
        f_fit = np.logspace(np.log10(data['frequency_hz'].min()), 
                           np.log10(data['frequency_hz'].max()), 100)
        alpha_fit = power_law(f_fit, A_fit, n_fit)
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(data['frequency_hz'], data['attenuation_coefficient_np_per_m'], 
                 'ro', markersize=8, label='Experimental Data')
        ax.loglog(f_fit, alpha_fit, 'b-', linewidth=2, 
                 label=f'Power Law Fit: α = {A_fit:.4f} × f^{n_fit:.2f}')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Attenuation Coefficient (Np/m)', fontsize=12)
        ax.set_title(f'Attenuation Model Fitting - {experiment_id}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate R-squared
        alpha_pred = power_law(data['frequency_hz'], A_fit, n_fit)
        ss_res = np.sum((data['attenuation_coefficient_np_per_m'] - alpha_pred) ** 2)
        ss_tot = np.sum((data['attenuation_coefficient_np_per_m'] - 
                        data['attenuation_coefficient_np_per_m'].mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
        
        print(f"Fitted parameters for {experiment_id}:")
        print(f"A = {A_fit:.6f} ± {A_err:.6f}")
        print(f"n = {n_fit:.3f} ± {n_err:.3f}")
        print(f"R² = {r_squared:.3f}")
        
        return A_fit, n_fit, r_squared
    
    def void_fraction_effect(self):
        """
        Analyze effect of void fraction on attenuation at different frequencies
        """
        # Select specific frequencies for analysis
        frequencies = [100, 250, 500, 1000]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for i, freq in enumerate(frequencies):
            freq_data = self.attenuation_data[self.attenuation_data['frequency_hz'] == freq]
            merged = freq_data.merge(self.flow_data, on='experiment_id')
            
            axes[i].scatter(merged['void_fraction_alpha'], 
                          merged['attenuation_coefficient_np_per_m'],
                          s=60, alpha=0.7)
            
            # Fit linear trend
            z = np.polyfit(merged['void_fraction_alpha'], 
                          merged['attenuation_coefficient_np_per_m'], 1)
            p = np.poly1d(z)
            alpha_range = np.linspace(merged['void_fraction_alpha'].min(),
                                    merged['void_fraction_alpha'].max(), 100)
            axes[i].plot(alpha_range, p(alpha_range), 'r--', alpha=0.8)
            
            axes[i].set_xlabel('Void Fraction', fontsize=10)
            axes[i].set_ylabel('Attenuation Coefficient (Np/m)', fontsize=10)
            axes[i].set_title(f'f = {freq} Hz', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # Calculate correlation
            r, p_val = pearsonr(merged['void_fraction_alpha'], 
                               merged['attenuation_coefficient_np_per_m'])
            axes[i].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[i].transAxes,
                        fontsize=10, verticalalignment='top')
        
        plt.suptitle('Effect of Void Fraction on Attenuation', fontsize=16)
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate analysis capabilities"""
    analyzer = AttenuationAnalyzer()
    
    print("=== Stratified Flow Acoustic Attenuation Analysis ===\n")
    
    # 1. Plot frequency-dependent attenuation
    print("1. Plotting frequency-dependent attenuation...")
    analyzer.plot_attenuation_vs_frequency()
    
    # 2. Analyze attenuation mechanisms
    print("2. Analyzing attenuation mechanisms...")
    analyzer.analyze_attenuation_mechanisms('EXP005')
    
    # 3. Correlation analysis
    print("3. Performing correlation analysis...")
    corr_matrix = analyzer.correlation_analysis()
    
    # 4. Fit attenuation model
    print("4. Fitting attenuation model...")
    A, n, r2 = analyzer.fit_attenuation_model('EXP005')
    
    # 5. Void fraction effect analysis
    print("5. Analyzing void fraction effects...")
    analyzer.void_fraction_effect()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()