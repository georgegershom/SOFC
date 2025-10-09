#!/usr/bin/env python3
"""
SOFC EIS (Electrochemical Impedance Spectroscopy) Visualization Script
Visualizes Nyquist plots and Bode plots for impedance data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import optimize

plt.style.use('seaborn-v0_8-darkgrid')

def load_eis_data(base_path='../eis_data'):
    """Load all EIS data files"""
    eis_files = list(Path(base_path).glob('*.csv'))
    
    datasets = {}
    for filepath in eis_files:
        # Extract temperature and current from filename
        filename = filepath.stem
        datasets[filename] = pd.read_csv(filepath)
        print(f"Loaded {filename}: {len(datasets[filename])} frequency points")
    
    return datasets

def plot_nyquist(datasets):
    """Create Nyquist plots for EIS data"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(datasets)))
    
    for (name, data), color in zip(datasets.items(), colors):
        # Plot experimental data
        ax.plot(data['Real_Impedance_Ohm_cm2'], 
               -data['Imaginary_Impedance_Ohm_cm2'],
               'o-', label=name.replace('_', ' '), 
               markersize=4, linewidth=1, color=color)
    
    ax.set_xlabel('Real Impedance (Ω·cm²)', fontsize=12)
    ax.set_ylabel('-Imaginary Impedance (Ω·cm²)', fontsize=12)
    ax.set_title('SOFC Impedance Spectroscopy - Nyquist Plot', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add frequency labels for key points
    for name, data in datasets.items():
        # Label high frequency (10kHz)
        idx_10k = np.argmin(np.abs(data['Frequency_Hz'] - 10000))
        ax.annotate('10 kHz', 
                   xy=(data['Real_Impedance_Ohm_cm2'].iloc[idx_10k], 
                       -data['Imaginary_Impedance_Ohm_cm2'].iloc[idx_10k]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Label low frequency (10Hz)
        idx_10 = np.argmin(np.abs(data['Frequency_Hz'] - 10))
        ax.annotate('10 Hz',
                   xy=(data['Real_Impedance_Ohm_cm2'].iloc[idx_10], 
                       -data['Imaginary_Impedance_Ohm_cm2'].iloc[idx_10]),
                   xytext=(5, -5), textcoords='offset points', fontsize=8, alpha=0.7)
        break  # Only annotate once
    
    plt.tight_layout()
    plt.savefig('../nyquist_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_bode(datasets):
    """Create Bode plots (magnitude and phase) for EIS data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(datasets)))
    
    for (name, data), color in zip(datasets.items(), colors):
        # Magnitude plot
        ax1.semilogx(data['Frequency_Hz'], data['Magnitude_Ohm_cm2'], 
                    'o-', label=name.replace('_', ' '), 
                    markersize=4, linewidth=1, color=color)
        
        # Phase plot
        ax2.semilogx(data['Frequency_Hz'], -data['Phase_Angle_deg'], 
                    'o-', label=name.replace('_', ' '), 
                    markersize=4, linewidth=1, color=color)
    
    ax1.set_ylabel('|Z| (Ω·cm²)', fontsize=12)
    ax1.set_title('SOFC Impedance Spectroscopy - Bode Plot', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('-Phase Angle (degrees)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('../bode_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def extract_impedance_parameters(datasets):
    """Extract key impedance parameters from EIS data"""
    parameters = {}
    
    for name, data in datasets.items():
        # High frequency resistance (ohmic resistance)
        r_ohmic = data['Real_Impedance_Ohm_cm2'].min()
        
        # Low frequency resistance (total resistance)
        r_total = data['Real_Impedance_Ohm_cm2'].max()
        
        # Polarization resistance
        r_pol = r_total - r_ohmic
        
        # Characteristic frequency (at maximum imaginary impedance)
        max_imag_idx = data['Imaginary_Impedance_Ohm_cm2'].abs().idxmax()
        f_char = data.loc[max_imag_idx, 'Frequency_Hz']
        
        # Calculate effective capacitance
        if f_char > 0:
            c_eff = 1 / (2 * np.pi * f_char * r_pol)
        else:
            c_eff = None
        
        parameters[name] = {
            'Ohmic Resistance (Ω·cm²)': r_ohmic,
            'Polarization Resistance (Ω·cm²)': r_pol,
            'Total Resistance (Ω·cm²)': r_total,
            'Characteristic Frequency (Hz)': f_char,
            'Effective Capacitance (F/cm²)': c_eff
        }
    
    # Display parameters
    print("\n" + "="*60)
    print("EXTRACTED EIS PARAMETERS")
    print("="*60)
    
    for name, params in parameters.items():
        print(f"\n{name}:")
        print("-"*30)
        for key, value in params.items():
            if value is not None:
                if 'Capacitance' in key:
                    print(f"  {key}: {value:.2e}")
                elif 'Frequency' in key:
                    print(f"  {key}: {value:.1f}")
                else:
                    print(f"  {key}: {value:.4f}")
    
    return parameters

def plot_drt_analysis(datasets):
    """Simple DRT (Distribution of Relaxation Times) visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(datasets)))
    
    for (name, data), color in zip(datasets.items(), colors):
        # Calculate relaxation time from frequency
        tau = 1 / (2 * np.pi * data['Frequency_Hz'])
        
        # Use negative imaginary impedance as a proxy for DRT
        # (This is simplified; real DRT requires deconvolution)
        drt_proxy = -data['Imaginary_Impedance_Ohm_cm2'] / data['Real_Impedance_Ohm_cm2'].max()
        
        ax.semilogx(tau, drt_proxy, 
                   'o-', label=name.replace('_', ' '), 
                   markersize=4, linewidth=1, color=color)
    
    ax.set_xlabel('Relaxation Time τ (s)', fontsize=12)
    ax.set_ylabel('DRT Intensity (normalized)', fontsize=12)
    ax.set_title('Distribution of Relaxation Times (Simplified)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('../drt_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("SOFC EIS Analysis Tool")
    print("="*40)
    
    # Load data
    datasets = load_eis_data()
    
    if datasets:
        # Create visualizations
        plot_nyquist(datasets)
        plot_bode(datasets)
        plot_drt_analysis(datasets)
        
        # Extract parameters
        parameters = extract_impedance_parameters(datasets)
        
        print("\nPlots saved to parent directory")
        print("Analysis complete!")
    else:
        print("No EIS data files found!")