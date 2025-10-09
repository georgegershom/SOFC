#!/usr/bin/env python3
"""
SOFC IV Curve Visualization Script
Visualizes Current-Voltage (IV) curves and power density for SOFC performance data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')

def load_iv_data(base_path='../iv_curves'):
    """Load all IV curve data files"""
    data_files = {
        '700°C': f'{base_path}/iv_curves_700C.csv',
        '750°C': f'{base_path}/iv_curves_750C.csv',
        '800°C': f'{base_path}/iv_curves_800C.csv'
    }
    
    datasets = {}
    for temp, filepath in data_files.items():
        if Path(filepath).exists():
            datasets[temp] = pd.read_csv(filepath)
            print(f"Loaded {temp} data: {len(datasets[temp])} points")
    
    return datasets

def plot_iv_curves(datasets):
    """Create IV curve and power density plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'700°C': '#2E86AB', '750°C': '#A23B72', '800°C': '#F18F01'}
    
    # Plot IV curves
    for temp, data in datasets.items():
        ax1.plot(data['Current_Density_A_cm2'], data['Voltage_V'], 
                label=temp, linewidth=2, color=colors[temp])
    
    ax1.set_xlabel('Current Density (A/cm²)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title('SOFC IV Curves at Different Temperatures', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(data['Current_Density_A_cm2'].max() for data in datasets.values())])
    
    # Plot Power Density curves
    for temp, data in datasets.items():
        ax2.plot(data['Current_Density_A_cm2'], data['Power_Density_W_cm2'], 
                label=temp, linewidth=2, color=colors[temp])
    
    ax2.set_xlabel('Current Density (A/cm²)', fontsize=12)
    ax2.set_ylabel('Power Density (W/cm²)', fontsize=12)
    ax2.set_title('SOFC Power Density at Different Temperatures', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../iv_curves_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_utilization_efficiency(datasets):
    """Plot fuel and air utilization vs current density"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'700°C': '#2E86AB', '750°C': '#A23B72', '800°C': '#F18F01'}
    
    # Fuel Utilization
    for temp, data in datasets.items():
        if 'Fuel_Utilization_%' in data.columns:
            ax1.plot(data['Current_Density_A_cm2'], data['Fuel_Utilization_%'], 
                    label=temp, linewidth=2, color=colors[temp])
    
    ax1.set_xlabel('Current Density (A/cm²)', fontsize=12)
    ax1.set_ylabel('Fuel Utilization (%)', fontsize=12)
    ax1.set_title('Fuel Utilization vs Current Density', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Air Utilization
    for temp, data in datasets.items():
        if 'Air_Utilization_%' in data.columns:
            ax2.plot(data['Current_Density_A_cm2'], data['Air_Utilization_%'], 
                    label=temp, linewidth=2, color=colors[temp])
    
    ax2.set_xlabel('Current Density (A/cm²)', fontsize=12)
    ax2.set_ylabel('Air Utilization (%)', fontsize=12)
    ax2.set_title('Air Utilization vs Current Density', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../utilization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def calculate_performance_metrics(datasets):
    """Calculate and display key performance metrics"""
    metrics = {}
    
    for temp, data in datasets.items():
        metrics[temp] = {
            'OCV': data['Voltage_V'].iloc[0],
            'Max Power Density': data['Power_Density_W_cm2'].max(),
            'Current at Max Power': data.loc[data['Power_Density_W_cm2'].idxmax(), 'Current_Density_A_cm2'],
            'Voltage at Max Power': data.loc[data['Power_Density_W_cm2'].idxmax(), 'Voltage_V'],
            'ASR (Area Specific Resistance)': None
        }
        
        # Calculate ASR from linear region (0.1-0.3 A/cm²)
        mask = (data['Current_Density_A_cm2'] >= 0.1) & (data['Current_Density_A_cm2'] <= 0.3)
        if mask.any():
            linear_data = data[mask]
            # ASR = ΔV/ΔI
            asr = -(linear_data['Voltage_V'].iloc[-1] - linear_data['Voltage_V'].iloc[0]) / \
                   (linear_data['Current_Density_A_cm2'].iloc[-1] - linear_data['Current_Density_A_cm2'].iloc[0])
            metrics[temp]['ASR (Area Specific Resistance)'] = f"{asr:.3f} Ω·cm²"
    
    # Display metrics
    print("\n" + "="*60)
    print("SOFC PERFORMANCE METRICS")
    print("="*60)
    
    for temp, metric in metrics.items():
        print(f"\n{temp}:")
        print("-"*30)
        for key, value in metric.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
    
    return metrics

if __name__ == "__main__":
    print("SOFC IV Curve Analysis Tool")
    print("="*40)
    
    # Load data
    datasets = load_iv_data()
    
    if datasets:
        # Create visualizations
        plot_iv_curves(datasets)
        plot_utilization_efficiency(datasets)
        
        # Calculate metrics
        metrics = calculate_performance_metrics(datasets)
        
        print("\nPlots saved to parent directory")
        print("Analysis complete!")
    else:
        print("No data files found!")