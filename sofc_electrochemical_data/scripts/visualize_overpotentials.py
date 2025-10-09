#!/usr/bin/env python3
"""
SOFC Overpotential Analysis and Visualization Script
Analyzes anode/cathode overpotentials and their effects on Ni oxidation and stress
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

def load_overpotential_data(base_path='../overpotentials'):
    """Load all overpotential data files"""
    data_files = {
        '700°C': f'{base_path}/overpotentials_700C.csv',
        '750°C': f'{base_path}/overpotentials_750C.csv',
        '800°C': f'{base_path}/overpotentials_800C.csv'
    }
    
    datasets = {}
    for temp, filepath in data_files.items():
        if Path(filepath).exists():
            datasets[temp] = pd.read_csv(filepath)
            print(f"Loaded {temp} overpotential data: {len(datasets[temp])} points")
    
    return datasets

def plot_overpotential_breakdown(datasets):
    """Plot breakdown of different overpotentials"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (temp, data) in enumerate(datasets.items()):
        ax = axes[idx]
        
        # Stack plot for overpotentials
        ax.fill_between(data['Current_Density_A_cm2'], 0, 
                        data['Anode_Overpotential_mV'],
                        alpha=0.7, label='Anode', color='#FF6B6B')
        
        ax.fill_between(data['Current_Density_A_cm2'], 
                        data['Anode_Overpotential_mV'],
                        data['Anode_Overpotential_mV'] + data['Cathode_Overpotential_mV'],
                        alpha=0.7, label='Cathode', color='#4ECDC4')
        
        ax.fill_between(data['Current_Density_A_cm2'],
                        data['Anode_Overpotential_mV'] + data['Cathode_Overpotential_mV'],
                        data['Total_Overpotential_mV'],
                        alpha=0.7, label='Ohmic', color='#45B7D1')
        
        ax.set_xlabel('Current Density (A/cm²)', fontsize=11)
        ax.set_ylabel('Overpotential (mV)', fontsize=11)
        ax.set_title(f'Overpotential Breakdown at {temp}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('SOFC Overpotential Components', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../overpotential_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_ni_oxidation_risk(datasets):
    """Plot Ni oxidation risk and local pO2"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'700°C': '#2E86AB', '750°C': '#A23B72', '800°C': '#F18F01'}
    
    # Plot Local pO2
    for temp, data in datasets.items():
        ax1.semilogy(data['Current_Density_A_cm2'], data['Local_pO2_atm'],
                    label=temp, linewidth=2, color=colors[temp])
    
    ax1.set_xlabel('Current Density (A/cm²)', fontsize=11)
    ax1.set_ylabel('Local pO₂ (atm)', fontsize=11)
    ax1.set_title('Oxygen Partial Pressure at Anode', fontsize=12, fontweight='bold')
    ax1.axhline(y=1e-18, color='red', linestyle='--', alpha=0.5, label='Ni/NiO boundary')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot Volume Change
    for temp, data in datasets.items():
        ax2.plot(data['Current_Density_A_cm2'], data['Volume_Change_%'],
                label=temp, linewidth=2, color=colors[temp])
    
    ax2.set_xlabel('Current Density (A/cm²)', fontsize=11)
    ax2.set_ylabel('Volume Change (%)', fontsize=11)
    ax2.set_title('Volume Change due to Ni Oxidation', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot Stress
    for temp, data in datasets.items():
        ax3.plot(data['Current_Density_A_cm2'], data['Stress_MPa'],
                label=temp, linewidth=2, color=colors[temp])
    
    ax3.set_xlabel('Current Density (A/cm²)', fontsize=11)
    ax3.set_ylabel('Stress (MPa)', fontsize=11)
    ax3.set_title('Mechanical Stress from Oxidation', fontsize=12, fontweight='bold')
    ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Moderate stress')
    ax3.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='High stress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot Risk Map
    risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    
    for temp, data in datasets.items():
        risk_numeric = [risk_mapping.get(risk, 0) for risk in data['Ni_Oxidation_Risk']]
        ax4.plot(data['Current_Density_A_cm2'], risk_numeric,
                label=temp, linewidth=2, marker='o', markersize=4, color=colors[temp])
    
    ax4.set_xlabel('Current Density (A/cm²)', fontsize=11)
    ax4.set_ylabel('Ni Oxidation Risk Level', fontsize=11)
    ax4.set_title('Ni Oxidation Risk Assessment', fontsize=12, fontweight='bold')
    ax4.set_yticks([1, 2, 3, 4])
    ax4.set_yticklabels(['Low', 'Medium', 'High', 'Very High'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Ni Oxidation Analysis and Stress Development', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../ni_oxidation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_correlation_heatmap(datasets):
    """Create correlation heatmap for different parameters"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (temp, data) in enumerate(datasets.items()):
        ax = axes[idx]
        
        # Select numerical columns for correlation
        corr_columns = ['Current_Density_A_cm2', 'Anode_Overpotential_mV', 
                       'Cathode_Overpotential_mV', 'Ohmic_Overpotential_mV',
                       'Volume_Change_%', 'Stress_MPa']
        
        corr_data = data[corr_columns].corr()
        
        # Create heatmap
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'Parameter Correlations at {temp}', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.suptitle('Correlation Analysis of Electrochemical Parameters', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('../correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_critical_points(datasets):
    """Identify critical operating points"""
    print("\n" + "="*60)
    print("CRITICAL OPERATING POINTS ANALYSIS")
    print("="*60)
    
    for temp, data in datasets.items():
        print(f"\n{temp}:")
        print("-"*30)
        
        # Find transition points for Ni oxidation risk
        risk_transitions = {}
        current_risk = data['Ni_Oxidation_Risk'].iloc[0]
        
        for idx, row in data.iterrows():
            if row['Ni_Oxidation_Risk'] != current_risk:
                risk_transitions[f"{current_risk} → {row['Ni_Oxidation_Risk']}"] = {
                    'Current Density': row['Current_Density_A_cm2'],
                    'Local pO2': row['Local_pO2_atm'],
                    'Stress': row['Stress_MPa']
                }
                current_risk = row['Ni_Oxidation_Risk']
        
        for transition, values in risk_transitions.items():
            print(f"\n  {transition}:")
            print(f"    Current: {values['Current Density']:.3f} A/cm²")
            print(f"    Local pO₂: {values['Local pO2']:.2e} atm")
            print(f"    Stress: {values['Stress']:.1f} MPa")
        
        # Find maximum safe operating current
        safe_data = data[data['Ni_Oxidation_Risk'].isin(['Low', 'Medium'])]
        if not safe_data.empty:
            max_safe_current = safe_data['Current_Density_A_cm2'].max()
            print(f"\n  Maximum safe current density: {max_safe_current:.3f} A/cm²")

def plot_operating_map(datasets):
    """Create operating map showing safe and critical regions"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create meshgrid for temperature and current density
    temps = [700, 750, 800]
    currents = np.linspace(0, 1, 50)
    
    # Create risk map
    risk_map = np.zeros((len(temps), len(currents)))
    risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    
    for i, temp in enumerate(['700°C', '750°C', '800°C']):
        if temp in datasets:
            data = datasets[temp]
            for j, current in enumerate(currents):
                # Interpolate risk level
                idx = np.argmin(np.abs(data['Current_Density_A_cm2'] - current))
                risk = data['Ni_Oxidation_Risk'].iloc[idx]
                risk_map[i, j] = risk_mapping.get(risk, 0)
    
    # Plot contour map
    X, Y = np.meshgrid(currents, temps)
    contour = ax.contourf(X, Y, risk_map, levels=[0.5, 1.5, 2.5, 3.5, 4.5],
                          colors=['green', 'yellow', 'orange', 'red'], alpha=0.7)
    
    # Add contour lines
    ax.contour(X, Y, risk_map, levels=[1.5, 2.5, 3.5], colors='black', linewidths=1, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Current Density (A/cm²)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('SOFC Operating Map - Ni Oxidation Risk Regions', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, ticks=[1, 2, 3, 4])
    cbar.set_label('Risk Level', fontsize=11)
    cbar.ax.set_yticklabels(['Low', 'Medium', 'High', 'Very High'])
    
    # Add safe operating boundary
    ax.text(0.15, 780, 'SAFE', fontsize=16, fontweight='bold', color='white', ha='center')
    ax.text(0.5, 780, 'CAUTION', fontsize=16, fontweight='bold', color='black', ha='center')
    ax.text(0.85, 780, 'DANGER', fontsize=16, fontweight='bold', color='white', ha='center')
    
    plt.tight_layout()
    plt.savefig('../operating_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("SOFC Overpotential Analysis Tool")
    print("="*40)
    
    # Load data
    datasets = load_overpotential_data()
    
    if datasets:
        # Create visualizations
        plot_overpotential_breakdown(datasets)
        plot_ni_oxidation_risk(datasets)
        plot_correlation_heatmap(datasets)
        plot_operating_map(datasets)
        
        # Analyze critical points
        analyze_critical_points(datasets)
        
        print("\nPlots saved to parent directory")
        print("Analysis complete!")
    else:
        print("No overpotential data files found!")