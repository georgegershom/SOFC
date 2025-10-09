#!/usr/bin/env python3
"""
Visualization script for SOFC Electrochemical Loading Dataset
Generates publication-quality plots demonstrating key phenomena
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 2

def plot_iv_and_power_curves():
    """Plot IV and power density curves"""
    df = pd.read_csv('sofc_iv_curve_800C.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # IV curve with overpotentials
    ax1.plot(df['Current_Density_A_cm2'], df['Voltage_V'], 
             'b-', linewidth=2.5, label='Cell Voltage')
    ax1.axhline(y=1.05, color='r', linestyle='--', alpha=0.7, label='OCV')
    ax1.set_xlabel('Current Density (A/cm²)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('IV Characteristic at 800°C')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, df['Current_Density_A_cm2'].max()])
    
    # Power density curve
    ax2.plot(df['Current_Density_A_cm2'], df['Power_Density_W_cm2'], 
             'g-', linewidth=2.5)
    max_power_idx = df['Power_Density_W_cm2'].idxmax()
    max_power = df.loc[max_power_idx, 'Power_Density_W_cm2']
    max_power_current = df.loc[max_power_idx, 'Current_Density_A_cm2']
    ax2.plot(max_power_current, max_power, 'ro', markersize=10, 
             label=f'Peak: {max_power:.3f} W/cm²')
    ax2.set_xlabel('Current Density (A/cm²)')
    ax2.set_ylabel('Power Density (W/cm²)')
    ax2.set_title('Power Density Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0, df['Current_Density_A_cm2'].max()])
    
    plt.tight_layout()
    plt.savefig('figure_1_iv_power_curves.png', dpi=300, bbox_inches='tight')
    print("Generated: figure_1_iv_power_curves.png")
    plt.close()

def plot_overpotential_breakdown():
    """Plot breakdown of overpotentials"""
    df = pd.read_csv('sofc_iv_curve_800C.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(df['Current_Density_A_cm2'], 0, 
                     df['Overpotential_Anode_V'], 
                     alpha=0.7, label='Anode Activation', color='#FF6B6B')
    ax.fill_between(df['Current_Density_A_cm2'], 
                     df['Overpotential_Anode_V'],
                     df['Overpotential_Anode_V'] + df['Overpotential_Cathode_V'],
                     alpha=0.7, label='Cathode Activation', color='#4ECDC4')
    ax.fill_between(df['Current_Density_A_cm2'],
                     df['Overpotential_Anode_V'] + df['Overpotential_Cathode_V'],
                     df['Overpotential_Anode_V'] + df['Overpotential_Cathode_V'] + 
                     df['Overpotential_Ohmic_V'],
                     alpha=0.7, label='Ohmic', color='#95E1D3')
    ax.fill_between(df['Current_Density_A_cm2'],
                     df['Overpotential_Anode_V'] + df['Overpotential_Cathode_V'] + 
                     df['Overpotential_Ohmic_V'],
                     df['Overpotential_Anode_V'] + df['Overpotential_Cathode_V'] + 
                     df['Overpotential_Ohmic_V'] + df['Overpotential_Concentration_V'],
                     alpha=0.7, label='Concentration', color='#F9A825')
    
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Overpotential (V)')
    ax.set_title('Breakdown of Overpotential Losses')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, df['Current_Density_A_cm2'].max()])
    
    plt.tight_layout()
    plt.savefig('figure_2_overpotential_breakdown.png', dpi=300, bbox_inches='tight')
    print("Generated: figure_2_overpotential_breakdown.png")
    plt.close()

def plot_eis_nyquist():
    """Plot EIS Nyquist plots"""
    df = pd.read_csv('sofc_eis_data.csv')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    current_densities = sorted(df['Current_Density_A_cm2'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(current_densities)))
    
    for i, i_cd in enumerate(current_densities):
        data = df[df['Current_Density_A_cm2'] == i_cd]
        ax.plot(data['Z_Real_Ohm_cm2'], -data['Z_Imag_Ohm_cm2'], 
                'o-', label=f'{i_cd:.1f} A/cm²', color=colors[i], 
                markersize=3, alpha=0.7)
    
    ax.set_xlabel('Z_Real (Ω·cm²)')
    ax.set_ylabel('-Z_Imag (Ω·cm²)')
    ax.set_title('EIS Nyquist Plots at Various Current Densities (800°C)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add annotations for characteristic features
    ax.annotate('High frequency\n(Ohmic)', xy=(0.12, 0.01), 
                xytext=(0.12, 0.05), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('Charge transfer\narc', xy=(0.25, 0.1), 
                xytext=(0.35, 0.15), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('figure_3_eis_nyquist.png', dpi=300, bbox_inches='tight')
    print("Generated: figure_3_eis_nyquist.png")
    plt.close()

def plot_overpotential_stress_coupling():
    """Plot electrochemical-mechanical coupling"""
    df = pd.read_csv('sofc_overpotential_stress_data.csv')
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Anode overpotential vs current
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['Current_Density_A_cm2'], df['Overpotential_Anode_V'], 
             'b-', linewidth=2)
    ax1.set_xlabel('Current Density (A/cm²)')
    ax1.set_ylabel('Anode Overpotential (V)')
    ax1.set_title('(a) Anode Overpotential')
    ax1.grid(True, alpha=0.3)
    
    # 2. Oxygen partial pressure at anode
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(df['Current_Density_A_cm2'], df['O2_Partial_Pressure_Anode_Pa'], 
                 'r-', linewidth=2)
    ax2.axhline(y=1e-15, color='k', linestyle='--', alpha=0.7, 
                label='Critical P_O2 for NiO')
    ax2.set_xlabel('Current Density (A/cm²)')
    ax2.set_ylabel('O₂ Partial Pressure (Pa)')
    ax2.set_title('(b) Local Oxygen Partial Pressure at Anode')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Oxygen chemical potential gradient
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['Current_Density_A_cm2'], 
             -df['O2_Chemical_Potential_Gradient_J_mol']/1e6, 
             'g-', linewidth=2)
    ax3.set_xlabel('Current Density (A/cm²)')
    ax3.set_ylabel('|Δμ_O₂| (MJ/mol)')
    ax3.set_title('(c) Oxygen Chemical Potential Gradient Across Electrolyte')
    ax3.grid(True, alpha=0.3)
    
    # 4. Oxidation risk factor
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['Current_Density_A_cm2'], df['Oxidation_Risk_Factor'], 
             'orange', linewidth=2)
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, 
                label='Critical threshold')
    ax4.fill_between(df['Current_Density_A_cm2'], 0, 1.0, 
                     alpha=0.2, color='green', label='Safe region')
    ax4.fill_between(df['Current_Density_A_cm2'], 1.0, 
                     df['Oxidation_Risk_Factor'].max()*1.1,
                     alpha=0.2, color='red', label='Oxidation risk')
    ax4.set_xlabel('Current Density (A/cm²)')
    ax4.set_ylabel('Oxidation Risk Factor')
    ax4.set_title('(d) Ni to NiO Oxidation Risk')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Ni fraction oxidized
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df['Current_Density_A_cm2'], df['Ni_Fraction_Oxidized']*100, 
             'm-', linewidth=2)
    ax5.set_xlabel('Current Density (A/cm²)')
    ax5.set_ylabel('Ni Oxidized (%)')
    ax5.set_title('(e) Fraction of Ni Converted to NiO')
    ax5.grid(True, alpha=0.3)
    
    # 6. Induced stress
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(df['Current_Density_A_cm2'], df['Stress_Induced_MPa'], 
             'darkred', linewidth=2)
    ax6.set_xlabel('Current Density (A/cm²)')
    ax6.set_ylabel('Induced Stress (MPa)')
    ax6.set_title('(f) Mechanical Stress from Volume Expansion')
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Electrochemical-Mechanical Coupling in SOFC Anode', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('figure_4_electrochemical_mechanical_coupling.png', 
                dpi=300, bbox_inches='tight')
    print("Generated: figure_4_electrochemical_mechanical_coupling.png")
    plt.close()

def plot_multi_temperature_comparison():
    """Plot IV curves at multiple temperatures"""
    df = pd.read_csv('sofc_multi_temperature_iv_curves.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    temperatures = sorted(df['Temperature_C'].unique())
    colors = plt.cm.hot_r(np.linspace(0.2, 0.8, len(temperatures)))
    
    for i, temp in enumerate(temperatures):
        data = df[df['Temperature_C'] == temp]
        ax1.plot(data['Current_Density_A_cm2'], data['Voltage_V'],
                '-', color=colors[i], linewidth=2, label=f'{temp}°C')
        ax2.plot(data['Current_Density_A_cm2'], data['Power_Density_W_cm2'],
                '-', color=colors[i], linewidth=2, label=f'{temp}°C')
    
    ax1.set_xlabel('Current Density (A/cm²)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('IV Curves at Different Temperatures')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Current Density (A/cm²)')
    ax2.set_ylabel('Power Density (W/cm²)')
    ax2.set_title('Power Curves at Different Temperatures')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_5_multi_temperature_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("Generated: figure_5_multi_temperature_comparison.png")
    plt.close()

def plot_degradation_analysis():
    """Plot degradation time series"""
    df = pd.read_csv('sofc_degradation_time_series.csv')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Voltage degradation
    ax1.plot(df['Time_hours'], df['Voltage_V'], 'b-', linewidth=2)
    ax1.set_xlabel('Operating Time (hours)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('(a) Voltage Degradation at 0.5 A/cm²')
    ax1.grid(True, alpha=0.3)
    
    # Overpotential increase
    ax2.plot(df['Time_hours'], df['Overpotential_Anode_V']*1000, 
             'r-', linewidth=2, label='Anode')
    ax2.plot(df['Time_hours'], df['Overpotential_Cathode_V']*1000, 
             'g-', linewidth=2, label='Cathode')
    ax2.plot(df['Time_hours'], df['Overpotential_Ohmic_V']*1000, 
             'b-', linewidth=2, label='Ohmic')
    ax2.set_xlabel('Operating Time (hours)')
    ax2.set_ylabel('Overpotential (mV)')
    ax2.set_title('(b) Overpotential Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ohmic resistance increase
    ax3.plot(df['Time_hours'], df['R_Ohmic_Ohm_cm2'], 
             'purple', linewidth=2)
    ax3.set_xlabel('Operating Time (hours)')
    ax3.set_ylabel('R_Ohmic (Ω·cm²)')
    ax3.set_title('(c) Ohmic Resistance Increase')
    ax3.grid(True, alpha=0.3)
    
    # Power degradation
    ax4.plot(df['Time_hours'], df['Power_Density_W_cm2'], 
             'orange', linewidth=2)
    ax4.set_xlabel('Operating Time (hours)')
    ax4.set_ylabel('Power Density (W/cm²)')
    ax4.set_title('(d) Power Density Degradation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_6_degradation_analysis.png', dpi=300, bbox_inches='tight')
    print("Generated: figure_6_degradation_analysis.png")
    plt.close()

def main():
    """Generate all visualizations"""
    print("Generating SOFC Dataset Visualizations...")
    print("="*60)
    
    plot_iv_and_power_curves()
    plot_overpotential_breakdown()
    plot_eis_nyquist()
    plot_overpotential_stress_coupling()
    plot_multi_temperature_comparison()
    plot_degradation_analysis()
    
    print("="*60)
    print("All visualizations generated successfully!")
    print("\nFigures created:")
    print("  1. figure_1_iv_power_curves.png")
    print("  2. figure_2_overpotential_breakdown.png")
    print("  3. figure_3_eis_nyquist.png")
    print("  4. figure_4_electrochemical_mechanical_coupling.png")
    print("  5. figure_5_multi_temperature_comparison.png")
    print("  6. figure_6_degradation_analysis.png")

if __name__ == "__main__":
    main()
