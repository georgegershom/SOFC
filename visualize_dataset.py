#!/usr/bin/env python3
"""
????????
??????????????
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os
import json

# ??????
os.makedirs('figures', exist_ok=True)

def plot_sintering_kinetics():
    """?????????"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sintering Kinetics - Shrinkage Strain vs Temperature & Time', fontsize=14)
    
    layers = ['anode', 'electrolyte', 'cathode']
    colors = ['blue', 'green', 'red']
    
    # Plot 1: Strain vs Temperature at different times
    ax1 = axes[0, 0]
    for i, layer in enumerate(layers):
        df = pd.read_csv(f'dataset/sintering_kinetics_{layer}.csv')
        # Select time = 120 minutes
        df_t120 = df[df['time_minutes'] == 120]
        ax1.plot(df_t120['temperature_C'], df_t120['strain_dL_L0'], 
                label=layer.capitalize(), color=colors[i], linewidth=2)
    ax1.set_xlabel('Temperature (?C)')
    ax1.set_ylabel('Shrinkage Strain (dL/L0)')
    ax1.set_title('Strain vs Temperature (at t=120 min)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Strain vs Time at different temperatures
    ax2 = axes[0, 1]
    for i, layer in enumerate(layers):
        df = pd.read_csv(f'dataset/sintering_kinetics_{layer}.csv')
        # Select temperature = 1200?C
        df_T1200 = df[df['temperature_C'] == 1200]
        ax2.plot(df_T1200['time_minutes'], df_T1200['strain_dL_L0'], 
                label=layer.capitalize(), color=colors[i], linewidth=2)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Shrinkage Strain (dL/L0)')
    ax2.set_title('Strain vs Time (at T=1200?C)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Relative density vs Temperature
    ax3 = axes[1, 0]
    for i, layer in enumerate(layers):
        df = pd.read_csv(f'dataset/sintering_kinetics_{layer}.csv')
        df_t120 = df[df['time_minutes'] == 120]
        ax3.plot(df_t120['temperature_C'], df_t120['relative_density'], 
                label=layer.capitalize(), color=colors[i], linewidth=2)
    ax3.set_xlabel('Temperature (?C)')
    ax3.set_ylabel('Relative Density')
    ax3.set_title('Relative Density vs Temperature (at t=120 min)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bilayer mismatch strain
    ax4 = axes[1, 1]
    df_anode_elec = pd.read_csv('dataset/sintering_kinetics_anode_electrolyte.csv')
    df_anode_elec_t120 = df_anode_elec[df_anode_elec['time_minutes'] == 120]
    ax4.plot(df_anode_elec_t120['temperature_C'], 
            df_anode_elec_t120['mismatch_strain_dL_L0'], 
            label='Anode-Electrolyte', color='purple', linewidth=2)
    
    df_elec_cath = pd.read_csv('dataset/sintering_kinetics_electrolyte_cathode.csv')
    df_elec_cath_t120 = df_elec_cath[df_elec_cath['time_minutes'] == 120]
    ax4.plot(df_elec_cath_t120['temperature_C'], 
            df_elec_cath_t120['mismatch_strain_dL_L0'], 
            label='Electrolyte-Cathode', color='orange', linewidth=2)
    ax4.set_xlabel('Temperature (?C)')
    ax4.set_ylabel('Mismatch Strain (dL/L0)')
    ax4.set_title('Bilayer Mismatch Strain (at t=120 min)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/sintering_kinetics.png', dpi=300)
    print("? Generated: figures/sintering_kinetics.png")
    plt.close()


def plot_cte():
    """??????? vs ??"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = ['anode', 'electrolyte', 'cathode', 'interconnect']
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['Ni-YSZ Anode', '8YSZ Electrolyte', 'LSM Cathode', 'Crofer Interconnect']
    
    for i, layer in enumerate(layers):
        df = pd.read_csv(f'dataset/cte_{layer}.csv')
        ax.plot(df['temperature_C'], df['cte_10minus6_per_K'], 
               label=labels[i], color=colors[i], linewidth=2)
    
    ax.set_xlabel('Temperature (?C)', fontsize=12)
    ax.set_ylabel('Coefficient of Thermal Expansion (?10?? K??)', fontsize=12)
    ax.set_title('CTE vs Temperature for SOFC Components', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/cte_vs_temperature.png', dpi=300)
    print("? Generated: figures/cte_vs_temperature.png")
    plt.close()


def plot_creep_data():
    """??????"""
    layers = ['anode', 'electrolyte', 'cathode']
    temperatures = [800, 1000, 1200]
    
    for layer in layers:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Creep Behavior: {layer.capitalize()}', fontsize=14)
        
        for idx, temp in enumerate(temperatures):
            ax = axes[idx // 2, idx % 2]
            
            df = pd.read_csv(f'dataset/creep_{layer}_{temp}C.csv')
            
            # Plot strain vs time for different stresses (sintered state only)
            df_sintered = df[df['state'] == 'sintered']
            stresses = sorted(df_sintered['stress_mpa'].unique())
            colors_plt = plt.cm.viridis(np.linspace(0, 1, len(stresses)))
            
            for i, stress in enumerate(stresses):
                df_stress = df_sintered[df_sintered['stress_mpa'] == stress]
                df_stress = df_stress.sort_values('time_hours')
                ax.plot(df_stress['time_hours'], df_stress['strain'] * 100, 
                       label=f'{stress} MPa', color=colors_plt[i], linewidth=2)
            
            ax.set_xlabel('Time (hours)', fontsize=11)
            ax.set_ylabel('Creep Strain (%)', fontsize=11)
            ax.set_title(f'Creep at {temp}?C (Sintered State)', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 100])
        
        plt.tight_layout()
        plt.savefig(f'figures/creep_{layer}.png', dpi=300)
        print(f"? Generated: figures/creep_{layer}.png")
        plt.close()


def plot_creep_parameters():
    """??????????"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Creep Parameter Fitting Verification', fontsize=14)
    
    R = 8.314e-3  # kJ/(mol?K)
    
    # Load creep parameters from JSON
    with open('material_property_dataset.json', 'r') as f:
        dataset = json.load(f)
    
    layers = ['anode', 'electrolyte', 'cathode']
    colors = ['blue', 'green', 'red']
    
    for idx, layer in enumerate(layers):
        ax = axes[idx // 2, idx % 2]
        
        # Get parameters (sintered state)
        params = dataset['layers'][layer]['creep_constitutive']['parameters']
        A = params['A_pre_exponential']['sintering_state']
        n = params['n_stress_exponent']['sintering_state']
        Q = params['Q_activation_energy']['sintering_state']
        
        # Load experimental data at 1000?C
        df = pd.read_csv(f'dataset/creep_{layer}_1000C.csv')
        df_sintered = df[df['state'] == 'sintered']
        
        # Get steady-state strain rates (from later time points)
        stresses = sorted(df_sintered['stress_mpa'].unique())
        strain_rates_exp = []
        
        for stress in stresses:
            df_stress = df_sintered[df_sintered['stress_mpa'] == stress]
            # Use average strain rate from last 20 hours
            df_late = df_stress[df_stress['time_hours'] > 80]
            avg_rate = df_late['strain_rate_per_s'].mean()
            strain_rates_exp.append(avg_rate)
        
        # Calculate theoretical strain rates
        T = 1000 + 273.15
        strain_rates_theory = [A * (s ** n) * np.exp(-Q / (R * T)) for s in stresses]
        
        # Plot
        ax.loglog(stresses, strain_rates_exp, 'o', label='Data (1000?C)', 
                 color=colors[idx], markersize=8)
        ax.loglog(stresses, strain_rates_theory, '--', label='Norton Law Fit', 
                 color=colors[idx], linewidth=2)
        
        ax.set_xlabel('Stress (MPa)', fontsize=11)
        ax.set_ylabel('Strain Rate (s??)', fontsize=11)
        ax.set_title(f'{layer.capitalize()}: A={A:.2e}, n={n:.1f}, Q={Q} kJ/mol', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    
    # Temperature dependence plot (electrolyte)
    ax = axes[1, 1]
    layer = 'electrolyte'
    params = dataset['layers'][layer]['creep_constitutive']['parameters']
    A = params['A_pre_exponential']['sintering_state']
    n = params['n_stress_exponent']['sintering_state']
    Q = params['Q_activation_energy']['sintering_state']
    
    stress = 10.0  # MPa
    temps = np.linspace(800, 1200, 50)
    strain_rates = [A * (stress ** n) * np.exp(-Q / (R * (T + 273.15))) for T in temps]
    
    ax.semilogy(temps, strain_rates, 'b-', linewidth=2)
    ax.set_xlabel('Temperature (?C)', fontsize=11)
    ax.set_ylabel('Strain Rate (s??)', fontsize=11)
    ax.set_title(f'{layer.capitalize()}: Temperature Dependence (?={stress} MPa)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/creep_parameters.png', dpi=300)
    print("? Generated: figures/creep_parameters.png")
    plt.close()


def plot_elastic_properties():
    """??????"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Elastic Properties vs Temperature', fontsize=14)
    
    layers = ['anode', 'electrolyte', 'cathode', 'interconnect']
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['Ni-YSZ Anode', '8YSZ Electrolyte', 'LSM Cathode', 'Crofer Interconnect']
    
    # Young's Modulus vs Temperature
    ax1 = axes[0, 0]
    for i, layer in enumerate(layers):
        df = pd.read_csv(f'dataset/elastic_{layer}.csv')
        ax1.plot(df['temperature_C'], df['youngs_modulus_gpa'], 
                label=labels[i], color=colors[i], linewidth=2)
    ax1.set_xlabel('Temperature (?C)', fontsize=11)
    ax1.set_ylabel("Young's Modulus (GPa)", fontsize=11)
    ax1.set_title("Young's Modulus vs Temperature", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Poisson's Ratio vs Temperature
    ax2 = axes[0, 1]
    for i, layer in enumerate(layers):
        df = pd.read_csv(f'dataset/elastic_{layer}.csv')
        ax2.plot(df['temperature_C'], df['poissons_ratio'], 
                label=labels[i], color=colors[i], linewidth=2)
    ax2.set_xlabel('Temperature (?C)', fontsize=11)
    ax2.set_ylabel("Poisson's Ratio", fontsize=11)
    ax2.set_title("Poisson's Ratio vs Temperature", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Young's Modulus vs Relative Density (Electrolyte at 800?C)
    ax3 = axes[1, 0]
    df = pd.read_csv('dataset/elastic_density_electrolyte.csv')
    df_800 = df[df['temperature_C'] == 800]
    ax3.plot(df_800['relative_density'], df_800['youngs_modulus_gpa'], 
            'g-', linewidth=2, label='Electrolyte at 800?C')
    ax3.set_xlabel('Relative Density', fontsize=11)
    ax3.set_ylabel("Young's Modulus (GPa)", fontsize=11)
    ax3.set_title("Young's Modulus vs Relative Density", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Temperature-Density Surface (Anode)
    ax4 = axes[1, 1]
    df = pd.read_csv('dataset/elastic_density_anode.csv')
    # Select subset for visualization
    temps_plot = np.linspace(25, 1350, 20)
    densities_plot = np.linspace(0.65, 0.99, 20)
    
    E_matrix = np.zeros((len(densities_plot), len(temps_plot)))
    for i, rho in enumerate(densities_plot):
        for j, T in enumerate(temps_plot):
            df_sub = df[(df['relative_density'] >= rho - 0.01) & 
                       (df['relative_density'] <= rho + 0.01) &
                       (df['temperature_C'] >= T - 10) & 
                       (df['temperature_C'] <= T + 10)]
            if len(df_sub) > 0:
                E_matrix[i, j] = df_sub['youngs_modulus_gpa'].mean()
    
    im = ax4.contourf(temps_plot, densities_plot, E_matrix, levels=20, cmap='viridis')
    ax4.set_xlabel('Temperature (?C)', fontsize=11)
    ax4.set_ylabel('Relative Density', fontsize=11)
    ax4.set_title('Anode: E vs T & Density', fontsize=12)
    plt.colorbar(im, ax=ax4, label='Young\'s Modulus (GPa)')
    
    plt.tight_layout()
    plt.savefig('figures/elastic_properties.png', dpi=300)
    print("? Generated: figures/elastic_properties.png")
    plt.close()


def main():
    """?????????"""
    print("??????????...\n")
    
    plot_sintering_kinetics()
    plot_cte()
    plot_creep_data()
    plot_creep_parameters()
    plot_elastic_properties()
    
    print("\n?????????????")
    print("????? 'figures/' ???")


if __name__ == '__main__':
    main()
