#!/usr/bin/env python3
"""
???????????
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# ???????????loader
sys.path.insert(0, str(Path(__file__).parent))
from load_material_data import MaterialDataLoader

# ??????
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_dilatometry_data():
    """????????"""
    loader = MaterialDataLoader()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ????
    ax1 = axes[0]
    for material in ["8YSZ", "NiYSZ"]:
        data = loader.load_dilatometry_data(material)
        ax1.plot(data['Temperature_C'], data['Strain_dL_L0'], 
                marker='o', label=material, linewidth=2, markersize=6)
    
    ax1.set_xlabel('?? (?C)', fontsize=12)
    ax1.set_ylabel('?? (?L/L?)', fontsize=12)
    ax1.set_title('???????? vs ??', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ????
    ax2 = axes[1]
    bilayer_data = loader.load_dilatometry_data("bilayer_anode_electrolyte")
    ax2.plot(bilayer_data['Temperature_C'], bilayer_data['Strain_dL_L0_Anode'], 
            marker='s', label='??', linewidth=2, markersize=6)
    ax2.plot(bilayer_data['Temperature_C'], bilayer_data['Strain_dL_L0_Electrolyte'], 
            marker='o', label='???', linewidth=2, markersize=6)
    ax2.plot(bilayer_data['Temperature_C'], bilayer_data['Mismatch_Strain'], 
            marker='^', label='????', linewidth=2, markersize=6, linestyle='--')
    
    ax2.set_xlabel('?? (?C)', fontsize=12)
    ax2.set_ylabel('?? (?L/L?)', fontsize=12)
    ax2.set_title('???????-?????? vs ??', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'documentation' / 'dilatometry_plots.png', 
                dpi=300, bbox_inches='tight')
    print("???: dilatometry_plots.png")

def plot_CTE_data():
    """?????????"""
    loader = MaterialDataLoader()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    materials = ["8YSZ", "Ni-YSZ", "LSM-YSZ"]
    colors = ['blue', 'red', 'green']
    
    for material, color in zip(materials, colors):
        cte_data = loader.load_CTE_data(material)
        ax.plot(cte_data['Temperature_C'], cte_data['CTE_10minus6_per_K'], 
               marker='o', label=material, linewidth=2, markersize=6, color=color)
        # ?????
        ax.errorbar(cte_data['Temperature_C'], cte_data['CTE_10minus6_per_K'],
                   yerr=cte_data['Standard_Deviation'], color=color, alpha=0.3)
    
    ax.set_xlabel('?? (?C)', fontsize=12)
    ax.set_ylabel('CTE (?10?? K??)', fontsize=12)
    ax.set_title('????? vs ??', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'documentation' / 'CTE_plots.png', 
                dpi=300, bbox_inches='tight')
    print("???: CTE_plots.png")

def plot_creep_data():
    """??????"""
    loader = MaterialDataLoader()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 8YSZ Green??
    ax1 = axes[0, 0]
    creep_data = loader.load_creep_test_data("8YSZ")
    green_800 = creep_data[(creep_data['State'] == 'Green') & (creep_data['Temperature_C'] == 800)]
    green_1200 = creep_data[(creep_data['State'] == 'Green') & (creep_data['Temperature_C'] == 1200)]
    
    for stress in [5, 10]:
        data_800 = green_800[green_800['Stress_MPa'] == stress]
        data_1200 = green_1200[green_1200['Stress_MPa'] == stress]
        ax1.plot(data_800['Time_hours'], data_800['Strain'], 
                marker='o', label=f'800?C, {stress}MPa', linewidth=2)
        ax1.plot(data_1200['Time_hours'], data_1200['Strain'], 
                marker='s', label=f'1200?C, {stress}MPa', linewidth=2)
    
    ax1.set_xlabel('?? (??)', fontsize=11)
    ax1.set_ylabel('??', fontsize=11)
    ax1.set_title('8YSZ Green????', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Ni-YSZ Green??
    ax2 = axes[0, 1]
    creep_data_ni = loader.load_creep_test_data("NiYSZ")
    green_800_ni = creep_data_ni[(creep_data_ni['State'] == 'Green') & (creep_data_ni['Temperature_C'] == 800)]
    green_1200_ni = creep_data_ni[(creep_data_ni['State'] == 'Green') & (creep_data_ni['Temperature_C'] == 1200)]
    
    for stress in [5, 10]:
        data_800 = green_800_ni[green_800_ni['Stress_MPa'] == stress]
        data_1200 = green_1200_ni[green_1200_ni['Stress_MPa'] == stress]
        ax2.plot(data_800['Time_hours'], data_800['Strain'], 
                marker='o', label=f'800?C, {stress}MPa', linewidth=2)
        ax2.plot(data_1200['Time_hours'], data_1200['Strain'], 
                marker='s', label=f'1200?C, {stress}MPa', linewidth=2)
    
    ax2.set_xlabel('?? (??)', fontsize=11)
    ax2.set_ylabel('??', fontsize=11)
    ax2.set_title('Ni-YSZ Green????', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ????? vs ??
    ax3 = axes[1, 0]
    temperatures = np.array([800, 1000, 1200])
    stresses = [5, 10]
    
    for stress in stresses:
        strain_rates_8ysz = []
        for T in temperatures:
            rate = loader.get_creep_strain_rate("8YSZ", "Green", T, stress)
            strain_rates_8ysz.append(rate)
        ax3.semilogy(temperatures, strain_rates_8ysz, 'o-', 
                    label=f'8YSZ, {stress}MPa', linewidth=2, markersize=8)
        
        strain_rates_ni = []
        for T in temperatures:
            rate = loader.get_creep_strain_rate("Ni-YSZ", "Green", T, stress)
            strain_rates_ni.append(rate)
        ax3.semilogy(temperatures, strain_rates_ni, 's--', 
                    label=f'Ni-YSZ, {stress}MPa', linewidth=2, markersize=8)
    
    ax3.set_xlabel('?? (?C)', fontsize=11)
    ax3.set_ylabel('????? (s??)', fontsize=11)
    ax3.set_title('????? vs ?? (Green??)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ???? vs ????
    ax4 = axes[1, 1]
    materials = ["8YSZ", "Ni-YSZ", "LSM-YSZ"]
    colors = ['blue', 'red', 'green']
    
    for material, color in zip(materials, colors):
        youngs = loader.load_youngs_modulus(material)
        room_temp = youngs[youngs['Temperature_C'] == 25]
        ax4.plot(room_temp['Relative_Density'], room_temp['Youngs_Modulus_GPa'], 
                marker='o', label=material, linewidth=2, markersize=6, color=color)
    
    ax4.set_xlabel('????', fontsize=11)
    ax4.set_ylabel('???? (GPa)', fontsize=11)
    ax4.set_title('???? vs ???? (25?C)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'documentation' / 'creep_and_elastic_plots.png', 
                dpi=300, bbox_inches='tight')
    print("???: creep_and_elastic_plots.png")

def main():
    """??????"""
    print("?????????????...\n")
    
    plot_dilatometry_data()
    plot_CTE_data()
    plot_creep_data()
    
    print("\n??????????")

if __name__ == "__main__":
    main()
