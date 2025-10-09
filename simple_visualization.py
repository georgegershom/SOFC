#!/usr/bin/env python3
"""
Simple visualization script for SOFC material properties dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def load_and_plot_data():
    """Load dataset and create simple plots"""
    
    # Load the dataset
    with open('/workspace/material_properties.json', 'r') as f:
        dataset = json.load(f)
    
    # Create a comprehensive summary
    print("=== SOFC Material Property Dataset Summary ===\n")
    
    # Temperature range
    temp_range = dataset['metadata']['temperature_range_K']
    print(f"Temperature Range: {temp_range[0]}K - {temp_range[-1]}K ({len(temp_range)} points)")
    print(f"Materials: {', '.join(dataset['metadata']['materials'])}")
    print()
    
    # Elastic properties summary
    print("=== ELASTIC PROPERTIES (300K) ===")
    print("Material        | Young's Modulus (GPa) | Poisson's Ratio")
    print("-" * 55)
    
    for material in ['YSZ', 'Ni', 'Ni-YSZ_composite']:
        data = dataset['elastic_properties'][material]
        E = data['Young_Modulus_Pa'][0] / 1e9  # First value (300K)
        nu = data['Poisson_Ratio'][0]
        print(f"{material:15} | {E:19.1f} | {nu:.3f}")
    
    print()
    
    # Fracture properties summary
    print("=== FRACTURE PROPERTIES (300K) ===")
    print("Material                | K_ic (MPa√m) | G_c (J/m²)")
    print("-" * 50)
    
    for material in ['YSZ', 'Ni', 'Ni-YSZ_interface', 'YSZ-electrolyte_interface']:
        data = dataset['fracture_properties'][material]
        K_ic = data['Fracture_Toughness_MPa_sqrt_m'][0]
        G_c = data['Critical_Energy_Release_Rate_J_per_m2'][0]
        print(f"{material:23} | {K_ic:11.2f} | {G_c:.1f}")
    
    print()
    
    # CTE properties summary
    print("=== THERMAL EXPANSION (300K) ===")
    print("Material        | CTE (ppm/K)")
    print("-" * 30)
    
    for material in ['YSZ', 'Ni', 'Ni-YSZ_composite']:
        data = dataset['thermo_physical_properties'][material]
        cte = data['CTE_per_K'][0] * 1e6  # Convert to ppm/K
        print(f"{material:15} | {cte:11.1f}")
    
    # CTE Mismatch
    mismatch_data = dataset['thermo_physical_properties']['CTE_Mismatch']
    cte_mismatch = mismatch_data['CTE_Difference_per_K'][0] * 1e6
    print(f"{'CTE Mismatch':15} | {cte_mismatch:11.1f}")
    
    print()
    
    # Chemical expansion summary
    print("=== CHEMICAL EXPANSION (300K) ===")
    print("Material/Process        | Expansion (%)")
    print("-" * 35)
    
    for material in ['Ni_to_NiO', 'YSZ_oxygen_vacancy', 'Ni-YSZ_composite_oxidation']:
        data = dataset['chemical_expansion_properties'][material]
        chem_exp = data['Chemical_Expansion_Coefficient'][0] * 100
        label = material.replace('_', ' ').replace('Ni to NiO', 'Ni→NiO')
        print(f"{label:23} | {chem_exp:11.2f}")
    
    print()
    
    # Key insights
    print("=== KEY INSIGHTS FOR MODELING ===")
    print("1. Interface fracture toughness is CRITICAL:")
    print("   - Ni-YSZ interface: K_ic ≈ 0.5 MPa√m (very low)")
    print("   - This is the most likely failure location")
    print()
    print("2. CTE Mismatch drives residual stresses:")
    print(f"   - Ni CTE: {dataset['thermo_physical_properties']['Ni']['CTE_per_K'][0]*1e6:.1f} ppm/K")
    print(f"   - YSZ CTE: {dataset['thermo_physical_properties']['YSZ']['CTE_per_K'][0]*1e6:.1f} ppm/K")
    print(f"   - Mismatch: {cte_mismatch:.1f} ppm/K (significant)")
    print()
    print("3. Chemical expansion is substantial:")
    print("   - Ni→NiO: 6.7% linear expansion")
    print("   - Critical for redox cycling analysis")
    print()
    print("4. Temperature effects are significant:")
    print("   - Young's modulus decreases with temperature")
    print("   - Fracture toughness may decrease at high T")
    print("   - CTE increases slightly with temperature")
    
    # Create simple plots
    create_simple_plots(dataset)

def create_simple_plots(dataset):
    """Create simple property plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Young's Modulus vs Temperature
    ax1 = axes[0, 0]
    materials = ['YSZ', 'Ni', 'Ni-YSZ_composite']
    colors = ['blue', 'red', 'green']
    
    for material, color in zip(materials, colors):
        data = dataset['elastic_properties'][material]
        temp = data['Temperature_K']
        E = [float(x) for x in data['Young_Modulus_Pa']]
        E = [x/1e9 for x in E]  # Convert to GPa
        
        ax1.plot(temp, E, label=material, color=color, linewidth=2)
    
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Young\'s Modulus (GPa)')
    ax1.set_title('Young\'s Modulus vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fracture Toughness vs Temperature
    ax2 = axes[0, 1]
    materials = ['YSZ', 'Ni', 'Ni-YSZ_interface']
    colors = ['blue', 'red', 'orange']
    
    for material, color in zip(materials, colors):
        data = dataset['fracture_properties'][material]
        temp = data['Temperature_K']
        K_ic = [float(x) for x in data['Fracture_Toughness_MPa_sqrt_m']]
        
        ax2.plot(temp, K_ic, label=material, color=color, linewidth=2)
    
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Fracture Toughness (MPa√m)')
    ax2.set_title('Fracture Toughness vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: CTE vs Temperature
    ax3 = axes[1, 0]
    materials = ['YSZ', 'Ni', 'Ni-YSZ_composite']
    colors = ['blue', 'red', 'green']
    
    for material, color in zip(materials, colors):
        data = dataset['thermo_physical_properties'][material]
        temp = data['Temperature_K']
        cte = [float(x) for x in data['CTE_per_K']]
        cte = [x*1e6 for x in cte]  # Convert to ppm/K
        
        ax3.plot(temp, cte, label=material, color=color, linewidth=2)
    
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('CTE (ppm/K)')
    ax3.set_title('Coefficient of Thermal Expansion vs Temperature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Chemical Expansion
    ax4 = axes[1, 1]
    materials = ['Ni_to_NiO', 'YSZ_oxygen_vacancy']
    colors = ['red', 'blue']
    labels = ['Ni→NiO', 'YSZ O₂ vacancy']
    
    for material, color, label in zip(materials, colors, labels):
        data = dataset['chemical_expansion_properties'][material]
        temp = data['Temperature_K']
        chem_exp = [float(x) for x in data['Chemical_Expansion_Coefficient']]
        chem_exp = [x*100 for x in chem_exp]  # Convert to %
        
        ax4.plot(temp, chem_exp, label=label, color=color, linewidth=2)
    
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Chemical Expansion (%)')
    ax4.set_title('Chemical Expansion vs Temperature')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/material_properties_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Summary plot saved as 'material_properties_summary.png'")

if __name__ == "__main__":
    load_and_plot_data()