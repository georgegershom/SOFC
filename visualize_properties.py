#!/usr/bin/env python3
"""
Visualization script for SOFC material properties dataset
Creates plots showing key material properties vs temperature
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_dataset():
    """Load the material properties dataset"""
    with open('/workspace/material_properties.json', 'r') as f:
        return json.load(f)

def plot_elastic_properties(dataset):
    """Plot Young's modulus vs temperature"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    materials = ['YSZ', 'Ni', 'Ni-YSZ_composite']
    colors = ['blue', 'red', 'green']
    
    for material, color in zip(materials, colors):
        data = dataset['elastic_properties'][material]
        temp = data['Temperature_K']
        E = np.array(data['Young_Modulus_Pa'], dtype=float) / 1e9  # Convert to GPa
        E_err = np.array(data['Uncertainty_E_Pa'], dtype=float) / 1e9
        
        ax.errorbar(temp, E, yerr=E_err, label=material, 
                   color=color, linewidth=2, capsize=3)
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Young\'s Modulus (GPa)')
    ax.set_title('Young\'s Modulus vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/workspace/youngs_modulus.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fracture_properties(dataset):
    """Plot fracture toughness vs temperature"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    materials = ['YSZ', 'Ni', 'Ni-YSZ_interface', 'YSZ-electrolyte_interface']
    colors = ['blue', 'red', 'orange', 'purple']
    
    for material, color in zip(materials, colors):
        data = dataset['fracture_properties'][material]
        temp = data['Temperature_K']
        K_ic = np.array(data['Fracture_Toughness_MPa_sqrt_m'], dtype=float)
        K_ic_err = np.array(data['Uncertainty_K_ic_MPa_sqrt_m'], dtype=float)
        
        ax.errorbar(temp, K_ic, yerr=K_ic_err, label=material, 
                   color=color, linewidth=2, capsize=3)
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Fracture Toughness (MPa√m)')
    ax.set_title('Fracture Toughness vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale due to large range
    plt.tight_layout()
    plt.savefig('/workspace/fracture_toughness.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cte_properties(dataset):
    """Plot CTE vs temperature"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    materials = ['YSZ', 'Ni', 'Ni-YSZ_composite']
    colors = ['blue', 'red', 'green']
    
    for material, color in zip(materials, colors):
        data = dataset['thermo_physical_properties'][material]
        temp = data['Temperature_K']
        cte = np.array(data['CTE_per_K'], dtype=float) * 1e6  # Convert to ppm/K
        cte_err = np.array(data['Uncertainty_CTE_per_K'], dtype=float) * 1e6
        
        ax.errorbar(temp, cte, yerr=cte_err, label=material, 
                   color=color, linewidth=2, capsize=3)
    
    # Add CTE mismatch
    mismatch_data = dataset['thermo_physical_properties']['CTE_Mismatch']
    temp = mismatch_data['Temperature_K']
    mismatch = np.array(mismatch_data['CTE_Difference_per_K'], dtype=float) * 1e6
    mismatch_err = np.array(mismatch_data['Uncertainty_CTE_Difference_per_K'], dtype=float) * 1e6
    
    ax.errorbar(temp, mismatch, yerr=mismatch_err, label='CTE Mismatch (Ni-YSZ)', 
               color='orange', linewidth=2, capsize=3, linestyle='--')
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('CTE (ppm/K)')
    ax.set_title('Coefficient of Thermal Expansion vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/workspace/cte_properties.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_chemical_expansion(dataset):
    """Plot chemical expansion coefficients"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    materials = ['Ni_to_NiO', 'YSZ_oxygen_vacancy', 'Ni-YSZ_composite_oxidation']
    colors = ['red', 'blue', 'green']
    labels = ['Ni→NiO', 'YSZ O₂ vacancy', 'Ni-YSZ Composite']
    
    for material, color, label in zip(materials, colors, labels):
        data = dataset['chemical_expansion_properties'][material]
        temp = data['Temperature_K']
        chem_exp = np.array(data['Chemical_Expansion_Coefficient'], dtype=float) * 100  # Convert to %
        chem_exp_err = np.array(data['Uncertainty_Chemical_Expansion'], dtype=float) * 100
        
        ax.errorbar(temp, chem_exp, yerr=chem_exp_err, label=label, 
                   color=color, linewidth=2, capsize=3)
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Chemical Expansion (%)')
    ax.set_title('Chemical Expansion Coefficients vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/workspace/chemical_expansion.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(dataset):
    """Create a summary table of key properties at room temperature"""
    summary_data = []
    
    # Room temperature data (300K)
    temp_idx = 0
    
    # Elastic properties
    for material in ['YSZ', 'Ni', 'Ni-YSZ_composite']:
        data = dataset['elastic_properties'][material]
        E = float(data['Young_Modulus_Pa'][temp_idx]) / 1e9  # GPa
        nu = float(data['Poisson_Ratio'][temp_idx])
        E_err = float(data['Uncertainty_E_Pa'][temp_idx]) / 1e9
        
        summary_data.append({
            'Material': material,
            'Property': 'Elastic',
            'E (GPa)': f"{E:.1f} ± {E_err:.1f}",
            'ν': f"{nu:.3f}",
            'Temperature': '300K'
        })
    
    # Fracture properties
    for material in ['YSZ', 'Ni', 'Ni-YSZ_interface']:
        data = dataset['fracture_properties'][material]
        K_ic = float(data['Fracture_Toughness_MPa_sqrt_m'][temp_idx])
        K_ic_err = float(data['Uncertainty_K_ic_MPa_sqrt_m'][temp_idx])
        
        summary_data.append({
            'Material': material,
            'Property': 'Fracture',
            'K_ic (MPa√m)': f"{K_ic:.2f} ± {K_ic_err:.2f}",
            'ν': '-',
            'Temperature': '300K'
        })
    
    # CTE properties
    for material in ['YSZ', 'Ni', 'Ni-YSZ_composite']:
        data = dataset['thermo_physical_properties'][material]
        cte = float(data['CTE_per_K'][temp_idx]) * 1e6  # ppm/K
        cte_err = float(data['Uncertainty_CTE_per_K'][temp_idx]) * 1e6
        
        summary_data.append({
            'Material': material,
            'Property': 'CTE',
            'E (GPa)': f"{cte:.1f} ± {cte_err:.1f}",
            'ν': 'ppm/K',
            'Temperature': '300K'
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv('/workspace/property_summary_300K.csv', index=False)
    
    return df

def main():
    """Main visualization function"""
    print("Loading material properties dataset...")
    dataset = load_dataset()
    
    print("Creating visualizations...")
    
    # Create plots
    plot_elastic_properties(dataset)
    print("✓ Young's modulus plot saved")
    
    plot_fracture_properties(dataset)
    print("✓ Fracture toughness plot saved")
    
    plot_cte_properties(dataset)
    print("✓ CTE properties plot saved")
    
    plot_chemical_expansion(dataset)
    print("✓ Chemical expansion plot saved")
    
    # Create summary table
    summary_df = create_summary_table(dataset)
    print("✓ Property summary table saved")
    
    print("\nVisualization complete!")
    print("Generated files:")
    print("- youngs_modulus.png")
    print("- fracture_toughness.png") 
    print("- cte_properties.png")
    print("- chemical_expansion.png")
    print("- property_summary_300K.csv")
    
    # Print summary table
    print("\nProperty Summary at 300K:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()