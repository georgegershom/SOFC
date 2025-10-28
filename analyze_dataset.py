#!/usr/bin/env python3
"""
Dataset Analysis Script
Demonstrates how to load and analyze the atomic simulation dataset
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_dataset():
    """Load the complete dataset"""
    data_dir = Path("atomic_simulation_data")
    
    # Load all datasets
    with open(data_dir / "dft_simulations.json", 'r') as f:
        dft_data = json.load(f)
    
    with open(data_dir / "md_simulations.json", 'r') as f:
        md_data = json.load(f)
    
    with open(data_dir / "defect_simulations.json", 'r') as f:
        defect_data = json.load(f)
    
    return dft_data, md_data, defect_data

def analyze_dft_correlations(dft_data):
    """Analyze correlations in DFT data"""
    print("=== DFT Data Analysis ===")
    
    # Extract key properties
    materials = [sim['material'] for sim in dft_data]
    formation_energies = [sim['results']['formation_energy'] for sim in dft_data]
    activation_barriers = [sim['results']['activation_barrier'] for sim in dft_data]
    surface_energies = [sim['results']['surface_energy'] for sim in dft_data]
    bulk_moduli = [sim['results']['bulk_modulus'] for sim in dft_data]
    
    # Create DataFrame
    df_dft = pd.DataFrame({
        'material': materials,
        'formation_energy': formation_energies,
        'activation_barrier': activation_barriers,
        'surface_energy': surface_energies,
        'bulk_modulus': bulk_moduli
    })
    
    # Print statistics
    print(f"Total DFT simulations: {len(dft_data)}")
    print(f"Materials: {df_dft['material'].unique()}")
    print(f"Formation energy range: {df_dft['formation_energy'].min():.3f} to {df_dft['formation_energy'].max():.3f} eV/atom")
    print(f"Activation barrier range: {df_dft['activation_barrier'].min():.3f} to {df_dft['activation_barrier'].max():.3f} eV")
    
    # Calculate correlations
    correlation_matrix = df_dft[['formation_energy', 'activation_barrier', 'surface_energy', 'bulk_modulus']].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))
    
    return df_dft

def analyze_md_temperature_dependence(md_data):
    """Analyze temperature dependence in MD data"""
    print("\n=== MD Data Analysis ===")
    
    # Extract properties
    materials = [sim['material'] for sim in md_data]
    temperatures = [sim['parameters']['temperature'] for sim in md_data]
    diffusion_coeffs = [sim['results']['diffusion_coefficient'] for sim in md_data]
    dislocation_mobilities = [sim['results']['dislocation_mobility'] for sim in md_data]
    viscosities = [sim['results']['viscosity'] for sim in md_data]
    
    # Create DataFrame
    df_md = pd.DataFrame({
        'material': materials,
        'temperature': temperatures,
        'diffusion_coefficient': diffusion_coeffs,
        'dislocation_mobility': dislocation_mobilities,
        'viscosity': viscosities
    })
    
    print(f"Total MD simulations: {len(md_data)}")
    print(f"Temperature range: {df_md['temperature'].min():.1f} to {df_md['temperature'].max():.1f} K")
    print(f"Diffusion coefficient range: {df_md['diffusion_coefficient'].min():.2e} to {df_md['diffusion_coefficient'].max():.2e} m²/s")
    
    # Analyze Arrhenius behavior
    for material in df_md['material'].unique():
        material_data = df_md[df_md['material'] == material]
        if len(material_data) > 10:  # Need enough points for fitting
            # Fit Arrhenius equation: D = D0 * exp(-Q/RT)
            T = material_data['temperature'].values
            D = material_data['diffusion_coefficient'].values
            
            # Linear fit in log space: ln(D) = ln(D0) - Q/(RT)
            log_D = np.log(D)
            inv_T = 1 / T
            
            # Simple linear regression
            coeffs = np.polyfit(inv_T, log_D, 1)
            Q = -coeffs[0] * 8.617e-5  # Convert to eV (R in eV/K)
            D0 = np.exp(coeffs[1])
            
            print(f"{material}: Q = {Q:.3f} eV, D0 = {D0:.2e} m²/s")
    
    return df_md

def analyze_defect_energies(defect_data):
    """Analyze defect formation energies"""
    print("\n=== Defect Data Analysis ===")
    
    # Extract properties
    materials = [sim['material'] for sim in defect_data]
    defect_types = [sim['defect_type'] for sim in defect_data]
    formation_energies = [sim['formation_energy'] for sim in defect_data]
    migration_barriers = [sim['migration_barrier'] for sim in defect_data]
    
    # Create DataFrame
    df_defect = pd.DataFrame({
        'material': materials,
        'defect_type': defect_types,
        'formation_energy': formation_energies,
        'migration_barrier': migration_barriers
    })
    
    print(f"Total defect simulations: {len(defect_data)}")
    print(f"Defect types: {df_defect['defect_type'].unique()}")
    
    # Analyze by defect type
    print("\nFormation energies by defect type:")
    defect_stats = df_defect.groupby('defect_type')['formation_energy'].agg(['mean', 'std', 'count'])
    print(defect_stats.round(3))
    
    return df_defect

def create_visualizations(df_dft, df_md, df_defect):
    """Create comprehensive visualizations"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Formation energy vs activation barrier
    for material in df_dft['material'].unique():
        material_data = df_dft[df_dft['material'] == material]
        axes[0, 0].scatter(material_data['formation_energy'], 
                          material_data['activation_barrier'], 
                          label=material, alpha=0.7)
    axes[0, 0].set_xlabel('Formation Energy (eV/atom)')
    axes[0, 0].set_ylabel('Activation Barrier (eV)')
    axes[0, 0].set_title('DFT: Formation Energy vs Activation Barrier')
    axes[0, 0].legend()
    
    # 2. Temperature vs diffusion coefficient
    for material in df_md['material'].unique():
        material_data = df_md[df_md['material'] == material]
        axes[0, 1].scatter(material_data['temperature'], 
                          material_data['diffusion_coefficient'], 
                          label=material, alpha=0.7)
    axes[0, 1].set_xlabel('Temperature (K)')
    axes[0, 1].set_ylabel('Diffusion Coefficient (m²/s)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('MD: Temperature vs Diffusion Coefficient')
    axes[0, 1].legend()
    
    # 3. Defect formation energies
    defect_types = df_defect['defect_type'].unique()
    defect_energies = [df_defect[df_defect['defect_type'] == dt]['formation_energy'].values 
                      for dt in defect_types]
    axes[0, 2].boxplot(defect_energies, labels=defect_types)
    axes[0, 2].set_ylabel('Formation Energy (eV)')
    axes[0, 2].set_title('Defect Formation Energies')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Material comparison - formation energies
    material_energies = [df_dft[df_dft['material'] == mat]['formation_energy'].values 
                        for mat in df_dft['material'].unique()]
    axes[1, 0].boxplot(material_energies, labels=df_dft['material'].unique())
    axes[1, 0].set_ylabel('Formation Energy (eV/atom)')
    axes[1, 0].set_title('Formation Energy by Material')
    
    # 5. Temperature vs dislocation mobility
    for material in df_md['material'].unique():
        material_data = df_md[df_md['material'] == material]
        axes[1, 1].scatter(material_data['temperature'], 
                          material_data['dislocation_mobility'], 
                          label=material, alpha=0.7)
    axes[1, 1].set_xlabel('Temperature (K)')
    axes[1, 1].set_ylabel('Dislocation Mobility (m/s/MPa)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Temperature vs Dislocation Mobility')
    axes[1, 1].legend()
    
    # 6. Surface energy vs bulk modulus
    for material in df_dft['material'].unique():
        material_data = df_dft[df_dft['material'] == material]
        axes[1, 2].scatter(material_data['surface_energy'], 
                          material_data['bulk_modulus'], 
                          label=material, alpha=0.7)
    axes[1, 2].set_xlabel('Surface Energy (J/m²)')
    axes[1, 2].set_ylabel('Bulk Modulus (GPa)')
    axes[1, 2].set_title('Surface Energy vs Bulk Modulus')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    print("Loading atomic simulation dataset...")
    dft_data, md_data, defect_data = load_dataset()
    
    print("Analyzing dataset...")
    df_dft = analyze_dft_correlations(dft_data)
    df_md = analyze_md_temperature_dependence(md_data)
    df_defect = analyze_defect_energies(defect_data)
    
    print("\nCreating visualizations...")
    create_visualizations(df_dft, df_md, df_defect)
    
    print("\nAnalysis complete! Check 'dataset_analysis.png' for visualizations.")
    
    # Save analysis results
    analysis_results = {
        'dft_summary': df_dft.describe().to_dict(),
        'md_summary': df_md.describe().to_dict(),
        'defect_summary': df_defect.describe().to_dict(),
        'correlations': df_dft[['formation_energy', 'activation_barrier', 'surface_energy', 'bulk_modulus']].corr().to_dict()
    }
    
    with open('analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print("Analysis results saved to 'analysis_results.json'")

if __name__ == "__main__":
    main()