"""
Example Usage of Atomic-Scale Simulation Data
Demonstrates how to load and use the generated datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path

def example_1_calculate_diffusion_coefficient():
    """
    Example 1: Calculate temperature-dependent diffusion coefficients
    using Arrhenius equation with activation barriers from DFT
    """
    print("=" * 70)
    print("EXAMPLE 1: Temperature-Dependent Diffusion Coefficients")
    print("=" * 70)
    print()
    
    # Load activation barrier data
    data = pd.read_csv('atomic_simulation_data/dft_activation_barriers.csv')
    
    # Filter for vacancy migration in Fe
    fe_vacancy = data[(data['material'] == 'Fe') & 
                      (data['mechanism'] == 'vacancy_migration')]
    
    if len(fe_vacancy) > 0:
        # Take average activation energy
        Q = fe_vacancy['activation_energy_eV'].mean()
        D0 = fe_vacancy['pre_exponential_factor_m2_s'].mean()
        
        print(f"Material: Fe")
        print(f"Mechanism: Vacancy Migration")
        print(f"Activation Energy (Q): {Q:.4f} eV")
        print(f"Pre-exponential Factor (D₀): {D0:.2e} m²/s")
        print()
        
        # Calculate D at different temperatures
        k_B = 8.617e-5  # eV/K
        temperatures = [500, 700, 900, 1100, 1300]
        
        print("Temperature-dependent diffusion coefficients:")
        print(f"{'T (K)':<10} {'D (m²/s)':<15}")
        print("-" * 25)
        
        for T in temperatures:
            D = D0 * np.exp(-Q / (k_B * T))
            print(f"{T:<10} {D:<15.3e}")
        print()


def example_2_gb_sliding_threshold():
    """
    Example 2: Determine grain boundary sliding threshold stress
    """
    print("=" * 70)
    print("EXAMPLE 2: Grain Boundary Sliding Threshold Analysis")
    print("=" * 70)
    print()
    
    # Load GB sliding data
    data = pd.read_csv('atomic_simulation_data/md_grain_boundary_sliding.csv')
    
    # Group by material and calculate mean critical stress
    gb_summary = data.groupby('material')['critical_shear_stress_MPa'].agg(['mean', 'std', 'count'])
    
    print("Critical Shear Stress for Grain Boundary Sliding:")
    print(gb_summary.round(2))
    print()
    
    # Temperature dependence for Ni
    ni_data = data[data['material'] == 'Ni'].copy()
    if len(ni_data) > 0:
        print("Temperature Dependence (Ni):")
        
        # Bin by temperature ranges
        ni_data['T_bin'] = pd.cut(ni_data['temperature_K'], 
                                   bins=[0, 500, 800, 1100, 1500],
                                   labels=['300-500K', '500-800K', '800-1100K', '1100-1500K'])
        
        temp_analysis = ni_data.groupby('T_bin')['critical_shear_stress_MPa'].agg(['mean', 'count'])
        print(temp_analysis.round(2))
        print()


def example_3_dislocation_velocity_law():
    """
    Example 3: Extract dislocation velocity power law parameters
    """
    print("=" * 70)
    print("EXAMPLE 3: Dislocation Velocity Power Law")
    print("=" * 70)
    print()
    
    # Load dislocation mobility data
    data = pd.read_csv('atomic_simulation_data/md_dislocation_mobility.csv')
    
    # Filter for edge dislocations in Al
    al_edge = data[(data['material'] == 'Al') & 
                   (data['dislocation_type'] == 'edge')]
    
    if len(al_edge) > 0:
        print("Material: Al")
        print("Dislocation Type: Edge")
        print()
        print(f"Average Peierls Stress: {al_edge['peierls_stress_MPa'].mean():.2f} ± {al_edge['peierls_stress_MPa'].std():.2f} MPa")
        print(f"Average Mobility Coefficient: {al_edge['mobility_coefficient'].mean():.2e}")
        print(f"Average Stress Exponent: {al_edge['stress_exponent'].mean():.2f}")
        print()
        
        # Power law: v = M * (τ - τ_p)^m
        print("Power Law Form: v = M × (τ - τₚ)^m")
        print("where:")
        print("  v  = dislocation velocity (m/s)")
        print("  M  = mobility coefficient")
        print("  τ  = applied stress (MPa)")
        print("  τₚ = Peierls stress (MPa)")
        print("  m  = stress exponent")
        print()


def example_4_creep_rate_parameterization():
    """
    Example 4: Parameterize a simple creep rate equation
    """
    print("=" * 70)
    print("EXAMPLE 4: Creep Rate Equation Parameterization")
    print("=" * 70)
    print()
    
    # Load activation barriers
    barriers = pd.read_csv('atomic_simulation_data/dft_activation_barriers.csv')
    
    # Use grain boundary diffusion as rate-limiting mechanism
    gb_diff = barriers[barriers['mechanism'] == 'grain_boundary_diffusion']
    
    if len(gb_diff) > 0:
        Q_creep = gb_diff['activation_energy_eV'].mean()
        
        # Convert to kJ/mol for engineering use
        Q_creep_kJ_mol = Q_creep * 96.485  # eV to kJ/mol conversion
        
        print("Creep Rate Equation: ε̇ = A × σⁿ × exp(-Q/RT)")
        print()
        print("Parameters from atomic simulations:")
        print(f"  Activation Energy (Q): {Q_creep:.4f} eV = {Q_creep_kJ_mol:.2f} kJ/mol")
        print()
        print("Typical values for other parameters:")
        print("  A (pre-exponential): ~10⁻⁶ to 10⁻³ (depends on material)")
        print("  n (stress exponent): 3-8 (from dislocation climb/glide)")
        print("  R (gas constant): 8.314 J/(mol·K)")
        print()
        
        # Example calculation
        T = 1000  # K
        sigma = 100  # MPa
        n = 5
        A = 1e-4
        R = 8.314  # J/(mol·K)
        
        creep_rate = A * (sigma ** n) * np.exp(-Q_creep_kJ_mol * 1000 / (R * T))
        
        print(f"Example at T={T}K, σ={sigma}MPa:")
        print(f"  Creep rate ε̇ = {creep_rate:.3e} s⁻¹")
        print()


def example_5_surface_energy_analysis():
    """
    Example 5: Analyze surface energy for cavity nucleation
    """
    print("=" * 70)
    print("EXAMPLE 5: Surface Energy for Cavity Nucleation")
    print("=" * 70)
    print()
    
    # Load surface energy data
    data = pd.read_csv('atomic_simulation_data/dft_surface_energies.csv')
    
    # Find minimum surface energy (most stable orientation)
    by_orientation = data.groupby('surface_orientation')['surface_energy_J_m2'].mean().sort_values()
    
    print("Average Surface Energy by Orientation:")
    for orientation, energy in by_orientation.items():
        print(f"  {orientation}: {energy:.4f} J/m²")
    print()
    
    # Critical cavity size calculation
    gamma = by_orientation.iloc[0]  # Use minimum surface energy
    sigma = 50e6  # Applied stress in Pa (50 MPa)
    
    # Critical radius: r* = 2γ/σ
    r_critical_nm = (2 * gamma / sigma) * 1e9  # Convert to nm
    
    print(f"Critical Cavity Radius (Nucleation Theory):")
    print(f"  Applied Stress: {sigma/1e6:.0f} MPa")
    print(f"  Surface Energy: {gamma:.4f} J/m²")
    print(f"  Critical Radius: {r_critical_nm:.2f} nm")
    print()


def main():
    """Run all examples."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ATOMIC-SCALE DATA USAGE EXAMPLES" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    example_1_calculate_diffusion_coefficient()
    example_2_gb_sliding_threshold()
    example_3_dislocation_velocity_law()
    example_4_creep_rate_parameterization()
    example_5_surface_energy_analysis()
    
    print("=" * 70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Explore the CSV files in atomic_simulation_data/")
    print("  2. Adapt these examples to your specific use case")
    print("  3. Integrate parameters into your continuum models")
    print("  4. Train ML surrogate models on this data")
    print()


if __name__ == '__main__':
    main()
