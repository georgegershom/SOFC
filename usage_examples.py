#!/usr/bin/env python3
"""
Usage Examples for Advanced Creep and Damage Simulation

This file demonstrates various ways to use the creep and damage simulation
code for different research scenarios and customizations.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from enhanced_creep_damage_simulation import EnhancedCreepDamageSimulator, generate_enhanced_figure_4a2

def example_1_basic_simulation():
    """Example 1: Basic simulation with default parameters"""
    print("="*60)
    print("EXAMPLE 1: Basic Simulation")
    print("="*60)
    
    # Initialize simulator
    sim = EnhancedCreepDamageSimulator()
    
    # Define conditions
    sigma = 120  # MPa
    T = 1050     # °C
    T_K = T + 273.15
    
    # Time array
    t = np.linspace(0.1, 100, 1000)
    
    # Calculate creep strain and damage
    epsilon_c, D = sim.enhanced_creep_strain(t, sigma, T_K)
    
    # Find threshold times
    t_creep = sim.find_threshold_time(sigma, T_K, 'creep')
    t_damage = sim.find_threshold_time(sigma, T_K, 'damage')
    t_nuc = min(t_creep, t_damage)
    
    print(f"Conditions: σ = {sigma} MPa, T = {T}°C")
    print(f"Creep threshold time: {t_creep:.1f} min")
    print(f"Damage threshold time: {t_damage:.1f} min")
    print(f"Nucleation time: {t_nuc:.1f} min")
    print(f"Final strain: {epsilon_c[-1]*1e6:.1f} µε")
    print(f"Final damage: {D[-1]:.3f}")

def example_2_parameter_sensitivity():
    """Example 2: Parameter sensitivity analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Parameter Sensitivity Analysis")
    print("="*60)
    
    sim = EnhancedCreepDamageSimulator()
    
    # Base conditions
    sigma = 100  # MPa
    T = 1000     # °C
    T_K = T + 273.15
    
    # Parameter variations
    A_values = [1e-12, 2.5e-12, 5e-12]  # Creep prefactor
    n_values = [4.0, 4.8, 5.5]          # Creep exponent
    Dc_values = [0.2, 0.25, 0.3]        # Critical damage
    
    print("Sensitivity Analysis Results:")
    print("-" * 40)
    
    # Creep prefactor sensitivity
    print("\nCreep Prefactor (A) Sensitivity:")
    for A in A_values:
        sim.A = A
        t_nuc = sim.find_threshold_time(sigma, T_K, 'creep')
        print(f"  A = {A:.1e} s⁻¹ MPa⁻ⁿ → t_nuc = {t_nuc:.1f} min")
    
    # Creep exponent sensitivity
    print("\nCreep Exponent (n) Sensitivity:")
    for n in n_values:
        sim.n = n
        t_nuc = sim.find_threshold_time(sigma, T_K, 'creep')
        print(f"  n = {n:.1f} → t_nuc = {t_nuc:.1f} min")
    
    # Critical damage sensitivity
    print("\nCritical Damage (Dc) Sensitivity:")
    for Dc in Dc_values:
        sim.Dc = Dc
        t_nuc = sim.find_threshold_time(sigma, T_K, 'damage')
        print(f"  Dc = {Dc:.2f} → t_nuc = {t_nuc:.1f} min")

def example_3_operating_envelope():
    """Example 3: Generate operating envelope for specific conditions"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Operating Envelope Generation")
    print("="*60)
    
    sim = EnhancedCreepDamageSimulator()
    
    # Define stress and temperature ranges
    sigma_range = np.linspace(60, 200, 20)
    T_range = np.linspace(900, 1200, 20)
    
    # Calculate safe operating conditions
    safe_conditions = []
    unsafe_conditions = []
    
    for sigma in sigma_range:
        for T in T_range:
            T_K = T + 273.15
            t_creep = sim.find_threshold_time(sigma, T_K, 'creep')
            t_damage = sim.find_threshold_time(sigma, T_K, 'damage')
            t_nuc = min(t_creep, t_damage)
            
            if t_nuc >= sim.t_target:
                safe_conditions.append((sigma, T))
            else:
                unsafe_conditions.append((sigma, T))
    
    print(f"Safe operating conditions: {len(safe_conditions)} points")
    print(f"Unsafe conditions: {len(unsafe_conditions)} points")
    print(f"Safe fraction: {len(safe_conditions)/(len(safe_conditions)+len(unsafe_conditions)):.1%}")
    
    # Find maximum safe stress for each temperature
    print("\nMaximum Safe Stress by Temperature:")
    print("-" * 40)
    for T in [900, 950, 1000, 1050, 1100, 1150, 1200]:
        max_safe_sigma = 0
        for sigma in sigma_range:
            T_K = T + 273.15
            t_creep = sim.find_threshold_time(sigma, T_K, 'creep')
            t_damage = sim.find_threshold_time(sigma, T_K, 'damage')
            t_nuc = min(t_creep, t_damage)
            
            if t_nuc >= sim.t_target:
                max_safe_sigma = sigma
        
        print(f"  T = {T}°C → σ_max = {max_safe_sigma:.0f} MPa")

def example_4_custom_material():
    """Example 4: Custom material parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Material Parameters")
    print("="*60)
    
    # Create custom simulator for different material
    sim = EnhancedCreepDamageSimulator()
    
    # Modify parameters for different material (e.g., steel)
    sim.A = 5e-10      # Higher creep rate
    sim.n = 3.5        # Lower stress sensitivity
    sim.Q = 250e3      # Lower activation energy
    sim.B = 2e-6       # Higher damage rate
    sim.m = 2.0        # Lower damage stress sensitivity
    sim.Dc = 0.4       # Higher critical damage
    sim.epsilon_dot_c_star = 1e-6  # Higher critical creep rate
    
    print("Custom Material Parameters:")
    print(f"  A = {sim.A:.1e} s⁻¹ MPa⁻ⁿ")
    print(f"  n = {sim.n:.1f}")
    print(f"  Q = {sim.Q/1000:.0f} kJ/mol")
    print(f"  B = {sim.B:.1e} s⁻¹ MPa⁻ᵐ")
    print(f"  m = {sim.m:.1f}")
    print(f"  Dc = {sim.Dc:.2f}")
    print(f"  ε̇c* = {sim.epsilon_dot_c_star:.1e} s⁻¹")
    
    # Test with same conditions
    sigma = 100  # MPa
    T = 1000     # °C
    T_K = T + 273.15
    
    t_creep = sim.find_threshold_time(sigma, T_K, 'creep')
    t_damage = sim.find_threshold_time(sigma, T_K, 'damage')
    t_nuc = min(t_creep, t_damage)
    
    print(f"\nResults for σ = {sigma} MPa, T = {T}°C:")
    print(f"  Creep threshold: {t_creep:.1f} min")
    print(f"  Damage threshold: {t_damage:.1f} min")
    print(f"  Nucleation time: {t_nuc:.1f} min")

def example_5_experimental_data_fitting():
    """Example 5: Fit parameters to experimental data"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Parameter Fitting to Experimental Data")
    print("="*60)
    
    # Simulate experimental data
    np.random.seed(42)
    
    # Experimental conditions and observed nucleation times
    exp_conditions = [
        (80, 900, 45.2),   # (σ, T, t_nuc_obs)
        (100, 950, 32.1),
        (120, 1000, 18.7),
        (140, 1050, 12.3),
        (160, 1100, 8.9),
    ]
    
    print("Experimental Data:")
    print("-" * 30)
    for sigma, T, t_obs in exp_conditions:
        print(f"σ = {sigma:3d} MPa, T = {T:4d}°C → t_nuc = {t_obs:5.1f} min")
    
    # Simple parameter fitting (least squares)
    sim = EnhancedCreepDamageSimulator()
    
    def objective_function(params):
        A, n, Q = params
        sim.A = A
        sim.n = n
        sim.Q = Q
        
        error = 0
        for sigma, T, t_obs in exp_conditions:
            T_K = T + 273.15
            t_pred = sim.find_threshold_time(sigma, T_K, 'creep')
            if np.isfinite(t_pred):
                error += (t_pred - t_obs) ** 2
            else:
                error += 1000  # Penalty for infinite times
        
        return error
    
    # Initial guess
    x0 = [sim.A, sim.n, sim.Q]
    
    print(f"\nInitial parameters:")
    print(f"  A = {x0[0]:.1e} s⁻¹ MPa⁻ⁿ")
    print(f"  n = {x0[1]:.1f}")
    print(f"  Q = {x0[2]/1000:.0f} kJ/mol")
    
    # Note: In practice, you would use scipy.optimize.minimize
    # from scipy.optimize import minimize
    # result = minimize(objective_function, x0, method='Nelder-Mead')
    
    print("\nNote: Use scipy.optimize.minimize for actual parameter fitting")
    print("This example shows the framework for parameter identification")

def example_6_batch_analysis():
    """Example 6: Batch analysis for multiple conditions"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Batch Analysis")
    print("="*60)
    
    sim = EnhancedCreepDamageSimulator()
    
    # Define batch of conditions
    sigma_values = [80, 100, 120, 140, 160, 180]
    T_values = [900, 950, 1000, 1050, 1100, 1150]
    
    # Results storage
    results = []
    
    print("Batch Analysis Results:")
    print("-" * 50)
    print("σ (MPa) | T (°C) | t_creep | t_damage | t_nuc  | Status")
    print("-" * 50)
    
    for sigma in sigma_values:
        for T in T_values:
            T_K = T + 273.15
            
            t_creep = sim.find_threshold_time(sigma, T_K, 'creep')
            t_damage = sim.find_threshold_time(sigma, T_K, 'damage')
            t_nuc = min(t_creep, t_damage)
            
            status = "SAFE" if t_nuc >= sim.t_target else "UNSAFE"
            
            results.append({
                'sigma': sigma,
                'T': T,
                't_creep': t_creep,
                't_damage': t_damage,
                't_nuc': t_nuc,
                'status': status
            })
            
            print(f"{sigma:7d} | {T:5d} | {t_creep:7.1f} | {t_damage:7.1f} | {t_nuc:5.1f} | {status}")
    
    # Summary statistics
    safe_count = sum(1 for r in results if r['status'] == 'SAFE')
    total_count = len(results)
    
    print("-" * 50)
    print(f"Summary: {safe_count}/{total_count} conditions are safe ({safe_count/total_count:.1%})")

def example_7_visualization_customization():
    """Example 7: Custom visualization and plotting"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Custom Visualization")
    print("="*60)
    
    sim = EnhancedCreepDamageSimulator()
    
    # Create custom plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Creep curves for different stresses
    sigma_values = [80, 100, 120, 140, 160]
    T = 1000  # °C
    T_K = T + 273.15
    t = np.linspace(0.1, 60, 1000)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_values)))
    
    for i, sigma in enumerate(sigma_values):
        epsilon_c, D = sim.enhanced_creep_strain(t, sigma, T_K)
        ax1.plot(t, epsilon_c * 1e6, color=colors[i], linewidth=2, 
                label=f'σ = {sigma} MPa')
    
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Creep Strain (µε)')
    ax1.set_title(f'Creep Behavior at T = {T}°C')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Damage evolution
    for i, sigma in enumerate(sigma_values):
        D = sim.damage_evolution(t, sigma, T_K)
        ax2.plot(t, D, color=colors[i], linewidth=2, 
                label=f'σ = {sigma} MPa')
    
    ax2.axhline(y=sim.Dc, color='red', linestyle='--', linewidth=2, 
               label=f'Dc = {sim.Dc}')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Damage D')
    ax2.set_title(f'Damage Evolution at T = {T}°C')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/workspace/custom_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Custom visualization saved as 'custom_visualization.png'")

def main():
    """Run all examples"""
    print("Advanced Creep and Damage Simulation - Usage Examples")
    print("=" * 70)
    
    # Run all examples
    example_1_basic_simulation()
    example_2_parameter_sensitivity()
    example_3_operating_envelope()
    example_4_custom_material()
    example_5_experimental_data_fitting()
    example_6_batch_analysis()
    example_7_visualization_customization()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()