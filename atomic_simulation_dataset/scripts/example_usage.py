#!/usr/bin/env python3
"""
Example usage of the atomic-scale simulation dataset.
Demonstrates how to load, analyze, and use the data for model parameterization.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_json_data(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def example_1_vacancy_analysis():
    """Example 1: Analyze vacancy formation energies"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Vacancy Formation Energy Analysis")
    print("="*60)
    
    # Load data
    data = load_json_data('dft_calculations/defect_energies/vacancy_formation.json')
    
    # Extract energies and site types
    energies = []
    site_types = []
    for config in data['configurations']:
        energies.append(config['formation_energy'])
        site_types.append(config['vacancy_site']['type'])
    
    # Statistical analysis
    energies = np.array(energies)
    print(f"\nVacancy Formation Energies:")
    print(f"  Mean: {energies.mean():.3f} eV")
    print(f"  Std Dev: {energies.std():.3f} eV")
    print(f"  Range: [{energies.min():.3f}, {energies.max():.3f}] eV")
    
    # Site-specific analysis
    site_energy_map = {}
    for site, energy in zip(site_types, energies):
        if site not in site_energy_map:
            site_energy_map[site] = []
        site_energy_map[site].append(energy)
    
    print(f"\nBy Site Type:")
    for site, site_energies in site_energy_map.items():
        print(f"  {site}: {np.mean(site_energies):.3f} ± {np.std(site_energies):.3f} eV")
    
    return energies

def example_2_arrhenius_fit():
    """Example 2: Fit Arrhenius relationship to GB sliding data"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Arrhenius Fit for GB Sliding")
    print("="*60)
    
    # Load data
    data = load_json_data('md_simulations/grain_boundary/gb_sliding.json')
    
    # Extract temperature and sliding rate data
    temperatures = []
    sliding_rates = []
    for sim in data['simulations']:
        temperatures.append(sim['temperature'])
        sliding_rates.append(sim['analysis']['average_sliding_rate'])
    
    temperatures = np.array(temperatures)
    sliding_rates = np.array(sliding_rates)
    
    # Arrhenius fit: rate = A * exp(-E_a / (k_B * T))
    def arrhenius(T, A, E_a):
        k_B = 8.617e-5  # eV/K
        return A * np.exp(-E_a / (k_B * T))
    
    # Perform fit (use log-linear for stability)
    inv_T = 1000 / temperatures
    log_rates = np.log(sliding_rates + 1e-10)  # Add small value to avoid log(0)
    
    # Linear fit in log space
    coeffs = np.polyfit(inv_T, log_rates, 1)
    E_a = -coeffs[0] * 8.617e-5 * 1000  # Convert to eV
    A = np.exp(coeffs[1])
    
    print(f"\nArrhenius Parameters:")
    print(f"  Activation Energy: {E_a:.3f} eV")
    print(f"  Pre-exponential Factor: {A:.3e} Å/ps")
    print(f"  R² value: {np.corrcoef(inv_T, log_rates)[0,1]**2:.4f}")
    
    return temperatures, sliding_rates, E_a, A

def example_3_dislocation_mobility():
    """Example 3: Analyze dislocation mobility by type"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Dislocation Mobility Analysis")
    print("="*60)
    
    # Load data
    data = load_json_data('md_simulations/dislocation/dislocation_mobility.json')
    
    # Organize by dislocation type
    mobility_by_type = {'edge': [], 'screw': [], 'mixed': []}
    stress_velocity = {'edge': [], 'screw': [], 'mixed': []}
    
    for sim in data['simulations']:
        disl_type = sim['dislocation']['type']
        mobility = sim['analysis']['mobility']
        velocity = sim['analysis']['average_velocity']
        stress = sim['applied_stress']
        
        if disl_type in mobility_by_type:
            mobility_by_type[disl_type].append(mobility)
            stress_velocity[disl_type].append((stress, velocity))
    
    # Statistical analysis
    print("\nMobility Statistics (Å·ps⁻¹·MPa⁻¹):")
    for disl_type, mobilities in mobility_by_type.items():
        mobilities = np.array(mobilities)
        print(f"  {disl_type:6s}: {mobilities.mean():.3e} ± {mobilities.std():.3e}")
    
    # Power law fit: v = B * σ^n
    print("\nPower Law Fits (v = B·σⁿ):")
    for disl_type, sv_data in stress_velocity.items():
        if sv_data:
            stresses, velocities = zip(*sv_data)
            stresses = np.array(stresses)
            velocities = np.array(velocities)
            
            # Log-log fit
            log_s = np.log(stresses)
            log_v = np.log(velocities + 1e-10)
            n, log_B = np.polyfit(log_s, log_v, 1)
            B = np.exp(log_B)
            
            print(f"  {disl_type:6s}: n = {n:.2f}, B = {B:.3e}")
    
    return mobility_by_type

def example_4_parameterize_phase_field():
    """Example 4: Extract parameters for phase-field model"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Phase-Field Model Parameters")
    print("="*60)
    
    # Load various data sources
    vacancy_data = load_json_data('dft_calculations/defect_energies/vacancy_formation.json')
    gb_data = load_json_data('dft_calculations/defect_energies/grain_boundary_energies.json')
    barrier_data = load_json_data('dft_calculations/activation_barriers/diffusion_barriers.json')
    surface_data = load_json_data('dft_calculations/surface_energies/surface_energies.json')
    
    # Extract key parameters
    parameters = {}
    
    # Vacancy parameters
    E_vac = np.mean([c['formation_energy'] for c in vacancy_data['configurations']])
    parameters['vacancy_formation_energy'] = E_vac
    
    # GB parameters
    gb_energies = [c['gb_energy'] for c in gb_data['configurations']]
    parameters['grain_boundary_energy'] = np.mean(gb_energies)
    parameters['grain_boundary_mobility_prefactor'] = np.mean(
        [c['mobility_prefactor'] for c in gb_data['configurations']]
    )
    
    # Diffusion parameters
    vacancy_barriers = [p['activation_energy'] 
                       for p in barrier_data['diffusion_paths'] 
                       if p['mechanism'] == 'vacancy']
    parameters['vacancy_migration_barrier'] = np.mean(vacancy_barriers)
    
    # Surface energy
    surface_energies = [s['surface_energy'] for s in surface_data['surfaces']]
    parameters['surface_energy'] = np.mean(surface_energies)
    
    # Temperature-dependent diffusivity
    k_B = 8.617e-5  # eV/K
    T_ref = 900  # K
    D_0 = 1e-4  # cm²/s (typical)
    E_m = parameters['vacancy_migration_barrier']
    parameters['diffusivity_900K'] = D_0 * np.exp(-(E_vac + E_m) / (k_B * T_ref))
    
    print("\nExtracted Phase-Field Parameters:")
    for param, value in parameters.items():
        if isinstance(value, float):
            if value < 1e-3 or value > 1e3:
                print(f"  {param}: {value:.3e}")
            else:
                print(f"  {param}: {value:.3f}")
    
    return parameters

def example_5_creep_model():
    """Example 5: Parameterize power-law creep model"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Creep Model Parameterization")
    print("="*60)
    
    # Load GB sliding and thermal activation data
    gb_data = load_json_data('md_simulations/grain_boundary/gb_sliding.json')
    thermal_data = load_json_data('md_simulations/thermal_activation.json')
    
    # Extract creep-relevant data
    stresses = []
    rates = []
    temperatures = []
    
    for sim in gb_data['simulations']:
        stresses.append(sim['applied_stress'])
        rates.append(sim['analysis']['average_sliding_rate'])
        temperatures.append(sim['temperature'])
    
    stresses = np.array(stresses)
    rates = np.array(rates)
    temperatures = np.array(temperatures)
    
    # Power-law creep: ε̇ = A * σⁿ * exp(-Q/RT)
    # Fit at median temperature first
    T_median = np.median(temperatures)
    mask = np.abs(temperatures - T_median) < 50  # Within 50K of median
    
    if np.any(mask):
        stress_subset = stresses[mask]
        rate_subset = rates[mask]
        
        # Fit power law exponent
        log_stress = np.log(stress_subset)
        log_rate = np.log(rate_subset + 1e-10)
        n, log_A = np.polyfit(log_stress, log_rate, 1)
        
        print(f"\nPower-Law Creep Parameters:")
        print(f"  Stress exponent (n): {n:.2f}")
        print(f"  Reference temperature: {T_median:.0f} K")
        
        # Estimate activation energy from thermal data
        thermal_df = pd.DataFrame(thermal_data['measurements'])
        if 'creep_rate' in thermal_df.columns:
            T_thermal = thermal_df['temperature'].values
            creep_thermal = thermal_df['creep_rate'].values
            
            # Arrhenius fit
            inv_T = 1000 / T_thermal
            log_creep = np.log(creep_thermal + 1e-20)
            Q_slope = np.polyfit(inv_T, log_creep, 1)[0]
            Q = -Q_slope * 8.314 / 1000  # kJ/mol
            
            print(f"  Activation energy (Q): {Q:.1f} kJ/mol ({Q/96.485:.3f} eV)")
        
        # Creep mechanism map regions
        print(f"\nCreep Mechanism Indicators:")
        print(f"  Low stress (<100 MPa): Diffusion creep dominant")
        print(f"  Medium stress (100-500 MPa): Power-law creep")
        print(f"  High stress (>500 MPa): Dislocation glide")
    
    return n if 'n' in locals() else None

def example_6_ml_ready_dataset():
    """Example 6: Prepare dataset for machine learning"""
    print("\n" + "="*60)
    print("EXAMPLE 6: ML-Ready Dataset Preparation")
    print("="*60)
    
    # Combine multiple data sources into training dataset
    features = []
    targets = []
    
    # Add vacancy data
    vacancy_data = load_json_data('dft_calculations/defect_energies/vacancy_formation.json')
    for config in vacancy_data['configurations']:
        feature_vec = [
            config['relaxation_volume'],
            config['magnetic_moment'],
            1 if config['vacancy_site']['type'] == 'bulk' else 0,
            1 if config['vacancy_site']['type'] == 'near_gb' else 0,
        ]
        features.append(feature_vec)
        targets.append(config['formation_energy'])
    
    features = np.array(features)
    targets = np.array(targets)
    
    print(f"\nML Dataset Summary:")
    print(f"  Number of samples: {len(features)}")
    print(f"  Number of features: {features.shape[1]}")
    print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}] eV")
    
    # Split into training and test sets
    n_train = int(0.8 * len(features))
    indices = np.random.permutation(len(features))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, y_train = features[train_idx], targets[train_idx]
    X_test, y_test = features[test_idx], targets[test_idx]
    
    print(f"\nTrain/Test Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Simple linear regression as baseline
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nBaseline Model Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.4f} eV")
    print(f"  R² Score: {r2:.4f}")
    
    return X_train, y_train, X_test, y_test

def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("# ATOMIC-SCALE SIMULATION DATASET USAGE EXAMPLES")
    print("#"*60)
    
    # Check if sklearn is available for ML example
    try:
        import sklearn
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        print("\nNote: scikit-learn not installed. Skipping ML example.")
    
    # Run examples
    try:
        # Example 1: Vacancy analysis
        energies = example_1_vacancy_analysis()
        
        # Example 2: Arrhenius fit
        T, rates, E_a, A = example_2_arrhenius_fit()
        
        # Example 3: Dislocation mobility
        mobility = example_3_dislocation_mobility()
        
        # Example 4: Phase-field parameters
        pf_params = example_4_parameterize_phase_field()
        
        # Example 5: Creep model
        n_creep = example_5_creep_model()
        
        # Example 6: ML dataset (if sklearn available)
        if has_sklearn:
            X_train, y_train, X_test, y_test = example_6_ml_ready_dataset()
        
        print("\n" + "="*60)
        print("EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nThe dataset is ready for use in:")
        print("  • Phase-field simulations")
        print("  • Crystal plasticity models")
        print("  • Machine learning training")
        print("  • Continuum creep models")
        print("  • Multiscale material simulations")
        
    except Exception as e:
        print(f"\nError in examples: {e}")
        print("Make sure to run from the atomic_simulation_dataset directory")

if __name__ == "__main__":
    main()