"""
Synthetic Dataset Generator for Process & Microstructure in Sintering
This script generates realistic synthetic data linking sintering parameters to microstructure characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_sintering_dataset(n_samples=200):
    """
    Generate synthetic sintering process and microstructure dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of experimental samples to generate
        
    Returns:
    --------
    pd.DataFrame : Dataset with sintering parameters and resulting microstructure
    """
    
    print(f"Generating {n_samples} synthetic sintering experiments...")
    
    # ============================================================================
    # SINTERING PARAMETERS (INPUTS)
    # ============================================================================
    
    # Temperature Profile Parameters
    ramp_rate = np.random.uniform(2, 10, n_samples)  # °C/min
    hold_temperature = np.random.uniform(1200, 1600, n_samples)  # °C
    hold_time = np.random.uniform(0.5, 6, n_samples)  # hours
    cooling_rate = np.random.uniform(3, 15, n_samples)  # °C/min
    
    # Applied Pressure
    applied_pressure = np.random.choice([0, 5, 10, 15, 20, 25, 30], n_samples)  # MPa
    
    # Green Body Characteristics
    initial_relative_density = np.random.uniform(0.45, 0.65, n_samples)  # fraction
    initial_mean_pore_size = np.random.uniform(0.5, 3.0, n_samples)  # μm
    initial_porosity = (1 - initial_relative_density) * 100  # %
    
    # Atmosphere (categorical)
    atmosphere_options = ['Air', 'Nitrogen', 'Argon', 'Vacuum']
    atmosphere = np.random.choice(atmosphere_options, n_samples)
    
    # Material batch (introduces slight variation)
    batch_id = np.random.randint(1, 6, n_samples)
    
    # ============================================================================
    # PHYSICS-BASED MICROSTRUCTURE CALCULATION (OUTPUTS)
    # ============================================================================
    
    # Effective sintering temperature (normalized)
    T_norm = (hold_temperature - 1200) / 400  # Normalized to [0,1]
    
    # Time-temperature integral (sintering effectiveness)
    sintering_effectiveness = T_norm * np.sqrt(hold_time) * (1 + applied_pressure/100)
    
    # --- Grain Size Distribution ---
    # Grain growth follows power law with temperature and time
    base_grain_size = 0.5  # μm
    grain_growth_factor = np.exp(0.8 * T_norm) * np.power(hold_time, 0.3)
    
    # Pressure slightly suppresses grain growth
    pressure_effect = 1 - 0.01 * applied_pressure
    
    mean_grain_size = base_grain_size * grain_growth_factor * pressure_effect
    # Add realistic noise (5-10% variation)
    mean_grain_size = mean_grain_size * np.random.uniform(0.92, 1.08, n_samples)
    
    # Standard deviation of grain size (typically 30-50% of mean)
    grain_size_std = mean_grain_size * np.random.uniform(0.3, 0.5, n_samples)
    
    # Grain size range
    d10_grain = mean_grain_size * np.random.uniform(0.4, 0.6, n_samples)  # 10th percentile
    d90_grain = mean_grain_size * np.random.uniform(1.6, 2.2, n_samples)  # 90th percentile
    
    # --- Porosity and Density ---
    # Densification is driven by temperature, time, and pressure
    densification_rate = 0.15 * T_norm + 0.08 * np.log1p(hold_time) + 0.002 * applied_pressure
    
    # Final relative density depends on initial density and densification
    final_relative_density = initial_relative_density + densification_rate * (1 - initial_relative_density)
    
    # Atmosphere effect (reducing atmosphere slightly improves densification)
    atmosphere_bonus = np.where(atmosphere == 'Air', 0, 
                                np.where(atmosphere == 'Vacuum', 0.03, 0.015))
    final_relative_density = np.minimum(final_relative_density + atmosphere_bonus, 0.99)
    
    # Add realistic measurement noise
    final_relative_density = final_relative_density * np.random.uniform(0.98, 1.02, n_samples)
    final_relative_density = np.clip(final_relative_density, 0.55, 0.99)
    
    # Calculate final porosity
    final_porosity = (1 - final_relative_density) * 100  # %
    
    # --- Pore Size Distribution ---
    # Pores shrink during sintering but some coarsen
    pore_shrinkage_factor = np.exp(-0.6 * sintering_effectiveness)
    mean_pore_size = initial_mean_pore_size * pore_shrinkage_factor
    mean_pore_size = mean_pore_size * np.random.uniform(0.9, 1.1, n_samples)
    mean_pore_size = np.maximum(mean_pore_size, 0.1)  # Minimum pore size
    
    pore_size_std = mean_pore_size * np.random.uniform(0.4, 0.7, n_samples)
    
    # --- Grain Boundary Characteristics ---
    # Grain boundary thickness (nm)
    gb_thickness = np.random.uniform(0.5, 2.0, n_samples)
    
    # Grain boundary energy (related to temperature and atmosphere)
    # Higher temperature annealing → lower grain boundary energy
    gb_energy = 0.8 - 0.15 * T_norm + np.random.normal(0, 0.05, n_samples)  # J/m²
    gb_energy = np.clip(gb_energy, 0.4, 1.0)
    
    # Grain boundary coverage by secondary phase (%)
    gb_phase_coverage = np.random.uniform(0, 15, n_samples)
    
    # --- Additional Microstructural Features ---
    # Coordination number (average number of grain neighbors)
    coordination_number = 6 + 8 * (final_relative_density - 0.5) / 0.5
    coordination_number = coordination_number + np.random.normal(0, 0.5, n_samples)
    coordination_number = np.clip(coordination_number, 4, 14)
    
    # Pore connectivity (higher in less dense samples)
    pore_connectivity = 100 * (1 - final_relative_density)**2
    pore_connectivity = pore_connectivity * np.random.uniform(0.8, 1.2, n_samples)
    
    # ============================================================================
    # CREATE DATAFRAME
    # ============================================================================
    
    data = {
        # Experimental ID
        'Sample_ID': [f'SINT_{i+1:03d}' for i in range(n_samples)],
        'Batch_ID': batch_id,
        
        # SINTERING PARAMETERS (INPUTS)
        'Ramp_Rate_C_per_min': np.round(ramp_rate, 2),
        'Hold_Temperature_C': np.round(hold_temperature, 1),
        'Hold_Time_hours': np.round(hold_time, 2),
        'Cooling_Rate_C_per_min': np.round(cooling_rate, 2),
        'Applied_Pressure_MPa': applied_pressure,
        'Atmosphere': atmosphere,
        'Initial_Relative_Density': np.round(initial_relative_density, 3),
        'Initial_Porosity_percent': np.round(initial_porosity, 2),
        'Initial_Mean_Pore_Size_um': np.round(initial_mean_pore_size, 2),
        
        # RESULTING MICROSTRUCTURE (OUTPUTS)
        'Final_Relative_Density': np.round(final_relative_density, 3),
        'Final_Porosity_percent': np.round(final_porosity, 2),
        'Mean_Grain_Size_um': np.round(mean_grain_size, 3),
        'Grain_Size_Std_um': np.round(grain_size_std, 3),
        'Grain_Size_D10_um': np.round(d10_grain, 3),
        'Grain_Size_D90_um': np.round(d90_grain, 3),
        'Mean_Pore_Size_um': np.round(mean_pore_size, 3),
        'Pore_Size_Std_um': np.round(pore_size_std, 3),
        'Pore_Connectivity_percent': np.round(pore_connectivity, 2),
        'GB_Thickness_nm': np.round(gb_thickness, 2),
        'GB_Energy_J_per_m2': np.round(gb_energy, 3),
        'GB_Phase_Coverage_percent': np.round(gb_phase_coverage, 2),
        'Coordination_Number': np.round(coordination_number, 1),
    }
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['Thermal_Load_C_hours'] = df['Hold_Temperature_C'] * df['Hold_Time_hours'] / 1000
    df['Densification_Percent'] = np.round((df['Final_Relative_Density'] - df['Initial_Relative_Density']) * 100, 2)
    df['Grain_Growth_Factor'] = np.round(df['Mean_Grain_Size_um'] / 0.5, 2)
    
    return df


def generate_summary_statistics(df):
    """Generate summary statistics for the dataset."""
    print("\n" + "="*80)
    print("DATASET SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    
    print("\n--- SINTERING PARAMETERS (INPUTS) ---")
    input_cols = ['Hold_Temperature_C', 'Hold_Time_hours', 'Applied_Pressure_MPa', 
                  'Initial_Relative_Density']
    print(df[input_cols].describe())
    
    print("\n--- MICROSTRUCTURE OUTPUTS ---")
    output_cols = ['Final_Relative_Density', 'Final_Porosity_percent', 
                   'Mean_Grain_Size_um', 'Mean_Pore_Size_um']
    print(df[output_cols].describe())
    
    print("\n--- ATMOSPHERE DISTRIBUTION ---")
    print(df['Atmosphere'].value_counts())
    
    print("\n--- CORRELATIONS (Key Relationships) ---")
    # Calculate some key correlations
    corr_pairs = [
        ('Hold_Temperature_C', 'Mean_Grain_Size_um'),
        ('Hold_Temperature_C', 'Final_Relative_Density'),
        ('Applied_Pressure_MPa', 'Final_Relative_Density'),
        ('Hold_Time_hours', 'Mean_Grain_Size_um'),
        ('Initial_Relative_Density', 'Final_Relative_Density'),
    ]
    
    for col1, col2 in corr_pairs:
        corr = df[col1].corr(df[col2])
        print(f"{col1} vs {col2}: {corr:.3f}")


if __name__ == "__main__":
    print("="*80)
    print("SINTERING PROCESS & MICROSTRUCTURE DATASET GENERATOR")
    print("="*80)
    
    # Generate dataset
    df = generate_sintering_dataset(n_samples=200)
    
    # Save to CSV
    output_file = 'sintering_process_microstructure_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved to: {output_file}")
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Save a sample of the data for quick viewing
    print("\n--- FIRST 5 SAMPLES ---")
    print(df.head())
    
    print("\n" + "="*80)
    print("Dataset generation complete!")
    print("="*80)