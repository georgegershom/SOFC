#!/usr/bin/env python3
"""
Deformation Properties Dataset Generator for Thermo-Mechanical Modeling
======================================================================

Generates temperature and time-dependent deformation properties for rubberized concrete mixes
with multi-physics coupling and statistical bounds.

Properties Generated:
- Creep Parameters (primary, secondary, tertiary creep)
- Shrinkage Parameters (autogenous, drying, thermal)
- Thermal Strain (instantaneous and time-dependent)
- Damage Evolution (temperature-dependent degradation)

Temperature Range: 20°C to 800°C
Time Range: 1 hour to 50 years (for time-dependent properties)
"""

import numpy as np
import pandas as pd
import os

# Constants and base parameters
TEMPERATURE_RANGE = np.arange(20, 801, 10)  # 20°C to 800°C in 10°C increments
TIME_RANGE = np.logspace(0, 5, 50)  # 1 hour to ~11 years (logarithmic scale)
MIX_TYPES = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']

# Base deformation properties at 20°C for control concrete
BASE_PROPERTIES = {
    # Creep parameters (Norton-Bailey law: ε̇ = A * σ^n * t^m * exp(-Q/RT))
    'creep_coefficient_A': 2.5e-12,     # 1/(MPa^n * s^(1-m))
    'creep_stress_exponent_n': 1.2,     # dimensionless
    'creep_time_exponent_m': 0.18,      # dimensionless
    'creep_activation_energy': 45000,   # J/mol
    
    # Shrinkage parameters
    'ultimate_autogenous_shrinkage': 120e-6,  # strain (120 microstrain)
    'ultimate_drying_shrinkage': 450e-6,      # strain (450 microstrain)
    'shrinkage_time_constant': 28,            # days
    'shrinkage_humidity_factor': 0.85,        # dimensionless
    
    # Thermal strain parameters
    'thermal_expansion_coeff': 12e-6,          # /K
    'thermal_expansion_nonlinearity': 0.15,   # dimensionless
    
    # Damage parameters
    'damage_threshold_temp': 300,             # °C
    'damage_evolution_rate': 0.002,           # 1/°C
    'max_damage_level': 0.8                   # dimensionless (0-1)
}

def rubber_deformation_modification(rubber_content, particle_size='small'):
    """
    Calculate deformation property modification factors based on rubber content and size.
    
    Parameters:
    -----------
    rubber_content : float
        Rubber content as percentage (0-20)
    particle_size : str
        'small' or 'large' rubber particles
    
    Returns:
    --------
    dict : Modification factors for different properties
    """
    # Size effect factors
    size_factor = 1.1 if particle_size == 'large' else 1.0
    
    # Property-specific modification factors
    factors = {
        'creep_coefficient_A': 1.0 + (rubber_content / 100) * 0.8 * size_factor,  # Rubber increases creep
        'creep_stress_exponent_n': 1.0 + (rubber_content / 100) * 0.15 * size_factor,
        'creep_time_exponent_m': 1.0 - (rubber_content / 100) * 0.1 * size_factor,  # Less time dependence
        'creep_activation_energy': 1.0 - (rubber_content / 100) * 0.12 * size_factor,  # Lower activation energy
        
        'ultimate_autogenous_shrinkage': 1.0 - (rubber_content / 100) * 0.4 * size_factor,  # Rubber reduces shrinkage
        'ultimate_drying_shrinkage': 1.0 - (rubber_content / 100) * 0.35 * size_factor,
        'shrinkage_time_constant': 1.0 + (rubber_content / 100) * 0.25 * size_factor,  # Slower shrinkage
        'shrinkage_humidity_factor': 1.0 - (rubber_content / 100) * 0.05 * size_factor,
        
        'thermal_expansion_coeff': 1.0 + (rubber_content / 100) * 0.6 * size_factor,  # Higher expansion
        'thermal_expansion_nonlinearity': 1.0 + (rubber_content / 100) * 0.3 * size_factor,
        
        'damage_threshold_temp': 1.0 - (rubber_content / 100) * 0.08 * size_factor,  # Lower threshold
        'damage_evolution_rate': 1.0 + (rubber_content / 100) * 0.2 * size_factor,   # Faster damage
        'max_damage_level': 1.0 + (rubber_content / 100) * 0.1 * size_factor         # Higher max damage
    }
    
    return factors

def generate_creep_parameters_data():
    """Generate creep parameters data for all mix types."""
    
    data = []
    
    for mix_id in MIX_TYPES:
        # Extract rubber content and particle size
        if mix_id == 'C':
            rubber_content = 0
            particle_size = 'small'
        else:
            rubber_content = float(mix_id[1:3]) if mix_id[1:3].isdigit() else 10
            particle_size = 'large' if 'L' in mix_id else 'small'
        
        # Get modification factors
        mod_factors = rubber_deformation_modification(rubber_content, particle_size)
        
        # Base parameters at 20°C
        base_A = BASE_PROPERTIES['creep_coefficient_A'] * mod_factors['creep_coefficient_A']
        base_n = BASE_PROPERTIES['creep_stress_exponent_n'] * mod_factors['creep_stress_exponent_n']
        base_m = BASE_PROPERTIES['creep_time_exponent_m'] * mod_factors['creep_time_exponent_m']
        base_Q = BASE_PROPERTIES['creep_activation_energy'] * mod_factors['creep_activation_energy']
        
        for temp in TEMPERATURE_RANGE:
            # Temperature-dependent modifications
            T_kelvin = temp + 273.15
            T_ref = 293.15  # 20°C reference
            
            # Arrhenius temperature dependence for creep coefficient
            A_temp = base_A * np.exp(-base_Q / 8.314 * (1/T_kelvin - 1/T_ref))
            
            # Stress exponent slightly increases with temperature
            n_temp = base_n * (1.0 + 0.1 * (temp - 20) / 780)
            
            # Time exponent decreases slightly with temperature
            m_temp = base_m * (1.0 - 0.05 * (temp - 20) / 780)
            
            # Add statistical variation
            A_std = A_temp * 0.30  # High variability for creep
            n_std = n_temp * 0.15
            m_std = m_temp * 0.12
            Q_std = base_Q * 0.10
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Deformation',
                'Creep_Coefficient_A_Mean': f"{A_temp:.2e}",
                'Creep_Coefficient_A_Std': f"{A_std:.2e}",
                'Stress_Exponent_n_Mean': round(n_temp, 3),
                'Stress_Exponent_n_Std': round(n_std, 3),
                'Time_Exponent_m_Mean': round(m_temp, 4),
                'Time_Exponent_m_Std': round(m_std, 4),
                'Activation_Energy_Mean_J_mol': round(base_Q, 0),
                'Activation_Energy_Std_J_mol': round(Q_std, 0),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_shrinkage_parameters_data():
    """Generate shrinkage parameters data for all mix types."""
    
    data = []
    
    for mix_id in MIX_TYPES:
        # Extract rubber content and particle size
        if mix_id == 'C':
            rubber_content = 0
            particle_size = 'small'
        else:
            rubber_content = float(mix_id[1:3]) if mix_id[1:3].isdigit() else 10
            particle_size = 'large' if 'L' in mix_id else 'small'
        
        # Get modification factors
        mod_factors = rubber_deformation_modification(rubber_content, particle_size)
        
        # Base parameters at 20°C
        base_autogenous = BASE_PROPERTIES['ultimate_autogenous_shrinkage'] * mod_factors['ultimate_autogenous_shrinkage']
        base_drying = BASE_PROPERTIES['ultimate_drying_shrinkage'] * mod_factors['ultimate_drying_shrinkage']
        base_time_const = BASE_PROPERTIES['shrinkage_time_constant'] * mod_factors['shrinkage_time_constant']
        base_humidity = BASE_PROPERTIES['shrinkage_humidity_factor'] * mod_factors['shrinkage_humidity_factor']
        
        for temp in TEMPERATURE_RANGE:
            # Temperature effects on shrinkage
            temp_factor = 1.0 + 0.5 * (temp - 20) / 780  # Increases with temperature
            
            # High temperature effects (>100°C)
            if temp > 100:
                additional_shrinkage = 200e-6 * ((temp - 100) / 700)**0.5  # Additional thermal shrinkage
            else:
                additional_shrinkage = 0
            
            autogenous_mean = base_autogenous * temp_factor + additional_shrinkage * 0.3
            drying_mean = base_drying * temp_factor + additional_shrinkage
            
            # Time constant decreases with temperature (faster shrinkage)
            time_const_mean = base_time_const * np.exp(-0.3 * (temp - 20) / 780)
            
            # Humidity factor decreases at high temperatures
            humidity_mean = base_humidity * np.exp(-0.2 * (temp - 20) / 780)
            
            # Add statistical variation
            autogenous_std = autogenous_mean * 0.25
            drying_std = drying_mean * 0.20
            time_const_std = time_const_mean * 0.18
            humidity_std = humidity_mean * 0.10
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Deformation',
                'Ultimate_Autogenous_Shrinkage_Mean': f"{autogenous_mean:.2e}",
                'Ultimate_Autogenous_Shrinkage_Std': f"{autogenous_std:.2e}",
                'Ultimate_Drying_Shrinkage_Mean': f"{drying_mean:.2e}",
                'Ultimate_Drying_Shrinkage_Std': f"{drying_std:.2e}",
                'Shrinkage_Time_Constant_Mean_days': round(time_const_mean, 1),
                'Shrinkage_Time_Constant_Std_days': round(time_const_std, 1),
                'Humidity_Factor_Mean': round(humidity_mean, 3),
                'Humidity_Factor_Std': round(humidity_std, 3),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_thermal_strain_data():
    """Generate thermal strain data for all mix types."""
    
    data = []
    
    for mix_id in MIX_TYPES:
        # Extract rubber content and particle size
        if mix_id == 'C':
            rubber_content = 0
            particle_size = 'small'
        else:
            rubber_content = float(mix_id[1:3]) if mix_id[1:3].isdigit() else 10
            particle_size = 'large' if 'L' in mix_id else 'small'
        
        # Get modification factors
        mod_factors = rubber_deformation_modification(rubber_content, particle_size)
        
        # Base parameters
        base_alpha = BASE_PROPERTIES['thermal_expansion_coeff'] * mod_factors['thermal_expansion_coeff']
        base_nonlinearity = BASE_PROPERTIES['thermal_expansion_nonlinearity'] * mod_factors['thermal_expansion_nonlinearity']
        
        for temp in TEMPERATURE_RANGE:
            # Temperature-dependent thermal expansion coefficient
            # α(T) = α₀ * (1 + β * (T - T₀))
            alpha_temp = base_alpha * (1.0 + base_nonlinearity * (temp - 20) / 780)
            
            # Instantaneous thermal strain (from reference temperature)
            thermal_strain_instant = alpha_temp * (temp - 20)
            
            # Time-dependent thermal strain (additional strain due to microstructural changes)
            if temp > 200:
                time_dependent_factor = 0.15 * ((temp - 200) / 600)**0.5
                thermal_strain_time_dep = thermal_strain_instant * time_dependent_factor
            else:
                thermal_strain_time_dep = 0
            
            # Total thermal strain
            total_thermal_strain = thermal_strain_instant + thermal_strain_time_dep
            
            # Add statistical variation
            alpha_std = alpha_temp * 0.12
            instant_std = thermal_strain_instant * 0.10
            time_dep_std = thermal_strain_time_dep * 0.25 if thermal_strain_time_dep > 0 else 0
            total_std = total_thermal_strain * 0.12
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Deformation',
                'Thermal_Expansion_Coeff_Mean_per_K': f"{alpha_temp:.2e}",
                'Thermal_Expansion_Coeff_Std_per_K': f"{alpha_std:.2e}",
                'Instantaneous_Thermal_Strain_Mean': f"{thermal_strain_instant:.2e}",
                'Instantaneous_Thermal_Strain_Std': f"{instant_std:.2e}",
                'Time_Dependent_Thermal_Strain_Mean': f"{thermal_strain_time_dep:.2e}",
                'Time_Dependent_Thermal_Strain_Std': f"{time_dep_std:.2e}",
                'Total_Thermal_Strain_Mean': f"{total_thermal_strain:.2e}",
                'Total_Thermal_Strain_Std': f"{total_std:.2e}",
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_damage_evolution_data():
    """Generate damage evolution data for all mix types."""
    
    data = []
    
    for mix_id in MIX_TYPES:
        # Extract rubber content and particle size
        if mix_id == 'C':
            rubber_content = 0
            particle_size = 'small'
        else:
            rubber_content = float(mix_id[1:3]) if mix_id[1:3].isdigit() else 10
            particle_size = 'large' if 'L' in mix_id else 'small'
        
        # Get modification factors
        mod_factors = rubber_deformation_modification(rubber_content, particle_size)
        
        # Base parameters
        base_threshold = BASE_PROPERTIES['damage_threshold_temp'] * mod_factors['damage_threshold_temp']
        base_rate = BASE_PROPERTIES['damage_evolution_rate'] * mod_factors['damage_evolution_rate']
        base_max_damage = BASE_PROPERTIES['max_damage_level'] * mod_factors['max_damage_level']
        
        # Ensure physical bounds
        base_max_damage = min(0.95, base_max_damage)  # Maximum 95% damage
        
        for temp in TEMPERATURE_RANGE:
            # Damage evolution: D = D_max * (1 - exp(-k * (T - T_threshold)))
            if temp <= base_threshold:
                damage_mean = 0.0
            else:
                damage_mean = base_max_damage * (1.0 - np.exp(-base_rate * (temp - base_threshold)))
            
            # Damage rate (dD/dT)
            if temp <= base_threshold:
                damage_rate = 0.0
            else:
                damage_rate = base_max_damage * base_rate * np.exp(-base_rate * (temp - base_threshold))
            
            # Residual strength factor (1 - D)
            residual_strength_factor = 1.0 - damage_mean
            
            # Residual stiffness factor (typically degrades faster than strength)
            residual_stiffness_factor = (1.0 - damage_mean)**1.5
            
            # Add statistical variation
            damage_std = damage_mean * 0.20 if damage_mean > 0 else 0
            rate_std = damage_rate * 0.25 if damage_rate > 0 else 0
            strength_std = residual_strength_factor * 0.08
            stiffness_std = residual_stiffness_factor * 0.10
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Deformation',
                'Damage_Level_Mean': round(damage_mean, 4),
                'Damage_Level_Std': round(damage_std, 4),
                'Damage_Rate_Mean_per_C': f"{damage_rate:.2e}",
                'Damage_Rate_Std_per_C': f"{rate_std:.2e}",
                'Residual_Strength_Factor_Mean': round(residual_strength_factor, 4),
                'Residual_Strength_Factor_Std': round(strength_std, 4),
                'Residual_Stiffness_Factor_Mean': round(residual_stiffness_factor, 4),
                'Residual_Stiffness_Factor_Std': round(stiffness_std, 4),
                'Damage_Threshold_Temp_C': round(base_threshold, 1),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def main():
    """Generate all deformation properties datasets."""
    
    print("Generating Deformation Properties Dataset...")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate datasets
    print("1. Generating creep parameters data...")
    creep_df = generate_creep_parameters_data()
    
    print("2. Generating shrinkage parameters data...")
    shrinkage_df = generate_shrinkage_parameters_data()
    
    print("3. Generating thermal strain data...")
    thermal_strain_df = generate_thermal_strain_data()
    
    print("4. Generating damage evolution data...")
    damage_df = generate_damage_evolution_data()
    
    # Create output directory
    output_dir = "/workspace/thermo_mechanical_dataset/deformation_properties"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    print("\nSaving datasets...")
    creep_df.to_csv(f"{output_dir}/creep_parameters.csv", index=False)
    shrinkage_df.to_csv(f"{output_dir}/shrinkage_parameters.csv", index=False)
    thermal_strain_df.to_csv(f"{output_dir}/thermal_strain.csv", index=False)
    damage_df.to_csv(f"{output_dir}/damage_evolution.csv", index=False)
    
    # Generate summary statistics
    print("\nDataset Summary:")
    print(f"- Creep Parameters: {len(creep_df)} data points")
    print(f"- Shrinkage Parameters: {len(shrinkage_df)} data points")
    print(f"- Thermal Strain: {len(thermal_strain_df)} data points")
    print(f"- Damage Evolution: {len(damage_df)} data points")
    print(f"- Temperature range: {TEMPERATURE_RANGE[0]}°C to {TEMPERATURE_RANGE[-1]}°C")
    print(f"- Mix types: {', '.join(MIX_TYPES)}")
    
    # Calculate calibration/validation split
    total_points = len(creep_df)
    cal_points = len(creep_df[creep_df['Data_Type'] == 'Calibration'])
    val_points = len(creep_df[creep_df['Data_Type'] == 'Validation'])
    
    print(f"\nData Split:")
    print(f"- Calibration: {cal_points} points ({cal_points/total_points*100:.1f}%)")
    print(f"- Validation: {val_points} points ({val_points/total_points*100:.1f}%)")
    
    # Display sample property ranges
    print(f"\nProperty Ranges at 20°C:")
    for mix_id in MIX_TYPES:
        creep_data = creep_df[(creep_df['Mix_ID'] == mix_id) & 
                             (creep_df['Temperature_C'] == 20)]
        if not creep_data.empty:
            stress_exp = creep_data.iloc[0]['Stress_Exponent_n_Mean']
            print(f"- {mix_id}: {stress_exp:.2f} stress exponent")
    
    print("\nDeformation properties dataset generation completed successfully!")

if __name__ == "__main__":
    main()