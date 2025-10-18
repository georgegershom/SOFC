#!/usr/bin/env python3
"""
Thermal Properties Dataset Generator for Thermo-Mechanical Modeling
==================================================================

Generates temperature-dependent thermal properties for rubberized concrete mixes
with physically consistent degradation functions and statistical bounds.

Mix Types:
- C: Control concrete (0% rubber)
- R5S: 5% small rubber particles
- R10S: 10% small rubber particles  
- R15S: 15% small rubber particles
- R20S: 20% small rubber particles
- R10L: 10% large rubber particles

Temperature Range: 20°C to 800°C
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Constants and base parameters
TEMPERATURE_RANGE = np.arange(20, 801, 10)  # 20°C to 800°C in 10°C increments
MIX_TYPES = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']

# Base properties at 20°C for control concrete
BASE_PROPERTIES = {
    'thermal_conductivity': 1.95,  # W/m·K
    'specific_heat': 880,          # J/kg·K
    'density': 2400,               # kg/m³
    'thermal_diffusivity': None    # Will be calculated
}

def rubber_modification_factor(rubber_content, particle_size='small'):
    """
    Calculate property modification factor based on rubber content and size.
    
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
    size_factor = 0.85 if particle_size == 'large' else 1.0
    
    # Property-specific modification factors
    factors = {
        'thermal_conductivity': 1.0 - (rubber_content / 100) * 0.45 * size_factor,
        'specific_heat': 1.0 + (rubber_content / 100) * 0.25 * size_factor,
        'density': 1.0 - (rubber_content / 100) * 0.35 * size_factor
    }
    
    return factors

def temperature_degradation_function(T, property_type, mix_type):
    """
    Temperature-dependent degradation functions for thermal properties.
    
    Parameters:
    -----------
    T : array-like
        Temperature in Celsius
    property_type : str
        Type of property ('conductivity', 'specific_heat', 'density')
    mix_type : str
        Mix type identifier
    
    Returns:
    --------
    array : Temperature modification factor
    """
    T_norm = (T - 20) / 780  # Normalize temperature range
    
    if property_type == 'thermal_conductivity':
        # Conductivity decreases with temperature due to phonon scattering
        # and microcracking at high temperatures
        if 'R' in mix_type:
            # Rubber mixes show more pronounced degradation
            rubber_content = float(mix_type[1:3]) if mix_type[1:3].isdigit() else 5
            degradation_factor = 1.2 + rubber_content * 0.02
        else:
            degradation_factor = 1.0
        
        return np.exp(-0.8 * T_norm * degradation_factor) * (1 - 0.3 * T_norm**2)
    
    elif property_type == 'specific_heat':
        # Specific heat increases with temperature due to lattice vibrations
        # Rubber content affects the rate of increase
        if 'R' in mix_type:
            rubber_content = float(mix_type[1:3]) if mix_type[1:3].isdigit() else 5
            enhancement_factor = 1.0 + rubber_content * 0.01
        else:
            enhancement_factor = 1.0
        
        return 1.0 + 0.4 * T_norm * enhancement_factor + 0.1 * T_norm**2
    
    elif property_type == 'density':
        # Density decreases slightly with temperature due to thermal expansion
        # and potential mass loss at very high temperatures
        thermal_expansion = 1.0 + 1.2e-5 * (T - 20)  # Linear expansion
        volume_expansion = thermal_expansion**3
        
        # Mass loss at high temperatures (>600°C)
        mass_loss_factor = 1.0
        if T > 600:
            mass_loss = 0.02 * ((T - 600) / 200)**2  # Up to 2% mass loss
            mass_loss_factor = 1.0 - mass_loss
        
        return mass_loss_factor / volume_expansion

def generate_thermal_conductivity_data():
    """Generate thermal conductivity data for all mix types."""
    
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
        mod_factors = rubber_modification_factor(rubber_content, particle_size)
        
        # Base conductivity at 20°C
        base_conductivity = BASE_PROPERTIES['thermal_conductivity'] * mod_factors['thermal_conductivity']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature degradation
            temp_factor = temperature_degradation_function(temp, 'thermal_conductivity', mix_id)
            conductivity_mean = base_conductivity * temp_factor
            
            # Add statistical variation (±10% standard deviation)
            conductivity_std = conductivity_mean * 0.10
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Thermal',
                'Thermal_Conductivity_Mean_W_m_K': round(conductivity_mean, 4),
                'Thermal_Conductivity_Std_W_m_K': round(conductivity_std, 4),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_specific_heat_data():
    """Generate specific heat data for all mix types."""
    
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
        mod_factors = rubber_modification_factor(rubber_content, particle_size)
        
        # Base specific heat at 20°C
        base_specific_heat = BASE_PROPERTIES['specific_heat'] * mod_factors['specific_heat']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature enhancement
            temp_factor = temperature_degradation_function(temp, 'specific_heat', mix_id)
            specific_heat_mean = base_specific_heat * temp_factor
            
            # Add statistical variation (±8% standard deviation)
            specific_heat_std = specific_heat_mean * 0.08
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Thermal',
                'Specific_Heat_Mean_J_kg_K': round(specific_heat_mean, 2),
                'Specific_Heat_Std_J_kg_K': round(specific_heat_std, 2),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_thermal_diffusivity_data():
    """Generate thermal diffusivity data (calculated from conductivity, specific heat, density)."""
    
    # Load conductivity and specific heat data
    conductivity_df = generate_thermal_conductivity_data()
    specific_heat_df = generate_specific_heat_data()
    
    data = []
    
    for mix_id in MIX_TYPES:
        # Extract rubber content and particle size
        if mix_id == 'C':
            rubber_content = 0
            particle_size = 'small'
        else:
            rubber_content = float(mix_id[1:3]) if mix_id[1:3].isdigit() else 10
            particle_size = 'large' if 'L' in mix_id else 'small'
        
        # Get modification factors for density
        mod_factors = rubber_modification_factor(rubber_content, particle_size)
        base_density = BASE_PROPERTIES['density'] * mod_factors['density']
        
        for temp in TEMPERATURE_RANGE:
            # Get corresponding conductivity and specific heat
            k_data = conductivity_df[(conductivity_df['Mix_ID'] == mix_id) & 
                                   (conductivity_df['Temperature_C'] == temp)].iloc[0]
            cp_data = specific_heat_df[(specific_heat_df['Mix_ID'] == mix_id) & 
                                     (specific_heat_df['Temperature_C'] == temp)].iloc[0]
            
            # Calculate density at temperature
            density_factor = temperature_degradation_function(temp, 'density', mix_id)
            density = base_density * density_factor
            
            # Calculate thermal diffusivity: α = k / (ρ * cp)
            thermal_diffusivity_mean = (k_data['Thermal_Conductivity_Mean_W_m_K'] / 
                                      (density * cp_data['Specific_Heat_Mean_J_kg_K'])) * 1e6  # Convert to mm²/s
            
            # Propagate uncertainty
            k_rel_std = k_data['Thermal_Conductivity_Std_W_m_K'] / k_data['Thermal_Conductivity_Mean_W_m_K']
            cp_rel_std = cp_data['Specific_Heat_Std_J_kg_K'] / cp_data['Specific_Heat_Mean_J_kg_K']
            density_rel_std = 0.05  # Assume 5% uncertainty in density
            
            # Combined relative standard deviation
            combined_rel_std = np.sqrt(k_rel_std**2 + cp_rel_std**2 + density_rel_std**2)
            thermal_diffusivity_std = thermal_diffusivity_mean * combined_rel_std
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Thermal',
                'Thermal_Diffusivity_Mean_mm2_s': round(thermal_diffusivity_mean, 6),
                'Thermal_Diffusivity_Std_mm2_s': round(thermal_diffusivity_std, 6),
                'Density_kg_m3': round(density, 1),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_thermal_expansion_data():
    """Generate thermal expansion coefficient data."""
    
    data = []
    
    for mix_id in MIX_TYPES:
        # Extract rubber content and particle size
        if mix_id == 'C':
            rubber_content = 0
            particle_size = 'small'
        else:
            rubber_content = float(mix_id[1:3]) if mix_id[1:3].isdigit() else 10
            particle_size = 'large' if 'L' in mix_id else 'small'
        
        # Base thermal expansion coefficient (concrete: ~12e-6 /K)
        base_alpha = 12.0e-6  # /K
        
        # Rubber modification (rubber has higher expansion coefficient)
        rubber_enhancement = 1.0 + (rubber_content / 100) * 0.8
        if particle_size == 'large':
            rubber_enhancement *= 1.1  # Large particles have slightly higher effect
        
        for temp in TEMPERATURE_RANGE:
            # Temperature-dependent expansion coefficient
            # Increases slightly with temperature
            temp_factor = 1.0 + 0.3 * ((temp - 20) / 780)
            
            alpha_mean = base_alpha * rubber_enhancement * temp_factor
            alpha_std = alpha_mean * 0.12  # ±12% standard deviation
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Thermal',
                'Thermal_Expansion_Coeff_Mean_per_K': f"{alpha_mean:.2e}",
                'Thermal_Expansion_Coeff_Std_per_K': f"{alpha_std:.2e}",
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def main():
    """Generate all thermal properties datasets."""
    
    print("Generating Thermal Properties Dataset...")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate datasets
    print("1. Generating thermal conductivity data...")
    conductivity_df = generate_thermal_conductivity_data()
    
    print("2. Generating specific heat data...")
    specific_heat_df = generate_specific_heat_data()
    
    print("3. Generating thermal diffusivity data...")
    diffusivity_df = generate_thermal_diffusivity_data()
    
    print("4. Generating thermal expansion data...")
    expansion_df = generate_thermal_expansion_data()
    
    # Create output directory
    output_dir = "/workspace/thermo_mechanical_dataset/thermal_properties"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    print("\nSaving datasets...")
    conductivity_df.to_csv(f"{output_dir}/thermal_conductivity.csv", index=False)
    specific_heat_df.to_csv(f"{output_dir}/specific_heat.csv", index=False)
    diffusivity_df.to_csv(f"{output_dir}/thermal_diffusivity.csv", index=False)
    expansion_df.to_csv(f"{output_dir}/thermal_expansion.csv", index=False)
    
    # Generate summary statistics
    print("\nDataset Summary:")
    print(f"- Thermal Conductivity: {len(conductivity_df)} data points")
    print(f"- Specific Heat: {len(specific_heat_df)} data points")
    print(f"- Thermal Diffusivity: {len(diffusivity_df)} data points")
    print(f"- Thermal Expansion: {len(expansion_df)} data points")
    print(f"- Temperature range: {TEMPERATURE_RANGE[0]}°C to {TEMPERATURE_RANGE[-1]}°C")
    print(f"- Mix types: {', '.join(MIX_TYPES)}")
    
    # Calculate calibration/validation split
    total_points = len(conductivity_df)
    cal_points = len(conductivity_df[conductivity_df['Data_Type'] == 'Calibration'])
    val_points = len(conductivity_df[conductivity_df['Data_Type'] == 'Validation'])
    
    print(f"\nData Split:")
    print(f"- Calibration: {cal_points} points ({cal_points/total_points*100:.1f}%)")
    print(f"- Validation: {val_points} points ({val_points/total_points*100:.1f}%)")
    
    print("\nThermal properties dataset generation completed successfully!")

if __name__ == "__main__":
    main()