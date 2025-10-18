#!/usr/bin/env python3
"""
Transport Properties Dataset Generator for Thermo-Mechanical Modeling
====================================================================

Generates temperature-dependent transport properties for rubberized concrete mixes
with physically consistent poro-mechanical coupling and statistical bounds.

Properties Generated:
- Permeability (intrinsic and relative)
- Porosity (total and effective)
- Moisture Transport (diffusion coefficient, sorption isotherms)
- Gas Transport (oxygen and CO2 diffusion)

Temperature Range: 20°C to 800°C
"""

import numpy as np
import pandas as pd
import os

# Constants and base parameters
TEMPERATURE_RANGE = np.arange(20, 801, 10)  # 20°C to 800°C in 10°C increments
MIX_TYPES = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']

# Base transport properties at 20°C for control concrete
BASE_PROPERTIES = {
    'intrinsic_permeability': 1.2e-18,    # m²
    'total_porosity': 0.12,               # dimensionless (12%)
    'effective_porosity': 0.08,           # dimensionless (8%)
    'moisture_diffusivity': 2.5e-12,      # m²/s
    'oxygen_diffusivity': 1.8e-6,         # m²/s
    'co2_diffusivity': 1.2e-6,            # m²/s
    'sorption_capacity': 0.045            # kg/kg (4.5% by mass)
}

def rubber_transport_modification(rubber_content, particle_size='small'):
    """
    Calculate transport property modification factors based on rubber content and size.
    
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
    # Size effect factors - larger particles create more interfacial transition zones
    size_factor = 1.2 if particle_size == 'large' else 1.0
    
    # Property-specific modification factors
    factors = {
        'intrinsic_permeability': 1.0 + (rubber_content / 100) * 2.8 * size_factor,  # Rubber increases permeability
        'total_porosity': 1.0 + (rubber_content / 100) * 0.45 * size_factor,
        'effective_porosity': 1.0 + (rubber_content / 100) * 0.65 * size_factor,     # More connected pores
        'moisture_diffusivity': 1.0 + (rubber_content / 100) * 1.8 * size_factor,
        'oxygen_diffusivity': 1.0 + (rubber_content / 100) * 1.5 * size_factor,
        'co2_diffusivity': 1.0 + (rubber_content / 100) * 1.6 * size_factor,
        'sorption_capacity': 1.0 - (rubber_content / 100) * 0.3 * size_factor       # Rubber is hydrophobic
    }
    
    return factors

def temperature_transport_modification(T, property_type, mix_type):
    """
    Temperature-dependent modification functions for transport properties.
    
    Parameters:
    -----------
    T : array-like
        Temperature in Celsius
    property_type : str
        Type of property
    mix_type : str
        Mix type identifier
    
    Returns:
    --------
    array : Temperature modification factor
    """
    T_norm = (T - 20) / 780  # Normalize temperature range
    T_kelvin = T + 273.15    # Convert to Kelvin for Arrhenius relationships
    
    # Extract rubber content for rubber-specific effects
    if 'R' in mix_type:
        rubber_content = float(mix_type[1:3]) if mix_type[1:3].isdigit() else 5
    else:
        rubber_content = 0
    
    if property_type == 'intrinsic_permeability':
        # Permeability increases with temperature due to:
        # 1. Thermal expansion creating microcracks
        # 2. Dehydration increasing pore connectivity
        # 3. Rubber degradation at high temperatures
        
        thermal_expansion_effect = 1.0 + 2.5 * T_norm  # Microcracking
        dehydration_effect = 1.0 if T < 100 else 1.0 + 0.8 * ((T - 100) / 700)**0.5
        
        # Rubber degradation accelerates permeability increase
        rubber_degradation = 1.0 + rubber_content * 0.02 * T_norm**2
        
        return thermal_expansion_effect * dehydration_effect * rubber_degradation
    
    elif property_type == 'total_porosity':
        # Porosity increases with temperature due to:
        # 1. Dehydration (loss of bound water)
        # 2. Thermal decomposition
        # 3. Rubber degradation
        
        dehydration_porosity = 0.0 if T < 105 else 0.03 * ((T - 105) / 695)**0.3
        decomposition_porosity = 0.0 if T < 400 else 0.02 * ((T - 400) / 400)**0.5
        rubber_degradation_porosity = rubber_content / 100 * 0.15 * T_norm**1.5
        
        additional_porosity = dehydration_porosity + decomposition_porosity + rubber_degradation_porosity
        return 1.0 + additional_porosity / 0.12  # Normalize by base porosity
    
    elif property_type == 'effective_porosity':
        # Effective porosity increases more than total due to pore connectivity
        total_factor = temperature_transport_modification(T, 'total_porosity', mix_type)
        connectivity_enhancement = 1.0 + 0.5 * T_norm  # Better connectivity at high T
        return total_factor * connectivity_enhancement
    
    elif property_type in ['moisture_diffusivity', 'oxygen_diffusivity', 'co2_diffusivity']:
        # Gas diffusion follows Arrhenius relationship with temperature
        # D = D0 * exp(-Ea/RT) * (T/T0)^n
        
        # Activation energies (J/mol)
        activation_energies = {
            'moisture_diffusivity': 25000,
            'oxygen_diffusivity': 15000,
            'co2_diffusivity': 18000
        }
        
        Ea = activation_energies[property_type]
        R = 8.314  # Gas constant
        T0 = 293.15  # Reference temperature (20°C)
        
        # Arrhenius factor
        arrhenius_factor = np.exp(-Ea/R * (1/T_kelvin - 1/T0))
        
        # Temperature dependence factor
        temp_factor = (T_kelvin / T0)**1.75
        
        # Porosity enhancement factor
        porosity_factor = temperature_transport_modification(T, 'effective_porosity', mix_type)
        
        return arrhenius_factor * temp_factor * porosity_factor**0.5
    
    elif property_type == 'sorption_capacity':
        # Sorption capacity decreases with temperature due to:
        # 1. Dehydration of binding sites
        # 2. Reduced physical adsorption
        # 3. Rubber degradation reducing hydrophilic sites
        
        T0 = 293.15  # Reference temperature (20°C)
        dehydration_factor = np.exp(-0.8 * T_norm)
        physical_adsorption_factor = (T0 / T_kelvin)**0.5
        rubber_effect = 1.0 - rubber_content * 0.01 * T_norm
        
        return dehydration_factor * physical_adsorption_factor * rubber_effect

def generate_permeability_data():
    """Generate permeability data for all mix types."""
    
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
        mod_factors = rubber_transport_modification(rubber_content, particle_size)
        
        # Base permeability at 20°C
        base_permeability = BASE_PROPERTIES['intrinsic_permeability'] * mod_factors['intrinsic_permeability']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature modification
            temp_factor = temperature_transport_modification(temp, 'intrinsic_permeability', mix_id)
            permeability_mean = base_permeability * temp_factor
            
            # Add statistical variation (±25% standard deviation - high for permeability)
            permeability_std = permeability_mean * 0.25
            
            # Calculate relative permeability (normalized by water permeability)
            water_permeability = 1e-18  # Reference water permeability
            relative_permeability = permeability_mean / water_permeability
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Transport',
                'Intrinsic_Permeability_Mean_m2': f"{permeability_mean:.2e}",
                'Intrinsic_Permeability_Std_m2': f"{permeability_std:.2e}",
                'Relative_Permeability': round(relative_permeability, 2),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_porosity_data():
    """Generate porosity data for all mix types."""
    
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
        mod_factors = rubber_transport_modification(rubber_content, particle_size)
        
        # Base porosities at 20°C
        base_total_porosity = BASE_PROPERTIES['total_porosity'] * mod_factors['total_porosity']
        base_effective_porosity = BASE_PROPERTIES['effective_porosity'] * mod_factors['effective_porosity']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature modifications
            total_temp_factor = temperature_transport_modification(temp, 'total_porosity', mix_id)
            effective_temp_factor = temperature_transport_modification(temp, 'effective_porosity', mix_id)
            
            total_porosity_mean = base_total_porosity * total_temp_factor
            effective_porosity_mean = base_effective_porosity * effective_temp_factor
            
            # Ensure physical bounds (0 < porosity < 1, effective < total)
            total_porosity_mean = max(0.05, min(0.45, total_porosity_mean))
            effective_porosity_mean = max(0.02, min(total_porosity_mean * 0.9, effective_porosity_mean))
            
            # Add statistical variation
            total_porosity_std = total_porosity_mean * 0.15
            effective_porosity_std = effective_porosity_mean * 0.18
            
            # Calculate connectivity factor
            connectivity_factor = effective_porosity_mean / total_porosity_mean
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Transport',
                'Total_Porosity_Mean': round(total_porosity_mean, 4),
                'Total_Porosity_Std': round(total_porosity_std, 4),
                'Effective_Porosity_Mean': round(effective_porosity_mean, 4),
                'Effective_Porosity_Std': round(effective_porosity_std, 4),
                'Connectivity_Factor': round(connectivity_factor, 3),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_moisture_transport_data():
    """Generate moisture transport data for all mix types."""
    
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
        mod_factors = rubber_transport_modification(rubber_content, particle_size)
        
        # Base properties at 20°C
        base_diffusivity = BASE_PROPERTIES['moisture_diffusivity'] * mod_factors['moisture_diffusivity']
        base_sorption = BASE_PROPERTIES['sorption_capacity'] * mod_factors['sorption_capacity']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature modifications
            diffusivity_temp_factor = temperature_transport_modification(temp, 'moisture_diffusivity', mix_id)
            sorption_temp_factor = temperature_transport_modification(temp, 'sorption_capacity', mix_id)
            
            diffusivity_mean = base_diffusivity * diffusivity_temp_factor
            sorption_mean = base_sorption * sorption_temp_factor
            
            # Add statistical variation
            diffusivity_std = diffusivity_mean * 0.22
            sorption_std = sorption_mean * 0.18
            
            # Calculate moisture permeability (combined transport parameter)
            moisture_permeability = diffusivity_mean * sorption_mean * 1000  # kg/m/s/Pa
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Transport',
                'Moisture_Diffusivity_Mean_m2_s': f"{diffusivity_mean:.2e}",
                'Moisture_Diffusivity_Std_m2_s': f"{diffusivity_std:.2e}",
                'Sorption_Capacity_Mean_kg_kg': round(sorption_mean, 5),
                'Sorption_Capacity_Std_kg_kg': round(sorption_std, 5),
                'Moisture_Permeability_kg_m_s_Pa': f"{moisture_permeability:.2e}",
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_gas_transport_data():
    """Generate gas transport data for all mix types."""
    
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
        mod_factors = rubber_transport_modification(rubber_content, particle_size)
        
        # Base diffusivities at 20°C
        base_o2_diffusivity = BASE_PROPERTIES['oxygen_diffusivity'] * mod_factors['oxygen_diffusivity']
        base_co2_diffusivity = BASE_PROPERTIES['co2_diffusivity'] * mod_factors['co2_diffusivity']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature modifications
            o2_temp_factor = temperature_transport_modification(temp, 'oxygen_diffusivity', mix_id)
            co2_temp_factor = temperature_transport_modification(temp, 'co2_diffusivity', mix_id)
            
            o2_diffusivity_mean = base_o2_diffusivity * o2_temp_factor
            co2_diffusivity_mean = base_co2_diffusivity * co2_temp_factor
            
            # Add statistical variation
            o2_diffusivity_std = o2_diffusivity_mean * 0.20
            co2_diffusivity_std = co2_diffusivity_mean * 0.20
            
            # Calculate gas permeability coefficients
            o2_permeability = o2_diffusivity_mean * 1.4e-3  # Solubility factor for O2
            co2_permeability = co2_diffusivity_mean * 3.3e-2  # Solubility factor for CO2
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Transport',
                'O2_Diffusivity_Mean_m2_s': f"{o2_diffusivity_mean:.2e}",
                'O2_Diffusivity_Std_m2_s': f"{o2_diffusivity_std:.2e}",
                'CO2_Diffusivity_Mean_m2_s': f"{co2_diffusivity_mean:.2e}",
                'CO2_Diffusivity_Std_m2_s': f"{co2_diffusivity_std:.2e}",
                'O2_Permeability_m2_s_Pa': f"{o2_permeability:.2e}",
                'CO2_Permeability_m2_s_Pa': f"{co2_permeability:.2e}",
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def main():
    """Generate all transport properties datasets."""
    
    print("Generating Transport Properties Dataset...")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate datasets
    print("1. Generating permeability data...")
    permeability_df = generate_permeability_data()
    
    print("2. Generating porosity data...")
    porosity_df = generate_porosity_data()
    
    print("3. Generating moisture transport data...")
    moisture_df = generate_moisture_transport_data()
    
    print("4. Generating gas transport data...")
    gas_df = generate_gas_transport_data()
    
    # Create output directory
    output_dir = "/workspace/thermo_mechanical_dataset/transport_properties"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    print("\nSaving datasets...")
    permeability_df.to_csv(f"{output_dir}/permeability.csv", index=False)
    porosity_df.to_csv(f"{output_dir}/porosity.csv", index=False)
    moisture_df.to_csv(f"{output_dir}/moisture_transport.csv", index=False)
    gas_df.to_csv(f"{output_dir}/gas_transport.csv", index=False)
    
    # Generate summary statistics
    print("\nDataset Summary:")
    print(f"- Permeability: {len(permeability_df)} data points")
    print(f"- Porosity: {len(porosity_df)} data points")
    print(f"- Moisture Transport: {len(moisture_df)} data points")
    print(f"- Gas Transport: {len(gas_df)} data points")
    print(f"- Temperature range: {TEMPERATURE_RANGE[0]}°C to {TEMPERATURE_RANGE[-1]}°C")
    print(f"- Mix types: {', '.join(MIX_TYPES)}")
    
    # Calculate calibration/validation split
    total_points = len(permeability_df)
    cal_points = len(permeability_df[permeability_df['Data_Type'] == 'Calibration'])
    val_points = len(permeability_df[permeability_df['Data_Type'] == 'Validation'])
    
    print(f"\nData Split:")
    print(f"- Calibration: {cal_points} points ({cal_points/total_points*100:.1f}%)")
    print(f"- Validation: {val_points} points ({val_points/total_points*100:.1f}%)")
    
    # Display sample property ranges
    print(f"\nProperty Ranges at 20°C:")
    for mix_id in MIX_TYPES:
        porosity_data = porosity_df[(porosity_df['Mix_ID'] == mix_id) & 
                                   (porosity_df['Temperature_C'] == 20)]
        if not porosity_data.empty:
            total_porosity = porosity_data.iloc[0]['Total_Porosity_Mean']
            print(f"- {mix_id}: {total_porosity:.3f} total porosity")
    
    print("\nTransport properties dataset generation completed successfully!")

if __name__ == "__main__":
    main()