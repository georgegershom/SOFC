#!/usr/bin/env python3
"""
Mechanical Properties Dataset Generator for Thermo-Mechanical Modeling
=====================================================================

Generates temperature-dependent mechanical properties for rubberized concrete mixes
with physically consistent degradation functions and statistical bounds.

Properties Generated:
- Compressive Strength
- Tensile Strength  
- Elastic Modulus
- Poisson's Ratio
- Fracture Properties (Fracture Toughness, Critical Strain Energy Release Rate)

Temperature Range: 20°C to 800°C
"""

import numpy as np
import pandas as pd
import os

# Constants and base parameters
TEMPERATURE_RANGE = np.arange(20, 801, 10)  # 20°C to 800°C in 10°C increments
MIX_TYPES = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']

# Base mechanical properties at 20°C for control concrete
BASE_PROPERTIES = {
    'compressive_strength': 45.0,    # MPa
    'tensile_strength': 4.2,         # MPa  
    'elastic_modulus': 32000,        # MPa
    'poissons_ratio': 0.18,          # dimensionless
    'fracture_toughness': 1.2,       # MPa·m^0.5
    'fracture_energy': 120           # J/m²
}

def rubber_mechanical_modification(rubber_content, particle_size='small'):
    """
    Calculate mechanical property modification factors based on rubber content and size.
    
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
    # Size effect factors - larger particles generally have more detrimental effects
    size_factor = 1.15 if particle_size == 'large' else 1.0
    
    # Property-specific modification factors based on literature
    factors = {
        'compressive_strength': 1.0 - (rubber_content / 100) * 0.65 * size_factor,
        'tensile_strength': 1.0 - (rubber_content / 100) * 0.45 * size_factor,
        'elastic_modulus': 1.0 - (rubber_content / 100) * 0.85 * size_factor,
        'poissons_ratio': 1.0 + (rubber_content / 100) * 0.35 * (1/size_factor),  # Inverse for Poisson's
        'fracture_toughness': 1.0 + (rubber_content / 100) * 0.25 * (1/size_factor),  # Rubber improves toughness
        'fracture_energy': 1.0 + (rubber_content / 100) * 0.45 * (1/size_factor)
    }
    
    return factors

def temperature_mechanical_degradation(T, property_type, mix_type):
    """
    Temperature-dependent degradation functions for mechanical properties.
    
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
    
    # Extract rubber content for rubber-specific effects
    if 'R' in mix_type:
        rubber_content = float(mix_type[1:3]) if mix_type[1:3].isdigit() else 5
    else:
        rubber_content = 0
    
    if property_type == 'compressive_strength':
        # Strength initially increases slightly (up to ~200°C) then decreases
        # Rubber mixes show more pronounced degradation at high temperatures
        degradation_factor = 1.0 + rubber_content * 0.015
        
        if T <= 200:
            # Slight increase due to accelerated hydration/curing
            return 1.0 + 0.15 * (T - 20) / 180
        else:
            # Degradation due to dehydration and microcracking
            high_temp_factor = (T - 200) / 600
            return 1.15 * np.exp(-1.8 * high_temp_factor * degradation_factor)
    
    elif property_type == 'tensile_strength':
        # Tensile strength degrades more rapidly than compressive
        degradation_factor = 1.2 + rubber_content * 0.02
        
        if T <= 150:
            return 1.0 + 0.05 * (T - 20) / 130
        else:
            high_temp_factor = (T - 150) / 650
            return 1.05 * np.exp(-2.2 * high_temp_factor * degradation_factor)
    
    elif property_type == 'elastic_modulus':
        # Modulus decreases almost linearly with temperature
        # Rubber content accelerates degradation
        degradation_factor = 1.0 + rubber_content * 0.012
        return np.exp(-0.9 * T_norm * degradation_factor)
    
    elif property_type == 'poissons_ratio':
        # Poisson's ratio increases slightly with temperature
        # Less affected by rubber content
        enhancement_factor = 1.0 - rubber_content * 0.005
        return 1.0 + 0.25 * T_norm * enhancement_factor
    
    elif property_type == 'fracture_toughness':
        # Fracture toughness initially stable, then decreases at high temperatures
        # Rubber provides some thermal stability
        stability_factor = 1.0 + rubber_content * 0.008
        
        if T <= 300:
            return 1.0 + 0.05 * T_norm * stability_factor
        else:
            high_temp_factor = (T - 300) / 500
            return (1.0 + 0.05 * (300 - 20) / 780 * stability_factor) * \
                   np.exp(-0.8 * high_temp_factor)
    
    elif property_type == 'fracture_energy':
        # Similar to fracture toughness but more sensitive to temperature
        stability_factor = 1.0 + rubber_content * 0.01
        
        if T <= 250:
            return 1.0 + 0.1 * T_norm * stability_factor
        else:
            high_temp_factor = (T - 250) / 550
            return (1.0 + 0.1 * (250 - 20) / 780 * stability_factor) * \
                   np.exp(-1.2 * high_temp_factor)

def generate_compressive_strength_data():
    """Generate compressive strength data for all mix types."""
    
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
        mod_factors = rubber_mechanical_modification(rubber_content, particle_size)
        
        # Base strength at 20°C
        base_strength = BASE_PROPERTIES['compressive_strength'] * mod_factors['compressive_strength']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature degradation
            temp_factor = temperature_mechanical_degradation(temp, 'compressive_strength', mix_id)
            strength_mean = base_strength * temp_factor
            
            # Add statistical variation (±12% standard deviation)
            strength_std = strength_mean * 0.12
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Mechanical',
                'Compressive_Strength_Mean_MPa': round(strength_mean, 2),
                'Compressive_Strength_Std_MPa': round(strength_std, 2),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_tensile_strength_data():
    """Generate tensile strength data for all mix types."""
    
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
        mod_factors = rubber_mechanical_modification(rubber_content, particle_size)
        
        # Base strength at 20°C
        base_strength = BASE_PROPERTIES['tensile_strength'] * mod_factors['tensile_strength']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature degradation
            temp_factor = temperature_mechanical_degradation(temp, 'tensile_strength', mix_id)
            strength_mean = base_strength * temp_factor
            
            # Add statistical variation (±15% standard deviation - higher for tensile)
            strength_std = strength_mean * 0.15
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Mechanical',
                'Tensile_Strength_Mean_MPa': round(strength_mean, 3),
                'Tensile_Strength_Std_MPa': round(strength_std, 3),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_elastic_modulus_data():
    """Generate elastic modulus data for all mix types."""
    
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
        mod_factors = rubber_mechanical_modification(rubber_content, particle_size)
        
        # Base modulus at 20°C
        base_modulus = BASE_PROPERTIES['elastic_modulus'] * mod_factors['elastic_modulus']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature degradation
            temp_factor = temperature_mechanical_degradation(temp, 'elastic_modulus', mix_id)
            modulus_mean = base_modulus * temp_factor
            
            # Add statistical variation (±10% standard deviation)
            modulus_std = modulus_mean * 0.10
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Mechanical',
                'Elastic_Modulus_Mean_MPa': round(modulus_mean, 0),
                'Elastic_Modulus_Std_MPa': round(modulus_std, 0),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_poissons_ratio_data():
    """Generate Poisson's ratio data for all mix types."""
    
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
        mod_factors = rubber_mechanical_modification(rubber_content, particle_size)
        
        # Base Poisson's ratio at 20°C
        base_poisson = BASE_PROPERTIES['poissons_ratio'] * mod_factors['poissons_ratio']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature modification
            temp_factor = temperature_mechanical_degradation(temp, 'poissons_ratio', mix_id)
            poisson_mean = base_poisson * temp_factor
            
            # Ensure physical bounds (0 < ν < 0.5)
            poisson_mean = max(0.05, min(0.45, poisson_mean))
            
            # Add statistical variation (±8% standard deviation)
            poisson_std = poisson_mean * 0.08
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Mechanical',
                'Poissons_Ratio_Mean': round(poisson_mean, 4),
                'Poissons_Ratio_Std': round(poisson_std, 4),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def generate_fracture_properties_data():
    """Generate fracture properties data for all mix types."""
    
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
        mod_factors = rubber_mechanical_modification(rubber_content, particle_size)
        
        # Base fracture properties at 20°C
        base_toughness = BASE_PROPERTIES['fracture_toughness'] * mod_factors['fracture_toughness']
        base_energy = BASE_PROPERTIES['fracture_energy'] * mod_factors['fracture_energy']
        
        for temp in TEMPERATURE_RANGE:
            # Apply temperature modifications
            toughness_temp_factor = temperature_mechanical_degradation(temp, 'fracture_toughness', mix_id)
            energy_temp_factor = temperature_mechanical_degradation(temp, 'fracture_energy', mix_id)
            
            toughness_mean = base_toughness * toughness_temp_factor
            energy_mean = base_energy * energy_temp_factor
            
            # Add statistical variation
            toughness_std = toughness_mean * 0.18  # ±18% for fracture toughness
            energy_std = energy_mean * 0.20        # ±20% for fracture energy
            
            # Split into calibration and validation data
            data_type = 'Calibration' if np.random.random() < 0.7 else 'Validation'
            
            data.append({
                'Mix_ID': mix_id,
                'Temperature_C': temp,
                'Data_Type': data_type,
                'Property_Type': 'Mechanical',
                'Fracture_Toughness_Mean_MPa_m05': round(toughness_mean, 4),
                'Fracture_Toughness_Std_MPa_m05': round(toughness_std, 4),
                'Fracture_Energy_Mean_J_m2': round(energy_mean, 1),
                'Fracture_Energy_Std_J_m2': round(energy_std, 1),
                'Rubber_Content_Percent': rubber_content,
                'Particle_Size': particle_size
            })
    
    return pd.DataFrame(data)

def main():
    """Generate all mechanical properties datasets."""
    
    print("Generating Mechanical Properties Dataset...")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate datasets
    print("1. Generating compressive strength data...")
    comp_strength_df = generate_compressive_strength_data()
    
    print("2. Generating tensile strength data...")
    tens_strength_df = generate_tensile_strength_data()
    
    print("3. Generating elastic modulus data...")
    modulus_df = generate_elastic_modulus_data()
    
    print("4. Generating Poisson's ratio data...")
    poisson_df = generate_poissons_ratio_data()
    
    print("5. Generating fracture properties data...")
    fracture_df = generate_fracture_properties_data()
    
    # Create output directory
    output_dir = "/workspace/thermo_mechanical_dataset/mechanical_properties"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    print("\nSaving datasets...")
    comp_strength_df.to_csv(f"{output_dir}/compressive_strength.csv", index=False)
    tens_strength_df.to_csv(f"{output_dir}/tensile_strength.csv", index=False)
    modulus_df.to_csv(f"{output_dir}/elastic_modulus.csv", index=False)
    poisson_df.to_csv(f"{output_dir}/poissons_ratio.csv", index=False)
    fracture_df.to_csv(f"{output_dir}/fracture_properties.csv", index=False)
    
    # Generate summary statistics
    print("\nDataset Summary:")
    print(f"- Compressive Strength: {len(comp_strength_df)} data points")
    print(f"- Tensile Strength: {len(tens_strength_df)} data points")
    print(f"- Elastic Modulus: {len(modulus_df)} data points")
    print(f"- Poisson's Ratio: {len(poisson_df)} data points")
    print(f"- Fracture Properties: {len(fracture_df)} data points")
    print(f"- Temperature range: {TEMPERATURE_RANGE[0]}°C to {TEMPERATURE_RANGE[-1]}°C")
    print(f"- Mix types: {', '.join(MIX_TYPES)}")
    
    # Calculate calibration/validation split
    total_points = len(comp_strength_df)
    cal_points = len(comp_strength_df[comp_strength_df['Data_Type'] == 'Calibration'])
    val_points = len(comp_strength_df[comp_strength_df['Data_Type'] == 'Validation'])
    
    print(f"\nData Split:")
    print(f"- Calibration: {cal_points} points ({cal_points/total_points*100:.1f}%)")
    print(f"- Validation: {val_points} points ({val_points/total_points*100:.1f}%)")
    
    # Display sample property ranges
    print(f"\nProperty Ranges at 20°C:")
    for mix_id in MIX_TYPES:
        mix_data = comp_strength_df[(comp_strength_df['Mix_ID'] == mix_id) & 
                                   (comp_strength_df['Temperature_C'] == 20)]
        if not mix_data.empty:
            comp_str = mix_data.iloc[0]['Compressive_Strength_Mean_MPa']
            print(f"- {mix_id}: {comp_str:.1f} MPa compressive strength")
    
    print("\nMechanical properties dataset generation completed successfully!")

if __name__ == "__main__":
    main()