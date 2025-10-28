#!/usr/bin/env python3
"""
Generate Thermo-Physical Properties Dataset
Simulates CTE, thermal conductivity, and specific heat capacity data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import json

# Set random seed for reproducibility
np.random.seed(42)

def generate_thermophysical_properties():
    """
    Generate temperature-dependent thermo-physical properties for superalloys
    """
    
    # Temperature range (Celsius) - finer resolution for thermal properties
    temperatures = np.arange(20, 1201, 10)  # 20°C to 1200°C
    
    # Coefficient of Thermal Expansion (CTE) - μm/m·K or ppm/K
    # Typically increases with temperature for metals
    cte_data = []
    for T in temperatures:
        if T < 100:
            cte = 11.5 + 0.8 * (T - 20) / 80
        elif T < 500:
            cte = 12.3 + 1.5 * (T - 100) / 400
        elif T < 800:
            cte = 13.8 + 2.0 * (T - 500) / 300
        else:
            cte = 15.8 + 1.2 * (T - 800) / 400
        
        # Add experimental scatter
        cte += np.random.normal(0, 0.1)
        cte_data.append(max(10, min(20, cte)))  # Physical bounds
    
    # Thermal Conductivity - W/m·K
    # Generally increases with temperature for superalloys
    thermal_conductivity = []
    for T in temperatures:
        # Base conductivity with temperature dependence
        k_base = 11.2  # W/m·K at room temperature
        
        if T < 200:
            k = k_base + 0.015 * (T - 20)
        elif T < 600:
            k = k_base + 0.015 * 180 + 0.020 * (T - 200)
        else:
            k = k_base + 0.015 * 180 + 0.020 * 400 + 0.010 * (T - 600)
        
        # Add experimental scatter
        k += np.random.normal(0, k * 0.02)  # 2% scatter
        thermal_conductivity.append(max(10, min(35, k)))
    
    # Specific Heat Capacity - J/kg·K
    # Generally increases with temperature
    specific_heat = []
    for T in temperatures:
        # Polynomial fit for Cp
        T_kelvin = T + 273.15
        
        # Shomate equation coefficients (simplified)
        cp = 420 + 0.15 * T - 2.5e-5 * T**2 + 1.2e-8 * T**3
        
        # Phase transition effects
        if 700 < T < 750:  # Gamma prime dissolution
            cp += 50 * np.exp(-((T - 725) / 15) ** 2)
        
        # Add experimental scatter
        cp += np.random.normal(0, cp * 0.01)  # 1% scatter
        specific_heat.append(max(400, min(700, cp)))
    
    # Thermal Diffusivity - mm²/s (derived property)
    # α = k / (ρ * Cp)
    density = 8190  # kg/m³ (typical for superalloys)
    thermal_diffusivity = []
    for k, cp in zip(thermal_conductivity, specific_heat):
        alpha = (k / (density * cp)) * 1e6  # Convert to mm²/s
        alpha += np.random.normal(0, alpha * 0.02)
        thermal_diffusivity.append(alpha)
    
    # Linear Thermal Expansion - % (integrated CTE)
    linear_expansion = []
    T_ref = 20  # Reference temperature
    for i, T in enumerate(temperatures):
        if T == T_ref:
            expansion = 0
        else:
            # Integrate CTE to get expansion
            expansion = np.trapz(cte_data[:i+1], temperatures[:i+1]) / 1e6 * 100
        linear_expansion.append(expansion)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature_C': temperatures,
        'CTE_ppm_per_K': np.round(cte_data, 3),
        'Thermal_Conductivity_W_per_mK': np.round(thermal_conductivity, 2),
        'Specific_Heat_J_per_kgK': np.round(specific_heat, 1),
        'Thermal_Diffusivity_mm2_per_s': np.round(thermal_diffusivity, 3),
        'Linear_Expansion_Percent': np.round(linear_expansion, 4),
        'Density_kg_per_m3': density,
        'Measurement_Method_CTE': 'Dilatometry',
        'Measurement_Method_k': 'Laser_Flash',
        'Measurement_Method_Cp': 'DSC',
        'Material': 'Superalloy_A',
        'Atmosphere': 'Argon',
        'Heating_Rate_K_per_min': 10
    })
    
    # Add some additional measurements at key temperatures with different methods
    key_temps = [20, 100, 200, 400, 600, 800, 1000]
    additional_measurements = []
    
    for temp in key_temps:
        idx = np.where(temperatures == temp)[0][0]
        
        # Steady-state measurement of thermal conductivity
        k_steady = thermal_conductivity[idx] + np.random.normal(0, 0.3)
        
        # Drop calorimetry for Cp
        cp_drop = specific_heat[idx] + np.random.normal(0, 5)
        
        additional_measurements.append({
            'Temperature_C': temp,
            'Thermal_Conductivity_Steady_State_W_per_mK': np.round(k_steady, 2),
            'Specific_Heat_Drop_Calorimetry_J_per_kgK': np.round(cp_drop, 1),
            'Measurement_Replicate': 'Validation'
        })
    
    df_validation = pd.DataFrame(additional_measurements)
    
    # Metadata
    metadata = {
        'material_specification': {
            'class': 'Nickel-based Superalloy',
            'density_room_temp_kg_per_m3': density,
            'melting_range_C': '1260-1336',
            'max_service_temperature_C': 1050
        },
        'measurement_standards': {
            'CTE': 'ASTM E831',
            'thermal_conductivity': 'ASTM E1461',
            'specific_heat': 'ASTM E1269',
            'density': 'ASTM B311'
        },
        'equipment': {
            'dilatometer': 'Model DIL 402C',
            'laser_flash': 'Model LFA 467',
            'DSC': 'Model DSC 404 F3',
            'balance': 'Precision 0.0001g'
        },
        'uncertainties': {
            'CTE': '±0.5 ppm/K',
            'thermal_conductivity': '±3%',
            'specific_heat': '±2%',
            'temperature': '±1°C'
        },
        'data_processing': {
            'baseline_correction': 'Applied',
            'thermal_lag_correction': 'Applied',
            'radiation_loss_correction': 'Applied above 500°C'
        },
        'generation_date': pd.Timestamp.now().isoformat()
    }
    
    return df, df_validation, metadata

def plot_thermophysical_properties(df, save_path):
    """Create comprehensive visualization of thermo-physical properties"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Temperature-Dependent Thermo-Physical Properties', fontsize=16, fontweight='bold')
    
    # CTE
    ax = axes[0, 0]
    ax.plot(df['Temperature_C'], df['CTE_ppm_per_K'], 'b-', linewidth=2)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('CTE (ppm/K)', fontsize=11)
    ax.set_title('Coefficient of Thermal Expansion')
    ax.grid(True, alpha=0.3)
    
    # Thermal Conductivity
    ax = axes[0, 1]
    ax.plot(df['Temperature_C'], df['Thermal_Conductivity_W_per_mK'], 'r-', linewidth=2)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=11)
    ax.set_title('Thermal Conductivity')
    ax.grid(True, alpha=0.3)
    
    # Specific Heat
    ax = axes[0, 2]
    ax.plot(df['Temperature_C'], df['Specific_Heat_J_per_kgK'], 'g-', linewidth=2)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Specific Heat (J/kg·K)', fontsize=11)
    ax.set_title('Specific Heat Capacity')
    ax.grid(True, alpha=0.3)
    
    # Thermal Diffusivity
    ax = axes[1, 0]
    ax.plot(df['Temperature_C'], df['Thermal_Diffusivity_mm2_per_s'], 'm-', linewidth=2)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Thermal Diffusivity (mm²/s)', fontsize=11)
    ax.set_title('Thermal Diffusivity')
    ax.grid(True, alpha=0.3)
    
    # Linear Expansion
    ax = axes[1, 1]
    ax.plot(df['Temperature_C'], df['Linear_Expansion_Percent'], 'c-', linewidth=2)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Linear Expansion (%)', fontsize=11)
    ax.set_title('Cumulative Linear Thermal Expansion')
    ax.grid(True, alpha=0.3)
    
    # Combined normalized plot
    ax = axes[1, 2]
    # Normalize all properties to 0-1 range for comparison
    props_norm = {
        'CTE': (df['CTE_ppm_per_K'] - df['CTE_ppm_per_K'].min()) / 
               (df['CTE_ppm_per_K'].max() - df['CTE_ppm_per_K'].min()),
        'k': (df['Thermal_Conductivity_W_per_mK'] - df['Thermal_Conductivity_W_per_mK'].min()) / 
             (df['Thermal_Conductivity_W_per_mK'].max() - df['Thermal_Conductivity_W_per_mK'].min()),
        'Cp': (df['Specific_Heat_J_per_kgK'] - df['Specific_Heat_J_per_kgK'].min()) / 
              (df['Specific_Heat_J_per_kgK'].max() - df['Specific_Heat_J_per_kgK'].min()),
        'α': (df['Thermal_Diffusivity_mm2_per_s'] - df['Thermal_Diffusivity_mm2_per_s'].min()) / 
             (df['Thermal_Diffusivity_mm2_per_s'].max() - df['Thermal_Diffusivity_mm2_per_s'].min())
    }
    
    for label, data in props_norm.items():
        ax.plot(df['Temperature_C'], data, label=label, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_title('Normalized Properties Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate data
    df, df_validation, metadata = generate_thermophysical_properties()
    
    # Save main dataset to CSV
    csv_path = '../thermophysical/thermophysical_properties.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved thermo-physical properties to {csv_path}")
    
    # Save validation dataset
    validation_path = '../thermophysical/thermophysical_validation.csv'
    df_validation.to_csv(validation_path, index=False)
    print(f"✅ Saved validation data to {validation_path}")
    
    # Save metadata to JSON
    json_path = '../thermophysical/thermophysical_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {json_path}")
    
    # Create visualization
    plot_path = '../thermophysical/thermophysical_properties_plot.png'
    plot_thermophysical_properties(df, plot_path)
    print(f"✅ Saved visualization to {plot_path}")
    
    # Display statistics
    print("\n📊 Data Statistics:")
    print(f"Temperature range: {df['Temperature_C'].min()}-{df['Temperature_C'].max()}°C")
    print(f"CTE range: {df['CTE_ppm_per_K'].min():.1f}-{df['CTE_ppm_per_K'].max():.1f} ppm/K")
    print(f"Thermal conductivity range: {df['Thermal_Conductivity_W_per_mK'].min():.1f}-{df['Thermal_Conductivity_W_per_mK'].max():.1f} W/m·K")
    print(f"Specific heat range: {df['Specific_Heat_J_per_kgK'].min():.0f}-{df['Specific_Heat_J_per_kgK'].max():.0f} J/kg·K")
    print(f"Max linear expansion: {df['Linear_Expansion_Percent'].max():.3f}%")
    print(f"Number of data points: {len(df)}")