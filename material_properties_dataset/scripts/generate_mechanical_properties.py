#!/usr/bin/env python3
"""
Generate Mechanical Properties Dataset
Simulates temperature-dependent mechanical properties for multi-physics modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import json

# Set random seed for reproducibility
np.random.seed(42)

def generate_mechanical_properties():
    """
    Generate temperature-dependent mechanical properties for a typical superalloy
    Based on realistic values for Inconel 718 or similar materials
    """
    
    # Temperature range (Celsius)
    temperatures = np.arange(20, 1001, 20)  # 20°C to 1000°C
    
    # Generate Young's Modulus (GPa) - decreases with temperature
    # Typical values: ~200 GPa at room temp, ~150 GPa at 1000°C
    E_room = 205.0  # GPa at room temperature
    E_coefficients = [-0.055, -0.00002]  # Linear + quadratic temperature dependence
    
    youngs_modulus = []
    for T in temperatures:
        E = E_room * (1 + E_coefficients[0] * (T - 20) / 1000 + 
                     E_coefficients[1] * ((T - 20) / 1000) ** 2)
        # Add some experimental scatter
        E += np.random.normal(0, E * 0.01)  # 1% scatter
        youngs_modulus.append(max(E, 50))  # Minimum bound
    
    # Generate Tensile Strength (MPa) - decreases with temperature
    # Typical values: ~1400 MPa at room temp, ~200 MPa at 1000°C
    tensile_room = 1380.0  # MPa at room temperature
    
    tensile_strength = []
    for T in temperatures:
        if T < 600:
            # Gradual decrease
            sigma = tensile_room * (1 - 0.3 * (T - 20) / 580)
        else:
            # Rapid decrease at high temperatures
            sigma = tensile_room * 0.7 * np.exp(-2.5 * (T - 600) / 400)
        
        # Add experimental scatter
        sigma += np.random.normal(0, sigma * 0.02)  # 2% scatter
        tensile_strength.append(max(sigma, 50))  # Minimum bound
    
    # Generate Poisson's Ratio - slight temperature dependence
    # Typical values: 0.29-0.31
    poisson_room = 0.294
    
    poissons_ratio = []
    for T in temperatures:
        nu = poisson_room + 0.02 * (T - 20) / 1000  # Slight increase with temp
        # Add small experimental scatter
        nu += np.random.normal(0, 0.002)
        nu = max(0.25, min(0.35, nu))  # Physical bounds
        poissons_ratio.append(nu)
    
    # Generate Yield Strength (MPa) - typically 80-90% of tensile strength
    yield_strength = [0.85 * ts + np.random.normal(0, 10) for ts in tensile_strength]
    
    # Generate Elongation at Break (%) - increases with temperature
    elongation = []
    for T in temperatures:
        if T < 600:
            elong = 15 + 10 * (T - 20) / 580
        else:
            elong = 25 + 25 * (T - 600) / 400
        
        elong += np.random.normal(0, elong * 0.05)
        elongation.append(max(5, min(60, elong)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature_C': temperatures,
        'Youngs_Modulus_GPa': np.round(youngs_modulus, 2),
        'Tensile_Strength_MPa': np.round(tensile_strength, 1),
        'Yield_Strength_MPa': np.round(yield_strength, 1),
        'Poissons_Ratio': np.round(poissons_ratio, 4),
        'Elongation_Percent': np.round(elongation, 1),
        'Test_Standard': 'ASTM E21',
        'Specimen_Type': 'Round Bar',
        'Strain_Rate_s-1': 0.001,
        'Material': 'Superalloy_A',
        'Heat_Treatment': 'Solution_Annealed_Aged',
        'Measurement_Uncertainty_Percent': 2.0
    })
    
    # Add metadata
    metadata = {
        'material_class': 'Nickel-based Superalloy',
        'nominal_composition': {
            'Ni': '50-55%',
            'Cr': '17-21%',
            'Fe': '17%',
            'Nb+Ta': '4.75-5.5%',
            'Mo': '2.8-3.3%',
            'Ti': '0.65-1.15%',
            'Al': '0.2-0.8%'
        },
        'test_conditions': {
            'atmosphere': 'Air',
            'humidity': '50% RH',
            'grip_type': 'Hydraulic',
            'extensometer': 'High-temp contact'
        },
        'data_source': 'Synthetic data based on literature values',
        'generation_date': pd.Timestamp.now().isoformat(),
        'units': {
            'Temperature': 'Celsius',
            'Youngs_Modulus': 'GPa',
            'Tensile_Strength': 'MPa',
            'Yield_Strength': 'MPa',
            'Poissons_Ratio': 'dimensionless',
            'Elongation': 'percent'
        }
    }
    
    return df, metadata

def plot_mechanical_properties(df, save_path):
    """Create visualization of mechanical properties vs temperature"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature-Dependent Mechanical Properties', fontsize=16, fontweight='bold')
    
    # Young's Modulus
    ax = axes[0, 0]
    ax.plot(df['Temperature_C'], df['Youngs_Modulus_GPa'], 'bo-', linewidth=2, markersize=4)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel("Young's Modulus (GPa)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title("Young's Modulus vs Temperature")
    
    # Tensile and Yield Strength
    ax = axes[0, 1]
    ax.plot(df['Temperature_C'], df['Tensile_Strength_MPa'], 'ro-', label='Tensile Strength', linewidth=2, markersize=4)
    ax.plot(df['Temperature_C'], df['Yield_Strength_MPa'], 'go-', label='Yield Strength', linewidth=2, markersize=4)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Strength (MPa)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Strength Properties vs Temperature')
    
    # Poisson's Ratio
    ax = axes[1, 0]
    ax.plot(df['Temperature_C'], df['Poissons_Ratio'], 'mo-', linewidth=2, markersize=4)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel("Poisson's Ratio", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title("Poisson's Ratio vs Temperature")
    
    # Elongation
    ax = axes[1, 1]
    ax.plot(df['Temperature_C'], df['Elongation_Percent'], 'co-', linewidth=2, markersize=4)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Elongation at Break (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('Ductility vs Temperature')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate data
    df, metadata = generate_mechanical_properties()
    
    # Save to CSV
    csv_path = '../mechanical/mechanical_properties.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved mechanical properties data to {csv_path}")
    
    # Save metadata to JSON
    json_path = '../mechanical/mechanical_properties_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {json_path}")
    
    # Create visualization
    plot_path = '../mechanical/mechanical_properties_plot.png'
    plot_mechanical_properties(df, plot_path)
    print(f"✅ Saved visualization to {plot_path}")
    
    # Display statistics
    print("\n📊 Data Statistics:")
    print(f"Temperature range: {df['Temperature_C'].min()}-{df['Temperature_C'].max()}°C")
    print(f"Young's Modulus range: {df['Youngs_Modulus_GPa'].min():.1f}-{df['Youngs_Modulus_GPa'].max():.1f} GPa")
    print(f"Tensile Strength range: {df['Tensile_Strength_MPa'].min():.0f}-{df['Tensile_Strength_MPa'].max():.0f} MPa")
    print(f"Number of data points: {len(df)}")