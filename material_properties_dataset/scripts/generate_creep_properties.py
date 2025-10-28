#!/usr/bin/env python3
"""
Generate Creep Properties Dataset
Simulates creep strain vs time curves for different temperatures and stress levels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

# Set random seed for reproducibility
np.random.seed(42)

def norton_bailey_creep(t, A, n, m, epsilon_0=0):
    """
    Norton-Bailey creep model
    ε = ε_0 + A * t^m * σ^n
    
    For this function, stress is embedded in A
    """
    return epsilon_0 + A * t**m

def generate_creep_curves():
    """
    Generate creep strain vs time curves at various temperatures and stress levels
    Based on typical superalloy behavior
    """
    
    # Test conditions
    temperatures = [600, 700, 750, 800, 850, 900, 950, 1000]  # °C
    stress_levels = [100, 150, 200, 250, 300, 350, 400]  # MPa
    
    # Time points (hours) - log spacing for better resolution
    time_points = np.logspace(-2, 3.5, 100)  # 0.01 to ~3162 hours
    
    all_data = []
    test_id = 0
    
    for temp in temperatures:
        for stress in stress_levels:
            test_id += 1
            
            # Temperature and stress dependent parameters
            # Higher temperature and stress = faster creep
            
            # Norton stress exponent (typically 3-8 for superalloys)
            n = 4.5 + 0.5 * np.random.normal(0, 1)
            
            # Time exponent (typically 0.3-0.7)
            m = 0.3 + 0.2 * (temp - 600) / 400 + 0.05 * np.random.normal(0, 1)
            m = max(0.2, min(0.8, m))
            
            # Creep coefficient (temperature and stress dependent)
            # Using Arrhenius-type temperature dependence
            Q = 280000  # Activation energy (J/mol)
            R = 8.314  # Gas constant (J/mol·K)
            T_kelvin = temp + 273.15
            
            A_base = 1e-15 * np.exp(-Q / (R * T_kelvin))
            A = A_base * (stress / 100) ** n
            
            # Generate strain data with three stages
            strain_data = []
            
            for t in time_points:
                # Primary creep (decreasing rate)
                primary = A * t**m * (1 - np.exp(-t/10))
                
                # Secondary creep (steady state)
                secondary_rate = A * (stress/200)**2 * np.exp(-Q/(R*T_kelvin))
                secondary = secondary_rate * t
                
                # Tertiary creep (accelerating, only at high stress/temp)
                tertiary = 0
                if temp > 800 and stress > 200:
                    t_tertiary = 1000 / (temp/800 * stress/200)  # Time to tertiary
                    if t > t_tertiary:
                        tertiary = 0.001 * ((t - t_tertiary) / 100) ** 2
                
                # Total strain (%) with some noise
                total_strain = (primary + secondary + tertiary) * 100
                noise = np.random.normal(0, 0.0001 + 0.01 * total_strain)
                total_strain += noise
                
                strain_data.append(total_strain)
            
            # Calculate derived properties
            min_creep_rate = np.min(np.gradient(strain_data) / np.gradient(time_points))
            rupture_strain = strain_data[-1]
            
            # Determine if specimen failed
            failed = rupture_strain > 20 or (temp > 900 and stress > 300)
            rupture_time = time_points[-1] if not failed else time_points[np.where(np.array(strain_data) > 20)[0][0]] if any(np.array(strain_data) > 20) else time_points[-1]
            
            # Create records for each time point
            for i, (t, strain) in enumerate(zip(time_points, strain_data)):
                all_data.append({
                    'Test_ID': f'CR_{test_id:04d}',
                    'Temperature_C': temp,
                    'Stress_MPa': stress,
                    'Time_Hours': np.round(t, 4),
                    'Creep_Strain_Percent': np.round(strain, 6),
                    'Strain_Rate_Per_Hour': np.round(np.gradient(strain_data)[i] / np.gradient(time_points)[i], 8),
                    'Test_Type': 'Constant_Load',
                    'Atmosphere': 'Air',
                    'Specimen_Failed': failed,
                    'Min_Creep_Rate_Per_Hour': np.round(min_creep_rate, 8),
                    'Rupture_Time_Hours': np.round(rupture_time, 2) if failed else None,
                    'Material': 'Superalloy_A',
                    'Specimen_Geometry': 'Cylindrical_6mm_dia'
                })
    
    df = pd.DataFrame(all_data)
    
    # Create summary statistics
    summary = df.groupby(['Temperature_C', 'Stress_MPa']).agg({
        'Creep_Strain_Percent': ['min', 'max'],
        'Min_Creep_Rate_Per_Hour': 'first',
        'Rupture_Time_Hours': 'first',
        'Specimen_Failed': 'first'
    }).round(6)
    
    # Metadata
    metadata = {
        'test_standard': 'ASTM E139',
        'material': 'Nickel-based Superalloy',
        'test_conditions': {
            'temperatures_C': temperatures,
            'stress_levels_MPa': stress_levels,
            'max_test_duration_hours': float(time_points[-1]),
            'loading_type': 'Constant Load',
            'specimen_preparation': 'Machined and polished to 0.4 μm'
        },
        'data_processing': {
            'strain_measurement': 'High-temperature extensometer',
            'data_acquisition_rate': '1 point per minute',
            'filtering': 'Moving average, window=5'
        },
        'models_applicable': [
            'Norton-Bailey',
            'Theta Projection',
            'Wilshire',
            'Larson-Miller Parameter'
        ],
        'generation_date': pd.Timestamp.now().isoformat()
    }
    
    return df, summary, metadata

def plot_creep_curves(df, save_path):
    """Create visualization of creep curves"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Creep Strain vs Time Curves', fontsize=16, fontweight='bold')
    
    # Select specific conditions for plotting
    plot_conditions = [
        (700, [150, 200, 250, 300]),  # Varying stress at 700°C
        (800, [150, 200, 250, 300]),  # Varying stress at 800°C
        ([700, 750, 800, 850], 200),  # Varying temperature at 200 MPa
        ([700, 750, 800, 850], 300),  # Varying temperature at 300 MPa
    ]
    
    for idx, (ax, condition) in enumerate(zip(axes.flat, plot_conditions)):
        if idx < 2:
            # Varying stress plots
            temp = condition[0]
            stresses = condition[1]
            
            for stress in stresses:
                data = df[(df['Temperature_C'] == temp) & (df['Stress_MPa'] == stress)]
                if not data.empty:
                    ax.loglog(data['Time_Hours'], data['Creep_Strain_Percent'], 
                             label=f'{stress} MPa', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Creep Strain (%)', fontsize=12)
            ax.set_title(f'Temperature: {temp}°C', fontsize=12)
            ax.legend(title='Stress Level')
            ax.grid(True, alpha=0.3, which='both')
            
        else:
            # Varying temperature plots
            temps = condition[0]
            stress = condition[1]
            
            for temp in temps:
                data = df[(df['Temperature_C'] == temp) & (df['Stress_MPa'] == stress)]
                if not data.empty:
                    ax.loglog(data['Time_Hours'], data['Creep_Strain_Percent'],
                             label=f'{temp}°C', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Creep Strain (%)', fontsize=12)
            ax.set_title(f'Stress: {stress} MPa', fontsize=12)
            ax.legend(title='Temperature')
            ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_creep_rate_map(df, save_path):
    """Create a contour map of minimum creep rates"""
    
    # Get unique values
    temps = sorted(df['Temperature_C'].unique())
    stresses = sorted(df['Stress_MPa'].unique())
    
    # Create grid
    creep_rates = np.zeros((len(stresses), len(temps)))
    
    for i, stress in enumerate(stresses):
        for j, temp in enumerate(temps):
            data = df[(df['Temperature_C'] == temp) & (df['Stress_MPa'] == stress)]
            if not data.empty:
                creep_rates[i, j] = abs(data['Min_Creep_Rate_Per_Hour'].iloc[0])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot
    X, Y = np.meshgrid(temps, stresses)
    levels = np.logspace(np.log10(creep_rates[creep_rates > 0].min()), 
                        np.log10(creep_rates.max()), 15)
    
    cs = ax.contourf(X, Y, creep_rates, levels=levels, cmap='hot', extend='both')
    ax.contour(X, Y, creep_rates, levels=levels, colors='black', linewidths=0.5, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, format='%.2e')
    cbar.set_label('Minimum Creep Rate (1/hour)', fontsize=12)
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Stress (MPa)', fontsize=12)
    ax.set_title('Minimum Creep Rate Map', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate data
    print("🔄 Generating creep data... This may take a moment...")
    df, summary, metadata = generate_creep_curves()
    
    # Save full dataset to CSV
    csv_path = '../creep/creep_curves_full.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved full creep data to {csv_path}")
    
    # Save summary to CSV
    summary_path = '../creep/creep_summary.csv'
    summary.to_csv(summary_path)
    print(f"✅ Saved summary data to {summary_path}")
    
    # Save metadata to JSON
    json_path = '../creep/creep_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {json_path}")
    
    # Create visualizations
    plot_path = '../creep/creep_curves_plot.png'
    plot_creep_curves(df, plot_path)
    print(f"✅ Saved creep curves plot to {plot_path}")
    
    rate_map_path = '../creep/creep_rate_map.png'
    plot_creep_rate_map(df, rate_map_path)
    print(f"✅ Saved creep rate map to {rate_map_path}")
    
    # Display statistics
    print("\n📊 Data Statistics:")
    print(f"Total data points: {len(df):,}")
    print(f"Unique test conditions: {df.groupby(['Temperature_C', 'Stress_MPa']).ngroups}")
    print(f"Temperature range: {df['Temperature_C'].min()}-{df['Temperature_C'].max()}°C")
    print(f"Stress range: {df['Stress_MPa'].min()}-{df['Stress_MPa'].max()} MPa")
    print(f"Max creep strain: {df['Creep_Strain_Percent'].max():.2f}%")
    print(f"Failed specimens: {df['Specimen_Failed'].sum() // len(df['Time_Hours'].unique())}")