#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Fire-Resistant Structural Elements Research
Development and Validation of a Thermo-Mechanical Model for Fire-Resistant 
Structural Elements Utilizing High-Performance Rubberized Concrete

This script generates a comprehensive synthetic dataset capturing:
- Multi-dimensional experimental matrix (Mix_ID, Curing_Age, Exposure_Type, etc.)
- Physically consistent degradation trends
- Stochastic realism with controlled scatter
- Multi-scale data linkage
- Advanced measurement simulation

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("SYNTHETIC FIRE-RESISTANCE DATASET GENERATOR")
print("Thermo-Mechanical Model for Rubberized Concrete")
print("="*80)
print(f"Generation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

def generate_ambient_data():
    """
    Generate ambient condition test data for all mix types and curing ages.
    Returns: DataFrame with ambient properties
    """
    print("Generating Ambient Condition Test Data...")
    
    # Define the base matrix: 6 mixes x 3 ages = 18 conditions. 3 specimens each = 54 data rows.
    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    ages = [7, 28, 56]
    specimens_per_condition = 3

    ambient_data = []

    for mix in mix_ids:
        # Base properties decrease with rubber content
        base_fc_28 = 65 - (mix_ids.index(mix) * 8)  # Control=65 MPa, dropping by 8 MPa per step
        base_ft_28 = base_fc_28 * 0.1  # Tensile strength ~10% of compressive
        base_E_28 = 35000 - (mix_ids.index(mix) * 3000)  # Modulus of Elasticity in MPa
        base_UPV = 4500 - (mix_ids.index(mix) * 150)  # UPV in m/s

        for age in ages:
            # Age factor: strength gain from 7 to 28 to 56 days
            if age == 7:
                age_factor_fc, age_factor_ft, age_factor_E = 0.75, 0.75, 0.85
            elif age == 28:
                age_factor_fc, age_factor_ft, age_factor_E = 1.0, 1.0, 1.0
            else:  # 56 days
                age_factor_fc, age_factor_ft, age_factor_E = 1.05, 1.03, 1.02

            for spec_num in range(1, specimens_per_condition + 1):
                specimen_id = f"{mix}-{age}-A-{spec_num}"
                # Introduce realistic variability (5% COV for strength is common)
                fc = (base_fc_28 * age_factor_fc) * np.random.normal(1, 0.05)
                ft = (base_ft_28 * age_factor_ft) * np.random.normal(1, 0.06)
                E = (base_E_28 * age_factor_E) * np.random.normal(1, 0.04)
                UPV = base_UPV * np.random.normal(1, 0.02)
                density = 2400 - (mix_ids.index(mix) * 40)  # kg/m³

                ambient_data.append({
                    'Specimen_ID': specimen_id,
                    'Mix_ID': mix,
                    'Curing_Age_days': age,
                    'Test_Type': 'Ambient',
                    'Compressive_Strength_MPa': max(0, fc),
                    'Tensile_Strength_MPa': max(0, ft),
                    'Modulus_of_Elasticity_MPa': max(0, E),
                    'Dry_Density_kgm3': density,
                    'UPV_mps': UPV
                })

    df_ambient = pd.DataFrame(ambient_data)
    print(f"✓ Generated {len(df_ambient)} ambient test specimens")
    return df_ambient

def generate_residual_data(df_ambient):
    """
    Generate residual high-temperature properties data with complex degradation patterns.
    Returns: DataFrame with residual properties
    """
    print("Generating Residual High-Temperature Properties Data...")
    
    # Expanded Experimental Matrix for Residual Tests
    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    peak_temps = [23, 200, 400, 600, 800]
    cooling_methods = ['Furnace', 'Quench']
    heating_rates = ['5_C_per_min']  # All mixes get standard heating
    # Add rapid heating for control and high-rubber mixes to study spalling
    spalling_mixes = ['C', 'R15S', 'R20S']
    specimens_per_condition = 3

    residual_data = []

    for mix in mix_ids:
        # Get the 28-day ambient strength for this mix to base degradation on
        ambient_fc = df_ambient[(df_ambient['Mix_ID'] == mix) & 
                               (df_ambient['Curing_Age_days'] == 28) & 
                               (df_ambient['Test_Type'] == 'Ambient')]['Compressive_Strength_MPa'].mean()
        
        for temp in peak_temps:
            for cooling in cooling_methods:
                hr_list = heating_rates
                if mix in spalling_mixes:
                    hr_list = ['5_C_per_min', '10_C_per_min']  # Add rapid heating for spalling study
                
                for hr in hr_list:
                    for spec_num in range(1, specimens_per_condition + 1):
                        specimen_id = f"{mix}-28-R-{temp}-{cooling}-{hr}-{spec_num}"
                        
                        # --- Calculate Degradation Factors ---
                        # Mass Loss: increases with temperature, more for rubber mixes
                        base_mass_loss = 0.0
                        if temp >= 105:
                            base_mass_loss = (temp / 1000) * 8  # Basic dehydration
                            if mix != 'C':
                                base_mass_loss += (mix_ids.index(mix) * 0.5) * (temp / 800)  # Rubber burns off
                        
                        # Strength Retention: complex function of T and mix
                        if temp <= 200:
                            strength_retention = 1.0 + np.random.normal(0, 0.05)  # Slight increase possible
                        elif temp <= 400:
                            # Rubber mixes show a "softening zone" then more rapid drop
                            if mix == 'C':
                                strength_retention = 0.75 - (temp - 200) * 0.002
                            else:
                                strength_retention = 0.65 - (temp - 200) * 0.0015
                        elif temp <= 600:
                            if mix == 'C':
                                strength_retention = 0.4 - (temp - 400) * 0.002
                            else:
                                # Rubber has combusted, leaving pores, but may reduce spalling
                                strength_retention = 0.3 - (temp - 400) * 0.0015
                        else:  # 800°C
                            strength_retention = 0.1 + np.random.normal(0, 0.03)

                        # Quenching causes additional 15% strength loss due to thermal shock
                        if cooling == 'Quench' and temp > 105:
                            strength_retention *= 0.85

                        # Rapid heating increases mass loss and strength loss for non-rubber mixes (spalling risk)
                        if hr == '10_C_per_min' and mix == 'C' and temp >= 400:
                            strength_retention *= 0.7  # Spalling reduces strength drastically
                            base_mass_loss *= 1.3  # More mass loss due to spalling

                        # UPV correlates with damage
                        UPV_retention = strength_retention ** 0.5  # Rough correlation

                        # --- Fabricate Data Points with Noise ---
                        fc_residual = (ambient_fc * strength_retention) * np.random.normal(1, 0.08)
                        mass_loss_pct = base_mass_loss * np.random.normal(1, 0.1)
                        UPV_residual = (4500 * UPV_retention) * np.random.normal(1, 0.05)

                        # Spalling Flag based on conditions
                        spalling_occurred = False
                        spalling_depth_mm = 0.0
                        if mix == 'C' and temp >= 400 and hr == '10_C_per_min':
                            spalling_occurred = True
                            spalling_depth_mm = np.random.uniform(5, 25)  # 5-25 mm spalling depth
                        elif mix in ['R15S', 'R20S'] and temp >= 400 and hr == '10_C_per_min':
                            # Rubber mixes might have minor spalling
                            if np.random.random() > 0.7:  # 30% chance
                                spalling_occurred = True
                                spalling_depth_mm = np.random.uniform(2, 10)

                        residual_data.append({
                            'Specimen_ID': specimen_id,
                            'Mix_ID': mix,
                            'Peak_Temperature_C': temp,
                            'Heating_Rate': hr,
                            'Cooling_Method': cooling,
                            'Test_Type': 'Residual',
                            'Mass_Loss_pct': max(0, mass_loss_pct),
                            'UPV_mps': max(500, UPV_residual),
                            'Residual_Compressive_Strength_MPa': max(0, fc_residual),
                            'Spalling_Occurred': spalling_occurred,
                            'Spalling_Depth_mm': spalling_depth_mm,
                            'Visual_Cracking_Rating': np.random.choice(['None', 'Minor', 'Moderate', 'Severe'], 
                                                                      p=[0.3, 0.4, 0.2, 0.1])
                        })

    df_residual = pd.DataFrame(residual_data)
    print(f"✓ Generated {len(df_residual)} residual high-temperature test specimens")
    return df_residual

def generate_insitu_data(df_ambient):
    """
    Generate in-situ high-temperature test data with stress-strain curves.
    Returns: DataFrame with in-situ properties and stress-strain curves dictionary
    """
    print("Generating In-Situ High-Temperature Test Data...")
    
    # In-Situ tests are complex, so we focus on key mixes and temperatures
    key_mixes = ['C', 'R10S', 'R20S']  # Control, medium rubber, high rubber
    in_situ_temps = [23, 200, 400, 600]  # 800°C is often too difficult to test in-situ
    specimens_per_condition = 2

    in_situ_data = []
    stress_strain_curves = {}  # Dictionary to store full stress-strain curves

    for mix in key_mixes:
        ambient_fc = df_ambient[(df_ambient['Mix_ID'] == mix) & 
                               (df_ambient['Curing_Age_days'] == 28) & 
                               (df_ambient['Test_Type'] == 'Ambient')]['Compressive_Strength_MPa'].mean()
        ambient_E = df_ambient[(df_ambient['Mix_ID'] == mix) & 
                              (df_ambient['Curing_Age_days'] == 28) & 
                              (df_ambient['Test_Type'] == 'Ambient')]['Modulus_of_Elasticity_MPa'].mean()
        
        for temp in in_situ_temps:
            for spec_num in range(1, specimens_per_condition + 1):
                specimen_id = f"{mix}-28-IS-{temp}-{spec_num}"
                
                # Strength retention in-situ is different from residual
                if temp <= 200:
                    strength_retention = 1.0
                    E_retention = 0.9
                elif temp <= 400:
                    strength_retention = 0.8 - (temp - 200) * 0.001
                    E_retention = 0.6
                else:  # 600°C
                    strength_retention = 0.4
                    E_retention = 0.2

                # Rubber mixes show more ductility in-situ
                peak_strain_factor = 1.0
                if mix != 'C':
                    peak_strain_factor = 1.0 + (['C', 'R10S', 'R20S'].index(mix) * 0.2)  # More rubber = more strain capacity

                fc_in_situ = ambient_fc * strength_retention * np.random.normal(1, 0.07)
                E_in_situ = ambient_E * E_retention * np.random.normal(1, 0.06)
                
                # Generate a synthetic stress-strain curve
                strain_points = np.linspace(0, 0.025, 50)
                peak_strain = 0.002 * peak_strain_factor * (1 + 0.005 * temp)  # Strain at peak increases with T
                stress_points = []
                
                for strain in strain_points:
                    if strain <= peak_strain:
                        # Parabolic ascending branch
                        stress = fc_in_situ * (2 * (strain/peak_strain) - (strain/peak_strain)**2)
                    else:
                        # Linear descending branch (more gradual for rubber mixes)
                        descent_factor = 0.3 if mix == 'C' else 0.15
                        stress = fc_in_situ * max(0, 1 - descent_factor * (strain - peak_strain) / peak_strain)
                    stress_points.append(stress * np.random.normal(1, 0.02))  # Add noise
                
                stress_strain_curves[specimen_id] = {'strain': strain_points, 'stress': stress_points}

                in_situ_data.append({
                    'Specimen_ID': specimen_id,
                    'Mix_ID': mix,
                    'Test_Temperature_C': temp,
                    'Test_Type': 'In-Situ',
                    'InSitu_Compressive_Strength_MPa': max(0, fc_in_situ),
                    'InSitu_Modulus_of_Elasticity_MPa': max(0, E_in_situ),
                    'Peak_Strain': peak_strain,
                    'Poissons_Ratio': max(0.1, 0.2 - (temp * 0.0002))  # Decreases with temperature
                })

    df_in_situ = pd.DataFrame(in_situ_data)
    print(f"✓ Generated {len(df_in_situ)} in-situ high-temperature test specimens")
    return df_in_situ, stress_strain_curves

def generate_advanced_measurements():
    """
    Generate pore pressure and transient strain data for spalling investigation.
    Returns: DataFrame with pore pressure data
    """
    print("Generating Advanced Measurements (Pore Pressure Data)...")
    
    # Pore pressure data for spalling investigation
    pore_pressure_data = []
    times = np.linspace(0, 120, 121)  # 2-hour heating timeline in minutes

    for mix in ['C', 'R20S']:  # Compare control vs high-rubber
        for depth in [10, 25, 40]:  # mm from exposed surface
            for run in [1, 2]:  # Two experimental runs
                pressures = []
                for t in times:
                    temp_at_t = min(800, t * 10)  # Rough temperature approximation
                    if mix == 'C':
                        # Control: high pore pressure build-up, sharp peak around 250°C
                        base_pressure = 0.5 * np.exp(-((t-25)/15)**2) * (depth/40)  # Gaussian peak
                        if temp_at_t > 300:
                            base_pressure *= 0.5  # Pressure release after dehydration
                    else:
                        # Rubber concrete: lower peak pressure, broader peak due to rubber melting
                        base_pressure = 0.3 * np.exp(-((t-30)/25)**2) * (depth/40)
                    
                    pressure = base_pressure + np.random.normal(0, 0.02)
                    pressures.append(max(0, pressure))
                
                pore_pressure_data.append({
                    'Mix_ID': mix,
                    'Depth_mm': depth,
                    'Run': run,
                    'Time_min': times,
                    'Pore_Pressure_MPa': pressures
                })

    # Create a summary DataFrame for peak pressures
    peak_pressures = []
    for pp in pore_pressure_data:
        peak_pressures.append({
            'Mix_ID': pp['Mix_ID'],
            'Depth_mm': pp['Depth_mm'],
            'Run': pp['Run'],
            'Peak_Pressure_MPa': max(pp['Pore_Pressure_MPa']),
            'Time_of_Peak_min': pp['Time_min'][np.argmax(pp['Pore_Pressure_MPa'])]
        })

    df_pore_pressure = pd.DataFrame(peak_pressures)
    print(f"✓ Generated {len(df_pore_pressure)} pore pressure measurement configurations")
    return df_pore_pressure, pore_pressure_data

def create_visualizations(df_ambient, df_residual, df_in_situ, stress_strain_curves, df_pore_pressure, pore_pressure_data):
    """
    Create comprehensive visualizations of the generated dataset.
    """
    print("Creating Comprehensive Visualizations...")
    
    # Create a master summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Residual strength vs temperature
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        strength_by_temp = mix_data.groupby('Peak_Temperature_C')['Residual_Compressive_Strength_MPa'].mean()
        axes[0,0].plot(strength_by_temp.index, strength_by_temp.values, 'o-', label=mix, linewidth=2, markersize=6)

    axes[0,0].set_xlabel('Peak Temperature (°C)')
    axes[0,0].set_ylabel('Residual Compressive Strength (MPa)')
    axes[0,0].set_title('A. Strength Degradation with Temperature\n(Furnace Cooled)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Plot 2: Effect of cooling method
    temp = 400
    for mix in ['C', 'R20S']:
        for cooling in ['Furnace', 'Quench']:
            cool_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                                   (df_residual['Peak_Temperature_C'] == temp) &
                                   (df_residual['Heating_Rate'] == '5_C_per_min') &
                                   (df_residual['Cooling_Method'] == cooling)]
            strength = cool_data['Residual_Compressive_Strength_MPa'].mean()
            axes[0,1].bar(f"{mix}\n{cooling}", strength, alpha=0.7, 
                         color='red' if cooling == 'Quench' else 'blue')

    axes[0,1].set_ylabel('Residual Compressive Strength (MPa)')
    axes[0,1].set_title(f'B. Effect of Cooling Method at {temp}°C\n(Thermal Shock Damage)')
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Mass loss correlation
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        axes[1,0].scatter(mix_data['Mass_Loss_pct'], mix_data['Residual_Compressive_Strength_MPa'], 
                         label=mix, alpha=0.6, s=50)

    axes[1,0].set_xlabel('Mass Loss (%)')
    axes[1,0].set_ylabel('Residual Compressive Strength (MPa)')
    axes[1,0].set_title('C. Strength vs. Mass Loss Correlation')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Spalling risk summary
    spalling_summary = df_residual[df_residual['Heating_Rate'] == '10_C_per_min']
    spalling_rates = spalling_summary.groupby(['Mix_ID', 'Peak_Temperature_C'])['Spalling_Occurred'].mean().reset_index()
    pivot_spalling = spalling_rates.pivot(index='Peak_Temperature_C', columns='Mix_ID', values='Spalling_Occurred')

    pivot_spalling.plot(kind='bar', ax=axes[1,1], width=0.8)
    axes[1,1].set_xlabel('Peak Temperature (°C)')
    axes[1,1].set_ylabel('Probability of Spalling')
    axes[1,1].set_title('D. Spalling Risk: Rapid Heating (10°C/min)')
    axes[1,1].legend(title='Mix ID')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comprehensive_high_temperature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot stress-strain curves
    plt.figure(figsize=(10, 6))
    for mix in ['C', 'R10S', 'R20S']:
        spec_id = f"{mix}-28-IS-400-1"
        if spec_id in stress_strain_curves:
            plt.plot(stress_strain_curves[spec_id]['strain']*100, 
                    stress_strain_curves[spec_id]['stress'], 
                    label=f'{mix}, 400°C', linewidth=2)

    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (MPa)')
    plt.title('In-Situ Compressive Stress-Strain Behavior at High Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('in_situ_stress_strain_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot pore pressure evolution
    plt.figure(figsize=(12, 6))
    for mix in ['C', 'R20S']:
        for depth in [10, 25, 40]:
            # Get the first run for plotting
            data = next(pp for pp in pore_pressure_data if pp['Mix_ID'] == mix and pp['Depth_mm'] == depth and pp['Run'] == 1)
            plt.plot(data['Time_min'], data['Pore_Pressure_MPa'], 
                    label=f'{mix}, {depth}mm', linewidth=2)

    plt.xlabel('Time (min)')
    plt.ylabel('Pore Pressure (MPa)')
    plt.title('Pore Pressure Evolution During Heating\n(10°C/min heating rate)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pore_pressure_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ All visualizations created and saved")

def export_datasets(df_ambient, df_residual, df_in_situ, stress_strain_curves, df_pore_pressure):
    """
    Export all datasets to CSV and JSON formats.
    """
    print("Exporting Datasets...")
    
    # Save all datasets to CSV files
    df_ambient.to_csv('ambient_properties.csv', index=False)
    df_residual.to_csv('residual_properties_high_temp.csv', index=False)
    df_in_situ.to_csv('in_situ_properties.csv', index=False)
    df_pore_pressure.to_csv('pore_pressure_summary.csv', index=False)

    # Save stress-strain curves as a JSON file for easy reloading
    # Convert numpy arrays to lists for JSON serialization
    ss_curves_serializable = {}
    for key, value in stress_strain_curves.items():
        # Check if the values are already lists or numpy arrays
        strain_data = value['strain'].tolist() if hasattr(value['strain'], 'tolist') else value['strain']
        stress_data = value['stress'].tolist() if hasattr(value['stress'], 'tolist') else value['stress']
        
        ss_curves_serializable[key] = {
            'strain': strain_data,
            'stress': stress_data
        }

    with open('stress_strain_curves.json', 'w') as f:
        json.dump(ss_curves_serializable, f, indent=2)

    print("✓ All datasets exported to CSV and JSON formats")

def main():
    """
    Main function to generate the complete synthetic dataset.
    """
    print("Starting comprehensive synthetic dataset generation...")
    print()
    
    # Generate all datasets
    df_ambient = generate_ambient_data()
    df_residual = generate_residual_data(df_ambient)
    df_in_situ, stress_strain_curves = generate_insitu_data(df_ambient)
    df_pore_pressure, pore_pressure_data = generate_advanced_measurements()
    
    # Create visualizations
    create_visualizations(df_ambient, df_residual, df_in_situ, stress_strain_curves, df_pore_pressure, pore_pressure_data)
    
    # Export datasets
    export_datasets(df_ambient, df_residual, df_in_situ, stress_strain_curves, df_pore_pressure)
    
    # Print summary
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"Ambient tests: {len(df_ambient)} specimens")
    print(f"Residual high-temperature tests: {len(df_residual)} specimens")
    print(f"In-situ high-temperature tests: {len(df_in_situ)} specimens")
    print(f"Pore pressure experiments: {len(df_pore_pressure)} configurations")
    print()
    print("Generated Files:")
    print("- ambient_properties.csv")
    print("- residual_properties_high_temp.csv")
    print("- in_situ_properties.csv")
    print("- stress_strain_curves.json")
    print("- pore_pressure_summary.csv")
    print("- comprehensive_high_temperature_analysis.png")
    print("- in_situ_stress_strain_curves.png")
    print("- pore_pressure_evolution.png")
    print()
    print("This comprehensive dataset is now ready for thermo-mechanical model development and validation.")
    print(f"Generation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()