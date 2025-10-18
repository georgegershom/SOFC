#!/usr/bin/env python3
"""
Comprehensive Synthetic Dataset Generator for Fire-Resistant Structural Elements Research

Title: "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant 
       Structural Elements Utilizing High-Performance Rubberized Concrete"

This script generates scientifically plausible, internally consistent synthetic data
for thermo-mechanical degradation pathways in rubberized concrete under fire conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("SYNTHETIC FIRE RESISTANCE DATASET GENERATOR")
print("="*80)
print(f"Generation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# PART A: AMBIENT CONDITION TESTS
# ============================================================================

def generate_ambient_data():
    """Generate ambient condition test data for baseline properties"""
    print("Generating Part A: Ambient Condition Tests...")
    
    # Define experimental matrix
    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    ages = [7, 28, 56]  # days
    specimens_per_condition = 3
    
    ambient_data = []
    
    for mix in mix_ids:
        # Base properties decrease with rubber content
        rubber_index = mix_ids.index(mix)
        base_fc_28 = 65 - (rubber_index * 8)  # Control=65 MPa, dropping by 8 MPa per step
        base_ft_28 = base_fc_28 * 0.1  # Tensile strength ~10% of compressive
        base_E_28 = 35000 - (rubber_index * 3000)  # Modulus of Elasticity in MPa
        base_UPV = 4500 - (rubber_index * 150)  # UPV in m/s
        base_density = 2400 - (rubber_index * 40)  # kg/m³
        
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
                density = base_density * np.random.normal(1, 0.01)
                
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
    print(f"Generated {len(df_ambient)} ambient condition specimens")
    return df_ambient

# ============================================================================
# PART B: HIGH-TEMPERATURE RESIDUAL PROPERTIES
# ============================================================================

def generate_residual_data(df_ambient):
    """Generate residual properties after high-temperature exposure"""
    print("Generating Part B: High-Temperature Residual Properties...")
    
    # Expanded experimental matrix
    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    peak_temps = [23, 200, 400, 600, 800]  # °C
    cooling_methods = ['Furnace', 'Quench']
    heating_rates = ['5_C_per_min']  # Standard heating
    spalling_mixes = ['C', 'R15S', 'R20S']  # Add rapid heating for spalling study
    specimens_per_condition = 3
    
    residual_data = []
    
    for mix in mix_ids:
        # Get the 28-day ambient strength for this mix to base degradation on
        ambient_fc = df_ambient[(df_ambient['Mix_ID'] == mix) & 
                               (df_ambient['Curing_Age_days'] == 28) & 
                               (df_ambient['Test_Type'] == 'Ambient')]['Compressive_Strength_MPa'].mean()
        
        for temp in peak_temps:
            for cooling in cooling_methods:
                hr_list = heating_rates[:]
                if mix in spalling_mixes:
                    hr_list.append('10_C_per_min')  # Add rapid heating for spalling study
                
                for hr in hr_list:
                    for spec_num in range(1, specimens_per_condition + 1):
                        specimen_id = f"{mix}-28-R-{temp}-{cooling[0]}-{hr.split('_')[0]}-{spec_num}"
                        
                        # Calculate degradation factors
                        mass_loss = calculate_mass_loss(temp, mix, mix_ids)
                        strength_retention = calculate_strength_retention(temp, mix, cooling, hr, mix_ids)
                        UPV_retention = strength_retention ** 0.5  # Rough correlation
                        
                        # Fabricate data points with noise
                        fc_residual = (ambient_fc * strength_retention) * np.random.normal(1, 0.08)
                        mass_loss_pct = mass_loss * np.random.normal(1, 0.1)
                        UPV_residual = (4500 * UPV_retention) * np.random.normal(1, 0.05)
                        
                        # Spalling assessment
                        spalling_occurred, spalling_depth = assess_spalling(mix, temp, hr, mix_ids)
                        
                        # Visual damage rating
                        visual_rating = assess_visual_damage(temp, strength_retention)
                        
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
                            'Spalling_Depth_mm': spalling_depth,
                            'Visual_Cracking_Rating': visual_rating
                        })
    
    df_residual = pd.DataFrame(residual_data)
    print(f"Generated {len(df_residual)} residual property specimens")
    return df_residual

def calculate_mass_loss(temp, mix, mix_ids):
    """Calculate mass loss based on temperature and mix composition"""
    base_mass_loss = 0.0
    if temp >= 105:
        base_mass_loss = (temp / 1000) * 8  # Basic dehydration
        if mix != 'C':
            # Rubber burns off at higher temperatures
            rubber_factor = mix_ids.index(mix) * 0.5
            base_mass_loss += rubber_factor * (temp / 800)
    return base_mass_loss

def calculate_strength_retention(temp, mix, cooling, hr, mix_ids):
    """Calculate strength retention based on multiple factors"""
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
    
    # Rapid heating increases strength loss for non-rubber mixes (spalling risk)
    if hr == '10_C_per_min' and mix == 'C' and temp >= 400:
        strength_retention *= 0.7  # Spalling reduces strength drastically
    
    return max(0, strength_retention)

def assess_spalling(mix, temp, hr, mix_ids):
    """Assess spalling occurrence and depth"""
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
    
    return spalling_occurred, spalling_depth_mm

def assess_visual_damage(temp, strength_retention):
    """Assess visual damage rating based on temperature and strength loss"""
    if temp <= 200:
        return np.random.choice(['None', 'Minor'], p=[0.8, 0.2])
    elif temp <= 400:
        return np.random.choice(['None', 'Minor', 'Moderate'], p=[0.3, 0.5, 0.2])
    elif temp <= 600:
        return np.random.choice(['Minor', 'Moderate', 'Severe'], p=[0.2, 0.5, 0.3])
    else:
        return np.random.choice(['Moderate', 'Severe'], p=[0.3, 0.7])

# ============================================================================
# PART C: IN-SITU HIGH-TEMPERATURE TESTS
# ============================================================================

def generate_in_situ_data(df_ambient):
    """Generate in-situ high-temperature test data with stress-strain curves"""
    print("Generating Part C: In-Situ High-Temperature Tests...")
    
    # In-situ tests are complex, so we focus on key mixes and temperatures
    key_mixes = ['C', 'R10S', 'R20S']  # Control, medium rubber, high rubber
    in_situ_temps = [23, 200, 400, 600]  # 800°C is often too difficult to test in-situ
    specimens_per_condition = 2
    
    in_situ_data = []
    stress_strain_curves = {}
    
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
                mix_index = ['C', 'R10S', 'R20S'].index(mix)
                peak_strain_factor = 1.0 + (mix_index * 0.2)  # More rubber = more strain capacity
                
                fc_in_situ = ambient_fc * strength_retention * np.random.normal(1, 0.07)
                E_in_situ = ambient_E * E_retention * np.random.normal(1, 0.06)
                
                # Generate synthetic stress-strain curve
                strain_points = np.linspace(0, 0.025, 50)
                peak_strain = 0.002 * peak_strain_factor * (1 + 0.005 * temp)
                stress_points = []
                
                for strain in strain_points:
                    if strain <= peak_strain:
                        # Parabolic ascending branch
                        stress = fc_in_situ * (2 * (strain/peak_strain) - (strain/peak_strain)**2)
                    else:
                        # Linear descending branch (more gradual for rubber mixes)
                        descent_factor = 0.3 if mix == 'C' else 0.15
                        stress = fc_in_situ * max(0, 1 - descent_factor * (strain - peak_strain) / peak_strain)
                    stress_points.append(max(0, stress * np.random.normal(1, 0.02)))
                
                stress_strain_curves[specimen_id] = {
                    'strain': strain_points.tolist(),
                    'stress': stress_points
                }
                
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
    print(f"Generated {len(df_in_situ)} in-situ test specimens")
    return df_in_situ, stress_strain_curves

# ============================================================================
# PART D: ADVANCED MEASUREMENTS - PORE PRESSURE AND TRANSIENT STRAIN
# ============================================================================

def generate_pore_pressure_data():
    """Generate pore pressure data for spalling investigation"""
    print("Generating Part D: Advanced Measurements - Pore Pressure...")
    
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
                    'Time_min': times.tolist(),
                    'Pore_Pressure_MPa': pressures
                })
    
    # Create summary DataFrame for peak pressures
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
    print(f"Generated pore pressure data for {len(df_pore_pressure)} configurations")
    return df_pore_pressure, pore_pressure_data

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(df_ambient, df_residual, df_in_situ, df_pore_pressure, 
                         stress_strain_curves, pore_pressure_data):
    """Create comprehensive analysis plots and visualizations"""
    print("Creating comprehensive visualizations...")
    
    # Set up the plotting environment
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Stress-strain curves for in-situ tests
    create_stress_strain_plot(stress_strain_curves)
    
    # Plot 2: Pore pressure evolution
    create_pore_pressure_plot(pore_pressure_data)
    
    # Plot 3: Comprehensive analysis (4 subplots)
    create_comprehensive_analysis_plot(df_residual, df_pore_pressure)
    
    # Plot 4: Temperature degradation trends
    create_temperature_degradation_plot(df_ambient, df_residual)
    
    print("All visualizations created and saved to 'plots/' directory")

def create_stress_strain_plot(stress_strain_curves):
    """Create stress-strain curves plot"""
    plt.figure(figsize=(10, 6))
    
    key_mixes = ['C', 'R10S', 'R20S']
    colors = ['red', 'blue', 'green']
    
    for i, mix in enumerate(key_mixes):
        spec_id = f"{mix}-28-IS-400-1"
        if spec_id in stress_strain_curves:
            strain_data = np.array(stress_strain_curves[spec_id]['strain']) * 100
            stress_data = stress_strain_curves[spec_id]['stress']
            plt.plot(strain_data, stress_data, 
                    label=f'{mix}, 400°C', linewidth=2, color=colors[i])
    
    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (MPa)')
    plt.title('In-Situ Compressive Stress-Strain Behavior at High Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/in_situ_stress_strain_curves.png', bbox_inches='tight')
    plt.close()

def create_pore_pressure_plot(pore_pressure_data):
    """Create pore pressure evolution plot"""
    plt.figure(figsize=(12, 6))
    
    colors = {'C': 'red', 'R20S': 'blue'}
    linestyles = {10: '-', 25: '--', 40: ':'}
    
    for pp in pore_pressure_data:
        if pp['Run'] == 1:  # Plot only first run for clarity
            mix = pp['Mix_ID']
            depth = pp['Depth_mm']
            plt.plot(pp['Time_min'], pp['Pore_Pressure_MPa'], 
                    color=colors[mix], linestyle=linestyles[depth],
                    label=f'{mix}, {depth}mm', linewidth=2)
    
    plt.xlabel('Time (min)')
    plt.ylabel('Pore Pressure (MPa)')
    plt.title('Pore Pressure Evolution During Heating\n(10°C/min heating rate)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/pore_pressure_evolution.png', bbox_inches='tight')
    plt.close()

def create_comprehensive_analysis_plot(df_residual, df_pore_pressure):
    """Create comprehensive analysis with 4 subplots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Residual strength vs temperature
    key_mixes = ['C', 'R10S', 'R20S']
    colors = ['red', 'blue', 'green']
    
    for i, mix in enumerate(key_mixes):
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        strength_by_temp = mix_data.groupby('Peak_Temperature_C')['Residual_Compressive_Strength_MPa'].mean()
        axes[0,0].plot(strength_by_temp.index, strength_by_temp.values, 
                      'o-', label=mix, linewidth=2, markersize=6, color=colors[i])
    
    axes[0,0].set_xlabel('Peak Temperature (°C)')
    axes[0,0].set_ylabel('Residual Compressive Strength (MPa)')
    axes[0,0].set_title('A. Strength Degradation with Temperature\n(Furnace Cooled)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Effect of cooling method
    temp = 400
    cooling_data = []
    for mix in ['C', 'R20S']:
        for cooling in ['Furnace', 'Quench']:
            cool_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                                   (df_residual['Peak_Temperature_C'] == temp) &
                                   (df_residual['Heating_Rate'] == '5_C_per_min') &
                                   (df_residual['Cooling_Method'] == cooling)]
            if not cool_data.empty:
                strength = cool_data['Residual_Compressive_Strength_MPa'].mean()
                cooling_data.append({'Mix': mix, 'Cooling': cooling, 'Strength': strength})
    
    if cooling_data:
        cooling_df = pd.DataFrame(cooling_data)
        x_pos = np.arange(len(cooling_data))
        bars = axes[0,1].bar(x_pos, cooling_df['Strength'], 
                            color=['blue' if 'Furnace' in str(row['Cooling']) else 'red' 
                                  for _, row in cooling_df.iterrows()], alpha=0.7)
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels([f"{row['Mix']}\n{row['Cooling']}" for _, row in cooling_df.iterrows()])
    
    axes[0,1].set_ylabel('Residual Compressive Strength (MPa)')
    axes[0,1].set_title(f'B. Effect of Cooling Method at {temp}°C\n(Thermal Shock Damage)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Mass loss correlation
    for i, mix in enumerate(key_mixes):
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        axes[1,0].scatter(mix_data['Mass_Loss_pct'], mix_data['Residual_Compressive_Strength_MPa'], 
                         label=mix, alpha=0.6, s=50, color=colors[i])
    
    axes[1,0].set_xlabel('Mass Loss (%)')
    axes[1,0].set_ylabel('Residual Compressive Strength (MPa)')
    axes[1,0].set_title('C. Strength vs. Mass Loss Correlation')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Pore pressure summary
    pore_summary = df_pore_pressure.groupby('Mix_ID')['Peak_Pressure_MPa'].mean()
    bars = axes[1,1].bar(pore_summary.index, pore_summary.values, 
                        color=['red', 'blue'], alpha=0.7)
    axes[1,1].set_xlabel('Mix Type')
    axes[1,1].set_ylabel('Average Peak Pore Pressure (MPa)')
    axes[1,1].set_title('D. Peak Pore Pressure Comparison')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/comprehensive_high_temperature_analysis.png', bbox_inches='tight')
    plt.close()

def create_temperature_degradation_plot(df_ambient, df_residual):
    """Create temperature degradation trends plot"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get ambient baseline for normalization
    ambient_baseline = df_ambient[df_ambient['Curing_Age_days'] == 28].groupby('Mix_ID').agg({
        'Compressive_Strength_MPa': 'mean',
        'UPV_mps': 'mean'
    }).reset_index()
    
    # Plot normalized strength retention
    key_mixes = ['C', 'R10S', 'R20S']
    colors = ['red', 'blue', 'green']
    
    for i, mix in enumerate(key_mixes):
        baseline_strength = ambient_baseline[ambient_baseline['Mix_ID'] == mix]['Compressive_Strength_MPa'].iloc[0]
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        
        temp_strength = mix_data.groupby('Peak_Temperature_C')['Residual_Compressive_Strength_MPa'].mean()
        retention_ratio = temp_strength / baseline_strength
        
        axes[0,0].plot(temp_strength.index, retention_ratio, 'o-', 
                      label=mix, linewidth=2, markersize=6, color=colors[i])
    
    axes[0,0].set_xlabel('Peak Temperature (°C)')
    axes[0,0].set_ylabel('Strength Retention Ratio')
    axes[0,0].set_title('Normalized Strength Retention vs Temperature')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1.2)
    
    # Plot mass loss trends
    for i, mix in enumerate(key_mixes):
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        mass_loss_by_temp = mix_data.groupby('Peak_Temperature_C')['Mass_Loss_pct'].mean()
        axes[0,1].plot(mass_loss_by_temp.index, mass_loss_by_temp.values, 'o-', 
                      label=mix, linewidth=2, markersize=6, color=colors[i])
    
    axes[0,1].set_xlabel('Peak Temperature (°C)')
    axes[0,1].set_ylabel('Mass Loss (%)')
    axes[0,1].set_title('Mass Loss vs Temperature')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot UPV degradation
    for i, mix in enumerate(key_mixes):
        baseline_UPV = ambient_baseline[ambient_baseline['Mix_ID'] == mix]['UPV_mps'].iloc[0]
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        
        temp_UPV = mix_data.groupby('Peak_Temperature_C')['UPV_mps'].mean()
        UPV_retention = temp_UPV / baseline_UPV
        
        axes[1,0].plot(temp_UPV.index, UPV_retention, 'o-', 
                      label=mix, linewidth=2, markersize=6, color=colors[i])
    
    axes[1,0].set_xlabel('Peak Temperature (°C)')
    axes[1,0].set_ylabel('UPV Retention Ratio')
    axes[1,0].set_title('UPV Degradation vs Temperature')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(0, 1.2)
    
    # Plot spalling occurrence
    spalling_data = df_residual[df_residual['Heating_Rate'] == '10_C_per_min']
    if not spalling_data.empty:
        spalling_rates = spalling_data.groupby(['Mix_ID', 'Peak_Temperature_C'])['Spalling_Occurred'].mean().reset_index()
        
        for i, mix in enumerate(['C', 'R15S', 'R20S']):  # Spalling study mixes
            if mix in spalling_rates['Mix_ID'].values:
                mix_spalling = spalling_rates[spalling_rates['Mix_ID'] == mix]
                axes[1,1].plot(mix_spalling['Peak_Temperature_C'], mix_spalling['Spalling_Occurred'], 
                              'o-', label=mix, linewidth=2, markersize=6)
    
    axes[1,1].set_xlabel('Peak Temperature (°C)')
    axes[1,1].set_ylabel('Probability of Spalling')
    axes[1,1].set_title('Spalling Risk: Rapid Heating (10°C/min)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('plots/temperature_degradation_trends.png', bbox_inches='tight')
    plt.close()

# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def export_datasets(df_ambient, df_residual, df_in_situ, df_pore_pressure, 
                   stress_strain_curves, pore_pressure_data):
    """Export all datasets to CSV and JSON files"""
    print("Exporting datasets to files...")
    
    # Create output directory
    os.makedirs('datasets', exist_ok=True)
    
    # Export CSV files
    df_ambient.to_csv('datasets/ambient_properties.csv', index=False)
    df_residual.to_csv('datasets/residual_properties_high_temp.csv', index=False)
    df_in_situ.to_csv('datasets/in_situ_properties.csv', index=False)
    df_pore_pressure.to_csv('datasets/pore_pressure_summary.csv', index=False)
    
    # Export stress-strain curves as JSON
    with open('datasets/stress_strain_curves.json', 'w') as f:
        json.dump(stress_strain_curves, f, indent=2)
    
    # Export full pore pressure time series as JSON
    with open('datasets/pore_pressure_time_series.json', 'w') as f:
        json.dump(pore_pressure_data, f, indent=2)
    
    # Create a comprehensive metadata file
    metadata = {
        "dataset_info": {
            "title": "Synthetic Dataset for Fire-Resistant Structural Elements Research",
            "subtitle": "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete",
            "generation_date": datetime.now().isoformat(),
            "random_seed": 42
        },
        "mix_compositions": {
            "C": "Control concrete (no rubber)",
            "R5S": "5% small rubber particles",
            "R10S": "10% small rubber particles", 
            "R15S": "15% small rubber particles",
            "R20S": "20% small rubber particles",
            "R10L": "10% large rubber particles"
        },
        "test_conditions": {
            "curing_ages": [7, 28, 56],
            "peak_temperatures": [23, 200, 400, 600, 800],
            "heating_rates": ["5_C_per_min", "10_C_per_min"],
            "cooling_methods": ["Furnace", "Quench"],
            "test_types": ["Ambient", "Residual", "In-Situ"]
        },
        "dataset_statistics": {
            "ambient_specimens": len(df_ambient),
            "residual_specimens": len(df_residual),
            "in_situ_specimens": len(df_in_situ),
            "pore_pressure_configurations": len(df_pore_pressure),
            "stress_strain_curves": len(stress_strain_curves)
        }
    }
    
    with open('datasets/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Exported {len(df_ambient)} ambient specimens to ambient_properties.csv")
    print(f"Exported {len(df_residual)} residual specimens to residual_properties_high_temp.csv")
    print(f"Exported {len(df_in_situ)} in-situ specimens to in_situ_properties.csv")
    print(f"Exported {len(df_pore_pressure)} pore pressure configs to pore_pressure_summary.csv")
    print("Exported stress-strain curves and metadata to JSON files")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("Starting comprehensive dataset generation...")
    
    # Generate all datasets
    df_ambient = generate_ambient_data()
    df_residual = generate_residual_data(df_ambient)
    df_in_situ, stress_strain_curves = generate_in_situ_data(df_ambient)
    df_pore_pressure, pore_pressure_data = generate_pore_pressure_data()
    
    # Create visualizations
    create_visualizations(df_ambient, df_residual, df_in_situ, df_pore_pressure, 
                         stress_strain_curves, pore_pressure_data)
    
    # Export datasets
    export_datasets(df_ambient, df_residual, df_in_situ, df_pore_pressure, 
                   stress_strain_curves, pore_pressure_data)
    
    # Print summary
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"Ambient tests: {len(df_ambient)} specimens")
    print(f"Residual high-temperature tests: {len(df_residual)} specimens")
    print(f"In-situ high-temperature tests: {len(df_in_situ)} specimens")
    print(f"Pore pressure experiments: {len(df_pore_pressure)} configurations")
    print(f"Stress-strain curves: {len(stress_strain_curves)} curves")
    print()
    print("Files generated:")
    print("- datasets/ambient_properties.csv")
    print("- datasets/residual_properties_high_temp.csv")
    print("- datasets/in_situ_properties.csv")
    print("- datasets/pore_pressure_summary.csv")
    print("- datasets/stress_strain_curves.json")
    print("- datasets/pore_pressure_time_series.json")
    print("- datasets/dataset_metadata.json")
    print("- plots/in_situ_stress_strain_curves.png")
    print("- plots/pore_pressure_evolution.png")
    print("- plots/comprehensive_high_temperature_analysis.png")
    print("- plots/temperature_degradation_trends.png")
    print()
    print("This comprehensive dataset is now ready for thermo-mechanical model development and validation.")
    print("="*80)

if __name__ == "__main__":
    main()