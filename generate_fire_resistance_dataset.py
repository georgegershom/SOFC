#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Fire-Resistant Rubberized Concrete Research
Title: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant 
       Structural Elements Utilizing High-Performance Rubberized Concrete

This script generates a comprehensive synthetic dataset capturing complex thermo-mechanical 
degradation pathways including effects of heating rate, peak temperature, cooling regime, 
and rubber content/size.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for results
output_dir = "fire_resistance_dataset"
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("FIRE-RESISTANT RUBBERIZED CONCRETE SYNTHETIC DATASET GENERATOR")
print("="*80)
print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {output_dir}")
print("="*80)

# =============================================================================
# PART A: AMBIENT CONDITION TESTS
# =============================================================================

def generate_ambient_data():
    """Generate ambient condition test data for 6 mixes x 3 ages x 3 specimens."""
    print("\n[Phase A] Generating Ambient Condition Test Data...")
    
    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    ages = [7, 28, 56]
    specimens_per_condition = 3
    
    ambient_data = []
    
    for mix in mix_ids:
        # Base properties decrease with rubber content
        mix_index = mix_ids.index(mix)
        base_fc_28 = 65 - (mix_index * 8)  # Control=65 MPa, dropping by 8 MPa per step
        base_ft_28 = base_fc_28 * 0.1  # Tensile strength ~10% of compressive
        base_E_28 = 35000 - (mix_index * 3000)  # Modulus of Elasticity in MPa
        base_UPV = 4500 - (mix_index * 150)  # UPV in m/s
        base_density = 2400 - (mix_index * 40)  # kg/m³
        
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
                
                # Additional properties
                flexural_strength = ft * 1.5 * np.random.normal(1, 0.05)
                porosity = 12 + (mix_index * 1.5) * np.random.normal(1, 0.1)
                
                ambient_data.append({
                    'Specimen_ID': specimen_id,
                    'Mix_ID': mix,
                    'Curing_Age_days': age,
                    'Test_Type': 'Ambient',
                    'Test_Date': f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
                    'Compressive_Strength_MPa': max(0, fc),
                    'Tensile_Strength_MPa': max(0, ft),
                    'Flexural_Strength_MPa': max(0, flexural_strength),
                    'Modulus_of_Elasticity_MPa': max(0, E),
                    'Dry_Density_kgm3': density,
                    'UPV_mps': UPV,
                    'Porosity_pct': porosity,
                    'Moisture_Content_pct': np.random.uniform(2.5, 4.5)
                })
    
    df_ambient = pd.DataFrame(ambient_data)
    print(f"  Generated {len(df_ambient)} ambient test records")
    return df_ambient

# =============================================================================
# PART B: HIGH-TEMPERATURE RESIDUAL PROPERTIES
# =============================================================================

def generate_residual_data(df_ambient):
    """Generate high-temperature residual property data with complex degradation patterns."""
    print("\n[Phase B] Generating High-Temperature Residual Properties Data...")
    
    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    peak_temps = [23, 200, 400, 600, 800]
    cooling_methods = ['Furnace', 'Quench']
    heating_rates = ['5_C_per_min']
    spalling_mixes = ['C', 'R15S', 'R20S']
    specimens_per_condition = 3
    
    residual_data = []
    
    for mix in mix_ids:
        # Get the 28-day ambient strength for this mix
        ambient_fc = df_ambient[(df_ambient['Mix_ID'] == mix) & 
                               (df_ambient['Curing_Age_days'] == 28)]['Compressive_Strength_MPa'].mean()
        ambient_E = df_ambient[(df_ambient['Mix_ID'] == mix) & 
                              (df_ambient['Curing_Age_days'] == 28)]['Modulus_of_Elasticity_MPa'].mean()
        
        for temp in peak_temps:
            for cooling in cooling_methods:
                hr_list = heating_rates.copy()
                if mix in spalling_mixes and temp >= 400:
                    hr_list.append('10_C_per_min')  # Add rapid heating for spalling study
                
                for hr in hr_list:
                    for spec_num in range(1, specimens_per_condition + 1):
                        specimen_id = f"{mix}-28-R-{temp}-{cooling[0]}-{hr.split('_')[0]}-{spec_num}"
                        
                        # Calculate complex degradation factors
                        mass_loss, strength_retention, E_retention = calculate_degradation(
                            mix, temp, cooling, hr
                        )
                        
                        # Apply stochastic variation
                        fc_residual = ambient_fc * strength_retention * np.random.normal(1, 0.08)
                        E_residual = ambient_E * E_retention * np.random.normal(1, 0.06)
                        mass_loss_actual = mass_loss * np.random.normal(1, 0.1)
                        
                        # UPV correlates with damage
                        UPV_retention = (strength_retention * E_retention) ** 0.5
                        UPV_residual = 4500 * UPV_retention * np.random.normal(1, 0.05)
                        
                        # Spalling assessment
                        spalling_occurred, spalling_depth = assess_spalling(mix, temp, hr)
                        
                        # Visual damage assessment
                        visual_rating = assess_visual_damage(temp, strength_retention)
                        
                        # Color change
                        color_change = get_color_change(temp, mix)
                        
                        residual_data.append({
                            'Specimen_ID': specimen_id,
                            'Mix_ID': mix,
                            'Peak_Temperature_C': temp,
                            'Heating_Rate': hr,
                            'Cooling_Method': cooling,
                            'Test_Type': 'Residual',
                            'Exposure_Duration_hours': 2 if temp <= 400 else 3,
                            'Mass_Loss_pct': max(0, mass_loss_actual),
                            'UPV_mps': max(500, UPV_residual),
                            'Residual_Compressive_Strength_MPa': max(0, fc_residual),
                            'Residual_Elastic_Modulus_MPa': max(0, E_residual),
                            'Strength_Retention_pct': max(0, strength_retention * 100),
                            'Spalling_Occurred': spalling_occurred,
                            'Spalling_Depth_mm': spalling_depth,
                            'Visual_Cracking_Rating': visual_rating,
                            'Color_Change': color_change,
                            'Test_Date': f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"
                        })
    
    df_residual = pd.DataFrame(residual_data)
    print(f"  Generated {len(df_residual)} residual property test records")
    return df_residual

def calculate_degradation(mix, temp, cooling, heating_rate):
    """Calculate complex degradation factors based on temperature and mix properties."""
    mix_index = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L'].index(mix)
    
    # Mass loss calculation
    mass_loss = 0.0
    if temp >= 105:
        # Basic dehydration
        mass_loss = (temp / 1000) * 8
        # Rubber decomposition
        if mix != 'C':
            rubber_loss = (mix_index * 0.5) * (temp / 800)
            if temp >= 200:  # Rubber starts degrading
                rubber_loss *= 1.5
            if temp >= 400:  # Rubber combustion
                rubber_loss *= 2.0
            mass_loss += rubber_loss
    
    # Strength retention - complex function
    if temp <= 100:
        strength_retention = 1.0 + np.random.uniform(-0.02, 0.02)
    elif temp <= 200:
        # Slight increase possible due to further hydration
        strength_retention = 1.0 + 0.05 * (1 - temp/200)
    elif temp <= 400:
        # Transition zone - rubber softening
        if mix == 'C':
            strength_retention = 0.85 - (temp - 200) * 0.0015
        else:
            # Rubber provides some protection
            strength_retention = 0.75 - (temp - 200) * 0.0012
    elif temp <= 600:
        # Significant degradation
        if mix == 'C':
            strength_retention = 0.5 - (temp - 400) * 0.002
        else:
            # Pores from burnt rubber may reduce spalling
            strength_retention = 0.45 - (temp - 400) * 0.0018
    else:  # 800°C
        strength_retention = 0.15 - mix_index * 0.01
    
    # Elastic modulus degrades faster than strength
    E_retention = strength_retention ** 1.3
    
    # Cooling effects
    if cooling == 'Quench' and temp > 105:
        strength_retention *= 0.85  # Thermal shock
        E_retention *= 0.80
        mass_loss *= 1.1  # Additional micro-cracking
    
    # Rapid heating effects
    if heating_rate == '10_C_per_min' and temp >= 400:
        if mix == 'C':
            strength_retention *= 0.7  # Spalling damage
            mass_loss *= 1.3
        else:
            strength_retention *= 0.85  # Less spalling in rubber mixes
            mass_loss *= 1.15
    
    return mass_loss, strength_retention, E_retention

def assess_spalling(mix, temp, heating_rate):
    """Assess spalling occurrence and depth based on conditions."""
    spalling_occurred = False
    spalling_depth = 0.0
    
    if temp >= 400 and heating_rate == '10_C_per_min':
        if mix == 'C':
            # High spalling risk for control mix
            if np.random.random() > 0.2:  # 80% chance
                spalling_occurred = True
                spalling_depth = np.random.uniform(10, 30)
        elif mix in ['R15S', 'R20S']:
            # Reduced spalling risk with rubber
            if np.random.random() > 0.7:  # 30% chance
                spalling_occurred = True
                spalling_depth = np.random.uniform(3, 12)
        else:
            # Low rubber content - intermediate risk
            if np.random.random() > 0.5:  # 50% chance
                spalling_occurred = True
                spalling_depth = np.random.uniform(5, 15)
    
    return spalling_occurred, spalling_depth

def assess_visual_damage(temp, strength_retention):
    """Assess visual damage rating based on temperature and strength loss."""
    if temp <= 200:
        return np.random.choice(['None', 'Minor'], p=[0.7, 0.3])
    elif temp <= 400:
        if strength_retention > 0.6:
            return np.random.choice(['Minor', 'Moderate'], p=[0.6, 0.4])
        else:
            return np.random.choice(['Moderate', 'Severe'], p=[0.7, 0.3])
    elif temp <= 600:
        return np.random.choice(['Moderate', 'Severe', 'Critical'], p=[0.2, 0.5, 0.3])
    else:
        return np.random.choice(['Severe', 'Critical'], p=[0.3, 0.7])

def get_color_change(temp, mix):
    """Determine color change based on temperature exposure."""
    if temp <= 200:
        return "No change"
    elif temp <= 400:
        return "Light gray to pink"
    elif temp <= 600:
        return "Pink to light red"
    else:
        return "Light red to buff/yellow"

# =============================================================================
# PART C: IN-SITU HIGH-TEMPERATURE TESTS
# =============================================================================

def generate_in_situ_data(df_ambient):
    """Generate in-situ high-temperature test data with stress-strain curves."""
    print("\n[Phase C] Generating In-Situ High-Temperature Test Data...")
    
    key_mixes = ['C', 'R10S', 'R20S']
    in_situ_temps = [23, 200, 400, 600]
    specimens_per_condition = 2
    
    in_situ_data = []
    stress_strain_curves = {}
    
    for mix in key_mixes:
        ambient_fc = df_ambient[(df_ambient['Mix_ID'] == mix) & 
                               (df_ambient['Curing_Age_days'] == 28)]['Compressive_Strength_MPa'].mean()
        ambient_E = df_ambient[(df_ambient['Mix_ID'] == mix) & 
                              (df_ambient['Curing_Age_days'] == 28)]['Modulus_of_Elasticity_MPa'].mean()
        
        for temp in in_situ_temps:
            for spec_num in range(1, specimens_per_condition + 1):
                specimen_id = f"{mix}-28-IS-{temp}-{spec_num}"
                
                # In-situ strength and stiffness retention
                strength_retention, E_retention, ductility_factor = get_in_situ_properties(mix, temp)
                
                fc_in_situ = ambient_fc * strength_retention * np.random.normal(1, 0.07)
                E_in_situ = ambient_E * E_retention * np.random.normal(1, 0.06)
                
                # Generate stress-strain curve
                strain_points, stress_points = generate_stress_strain_curve(
                    fc_in_situ, E_in_situ, temp, ductility_factor
                )
                
                stress_strain_curves[specimen_id] = {
                    'strain': strain_points.tolist(),
                    'stress': stress_points.tolist()
                }
                
                # Calculate additional properties
                peak_strain = strain_points[np.argmax(stress_points)]
                ultimate_strain = strain_points[-1]
                toughness = np.trapezoid(stress_points, strain_points)  # Area under curve
                
                in_situ_data.append({
                    'Specimen_ID': specimen_id,
                    'Mix_ID': mix,
                    'Test_Temperature_C': temp,
                    'Test_Type': 'In-Situ',
                    'InSitu_Compressive_Strength_MPa': max(0, fc_in_situ),
                    'InSitu_Modulus_of_Elasticity_MPa': max(0, E_in_situ),
                    'Peak_Strain': peak_strain,
                    'Ultimate_Strain': ultimate_strain,
                    'Toughness_MPa': toughness,
                    'Poissons_Ratio': max(0.1, 0.2 - (temp * 0.0002)),
                    'Thermal_Strain_pct': calculate_thermal_strain(temp, mix),
                    'Test_Duration_min': np.random.uniform(15, 25)
                })
    
    df_in_situ = pd.DataFrame(in_situ_data)
    print(f"  Generated {len(df_in_situ)} in-situ test records")
    print(f"  Generated {len(stress_strain_curves)} stress-strain curves")
    return df_in_situ, stress_strain_curves

def get_in_situ_properties(mix, temp):
    """Calculate in-situ properties at elevated temperature."""
    # Base retention factors
    if temp <= 100:
        strength_retention = 1.0
        E_retention = 0.95
    elif temp <= 200:
        strength_retention = 0.95
        E_retention = 0.85
    elif temp <= 400:
        strength_retention = 0.75
        E_retention = 0.55
    else:  # 600°C
        strength_retention = 0.35
        E_retention = 0.20
    
    # Rubber effect on ductility
    mix_index = ['C', 'R10S', 'R20S'].index(mix)
    ductility_factor = 1.0 + (mix_index * 0.25)  # More rubber = more ductile
    
    # Rubber provides slight strength benefit at high temp
    if mix != 'C' and temp >= 400:
        strength_retention *= 1.1
    
    return strength_retention, E_retention, ductility_factor

def generate_stress_strain_curve(fc, E, temp, ductility_factor):
    """Generate realistic stress-strain curve for given conditions."""
    # Strain points (more points for smoother curve)
    max_strain = 0.035 * (1 + temp/1000) * ductility_factor
    strain_points = np.linspace(0, max_strain, 100)
    
    # Peak strain increases with temperature and rubber content
    peak_strain = 0.002 * (1 + temp/500) * ductility_factor
    
    # Generate stress points using Popovics model
    stress_points = []
    n = E * peak_strain / fc  # Shape parameter
    
    for strain in strain_points:
        if strain == 0:
            stress = 0
        elif strain <= peak_strain:
            # Ascending branch - Popovics equation
            r = strain / peak_strain
            stress = fc * r * n / (n - 1 + r**n)
        else:
            # Descending branch - modified exponential
            alpha = 0.15 / ductility_factor  # Descent rate
            stress = fc * np.exp(-alpha * (strain - peak_strain) / peak_strain)
        
        # Add realistic noise
        stress = stress * np.random.normal(1, 0.02)
        stress_points.append(max(0, stress))
    
    return strain_points, np.array(stress_points)

def calculate_thermal_strain(temp, mix):
    """Calculate thermal strain based on temperature and mix."""
    # Coefficient of thermal expansion (CTE) in microstrain/°C
    base_CTE = 10.0  # Typical for concrete
    
    # Rubber slightly increases CTE
    mix_index = ['C', 'R10S', 'R20S'].index(mix) if mix in ['C', 'R10S', 'R20S'] else 0
    CTE = base_CTE * (1 + mix_index * 0.1)
    
    # Thermal strain calculation
    thermal_strain = CTE * (temp - 20) * 1e-6 * 100  # Convert to percentage
    
    return thermal_strain * np.random.normal(1, 0.1)

# =============================================================================
# PART D: ADVANCED MEASUREMENTS - PORE PRESSURE AND TRANSIENT STRAIN
# =============================================================================

def generate_advanced_measurements():
    """Generate pore pressure and transient strain data."""
    print("\n[Phase D] Generating Advanced Measurements Data...")
    
    # Pore pressure measurements
    pore_pressure_data = []
    transient_strain_data = []
    
    # Time points for measurements (minutes)
    times = np.linspace(0, 120, 121)  # 2-hour heating
    
    # Pore pressure experiments
    for mix in ['C', 'R10S', 'R20S']:
        for depth in [10, 25, 40]:  # mm from surface
            for run in [1, 2]:
                pressures = []
                temperatures = []
                
                for t in times:
                    # Temperature at depth (simplified heat transfer)
                    surface_temp = min(800, t * 10)  # 10°C/min heating
                    depth_factor = np.exp(-depth / 50)
                    temp_at_depth = 20 + (surface_temp - 20) * depth_factor
                    temperatures.append(temp_at_depth)
                    
                    # Pore pressure calculation
                    pressure = calculate_pore_pressure(mix, temp_at_depth, depth, t)
                    pressures.append(max(0, pressure))
                
                # Find peak values
                peak_pressure = max(pressures)
                peak_time = times[np.argmax(pressures)]
                peak_temp = temperatures[np.argmax(pressures)]
                
                pore_pressure_data.append({
                    'Mix_ID': mix,
                    'Depth_mm': depth,
                    'Run': run,
                    'Peak_Pressure_MPa': peak_pressure,
                    'Time_of_Peak_min': peak_time,
                    'Temp_at_Peak_C': peak_temp,
                    'Time_min': times,  # Add this for JSON export
                    'Pressure_Profile': pressures  # Full time series
                })
    
    # Transient strain experiments
    for mix in ['C', 'R20S']:
        for load_level in [0.2, 0.4]:  # Fraction of ambient strength
            for temp_rate in [2, 5]:  # °C/min
                max_temp = 600
                time_points = np.linspace(0, max_temp/temp_rate, 100)
                temperatures = np.minimum(time_points * temp_rate, max_temp)
                
                strains = []
                for i, temp in enumerate(temperatures):
                    strain = calculate_transient_strain(mix, temp, load_level, time_points[i])
                    strains.append(strain)
                
                transient_strain_data.append({
                    'Mix_ID': mix,
                    'Load_Level': load_level,
                    'Heating_Rate_C_per_min': temp_rate,
                    'Max_Temperature_C': max_temp,
                    'Total_Transient_Strain_pct': max(strains),
                    'Strain_Profile': strains  # Full profile
                })
    
    df_pore_pressure = pd.DataFrame([{k: v for k, v in d.items() if k != 'Pressure_Profile'} 
                                    for d in pore_pressure_data])
    df_transient_strain = pd.DataFrame([{k: v for k, v in d.items() if k != 'Strain_Profile'} 
                                       for d in transient_strain_data])
    
    print(f"  Generated {len(df_pore_pressure)} pore pressure measurements")
    print(f"  Generated {len(df_transient_strain)} transient strain measurements")
    
    return df_pore_pressure, df_transient_strain, pore_pressure_data, transient_strain_data

def calculate_pore_pressure(mix, temp, depth, time):
    """Calculate pore pressure based on mix, temperature, depth, and time."""
    # Base pressure from moisture evaporation
    if temp < 100:
        base_pressure = 0.0
    elif temp < 250:
        # Peak pressure zone for moisture
        base_pressure = 0.4 * np.exp(-((temp - 150)/50)**2)
    else:
        # Pressure relief after dehydration
        base_pressure = 0.1 * np.exp(-(temp - 250)/100)
    
    # Depth effect
    depth_factor = (depth / 40) * np.exp(-depth / 100)
    base_pressure *= depth_factor
    
    # Mix effect - rubber creates pathways for pressure relief
    if mix == 'R10S':
        base_pressure *= 0.7
    elif mix == 'R20S':
        base_pressure *= 0.5
    
    # Add time-dependent effects
    if time < 30:
        time_factor = time / 30
    else:
        time_factor = 1.0
    
    pressure = base_pressure * time_factor * np.random.normal(1, 0.1)
    
    return pressure

def calculate_transient_strain(mix, temp, load_level, time):
    """Calculate transient strain (LITS - Load Induced Thermal Strain)."""
    # Basic thermal strain
    thermal_strain = 10e-6 * (temp - 20) * 100  # In percentage
    
    # Load-induced thermal strain (LITS)
    if temp < 100:
        lits_factor = 0.5
    elif temp < 400:
        lits_factor = 2.0
    else:
        lits_factor = 3.5
    
    lits = load_level * lits_factor * (temp / 600) * 0.5  # In percentage
    
    # Creep component
    creep = load_level * 0.1 * np.log(1 + time/10) * (temp/600)
    
    # Mix effect - rubber increases strain
    if mix == 'R20S':
        strain_multiplier = 1.3
    else:
        strain_multiplier = 1.0
    
    total_strain = (thermal_strain + lits + creep) * strain_multiplier
    
    return total_strain * np.random.normal(1, 0.05)

# =============================================================================
# DATA VISUALIZATION AND ANALYSIS
# =============================================================================

def create_comprehensive_plots(df_ambient, df_residual, df_in_situ, df_pore_pressure):
    """Create comprehensive visualization plots."""
    print("\n[Visualization] Creating comprehensive analysis plots...")
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)
    
    # Plot 1: Ambient strength by mix and age
    ax1 = fig.add_subplot(gs[0, 0])
    for mix in ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']:
        mix_data = df_ambient[df_ambient['Mix_ID'] == mix]
        strength_by_age = mix_data.groupby('Curing_Age_days')['Compressive_Strength_MPa'].mean()
        ax1.plot(strength_by_age.index, strength_by_age.values, 'o-', label=mix, linewidth=2, markersize=8)
    ax1.set_xlabel('Curing Age (days)', fontsize=11)
    ax1.set_ylabel('Compressive Strength (MPa)', fontsize=11)
    ax1.set_title('A. Strength Development with Age', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residual strength vs temperature
    ax2 = fig.add_subplot(gs[0, 1])
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        strength_by_temp = mix_data.groupby('Peak_Temperature_C')['Residual_Compressive_Strength_MPa'].mean()
        ax2.plot(strength_by_temp.index, strength_by_temp.values, 'o-', label=mix, linewidth=2, markersize=8)
    ax2.set_xlabel('Peak Temperature (°C)', fontsize=11)
    ax2.set_ylabel('Residual Strength (MPa)', fontsize=11)
    ax2.set_title('B. Strength Degradation with Temperature', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mass loss vs temperature
    ax3 = fig.add_subplot(gs[0, 2])
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace')]
        mass_loss_by_temp = mix_data.groupby('Peak_Temperature_C')['Mass_Loss_pct'].mean()
        ax3.plot(mass_loss_by_temp.index, mass_loss_by_temp.values, 's-', label=mix, linewidth=2, markersize=8)
    ax3.set_xlabel('Peak Temperature (°C)', fontsize=11)
    ax3.set_ylabel('Mass Loss (%)', fontsize=11)
    ax3.set_title('C. Mass Loss Evolution', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effect of cooling method
    ax4 = fig.add_subplot(gs[1, 0])
    cooling_effect = []
    labels = []
    for mix in ['C', 'R20S']:
        for temp in [400, 600]:
            for cooling in ['Furnace', 'Quench']:
                data = df_residual[(df_residual['Mix_ID'] == mix) & 
                                 (df_residual['Peak_Temperature_C'] == temp) &
                                 (df_residual['Cooling_Method'] == cooling)]
                strength = data['Residual_Compressive_Strength_MPa'].mean()
                cooling_effect.append(strength)
                labels.append(f"{mix}\n{temp}°C\n{cooling}")
    
    colors = ['blue' if 'Furnace' in l else 'red' for l in labels]
    bars = ax4.bar(range(len(cooling_effect)), cooling_effect, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Residual Strength (MPa)', fontsize=11)
    ax4.set_title('D. Cooling Method Impact', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Spalling occurrence
    ax5 = fig.add_subplot(gs[1, 1])
    spalling_data = df_residual[df_residual['Heating_Rate'] == '10_C_per_min']
    spalling_summary = spalling_data.groupby(['Mix_ID', 'Peak_Temperature_C'])['Spalling_Occurred'].mean()
    spalling_pivot = spalling_summary.unstack()
    spalling_pivot.T.plot(kind='bar', ax=ax5, width=0.8)
    ax5.set_xlabel('Peak Temperature (°C)', fontsize=11)
    ax5.set_ylabel('Spalling Probability', fontsize=11)
    ax5.set_title('E. Spalling Risk (Rapid Heating)', fontsize=12, fontweight='bold')
    ax5.legend(title='Mix ID', framealpha=0.9)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: In-situ vs residual strength
    ax6 = fig.add_subplot(gs[1, 2])
    for mix in ['C', 'R20S']:
        # In-situ data
        in_situ = df_in_situ[df_in_situ['Mix_ID'] == mix]
        in_situ_strength = in_situ.groupby('Test_Temperature_C')['InSitu_Compressive_Strength_MPa'].mean()
        
        # Residual data (comparable conditions)
        residual = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
        residual_strength = residual.groupby('Peak_Temperature_C')['Residual_Compressive_Strength_MPa'].mean()
        
        ax6.plot(in_situ_strength.index, in_situ_strength.values, 'o-', 
                label=f'{mix} In-situ', linewidth=2, markersize=8)
        ax6.plot(residual_strength.index, residual_strength.values, 's--', 
                label=f'{mix} Residual', linewidth=2, markersize=6, alpha=0.7)
    
    ax6.set_xlabel('Temperature (°C)', fontsize=11)
    ax6.set_ylabel('Compressive Strength (MPa)', fontsize=11)
    ax6.set_title('F. In-situ vs Residual Properties', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', framealpha=0.9)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Strength-Mass Loss Correlation
    ax7 = fig.add_subplot(gs[2, 0])
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace')]
        ax7.scatter(mix_data['Mass_Loss_pct'], 
                   mix_data['Residual_Compressive_Strength_MPa'],
                   label=mix, alpha=0.6, s=30)
    ax7.set_xlabel('Mass Loss (%)', fontsize=11)
    ax7.set_ylabel('Residual Strength (MPa)', fontsize=11)
    ax7.set_title('G. Strength-Mass Loss Correlation', fontsize=12, fontweight='bold')
    ax7.legend(loc='best', framealpha=0.9)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: UPV correlation
    ax8 = fig.add_subplot(gs[2, 1])
    for mix in ['C', 'R20S']:
        mix_data = df_residual[df_residual['Mix_ID'] == mix]
        ax8.scatter(mix_data['UPV_mps'], 
                   mix_data['Residual_Compressive_Strength_MPa'],
                   label=mix, alpha=0.5, s=20)
    ax8.set_xlabel('UPV (m/s)', fontsize=11)
    ax8.set_ylabel('Residual Strength (MPa)', fontsize=11)
    ax8.set_title('H. UPV-Strength Relationship', fontsize=12, fontweight='bold')
    ax8.legend(loc='best', framealpha=0.9)
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Pore pressure peaks
    ax9 = fig.add_subplot(gs[2, 2])
    pore_data = df_pore_pressure.groupby(['Mix_ID', 'Depth_mm'])['Peak_Pressure_MPa'].mean()
    pore_pivot = pore_data.unstack()
    pore_pivot.plot(kind='bar', ax=ax9, width=0.8)
    ax9.set_xlabel('Mix ID', fontsize=11)
    ax9.set_ylabel('Peak Pore Pressure (MPa)', fontsize=11)
    ax9.set_title('I. Peak Pore Pressure by Depth', fontsize=12, fontweight='bold')
    ax9.legend(title='Depth (mm)', framealpha=0.9)
    ax9.set_xticklabels(ax9.get_xticklabels(), rotation=0)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Plot 10: Modulus degradation
    ax10 = fig.add_subplot(gs[3, 0])
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) & 
                              (df_residual['Cooling_Method'] == 'Furnace')]
        E_by_temp = mix_data.groupby('Peak_Temperature_C')['Residual_Elastic_Modulus_MPa'].mean()
        ax10.plot(E_by_temp.index, E_by_temp.values/1000, 'D-', label=mix, linewidth=2, markersize=7)
    ax10.set_xlabel('Peak Temperature (°C)', fontsize=11)
    ax10.set_ylabel('Residual E-Modulus (GPa)', fontsize=11)
    ax10.set_title('J. Elastic Modulus Degradation', fontsize=12, fontweight='bold')
    ax10.legend(loc='best', framealpha=0.9)
    ax10.grid(True, alpha=0.3)
    
    # Plot 11: Visual damage distribution
    ax11 = fig.add_subplot(gs[3, 1])
    damage_counts = df_residual.groupby(['Peak_Temperature_C', 'Visual_Cracking_Rating']).size().unstack(fill_value=0)
    damage_counts.plot(kind='bar', stacked=True, ax=ax11, 
                      color=['green', 'yellow', 'orange', 'red', 'darkred'])
    ax11.set_xlabel('Peak Temperature (°C)', fontsize=11)
    ax11.set_ylabel('Number of Specimens', fontsize=11)
    ax11.set_title('K. Visual Damage Assessment', fontsize=12, fontweight='bold')
    ax11.legend(title='Damage Rating', framealpha=0.9)
    ax11.set_xticklabels(ax11.get_xticklabels(), rotation=0)
    ax11.grid(True, alpha=0.3, axis='y')
    
    # Plot 12: Statistical summary
    ax12 = fig.add_subplot(gs[3, 2])
    summary_stats = []
    for mix in ['C', 'R10S', 'R20S']:
        for temp in [23, 400, 600]:
            data = df_residual[(df_residual['Mix_ID'] == mix) & 
                             (df_residual['Peak_Temperature_C'] == temp)]
            if len(data) > 0:
                summary_stats.append({
                    'Mix': mix,
                    'Temp': temp,
                    'Mean': data['Residual_Compressive_Strength_MPa'].mean(),
                    'Std': data['Residual_Compressive_Strength_MPa'].std(),
                    'COV': (data['Residual_Compressive_Strength_MPa'].std() / 
                           data['Residual_Compressive_Strength_MPa'].mean() * 100)
                })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_pivot = summary_df.pivot_table(values='COV', index='Temp', columns='Mix')
    summary_pivot.plot(kind='bar', ax=ax12, width=0.8)
    ax12.set_xlabel('Temperature (°C)', fontsize=11)
    ax12.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax12.set_title('L. Data Variability (COV)', fontsize=12, fontweight='bold')
    ax12.legend(title='Mix ID', framealpha=0.9)
    ax12.set_xticklabels(ax12.get_xticklabels(), rotation=0)
    ax12.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('COMPREHENSIVE FIRE RESISTANCE ANALYSIS - RUBBERIZED CONCRETE', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'comprehensive_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comprehensive analysis plot to {output_path}")
    plt.close()

def create_stress_strain_plots(stress_strain_curves):
    """Create stress-strain curve visualizations."""
    print("\n[Visualization] Creating stress-strain curve plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot curves for different temperatures
    temps = [23, 200, 400, 600]
    for i, temp in enumerate(temps):
        ax = axes[i//2, i%2]
        
        for mix in ['C', 'R10S', 'R20S']:
            spec_id = f"{mix}-28-IS-{temp}-1"
            if spec_id in stress_strain_curves:
                strains = np.array(stress_strain_curves[spec_id]['strain']) * 100  # Convert to %
                stresses = stress_strain_curves[spec_id]['stress']
                ax.plot(strains, stresses, linewidth=2.5, label=mix)
        
        ax.set_xlabel('Strain (%)', fontsize=11)
        ax.set_ylabel('Stress (MPa)', fontsize=11)
        ax.set_title(f'Temperature: {temp}°C', fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 4)
    
    plt.suptitle('IN-SITU STRESS-STRAIN BEHAVIOR AT ELEVATED TEMPERATURES', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'stress_strain_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved stress-strain curves to {output_path}")
    plt.close()

def create_pore_pressure_plots(pore_pressure_data):
    """Create pore pressure evolution plots."""
    print("\n[Visualization] Creating pore pressure evolution plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Pressure evolution for different mixes
    ax1 = axes[0]
    times = np.linspace(0, 120, 121)
    
    for mix in ['C', 'R10S', 'R20S']:
        # Get data for 25mm depth, run 1
        data = next((d for d in pore_pressure_data 
                    if d['Mix_ID'] == mix and d['Depth_mm'] == 25 and d['Run'] == 1), None)
        if data:
            ax1.plot(times, data['Pressure_Profile'], linewidth=2.5, label=f'{mix}')
    
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('Pore Pressure (MPa)', fontsize=11)
    ax1.set_title('Pore Pressure Evolution at 25mm Depth', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pressure vs depth for control mix
    ax2 = axes[1]
    for depth in [10, 25, 40]:
        data = next((d for d in pore_pressure_data 
                    if d['Mix_ID'] == 'C' and d['Depth_mm'] == depth and d['Run'] == 1), None)
        if data:
            ax2.plot(times, data['Pressure_Profile'], linewidth=2.5, label=f'{depth}mm')
    
    ax2.set_xlabel('Time (min)', fontsize=11)
    ax2.set_ylabel('Pore Pressure (MPa)', fontsize=11)
    ax2.set_title('Pore Pressure vs Depth (Control Mix)', fontsize=12, fontweight='bold')
    ax2.legend(title='Depth', loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('PORE PRESSURE DEVELOPMENT DURING HEATING', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'pore_pressure_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved pore pressure evolution to {output_path}")
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate the complete synthetic dataset."""
    
    # Generate all data components
    df_ambient = generate_ambient_data()
    df_residual = generate_residual_data(df_ambient)
    df_in_situ, stress_strain_curves = generate_in_situ_data(df_ambient)
    df_pore_pressure, df_transient_strain, pore_pressure_data, transient_strain_data = generate_advanced_measurements()
    
    # Save all datasets to CSV files
    print("\n[Export] Saving datasets to CSV files...")
    df_ambient.to_csv(os.path.join(output_dir, 'ambient_properties.csv'), index=False)
    df_residual.to_csv(os.path.join(output_dir, 'residual_properties_high_temp.csv'), index=False)
    df_in_situ.to_csv(os.path.join(output_dir, 'in_situ_properties.csv'), index=False)
    df_pore_pressure.to_csv(os.path.join(output_dir, 'pore_pressure_summary.csv'), index=False)
    df_transient_strain.to_csv(os.path.join(output_dir, 'transient_strain_summary.csv'), index=False)
    
    # Save stress-strain curves to JSON
    with open(os.path.join(output_dir, 'stress_strain_curves.json'), 'w') as f:
        json.dump(stress_strain_curves, f, indent=2)
    
    # Save complete pore pressure profiles to JSON
    pore_pressure_json = []
    for data in pore_pressure_data:
        pore_pressure_json.append({
            'Mix_ID': data['Mix_ID'],
            'Depth_mm': data['Depth_mm'],
            'Run': data['Run'],
            'Time_min': data['Time_min'].tolist(),
            'Pore_Pressure_MPa': [float(p) for p in data['Pressure_Profile']]
        })
    with open(os.path.join(output_dir, 'pore_pressure_profiles.json'), 'w') as f:
        json.dump(pore_pressure_json, f, indent=2)
    
    # Save transient strain profiles to JSON
    transient_strain_json = []
    for data in transient_strain_data:
        transient_strain_json.append({
            'Mix_ID': data['Mix_ID'],
            'Load_Level': data['Load_Level'],
            'Heating_Rate_C_per_min': data['Heating_Rate_C_per_min'],
            'Max_Temperature_C': data['Max_Temperature_C'],
            'Total_Transient_Strain_pct': data['Total_Transient_Strain_pct'],
            'Strain_Profile': [float(s) for s in data['Strain_Profile']]
        })
    with open(os.path.join(output_dir, 'transient_strain_profiles.json'), 'w') as f:
        json.dump(transient_strain_json, f, indent=2)
    
    print("  All datasets exported successfully!")
    
    # Create visualizations
    create_comprehensive_plots(df_ambient, df_residual, df_in_situ, df_pore_pressure)
    create_stress_strain_plots(stress_strain_curves)
    create_pore_pressure_plots(pore_pressure_data)
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE - SUMMARY STATISTICS")
    print("="*80)
    print(f"Output Directory: {output_dir}")
    print(f"\nDataset Sizes:")
    print(f"  - Ambient tests: {len(df_ambient)} specimens")
    print(f"  - Residual high-temperature tests: {len(df_residual)} specimens")
    print(f"  - In-situ high-temperature tests: {len(df_in_situ)} specimens")
    print(f"  - Pore pressure experiments: {len(df_pore_pressure)} configurations")
    print(f"  - Transient strain experiments: {len(df_transient_strain)} configurations")
    print(f"  - Stress-strain curves: {len(stress_strain_curves)} curves")
    
    print(f"\nFiles Generated:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"  - {file}: {file_size:.1f} KB")
    
    print("\n" + "="*80)
    print("Dataset ready for thermo-mechanical model development and validation!")
    print("="*80)

if __name__ == "__main__":
    main()