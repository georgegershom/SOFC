#!/usr/bin/env python3
"""
FEA Software Format Generator
============================

Converts the generated dataset into formats directly usable by commercial FEA software:
- ABAQUS (.inp material definitions and property tables)
- ANSYS (.mac macros and material property definitions)
- COMSOL (.mph material property functions and tables)

Each format includes proper units, temperature dependencies, and material model definitions.
"""

import numpy as np
import pandas as pd
import os

def load_datasets():
    """Load all generated datasets."""
    
    base_path = "/workspace/thermo_mechanical_dataset"
    
    datasets = {}
    
    # Load all datasets
    property_types = ['thermal_properties', 'mechanical_properties', 'transport_properties', 'deformation_properties']
    
    for prop_type in property_types:
        prop_path = f"{base_path}/{prop_type}"
        if os.path.exists(prop_path):
            for file in os.listdir(prop_path):
                if file.endswith('.csv'):
                    dataset_name = file.replace('.csv', '')
                    datasets[dataset_name] = pd.read_csv(f"{prop_path}/{file}")
    
    return datasets

def generate_abaqus_materials():
    """Generate ABAQUS material definitions."""
    
    datasets = load_datasets()
    
    abaqus_output = []
    
    # Header
    abaqus_output.append("** ABAQUS Material Definitions for Rubberized Concrete")
    abaqus_output.append("** Generated for Thermo-Mechanical Fire Modeling")
    abaqus_output.append("** Temperature Range: 20°C to 800°C")
    abaqus_output.append("**")
    
    mix_types = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    
    for mix_id in mix_types:
        abaqus_output.append(f"**")
        abaqus_output.append(f"** Material: {mix_id}")
        abaqus_output.append(f"**")
        abaqus_output.append(f"*MATERIAL, NAME={mix_id}_CONCRETE")
        
        # Density (assumed constant)
        if 'thermal_diffusivity' in datasets:
            density_data = datasets['thermal_diffusivity'][datasets['thermal_diffusivity']['Mix_ID'] == mix_id]
            if not density_data.empty:
                avg_density = density_data['Density_kg_m3'].mean()
                abaqus_output.append(f"*DENSITY")
                abaqus_output.append(f"{avg_density:.1f},")
        
        # Elastic properties
        if 'elastic_modulus' in datasets and 'poissons_ratio' in datasets:
            modulus_data = datasets['elastic_modulus'][datasets['elastic_modulus']['Mix_ID'] == mix_id]
            poisson_data = datasets['poissons_ratio'][datasets['poissons_ratio']['Mix_ID'] == mix_id]
            
            if not modulus_data.empty and not poisson_data.empty:
                abaqus_output.append(f"*ELASTIC, TYPE=ISOTROPIC")
                
                # Merge data by temperature
                merged_data = pd.merge(
                    modulus_data[['Temperature_C', 'Elastic_Modulus_Mean_MPa']], 
                    poisson_data[['Temperature_C', 'Poissons_Ratio_Mean']], 
                    on='Temperature_C'
                )
                
                for _, row in merged_data.iterrows():
                    temp = row['Temperature_C']
                    E = row['Elastic_Modulus_Mean_MPa']
                    nu = row['Poissons_Ratio_Mean']
                    abaqus_output.append(f"{E:.0f}, {nu:.4f}, {temp:.0f}")
        
        # Thermal expansion
        if 'thermal_strain' in datasets:
            expansion_data = datasets['thermal_strain'][datasets['thermal_strain']['Mix_ID'] == mix_id]
            
            if not expansion_data.empty:
                abaqus_output.append(f"*EXPANSION, TYPE=ISOTROPIC")
                
                for _, row in expansion_data.iterrows():
                    temp = row['Temperature_C']
                    # Convert from scientific notation string to float
                    alpha_str = row['Thermal_Expansion_Coeff_Mean_per_K']
                    alpha = float(alpha_str)
                    abaqus_output.append(f"{alpha:.2e}, {temp:.0f}")
        
        # Thermal conductivity
        if 'thermal_conductivity' in datasets:
            conductivity_data = datasets['thermal_conductivity'][datasets['thermal_conductivity']['Mix_ID'] == mix_id]
            
            if not conductivity_data.empty:
                abaqus_output.append(f"*CONDUCTIVITY")
                
                for _, row in conductivity_data.iterrows():
                    temp = row['Temperature_C']
                    k = row['Thermal_Conductivity_Mean_W_m_K']
                    abaqus_output.append(f"{k:.4f}, {temp:.0f}")
        
        # Specific heat
        if 'specific_heat' in datasets:
            heat_data = datasets['specific_heat'][datasets['specific_heat']['Mix_ID'] == mix_id]
            
            if not heat_data.empty:
                abaqus_output.append(f"*SPECIFIC HEAT")
                
                for _, row in heat_data.iterrows():
                    temp = row['Temperature_C']
                    cp = row['Specific_Heat_Mean_J_kg_K']
                    abaqus_output.append(f"{cp:.1f}, {temp:.0f}")
        
        # Creep properties (if available)
        if 'creep_parameters' in datasets:
            creep_data = datasets['creep_parameters'][datasets['creep_parameters']['Mix_ID'] == mix_id]
            
            if not creep_data.empty:
                abaqus_output.append(f"*CREEP, LAW=POWER")
                
                # Use representative values (e.g., at 400°C)
                mid_temp_data = creep_data[creep_data['Temperature_C'] == 400]
                if not mid_temp_data.empty:
                    row = mid_temp_data.iloc[0]
                    A = float(row['Creep_Coefficient_A_Mean'])
                    n = row['Stress_Exponent_n_Mean']
                    m = row['Time_Exponent_m_Mean']
                    abaqus_output.append(f"{A:.2e}, {n:.3f}, {m:.4f}")
        
        abaqus_output.append("")
    
    return "\n".join(abaqus_output)

def generate_ansys_materials():
    """Generate ANSYS material definitions."""
    
    datasets = load_datasets()
    
    ansys_output = []
    
    # Header
    ansys_output.append("! ANSYS Material Definitions for Rubberized Concrete")
    ansys_output.append("! Generated for Thermo-Mechanical Fire Modeling")
    ansys_output.append("! Temperature Range: 20°C to 800°C")
    ansys_output.append("!")
    
    mix_types = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    
    for i, mix_id in enumerate(mix_types, 1):
        ansys_output.append(f"!")
        ansys_output.append(f"! Material {i}: {mix_id}")
        ansys_output.append(f"!")
        ansys_output.append(f"MP,DELETE,ALL,{i}")
        
        # Density
        if 'thermal_diffusivity' in datasets:
            density_data = datasets['thermal_diffusivity'][datasets['thermal_diffusivity']['Mix_ID'] == mix_id]
            if not density_data.empty:
                avg_density = density_data['Density_kg_m3'].mean()
                ansys_output.append(f"MP,DENS,{i},{avg_density:.1f}")
        
        # Temperature-dependent properties
        temperatures = np.arange(20, 801, 20)  # Every 20°C for ANSYS
        
        # Elastic modulus
        if 'elastic_modulus' in datasets:
            modulus_data = datasets['elastic_modulus'][datasets['elastic_modulus']['Mix_ID'] == mix_id]
            
            if not modulus_data.empty:
                ansys_output.append(f"! Elastic Modulus (Pa)")
                for temp in temperatures:
                    temp_data = modulus_data[modulus_data['Temperature_C'] == temp]
                    if not temp_data.empty:
                        E = temp_data.iloc[0]['Elastic_Modulus_Mean_MPa'] * 1e6  # Convert to Pa
                        ansys_output.append(f"MPTEMP,{int((temp-20)/20)+1},{temp}")
                        ansys_output.append(f"MPDATA,EX,{i},{int((temp-20)/20)+1},{E:.0f}")
        
        # Poisson's ratio
        if 'poissons_ratio' in datasets:
            poisson_data = datasets['poissons_ratio'][datasets['poissons_ratio']['Mix_ID'] == mix_id]
            
            if not poisson_data.empty:
                ansys_output.append(f"! Poisson's Ratio")
                for temp in temperatures:
                    temp_data = poisson_data[poisson_data['Temperature_C'] == temp]
                    if not temp_data.empty:
                        nu = temp_data.iloc[0]['Poissons_Ratio_Mean']
                        ansys_output.append(f"MPDATA,PRXY,{i},{int((temp-20)/20)+1},{nu:.4f}")
        
        # Thermal expansion
        if 'thermal_strain' in datasets:
            expansion_data = datasets['thermal_strain'][datasets['thermal_strain']['Mix_ID'] == mix_id]
            
            if not expansion_data.empty:
                ansys_output.append(f"! Thermal Expansion Coefficient (1/K)")
                for temp in temperatures:
                    temp_data = expansion_data[expansion_data['Temperature_C'] == temp]
                    if not temp_data.empty:
                        alpha_str = temp_data.iloc[0]['Thermal_Expansion_Coeff_Mean_per_K']
                        alpha = float(alpha_str)
                        ansys_output.append(f"MPDATA,ALPX,{i},{int((temp-20)/20)+1},{alpha:.2e}")
        
        # Thermal conductivity
        if 'thermal_conductivity' in datasets:
            conductivity_data = datasets['thermal_conductivity'][datasets['thermal_conductivity']['Mix_ID'] == mix_id]
            
            if not conductivity_data.empty:
                ansys_output.append(f"! Thermal Conductivity (W/m·K)")
                for temp in temperatures:
                    temp_data = conductivity_data[conductivity_data['Temperature_C'] == temp]
                    if not temp_data.empty:
                        k = temp_data.iloc[0]['Thermal_Conductivity_Mean_W_m_K']
                        ansys_output.append(f"MPDATA,KXX,{i},{int((temp-20)/20)+1},{k:.4f}")
        
        # Specific heat
        if 'specific_heat' in datasets:
            heat_data = datasets['specific_heat'][datasets['specific_heat']['Mix_ID'] == mix_id]
            
            if not heat_data.empty:
                ansys_output.append(f"! Specific Heat (J/kg·K)")
                for temp in temperatures:
                    temp_data = heat_data[heat_data['Temperature_C'] == temp]
                    if not temp_data.empty:
                        cp = temp_data.iloc[0]['Specific_Heat_Mean_J_kg_K']
                        ansys_output.append(f"MPDATA,C,{i},{int((temp-20)/20)+1},{cp:.1f}")
        
        ansys_output.append("")
    
    return "\n".join(ansys_output)

def generate_comsol_materials():
    """Generate COMSOL material definitions."""
    
    datasets = load_datasets()
    
    comsol_output = []
    
    # Header - COMSOL uses Java-like syntax for material definitions
    comsol_output.append("% COMSOL Material Definitions for Rubberized Concrete")
    comsol_output.append("% Generated for Thermo-Mechanical Fire Modeling")
    comsol_output.append("% Temperature Range: 20°C to 800°C")
    comsol_output.append("%")
    comsol_output.append("% Usage: Copy these functions into COMSOL material property definitions")
    comsol_output.append("%")
    
    mix_types = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    
    for mix_id in mix_types:
        comsol_output.append(f"%")
        comsol_output.append(f"% Material: {mix_id}")
        comsol_output.append(f"%")
        
        # Density function
        if 'thermal_diffusivity' in datasets:
            density_data = datasets['thermal_diffusivity'][datasets['thermal_diffusivity']['Mix_ID'] == mix_id]
            if not density_data.empty:
                avg_density = density_data['Density_kg_m3'].mean()
                comsol_output.append(f"% Density (kg/m³)")
                comsol_output.append(f"rho_{mix_id} = {avg_density:.1f}")
        
        # Elastic modulus function
        if 'elastic_modulus' in datasets:
            modulus_data = datasets['elastic_modulus'][datasets['elastic_modulus']['Mix_ID'] == mix_id]
            
            if not modulus_data.empty:
                comsol_output.append(f"% Elastic Modulus (Pa) - Temperature dependent")
                
                # Create interpolation points
                temps = modulus_data['Temperature_C'].values
                E_values = modulus_data['Elastic_Modulus_Mean_MPa'].values * 1e6  # Convert to Pa
                
                # Generate piecewise linear function
                comsol_output.append(f"E_{mix_id}(T) = piecewise(")
                for i in range(len(temps)-1):
                    T1, T2 = temps[i], temps[i+1]
                    E1, E2 = E_values[i], E_values[i+1]
                    slope = (E2 - E1) / (T2 - T1)
                    comsol_output.append(f"  (T >= {T1}[degC]) && (T < {T2}[degC]), {E1:.0f} + {slope:.0f}*(T-{T1}[degC]),")
                
                # Last interval
                comsol_output.append(f"  T >= {temps[-1]}[degC], {E_values[-1]:.0f})")
        
        # Poisson's ratio function
        if 'poissons_ratio' in datasets:
            poisson_data = datasets['poissons_ratio'][datasets['poissons_ratio']['Mix_ID'] == mix_id]
            
            if not poisson_data.empty:
                comsol_output.append(f"% Poisson's Ratio - Temperature dependent")
                
                temps = poisson_data['Temperature_C'].values
                nu_values = poisson_data['Poissons_Ratio_Mean'].values
                
                comsol_output.append(f"nu_{mix_id}(T) = piecewise(")
                for i in range(len(temps)-1):
                    T1, T2 = temps[i], temps[i+1]
                    nu1, nu2 = nu_values[i], nu_values[i+1]
                    slope = (nu2 - nu1) / (T2 - T1)
                    comsol_output.append(f"  (T >= {T1}[degC]) && (T < {T2}[degC]), {nu1:.4f} + {slope:.6f}*(T-{T1}[degC]),")
                
                comsol_output.append(f"  T >= {temps[-1]}[degC], {nu_values[-1]:.4f})")
        
        # Thermal expansion function
        if 'thermal_strain' in datasets:
            expansion_data = datasets['thermal_strain'][datasets['thermal_strain']['Mix_ID'] == mix_id]
            
            if not expansion_data.empty:
                comsol_output.append(f"% Thermal Expansion Coefficient (1/K)")
                
                temps = expansion_data['Temperature_C'].values
                alpha_values = []
                
                for _, row in expansion_data.iterrows():
                    alpha_str = row['Thermal_Expansion_Coeff_Mean_per_K']
                    alpha_values.append(float(alpha_str))
                
                alpha_values = np.array(alpha_values)
                
                comsol_output.append(f"alpha_{mix_id}(T) = piecewise(")
                for i in range(len(temps)-1):
                    T1, T2 = temps[i], temps[i+1]
                    a1, a2 = alpha_values[i], alpha_values[i+1]
                    slope = (a2 - a1) / (T2 - T1)
                    comsol_output.append(f"  (T >= {T1}[degC]) && (T < {T2}[degC]), {a1:.2e} + {slope:.2e}*(T-{T1}[degC]),")
                
                comsol_output.append(f"  T >= {temps[-1]}[degC], {alpha_values[-1]:.2e})")
        
        # Thermal conductivity function
        if 'thermal_conductivity' in datasets:
            conductivity_data = datasets['thermal_conductivity'][datasets['thermal_conductivity']['Mix_ID'] == mix_id]
            
            if not conductivity_data.empty:
                comsol_output.append(f"% Thermal Conductivity (W/m·K)")
                
                temps = conductivity_data['Temperature_C'].values
                k_values = conductivity_data['Thermal_Conductivity_Mean_W_m_K'].values
                
                comsol_output.append(f"k_{mix_id}(T) = piecewise(")
                for i in range(len(temps)-1):
                    T1, T2 = temps[i], temps[i+1]
                    k1, k2 = k_values[i], k_values[i+1]
                    slope = (k2 - k1) / (T2 - T1)
                    comsol_output.append(f"  (T >= {T1}[degC]) && (T < {T2}[degC]), {k1:.4f} + {slope:.6f}*(T-{T1}[degC]),")
                
                comsol_output.append(f"  T >= {temps[-1]}[degC], {k_values[-1]:.4f})")
        
        # Specific heat function
        if 'specific_heat' in datasets:
            heat_data = datasets['specific_heat'][datasets['specific_heat']['Mix_ID'] == mix_id]
            
            if not heat_data.empty:
                comsol_output.append(f"% Specific Heat (J/kg·K)")
                
                temps = heat_data['Temperature_C'].values
                cp_values = heat_data['Specific_Heat_Mean_J_kg_K'].values
                
                comsol_output.append(f"Cp_{mix_id}(T) = piecewise(")
                for i in range(len(temps)-1):
                    T1, T2 = temps[i], temps[i+1]
                    cp1, cp2 = cp_values[i], cp_values[i+1]
                    slope = (cp2 - cp1) / (T2 - T1)
                    comsol_output.append(f"  (T >= {T1}[degC]) && (T < {T2}[degC]), {cp1:.1f} + {slope:.3f}*(T-{T1}[degC]),")
                
                comsol_output.append(f"  T >= {temps[-1]}[degC], {cp_values[-1]:.1f})")
        
        comsol_output.append("")
    
    return "\n".join(comsol_output)

def main():
    """Generate FEA software format files."""
    
    print("Generating FEA Software Format Files...")
    print("=" * 50)
    
    # Create output directories
    base_output_dir = "/workspace/thermo_mechanical_dataset/fea_formats"
    
    for software in ['abaqus', 'ansys', 'comsol']:
        os.makedirs(f"{base_output_dir}/{software}", exist_ok=True)
    
    print("1. Generating ABAQUS material definitions...")
    abaqus_content = generate_abaqus_materials()
    with open(f"{base_output_dir}/abaqus/rubberized_concrete_materials.inp", 'w') as f:
        f.write(abaqus_content)
    
    print("2. Generating ANSYS material definitions...")
    ansys_content = generate_ansys_materials()
    with open(f"{base_output_dir}/ansys/rubberized_concrete_materials.mac", 'w') as f:
        f.write(ansys_content)
    
    print("3. Generating COMSOL material definitions...")
    comsol_content = generate_comsol_materials()
    with open(f"{base_output_dir}/comsol/rubberized_concrete_materials.txt", 'w') as f:
        f.write(comsol_content)
    
    # Generate usage instructions
    print("4. Generating usage instructions...")
    
    usage_instructions = """# FEA Software Material Definitions Usage Guide

## ABAQUS (.inp file)
1. Copy the contents of `rubberized_concrete_materials.inp` into your ABAQUS input file
2. Reference materials in your model using names like `C_CONCRETE`, `R5S_CONCRETE`, etc.
3. Ensure temperature-dependent analysis is enabled: `*STEP, INC=100, NLGEOM=YES`

## ANSYS (.mac file)  
1. Load the macro file: `/INPUT,rubberized_concrete_materials,mac`
2. Materials are numbered 1-6 corresponding to C, R5S, R10S, R15S, R20S, R10L
3. Use `MAT,1` to `MAT,6` to assign materials to elements
4. Enable temperature-dependent analysis: `TREF,20` (reference temperature)

## COMSOL (.txt file)
1. Copy the function definitions into COMSOL material property fields
2. Use functions like `E_C(T)`, `nu_R5S(T)`, etc. in material property expressions
3. Ensure temperature variable `T` is properly defined in your physics
4. Units are consistent: Pa for modulus, kg/m³ for density, etc.

## Temperature Range
All definitions are valid from 20°C to 800°C with 10°C increments.

## Material Naming Convention
- C: Control concrete (0% rubber)
- R5S: 5% small rubber particles
- R10S: 10% small rubber particles  
- R15S: 15% small rubber particles
- R20S: 20% small rubber particles
- R10L: 10% large rubber particles
"""
    
    with open(f"{base_output_dir}/usage_instructions.md", 'w') as f:
        f.write(usage_instructions)
    
    print("\nFEA Format Generation Summary:")
    print(f"- ABAQUS: Material definitions with temperature-dependent properties")
    print(f"- ANSYS: Macro file with material property tables")
    print(f"- COMSOL: Piecewise functions for temperature dependence")
    print(f"- Usage instructions provided for all formats")
    
    print("\nFiles Generated:")
    print(f"- abaqus/rubberized_concrete_materials.inp")
    print(f"- ansys/rubberized_concrete_materials.mac")
    print(f"- comsol/rubberized_concrete_materials.txt")
    print(f"- usage_instructions.md")
    
    print("\nFEA software format generation completed successfully!")

if __name__ == "__main__":
    main()