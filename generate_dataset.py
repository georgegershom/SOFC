#!/usr/bin/env python3
"""
Material Property & Constitutive Model Dataset Generator
for SOFC FEM Analysis

Generates comprehensive material property datasets including:
- Thermo-physical properties (green state, sintering kinetics, CTE)
- Viscoplastic (creep) constitutive data
- Elastic properties as function of temperature and relative density
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import curve_fit

# Gas constant
R = 8.314e-3  # kJ/(mol?K)

def generate_sintering_kinetics(layer_name, temp_range, params):
    """
    Generate dilatometry data (dL/L0 vs T, t) for sintering kinetics
    
    Parameters:
    - layer_name: 'anode', 'electrolyte', 'cathode'
    - temp_range: [min_temp, max_temp] in Celsius
    - params: dictionary with sintering parameters
    """
    temps = np.linspace(temp_range[0], temp_range[1], 200)
    times = np.linspace(0, 180, 181)  # 3 hours at max temp
    
    # Sintering strain model: dL/L0 = -f(T, t)
    # Simplified model: exponential decay with temperature-dependent rate
    data = []
    
    for t_time in times:
        for T in temps:
            # Sintering rate increases with temperature
            T_k = T + 273.15
            sintering_rate = params['A0'] * np.exp(-params['Q'] / (R * T_k))
            
            # Accumulated shrinkage strain
            strain = -params['max_shrinkage'] * (1 - np.exp(-sintering_rate * t_time / 60))
            
            # Relative density increases with shrinkage
            initial_density = params['initial_relative_density']
            relative_density = initial_density + (1 - initial_density) * abs(strain) / params['max_shrinkage']
            relative_density = min(relative_density, 0.99)
            
            strain_rate = params['max_shrinkage'] * sintering_rate * np.exp(-sintering_rate * t_time / 60) / 60
            
            data.append({
                'temperature_C': T,
                'time_minutes': t_time,
                'strain_dL_L0': strain,
                'strain_rate_per_s': strain_rate,
                'relative_density': relative_density
            })
    
    df = pd.DataFrame(data)
    return df


def generate_bilayer_sintering(layer1_name, layer2_name, params1, params2):
    """Generate bilayer sintering kinetics (anode-electrolyte, electrolyte-cathode)"""
    temp_range = [1000, 1350]
    temps = np.linspace(temp_range[0], temp_range[1], 200)
    times = np.linspace(0, 180, 181)
    
    data = []
    
    for t_time in times:
        for T in temps:
            T_k = T + 273.15
            
            # Individual layer strains
            rate1 = params1['A0'] * np.exp(-params1['Q'] / (R * T_k))
            rate2 = params2['A0'] * np.exp(-params2['Q'] / (R * T_k))
            
            strain1 = -params1['max_shrinkage'] * (1 - np.exp(-rate1 * t_time / 60))
            strain2 = -params2['max_shrinkage'] * (1 - np.exp(-rate2 * t_time / 60))
            
            # Mismatch strain (drives stress)
            mismatch_strain = strain1 - strain2
            
            data.append({
                'temperature_C': T,
                'time_minutes': t_time,
                'strain_layer1_dL_L0': strain1,
                'strain_layer2_dL_L0': strain2,
                'mismatch_strain_dL_L0': mismatch_strain,
                'relative_density_layer1': min(params1['initial_relative_density'] + 
                                               abs(strain1) / params1['max_shrinkage'] * 
                                               (1 - params1['initial_relative_density']), 0.99),
                'relative_density_layer2': min(params2['initial_relative_density'] + 
                                               abs(strain2) / params2['max_shrinkage'] * 
                                               (1 - params2['initial_relative_density']), 0.99)
            })
    
    df = pd.DataFrame(data)
    return df


def generate_cte_data(layer_name, cte_params):
    """Generate Coefficient of Thermal Expansion data vs temperature"""
    temps = np.linspace(25, 1350, 300)
    
    # CTE typically increases slightly with temperature
    cte_rt = cte_params['RT']
    cte_1350 = cte_params['1350C']
    
    # Linear interpolation with slight nonlinearity
    data = []
    for T in temps:
        # Polynomial fit for smooth variation
        normalized_T = (T - 25) / 1325
        cte = cte_rt + (cte_1350 - cte_rt) * normalized_T * (1 + 0.1 * normalized_T)
        
        data.append({
            'temperature_C': T,
            'cte_10minus6_per_K': cte
        })
    
    df = pd.DataFrame(data)
    return df


def generate_creep_data(layer_name, temperature_C, stress_mpa, creep_params, green_state=True):
    """
    Generate constant-stress creep test data
    
    Norton's Law: epsilon_dot = A * sigma^n * exp(-Q/(R*T))
    Integrating: epsilon(t) = A * sigma^n * exp(-Q/(R*T)) * t
    """
    T_k = temperature_C + 273.15
    
    if green_state:
        A = creep_params['green_state']['A']
        n = creep_params['green_state']['n']
        Q = creep_params['green_state']['Q']
    else:
        A = creep_params['sintering_state']['A']
        n = creep_params['sintering_state']['n']
        Q = creep_params['sintering_state']['Q']
    
    # Time array: 0 to 100 hours
    time_hours = np.linspace(0, 100, 1000)
    time_seconds = time_hours * 3600
    
    # Calculate strain rate
    strain_rate = A * (stress_mpa ** n) * np.exp(-Q / (R * T_k))
    
    # Accumulated strain (primary + secondary creep)
    # Add primary creep contribution: epsilon_p = A_p * (1 - exp(-t/tau))
    A_primary = strain_rate * 0.3  # 30% of total strain is primary
    tau = 3600  # 1 hour relaxation time
    
    strain = strain_rate * time_seconds + A_primary * (1 - np.exp(-time_seconds / tau))
    
    data = pd.DataFrame({
        'time_seconds': time_seconds,
        'time_hours': time_hours,
        'strain': strain,
        'strain_rate_per_s': strain_rate + A_primary * np.exp(-time_seconds / tau) / tau,
        'stress_mpa': stress_mpa,
        'temperature_C': temperature_C
    })
    
    return data


def generate_elastic_properties(layer_name, elastic_params):
    """
    Generate Young's modulus and Poisson's ratio as function of 
    temperature and relative density
    """
    temps = np.linspace(25, 1350, 100)
    relative_densities = np.linspace(0.60, 0.99, 40)
    
    data = []
    
    for T in temps:
        for rho_rel in relative_densities:
            # Temperature dependence of modulus
            E_rt = elastic_params['E_RT_GPa']
            E_800 = elastic_params['E_800C_GPa']
            E_1350 = E_800 * 0.85  # Further decrease at higher temp
            
            if T <= 800:
                E_T = E_rt + (E_800 - E_rt) * (T - 25) / 775
            else:
                E_T = E_800 + (E_1350 - E_800) * (T - 800) / 550
            
            # Relative density dependence (power law)
            # E = E_fully_dense * (rho_rel)^m, where m ~ 2-3
            m = 2.5
            E = E_T * (rho_rel ** m)
            
            # Poisson's ratio (slight dependence on density)
            nu_0 = elastic_params['Poissons_ratio']
            nu = nu_0 * (0.95 + 0.05 * rho_rel)  # Small increase with density
            
            data.append({
                'temperature_C': T,
                'relative_density': rho_rel,
                'youngs_modulus_gpa': E,
                'poissons_ratio': nu
            })
    
    df = pd.DataFrame(data)
    return df


def generate_simple_elastic_properties(layer_name, elastic_params):
    """Generate temperature-dependent elastic properties (no density variation)"""
    temps = np.linspace(25, 1350, 300)
    
    data = []
    for T in temps:
        E_rt = elastic_params['E_RT_GPa']
        E_800 = elastic_params['E_800C_GPa']
        E_1350 = E_800 * 0.85 if 'E_1350C_GPa' not in elastic_params else elastic_params['E_1350C_GPa']
        
        if T <= 800:
            E = E_rt + (E_800 - E_rt) * (T - 25) / 775
        else:
            E = E_800 + (E_1350 - E_800) * (T - 800) / 550
        
        nu = elastic_params['Poissons_ratio']
        
        data.append({
            'temperature_C': T,
            'youngs_modulus_gpa': E,
            'poissons_ratio': nu
        })
    
    df = pd.DataFrame(data)
    return df


def main():
    """Generate all dataset files"""
    
    # Create data directory
    os.makedirs('dataset', exist_ok=True)
    
    # Load dataset configuration
    with open('material_property_dataset.json', 'r') as f:
        dataset = json.load(f)
    
    # Material parameters for each layer
    materials = {
        'anode': {
            'sintering': {
                'A0': 1e-5,  # Rate constant
                'Q': 380,    # Activation energy (kJ/mol)
                'max_shrinkage': 0.18,  # Maximum shrinkage strain
                'initial_relative_density': 0.65
            },
            'cte': {
                'RT': 12.5,
                '800C': 13.3,
                '1350C': 13.8
            },
            'elastic': {
                'E_RT_GPa': 55,
                'E_800C_GPa': 29,
                'Poissons_ratio': 0.29
            },
            'creep': {
                'green_state': {'A': 1.2e-10, 'n': 2.1, 'Q': 420},
                'sintering_state': {'A': 5.8e-12, 'n': 1.9, 'Q': 395}
            }
        },
        'electrolyte': {
            'sintering': {
                'A0': 8e-6,
                'Q': 400,
                'max_shrinkage': 0.15,
                'initial_relative_density': 0.75
            },
            'cte': {
                'RT': 10.0,
                '800C': 10.5,
                '1350C': 11.2
            },
            'elastic': {
                'E_RT_GPa': 200,
                'E_800C_GPa': 170,
                'Poissons_ratio': 0.23
            },
            'creep': {
                'green_state': {'A': 2.5e-11, 'n': 1.9, 'Q': 410},
                'sintering_state': {'A': 8.5e-12, 'n': 1.8, 'Q': 385}
            }
        },
        'cathode': {
            'sintering': {
                'A0': 1.2e-5,
                'Q': 390,
                'max_shrinkage': 0.17,
                'initial_relative_density': 0.70
            },
            'cte': {
                'RT': 11.5,
                '800C': 12.0,
                '1350C': 12.8
            },
            'elastic': {
                'E_RT_GPa': 45,
                'E_800C_GPa': 40,
                'Poissons_ratio': 0.25
            },
            'creep': {
                'green_state': {'A': 9.5e-11, 'n': 2.3, 'Q': 435},
                'sintering_state': {'A': 3.2e-12, 'n': 2.0, 'Q': 405}
            }
        },
        'interconnect': {
            'cte': {
                'RT': 11.5,
                '800C': 11.9,
                '1350C': 12.5
            },
            'elastic': {
                'E_RT_GPa': 160,
                'E_800C_GPa': 140,
                'Poissons_ratio': 0.30
            },
            'creep': {
                'green_state': {'A': 2.1e-8, 'n': 3.2, 'Q': 285},
                'sintering_state': {'A': 2.1e-8, 'n': 3.2, 'Q': 285}  # Same for metal
            }
        }
    }
    
    print("?????????...")
    # Generate sintering kinetics
    for layer in ['anode', 'electrolyte', 'cathode']:
        temp_range = [800 if layer != 'electrolyte' else 1000, 1350]
        df = generate_sintering_kinetics(layer, temp_range, materials[layer]['sintering'])
        df.to_csv(f'dataset/sintering_kinetics_{layer}.csv', index=False)
        print(f"  ? sintering_kinetics_{layer}.csv")
    
    # Generate bilayer sintering
    print("\n????????...")
    bilayer_anode_elec = generate_bilayer_sintering(
        'anode', 'electrolyte',
        materials['anode']['sintering'],
        materials['electrolyte']['sintering']
    )
    bilayer_anode_elec.to_csv('dataset/sintering_kinetics_anode_electrolyte.csv', index=False)
    print("  ? sintering_kinetics_anode_electrolyte.csv")
    
    bilayer_elec_cath = generate_bilayer_sintering(
        'electrolyte', 'cathode',
        materials['electrolyte']['sintering'],
        materials['cathode']['sintering']
    )
    bilayer_elec_cath.to_csv('dataset/sintering_kinetics_electrolyte_cathode.csv', index=False)
    print("  ? sintering_kinetics_electrolyte_cathode.csv")
    
    print("\n?????????...")
    # Generate CTE data
    for layer in ['anode', 'electrolyte', 'cathode', 'interconnect']:
        df = generate_cte_data(layer, materials[layer]['cte'])
        df.to_csv(f'dataset/cte_{layer}.csv', index=False)
        print(f"  ? cte_{layer}.csv")
    
    print("\n????????...")
    # Generate creep data for each layer at multiple temperatures and stresses
    temperatures = {'anode': [800, 1000, 1200],
                    'electrolyte': [800, 1000, 1200],
                    'cathode': [800, 1000, 1200],
                    'interconnect': [600, 800, 1000]}
    
    stress_levels = {'anode': [0.5, 1.0, 2.0, 5.0, 10.0],
                     'electrolyte': [1.0, 2.0, 5.0, 10.0, 20.0],
                     'cathode': [0.5, 1.0, 2.0, 5.0, 10.0],
                     'interconnect': [5.0, 10.0, 20.0, 50.0]}
    
    for layer in ['anode', 'electrolyte', 'cathode', 'interconnect']:
        for temp in temperatures[layer]:
            # Combine all stress levels for this temperature
            all_data = []
            for stress in stress_levels[layer]:
                # Generate for both green and sintering states (except interconnect)
                if layer != 'interconnect':
                    data_green = generate_creep_data(layer, temp, stress, 
                                                    materials[layer]['creep'], 
                                                    green_state=True)
                    data_green['state'] = 'green'
                    all_data.append(data_green)
                
                data_sintered = generate_creep_data(layer, temp, stress, 
                                                   materials[layer]['creep'], 
                                                   green_state=False)
                data_sintered['state'] = 'sintered'
                all_data.append(data_sintered)
            
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(f'dataset/creep_{layer}_{temp}C.csv', index=False)
            print(f"  ? creep_{layer}_{temp}C.csv")
    
    print("\n????????...")
    # Generate elastic properties
    for layer in ['anode', 'electrolyte', 'cathode']:
        df = generate_elastic_properties(layer, materials[layer]['elastic'])
        df.to_csv(f'dataset/elastic_density_{layer}.csv', index=False)
        print(f"  ? elastic_density_{layer}.csv")
        
        # Simple temperature-dependent (no density)
        df_simple = generate_simple_elastic_properties(layer, materials[layer]['elastic'])
        df_simple.to_csv(f'dataset/elastic_{layer}.csv', index=False)
        print(f"  ? elastic_{layer}.csv")
    
    # Interconnect (no density dependence)
    df_inter = generate_simple_elastic_properties('interconnect', materials['interconnect']['elastic'])
    df_inter.to_csv('dataset/elastic_interconnect.csv', index=False)
    print("  ? elastic_interconnect.csv")
    
    print("\n????????")
    print(f"\n??????? 'dataset/' ??:")
    files = os.listdir('dataset')
    for f in sorted(files):
        if f.endswith('.csv'):
            size = os.path.getsize(f'dataset/{f}') / 1024  # KB
            print(f"  - {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()
