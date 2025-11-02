#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOFC???????????
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_dataset():
    """?????"""
    with open('material_dataset/material_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def example_1_basic_properties():
    """??1: ????????"""
    print("=" * 60)
    print("??1: ????????")
    print("=" * 60)
    
    dataset = load_dataset()
    
    # ????????
    green_props = dataset['green_state_properties']
    
    print("\n??????:")
    for layer, props in green_props.items():
        print(f"\n{layer.upper()} ({props['material']}):")
        print(f"  ????: {props['green_density_kg_m3']} kg/m?")
        print(f"  ????: {props['sintered_density_kg_m3']} kg/m?")
        print(f"  ?????: {props['initial_porosity']*100:.1f}%")
        print(f"  ?????: {props['binder_content_wt_percent']} wt%")

def example_2_creep_parameters():
    """??2: ????????"""
    print("\n" + "=" * 60)
    print("??2: ???????? (Norton??)")
    print("=" * 60)
    
    dataset = load_dataset()
    
    # ??????????????
    creep_params = dataset['creep_constitutive_data']['electrolyte']['sintered']['creep_parameters']
    
    print("\n??? (8YSZ) - ????????:")
    print(f"  A (?????): {creep_params['A_pre_exponential']:.2e} s?? MPa??")
    print(f"  n (????): {creep_params['n_stress_exponent']}")
    print(f"  Q (???): {creep_params['Q_activation_energy_kJ_mol']:.1f} kJ/mol")
    
    # ??Norton?????????
    print("\n??Norton?????????:")
    print("  ?? = A ? ?? ? exp(-Q/RT)")
    
    A = creep_params['A_pre_exponential']
    n = creep_params['n_stress_exponent']
    Q = creep_params['Q_activation_energy_J_mol']
    R = 8.314  # J/mol?K
    
    # ????
    T = 1073  # K (800?C)
    sigma = 50  # MPa
    
    epsilon_dot = A * (sigma ** n) * np.exp(-Q / (R * T))
    print(f"\n  ??: {T-273.15:.1f}?C ({T} K)")
    print(f"  ??: {sigma} MPa")
    print(f"  ?????: {epsilon_dot:.2e} s??")

def example_3_cte_interpolation():
    """??3: CTE????"""
    print("\n" + "=" * 60)
    print("??3: CTE????")
    print("=" * 60)
    
    dataset = load_dataset()
    
    # ?????CTE??
    cte_data = dataset['cte_data']['electrolyte']
    T = np.array(cte_data['temperature_K'])
    CTE = np.array(cte_data['CTE_K_inv'])
    
    # ??????
    cte_interp = interp1d(T, CTE, kind='linear', fill_value='extrapolate')
    
    # ???????CTE
    target_temps = [298.15, 573.15, 873.15, 1073.15]  # 25?C, 300?C, 600?C, 800?C
    
    print("\n???CTE????:")
    print(f"{'?? (?C)':<15} {'CTE (?10?? K??)':<20}")
    print("-" * 35)
    for T_K in target_temps:
        cte_value = cte_interp(T_K)
        print(f"{T_K-273.15:<15.1f} {cte_value*1e6:<20.4f}")

def example_4_elastic_properties():
    """??4: ??????"""
    print("\n" + "=" * 60)
    print("??4: ??????")
    print("=" * 60)
    
    dataset = load_dataset()
    
    # ??CSV??
    elastic_df = pd.read_csv('material_dataset/csv_files/elastic_properties_T_electrolyte.csv')
    
    print("\n????????? (?5?):")
    print(elastic_df.head())
    
    # ??????
    T = elastic_df['temperature_K'].values
    E = elastic_df['youngs_modulus_GPa'].values
    nu = elastic_df['poissons_ratio'].values
    
    E_interp = interp1d(T, E, kind='linear', fill_value='extrapolate')
    nu_interp = interp1d(T, nu, kind='linear', fill_value='extrapolate')
    
    # ???????????
    target_temp = 1073.15  # 800?C
    
    E_value = E_interp(target_temp)
    nu_value = nu_interp(target_temp)
    
    print(f"\n? {target_temp-273.15:.1f}?C ??????:")
    print(f"  ????: {E_value:.1f} GPa")
    print(f"  ???: {nu_value:.3f}")

def example_5_sintering_kinetics():
    """??5: ???????"""
    print("\n" + "=" * 60)
    print("??5: ???????")
    print("=" * 60)
    
    dataset = load_dataset()
    
    # ?????????
    sintering_df = pd.read_csv('material_dataset/csv_files/sintering_kinetics_electrolyte.csv')
    
    print("\n??????????:")
    print(f"  ????: {len(sintering_df)}")
    print(f"  ????: {sintering_df['temperature_K'].min()-273.15:.1f} - {sintering_df['temperature_K'].max()-273.15:.1f} ?C")
    print(f"  ?????: {sintering_df['shrinkage'].max()*100:.2f}%")
    
    # ???????
    print("\n?????????:")
    key_temps = [1000, 1200, 1400]  # ?C
    for T_C in key_temps:
        T_K = T_C + 273.15
        closest_idx = (sintering_df['temperature_K'] - T_K).abs().idxmin()
        shrinkage = sintering_df.loc[closest_idx, 'shrinkage'] * 100
        print(f"  {T_C}?C: {shrinkage:.2f}%")

def example_6_creep_data_analysis():
    """??6: ??????"""
    print("\n" + "=" * 60)
    print("??6: ??????")
    print("=" * 60)
    
    # ??????
    creep_df = pd.read_csv('material_dataset/csv_files/creep_electrolyte_sintered.csv')
    
    print("\n???????????:")
    print(f"  ?????: {len(creep_df)}")
    print(f"  ????: {creep_df['temperature_C'].min():.1f} - {creep_df['temperature_C'].max():.1f} ?C")
    print(f"  ????: {creep_df['stress_MPa'].min():.1f} - {creep_df['stress_MPa'].max():.1f} MPa")
    
    # ???????
    print("\n??????????? (50 MPa):")
    for T_C in [800, 1000, 1200]:
        T_K = T_C + 273.15
        subset = creep_df[(creep_df['temperature_K'] == T_K) & (creep_df['stress_MPa'] == 50)]
        if len(subset) > 0:
            strain_rate = subset['strain_rate_s_inv'].values[0]
            print(f"  {T_C}?C: {strain_rate:.2e} s??")

def example_7_plot_example():
    """??7: ???????"""
    print("\n" + "=" * 60)
    print("??7: ???????")
    print("=" * 60)
    
    # ??CTE??
    cte_df = pd.read_csv('material_dataset/csv_files/cte_electrolyte.csv')
    
    # ??????
    plt.figure(figsize=(10, 6))
    plt.plot(cte_df['temperature_K'] - 273.15, cte_df['CTE_K_inv'] * 1e6, 
             linewidth=2, label='Electrolyte (8YSZ)')
    plt.xlabel('Temperature (?C)')
    plt.ylabel('CTE (?10?? K??)')
    plt.title('Coefficient of Thermal Expansion vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('material_dataset/cte_example_plot.png', dpi=150)
    plt.close()
    
    print("\n??????: material_dataset/cte_example_plot.png")

def main():
    """??????"""
    print("\n" + "=" * 60)
    print("SOFC?????????")
    print("=" * 60)
    
    try:
        example_1_basic_properties()
        example_2_creep_parameters()
        example_3_cte_interpolation()
        example_4_elastic_properties()
        example_5_sintering_kinetics()
        example_6_creep_data_analysis()
        example_7_plot_example()
        
        print("\n" + "=" * 60)
        print("?????????")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n??: ????????!")
        print("???? generate_material_dataset.py ?????")
        print(f"????: {e}")
    except Exception as e:
        print(f"\n??: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
