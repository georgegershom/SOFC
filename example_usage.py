#!/usr/bin/env python3
"""
SOFC???????
???????????????
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("SOFC???????")
print("="*60)

# ============================================================
# 1. ????????
# ============================================================
print("\n1. ????????")
print("-"*60)

with open('sofc_dataset/thermal_physical/green_state_properties.json', 'r') as f:
    green_props = json.load(f)

for layer, props in green_props.items():
    print(f"\n{layer.upper()}:")
    print(f"  ??: {props['material']}")
    print(f"  ????: {props['density_green']:.2f} g/cm?")
    print(f"  ???: {props['porosity_green']:.1f} %")
    print(f"  ????: {props['relative_density']:.1f} %")

# ============================================================
# 2. ??Norton????
# ============================================================
print("\n\n2. Norton??????")
print("-"*60)
print("?? = A ? ?? ? exp(-Q/RT)\n")

with open('sofc_dataset/constitutive_parameters/norton_law_parameters.json', 'r') as f:
    norton = json.load(f)

for layer, params in norton.items():
    print(f"{layer.upper()}:")
    print(f"  A = {params['A_pre_exponential']:.2e} 1/(MPa^n?s)")
    print(f"  n = {params['n_stress_exponent']:.2f}")
    print(f"  Q = {params['Q_activation_energy']:.1f} kJ/mol")
    print()

# ============================================================
# 3. ????????????
# ============================================================
print("\n3. ????????")
print("-"*60)

# ????
T_celsius = 1200
sigma_MPa = 2.0

print(f"????: T = {T_celsius}?C, ? = {sigma_MPa} MPa\n")

R = 8.314  # J/(mol?K)
T_K = T_celsius + 273.15

for layer in ['anode', 'electrolyte', 'cathode']:
    params = norton[layer]
    A = params['A_pre_exponential']
    n = params['n_stress_exponent']
    Q = params['Q_activation_energy']
    
    # Norton??
    creep_rate = A * (sigma_MPa ** n) * np.exp(-Q * 1000 / (R * T_K))
    
    print(f"{layer.capitalize()}: {creep_rate:.3e} 1/s = {creep_rate*3600:.3e} 1/h")

# ============================================================
# 4. ?????CTE??
# ============================================================
print("\n\n4. ?????(CTE)??")
print("-"*60)

cte_at_1000 = {}

for layer in ['anode', 'electrolyte', 'cathode', 'interconnect']:
    df = pd.read_csv(f'sofc_dataset/thermal_physical/{layer}_CTE.csv')
    
    # ??1000?C???CTE?
    idx = np.argmin(np.abs(df['temperature_C'] - 1000))
    cte = df.loc[idx, 'CTE_ppm_per_K']
    cte_at_1000[layer] = cte
    
    print(f"{layer.capitalize():15s}: {cte:.2f} ppm/K @ 1000?C")

# ??CTE???
print("\nCTE?????:")
mismatch_ae = cte_at_1000['anode'] - cte_at_1000['electrolyte']
mismatch_ec = cte_at_1000['electrolyte'] - cte_at_1000['cathode']

print(f"  ??-???: {mismatch_ae:+.2f} ppm/K")
print(f"  ???-??: {mismatch_ec:+.2f} ppm/K")

# ============================================================
# 5. ????????
# ============================================================
print("\n\n5. ??????")
print("-"*60)

for layer in ['anode', 'electrolyte', 'cathode']:
    df = pd.read_csv(f'sofc_dataset/sintering_kinetics/{layer}_dilatometry.csv')
    
    max_shrinkage = df['linear_shrinkage_percent'].max()
    
    # ????????(???>1%)
    idx_start = df[df['linear_shrinkage_percent'] > 1.0].index[0]
    T_start = df.loc[idx_start, 'temperature_C']
    
    # ????90%???????
    target_shrinkage = 0.9 * max_shrinkage
    idx_90 = np.argmin(np.abs(df['linear_shrinkage_percent'] - target_shrinkage))
    T_90 = df.loc[idx_90, 'temperature_C']
    
    print(f"{layer.capitalize()}:")
    print(f"  ????: {max_shrinkage:.1f} %")
    print(f"  ????: {T_start:.0f}?C")
    print(f"  90%????: {T_90:.0f}?C")

# ============================================================
# 6. ??????
# ============================================================
print("\n\n6. ?????????")
print("-"*60)

bilayers = [
    ('anode', 'electrolyte'),
    ('electrolyte', 'cathode')
]

for layer1, layer2 in bilayers:
    filename = f'sofc_dataset/sintering_kinetics/{layer1}_{layer2}_bilayer.csv'
    df = pd.read_csv(filename)
    
    max_stress = df['estimated_stress_MPa'].max()
    min_stress = df['estimated_stress_MPa'].min()
    idx_max = df['estimated_stress_MPa'].idxmax()
    T_max_stress = df.loc[idx_max, 'temperature_C']
    
    print(f"{layer1.capitalize()}-{layer2.capitalize()}:")
    print(f"  ?????: {max_stress:.1f} MPa @ {T_max_stress:.0f}?C")
    print(f"  ?????: {min_stress:.1f} MPa")
    print(f"  ????: {max_stress - min_stress:.1f} MPa")

# ============================================================
# 7. ??????
# ============================================================
print("\n\n7. ???? (???, 1000?C)")
print("-"*60)

for layer in ['anode', 'electrolyte', 'cathode', 'interconnect']:
    df = pd.read_csv(f'sofc_dataset/elastic_properties/{layer}_elastic_properties.csv')
    
    # ??: ????>0.95, ????1000?C
    df_filtered = df[(df['relative_density'] > 0.95) & 
                      (np.abs(df['temperature_C'] - 1000) < 50)]
    
    if len(df_filtered) > 0:
        E = df_filtered['youngs_modulus_GPa'].mean()
        nu = df_filtered['poisson_ratio'].mean()
        G = df_filtered['shear_modulus_GPa'].mean()
        
        print(f"{layer.capitalize()}:")
        print(f"  ???? E = {E:.1f} GPa")
        print(f"  ??? ? = {nu:.3f}")
        print(f"  ???? G = {G:.1f} GPa")

# ============================================================
# 8. ????????
# ============================================================
print("\n\n8. ????????")
print("-"*60)

for layer in ['anode', 'electrolyte', 'cathode']:
    with open(f'sofc_dataset/creep_data/{layer}_creep_tests.json', 'r') as f:
        creep_tests = json.load(f)
    
    # ??????
    temps = set()
    stresses = set()
    
    for key in creep_tests.keys():
        temp = creep_tests[key]['temperature_C']
        stress = creep_tests[key]['stress_MPa']
        temps.add(temp)
        stresses.add(stress)
    
    print(f"{layer.capitalize()}:")
    print(f"  ????: {len(creep_tests)}")
    print(f"  ???: {len(temps)} ({min(temps):.0f}-{max(temps):.0f}?C)")
    print(f"  ???: {len(stresses)} ({min(stresses):.1f}-{max(stresses):.1f} MPa)")

# ============================================================
# 9. ??????
# ============================================================
print("\n\n9. ??????")
print("-"*60)

for layer in ['anode', 'electrolyte', 'cathode']:
    df = pd.read_csv(f'sofc_dataset/sintering_kinetics/{layer}_density_evolution.csv')
    
    # ????95%?????????
    df_95 = df[df['relative_density'] >= 0.95]
    
    if len(df_95) > 0:
        idx_95 = df_95.index[0]
        time_95 = df.loc[idx_95, 'time_minutes']
        temp_95 = df.loc[idx_95, 'temperature_C']
        
        print(f"{layer.capitalize()}: ??95%???")
        print(f"  ??: {time_95:.0f} ?? ({time_95/60:.1f} ??)")
        print(f"  ??: {temp_95:.0f}?C")

# ============================================================
# 10. ???????
# ============================================================
print("\n\n10. ????????")
print("-"*60)

import os

file_checks = {
    '????': 'sofc_dataset/thermal_physical/green_state_properties.json',
    'Norton??': 'sofc_dataset/constitutive_parameters/norton_law_parameters.json',
    '???': 'sofc_dataset/dataset_metadata.json',
}

print("??????:")
for name, path in file_checks.items():
    exists = "?" if os.path.exists(path) else "?"
    print(f"  {exists} {name}")

# ??????
file_counts = {
    'CTE??': len([f for f in os.listdir('sofc_dataset/thermal_physical') if f.endswith('_CTE.csv')]),
    '????': len([f for f in os.listdir('sofc_dataset/sintering_kinetics') if 'dilatometry' in f]),
    '????': len([f for f in os.listdir('sofc_dataset/creep_data') if f.endswith('.json')]),
    '????': len([f for f in os.listdir('sofc_dataset/elastic_properties') if f.endswith('.csv')]),
    '?????': len([f for f in os.listdir('sofc_dataset/visualization') if f.endswith('.png')]),
}

print("\n??????:")
for name, count in file_counts.items():
    print(f"  {name}: {count} ?")

print("\n" + "="*60)
print("?????????!")
print("="*60)
print("\n??:")
print("  - ?? QUICK_START_GUIDE.md ????????")
print("  - ?? DATASET_README.md ????????")
print("  - ?? sofc_dataset/visualization/ ?????????")
print("\n??????,???FEM??! ??")
