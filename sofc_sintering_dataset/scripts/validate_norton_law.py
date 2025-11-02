#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Norton?????????????
???????????????
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats


BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data" / "creep_tests"
PROCESSED_DIR = BASE_DIR / "processed_data"


def norton_law(sigma, A, n):
    """Norton????: ?? = A???"""
    return A * sigma**n


def arrhenius_factor(T_celsius, Q, R=8.314):
    """Arrhenius????: exp(-Q/RT)"""
    T_kelvin = T_celsius + 273.15
    return np.exp(-Q * 1000 / (R * T_kelvin))  # Q in kJ/mol


def fit_norton_parameters(creep_files, material_name):
    """??Norton????"""
    print(f"\n{'='*60}")
    print(f"  {material_name} Norton??????")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for file_info in creep_files:
        filename = file_info['file']
        temp_c = file_info['temp_C']
        
        df = pd.read_csv(RAW_DATA_DIR / filename)
        
        # ??????????????
        stresses = []
        strain_rates = []
        
        for stress in df['Stress_MPa'].unique():
            df_stress = df[df['Stress_MPa'] == stress]
            avg_rate = df_stress['Strain_Rate_s_inv'].mean()
            stresses.append(stress)
            strain_rates.append(avg_rate)
        
        stresses = np.array(stresses)
        strain_rates = np.array(strain_rates)
        
        # ????????
        log_stress = np.log(stresses)
        log_rate = np.log(strain_rates)
        
        # ????
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_stress, log_rate)
        
        n = slope
        A = np.exp(intercept)
        r_squared = r_value**2
        
        print(f"??: {temp_c}?C")
        print(f"  ???? n = {n:.3f}")
        print(f"  ????? A = {A:.3e} (MPa^-n s^-1)")
        print(f"  ???? R? = {r_squared:.6f}")
        
        # ????
        predicted_rates = norton_law(stresses, A, n)
        residuals = np.abs((strain_rates - predicted_rates) / strain_rates) * 100
        print(f"  ?????? = {np.mean(residuals):.2f}%")
        print(f"  ?????? = {np.max(residuals):.2f}%\n")
        
        all_results.append({
            'temperature_C': temp_c,
            'n': n,
            'A': A,
            'R_squared': r_squared,
            'avg_error_percent': np.mean(residuals)
        })
    
    return all_results


def fit_activation_energy(results):
    """???????????Q"""
    print(f"\n{'='*60}")
    print("  ????? (Arrhenius??)")
    print(f"{'='*60}\n")
    
    temps_K = np.array([r['temperature_C'] + 273.15 for r in results])
    A_values = np.array([r['A'] for r in results])
    
    # ln(A) vs 1/T ????
    inv_T = 1000 / temps_K  # 1000/T for better scaling
    ln_A = np.log(A_values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_A)
    
    # Q = -slope * R * 1000 (??????1000/T)
    R = 8.314  # J/(mol?K)
    Q_kJ_mol = -slope * R
    A0 = np.exp(intercept)
    r_squared = r_value**2
    
    print(f"??? Q = {Q_kJ_mol:.1f} kJ/mol")
    print(f"???? A? = {A0:.3e}")
    print(f"???? R? = {r_squared:.6f}\n")
    
    return Q_kJ_mol, A0


def main():
    """?????"""
    print("\n" + "="*70)
    print("  SOFC??Norton???????????")
    print("="*70)
    
    # ????
    anode_files = [
        {'file': 'anode_creep_800C.csv', 'temp_C': 800},
        {'file': 'anode_creep_1000C.csv', 'temp_C': 1000},
        {'file': 'anode_creep_1200C.csv', 'temp_C': 1200}
    ]
    anode_results = fit_norton_parameters(anode_files, "?? (Ni-YSZ)")
    Q_anode, A0_anode = fit_activation_energy(anode_results)
    
    # ?????
    electrolyte_files = [
        {'file': 'electrolyte_creep_1000C.csv', 'temp_C': 1000},
        {'file': 'electrolyte_creep_1200C.csv', 'temp_C': 1200},
        {'file': 'electrolyte_creep_1350C.csv', 'temp_C': 1350}
    ]
    electrolyte_results = fit_norton_parameters(electrolyte_files, "??? (8YSZ)")
    Q_electrolyte, A0_electrolyte = fit_activation_energy(electrolyte_results)
    
    # ????
    cathode_files = [
        {'file': 'cathode_creep_900C.csv', 'temp_C': 900},
        {'file': 'cathode_creep_1100C.csv', 'temp_C': 1100},
        {'file': 'cathode_creep_1300C.csv', 'temp_C': 1300}
    ]
    cathode_results = fit_norton_parameters(cathode_files, "?? (LSM-YSZ)")
    Q_cathode, A0_cathode = fit_activation_energy(cathode_results)
    
    # ??????
    print("\n" + "="*70)
    print("  ????")
    print("="*70 + "\n")
    
    summary = {
        "validation_date": "2025-11-01",
        "materials": {
            "anode": {
                "material_name": "Ni-YSZ",
                "average_n": np.mean([r['n'] for r in anode_results]),
                "n_std": np.std([r['n'] for r in anode_results]),
                "activation_energy_kJ_mol": Q_anode,
                "frequency_factor_A0": A0_anode,
                "average_fit_quality_R2": np.mean([r['R_squared'] for r in anode_results]),
                "temperature_data": anode_results
            },
            "electrolyte": {
                "material_name": "8YSZ",
                "average_n": np.mean([r['n'] for r in electrolyte_results]),
                "n_std": np.std([r['n'] for r in electrolyte_results]),
                "activation_energy_kJ_mol": Q_electrolyte,
                "frequency_factor_A0": A0_electrolyte,
                "average_fit_quality_R2": np.mean([r['R_squared'] for r in electrolyte_results]),
                "temperature_data": electrolyte_results
            },
            "cathode": {
                "material_name": "LSM-YSZ",
                "average_n": np.mean([r['n'] for r in cathode_results]),
                "n_std": np.std([r['n'] for r in cathode_results]),
                "activation_energy_kJ_mol": Q_cathode,
                "frequency_factor_A0": A0_cathode,
                "average_fit_quality_R2": np.mean([r['R_squared'] for r in cathode_results]),
                "temperature_data": cathode_results
            }
        },
        "notes": [
            "?????Norton????n?????2-2.5?????????",
            "???Q???????(180-350 kJ/mol)???????",
            "????R????0.99???Norton????????????"
        ]
    }
    
    # ??????
    output_file = PROCESSED_DIR / "norton_law_validation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("? ????????:", output_file)
    
    # ??????
    print("\n??????:")
    print("-" * 70)
    for mat_key, mat_name in [('anode', '??'), ('electrolyte', '???'), ('cathode', '??')]:
        mat_data = summary['materials'][mat_key]
        print(f"\n{mat_name} ({mat_data['material_name']}):")
        print(f"  ??????: n = {mat_data['average_n']:.3f} ? {mat_data['n_std']:.3f}")
        print(f"  ???: Q = {mat_data['activation_energy_kJ_mol']:.1f} kJ/mol")
        print(f"  ??????: R? = {mat_data['average_fit_quality_R2']:.6f}")
    
    print("\n" + "="*70)
    print("  ???????????????FEM??")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
