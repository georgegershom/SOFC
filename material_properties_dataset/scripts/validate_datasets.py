#!/usr/bin/env python3
"""
Validate the generated datasets for consistency and physical realism
"""

import pandas as pd
import numpy as np
import json
import os

def validate_mechanical_properties():
    """Validate mechanical properties dataset"""
    print("\n🔍 Validating Mechanical Properties...")
    
    df = pd.read_csv('../mechanical/mechanical_properties.csv')
    
    checks = []
    
    # Check 1: Young's modulus decreases with temperature
    E_trend = np.corrcoef(df['Temperature_C'], df['Youngs_Modulus_GPa'])[0,1]
    checks.append(('Young\'s modulus decreases with T', E_trend < 0))
    
    # Check 2: Tensile strength decreases with temperature
    TS_trend = np.corrcoef(df['Temperature_C'], df['Tensile_Strength_MPa'])[0,1]
    checks.append(('Tensile strength decreases with T', TS_trend < 0))
    
    # Check 3: Poisson's ratio in physical range
    nu_valid = (df['Poissons_Ratio'].min() > 0) and (df['Poissons_Ratio'].max() < 0.5)
    checks.append(('Poisson\'s ratio in (0, 0.5)', nu_valid))
    
    # Check 4: Yield < Tensile strength
    yield_valid = (df['Yield_Strength_MPa'] < df['Tensile_Strength_MPa']).all()
    checks.append(('Yield strength < Tensile strength', yield_valid))
    
    # Check 5: No negative values
    no_negatives = (df.select_dtypes(include=[np.number]) >= 0).all().all()
    checks.append(('No negative values', no_negatives))
    
    return checks

def validate_creep_properties():
    """Validate creep properties dataset"""
    print("\n🔍 Validating Creep Properties...")
    
    df = pd.read_csv('../creep/creep_curves_full.csv')
    
    checks = []
    
    # Check 1: Creep strain increases with time
    for test_id in df['Test_ID'].unique()[:5]:  # Check first 5 tests
        test_data = df[df['Test_ID'] == test_id]
        strain_increasing = test_data['Creep_Strain_Percent'].diff().dropna() >= -0.001  # Allow tiny noise
        checks.append((f'{test_id}: Strain increases', strain_increasing.mean() > 0.95))
    
    # Check 2: Higher temperature = higher creep rate (at same stress)
    stress = 200  # MPa
    temps = [700, 800]
    rates = []
    for temp in temps:
        data = df[(df['Temperature_C'] == temp) & (df['Stress_MPa'] == stress)]
        if not data.empty:
            rates.append(abs(data['Min_Creep_Rate_Per_Hour'].iloc[0]))
    if len(rates) == 2:
        checks.append(('Higher T = higher creep rate', rates[1] > rates[0]))
    
    # Check 3: No negative strain
    no_negative_strain = (df['Creep_Strain_Percent'] >= 0).all()
    checks.append(('No negative creep strain', no_negative_strain))
    
    return checks

def validate_thermophysical_properties():
    """Validate thermophysical properties dataset"""
    print("\n🔍 Validating Thermophysical Properties...")
    
    df = pd.read_csv('../thermophysical/thermophysical_properties.csv')
    
    checks = []
    
    # Check 1: CTE increases with temperature
    CTE_trend = np.corrcoef(df['Temperature_C'], df['CTE_ppm_per_K'])[0,1]
    checks.append(('CTE increases with T', CTE_trend > 0))
    
    # Check 2: Thermal conductivity positive trend
    k_trend = np.corrcoef(df['Temperature_C'], df['Thermal_Conductivity_W_per_mK'])[0,1]
    checks.append(('Thermal conductivity increases with T', k_trend > 0))
    
    # Check 3: Specific heat in reasonable range
    Cp_valid = (df['Specific_Heat_J_per_kgK'] > 300) & (df['Specific_Heat_J_per_kgK'] < 1000)
    checks.append(('Specific heat in (300, 1000) J/kg·K', Cp_valid.all()))
    
    # Check 4: Linear expansion is cumulative
    expansion_increasing = df['Linear_Expansion_Percent'].diff().dropna() >= 0
    checks.append(('Linear expansion increases', expansion_increasing.all()))
    
    # Check 5: Thermal diffusivity calculated correctly
    # α = k / (ρ * Cp)
    density = df['Density_kg_per_m3'].iloc[0]
    calculated_alpha = df['Thermal_Conductivity_W_per_mK'] / (density * df['Specific_Heat_J_per_kgK']) * 1e6
    alpha_match = np.allclose(df['Thermal_Diffusivity_mm2_per_s'], calculated_alpha, rtol=0.1)
    checks.append(('Thermal diffusivity consistent', alpha_match))
    
    return checks

def validate_electrochemical_properties():
    """Validate electrochemical properties dataset"""
    print("\n🔍 Validating Electrochemical Properties...")
    
    df_cond = pd.read_csv('../electrochemical/electrochemical_conductivity.csv')
    df_corr = pd.read_csv('../electrochemical/electrochemical_corrosion.csv')
    
    checks = []
    
    # Check 1: Transference numbers sum to 1
    t_sum = df_cond['Ionic_Transference_Number'] + df_cond['Electronic_Transference_Number']
    t_valid = np.allclose(t_sum, 1.0, atol=0.01)
    checks.append(('Transference numbers sum to 1', t_valid))
    
    # Check 2: Conductivities positive
    cond_positive = (df_cond['Electronic_Conductivity_S_per_cm'] > 0).all() and \
                   (df_cond['Ionic_Conductivity_S_per_cm'] > 0).all()
    checks.append(('All conductivities positive', cond_positive))
    
    # Check 3: Ionic conductivity increases with temperature
    air_data = df_cond[df_cond['Environment'] == 'Air']
    if not air_data.empty:
        ionic_trend = np.corrcoef(air_data['Temperature_K'], 
                                  np.log10(air_data['Ionic_Conductivity_S_per_cm']))[0,1]
        checks.append(('Ionic conductivity Arrhenius behavior', ionic_trend > 0))
    
    # Check 4: Corrosion rate increases with temperature
    seawater = df_corr[df_corr['Electrolyte'] == 'Simulated_Seawater']
    if not seawater.empty:
        corr_trend = np.corrcoef(seawater['Temperature_C'], 
                                 seawater['Corrosion_Rate_mm_per_year'])[0,1]
        checks.append(('Corrosion rate increases with T', corr_trend > 0))
    
    # Check 5: pH affects corrosion potential
    pH_values = df_corr['pH'].unique()
    if len(pH_values) > 2:
        pH_effect = len(df_corr.groupby('pH')['Corrosion_Potential_V_vs_SCE'].mean().unique()) > 1
        checks.append(('pH affects corrosion potential', pH_effect))
    
    return checks

def main():
    """Run all validation checks"""
    
    print("="*60)
    print("🔬 Material Properties Dataset Validation")
    print("="*60)
    
    all_checks = []
    
    # Run all validations
    if os.path.exists('../mechanical/mechanical_properties.csv'):
        all_checks.extend(validate_mechanical_properties())
    
    if os.path.exists('../creep/creep_curves_full.csv'):
        all_checks.extend(validate_creep_properties())
    
    if os.path.exists('../thermophysical/thermophysical_properties.csv'):
        all_checks.extend(validate_thermophysical_properties())
    
    if os.path.exists('../electrochemical/electrochemical_conductivity.csv'):
        all_checks.extend(validate_electrochemical_properties())
    
    # Summary
    print("\n" + "="*60)
    print("📋 Validation Summary")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for check_name, check_result in all_checks:
        status = "✅ PASS" if check_result else "❌ FAIL"
        print(f"{status}: {check_name}")
        if check_result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"Total Checks: {len(all_checks)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All validation checks passed! Dataset is physically consistent.")
    else:
        print(f"\n⚠️ {failed} checks failed. Please review the data generation.")
    
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()