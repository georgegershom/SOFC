#!/usr/bin/env python3
"""
Dataset Validation Script for Fire-Resistant Structural Elements Research

This script performs comprehensive validation checks on the generated synthetic dataset
to ensure scientific plausibility and internal consistency.
"""

import pandas as pd
import numpy as np
import json

def validate_dataset():
    """Perform comprehensive dataset validation"""
    print("="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    
    # Load datasets
    df_ambient = pd.read_csv('datasets/ambient_properties.csv')
    df_residual = pd.read_csv('datasets/residual_properties_high_temp.csv')
    df_in_situ = pd.read_csv('datasets/in_situ_properties.csv')
    df_pore_pressure = pd.read_csv('datasets/pore_pressure_summary.csv')
    
    with open('datasets/stress_strain_curves.json', 'r') as f:
        stress_strain_curves = json.load(f)
    
    print(f"Loaded datasets:")
    print(f"- Ambient: {len(df_ambient)} specimens")
    print(f"- Residual: {len(df_residual)} specimens") 
    print(f"- In-Situ: {len(df_in_situ)} specimens")
    print(f"- Pore Pressure: {len(df_pore_pressure)} configurations")
    print(f"- Stress-Strain Curves: {len(stress_strain_curves)} curves")
    print()
    
    # Validation checks
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Strength decreases with temperature
    print("Check 1: Strength degradation with temperature")
    total_checks += 1
    
    control_data = df_residual[(df_residual['Mix_ID'] == 'C') & 
                              (df_residual['Cooling_Method'] == 'Furnace') &
                              (df_residual['Heating_Rate'] == '5_C_per_min')]
    
    temp_strength = control_data.groupby('Peak_Temperature_C')['Residual_Compressive_Strength_MPa'].mean()
    strength_decreasing = all(temp_strength.iloc[i] >= temp_strength.iloc[i+1] 
                             for i in range(len(temp_strength)-1))
    
    if strength_decreasing:
        print("✅ PASS: Strength decreases monotonically with temperature")
        checks_passed += 1
    else:
        print("❌ FAIL: Strength does not decrease monotonically")
    
    # Check 2: Mass loss increases with temperature
    print("\nCheck 2: Mass loss increases with temperature")
    total_checks += 1
    
    temp_mass_loss = control_data.groupby('Peak_Temperature_C')['Mass_Loss_pct'].mean()
    mass_loss_increasing = all(temp_mass_loss.iloc[i] <= temp_mass_loss.iloc[i+1] 
                              for i in range(len(temp_mass_loss)-1))
    
    if mass_loss_increasing:
        print("✅ PASS: Mass loss increases with temperature")
        checks_passed += 1
    else:
        print("❌ FAIL: Mass loss does not increase with temperature")
    
    # Check 3: Rubber content affects properties
    print("\nCheck 3: Rubber content effects on ambient strength")
    total_checks += 1
    
    ambient_28d = df_ambient[df_ambient['Curing_Age_days'] == 28]
    mix_strength = ambient_28d.groupby('Mix_ID')['Compressive_Strength_MPa'].mean()
    
    # Control should be strongest
    control_strongest = mix_strength['C'] == mix_strength.max()
    
    if control_strongest:
        print("✅ PASS: Control mix has highest ambient strength")
        checks_passed += 1
    else:
        print("❌ FAIL: Control mix is not the strongest")
    
    # Check 4: Quenching causes additional damage
    print("\nCheck 4: Quenching vs furnace cooling effects")
    total_checks += 1
    
    temp_400_data = df_residual[(df_residual['Peak_Temperature_C'] == 400) &
                               (df_residual['Mix_ID'] == 'C') &
                               (df_residual['Heating_Rate'] == '5_C_per_min')]
    
    furnace_strength = temp_400_data[temp_400_data['Cooling_Method'] == 'Furnace']['Residual_Compressive_Strength_MPa'].mean()
    quench_strength = temp_400_data[temp_400_data['Cooling_Method'] == 'Quench']['Residual_Compressive_Strength_MPa'].mean()
    
    quench_more_damage = quench_strength < furnace_strength
    
    if quench_more_damage:
        print("✅ PASS: Quenching causes more damage than furnace cooling")
        checks_passed += 1
    else:
        print("❌ FAIL: Quenching effect not evident")
    
    # Check 5: Spalling occurs mainly in control mix
    print("\nCheck 5: Spalling occurrence patterns")
    total_checks += 1
    
    rapid_heating_data = df_residual[df_residual['Heating_Rate'] == '10_C_per_min']
    if not rapid_heating_data.empty:
        spalling_by_mix = rapid_heating_data.groupby('Mix_ID')['Spalling_Occurred'].mean()
        control_spalling_highest = spalling_by_mix['C'] == spalling_by_mix.max()
        
        if control_spalling_highest:
            print("✅ PASS: Control mix shows highest spalling tendency")
            checks_passed += 1
        else:
            print("❌ FAIL: Spalling pattern not as expected")
    else:
        print("⚠️  SKIP: No rapid heating data available")
    
    # Check 6: UPV correlates with strength
    print("\nCheck 6: UPV-strength correlation")
    total_checks += 1
    
    correlation = df_residual['UPV_mps'].corr(df_residual['Residual_Compressive_Strength_MPa'])
    strong_correlation = correlation > 0.7
    
    if strong_correlation:
        print(f"✅ PASS: Strong UPV-strength correlation (r = {correlation:.3f})")
        checks_passed += 1
    else:
        print(f"❌ FAIL: Weak UPV-strength correlation (r = {correlation:.3f})")
    
    # Check 7: Realistic property ranges
    print("\nCheck 7: Realistic property ranges")
    total_checks += 1
    
    # Check ambient strength range (should be 20-70 MPa for concrete)
    ambient_strength_range = (df_ambient['Compressive_Strength_MPa'].min() > 20 and 
                             df_ambient['Compressive_Strength_MPa'].max() < 80)
    
    # Check UPV range (should be 3000-5000 m/s for concrete)
    upv_range = (df_ambient['UPV_mps'].min() > 3000 and 
                df_ambient['UPV_mps'].max() < 5500)
    
    if ambient_strength_range and upv_range:
        print("✅ PASS: All properties within realistic ranges")
        checks_passed += 1
    else:
        print("❌ FAIL: Some properties outside realistic ranges")
    
    # Check 8: Coefficient of variation
    print("\nCheck 8: Realistic variability (COV)")
    total_checks += 1
    
    control_ambient = df_ambient[(df_ambient['Mix_ID'] == 'C') & 
                                (df_ambient['Curing_Age_days'] == 28)]
    
    cov_strength = control_ambient['Compressive_Strength_MPa'].std() / control_ambient['Compressive_Strength_MPa'].mean()
    realistic_cov = 0.03 < cov_strength < 0.10  # 3-10% is realistic for concrete
    
    if realistic_cov:
        print(f"✅ PASS: Realistic COV for strength ({cov_strength:.3f})")
        checks_passed += 1
    else:
        print(f"❌ FAIL: Unrealistic COV for strength ({cov_strength:.3f})")
    
    # Check 9: Stress-strain curve validity
    print("\nCheck 9: Stress-strain curve validity")
    total_checks += 1
    
    valid_curves = 0
    for specimen_id, curve_data in stress_strain_curves.items():
        strains = np.array(curve_data['strain'])
        stresses = np.array(curve_data['stress'])
        
        # Check monotonic strain increase
        strain_monotonic = all(strains[i] <= strains[i+1] for i in range(len(strains)-1))
        
        # Check peak stress occurs before end
        peak_index = np.argmax(stresses)
        peak_not_at_end = peak_index < len(stresses) - 5
        
        if strain_monotonic and peak_not_at_end:
            valid_curves += 1
    
    curve_validity = valid_curves / len(stress_strain_curves) > 0.9
    
    if curve_validity:
        print(f"✅ PASS: {valid_curves}/{len(stress_strain_curves)} curves are valid")
        checks_passed += 1
    else:
        print(f"❌ FAIL: Only {valid_curves}/{len(stress_strain_curves)} curves are valid")
    
    # Check 10: Data completeness
    print("\nCheck 10: Data completeness")
    total_checks += 1
    
    missing_ambient = df_ambient.isnull().sum().sum()
    missing_residual = df_residual.isnull().sum().sum()
    missing_in_situ = df_in_situ.isnull().sum().sum()
    
    no_missing_data = (missing_ambient == 0 and missing_residual == 0 and missing_in_situ == 0)
    
    if no_missing_data:
        print("✅ PASS: No missing data found")
        checks_passed += 1
    else:
        print(f"❌ FAIL: Missing data found (A:{missing_ambient}, R:{missing_residual}, I:{missing_in_situ})")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Checks Passed: {checks_passed}/{total_checks}")
    print(f"Success Rate: {checks_passed/total_checks*100:.1f}%")
    
    if checks_passed == total_checks:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("Dataset is scientifically plausible and internally consistent.")
    elif checks_passed >= total_checks * 0.8:
        print("✅ DATASET QUALITY: GOOD")
        print("Minor issues detected but dataset is suitable for research.")
    else:
        print("⚠️  DATASET QUALITY: NEEDS REVIEW")
        print("Multiple validation issues detected.")
    
    print("="*60)
    
    return checks_passed, total_checks

if __name__ == "__main__":
    validate_dataset()