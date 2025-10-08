"""
Dataset Validation Script
Performs sanity checks on the YSZ material properties dataset
"""

import pandas as pd
import numpy as np
import sys

def validate_dataset():
    """Run validation checks on the material properties dataset."""
    
    print("="*70)
    print("VALIDATING YSZ MATERIAL PROPERTIES DATASET")
    print("="*70)
    
    passed = 0
    failed = 0
    warnings = 0
    
    # Load datasets
    try:
        props_df = pd.read_csv("ysz_material_properties.csv")
        weibull_df = pd.read_csv("weibull_parameters.csv")
        creep_df = pd.read_csv("creep_model_parameters.csv")
        print("✓ All CSV files loaded successfully\n")
        passed += 1
    except Exception as e:
        print(f"✗ Failed to load CSV files: {e}\n")
        failed += 1
        return
    
    # Check 1: Temperature range
    print("Check 1: Temperature Range")
    temp_min = props_df['Temperature_C'].min()
    temp_max = props_df['Temperature_C'].max()
    if temp_min <= 25 and temp_max >= 1400:
        print(f"  ✓ Temperature range: {temp_min}°C to {temp_max}°C (adequate)")
        passed += 1
    else:
        print(f"  ⚠ Temperature range: {temp_min}°C to {temp_max}°C (may be insufficient)")
        warnings += 1
    
    # Check 2: Monotonicity checks
    print("\nCheck 2: Physical Monotonicity")
    
    # Young's modulus should decrease with temperature
    E_diff = np.diff(props_df['Youngs_Modulus_GPa'])
    if np.all(E_diff <= 0):
        print("  ✓ Young's Modulus decreases with temperature (physically correct)")
        passed += 1
    else:
        print("  ✗ Young's Modulus does not monotonically decrease")
        failed += 1
    
    # CTE should increase with temperature
    CTE_diff = np.diff(props_df['CTE_1e-6_K'])
    if np.all(CTE_diff >= 0):
        print("  ✓ CTE increases with temperature (physically correct)")
        passed += 1
    else:
        print("  ⚠ CTE does not monotonically increase (unusual but possible)")
        warnings += 1
    
    # Density should decrease with temperature
    rho_diff = np.diff(props_df['Density_kg_m3'])
    if np.all(rho_diff <= 0):
        print("  ✓ Density decreases with temperature (physically correct)")
        passed += 1
    else:
        print("  ✗ Density does not monotonically decrease")
        failed += 1
    
    # Check 3: Value ranges
    print("\nCheck 3: Realistic Value Ranges")
    
    E_RT = props_df.loc[props_df['Temperature_C'] == 25, 'Youngs_Modulus_GPa'].values[0]
    if 180 <= E_RT <= 220:
        print(f"  ✓ Young's Modulus at RT: {E_RT} GPa (typical for YSZ)")
        passed += 1
    else:
        print(f"  ⚠ Young's Modulus at RT: {E_RT} GPa (verify against literature)")
        warnings += 1
    
    CTE_RT = props_df.loc[props_df['Temperature_C'] == 25, 'CTE_1e-6_K'].values[0]
    if 9.5 <= CTE_RT <= 11.0:
        print(f"  ✓ CTE at RT: {CTE_RT}×10⁻⁶/K (typical for 8YSZ)")
        passed += 1
    else:
        print(f"  ⚠ CTE at RT: {CTE_RT}×10⁻⁶/K (verify composition)")
        warnings += 1
    
    rho_RT = props_df.loc[props_df['Temperature_C'] == 25, 'Density_kg_m3'].values[0]
    if 5900 <= rho_RT <= 6100:
        print(f"  ✓ Density at RT: {rho_RT} kg/m³ (typical for dense YSZ)")
        passed += 1
    else:
        print(f"  ⚠ Density at RT: {rho_RT} kg/m³ (check porosity)")
        warnings += 1
    
    # Check 4: Poisson's ratio bounds
    print("\nCheck 4: Poisson's Ratio Bounds")
    nu_min = props_df['Poissons_Ratio'].min()
    nu_max = props_df['Poissons_Ratio'].max()
    if 0 < nu_min and nu_max < 0.5:
        print(f"  ✓ Poisson's ratio range: {nu_min:.3f} to {nu_max:.3f} (physically valid)")
        passed += 1
    else:
        print(f"  ✗ Poisson's ratio range: {nu_min:.3f} to {nu_max:.3f} (unphysical)")
        failed += 1
    
    # Check 5: Weibull parameters
    print("\nCheck 5: Weibull Statistical Parameters")
    m_RT = weibull_df.loc[weibull_df['Temperature_C'] == 25, 'Weibull_Modulus_m'].values[0]
    if 5 <= m_RT <= 15:
        print(f"  ✓ Weibull modulus at RT: {m_RT} (typical for ceramics)")
        passed += 1
    else:
        print(f"  ⚠ Weibull modulus at RT: {m_RT} (verify with fracture tests)")
        warnings += 1
    
    sigma0_RT = weibull_df.loc[weibull_df['Temperature_C'] == 25, 'Characteristic_Strength_MPa'].values[0]
    if 300 <= sigma0_RT <= 600:
        print(f"  ✓ Characteristic strength at RT: {sigma0_RT} MPa (reasonable)")
        passed += 1
    else:
        print(f"  ⚠ Characteristic strength at RT: {sigma0_RT} MPa (verify)")
        warnings += 1
    
    # Check 6: No missing values
    print("\nCheck 6: Data Completeness")
    if not props_df.isnull().any().any():
        print("  ✓ No missing values in main properties dataset")
        passed += 1
    else:
        print("  ✗ Missing values detected in main properties dataset")
        failed += 1
    
    if not weibull_df.isnull().any().any():
        print("  ✓ No missing values in Weibull parameters dataset")
        passed += 1
    else:
        print("  ✗ Missing values detected in Weibull parameters dataset")
        failed += 1
    
    # Check 7: Temperature consistency
    print("\nCheck 7: Temperature Spacing")
    temp_spacing = np.diff(props_df['Temperature_C'])
    if np.std(temp_spacing) / np.mean(temp_spacing) < 0.3:
        print(f"  ✓ Temperature spacing is reasonably uniform (avg: {np.mean(temp_spacing):.1f}°C)")
        passed += 1
    else:
        print(f"  ⚠ Temperature spacing is irregular (may need refinement)")
        warnings += 1
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"  ✓ Passed:   {passed}")
    print(f"  ✗ Failed:   {failed}")
    print(f"  ⚠ Warnings: {warnings}")
    
    if failed == 0 and warnings <= 2:
        print("\n✓ Dataset is VALID and ready for FEM use")
        print("  (Remember: This is fabricated data - validate experimentally for production)")
        return 0
    elif failed == 0:
        print("\n⚠ Dataset is ACCEPTABLE but has some warnings")
        print("  Review the warnings above and verify against literature")
        return 0
    else:
        print("\n✗ Dataset has ERRORS that must be corrected")
        return 1

if __name__ == "__main__":
    sys.exit(validate_dataset())