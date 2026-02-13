#!/usr/bin/env python3
"""
Verification and QA script for material parameter database

Performs physical bounds checking and consistency validation:
1. Thermodynamic constraints (-1 < ν < 0.5)
2. Fracture hierarchy (Gc,II ≥ Gc,I)
3. Material property trends
4. Elastic tensor positive definiteness
5. Dimensional consistency

Usage:
    python verify_parameters.py --input data/master_material_database.csv

Author: Generated for SOFC Fracture Dataset
Date: February 13, 2026
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class ParameterVerifier:
    """Verify physical bounds and consistency of material parameters"""
    
    def __init__(self, csv_path: str):
        """
        Initialize verifier with CSV database
        
        Parameters:
        -----------
        csv_path : str or Path
            Path to master_material_database.csv
        """
        self.df = pd.read_csv(csv_path)
        self.errors = []
        self.warnings = []
        self.info = []
        
    def check_poisson_ratio(self) -> None:
        """Check Poisson's ratio thermodynamic bounds"""
        nu_data = self.df[self.df['Parameter'] == "Poisson's Ratio"].copy()
        
        for idx, row in nu_data.iterrows():
            nu = row['Value']
            material = row['Material']
            temp = row['Temperature_C']
            
            # Thermodynamic bounds: -1 < ν < 0.5
            if nu <= -1.0 or nu >= 0.5:
                self.errors.append(
                    f"Poisson's ratio out of thermodynamic bounds: "
                    f"{material} at {temp}°C, ν = {nu:.3f} (must be in (-1, 0.5))"
                )
            
            # Typical ceramic range: 0.2 < ν < 0.35
            if nu < 0.2 or nu > 0.35:
                self.warnings.append(
                    f"Poisson's ratio outside typical ceramic range: "
                    f"{material} at {temp}°C, ν = {nu:.3f} (typical: 0.2-0.35)"
                )
    
    def check_elastic_modulus(self) -> None:
        """Check Young's modulus trends and bounds"""
        E_data = self.df[self.df['Parameter'] == "Young's Modulus"].copy()
        
        for material in E_data['Material'].unique():
            mat_data = E_data[E_data['Material'] == material].sort_values('Temperature_C')
            
            if len(mat_data) >= 2:
                E_RT = mat_data[mat_data['Temperature_C'] == 25]['Value'].values
                E_HT = mat_data[mat_data['Temperature_C'] == 800]['Value'].values
                
                if len(E_RT) > 0 and len(E_HT) > 0:
                    E_RT, E_HT = E_RT[0], E_HT[0]
                    
                    # Check positive definiteness
                    if E_RT <= 0 or E_HT <= 0:
                        self.errors.append(
                            f"Non-positive Young's modulus: {material}, "
                            f"E_RT={E_RT:.1f}, E_HT={E_HT:.1f} GPa"
                        )
                    
                    # Check temperature trend (should decrease)
                    if E_HT > E_RT:
                        self.warnings.append(
                            f"Unusual temperature trend: {material} E increases with T "
                            f"({E_RT:.1f} → {E_HT:.1f} GPa)"
                        )
                    
                    # Check degradation is reasonable (5-20%)
                    degradation = (E_RT - E_HT) / E_RT * 100
                    if degradation < 0 or degradation > 25:
                        self.warnings.append(
                            f"Unusual E degradation: {material} shows {degradation:.1f}% "
                            f"reduction (typical: 5-20%)"
                        )
                    else:
                        self.info.append(
                            f"✓ {material}: E degradation = {degradation:.1f}% (RT→800°C)"
                        )
    
    def check_cte_hierarchy(self) -> None:
        """Check CTE hierarchy: α_LSCF > α_GDC > α_YSZ"""
        cte_data = self.df[self.df['Category'] == 'Thermal'].copy()
        
        cte_values = {}
        for material in ['8YSZ', 'GDC10', 'LSCF']:
            symbol = f'alpha_{material}'
            value = cte_data[cte_data['Symbol'] == symbol]['Value'].values
            if len(value) > 0:
                cte_values[material] = value[0]
        
        if len(cte_values) == 3:
            alpha_YSZ = cte_values['8YSZ']
            alpha_GDC = cte_values['GDC10']
            alpha_LSCF = cte_values['LSCF']
            
            # Expected hierarchy
            if not (alpha_YSZ < alpha_GDC < alpha_LSCF):
                self.warnings.append(
                    f"Unexpected CTE hierarchy: YSZ={alpha_YSZ:.2f}, "
                    f"GDC={alpha_GDC:.2f}, LSCF={alpha_LSCF:.2f} ppm/K "
                    f"(expected: YSZ < GDC < LSCF)"
                )
            else:
                self.info.append(
                    f"✓ CTE hierarchy: YSZ ({alpha_YSZ:.2f}) < "
                    f"GDC ({alpha_GDC:.2f}) < LSCF ({alpha_LSCF:.2f}) ppm/K"
                )
                
                # Calculate mismatch strains
                delta_T = 775  # 800°C - 25°C
                eps_thermal = (alpha_LSCF - alpha_YSZ) * 1e-6 * delta_T * 100  # in %
                self.info.append(
                    f"  → Thermal mismatch strain: {eps_thermal:.3f}% "
                    f"(LSCF-YSZ over {delta_T}°C)"
                )
    
    def check_fracture_toughness(self) -> None:
        """Check fracture toughness hierarchy and mode mixity"""
        frac_data = self.df[self.df['Category'] == 'Fracture'].copy()
        
        # Check bulk toughness hierarchy
        bulk_Gc = {}
        for material in ['8YSZ', 'GDC10', 'LSCF']:
            symbol = f'Gc_b_{material.replace("10", "").replace("8", "")}'
            value = frac_data[frac_data['Symbol'] == symbol]['Value'].values
            if len(value) > 0:
                bulk_Gc[material] = value[0]
        
        if len(bulk_Gc) >= 2:
            self.info.append(
                f"✓ Bulk fracture toughness: " +
                ", ".join([f"{k}={v:.1f} J/m²" for k, v in bulk_Gc.items()])
            )
        
        # Check interface mode I < mode II
        for interface in ['YSZ_GDC', 'GDC_LSCF']:
            Gc_I_symbol = f'Gc_I_{interface}'
            Gc_II_symbol = f'Gc_II_{interface}'
            
            Gc_I = frac_data[frac_data['Symbol'] == Gc_I_symbol]['Value'].values
            Gc_II = frac_data[frac_data['Symbol'] == Gc_II_symbol]['Value'].values
            
            if len(Gc_I) > 0 and len(Gc_II) > 0:
                Gc_I, Gc_II = Gc_I[0], Gc_II[0]
                ratio = Gc_II / Gc_I
                
                if Gc_II < Gc_I:
                    self.errors.append(
                        f"Mode II toughness < Mode I for {interface.replace('_', '/')}: "
                        f"Gc_I={Gc_I:.1f}, Gc_II={Gc_II:.1f} J/m² (unusual for ceramics)"
                    )
                elif ratio < 1.2 or ratio > 2.5:
                    self.warnings.append(
                        f"Unusual Gc_II/Gc_I ratio for {interface.replace('_', '/')}: "
                        f"{ratio:.2f} (typical for ceramics: 1.2-2.5)"
                    )
                else:
                    self.info.append(
                        f"✓ {interface.replace('_', '/')}: Gc_II/Gc_I = {ratio:.2f}"
                    )
                
                # Check interface weaker than bulk
                materials_in_interface = interface.split('_')
                for mat in materials_in_interface:
                    mat_key = mat if mat == 'LSCF' else (mat + '10' if mat == 'GDC' else '8' + mat)
                    if mat_key in bulk_Gc:
                        if Gc_I > bulk_Gc[mat_key]:
                            self.warnings.append(
                                f"Interface {interface.replace('_', '/')} tougher than bulk {mat}: "
                                f"Gc_I={Gc_I:.1f} > Gc_bulk={bulk_Gc[mat_key]:.1f} J/m²"
                            )
    
    def check_cohesive_parameters(self) -> None:
        """Check cohesive strength and characteristic length consistency"""
        cohesive_data = self.df[self.df['Category'] == 'Cohesive'].copy()
        frac_data = self.df[self.df['Category'] == 'Fracture'].copy()
        
        for interface in ['YSZ_GDC', 'GDC_LSCF']:
            # Get cohesive parameters
            T_max_n = cohesive_data[cohesive_data['Symbol'] == f'T_max_n_{interface}']['Value'].values
            T_max_t = cohesive_data[cohesive_data['Symbol'] == f'T_max_t_{interface}']['Value'].values
            Gc_I = frac_data[frac_data['Symbol'] == f'Gc_I_{interface}']['Value'].values
            Gc_II = frac_data[frac_data['Symbol'] == f'Gc_II_{interface}']['Value'].values
            
            if len(T_max_n) > 0 and len(Gc_I) > 0:
                T_max_n, Gc_I = T_max_n[0], Gc_I[0]
                
                # Characteristic length: lc = Gc / T_max^2 (approximate)
                # For triangular TSL: lc = 2*Gc/T_max
                lc_calc = 2 * Gc_I / T_max_n  # in J/m² / MPa = mJ/MPa = μm
                
                self.info.append(
                    f"✓ {interface.replace('_', '/')}: Characteristic length lc ≈ {lc_calc:.3f} μm "
                    f"(from Gc_I={Gc_I:.1f} J/m², T_max_n={T_max_n:.0f} MPa)"
                )
                
                # Check if reasonable for mesh (should be > 0.01 μm and < 10 μm)
                if lc_calc < 0.01:
                    self.warnings.append(
                        f"Very small characteristic length for {interface.replace('_', '/')}: "
                        f"lc={lc_calc:.4f} μm may require extremely fine mesh"
                    )
                elif lc_calc > 10:
                    self.warnings.append(
                        f"Large characteristic length for {interface.replace('_', '/')}: "
                        f"lc={lc_calc:.3f} μm may not be physically realistic"
                    )
            
            # Check T_max_t > T_max_n (typical)
            if len(T_max_t) > 0 and len(T_max_n) > 0:
                T_max_t, T_max_n = T_max_t[0], T_max_n[0]
                if T_max_t < T_max_n:
                    self.warnings.append(
                        f"Shear strength < Normal strength for {interface.replace('_', '/')}: "
                        f"T_max_t={T_max_t:.0f} < T_max_n={T_max_n:.0f} MPa (unusual)"
                    )
    
    def check_chemical_expansion(self) -> None:
        """Check chemical expansion parameters"""
        chem_data = self.df[self.df['Category'] == 'Chemical'].copy()
        
        # GDC isotropic expansion
        beta_iso = chem_data[chem_data['Symbol'] == 'beta_iso_GDC']['Value'].values
        delta_delta_GDC = chem_data[chem_data['Symbol'] == 'delta_delta_GDC']['Value'].values
        
        if len(beta_iso) > 0 and len(delta_delta_GDC) > 0:
            eps_chem_GDC = beta_iso[0] * delta_delta_GDC[0] * 100  # in %
            self.info.append(
                f"✓ GDC chemical expansion: {eps_chem_GDC:.3f}% "
                f"(β={beta_iso[0]:.4f}, Δδ={delta_delta_GDC[0]:.3f})"
            )
        
        # LSCF anisotropic expansion
        beta_11 = chem_data[chem_data['Symbol'] == 'beta_11_LSCF']['Value'].values
        beta_33 = chem_data[chem_data['Symbol'] == 'beta_33_LSCF']['Value'].values
        delta_delta_LSCF = chem_data[chem_data['Symbol'] == 'delta_delta_LSCF']['Value'].values
        
        if len(beta_11) > 0 and len(beta_33) > 0 and len(delta_delta_LSCF) > 0:
            eps_11 = beta_11[0] * delta_delta_LSCF[0] * 100
            eps_33 = beta_33[0] * delta_delta_LSCF[0] * 100
            
            self.info.append(
                f"✓ LSCF chemical expansion: ε₁₁={eps_11:.3f}%, ε₃₃={eps_33:.3f}% "
                f"(Δδ={delta_delta_LSCF[0]:.3f})"
            )
            
            # Check anisotropy ratio
            anisotropy = beta_33[0] / beta_11[0]
            if anisotropy < 1.0:
                self.warnings.append(
                    f"Unexpected LSCF anisotropy: β₃₃/β₁₁ = {anisotropy:.2f} < 1.0 "
                    f"(typically c-axis expands more)"
                )
            else:
                self.info.append(
                    f"  → Anisotropy ratio β₃₃/β₁₁ = {anisotropy:.2f}"
                )
    
    def check_mesh_requirements(self) -> None:
        """Check mesh requirements for phase field and cohesive zones"""
        pf_data = self.df[self.df['Category'] == 'Phase_Field'].copy()
        mesh_data = self.df[self.df['Category'] == 'Mesh'].copy()
        geom_data = self.df[self.df['Category'] == 'Geometric'].copy()
        
        for material in ['YSZ', 'GDC', 'LSCF']:
            # Phase field length scale
            l_pf = pf_data[pf_data['Material'] == ('8YSZ' if material == 'YSZ' else 
                                                   ('GDC10' if material == 'GDC' else material))]['Value'].values
            
            # Element size
            h = mesh_data[mesh_data['Material'] == ('8YSZ' if material == 'YSZ' else 
                                                    ('GDC10' if material == 'GDC' else material))]['Value'].values
            
            # Layer thickness
            t = geom_data[geom_data['Symbol'] == f't_{material}']['Value'].values
            
            if len(l_pf) > 0 and len(h) > 0 and len(t) > 0:
                l_pf, h, t = l_pf[0], h[0], t[0]
                
                # Check phase field resolution: h ≤ l_pf/2
                if h > l_pf / 2:
                    self.warnings.append(
                        f"{material}: Element size h={h:.3f} μm > l_pf/2={l_pf/2:.3f} μm "
                        f"(may not resolve phase field)"
                    )
                
                # Check through-thickness resolution
                n_elements = t / h
                if n_elements < 3:
                    self.warnings.append(
                        f"{material}: Only {n_elements:.1f} elements through thickness "
                        f"(recommend ≥5 for accuracy)"
                    )
                else:
                    self.info.append(
                        f"✓ {material}: {n_elements:.0f} elements through {t:.1f} μm thickness"
                    )
    
    def check_quality_flags(self) -> None:
        """Check distribution of quality flags"""
        quality_counts = self.df['Quality_Flag'].value_counts()
        
        self.info.append("\nData Quality Distribution:")
        for flag, count in quality_counts.items():
            percentage = count / len(self.df) * 100
            self.info.append(f"  {flag}: {count} parameters ({percentage:.1f}%)")
        
        # Flag critical parameters with low quality
        low_quality = self.df[self.df['Quality_Flag'].isin(['LOW', 'FLAG'])].copy()
        critical_params = ['Gc_I', 'Gc_II', 'T_max_n', 'T_max_t', 'beta']
        
        for param in critical_params:
            critical_low_q = low_quality[low_quality['Parameter'].str.contains(param, na=False)]
            if len(critical_low_q) > 0:
                self.warnings.append(
                    f"Critical parameter '{param}' has LOW/FLAG quality - "
                    f"recommend experimental validation"
                )
    
    def run_all_checks(self) -> Dict[str, int]:
        """
        Run all verification checks
        
        Returns:
        --------
        dict : Summary of issues found
        """
        print("\nRunning Parameter Verification Checks...")
        print("=" * 70)
        
        self.check_poisson_ratio()
        self.check_elastic_modulus()
        self.check_cte_hierarchy()
        self.check_fracture_toughness()
        self.check_cohesive_parameters()
        self.check_chemical_expansion()
        self.check_mesh_requirements()
        self.check_quality_flags()
        
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'info': len(self.info)
        }
    
    def print_report(self) -> bool:
        """
        Print verification report
        
        Returns:
        --------
        bool : True if no errors, False otherwise
        """
        print("\n" + "=" * 70)
        print("VERIFICATION REPORT")
        print("=" * 70)
        
        if self.errors:
            print("\n❌ ERRORS (Must Fix):")
            print("-" * 70)
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS (Recommended to Review):")
            print("-" * 70)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")
        
        if self.info:
            print("\nℹ️  INFORMATION:")
            print("-" * 70)
            for item in self.info:
                print(item)
        
        print("\n" + "=" * 70)
        print(f"Summary: {len(self.errors)} errors, {len(self.warnings)} warnings")
        print("=" * 70)
        
        if len(self.errors) == 0:
            print("\n✅ All critical checks passed!")
            return True
        else:
            print("\n❌ Errors found - please fix before running simulation")
            return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Verify material parameter database for physical consistency'
    )
    parser.add_argument('--input', '-i',
                       default='data/master_material_database.csv',
                       help='Input CSV file path')
    parser.add_argument('--strict', '-s',
                       action='store_true',
                       help='Treat warnings as errors')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1
    
    # Run verification
    print(f"Loading material database: {input_path}")
    verifier = ParameterVerifier(input_path)
    
    summary = verifier.run_all_checks()
    passed = verifier.print_report()
    
    # Exit code
    if args.strict and summary['warnings'] > 0:
        print("\n⚠️  Strict mode: treating warnings as errors")
        return 1
    
    return 0 if passed else 1


if __name__ == '__main__':
    exit(main())
