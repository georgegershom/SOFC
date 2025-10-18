#!/usr/bin/env python3
"""
Dataset Validation Script for Thermo-Mechanical Modeling Dataset
Fire-Resistant Structural Elements with High-Performance Rubberized Concrete

This script performs comprehensive validation of the generated dataset to ensure
physical consistency, statistical validity, and FEA software compatibility.
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DatasetValidator:
    """
    Comprehensive validator for thermo-mechanical modeling dataset.
    Performs physical consistency, statistical, and compatibility checks.
    """
    
    def __init__(self, data_dir='.'):
        """Initialize validator with dataset location"""
        self.data_dir = data_dir
        self.validation_results = {}
        self.load_data()
    
    def load_data(self):
        """Load all dataset files for validation"""
        try:
            self.thermal_df = pd.read_csv(f'{self.data_dir}/thermal_properties.csv')
            self.mechanical_df = pd.read_csv(f'{self.data_dir}/mechanical_properties.csv')
            self.transport_df = pd.read_csv(f'{self.data_dir}/transport_properties.csv')
            self.deformation_df = pd.read_csv(f'{self.data_dir}/deformation_properties.csv')
            
            with open(f'{self.data_dir}/stress_strain_curves.json', 'r') as f:
                self.stress_strain_curves = json.load(f)
            
            print("Dataset loaded for validation")
            
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def validate_data_completeness(self):
        """Validate that all required data is present"""
        print("\n1. Validating Data Completeness...")
        
        results = {
            'total_data_points': 0,
            'missing_values': 0,
            'expected_mixes': ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L'],
            'expected_temps': 157,  # 20°C to 800°C in 5°C increments
            'expected_properties': {}
        }
        
        # Check each dataframe
        dataframes = {
            'thermal': self.thermal_df,
            'mechanical': self.mechanical_df,
            'transport': self.transport_df,
            'deformation': self.deformation_df
        }
        
        for name, df in dataframes.items():
            results['total_data_points'] += len(df)
            results['missing_values'] += df.isnull().sum().sum()
            
            # Check mix coverage
            unique_mixes = df['Mix_ID'].unique()
            missing_mixes = set(results['expected_mixes']) - set(unique_mixes)
            if missing_mixes:
                print(f"  WARNING: {name} missing mixes: {missing_mixes}")
            
            # Check temperature coverage
            unique_temps = df['Temperature_C'].nunique()
            if unique_temps != results['expected_temps']:
                print(f"  WARNING: {name} has {unique_temps} temperature points, expected {results['expected_temps']}")
            
            # Check data type distribution
            cal_ratio = (df['Data_Type'] == 'Calibration').mean()
            if not (0.6 <= cal_ratio <= 0.8):
                print(f"  WARNING: {name} calibration ratio {cal_ratio:.2f} outside expected range [0.6, 0.8]")
        
        # Check stress-strain curves
        for mix_id in results['expected_mixes']:
            if mix_id not in self.stress_strain_curves:
                print(f"  WARNING: Missing stress-strain curves for mix {mix_id}")
        
        results['completeness_score'] = 1.0 - (results['missing_values'] / results['total_data_points'])
        
        print(f"  Total data points: {results['total_data_points']}")
        print(f"  Missing values: {results['missing_values']}")
        print(f"  Completeness score: {results['completeness_score']:.3f}")
        
        self.validation_results['completeness'] = results
        return results['completeness_score'] > 0.99
    
    def validate_physical_consistency(self):
        """Validate physical consistency of material properties"""
        print("\n2. Validating Physical Consistency...")
        
        results = {
            'violations': [],
            'warnings': [],
            'consistency_score': 1.0
        }
        
        # Check temperature monotonicity for properties that should decrease
        decreasing_props = [
            ('thermal', 'Thermal_Conductivity_W_mK'),
            ('mechanical', 'Compressive_Strength_MPa'),
            ('mechanical', 'Elastic_Modulus_GPa'),
            ('mechanical', 'Tensile_Strength_MPa'),
            ('mechanical', 'Fracture_Energy_N_m'),
            ('deformation', 'Creep_Modulus_GPa')
        ]
        
        for prop_type, prop_name in decreasing_props:
            df = getattr(self, f'{prop_type}_df')
            
            for mix_id in df['Mix_ID'].unique():
                mix_data = df[df['Mix_ID'] == mix_id].sort_values('Temperature_C')
                values = mix_data[prop_name].values
                
                # Check for non-decreasing trend
                if not np.all(np.diff(values) <= 0):
                    results['warnings'].append(f"{mix_id} {prop_name} not monotonically decreasing")
        
        # Check temperature monotonicity for properties that should increase
        increasing_props = [
            ('thermal', 'Specific_Heat_J_kgK'),
            ('thermal', 'Thermal_Expansion_1_K'),
            ('mechanical', 'Poissons_Ratio'),
            ('transport', 'Porosity'),
            ('transport', 'Permeability_m2'),
            ('transport', 'Water_Diffusivity_m2_s'),
            ('transport', 'Vapor_Diffusivity_m2_s'),
            ('deformation', 'Creep_Coefficient'),
            ('deformation', 'Shrinkage_Strain'),
            ('deformation', 'Thermal_Strain')
        ]
        
        for prop_type, prop_name in increasing_props:
            df = getattr(self, f'{prop_type}_df')
            
            for mix_id in df['Mix_ID'].unique():
                mix_data = df[df['Mix_ID'] == mix_id].sort_values('Temperature_C')
                values = mix_data[prop_name].values
                
                # Check for non-increasing trend
                if not np.all(np.diff(values) >= 0):
                    results['warnings'].append(f"{mix_id} {prop_name} not monotonically increasing")
        
        # Check value ranges
        range_checks = [
            ('thermal', 'Thermal_Conductivity_W_mK', (0.5, 3.0)),
            ('thermal', 'Specific_Heat_J_kgK', (800, 1500)),
            ('thermal', 'Thermal_Expansion_1_K', (5e-6, 25e-6)),
            ('thermal', 'Density_kg_m3', (2000, 2500)),
            ('mechanical', 'Compressive_Strength_MPa', (0, 60)),
            ('mechanical', 'Elastic_Modulus_GPa', (0, 50)),
            ('mechanical', 'Tensile_Strength_MPa', (0, 5)),
            ('mechanical', 'Poissons_Ratio', (0.1, 0.4)),
            ('transport', 'Porosity', (0.05, 0.5)),
            ('transport', 'Permeability_m2', (1e-18, 1e-12)),
        ]
        
        for prop_type, prop_name, (min_val, max_val) in range_checks:
            df = getattr(self, f'{prop_type}_df')
            values = df[prop_name].values
            
            if np.any(values < min_val) or np.any(values > max_val):
                results['violations'].append(f"{prop_name} values outside range [{min_val}, {max_val}]")
        
        # Check mix ordering (rubber content effects)
        rubber_contents = {'C': 0, 'R5S': 5, 'R10S': 10, 'R15S': 15, 'R20S': 20, 'R10L': 10}
        
        for prop_type in ['thermal', 'mechanical', 'transport', 'deformation']:
            df = getattr(self, f'{prop_type}_df')
            
            # Get properties at 20°C for comparison
            room_temp_data = df[df['Temperature_C'] == 20.0]
            
            for prop_name in df.columns:
                if prop_name.endswith('_Std') or prop_name in ['Mix_ID', 'Temperature_C', 'Data_Type', 'Property_Type']:
                    continue
                
                # Check if property decreases with rubber content (for most properties)
                if prop_name in ['Thermal_Conductivity_W_mK', 'Compressive_Strength_MPa', 
                               'Elastic_Modulus_GPa', 'Tensile_Strength_MPa', 'Density_kg_m3']:
                    values = room_temp_data.groupby('Mix_ID')[prop_name].mean()
                    rubber_ordered = [rubber_contents[mix] for mix in values.index]
                    
                    if not np.all(np.diff(values.values[np.argsort(rubber_ordered)]) <= 0):
                        results['warnings'].append(f"{prop_name} not properly ordered by rubber content")
        
        # Calculate consistency score
        total_checks = len(decreasing_props) + len(increasing_props) + len(range_checks)
        violations = len(results['violations'])
        warnings = len(results['warnings'])
        
        results['consistency_score'] = 1.0 - (violations + 0.5 * warnings) / total_checks
        
        print(f"  Violations: {violations}")
        print(f"  Warnings: {warnings}")
        print(f"  Consistency score: {results['consistency_score']:.3f}")
        
        if violations > 0:
            print("  VIOLATIONS:")
            for violation in results['violations']:
                print(f"    - {violation}")
        
        if warnings > 0:
            print("  WARNINGS:")
            for warning in results['warnings']:
                print(f"    - {warning}")
        
        self.validation_results['physical_consistency'] = results
        return results['consistency_score'] > 0.8
    
    def validate_statistical_properties(self):
        """Validate statistical properties of the dataset"""
        print("\n3. Validating Statistical Properties...")
        
        results = {
            'normality_tests': {},
            'cv_checks': {},
            'correlation_checks': {},
            'statistical_score': 1.0
        }
        
        # Check normality of property distributions
        dataframes = {
            'thermal': self.thermal_df,
            'mechanical': self.mechanical_df,
            'transport': self.transport_df,
            'deformation': self.deformation_df
        }
        
        for name, df in dataframes.items():
            results['normality_tests'][name] = {}
            
            for col in df.columns:
                if col.endswith('_Std') or col in ['Mix_ID', 'Temperature_C', 'Data_Type', 'Property_Type']:
                    continue
                
                # Test normality using Shapiro-Wilk test
                if len(df[col].dropna()) > 3:
                    stat, p_value = stats.shapiro(df[col].dropna())
                    results['normality_tests'][name][col] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
        
        # Check coefficient of variation ranges
        cv_ranges = {
            'Thermal_Conductivity_W_mK': (0.03, 0.10),
            'Specific_Heat_J_kgK': (0.05, 0.15),
            'Compressive_Strength_MPa': (0.08, 0.20),
            'Elastic_Modulus_GPa': (0.10, 0.25),
            'Porosity': (0.08, 0.20),
            'Permeability_m2': (0.15, 0.40)
        }
        
        for prop_name, (min_cv, max_cv) in cv_ranges.items():
            # Find the dataframe containing this property
            for name, df in dataframes.items():
                if prop_name in df.columns:
                    std_col = prop_name + '_Std'
                    if std_col in df.columns:
                        cv_values = df[std_col] / df[prop_name]
                        mean_cv = cv_values.mean()
                        
                        results['cv_checks'][prop_name] = {
                            'mean_cv': mean_cv,
                            'in_range': min_cv <= mean_cv <= max_cv
                        }
                    break
        
        # Check correlations between related properties
        correlation_checks = [
            ('Compressive_Strength_MPa', 'Elastic_Modulus_GPa'),
            ('Thermal_Conductivity_W_mK', 'Density_kg_m3'),
            ('Porosity', 'Permeability_m2')
        ]
        
        for prop1, prop2 in correlation_checks:
            # Find properties in the same dataframe
            for name, df in dataframes.items():
                if prop1 in df.columns and prop2 in df.columns:
                    correlation = df[prop1].corr(df[prop2])
                    results['correlation_checks'][f"{prop1}_vs_{prop2}"] = {
                        'correlation': correlation,
                        'is_positive': correlation > 0.3  # Expected positive correlation
                    }
                    break
        
        # Calculate statistical score
        total_checks = 0
        passed_checks = 0
        
        # Normality checks
        for name, tests in results['normality_tests'].items():
            for prop, test_result in tests.items():
                total_checks += 1
                if test_result['is_normal']:
                    passed_checks += 1
        
        # CV checks
        for prop, cv_result in results['cv_checks'].items():
            total_checks += 1
            if cv_result['in_range']:
                passed_checks += 1
        
        # Correlation checks
        for prop_pair, corr_result in results['correlation_checks'].items():
            total_checks += 1
            if corr_result['is_positive']:
                passed_checks += 1
        
        results['statistical_score'] = passed_checks / total_checks if total_checks > 0 else 1.0
        
        print(f"  Normality tests passed: {sum(1 for tests in results['normality_tests'].values() for test in tests.values() if test['is_normal'])}")
        print(f"  CV checks passed: {sum(1 for check in results['cv_checks'].values() if check['in_range'])}")
        print(f"  Correlation checks passed: {sum(1 for check in results['correlation_checks'].values() if check['is_positive'])}")
        print(f"  Statistical score: {results['statistical_score']:.3f}")
        
        self.validation_results['statistical'] = results
        return results['statistical_score'] > 0.7
    
    def validate_fea_compatibility(self):
        """Validate FEA software compatibility"""
        print("\n4. Validating FEA Software Compatibility...")
        
        results = {
            'abaqus_compatibility': True,
            'ansys_compatibility': True,
            'comsol_compatibility': True,
            'compatibility_score': 1.0
        }
        
        # Check ABAQUS compatibility
        abaqus_required_props = [
            'Elastic_Modulus_GPa', 'Poissons_Ratio', 'Density_kg_m3',
            'Thermal_Conductivity_W_mK', 'Specific_Heat_J_kgK', 'Thermal_Expansion_1_K'
        ]
        
        for prop in abaqus_required_props:
            found = False
            for df in [self.thermal_df, self.mechanical_df, self.transport_df, self.deformation_df]:
                if prop in df.columns:
                    found = True
                    break
            if not found:
                results['abaqus_compatibility'] = False
                print(f"  WARNING: Missing ABAQUS required property: {prop}")
        
        # Check ANSYS compatibility
        ansys_required_props = [
            'Elastic_Modulus_GPa', 'Poissons_Ratio', 'Density_kg_m3',
            'Thermal_Conductivity_W_mK', 'Specific_Heat_J_kgK', 'Thermal_Expansion_1_K'
        ]
        
        for prop in ansys_required_props:
            found = False
            for df in [self.thermal_df, self.mechanical_df, self.transport_df, self.deformation_df]:
                if prop in df.columns:
                    found = True
                    break
            if not found:
                results['ansys_compatibility'] = False
                print(f"  WARNING: Missing ANSYS required property: {prop}")
        
        # Check COMSOL compatibility
        comsol_required_props = [
            'Elastic_Modulus_GPa', 'Poissons_Ratio', 'Density_kg_m3',
            'Thermal_Conductivity_W_mK', 'Specific_Heat_J_kgK', 'Thermal_Expansion_1_K',
            'Porosity', 'Permeability_m2'
        ]
        
        for prop in comsol_required_props:
            found = False
            for df in [self.thermal_df, self.mechanical_df, self.transport_df, self.deformation_df]:
                if prop in df.columns:
                    found = True
                    break
            if not found:
                results['comsol_compatibility'] = False
                print(f"  WARNING: Missing COMSOL required property: {prop}")
        
        # Check stress-strain curve compatibility
        if not self.stress_strain_curves:
            results['abaqus_compatibility'] = False
            results['ansys_compatibility'] = False
            results['comsol_compatibility'] = False
            print("  WARNING: Missing stress-strain curves")
        
        # Calculate compatibility score
        compatibility_checks = [
            results['abaqus_compatibility'],
            results['ansys_compatibility'],
            results['comsol_compatibility']
        ]
        
        results['compatibility_score'] = sum(compatibility_checks) / len(compatibility_checks)
        
        print(f"  ABAQUS compatible: {results['abaqus_compatibility']}")
        print(f"  ANSYS compatible: {results['ansys_compatibility']}")
        print(f"  COMSOL compatible: {results['comsol_compatibility']}")
        print(f"  Compatibility score: {results['compatibility_score']:.3f}")
        
        self.validation_results['fea_compatibility'] = results
        return results['compatibility_score'] > 0.8
    
    def validate_stress_strain_curves(self):
        """Validate stress-strain curve data"""
        print("\n5. Validating Stress-Strain Curves...")
        
        results = {
            'curve_checks': {},
            'curve_score': 1.0
        }
        
        expected_mixes = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
        expected_temps = list(range(20, 801, 50))  # Every 50°C
        
        for mix_id in expected_mixes:
            if mix_id not in self.stress_strain_curves:
                results['curve_checks'][mix_id] = {'present': False, 'temp_coverage': 0}
                continue
            
            curve_data = self.stress_strain_curves[mix_id]
            available_temps = [int(float(t)) for t in curve_data.keys()]
            
            results['curve_checks'][mix_id] = {
                'present': True,
                'temp_coverage': len(available_temps) / len(expected_temps),
                'has_required_temps': all(t in available_temps for t in [20, 200, 400, 600, 800]) or len(available_temps) >= 10
            }
            
            # Check curve validity
            for temp, curve in curve_data.items():
                strain = np.array(curve['strain'])
                stress = np.array(curve['stress'])
                
                # Check for negative stresses
                if np.any(stress < 0):
                    print(f"  WARNING: {mix_id} at {temp}°C has negative stresses")
                
                # Check for increasing strain
                if not np.all(np.diff(strain) > 0):
                    print(f"  WARNING: {mix_id} at {temp}°C has non-increasing strain")
                
                # Check for reasonable peak stress
                peak_stress = np.max(stress)
                if peak_stress < 0.1 or peak_stress > 100.0:
                    print(f"  WARNING: {mix_id} at {temp}°C has unreasonable peak stress: {peak_stress:.2f} MPa")
        
        # Calculate curve score
        total_checks = len(expected_mixes)
        passed_checks = 0
        
        for mix_id, checks in results['curve_checks'].items():
            if checks['present'] and checks['temp_coverage'] > 0.8 and checks['has_required_temps']:
                passed_checks += 1
        
        results['curve_score'] = passed_checks / total_checks
        
        print(f"  Mixes with curves: {sum(1 for checks in results['curve_checks'].values() if checks['present'])}")
        print(f"  Average temperature coverage: {np.mean([checks['temp_coverage'] for checks in results['curve_checks'].values() if checks['present']]):.3f}")
        print(f"  Curve score: {results['curve_score']:.3f}")
        
        self.validation_results['stress_strain_curves'] = results
        return results['curve_score'] > 0.8
    
    def run_complete_validation(self):
        """Run all validation checks and generate summary report"""
        print("Thermo-Mechanical Dataset Validation Report")
        print("=" * 50)
        
        # Run all validation checks
        completeness_ok = self.validate_data_completeness()
        consistency_ok = self.validate_physical_consistency()
        statistical_ok = self.validate_statistical_properties()
        fea_ok = self.validate_fea_compatibility()
        curves_ok = self.validate_stress_strain_curves()
        
        # Calculate overall score
        scores = [
            self.validation_results['completeness']['completeness_score'],
            self.validation_results['physical_consistency']['consistency_score'],
            self.validation_results['statistical']['statistical_score'],
            self.validation_results['fea_compatibility']['compatibility_score'],
            self.validation_results['stress_strain_curves']['curve_score']
        ]
        
        overall_score = np.mean(scores)
        
        # Generate summary report
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        print(f"1. Data Completeness:     {'PASS' if completeness_ok else 'FAIL'} ({scores[0]:.3f})")
        print(f"2. Physical Consistency:  {'PASS' if consistency_ok else 'FAIL'} ({scores[1]:.3f})")
        print(f"3. Statistical Properties: {'PASS' if statistical_ok else 'FAIL'} ({scores[2]:.3f})")
        print(f"4. FEA Compatibility:     {'PASS' if fea_ok else 'FAIL'} ({scores[3]:.3f})")
        print(f"5. Stress-Strain Curves:  {'PASS' if curves_ok else 'FAIL'} ({scores[4]:.3f})")
        
        print(f"\nOverall Score: {overall_score:.3f}")
        
        if overall_score >= 0.8:
            print("✅ DATASET VALIDATION PASSED")
            print("The dataset is ready for use in thermo-mechanical modeling.")
        else:
            print("❌ DATASET VALIDATION FAILED")
            print("Please review the validation results and address the identified issues.")
        
        # Save validation report
        with open('validation_report.txt', 'w') as f:
            f.write("Thermo-Mechanical Dataset Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for category, results in self.validation_results.items():
                f.write(f"{category.upper()}:\n")
                f.write(f"  Score: {results.get('completeness_score', results.get('consistency_score', results.get('statistical_score', results.get('compatibility_score', results.get('curve_score', 'N/A'))))):.3f}\n")
                f.write("\n")
        
        return overall_score >= 0.8

def main():
    """Run complete dataset validation"""
    validator = DatasetValidator()
    success = validator.run_complete_validation()
    
    if success:
        print("\n🎉 Dataset validation completed successfully!")
    else:
        print("\n⚠️  Dataset validation found issues that need attention.")
    
    return success

if __name__ == "__main__":
    main()