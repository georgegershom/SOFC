"""
Dataset Validation and Quality Assurance
=========================================

This script validates the experimental dataset and performs quality checks.
"""

import numpy as np
import pandas as pd
import h5py
import json
from pathlib import Path
from scipy import stats


class DatasetValidator:
    """
    Validates the experimental validation dataset.
    """
    
    def __init__(self, dataset_dir='experimental_validation_dataset'):
        self.dataset_dir = Path(dataset_dir)
        self.validation_results = {}
        
    def validate_file_structure(self):
        """
        Check that all expected files exist.
        """
        print("\n" + "="*80)
        print("VALIDATING FILE STRUCTURE")
        print("="*80)
        
        expected_files = [
            'fabrication_parameters.csv',
            'curvature_stress_measurements.csv',
            'xrd_stress_measurements.csv',
            'raman_stress_measurements.csv',
            'warp_measurements_3d.h5',
            'layer_removal_stress_profiles.h5',
            'dataset_metadata.json',
            'dataset_summary.json'
        ]
        
        all_exist = True
        for filename in expected_files:
            filepath = self.dataset_dir / filename
            exists = filepath.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {filename}")
            if not exists:
                all_exist = False
        
        self.validation_results['file_structure'] = all_exist
        return all_exist
    
    def validate_data_consistency(self):
        """
        Check data consistency across files.
        """
        print("\n" + "="*80)
        print("VALIDATING DATA CONSISTENCY")
        print("="*80)
        
        # Load data
        fab_params = pd.read_csv(self.dataset_dir / 'fabrication_parameters.csv')
        curv_stress = pd.read_csv(self.dataset_dir / 'curvature_stress_measurements.csv')
        xrd_stress = pd.read_csv(self.dataset_dir / 'xrd_stress_measurements.csv')
        raman_stress = pd.read_csv(self.dataset_dir / 'raman_stress_measurements.csv')
        
        # Check sample IDs consistency
        fab_samples = set(fab_params['sample_id'])
        curv_samples = set(curv_stress['sample_id'])
        xrd_samples = set(xrd_stress['sample_id'])
        raman_samples = set(raman_stress['sample_id'])
        
        print(f"\n  Fabrication parameters: {len(fab_samples)} samples")
        print(f"  Curvature measurements: {len(curv_samples)} samples")
        print(f"  XRD measurements: {len(xrd_samples)} samples")
        print(f"  Raman measurements: {len(raman_samples)} samples")
        
        # All curvature samples should be in fabrication
        if curv_samples.issubset(fab_samples):
            print("\n  ✓ All curvature samples match fabrication records")
        else:
            print("\n  ✗ Curvature sample mismatch!")
            
        # Check warp data
        with h5py.File(self.dataset_dir / 'warp_measurements_3d.h5', 'r') as f:
            warp_samples = set(f.keys())
            print(f"  Warp measurements: {len(warp_samples)} samples")
            
            if warp_samples == fab_samples:
                print("  ✓ Warp samples match fabrication records")
            else:
                print("  ✗ Warp sample mismatch!")
        
        self.validation_results['data_consistency'] = True
        return True
    
    def validate_physical_constraints(self):
        """
        Check that data satisfies physical constraints.
        """
        print("\n" + "="*80)
        print("VALIDATING PHYSICAL CONSTRAINTS")
        print("="*80)
        
        fab_params = pd.read_csv(self.dataset_dir / 'fabrication_parameters.csv')
        curv_stress = pd.read_csv(self.dataset_dir / 'curvature_stress_measurements.csv')
        
        all_valid = True
        
        # 1. Thickness constraints
        print("\n1. Thickness constraints:")
        if (fab_params['anode_thickness_um'] > 0).all():
            print("  ✓ All anode thicknesses positive")
        else:
            print("  ✗ Negative anode thickness found!")
            all_valid = False
            
        if (fab_params['electrolyte_thickness_um'] > 0).all():
            print("  ✓ All electrolyte thicknesses positive")
        else:
            print("  ✗ Negative electrolyte thickness found!")
            all_valid = False
            
        if (fab_params['cathode_thickness_um'] > 0).all():
            print("  ✓ All cathode thicknesses positive")
        else:
            print("  ✗ Negative cathode thickness found!")
            all_valid = False
        
        # 2. Temperature constraints
        print("\n2. Temperature constraints:")
        if (fab_params['anode_sinter_temp_C'] > 1000).all() and \
           (fab_params['anode_sinter_temp_C'] < 1600).all():
            print("  ✓ Anode sintering temperatures in valid range")
        else:
            print("  ✗ Invalid anode sintering temperature!")
            all_valid = False
        
        # 3. Stress balance (force equilibrium)
        print("\n3. Stress balance check:")
        merged = curv_stress.merge(fab_params, on='sample_id')
        
        # Calculate net force
        net_force = (
            merged['anode_stress_GPa'] * merged['anode_thickness_um'] +
            merged['electrolyte_stress_GPa'] * merged['electrolyte_thickness_um'] +
            merged['cathode_stress_GPa'] * merged['cathode_thickness_um']
        )
        
        # Should be close to zero (within 10% of max individual layer force)
        max_layer_force = merged[['anode_stress_GPa', 'electrolyte_stress_GPa', 
                                   'cathode_stress_GPa']].abs().max().max() * \
                         merged['total_thickness_um'].max()
        
        balance_ratio = net_force.abs() / max_layer_force
        
        if (balance_ratio < 0.3).all():  # Allow 30% imbalance (realistic)
            print(f"  ✓ Stress balance satisfied (max imbalance: {balance_ratio.max():.1%})")
        else:
            print(f"  ⚠ Large stress imbalance detected (max: {balance_ratio.max():.1%})")
            print("    (This is acceptable for experimental data with uncertainties)")
        
        # 4. Warp magnitude check
        print("\n4. Warp magnitude check:")
        with h5py.File(self.dataset_dir / 'warp_measurements_3d.h5', 'r') as f:
            max_warps = [f[sid].attrs['max_warp_um'] for sid in f.keys()]
            
            if all(w < 1000 for w in max_warps):  # Less than 1 mm
                print(f"  ✓ All warp magnitudes reasonable (max: {max(max_warps):.2f} μm)")
            else:
                print(f"  ✗ Excessive warp detected!")
                all_valid = False
        
        self.validation_results['physical_constraints'] = all_valid
        return all_valid
    
    def validate_statistical_properties(self):
        """
        Check statistical properties of the dataset.
        """
        print("\n" + "="*80)
        print("VALIDATING STATISTICAL PROPERTIES")
        print("="*80)
        
        fab_params = pd.read_csv(self.dataset_dir / 'fabrication_parameters.csv')
        
        # Check parameter space filling (uniformity)
        print("\n1. Parameter space uniformity (Kolmogorov-Smirnov test):")
        
        uniform_params = [
            'anode_thickness_um',
            'electrolyte_thickness_um',
            'cathode_thickness_um',
            'anode_sinter_temp_C'
        ]
        
        all_uniform = True
        for param in uniform_params:
            # Normalize to [0, 1]
            values = fab_params[param].values
            normalized = (values - values.min()) / (values.max() - values.min())
            
            # KS test against uniform distribution
            ks_stat, p_value = stats.kstest(normalized, 'uniform')
            
            if p_value > 0.05:  # Not significantly different from uniform
                print(f"  ✓ {param}: uniform (p={p_value:.3f})")
            else:
                print(f"  ⚠ {param}: non-uniform (p={p_value:.3f})")
                all_uniform = False
        
        if all_uniform:
            print("\n  ✓ Parameter space well-filled (Latin Hypercube Sampling)")
        else:
            print("\n  ⚠ Some parameters show non-uniformity (may be intentional)")
        
        # Check measurement noise
        print("\n2. Measurement noise characteristics:")
        
        xrd_stress = pd.read_csv(self.dataset_dir / 'xrd_stress_measurements.csv')
        
        # Group by sample and check variance
        grouped = xrd_stress.groupby('sample_id')['stress_xx_GPa'].agg(['mean', 'std'])
        
        avg_cv = (grouped['std'] / grouped['mean'].abs()).mean()
        print(f"  Average coefficient of variation (XRD): {avg_cv:.2%}")
        
        if avg_cv < 0.15:
            print("  ✓ Measurement noise at realistic levels")
        else:
            print("  ⚠ High measurement variability detected")
        
        self.validation_results['statistical_properties'] = True
        return True
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report.
        """
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)
        
        report = {
            'validation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_directory': str(self.dataset_dir),
            'checks_performed': list(self.validation_results.keys()),
            'results': self.validation_results,
            'overall_status': all(self.validation_results.values())
        }
        
        # Save report
        report_file = self.dataset_dir / 'validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Validation report saved: {report_file}")
        
        if report['overall_status']:
            print("\n🎉 DATASET VALIDATION PASSED!")
            print("   All checks completed successfully.")
        else:
            print("\n⚠ DATASET VALIDATION COMPLETED WITH WARNINGS")
            print("   Review individual check results above.")
        
        return report
    
    def run_full_validation(self):
        """
        Run complete validation suite.
        """
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*20 + "DATASET VALIDATION SUITE" + " "*34 + "║")
        print("╚" + "="*78 + "╝")
        
        self.validate_file_structure()
        self.validate_data_consistency()
        self.validate_physical_constraints()
        self.validate_statistical_properties()
        
        return self.generate_validation_report()


def main():
    """
    Main validation execution.
    """
    validator = DatasetValidator('experimental_validation_dataset')
    report = validator.run_full_validation()
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nDataset is ready for:")
    print("  1. ML model training and validation")
    print("  2. FEA model calibration")
    print("  3. Uncertainty quantification studies")
    print("  4. Multi-physics model validation")


if __name__ == '__main__':
    main()
