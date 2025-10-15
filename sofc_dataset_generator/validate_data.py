"""
Data validation and quality checks for SOFC dataset
"""

import numpy as np
import pandas as pd
import h5py
import json
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

class SOFCDataValidator:
    """Validate and check quality of SOFC dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.validation_results = {}
        self.warnings = []
        self.errors = []
        
    def validate_dataset(self) -> Dict:
        """Run complete validation suite"""
        
        print("Running dataset validation...")
        print("-" * 50)
        
        # 1. Check file integrity
        self._check_file_integrity()
        
        # 2. Validate data structure
        self._validate_structure()
        
        # 3. Check parameter ranges
        self._check_parameter_ranges()
        
        # 4. Validate physical consistency
        self._check_physical_consistency()
        
        # 5. Check data quality
        self._check_data_quality()
        
        # 6. Validate fidelity hierarchy
        self._validate_fidelity_hierarchy()
        
        # 7. Check response variables
        self._check_response_validity()
        
        # Compile results
        self.validation_results['summary'] = {
            'total_checks': len(self.validation_results),
            'passed': sum(1 for v in self.validation_results.values() 
                         if isinstance(v, dict) and v.get('status') == 'PASS'),
            'warnings': len(self.warnings),
            'errors': len(self.errors)
        }
        
        self._print_validation_summary()
        
        return self.validation_results
    
    def _check_file_integrity(self):
        """Check HDF5 file integrity"""
        result = {'status': 'PASS', 'details': {}}
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                # Check required groups
                required_groups = ['LF', 'MF', 'HF']
                missing_groups = [g for g in required_groups if g not in f]
                
                if missing_groups:
                    result['status'] = 'FAIL'
                    result['details']['missing_groups'] = missing_groups
                    self.errors.append(f"Missing fidelity groups: {missing_groups}")
                
                # Check time vector
                if 'time_hours' not in f:
                    result['status'] = 'FAIL'
                    result['details']['time_vector'] = 'Missing'
                    self.errors.append("Time vector missing")
                else:
                    time = f['time_hours'][:]
                    result['details']['time_points'] = len(time)
                    result['details']['time_range'] = [float(time[0]), float(time[-1])]
                
                # Check data sizes
                for fidelity in ['LF', 'MF', 'HF']:
                    if fidelity in f:
                        n_samples = len(f[fidelity]['responses'].keys())
                        result['details'][f'{fidelity}_samples'] = n_samples
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['details']['error'] = str(e)
            self.errors.append(f"File integrity check failed: {e}")
        
        self.validation_results['file_integrity'] = result
    
    def _validate_structure(self):
        """Validate dataset structure"""
        result = {'status': 'PASS', 'details': {}}
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                for fidelity in ['LF', 'MF', 'HF']:
                    if fidelity in f:
                        fid_details = {}
                        
                        # Check inputs structure
                        if 'inputs' not in f[fidelity]:
                            fid_details['inputs'] = 'Missing'
                            self.errors.append(f"{fidelity}: Missing inputs")
                            result['status'] = 'FAIL'
                        else:
                            n_params = len(f[fidelity]['inputs'].keys())
                            fid_details['n_parameters'] = n_params
                        
                        # Check responses structure
                        if 'responses' not in f[fidelity]:
                            fid_details['responses'] = 'Missing'
                            self.errors.append(f"{fidelity}: Missing responses")
                            result['status'] = 'FAIL'
                        else:
                            # Check sample structure
                            sample_keys = list(f[fidelity]['responses'].keys())
                            if sample_keys:
                                sample = f[fidelity]['responses'][sample_keys[0]]
                                fid_details['response_groups'] = list(sample.keys())
                        
                        result['details'][fidelity] = fid_details
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['details']['error'] = str(e)
            self.errors.append(f"Structure validation failed: {e}")
        
        self.validation_results['data_structure'] = result
    
    def _check_parameter_ranges(self):
        """Check if parameters are within valid ranges"""
        result = {'status': 'PASS', 'details': {}}
        
        # Define physical limits
        param_limits = {
            'temperature': (773, 1273),  # K
            'pressure': (0.5, 10),  # bar
            'current_density': (0, 2.0),  # A/cm²
            'fuel_utilization': (0, 1),
            'oxidant_utilization': (0, 1),
            'porosity': (0, 0.7),
            'thickness': (0, 5000),  # μm
        }
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                for fidelity in ['LF', 'MF', 'HF']:
                    if fidelity in f and 'inputs' in f[fidelity]:
                        violations = []
                        
                        for param_key in f[fidelity]['inputs'].keys():
                            data = f[fidelity]['inputs'][param_key][:]
                            
                            # Check against known limits
                            for limit_key, (min_val, max_val) in param_limits.items():
                                if limit_key in param_key.lower():
                                    if np.any(data < min_val) or np.any(data > max_val):
                                        violations.append({
                                            'parameter': param_key,
                                            'min': float(np.min(data)),
                                            'max': float(np.max(data)),
                                            'expected_range': (min_val, max_val)
                                        })
                                        self.warnings.append(
                                            f"{fidelity}/{param_key}: Values outside expected range"
                                        )
                            
                            # Check for NaN or Inf
                            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                                violations.append({
                                    'parameter': param_key,
                                    'issue': 'Contains NaN or Inf values'
                                })
                                self.errors.append(f"{fidelity}/{param_key}: Contains invalid values")
                                result['status'] = 'FAIL'
                        
                        if violations:
                            result['details'][f'{fidelity}_violations'] = violations
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['details']['error'] = str(e)
            self.errors.append(f"Parameter range check failed: {e}")
        
        self.validation_results['parameter_ranges'] = result
    
    def _check_physical_consistency(self):
        """Check physical consistency of parameters"""
        result = {'status': 'PASS', 'details': {}}
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                for fidelity in ['LF', 'MF', 'HF']:
                    if fidelity in f and 'inputs' in f[fidelity]:
                        consistency_checks = []
                        
                        # Collect parameter data
                        params = {}
                        for key in f[fidelity]['inputs'].keys():
                            params[key] = f[fidelity]['inputs'][key][:]
                        
                        # Check 1: Volume fractions should sum to ~1
                        volume_params = [k for k in params.keys() if 'volume_fraction' in k or 'porosity' in k]
                        if len(volume_params) >= 2:
                            # Group by component (anode, cathode)
                            for component in ['anode', 'cathode']:
                                comp_params = [k for k in volume_params if component in k]
                                if len(comp_params) >= 2:
                                    total = sum(params[k] for k in comp_params)
                                    if np.any(total > 1.1) or np.any(total < 0.9):
                                        consistency_checks.append({
                                            'check': f'{component} volume fractions',
                                            'issue': 'Sum deviates from 1.0',
                                            'range': [float(np.min(total)), float(np.max(total))]
                                        })
                                        self.warnings.append(
                                            f"{fidelity}: {component} volume fractions sum issue"
                                        )
                        
                        # Check 2: Thickness ratios
                        thickness_params = [k for k in params.keys() if 'thickness' in k]
                        if 'electrolyte_thickness' in params and 'anode_thickness' in params:
                            ratio = params['anode_thickness'] / params['electrolyte_thickness']
                            if np.any(ratio < 10) or np.any(ratio > 200):
                                consistency_checks.append({
                                    'check': 'Anode/Electrolyte thickness ratio',
                                    'issue': 'Unusual ratio',
                                    'range': [float(np.min(ratio)), float(np.max(ratio))]
                                })
                                self.warnings.append(f"{fidelity}: Unusual thickness ratio")
                        
                        # Check 3: Conductivity relationships
                        if 'temperature' in params:
                            temp_data = params['temperature']
                            # Ionic conductivity should increase with temperature
                            ionic_params = [k for k in params.keys() if 'ionic_conductivity' in k]
                            for ionic_param in ionic_params:
                                correlation = np.corrcoef(temp_data, params[ionic_param])[0, 1]
                                if correlation < 0:
                                    consistency_checks.append({
                                        'check': f'Temperature vs {ionic_param}',
                                        'issue': 'Negative correlation',
                                        'correlation': float(correlation)
                                    })
                                    self.warnings.append(
                                        f"{fidelity}: Unexpected temperature-conductivity relationship"
                                    )
                        
                        if consistency_checks:
                            result['details'][f'{fidelity}_consistency'] = consistency_checks
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['details']['error'] = str(e)
            self.errors.append(f"Physical consistency check failed: {e}")
        
        self.validation_results['physical_consistency'] = result
    
    def _check_data_quality(self):
        """Check data quality metrics"""
        result = {'status': 'PASS', 'details': {}}
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                for fidelity in ['LF', 'MF', 'HF']:
                    if fidelity in f and 'inputs' in f[fidelity]:
                        quality_metrics = {}
                        
                        # Check for duplicates
                        all_params = []
                        for key in f[fidelity]['inputs'].keys():
                            if 'sample_id' not in key:
                                all_params.append(f[fidelity]['inputs'][key][:])
                        
                        if all_params:
                            param_matrix = np.column_stack(all_params)
                            unique_rows = np.unique(param_matrix, axis=0)
                            duplicate_ratio = 1 - len(unique_rows) / len(param_matrix)
                            
                            quality_metrics['duplicate_ratio'] = float(duplicate_ratio)
                            if duplicate_ratio > 0.01:
                                self.warnings.append(
                                    f"{fidelity}: {duplicate_ratio:.1%} duplicate samples"
                                )
                        
                        # Check distribution quality (uniformity for LHS)
                        for key in f[fidelity]['inputs'].keys():
                            if 'sample_id' not in key and 'fidelity' not in key:
                                data = f[fidelity]['inputs'][key][:]
                                
                                # Kolmogorov-Smirnov test for uniformity
                                if len(data) > 10:
                                    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
                                    ks_stat, ks_p = stats.kstest(normalized, 'uniform')
                                    
                                    if ks_p < 0.05:
                                        quality_metrics[f'{key}_uniformity'] = {
                                            'ks_statistic': float(ks_stat),
                                            'p_value': float(ks_p)
                                        }
                        
                        # Check for outliers
                        outlier_params = []
                        for key in f[fidelity]['inputs'].keys():
                            if 'sample_id' not in key and 'fidelity' not in key:
                                data = f[fidelity]['inputs'][key][:]
                                q1, q3 = np.percentile(data, [25, 75])
                                iqr = q3 - q1
                                outliers = np.sum((data < q1 - 3*iqr) | (data > q3 + 3*iqr))
                                
                                if outliers > 0:
                                    outlier_params.append({
                                        'parameter': key,
                                        'n_outliers': int(outliers),
                                        'percentage': float(100 * outliers / len(data))
                                    })
                        
                        if outlier_params:
                            quality_metrics['outliers'] = outlier_params
                        
                        result['details'][fidelity] = quality_metrics
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['details']['error'] = str(e)
            self.errors.append(f"Data quality check failed: {e}")
        
        self.validation_results['data_quality'] = result
    
    def _validate_fidelity_hierarchy(self):
        """Validate multi-fidelity hierarchy"""
        result = {'status': 'PASS', 'details': {}}
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                # Check nested structure
                n_lf = len(f['LF']['responses'].keys()) if 'LF' in f else 0
                n_mf = len(f['MF']['responses'].keys()) if 'MF' in f else 0
                n_hf = len(f['HF']['responses'].keys()) if 'HF' in f else 0
                
                result['details']['sample_counts'] = {
                    'LF': n_lf,
                    'MF': n_mf,
                    'HF': n_hf
                }
                
                # Check hierarchy (LF > MF > HF)
                if not (n_lf >= n_mf >= n_hf):
                    result['status'] = 'WARNING'
                    result['details']['hierarchy_violated'] = True
                    self.warnings.append("Fidelity hierarchy not maintained (expected LF > MF > HF)")
                
                # Check parameter consistency across fidelities
                param_consistency = {}
                lf_params = set(f['LF']['inputs'].keys()) if 'LF' in f else set()
                mf_params = set(f['MF']['inputs'].keys()) if 'MF' in f else set()
                hf_params = set(f['HF']['inputs'].keys()) if 'HF' in f else set()
                
                # LF params should be subset of MF and HF
                if lf_params and mf_params:
                    missing_in_mf = lf_params - mf_params
                    if missing_in_mf:
                        param_consistency['lf_not_in_mf'] = list(missing_in_mf)
                        self.warnings.append(f"LF parameters missing in MF: {missing_in_mf}")
                
                if lf_params and hf_params:
                    missing_in_hf = lf_params - hf_params
                    if missing_in_hf:
                        param_consistency['lf_not_in_hf'] = list(missing_in_hf)
                        self.warnings.append(f"LF parameters missing in HF: {missing_in_hf}")
                
                if param_consistency:
                    result['details']['parameter_consistency'] = param_consistency
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['details']['error'] = str(e)
            self.errors.append(f"Fidelity hierarchy validation failed: {e}")
        
        self.validation_results['fidelity_hierarchy'] = result
    
    def _check_response_validity(self):
        """Check validity of response variables"""
        result = {'status': 'PASS', 'details': {}}
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                time = f['time_hours'][:]
                
                for fidelity in ['LF', 'MF', 'HF']:
                    if fidelity in f and 'responses' in f[fidelity]:
                        response_checks = []
                        
                        # Sample first few responses
                        sample_keys = list(f[fidelity]['responses'].keys())[:5]
                        
                        for sample_key in sample_keys:
                            sample = f[fidelity]['responses'][sample_key]
                            
                            # Check degradation data
                            if 'degradation' in sample:
                                deg = sample['degradation']
                                
                                # Voltage should be positive and decreasing
                                if 'voltage' in deg:
                                    voltage = deg['voltage'][:]
                                    if np.any(voltage <= 0):
                                        response_checks.append({
                                            'sample': sample_key,
                                            'issue': 'Negative voltage detected'
                                        })
                                        self.errors.append(f"{fidelity}/{sample_key}: Invalid voltage")
                                        result['status'] = 'FAIL'
                                    
                                    # Check for monotonic degradation (mostly)
                                    if len(voltage) > 1:
                                        increasing_points = np.sum(np.diff(voltage) > 0)
                                        if increasing_points > len(voltage) * 0.1:
                                            response_checks.append({
                                                'sample': sample_key,
                                                'issue': 'Non-monotonic voltage degradation',
                                                'increasing_ratio': float(increasing_points / len(voltage))
                                            })
                                            self.warnings.append(
                                                f"{fidelity}/{sample_key}: Unexpected voltage increase"
                                            )
                                
                                # ASR should be positive and increasing
                                if 'ASR' in deg:
                                    asr = deg['ASR'][:]
                                    if np.any(asr <= 0):
                                        response_checks.append({
                                            'sample': sample_key,
                                            'issue': 'Negative ASR detected'
                                        })
                                        self.errors.append(f"{fidelity}/{sample_key}: Invalid ASR")
                                        result['status'] = 'FAIL'
                                
                                # Efficiency should be between 0 and 1
                                if 'efficiency' in deg:
                                    eff = deg['efficiency'][:]
                                    if np.any(eff < 0) or np.any(eff > 1):
                                        response_checks.append({
                                            'sample': sample_key,
                                            'issue': 'Efficiency outside [0, 1]',
                                            'range': [float(np.min(eff)), float(np.max(eff))]
                                        })
                                        self.warnings.append(f"{fidelity}/{sample_key}: Efficiency out of range")
                        
                        if response_checks:
                            result['details'][f'{fidelity}_response_issues'] = response_checks
        
        except Exception as e:
            result['status'] = 'FAIL'
            result['details']['error'] = str(e)
            self.errors.append(f"Response validation failed: {e}")
        
        self.validation_results['response_validity'] = result
    
    def _print_validation_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        for check_name, result in self.validation_results.items():
            if check_name == 'summary':
                continue
            
            if isinstance(result, dict):
                status = result.get('status', 'UNKNOWN')
                status_symbol = {
                    'PASS': '✓',
                    'WARNING': '⚠',
                    'FAIL': '✗',
                    'UNKNOWN': '?'
                }.get(status, '?')
                
                print(f"{status_symbol} {check_name:30s} {status}")
        
        print("\n" + "-" * 70)
        summary = self.validation_results.get('summary', {})
        print(f"Total Checks: {summary.get('total_checks', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Warnings: {summary.get('warnings', 0)}")
        print(f"Errors: {summary.get('errors', 0)}")
        
        if self.errors:
            print("\nERRORS:")
            for error in self.errors[:5]:
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings[:5]:
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        print("=" * 70)
    
    def export_validation_report(self, output_path: str = 'validation_report.json'):
        """Export validation results to JSON"""
        report = {
            'dataset_path': self.dataset_path,
            'validation_results': self.validation_results,
            'warnings': self.warnings,
            'errors': self.errors
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nValidation report saved to: {output_path}")

def validate_dataset(dataset_path: str) -> Dict:
    """Convenience function to validate a dataset"""
    validator = SOFCDataValidator(dataset_path)
    results = validator.validate_dataset()
    validator.export_validation_report()
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate SOFC dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to HDF5 dataset file')
    
    args = parser.parse_args()
    
    validate_dataset(args.dataset)