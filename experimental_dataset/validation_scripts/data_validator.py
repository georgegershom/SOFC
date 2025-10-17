#!/usr/bin/env python3
"""
Data Validation Script for High-Temperature Experimental Dataset
Validates data quality, consistency, and completeness
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.validation_results = {}
        
    def validate_file_structure(self):
        """Validate the overall file structure"""
        print("Validating file structure...")
        
        expected_structure = {
            'thermal_properties': ['tga_dsc', 'thermal_conductivity', 'cte', 'mass_loss'],
            'mechanical_testing': ['tts_curves', 'stt_tests', 'residual_properties'],
            'spalling_durability': ['visual_audio', 'vapor_pressure', 'gas_permeability', 'microstructural'],
            'metadata': ['sample_specifications.json', 'test_protocols.json', 'data_dictionary.json']
        }
        
        structure_valid = True
        missing_dirs = []
        
        for category, subdirs in expected_structure.items():
            category_path = self.dataset_path / category
            if not category_path.exists():
                missing_dirs.append(f"Missing category: {category}")
                structure_valid = False
                continue
                
            for subdir in subdirs:
                subdir_path = category_path / subdir
                if not subdir_path.exists():
                    missing_dirs.append(f"Missing subdirectory: {category}/{subdir}")
                    structure_valid = False
        
        self.validation_results['file_structure'] = {
            'valid': structure_valid,
            'missing_dirs': missing_dirs
        }
        
        return structure_valid
    
    def validate_data_completeness(self):
        """Validate data completeness for all categories"""
        print("Validating data completeness...")
        
        completeness_results = {}
        
        # Check thermal properties
        thermal_path = self.dataset_path / 'thermal_properties'
        if thermal_path.exists():
            completeness_results['thermal_properties'] = self._check_thermal_completeness(thermal_path)
        
        # Check mechanical testing
        mechanical_path = self.dataset_path / 'mechanical_testing'
        if mechanical_path.exists():
            completeness_results['mechanical_testing'] = self._check_mechanical_completeness(mechanical_path)
        
        # Check spalling durability
        spalling_path = self.dataset_path / 'spalling_durability'
        if spalling_path.exists():
            completeness_results['spalling_durability'] = self._check_spalling_completeness(spalling_path)
        
        self.validation_results['data_completeness'] = completeness_results
        return completeness_results
    
    def _check_thermal_completeness(self, thermal_path):
        """Check thermal properties data completeness"""
        results = {}
        
        # Check TGA/DSC data
        tga_path = thermal_path / 'tga_dsc'
        if tga_path.exists():
            json_files = list(tga_path.glob('*.json'))
            csv_files = list(tga_path.glob('*.csv'))
            results['tga_dsc'] = {
                'json_files': len(json_files),
                'csv_files': len(csv_files),
                'complete': len(json_files) > 0 and len(csv_files) > 0
            }
        
        # Check thermal conductivity
        tc_path = thermal_path / 'thermal_conductivity'
        if tc_path.exists():
            json_files = list(tc_path.glob('*.json'))
            csv_files = list(tc_path.glob('*.csv'))
            results['thermal_conductivity'] = {
                'json_files': len(json_files),
                'csv_files': len(csv_files),
                'complete': len(json_files) > 0 and len(csv_files) > 0
            }
        
        return results
    
    def _check_mechanical_completeness(self, mechanical_path):
        """Check mechanical testing data completeness"""
        results = {}
        
        # Check TTS curves
        tts_path = mechanical_path / 'tts_curves'
        if tts_path.exists():
            json_files = list(tts_path.glob('*.json'))
            csv_files = list(tts_path.glob('*.csv'))
            results['tts_curves'] = {
                'json_files': len(json_files),
                'csv_files': len(csv_files),
                'complete': len(json_files) > 0 and len(csv_files) > 0
            }
        
        # Check STT tests
        stt_path = mechanical_path / 'stt_tests'
        if stt_path.exists():
            json_files = list(stt_path.glob('*.json'))
            csv_files = list(stt_path.glob('*.csv'))
            results['stt_tests'] = {
                'json_files': len(json_files),
                'csv_files': len(csv_files),
                'complete': len(json_files) > 0 and len(csv_files) > 0
            }
        
        return results
    
    def _check_spalling_completeness(self, spalling_path):
        """Check spalling durability data completeness"""
        results = {}
        
        # Check vapor pressure
        vp_path = spalling_path / 'vapor_pressure'
        if vp_path.exists():
            json_files = list(vp_path.glob('*.json'))
            csv_files = list(vp_path.glob('*.csv'))
            results['vapor_pressure'] = {
                'json_files': len(json_files),
                'csv_files': len(csv_files),
                'complete': len(json_files) > 0 and len(csv_files) > 0
            }
        
        # Check permeability
        perm_path = spalling_path / 'gas_permeability'
        if perm_path.exists():
            json_files = list(perm_path.glob('*.json'))
            csv_files = list(perm_path.glob('*.csv'))
            results['gas_permeability'] = {
                'json_files': len(json_files),
                'csv_files': len(csv_files),
                'complete': len(json_files) > 0 and len(csv_files) > 0
            }
        
        return results
    
    def validate_data_quality(self):
        """Validate data quality and consistency"""
        print("Validating data quality...")
        
        quality_results = {}
        
        # Check thermal properties data quality
        thermal_path = self.dataset_path / 'thermal_properties'
        if thermal_path.exists():
            quality_results['thermal_properties'] = self._check_thermal_quality(thermal_path)
        
        # Check mechanical testing data quality
        mechanical_path = self.dataset_path / 'mechanical_testing'
        if mechanical_path.exists():
            quality_results['mechanical_testing'] = self._check_mechanical_quality(mechanical_path)
        
        self.validation_results['data_quality'] = quality_results
        return quality_results
    
    def _check_thermal_quality(self, thermal_path):
        """Check thermal properties data quality"""
        results = {}
        
        # Check TGA/DSC data
        tga_path = thermal_path / 'tga_dsc'
        if tga_path.exists():
            json_file = tga_path / 'tga_dsc_data.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check data ranges
                results['tga_dsc'] = self._validate_tga_data(data)
        
        return results
    
    def _validate_tga_data(self, data):
        """Validate TGA/DSC data quality"""
        issues = []
        
        try:
            for mix, mix_data in data.items():
                if mix == 'metadata':
                    continue
                    
                for replicate in mix_data.get('tga', []):
                    tga_data = replicate['data']
                    
                    # Check temperature range
                    temps = tga_data['temperature']
                    if isinstance(temps, list) and len(temps) > 0:
                        temps_array = np.array(temps)
                        if np.min(temps_array) < 20 or np.max(temps_array) > 850:
                            issues.append(f"Temperature range issue in {mix}")
                    
                    # Check mass loss values
                    mass_loss = tga_data['mass_loss_percent']
                    if isinstance(mass_loss, list) and len(mass_loss) > 0:
                        mass_loss_array = np.array(mass_loss)
                        if np.min(mass_loss_array) < 0 or np.max(mass_loss_array) > 100:
                            issues.append(f"Mass loss range issue in {mix}")
                        
                        # Check for NaN values
                        if np.any(np.isnan(mass_loss_array)):
                            issues.append(f"NaN values in mass loss data for {mix}")
        except Exception as e:
            issues.append(f"Error validating TGA data: {str(e)}")
        
        return {
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def _check_mechanical_quality(self, mechanical_path):
        """Check mechanical testing data quality"""
        results = {}
        
        # Check TTS curves
        tts_path = mechanical_path / 'tts_curves'
        if tts_path.exists():
            json_file = tts_path / 'tts_data.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                results['tts_curves'] = self._validate_tts_data(data)
        
        return results
    
    def _validate_tts_data(self, data):
        """Validate TTS data quality"""
        issues = []
        
        for mix, mix_data in data.items():
            if mix == 'metadata':
                continue
                
            for temp, temp_data in mix_data.get('temperatures', {}).items():
                for replicate in temp_data.get('replicates', []):
                    # Check stress-strain data
                    stress = replicate['stress']
                    strain = replicate['strain']
                    
                    # Check for negative stress
                    if any(s < 0 for s in stress):
                        issues.append(f"Negative stress values in {mix} at {temp}°C")
                    
                    # Check for increasing strain
                    if not all(strain[i] <= strain[i+1] for i in range(len(strain)-1)):
                        issues.append(f"Non-increasing strain in {mix} at {temp}°C")
                    
                    # Check peak strength
                    peak_strength = replicate['peak_strength']
                    if peak_strength <= 0:
                        issues.append(f"Invalid peak strength in {mix} at {temp}°C")
        
        return {
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def generate_validation_report(self, output_path):
        """Generate comprehensive validation report"""
        print("Generating validation report...")
        
        # Run all validations
        self.validate_file_structure()
        self.validate_data_completeness()
        self.validate_data_quality()
        
        # Generate report
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'validation_results': self.validation_results
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        
        # File structure
        fs_result = self.validation_results.get('file_structure', {})
        print(f"File Structure: {'✓ PASS' if fs_result.get('valid', False) else '✗ FAIL'}")
        if fs_result.get('missing_dirs'):
            print("Missing directories:")
            for missing in fs_result['missing_dirs']:
                print(f"  - {missing}")
        
        # Data completeness
        dc_result = self.validation_results.get('data_completeness', {})
        print(f"\nData Completeness:")
        for category, results in dc_result.items():
            print(f"  {category}:")
            for subcategory, details in results.items():
                status = "✓" if details.get('complete', False) else "✗"
                print(f"    {subcategory}: {status} ({details.get('json_files', 0)} JSON, {details.get('csv_files', 0)} CSV)")
        
        # Data quality
        dq_result = self.validation_results.get('data_quality', {})
        print(f"\nData Quality:")
        for category, results in dq_result.items():
            print(f"  {category}:")
            for subcategory, details in results.items():
                status = "✓" if details.get('valid', False) else "✗"
                print(f"    {subcategory}: {status}")
                if details.get('issues'):
                    print(f"      Issues: {len(details['issues'])}")
                    for issue in details['issues'][:3]:  # Show first 3 issues
                        print(f"        - {issue}")
                    if len(details['issues']) > 3:
                        print(f"        ... and {len(details['issues']) - 3} more")
        
        print(f"\nValidation report saved to: {output_path}")
        return report

if __name__ == "__main__":
    validator = DatasetValidator("/workspace/experimental_dataset")
    report = validator.generate_validation_report("/workspace/experimental_dataset/validation_report.json")
    print("Validation completed!")