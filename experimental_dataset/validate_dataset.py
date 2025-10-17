#!/usr/bin/env python3
"""
Dataset Validation Script
High-Temperature Experimental Dataset for Fire-Resistant Rubberized Concrete

This script validates the quality and consistency of the generated experimental dataset.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

class DatasetValidator:
    def __init__(self, dataset_path='/workspace/experimental_dataset'):
        self.dataset_path = Path(dataset_path)
        self.dataset = None
        self.validation_results = {
            'passed_checks': 0,
            'total_checks': 0,
            'issues': [],
            'warnings': []
        }
        self.load_dataset()
    
    def load_dataset(self):
        """Load the complete experimental dataset"""
        with open(self.dataset_path / 'complete_experimental_dataset.json', 'r') as f:
            self.dataset = json.load(f)
        print("Dataset loaded for validation...")
    
    def check(self, condition, message, is_warning=False):
        """Generic check function"""
        self.validation_results['total_checks'] += 1
        if condition:
            self.validation_results['passed_checks'] += 1
            print(f"✓ {message}")
        else:
            if is_warning:
                self.validation_results['warnings'].append(message)
                print(f"⚠ {message}")
            else:
                self.validation_results['issues'].append(message)
                print(f"✗ {message}")
    
    def validate_data_completeness(self):
        """Validate that all expected data is present"""
        print("\n=== Data Completeness Validation ===")
        
        expected_mix_types = ['Control', 'R5', 'R10', 'R15', 'R20', 'Raw_Rubber']
        
        for mix_type in expected_mix_types:
            self.check(mix_type in self.dataset['thermal_properties'], 
                      f"Thermal properties data present for {mix_type}")
            
            if mix_type != 'Raw_Rubber':  # Raw rubber doesn't have mechanical/spalling data
                self.check(mix_type in self.dataset['mechanical_testing'], 
                          f"Mechanical testing data present for {mix_type}")
                self.check(mix_type in self.dataset['spalling_durability'], 
                          f"Spalling/durability data present for {mix_type}")
    
    def validate_temperature_consistency(self):
        """Validate temperature ranges are consistent across datasets"""
        print("\n=== Temperature Consistency Validation ===")
        
        # Check thermal properties temperature range
        thermal_temp_range = self.dataset['thermal_properties']['Control']['tga']['temperature']
        expected_temp_range = list(range(20, 801))  # 20°C to 800°C
        
        # Handle the case where temperature data might be stored as string representation
        if isinstance(thermal_temp_range, str):
            # Try to evaluate the string as a numpy array
            try:
                thermal_temp_array = eval(thermal_temp_range)
            except:
                thermal_temp_array = np.array([float(x) for x in thermal_temp_range.strip('[]').split()])
        else:
            thermal_temp_array = np.array(thermal_temp_range)
        
        self.check(len(thermal_temp_array) == len(expected_temp_range), 
                  f"Thermal properties temperature range has correct number of points: {len(thermal_temp_array)}")
        
        self.check(abs(thermal_temp_array[0] - 20) < 1, 
                  f"Thermal properties start temperature: {thermal_temp_array[0]:.1f}°C")
        
        self.check(abs(thermal_temp_array[-1] - 800) < 1, 
                  f"Thermal properties end temperature: {thermal_temp_array[-1]:.1f}°C")
        
        # Check mechanical testing temperatures
        mechanical_temps = [25, 100, 200, 400, 600, 800]
        for mix_type in ['Control', 'R5', 'R10', 'R15', 'R20']:
            for temp in mechanical_temps:
                temp_key = f'{temp}C'
                self.check(temp_key in self.dataset['mechanical_testing'][mix_type]['tts_compressive'], 
                          f"Mechanical data present for {mix_type} at {temp}°C")
    
    def validate_rubber_content_consistency(self):
        """Validate rubber content values are consistent"""
        print("\n=== Rubber Content Consistency Validation ===")
        
        expected_rubber_contents = {
            'Control': 0.0,
            'R5': 0.05,
            'R10': 0.10,
            'R15': 0.15,
            'R20': 0.20,
            'Raw_Rubber': 1.0
        }
        
        for mix_type, expected_content in expected_rubber_contents.items():
            actual_content = self.dataset['metadata']['mix_types'][mix_type]['rubber_content']
            self.check(abs(actual_content - expected_content) < 0.001, 
                      f"Rubber content for {mix_type}: {actual_content} (expected: {expected_content})")
    
    def validate_data_ranges(self):
        """Validate that data values are within reasonable ranges"""
        print("\n=== Data Range Validation ===")
        
        # Check thermal conductivity values
        for mix_type in ['Control', 'R5', 'R10', 'R15', 'R20']:
            k_values = self.dataset['thermal_properties'][mix_type]['thermal_conductivity']['thermal_conductivity']
            self.check(all(0.1 <= k <= 5.0 for k in k_values), 
                      f"Thermal conductivity values reasonable for {mix_type}: {min(k_values):.3f}-{max(k_values):.3f} W/m·K")
        
        # Check compressive strength values
        for mix_type in ['Control', 'R5', 'R10', 'R15', 'R20']:
            for temp_key, temp_data in self.dataset['mechanical_testing'][mix_type]['tts_compressive'].items():
                strength = temp_data['peak_strength']
                self.check(0 <= strength <= 100, 
                          f"Compressive strength reasonable for {mix_type} at {temp_key}: {strength:.2f} MPa")
        
        # Check mass loss values
        for mix_type in ['Control', 'R5', 'R10', 'R15', 'R20']:
            mass_loss = self.dataset['thermal_properties'][mix_type]['tga']['mass_loss_percent']
            self.check(all(0 <= ml <= 100 for ml in mass_loss), 
                      f"Mass loss values reasonable for {mix_type}: {min(mass_loss):.2f}-{max(mass_loss):.2f}%")
    
    def validate_temperature_dependencies(self):
        """Validate that data shows expected temperature dependencies"""
        print("\n=== Temperature Dependency Validation ===")
        
        # Check that thermal conductivity generally decreases with temperature
        for mix_type in ['Control', 'R5', 'R10', 'R15', 'R20']:
            k_data = self.dataset['thermal_properties'][mix_type]['thermal_conductivity']
            k_values = k_data['thermal_conductivity']
            # Should generally decrease (allow for some noise)
            decreasing_trend = sum(k_values[i] >= k_values[i+1] for i in range(len(k_values)-1)) / (len(k_values)-1)
            self.check(decreasing_trend >= 0.6, 
                      f"Thermal conductivity shows decreasing trend for {mix_type}: {decreasing_trend:.2f}")
        
        # Check that compressive strength decreases with temperature
        for mix_type in ['Control', 'R5', 'R10', 'R15', 'R20']:
            strengths = []
            temps = []
            for temp_key, temp_data in self.dataset['mechanical_testing'][mix_type]['tts_compressive'].items():
                temps.append(temp_data['temperature'])
                strengths.append(temp_data['peak_strength'])
            temps, strengths = zip(*sorted(zip(temps, strengths)))
            
            # Should generally decrease
            decreasing_trend = sum(strengths[i] >= strengths[i+1] for i in range(len(strengths)-1)) / (len(strengths)-1)
            self.check(decreasing_trend >= 0.5, 
                      f"Compressive strength shows decreasing trend for {mix_type}: {decreasing_trend:.2f}")
    
    def validate_rubber_effects(self):
        """Validate that rubber content has expected effects"""
        print("\n=== Rubber Content Effects Validation ===")
        
        # Check that thermal conductivity decreases with rubber content
        k_values_25C = []
        rubber_contents = []
        for mix_type in ['Control', 'R5', 'R10', 'R15', 'R20']:
            k_25C = self.dataset['thermal_properties'][mix_type]['thermal_conductivity']['thermal_conductivity'][0]
            rubber_content = self.dataset['metadata']['mix_types'][mix_type]['rubber_content']
            k_values_25C.append(k_25C)
            rubber_contents.append(rubber_content)
        
        # Should generally decrease with rubber content
        correlation = np.corrcoef(rubber_contents, k_values_25C)[0, 1]
        self.check(correlation < -0.5, 
                  f"Thermal conductivity decreases with rubber content: correlation = {correlation:.3f}")
        
        # Check that spalling events decrease with rubber content
        spalling_events = []
        for mix_type in ['Control', 'R5', 'R10', 'R15', 'R20']:
            events = len(self.dataset['spalling_durability'][mix_type]['spalling_events']['spalling_events'])
            spalling_events.append(events)
        
        correlation = np.corrcoef(rubber_contents, spalling_events)[0, 1]
        self.check(correlation < -0.3, 
                  f"Spalling events decrease with rubber content: correlation = {correlation:.3f}")
    
    def validate_file_structure(self):
        """Validate that all expected files are present"""
        print("\n=== File Structure Validation ===")
        
        # Check main files
        main_files = [
            'complete_experimental_dataset.json',
            'dataset_metadata.json',
            'experimental_data_summary.csv',
            'summary_statistics.json',
            'data_quality_report.json'
        ]
        
        for file in main_files:
            self.check((self.dataset_path / file).exists(), 
                      f"Main file present: {file}")
        
        # Check directory structure
        expected_dirs = ['thermal_properties', 'mechanical_testing', 'spalling_durability', 'comprehensive_plots']
        for dir_name in expected_dirs:
            self.check((self.dataset_path / dir_name).exists(), 
                      f"Directory present: {dir_name}")
    
    def generate_validation_report(self):
        """Generate a comprehensive validation report"""
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        
        total_checks = self.validation_results['total_checks']
        passed_checks = self.validation_results['passed_checks']
        success_rate = (passed_checks / total_checks) * 100
        
        print(f"\nValidation Summary:")
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.validation_results['issues']:
            print(f"\nIssues Found ({len(self.validation_results['issues'])}):")
            for issue in self.validation_results['issues']:
                print(f"  • {issue}")
        
        if self.validation_results['warnings']:
            print(f"\nWarnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings']:
                print(f"  • {warning}")
        
        if success_rate >= 90:
            print(f"\n✓ Dataset validation PASSED! ({success_rate:.1f}% success rate)")
            print("Dataset is ready for use in thermo-mechanical model validation.")
        elif success_rate >= 80:
            print(f"\n⚠ Dataset validation PASSED with warnings ({success_rate:.1f}% success rate)")
            print("Dataset is usable but review warnings.")
        else:
            print(f"\n✗ Dataset validation FAILED ({success_rate:.1f}% success rate)")
            print("Dataset needs attention before use.")
        
        # Save validation report
        validation_report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'success_rate': success_rate,
            'issues': self.validation_results['issues'],
            'warnings': self.validation_results['warnings']
        }
        
        with open(self.dataset_path / 'validation_report.json', 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"\nValidation report saved to: {self.dataset_path / 'validation_report.json'}")
    
    def run_full_validation(self):
        """Run complete dataset validation"""
        print("Starting comprehensive dataset validation...")
        print("="*60)
        
        self.validate_data_completeness()
        self.validate_temperature_consistency()
        self.validate_rubber_content_consistency()
        self.validate_data_ranges()
        self.validate_temperature_dependencies()
        self.validate_rubber_effects()
        self.validate_file_structure()
        
        self.generate_validation_report()

def main():
    """Main function to run dataset validation"""
    validator = DatasetValidator()
    validator.run_full_validation()

if __name__ == "__main__":
    main()