#!/usr/bin/env python3
"""
Data Validation Script for Retrofit Intervention & Lifecycle Data
Validates the integrity and consistency of the dataset
"""

import json
import os
import sys
from typing import Dict, List, Any

class RetrofitDataValidator:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.validation_results = {}
        
    def validate_json_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate JSON file structure and content"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            validation_result = {
                'file_path': file_path,
                'is_valid_json': True,
                'has_metadata': 'metadata' in data,
                'data_size': len(str(data)),
                'errors': []
            }
            
            # Check for required fields based on file type
            if 'retrofit_measures' in file_path:
                required_fields = ['retrofit_measures', 'metadata']
                for field in required_fields:
                    if field not in data:
                        validation_result['errors'].append(f"Missing required field: {field}")
            
            elif 'cost_data' in file_path:
                required_fields = ['cost_data', 'metadata']
                for field in required_fields:
                    if field not in data:
                        validation_result['errors'].append(f"Missing required field: {field}")
            
            elif 'embodied_carbon' in file_path:
                required_fields = ['embodied_carbon', 'metadata']
                for field in required_fields:
                    if field not in data:
                        validation_result['errors'].append(f"Missing required field: {field}")
            
            return validation_result
            
        except json.JSONDecodeError as e:
            return {
                'file_path': file_path,
                'is_valid_json': False,
                'error': str(e)
            }
        except Exception as e:
            return {
                'file_path': file_path,
                'is_valid_json': False,
                'error': str(e)
            }
    
    def validate_all_files(self) -> Dict[str, Any]:
        """Validate all files in the dataset"""
        validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'file_results': {},
            'overall_errors': []
        }
        
        # List all JSON files
        json_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        validation_results['total_files'] = len(json_files)
        
        # Validate each file
        for file_path in json_files:
            file_result = self.validate_json_structure(file_path)
            validation_results['file_results'][file_path] = file_result
            
            if file_result.get('is_valid_json', False):
                validation_results['valid_files'] += 1
            else:
                validation_results['invalid_files'] += 1
                validation_results['overall_errors'].append(f"Invalid JSON in {file_path}")
        
        return validation_results
    
    def generate_validation_report(self, output_file: str = None):
        """Generate a comprehensive validation report"""
        results = self.validate_all_files()
        
        report = f"""
# Retrofit Dataset Validation Report

## Summary
- Total files: {results['total_files']}
- Valid files: {results['valid_files']}
- Invalid files: {results['invalid_files']}
- Overall errors: {len(results['overall_errors'])}

## File Validation Results
"""
        
        for file_path, file_result in results['file_results'].items():
            report += f"\n### {file_path}\n"
            report += f"- Valid JSON: {file_result.get('is_valid_json', False)}\n"
            if 'errors' in file_result:
                report += f"- Errors: {len(file_result['errors'])}\n"
                for error in file_result['errors']:
                    report += f"  - {error}\n"
        
        if results['overall_errors']:
            report += "\n## Overall Errors\n"
            for error in results['overall_errors']:
                report += f"- {error}\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        else:
            print(report)
        
        return results

def main():
    """Main function"""
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    validator = RetrofitDataValidator(data_dir)
    results = validator.generate_validation_report()
    
    # Exit with error code if there are validation errors
    if results['invalid_files'] > 0 or len(results['overall_errors']) > 0:
        sys.exit(1)
    else:
        print("All validations passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
