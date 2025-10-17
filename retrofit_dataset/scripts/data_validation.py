#!/usr/bin/env python3
"""
Data Validation Script for Retrofit Intervention & Lifecycle Data
Validates the integrity and consistency of the dataset
"""

import json
import os
import sys
from typing import Dict, List, Any
# import pandas as pd
# import numpy as np

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
                required_fields = ['capex_costs', 'opex_costs', 'metadata']
                for field in required_fields:
                    if field not in data:
                        validation_result['errors'].append(f"Missing required field: {field}")
            
            elif 'embodied_carbon' in file_path:
                required_fields = ['materials', 'systems', 'metadata']
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
    
    def validate_cost_data_consistency(self, cost_data: Dict) -> List[str]:
        """Validate cost data consistency"""
        errors = []
        
        # Check if all measures have cost data
        measures_file = os.path.join(self.data_dir, 'measures', 'retrofit_measures.json')
        if os.path.exists(measures_file):
            with open(measures_file, 'r') as f:
                measures_data = json.load(f)
            
            # Extract measure IDs
            measure_ids = []
            for category, measures in measures_data['retrofit_measures'].items():
                if isinstance(measures, dict):
                    for subcategory, measure_list in measures.items():
                        if isinstance(measure_list, list):
                            for measure in measure_list:
                                if 'id' in measure:
                                    measure_ids.append(measure['id'])
                        elif isinstance(measure_list, dict) and 'id' in measure_list:
                            measure_ids.append(measure_list['id'])
                elif isinstance(measures, list):
                    for measure in measures:
                        if 'id' in measure:
                            measure_ids.append(measure['id'])
            
            # Check if all measures have cost data
            for measure_id in measure_ids:
                if measure_id not in cost_data.get('capex_costs', {}):
                    errors.append(f"Missing cost data for measure: {measure_id}")
        
        return errors
    
    def validate_carbon_data_consistency(self, carbon_data: Dict) -> List[str]:
        """Validate carbon data consistency"""
        errors = []
        
        # Check if all materials have carbon data
        required_materials = ['polyisocyanurate', 'extruded_polystyrene', 'cellulose_insulation']
        for material in required_materials:
            if material not in carbon_data.get('materials', {}):
                errors.append(f"Missing carbon data for material: {material}")
        
        # Check if carbon values are reasonable
        for material_id, material_data in carbon_data.get('materials', {}).items():
            if 'total_lifecycle_gwp' in material_data:
                gwp = material_data['total_lifecycle_gwp'].get('total_kg_co2e_per_sqft', 0)
                if gwp < 0 or gwp > 100:  # Reasonable range check
                    errors.append(f"Unreasonable GWP value for {material_id}: {gwp}")
        
        return errors
    
    def validate_maintenance_data_consistency(self, maintenance_data: Dict) -> List[str]:
        """Validate maintenance data consistency"""
        errors = []
        
        # Check if maintenance costs are reasonable
        for system_type, systems in maintenance_data.get('maintenance_schedules', {}).items():
            for system_id, system_data in systems.items():
                if 'total_annual_maintenance_cost' in system_data:
                    cost = system_data['total_annual_maintenance_cost']
                    if cost < 0 or cost > 10000:  # Reasonable range check
                        errors.append(f"Unreasonable maintenance cost for {system_id}: {cost}")
        
        return errors
    
    def validate_measure_interactions(self, interactions_data: Dict) -> List[str]:
        """Validate measure interactions data"""
        errors = []
        
        # Check if synergy factors are reasonable
        for interaction_type, interactions in interactions_data.get('measure_interactions', {}).items():
            if interaction_type == 'synergistic_effects':
                for interaction_id, interaction_data in interactions.items():
                    synergy_factor = interaction_data.get('synergy_factor', 1.0)
                    if synergy_factor < 1.0 or synergy_factor > 2.0:
                        errors.append(f"Unreasonable synergy factor for {interaction_id}: {synergy_factor}")
        
        return errors
    
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
        
        # Additional consistency checks
        try:
            # Load cost data for consistency check
            cost_file = os.path.join(self.data_dir, 'measures', 'cost_data.json')
            if os.path.exists(cost_file):
                with open(cost_file, 'r') as f:
                    cost_data = json.load(f)
                cost_errors = self.validate_cost_data_consistency(cost_data)
                validation_results['overall_errors'].extend(cost_errors)
            
            # Load carbon data for consistency check
            carbon_file = os.path.join(self.data_dir, 'lci', 'embodied_carbon.json')
            if os.path.exists(carbon_file):
                with open(carbon_file, 'r') as f:
                    carbon_data = json.load(f)
                carbon_errors = self.validate_carbon_data_consistency(carbon_data)
                validation_results['overall_errors'].extend(carbon_errors)
            
            # Load maintenance data for consistency check
            maintenance_file = os.path.join(self.data_dir, 'maintenance', 'maintenance_schedules.json')
            if os.path.exists(maintenance_file):
                with open(maintenance_file, 'r') as f:
                    maintenance_data = json.load(f)
                maintenance_errors = self.validate_maintenance_data_consistency(maintenance_data)
                validation_results['overall_errors'].extend(maintenance_errors)
            
            # Load interactions data for consistency check
            interactions_file = os.path.join(self.data_dir, 'integration', 'measure_interactions.json')
            if os.path.exists(interactions_file):
                with open(interactions_file, 'r') as f:
                    interactions_data = json.load(f)
                interaction_errors = self.validate_measure_interactions(interactions_data)
                validation_results['overall_errors'].extend(interaction_errors)
                
        except Exception as e:
            validation_results['overall_errors'].append(f"Error during consistency checks: {str(e)}")
        
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
