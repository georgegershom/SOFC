#!/usr/bin/env python3
"""
Building DNA Dataset Validation Tool
====================================

This script validates the integrity, completeness, and consistency of the 
building DNA dataset for the Dynamic Digital Twin Framework.

Author: AI Assistant
Date: 2025-10-16
Version: 1.0.0
"""

import json
import os
import sys
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_report.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BuildingDNAValidator:
    """
    Comprehensive validator for building DNA dataset
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the validator
        
        Args:
            dataset_path: Path to the building DNA dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'overall_status': 'PENDING',
            'validation_summary': {},
            'detailed_results': {},
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Define expected file structure
        self.expected_structure = {
            'building_metadata.json': 'root',
            'geometric_data/bim_geometry.json': 'geometric',
            'geometric_data/floor_plans.json': 'geometric',
            'construction_materials/wall_assemblies.json': 'construction',
            'construction_materials/roof_floor_assemblies.json': 'construction',
            'system_data/hvac_systems.json': 'systems',
            'system_data/dhw_systems.json': 'systems',
            'system_data/lighting_systems.json': 'systems',
            'system_data/renewable_energy_systems.json': 'systems',
            'environmental_data/lidar_aerial_data.json': 'environmental',
            'environmental_data/air_tightness_data.json': 'environmental'
        }
        
        # Define validation thresholds
        self.thresholds = {
            'min_completeness_percent': 95.0,
            'max_missing_critical_fields': 0,
            'min_data_quality_score': 0.9,
            'max_inconsistencies': 5,
            'min_thermal_performance_r_value': 1.0,
            'max_thermal_performance_u_value': 2.0,
            'min_hvac_efficiency_cop': 2.0,
            'max_air_leakage_ach50': 5.0
        }
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of the entire dataset
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Starting comprehensive dataset validation...")
        
        try:
            # 1. Validate file structure
            self._validate_file_structure()
            
            # 2. Validate JSON syntax and structure
            self._validate_json_files()
            
            # 3. Validate data completeness
            self._validate_data_completeness()
            
            # 4. Validate data consistency
            self._validate_data_consistency()
            
            # 5. Validate physical constraints
            self._validate_physical_constraints()
            
            # 6. Validate performance metrics
            self._validate_performance_metrics()
            
            # 7. Calculate overall quality score
            self._calculate_quality_score()
            
            # 8. Generate recommendations
            self._generate_recommendations()
            
            # Determine overall status
            if len(self.validation_results['errors']) == 0:
                if len(self.validation_results['warnings']) <= 5:
                    self.validation_results['overall_status'] = 'PASSED'
                else:
                    self.validation_results['overall_status'] = 'PASSED_WITH_WARNINGS'
            else:
                self.validation_results['overall_status'] = 'FAILED'
            
            logger.info(f"Validation completed with status: {self.validation_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {str(e)}")
            self.validation_results['overall_status'] = 'ERROR'
            self.validation_results['errors'].append(f"Validation exception: {str(e)}")
        
        return self.validation_results
    
    def _validate_file_structure(self):
        """Validate that all expected files exist"""
        logger.info("Validating file structure...")
        
        missing_files = []
        existing_files = []
        
        for file_path, category in self.expected_structure.items():
            full_path = self.dataset_path / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        self.validation_results['detailed_results']['file_structure'] = {
            'total_expected': len(self.expected_structure),
            'existing_files': len(existing_files),
            'missing_files': missing_files,
            'completeness_percent': (len(existing_files) / len(self.expected_structure)) * 100
        }
        
        if missing_files:
            self.validation_results['errors'].extend([f"Missing file: {f}" for f in missing_files])
        
        logger.info(f"File structure validation: {len(existing_files)}/{len(self.expected_structure)} files found")
    
    def _validate_json_files(self):
        """Validate JSON syntax and basic structure"""
        logger.info("Validating JSON files...")
        
        json_results = {}
        
        for file_path, category in self.expected_structure.items():
            full_path = self.dataset_path / file_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                json_results[file_path] = {
                    'valid_json': True,
                    'size_bytes': full_path.stat().st_size,
                    'top_level_keys': len(data) if isinstance(data, dict) else 0,
                    'data_type': type(data).__name__
                }
                
                # Basic structure validation
                if isinstance(data, dict) and len(data) == 0:
                    self.validation_results['warnings'].append(f"Empty JSON object in {file_path}")
                
            except json.JSONDecodeError as e:
                json_results[file_path] = {
                    'valid_json': False,
                    'error': str(e)
                }
                self.validation_results['errors'].append(f"Invalid JSON in {file_path}: {str(e)}")
            except Exception as e:
                json_results[file_path] = {
                    'valid_json': False,
                    'error': f"File read error: {str(e)}"
                }
                self.validation_results['errors'].append(f"Error reading {file_path}: {str(e)}")
        
        self.validation_results['detailed_results']['json_validation'] = json_results
        logger.info(f"JSON validation completed for {len(json_results)} files")
    
    def _validate_data_completeness(self):
        """Validate data completeness and required fields"""
        logger.info("Validating data completeness...")
        
        completeness_results = {}
        
        # Define critical fields for each file type
        critical_fields = {
            'building_metadata.json': [
                'building_id', 'building_type', 'location', 'building_characteristics'
            ],
            'geometric_data/bim_geometry.json': [
                'building_envelope', 'interior_spaces', 'structural_elements'
            ],
            'construction_materials/wall_assemblies.json': [
                'wall_assemblies'
            ],
            'system_data/hvac_systems.json': [
                'hvac_systems'
            ]
        }
        
        for file_path, required_fields in critical_fields.items():
            full_path = self.dataset_path / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                missing_fields = []
                present_fields = []
                
                for field in required_fields:
                    if field in data:
                        present_fields.append(field)
                    else:
                        missing_fields.append(field)
                
                completeness_percent = (len(present_fields) / len(required_fields)) * 100
                
                completeness_results[file_path] = {
                    'completeness_percent': completeness_percent,
                    'missing_critical_fields': missing_fields,
                    'present_fields': present_fields
                }
                
                if missing_fields:
                    self.validation_results['errors'].extend([
                        f"Missing critical field '{field}' in {file_path}" for field in missing_fields
                    ])
                
            except Exception as e:
                completeness_results[file_path] = {
                    'error': str(e)
                }
        
        self.validation_results['detailed_results']['completeness'] = completeness_results
        logger.info("Data completeness validation completed")
    
    def _validate_data_consistency(self):
        """Validate data consistency across files"""
        logger.info("Validating data consistency...")
        
        consistency_results = {
            'cross_references': [],
            'unit_consistency': [],
            'value_ranges': []
        }
        
        try:
            # Load key files for cross-reference validation
            metadata_path = self.dataset_path / 'building_metadata.json'
            geometry_path = self.dataset_path / 'geometric_data/bim_geometry.json'
            
            if metadata_path.exists() and geometry_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                with open(geometry_path, 'r') as f:
                    geometry = json.load(f)
                
                # Check building area consistency
                if 'building_characteristics' in metadata and 'building_envelope' in geometry:
                    metadata_area = metadata.get('building_characteristics', {}).get('total_floor_area_m2', 0)
                    
                    # Calculate geometry area (simplified)
                    geometry_area = 0
                    if 'interior_spaces' in geometry:
                        for space in geometry['interior_spaces']:
                            geometry_area += space.get('area_m2', 0)
                    
                    if abs(metadata_area - geometry_area) > (metadata_area * 0.1):  # 10% tolerance
                        consistency_results['cross_references'].append({
                            'issue': 'Area mismatch between metadata and geometry',
                            'metadata_area': metadata_area,
                            'geometry_area': geometry_area,
                            'difference_percent': abs(metadata_area - geometry_area) / metadata_area * 100
                        })
                        self.validation_results['warnings'].append(
                            f"Building area mismatch: metadata={metadata_area}m², geometry={geometry_area}m²"
                        )
            
            # Validate thermal properties consistency
            wall_assemblies_path = self.dataset_path / 'construction_materials/wall_assemblies.json'
            if wall_assemblies_path.exists():
                with open(wall_assemblies_path, 'r') as f:
                    wall_data = json.load(f)
                
                if 'wall_assemblies' in wall_data:
                    for assembly in wall_data['wall_assemblies']:
                        r_value = assembly.get('total_r_value_m2k_w', 0)
                        u_value = assembly.get('total_u_value_w_m2k', 0)
                        
                        # Check R-value and U-value consistency (U = 1/R)
                        if r_value > 0 and u_value > 0:
                            calculated_u = 1.0 / r_value
                            if abs(calculated_u - u_value) > 0.1:
                                consistency_results['value_ranges'].append({
                                    'assembly_id': assembly.get('assembly_id', 'unknown'),
                                    'issue': 'R-value and U-value inconsistency',
                                    'r_value': r_value,
                                    'u_value': u_value,
                                    'calculated_u': calculated_u
                                })
                                self.validation_results['warnings'].append(
                                    f"Thermal property inconsistency in {assembly.get('assembly_id', 'unknown')}"
                                )
        
        except Exception as e:
            consistency_results['error'] = str(e)
            self.validation_results['warnings'].append(f"Consistency validation error: {str(e)}")
        
        self.validation_results['detailed_results']['consistency'] = consistency_results
        logger.info("Data consistency validation completed")
    
    def _validate_physical_constraints(self):
        """Validate physical constraints and realistic values"""
        logger.info("Validating physical constraints...")
        
        constraints_results = {
            'thermal_properties': [],
            'geometric_constraints': [],
            'system_capacities': []
        }
        
        # Validate thermal properties
        wall_assemblies_path = self.dataset_path / 'construction_materials/wall_assemblies.json'
        if wall_assemblies_path.exists():
            try:
                with open(wall_assemblies_path, 'r') as f:
                    wall_data = json.load(f)
                
                if 'wall_assemblies' in wall_data:
                    for assembly in wall_data['wall_assemblies']:
                        assembly_id = assembly.get('assembly_id', 'unknown')
                        
                        # Check R-value range
                        r_value = assembly.get('total_r_value_m2k_w', 0)
                        if r_value < self.thresholds['min_thermal_performance_r_value']:
                            constraints_results['thermal_properties'].append({
                                'assembly_id': assembly_id,
                                'issue': 'R-value below minimum threshold',
                                'value': r_value,
                                'threshold': self.thresholds['min_thermal_performance_r_value']
                            })
                            self.validation_results['warnings'].append(
                                f"Low R-value ({r_value}) in assembly {assembly_id}"
                            )
                        
                        # Check U-value range
                        u_value = assembly.get('total_u_value_w_m2k', 0)
                        if u_value > self.thresholds['max_thermal_performance_u_value']:
                            constraints_results['thermal_properties'].append({
                                'assembly_id': assembly_id,
                                'issue': 'U-value above maximum threshold',
                                'value': u_value,
                                'threshold': self.thresholds['max_thermal_performance_u_value']
                            })
                            self.validation_results['warnings'].append(
                                f"High U-value ({u_value}) in assembly {assembly_id}"
                            )
            
            except Exception as e:
                constraints_results['thermal_properties'].append({'error': str(e)})
        
        # Validate HVAC system constraints
        hvac_path = self.dataset_path / 'system_data/hvac_systems.json'
        if hvac_path.exists():
            try:
                with open(hvac_path, 'r') as f:
                    hvac_data = json.load(f)
                
                if 'hvac_systems' in hvac_data:
                    for system in hvac_data['hvac_systems']:
                        system_id = system.get('system_id', 'unknown')
                        
                        # Check COP values
                        if 'efficiency_ratings' in system:
                            cooling_cop = system['efficiency_ratings'].get('cooling_cop', 0)
                            heating_cop = system['efficiency_ratings'].get('heating_cop', 0)
                            
                            if cooling_cop > 0 and cooling_cop < self.thresholds['min_hvac_efficiency_cop']:
                                constraints_results['system_capacities'].append({
                                    'system_id': system_id,
                                    'issue': 'Cooling COP below minimum threshold',
                                    'value': cooling_cop,
                                    'threshold': self.thresholds['min_hvac_efficiency_cop']
                                })
                                self.validation_results['warnings'].append(
                                    f"Low cooling COP ({cooling_cop}) in system {system_id}"
                                )
            
            except Exception as e:
                constraints_results['system_capacities'].append({'error': str(e)})
        
        self.validation_results['detailed_results']['physical_constraints'] = constraints_results
        logger.info("Physical constraints validation completed")
    
    def _validate_performance_metrics(self):
        """Validate performance metrics and benchmarks"""
        logger.info("Validating performance metrics...")
        
        performance_results = {
            'air_tightness': [],
            'energy_efficiency': [],
            'renewable_energy': []
        }
        
        # Validate air tightness data
        air_tightness_path = self.dataset_path / 'environmental_data/air_tightness_data.json'
        if air_tightness_path.exists():
            try:
                with open(air_tightness_path, 'r') as f:
                    air_data = json.load(f)
                
                if 'blower_door_tests' in air_data:
                    for test in air_data['blower_door_tests']:
                        test_id = test.get('test_id', 'unknown')
                        
                        if 'test_results' in test:
                            ach50 = test['test_results'].get('air_changes_per_hour_50pa', 0)
                            
                            if ach50 > self.thresholds['max_air_leakage_ach50']:
                                performance_results['air_tightness'].append({
                                    'test_id': test_id,
                                    'issue': 'Air leakage above maximum threshold',
                                    'value': ach50,
                                    'threshold': self.thresholds['max_air_leakage_ach50']
                                })
                                self.validation_results['warnings'].append(
                                    f"High air leakage ({ach50} ACH50) in test {test_id}"
                                )
                            elif ach50 < 0.5:
                                performance_results['air_tightness'].append({
                                    'test_id': test_id,
                                    'issue': 'Unusually low air leakage (may indicate measurement error)',
                                    'value': ach50
                                })
                                self.validation_results['warnings'].append(
                                    f"Unusually low air leakage ({ach50} ACH50) in test {test_id}"
                                )
            
            except Exception as e:
                performance_results['air_tightness'].append({'error': str(e)})
        
        self.validation_results['detailed_results']['performance_metrics'] = performance_results
        logger.info("Performance metrics validation completed")
    
    def _calculate_quality_score(self):
        """Calculate overall data quality score"""
        logger.info("Calculating data quality score...")
        
        # Initialize scoring components
        scores = {
            'completeness': 0.0,
            'consistency': 0.0,
            'accuracy': 0.0,
            'validity': 0.0
        }
        
        # Completeness score (based on file structure and critical fields)
        file_structure = self.validation_results['detailed_results'].get('file_structure', {})
        completeness_percent = file_structure.get('completeness_percent', 0)
        scores['completeness'] = min(completeness_percent / 100.0, 1.0)
        
        # Consistency score (based on cross-reference validation)
        consistency_issues = len(self.validation_results['detailed_results'].get('consistency', {}).get('cross_references', []))
        scores['consistency'] = max(0.0, 1.0 - (consistency_issues * 0.1))
        
        # Accuracy score (based on physical constraints)
        constraint_issues = sum([
            len(self.validation_results['detailed_results'].get('physical_constraints', {}).get('thermal_properties', [])),
            len(self.validation_results['detailed_results'].get('physical_constraints', {}).get('system_capacities', []))
        ])
        scores['accuracy'] = max(0.0, 1.0 - (constraint_issues * 0.05))
        
        # Validity score (based on JSON validation and required fields)
        json_errors = len([r for r in self.validation_results['detailed_results'].get('json_validation', {}).values() 
                          if not r.get('valid_json', True)])
        scores['validity'] = max(0.0, 1.0 - (json_errors * 0.2))
        
        # Calculate weighted overall score
        weights = {
            'completeness': 0.3,
            'consistency': 0.25,
            'accuracy': 0.25,
            'validity': 0.2
        }
        
        overall_score = sum(scores[component] * weights[component] for component in scores)
        
        self.validation_results['validation_summary'] = {
            'overall_quality_score': round(overall_score, 3),
            'component_scores': {k: round(v, 3) for k, v in scores.items()},
            'weights': weights,
            'total_errors': len(self.validation_results['errors']),
            'total_warnings': len(self.validation_results['warnings'])
        }
        
        logger.info(f"Data quality score: {overall_score:.3f}")
    
    def _generate_recommendations(self):
        """Generate recommendations for data quality improvement"""
        logger.info("Generating recommendations...")
        
        recommendations = []
        
        # File structure recommendations
        file_structure = self.validation_results['detailed_results'].get('file_structure', {})
        if file_structure.get('completeness_percent', 100) < 100:
            recommendations.append({
                'category': 'File Structure',
                'priority': 'High',
                'recommendation': 'Complete missing data files to ensure full dataset coverage',
                'missing_files': file_structure.get('missing_files', [])
            })
        
        # Data quality recommendations
        quality_score = self.validation_results['validation_summary'].get('overall_quality_score', 1.0)
        if quality_score < self.thresholds['min_data_quality_score']:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'recommendation': f'Improve data quality score from {quality_score:.3f} to above {self.thresholds["min_data_quality_score"]:.3f}',
                'focus_areas': [k for k, v in self.validation_results['validation_summary']['component_scores'].items() if v < 0.8]
            })
        
        # Performance recommendations
        if len(self.validation_results['warnings']) > 10:
            recommendations.append({
                'category': 'Data Validation',
                'priority': 'Medium',
                'recommendation': 'Address validation warnings to improve dataset reliability',
                'warning_count': len(self.validation_results['warnings'])
            })
        
        # Consistency recommendations
        consistency_data = self.validation_results['detailed_results'].get('consistency', {})
        if consistency_data.get('cross_references') or consistency_data.get('value_ranges'):
            recommendations.append({
                'category': 'Data Consistency',
                'priority': 'Medium',
                'recommendation': 'Resolve data inconsistencies between related files and values',
                'inconsistency_count': len(consistency_data.get('cross_references', [])) + len(consistency_data.get('value_ranges', []))
            })
        
        self.validation_results['recommendations'] = recommendations
        logger.info(f"Generated {len(recommendations)} recommendations")
    
    def save_validation_report(self, output_path: str = None):
        """Save validation results to JSON file"""
        if output_path is None:
            output_path = self.dataset_path / 'validation_report.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print validation summary to console"""
        print("\n" + "="*80)
        print("BUILDING DNA DATASET VALIDATION SUMMARY")
        print("="*80)
        
        summary = self.validation_results['validation_summary']
        print(f"Overall Status: {self.validation_results['overall_status']}")
        print(f"Quality Score: {summary.get('overall_quality_score', 0):.3f}")
        print(f"Total Errors: {summary.get('total_errors', 0)}")
        print(f"Total Warnings: {summary.get('total_warnings', 0)}")
        
        print("\nComponent Scores:")
        for component, score in summary.get('component_scores', {}).items():
            print(f"  {component.capitalize()}: {score:.3f}")
        
        if self.validation_results['recommendations']:
            print(f"\nRecommendations ({len(self.validation_results['recommendations'])}):")
            for i, rec in enumerate(self.validation_results['recommendations'], 1):
                print(f"  {i}. [{rec['priority']}] {rec['recommendation']}")
        
        print("="*80)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Building DNA Dataset')
    parser.add_argument('dataset_path', help='Path to the building DNA dataset directory')
    parser.add_argument('--output', '-o', help='Output path for validation report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run validator
    validator = BuildingDNAValidator(args.dataset_path)
    results = validator.validate_dataset()
    
    # Save report
    report_path = validator.save_validation_report(args.output)
    
    # Print summary
    validator.print_summary()
    
    # Exit with appropriate code
    if results['overall_status'] in ['PASSED', 'PASSED_WITH_WARNINGS']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()