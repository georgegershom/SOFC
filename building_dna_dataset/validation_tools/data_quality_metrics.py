#!/usr/bin/env python3
"""
Building DNA Dataset Quality Metrics Calculator
==============================================

Advanced quality metrics and statistical analysis for the building DNA dataset.
Provides detailed insights into data quality, completeness, and consistency.

Author: AI Assistant
Date: 2025-10-16
Version: 1.0.0
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)

class DataQualityMetrics:
    """
    Advanced quality metrics calculator for building DNA dataset
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the quality metrics calculator
        
        Args:
            dataset_path: Path to the building DNA dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'completeness_metrics': {},
            'consistency_metrics': {},
            'accuracy_metrics': {},
            'timeliness_metrics': {},
            'uniqueness_metrics': {},
            'validity_metrics': {},
            'overall_quality_index': 0.0
        }
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics
        
        Returns:
            Dictionary containing all quality metrics
        """
        logger.info("Calculating comprehensive quality metrics...")
        
        self._calculate_completeness_metrics()
        self._calculate_consistency_metrics()
        self._calculate_accuracy_metrics()
        self._calculate_timeliness_metrics()
        self._calculate_uniqueness_metrics()
        self._calculate_validity_metrics()
        self._calculate_overall_quality_index()
        
        return self.metrics
    
    def _calculate_completeness_metrics(self):
        """Calculate data completeness metrics"""
        logger.info("Calculating completeness metrics...")
        
        completeness = {
            'file_completeness': {},
            'field_completeness': {},
            'data_density': {},
            'missing_data_patterns': {}
        }
        
        # Define expected files and their critical fields
        expected_data = {
            'building_metadata.json': {
                'critical_fields': ['building_id', 'building_type', 'location', 'building_characteristics'],
                'optional_fields': ['dataset_metadata']
            },
            'geometric_data/bim_geometry.json': {
                'critical_fields': ['building_envelope', 'interior_spaces'],
                'optional_fields': ['structural_elements']
            },
            'construction_materials/wall_assemblies.json': {
                'critical_fields': ['wall_assemblies'],
                'optional_fields': ['window_door_specifications']
            },
            'system_data/hvac_systems.json': {
                'critical_fields': ['hvac_systems'],
                'optional_fields': ['ventilation_systems', 'zone_systems']
            }
        }
        
        total_files = len(expected_data)
        existing_files = 0
        
        for file_path, field_info in expected_data.items():
            full_path = self.dataset_path / file_path
            
            if full_path.exists():
                existing_files += 1
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Calculate field completeness
                    critical_fields = field_info['critical_fields']
                    optional_fields = field_info.get('optional_fields', [])
                    all_fields = critical_fields + optional_fields
                    
                    present_critical = sum(1 for field in critical_fields if field in data)
                    present_optional = sum(1 for field in optional_fields if field in data)
                    present_total = present_critical + present_optional
                    
                    completeness['field_completeness'][file_path] = {
                        'critical_completeness_percent': (present_critical / len(critical_fields)) * 100 if critical_fields else 100,
                        'optional_completeness_percent': (present_optional / len(optional_fields)) * 100 if optional_fields else 100,
                        'overall_completeness_percent': (present_total / len(all_fields)) * 100 if all_fields else 100,
                        'missing_critical_fields': [f for f in critical_fields if f not in data],
                        'missing_optional_fields': [f for f in optional_fields if f not in data]
                    }
                    
                    # Calculate data density (non-null values in nested structures)
                    density_score = self._calculate_data_density(data)
                    completeness['data_density'][file_path] = density_score
                    
                except Exception as e:
                    completeness['field_completeness'][file_path] = {'error': str(e)}
                    completeness['data_density'][file_path] = 0.0
            else:
                completeness['field_completeness'][file_path] = {
                    'critical_completeness_percent': 0.0,
                    'optional_completeness_percent': 0.0,
                    'overall_completeness_percent': 0.0,
                    'missing_critical_fields': field_info['critical_fields'],
                    'missing_optional_fields': field_info.get('optional_fields', [])
                }
                completeness['data_density'][file_path] = 0.0
        
        # Overall file completeness
        completeness['file_completeness'] = {
            'total_expected_files': total_files,
            'existing_files': existing_files,
            'file_completeness_percent': (existing_files / total_files) * 100
        }
        
        # Calculate average completeness scores
        field_scores = [fc.get('overall_completeness_percent', 0) for fc in completeness['field_completeness'].values() if 'error' not in fc]
        density_scores = [d for d in completeness['data_density'].values() if isinstance(d, (int, float))]
        
        completeness['summary'] = {
            'average_field_completeness_percent': statistics.mean(field_scores) if field_scores else 0.0,
            'average_data_density_percent': statistics.mean(density_scores) if density_scores else 0.0,
            'overall_completeness_score': (
                completeness['file_completeness']['file_completeness_percent'] * 0.3 +
                (statistics.mean(field_scores) if field_scores else 0.0) * 0.4 +
                (statistics.mean(density_scores) if density_scores else 0.0) * 0.3
            )
        }
        
        self.metrics['completeness_metrics'] = completeness
    
    def _calculate_data_density(self, data: Any, max_depth: int = 5, current_depth: int = 0) -> float:
        """
        Calculate data density (percentage of non-null/non-empty values)
        
        Args:
            data: Data structure to analyze
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            Data density score (0.0 to 100.0)
        """
        if current_depth >= max_depth:
            return 100.0 if data is not None else 0.0
        
        if data is None or data == "" or data == []:
            return 0.0
        
        if isinstance(data, dict):
            if not data:
                return 0.0
            
            scores = []
            for value in data.values():
                scores.append(self._calculate_data_density(value, max_depth, current_depth + 1))
            
            return statistics.mean(scores) if scores else 0.0
        
        elif isinstance(data, list):
            if not data:
                return 0.0
            
            scores = []
            for item in data:
                scores.append(self._calculate_data_density(item, max_depth, current_depth + 1))
            
            return statistics.mean(scores) if scores else 0.0
        
        else:
            # Primitive value - check if it's meaningful
            if isinstance(data, str) and len(data.strip()) == 0:
                return 0.0
            elif isinstance(data, (int, float)) and data == 0:
                return 50.0  # Zero might be meaningful
            else:
                return 100.0
    
    def _calculate_consistency_metrics(self):
        """Calculate data consistency metrics"""
        logger.info("Calculating consistency metrics...")
        
        consistency = {
            'cross_file_consistency': {},
            'unit_consistency': {},
            'format_consistency': {},
            'value_range_consistency': {}
        }
        
        # Check cross-file consistency
        try:
            # Load key files for comparison
            metadata_path = self.dataset_path / 'building_metadata.json'
            geometry_path = self.dataset_path / 'geometric_data/bim_geometry.json'
            
            if metadata_path.exists() and geometry_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                with open(geometry_path, 'r') as f:
                    geometry = json.load(f)
                
                # Building area consistency
                metadata_area = metadata.get('building_characteristics', {}).get('total_floor_area_m2', 0)
                
                geometry_area = 0
                if 'interior_spaces' in geometry:
                    for space in geometry['interior_spaces']:
                        geometry_area += space.get('area_m2', 0)
                
                area_consistency = 100.0 - min(100.0, abs(metadata_area - geometry_area) / max(metadata_area, 1) * 100)
                
                consistency['cross_file_consistency']['building_area'] = {
                    'metadata_area_m2': metadata_area,
                    'geometry_area_m2': geometry_area,
                    'consistency_percent': area_consistency,
                    'absolute_difference_m2': abs(metadata_area - geometry_area)
                }
            
            # Unit consistency check
            unit_patterns = self._analyze_unit_consistency()
            consistency['unit_consistency'] = unit_patterns
            
            # Format consistency
            format_patterns = self._analyze_format_consistency()
            consistency['format_consistency'] = format_patterns
            
        except Exception as e:
            consistency['error'] = str(e)
        
        # Calculate overall consistency score
        consistency_scores = []
        
        if 'building_area' in consistency['cross_file_consistency']:
            consistency_scores.append(consistency['cross_file_consistency']['building_area']['consistency_percent'])
        
        unit_score = consistency['unit_consistency'].get('overall_consistency_percent', 100.0)
        format_score = consistency['format_consistency'].get('overall_consistency_percent', 100.0)
        
        consistency_scores.extend([unit_score, format_score])
        
        consistency['summary'] = {
            'overall_consistency_score': statistics.mean(consistency_scores) if consistency_scores else 100.0,
            'component_scores': {
                'cross_file_consistency': consistency_scores[0] if consistency_scores else 100.0,
                'unit_consistency': unit_score,
                'format_consistency': format_score
            }
        }
        
        self.metrics['consistency_metrics'] = consistency
    
    def _analyze_unit_consistency(self) -> Dict[str, Any]:
        """Analyze consistency of units across the dataset"""
        unit_analysis = {
            'area_units': [],
            'temperature_units': [],
            'pressure_units': [],
            'energy_units': [],
            'overall_consistency_percent': 100.0
        }
        
        # This would be expanded to check all files for unit consistency
        # For now, return a placeholder
        return unit_analysis
    
    def _analyze_format_consistency(self) -> Dict[str, Any]:
        """Analyze format consistency across the dataset"""
        format_analysis = {
            'date_formats': [],
            'id_formats': [],
            'coordinate_formats': [],
            'overall_consistency_percent': 100.0
        }
        
        # This would be expanded to check format consistency
        # For now, return a placeholder
        return format_analysis
    
    def _calculate_accuracy_metrics(self):
        """Calculate data accuracy metrics"""
        logger.info("Calculating accuracy metrics...")
        
        accuracy = {
            'physical_constraints': {},
            'engineering_limits': {},
            'statistical_outliers': {},
            'reference_validation': {}
        }
        
        # Check physical constraints
        constraints_score = self._validate_physical_constraints()
        accuracy['physical_constraints'] = constraints_score
        
        # Check engineering limits
        engineering_score = self._validate_engineering_limits()
        accuracy['engineering_limits'] = engineering_score
        
        # Statistical outlier detection
        outliers_score = self._detect_statistical_outliers()
        accuracy['statistical_outliers'] = outliers_score
        
        # Calculate overall accuracy score
        scores = [
            constraints_score.get('overall_score', 100.0),
            engineering_score.get('overall_score', 100.0),
            outliers_score.get('overall_score', 100.0)
        ]
        
        accuracy['summary'] = {
            'overall_accuracy_score': statistics.mean(scores),
            'component_scores': {
                'physical_constraints': scores[0],
                'engineering_limits': scores[1],
                'statistical_outliers': scores[2]
            }
        }
        
        self.metrics['accuracy_metrics'] = accuracy
    
    def _validate_physical_constraints(self) -> Dict[str, Any]:
        """Validate physical constraints in the data"""
        constraints = {
            'thermal_properties': [],
            'geometric_properties': [],
            'system_properties': [],
            'overall_score': 100.0
        }
        
        violations = 0
        total_checks = 0
        
        # Check thermal properties
        wall_assemblies_path = self.dataset_path / 'construction_materials/wall_assemblies.json'
        if wall_assemblies_path.exists():
            try:
                with open(wall_assemblies_path, 'r') as f:
                    wall_data = json.load(f)
                
                if 'wall_assemblies' in wall_data:
                    for assembly in wall_data['wall_assemblies']:
                        total_checks += 1
                        
                        # R-value should be positive and reasonable
                        r_value = assembly.get('total_r_value_m2k_w', 0)
                        if r_value <= 0 or r_value > 20:  # Reasonable range for building assemblies
                            violations += 1
                            constraints['thermal_properties'].append({
                                'assembly_id': assembly.get('assembly_id'),
                                'issue': 'R-value out of reasonable range',
                                'value': r_value
                            })
                        
                        # U-value should be positive and inverse of R-value
                        u_value = assembly.get('total_u_value_w_m2k', 0)
                        if u_value <= 0 or (r_value > 0 and abs(1/r_value - u_value) > 0.1):
                            violations += 1
                            constraints['thermal_properties'].append({
                                'assembly_id': assembly.get('assembly_id'),
                                'issue': 'U-value inconsistent with R-value',
                                'r_value': r_value,
                                'u_value': u_value
                            })
            
            except Exception as e:
                constraints['thermal_properties'].append({'error': str(e)})
        
        # Calculate score
        if total_checks > 0:
            constraints['overall_score'] = max(0.0, 100.0 - (violations / total_checks * 100))
        
        return constraints
    
    def _validate_engineering_limits(self) -> Dict[str, Any]:
        """Validate engineering limits and realistic values"""
        limits = {
            'hvac_efficiency': [],
            'system_capacities': [],
            'performance_metrics': [],
            'overall_score': 100.0
        }
        
        # This would be expanded with specific engineering limit checks
        return limits
    
    def _detect_statistical_outliers(self) -> Dict[str, Any]:
        """Detect statistical outliers in numerical data"""
        outliers = {
            'detected_outliers': [],
            'outlier_statistics': {},
            'overall_score': 100.0
        }
        
        # This would be expanded with statistical outlier detection
        return outliers
    
    def _calculate_timeliness_metrics(self):
        """Calculate data timeliness metrics"""
        logger.info("Calculating timeliness metrics...")
        
        timeliness = {
            'data_freshness': {},
            'update_frequency': {},
            'temporal_consistency': {},
            'overall_timeliness_score': 100.0
        }
        
        # Analyze file modification times and data timestamps
        current_time = datetime.now()
        
        for file_path in self.dataset_path.rglob('*.json'):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.dataset_path)
                
                # File system timestamp
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                age_days = (current_time - mod_time).days
                
                timeliness['data_freshness'][str(relative_path)] = {
                    'last_modified': mod_time.isoformat(),
                    'age_days': age_days,
                    'freshness_score': max(0.0, 100.0 - (age_days * 0.5))  # Decrease 0.5% per day
                }
        
        # Calculate overall timeliness score
        freshness_scores = [df['freshness_score'] for df in timeliness['data_freshness'].values()]
        timeliness['overall_timeliness_score'] = statistics.mean(freshness_scores) if freshness_scores else 100.0
        
        self.metrics['timeliness_metrics'] = timeliness
    
    def _calculate_uniqueness_metrics(self):
        """Calculate data uniqueness metrics"""
        logger.info("Calculating uniqueness metrics...")
        
        uniqueness = {
            'duplicate_detection': {},
            'id_uniqueness': {},
            'overall_uniqueness_score': 100.0
        }
        
        # Check for duplicate IDs across the dataset
        all_ids = {}
        
        for file_path in self.dataset_path.rglob('*.json'):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract IDs from the data structure
                    ids = self._extract_ids(data)
                    
                    for id_value in ids:
                        if id_value in all_ids:
                            all_ids[id_value].append(str(file_path.relative_to(self.dataset_path)))
                        else:
                            all_ids[id_value] = [str(file_path.relative_to(self.dataset_path))]
                
                except Exception as e:
                    continue
        
        # Find duplicates
        duplicates = {id_val: files for id_val, files in all_ids.items() if len(files) > 1}
        
        uniqueness['duplicate_detection'] = {
            'total_ids': len(all_ids),
            'duplicate_ids': len(duplicates),
            'duplicates': duplicates
        }
        
        # Calculate uniqueness score
        if all_ids:
            uniqueness_score = max(0.0, 100.0 - (len(duplicates) / len(all_ids) * 100))
        else:
            uniqueness_score = 100.0
        
        uniqueness['overall_uniqueness_score'] = uniqueness_score
        
        self.metrics['uniqueness_metrics'] = uniqueness
    
    def _extract_ids(self, data: Any, id_fields: List[str] = None) -> List[str]:
        """Extract ID values from data structure"""
        if id_fields is None:
            id_fields = ['id', '_id', 'system_id', 'assembly_id', 'fixture_id', 'zone_id', 'building_id']
        
        ids = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if any(id_field in key.lower() for id_field in id_fields):
                    if isinstance(value, str):
                        ids.append(value)
                elif isinstance(value, (dict, list)):
                    ids.extend(self._extract_ids(value, id_fields))
        
        elif isinstance(data, list):
            for item in data:
                ids.extend(self._extract_ids(item, id_fields))
        
        return ids
    
    def _calculate_validity_metrics(self):
        """Calculate data validity metrics"""
        logger.info("Calculating validity metrics...")
        
        validity = {
            'schema_compliance': {},
            'data_type_validation': {},
            'format_validation': {},
            'overall_validity_score': 100.0
        }
        
        # Check JSON validity and basic data types
        valid_files = 0
        total_files = 0
        
        for file_path in self.dataset_path.rglob('*.json'):
            if file_path.is_file():
                total_files += 1
                relative_path = file_path.relative_to(self.dataset_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    valid_files += 1
                    validity['schema_compliance'][str(relative_path)] = {
                        'valid_json': True,
                        'data_type': type(data).__name__,
                        'structure_score': self._calculate_structure_score(data)
                    }
                
                except json.JSONDecodeError as e:
                    validity['schema_compliance'][str(relative_path)] = {
                        'valid_json': False,
                        'error': str(e)
                    }
                except Exception as e:
                    validity['schema_compliance'][str(relative_path)] = {
                        'valid_json': False,
                        'error': f"File read error: {str(e)}"
                    }
        
        # Calculate overall validity score
        if total_files > 0:
            validity['overall_validity_score'] = (valid_files / total_files) * 100
        
        self.metrics['validity_metrics'] = validity
    
    def _calculate_structure_score(self, data: Any) -> float:
        """Calculate structure quality score for data"""
        if not isinstance(data, dict):
            return 50.0
        
        if not data:
            return 0.0
        
        # Score based on depth and organization
        max_depth = self._get_max_depth(data)
        key_count = len(data)
        
        # Reasonable structure should have 2-6 levels and organized keys
        depth_score = min(100.0, max(0.0, 100.0 - abs(max_depth - 4) * 10))
        key_score = min(100.0, max(0.0, 100.0 - abs(key_count - 10) * 2))
        
        return (depth_score + key_score) / 2
    
    def _get_max_depth(self, data: Any, current_depth: int = 0) -> int:
        """Get maximum depth of nested data structure"""
        if not isinstance(data, (dict, list)):
            return current_depth
        
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._get_max_depth(value, current_depth + 1) for value in data.values())
        
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._get_max_depth(item, current_depth + 1) for item in data)
        
        return current_depth
    
    def _calculate_overall_quality_index(self):
        """Calculate overall quality index"""
        logger.info("Calculating overall quality index...")
        
        # Weight factors for different quality dimensions
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.20,
            'timeliness': 0.10,
            'uniqueness': 0.10,
            'validity': 0.15
        }
        
        # Get scores from each dimension
        scores = {
            'completeness': self.metrics['completeness_metrics'].get('summary', {}).get('overall_completeness_score', 0.0),
            'consistency': self.metrics['consistency_metrics'].get('summary', {}).get('overall_consistency_score', 100.0),
            'accuracy': self.metrics['accuracy_metrics'].get('summary', {}).get('overall_accuracy_score', 100.0),
            'timeliness': self.metrics['timeliness_metrics'].get('overall_timeliness_score', 100.0),
            'uniqueness': self.metrics['uniqueness_metrics'].get('overall_uniqueness_score', 100.0),
            'validity': self.metrics['validity_metrics'].get('overall_validity_score', 100.0)
        }
        
        # Calculate weighted overall quality index
        overall_quality = sum(scores[dimension] * weights[dimension] for dimension in weights)
        
        self.metrics['overall_quality_index'] = {
            'quality_index': round(overall_quality, 2),
            'dimension_scores': scores,
            'weights': weights,
            'quality_grade': self._get_quality_grade(overall_quality)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'A-'
        elif score >= 80:
            return 'B+'
        elif score >= 75:
            return 'B'
        elif score >= 70:
            return 'B-'
        elif score >= 65:
            return 'C+'
        elif score >= 60:
            return 'C'
        elif score >= 55:
            return 'C-'
        else:
            return 'F'
    
    def save_metrics_report(self, output_path: str = None) -> str:
        """Save quality metrics to JSON file"""
        if output_path is None:
            output_path = self.dataset_path / 'quality_metrics_report.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality metrics report saved to: {output_path}")
        return str(output_path)
    
    def print_quality_summary(self):
        """Print quality metrics summary"""
        print("\n" + "="*80)
        print("BUILDING DNA DATASET QUALITY METRICS SUMMARY")
        print("="*80)
        
        overall = self.metrics.get('overall_quality_index', {})
        print(f"Overall Quality Index: {overall.get('quality_index', 0):.2f}")
        print(f"Quality Grade: {overall.get('quality_grade', 'N/A')}")
        
        print("\nDimension Scores:")
        for dimension, score in overall.get('dimension_scores', {}).items():
            print(f"  {dimension.capitalize()}: {score:.1f}%")
        
        print("\nKey Findings:")
        
        # Completeness findings
        completeness = self.metrics.get('completeness_metrics', {}).get('summary', {})
        print(f"  • File Completeness: {self.metrics.get('completeness_metrics', {}).get('file_completeness', {}).get('file_completeness_percent', 0):.1f}%")
        print(f"  • Average Field Completeness: {completeness.get('average_field_completeness_percent', 0):.1f}%")
        
        # Consistency findings
        consistency = self.metrics.get('consistency_metrics', {}).get('summary', {})
        print(f"  • Overall Consistency: {consistency.get('overall_consistency_score', 100):.1f}%")
        
        # Uniqueness findings
        uniqueness = self.metrics.get('uniqueness_metrics', {})
        duplicates = uniqueness.get('duplicate_detection', {}).get('duplicate_ids', 0)
        if duplicates > 0:
            print(f"  • Duplicate IDs Found: {duplicates}")
        
        print("="*80)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate Building DNA Dataset Quality Metrics')
    parser.add_argument('dataset_path', help='Path to the building DNA dataset directory')
    parser.add_argument('--output', '-o', help='Output path for metrics report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run metrics calculator
    calculator = DataQualityMetrics(args.dataset_path)
    metrics = calculator.calculate_all_metrics()
    
    # Save report
    report_path = calculator.save_metrics_report(args.output)
    
    # Print summary
    calculator.print_quality_summary()
    
    print(f"\nDetailed metrics report saved to: {report_path}")


if __name__ == '__main__':
    main()